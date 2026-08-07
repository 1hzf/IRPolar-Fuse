"""
Loss-function module for fusion only.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.transforms.functional import normalize

# Import the only required utility function: gradient.
from utils import gradient


# ============================================================
# Utility Functions
# ============================================================

def align_spatial_size(a, b, mode='bilinear', align_corners=False):
    """
    Ensure that two tensors have matching spatial dimensions.
    
    Args:
        a: reference tensor (B, C, H, W)
        b: tensor to align (B, C, H', W')
        mode: interpolation mode, default 'bilinear'
        align_corners: whether to align corners, default False
    
    Returns:
        a: returned unchanged
        b: aligned tensor with the same spatial dimensions as a
    """
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode=mode, align_corners=align_corners)
    return a, b


# ============================================================
# Similarity Loss Functions
# ============================================================

def ncc_loss(a, b, eps=1e-6):
    """
    Normalized cross-correlation loss.
    Its approximate range is [0,2], where 0 indicates perfect correlation and larger values indicate weaker correlation.
    """
    # Ensure matching spatial dimensions.
    a, b = align_spatial_size(a, b)

    a_mean = a.mean(dim=[2, 3], keepdim=True)
    b_mean = b.mean(dim=[2, 3], keepdim=True)

    a_centered = a - a_mean
    b_centered = b - b_mean

    num = (a_centered * b_centered).mean(dim=[2, 3], keepdim=True)
    den = torch.sqrt(a_centered.pow(2).mean(dim=[2, 3], keepdim=True) * b_centered.pow(2).mean(dim=[2, 3], keepdim=True) + eps)

    ncc = num / (den + eps)  # Values closer to 1 indicate greater consistency.
    # The loss is the mean of (1 - ncc).
    return (1.0 - ncc).mean()


def ssim_loss(a, b, C1=0.01 ** 2, C2=0.03 ** 2, eps=1e-6):
    """
    Simplified differentiable SSIM loss using global statistics:
    loss = 1 - SSIM(a, b), with an approximate range of [0,2].
    """
    # Ensure matching spatial dimensions.
    a, b = align_spatial_size(a, b)

    mu_a = a.mean(dim=[2, 3], keepdim=True)
    mu_b = b.mean(dim=[2, 3], keepdim=True)

    sigma_a = a.var(dim=[2, 3], keepdim=True, unbiased=False)
    sigma_b = b.var(dim=[2, 3], keepdim=True, unbiased=False)
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean(dim=[2, 3], keepdim=True)

    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2) + eps

    ssim = num / den
    return (1.0 - ssim).mean()


def contrast_loss(a, b, eps=1e-6):
    """
    Contrast loss that encourages similar overall image contrast, measured by standard deviation.
    """
    # Ensure matching spatial dimensions.
    a, b = align_spatial_size(a, b)

    std_a = a.std(dim=[2, 3], keepdim=True)
    std_b = b.std(dim=[2, 3], keepdim=True)

    return F.l1_loss(std_a, std_b)


# ============================================================
# Fusion Loss Functions
# ============================================================

def fusion_loss(fused, ir, pol):
    """
    Fusion loss design:
    - Emphasize human and target details in highlighted INF (ir) regions
    - Emphasize texture and edge details in POL (pol)
    - Construct adaptive spatial weights from gradients and intensities
    """
    # Match the spatial sizes of ir and pol to fused to avoid broadcasting or size mismatches.
    fused, ir = align_spatial_size(fused, ir)
    fused, pol = align_spatial_size(fused, pol)

    fx, fy = gradient(fused)
    ix, iy = gradient(ir)
    px, py = gradient(pol)

    # ---------------- INF: highlight-region weights, where people and targets usually appear ----------------
    # Normalize IR intensity to [0,1] as the highlight weight map.
    with torch.no_grad():
        ir_min = ir.amin(dim=[2, 3], keepdim=True)
        ir_max = ir.amax(dim=[2, 3], keepdim=True)
        ir_range = (ir_max - ir_min).clamp(min=1e-6)
        ir_norm = (ir - ir_min) / ir_range          # [0,1], with larger weights for brighter pixels

    # ---------------- POL: texture and edge weights ----------------
    # Construct the texture weight map from the POL gradient magnitude.
    with torch.no_grad():
        g_pol = torch.sqrt(px.pow(2) + py.pow(2) + 1e-6)
        g_min = g_pol.amin(dim=[2, 3], keepdim=True)
        g_max = g_pol.amax(dim=[2, 3], keepdim=True)
        g_range = (g_max - g_min).clamp(min=1e-6)
        g_pol_norm = (g_pol - g_min) / g_range      # [0,1], with larger weights for stronger textures

    # ---------------- Intensity term: balance IR highlights and POL texture while protecting IR details ----------------
    # Emphasize infrared in highlighted regions and polarization in textured regions.
    alpha_ir = 3.5   # Increase IR weight substantially to preserve highlight information.
    beta_pol = 2.0   # Moderately reduce POL weight to avoid suppressing IR details excessively.

    w_ir_int = 1.0 + alpha_ir * ir_norm  # Larger weights in highlighted regions
    w_pol_int = 1.0 + beta_pol * g_pol_norm  # Larger weights in textured regions

    L_intensity = torch.mean(w_ir_int * torch.abs(fused - ir) +
                             w_pol_int * torch.abs(fused - pol))

    # ---------------- Structure term: balance IR structure and POL texture while protecting IR structure ----------------
    # Emphasize IR structure in highlighted regions and POL structure in textured regions.
    gamma_ir = 2.5  # Increase the IR structure weight substantially to retain structural details.
    gamma_pol = 1.5  # Moderately reduce the POL structure weight.
    w_ir_str = 1.0 + gamma_ir * ir_norm  # Emphasize IR structure in highlighted regions.
    w_pol_str = 1.0 + gamma_pol * g_pol_norm  # Emphasize POL structure in textured regions.

    L_structure_ir = torch.mean(w_ir_str * (torch.abs(fx - ix) + torch.abs(fy - iy)))
    L_structure_pol = torch.mean(w_pol_str * (torch.abs(fx - px) + torch.abs(fy - py)))
    # Increase the IR structure-loss weight substantially to preserve IR details.
    L_structure = 1.8 * L_structure_ir + 1.0 * L_structure_pol

    # ---------------- Maximum-gradient constraint: preserve strong edges from max(ir, pol) ----------------
    max_x = torch.max(torch.abs(ix), torch.abs(px))
    max_y = torch.max(torch.abs(iy), torch.abs(py))
    L_max = F.l1_loss(torch.abs(fx), max_x) + F.l1_loss(torch.abs(fy), max_y)

    # ---------------- Brightness constraint: balance IR and POL brightness to avoid excessive darkness ----------------
    # Use a weighted average of IR and POL as the target brightness rather than POL alone.
    fused_mean = fused.mean()
    ir_mean = ir.mean()
    pol_mean = pol.mean()
    target_mean = 0.4 * ir_mean + 0.6 * pol_mean  # Balance IR and POL brightness.
    L_brightness = F.mse_loss(fused_mean, target_mean)  # Bring fusion brightness closer to the balanced target.

    return L_intensity + L_structure + L_max + 0.2 * L_brightness  # Reduce the brightness-constraint weight.


# ============================================================
# VGG Perceptual Loss
# ============================================================

class VGGPerceptualLoss(nn.Module):
    """
    VGG perceptual loss using pretrained VGG features to measure image similarity.
    It emphasizes semantic alignment rather than pixel alignment alone and can be
    used to compute fusion loss. The implementation follows CPMFusion's perceptual loss.
    """
    def __init__(self, layer_idx=8):  # layer_idx=8 corresponds to relu2_2.
        super().__init__()
        # Use the new weights API instead of the deprecated pretrained argument.
        try:
            from torchvision.models import VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:layer_idx].eval()
        except ImportError:
            # Support older torchvision versions.
            vgg = vgg16(pretrained=True).features[:layer_idx].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        # Keep the VGG model in float32 rather than half precision.
        vgg = vgg.float()
        self.vgg = vgg
    
    def forward(self, img1, img2):
        """
        Compute the perceptual loss between two images.
        
        Args:
            img1: (B, C, H, W) first image with values in [0,1]
            img2: (B, C, H, W) second image with values in [0,1]
        
        Returns:
            loss: scalar perceptual loss
        """
        # Disable autocast so VGG runs in float32 precision.
        # VGG parameters are float32, while autocast could produce incompatible float16 inputs.
        with torch.cuda.amp.autocast(enabled=False):
            # Convert to float32 to avoid type mismatches during mixed-precision training.
            img1 = img1.float()
            img2 = img2.float()
            
            # Ensure that the VGG model is on the same device as the inputs.
            device = img1.device
            if next(self.vgg.parameters()).device != device:
                self.vgg = self.vgg.to(device)
            
            # Expand single-channel inputs to three channels as required by VGG.
            if img1.shape[1] == 1:
                img1 = img1.repeat(1, 3, 1, 1)
            if img2.shape[1] == 1:
                img2 = img2.repeat(1, 3, 1, 1)
            
            # Match input image sizes to avoid mismatched VGG feature dimensions.
            # Use the larger dimensions as the target size.
            if img1.shape[-2:] != img2.shape[-2:]:
                target_size = (max(img1.shape[-2], img2.shape[-2]), 
                              max(img1.shape[-1], img2.shape[-1]))
                if img1.shape[-2:] != target_size:
                    img1 = F.interpolate(img1, size=target_size, mode='bilinear', align_corners=False)
                if img2.shape[-2:] != target_size:
                    img2 = F.interpolate(img2, size=target_size, mode='bilinear', align_corners=False)
            # Do not use align_spatial_size here because both inputs must align to the larger size, not the first tensor.
            
            # Normalize images from [0,1] to the ImageNet distribution required by VGG.
            def preprocess(x):
                return normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            feat1 = self.vgg(preprocess(img1))
            feat2 = self.vgg(preprocess(img2))
            
            # Recheck feature-map sizes in case VGG introduces an internal size difference.
            if feat1.shape != feat2.shape:
                # Interpolate the smaller feature map to the size of the larger one.
                if feat1.numel() < feat2.numel():
                    feat1 = F.interpolate(feat1, size=feat2.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    feat2 = F.interpolate(feat2, size=feat1.shape[-2:], mode='bilinear', align_corners=False)
            
            return F.l1_loss(feat1, feat2)


def compute_fusion_core_loss(fused, ir_reg, pol_reg):
    """
    Core fusion loss that evaluates basic fusion quality.
    It includes intensity, structure, gradient, and brightness losses.
    """
    # Ensure matching spatial dimensions.
    fused, ir_reg = align_spatial_size(fused, ir_reg)
    fused, pol_reg = align_spatial_size(fused, pol_reg)
    
    # Numerical stability: check inputs for NaN or Inf values.
    if torch.isnan(fused).any() or torch.isinf(fused).any():
        fused = torch.where(torch.isnan(fused) | torch.isinf(fused), 
                            torch.zeros_like(fused), fused)
    if torch.isnan(ir_reg).any() or torch.isinf(ir_reg).any():
        ir_reg = torch.where(torch.isnan(ir_reg) | torch.isinf(ir_reg), 
                            torch.zeros_like(ir_reg), ir_reg)
    if torch.isnan(pol_reg).any() or torch.isinf(pol_reg).any():
        pol_reg = torch.where(torch.isnan(pol_reg) | torch.isinf(pol_reg), 
                             torch.zeros_like(pol_reg), pol_reg)

    fx, fy = gradient(fused)
    ix, iy = gradient(ir_reg)
    px, py = gradient(pol_reg)

    # 1. Intensity loss: preserve IR and POL intensity information.
    L_intensity = 0.5 * F.l1_loss(fused, ir_reg) + 0.5 * F.l1_loss(fused, pol_reg)

    # 2. Structure loss: preserve IR and POL structural information.
    L_structure = 0.5 * (F.l1_loss(torch.abs(fx - ix) + torch.abs(fy - iy), torch.zeros_like(fx))) + \
                   0.5 * (F.l1_loss(torch.abs(fx - px) + torch.abs(fy - py), torch.zeros_like(fx)))

    # 3. Gradient loss: preserve maximum-gradient information.
    max_x = torch.max(torch.abs(ix), torch.abs(px))
    max_y = torch.max(torch.abs(iy), torch.abs(py))
    L_gradient = F.l1_loss(torch.abs(fx), max_x) + F.l1_loss(torch.abs(fy), max_y)

    # 4. Brightness loss: balance IR and POL brightness.
    fused_mean = fused.mean()
    ir_mean = ir_reg.mean()
    pol_mean = pol_reg.mean()
    target_mean = 0.4 * ir_mean + 0.6 * pol_mean
    L_brightness = F.mse_loss(fused_mean, target_mean)

    return L_intensity + L_structure + L_gradient + 0.2 * L_brightness


def compute_ir_bias_loss(fused, ir_reg, pol_reg):
    """
    Optimized IR bias loss that preserves more IR information in highlighted human regions.
    It reduces IR weights outside highlights so backgrounds come primarily from the polarization image.
    """
    # Ensure matching spatial dimensions.
    fused, ir_reg = align_spatial_size(fused, ir_reg)
    fused, pol_reg = align_spatial_size(fused, pol_reg)
    
    # Numerical stability: check inputs for NaN or Inf values.
    if torch.isnan(fused).any() or torch.isinf(fused).any():
        fused = torch.where(torch.isnan(fused) | torch.isinf(fused), 
                            torch.zeros_like(fused), fused)
    if torch.isnan(ir_reg).any() or torch.isinf(ir_reg).any():
        ir_reg = torch.where(torch.isnan(ir_reg) | torch.isinf(ir_reg), 
                            torch.zeros_like(ir_reg), ir_reg)
    if torch.isnan(pol_reg).any() or torch.isinf(pol_reg).any():
        pol_reg = torch.where(torch.isnan(pol_reg) | torch.isinf(pol_reg), 
                             torch.zeros_like(pol_reg), pol_reg)

    # 1. IR protection loss for true highlight regions using a stricter threshold.
    ir_reg_min = ir_reg.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    ir_reg_max = ir_reg.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    ir_reg_range = (ir_reg_max - ir_reg_min).clamp(min=1e-6)  # Ensure a nonzero range.
    ir_reg_norm = (ir_reg - ir_reg_min) / ir_reg_range
    
    # Use a stricter threshold of 0.8 to focus on true human highlights and avoid background false positives.
    # The higher threshold prevents small background heat sources from being misclassified.
    highlight_mask_strict = (ir_reg_norm > 0.8).float()  # A threshold of 0.8 identifies people more precisely.
    highlight_mask_medium = (ir_reg_norm > 0.6).float()  # Medium-highlight regions
    
    # Additional spatial-continuity filtering removes small isolated highlights that may be background heat sources.
    # Use a morphological operation to retain only large connected human regions.
    with torch.no_grad():
        # Apply morphological opening to remove small isolated points from the highlight mask.
        highlight_opened = F.avg_pool2d(highlight_mask_strict, kernel_size=7, stride=1, padding=3)
        highlight_opened = (highlight_opened > 0.3).float()
        highlight_opened = F.avg_pool2d(highlight_opened, kernel_size=7, stride=1, padding=3)
        highlight_opened = (highlight_opened > 0.3).float()
        highlight_mask_strict = highlight_mask_strict * highlight_opened  # Retain only large connected regions.
    
    # Highlight-region L1 intensity loss keeps highlighted human regions closer to IR.
    # Increase the weight to preserve IR information more strongly in these regions.
    highlight_mask_sum = highlight_mask_strict.sum()
    if highlight_mask_sum > 1e-6:
        L_highlight_strict = torch.mean(highlight_mask_strict * torch.abs(fused - ir_reg)) * 3.0  # Increased weight
    else:
        L_highlight_strict = torch.tensor(0.0, device=fused.device, requires_grad=True)
    
    # Highlight-region structure loss aligns gradients so structural details come from IR.
    fx, fy = gradient(fused)
    ix, iy = gradient(ir_reg)
    if highlight_mask_sum > 1e-6:
        L_highlight_structure = torch.mean(highlight_mask_strict * (
            torch.abs(fx - ix) + torch.abs(fy - iy)
        )) * 2.0  # Increased weight
    else:
        L_highlight_structure = torch.tensor(0.0, device=fused.device, requires_grad=True)
    
    # 2. Weighted overall IR preservation loss with larger highlight and smaller non-highlight weights.
    # Nonlinear weights increase substantially in highlighted regions and decrease elsewhere.
    # Raise IR weights strongly in highlights to preserve infrared information.
    ir_weight = 1.0 + 8.0 * highlight_mask_strict + 3.0 * (highlight_mask_medium - highlight_mask_strict)  # Increased weight
    # Reduce IR weights substantially in background regions to prevent infrared highlights from entering the background.
    non_highlight_mask = (ir_reg_norm < 0.5).float()  # Non-highlight background regions with threshold raised to 0.5
    ir_weight = ir_weight * (1.0 - 0.8 * non_highlight_mask)  # Increase background suppression from 0.6 to 0.8.
    # Numerical stability: ensure weights are neither NaN nor Inf.
    ir_weight = torch.clamp(ir_weight, min=0.0, max=100.0)  # Limit the weight range.
    L_ir_preserve = torch.mean(ir_weight * torch.abs(fused - ir_reg))
    
    # 3. Highlight-region density constraint preserves information density in highlighted regions.
    def local_variance(x, kernel_size=5):
        local_mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return local_var
    
    ir_density = local_variance(ir_reg)
    fused_density = local_variance(fused)
    # In highlighted regions, fusion density should remain close to infrared image density.
    L_density = torch.mean(highlight_mask_strict * torch.abs(fused_density - ir_density))
    
    # 4. Non-highlight backgrounds should be closer to polarization than infrared images.
    # The fused background should follow POL without introducing infrared highlights.
    # Increase the weight so backgrounds primarily come from POL.
    L_non_highlight_pol = torch.mean(non_highlight_mask * torch.abs(fused - pol_reg)) * 1.2  # Higher weight ensures POL-derived backgrounds.
    L_non_highlight_suppress = torch.mean(non_highlight_mask * torch.abs(fused - ir_reg)) * 0.5  # Moderate increase prevents IR leakage into backgrounds.
    
    # 5. Background structure-alignment loss ensures structural details come from the polarization image.
    px, py = gradient(pol_reg)
    L_background_structure = torch.mean(non_highlight_mask * (
        torch.abs(fx - px) + torch.abs(fy - py)
    )) * 0.6  # Ensure that background structure comes from POL.

    return (L_highlight_strict + 0.8 * L_highlight_structure + 
            L_ir_preserve + 0.5 * L_density + L_non_highlight_pol + L_non_highlight_suppress + L_background_structure)


class PolarTextureLoss(nn.Module):
    """
    Polarization texture-preservation loss module that stores the Laplacian kernel
    as a buffer to avoid recreating it.
    """
    def __init__(self):
        super().__init__()
        # Register the Laplacian kernel as a buffer to avoid recreating it on every forward pass.
        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
    
    def forward(self, fused, pol_reg):
        """
        Compute the enhanced polarization texture-preservation loss with special attention to dark-region details.
        
        Args:
            fused: fusion result (B, C, H, W)
            pol_reg: polarization image (B, C, H, W)
        
        Returns:
            loss: polarization texture-preservation loss
        """
        # Ensure matching spatial dimensions.
        fused, pol_reg = align_spatial_size(fused, pol_reg)
        
        # 0. Detect dark regions and emphasize detail preservation within them.
        with torch.no_grad():
            pol_min = pol_reg.amin(dim=[2, 3], keepdim=True)
            pol_max = pol_reg.amax(dim=[2, 3], keepdim=True)
            pol_range = (pol_max - pol_min).clamp(min=1e-6)
            pol_norm = (pol_reg - pol_min) / pol_range
            # Dark-region mask: larger values indicate darker low-brightness regions.
            dark_mask = 1.0 - pol_norm
            # Use a threshold to focus only on genuinely dark regions.
            dark_threshold = 0.4  # Regions below 40% brightness are considered dark.
            dark_mask = (dark_mask > dark_threshold).float()
        
        # 1. First-order gradient preservation for edge and texture intensity.
        fx, fy = gradient(fused)
        px, py = gradient(pol_reg)
        g_fused = torch.sqrt(fx.pow(2) + fy.pow(2) + 1e-6)
        g_pol = torch.sqrt(px.pow(2) + py.pow(2) + 1e-6)
        
        # Emphasize polarization texture preservation in high-gradient texture regions.
        # Add boundary checks for improved numerical stability.
        try:
            quantile_val = g_pol.quantile(0.3)
            # Check whether the quantile value is valid.
            if torch.isnan(quantile_val) or torch.isinf(quantile_val) or quantile_val <= 0:
                # Fall back to the median.
                quantile_val = g_pol.median()
                if torch.isnan(quantile_val) or torch.isinf(quantile_val) or quantile_val <= 0:
                    # Use the mean if the median is also invalid.
                    quantile_val = g_pol.mean()
                    if torch.isnan(quantile_val) or torch.isinf(quantile_val) or quantile_val <= 0:
                        # Use a fixed value if the mean is also invalid.
                        quantile_val = torch.tensor(1e-6, device=g_pol.device, dtype=g_pol.dtype)
        except Exception:
            # Use the median if quantile computation fails.
            quantile_val = g_pol.median()
            if torch.isnan(quantile_val) or torch.isinf(quantile_val) or quantile_val <= 0:
                quantile_val = g_pol.mean()
                if torch.isnan(quantile_val) or torch.isinf(quantile_val) or quantile_val <= 0:
                    quantile_val = torch.tensor(1e-6, device=g_pol.device, dtype=g_pol.dtype)
        
        # Ensure that quantile_val is a valid scalar.
        if quantile_val.dim() > 0:
            quantile_val = quantile_val.item() if quantile_val.numel() == 1 else quantile_val.mean()
        quantile_val = max(float(quantile_val), 1e-6)  # Ensure a minimum value of 1e-6.
        
        texture_mask = (g_pol > quantile_val).float()  # Texture-region mask
        # Increase weights where dark and textured regions overlap, while limiting enhancement to avoid whitening.
        combined_mask = texture_mask * (1.0 + 1.5 * dark_mask) + dark_mask * 1.0  # Lower weight coefficient
        L_gradient = torch.mean(combined_mask * torch.abs(g_fused - g_pol))
        
        # 2. Second-order Laplacian texture preservation emphasizes fine textures.
        # Use the registered buffer to avoid repeated creation.
        lap_fused = F.conv2d(fused, self.laplacian_kernel, padding=1)
        lap_pol = F.conv2d(pol_reg, self.laplacian_kernel, padding=1)
        # Emphasize Laplacian texture preservation in dark regions with a lower weight to avoid over-enhancement.
        laplacian_weight = 1.0 + 1.2 * dark_mask  # Lower weight coefficient
        L_laplacian = torch.mean(laplacian_weight * torch.abs(torch.abs(lap_fused) - torch.abs(lap_pol)))
    
        # 3. Texture orientation consistency through gradient-direction alignment.
        # Compute gradient directions.
        fused_dir = torch.atan2(fy, fx + 1e-6)
        pol_dir = torch.atan2(py, px + 1e-6)
        # Direction difference in strongly textured regions.
        dir_diff = torch.abs(fused_dir - pol_dir)
        dir_diff = torch.min(dir_diff, 2 * math.pi - dir_diff)  # Handle periodicity.
        # Emphasize orientation consistency in dark regions with a reduced weight.
        direction_weight = texture_mask * (1.0 + 1.2 * dark_mask)  # Lower weight coefficient
        L_direction = torch.mean(direction_weight * dir_diff)
        
        # 4. Preserve local contrast in textured regions.
        # Use local standard deviation as the contrast measure.
        def local_std(x, kernel_size=5):
            local_mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            return torch.sqrt(local_var + 1e-6)
        
        contrast_fused = local_std(fused)
        contrast_pol = local_std(pol_reg)
        # Emphasize contrast preservation in dark regions with a lower weight to avoid over-enhancement.
        contrast_weight = texture_mask * (1.0 + 1.5 * dark_mask)  # Lower weight coefficient
        L_contrast = torch.mean(contrast_weight * torch.abs(contrast_fused - contrast_pol))
        
        # 5. Preserve dark-region detail intensity with a lower weight to avoid whitening.
        # In dark regions, the fused result should preserve polarization detail intensity.
        dark_intensity_weight = dark_mask * 1.2  # Lower weight to avoid over-enhancement.
        L_dark_intensity = torch.mean(dark_intensity_weight * torch.abs(fused - pol_reg))
        
        # 6. Detail-region focus loss directs polarization information toward detail-rich background regions.
        # Use gradient magnitude as the detail-richness measure.
        detail_mask = (g_pol > g_pol.quantile(0.5)).float()  # Regions with gradient magnitude above the median
        # Keep fused detail regions closer to the polarization image with a lower weight to avoid over-enhancement.
        detail_weight = 1.0 + 1.8 * detail_mask  # Lower weight coefficient to avoid whitening.
        L_detail_focus = torch.mean(detail_weight * torch.abs(fused - pol_reg))
        
        # 7. Detail-region structure-alignment loss aligns gradient structure in detailed regions.
        L_detail_structure = torch.mean(detail_mask * (
            torch.abs(fx - px) + torch.abs(fy - py)
        ))
        
        # 8. Non-detail highlight regions should be closer to infrared than polarization images.
        # Reduce the influence of polarization information in low-detail human highlight regions.
        non_detail_mask = (g_pol < g_pol.quantile(0.3)).float()  # Low-detail regions, usually highlights
        # In non-detail regions, keep the fused result closer to the infrared image than the polarization image.
        L_non_detail_suppress = torch.mean(non_detail_mask * torch.abs(fused - pol_reg)) * 0.3  # Lower weight
        
        # Combined weighted loss with reduced component weights to avoid whitening from over-enhancement.
        return (0.25 * L_gradient + 0.18 * L_laplacian + 0.1 * L_direction + 
                0.08 * L_contrast + 0.1 * L_dark_intensity + 
                0.15 * L_detail_focus + 0.08 * L_detail_structure + 0.06 * L_non_detail_suppress)


# Create a global instance to avoid repeated construction.
_polar_texture_loss_module = None

def compute_polar_texture_preservation_loss(fused, pol_reg):
    """
    Polarization texture-preservation loss specialized for retaining texture details.
    It constrains multi-scale, multi-directional texture features.
    
    Note: PolarTextureLoss is used internally to avoid recreating the Laplacian kernel.
    """
    global _polar_texture_loss_module
    if _polar_texture_loss_module is None:
        _polar_texture_loss_module = PolarTextureLoss()
    
    # Ensure the module is on the correct device by checking its buffer, as it has no parameters.
    if _polar_texture_loss_module.laplacian_kernel.device != fused.device:
        _polar_texture_loss_module = _polar_texture_loss_module.to(fused.device)
    
    return _polar_texture_loss_module(fused, pol_reg)


# ============================================================
# Total Loss Function (Reorganized)
# ============================================================

def compute_intermediate_supervision_loss(outputs, ir, pol):
    """
    Intermediate supervision loss ensures that feature extraction focuses on the correct regions:
    1. Infrared features should focus on highlighted regions.
    2. Polarization features should focus on detailed regions.
    """
    # Retrieve intermediate supervision masks.
    ir_highlight_mask = outputs.get("ir_highlight_focus_mask", None)
    pol_detail_mask = outputs.get("pol_detail_focus_mask", None)
    
    if ir_highlight_mask is None or pol_detail_mask is None:
        # Return zero loss if intermediate supervision masks are unavailable.
        return torch.tensor(0.0, device=ir.device, requires_grad=True)
    
    # Align spatial dimensions.
    ir_highlight_mask, ir = align_spatial_size(ir_highlight_mask, ir)
    pol_detail_mask, pol = align_spatial_size(pol_detail_mask, pol)
    
    # 1. Infrared highlight-focus supervision loss.
    # Ensure a strong mask response in highlighted infrared regions.
    with torch.no_grad():
        ir_min = ir.amin(dim=[2, 3], keepdim=True)
        ir_max = ir.amax(dim=[2, 3], keepdim=True)
        ir_norm = (ir - ir_min) / (ir_max - ir_min + 1e-6)
        ir_highlight_gt = (ir_norm > 0.6).float()  # Ground-truth highlight regions
    
    # The highlight mask should respond strongly in ground-truth highlight regions.
    L_ir_focus = F.mse_loss(ir_highlight_mask, ir_highlight_gt)
    
    # 2. Polarization detail-focus supervision loss.
    # Ensure a strong detail-mask response in detailed polarization regions.
    with torch.no_grad():
        pol_dx, pol_dy = gradient(pol)
        pol_grad_mag = torch.sqrt(pol_dx.pow(2) + pol_dy.pow(2) + 1e-6)
        try:
            detail_threshold = pol_grad_mag.quantile(0.5)
            if torch.isnan(detail_threshold) or torch.isinf(detail_threshold) or detail_threshold <= 0:
                detail_threshold = pol_grad_mag.median()
                if torch.isnan(detail_threshold) or torch.isinf(detail_threshold) or detail_threshold <= 0:
                    detail_threshold = pol_grad_mag.mean()
                    if torch.isnan(detail_threshold) or torch.isinf(detail_threshold) or detail_threshold <= 0:
                        detail_threshold = torch.tensor(1e-6, device=pol_grad_mag.device, dtype=pol_grad_mag.dtype)
        except:
            detail_threshold = pol_grad_mag.median()
            if torch.isnan(detail_threshold) or torch.isinf(detail_threshold) or detail_threshold <= 0:
                detail_threshold = pol_grad_mag.mean()
                if torch.isnan(detail_threshold) or torch.isinf(detail_threshold) or detail_threshold <= 0:
                    detail_threshold = torch.tensor(1e-6, device=pol_grad_mag.device, dtype=pol_grad_mag.dtype)
        
        # Ensure that detail_threshold is a valid scalar.
        if detail_threshold.dim() > 0:
            detail_threshold = detail_threshold.item() if detail_threshold.numel() == 1 else detail_threshold.mean()
        detail_threshold = max(float(detail_threshold), 1e-6)  # Ensure a minimum value of 1e-6.
        pol_detail_gt = (pol_grad_mag > detail_threshold).float()  # Ground-truth detail regions
    
    # The detail mask should respond strongly in ground-truth detail regions.
    L_pol_focus = F.mse_loss(pol_detail_mask, pol_detail_gt)
    
    return L_ir_focus + L_pol_focus


def total_loss(outputs, ir, pol,
               lambda_fusion_core=1.0,
               # Reduce the IR bias-loss weight to limit infrared whitening in non-target regions such as pillars and walls.
               lambda_ir_bias=2.0,
               # Raise the polarization texture-loss weight slightly so backgrounds and structures follow POL and suppress pillar whitening.
               lambda_polar_texture=2.5,
               lambda_regularization=0.3,
               lambda_intermediate=1.0):  # Intermediate supervision-loss weight
    """
    Fusion-only loss function without registration:
    L_total = λ1 * L_fusion_core + λ2 * L_IR_bias + λ3 * L_polar_texture + λ4 * L_regularization
    
    Args:
        outputs: model output dictionary containing:
            - fusion: fusion result
            - ir_reg: original IR image for compatibility
            - pol_reg: original POL image for compatibility
        ir: original IR image
        pol: original POL image
        lambda_fusion_core: core fusion-loss weight, default 1.0
        lambda_ir_bias: IR bias-loss weight, default 1.5
        lambda_polar_texture: polarization texture-preservation weight, default 0.8
        lambda_regularization: regularization-loss weight, default 0.3
    
    Returns:
        L_total: total loss
        loss_dict: detailed loss dictionary
    """
    fuse = outputs["fusion"]
    ir_reg = outputs.get("ir_reg", ir)  # Compatibility: use the original IR image when registration is absent.
    pol_reg = outputs.get("pol_reg", pol)  # Compatibility: use the original POL image when registration is absent.

    # Match all input spatial dimensions to the fusion result.
    fuse, ir_reg = align_spatial_size(fuse, ir_reg)
    fuse, pol_reg = align_spatial_size(fuse, pol_reg)

    # 1. Fusion Core Loss (L_fusion_core)
    L_fusion_core = compute_fusion_core_loss(fuse, ir_reg, pol_reg)

    # 2. IR Bias Loss (L_IR_bias) preserves IR information in highlighted regions.
    L_ir_bias = compute_ir_bias_loss(fuse, ir_reg, pol_reg)

    # 3. Polar Texture Preservation Loss (L_polar_texture)
    L_polar_texture = compute_polar_texture_preservation_loss(fuse, pol_reg)

    # 4. Regularization Loss (L_regularization) uses only a simplified fusion-quality constraint.
    # Remove registration-related regularization and retain only the fusion-quality constraint.
    L_regularization = 0.0
    # Consistency constraint between the aligned fusion result and inputs.
    L_cons = (
        F.l1_loss(fuse, ir_reg) +
        F.l1_loss(fuse, pol_reg)
    )
    L_regularization = 0.5 * L_cons

    # 5. Intermediate Supervision Loss (L_intermediate)
    # Ensure that feature extraction focuses on the correct regions.
    L_intermediate = compute_intermediate_supervision_loss(outputs, ir, pol)

    # 6. Total Loss: L_total = λ1 * L_fusion_core + λ2 * L_IR_bias + λ3 * L_polar_texture + λ4 * L_regularization + λ5 * L_intermediate
    # Numerical stability: check each loss component for NaN or Inf values.
    loss_components = [
        lambda_fusion_core * L_fusion_core,
        lambda_ir_bias * L_ir_bias,
        lambda_polar_texture * L_polar_texture,
        lambda_regularization * L_regularization,
        lambda_intermediate * L_intermediate
    ]
    
    # Replace NaN and Inf values with zero.
    loss_components_clean = []
    for comp in loss_components:
        if torch.isnan(comp) or torch.isinf(comp):
            comp = torch.tensor(0.0, device=comp.device, requires_grad=True)
        loss_components_clean.append(comp)
    
    L_total = sum(loss_components_clean)
    
    # Final check: return a small nonzero loss if the total is still NaN or Inf.
    if torch.isnan(L_total) or torch.isinf(L_total):
        L_total = torch.tensor(1e-6, device=fuse.device, requires_grad=True)
    
    # Return the total loss and detailed loss dictionary for training monitoring.
    with torch.no_grad():
        loss_dict = {
            "total": float(L_total.item() if isinstance(L_total, torch.Tensor) else L_total),
            "L_fusion_core": float(L_fusion_core.item() if isinstance(L_fusion_core, torch.Tensor) else L_fusion_core),
            "L_ir_bias": float(L_ir_bias.item() if isinstance(L_ir_bias, torch.Tensor) else L_ir_bias),
            "L_polar_texture": float(L_polar_texture.item() if isinstance(L_polar_texture, torch.Tensor) else L_polar_texture),
            "L_regularization": float(L_regularization.item() if isinstance(L_regularization, torch.Tensor) else L_regularization),
            "L_intermediate": float(L_intermediate.item() if isinstance(L_intermediate, torch.Tensor) else L_intermediate),
            "weighted_L_fusion_core": float((lambda_fusion_core * L_fusion_core).item() if isinstance(L_fusion_core, torch.Tensor) else lambda_fusion_core * L_fusion_core),
            "weighted_L_ir_bias": float((lambda_ir_bias * L_ir_bias).item() if isinstance(L_ir_bias, torch.Tensor) else lambda_ir_bias * L_ir_bias),
            "weighted_L_polar_texture": float((lambda_polar_texture * L_polar_texture).item() if isinstance(L_polar_texture, torch.Tensor) else lambda_polar_texture * L_polar_texture),
            "weighted_L_regularization": float((lambda_regularization * L_regularization).item() if isinstance(L_regularization, torch.Tensor) else lambda_regularization * L_regularization),
            "weighted_L_intermediate": float((lambda_intermediate * L_intermediate).item() if isinstance(L_intermediate, torch.Tensor) else lambda_intermediate * L_intermediate),
        }
    
    return L_total, loss_dict
