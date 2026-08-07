import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba
from losses import total_loss
from utils import gradient

# ============================================================
# Enhanced Encoder (enhanced feature extraction)
# ============================================================

class EnhancedConvEncoder(nn.Module):
    """
    Enhanced feature extractor:
    - Multi-scale feature extraction that preserves features at different resolutions
    - Residual connections for improved gradient flow
    - An attention mechanism that emphasizes important features
    """
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        # First layer: initial feature extraction
        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )
        
        # Second layer: downsampling and feature enhancement
        self.c2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, 1, 1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True)
        )
        
        # Third layer: further downsampling and deep feature extraction
        self.c3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1),  # Additional layer for feature enhancement
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention: emphasize important feature channels
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base * 4, base * 4 // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4 // 4, base * 4, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        f1 = self.c1(x)  # (B, base, H, W)
        f2 = self.c2(f1)  # (B, base*2, H/2, W/2)
        f3 = self.c3(f2)  # (B, base*4, H/4, W/4)
        
        # Apply channel attention.
        attn = self.channel_attn(f3)
        f3 = f3 * attn
        
        return f1, f2, f3

# ============================================================
# Multi-Scale Feature Extractor
# ============================================================

class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extractor that extracts features at different scales for fusion.
    """
    def __init__(self, ch):
        super().__init__()
        # Multi-scale convolutions extract features at different scales.
        self.scale1 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(ch, ch, 5, 1, 2),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(ch, ch, 7, 1, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
        # Fuse multi-scale features.
        self.fusion = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1, 1, 0),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        fused = self.fusion(torch.cat([s1, s2, s3], dim=1))
        return fused


# ============================================================
# Mamba-based Cross-modal Fusion Block
# ============================================================

class MambaFusionBlock(nn.Module):
    """
    Use Mamba to model long-range dependencies in the joint space of the two
    modalities so that INF and POL details are fully exchanged before fusion.
    """
    def __init__(self, dim):
        super().__init__()
        # Concatenate the two modalities along the channel dimension, so d_model = 2 * dim.
        self.mamba = Mamba(
            d_model=dim * 2,
            d_state=dim * 2,
            d_conv=4,
            expand=2
        )

    def forward(self, f_ir, f_pol):
        """
        f_ir, f_pol: (B, C, H, W)
        Return two enhanced modal features, each containing information from the other modality.
        """
        B, C, H, W = f_ir.shape

        # If the spatial sizes differ, align f_pol with the size of f_ir.
        if f_pol.shape[2:] != (H, W):
            f_pol = F.interpolate(f_pol, size=(H, W), mode='bilinear', align_corners=False)

        x = torch.cat([f_ir, f_pol], dim=1)      # (B, 2C, H, W)
        seq = x.flatten(2).transpose(1, 2)       # (B, HW, 2C)
        # Process overly long sequences in chunks to avoid cuDNN errors.
        if seq.size(1) > 65536:  # Use chunks when the sequence length exceeds 65,536.
            chunk_size = 32768
            chunks = []
            for i in range(0, seq.size(1), chunk_size):
                chunk = seq[:, i:i+chunk_size, :]
                chunk_out = self.mamba(chunk)
                chunks.append(chunk_out)
            seq = torch.cat(chunks, dim=1)
        else:
            seq = self.mamba(seq)                    # (B, HW, 2C)
        x_enh = seq.transpose(1, 2).view(B, 2 * C, H, W)
        f_ir_enh, f_pol_enh = torch.chunk(x_enh, 2, dim=1)
        return f_ir_enh, f_pol_enh

# ============================================================
# IR Highlight Injection (human/highlight preservation)
# ============================================================

class IRHighlightInjector(nn.Module):
    """
    Inject highlighted regions of the infrared image, which usually correspond to
    people or targets, into the final fusion result using a soft mask:
      fused_out = (1 - m) * fused + m * ir
    The mask m is derived adaptively from IR intensity. The threshold and sharpness
    are validated fixed hyperparameters that avoid artifacts caused by hard thresholds.
    For the current scenes, a slightly higher threshold and lower base injection ratio
    reduce whitening near background objects such as pillars and walls.
    """
    def __init__(self, init_thresh=0.62, init_sharpness=15.0, blur_kernel=3, inject_ratio_base=0.6):
        super().__init__()
        # Raise the threshold slightly to 0.62 to reduce false highlights in medium-bright backgrounds.
        # These values are fixed for a run and persisted in the state dict as buffers.
        # Treating them as fixed removes the ambiguity between the architecture diagram
        # and the implementation, and makes sensitivity experiments directly reproducible.
        self.register_buffer("thresh", torch.tensor(float(init_thresh)))
        self.register_buffer("sharpness", torch.tensor(float(init_sharpness)))
        self.inject_ratio_base = float(inject_ratio_base)
        k = int(blur_kernel)
        k = max(1, k)
        # Smooth the mask slightly to suppress block boundaries, using a smaller kernel to avoid excessive blur.
        self.blur = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2) if k > 1 else nn.Identity()

    def forward(self, fused, ir):
        # Align spatial dimensions.
        if ir.shape[-2:] != fused.shape[-2:]:
            ir = F.interpolate(ir, size=fused.shape[-2:], mode='bilinear', align_corners=False)

        # Normalize each image to [0,1] to extract highlighted regions.
        with torch.no_grad():
            ir_min = ir.amin(dim=[2, 3], keepdim=True)
            ir_max = ir.amax(dim=[2, 3], keepdim=True)
            ir_norm = (ir - ir_min) / (ir_max - ir_min + 1e-6)

        # Fixed-parameter soft-threshold mask: raise the threshold slightly to exclude broad background areas.
        sharp = self.sharpness.clamp(1.0, 50.0)
        thr = self.thresh.clamp(0.5, 0.85)
        m = torch.sigmoid(sharp * (ir_norm - thr))
        m = self.blur(m)
        
        # Improved spatial-continuity filtering removes small isolated highlights more gently.
        # Use a smaller kernel and looser threshold to remove only very small isolated points.
        with torch.no_grad():
            # Binarize the mask with a threshold of 0.3 to retain more medium-bright regions.
            m_binary = (m > 0.3).float()
            # Apply a smaller 3x3 morphological opening to remove only very small isolated points.
            m_opened = F.avg_pool2d(m_binary, kernel_size=3, stride=1, padding=1)
            m_opened = (m_opened > 0.2).float()  # Lower the threshold to retain more regions.
            m_opened = F.avg_pool2d(m_opened, kernel_size=3, stride=1, padding=1)
            m_opened = (m_opened > 0.2).float()
            # Combine the morphologically processed mask with the original using a gentler strategy.
            # Suppress only very small isolated points while preserving medium-sized connected regions.
            m_filtered = m * (0.7 + 0.3 * m_opened)  # Combine gently to avoid excessive removal.
            m = m_filtered

        # Improved injection strategy: lower the base ratio slightly to reduce IR whitening in backgrounds.
        # Highlights (m>0.7) retain strong injection, while medium regions (0.3<m<0.7) receive less.
        inject_ratio_base = self.inject_ratio_base
        # Use a smoother mapping with stronger highlight injection and moderate injection elsewhere.
        # Combine linear and squared terms so medium regions still receive sufficient injection.
        m_linear = m  # Linear term ensures injection in medium regions.
        m_squared = m.pow(2)  # Squared term emphasizes highlighted regions.
        inject_ratio_adaptive = inject_ratio_base * (0.4 + 0.4 * m_linear + 0.2 * m_squared)  # Composite mapping better preserves medium-brightness features.
        
        # Background suppression applies only to truly dark regions (m<0.15), allowing medium brightness through.
        background_suppress = (m < 0.15).float()
        inject_ratio_adaptive = inject_ratio_adaptive * (1.0 - 0.8 * background_suppress)
        
        fused_out = fused * (1.0 - inject_ratio_adaptive * m) + ir * (inject_ratio_adaptive * m)
        return fused_out, m

# ============================================================
# Wavelet Attention Module (based on the STANet attention mechanism)
# ============================================================

class WaveletAttention(nn.Module):
    """
    Wavelet attention module based on STANet's spatiotemporal attention concept.
    It decomposes an image into frequency subbands with a wavelet transform and
    applies attention to each subband, making it especially suitable for extracting
    texture details from polarization images.
    """
    def __init__(self, in_ch=1, reduction=4):
        super().__init__()
        # Approximation (low-frequency) and detail (high-frequency) branches of the wavelet decomposition.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Channel attention learns the importance of different frequency subbands.
        # Keep at least one intermediate channel to avoid division-by-zero errors.
        mid_ch = max(1, in_ch // reduction)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention learns the importance of spatial locations.
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3),  # Input: concatenated average- and max-pooled features
            nn.Sigmoid()
        )
        
    def dwt_2d(self, x):
        """
        Simplified 2D discrete wavelet transform using separable Haar wavelets.
        Decompose the image into four subbands: LL (low frequency), LH (horizontal
        high frequency), HL (vertical high frequency), and HH (diagonal high frequency).
        This implementation prioritizes stability.
        """
        B, C, H, W = x.shape
        # Ensure that the dimensions are even.
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, H, W = x.shape
        
        # Normalized Haar wavelet low-pass and high-pass filters.
        h_low = torch.tensor([[1.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 1, 2) / 2.0
        h_high = torch.tensor([[1.0, -1.0]], device=x.device, dtype=x.dtype).view(1, 1, 1, 2) / 2.0
        
        # Horizontal wavelet decomposition.
        x_low_h = F.conv2d(x, h_low, padding=(0, 0), stride=(1, 2))  # (B, C, H, W/2)
        x_high_h = F.conv2d(x, h_high, padding=(0, 0), stride=(1, 2))  # (B, C, H, W/2)
        
        # Vertical wavelet decomposition.
        h_low_t = h_low.transpose(2, 3)  # (1, 1, 2, 1)
        h_high_t = h_high.transpose(2, 3)  # (1, 1, 2, 1)
        
        LL = F.conv2d(x_low_h, h_low_t, padding=(0, 0), stride=(2, 1))  # Low frequency (B, C, H/2, W/2)
        LH = F.conv2d(x_low_h, h_high_t, padding=(0, 0), stride=(2, 1))  # Horizontal high frequency
        HL = F.conv2d(x_high_h, h_low_t, padding=(0, 0), stride=(2, 1))  # Vertical high frequency
        HH = F.conv2d(x_high_h, h_high_t, padding=(0, 0), stride=(2, 1))  # Diagonal high frequency
        
        return LL, LH, HL, HH
    
    def idwt_2d(self, LL, LH, HL, HH):
        """
        Inverse wavelet transform for image reconstruction.
        Use transposed convolutions for reconstruction.
        """
        # Haar wavelet reconstruction filters, identical to the decomposition filters.
        g_low = torch.tensor([[1.0, 1.0]], device=LL.device, dtype=LL.dtype).view(1, 1, 1, 2)
        g_high = torch.tensor([[1.0, -1.0]], device=LL.device, dtype=LL.dtype).view(1, 1, 1, 2)
        
        # Vertical reconstruction.
        g_low_t = g_low.transpose(2, 3)  # (1, 1, 2, 1)
        g_high_t = g_high.transpose(2, 3)  # (1, 1, 2, 1)
        
        # Use transposed convolutions for upsampling and filtering.
        x_low_h = F.conv_transpose2d(LL, g_low_t, stride=(2, 1), padding=(0, 0), output_padding=(0, 0)) + \
                  F.conv_transpose2d(LH, g_high_t, stride=(2, 1), padding=(0, 0), output_padding=(0, 0))
        x_high_h = F.conv_transpose2d(HL, g_low_t, stride=(2, 1), padding=(0, 0), output_padding=(0, 0)) + \
                   F.conv_transpose2d(HH, g_high_t, stride=(2, 1), padding=(0, 0), output_padding=(0, 0))
        
        # Horizontal reconstruction.
        x = F.conv_transpose2d(x_low_h, g_low, stride=(1, 2), padding=(0, 0), output_padding=(0, 0)) + \
            F.conv_transpose2d(x_high_h, g_high, stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        
        return x
    
    def forward(self, x):
        """
        x: (B, C, H, W) input features
        Returns: enhanced features
        """
        B, C, H, W = x.shape
        
        # 1. Wavelet decomposition: split the image into four subbands.
        LL, LH, HL, HH = self.dwt_2d(x)
        
        # 2. Apply channel attention to each subband.
        # Each subband has C channels and receives channel attention independently.
        LL_avg = self.avg_pool(LL)
        LL_max = self.max_pool(LL)
        LL_att_weight = (self.channel_attention(LL_avg) + self.channel_attention(LL_max)) / 2.0
        LL_att = LL * LL_att_weight
        
        LH_avg = self.avg_pool(LH)
        LH_max = self.max_pool(LH)
        LH_att_weight = (self.channel_attention(LH_avg) + self.channel_attention(LH_max)) / 2.0
        LH_att = LH * LH_att_weight
        
        HL_avg = self.avg_pool(HL)
        HL_max = self.max_pool(HL)
        HL_att_weight = (self.channel_attention(HL_avg) + self.channel_attention(HL_max)) / 2.0
        HL_att = HL * HL_att_weight
        
        HH_avg = self.avg_pool(HH)
        HH_max = self.max_pool(HH)
        HH_att_weight = (self.channel_attention(HH_avg) + self.channel_attention(HH_max)) / 2.0
        HH_att = HH * HH_att_weight
        
        # 3. Apply spatial attention to high-frequency subbands (LH, HL, HH) to emphasize texture regions.
        # Concatenate the high-frequency subbands.
        high_freq = torch.cat([LH_att, HL_att, HH_att], dim=1)  # (B, 3C, H/2, W/2)
        # Spatial attention.
        avg_out_spatial = torch.mean(high_freq, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(high_freq, dim=1, keepdim=True)
        spatial_att_input = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = self.spatial_attention(spatial_att_input)
        # Apply spatial attention only to the high-frequency subbands.
        LH_att = LH_att * spatial_att
        HL_att = HL_att * spatial_att
        HH_att = HH_att * spatial_att
        
        # 4. Reconstruct with the inverse wavelet transform.
        x_enhanced = self.idwt_2d(LL_att, LH_att, HL_att, HH_att)
        
        # 5. Match the input size because the wavelet transform may introduce a one-pixel difference.
        if x_enhanced.shape[-2:] != (H, W):
            x_enhanced = F.interpolate(x_enhanced, size=(H, W), mode='bilinear', align_corners=False)
        
        return x_enhanced

# ============================================================
# Polar Texture Enhancer: multi-scale polarization texture enhancement with wavelet attention
# ============================================================

class PolarTextureEnhancer(nn.Module):
    """
    Enhanced polarization texture-detail extraction module:
      - Multi-scale texture extraction with fine- and coarse-scale convolutions
      - Multi-directional gradient extraction using first-order, second-order, and Laplacian operators
      - Texture orientation analysis using gradient orientation histograms
      - Adaptive CLAHE-style contrast enhancement
      - Frequency-domain texture enhancement through high-frequency component extraction
      - Adaptive weighted fusion that emphasizes polarization texture outside highlighted regions
    """
    def __init__(self, mid_ch=16):
        super().__init__()
        # Wavelet attention module: extract texture details in the frequency domain.
        self.wavelet_attention = WaveletAttention(in_ch=1, reduction=4)
        
        # Multi-scale texture extraction.
        self.conv_pol_fine = nn.Conv2d(1, mid_ch, 3, 1, 1)      # Fine-scale texture
        self.conv_pol_coarse = nn.Conv2d(1, mid_ch, 5, 1, 2)   # Coarse-scale texture
        self.conv_pol_medium = nn.Conv2d(1, mid_ch, 7, 1, 3)    # Medium-scale texture
        
        # Feature extraction after wavelet enhancement in the wavelet domain.
        self.wavelet_conv = nn.Conv2d(1, mid_ch, 3, 1, 1)
        
        # Laplacian (second-order gradient) extraction.
        self.laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Texture orientation analysis using a learnable directional filter.
        self.direction_conv = nn.Conv2d(1, mid_ch // 2, 3, 1, 1)
        
        # Lightweight adaptive contrast enhancement.
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(1, mid_ch // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 4, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # Enhanced texture fusion network combining additional features, including wavelet features.
        # Input: fine+coarse+medium scales (3*mid_ch) + wavelet features (mid_ch) + gradient magnitude (1) + Laplacian (1) + directional features (mid_ch//2) + contrast enhancement (1)
        fusion_in_ch = mid_ch * 4 + 1 + 1 + mid_ch // 2 + 1
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_ch, mid_ch * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch * 2, mid_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1)
        )
        
    def compute_laplacian(self, x):
        """Compute the Laplacian operator (second-order gradient)."""
        if self.laplacian_kernel.device != x.device:
            self.laplacian_kernel = self.laplacian_kernel.to(x.device)
        laplacian = F.conv2d(x, self.laplacian_kernel, padding=1)
        return torch.abs(laplacian)  # Use the absolute value to emphasize edges.
        
    def compute_texture_direction(self, pol):
        """Compute texture orientation from gradient direction."""
        pol_dx, pol_dy = gradient(pol)
        # Compute the gradient direction as an angle.
        direction = torch.atan2(pol_dy, pol_dx + 1e-6)  # [-π, π]
        # Normalize to [0,1].
        direction_norm = (direction + math.pi) / (2 * math.pi)
        # Extract directional features with a learnable convolution.
        direction_feat = F.relu(self.direction_conv(direction_norm), inplace=True)
        return direction_feat
        
    def adaptive_contrast_enhance(self, pol):
        """Perform adaptive CLAHE-like contrast enhancement."""
        # Local contrast-enhancement weight.
        contrast_weight = self.contrast_enhancer(pol)
        # Local mean used for adaptive enhancement.
        local_mean = F.avg_pool2d(pol, kernel_size=5, stride=1, padding=2)
        # Apply stronger adaptive enhancement in low-contrast regions.
        enhanced = pol + contrast_weight * (pol - local_mean) * 0.5
        return enhanced
        
    def forward(self, fused, pol, ir_highlight_mask):
        """
        fused: current fusion result
        pol: polarization image
        ir_highlight_mask: optional infrared highlight mask used to protect highlighted regions
        """
        H, W = fused.shape[-2], fused.shape[-1]
        if pol.shape[-2:] != (H, W):
            pol = F.interpolate(pol, size=(H, W), mode='bilinear', align_corners=False)
        
        # 0. Wavelet attention enhancement extracts texture details in the frequency domain.
        pol_wavelet = self.wavelet_attention(pol)  # Wavelet-domain enhancement
        pol_wavelet_feat = F.relu(self.wavelet_conv(pol_wavelet), inplace=True)  # Extract wavelet features
        
        # 1. Multi-scale texture extraction at fine, medium, and coarse scales.
        pol_fine = F.relu(self.conv_pol_fine(pol), inplace=True)
        pol_medium = F.relu(self.conv_pol_medium(pol), inplace=True)
        pol_coarse = F.relu(self.conv_pol_coarse(pol), inplace=True)
        
        # 2. First-order gradients for edge and texture intensity.
        pol_dx, pol_dy = gradient(pol)
        pol_edge = torch.sqrt(pol_dx.pow(2) + pol_dy.pow(2) + 1e-6)
        
        # 3. Second-order Laplacian gradients emphasize finer textures.
        pol_laplacian = self.compute_laplacian(pol)
        
        # 4. Texture orientation analysis.
        pol_direction = self.compute_texture_direction(pol)
        
        # 5. Adaptive contrast enhancement.
        pol_enhanced = self.adaptive_contrast_enhance(pol)
        
        # 7. Fuse all texture features, including wavelet features.
        x = torch.cat([
            pol_fine, pol_medium, pol_coarse,  # Multi-scale features
            pol_wavelet_feat,                  # Wavelet-domain features
            pol_edge,                         # First-order gradient
            pol_laplacian,                    # Second-order Laplacian gradient
            pol_direction,                    # Directional features
            pol_enhanced                      # Contrast enhancement
        ], dim=1)
        
        texture_residual = self.fusion_conv(x)
        
        # 8. Adaptively enhance polarization texture residuals outside highlights while limiting whitening.
        if ir_highlight_mask is not None:
            # Enhance texture residuals outside highlights with a conservative coefficient to avoid whitening.
            # Apply more enhancement to background regions while reducing the overall enhancement strength.
            texture_scale = 1.5 + 0.8 * (1.0 - ir_highlight_mask)  # Substantially lower the coefficient to avoid whitening.
        else:
            texture_scale = 1.8  # Lower the default enhancement ratio to avoid over-enhancement.
        
        texture_residual = texture_scale * texture_residual
        
        # 9. Residual injection adds polarization texture details to the original fusion result.
        # Reduce texture residuals in highlighted human regions to avoid interfering with IR details.
        # Use polarization texture fully in background regions without introducing infrared highlights.
        if ir_highlight_mask is not None:
            # Strongly reduce texture residuals in highlights (m>0.5) and use polarization texture in backgrounds (m<0.2).
            highlight_suppress = (ir_highlight_mask > 0.5).float()
            background_enhance = (ir_highlight_mask < 0.2).float()
            texture_residual = texture_residual * (1.0 - 0.8 * highlight_suppress)  # Strongly reduce texture residuals in highlighted regions.
            texture_residual = texture_residual * (1.0 + 0.3 * background_enhance)  # Enhance polarization texture in background regions.
        
        # 10. Noise suppression: smooth texture residuals slightly to avoid introducing noise.
        texture_residual = F.avg_pool2d(texture_residual, kernel_size=3, stride=1, padding=1) * 0.7 + texture_residual * 0.3
        
        return fused + texture_residual

# ============================================================
# Task 2: Fusion Head (Explainable Polar Attention)
# ============================================================

class PolarFusionAttention(nn.Module):
    """
    Mamba-SSM-based intelligent fusion attention module:
    - Use Mamba-SSM to model spatial dependencies and locate highlight and texture regions
    - Assign high IR weights to highlighted regions such as people and heat sources
    - Assign high POL weights to complex textures such as leaves, railings, and fine details
    - Use Mamba's long-range dependency modeling to guide comprehensive, high-quality fusion
    """
    def __init__(self, ch):
        super().__init__()
        # IR feature branch specialized for infrared highlight information.
        self.ir_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),  # Additional layer improves highlight feature extraction.
            nn.ReLU(inplace=True)
        )
        # POL feature branch specialized for polarization texture information.
        self.pol_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),  # Additional layer improves texture feature extraction.
            nn.ReLU(inplace=True)
        )
        
        # Mamba-SSM models spatial dependencies to locate highlight and texture regions.
        # Reduce d_state to lower memory use and computational complexity.
        self.mamba_attn = Mamba(
            d_model=ch * 2,  # Input consists of concatenated IR and POL features.
            d_state=ch // 2,  # Lower state dimension reduces memory use.
            d_conv=4,
            expand=2
        )
        
        # Highlight detection head extracts highlight features from the original IR image.
        self.highlight_head = nn.Sequential(
            nn.Conv2d(1, ch, 3, 1, 1),  # Input: single-channel original image
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch // 2, 3, 1, 1),  # Additional intermediate layer
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 1, 1)  # Output: highlight map
        )
        
        # Texture-complexity detection head extracts texture features from the original POL image.
        self.texture_head = nn.Sequential(
            nn.Conv2d(1, ch, 3, 1, 1),  # Input: single-channel original image
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch // 2, 3, 1, 1),  # Additional intermediate layer
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 1, 1)  # Output: texture-complexity map
        )
        
        # Adaptive weight-generation network uses highlight and texture information to produce fusion weights.
        self.weight_gen = nn.Sequential(
            nn.Conv2d(ch * 2 + 2, ch, 3, 1, 1),  # Input: IR+POL features, highlight map, and texture map
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch // 2, 3, 1, 1),  # Additional intermediate layer improves weight generation.
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 2, 1)  # Output: IR and POL weights
        )
        
        # Polarization texture-enhancement branch for lightweight texture extraction.
        self.pol_texture = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, ir, pol, ir_img, pol_img):
        """
        Input order is fixed: ir first, followed by pol.
        Mamba-SSM locates highlight and texture regions and dynamically adjusts fusion weights.
        Args:
            ir: (B, C, H, W) IR features
            pol: (B, C, H, W) POL features
            ir_img: (B, 1, H, W) optional original IR image
            pol_img: (B, 1, H, W) optional original POL image
        """
        # Process IR and POL features separately.
        ir_processed = self.ir_branch(ir)  # IR branch processes highlight information.
        pol_processed = self.pol_branch(pol)  # POL branch processes texture information.
        
        # Concatenate IR and POL features.
        combined = torch.cat([ir_processed, pol_processed], dim=1)  # [B, 2C, H, W]
        
        # Use Mamba-SSM to model spatial dependencies and locate highlight and texture regions.
        B, C2, H, W = combined.shape
        # Downsample large spatial inputs to shorten the Mamba sequence and avoid cuDNN errors.
        # Use a lower threshold so downsampling occurs earlier.
        if H * W > 128 * 128:  # Downsample inputs larger than 128x128.
            combined_small = F.interpolate(combined, size=(H//2, W//2), mode='bilinear', align_corners=False)
            seq = combined_small.flatten(2).transpose(1, 2)  # [B, HW/4, 2C]
            # Process the sequence in chunks if it remains too long.
            if seq.size(1) > 65536:
                chunk_size = 32768
                chunks = []
                for i in range(0, seq.size(1), chunk_size):
                    chunk = seq[:, i:i+chunk_size, :]
                    chunk_out = self.mamba_attn(chunk)
                    chunks.append(chunk_out)
                seq_enhanced = torch.cat(chunks, dim=1)
            else:
                seq_enhanced = self.mamba_attn(seq)  # Mamba models long-range dependencies.
            combined_enhanced_small = seq_enhanced.transpose(1, 2).view(B, C2, H//2, W//2)  # [B, 2C, H/2, W/2]
            combined_enhanced = F.interpolate(combined_enhanced_small, size=(H, W), mode='bilinear', align_corners=False)
        else:
            seq = combined.flatten(2).transpose(1, 2)  # [B, HW, 2C]
            # Process the sequence in chunks if it is too long.
            if seq.size(1) > 65536:
                chunk_size = 32768
                chunks = []
                for i in range(0, seq.size(1), chunk_size):
                    chunk = seq[:, i:i+chunk_size, :]
                    chunk_out = self.mamba_attn(chunk)
                    chunks.append(chunk_out)
                seq_enhanced = torch.cat(chunks, dim=1)
            else:
                seq_enhanced = self.mamba_attn(seq)  # Mamba models long-range dependencies.
            combined_enhanced = seq_enhanced.transpose(1, 2).view(B, C2, H, W)  # [B, 2C, H, W]
        
        # Extract a highlight map from the original IR image.
        if ir_img is None:
            # If the original image is unavailable, use the mean IR feature as a substitute.
            ir_img = ir.mean(dim=1, keepdim=True)  # [B, 1, H_ir, W_ir]
        # Ensure that the spatial size of ir_img matches combined_enhanced.
        if ir_img.shape[2:] != (H, W):
            ir_img = F.interpolate(ir_img, size=(H, W), mode='bilinear', align_corners=False)
        highlight_map = torch.sigmoid(self.highlight_head(ir_img))  # [B, 1, H, W]
        
        # Extract a texture-complexity map from the original POL image.
        if pol_img is None:
            # If the original image is unavailable, use the mean POL feature as a substitute.
            pol_img = pol.mean(dim=1, keepdim=True)  # [B, 1, H_pol, W_pol]
        # Ensure that the spatial size of pol_img matches combined_enhanced.
        if pol_img.shape[2:] != (H, W):
            pol_img = F.interpolate(pol_img, size=(H, W), mode='bilinear', align_corners=False)
        texture_map = torch.sigmoid(self.texture_head(pol_img))  # [B, 1, H, W]
        
        # Generate adaptive fusion weights from highlight and texture information.
        # Highlight regions receive high IR weights; texture regions receive high POL weights.
        weight_input = torch.cat([combined_enhanced, highlight_map, texture_map], dim=1)  # [B, 2C+2, H, W]
        w_raw = self.weight_gen(weight_input)  # [B, 2, H, W]
        
        
        w = torch.softmax(w_raw, dim=1)  # [B, 2, H, W]
        
        # Enhance polarization features by extracting texture information.
        pol_enhanced =self.pol_texture(pol_processed)  # Increase texture-enhancement strength.
        
        # Intelligent fusion uses weights guided by Mamba-SSM.
        # w[:, 0:1] contains IR weights, and w[:, 1:2] contains POL weights.
        fused = w[:, 0:1] * ir_processed + w[:, 1:2] * pol_enhanced
        
        return fused, w

# ============================================================
# Retinex Decomposition Fusion
# ============================================================

class FusionHead(nn.Module):
    def __init__(self, ch, use_retinex=False, use_paf=True):
        super().__init__()
        self.use_retinex = use_retinex
        self.use_paf = bool(use_paf)
        
        # Mamba attention fusion module.
        self.att = PolarFusionAttention(ch) if self.use_paf else None
        
        self.out = nn.Conv2d(ch, 1, 3, 1, 1)
        


    def forward(self, ir, pol, ir_img, pol_img):
        """
        Args:
            ir: (B, C, H, W) IR features
            pol: (B, C, H, W) POL features
            ir_img: (B, 1, H, W) optional original IR image for Retinex
            pol_img: (B, 1, H, W) optional original POL image for Retinex
        Returns:
            fused: (B, 1, H, W) fusion result
            w: (B, 2, H, W) fusion weights, where w[:, 0:1] is IR and w[:, 1:2] is POL
        """
        if self.use_paf:
            # Use only Mamba attention fusion and pass original images to extract highlight and texture maps.
            fused, w = self.att(ir, pol, ir_img, pol_img)
        else:
            fused = 0.5 * (ir + pol)
            B, _, H, W = fused.shape
            w = fused.new_full((B, 2, H, W), 0.5)
        
        # Produce the fusion result.
        base_fused = self.out(fused)
        
        
        fused = base_fused
        
        return fused, w

# ============================================================
# Full Model
# ============================================================
class PolarIRS4FusionMamba(nn.Module):
    """
    Fusion-only model without registration:
    - Enhanced feature extraction uses an improved encoder for multi-scale features
    - Optimized fusion uses Mamba-SSM to model long-range dependencies and fuse IR and POL features intelligently
    """
    def __init__(
        self,
        use_checkpoint: bool = True,
        save_intermediate: bool = False,
        ihj_thresh: float = 0.62,
        ihj_sharpness: float = 15.0,
        ihj_inject_ratio: float = 0.6,
        use_multiscale: bool = True,
        use_cross_mamba: bool = True,
        use_paf: bool = True,
        use_ihj: bool = True,
        use_ptj: bool = True,
    ):
        super().__init__()
        self.use_multiscale = bool(use_multiscale)
        self.use_cross_mamba = bool(use_cross_mamba)
        self.use_ihj = bool(use_ihj)
        self.use_ptj = bool(use_ptj)

        # Enhanced feature extractor.
        self.encoder = EnhancedConvEncoder()
        
        # Multi-scale feature extractors for further feature enhancement.
        self.multiscale_extractor_ir = MultiScaleFeatureExtractor(128) if self.use_multiscale else nn.Identity()
        self.multiscale_extractor_pol = MultiScaleFeatureExtractor(128) if self.use_multiscale else nn.Identity()

        # Mamba-based cross-modal fusion module for deep interaction at the feature level.
        self.fusion_mamba = MambaFusionBlock(128) if self.use_cross_mamba else None
        
        # Fusion head with Retinex enhancement.
        self.fuse_head = FusionHead(128, use_retinex=True, use_paf=use_paf)
        
        # Infrared highlight injection makes people and highlighted targets more prominent in the final result.
        self.ir_inject = (
            IRHighlightInjector(
                init_thresh=ihj_thresh,
                init_sharpness=ihj_sharpness,
                inject_ratio_base=ihj_inject_ratio,
            )
            if self.use_ihj
            else None
        )
        
        # Polarization texture-enhancement module specialized for texture details.
        self.polar_texture_enhancer = PolarTextureEnhancer() if self.use_ptj else None

        # Whether to save intermediate results.
        self.save_intermediate = save_intermediate

        # Reduce memory use by optionally applying gradient checkpointing to expensive blocks.
        self.use_checkpoint = use_checkpoint

    def forward(self, ir, pol):
        """
        Fusion-only forward pass without registration:
        1. Enhanced feature extraction
        2. Multi-scale feature enhancement
        3. Cross-modal Mamba interaction
        4. Intelligent fusion
        5. Post-processing enhancement
        """
        # ========== Stage 1: Enhanced feature extraction ==========
        # Extract multi-scale features.
        f1_ir, f2_ir, f3_ir = self.encoder(ir)
        f1_pol, f2_pol, f3_pol = self.encoder(pol)
        
        # Use the highest-level features (f3) for fusion.
        f_ir = f3_ir  # (B, 128, H/4, W/4)
        f_pol = f3_pol  # (B, 128, H/4, W/4)

        # ========== Stage 2: Multi-scale feature enhancement ==========
        # Apply multi-scale enhancement to IR and POL features.
        f_ir = self.multiscale_extractor_ir(f_ir)
        f_pol = self.multiscale_extractor_pol(f_pol)

        # ========== Stage 4: Cross-modal Mamba fusion ==========
        # Perform deep interaction at the feature level.
        if self.use_cross_mamba and self.use_checkpoint:
            f_ir_fused, f_pol_fused = checkpoint(
                lambda a, b: self.fusion_mamba(a, b), f_ir, f_pol
            )
        elif self.use_cross_mamba:
            f_ir_fused, f_pol_fused = self.fusion_mamba(f_ir, f_pol)
        else:
            f_ir_fused, f_pol_fused = f_ir, f_pol

        # ========== Stage 5: Upsample to the input resolution ==========
        H, W = ir.shape[-2], ir.shape[-1]
        f_ir_fused = F.interpolate(f_ir_fused, size=(H, W), mode='bilinear', align_corners=False)
        f_pol_fused = F.interpolate(f_pol_fused, size=(H, W), mode='bilinear', align_corners=False)

        # ========== Stage 6: Intelligent fusion ==========
        # Use the fusion head and pass the original images for Retinex processing.
        if self.use_checkpoint:
            fuse_feat, w = checkpoint(
                lambda a, b, c, d: self.fuse_head(a, b, c, d), 
                f_ir_fused, f_pol_fused, ir, pol
            )
        else:
            fuse_feat, w = self.fuse_head(f_ir_fused, f_pol_fused, ir, pol)
        
        # ========== Stage 7: Post-processing enhancement ==========
        # Inject infrared highlights to emphasize people and highlighted targets.
        if self.use_ihj:
            fuse_feat, m_ir = self.ir_inject(fuse_feat, ir)
        else:
            m_ir = None

        # Enhance polarization texture outside highlighted regions.
        if self.use_ptj:
            fuse_feat = self.polar_texture_enhancer(fuse_feat, pol, m_ir)


        # Brightness normalization balances IR and POL brightness to prevent an overly dark result.
        ir_mean = ir.mean(dim=[2, 3], keepdim=True).detach()
        pol_mean = pol.mean(dim=[2, 3], keepdim=True).detach()
        ir_std = ir.std(dim=[2, 3], keepdim=True).detach()
        pol_std = pol.std(dim=[2, 3], keepdim=True).detach()

        if m_ir is not None:
            # Stay closer to IR in highlighted human regions and closer to POL in backgrounds.
            target_mean = m_ir * (0.6 * ir_mean + 0.4 * pol_mean) + (1.0 - m_ir) * (0.2 * ir_mean + 0.8 * pol_mean)
            target_std = m_ir * (0.6 * ir_std + 0.4 * pol_std) + (1.0 - m_ir) * (0.2 * ir_std + 0.8 * pol_std)
        else:
            # Without a highlight mask, use balanced weights with backgrounds closer to POL.
            target_mean = 0.3 * ir_mean + 0.7 * pol_mean  # Keep the background closer to POL.
            target_std = 0.3 * ir_std + 0.7 * pol_std
        
        fuse_mean = fuse_feat.mean(dim=[2, 3], keepdim=True)
        fuse_std = fuse_feat.std(dim=[2, 3], keepdim=True)
        
        # Normalize to the target mean and standard deviation while preserving relative brightness.
        fuse_feat_normalized = (fuse_feat - fuse_mean) / (fuse_std + 1e-6) * target_std + target_mean
        
        # If intermediate-result saving is enabled, store values before and after normalization.
        if self.save_intermediate:
            # Save pre-normalization fuse_mean and fuse_std and post-normalization fuse_feat.
            if not hasattr(self, '_intermediate_buffer'):
                self._intermediate_buffer = {}
            self._intermediate_buffer['polar_texture_enhancer'] = {
                "fuse_mean": fuse_mean.detach().clone(),
                "fuse_std": fuse_std.detach().clone(),
                "fuse_feat": fuse_feat_normalized.detach().clone()
            }
        
        fuse_feat = fuse_feat_normalized
        
        # Clamp to a reasonable range to avoid excessive brightness or darkness without blocking gradients.
        fuse_feat = torch.clamp(fuse_feat, 0.0, 1.0)
        
        # Numerical stability: detect and repair NaN and Inf values.
        if torch.isnan(fuse_feat).any() or torch.isinf(fuse_feat).any():
            # Use a weighted average of IR and POL as the fallback.
            # Ensure that the spatial sizes of ir and pol match fuse_feat.
            target_size = fuse_feat.shape[-2:]  # (H, W)
            
            # Interpolate ir if its spatial size does not match.
            if ir.shape[-2:] != target_size:
                ir = F.interpolate(ir, size=target_size, mode='bilinear', align_corners=False)
            
            # Interpolate pol if its spatial size does not match.
            if pol.shape[-2:] != target_size:
                pol = F.interpolate(pol, size=target_size, mode='bilinear', align_corners=False)
            
            # Ensure that ir and pol have matching spatial sizes.
            if ir.shape[-2:] != pol.shape[-2:]:
                # Use the size of fuse_feat as the reference.
                if ir.shape[-2:] == target_size:
                    pol = F.interpolate(pol, size=target_size, mode='bilinear', align_corners=False)
                elif pol.shape[-2:] == target_size:
                    ir = F.interpolate(ir, size=target_size, mode='bilinear', align_corners=False)
                else:
                    # If neither matches, resize both to the size of fuse_feat.
                    ir = F.interpolate(ir, size=target_size, mode='bilinear', align_corners=False)
                    pol = F.interpolate(pol, size=target_size, mode='bilinear', align_corners=False)
            
            ir_weight_fallback = 0.5
            pol_weight_fallback = 0.5
            fallback = ir_weight_fallback * ir + pol_weight_fallback * pol
            fuse_feat = torch.where(torch.isnan(fuse_feat) | torch.isinf(fuse_feat), 
                                    fallback, fuse_feat)
            fuse_feat = torch.clamp(fuse_feat, 0.0, 1.0)

        
        # ========== Stage 8: Build the output dictionary ==========
        output_dict = {
            "fusion": fuse_feat,  # Fusion result
            "ir_reg": ir,  # Original IR image for loss-computation compatibility
            "pol_reg": pol,  # Original POL image for loss-computation compatibility
            "attn": w,  # Fusion attention weights: w[:, 0:1] for IR and w[:, 1:2] for POL
            "ir_highlight_mask": m_ir  # Infrared highlight-injection mask for visualization and debugging
        }
        
        # Add intermediate results when intermediate-result saving is enabled.
        if self.save_intermediate:
            intermediate_results = {}
            
            # Retrieve saved intermediate results from the buffer.
            if hasattr(self, '_intermediate_buffer') and 'polar_texture_enhancer' in self._intermediate_buffer:
                intermediate_results["polar_texture_enhancer"] = self._intermediate_buffer['polar_texture_enhancer']
            
            output_dict["intermediate_results"] = intermediate_results
        
        return output_dict

# ============================================================
# Debug
# ============================================================

if __name__ == "__main__":
    model = PolarIRS4FusionMamba()
    ir = torch.randn(1, 1, 256, 256)
    pol = torch.randn(1, 1, 256, 256)
    out = model(ir, pol)
    loss, loss_dict = total_loss(out, ir, pol)
    print("OK | Total Loss:", loss.item())
    print("Loss Components:", loss_dict)
#best model
