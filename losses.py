"""
损失函数模块（纯融合版本）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.transforms.functional import normalize

# 导入工具函数（仅需梯度）
from utils import gradient


# ============================================================
# Utility Functions (工具函数)
# ============================================================

def align_spatial_size(a, b, mode='bilinear', align_corners=False):
    """
    确保两个tensor的空间尺寸一致
    
    Args:
        a: 参考tensor (B, C, H, W)
        b: 需要对齐的tensor (B, C, H', W')
        mode: 插值模式，默认'bilinear'
        align_corners: 是否对齐角点，默认False
    
    Returns:
        a: 原样返回
        b: 对齐后的tensor，空间尺寸与a一致
    """
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode=mode, align_corners=align_corners)
    return a, b


# ============================================================
# Similarity Loss Functions
# ============================================================

def ncc_loss(a, b, eps=1e-6):
    """
    归一化互相关 (Normalized Cross-Correlation) 损失
    值域约在 [0,2]，0 表示完全相关，越大相关性越差
    """
    # 保证空间尺寸一致
    a, b = align_spatial_size(a, b)

    a_mean = a.mean(dim=[2, 3], keepdim=True)
    b_mean = b.mean(dim=[2, 3], keepdim=True)

    a_centered = a - a_mean
    b_centered = b - b_mean

    num = (a_centered * b_centered).mean(dim=[2, 3], keepdim=True)
    den = torch.sqrt(a_centered.pow(2).mean(dim=[2, 3], keepdim=True) * b_centered.pow(2).mean(dim=[2, 3], keepdim=True) + eps)

    ncc = num / (den + eps)  # 越接近 1 说明越一致
    # 损失为 (1 - ncc)，并取平均
    return (1.0 - ncc).mean()


def ssim_loss(a, b, C1=0.01 ** 2, C2=0.03 ** 2, eps=1e-6):
    """
    简化版 SSIM 损失（全局统计），可反向传播：
    loss = 1 - SSIM(a, b)，范围约为 [0,2]
    """
    # 保证空间尺寸一致
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
    对比度损失：约束两幅图像的整体对比度（标准差）接近
    """
    # 保证空间尺寸一致
    a, b = align_spatial_size(a, b)

    std_a = a.std(dim=[2, 3], keepdim=True)
    std_b = b.std(dim=[2, 3], keepdim=True)

    return F.l1_loss(std_a, std_b)


# ============================================================
# Fusion Loss Functions
# ============================================================

def fusion_loss(fused, ir, pol):
    """
    融合损失设计：
    - 对 INF (ir) 更强调高亮区域的人体/目标细节
    - 对 POL (pol) 更强调纹理/边缘细节
    - 利用梯度与强度构造自适应的空间权重
    """
    # 保证 ir / pol 与 fused 空间尺寸一致，避免广播/尺寸不匹配
    fused, ir = align_spatial_size(fused, ir)
    fused, pol = align_spatial_size(fused, pol)

    fx, fy = gradient(fused)
    ix, iy = gradient(ir)
    px, py = gradient(pol)

    # ---------------- INF: 高亮区域权重（人影/目标通常在高亮区） ----------------
    # 将 ir 强度归一化到 [0,1]，作为高亮权重图
    with torch.no_grad():
        ir_min = ir.amin(dim=[2, 3], keepdim=True)
        ir_max = ir.amax(dim=[2, 3], keepdim=True)
        ir_range = (ir_max - ir_min).clamp(min=1e-6)
        ir_norm = (ir - ir_min) / ir_range          # [0,1]，越亮权重越大

    # ---------------- POL: 纹理/边缘权重 ----------------
    # 利用 pol 的梯度幅度构造纹理权重图
    with torch.no_grad():
        g_pol = torch.sqrt(px.pow(2) + py.pow(2) + 1e-6)
        g_min = g_pol.amin(dim=[2, 3], keepdim=True)
        g_max = g_pol.amax(dim=[2, 3], keepdim=True)
        g_range = (g_max - g_min).clamp(min=1e-6)
        g_pol_norm = (g_pol - g_min) / g_range      # [0,1]，纹理越强权重越大

    # ---------------- 强度项：平衡红外高亮和偏振纹理（增强IR细节保护）-------------
    # 在高亮区域更重视红外，在纹理区域更重视偏振
    alpha_ir = 3.5   # 大幅提高红外权重，确保高亮信息充分保留
    beta_pol = 2.0   # 适度降低偏振权重，避免过度压制IR细节

    w_ir_int = 1.0 + alpha_ir * ir_norm  # 高亮区域权重更大
    w_pol_int = 1.0 + beta_pol * g_pol_norm  # 纹理区域权重更大

    L_intensity = torch.mean(w_ir_int * torch.abs(fused - ir) +
                             w_pol_int * torch.abs(fused - pol))

    # ---------------- 结构项：平衡红外结构和偏振纹理（增强IR结构保护）-------------
    # 在高亮区域更重视红外结构，在纹理区域更重视偏振结构
    gamma_ir = 2.5  # 大幅提高红外结构权重，确保IR结构细节保留
    gamma_pol = 1.5  # 适度降低偏振结构权重
    w_ir_str = 1.0 + gamma_ir * ir_norm  # 高亮区域更重视IR结构
    w_pol_str = 1.0 + gamma_pol * g_pol_norm  # 纹理区域更重视POL结构

    L_structure_ir = torch.mean(w_ir_str * (torch.abs(fx - ix) + torch.abs(fy - iy)))
    L_structure_pol = torch.mean(w_pol_str * (torch.abs(fx - px) + torch.abs(fy - py)))
    # 大幅提高IR结构损失的权重，确保IR细节不丢失
    L_structure = 1.8 * L_structure_ir + 1.0 * L_structure_pol

    # ---------------- 最大梯度约束：在 max( ir, pol ) 上保持强边缘 ----------------
    max_x = torch.max(torch.abs(ix), torch.abs(px))
    max_y = torch.max(torch.abs(iy), torch.abs(py))
    L_max = F.l1_loss(torch.abs(fx), max_x) + F.l1_loss(torch.abs(fy), max_y)

    # ---------------- 亮度约束：平衡IR和POL的亮度，避免过度偏暗 ----------------
    # 使用IR和POL的加权平均作为目标亮度，而不是只用POL
    fused_mean = fused.mean()
    ir_mean = ir.mean()
    pol_mean = pol.mean()
    target_mean = 0.4 * ir_mean + 0.6 * pol_mean  # 平衡IR和POL的亮度
    L_brightness = F.mse_loss(fused_mean, target_mean)  # 让融合结果亮度接近平衡后的目标

    return L_intensity + L_structure + L_max + 0.2 * L_brightness  # 降低亮度约束权重


# ============================================================
# VGG Perceptual Loss (感知损失)
# ============================================================

class VGGPerceptualLoss(nn.Module):
    """
    VGG感知损失：使用预训练VGG特征计算图像相似度损失
    更关注语义对齐，而不仅仅是像素对齐
    可用于融合损失计算
    借鉴CPMFusion的感知损失实现
    """
    def __init__(self, layer_idx=8):  # layer_idx=8 对应 relu2_2
        super().__init__()
        # 使用新的weights API替代deprecated的pretrained参数
        try:
            from torchvision.models import VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:layer_idx].eval()
        except ImportError:
            # 兼容旧版本torchvision
            vgg = vgg16(pretrained=True).features[:layer_idx].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        # 确保VGG模型在float32模式（不使用half precision）
        vgg = vgg.float()
        self.vgg = vgg
    
    def forward(self, img1, img2):
        """
        计算两张图像的感知损失
        
        Args:
            img1: (B, C, H, W) 第一张图像，值域应该在[0,1]
            img2: (B, C, H, W) 第二张图像，值域应该在[0,1]
        
        Returns:
            loss: 感知损失标量
        """
        # 禁用autocast以确保VGG在float32精度下运行
        # VGG模型参数是float32，autocast会产生float16输入导致类型不匹配
        with torch.cuda.amp.autocast(enabled=False):
            # 转换为float32以避免混合精度训练时的类型不匹配问题
            img1 = img1.float()
            img2 = img2.float()
            
            # 确保VGG模型在输入所在的设备上
            device = img1.device
            if next(self.vgg.parameters()).device != device:
                self.vgg = self.vgg.to(device)
            
            # 如果输入是单通道，扩展为3通道（VGG需要3通道输入）
            if img1.shape[1] == 1:
                img1 = img1.repeat(1, 3, 1, 1)
            if img2.shape[1] == 1:
                img2 = img2.repeat(1, 3, 1, 1)
            
            # 确保输入图像尺寸一致，避免VGG提取的特征尺寸不匹配
            # 使用较大的尺寸作为目标尺寸
            if img1.shape[-2:] != img2.shape[-2:]:
                target_size = (max(img1.shape[-2], img2.shape[-2]), 
                              max(img1.shape[-1], img2.shape[-1]))
                if img1.shape[-2:] != target_size:
                    img1 = F.interpolate(img1, size=target_size, mode='bilinear', align_corners=False)
                if img2.shape[-2:] != target_size:
                    img2 = F.interpolate(img2, size=target_size, mode='bilinear', align_corners=False)
            # 注意：这里不能直接用align_spatial_size，因为需要对齐到较大的尺寸，而不是第一个tensor的尺寸
            
            # 图像归一化到VGG需要的分布 [0,1] -> ImageNet归一化
            def preprocess(x):
                return normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            feat1 = self.vgg(preprocess(img1))
            feat2 = self.vgg(preprocess(img2))
            
            # 再次确保特征图尺寸一致（防止VGG内部可能导致的尺寸差异）
            if feat1.shape != feat2.shape:
                # 将较小的特征图插值到较大的特征图尺寸
                if feat1.numel() < feat2.numel():
                    feat1 = F.interpolate(feat1, size=feat2.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    feat2 = F.interpolate(feat2, size=feat1.shape[-2:], mode='bilinear', align_corners=False)
            
            return F.l1_loss(feat1, feat2)


def compute_fusion_core_loss(fused, ir_reg, pol_reg):
    """
    核心融合损失：评估融合结果的基本质量
    包括：强度损失、结构损失、梯度损失、亮度损失
    """
    # 保证空间尺寸一致
    fused, ir_reg = align_spatial_size(fused, ir_reg)
    fused, pol_reg = align_spatial_size(fused, pol_reg)
    
    # 数值稳定性：检查输入是否包含NaN或Inf
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

    # 1. 强度损失：融合结果应该保留IR和POL的强度信息
    L_intensity = 0.5 * F.l1_loss(fused, ir_reg) + 0.5 * F.l1_loss(fused, pol_reg)

    # 2. 结构损失：融合结果应该保留IR和POL的结构信息
    L_structure = 0.5 * (F.l1_loss(torch.abs(fx - ix) + torch.abs(fy - iy), torch.zeros_like(fx))) + \
                   0.5 * (F.l1_loss(torch.abs(fx - px) + torch.abs(fy - py), torch.zeros_like(fx)))

    # 3. 梯度损失：融合结果应该保留最大梯度信息
    max_x = torch.max(torch.abs(ix), torch.abs(px))
    max_y = torch.max(torch.abs(iy), torch.abs(py))
    L_gradient = F.l1_loss(torch.abs(fx), max_x) + F.l1_loss(torch.abs(fy), max_y)

    # 4. 亮度损失：平衡IR和POL的亮度
    fused_mean = fused.mean()
    ir_mean = ir_reg.mean()
    pol_mean = pol_reg.mean()
    target_mean = 0.4 * ir_mean + 0.6 * pol_mean
    L_brightness = F.mse_loss(fused_mean, target_mean)

    return L_intensity + L_structure + L_gradient + 0.2 * L_brightness


def compute_ir_bias_loss(fused, ir_reg, pol_reg):
    """
    IR偏向损失（优化版）：确保融合结果在高亮区域（人体）保留更多IR信息
    在非高亮区域（背景）降低IR权重，确保背景主要来自偏振图像
    """
    # 保证空间尺寸一致
    fused, ir_reg = align_spatial_size(fused, ir_reg)
    fused, pol_reg = align_spatial_size(fused, pol_reg)
    
    # 数值稳定性：检查输入是否包含NaN或Inf
    if torch.isnan(fused).any() or torch.isinf(fused).any():
        fused = torch.where(torch.isnan(fused) | torch.isinf(fused), 
                            torch.zeros_like(fused), fused)
    if torch.isnan(ir_reg).any() or torch.isinf(ir_reg).any():
        ir_reg = torch.where(torch.isnan(ir_reg) | torch.isinf(ir_reg), 
                            torch.zeros_like(ir_reg), ir_reg)
    if torch.isnan(pol_reg).any() or torch.isinf(pol_reg).any():
        pol_reg = torch.where(torch.isnan(pol_reg) | torch.isinf(pol_reg), 
                             torch.zeros_like(pol_reg), pol_reg)

    # 1. 高亮区域IR保护损失（更严格的阈值，只针对真正的高亮区域）
    ir_reg_min = ir_reg.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    ir_reg_max = ir_reg.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    ir_reg_range = (ir_reg_max - ir_reg_min).clamp(min=1e-6)  # 确保范围不为0
    ir_reg_norm = (ir_reg - ir_reg_min) / ir_reg_range
    
    # 使用更严格的阈值（0.8），只关注真正的高亮区域（人体），避免背景被误判
    # 进一步提高阈值，确保只有人体高亮区域被识别，背景中的小热源不被误判
    highlight_mask_strict = (ir_reg_norm > 0.8).float()  # 进一步提高阈值到0.8，更精确地只识别人体
    highlight_mask_medium = (ir_reg_norm > 0.6).float()  # 中等高亮区域
    
    # 额外的空间连续性过滤：去除小的孤立高亮点（可能是背景中的小热源）
    # 使用形态学操作，只保留大的连通区域（人体）
    with torch.no_grad():
        # 对高亮掩码进行形态学开运算，去除小的孤立点
        highlight_opened = F.avg_pool2d(highlight_mask_strict, kernel_size=7, stride=1, padding=3)
        highlight_opened = (highlight_opened > 0.3).float()
        highlight_opened = F.avg_pool2d(highlight_opened, kernel_size=7, stride=1, padding=3)
        highlight_opened = (highlight_opened > 0.3).float()
        highlight_mask_strict = highlight_mask_strict * highlight_opened  # 只保留大的连通区域
    
    # 高亮区域强度损失（L1）：确保高亮区域（人体）更接近IR
    # 增加权重，确保高亮区域（人体）更强烈地保留IR信息
    highlight_mask_sum = highlight_mask_strict.sum()
    if highlight_mask_sum > 1e-6:
        L_highlight_strict = torch.mean(highlight_mask_strict * torch.abs(fused - ir_reg)) * 3.0  # 提高权重
    else:
        L_highlight_strict = torch.tensor(0.0, device=fused.device, requires_grad=True)
    
    # 高亮区域结构损失（梯度对齐）：确保高亮区域的结构细节来自IR
    fx, fy = gradient(fused)
    ix, iy = gradient(ir_reg)
    if highlight_mask_sum > 1e-6:
        L_highlight_structure = torch.mean(highlight_mask_strict * (
            torch.abs(fx - ix) + torch.abs(fy - iy)
        )) * 2.0  # 提高权重
    else:
        L_highlight_structure = torch.tensor(0.0, device=fused.device, requires_grad=True)
    
    # 2. 整体IR信息保留损失（加权，高亮区域权重更大，非高亮区域权重更小）
    # 使用非线性权重，高亮区域权重显著增大，非高亮区域权重降低
    # 大幅提高高亮区域的IR权重，确保红外信息被强烈保留
    ir_weight = 1.0 + 8.0 * highlight_mask_strict + 3.0 * (highlight_mask_medium - highlight_mask_strict)  # 提高权重
    # 在非高亮区域（背景）大幅降低IR权重，确保背景不融入红外高亮
    non_highlight_mask = (ir_reg_norm < 0.5).float()  # 非高亮区域（背景），提高阈值到0.5
    ir_weight = ir_weight * (1.0 - 0.8 * non_highlight_mask)  # 在背景区域大幅降低IR权重（从0.6提高到0.8）
    # 数值稳定性：确保权重不为NaN或Inf
    ir_weight = torch.clamp(ir_weight, min=0.0, max=100.0)  # 限制权重范围
    L_ir_preserve = torch.mean(ir_weight * torch.abs(fused - ir_reg))
    
    # 3. 高亮区域密集度约束：确保高亮区域的信息密度得到保留
    def local_variance(x, kernel_size=5):
        local_mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return local_var
    
    ir_density = local_variance(ir_reg)
    fused_density = local_variance(fused)
    # 在高亮区域，融合结果的密集度应该接近红外图像的密集度
    L_density = torch.mean(highlight_mask_strict * torch.abs(fused_density - ir_density))
    
    # 4. 非高亮区域（背景）应该更接近偏振图像，而不是红外图像
    # 在非高亮区域，融合结果应该更接近偏振图像，完全不融入红外高亮
    # 提高权重，确保背景主要来自POL，不融入任何红外高亮信息
    L_non_highlight_pol = torch.mean(non_highlight_mask * torch.abs(fused - pol_reg)) * 1.2  # 进一步提高权重，确保背景来自POL
    L_non_highlight_suppress = torch.mean(non_highlight_mask * torch.abs(fused - ir_reg)) * 0.5  # 适度提高，确保背景不融入IR
    
    # 5. 背景区域结构对齐损失（新增）：确保背景区域的结构细节来自偏振图像
    px, py = gradient(pol_reg)
    L_background_structure = torch.mean(non_highlight_mask * (
        torch.abs(fx - px) + torch.abs(fy - py)
    )) * 0.6  # 确保背景区域的结构来自POL

    return (L_highlight_strict + 0.8 * L_highlight_structure + 
            L_ir_preserve + 0.5 * L_density + L_non_highlight_pol + L_non_highlight_suppress + L_background_structure)


class PolarTextureLoss(nn.Module):
    """
    偏振纹理保留损失模块：使用buffer存储拉普拉斯核，避免重复创建
    """
    def __init__(self):
        super().__init__()
        # 注册拉普拉斯核为buffer，避免每次forward都创建
        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
    
    def forward(self, fused, pol_reg):
        """
        计算偏振纹理保留损失（增强版：特别关注暗处细节）
        
        Args:
            fused: 融合结果 (B, C, H, W)
            pol_reg: 偏振图像 (B, C, H, W)
        
        Returns:
            loss: 偏振纹理保留损失
        """
        # 保证空间尺寸一致
        fused, pol_reg = align_spatial_size(fused, pol_reg)
        
        # 0. 检测暗处区域（新增）：在暗处区域特别强调细节保留
        with torch.no_grad():
            pol_min = pol_reg.amin(dim=[2, 3], keepdim=True)
            pol_max = pol_reg.amax(dim=[2, 3], keepdim=True)
            pol_range = (pol_max - pol_min).clamp(min=1e-6)
            pol_norm = (pol_reg - pol_min) / pol_range
            # 暗处掩码：低亮度区域（值越大表示越暗）
            dark_mask = 1.0 - pol_norm
            # 使用阈值，只关注真正的暗处区域
            dark_threshold = 0.4  # 低于40%亮度的区域视为暗处
            dark_mask = (dark_mask > dark_threshold).float()
        
        # 1. 一阶梯度纹理保留（边缘/纹理强度）
        fx, fy = gradient(fused)
        px, py = gradient(pol_reg)
        g_fused = torch.sqrt(fx.pow(2) + fy.pow(2) + 1e-6)
        g_pol = torch.sqrt(px.pow(2) + py.pow(2) + 1e-6)
        
        # 在纹理区域（高梯度区域）更重视偏振纹理保留
        # 添加边界检查，提高数值稳定性
        try:
            quantile_val = g_pol.quantile(0.3)
            # 检查quantile值是否有效
            if torch.isnan(quantile_val) or torch.isinf(quantile_val) or quantile_val <= 0:
                # 回退到中位数
                quantile_val = g_pol.median()
                if torch.isnan(quantile_val) or torch.isinf(quantile_val) or quantile_val <= 0:
                    # 如果中位数也无效，使用均值
                    quantile_val = g_pol.mean()
                    if torch.isnan(quantile_val) or torch.isinf(quantile_val) or quantile_val <= 0:
                        # 如果均值也无效，使用固定值
                        quantile_val = torch.tensor(1e-6, device=g_pol.device, dtype=g_pol.dtype)
        except Exception:
            # 如果quantile计算失败，使用中位数
            quantile_val = g_pol.median()
            if torch.isnan(quantile_val) or torch.isinf(quantile_val) or quantile_val <= 0:
                quantile_val = g_pol.mean()
                if torch.isnan(quantile_val) or torch.isinf(quantile_val) or quantile_val <= 0:
                    quantile_val = torch.tensor(1e-6, device=g_pol.device, dtype=g_pol.dtype)
        
        # 确保quantile_val是有效的标量
        if quantile_val.dim() > 0:
            quantile_val = quantile_val.item() if quantile_val.numel() == 1 else quantile_val.mean()
        quantile_val = max(float(quantile_val), 1e-6)  # 确保至少为1e-6
        
        texture_mask = (g_pol > quantile_val).float()  # 纹理区域掩码
        # 在暗处区域和纹理区域的交集处，权重更大，但降低增强强度避免泛白
        combined_mask = texture_mask * (1.0 + 1.5 * dark_mask) + dark_mask * 1.0  # 降低权重系数
        L_gradient = torch.mean(combined_mask * torch.abs(g_fused - g_pol))
        
        # 2. 二阶梯度（拉普拉斯）纹理保留 - 突出细微纹理
        # 使用注册的buffer，避免重复创建
        lap_fused = F.conv2d(fused, self.laplacian_kernel, padding=1)
        lap_pol = F.conv2d(pol_reg, self.laplacian_kernel, padding=1)
        # 在暗处区域更重视拉普拉斯纹理保留，但降低权重避免过度增强
        laplacian_weight = 1.0 + 1.2 * dark_mask  # 降低权重系数
        L_laplacian = torch.mean(laplacian_weight * torch.abs(torch.abs(lap_fused) - torch.abs(lap_pol)))
    
        # 3. 纹理方向一致性（梯度方向对齐）
        # 计算梯度方向
        fused_dir = torch.atan2(fy, fx + 1e-6)
        pol_dir = torch.atan2(py, px + 1e-6)
        # 方向差异（在强纹理区域）
        dir_diff = torch.abs(fused_dir - pol_dir)
        dir_diff = torch.min(dir_diff, 2 * math.pi - dir_diff)  # 处理周期性
        # 在暗处区域更重视方向一致性，但降低权重
        direction_weight = texture_mask * (1.0 + 1.2 * dark_mask)  # 降低权重系数
        L_direction = torch.mean(direction_weight * dir_diff)
        
        # 4. 局部对比度保留（纹理区域的对比度应该保留）
        # 使用局部标准差作为对比度度量
        def local_std(x, kernel_size=5):
            local_mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            return torch.sqrt(local_var + 1e-6)
        
        contrast_fused = local_std(fused)
        contrast_pol = local_std(pol_reg)
        # 在暗处区域更重视对比度保留，但降低权重避免过度增强
        contrast_weight = texture_mask * (1.0 + 1.5 * dark_mask)  # 降低权重系数
        L_contrast = torch.mean(contrast_weight * torch.abs(contrast_fused - contrast_pol))
        
        # 5. 暗处细节强度保留：确保暗处的细节强度得到保留，但降低权重避免泛白
        # 在暗处区域，融合结果应该保留偏振图像的细节强度
        dark_intensity_weight = dark_mask * 1.2  # 降低权重，避免过度增强
        L_dark_intensity = torch.mean(dark_intensity_weight * torch.abs(fused - pol_reg))
        
        # 6. 细节区域聚焦损失：确保偏振信息聚焦于细节丰富的区域（背景区域）
        # 使用梯度幅值作为细节丰富度的指标
        detail_mask = (g_pol > g_pol.quantile(0.5)).float()  # 细节丰富的区域（梯度幅值大于中位数）
        # 在细节区域，融合结果应该更接近偏振图像，但降低权重避免过度增强
        detail_weight = 1.0 + 1.8 * detail_mask  # 降低权重系数，避免泛白
        L_detail_focus = torch.mean(detail_weight * torch.abs(fused - pol_reg))
        
        # 7. 细节区域结构对齐损失：在细节区域，梯度结构应该对齐
        L_detail_structure = torch.mean(detail_mask * (
            torch.abs(fx - px) + torch.abs(fy - py)
        ))
        
        # 8. 非细节区域（高亮区域）应该更接近红外图像，而不是偏振图像
        # 在非细节区域（通常是高亮区域，人体），降低偏振信息的影响
        non_detail_mask = (g_pol < g_pol.quantile(0.3)).float()  # 细节较少的区域（通常是高亮区域）
        # 在非细节区域，融合结果应该更接近红外图像，而不是偏振图像
        L_non_detail_suppress = torch.mean(non_detail_mask * torch.abs(fused - pol_reg)) * 0.3  # 降低权重
        
        # 综合损失：加权组合（降低各组件权重，避免过度增强导致的泛白）
        return (0.25 * L_gradient + 0.18 * L_laplacian + 0.1 * L_direction + 
                0.08 * L_contrast + 0.1 * L_dark_intensity + 
                0.15 * L_detail_focus + 0.08 * L_detail_structure + 0.06 * L_non_detail_suppress)


# 创建全局实例，避免重复创建
_polar_texture_loss_module = None

def compute_polar_texture_preservation_loss(fused, pol_reg):
    """
    偏振纹理保留损失：专门强化偏振纹理细节的保留
    使用多尺度、多方向的纹理特征进行约束
    
    注意：内部使用PolarTextureLoss模块来避免重复创建拉普拉斯核
    """
    global _polar_texture_loss_module
    if _polar_texture_loss_module is None:
        _polar_texture_loss_module = PolarTextureLoss()
    
    # 确保模块在正确的设备上（检查buffer而不是parameters，因为模块只有buffer没有参数）
    if _polar_texture_loss_module.laplacian_kernel.device != fused.device:
        _polar_texture_loss_module = _polar_texture_loss_module.to(fused.device)
    
    return _polar_texture_loss_module(fused, pol_reg)


# ============================================================
# Total Loss Function (Reorganized)
# ============================================================

def compute_intermediate_supervision_loss(outputs, ir, pol):
    """
    中间监督损失：确保特征提取阶段就聚焦于正确区域
    1. 红外特征应该聚焦于高亮区域
    2. 偏振特征应该聚焦于细节区域
    """
    # 获取中间监督掩码
    ir_highlight_mask = outputs.get("ir_highlight_focus_mask", None)
    pol_detail_mask = outputs.get("pol_detail_focus_mask", None)
    
    if ir_highlight_mask is None or pol_detail_mask is None:
        # 如果没有中间监督掩码，返回零损失
        return torch.tensor(0.0, device=ir.device, requires_grad=True)
    
    # 对齐空间尺寸
    ir_highlight_mask, ir = align_spatial_size(ir_highlight_mask, ir)
    pol_detail_mask, pol = align_spatial_size(pol_detail_mask, pol)
    
    # 1. 红外高亮聚焦监督损失
    # 确保高亮掩码在红外图像的高亮区域有高响应
    with torch.no_grad():
        ir_min = ir.amin(dim=[2, 3], keepdim=True)
        ir_max = ir.amax(dim=[2, 3], keepdim=True)
        ir_norm = (ir - ir_min) / (ir_max - ir_min + 1e-6)
        ir_highlight_gt = (ir_norm > 0.6).float()  # 真实高亮区域
    
    # 高亮掩码应该在真实高亮区域有高响应
    L_ir_focus = F.mse_loss(ir_highlight_mask, ir_highlight_gt)
    
    # 2. 偏振细节聚焦监督损失
    # 确保细节掩码在偏振图像的细节区域有高响应
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
        
        # 确保detail_threshold是有效的标量
        if detail_threshold.dim() > 0:
            detail_threshold = detail_threshold.item() if detail_threshold.numel() == 1 else detail_threshold.mean()
        detail_threshold = max(float(detail_threshold), 1e-6)  # 确保至少为1e-6
        pol_detail_gt = (pol_grad_mag > detail_threshold).float()  # 真实细节区域
    
    # 细节掩码应该在真实细节区域有高响应
    L_pol_focus = F.mse_loss(pol_detail_mask, pol_detail_gt)
    
    return L_ir_focus + L_pol_focus


def total_loss(outputs, ir, pol,
               lambda_fusion_core=1.0,
               # 将IR偏向损失权重适当降低，减弱红外在非目标区域（如柱子、墙面）的“泛白”影响
               lambda_ir_bias=2.0,
               # 略微提高偏振纹理损失权重，让背景和结构区域更多地跟随POL，从而压制柱子发白
               lambda_polar_texture=2.5,
               lambda_regularization=0.3,
               lambda_intermediate=1.0):  # 中间监督损失权重
    """
    纯融合损失函数（无配准）：
    L_total = λ1 * L_fusion_core + λ2 * L_IR_bias + λ3 * L_polar_texture + λ4 * L_regularization
    
    Args:
        outputs: 模型输出字典，包含：
            - fusion: 融合结果
            - ir_reg: 原始IR图像（用于兼容性）
            - pol_reg: 原始POL图像（用于兼容性）
        ir: 原始IR图像
        pol: 原始POL图像
        lambda_fusion_core: 核心融合损失权重（默认1.0）
        lambda_ir_bias: IR偏向损失权重（默认1.5）
        lambda_polar_texture: 偏振纹理保留损失权重（默认0.8，新增）
        lambda_regularization: 正则化损失权重（默认0.3）
    
    Returns:
        L_total: 总损失
        loss_dict: 详细损失字典
    """
    fuse = outputs["fusion"]
    ir_reg = outputs.get("ir_reg", ir)  # 兼容性：如果没有配准，使用原始IR
    pol_reg = outputs.get("pol_reg", pol)  # 兼容性：如果没有配准，使用原始POL

    # 确保所有输入的空间尺寸一致（以融合结果为基准）
    fuse, ir_reg = align_spatial_size(fuse, ir_reg)
    fuse, pol_reg = align_spatial_size(fuse, pol_reg)

    # 1. Fusion Core Loss (L_fusion_core) - 核心融合损失
    L_fusion_core = compute_fusion_core_loss(fuse, ir_reg, pol_reg)

    # 2. IR Bias Loss (L_IR_bias) - IR偏向损失（确保高亮区域保留IR信息）
    L_ir_bias = compute_ir_bias_loss(fuse, ir_reg, pol_reg)

    # 3. Polar Texture Preservation Loss (L_polar_texture) - 偏振纹理保留损失（新增）
    L_polar_texture = compute_polar_texture_preservation_loss(fuse, pol_reg)

    # 4. Regularization Loss (L_regularization) - 简化版正则化（仅融合质量约束）
    # 移除配准相关的正则化，只保留融合质量约束
    L_regularization = 0.0
    # 融合结果与输入的一致性约束（此时尺寸已对齐）
    L_cons = (
        F.l1_loss(fuse, ir_reg) +
        F.l1_loss(fuse, pol_reg)
    )
    L_regularization = 0.5 * L_cons

    # 5. Intermediate Supervision Loss (L_intermediate) - 中间监督损失（新增）
    # 确保特征提取阶段就聚焦于正确区域
    L_intermediate = compute_intermediate_supervision_loss(outputs, ir, pol)

    # 6. Total Loss: L_total = λ1 * L_fusion_core + λ2 * L_IR_bias + λ3 * L_polar_texture + λ4 * L_regularization + λ5 * L_intermediate
    # 数值稳定性：检查每个损失组件是否包含NaN或Inf
    loss_components = [
        lambda_fusion_core * L_fusion_core,
        lambda_ir_bias * L_ir_bias,
        lambda_polar_texture * L_polar_texture,
        lambda_regularization * L_regularization,
        lambda_intermediate * L_intermediate
    ]
    
    # 替换NaN和Inf为0
    loss_components_clean = []
    for comp in loss_components:
        if torch.isnan(comp) or torch.isinf(comp):
            comp = torch.tensor(0.0, device=comp.device, requires_grad=True)
        loss_components_clean.append(comp)
    
    L_total = sum(loss_components_clean)
    
    # 最终检查：如果总损失仍然是NaN或Inf，返回一个小的非零损失
    if torch.isnan(L_total) or torch.isinf(L_total):
        L_total = torch.tensor(1e-6, device=fuse.device, requires_grad=True)
    
    # 返回总损失和详细损失字典（用于训练监控）
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

