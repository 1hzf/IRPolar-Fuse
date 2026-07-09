import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba
from losses import total_loss
from utils import gradient

# ============================================================
# Enhanced Encoder (强化特征提取)
# ============================================================

class EnhancedConvEncoder(nn.Module):
    """
    增强的特征提取器：
    - 多尺度特征提取（保留不同分辨率的特征）
    - 残差连接增强梯度流
    - 注意力机制强化重要特征
    """
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        # 第一层：初始特征提取
        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )
        
        # 第二层：下采样 + 特征增强
        self.c2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, 1, 1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True)
        )
        
        # 第三层：进一步下采样 + 深度特征提取
        self.c3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1),  # 额外一层增强特征
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True)
        )
        
        # 通道注意力：强化重要特征通道
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
        
        # 应用通道注意力
        attn = self.channel_attn(f3)
        f3 = f3 * attn
        
        return f1, f2, f3

# ============================================================
# Multi-Scale Feature Extractor (多尺度特征提取)
# ============================================================

class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器：提取不同尺度的特征用于融合
    """
    def __init__(self, ch):
        super().__init__()
        # 多尺度卷积：提取不同尺度的特征
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
        # 融合多尺度特征
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
    利用 Mamba 在两个模态特征的联合空间上建模长程依赖，
    让融合前的特征已经充分交换来自 INF 和 POL 的细节信息。
    """
    def __init__(self, dim):
        super().__init__()
        # 在通道维拼接两个模态，因此 d_model = 2 * dim
        self.mamba = Mamba(
            d_model=dim * 2,
            d_state=dim * 2,
            d_conv=4,
            expand=2
        )

    def forward(self, f_ir, f_pol):
        """
        f_ir, f_pol: (B, C, H, W)
        输出两个“增强后的”模态特征，已经包含对方的信息
        """
        B, C, H, W = f_ir.shape

        # 若两路特征空间尺寸不一致，将 f_pol 对齐到 f_ir 的尺寸
        if f_pol.shape[2:] != (H, W):
            f_pol = F.interpolate(f_pol, size=(H, W), mode='bilinear', align_corners=False)

        x = torch.cat([f_ir, f_pol], dim=1)      # (B, 2C, H, W)
        seq = x.flatten(2).transpose(1, 2)       # (B, HW, 2C)
        # 如果序列太长，分块处理以避免cuDNN错误
        if seq.size(1) > 65536:  # 如果序列长度超过65536，分块处理
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
    将红外图像的高亮区域（通常对应人体/目标）以"软掩码"的方式注入最终融合结果中：
      fused_out = (1 - m) * fused + m * ir
    m 由 ir 强度自适应得到；阈值和陡峭度是经验证选择的固定超参数，避免硬阈值导致伪影。
    当前场景下，适当提高阈值、减弱基础注入比例，可以减少靠近背景（如柱子、墙面）的泛白。
    """
    def __init__(self, init_thresh=0.62, init_sharpness=15.0, blur_kernel=3, inject_ratio_base=0.6):
        super().__init__()
        # 略微提高阈值到0.62，减少中等亮度背景区域被误判为高亮目标
        # These values are fixed for a run and persisted in the state dict as buffers.
        # Treating them as fixed removes the ambiguity between the architecture diagram
        # and the implementation, and makes sensitivity experiments directly reproducible.
        self.register_buffer("thresh", torch.tensor(float(init_thresh)))
        self.register_buffer("sharpness", torch.tensor(float(init_sharpness)))
        self.inject_ratio_base = float(inject_ratio_base)
        k = int(blur_kernel)
        k = max(1, k)
        # 轻微平滑掩码，抑制块状边界（减小kernel size，避免过度模糊）
        self.blur = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2) if k > 1 else nn.Identity()

    def forward(self, fused, ir):
        # 对齐空间尺寸
        if ir.shape[-2:] != fused.shape[-2:]:
            ir = F.interpolate(ir, size=fused.shape[-2:], mode='bilinear', align_corners=False)

        # 每张图归一化到 [0,1]，提取高亮区域
        with torch.no_grad():
            ir_min = ir.amin(dim=[2, 3], keepdim=True)
            ir_max = ir.amax(dim=[2, 3], keepdim=True)
            ir_norm = (ir - ir_min) / (ir_max - ir_min + 1e-6)

        # 固定参数的软阈值掩码：略抬高阈值，避免大面积背景被纳入高亮区域
        sharp = self.sharpness.clamp(1.0, 50.0)
        thr = self.thresh.clamp(0.5, 0.85)
        m = torch.sigmoid(sharp * (ir_norm - thr))
        m = self.blur(m)
        
        # 改进的空间连续性过滤：更温和地去除小的孤立高亮点，避免过度去除有效特征
        # 使用更小的kernel和更宽松的阈值，只去除非常小的孤立点
        with torch.no_grad():
            # 二值化掩码（降低阈值到0.3，保留更多中等亮度区域）
            m_binary = (m > 0.3).float()
            # 使用更小的kernel（3x3）进行形态学开运算，只去除非常小的孤立点
            m_opened = F.avg_pool2d(m_binary, kernel_size=3, stride=1, padding=1)
            m_opened = (m_opened > 0.2).float()  # 降低阈值，保留更多区域
            m_opened = F.avg_pool2d(m_opened, kernel_size=3, stride=1, padding=1)
            m_opened = (m_opened > 0.2).float()
            # 将形态学处理后的掩码与原始掩码结合，但使用更温和的策略
            # 只对非常小的孤立点进行抑制，保留中等大小的连通区域
            m_filtered = m * (0.7 + 0.3 * m_opened)  # 温和地结合，避免过度去除
            m = m_filtered

        # 改进的注入策略：略微降低基础注入比例，减弱IR对背景区域的“漂白”作用
        # 高亮区域（m>0.7）仍然有明显注入，中等区域（0.3<m<0.7）注入适度减弱
        inject_ratio_base = self.inject_ratio_base
        # 使用更平滑的映射：高亮区域注入更多，中等区域也适度注入
        # 使用线性+平方的组合，让中等区域也能得到足够的注入
        m_linear = m  # 线性部分，保证中等区域也有注入
        m_squared = m.pow(2)  # 平方部分，让高亮区域更突出
        inject_ratio_adaptive = inject_ratio_base * (0.4 + 0.4 * m_linear + 0.2 * m_squared)  # composite mapping，更好地保留中等亮度特征
        
        # 背景抑制：只对真正的低亮度区域（m<0.15）进行抑制，允许中等亮度通过
        background_suppress = (m < 0.15).float()
        inject_ratio_adaptive = inject_ratio_adaptive * (1.0 - 0.8 * background_suppress)
        
        fused_out = fused * (1.0 - inject_ratio_adaptive * m) + ir * (inject_ratio_adaptive * m)
        return fused_out, m

# ============================================================
# Wavelet Attention Module (参考STANet的注意力机制)
# ============================================================

class WaveletAttention(nn.Module):
    """
    小波注意力模块：参考STANet的时空注意力思想
    使用小波变换将图像分解为不同频率子带，然后在各子带上应用注意力机制
    特别适合提取偏振图像的纹理细节
    """
    def __init__(self, in_ch=1, reduction=4):
        super().__init__()
        # 小波分解的近似（低频）和细节（高频）分支
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 通道注意力：学习不同频率子带的重要性
        # 确保中间通道数至少为1，避免除零错误
        mid_ch = max(1, in_ch // reduction)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力：学习空间位置的重要性
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3),  # 输入：平均池化和最大池化的拼接
            nn.Sigmoid()
        )
        
    def dwt_2d(self, x):
        """
        简化的2D离散小波变换（使用可分离的Haar小波）
        将图像分解为4个子带：LL(低频), LH(水平高频), HL(垂直高频), HH(对角高频)
        使用更稳定的实现方式
        """
        B, C, H, W = x.shape
        # 确保尺寸是偶数
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, H, W = x.shape
        
        # Haar小波低通和高通滤波器（归一化）
        h_low = torch.tensor([[1.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 1, 2) / 2.0
        h_high = torch.tensor([[1.0, -1.0]], device=x.device, dtype=x.dtype).view(1, 1, 1, 2) / 2.0
        
        # 水平方向小波分解
        x_low_h = F.conv2d(x, h_low, padding=(0, 0), stride=(1, 2))  # (B, C, H, W/2)
        x_high_h = F.conv2d(x, h_high, padding=(0, 0), stride=(1, 2))  # (B, C, H, W/2)
        
        # 垂直方向小波分解
        h_low_t = h_low.transpose(2, 3)  # (1, 1, 2, 1)
        h_high_t = h_high.transpose(2, 3)  # (1, 1, 2, 1)
        
        LL = F.conv2d(x_low_h, h_low_t, padding=(0, 0), stride=(2, 1))  # 低频 (B, C, H/2, W/2)
        LH = F.conv2d(x_low_h, h_high_t, padding=(0, 0), stride=(2, 1))  # 水平高频
        HL = F.conv2d(x_high_h, h_low_t, padding=(0, 0), stride=(2, 1))  # 垂直高频
        HH = F.conv2d(x_high_h, h_high_t, padding=(0, 0), stride=(2, 1))  # 对角高频
        
        return LL, LH, HL, HH
    
    def idwt_2d(self, LL, LH, HL, HH):
        """
        逆小波变换：重构图像
        使用转置卷积进行重构
        """
        # Haar小波重构滤波器（与分解滤波器相同）
        g_low = torch.tensor([[1.0, 1.0]], device=LL.device, dtype=LL.dtype).view(1, 1, 1, 2)
        g_high = torch.tensor([[1.0, -1.0]], device=LL.device, dtype=LL.dtype).view(1, 1, 1, 2)
        
        # 垂直方向重构
        g_low_t = g_low.transpose(2, 3)  # (1, 1, 2, 1)
        g_high_t = g_high.transpose(2, 3)  # (1, 1, 2, 1)
        
        # 使用转置卷积进行上采样和滤波
        x_low_h = F.conv_transpose2d(LL, g_low_t, stride=(2, 1), padding=(0, 0), output_padding=(0, 0)) + \
                  F.conv_transpose2d(LH, g_high_t, stride=(2, 1), padding=(0, 0), output_padding=(0, 0))
        x_high_h = F.conv_transpose2d(HL, g_low_t, stride=(2, 1), padding=(0, 0), output_padding=(0, 0)) + \
                   F.conv_transpose2d(HH, g_high_t, stride=(2, 1), padding=(0, 0), output_padding=(0, 0))
        
        # 水平方向重构
        x = F.conv_transpose2d(x_low_h, g_low, stride=(1, 2), padding=(0, 0), output_padding=(0, 0)) + \
            F.conv_transpose2d(x_high_h, g_high, stride=(1, 2), padding=(0, 0), output_padding=(0, 0))
        
        return x
    
    def forward(self, x):
        """
        x: (B, C, H, W) 输入特征
        返回：增强后的特征
        """
        B, C, H, W = x.shape
        
        # 1. 小波分解：将图像分解为4个子带
        LL, LH, HL, HH = self.dwt_2d(x)
        
        # 2. 对每个子带应用通道注意力
        # 对每个子带分别应用通道注意力（每个子带都是C通道）
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
        
        # 3. 对高频子带（LH, HL, HH）应用空间注意力（强调纹理区域）
        # 拼接高频子带
        high_freq = torch.cat([LH_att, HL_att, HH_att], dim=1)  # (B, 3C, H/2, W/2)
        # 空间注意力
        avg_out_spatial = torch.mean(high_freq, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(high_freq, dim=1, keepdim=True)
        spatial_att_input = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = self.spatial_attention(spatial_att_input)
        # 只对高频子带应用空间注意力
        LH_att = LH_att * spatial_att
        HL_att = HL_att * spatial_att
        HH_att = HH_att * spatial_att
        
        # 4. 逆小波变换重构
        x_enhanced = self.idwt_2d(LL_att, LH_att, HL_att, HH_att)
        
        # 5. 确保输出尺寸与输入一致（由于小波变换可能有1像素差异）
        if x_enhanced.shape[-2:] != (H, W):
            x_enhanced = F.interpolate(x_enhanced, size=(H, W), mode='bilinear', align_corners=False)
        
        return x_enhanced

# ============================================================
# Polar Texture Enhancer: 多尺度偏振纹理增强模块（集成小波注意力）
# ============================================================

class PolarTextureEnhancer(nn.Module):
    """
    增强版偏振纹理细节挖掘模块：
      - 多尺度纹理提取（细/粗尺度卷积）
      - 多方向梯度提取（一阶、二阶梯度、拉普拉斯算子）
      - 纹理方向性分析（梯度方向直方图）
      - 自适应对比度增强（CLAHE风格）
      - 频域纹理增强（高频成分提取）
      - 自适应权重融合，在非高亮区域更强调偏振纹理
    """
    def __init__(self, mid_ch=16):
        super().__init__()
        # 小波注意力模块（新增）：在频域提取纹理细节
        self.wavelet_attention = WaveletAttention(in_ch=1, reduction=4)
        
        # 多尺度纹理提取
        self.conv_pol_fine = nn.Conv2d(1, mid_ch, 3, 1, 1)      # 细尺度纹理
        self.conv_pol_coarse = nn.Conv2d(1, mid_ch, 5, 1, 2)   # 粗尺度纹理
        self.conv_pol_medium = nn.Conv2d(1, mid_ch, 7, 1, 3)    # 中尺度纹理
        
        # 小波增强后的特征提取（在小波域提取纹理）
        self.wavelet_conv = nn.Conv2d(1, mid_ch, 3, 1, 1)
        
        # 拉普拉斯算子（二阶梯度）提取
        self.laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 纹理方向性分析（使用可学习的方向滤波器）
        self.direction_conv = nn.Conv2d(1, mid_ch // 2, 3, 1, 1)
        
        # 自适应对比度增强（轻量级）
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(1, mid_ch // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 4, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # 增强的纹理融合网络（融合更多特征，包括小波特征）
        # 输入：细+粗+中尺度(3*mid_ch) + 小波特征(mid_ch) + 梯度幅值(1) + 拉普拉斯(1) + 方向特征(mid_ch//2) + 对比度增强(1)
        fusion_in_ch = mid_ch * 4 + 1 + 1 + mid_ch // 2 + 1
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_ch, mid_ch * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch * 2, mid_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1)
        )
        
    def compute_laplacian(self, x):
        """计算拉普拉斯算子（二阶梯度）"""
        if self.laplacian_kernel.device != x.device:
            self.laplacian_kernel = self.laplacian_kernel.to(x.device)
        laplacian = F.conv2d(x, self.laplacian_kernel, padding=1)
        return torch.abs(laplacian)  # 取绝对值，突出边缘
        
    def compute_texture_direction(self, pol):
        """计算纹理方向性（梯度方向）"""
        pol_dx, pol_dy = gradient(pol)
        # 计算梯度方向（角度）
        direction = torch.atan2(pol_dy, pol_dx + 1e-6)  # [-π, π]
        # 归一化到[0, 1]
        direction_norm = (direction + math.pi) / (2 * math.pi)
        # 使用可学习的卷积提取方向特征
        direction_feat = F.relu(self.direction_conv(direction_norm), inplace=True)
        return direction_feat
        
    def adaptive_contrast_enhance(self, pol):
        """自适应对比度增强（类似CLAHE）"""
        # 局部对比度增强权重
        contrast_weight = self.contrast_enhancer(pol)
        # 局部均值（用于自适应增强）
        local_mean = F.avg_pool2d(pol, kernel_size=5, stride=1, padding=2)
        # 自适应增强：在低对比度区域增强更多
        enhanced = pol + contrast_weight * (pol - local_mean) * 0.5
        return enhanced
        
    def forward(self, fused, pol, ir_highlight_mask):
        """
        fused: 当前融合结果
        pol: 偏振图像
        ir_highlight_mask: 红外高亮掩码（可选），用于保护高亮区域
        """
        H, W = fused.shape[-2], fused.shape[-1]
        if pol.shape[-2:] != (H, W):
            pol = F.interpolate(pol, size=(H, W), mode='bilinear', align_corners=False)
        
        # 0. 小波注意力增强（新增）：在频域提取纹理细节
        pol_wavelet = self.wavelet_attention(pol)  # 小波域增强
        pol_wavelet_feat = F.relu(self.wavelet_conv(pol_wavelet), inplace=True)  # 提取小波特征
        
        # 1. 多尺度纹理提取（细/中/粗）
        pol_fine = F.relu(self.conv_pol_fine(pol), inplace=True)
        pol_medium = F.relu(self.conv_pol_medium(pol), inplace=True)
        pol_coarse = F.relu(self.conv_pol_coarse(pol), inplace=True)
        
        # 2. 一阶梯度（边缘/纹理强度）
        pol_dx, pol_dy = gradient(pol)
        pol_edge = torch.sqrt(pol_dx.pow(2) + pol_dy.pow(2) + 1e-6)
        
        # 3. 二阶梯度（拉普拉斯算子）- 突出更细微的纹理
        pol_laplacian = self.compute_laplacian(pol)
        
        # 4. 纹理方向性分析
        pol_direction = self.compute_texture_direction(pol)
        
        # 5. 自适应对比度增强
        pol_enhanced = self.adaptive_contrast_enhance(pol)
        
        # 7. 融合所有纹理特征（包括小波特征）
        x = torch.cat([
            pol_fine, pol_medium, pol_coarse,  # 多尺度特征
            pol_wavelet_feat,                  # 小波域特征（新增）
            pol_edge,                         # 一阶梯度
            pol_laplacian,                    # 二阶梯度（拉普拉斯）
            pol_direction,                    # 方向特征
            pol_enhanced                      # 对比度增强
        ], dim=1)
        
        texture_residual = self.fusion_conv(x)
        
        # 8. 自适应增强偏振纹理残差：在非高亮区域更强调纹理，但降低增强强度避免泛白
        if ir_highlight_mask is not None:
            # 在非高亮区域增强纹理残差，但使用更保守的增强系数，避免泛白
            # 背景区域（非高亮）增强更多，但总体降低增强强度
            texture_scale = 1.5 + 0.8 * (1.0 - ir_highlight_mask)  # 大幅降低增强系数，避免泛白
        else:
            texture_scale = 1.8  # 降低默认增强比例，避免过度增强
        
        texture_residual = texture_scale * texture_residual
        
        # 9. 残差注入：在原融合结果基础上叠加偏振纹理细节
        # 在高亮区域（人体）降低纹理残差的影响，避免干扰IR细节
        # 在背景区域（非高亮）完全使用偏振纹理，不融入任何红外高亮
        if ir_highlight_mask is not None:
            # 在高亮区域（m>0.5）大幅降低纹理残差，在背景区域（m<0.2）完全使用偏振纹理
            highlight_suppress = (ir_highlight_mask > 0.5).float()
            background_enhance = (ir_highlight_mask < 0.2).float()
            texture_residual = texture_residual * (1.0 - 0.8 * highlight_suppress)  # 在高亮区域大幅降低纹理残差
            texture_residual = texture_residual * (1.0 + 0.3 * background_enhance)  # 在背景区域增强偏振纹理
        
        # 10. 噪声抑制：对纹理残差进行轻微平滑，避免引入噪声
        texture_residual = F.avg_pool2d(texture_residual, kernel_size=3, stride=1, padding=1) * 0.7 + texture_residual * 0.3
        
        return fused + texture_residual

# ============================================================
# Task 2: Fusion Head (Explainable Polar Attention)
# ============================================================

class PolarFusionAttention(nn.Module):
    """
    基于Mamba-SSM的智能融合注意力模块：
    - 使用Mamba-SSM建模空间依赖关系，搜寻高亮和纹理区域
    - 高亮区域：给予IR高权重（人体、热源等）
    - 复杂纹理区域：给予POL高权重（树叶、栏杆、细节等）
    - 通过Mamba的长程依赖建模，引导融合结果更加全面高质量
    """
    def __init__(self, ch):
        super().__init__()
        # IR特征处理分支：专门处理红外高亮信息（增强版）
        self.ir_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),  # 增加一层，增强高亮特征提取
            nn.ReLU(inplace=True)
        )
        # POL特征处理分支：专门处理偏振纹理信息（增强版）
        self.pol_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),  # 增加一层，增强纹理特征提取
            nn.ReLU(inplace=True)
        )
        
        # Mamba-SSM模块：建模空间依赖，搜寻高亮和纹理区域
        # 降低d_state以减少显存占用和计算复杂度
        self.mamba_attn = Mamba(
            d_model=ch * 2,  # 输入IR和POL的拼接特征
            d_state=ch // 2,  # 降低状态维度，减少显存占用
            d_conv=4,
            expand=2
        )
        
        # 高亮区域检测头：从IR原图中提取高亮特征（增强版）
        self.highlight_head = nn.Sequential(
            nn.Conv2d(1, ch, 3, 1, 1),  # 输入1通道原图
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch // 2, 3, 1, 1),  # 增加中间层
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 1, 1)  # 输出高亮图
        )
        
        # 纹理复杂度检测头：从POL原图中提取纹理特征（增强版）
        self.texture_head = nn.Sequential(
            nn.Conv2d(1, ch, 3, 1, 1),  # 输入1通道原图
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch // 2, 3, 1, 1),  # 增加中间层
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 1, 1)  # 输出纹理复杂度图
        )
        
        # 自适应权重生成网络：基于高亮和纹理信息生成融合权重（增强版）
        self.weight_gen = nn.Sequential(
            nn.Conv2d(ch * 2 + 2, ch, 3, 1, 1),  # 输入：IR+POL特征 + 高亮图+纹理图
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch // 2, 3, 1, 1),  # 增加中间层，增强权重生成能力
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 2, 1)  # 输出IR和POL的权重
        )
        
        # 偏振纹理增强分支：轻量级纹理提取
        self.pol_texture = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, ir, pol, ir_img, pol_img):
        """
        输入顺序固定：ir在前，pol在后
        使用Mamba-SSM智能搜寻高亮和纹理区域，动态调整融合权重
        Args:
            ir: (B, C, H, W) IR特征
            pol: (B, C, H, W) POL特征
            ir_img: (B, 1, H, W) 可选，原始IR图像
            pol_img: (B, 1, H, W) 可选，原始POL图像
        """
        # 分别处理IR和POL特征
        ir_processed = self.ir_branch(ir)  # IR分支：处理高亮信息
        pol_processed = self.pol_branch(pol)  # POL分支：处理纹理信息
        
        # 拼接IR和POL特征
        combined = torch.cat([ir_processed, pol_processed], dim=1)  # [B, 2C, H, W]
        
        # 使用Mamba-SSM建模空间依赖关系，搜寻高亮和纹理区域
        B, C2, H, W = combined.shape
        # 如果空间尺寸太大，先下采样以减少Mamba的序列长度，避免cuDNN错误
        # 降低阈值，更早进行下采样以避免cuDNN错误
        if H * W > 128 * 128:  # 如果超过128x128，先下采样（降低阈值）
            combined_small = F.interpolate(combined, size=(H//2, W//2), mode='bilinear', align_corners=False)
            seq = combined_small.flatten(2).transpose(1, 2)  # [B, HW/4, 2C]
            # 如果序列仍然太长，分块处理
            if seq.size(1) > 65536:
                chunk_size = 32768
                chunks = []
                for i in range(0, seq.size(1), chunk_size):
                    chunk = seq[:, i:i+chunk_size, :]
                    chunk_out = self.mamba_attn(chunk)
                    chunks.append(chunk_out)
                seq_enhanced = torch.cat(chunks, dim=1)
            else:
                seq_enhanced = self.mamba_attn(seq)  # Mamba处理，建模长程依赖
            combined_enhanced_small = seq_enhanced.transpose(1, 2).view(B, C2, H//2, W//2)  # [B, 2C, H/2, W/2]
            combined_enhanced = F.interpolate(combined_enhanced_small, size=(H, W), mode='bilinear', align_corners=False)
        else:
            seq = combined.flatten(2).transpose(1, 2)  # [B, HW, 2C]
            # 如果序列太长，分块处理
            if seq.size(1) > 65536:
                chunk_size = 32768
                chunks = []
                for i in range(0, seq.size(1), chunk_size):
                    chunk = seq[:, i:i+chunk_size, :]
                    chunk_out = self.mamba_attn(chunk)
                    chunks.append(chunk_out)
                seq_enhanced = torch.cat(chunks, dim=1)
            else:
                seq_enhanced = self.mamba_attn(seq)  # Mamba处理，建模长程依赖
            combined_enhanced = seq_enhanced.transpose(1, 2).view(B, C2, H, W)  # [B, 2C, H, W]
        
        # 从IR原图中提取高亮区域图
        if ir_img is None:
            # 如果没有提供原图，使用IR特征的平均值作为替代
            ir_img = ir.mean(dim=1, keepdim=True)  # [B, 1, H_ir, W_ir]
        # 确保ir_img的空间尺寸与combined_enhanced匹配
        if ir_img.shape[2:] != (H, W):
            ir_img = F.interpolate(ir_img, size=(H, W), mode='bilinear', align_corners=False)
        highlight_map = torch.sigmoid(self.highlight_head(ir_img))  # [B, 1, H, W]
        
        # 从POL原图中提取纹理复杂度图
        if pol_img is None:
            # 如果没有提供原图，使用POL特征的平均值作为替代
            pol_img = pol.mean(dim=1, keepdim=True)  # [B, 1, H_pol, W_pol]
        # 确保pol_img的空间尺寸与combined_enhanced匹配
        if pol_img.shape[2:] != (H, W):
            pol_img = F.interpolate(pol_img, size=(H, W), mode='bilinear', align_corners=False)
        texture_map = torch.sigmoid(self.texture_head(pol_img))  # [B, 1, H, W]
        
        # 基于高亮和纹理信息生成自适应融合权重
        # 高亮区域 -> IR高权重，纹理区域 -> POL高权重
        weight_input = torch.cat([combined_enhanced, highlight_map, texture_map], dim=1)  # [B, 2C+2, H, W]
        w_raw = self.weight_gen(weight_input)  # [B, 2, H, W]
        
        
        w = torch.softmax(w_raw, dim=1)  # [B, 2, H, W]
        
        # 增强偏振特征：提取纹理信息（增强版）
        pol_enhanced =self.pol_texture(pol_processed)  # 提高纹理增强强度
        
        # 智能融合：基于Mamba-SSM引导的权重进行融合
        # w[:, 0:1] 对应IR权重，w[:, 1:2] 对应POL权重
        fused = w[:, 0:1] * ir_processed + w[:, 1:2] * pol_enhanced
        
        return fused, w

# ============================================================
# Retinex Decomposition Fusion (Retinex分解融合)
# ============================================================

class FusionHead(nn.Module):
    def __init__(self, ch, use_retinex=False, use_paf=True):
        super().__init__()
        self.use_retinex = use_retinex
        self.use_paf = bool(use_paf)
        
        # Mamba注意力融合模块
        self.att = PolarFusionAttention(ch) if self.use_paf else None
        
        self.out = nn.Conv2d(ch, 1, 3, 1, 1)
        


    def forward(self, ir, pol, ir_img, pol_img):
        """
        Args:
            ir: (B, C, H, W) IR特征
            pol: (B, C, H, W) POL特征
            ir_img: (B, 1, H, W) 可选，原始IR图像（用于Retinex）
            pol_img: (B, 1, H, W) 可选，原始POL图像（用于Retinex）
        Returns:
            fused: (B, 1, H, W) 融合结果
            w: (B, 2, H, W) 融合权重（w[:, 0:1]对应IR，w[:, 1:2]对应POL）
        """
        if self.use_paf:
            # 仅使用 Mamba 注意力融合，传入原图用于提取高亮和纹理图
            fused, w = self.att(ir, pol, ir_img, pol_img)
        else:
            fused = 0.5 * (ir + pol)
            B, _, H, W = fused.shape
            w = fused.new_full((B, 2, H, W), 0.5)
        
        # 输出融合结果
        base_fused = self.out(fused)
        
        
        fused = base_fused
        
        return fused, w

# ============================================================
# Full Model
# ============================================================
class PolarIRS4FusionMamba(nn.Module):
    """
    纯融合模型（无配准）：
    - 强化特征提取：使用增强的编码器提取多尺度特征
    - 优化融合过程：使用Mamba-SSM建模长程依赖，智能融合IR和POL特征
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

        # 增强的特征提取器
        self.encoder = EnhancedConvEncoder()
        
        # 多尺度特征提取器（用于进一步强化特征）
        self.multiscale_extractor_ir = MultiScaleFeatureExtractor(128) if self.use_multiscale else nn.Identity()
        self.multiscale_extractor_pol = MultiScaleFeatureExtractor(128) if self.use_multiscale else nn.Identity()

        # 基于 Mamba 的跨模态融合模块，在特征层面进行深度交互
        self.fusion_mamba = MambaFusionBlock(128) if self.use_cross_mamba else None
        
        # 融合头（使用Retinex增强）
        self.fuse_head = FusionHead(128, use_retinex=True, use_paf=use_paf)
        
        # 红外高亮注入模块，确保人体/高亮目标更明显地进入最终融合结果
        self.ir_inject = (
            IRHighlightInjector(
                init_thresh=ihj_thresh,
                init_sharpness=ihj_sharpness,
                inject_ratio_base=ihj_inject_ratio,
            )
            if self.use_ihj
            else None
        )
        
        # 偏振纹理增强模块，专门强化偏振图像的纹理细节
        self.polar_texture_enhancer = PolarTextureEnhancer() if self.use_ptj else None

        # 是否保存中间结果
        self.save_intermediate = save_intermediate

        # 减少显存占用：可选地对重算开销较大的块使用 gradient checkpoint
        self.use_checkpoint = use_checkpoint

    def forward(self, ir, pol):
        """
        纯融合前向传播（无配准）：
        1. 增强特征提取
        2. 多尺度特征强化
        3. Mamba跨模态交互
        4. 智能融合
        5. 后处理增强
        """
        # ========== 阶段1：增强特征提取 ==========
        # 提取多尺度特征
        f1_ir, f2_ir, f3_ir = self.encoder(ir)
        f1_pol, f2_pol, f3_pol = self.encoder(pol)
        
        # 使用最高层特征（f3）进行融合
        f_ir = f3_ir  # (B, 128, H/4, W/4)
        f_pol = f3_pol  # (B, 128, H/4, W/4)

        # ========== 阶段2：多尺度特征强化 ==========
        # 对IR和POL特征进行多尺度增强
        f_ir = self.multiscale_extractor_ir(f_ir)
        f_pol = self.multiscale_extractor_pol(f_pol)

        # ========== 阶段4：跨模态Mamba融合 ==========
        # 在特征层面进行深度交互
        if self.use_cross_mamba and self.use_checkpoint:
            f_ir_fused, f_pol_fused = checkpoint(
                lambda a, b: self.fusion_mamba(a, b), f_ir, f_pol
            )
        elif self.use_cross_mamba:
            f_ir_fused, f_pol_fused = self.fusion_mamba(f_ir, f_pol)
        else:
            f_ir_fused, f_pol_fused = f_ir, f_pol

        # ========== 阶段5：上采样到输入分辨率 ==========
        H, W = ir.shape[-2], ir.shape[-1]
        f_ir_fused = F.interpolate(f_ir_fused, size=(H, W), mode='bilinear', align_corners=False)
        f_pol_fused = F.interpolate(f_pol_fused, size=(H, W), mode='bilinear', align_corners=False)

        # ========== 阶段6：智能融合 ==========
        # 使用融合头进行智能融合（传入原始图像用于Retinex）
        if self.use_checkpoint:
            fuse_feat, w = checkpoint(
                lambda a, b, c, d: self.fuse_head(a, b, c, d), 
                f_ir_fused, f_pol_fused, ir, pol
            )
        else:
            fuse_feat, w = self.fuse_head(f_ir_fused, f_pol_fused, ir, pol)
        
        # ========== 阶段7：后处理增强 ==========
        # 红外高亮注入（突出人体/高亮目标）
        if self.use_ihj:
            fuse_feat, m_ir = self.ir_inject(fuse_feat, ir)
        else:
            m_ir = None

        # 偏振纹理增强（在非高亮区域强化纹理）
        if self.use_ptj:
            fuse_feat = self.polar_texture_enhancer(fuse_feat, pol, m_ir)


        # 亮度归一化：平衡IR和POL的亮度，避免整体过暗
        ir_mean = ir.mean(dim=[2, 3], keepdim=True).detach()
        pol_mean = pol.mean(dim=[2, 3], keepdim=True).detach()
        ir_std = ir.std(dim=[2, 3], keepdim=True).detach()
        pol_std = pol.std(dim=[2, 3], keepdim=True).detach()

        if m_ir is not None:
            # 在高亮区域（人体）更接近IR，在背景区域更接近POL
            target_mean = m_ir * (0.6 * ir_mean + 0.4 * pol_mean) + (1.0 - m_ir) * (0.2 * ir_mean + 0.8 * pol_mean)
            target_std = m_ir * (0.6 * ir_std + 0.4 * pol_std) + (1.0 - m_ir) * (0.2 * ir_std + 0.8 * pol_std)
        else:
            # 如果没有高亮掩码，使用平衡的权重（背景更接近POL）
            target_mean = 0.3 * ir_mean + 0.7 * pol_mean  # 背景更接近POL
            target_std = 0.3 * ir_std + 0.7 * pol_std
        
        fuse_mean = fuse_feat.mean(dim=[2, 3], keepdim=True)
        fuse_std = fuse_feat.std(dim=[2, 3], keepdim=True)
        
        # 归一化到目标均值和标准差范围，但保留相对亮度关系
        fuse_feat_normalized = (fuse_feat - fuse_mean) / (fuse_std + 1e-6) * target_std + target_mean
        
        # 如果启用中间结果保存，在这里保存归一化之前和之后的值
        if self.save_intermediate:
            # 保存归一化之前的 fuse_mean 和 fuse_std，以及归一化之后的 fuse_feat
            if not hasattr(self, '_intermediate_buffer'):
                self._intermediate_buffer = {}
            self._intermediate_buffer['polar_texture_enhancer'] = {
                "fuse_mean": fuse_mean.detach().clone(),
                "fuse_std": fuse_std.detach().clone(),
                "fuse_feat": fuse_feat_normalized.detach().clone()
            }
        
        fuse_feat = fuse_feat_normalized
        
        # 限制在合理范围内，避免过亮或过暗（clamp 不会阻断梯度）
        fuse_feat = torch.clamp(fuse_feat, 0.0, 1.0)
        
        # 数值稳定性：检查并修复NaN和Inf
        if torch.isnan(fuse_feat).any() or torch.isinf(fuse_feat).any():
            # 使用IR和POL的加权平均作为fallback
            # 确保ir和pol的空间尺寸与fuse_feat一致
            target_size = fuse_feat.shape[-2:]  # (H, W)
            
            # 如果ir的空间尺寸不匹配，进行插值
            if ir.shape[-2:] != target_size:
                ir = F.interpolate(ir, size=target_size, mode='bilinear', align_corners=False)
            
            # 如果pol的空间尺寸不匹配，进行插值
            if pol.shape[-2:] != target_size:
                pol = F.interpolate(pol, size=target_size, mode='bilinear', align_corners=False)
            
            # 确保ir和pol的空间尺寸一致（以防万一）
            if ir.shape[-2:] != pol.shape[-2:]:
                # 以fuse_feat的尺寸为准
                if ir.shape[-2:] == target_size:
                    pol = F.interpolate(pol, size=target_size, mode='bilinear', align_corners=False)
                elif pol.shape[-2:] == target_size:
                    ir = F.interpolate(ir, size=target_size, mode='bilinear', align_corners=False)
                else:
                    # 如果都不匹配，统一调整到fuse_feat的尺寸
                    ir = F.interpolate(ir, size=target_size, mode='bilinear', align_corners=False)
                    pol = F.interpolate(pol, size=target_size, mode='bilinear', align_corners=False)
            
            ir_weight_fallback = 0.5
            pol_weight_fallback = 0.5
            fallback = ir_weight_fallback * ir + pol_weight_fallback * pol
            fuse_feat = torch.where(torch.isnan(fuse_feat) | torch.isinf(fuse_feat), 
                                    fallback, fuse_feat)
            fuse_feat = torch.clamp(fuse_feat, 0.0, 1.0)

        
        # ========== 阶段8：构建返回字典 ==========
        output_dict = {
            "fusion": fuse_feat,  # 融合结果
            "ir_reg": ir,  # 原始IR图像（用于损失计算兼容性）
            "pol_reg": pol,  # 原始POL图像（用于损失计算兼容性）
            "attn": w,  # 融合注意力权重（w[:, 0:1]对应IR，w[:, 1:2]对应POL）
            "ir_highlight_mask": m_ir  # 红外高亮注入掩码（用于可视化/调试）
        }
        
        # 如果启用中间结果保存，添加中间结果
        if self.save_intermediate:
            intermediate_results = {}
            
            # 从缓冲区获取保存的中间结果
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
