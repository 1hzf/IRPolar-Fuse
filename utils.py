"""
工具函数模块
包含图像处理和配准相关的工具函数
"""

import torch
import torch.nn.functional as F


def gradient(x):
    """
    计算 x 在 x / y 方向的一阶差分梯度，并通过 padding 保持与原图相同的空间尺寸。
    这样后续在同时使用 dx, dy 时不会出现尺寸不匹配的问题。
    """
    # x 方向梯度：在宽度方向做差分，然后在最右侧补 0
    dx = x[..., :, 1:] - x[..., :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0))  # pad (left, right, top, bottom)

    # y 方向梯度：在高度方向做差分，然后在最下方补 0
    dy = x[..., 1:, :] - x[..., :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))

    return dx, dy


def warp(x, flow):
    """
    使用光流对图像进行warp操作
    
    Args:
        x: 输入图像 (B, C, H, W)
        flow: 光流场 (B, 2, H, W)，其中通道0是x方向，通道1是y方向
    
    Returns:
        warped: warp后的图像 (B, C, H, W)
    """
    B, C, H, W = x.shape
    # 如果 flow 的空间尺寸与输入不一致，则先插值到输入尺寸并按比例缩放像素位移
    if flow.shape[2] != H or flow.shape[3] != W:
        Hf, Wf = flow.shape[2], flow.shape[3]
        # 计算缩放因子（x 通道对应宽度，y 通道对应高度）
        scale_x = float(W) / float(Wf)
        scale_y = float(H) / float(Hf)
        # 双线性插值到目标空间
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        # 按方向缩放位移量（注意通道顺序为 [B,2,H,W]：0->x, 1->y）
        flow[:, 0, :, :] = flow[:, 0, :, :] * scale_x
        flow[:, 1, :, :] = flow[:, 1, :, :] * scale_y

    # 构造采样网格（与 x 的空间尺寸一致）
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2) x,y order

    # flow: (B,2,H,W) -> (B,H,W,2) with x,y channels
    flow = flow.permute(0, 2, 3, 1)

    # convert pixel flow -> normalized [-1,1] relative to width/height
    flow_x = flow[..., 0]  # (B,H,W)
    flow_y = flow[..., 1]  # (B,H,W)
    norm_x = flow_x / ((W - 1) / 2.0 + 1e-8)
    norm_y = flow_y / ((H - 1) / 2.0 + 1e-8)
    flow_norm = torch.stack([norm_x, norm_y], dim=-1)  # (B,H,W,2)

    # 使用 align_corners=False 与 affine_grid 保持一致，避免几何偏移
    return F.grid_sample(
        x,
        grid + flow_norm,
        mode='bilinear',
        align_corners=False,
        padding_mode='border'
    )

