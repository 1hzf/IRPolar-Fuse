"""
Utility-function module.
Contains utilities related to image processing and registration.
"""

import torch
import torch.nn.functional as F


def gradient(x):
    """
    Compute first-order finite-difference gradients of x in the x and y directions,
    using padding to preserve the original spatial dimensions. This prevents size
    mismatches when dx and dy are used together later.
    """
    # X-direction gradient: take differences along the width and pad zero on the right.
    dx = x[..., :, 1:] - x[..., :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0))  # pad (left, right, top, bottom)

    # Y-direction gradient: take differences along the height and pad zero at the bottom.
    dy = x[..., 1:, :] - x[..., :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))

    return dx, dy


def warp(x, flow):
    """
    Warp an image using optical flow.
    
    Args:
        x: input image (B, C, H, W)
        flow: optical-flow field (B, 2, H, W), with channel 0 for x and channel 1 for y
    
    Returns:
        warped: warped image (B, C, H, W)
    """
    B, C, H, W = x.shape
    # If the flow size differs from the input, interpolate it and scale pixel displacements proportionally.
    if flow.shape[2] != H or flow.shape[3] != W:
        Hf, Wf = flow.shape[2], flow.shape[3]
        # Compute scale factors: the x channel corresponds to width and the y channel to height.
        scale_x = float(W) / float(Wf)
        scale_y = float(H) / float(Hf)
        # Bilinearly interpolate to the target spatial size.
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        # Scale displacement by direction; channel order [B,2,H,W] is 0->x and 1->y.
        flow[:, 0, :, :] = flow[:, 0, :, :] * scale_x
        flow[:, 1, :, :] = flow[:, 1, :, :] * scale_y

    # Construct a sampling grid with the same spatial dimensions as x.
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

    # Use align_corners=False consistently with affine_grid to avoid geometric shifts.
    return F.grid_sample(
        x,
        grid + flow_norm,
        mode='bilinear',
        align_corners=False,
        padding_mode='border'
    )
