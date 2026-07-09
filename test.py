import os
import argparse
from pathlib import Path
# 限制本进程只可见物理 GPU 2（进程内该 GPU 将成为 cuda:0）
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')

import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from thop import profile

from polar_ir_s4fusion_mamba import PolarIRS4FusionMamba

# ============================================================
# 图像加载工具函数
# ============================================================

def load_image_as_tensor(path: str):
    """Load grayscale image to tensor [1,1,H,W] in [0,1]."""
    img = Image.open(path).convert("L")
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t


# ============================================================
# 融合推理函数
# ============================================================

@torch.no_grad()
def run_fusion(model, data_root, out_dir, device):
    """
    对指定路径下的图像进行融合并保存结果
    输入路径应包含 inf/ 和 pol/ 两个子文件夹
    输出：保存单张融合结果图和中间模块的输入输出
    """
    model.eval()
    inf_dir = Path(data_root) / "ir"
    pol_dir = Path(data_root) / "vi"
    
    # 检查输入目录是否存在
    if not inf_dir.exists():
        raise ValueError(f"输入目录不存在: {inf_dir}")
    if not pol_dir.exists():
        raise ValueError(f"输入目录不存在: {pol_dir}")
    
    # 获取所有图像文件名
    names = sorted([p.name for p in inf_dir.iterdir() 
                   if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']])
    
    if len(names) == 0:
        print(f"警告: 在 {inf_dir} 中未找到图像文件")
        return
    
    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    print(f"找到 {len(names)} 张图像，开始融合...")
    
    for name in tqdm(names, desc="处理图像"):
        inf_path = inf_dir / name
        pol_path = pol_dir / name
        
        if not pol_path.exists():
            print(f"跳过 {name}，POL文件不存在")
            continue
        
        try:
            # 加载图像
            ir = load_image_as_tensor(str(inf_path)).to(device)
            pol = load_image_as_tensor(str(pol_path)).to(device)
            
            # 每组数据计算模型参数量与 FLOPs
            try:
                flops, params = profile(model, inputs=(ir, pol), verbose=False)
                print(f"{name} 计算复杂度: {flops / 1e9:.4f} GFLOPs，参数量: {params / 1e6:.4f} M")
            except Exception as e:
                print(f"{name} 计算 FLOPs/参数量失败: {e}")
            
            # 执行融合
            t_start = time.perf_counter()
            out = model(ir, pol)
            duration_ms = (time.perf_counter() - t_start) * 1000
            fusion = out["fusion"].clamp(0, 1).cpu()
            
            # 检查融合结果
            fusion_min = fusion.min().item()
            fusion_max = fusion.max().item()
            fusion_mean = fusion.mean().item()
            
            # 保存单张融合结果（不拼接）
            save_image(fusion[0], os.path.join(out_dir, f"fusion_{name}"), normalize=False)
            
            print(f"  {name}: 融合完成 [范围: {fusion_min:.4f}~{fusion_max:.4f}, 均值: {fusion_mean:.4f}]，处理时间: {duration_ms:.2f} ms")
            
        except Exception as e:
            print(f"处理 {name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n融合完成！结果保存在: {out_dir}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="使用训练好的模型进行图像融合")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重文件路径 (.pth)")
    parser.add_argument("--data_root", type=str, required=True, help="输入数据根目录（包含 inf/ 和 pol/ 子文件夹）")
    parser.add_argument("--out_dir", type=str, default="MSRS", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="运行设备")
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"模型文件不存在: {args.ckpt}")
    
    # 设置设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    print(f"加载模型权重: {args.ckpt}")
    
    # 创建模型（不保存中间结果）
    model = PolarIRS4FusionMamba(save_intermediate=False).to(device)
    
    # 加载权重
    try:
        checkpoint = torch.load(args.ckpt, map_location=device)
        
        # 兼容不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("  ✓ 加载模型权重 (model_state_dict)")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("  ✓ 加载模型权重 (state_dict)")
            else:
                # 尝试直接加载整个字典作为state_dict
                model.load_state_dict(checkpoint, strict=False)
                print("  ✓ 加载模型权重 (直接字典)")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("  ✓ 加载模型权重 (直接state_dict)")
        
        print("模型加载成功！")
        
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        raise
    
    # 执行融合
    print(f"\n开始融合，输入路径: {args.data_root}")
    run_fusion(model, args.data_root, args.out_dir, device)


if __name__ == "__main__":
    main()



# import os
# import argparse
# from pathlib import Path
# # 限制本进程只可见物理 GPU 2（进程内该 GPU 将成为 cuda:0）
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')

# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from torchvision.utils import save_image
# from PIL import Image
# from tqdm import tqdm

# from polar_ir_s4fusion_mamba import PolarIRS4FusionMamba

# # ============================================================
# # 图像加载工具函数
# # ============================================================

# def load_image_as_tensor(path: str):
#     """Load grayscale image to tensor [1,1,H,W] in [0,1]."""
#     img = Image.open(path).convert("L")
#     t = transforms.ToTensor()(img).unsqueeze(0)
#     return t


# # ============================================================
# # 融合推理函数
# # ============================================================

# @torch.no_grad()
# def run_fusion(model, data_root, out_dir, device):
#     """
#     对指定路径下的图像进行融合并保存结果
#     输入路径应包含 inf/ 和 pol/ 两个子文件夹
#     输出：保存单张融合结果图和中间模块的输入输出
#     """
#     model.eval()
#     inf_dir = Path(data_root) / "MSRS/ir"
#     pol_dir = Path(data_root) / "MSRS/vi"
    
#     # 检查输入目录是否存在
#     if not inf_dir.exists():
#         raise ValueError(f"输入目录不存在: {inf_dir}")
#     if not pol_dir.exists():
#         raise ValueError(f"输入目录不存在: {pol_dir}")
    
#     # 获取所有图像文件名
#     names = sorted([p.name for p in inf_dir.iterdir() 
#                    if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']])
    
#     if len(names) == 0:
#         print(f"警告: 在 {inf_dir} 中未找到图像文件")
#         return
    
#     # 创建输出目录
#     os.makedirs(out_dir, exist_ok=True)
#     # 创建中间结果目录
#     intermediate_dir = os.path.join(out_dir, "intermediate")
#     os.makedirs(intermediate_dir, exist_ok=True)
    
#     print(f"找到 {len(names)} 张图像，开始融合...")
    
#     for name in tqdm(names, desc="处理图像"):
#         inf_path = inf_dir / name
#         pol_path = pol_dir / name
        
#         if not pol_path.exists():
#             print(f"跳过 {name}，POL文件不存在")
#             continue
        
#         try:
#             # 加载图像
#             ir = load_image_as_tensor(str(inf_path)).to(device)
#             pol = load_image_as_tensor(str(pol_path)).to(device)
            
#             # 执行融合
#             out = model(ir, pol)
#             fusion = out["fusion"].clamp(0, 1).cpu()
            
#             # 检查融合结果
#             fusion_min = fusion.min().item()
#             fusion_max = fusion.max().item()
#             fusion_mean = fusion.mean().item()
            
#             # 保存单张融合结果（不拼接）
#             save_image(fusion[0], os.path.join(out_dir, f"fusion_{name}"), normalize=False)
            
#             # 保存中间模块的输入输出
#             if "intermediate_results" in out:
#                 base_name = os.path.splitext(name)[0]
                
#                 # 辅助函数：将特征图归一化到[0,1]范围
#                 def normalize_feat(feat):
#                     """将特征图归一化到[0,1]范围"""
#                     if feat is None:
#                         return None
#                     feat = feat.cpu()
#                     if feat.shape[1] > 1:
#                         feat = feat.mean(dim=1, keepdim=True)
#                     feat_min = feat.min()
#                     feat_max = feat.max()
#                     if feat_max > feat_min:
#                         feat = (feat - feat_min) / (feat_max - feat_min)
#                     else:
#                         feat = torch.zeros_like(feat)
#                     return feat.clamp(0, 1)
                
#                 # 1. FusionHead 的输入输出
#                 if "fusion_head" in out["intermediate_results"]:
#                     fh = out["intermediate_results"]["fusion_head"]
#                     fh_dir = os.path.join(intermediate_dir, "fusion_head", base_name)
#                     os.makedirs(fh_dir, exist_ok=True)
                    
#                     # 输入：IR特征和POL特征（需要转换为单通道图像）
#                     if "input_ir" in fh:
#                         ir_feat = normalize_feat(fh["input_ir"])
#                         if ir_feat is not None:
#                             save_image(ir_feat[0], os.path.join(fh_dir, "input_ir.png"), normalize=False)
                    
#                     if "input_pol" in fh:
#                         pol_feat = normalize_feat(fh["input_pol"])
#                         if pol_feat is not None:
#                             save_image(pol_feat[0], os.path.join(fh_dir, "input_pol.png"), normalize=False)
                    
#                     # 输入：原始IR和POL图像
#                     if "input_ir_img" in fh and fh["input_ir_img"] is not None:
#                         save_image(fh["input_ir_img"][0].cpu().clamp(0, 1), 
#                                   os.path.join(fh_dir, "input_ir_img.png"), normalize=False)
                    
#                     if "input_pol_img" in fh and fh["input_pol_img"] is not None:
#                         save_image(fh["input_pol_img"][0].cpu().clamp(0, 1), 
#                                   os.path.join(fh_dir, "input_pol_img.png"), normalize=False)
                    
#                     # 输出：融合结果
#                     if "output_fused" in fh:
#                         save_image(fh["output_fused"][0].cpu().clamp(0, 1), 
#                                   os.path.join(fh_dir, "output_fused.png"), normalize=False)
                    
#                     # 权重图：IR权重和POL权重（已经是[0,1]范围）
#                     if "output_weights" in fh:
#                         w = fh["output_weights"].cpu()
#                         save_image(w[0, 0:1], os.path.join(fh_dir, "weight_ir.png"), normalize=False)
#                         save_image(w[0, 1:2], os.path.join(fh_dir, "weight_pol.png"), normalize=False)
                
#                 # 2. IRHighlightInjector 的输入输出
#                 if "ir_highlight_injector" in out["intermediate_results"]:
#                     iri = out["intermediate_results"]["ir_highlight_injector"]
#                     iri_dir = os.path.join(intermediate_dir, "ir_highlight_injector", base_name)
#                     os.makedirs(iri_dir, exist_ok=True)
                    
#                     # 输入：融合结果和IR图像
#                     if "input_fused" in iri:
#                         save_image(iri["input_fused"][0].cpu().clamp(0, 1), 
#                                   os.path.join(iri_dir, "input_fused.png"), normalize=False)
#                     if "input_ir" in iri:
#                         save_image(iri["input_ir"][0].cpu().clamp(0, 1), 
#                                   os.path.join(iri_dir, "input_ir.png"), normalize=False)
                    
#                     # 输出：融合结果和掩码
#                     if "output_fused" in iri:
#                         save_image(iri["output_fused"][0].cpu().clamp(0, 1), 
#                                   os.path.join(iri_dir, "output_fused.png"), normalize=False)
#                     if "output_mask" in iri and iri["output_mask"] is not None:
#                         save_image(iri["output_mask"][0].cpu().clamp(0, 1), 
#                                   os.path.join(iri_dir, "output_mask.png"), normalize=False)
                
#                 # 3. PolarTextureEnhancer 的输入输出
#                 if "polar_texture_enhancer" in out["intermediate_results"]:
#                     pte = out["intermediate_results"]["polar_texture_enhancer"]
#                     pte_dir = os.path.join(intermediate_dir, "polar_texture_enhancer", base_name)
#                     os.makedirs(pte_dir, exist_ok=True)
                    
#                     # 输入：融合结果、POL图像和掩码
#                     if "input_fused" in pte:
#                         save_image(pte["input_fused"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "input_fused.png"), normalize=False)
#                     if "input_pol" in pte:
#                         save_image(pte["input_pol"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "input_pol.png"), normalize=False)
#                     if "input_mask" in pte and pte["input_mask"] is not None:
#                         save_image(pte["input_mask"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "input_mask.png"), normalize=False)
                    
#                     # 输出：融合结果
#                     if "output_fused" in pte:
#                         save_image(pte["output_fused"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "output_fused.png"), normalize=False)
                    
#                     # 输出：纹理残差
#                     if "texture_residual" in pte:
#                         save_image(pte["texture_residual"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "texture_residual.png"), normalize=False)
                    
#                     # 输出：fuse_mean, fuse_std 和 fuse_feat（在 polar_texture_enhancer 之后）
#                     if "fuse_mean" in pte:
#                         # fuse_mean 是 [B, 1, 1, 1] 的形状
#                         fuse_mean_val = pte["fuse_mean"][0, 0, 0, 0].cpu().item()
#                         # 创建一个可视化图像
#                         fuse_mean_img = torch.full((1, 1, 64, 64), fuse_mean_val, dtype=torch.float32)
#                         save_image(fuse_mean_img, os.path.join(pte_dir, "fuse_mean.png"), normalize=False)
#                         # 同时保存数值到文本文件
#                         with open(os.path.join(pte_dir, "fuse_mean.txt"), "w") as f:
#                             f.write(f"{fuse_mean_val:.6f}\n")
                    
#                     if "fuse_std" in pte:
#                         # fuse_std 是 [B, 1, 1, 1] 的形状
#                         fuse_std_val = pte["fuse_std"][0, 0, 0, 0].cpu().item()
#                         # 创建一个可视化图像
#                         fuse_std_img = torch.full((1, 1, 64, 64), fuse_std_val, dtype=torch.float32)
#                         save_image(fuse_std_img, os.path.join(pte_dir, "fuse_std.png"), normalize=False)
#                         # 同时保存数值到文本文件
#                         with open(os.path.join(pte_dir, "fuse_std.txt"), "w") as f:
#                             f.write(f"{fuse_std_val:.6f}\n")
                    
#                     # fuse_feat 是在注释 "# 归一化到目标均值和标准差范围，但保留相对亮度关系" 之后（归一化之后）的特征
#                     if "fuse_feat" in pte:
#                         # fuse_feat 是 [B, 1, H, W] 的形状，归一化之后的特征
#                         save_image(pte["fuse_feat"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "fuse_feat.png"), normalize=False)
            
#             print(f"  {name}: 融合完成 [范围: {fusion_min:.4f}~{fusion_max:.4f}, 均值: {fusion_mean:.4f}]")
            
#         except Exception as e:
#             print(f"处理 {name} 时出错: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
    
#     print(f"\n融合完成！结果保存在: {out_dir}")
#     print(f"中间结果保存在: {intermediate_dir}")


# # ============================================================
# # 主函数
# # ============================================================

# def main():
#     parser = argparse.ArgumentParser(description="使用训练好的模型进行图像融合")
#     parser.add_argument("--ckpt", type=str, required=True, help="模型权重文件路径 (.pth)")
#     parser.add_argument("--data_root", type=str, required=True, help="输入数据根目录（包含 inf/ 和 pol/ 子文件夹）")
#     parser.add_argument("--out_dir", type=str, default="test_results", help="输出目录")
#     parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="运行设备")
    
#     args = parser.parse_args()
    
#     # 检查模型文件是否存在
#     if not os.path.exists(args.ckpt):
#         raise FileNotFoundError(f"模型文件不存在: {args.ckpt}")
    
#     # 设置设备
#     if args.device == "cuda" and not torch.cuda.is_available():
#         print("警告: CUDA不可用，使用CPU")
#         device = "cpu"
#     else:
#         device = args.device
    
#     print(f"使用设备: {device}")
#     print(f"加载模型权重: {args.ckpt}")
    
#     # 创建模型（启用中间结果保存）
#     model = PolarIRS4FusionMamba(save_intermediate=True).to(device)
    
#     # 加载权重
#     try:
#         checkpoint = torch.load(args.ckpt, map_location=device)
        
#         # 兼容不同的checkpoint格式
#         if isinstance(checkpoint, dict):
#             if 'model_state_dict' in checkpoint:
#                 model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#                 print("  ✓ 加载模型权重 (model_state_dict)")
#             elif 'state_dict' in checkpoint:
#                 model.load_state_dict(checkpoint['state_dict'], strict=False)
#                 print("  ✓ 加载模型权重 (state_dict)")
#             else:
#                 # 尝试直接加载整个字典作为state_dict
#                 model.load_state_dict(checkpoint, strict=False)
#                 print("  ✓ 加载模型权重 (直接字典)")
#         else:
#             model.load_state_dict(checkpoint, strict=False)
#             print("  ✓ 加载模型权重 (直接state_dict)")
        
#         print("模型加载成功！")
        
#     except Exception as e:
#         print(f"加载模型权重时出错: {e}")
#         raise
    
#     # 执行融合
#     print(f"\n开始融合，输入路径: {args.data_root}")
#     run_fusion(model, args.data_root, args.out_dir, device)


# if __name__ == "__main__":
#     main()

