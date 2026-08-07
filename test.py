import os
import argparse
from pathlib import Path
# Restrict this process to physical GPU 2, which is exposed as cuda:0 within the process.
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
# Image-loading utility function
# ============================================================

def load_image_as_tensor(path: str):
    """Load grayscale image to tensor [1,1,H,W] in [0,1]."""
    img = Image.open(path).convert("L")
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t


# ============================================================
# Fusion inference function
# ============================================================

@torch.no_grad()
def run_fusion(model, data_root, out_dir, device):
    """
    Fuse images under the specified path and save the results.
    The input path should contain inf/ and pol/ subdirectories.
    Output: individual fusion results and the inputs and outputs of intermediate modules.
    """
    model.eval()
    inf_dir = Path(data_root) / "ir"
    pol_dir = Path(data_root) / "vi"
    
    # Check whether the input directories exist.
    if not inf_dir.exists():
        raise ValueError(f"Input directory does not exist: {inf_dir}")
    if not pol_dir.exists():
        raise ValueError(f"Input directory does not exist: {pol_dir}")
    
    # Get all image filenames.
    names = sorted([p.name for p in inf_dir.iterdir() 
                   if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']])
    
    if len(names) == 0:
        print(f"Warning: No image files were found in {inf_dir}")
        return
    
    # Create the output directory.
    os.makedirs(out_dir, exist_ok=True)
    print(f"Found {len(names)} images; starting fusion...")
    
    for name in tqdm(names, desc="Processing images"):
        inf_path = inf_dir / name
        pol_path = pol_dir / name
        
        if not pol_path.exists():
            print(f"Skipping {name}: POL file does not exist")
            continue
        
        try:
            # Load the images.
            ir = load_image_as_tensor(str(inf_path)).to(device)
            pol = load_image_as_tensor(str(pol_path)).to(device)
            
            # Compute the model parameter count and FLOPs for each image pair.
            try:
                flops, params = profile(model, inputs=(ir, pol), verbose=False)
                print(f"{name} computational complexity: {flops / 1e9:.4f} GFLOPs, parameter count: {params / 1e6:.4f} M")
            except Exception as e:
                print(f"Failed to compute FLOPs/parameter count for {name}: {e}")
            
            # Perform fusion.
            t_start = time.perf_counter()
            out = model(ir, pol)
            duration_ms = (time.perf_counter() - t_start) * 1000
            fusion = out["fusion"].clamp(0, 1).cpu()
            
            # Check the fusion result.
            fusion_min = fusion.min().item()
            fusion_max = fusion.max().item()
            fusion_mean = fusion.mean().item()
            
            # Save the individual fusion result without concatenation.
            save_image(fusion[0], os.path.join(out_dir, f"fusion_{name}"), normalize=False)
            
            print(f"  {name}: Fusion complete [range: {fusion_min:.4f}~{fusion_max:.4f}, mean: {fusion_mean:.4f}], processing time: {duration_ms:.2f} ms")
            
        except Exception as e:
            print(f"Error while processing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nFusion complete. Results were saved to: {out_dir}")


# ============================================================
# Main function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Perform image fusion using a trained model")
    parser.add_argument("--ckpt", type=str, required=True, help="path to the model weights file (.pth)")
    parser.add_argument("--data_root", type=str, required=True, help="input data root directory containing inf/ and pol/ subdirectories")
    parser.add_argument("--out_dir", type=str, default="MSRS", help="output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="execution device")
    
    args = parser.parse_args()
    
    # Check whether the model file exists.
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Model file does not exist: {args.ckpt}")
    
    # Configure the device.
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is unavailable; using CPU")
        device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Loading model weights: {args.ckpt}")
    
    # Create the model without saving intermediate results.
    model = PolarIRS4FusionMamba(save_intermediate=False).to(device)
    
    # Load the weights.
    try:
        checkpoint = torch.load(args.ckpt, map_location=device)
        
        # Support different checkpoint formats.
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("  ✓ Loaded model weights (model_state_dict)")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("  ✓ Loaded model weights (state_dict)")
            else:
                # Attempt to load the entire dictionary directly as a state_dict.
                model.load_state_dict(checkpoint, strict=False)
                print("  ✓ Loaded model weights (dictionary directly)")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("  ✓ Loaded model weights (state_dict directly)")
        
        print("Model loaded successfully.")
        
    except Exception as e:
        print(f"Error while loading model weights: {e}")
        raise
    
    # Perform fusion.
    print(f"\nStarting fusion; input path: {args.data_root}")
    run_fusion(model, args.data_root, args.out_dir, device)


if __name__ == "__main__":
    main()



# import os
# import argparse
# from pathlib import Path
# # Restrict this process to physical GPU 2, which is exposed as cuda:0 within the process.
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
# # Image-loading utility function
# # ============================================================

# def load_image_as_tensor(path: str):
#     """Load grayscale image to tensor [1,1,H,W] in [0,1]."""
#     img = Image.open(path).convert("L")
#     t = transforms.ToTensor()(img).unsqueeze(0)
#     return t


# # ============================================================
# # Fusion inference function
# # ============================================================

# @torch.no_grad()
# def run_fusion(model, data_root, out_dir, device):
#     """
#     Fuse images under the specified path and save the results.
#     The input path should contain inf/ and pol/ subdirectories.
#     Output: individual fusion results and the inputs and outputs of intermediate modules.
#     """
#     model.eval()
#     inf_dir = Path(data_root) / "MSRS/ir"
#     pol_dir = Path(data_root) / "MSRS/vi"
    
#     # Check whether the input directories exist.
#     if not inf_dir.exists():
#         raise ValueError(f"Input directory does not exist: {inf_dir}")
#     if not pol_dir.exists():
#         raise ValueError(f"Input directory does not exist: {pol_dir}")
    
#     # Get all image filenames.
#     names = sorted([p.name for p in inf_dir.iterdir() 
#                    if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']])
    
#     if len(names) == 0:
#         print(f"Warning: No image files were found in {inf_dir}")
#         return
    
#     # Create the output directory.
#     os.makedirs(out_dir, exist_ok=True)
#     # Create the intermediate-results directory.
#     intermediate_dir = os.path.join(out_dir, "intermediate")
#     os.makedirs(intermediate_dir, exist_ok=True)
    
#     print(f"Found {len(names)} images; starting fusion...")
    
#     for name in tqdm(names, desc="Processing images"):
#         inf_path = inf_dir / name
#         pol_path = pol_dir / name
        
#         if not pol_path.exists():
#             print(f"Skipping {name}: POL file does not exist")
#             continue
        
#         try:
#             # Load the images.
#             ir = load_image_as_tensor(str(inf_path)).to(device)
#             pol = load_image_as_tensor(str(pol_path)).to(device)
            
#             # Perform fusion.
#             out = model(ir, pol)
#             fusion = out["fusion"].clamp(0, 1).cpu()
            
#             # Check the fusion result.
#             fusion_min = fusion.min().item()
#             fusion_max = fusion.max().item()
#             fusion_mean = fusion.mean().item()
            
#             # Save the individual fusion result without concatenation.
#             save_image(fusion[0], os.path.join(out_dir, f"fusion_{name}"), normalize=False)
            
#             # Save the inputs and outputs of intermediate modules.
#             if "intermediate_results" in out:
#                 base_name = os.path.splitext(name)[0]
                
#                 # Helper function: normalize feature maps to [0,1].
#                 def normalize_feat(feat):
#                     """Normalize feature maps to [0,1]."""
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
                
#                 # 1. FusionHead inputs and outputs
#                 if "fusion_head" in out["intermediate_results"]:
#                     fh = out["intermediate_results"]["fusion_head"]
#                     fh_dir = os.path.join(intermediate_dir, "fusion_head", base_name)
#                     os.makedirs(fh_dir, exist_ok=True)
                    
#                     # Inputs: IR and POL features, which must be converted to single-channel images.
#                     if "input_ir" in fh:
#                         ir_feat = normalize_feat(fh["input_ir"])
#                         if ir_feat is not None:
#                             save_image(ir_feat[0], os.path.join(fh_dir, "input_ir.png"), normalize=False)
                    
#                     if "input_pol" in fh:
#                         pol_feat = normalize_feat(fh["input_pol"])
#                         if pol_feat is not None:
#                             save_image(pol_feat[0], os.path.join(fh_dir, "input_pol.png"), normalize=False)
                    
#                     # Inputs: original IR and POL images.
#                     if "input_ir_img" in fh and fh["input_ir_img"] is not None:
#                         save_image(fh["input_ir_img"][0].cpu().clamp(0, 1), 
#                                   os.path.join(fh_dir, "input_ir_img.png"), normalize=False)
                    
#                     if "input_pol_img" in fh and fh["input_pol_img"] is not None:
#                         save_image(fh["input_pol_img"][0].cpu().clamp(0, 1), 
#                                   os.path.join(fh_dir, "input_pol_img.png"), normalize=False)
                    
#                     # Output: fusion result.
#                     if "output_fused" in fh:
#                         save_image(fh["output_fused"][0].cpu().clamp(0, 1), 
#                                   os.path.join(fh_dir, "output_fused.png"), normalize=False)
                    
#                     # Weight maps: IR and POL weights, already within [0,1].
#                     if "output_weights" in fh:
#                         w = fh["output_weights"].cpu()
#                         save_image(w[0, 0:1], os.path.join(fh_dir, "weight_ir.png"), normalize=False)
#                         save_image(w[0, 1:2], os.path.join(fh_dir, "weight_pol.png"), normalize=False)
                
#                 # 2. IRHighlightInjector inputs and outputs
#                 if "ir_highlight_injector" in out["intermediate_results"]:
#                     iri = out["intermediate_results"]["ir_highlight_injector"]
#                     iri_dir = os.path.join(intermediate_dir, "ir_highlight_injector", base_name)
#                     os.makedirs(iri_dir, exist_ok=True)
                    
#                     # Inputs: fusion result and IR image.
#                     if "input_fused" in iri:
#                         save_image(iri["input_fused"][0].cpu().clamp(0, 1), 
#                                   os.path.join(iri_dir, "input_fused.png"), normalize=False)
#                     if "input_ir" in iri:
#                         save_image(iri["input_ir"][0].cpu().clamp(0, 1), 
#                                   os.path.join(iri_dir, "input_ir.png"), normalize=False)
                    
#                     # Outputs: fusion result and mask.
#                     if "output_fused" in iri:
#                         save_image(iri["output_fused"][0].cpu().clamp(0, 1), 
#                                   os.path.join(iri_dir, "output_fused.png"), normalize=False)
#                     if "output_mask" in iri and iri["output_mask"] is not None:
#                         save_image(iri["output_mask"][0].cpu().clamp(0, 1), 
#                                   os.path.join(iri_dir, "output_mask.png"), normalize=False)
                
#                 # 3. PolarTextureEnhancer inputs and outputs
#                 if "polar_texture_enhancer" in out["intermediate_results"]:
#                     pte = out["intermediate_results"]["polar_texture_enhancer"]
#                     pte_dir = os.path.join(intermediate_dir, "polar_texture_enhancer", base_name)
#                     os.makedirs(pte_dir, exist_ok=True)
                    
#                     # Inputs: fusion result, POL image, and mask.
#                     if "input_fused" in pte:
#                         save_image(pte["input_fused"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "input_fused.png"), normalize=False)
#                     if "input_pol" in pte:
#                         save_image(pte["input_pol"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "input_pol.png"), normalize=False)
#                     if "input_mask" in pte and pte["input_mask"] is not None:
#                         save_image(pte["input_mask"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "input_mask.png"), normalize=False)
                    
#                     # Output: fusion result.
#                     if "output_fused" in pte:
#                         save_image(pte["output_fused"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "output_fused.png"), normalize=False)
                    
#                     # Output: texture residual.
#                     if "texture_residual" in pte:
#                         save_image(pte["texture_residual"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "texture_residual.png"), normalize=False)
                    
#                     # Outputs: fuse_mean, fuse_std, and fuse_feat after polar_texture_enhancer.
#                     if "fuse_mean" in pte:
#                         # fuse_mean has shape [B, 1, 1, 1].
#                         fuse_mean_val = pte["fuse_mean"][0, 0, 0, 0].cpu().item()
#                         # Create a visualization image.
#                         fuse_mean_img = torch.full((1, 1, 64, 64), fuse_mean_val, dtype=torch.float32)
#                         save_image(fuse_mean_img, os.path.join(pte_dir, "fuse_mean.png"), normalize=False)
#                         # Also save the value to a text file.
#                         with open(os.path.join(pte_dir, "fuse_mean.txt"), "w") as f:
#                             f.write(f"{fuse_mean_val:.6f}\n")
                    
#                     if "fuse_std" in pte:
#                         # fuse_std has shape [B, 1, 1, 1].
#                         fuse_std_val = pte["fuse_std"][0, 0, 0, 0].cpu().item()
#                         # Create a visualization image.
#                         fuse_std_img = torch.full((1, 1, 64, 64), fuse_std_val, dtype=torch.float32)
#                         save_image(fuse_std_img, os.path.join(pte_dir, "fuse_std.png"), normalize=False)
#                         # Also save the value to a text file.
#                         with open(os.path.join(pte_dir, "fuse_std.txt"), "w") as f:
#                             f.write(f"{fuse_std_val:.6f}\n")
                    
#                     # fuse_feat is the feature after the "normalize to the target mean and standard-deviation range while preserving relative brightness relationships" step.
#                     if "fuse_feat" in pte:
#                         # fuse_feat has shape [B, 1, H, W] and contains the normalized features.
#                         save_image(pte["fuse_feat"][0].cpu().clamp(0, 1), 
#                                   os.path.join(pte_dir, "fuse_feat.png"), normalize=False)
            
#             print(f"  {name}: Fusion complete [range: {fusion_min:.4f}~{fusion_max:.4f}, mean: {fusion_mean:.4f}]")
            
#         except Exception as e:
#             print(f"Error while processing {name}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
    
#     print(f"\nFusion complete. Results were saved to: {out_dir}")
#     print(f"Intermediate results were saved to: {intermediate_dir}")


# # ============================================================
# # Main function
# # ============================================================

# def main():
#     parser = argparse.ArgumentParser(description="Perform image fusion using a trained model")
#     parser.add_argument("--ckpt", type=str, required=True, help="path to the model weights file (.pth)")
#     parser.add_argument("--data_root", type=str, required=True, help="input data root directory containing inf/ and pol/ subdirectories")
#     parser.add_argument("--out_dir", type=str, default="test_results", help="output directory")
#     parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="execution device")
    
#     args = parser.parse_args()
    
#     # Check whether the model file exists.
#     if not os.path.exists(args.ckpt):
#         raise FileNotFoundError(f"Model file does not exist: {args.ckpt}")
    
#     # Configure the device.
#     if args.device == "cuda" and not torch.cuda.is_available():
#         print("Warning: CUDA is unavailable; using CPU")
#         device = "cpu"
#     else:
#         device = args.device
    
#     print(f"Using device: {device}")
#     print(f"Loading model weights: {args.ckpt}")
    
#     # Create the model with intermediate-result saving enabled.
#     model = PolarIRS4FusionMamba(save_intermediate=True).to(device)
    
#     # Load the weights.
#     try:
#         checkpoint = torch.load(args.ckpt, map_location=device)
        
#         # Support different checkpoint formats.
#         if isinstance(checkpoint, dict):
#             if 'model_state_dict' in checkpoint:
#                 model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#                 print("  ✓ Loaded model weights (model_state_dict)")
#             elif 'state_dict' in checkpoint:
#                 model.load_state_dict(checkpoint['state_dict'], strict=False)
#                 print("  ✓ Loaded model weights (state_dict)")
#             else:
#                 # Attempt to load the entire dictionary directly as a state_dict.
#                 model.load_state_dict(checkpoint, strict=False)
#                 print("  ✓ Loaded model weights (dictionary directly)")
#         else:
#             model.load_state_dict(checkpoint, strict=False)
#             print("  ✓ Loaded model weights (state_dict directly)")
        
#         print("Model loaded successfully.")
        
#     except Exception as e:
#         print(f"Error while loading model weights: {e}")
#         raise
    
#     # Perform fusion.
#     print(f"\nStarting fusion; input path: {args.data_root}")
#     run_fusion(model, args.data_root, args.out_dir, device)


# if __name__ == "__main__":
#     main()

