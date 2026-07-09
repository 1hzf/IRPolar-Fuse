import os
import argparse
import random
from pathlib import Path
# 默认使用物理 GPU 2；外部设置 CUDA_VISIBLE_DEVICES 时优先使用外部配置。
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# 设置cuDNN以自动选择最佳算法，避免算法选择错误；同时限制工作空间以减少显存碎片
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64,garbage_collection_threshold:0.8')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from polar_ir_s4fusion_mamba import (
    PolarIRS4FusionMamba,
    total_loss
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Dataset
# ============================================================

class PolarIRDataset(Dataset):
    def __init__(self, root, split="train"):
        self.ir_dir = os.path.join(root, "inf")
        self.pol_dir = os.path.join(root, "pol")

        self.names = sorted(os.listdir(self.ir_dir))

        self.transform = transforms.Compose([
            transforms.ToTensor(),   # [0,1]
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        ir = Image.open(os.path.join(self.ir_dir, name)).convert("L")
        pol = Image.open(os.path.join(self.pol_dir, name)).convert("L")

        ir = self.transform(ir)
        pol = self.transform(pol)

        return ir, pol


# ============================================================
# Inference utilities
# ============================================================

def load_image_as_tensor(path: str):
    """Load grayscale image to tensor [1,1,H,W] in [0,1]."""
    img = Image.open(path).convert("L")
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t


def resolve_pair_dirs(root: str):
    """Resolve paired input dirs. Prefer train-style inf/pol, fallback to ir/vi."""
    root_path = Path(root)
    candidates = [
        (root_path / "inf", root_path / "pol"),
        (root_path / "ir", root_path / "vi"),
    ]
    for first_dir, second_dir in candidates:
        if first_dir.exists() and second_dir.exists():
            return first_dir, second_dir
    return root_path / "inf", root_path / "pol"

@torch.no_grad()
def run_inference(model, data_root, out_dir, device):
    """
    Run fusion on a folder and save concatenated results:
    [IR_raw | POL_raw | Fusion]
    """
    model.eval()
    inf_dir = Path(data_root) / "inf"
    pol_dir = Path(data_root) / "pol"
    names = sorted([p.name for p in inf_dir.iterdir() if p.is_file()])
    os.makedirs(out_dir, exist_ok=True)

    for name in names:
        inf_path = inf_dir / name
        pol_path = pol_dir / name
        if not pol_path.exists():
            print(f"Skip {name}, pol file missing.")
            continue

        ir = load_image_as_tensor(str(inf_path)).to(device)
        pol = load_image_as_tensor(str(pol_path)).to(device)

        out = model(ir, pol)
        fusion = out["fusion"].clamp(0, 1).cpu()

        # 调试信息：检查融合结果的值范围
        fusion_min = fusion.min().item()
        fusion_max = fusion.max().item()
        fusion_mean = fusion.mean().item()
        print(f"  Fusion range: [{fusion_min:.4f}, {fusion_max:.4f}], mean: {fusion_mean:.4f}")

        imgs = [ir.cpu()[0], pol.cpu()[0], fusion[0]]
        cat = torch.cat(imgs, dim=-1)
        # 不使用normalize，因为输入已经在[0,1]范围内
        # 如果融合结果全黑，normalize=True会导致问题
        save_image(cat, os.path.join(out_dir, f"result_{name}"), normalize=False)
        # 单独保存融合结果图
        save_image(fusion[0], os.path.join(out_dir, f"fusion_{name}"), normalize=False)
        print(f"Saved {name}")

    model.train()


# ============================================================
# Train Function
# ============================================================

def train_one_epoch(model, loader, optimizer, device, scaler, loss_kwargs):
    model.train()
    total = 0.0
    # 用于累积所有损失组件的字典
    loss_components = {}

    for ir, pol in tqdm(loader, desc="Train", leave=False):
        ir = ir.to(device)
        pol = pol.to(device)

        with autocast(enabled=(device == "cuda")):
            out = model(ir, pol)
            
            # 检查模型输出是否包含NaN或Inf
            fusion = out.get("fusion", None)
            if fusion is not None:
                if torch.isnan(fusion).any() or torch.isinf(fusion).any():
                    print(f"警告: 模型输出包含NaN/Inf，跳过此batch")
                    continue
                # 修复NaN和Inf
                fusion = torch.where(torch.isnan(fusion) | torch.isinf(fusion), 
                                     torch.zeros_like(fusion), fusion)
                out["fusion"] = fusion
            
            loss, loss_dict = total_loss(out, ir, pol, **loss_kwargs)
            # 确保 loss 是 tensor，不是 tuple
            if isinstance(loss, tuple):
                loss = loss[0]
            
            # 检查损失是否包含NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 损失为NaN/Inf，跳过此batch")
                print(f"  损失组件: {loss_dict}")
                continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # 梯度裁剪：防止梯度爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        # 累积各个损失组件
        for key, value in loss_dict.items():
            if key not in loss_components:
                loss_components[key] = []
            loss_components[key].append(value)

    # 计算平均损失
    avg_loss = total / len(loader)
    avg_loss_dict = {key: sum(values) / len(values) for key, values in loss_components.items()}
    
    return avg_loss, avg_loss_dict


# ============================================================
# Main
# ============================================================

def main(args):
    # -------------------------------
    # Config
    # -------------------------------
    data_root = args.data_root
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # -------------------------------
    # Data
    # -------------------------------
    train_set = PolarIRDataset(data_root, "train")
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=data_generator
    )

    # -------------------------------
    # Model
    # -------------------------------
    model = PolarIRS4FusionMamba(
        ihj_thresh=args.ihj_thresh,
        ihj_sharpness=args.ihj_sharpness,
        ihj_inject_ratio=args.ihj_inject_ratio,
        use_multiscale=not args.ablate_multiscale,
        use_cross_mamba=not args.ablate_cross_mamba,
        use_paf=not args.ablate_paf,
        use_ihj=not args.ablate_ihj,
        use_ptj=not args.ablate_ptj,
    ).to(device)
    ablation_config = {
        "use_multiscale": not args.ablate_multiscale,
        "use_cross_mamba": not args.ablate_cross_mamba,
        "use_paf": not args.ablate_paf,
        "use_ihj": not args.ablate_ihj,
        "use_ptj": not args.ablate_ptj,
    }
    print(f"Ablation config: {ablation_config}")
    scaler = GradScaler(enabled=(device == "cuda"))
    loss_kwargs = {
        "lambda_fusion_core": args.lambda_fusion_core,
        "lambda_ir_bias": args.lambda_ir_bias,
        "lambda_polar_texture": args.lambda_polar_texture,
        "lambda_regularization": args.lambda_regularization,
        "lambda_intermediate": args.lambda_intermediate,
    }
    
    # 推理模式：仅做融合并保存结果
    if args.infer_root:
        if args.ckpt:
            state = torch.load(args.ckpt, map_location=device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint for inference: {args.ckpt}")
        run_inference(model, args.infer_root, args.out_dir, device)
        return

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    # -------------------------------
    # Load checkpoint for resuming training
    # -------------------------------
    start_epoch = 0
    if args.resume_ckpt:
        # 支持使用 'latest' 关键字
        if args.resume_ckpt.lower() == 'latest':
            resume_path = "checkpoints/latest.pth"
        else:
            resume_path = args.resume_ckpt
        
        if not os.path.exists(resume_path):
            print(f"警告: Checkpoint文件不存在: {resume_path}")
            print("  将从头开始训练")
            start_epoch = 0
        else:
            print(f"Loading checkpoint for resuming training: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("  ✓ Loaded model weights")
            else:
                # 兼容旧格式（只有模型权重）
                model.load_state_dict(checkpoint, strict=False)
                print("  ✓ Loaded model weights (old format)")
            
            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("  ✓ Loaded optimizer state")
            
            # 加载调度器状态
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("  ✓ Loaded scheduler state")
            
            # 加载scaler状态
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("  ✓ Loaded scaler state")
            
            # 恢复epoch计数
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"  ✓ Resuming from epoch {start_epoch}")
            else:
                # 尝试从文件名推断epoch
                import re
                match = re.search(r'epoch[_\s]*(\d+)', resume_path)
                if match:
                    start_epoch = int(match.group(1))
                    print(f"  ✓ Inferred starting epoch from filename: {start_epoch}")
            
            # 恢复最佳损失（如果有）
            if 'best_loss' in checkpoint:
                print(f"  ✓ Previous best loss: {checkpoint['best_loss']:.6f}")
    elif args.ckpt:
        # 兼容旧参数：只加载模型权重，不恢复训练状态
        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint (weights only): {args.ckpt}")
        print("  Note: Use --resume_ckpt to resume training with optimizer/scheduler state")

    # -------------------------------
    # Training Loop
    # -------------------------------
    checkpoint_root = args.save_dir
    os.makedirs(checkpoint_root, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        loss, loss_dict = train_one_epoch(model, train_loader, optimizer, device, scaler, loss_kwargs)
        scheduler.step()

        # 打印总损失和所有损失组件
        print(f"\n{'='*80}")
        print(f"[Epoch {epoch:03d}] Total Loss: {loss:.6f}")
        print(f"{'-'*80}")
        print("Loss Components (Fusion Only):")
        print(f"  L_fusion_core (核心融合损失):      {loss_dict.get('L_fusion_core', 0.0):.6f}")
        print(f"  L_ir_bias (IR偏向损失):            {loss_dict.get('L_ir_bias', 0.0):.6f}")
        print(f"  L_polar_texture (偏振纹理损失):    {loss_dict.get('L_polar_texture', 0.0):.6f}")
        print(f"  L_intermediate (中间监督损失):     {loss_dict.get('L_intermediate', 0.0):.6f}")
        print(f"  L_regularization (正则化损失):     {loss_dict.get('L_regularization', 0.0):.6f}")
        print(f"{'-'*80}")
        print("Weighted Loss Components:")
        print(f"  λ1 * L_fusion_core:               {loss_dict.get('weighted_L_fusion_core', 0.0):.6f}")
        print(f"  λ2 * L_ir_bias:                   {loss_dict.get('weighted_L_ir_bias', 0.0):.6f}")
        print(f"  λ3 * L_polar_texture:             {loss_dict.get('weighted_L_polar_texture', 0.0):.6f}")
        print(f"  λ4 * L_intermediate:               {loss_dict.get('weighted_L_intermediate', 0.0):.6f}")
        print(f"  λ5 * L_regularization:            {loss_dict.get('weighted_L_regularization', 0.0):.6f}")
        print(f"{'-'*80}")
        if 'confidence' in loss_dict:
            print(f"  Global Confidence (平均):         {loss_dict.get('confidence', 0.0):.4f}")
        print(f"{'='*80}\n")

        # -------------------------------------------------------
        # 每个 epoch 结束后：使用test文件夹中的图片进行融合验证
        # 默认使用 ./test 路径
        # -------------------------------------------------------
        should_validate = (not args.save_final_only) or (epoch + 1 == num_epochs)
        test_root = args.test_root if args.test_root else "./test"

        if should_validate and os.path.exists(test_root):
            model.eval()
            with torch.no_grad():
                # 本 epoch 保存目录
                out_dir = f"{checkpoint_root}/epoch_{epoch+1}"
                os.makedirs(f"{out_dir}/fusion", exist_ok=True)
                
                # 加载test数据集；支持 inf/pol 或 ir/vi 两种目录命名
                test_inf_dir, test_pol_dir = resolve_pair_dirs(test_root)
                
                if test_inf_dir.exists() and test_pol_dir.exists():
                    test_suffixes = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
                    test_names = sorted([p.name for p in test_inf_dir.iterdir()
                                        if p.is_file() and p.suffix.lower() in test_suffixes])
                    
                    print(f"\n处理test文件夹中的 {len(test_names)} 张图片...")
                    
                    for idx, name in enumerate(test_names):
                        inf_path = test_inf_dir / name
                        pol_path = test_pol_dir / name
                        
                        if not pol_path.exists():
                            print(f"跳过 {name}，POL文件不存在")
                            continue
                        
                        # 加载图像
                        ir = load_image_as_tensor(str(inf_path)).to(device)
                        pol = load_image_as_tensor(str(pol_path)).to(device)
                        
                        # 仅执行融合
                        out = model(ir, pol)
                        
                        # 获取结果
                        fusion = out["fusion"].clamp(0, 1).cpu()
                        # 调试信息：检查融合结果的值范围（仅第一张图片）
                        if idx == 0:
                            fusion_min = fusion.min().item()
                            fusion_max = fusion.max().item()
                            fusion_mean = fusion.mean().item()
                            print(f"  融合结果统计 (第一张): min={fusion_min:.4f}, max={fusion_max:.4f}, mean={fusion_mean:.4f}")
                            # 检查是否全黑
                            if fusion_max < 0.01:
                                print(f"  警告: 融合结果可能全黑！最大值仅为 {fusion_max:.6f}")
                        
                        if not args.save_fusion_only:
                            ir_original = ir.cpu()[0]
                            pol_original = pol.cpu()[0]
                            # 保存融合结果（拼接：IR原始 | POL原始 | Fusion）
                            imgs_concat = torch.cat([ir_original, pol_original, fusion[0]], dim=-1)
                            # 不使用normalize，因为输入已经在[0,1]范围内
                            # 如果融合结果全黑，normalize=True会导致问题
                            save_image(imgs_concat, f"{out_dir}/fusion/result_{name}", normalize=False)
                        # 单独保存融合结果图
                        save_image(fusion[0], f"{out_dir}/fusion/fusion_{name}", normalize=False)
                        
                        print(f"  已保存: {name}")
                    
                    print(f"所有test图片已处理完成，结果保存在: {out_dir}")
                else:
                    print(f"警告: test文件夹不存在 ({test_root})，跳过验证")
        elif should_validate:
            print(f"警告: test文件夹不存在 ({test_root})，跳过验证")
        
        model.train()

        # 每个epoch结束后都保存完整的checkpoint（包含训练状态）
        should_save_checkpoint = (not args.save_final_only) or (epoch + 1 == num_epochs)
        if should_save_checkpoint:
            checkpoint_path = f"{checkpoint_root}/epoch_{epoch+1}.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': loss,
                'loss_dict': loss_dict,
                'loss_kwargs': loss_kwargs,
                'ihj_params': {
                    'ihj_thresh': args.ihj_thresh,
                    'ihj_sharpness': args.ihj_sharpness,
                    'ihj_inject_ratio': args.ihj_inject_ratio,
                },
                'ablation_config': ablation_config,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"完整checkpoint已保存: {checkpoint_path}")

            # 同时保存一个latest checkpoint，方便恢复
            latest_path = f"{checkpoint_root}/latest.pth"
            torch.save(checkpoint, latest_path)

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or infer PolarIRS4FusionMamba")
    parser.add_argument("--data_root", type=str, default="./registration-fusion", help="train data root (contains inf/ and pol/)")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="训练的总epoch数 (可通过命令行参数 --num_epochs 设置，例如: --num_epochs 200)")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--ckpt", type=str, default="", help="optional checkpoint path (weights only, for inference or fine-tuning)")
    parser.add_argument("--resume_ckpt", type=str, default="", help="checkpoint path to resume training from (includes optimizer/scheduler state). Use 'latest' to resume from checkpoints/latest.pth")
    parser.add_argument("--infer_root", type=str, default="", help="if set, run inference on this root (expects inf/ and pol/)")
    parser.add_argument("--out_dir", type=str, default="test_results", help="output dir for inference results")
    parser.add_argument("--test_root", type=str, default="./test", help="test data root for validation after each epoch (contains inf/ and pol/), default: ./test")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="directory for checkpoints and epoch visual results")
    parser.add_argument("--save_final_only", action="store_true", help="only validate/save checkpoint at the final epoch")
    parser.add_argument("--save_fusion_only", action="store_true", help="save only fusion images, not IR/POL/Fusion concatenations")
    parser.add_argument("--seed", type=int, default=2026, help="random seed for reproducible sensitivity experiments")
    parser.add_argument("--lambda_fusion_core", type=float, default=1.0, help="weight for core fusion loss")
    parser.add_argument("--lambda_ir_bias", type=float, default=2.0, help="weight for infrared target preservation loss")
    parser.add_argument("--lambda_polar_texture", type=float, default=2.5, help="weight for polarization texture preservation loss")
    parser.add_argument("--lambda_regularization", type=float, default=0.3, help="weight for regularization loss")
    parser.add_argument("--lambda_intermediate", type=float, default=1.0, help="weight for intermediate supervision loss")
    parser.add_argument("--ihj_thresh", type=float, default=0.62, help="fixed soft-mask threshold for IR highlight injection")
    parser.add_argument("--ihj_sharpness", type=float, default=15.0, help="fixed soft-mask sharpness for IR highlight injection")
    parser.add_argument("--ihj_inject_ratio", type=float, default=0.6, help="fixed base injection ratio for IR highlight injection")
    parser.add_argument("--ablate_multiscale", action="store_true", help="leave-one-out: remove multi-scale feature extraction")
    parser.add_argument("--ablate_cross_mamba", action="store_true", help="leave-one-out: remove cross-modal Mamba feature interaction")
    parser.add_argument("--ablate_paf", action="store_true", help="leave-one-out: remove PAF fusion attention")
    parser.add_argument("--ablate_ihj", action="store_true", help="leave-one-out: remove IR highlight injection")
    parser.add_argument("--ablate_ptj", action="store_true", help="leave-one-out: remove polar texture joint/enhancement module")
    args = parser.parse_args()
    main(args)
