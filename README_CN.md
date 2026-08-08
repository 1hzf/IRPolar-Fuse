# IRPol-Fuse：面向低能见度场景的红外偏振图像融合能量–结构协同方法

<p align="center">
  🌐 <a href="./README.md">English</a> | <b>简体中文</b>
</p>

<p align="center">
  <b>Zhuangfan Huang</b>,
  Chusheng Fang,
  Xiaosong Li,
  Yang Liu,
  Xiaoqi Cheng,
  Haishu Tan
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.infrared.2026.106788">
    <img src="https://img.shields.io/badge/DOI-10.1016%2Fj.infrared.2026.106788-blue.svg">
  </a>
  <a href="YOUR_ARXIV_LINK">
    <img src="https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg">
  </a>
  <a href="https://pan.baidu.com/s/13ObQ8VmiruiBUz1A_9YMcA?pwd=a79c">
    <img src="https://img.shields.io/badge/Dataset-LI--PI-blue.svg">
  </a>
  <a href="https://pan.baidu.com/s/1xmbyjUPe4AkXTUbiitNIHw?pwd=mn32">
    <img src="https://img.shields.io/badge/Model-Pretrained-green.svg">
  </a>
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.infrared.2026.106788">论文</a> |
  <a href="YOUR_ARXIV_LINK">arXiv</a> |
  <a href="#li-pi-数据集">数据集</a> |
  <a href="#实验结果">实验结果</a> |
  <a href="#下载">下载</a> |
  <a href="#环境安装">环境安装</a> |
  <a href="#训练">训练</a> |
  <a href="#测试">测试</a>
</p>

---

# 项目简介

低能见度环境下的鲁棒视觉感知要求融合图像能够同时保留红外图像中的热目标显著性以及偏振图像中的结构细节信息。然而，现有红外–偏振图像融合方法往往容易过度强调占主导地位的红外响应，从而导致暗区域或视觉遮蔽区域中较弱但具有重要信息的偏振纹理受到抑制。

为解决这一问题，我们提出 **IRPol-Fuse**，一种面向复杂低能见度场景的能量–结构协同红外偏振图像融合框架。

IRPol-Fuse 主要包含三个核心模块：

- **偏振注意力融合模块（Polarization Attention Fusion, PAF）**：用于实现红外与偏振信息的自适应分配。
- **红外高亮注入模块（Infrared Highlight Injector, IHJ）**：通过高亮区域引导的方式增强和保持红外热目标响应。
- **偏振纹理注入模块（Polarization Texture Injector, PTJ）**：用于恢复偏振纹理信息以及细粒度结构细节。

此外，我们构建了 **LI-PI** 数据集，这是一个专门面向低能见度和视觉遮蔽场景的红外–偏振图像融合数据集。

---

# 网络结构

<p align="center">
  <img src="./fig/fig8.png" width="95%">
</p>

<p align="center">
  <b>IRPol-Fuse 整体网络结构。</b>
</p>

IRPol-Fuse 整体采用三级融合流程：

1. **模态特定特征编码**
2. **跨模态长程交互**
3. **基于 PAF、IHJ 和 PTJ 的逐级能量–结构协同**

在完成跨模态交互后，PAF 首先生成初步融合表征。随后，IHJ 通过自适应红外信息重注入进一步保持具有显著热响应的目标区域。最后，PTJ 恢复偏振图像中的结构与纹理信息，从而得到最终融合结果。

---

# 核心模块

<p align="center">
  <img src="./fig/fig9.png" width="95%">
</p>

<p align="center">
  <b>PAF、IHJ 和 PTJ 三个核心模块的详细结构。</b>
</p>

## 偏振注意力融合模块（PAF）

PAF 根据跨模态特征响应，自适应协调红外热响应与偏振结构信息，为后续细化阶段提供初步融合表征。

## 红外高亮注入模块（IHJ）

IHJ 采用红外响应引导的自适应重注入策略，显式保持具有高热响应的红外目标，同时避免向低响应背景区域中过度注入红外信息。

## 偏振纹理注入模块（PTJ）

PTJ 通过四个互补分支恢复偏振图像中的结构信息：

- 频域增强（Frequency Domain Enhancement）
- 二阶梯度（Second-Order Gradient）
- 自适应对比度增强（Adaptive Contrast Enhancement）
- 多尺度纹理提取（Multiscale Texture Extraction）

随后，根据红外响应对生成的偏振纹理残差进行进一步调制，从而在红外目标保持和偏振结构恢复之间取得平衡。

---

# LI-PI 数据集

我们构建了 **LI-PI**，一个专门面向复杂低能见度和视觉遮蔽场景的红外–偏振图像融合数据集。

LI-PI 共包含 **110 对严格配准的红外–偏振图像**，具体划分如下：

| 数据划分 | 图像对数量 |
|---|---:|
| 训练集 | 90 |
| 验证集 | 10 |
| 测试集 | 10 |
| **总计** | **110** |

其中，验证集用于模型选择和超参数分析，独立测试集仅用于最终性能评估。

LI-PI 覆盖多种具有代表性的复杂场景，包括：

- 地下停车场
- 阳台
- 森林
- 走廊
- 局部强光干扰
- 弱背景结构
- 视觉遮蔽目标
- 低能见度环境

## LI-PI 数据集示例

<p align="center">
  <img src="./fig/fig5.png" width="95%">
</p>

<p align="center">
  <b>LI-PI 数据集及典型低能见度场景示例。</b>
</p>

---

# 下载

| 资源 | 说明 | 下载链接 | 提取码 |
|---|---|---|---|
| LI-PI 数据集 | 本文实验使用的训练集、验证集和测试集 | [百度网盘](https://pan.baidu.com/s/13ObQ8VmiruiBUz1A_9YMcA?pwd=a79c) | `a79c` |
| 预训练模型 | IRPol-Fuse 预训练模型权重 | [百度网盘](https://pan.baidu.com/s/1xmbyjUPe4AkXTUbiitNIHw?pwd=mn32) | `mn32` |

---

# 环境安装

请根据 `requirements.txt` 安装项目所需依赖：

```bash
pip install -r requirements.txt
```

---

# 训练

训练前，请根据本地数据集存放位置修改 `--data_root` 和 `--test_root`。

```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
    --data_root YOUR_TRAIN_DATA_PATH \
    --test_root YOUR_TEST_DATA_PATH \
    --save_dir checkpoints_retrain
```

本文使用的 LI-PI 数据集划分为：

```text
训练集   : 90 对图像
验证集   : 10 对图像
测试集   : 10 对图像
```

---

# 测试

首先下载预训练模型权重，然后根据本地环境修改测试数据集路径和输出路径。

```bash
python test.py \
    --ckpt ./epoch_11.pth \
    --data_root YOUR_TEST_DATA_PATH \
    --out_dir YOUR_OUTPUT_PATH \
    --device cuda
```

最终融合图像将保存至 `--out_dir` 指定的目录中。

---

# 实验结果

## LI-PI 数据集定量对比

| 方法 | QP ↑ | QS ↑ | QCB ↑ | QCV ↓ | QAB/F ↑ | MS-SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **IRPol-Fuse** | **0.355** | **0.805** | **0.532** | **171.583** | **0.628** | **0.930** |
| TIPFNet | 0.346 | 0.770 | 0.487 | 199.669 | 0.624 | 0.929 |
| CPIFuse | 0.176 | 0.475 | 0.218 | 216.936 | 0.248 | 0.744 |
| LFDT | 0.235 | 0.472 | 0.257 | 218.473 | 0.335 | 0.721 |
| LUT-Fuse | 0.149 | 0.367 | 0.240 | 211.443 | 0.266 | 0.624 |
| PIPFNet | 0.137 | 0.368 | 0.225 | 229.187 | 0.144 | 0.592 |
| DT-F | 0.354 | 0.501 | 0.332 | 322.379 | 0.624 | 0.928 |
| FusionMamba | 0.113 | 0.436 | 0.170 | 263.748 | 0.208 | 0.746 |
| CDDFuse | 0.295 | 0.555 | 0.262 | 186.820 | 0.354 | 0.825 |
| SeAFusion | 0.178 | 0.569 | 0.283 | 240.403 | 0.450 | 0.841 |

IRPol-Fuse 在 LI-PI 测试集的六项全局融合指标上均取得最优性能，表明所提出的能量–结构协同策略能够有效协调红外热目标显著性与偏振结构信息。

---

## LI-PI 数据集主观视觉对比

<p align="center">
  <img src="./fig/fig1.png" width="100%">
</p>

<p align="center">
  <b>LI-PI 测试集上的主观视觉对比结果。</b>
</p>

IRPol-Fuse 在复杂低能见度环境下能够有效保持显著红外目标，同时保留更清晰的偏振纹理和结构细节。

---

# 区域感知评价

为了更加直接地评价红外目标保持能力和偏振纹理恢复能力，我们进一步采用了基于红外响应引导的区域感知评价协议。

| 方法 | IR ROI Corr ↑ | IR CNR ↑ | Pol Grad Corr ↑ | Pol Grad MAE ↓ |
|---|---:|---:|---:|---:|
| **IRPol-Fuse** | 0.886 | 3.306 | **0.959** | **0.012** |
| TIPFNet | 0.487 | 0.285 | 0.197 | 0.063 |
| CPIFuse | 0.848 | **3.544** | 0.800 | 0.025 |
| LFDT | **0.995** | 3.192 | 0.779 | 0.033 |
| LUT-Fuse | 0.964 | 3.065 | 0.788 | 0.034 |
| PIPFNet | 0.501 | 2.823 | 0.441 | 0.041 |
| DT-F | 0.534 | 2.207 | 0.859 | 0.028 |
| FusionMamba | 0.880 | 2.443 | 0.895 | 0.024 |
| CDDFuse | 0.941 | 2.257 | 0.894 | 0.029 |
| SeAFusion | 0.528 | 3.267 | 0.857 | 0.021 |

IRPol-Fuse 在两项偏振纹理评价指标上均取得最优结果，其中：

- **Pol Grad Corr：0.959**
- **Pol Grad MAE：0.012**

结果表明，IRPol-Fuse 能够有效恢复偏振图像中的结构信息，同时保持具有竞争力的红外目标显著性。

---

# LDDRS 泛化实验

为了验证模型在常规红外–偏振成像条件下的泛化能力，我们进一步在外部 **LDDRS** 数据集上对 IRPol-Fuse 进行直接测试。

## 定量对比

| 方法 | QP ↑ | QS ↑ | QCB ↑ | QCV ↓ | QAB/F ↑ | MS-SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **IRPol-Fuse** | 0.484 | **0.728** | 0.499 | 393.908 | **0.534** | 0.927 |
| TIPFNet | 0.329 | 0.579 | 0.320 | 982.441 | 0.430 | 0.924 |
| CPIFuse | **0.509** | 0.676 | **0.507** | 1961.014 | 0.442 | 0.907 |
| LFDT | 0.374 | 0.468 | 0.224 | 79.728 | 0.408 | 0.809 |
| LUT-Fuse | 0.208 | 0.438 | 0.222 | 94.077 | 0.356 | 0.761 |
| PIPFNet | 0.094 | 0.311 | 0.335 | 1920.085 | 0.167 | 0.300 |
| DT-F | 0.343 | 0.418 | 0.256 | 346.184 | 0.397 | **0.929** |
| FusionMamba | 0.123 | 0.485 | 0.250 | **75.322** | 0.227 | 0.880 |
| CDDFuse | 0.287 | 0.505 | 0.223 | 150.082 | 0.332 | 0.887 |
| SeAFusion | 0.268 | 0.510 | 0.251 | 77.722 | 0.374 | 0.895 |

IRPol-Fuse 在 QS 和 QAB/F 指标上取得最优性能，并在其余指标上保持较强竞争力，体现了其在外部红外–偏振数据集上的良好泛化能力。

---

## LDDRS 主观视觉对比

<p align="center">
  <img src="./fig/fig2.png" width="100%">
</p>

<p align="center">
  <b>LDDRS 数据集上的主观视觉对比结果。</b>
</p>

结果表明，在外部红外–偏振成像场景下，IRPol-Fuse 仍能够在红外热响应保持和偏振结构细节恢复之间取得良好平衡。

---

# 跨模态泛化

为了进一步验证所提出能量–结构协同策略的迁移能力，我们还在两个红外–可见光图像融合数据集上进行了实验：

- **MSRS**
- **M3FD**

这些实验作为补充验证，用于考察所提出框架在红外–偏振融合任务之外的跨模态泛化能力。

## MSRS 主观视觉对比

<p align="center">
  <img src="./fig/fig3.png" width="100%">
</p>

<p align="center">
  <b>MSRS 数据集上的主观视觉对比结果。</b>
</p>

---

## M3FD 主观视觉对比

<p align="center">
  <img src="./fig/fig4.png" width="100%">
</p>

<p align="center">
  <b>M3FD 数据集上的主观视觉对比结果。</b>
</p>

更详细的定量实验结果请参阅论文正文。

---

# 消融实验

我们在 LI-PI 测试集上进一步验证了跨模态交互、IHJ 和 PTJ 等核心组成部分的有效性。

| 方法 | QP ↑ | QS ↑ | QCB ↑ | QCV ↓ | QAB/F ↑ | MS-SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **IRPol-Fuse** | **0.355** | **0.805** | **0.532** | **171.583** | **0.628** | **0.930** |
| w/o IHJ | 0.332 | 0.756 | 0.453 | 188.761 | 0.580 | 0.911 |
| w/o PTJ | 0.345 | 0.804 | 0.525 | 179.074 | 0.620 | 0.928 |
| w/o Cross + IHJ | 0.350 | 0.766 | 0.468 | 184.127 | 0.610 | 0.926 |
| w/o Cross + PTJ | 0.353 | 0.805 | 0.523 | 189.215 | 0.616 | 0.926 |
| w/o IHJ + PTJ | 0.343 | 0.764 | 0.472 | 198.403 | 0.604 | 0.920 |
| Baseline | 0.346 | 0.762 | 0.457 | 188.551 | 0.591 | 0.917 |

完整模型在各项指标上呈现出最均衡的整体性能，验证了红外信息保持与偏振纹理恢复两个过程之间的互补作用。

---

# 超参数敏感性

IRPol-Fuse 最终采用的参数配置如下：

```text
lambda_fusion = 1.0
lambda_ir     = 2.0
lambda_pol    = 2.5

ihj_ratio     = 0.60
ihj_sharpness = 15
ihj_thresh    = 0.62
```

更详细的超参数敏感性分析请参阅论文正文。

---

# 引用

如果 **IRPol-Fuse** 或 **LI-PI 数据集** 对您的研究有所帮助，欢迎引用我们的论文：

**论文：** [IRPol-Fuse: Energy–structure coordination for infrared polarization fusion under low visibility](https://doi.org/10.1016/j.infrared.2026.106788)  
**期刊：** *Infrared Physics & Technology*, 2026, Article 106788.

```bibtex
@article{HUANG2026106788,
  title    = {IRPol-Fuse: Energy--structure coordination for infrared polarization fusion under low visibility},
  journal  = {Infrared Physics \& Technology},
  pages    = {106788},
  year     = {2026},
  issn     = {1350-4495},
  doi      = {10.1016/j.infrared.2026.106788},
  url      = {https://www.sciencedirect.com/science/article/pii/S1350449526004238},
  author   = {Zhuangfan Huang and Chusheng Fang and Xiaosong Li and Yang Liu and Xiaoqi Cheng and Haishu Tan},
  keywords = {Infrared-polarization image fusion, Energy-structure coordination, Low-visibility imaging}
}
```

---

# 致谢

感谢本文实验中所使用公开数据集及对比方法的作者，包括 **TIPFNet、CPIFuse、DT-F、PIPFNet、LFDT-Fusion、FusionMamba、CDDFuse、LUT-Fuse 和 SeAFusion**。

---

# 联系方式

如果您对论文、源代码、预训练模型或 LI-PI 数据集有任何问题，欢迎在本仓库中提交 Issue。
