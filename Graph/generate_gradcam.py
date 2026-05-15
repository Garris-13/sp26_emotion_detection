"""
Grad-CAM 可视化脚本
使用 Plan A 最优权重（best_rafdb_model_a.pth）对 RAF-DB 测试集图像生成热力图。
生成两张合图：
  1. gradcam_correct_grid.png  — 7 类各取 2 张正确预测样本
  2. gradcam_misclassify.png   — fear/disgust 典型误分样本（fear→surprised, disgust→anger）
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "backend", "model_njb", "best_rafdb_model_a.pth")
DATA_DIR   = os.path.join(ROOT, "data", "RAF-DB", "test")
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
IDX2CLS = {i: c for i, c in enumerate(CLASSES)}
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── 模型加载 ──────────────────────────────────────────────────────────────────
def load_resnet18(path, num_classes=7, device="cpu"):
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(path, map_location=device, weights_only=True)
    # 兼容带 backbone. 前缀的权重
    if next(iter(state)).startswith("backbone."):
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ── 图像预处理 ────────────────────────────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

INV_NORM = T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225],
)


# ── Grad-CAM 核心 ─────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[0, class_idx]
        score.backward()

        # GAP 加权
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)  # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        # 归一化到 [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def cam_to_heatmap(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """将 cam 插值到目标尺寸并转成 RGBA heatmap。"""
    from PIL import Image as PILImage
    pil = PILImage.fromarray((cam * 255).astype(np.uint8))
    pil = pil.resize(size, resample=PILImage.BILINEAR)
    arr = np.array(pil) / 255.0
    cmap = plt.get_cmap("jet")
    return cmap(arr)  # RGBA


def overlay(img_tensor: torch.Tensor, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """将 Grad-CAM 热力图叠加到原始图像上。"""
    # 反归一化 → numpy HWC [0,1]
    img_np = INV_NORM(img_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    h, w = img_np.shape[:2]
    heat = cam_to_heatmap(cam, (w, h))[:, :, :3]   # RGB
    blended = (1 - alpha) * img_np + alpha * heat
    return np.clip(blended, 0, 1)


# ── 收集预测结果 ──────────────────────────────────────────────────────────────
def collect_predictions(model, device):
    """遍历测试集，返回 {cls: {"correct": [...], "wrong": [(pred, path), ...]}}"""
    results = {c: {"correct": [], "wrong": []} for c in CLASSES}
    for cls_name in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        true_idx = CLS2IDX[cls_name]
        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".png"))])
        for fname in files:
            path = os.path.join(cls_dir, fname)
            img = Image.open(path).convert("RGB")
            x = TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = model(x).argmax(dim=1).item()
            entry = (path, x)
            if pred_idx == true_idx:
                results[cls_name]["correct"].append(entry)
            else:
                results[cls_name]["wrong"].append((IDX2CLS[pred_idx], path, x))
    return results


# ── 图 1：正确预测 7 类 2×7 网格 ─────────────────────────────────────────────
def plot_correct_grid(gradcam, results, device):
    n_per_class = 2
    fig, axes = plt.subplots(
        n_per_class * 2, len(CLASSES),
        figsize=(len(CLASSES) * 2.2, n_per_class * 2 * 2.2)
    )

    for col, cls_name in enumerate(CLASSES):
        samples = results[cls_name]["correct"]
        picked = random.sample(samples, min(n_per_class, len(samples)))
        # 如果某类样本不足则重复
        while len(picked) < n_per_class:
            picked.append(picked[-1])

        for row_pair, (path, x) in enumerate(picked):
            x_grad = x.clone().requires_grad_(False)
            x_for_cam = x_grad.to(device)
            x_for_cam.requires_grad_(True)
            # 重新过一遍（需要梯度）
            img_pil = Image.open(path).convert("RGB")
            x_g = TRANSFORM(img_pil).unsqueeze(0).to(device)

            cam = gradcam(x_g, class_idx=CLS2IDX[cls_name])
            blend = overlay(x_g, cam)

            orig_np = INV_NORM(x_g.squeeze(0)).permute(1, 2, 0).cpu().numpy()
            orig_np = np.clip(orig_np, 0, 1)

            r_orig = row_pair * 2
            r_cam  = row_pair * 2 + 1

            axes[r_orig][col].imshow(orig_np)
            axes[r_orig][col].axis("off")
            if row_pair == 0:
                axes[r_orig][col].set_title(cls_name, fontsize=9, fontweight="bold")

            axes[r_cam][col].imshow(blend)
            axes[r_cam][col].axis("off")

    # 行标注
    for row_pair in range(n_per_class):
        axes[row_pair * 2][0].set_ylabel("Original", fontsize=8, rotation=90, labelpad=4)
        axes[row_pair * 2 + 1][0].set_ylabel("Grad-CAM", fontsize=8, rotation=90, labelpad=4)

    fig.suptitle(
        "Grad-CAM: Correctly Classified Samples — Plan A (ResNet18 RAF-DB)\n"
        "Each column = emotion class; odd rows = original, even rows = Grad-CAM overlay",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "gradcam_correct_grid.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── 图 2：典型误分样本（fear→*, disgust→*）────────────────────────────────────
def plot_misclassify(gradcam, results, device):
    TARGET_PAIRS = [
        ("fear",    "surprised"),
        ("fear",    "sad"),
        ("disgust", "anger"),
        ("disgust", "sad"),
    ]
    samples_for_pair = {}
    for true_cls, pred_cls in TARGET_PAIRS:
        matches = [(p, x) for (pc, p, x) in results[true_cls]["wrong"] if pc == pred_cls]
        samples_for_pair[(true_cls, pred_cls)] = matches

    # 每对取 2 个（不足则取 1 或跳过）
    valid_pairs = [(pair, s) for pair, s in samples_for_pair.items() if len(s) > 0]
    if not valid_pairs:
        print("  No misclassification samples found, skipping misclassify plot.")
        return

    n_cols = len(valid_pairs)
    n_pick = 2
    fig, axes = plt.subplots(n_pick * 2, n_cols, figsize=(n_cols * 2.4, n_pick * 2 * 2.4))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, ((true_cls, pred_cls), samps) in enumerate(valid_pairs):
        picked = random.sample(samps, min(n_pick, len(samps)))
        while len(picked) < n_pick:
            picked.append(picked[-1])

        true_idx = CLS2IDX[true_cls]

        for row_pair, (path, x) in enumerate(picked):
            img_pil = Image.open(path).convert("RGB")
            x_g = TRANSFORM(img_pil).unsqueeze(0).to(device)

            # Grad-CAM for predicted class
            cam_pred = gradcam(x_g, class_idx=CLS2IDX[pred_cls])
            # Grad-CAM for true class
            cam_true = gradcam(x_g, class_idx=true_idx)

            blend_pred = overlay(x_g, cam_pred)
            blend_true = overlay(x_g, cam_true)

            r_pred = row_pair * 2
            r_true = row_pair * 2 + 1

            axes[r_pred][col].imshow(blend_pred)
            axes[r_pred][col].axis("off")
            axes[r_true][col].imshow(blend_true)
            axes[r_true][col].axis("off")

            if row_pair == 0:
                axes[r_pred][col].set_title(
                    f"True: {true_cls}\nPred: {pred_cls}",
                    fontsize=8.5, color="#d62728", fontweight="bold"
                )

        axes[0][col].set_ylabel(f"CAM@pred\n({pred_cls})", fontsize=7.5, rotation=90, labelpad=3)
        axes[1][col].set_ylabel(f"CAM@true\n({true_cls})", fontsize=7.5, rotation=90, labelpad=3)

    fig.suptitle(
        "Grad-CAM on Misclassified Samples — fear & disgust (Plan A)\n"
        "Top row: attention for predicted class | Bottom row: attention for true class",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "gradcam_misclassify.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_PATH}")

    model = load_resnet18(MODEL_PATH, device=device)
    # ResNet18 最后一个卷积块 layer4[1].conv2 是标准 Grad-CAM target
    target_layer = model.layer4[1].conv2
    gradcam = GradCAM(model, target_layer)

    print("Collecting predictions on test set...")
    results = collect_predictions(model, device)

    # 统计
    for cls in CLASSES:
        nc = len(results[cls]["correct"])
        nw = len(results[cls]["wrong"])
        print(f"  {cls:10s}: correct={nc:3d}  wrong={nw:3d}")

    print("\nGenerating gradcam_correct_grid.png ...")
    plot_correct_grid(gradcam, results, device)

    print("Generating gradcam_misclassify.png ...")
    plot_misclassify(gradcam, results, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
