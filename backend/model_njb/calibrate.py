"""
后处理 logit bias 校准脚本（方案 J：置信度补偿）
==================================================
策略：在模型最终 logit 上加一个固定偏置向量 b，
      调整后预测 = argmax(logit + b)。
      b 通过在 RAF-DB 训练集 10% 验证分割上最大化 macro-accuracy 确定，
      然后在完整测试集上评估效果。

用法（从项目根目录执行）：
    python "backend/model_njb/calibrate.py"

输出：
  - backend/model_njb/bias_a.json       最优偏置向量
  - Graph/calibration_comparison.png    校准前后对比图
"""
import os, sys, json, itertools
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
for p in (BACKEND_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.emotion_model import load_model

# ── 路径 ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(SCRIPT_DIR, "best_rafdb_model_a.pth")
TRAIN_DIR   = os.path.join(REPO_ROOT, "data", "RAF-DB", "train")
TEST_DIR    = os.path.join(REPO_ROOT, "data", "RAF-DB", "test")
BIAS_OUT    = os.path.join(SCRIPT_DIR, "bias_a.json")
GRAPH_DIR   = os.path.join(REPO_ROOT, "Graph")
CLASSES     = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
N_CLS       = 7
VAL_FRAC    = 0.10   # 从训练集中划出 10% 作校准集

# ── 模型与预处理 ──────────────────────────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH, num_classes=7, device=device)
model.eval()


# ── 收集 logits ────────────────────────────────────────────────────────────────
def collect_logits(data_dir, subset_indices=None, batch_size=128):
    """返回 (logits_np: [N, 7], labels_np: [N])"""
    ds = datasets.ImageFolder(data_dir, transform=TRANSFORM)
    if subset_indices is not None:
        ds = Subset(ds, subset_indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def macro_acc(logits, labels, bias):
    preds = np.argmax(logits + bias, axis=1)
    per_cls = []
    for c in range(N_CLS):
        mask = labels == c
        if mask.sum() == 0:
            continue
        per_cls.append((preds[mask] == c).mean())
    return np.mean(per_cls)


def per_class_acc(logits, labels, bias):
    preds = np.argmax(logits + bias, axis=1)
    accs = []
    for c in range(N_CLS):
        mask = labels == c
        acc = (preds[mask] == c).mean() * 100 if mask.sum() else 0.0
        accs.append(acc)
    return accs


# ── 划分 val 集 ────────────────────────────────────────────────────────────────
full_ds = datasets.ImageFolder(TRAIN_DIR, transform=TRANSFORM)
n_total = len(full_ds)
rng = np.random.default_rng(42)
indices = rng.permutation(n_total)
n_val = int(n_total * VAL_FRAC)
val_idx   = indices[:n_val].tolist()
print(f"\nTrain split: {n_total - n_val} for train, {n_val} for calibration val")

print("Collecting val logits...")
val_logits, val_labels = collect_logits(TRAIN_DIR, subset_indices=val_idx)
print("Collecting test logits...")
test_logits, test_labels = collect_logits(TEST_DIR)

# ── 基准（无 bias）────────────────────────────────────────────────────────────
zero_bias = np.zeros(N_CLS)
base_macro   = macro_acc(val_logits, val_labels, zero_bias)
base_test_pc = per_class_acc(test_logits, test_labels, zero_bias)
base_overall = np.mean(base_test_pc)
print(f"\nBaseline val macro-acc: {base_macro*100:.2f}%")
print(f"Baseline test overall:  {base_overall:.2f}%")
print("Baseline test per-class:")
for c, a in zip(CLASSES, base_test_pc):
    print(f"  {c:<12} {a:.2f}%")

# ── 网格搜索 ──────────────────────────────────────────────────────────────────
# 仅搜索 fear(idx=2) 和 disgust(idx=1) 的偏置，其他类固定为 0
SEARCH = np.arange(-1.0, 4.5, 0.5)   # -1, -0.5, ..., 4.0

print(f"\nGrid search over fear/disgust bias ({len(SEARCH)}x{len(SEARCH)} = {len(SEARCH)**2} points)...")
best_score = -np.inf
best_b = zero_bias.copy()

for b_disgust, b_fear in itertools.product(SEARCH, SEARCH):
    b = zero_bias.copy()
    b[1] = b_disgust  # disgust
    b[2] = b_fear     # fear
    score = macro_acc(val_logits, val_labels, b)
    if score > best_score:
        best_score = score
        best_b = b.copy()

print(f"Best val macro-acc: {best_score*100:.2f}%")
print(f"Best bias  — disgust: {best_b[1]:+.1f}, fear: {best_b[2]:+.1f}")

# ── 可选：微调其他少数类 (anger) ──────────────────────────────────────────────
SEARCH_SM = np.arange(-1.0, 3.0, 0.5)
print("\nFine-tuning bias for anger (idx=0) around best_b...")
best_b2 = best_b.copy()
best_score2 = best_score
for b_anger in SEARCH_SM:
    b = best_b.copy()
    b[0] = b_anger
    score = macro_acc(val_logits, val_labels, b)
    if score > best_score2:
        best_score2 = score
        best_b2 = b.copy()
        best_b2[0] = b_anger

best_b = best_b2
print(f"After anger tuning — val macro-acc: {best_score2*100:.2f}%")
print(f"Final bias: {dict(zip(CLASSES, [f'{v:+.1f}' for v in best_b]))}")

# ── 保存 bias ─────────────────────────────────────────────────────────────────
bias_dict = {cls: float(best_b[i]) for i, cls in enumerate(CLASSES)}
with open(BIAS_OUT, "w") as f:
    json.dump(bias_dict, f, indent=2)
print(f"\nSaved bias to: {BIAS_OUT}")

# ── 测试集评估对比 ─────────────────────────────────────────────────────────────
calib_test_pc = per_class_acc(test_logits, test_labels, best_b)
calib_overall = np.mean(calib_test_pc)

print(f"\n{'='*55}")
print(f"{'类别':<12} {'基准 (%)':>10} {'校准后 (%)':>12} {'Δ (pp)':>8}")
print(f"{'-'*55}")
for c, base, calib in zip(CLASSES, base_test_pc, calib_test_pc):
    delta = calib - base
    mark = " ◀" if abs(delta) >= 1.0 else ""
    print(f"{c:<12} {base:>10.2f} {calib:>12.2f} {delta:>+8.2f}{mark}")
print(f"{'-'*55}")
print(f"{'Overall':<12} {base_overall:>10.2f} {calib_overall:>12.2f} {calib_overall-base_overall:>+8.2f}")
print(f"{'='*55}")

# ── 对比图 ─────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# 左：逐类对比柱状图
x = np.arange(N_CLS)
w = 0.38
ax = axes[0]
b1 = ax.bar(x - w/2, base_test_pc,  w, label="Baseline (no bias)",  color="#9ecae1", alpha=0.9)
b2 = ax.bar(x + w/2, calib_test_pc, w, label="Calibrated (+ logit bias)", color="#2166ac", alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=18, ha="right", fontsize=9.5)
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_title("Per-class Accuracy: Before vs After Calibration\n(Plan A backbone + post-hoc logit bias)", fontsize=10)
ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=9)

# 标注 fear/disgust 的差值
for i, (base, calib) in enumerate(zip(base_test_pc, calib_test_pc)):
    delta = calib - base
    if abs(delta) >= 1.0:
        top = max(base, calib) + 1.2
        ax.annotate(f"{delta:+.1f}pp", xy=(i + w/2, calib),
                    xytext=(i + w/2, top + 1.5), fontsize=8,
                    ha="center", color="#d62728" if delta < 0 else "#2166ac",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))

# 右：偏置值条形图（只显示非零 bias）
ax2 = axes[1]
bias_vals = [best_b[i] for i in range(N_CLS)]
colors_b  = ["#d62728" if v < 0 else "#2166ac" for v in bias_vals]
bars = ax2.barh(CLASSES[::-1], bias_vals[::-1], color=colors_b[::-1], alpha=0.85)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Logit Bias Value", fontsize=11)
ax2.set_title("Learned Logit Bias per Class\n(added to raw logits before argmax)", fontsize=10)
ax2.grid(axis="x", alpha=0.25)
for bar, val in zip(bars, bias_vals[::-1]):
    if abs(val) > 0.05:
        ax2.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
                 f"{val:+.1f}", va="center", ha="left" if val >= 0 else "right",
                 fontsize=9, color="#333333")

plt.tight_layout()
out_path = os.path.join(GRAPH_DIR, "calibration_comparison.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved figure: {out_path}")
print("\nDone.")
