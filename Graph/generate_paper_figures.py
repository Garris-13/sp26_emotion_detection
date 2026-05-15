"""Generate all figures for the CS289A final report (English, paper-ready).

Produces 8 matplotlib figures into Graph/. Two additional Grad-CAM figures
(gradcam_correct_grid.png, gradcam_misclassify.png) are produced by
generate_gradcam.py and reused as-is.

Run:
    python Graph/generate_paper_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Data tables (RAF-DB test set, 3068 images) ───────────────────────────────
CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

SCHEMES  = ["Base", "A", "B", "C", "D", "E", "F", "G", "H", "I (LP)", "J (bias)"]
OVERALL  = [82.11, 84.71, 83.70, 82.53, 83.28, 82.17, 70.80, 71.35, 84.00, 49.93, 84.52]
FEAR_ACC = [60.81, 54.05, 47.30, 58.11, 60.81, 60.81, 63.51, 52.70, 50.00, 29.73, 62.16]

PER_CLASS = {
    "Base":     [73.46, 53.75, 60.81, 92.15, 76.62, 80.13, 82.98],
    "A":        [74.69, 58.12, 54.05, 92.24, 81.76, 85.36, 87.54],
    "B":        [77.78, 47.50, 47.30, 91.31, 83.09, 83.68, 86.32],
    "C":        [77.16, 65.62, 58.11, 87.00, 81.32, 82.22, 85.71],
    "D":        [73.46, 52.50, 60.81, 88.86, 84.56, 81.38, 88.15],
    "E":        [82.10, 58.12, 60.81, 87.17, 79.85, 81.80, 86.02],
    "F":        [79.63, 58.12, 63.51, 74.26, 65.00, 67.78, 78.12],
    "G":        [77.16, 58.75, 52.70, 74.85, 65.74, 66.74, 84.50],
    "H":        [82.10, 48.75, 50.00, 90.55, 80.29, 88.28, 87.54],
    "I (LP)":   [24.69, 14.38, 29.73, 81.69, 29.71, 22.59, 51.37],
    "J (bias)": [71.60, 58.13, 62.16, 91.98, 81.76, 84.73, 87.23],
}

FEAR_GAP = {
    "Base":   (96.80, 60.81),
    "A":      (95.37, 54.05),
    "B":      (95.37, 47.30),
    "C":      (97.51, 58.11),
    "D":      (96.80, 60.81),
    "E":      (96.09, 60.81),
    "F":      (85.05, 63.51),
    "G":      (64.41, 52.70),
    "H":      (69.75, 50.00),
    "I (LP)": (26.69, 29.73),
}

# RAF-DB class counts (from train/ and test/ subdirs)
TRAIN_COUNTS = [705, 717, 281, 4772, 2524, 1982, 1290]
TEST_COUNTS  = [162, 160,  74, 1185,  680,  478,  329]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved: {name}")


# ── Fig 1: end-to-end pipeline schematic ─────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")

    boxes = [
        ("Webcam frame\n(base64)",            "#cfe2f3", 0.30),
        ("YOLOv8-face\n(conf $\\geq$ 0.45)",  "#9fc5e8", 2.30),
        ("Crop + 25% pad",                    "#9fc5e8", 4.30),
        ("Resize 256 $\\to$\nCenterCrop 224", "#9fc5e8", 6.30),
        ("ResNet18\n(ImageNet $\\to$ FT)",    "#f9cb9c", 8.30),
        ("softmax\n($+$ logit bias, opt.)",   "#b6d7a8", 10.30),
    ]
    for txt, color, x0 in boxes:
        rect = FancyBboxPatch((x0, 1.0), 1.55, 1.1,
                              boxstyle="round,pad=0.04,rounding_size=0.12",
                              fc=color, ec="#444", lw=0.9)
        ax.add_patch(rect)
        ax.text(x0 + 0.78, 1.55, txt, ha="center", va="center",
                fontsize=9.2)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][2] + 1.55
        x1 = boxes[i + 1][2]
        arr = FancyArrowPatch((x0 + 0.02, 1.55), (x1 - 0.02, 1.55),
                              arrowstyle="->", mutation_scale=14, lw=1.1, color="#333")
        ax.add_patch(arr)

    ax.text(6.0, 0.55,
            "Output: $\\hat{y} \\in \\mathbb{R}^7$  (anger, disgust, fear, happy, neutral, sad, surprised)",
            ha="center", va="center", fontsize=9.3, color="#222")
    ax.text(6.0, 2.55, "End-to-end real-time inference pipeline",
            ha="center", va="center", fontsize=10.5, fontweight="bold")
    save("fig_pipeline.png")


# ── Fig 2: class distribution ────────────────────────────────────────────────
def fig_class_distribution():
    x = np.arange(len(CLASSES)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(x - w/2, TRAIN_COUNTS, w, label="Train (12,271)", color="#6baed6")
    ax.bar(x + w/2, TEST_COUNTS,  w, label="Test (3,068)",   color="#fd8d3c")
    for i, (t, e) in enumerate(zip(TRAIN_COUNTS, TEST_COUNTS)):
        ax.text(i - w/2, t + 80, f"{t}",  ha="center", fontsize=8)
        ax.text(i + w/2, e + 80, f"{e}",  ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("RAF-DB class distribution: fear and disgust are severely under-represented")
    ax.set_ylim(0, max(TRAIN_COUNTS) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    # Annotate the bottleneck classes
    ax.annotate("fear: 281 train / 74 test\n(2.3% of train images)",
                xy=(2 - w/2, 281), xytext=(0.3, 3700),
                arrowprops=dict(arrowstyle="->", color="#b00", lw=1.0),
                fontsize=9, color="#b00")
    save("fig_class_distribution.png")


# ── Fig 3: overall accuracy bar (all 11 schemes) ─────────────────────────────
def fig_overall_bar():
    colors = []
    for s in SCHEMES:
        if s == "A":            colors.append("#2166ac")
        elif s == "J (bias)":   colors.append("#1a9850")
        elif s == "I (LP)":     colors.append("#999999")
        elif s == "H":          colors.append("#d6604d")
        elif s in ("F", "G"):   colors.append("#fcae91")
        elif s == "Base":       colors.append("#bbbbbb")
        else:                   colors.append("#9ecae1")

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    bars = ax.bar(SCHEMES, OVERALL, color=colors, edgecolor="#333", lw=0.4)
    ax.set_ylabel("Test overall accuracy (%)", fontsize=11)
    ax.set_title("Overall accuracy across 10 controlled schemes (RAF-DB test, 3,068 images)",
                 fontsize=11.5)
    ax.set_ylim(40, 98)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(84.71, color="#2166ac", lw=1.0, ls="--", alpha=0.7,
               label="Plan A (84.71%)")
    for bar, v in zip(bars, OVERALL):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.6, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    legend = [
        mpatches.Patch(color="#2166ac", label="A: cosine + augmentation (best overall)"),
        mpatches.Patch(color="#1a9850", label="J: post-hoc fear-only logit bias"),
        mpatches.Patch(color="#d6604d", label="H: cross-domain (FER2013) mix"),
        mpatches.Patch(color="#999999", label="I: linear probe (frozen backbone)"),
        mpatches.Patch(color="#fcae91", label="F, G: two-stage fine-tuning (failed)"),
        mpatches.Patch(color="#9ecae1", label="B--E: sampling / loss variants"),
        mpatches.Patch(color="#bbbbbb", label="Base: pre-correction"),
    ]
    ax.legend(handles=legend, fontsize=8.6, loc="upper center",
              ncol=4, framealpha=0.95, bbox_to_anchor=(0.5, 1.0))
    save("fig_overall_bar.png")


# ── Fig 4: Pareto: overall vs fear accuracy ──────────────────────────────────
def fig_pareto():
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    cm = {"A": "#d62728", "J (bias)": "#1a9850", "H": "#ff7f0e", "I (LP)": "#777777"}
    for s, x, y in zip(SCHEMES, FEAR_ACC, OVERALL):
        c = cm.get(s, "#1f77b4")
        ax.scatter(x, y, s=110, c=c, zorder=3, edgecolor="white", lw=0.8)
        dx, dy = 0.6, 0.4
        if s == "I (LP)":  dy = -1.8
        if s == "F":       dx, dy = -2.4, -0.2
        if s == "B":       dx, dy = -2.0, -0.2
        if s == "Base":    dy = -1.4
        if s == "D":       dx, dy = 0.5, -1.3
        if s == "E":       dx, dy = 0.5, 0.6
        ax.annotate(s, (x, y), xytext=(x + dx, y + dy), fontsize=9.5,
                    fontweight="bold" if s in cm else "normal")

    # Pareto frontier (upper envelope)
    pts = sorted(zip(FEAR_ACC, OVERALL))
    front = []
    best = -np.inf
    for fx, fy in pts[::-1]:
        if fy > best:
            front.append((fx, fy)); best = fy
    front = sorted(front)
    fx, fy = zip(*front)
    ax.plot(fx, fy, "--", color="#666", alpha=0.5, label="Pareto frontier")

    ax.axvline(54.05, color="#d62728", lw=0.8, ls=":", alpha=0.4)
    ax.axhline(84.71, color="#d62728", lw=0.8, ls=":", alpha=0.4)
    ax.set_xlabel("fear test accuracy (%)", fontsize=11)
    ax.set_ylabel("Overall test accuracy (%)", fontsize=11)
    ax.set_title("Pareto trade-off: overall vs fear accuracy", fontsize=11.5)
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(25, 70); ax.set_ylim(45, 88)
    save("fig_pareto.png")


# ── Fig 5: Per-class Δ vs Plan A heatmap ─────────────────────────────────────
def fig_per_class_delta():
    ref = np.array(PER_CLASS["A"])
    rows, row_names = [], []
    order = ["Base", "B", "C", "D", "E", "F", "G", "H", "I (LP)", "J (bias)"]
    for s in order:
        rows.append(np.array(PER_CLASS[s]) - ref)
        row_names.append(s)
    M = np.array(rows)
    vmax = max(float(np.max(np.abs(M))), 5.0)

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_xticklabels(CLASSES, rotation=20, ha="right", fontsize=10)
    ax.set_yticklabels(row_names, fontsize=10)
    ax.set_title("Per-class accuracy $\\Delta$ vs Plan A (pp). Blue = better, red = worse.",
                 fontsize=11)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("$\\Delta$ accuracy (pp)", fontsize=9)

    # Phase dividers
    # rows: Base(0) | B-G tricks(1-6) | H(7) | I(8) | J(9)
    for y in (0.5, 6.5, 7.5, 8.5):
        ax.axhline(y, color="black", lw=1.0, ls="--", alpha=0.6)
    save("fig_per_class_delta.png")


# ── Fig 6: Fear train/test/gap across schemes ────────────────────────────────
def fig_fear_gap():
    labels = list(FEAR_GAP.keys())
    tr = [FEAR_GAP[k][0] for k in labels]
    te = [FEAR_GAP[k][1] for k in labels]
    gap = [a - b for a, b in zip(tr, te)]
    x = np.arange(len(labels)); w = 0.32

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(x - w/2, tr, w, label="fear train acc (%)", color="#6baed6")
    ax.bar(x + w/2, te, w, label="fear test acc (%)",  color="#fd8d3c")
    ax2 = ax.twinx()
    ax2.plot(x, gap, "D-", color="#d62728", lw=1.5, ms=5,
             label="train$-$test gap (pp)", zorder=5)
    ax2.set_ylabel("train $-$ test gap (pp)", color="#d62728", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.axhline(0, color="#d62728", lw=0.6, ls=":")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("fear accuracy (%)", fontsize=11)
    ax.set_title("fear class: train/test accuracy and generalization gap across schemes",
                 fontsize=11.5)
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    save("fig_fear_gap.png")


# ── Fig 7: Linear Probe vs Full FT per class ─────────────────────────────────
def fig_lp_vs_ft():
    ft = np.array(PER_CLASS["A"])
    lp = np.array(PER_CLASS["I (LP)"])
    x = np.arange(len(CLASSES)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.bar(x - w/2, ft, w, label="Plan A: full fine-tuning",
           color="#2166ac")
    ax.bar(x + w/2, lp, w, label="Plan I: linear probe (frozen backbone)",
           color="#d1e5f0", edgecolor="#4393c3", lw=0.8)
    for i, (f, l) in enumerate(zip(ft, lp)):
        ax.annotate(f"$-${f - l:.1f}", xy=(i + w/2, l),
                    xytext=(i + w/2, l + 2.5), ha="center",
                    fontsize=8.2, color="#b00")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title("Linear probe vs full fine-tuning — per class (RAF-DB test)",
                 fontsize=11.5)
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9.5, loc="upper left")

    ax.annotate("ImageNet features\nhardly separate\nfear / disgust",
                xy=(1.5, 14), xytext=(3.4, 22),
                arrowprops=dict(arrowstyle="->", color="#b00"),
                fontsize=9, color="#b00")
    save("fig_lp_vs_ft.png")


# ── Fig 8: Post-hoc calibration (Plan A vs Fear-only bias) ───────────────────
def fig_calibration():
    a = np.array(PER_CLASS["A"])
    j = np.array(PER_CLASS["J (bias)"])
    overall = {"Plan A": 84.71, "+ fear bias (+2.0)": 84.52, "Full bias": 83.87}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                             gridspec_kw={"width_ratios": [2.2, 1]})

    # Left: per-class bars
    ax = axes[0]
    x = np.arange(len(CLASSES)); w = 0.38
    ax.bar(x - w/2, a, w, label="Plan A (no bias)",  color="#2166ac")
    ax.bar(x + w/2, j, w, label="+ fear bias (+2.0)", color="#1a9850")
    for i, (av, jv) in enumerate(zip(a, j)):
        d = jv - av
        ax.annotate(f"{d:+.1f}", xy=(i + w/2, jv),
                    xytext=(i + w/2, jv + 1.6), ha="center",
                    fontsize=8.2, color=("#1a9850" if d >= 0 else "#b00"))
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title("Per-class effect of post-hoc fear-only logit bias", fontsize=11.5)
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9.5, loc="upper left")

    # Right: overall accuracy comparison incl. full bias
    ax = axes[1]
    names = list(overall.keys())
    vals  = list(overall.values())
    colors = ["#2166ac", "#1a9850", "#d6604d"]
    bars = ax.bar(names, vals, color=colors, edgecolor="#333", lw=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(82, 86)
    ax.set_ylabel("Overall accuracy (%)", fontsize=11)
    ax.set_title("Overall: $\\leq$ 0.84 pp cost", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9.5)

    save("fig_calibration.png")


def main():
    print("Generating paper figures into Graph/ ...")
    fig_pipeline()
    fig_class_distribution()
    fig_overall_bar()
    fig_pareto()
    fig_per_class_delta()
    fig_fear_gap()
    fig_lp_vs_ft()
    fig_calibration()
    print("Done.")


if __name__ == "__main__":
    main()
