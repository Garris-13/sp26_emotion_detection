import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

# ── 所有方案数据（含 H 和 Linear Probe I）────────────────────────────────────
SCHEMES  = ["baseline", "A", "B", "C", "D", "E", "F", "G", "H", "LP(I)"]
OVERALL  = [82.11,      84.71, 83.70, 82.53, 83.28, 82.17, 70.80, 71.35, 84.00, 49.93]
FEAR_ACC = [60.81,      54.05, 47.30, 58.11, 60.81, 60.81, 63.51, 52.70, 50.00, 29.73]

PER_CLASS = {
    "baseline": [73.46, 53.75, 60.81, 92.15, 76.62, 80.13, 82.98],
    "A":        [74.69, 58.12, 54.05, 92.24, 81.76, 85.36, 87.54],
    "B":        [77.78, 47.50, 47.30, 91.31, 83.09, 83.68, 86.32],
    "C":        [77.16, 65.62, 58.11, 87.00, 81.32, 82.22, 85.71],
    "D":        [73.46, 52.50, 60.81, 88.86, 84.56, 81.38, 88.15],
    "E":        [82.10, 58.12, 60.81, 87.17, 79.85, 81.80, 86.02],
    "F":        [79.63, 58.12, 63.51, 74.26, 65.00, 67.78, 78.12],
    "G":        [77.16, 58.75, 52.70, 74.85, 65.74, 66.74, 84.50],
    "H":        [82.10, 48.75, 50.00, 90.55, 80.29, 88.28, 87.54],
    "LP(I)":    [24.69, 14.38, 29.73, 81.69, 29.71, 22.59, 51.37],
}

# fear 类训练集 / 测试集准确率（用于间隙图）
FEAR_GAP = {
    #          train   test
    "baseline": (96.80, 60.81),
    "A":        (95.37, 54.05),
    "B":        (95.37, 47.30),
    "C":        (97.51, 58.11),
    "D":        (96.80, 60.81),
    "E":        (96.09, 60.81),
    "F":        (85.05, 63.51),
    "G":        (64.41, 52.70),
    "H":        (69.75, 50.00),
    "LP(I)":    (26.69, 29.73),
}

OUT = os.path.dirname(os.path.abspath(__file__))


# ── 1. 总体准确率柱状图（含 H 和 LP）────────────────────────────────────────
def plot_overall_bar():
    schemes = SCHEMES
    overall = OVERALL
    colors = []
    for s in schemes:
        if s == "A":        colors.append("#2166ac")
        elif s in ("H",):   colors.append("#d6604d")
        elif s == "LP(I)":  colors.append("#999999")
        else:               colors.append("#9ecae1")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.bar(schemes, overall, color=colors)
    ax.set_ylabel("Test Overall Accuracy (%)", fontsize=11)
    ax.set_title("Overall Accuracy by Scheme (RAF-DB Test Set, A–I)", fontsize=12)
    ax.set_ylim(40, 92)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(84.71, color="#2166ac", linewidth=1.2, linestyle="--", alpha=0.6, label="Plan A best (84.71%)")
    for bar, val in zip(bars, overall):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    legend_patches = [
        mpatches.Patch(color="#2166ac", label="Plan A (best overall)"),
        mpatches.Patch(color="#d6604d", label="Plan H (cross-domain mix)"),
        mpatches.Patch(color="#999999", label="Plan I (Linear Probe)"),
        mpatches.Patch(color="#9ecae1", label="Other"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "overall_accuracy_bar.png"), dpi=300)
    plt.close()


# ── 2. Pareto 散点图（总体 vs fear，含 H 和 LP）──────────────────────────────
def plot_pareto_scatter():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    color_map = {"A": "#d62728", "H": "#ff7f0e", "LP(I)": "#999999"}
    for s, x, y in zip(SCHEMES, FEAR_ACC, OVERALL):
        c = color_map.get(s, "#1f77b4")
        ax.scatter(x, y, s=80, c=c, zorder=3)
        offset_x, offset_y = 0.5, 0.3
        if s == "LP(I)": offset_y = -1.5
        if s == "F":     offset_y = -1.5
        ax.annotate(s, (x, y), xytext=(x + offset_x, y + offset_y), fontsize=9)

    ax.set_xlabel("fear Accuracy (%)", fontsize=11)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
    ax.set_title("Trade-off: Overall vs. fear Accuracy (All Schemes)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.axvline(54.05, color="#d62728", linewidth=1, linestyle=":", alpha=0.5)
    ax.axhline(84.71, color="#d62728", linewidth=1, linestyle=":", alpha=0.5)
    legend_patches = [
        mpatches.Patch(color="#d62728", label="Plan A"),
        mpatches.Patch(color="#ff7f0e", label="Plan H (cross-domain)"),
        mpatches.Patch(color="#999999", label="Plan I (Linear Probe)"),
        mpatches.Patch(color="#1f77b4", label="Other plans"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pareto_overall_vs_fear.png"), dpi=300)
    plt.close()


# ── 3. per-class Δ 热力图（相对方案 A，含 H 和 LP）──────────────────────────
def plot_per_class_delta_heatmap():
    ref = np.array(PER_CLASS["A"])
    row_names, rows = [], []
    for s in SCHEMES:
        if s in ("A", "baseline"):
            continue
        rows.append(np.array(PER_CLASS[s]) - ref)
        row_names.append(s)

    matrix = np.array(rows)
    vmax = max(np.max(np.abs(matrix)), 5)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_xticklabels(CLASSES, rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(row_names, fontsize=10)
    ax.set_title("Per-class Accuracy Δ vs Plan A (pp)  |  Blue=better, Red=worse", fontsize=11)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Δ Accuracy (pp)", fontsize=9)

    # 分割线区分技巧实验 vs H/LP
    ax.axhline(6.5, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.text(len(CLASSES) - 0.45, 6.7, "↓ Data/Probe Exp.", fontsize=7.5,
            ha="right", color="black", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "per_class_delta_vs_A_heatmap.png"), dpi=300)
    plt.close()


# ── 4. RAF-DB 类别分布柱状图（不变）────────────────────────────────────────
def plot_class_distribution():
    train_counts = [705, 717, 281, 4772, 2524, 1982, 1290]
    test_counts  = [162, 160,  74, 1185,  680,  478,  329]
    x = np.arange(len(CLASSES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(x - width / 2, train_counts, width, label="Train", color="#6baed6")
    ax.bar(x + width / 2, test_counts,  width, label="Test",  color="#fd8d3c")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_title("RAF-DB Class Distribution (Train / Test)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "rafdb_class_distribution.png"), dpi=300)
    plt.close()


# ── 5. [NEW] Linear Probe vs Full Fine-tuning 逐类对比 ──────────────────────
def plot_lp_vs_finetune():
    lp  = np.array(PER_CLASS["LP(I)"])
    ft  = np.array(PER_CLASS["A"])
    x   = np.arange(len(CLASSES))
    w   = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x - w / 2, ft,  w, label="Plan A (Full Fine-tuning)",  color="#2166ac", alpha=0.85)
    b2 = ax.bar(x + w / 2, lp,  w, label="Plan I (Linear Probe)",      color="#d1e5f0", edgecolor="#4393c3", linewidth=0.8)

    # 差值标注
    for i, (f, l) in enumerate(zip(ft, lp)):
        delta = f - l
        ax.annotate(f"−{delta:.1f}", xy=(i + w / 2, l),
                    xytext=(i + w / 2, l + 1.5), fontsize=7.5,
                    ha="center", color="#d62728")

    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Linear Probe vs Full Fine-tuning — Per-class Accuracy (RAF-DB Test)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)

    ax.annotate("ImageNet features\nbarely separate\nfear/disgust",
                xy=(1.5, 14), fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"),
                xytext=(3.2, 25))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lp_vs_finetune.png"), dpi=300)
    plt.close()


# ── 6. [NEW] fear 类训练集/测试集间隙图（所有方案）──────────────────────────
def plot_fear_gap():
    labels = list(FEAR_GAP.keys())
    trains = [FEAR_GAP[k][0] for k in labels]
    tests  = [FEAR_GAP[k][1] for k in labels]
    gaps   = [t - e for t, e in zip(trains, tests)]

    x   = np.arange(len(labels))
    w   = 0.3

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, trains, w, label="Train fear acc (%)", color="#6baed6", alpha=0.9)
    ax.bar(x + w / 2, tests,  w, label="Test fear acc (%)",  color="#fd8d3c", alpha=0.9)

    ax2 = ax.twinx()
    ax2.plot(x, gaps, "D-", color="#d62728", linewidth=1.5, markersize=5,
             label="Train−Test Gap (pp)", zorder=5)
    ax2.axhline(0, color="#d62728", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Train − Test Gap (pp)", fontsize=10, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("fear Accuracy (%)", fontsize=11)
    ax.set_title("fear Class: Train / Test Accuracy and Generalization Gap Across All Schemes", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fear_train_test_gap.png"), dpi=300)
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    plot_overall_bar()
    plot_pareto_scatter()
    plot_per_class_delta_heatmap()
    plot_class_distribution()
    plot_lp_vs_finetune()
    plot_fear_gap()

    print("Done. Saved figures:")
    for name in ["overall_accuracy_bar.png", "pareto_overall_vs_fear.png",
                 "per_class_delta_vs_A_heatmap.png", "rafdb_class_distribution.png",
                 "lp_vs_finetune.png", "fear_train_test_gap.png"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
