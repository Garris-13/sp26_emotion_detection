# Experimental narrative & reproducibility notes

This document complements [`README.md`](./README.md) (how to run the stack). It summarizes the **controlled diagnostic study** documented in the course final report: *where minority-class facial expression recognition (FER) bottlenecks come from* under a fixed setting—**ImageNet-pretrained ResNet18**, **RAF-DB** (12,271 train / 3,068 test, 7 classes), and **YOLOv8-face** for real-time webcam crops.

**Full write-up (LaTeX):**

- [`paper/Final_Report_EN (1).tex`](./paper/Final_Report_EN%20(1).tex) — English report (ICML-style single column; compile with pdfLaTeX).
- [`paper/Final_Report_CN.tex`](./paper/Final_Report_CN.tex) — Chinese version (XeLaTeX + `ctex`).

**Chronological engineering + metrics log:** [`EXPERIMENT_LOG.md`](./EXPERIMENT_LOG.md) (append-only entries; **#N** template).

---

## Research question

RAF-DB is **highly class-imbalanced** (e.g. **281** training images for *fear*, **717** for *disgust*, **4,772** for *happy*). After pilot runs, majority classes exceed ~90% accuracy while *fear* / *disgust* stall below ~60%. We ask:

> **When a model under-performs on specific minority classes, where is the limit—optimization, representation, data volume, or something else?**

We structure **ten matched training or inference schemes** (Plans **Base, A–G, H, I, J**) into **four diagnostic phases** so each phase falsifies or refines a candidate explanation.

---

## Hypotheses (treated as separable levers)

| ID | Hypothesis | Intuition |
|----|------------|-----------|
| **H1** | Optimization / regularization is insufficient | Better schedules, augmentations, re-balancing, robust losses, Mixup, two-stage fine-tuning should break the ceiling if this is the only issue. |
| **H2** | ImageNet features are not **linearly separable** for minority classes | A **linear probe** (frozen backbone, train only the FC head) should approach full fine-tuning if features are already aligned. |
| **H3** | Raw **sample count** is the binding constraint | Adding many more *fear* images should monotonically help *fear* test accuracy if count alone matters. |

The report argues that **H1 alone** cannot explain the full story, **H2** is partially supported (probe collapses on *fear* / *disgust*), and **H3** in the strong form “more images always help” fails when new data are **domain-mismatched** (Plan H).

---

## Four phases (Plans)

| Phase | Plans | What is tested | Takeaway (high level) |
|-------|-------|----------------|------------------------|
| **P1** | A–G | H1: optimization & long-tail tricks | Strong single-stage tuning (especially **Plan A**) reaches high **overall** RAF-DB test accuracy, but *fear* hits a **~60–63%** band; aggressive re-balancing (**Plan B**) can **hurt** minority classes. |
| **P2** | **I** (linear probe) | H2: linear separability of frozen features | Overall accuracy **49.93%**; *fear* **29.73%**, *disgust* **14.38%** vs *happy* still ~82%—fine-tuning the backbone is doing most of the work for hard classes. |
| **P3** | **H** (FER2013 *fear* ×17) | H3: raw count vs **domain alignment** | *fear* **drops** from Plan-A level (**54.05% → 50.00%**); cross-domain mixing hurts—**domain alignment** matters more than raw count. |
| **P4** | **J** (post-hoc **logit bias**) | Inference-only remedy | **Fear-only** offset **+2.0** on logits before softmax: *fear* **62.16%** (best *fear* among all ten setups in the paper’s table) with **−0.19 pp** overall vs Plan A (**84.71% → 84.52%**). |

**Grad-CAM** (report §5.5) links residual *fear* ↔ *surprised* confusion to **shared upper-face / AU5** attention when training signal is scarce.

---

## Scheme cheat-sheet (relative to bug-fixed **Base**)

| Plan | Core idea |
|------|-----------|
| **Base** | Label-order & class-weight fixes only; baseline reproduction. |
| **A** | Cosine annealing + stronger augmentations (main “strong single-stage” reference). |
| **B** | WeightedRandomSampler + Focal Loss (illustrates over-aggressive minority oversampling). |
| **C–E** | Global vs selective **Mixup** (+ label smoothing in C). |
| **F–G** | Two-stage fine-tune (head then backbone); **F** peaks *fear* training-side but **hurts overall** badly. |
| **H** | Concatenate **FER2013** *fear* images (17× *fear* count); tests domain shift. |
| **I** | **Linear probe**—frozen ResNet18, train FC only. |
| **J** | **Post-hoc** logit bias at inference; **fear-only +2.0** is the shipped default (see below). |

Full factorization table: report **Table “Ten controlled schemes”** (`Approach` → `The ten schemes`).

---

## Headline test-set numbers (RAF-DB official test, **N = 3,068**)

Reported **overall** and selected **minority** cells (see report Table “Per-class and overall test accuracy” for the full matrix):

| Setup | Overall | *fear* | Notes |
|-------|---------|--------|--------|
| **Base** (fixed) | 82.11% | 60.81% | Before tricks; *fear* higher than Plan A but overall lower. |
| **Plan A** | **84.71%** | 54.05% | Best overall in the study’s main single-stage cluster. |
| **Plan I** (linear probe) | 49.93% | 29.73% | Probes H2. |
| **Plan H** | 84.00% | 50.00% | Probes H3 / domain. |
| **Plan J** (fear-only bias) | 84.52% | **62.16%** | Inference-only; uses Plan A weights + bias. |

Figures (bar charts, Pareto, Δ heatmaps, train–test gap, calibration panels, Grad-CAM grids) are produced under **`Graph/`**; the driver script is [`Graph/generate_paper_figures.py`](./Graph/generate_paper_figures.py).

---

## Inference calibration (Plan J in code)

The API applies an optional **per-class logit bias** before softmax, loaded from:

- [`backend/model_njb/bias_fear_only.json`](./backend/model_njb/bias_fear_only.json) — **fear +2.0**, other classes **0** (production-friendly per `EXPERIMENT_LOG.md` entry on calibration).

Implementation: [`backend/api/api_server.py`](./backend/api/api_server.py) (search `LOGIT_BIAS` / `bias_fear_only`).  
Offline search for bias vectors: [`backend/model_njb/calibrate.py`](./backend/model_njb/calibrate.py).

If the JSON file is missing, bias defaults to **zero** (Plan-A–equivalent behaviour).

---

## Training vs report settings

- **Entry script:** [`backend/model_njb/Transfer Learning.py`](./backend/model_njb/Transfer%20Learning.py) — **line ~15 `data_dir`** must point at your local `data/RAF-DB` (repo default may still reference an old absolute path; see [`README.md`](./README.md) §Training).
- The report’s **Plan A** and follow-on plans use a **matched protocol** (90/10 train split from official train, official test held out, Adam vs SGD where noted, 50 epochs, etc.); exact knobs are in the LaTeX **Shared configuration** subsection—not every knob is duplicated in the training script comments, so treat **`EXPERIMENT_LOG.md` + paper** as the source of truth for ablation semantics.

---

## Known limitations (aligned with report + log)

1. **Small distant faces:** ~40×40 YOLO boxes → `Resize(256)` → heavy blur → softmax near uniform (~1/7). Mitigation direction: **blur / resolution-down-up augmentation** at train time (tracked in `EXPERIMENT_LOG.md`).
2. **Cross-domain *fear* augmentation** without alignment (Plan H narrative) can **hurt** *fear* and *disgust*—data **quality and domain** beat naive count.

---

## Citation bundle (report `thebibliography`)

RAF-DB, FER2013, ResNet, Focal Loss, Mixup, YOLO/Ultralytics, Grad-CAM—see the `.tex` **References** section for full bib entries.

---

## Quick links

| Artifact | Role |
|----------|------|
| [`README.md`](./README.md) | Install, dual-server demo, API curl examples |
| [`README_cn.md`](./README_cn.md) | 中文运行说明 |
| [`EXPERIMENT_LOG.md`](./EXPERIMENT_LOG.md) | Dated changes, calibration numbers, file touch list |
| `paper/*.tex` | Full narrative, tables, figure captions |
| `Graph/*.png` | Paper-facing plots (regenerate via `generate_paper_figures.py` where applicable) |
