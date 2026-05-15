---
name: Plan F — Training improvements
overview: Two-stage fine-tuning with layer-wise learning rates. Phase 1 freezes backbone and trains FC only (10 epochs, lr=0.01); Phase 2 unfreezes all with backbone lr=1e-4 and FC lr=1e-3 (40 epochs, CosineAnnealingLR), keeping Plan E’s fear-only Mixup and label_smoothing=0.1. Goal: preserve ImageNet features before sparse fear gradients damage the backbone.
todos:
  - id: step1-modify-script
    content: "Edit Transfer Learning.py: train_model(use_mixup); two-stage section4 (Phase1 frozen no Mixup, Phase2 unfreeze + discriminative lr + Mixup); best_rafdb_model_f.pth"
    status: pending
  - id: step2-run-training
    content: Run full schedule (10+40 epochs)
    status: pending
  - id: step3-evaluate
    content: evaluate.py → best_rafdb_model_f.pth
    status: pending
  - id: step4-log
    content: Append #10 to EXPERIMENT_LOG.md
    status: pending
isProject: false
---

# Plan F — Two-stage fine-tuning (discriminative LR)

## Background

Runs #5–#9 all used **full fine-tuning** from the start at lr=0.001. Fear (~2.3% of batches) injects sparse gradients early and may harm ImageNet-pretrained backbone features.

## Design

|        | Phase 1 | Phase 2 |
|--------|---------|---------|
| Epochs | 10      | 40      |
| Freeze | Full backbone, train FC only | Unfreeze all |
| backbone lr | — | 1e-4 |
| FC lr  | 0.01    | 1e-3    |
| Scheduler | Constant (LambdaLR × 1.0) | CosineAnnealingLR(T_max=40) |
| Loss   | CrossEntropyLoss(weight=) | + label_smoothing=0.1 |
| Mixup  | Off     | Fear-only (MINORITY_CLASSES={2}, α=0.2) |

Phase 1: no label smoothing (FC must align first). Phase 2: same recipe as Plan E.

## Stop

1. `best_rafdb_model_f.pth` + full evaluation.
2. EXPERIMENT_LOG.md **#10** complete.

Do not change `api_server.py`, `CLAUDE.md`, frontend; do not delete old checkpoints.
