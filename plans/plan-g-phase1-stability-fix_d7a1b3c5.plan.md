---
name: Plan G — Training improvements
overview: Fix Plan F Phase 1 instability. Lower Phase 1 lr 0.01→0.001; replace fixed 10 epochs with patience=5 (no val-acc improvement) before Phase 2; Phase 2 unchanged (backbone 1e-4, FC 1e-3, CosineAnnealingLR T_max=40, fear-only Mixup, label_smoothing=0.1). New checkpoint best_rafdb_model_g.pth.
todos:
  - id: step1-modify-script
    content: "Edit Transfer Learning.py: train_model patience early-stop; Phase1 lr=0.001, max_epochs=30, patience=5; best_rafdb_model_g.pth"
    status: done
  - id: step2-run-training
    content: Phase1 early-stopped at epoch 14 (val acc 0.4540); Phase2 40 epochs (best val acc 0.7123)
    status: done
  - id: step3-evaluate
    content: evaluate.py → best_rafdb_model_g.pth
    status: done
  - id: step4-log
    content: Append #11 to EXPERIMENT_LOG.md
    status: done
isProject: false
---

# Plan G — Two-stage fine-tuning + Phase 1 convergence fix

## Background

Plan F (#10): Phase 1 at lr=0.01 made validation accuracy oscillate ~22–47%; Phase 2 started from a bad point → overall collapse ~70.80%. Hypothesis: stabilizing Phase 1 lets Phase 2 fear-Mixup work on better features.

## Design

|        | Phase 1 | Phase 2 |
|--------|---------|---------|
| Epochs | max 30, **patience=5** early stop | 40 |
| Freeze | Backbone frozen, FC only | Unfreeze all |
| backbone lr | — | 1e-4 |
| FC lr  | **0.001** (10× lower than Plan F 0.01) | 1e-3 |
| Scheduler | Constant LambdaLR×1.0 | CosineAnnealingLR(T_max=40) |
| Loss   | CrossEntropyLoss(weight=) | + label_smoothing=0.1 |
| Mixup  | Off | Fear-only (MINORITY_CLASSES={2}, α=0.2) |
| Early stop | patience=5 | none |

**Only difference vs Plan F**: Phase 1 lr 10× lower + patience-based stop.

## Results (as recorded in source plan)

- Phase 1: early stop at epoch 14, val acc **45.40%** (stable, no wild oscillation).
- Phase 2: full 40 epochs, best val acc **71.23%**.

| Class     | Train  | Test   |
|-----------|--------|--------|
| anger     | 88.79% | 77.16% |
| disgust   | 76.15% | 58.75% |
| fear      | 64.41% | 52.70% |
| happy     | 77.79% | 74.85% |
| neutral   | 70.09% | 65.74% |
| sad       | 71.04% | 66.74% |
| surprised | 89.46% | 84.50% |
| **overall** | **76.57%** | **71.35%** |

Notable errors: happy→fear 42, sad→fear 17, neutral→fear 11.

## Conclusion

Phase 1 stability fix worked, but **overall test 71.35%** ≈ Plan F (70.80%); **fear fell to 52.70%**. Fear-Mixup in Phase 2 still hurts when the root issue is **too few fear samples (254)**. Pure training tricks hit a ceiling; consider data augmentation (e.g. Plan H) next.

## Stop

1. `best_rafdb_model_g.pth` + evaluation done.
2. EXPERIMENT_LOG.md **#11** filled.

Do not change `api_server.py`, `CLAUDE.md`, frontend; do not delete old checkpoints.
