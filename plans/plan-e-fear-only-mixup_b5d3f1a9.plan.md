---
name: Plan E — Training improvements
overview: On Plan D, shrink MINORITY_CLASSES from {anger,disgust,fear} to {fear} only to isolate Mixup from disgust noise while keeping fear gains.
todos:
  - id: step1-modify-script
    content: "Edit Transfer Learning.py: MINORITY_CLASSES = frozenset({2}), checkpoint best_rafdb_model_e.pth"
    status: pending
  - id: step2-run-training
    content: Run 50 epochs
    status: pending
  - id: step3-evaluate
    content: evaluate.py → best_rafdb_model_e.pth; train+test + confusion matrix
    status: pending
  - id: step4-log
    content: Append #9 to EXPERIMENT_LOG.md, then stop
    status: pending
isProject: false
---

# Plan E — Fear-only selective Mixup

## Background

Plan D (#8):

- Fear improved across C/D (54.05% → 58.11% → 60.81%); train–test gap shrank ~41pp → ~36pp.
- Disgust regressed in D (65.62% → 52.50%); train disgust dropped to ~89.40% — Mixup over-perturbed disgust.
- **Conclusion**: anger and disgust should not use Mixup; **fear** is the stable beneficiary.

## Changes (minimal — two edits)

1. `MINORITY_CLASSES = frozenset({2})`  # fear = 2 only  
2. `CHECKPOINT_PATH = ... 'best_rafdb_model_e.pth'`

Everything else matches Plan D: label_smoothing=0.1, `weights_tensor`, shuffle=True, CosineAnnealingLR(T_max=50), strong augmentation, MIXUP_ALPHA=0.2.

## Expectations

- Fear: keep improving, target ≥ 60.81%.
- Disgust: recover toward Plan C (~65%+).
- Anger: stable without Mixup.
- happy / neutral / sad / surprised: near Plan A levels.

## Stop

1. `best_rafdb_model_e.pth` trained; full `evaluate.py` run.
2. EXPERIMENT_LOG.md entry **#9** with real numbers.

Do not change `api_server.py`, `CLAUDE.md`, frontend; do not delete old checkpoints.
