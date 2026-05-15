---
name: Plan I — Linear probe comparison
overview: Freeze full ResNet18 backbone; train only the final FC (linear probe) on RAF-DB as a controlled comparison to full fine-tuning (Plan A). Quantifies how far **ImageNet features alone** linearly separate the 7 RAF emotions; diagnoses whether fear/disgust limits are representation vs optimization. Checkpoint best_rafdb_model_lp.pth.
todos:
  - id: step1-modify-script
    content: "Edit Transfer Learning.py: freeze backbone; SGD on fc only (lr=0.1, momentum=0.9, wd=1e-4); CosineAnnealingLR(T_max=50, eta_min=1e-3); same aug as Plan A; best_rafdb_model_lp.pth"
    status: pending
  - id: step2-run-training
    content: 50 epochs (very fast); log convergence
    status: pending
  - id: step3-evaluate
    content: evaluate.py → best_rafdb_model_lp.pth; train+test + confusion matrix
    status: pending
  - id: step4-log
    content: Append #13 to EXPERIMENT_LOG.md
    status: pending
isProject: false
---

# Plan I — Linear probe (Linear Probe) experiment

## Background

Plans A–G all used **full fine-tuning**. Unanswered question:

> If the backbone stays frozen and **only a linear head** is trained, how well do ImageNet features separate RAF-DB’s 7 expressions?

Standard in representation learning; here it helps:

1. **Course/report angle**: classic “linear separability of pretrained features” narrative.
2. **Diagnose fear**:
   - If linear-probe fear ≈ full fine-tune (~54–60%), ImageNet features may already be near a **linear ceiling** for fear → data / representation limit.
   - If linear probe fear is **much lower**, fear needs **backbone adaptation**; bottleneck is weak gradient in full FT, not just the head.
3. **Cost**: only 512×7 FC params (~3584); epochs are minutes vs tens of minutes for full FT.

---

## Design

|              | Linear probe (I)     | Full FT (Plan A baseline) |
|--------------|----------------------|---------------------------|
| Frozen       | All backbone + pool  | None                      |
| Trainable    | `model.fc` only      | All (~11M)                |
| Optimizer    | SGD(fc, lr=0.1, mom=0.9, wd=1e-4) | Adam(all, 1e-4) |
| Scheduler    | Cosine T_max=50, eta_min=1e-3 | Cosine, eta_min=1e-6 |
| Loss / aug / epochs | Same weighted CE + Plan A aug, 50 | Same |
| Checkpoint   | best_rafdb_model_lp.pth | best_rafdb_model_2.pth |

**Why higher LR (0.1) for linear probe?** Backbone is fixed; FC starts random — needs larger step for convex-ish problem. Common in MoCo/SimCLR-style probes.

---

## Step 1 — `backend/model_njb/Transfer Learning.py`

**After** `model_ft = models.resnet18(pretrained=True)`, **before** replacing `model_ft.fc`:

```python
# Linear probe: freeze backbone
for param in model_ft.parameters():
    param.requires_grad = False
# New fc layer remains trainable by default
```

**Optimizer** — replace Adam on all params:

```python
optimizer_ft = optim.SGD(
    model_ft.fc.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)
```

**Scheduler**

```python
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer_ft, T_max=num_epochs, eta_min=1e-3
)
```

**Checkpoint**

```python
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'best_rafdb_model_lp.pth')
```

Keep augmentation, class weights, val split identical to Plan A.

---

## Step 2 — Train

```bash
cd /home/wumenglin/Emotion_detection
python "backend/model_njb/Transfer Learning.py"
```

Expect fast epochs; val acc rises quickly early.

---

## Step 3 — Evaluate

Point `evaluate.py` to `best_rafdb_model_lp.pth`, run full train+test + confusion matrix. Compare fear/disgust vs Plan A (84.71% overall baseline).

---

## Step 4 — EXPERIMENT_LOG.md **#13**

Use template from source (motivation, edits, Plan A test numbers, LP train/test X%, interpretation).

---

## Interpretation cheat sheet

| LP fear outcome   | Interpretation                         | Next step suggestion        |
|-------------------|----------------------------------------|------------------------------|
| ~50–55% (near A)  | Linear feature ceiling near current FT | Data / Plan H               |
| <40%              | Weak linear separability; backbone must move | Full FT justified      |
| >60% (beats A)    | Strong frozen features; FT may “damage” them | Try milder FT (e.g. 1e-5) |

---

## Stop

1. `best_rafdb_model_lp.pth` + full evaluation.
2. EXPERIMENT_LOG.md **#13** complete.

Do not switch inference in `api_server.py` (analysis only); do not change `CLAUDE.md`/frontend; do not delete other checkpoints.
