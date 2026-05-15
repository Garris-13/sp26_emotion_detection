---
name: Plan C — Training improvements
overview: On Plan A, add Label Smoothing (ε=0.1) and Mixup (α=0.2) to reduce fear/disgust overfitting via loss and synthetic data, avoiding Plan B’s oversampling memorization.
todos:
  - id: step0-verify-base
    content: Confirm script is Plan B state (FocalLoss, WeightedRandomSampler); this plan partially reverts and adds new pieces
    status: pending
  - id: step1-modify-script
    content: "Edit Transfer Learning.py: remove sampler, shuffle=True; CrossEntropyLoss with label_smoothing=0.1 and weights; MIXUP_ALPHA; Mixup in train loop; best_rafdb_model_4.pth"
    status: pending
  - id: step2-run-training
    content: Run 50 epochs; save best_rafdb_model_4.pth
    status: pending
  - id: step3-evaluate
    content: Point evaluate.py to best_rafdb_model_4.pth; train+test metrics and confusion matrix
    status: pending
  - id: step4-log
    content: Append #7 to EXPERIMENT_LOG.md with all metrics, then stop
    status: pending
isProject: false
---

# Plan C — Label Smoothing + Mixup

## Background

Plan B (#6) worsened fear/disgust on test:

| Class   | Plan A test | Plan B test | Δ        |
|---------|-------------|-------------|----------|
| fear    | 54.05%      | 47.30%      | −6.75pp  |
| disgust | 58.12%      | 47.50%      | −10.62pp |
| overall | 84.71%      | 83.70%      | −1.01pp  |

Why Plan B hurt:

- **Oversampling → memorization**: with replacement, fear (~281) is seen ~15× per epoch; model memorizes train, poor generalization.
- **No class_weights in Focal**: majority classes less constrained; minority signal still weak.
- **Data scarcity unchanged**: 281 train / 74 test for fear — still memorization vs learning.

Confusion (Plan B test, 74 fear): fear→surprised 15 (38%), fear→sad 10 (26%) — boundaries overlap; need **softer** boundaries, not stronger resampling.

---

## Design

On Plan A, **restore shuffle=True (no sampler)** and add:

1. **Label smoothing 0.1**: `nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)` — soft targets, less overconfident fits; PyTorch ≥1.10; keep class weights.
2. **Mixup α=0.2**: train-only `mixed_x = λ x_i + (1−λ) x_j`, dual CE on labels — smoother boundaries between fear / surprised / sad.

Smoothing limits peak confidence on train; Mixup smooths class boundaries where confusion is high.

---

## Steps

### Step 0

Current code = Plan B (sampler + Focal). Plan C: remove sampler, remove FocalLoss, use weighted CE + label_smoothing, add Mixup.

---

### Step 1 — `backend/model_njb/Transfer Learning.py`

**Checkpoint**

```python
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'best_rafdb_model_4.pth')
```

**Config**

```python
MIXUP_ALPHA = 0.2  # 0 disables Mixup
```

**Remove** WeightedRandomSampler block; **restore**

```python
'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
```

Keep `_train_counts` / class-weight printing (recompute Counter where needed).

**Train loop** — after `for inputs, labels in dataloaders[phase]:`, before zero_grad:

```python
if phase == 'train' and MIXUP_ALPHA > 0:
    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    rand_idx = torch.randperm(inputs.size(0), device=device)
    inputs_b = inputs[rand_idx]
    labels_b = labels[rand_idx]
    inputs = lam * inputs + (1 - lam) * inputs_b
```

Forward / loss:

```python
with torch.set_grad_enabled(phase == 'train'):
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    if phase == 'train' and MIXUP_ALPHA > 0:
        loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels_b)
    else:
        loss = criterion(outputs, labels)
```

**Criterion**

```python
criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
```

(remove FocalLoss class and `criterion = FocalLoss(...)`)

---

### Step 2 — Train

```bash
cd /home/wumenglin/Emotion_detection
source .venv/bin/activate
python "backend/model_njb/Transfer Learning.py"
```

---

### Step 3 — Evaluate

`MODEL_PATH` → `best_rafdb_model_4.pth`, run `evaluate.py`.

---

### Step 4 — EXPERIMENT_LOG.md **#7**

Fill all measured numbers.

---

## Stop

1. `best_rafdb_model_4.pth` trained; evaluate.py done for train+test + confusion matrix.
2. #7 in EXPERIMENT_LOG.md complete.

Do not change `api_server.py`, `CLAUDE.md`, frontend; do not delete old checkpoints.
