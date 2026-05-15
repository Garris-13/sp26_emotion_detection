---
name: Plan D — Training improvements
overview: On Plan C, apply Mixup only to minority classes (anger, disgust, fear); majority classes use clean images and standard CE to keep happy accuracy while retaining boundary smoothing for fear/disgust.
todos:
  - id: step0-verify-base
    content: Confirm Plan C (global Mixup α=0.2, label_smoothing=0.1, shuffle=True)
    status: pending
  - id: step1-modify-script
    content: "Edit Transfer Learning.py: MINORITY_CLASSES config, selective Mixup on images, per-sample selective loss via F.cross_entropy reduction='none', checkpoint best_rafdb_model_d.pth"
    status: pending
  - id: step2-run-training
    content: Run 50 epochs; save best_rafdb_model_d.pth
    status: pending
  - id: step3-evaluate
    content: evaluate.py → best_rafdb_model_d.pth; train+test + confusion matrix
    status: pending
  - id: step4-log
    content: Append #8 to EXPERIMENT_LOG.md
    status: pending
isProject: false
---

# Plan D — Selective Mixup (minority-only)

## Background

Plan C (#7): global Mixup + label smoothing helped disgust/fear but hurt happy:

| Class   | Plan A test | Plan C test | Δ        |
|---------|-------------|-------------|----------|
| disgust | 58.12%      | 65.62%      | +7.50pp  |
| fear    | 54.05%      | 58.11%      | +4.06pp  |
| happy   | 92.24%      | 87.00%      | −5.24pp  |
| overall | 84.71%      | 82.53%      | −2.18pp  |

Global Mixup mixes happy (4310) with others; happy has a clear signal, mixing blurs its region.

**Plan D**: Mixup only when `label ∈ {anger=0, disgust=1, fear=2}` (ImageFolder alphabetical order). Majority classes (happy, neutral, sad, surprised) keep original images and single-label CE.

---

## Design

1. **Selective image mix**: `is_minority` mask; only minority rows get `λ x + (1−λ) x'`.
2. **Per-sample loss**: `F.cross_entropy(..., reduction='none')`; minority: `λ ce_a + (1−λ) ce_b`; majority: `ce_a`.
3. Keep **label_smoothing=0.1** and **class weights** in `F.cross_entropy`.

---

## Steps

### Step 1 — `backend/model_njb/Transfer Learning.py`

**Checkpoint**

```python
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'best_rafdb_model_d.pth')
```

**Config**

```python
MINORITY_CLASSES = frozenset({0, 1, 2})  # anger, disgust, fear
```

**Replace global Mixup block** with:

```python
labels_b = labels
lam = 1.0
is_minority = torch.zeros(inputs.size(0), device=device)
if phase == 'train' and MIXUP_ALPHA > 0:
    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    rand_idx = torch.randperm(inputs.size(0), device=device)
    labels_b = labels[rand_idx]
    is_minority = torch.tensor(
        [1.0 if l.item() in MINORITY_CLASSES else 0.0 for l in labels],
        device=device
    )
    mask = is_minority.view(-1, 1, 1, 1)
    inputs_mixed = lam * inputs + (1 - lam) * inputs[rand_idx]
    inputs = mask * inputs_mixed + (1 - mask) * inputs
```

**Loss**

```python
with torch.set_grad_enabled(phase == 'train'):
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    if phase == 'train' and MIXUP_ALPHA > 0:
        ce_a = F.cross_entropy(outputs, labels, weight=weights_tensor,
                               label_smoothing=0.1, reduction='none')
        ce_b = F.cross_entropy(outputs, labels_b, weight=weights_tensor,
                               label_smoothing=0.1, reduction='none')
        loss_per = is_minority * (lam * ce_a + (1 - lam) * ce_b) + (1 - is_minority) * ce_a
        loss = loss_per.mean()
    else:
        loss = criterion(outputs, labels)

    if phase == 'train':
        loss.backward()
        optimizer.step()
```

---

### Step 2–3

Train; set `MODEL_PATH` to `best_rafdb_model_d.pth`; run `evaluate.py`.

---

### Step 4

Append **#8** to EXPERIMENT_LOG.md.

---

## Stop

1. `best_rafdb_model_d.pth` + full evaluation done.
2. #8 complete.

Do not change `api_server.py`, `CLAUDE.md`, frontend; do not delete old weights.
