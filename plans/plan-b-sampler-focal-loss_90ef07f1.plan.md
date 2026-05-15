---
name: Plan B — Training improvements
overview: On top of Plan A, add WeightedRandomSampler (balanced mini-batches) and Focal Loss (replacing weighted CrossEntropy) to address data imbalance and hard examples, focusing on fear 54.05% and disgust 58.12%.
todos:
  - id: step0-verify-base
    content: Confirm training script matches Plan A (CosineAnnealingLR + expanded augmentation)
    status: pending
  - id: step1-modify-script
    content: "Edit Transfer Learning.py: WeightedRandomSampler, DataLoader with sampler, FocalLoss as criterion, checkpoint best_rafdb_model_3.pth"
    status: pending
  - id: step2-run-training
    content: Run training for 50 epochs; best_rafdb_model_3.pth saved automatically
    status: pending
  - id: step3-evaluate
    content: Point evaluate.py to best_rafdb_model_3.pth; run train+test metrics and confusion matrix
    status: pending
  - id: step4-log
    content: "Append entry #6 to EXPERIMENT_LOG.md with all numbers, then stop"
    status: pending
isProject: false
---

# Plan B — WeightedRandomSampler + Focal Loss

## Background

After Plan A (#5), overall test acc improved **+2.60pp** (82.11% → 84.71%), but **fear dropped 60.81% → 54.05% (−6.76pp)**; train 95.37% vs test 54.05% (~41pp gap). Augmentation alone does not fix fear (281 train images). Root causes:

**Issue 1 — Unbalanced mini-batches: fear is rarely seen**

`shuffle=True` samples by natural class frequency. Fear is ~2.3% of training; with batch_size=32, most batches have **zero** fear even if loss is weighted.

**Issue 2 — Weighted CE cannot fix “invisible” samples**

Weighted CE only rescales loss; it does not increase presence in the batch. Inverse-frequency weights (~6.24 for fear) combined with a sampler need careful tuning to avoid double correction.

**Issue 3 — No explicit emphasis on hard examples**

~22% mutual confusion between fear and disgust (#3). Standard CE treats hard and easy samples similarly.

---

## Design

On top of **all Plan A changes** (do **not** remove stronger aug or CosineAnnealingLR):

1. **WeightedRandomSampler**: roughly balanced classes per batch so fear/disgust appear as often as happy.
2. **Focal Loss (γ=2)**: down-weights easy, high-confidence predictions (e.g. happy), up-weights hard examples.

Sampler fixes “not seen”; Focal Loss fixes “seen but under-weighted”. With the sampler balancing batches, **do not pass class_weights into Focal Loss** to avoid double correction on minorities.

---

## Steps

### Step 0 — Verify starting state

Script must be Plan A version: line ~229 `CosineAnnealingLR`, train transforms include ColorJitter / RandomPerspective / RandomErasing.

---

### Step 1 — Edit [`backend/model_njb/Transfer Learning.py`](backend/model_njb/Transfer%20Learning.py)

**Imports**

```python
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
```

Append `WeightedRandomSampler` to existing `DataLoader, Subset` import.

**Before building `dataloaders`**

```python
# WeightedRandomSampler: balance classes in each batch
# Per-sample weight = 1 / class count (within 90% train split)
_sample_weights = [1.0 / _train_counts[full_train_aug.targets[i]] for i in train_idx]
_sampler = WeightedRandomSampler(
    weights=_sample_weights,
    num_samples=len(_sample_weights),
    replacement=True,
)
```

**DataLoader**

```python
# Before (line ~101)
DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# After (sampler and shuffle are mutually exclusive)
DataLoader(train_dataset, batch_size=batch_size, sampler=_sampler, num_workers=4)
```

**Replace criterion (~line 222)**

```python
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017).
    gamma=2 is standard; focuses on hard (low-confidence) examples.
    No class weights — WeightedRandomSampler already balances batches.
    """
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

criterion = FocalLoss(gamma=2.0)
```

Keep `weights_tensor` / `class_weights` computation for logging only.

**Checkpoint**

```python
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'best_rafdb_model_3.pth')
```

---

### Step 2 — Train

```bash
cd /home/wumenglin/Emotion_detection
python "backend/model_njb/Transfer Learning.py"
```

**Done when**: “training complete” and final test accuracy printed.

---

### Step 3 — Evaluate

In [`backend/model_njb/evaluate.py`](backend/model_njb/evaluate.py):

```python
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_rafdb_model_3.pth')
```

```bash
cd /home/wumenglin/Emotion_detection
python "backend/model_njb/evaluate.py"
```

---

### Step 4 — EXPERIMENT_LOG.md entry **#6**

```markdown
### #6 — Plan B: WeightedRandomSampler + Focal Loss (YYYY-MM-DD)
- **Motivation**: After Plan A, fear test 54.05% (train 95.37%, 41pp gap); rare fear in batches; weighted CE cannot fix visibility. Sampler + Focal, no class_weights in loss.
- **Changes**: `backend/model_njb/Transfer Learning.py` — sampler, FocalLoss(gamma=2), rest as Plan A, best_rafdb_model_3.pth
- **Metrics**: (Plan A model train/test as listed in source plan; new model X% …)
- **Conclusion**: (fear/disgust, train-test gap, happy underfit?, next steps)
```

---

## Stop conditions

1. Script updated as above; `best_rafdb_model_3.pth` trained; `evaluate.py` run with full metrics.
2. EXPERIMENT_LOG.md has #6 with real numbers.

Do not change `api_server.py`, `CLAUDE.md`, frontend; do not delete `best_rafdb_model_1.pth` or `best_rafdb_model_2.pth`.
