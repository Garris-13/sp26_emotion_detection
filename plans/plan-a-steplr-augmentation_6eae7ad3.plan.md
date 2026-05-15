---
name: Plan A — Training improvements
overview: Fix two core issues in the current training script—overly aggressive StepLR decay and insufficient augmentation—measure training-set accuracy of the current checkpoint as a baseline, then retrain, evaluate, and record results in EXPERIMENT_LOG.md.
todos:
  - id: step0-baseline-train-acc
    content: Extend evaluate.py with training-set evaluation; run and record per-class train accuracy for best_rafdb_model_1.pth
    status: pending
  - id: step1-fix-training-script
    content: "Edit Transfer Learning.py: StepLR → CosineAnnealingLR, expand train transforms, save weights as best_rafdb_model_2.pth"
    status: pending
  - id: step2-run-training
    content: Run the training script for 50 epochs; best_rafdb_model_2.pth is saved automatically
    status: pending
  - id: step3-evaluate-new-model
    content: Point evaluate.py MODEL_PATH to best_rafdb_model_2.pth; run to get train+test per-class accuracy
    status: pending
  - id: step4-log
    content: "Append entry #5 to EXPERIMENT_LOG.md with all measured numbers, then stop"
    status: pending
isProject: false
---

# Plan A — Fix StepLR + Stronger Augmentation

## Background and problem statement

The current weights `backend/model_njb/best_rafdb_model_1.pth` reach **82.11%** overall on the test set, but **disgust 53.75%** and **fear 60.81%** are very low. Two major issues in [`backend/model_njb/Transfer Learning.py`](backend/model_njb/Transfer%20Learning.py):

**Issue 1: StepLR decays too aggressively; the last ~30 epochs barely learn**

```python
# Line 229 (current)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

- After epoch 7, lr = 1e-4; after 14, lr = 1e-5; after 21, lr = **1e-6**
- For the remaining ~29 epochs the learning rate is effectively zero; **fear** (253 training images) gets almost no useful gradient updates.

**Issue 2: Training augmentation is too weak; minority classes see few effective variations**

```python
# Lines 52–59 (current)
transforms.Resize((256, 256)),
transforms.RandomRotation(15),
transforms.RandomHorizontalFlip(),
transforms.CenterCrop((224, 224)),
```

Only rotation and flip. With **253 fear** samples and no color, blur, or occlusion augmentations, over 50 epochs the model sees very limited diversity and tends to over- or under-fit on that class.

**Missing baseline: training-set accuracy**

The experiment log only has test accuracy. Without **train accuracy**, we cannot tell underfitting (train also low) from overfitting (train high, test low). **Always log both train and test accuracy** for minority-class analysis.

---

## Execution steps

### Step 0 — Measure training-set accuracy for the current checkpoint (baseline)

**Goal**: Before any change, evaluate `best_rafdb_model_1.pth` on the **full training set (12,271 images)** with the same preprocessing as test (no augmentation): Resize(256) → CenterCrop(224) → ToTensor → Normalize.

**Action**: Extend [`backend/model_njb/evaluate.py`](backend/model_njb/evaluate.py) after the existing test logic with the same pipeline for `data/RAF-DB/train/`.

Then run:

```bash
cd /home/wumenglin/Emotion_detection
python "backend/model_njb/evaluate.py"
```

Record per-class train accuracy, compare with known test results, and **add this block to EXPERIMENT_LOG.md under entry #4** as a “baseline training-set accuracy” subsection.

---

### Step 1 — Edit the training script

**File**: [`backend/model_njb/Transfer Learning.py`](backend/model_njb/Transfer%20Learning.py)

**Change 1 — Replace the LR scheduler (line 229)**

```python
# Remove
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Replace with
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=num_epochs, eta_min=1e-6)
```

`CosineAnnealingLR(T_max=50)` keeps a smooth, useful LR over all 50 epochs instead of collapsing near zero after epoch 21.

**Change 2 — Expand training transforms (lines 52–59)**

```python
# Replace the 'train' transforms.Compose with:
'train': transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
]),
```

- **ColorJitter**: lighting variation, less reliance on fixed brightness/contrast
- **RandomGrayscale**: low-quality camera simulation; more color-invariant emotion cues
- **RandomPerspective**: viewpoint change, less overfit to frontal faces
- **RandomErasing**: partial occlusion (mask, hand, etc.)

**Note**: `RandomErasing` must come **after** `ToTensor()`. Do **not** change val/test transforms.

**Change 3 — Save new weights as `best_rafdb_model_2.pth` (line 31)**

```python
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'best_rafdb_model_2.pth')
```

Avoids overwriting `best_rafdb_model_1.pth` for comparison.

---

### Step 2 — Run training

```bash
cd /home/wumenglin/Emotion_detection
python "backend/model_njb/Transfer Learning.py"
```

The script prints train/valid loss and acc each epoch, saves the best val checkpoint to `backend/model_njb/best_rafdb_model_2.pth`, and evaluates on the test set at the end.

**Done when**: output shows training finished and **final test-set accuracy** is printed.

---

### Step 3 — Evaluate the new model (train + test)

Set `MODEL_PATH` in `evaluate.py` to the new checkpoint:

```python
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_rafdb_model_2.pth')
```

Run (assuming train-set evaluation was added in Step 0):

```bash
python "backend/model_njb/evaluate.py"
```

Save full output: per-class train + test accuracy and confusion matrix.

---

### Step 4 — Log in EXPERIMENT_LOG.md

Append entry **#5** (increment N and use today’s date):

```markdown
### #5 — Plan A: StepLR fix + stronger augmentation retrain (YYYY-MM-DD)
- **Motivation**: StepLR(step_size=7,gamma=0.1) makes lr≈1e-6 after epoch 21 (little learning); augmentations were only rotation+flip, under-augmenting fear/disgust
- **Changes**: `backend/model_njb/Transfer Learning.py`
  - Scheduler: StepLR → CosineAnnealingLR(T_max=50, eta_min=1e-6)
  - Train transforms: ColorJitter / RandomGrayscale / RandomPerspective / RandomErasing
  - New checkpoint: best_rafdb_model_2.pth
- **Metrics**:
  - Old model (best_rafdb_model_1.pth) train: overall X% | anger X% / disgust X% / fear X% / happy X% / neutral X% / sad X% / surprised X%
  - Old model test: (already in #4) overall 82.11% | ...
  - New model (best_rafdb_model_2.pth) train: ...
  - New model test: ...
- **Conclusion**: (fill from results: minority classes, over/underfitting, next steps)
```

---

## Stop conditions (explicit for the agent)

**Stop immediately after all three are true; do not make extra changes:**

1. `evaluate.py` has successfully evaluated `best_rafdb_model_1.pth` on the train set with per-class accuracy printed.
2. Training script is updated, `best_rafdb_model_2.pth` is trained, and `evaluate.py` has been run for train+test on the new weights.
3. EXPERIMENT_LOG.md has entry #5 with all `X%` replaced by real numbers.

Do **not** change `api_server.py` (wait for user before switching inference weights), `CLAUDE.md`, or any frontend files.
