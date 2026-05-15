---
name: Plan H — Mixed-dataset training (FER2013 fear boost)
overview: Sample ~1024 fear images from FER2013, mix with RAF-DB training to grow fear training samples from ~254 to ~1278 (~5×), addressing the data bottleneck none of rounds A–G could fix. Baseline comparison is Plan A (best_rafdb_model_2.pth, 84.71% overall). New weights best_rafdb_model_h.pth; evaluation stays on **pure RAF-DB test** only.
todos:
  - id: step0-download-fer2013
    content: Download msambare/fer2013 via Kaggle CLI; verify data/FER2013/train/fear/ (~1024 grayscale 48×48 PNGs)
    status: pending
  - id: step1-modify-script
    content: "Edit Transfer Learning.py: GrayscaleToRGBImageFolder; build_mixed_trainset() ConcatDataset RAF 90% + FER fear; recomputed class weights; best_rafdb_model_h.pth; rest as Plan A (Cosine + strong aug)"
    status: pending
  - id: step2-run-training
    content: 50 epochs; watch fear val trend
    status: pending
  - id: step3-evaluate
    content: evaluate.py → best_rafdb_model_h.pth on RAF-DB test (3068) only
    status: pending
  - id: step4-log
    content: Append #12 to EXPERIMENT_LOG.md
    status: pending
isProject: false
---

# Plan H — FER2013 fear augmentation (mixed training)

## Background

Seven training rounds (Plans A–G) agreed: **fear tops out ~60–63%** because **only ~254 train images**. Regularization, sampling, and loss tricks cannot replace data. This plan adds FER2013 **fear** images.

**Why FER2013 over AffectNet?**

|            | FER2013              | AffectNet        |
|------------|----------------------|------------------|
| Access     | Kaggle, CLI download | Email, ~40 GB    |
| fear count | ~1024 train          | ~25000+          |
| Format     | 48×48 grayscale PNG  | Color, multi-res |
| Risk       | Domain shift         | Hard to obtain   |

FER2013 is easy to fetch; enough fear for a first domain-mix experiment. AffectNet can follow later.

**Domain shift**: FER2013 is grayscale 48×48; RAF-DB is color, variable resolution. Use `convert('RGB')` in the dataset so grayscale is replicated to 3 channels, then the same `Resize(256)→CenterCrop(224)→Normalize` as RAF-DB.

---

## Step 0 — Download FER2013 and verify layout

Requires `~/.kaggle/kaggle.json`.

```bash
cd /home/wumenglin/Emotion_detection
kaggle datasets download -d msambare/fer2013 -p data/FER2013 --unzip
```

Expected tree:

```
data/FER2013/
  train/
    angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  test/
    ...
```

Verify:

```bash
ls data/FER2013/train/fear/ | wc -l   # ~1024
file data/FER2013/train/fear/$(ls data/FER2013/train/fear/ | head -1)
# PNG 48 x 48, 8-bit grayscale
```

---

## Step 1 — Edit `backend/model_njb/Transfer Learning.py`

### Imports

```python
from torch.utils.data import ConcatDataset, Subset
```

### Class `GrayscaleToRGBImageFolder`

Subclass `datasets.ImageFolder`; in `__getitem__`, `img.convert('RGB')` after load so FER2013 matches the color pipeline.

### Function `build_mixed_trainset(raf_train_dir, fer_fear_dir, train_transform)`

- Full RAF `ImageFolder` on `train/`; 90/10 split (seed 42); **val = RAF only** (no FER in val).
- FER fear folder with `GrayscaleToRGBImageFolder` + `target_transform` mapping all to **fear index 2** (alphabetical RAF order anger/disgust/fear/…).
- `ConcatDataset([raf_train_subset, fer_fear])`.
- Recompute `class_counts` including extra fear for `CrossEntropyLoss` weights.

### Section 3 wiring

Set `FER2013_FEAR_DIR` to `.../data/FER2013/train/fear`, call `build_mixed_trainset`, build loaders, print mixed distribution (e.g. fear count ~1278).

### Checkpoint

```python
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'best_rafdb_model_h.pth')
```

### Unchanged vs Plan A

- CosineAnnealingLR(T_max=50, eta_min=1e-6)
- Strong augmentation stack
- 50 epochs, batch 32, Adam, lr 1e-4 (as in your current Plan A script)

---

## Step 2 — Train

```bash
cd /home/wumenglin/Emotion_detection
python "backend/model_njb/Transfer Learning.py"
```

Watch: printed class table (fear ≈ 1278), fear weight drops (~6.24 → ~1.33), smooth early val curve.

---

## Step 3 — Evaluate

`MODEL_PATH` → `best_rafdb_model_h.pth`. **Test set = RAF-DB only (3068)** — no FER2013 in eval for fair comparison to A–G.

```bash
python "backend/model_njb/evaluate.py"
```

---

## Step 4 — EXPERIMENT_LOG.md **#12**

Use the template from the Chinese plan (motivation, code changes, Plan A baseline numbers, new train/test X%, conclusion).

---

## Expectations and risks

| Hope                         | Risk                                      |
|-----------------------------|-------------------------------------------|
| Fear test >65%              | Grayscale vs color domain gap            |
| Overall ≥83%                | Smaller fear weight may slightly hurt disgust |
| Lower fear→surprised errors | RGB replication may add texture noise     |

If it fails: subset FER fear (~300), extra blur aug on FER, or AffectNet later.

---

## Stop

1. FER2013 extracted; `data/FER2013/train/fear/` count ≥ ~900.
2. `best_rafdb_model_h.pth` trained; `evaluate.py` on RAF test complete.
3. EXPERIMENT_LOG.md **#12** with real numbers.

Do not change `api_server.py` until user approves; do not change `CLAUDE.md`/frontend; do not delete existing `best_rafdb_model_*.pth` files.
