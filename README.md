# Real-Time Facial Expression Recognition (ResNet18 + YOLOv8-face + MTCNN)

Real-time emotion recognition using a ResNet18 classifier trained on RAF-DB. The browser captures webcam frames; the backend runs YOLOv8-face for face detection and ResNet18 for 7-class emotion inference.

> **The key topic is dereived and stretched from the Undergraduate Innovation & Entrepreneurship Training Program of Xi’an Jiaotong University**

> **中文说明：** [`README_cn.md`](./README_cn.md)

> **Experiments & paper alignment (diagnostic study, plans A–J, figures):** [`README_exp.md`](./README_exp.md)

---

## Architecture

| Component | Implementation | Files |
|-----------|----------------|--------|
| Emotion classification | ResNet18 (PyTorch, 7 classes) | `backend/models/emotion_model.py` + `best_model.pth` |
| Real-time face detection | YOLOv8-face (nano) | `backend/api/yolo_face_detector.py` + `backend/yolov8n-face.pt` |
| Single-image face detection | facenet MTCNN (fallback: OpenCV Haar) | `backend/api/api_server.py` |
| API | Flask + flask-cors | `backend/api/api_server.py` |
| Training | PyTorch + RAF-DB | `backend/model_njb/Transfer Learning.py` |
| Frontend | Plain HTML/JS (no framework) | `frontend/examples/realtime_emotion.html` |

**Emotion labels (7):** `surprised, fear, disgust, happy, sad, anger, neutral`

---

## Repository layout

```
Emotion_detection/
├── backend/
│   ├── api/
│   │   ├── api_server.py         # Flask API (core inference)
│   │   └── yolo_face_detector.py # YOLOv8-face wrapper
│   ├── models/
│   │   └── emotion_model.py      # ResNet18 definition + load_model()
│   ├── model_njb/
│   │   └── Transfer Learning.py  # Training script (RAF-DB)
│   ├── ultralytics/              # Local YOLOv8 dependency
│   ├── yolov8n-face.pt         # YOLO face-detection weights
│   └── requirements.txt
├── frontend/
│   └── examples/
│       └── realtime_emotion.html # Live webcam UI
├── data/
│   └── RAF-DB/                   # Dataset (train/ + test/)
├── facenet/                      # Optional local MTCNN source
├── best_model.pth                # ResNet18 inference weights (~42.7 MB)
├── EXPERIMENT_LOG.md             # Experiment log (append on changes)
├── README.md                     # This file (English, default)
├── README_cn.md                  # Chinese README
└── README_exp.md                 # Experiments / paper narrative (English)
```

---

## Quick start

### 1. Environment

```bash
# Python 3.10+; venv or conda recommended
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r backend/requirements.txt

# Optional: facenet-pytorch for MTCNN on /predict
# Without it, /predict falls back to OpenCV Haar cascades
pip install facenet-pytorch
```

### 2. Run inference (web demo)

You need **two terminals**: API server + static file server. **Run both from the project root** `Emotion_detection/` (paths to weights are resolved relative to the repo; running from the root is the most reliable).

**Terminal 1 — API server (port 7860):**

```bash
cd /path/to/Emotion_detection
python backend/api/api_server.py
```

On success you should see something like:

```
✅ 找到模型文件: .../best_model.pth (42.7 MB)
✅ 模型加载成功
✅ YOLOv8-face 初始化成功: .../backend/yolov8n-face.pt
🌐 Running on http://0.0.0.0:7860
```

**Smoke-test the backend** before opening the browser:

```bash
curl -s http://localhost:7860/health
# Expect: {"status":"healthy","model_loaded":true,"yolo_available":true,...}
```

**Terminal 2 — static HTTP server (port 8000):**

```bash
cd /path/to/Emotion_detection
python -m http.server 8000
```

**Open in the browser:**

```
http://localhost:8000/frontend/examples/realtime_emotion.html
```

Grant webcam access to see live boxes, per-emotion bars, and inference timing. The frontend `API_BASE` is hard-coded to `http://localhost:7860`; keep the two ports aligned.

### 3. Training

```bash
# 1. In backend/model_njb/Transfer Learning.py line ~15, set data_dir to your RAF-DB root, e.g.:
#       data_dir = '/path/to/Emotion_detection/data/RAF-DB'
#    (The repo default may still point to /data/njb/Emotion/RAF-DB and will not find data.)
# 2. From the project root (quote the path because of the space in the filename):
cd /path/to/Emotion_detection
python "backend/model_njb/Transfer Learning.py"
```

**Training hyperparameters** (edit at the top of the script):

| Parameter | Default | Notes |
|-----------|---------|--------|
| `data_dir` | `/data/njb/Emotion/RAF-DB` | RAF-DB root with `train/` and `test/` |
| `batch_size` | 32 | Batch size |
| `num_epochs` | 50 | Epochs |
| `learning_rate` | 0.001 | Initial SGD learning rate |
| `val_split` | 0.1 | 10% of train for validation |
| `random_seed` | 42 | Seed for the train/val split |

CUDA is used automatically when available. The best checkpoint on the validation split is written to `best_rafdb_model_1.pth` in the **current working directory** (typically `Emotion_detection/best_rafdb_model_1.pth`). To use it for inference, copy over the root `best_model.pth`:

```bash
cp best_rafdb_model_1.pth best_model.pth
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info + load status |
| GET | `/health` | Health check |
| GET | `/emotions` | Seven labels (EN + ZH) |
| POST | `/predict` | Single image (MTCNN + ResNet18) |
| POST | `/predict_realtime` | Real-time multi-face (YOLOv8-face + ResNet18) |

**Single-image example** (multipart upload):

```bash
curl -X POST http://localhost:7860/predict -F "image=@test.jpg"
```

**Real-time example** (base64 JSON):

```bash
curl -X POST http://localhost:7860/predict_realtime \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

Response: `faces[]`; each item includes `bbox`, `emotion`, `emotion_zh`, `confidence`, `all_probs`, `yolo_conf`.

---

## Dataset

RAF-DB (Real-world Affective Faces Database), Kaggle mirror `shuvoalok/raf-db-dataset`.

```
data/RAF-DB/
├── train/        12,271 images (7 class subfolders)
└── test/          3,068 images (7 class subfolders)
```

Download with Kaggle CLI:

```bash
kaggle datasets download -d shuvoalok/raf-db-dataset -p data/
unzip -q data/raf-db-dataset.zip -d data/RAF-DB
```

---

## Experiment log

Model, training, and inference changes are appended to [`EXPERIMENT_LOG.md`](./EXPERIMENT_LOG.md). Entry template:

```markdown
### #N — Title (YYYY-MM-DD)
- **Motivation**: why the change
- **Changes**: what changed (files + key edits)
- **Metrics**: train/infer results if applicable
- **Conclusion**: next steps
```

---

## FAQ

**Q: Webcam does not start?**  
Browsers require HTTPS or `localhost` for camera access. Use `http://localhost:8000/...`, not a raw LAN IP.

**Q: `/predict_realtime` shows `yolo_available: false`?**  
Ensure `backend/yolov8n-face.pt` exists and `backend/ultralytics/` is intact.

**Q: Distant small faces yield nearly uniform probabilities (~1/7)?**  
Known limitation: ~40×40 YOLO boxes upscaled through Resize(256) look blurry to ResNet18. Planned mitigation: GaussianBlur / down–up resize in training augmentations. See `EXPERIMENT_LOG.md`.
