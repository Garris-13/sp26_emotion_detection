# 表情识别系统（ResNet18 + YOLOv8-face + MTCNN）

实时情绪识别项目，基于 RAF-DB 训练的 ResNet18 模型，前端通过摄像头采集画面，后端用 YOLOv8-face 检测人脸 + ResNet18 推理 7 类情绪。

> **实验想法来源：西安交通大学 大学生创新创业训练项目**

> **English documentation:** [`README.md`](./README.md)

> **实验与论文说明（英文）：** [`README_exp.md`](./README_exp.md)

---

## 技术架构

| 组件 | 实现 | 文件 |
|------|------|------|
| 情绪分类 | ResNet18（PyTorch，7 类） | `backend/models/emotion_model.py` + `best_model.pth` |
| 实时人脸检测 | YOLOv8-face（nano） | `backend/api/yolo_face_detector.py` + `backend/yolov8n-face.pt` |
| 单图人脸检测 | facenet MTCNN（回退 OpenCV Haar） | `backend/api/api_server.py` |
| API 服务 | Flask + flask-cors | `backend/api/api_server.py` |
| 训练 | PyTorch + RAF-DB | `backend/model_njb/Transfer Learning.py` |
| 前端 | 原生 HTML/JS（无框架） | `frontend/examples/realtime_emotion.html` |

**情绪标签**（7 类）：`surprised, fear, disgust, happy, sad, anger, neutral`

---

## 目录结构

```
Emotion_detection/
├── backend/
│   ├── api/
│   │   ├── api_server.py         # Flask API（核心推理服务）
│   │   └── yolo_face_detector.py # YOLOv8-face 封装
│   ├── models/
│   │   └── emotion_model.py      # ResNet18 模型定义 + load_model()
│   ├── model_njb/
│   │   └── Transfer Learning.py  # 训练脚本（RAF-DB）
│   ├── ultralytics/              # YOLOv8 本地实现（依赖）
│   ├── yolov8n-face.pt           # YOLO 人脸检测权重
│   └── requirements.txt
├── frontend/
│   └── examples/
│       └── realtime_emotion.html # 实时摄像头识别 UI
├── data/
│   └── RAF-DB/                   # 训练数据（train/ + test/）
├── facenet/                      # MTCNN 本地源码（可选）
├── best_model.pth                # ResNet18 推理权重（42.7 MB）
├── EXPERIMENT_LOG.md             # 实验记录（每次改动追加）
├── README.md                     # 英文说明（默认）
├── README_cn.md                  # 本文件（中文）
└── README_exp.md                 # 实验与论文叙事（英文）
```

---

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+，建议使用 venv 或 conda
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

pip install -r backend/requirements.txt

# （可选）安装 facenet-pytorch 启用 MTCNN 单图检测
# 否则 /predict 会自动回退到 OpenCV Haar
pip install facenet-pytorch
```

### 2. 启动推理（Web 可视化）

需要同时开两个进程：API 服务器 + 静态文件服务器。**两条命令都在项目根目录 `Emotion_detection/` 下执行**（脚本会用 `__file__` 推算权重路径，从根目录跑最稳妥）。

**终端 1 — 启动 API 服务器（端口 7860）：**

```bash
cd /path/to/Emotion_detection
python backend/api/api_server.py
```

启动成功后会打印：
```
✅ 找到模型文件: .../best_model.pth (42.7 MB)
✅ 模型加载成功
✅ YOLOv8-face 初始化成功: .../backend/yolov8n-face.pt
🌐 Running on http://0.0.0.0:7860
```

**烟测一下后端是否就绪**（再开浏览器前先确认）：

```bash
curl -s http://localhost:7860/health
# 期望: {"status":"healthy","model_loaded":true,"yolo_available":true,...}
```

**终端 2 — 启动静态文件服务器（端口 8000）：**

```bash
cd /path/to/Emotion_detection
python -m http.server 8000
```

**浏览器访问：**
```
http://localhost:8000/frontend/examples/realtime_emotion.html
```

允许浏览器使用摄像头权限即可看到实时识别画面（人脸框 + 各情绪概率条 + 推理速度）。前端 `API_BASE` 已硬编码为 `http://localhost:7860`，两个端口务必一致。

### 3. 启动训练

```bash
# 1. 把 backend/model_njb/Transfer Learning.py 第 15 行的 data_dir 改为：
#       data_dir = '/path/to/Emotion_detection/data/RAF-DB'
#    （仓库默认值仍是旧路径 /data/njb/Emotion/RAF-DB，会找不到数据）
# 2. 在项目根目录运行训练（路径里有空格，必须用引号）：
cd /path/to/Emotion_detection
python "backend/model_njb/Transfer Learning.py"
```

**训练参数**（在脚本顶部修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | `/data/njb/Emotion/RAF-DB` | RAF-DB 根目录（含 `train/` 和 `test/`） |
| `batch_size` | 32 | 批大小 |
| `num_epochs` | 50 | 训练轮数 |
| `learning_rate` | 0.001 | SGD 初始学习率 |
| `val_split` | 0.1 | 从 train 切分 10% 做验证 |
| `random_seed` | 42 | 固定数据划分随机种子 |

训练时会自动检测 CUDA。脚本会把验证集最优权重写到当前工作目录下的 `best_rafdb_model_1.pth`（即 `Emotion_detection/best_rafdb_model_1.pth`）。要让推理用上新模型，把它复制覆盖根目录的 `best_model.pth` 即可：

```bash
cp best_rafdb_model_1.pth best_model.pth
```

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | API 信息 + 加载状态 |
| GET | `/health` | 健康检查 |
| GET | `/emotions` | 7 个情绪标签（中英对照） |
| POST | `/predict` | 单张图像情绪识别（MTCNN + ResNet18） |
| POST | `/predict_realtime` | 实时多人脸识别（YOLOv8-face + ResNet18） |

**单图识别示例**（multipart 上传）：
```bash
curl -X POST http://localhost:7860/predict -F "image=@test.jpg"
```

**实时识别示例**（base64 JSON）：
```bash
curl -X POST http://localhost:7860/predict_realtime \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

返回字段：`faces[]` 数组，每个元素含 `bbox / emotion / emotion_zh / confidence / all_probs / yolo_conf`。

---

## 数据集

RAF-DB（Real-world Affective Faces Database），来源 Kaggle 镜像 `shuvoalok/raf-db-dataset`。

```
data/RAF-DB/
├── train/        12271 张（按 7 类子目录组织）
└── test/          3068 张（按 7 类子目录组织）
```

下载方式（Kaggle CLI）：
```bash
kaggle datasets download -d shuvoalok/raf-db-dataset -p data/
unzip -q data/raf-db-dataset.zip -d data/RAF-DB
```

---

## 实验日志

**每次模型改动、训练、调参都会追加到 [`EXPERIMENT_LOG.md`](./EXPERIMENT_LOG.md)。** 模板：

```markdown
### #N — 标题 (YYYY-MM-DD)
- **动机**: 为什么改
- **改动**: 改了什么（文件 + 关键修改）
- **指标**: 训练/推理结果（如适用）
- **结论**: 下一步
```

---

## 常见问题

**Q: 摄像头无法开启？**
浏览器需要 HTTPS 或 `localhost` 才能访问摄像头。访问 `http://localhost:8000/...` 而不是 IP。

**Q: `/predict_realtime` 返回 `yolo_available: false`？**
确认 `backend/yolov8n-face.pt` 存在，且 `backend/ultralytics/` 目录完整。

**Q: 远距离小脸识别概率分布接近均匀（约 1/7）？**
已知问题。YOLO 给出 ~40×40 的 bbox 经 Resize(256) 后高度模糊，ResNet18 失去判别能力。计划改进：在训练 transform 中加入 GaussianBlur / Resize-down-up 增广。详见 `EXPERIMENT_LOG.md`。
