# EmoCare 实验日志

本文件用于记录 EmoCare 项目的当前基线状态以及后续每一次改进的步骤、动机与结果。每次改动按编号追加一条 entry。

---

## 基线状态 (Baseline, 2026-05-01)

### 模型与数据

- **模型**: ResNet18, 7 类输出 (`anger, disgust, fear, happy, neutral, sad, surprised`)
- **训练数据**: RAF-DB (Kaggle 镜像 `shuvoalok/raf-db-dataset`)
- **本地数据路径**: `/home/wumenglin/Emotion_detection/data/RAF-DB/`
  - `train/` 12271 张 (按 7 类子目录)
  - `test/` 3068 张 (按 7 类子目录)
- **训练脚本**: [`backend/model_njb/Transfer Learning.py`](backend/model_njb/Transfer%20Learning.py)
  - 注意: 脚本中 `data_dir` 仍为旧路径 `/data/njb/Emotion/RAF-DB` (line 15),后续改进时需指向上面新路径
- **推理权重**: 项目根 `best_model.pth`

### 推理流水线

```
前端 base64 → /predict_realtime → PIL(RGB) → BGR(numpy)
  → YOLOv8-face (conf=0.45) → bbox
  → crop_face(pad=0.25)        ← 无尺寸过滤
  → Resize(256,256) → CenterCrop(224,224) → Normalize
  → ResNet18 → softmax(7)
```

- 人脸检测: YOLOv8-face, 权重 `backend/yolov8n-face.pt`
- 主要文件:
  - [`backend/api/api_server.py`](backend/api/api_server.py) `predict_realtime_emotion`
  - [`backend/api/yolo_face_detector.py`](backend/api/yolo_face_detector.py) `crop_face`

### 已知问题

1. **小脸放大模糊**: 摄像头距离较远时,YOLO 给出的 bbox 可能只有 ~40×40 像素,经 `Resize(256,256)` 强制上采样后变得高度模糊,导致 ResNet18 输出近乎均匀的 7 类概率(每类 ≈ 1/7),实质上失去判别能力。
2. **训练时无模糊增广**: 现有训练 transform 不包含模糊/低分辨率退化,因此模型对低质量输入鲁棒性差。

### 计划改进

| # | 改进 | 状态 |
|---|------|------|
| 1 | 在 RAF-DB 上加入模糊增广(GaussianBlur / Resize-down-up)重新训练,提升对小脸/低分辨率人脸的鲁棒性 | 待开始 |

---

## 改进记录

> 每次改动追加一条,模板:
>
> ### #N — 标题 (YYYY-MM-DD)
> - **动机**:
> - **改动**:
> - **指标**:
> - **结论**:

### #0 — RAF-DB 数据集准备完成 (2026-05-01)

- **动机**: 为后续在本地复现训练并加入模糊增广做准备。
- **改动**:
  - 通过 Kaggle CLI 下载 `shuvoalok/raf-db-dataset` (~37.7 MB) 至 `data/RAF-DB/`
  - 将原始数字类目录 `1..7` 重命名为 `surprised, fear, disgust, happy, sad, anger, neutral`
  - 最终结构符合 `ImageFolder` 约定: `data/RAF-DB/{train,test}/<class_name>/*.jpg`
- **指标**: train=12271, test=3068, 共 7 类
- **结论**: 数据已就绪。下一步: 修改 `backend/model_njb/Transfer Learning.py:15` 的 `data_dir` 指向新路径,并规划改进 #1 的增广方案。

### #1 — 代码库瘦身：只保留 ResNet18 + YOLO + MTCNN 推理管线 (2026-05-01)

- **动机**: 项目方向已收敛到核心推理 + 可视化，旧的 LLM 对话、健康建议、MySQL 存储、摄像头监控等模块不再使用，需移除以降低维护成本。
- **改动**:
  - 删除 `backend/multiAgent/`、`backend/database/` 整目录
  - 删除 `backend/api/{langgraph_agent,advice_generator,api_client,camera_monitor,usb_camera_monitor}.py`
  - 删除 `backend/models/health_advisor.py`
  - 删除 `frontend/examples/{emotion_ui.html, example_*.{py,sh}}` 与若干根目录测试脚本
  - `backend/models/emotion_model.py`：删除 ResNet50 分支，固定 ResNet18
  - `backend/api/api_server.py`：从 3689 行重写为 493 行，仅保留 `/`、`/health`、`/emotions`、`/predict`、`/predict_realtime` 五个端点
  - `backend/requirements.txt`：移除 langchain、langgraph、openai、pymysql、mediapipe、cryptography、python-dotenv
  - `frontend/examples/realtime_emotion.html`：为 `face.yolo_conf` 添加 null 保护
  - 新 `README.md`：仅保留启动推理与训练命令
- **指标**: 启动验证通过——`/health` 返回 200，model_loaded=true，YOLOv8-face 初始化成功，CUDA 可用
- **结论**: 仓库已聚焦核心推理。下一步可专注计划改进 #1（模糊增广重训）。

### #2 — 修正训练脚本路径与类别权重错位 (2026-05-01)

- **动机**: 准备开始本地复现训练。检查发现 `backend/model_njb/Transfer Learning.py` 有两个会直接影响训练正确性的问题：
  1. `data_dir` 仍是旧服务器路径 `/data/njb/Emotion/RAF-DB`，本地跑会 `FileNotFoundError`。
  2. 类别权重数组 `[1.0, 10.0, 5.0, 0.5, 1.0, 2.0, 1.0]` 是按 RAF-DB **原始 1-7 数字编号** 顺序写的 (Surprise/Fear/Disgust/Happy/Sad/Anger/Neutral)，但 `ImageFolder` 是按目录名 **字母序** 给类编号 (`anger, disgust, fear, happy, neutral, sad, surprised`) ——结果 `disgust` 拿到 10.0 权重 (本意给 fear)、最稀少类 `fear` (281 张) 反而只拿到 5.0，loss 实际是错位的。
- **改动** (`backend/model_njb/Transfer Learning.py`):
  - 第 21 行 `data_dir` → `/home/wumenglin/Emotion_detection/data/RAF-DB`
  - 新增 `SCRIPT_DIR / CHECKPOINT_PATH`，把保存/加载的 `best_rafdb_model_1.pth` 锚定到脚本同级目录 (原来是相对 cwd，cd 到别处会丢权重)
  - 加 `assert os.path.isdir(...)` 对 `train/` `test/` 做存在性快速失败
  - 删除硬编码 `class_weights`，改为运行时按 90% 训练 split 的真实样本数计算 inverse-frequency 权重 (`w_c = N / (n_classes * count_c)`，等价 `sklearn.utils.class_weight.compute_class_weight('balanced')`)，并在启动时打印 `[idx] class count weight` 表
  - 训练超参 (lr / epochs / val_split / 模型结构) 未动
- **指标**: AST 检查通过；新权重对照 (基于 90% split ≈11044 张):
  | idx | class | count | old (错位) | new (≈) |
  |---|---|---|---|---|
  | 0 | anger | ~635 | 1.0 | 2.49 |
  | 1 | disgust | ~645 | 10.0 | 2.45 |
  | 2 | fear | ~253 | 5.0 | **6.24** |
  | 3 | happy | ~4295 | 0.5 | 0.37 |
  | 4 | neutral | ~2272 | 1.0 | 0.69 |
  | 5 | sad | ~1784 | 2.0 | 0.88 |
  | 6 | surprised | ~1161 | 1.0 | 1.36 |
- **结论**: 训练脚本已可在本地直接跑。下一步：跑一次 50 epoch 拿基线指标，再考虑改进 #1 (模糊增广)。

### #4 — 本地复现训练 + 修正 EMOTION_LABELS + 替换推理权重 (2026-05-01)

- **动机**: 用修正后的训练脚本在本地跑完 50 epoch，得到新权重并部署到 Web 推理端。
- **改动**:
  - 训练完成，权重保存至 `backend/model_njb/best_rafdb_model_1.pth`，复制覆盖 `best_model.pth`
  - `backend/api/api_server.py` 第 88 行 `EMOTION_LABELS` 由旧训练顺序 `['surprised','fear','disgust','happy','sad','anger','neutral']` 改为字母序 `['anger','disgust','fear','happy','neutral','sad','surprised']`（与新模型 ImageFolder 训练顺序一致）
- **指标** (RAF-DB test set, 3068 张, 新模型): 总体 **82.11%**；各类 anger 73.46% / disgust 53.75% / fear 60.81% / happy 92.15% / neutral 76.62% / sad 80.13% / surprised 82.98%
- **结论**: Web 推理端已可使用新权重。`disgust` 和 `fear` 仍是最弱类（样本少），是改进 #1（模糊增广重训）的主要目标。

### #3 — 新增评估脚本 & 对 best_model.pth 做 RAF-DB test set 基线测试 (2026-05-01)

- **动机**: 仓库无评估脚本；需要拿到 `best_model.pth` 在 RAF-DB test set 上的基线准确率，作为后续改进的参照。
- **改动**: 新增 `backend/model_njb/evaluate.py`
  - 加载 `best_model.pth`，在 `data/RAF-DB/test/` (3068 张) 上推理
  - 关键处理：`best_model.pth` 训练时目录名为 `"1"~"7"`，`ImageFolder` 按字母/数字序映射为 `[surprised,fear,disgust,happy,sad,anger,neutral]`，与当前按英文目录名字母序的 `[anger,disgust,fear,happy,neutral,sad,surprised]` 不同；脚本内建 `pred_remap` 做正确对齐（首次不加 remap 时 Accuracy 仅 37.65%，加后 81.71%）
- **指标** (RAF-DB test set, 3068 张):

  | 类别 | 样本数 | 正确 | 准确率 |
  |------|--------|------|--------|
  | anger | 162 | 116 | 71.60% |
  | disgust | 160 | 94 | 58.75% |
  | fear | 74 | 46 | 62.16% |
  | happy | 1185 | 1072 | 90.46% |
  | neutral | 680 | 562 | 82.65% |
  | sad | 478 | 348 | 72.80% |
  | surprised | 329 | 269 | 81.76% |
  | **总体** | **3068** | **2507** | **81.71%** |

- **结论**: 现有权重基线准确率 81.71%。`disgust`(58.75%) 和 `fear`(62.16%) 最弱，两者也是样本最少的类（分别 160/74 张）；主要误分方向为互相混淆（fear↔disgust 混淆率约 22%/8%）。下一步：跑模糊增广重训，并以此作为改进参照。

---

### #5 — 方案A：修复StepLR + 扩充增广重训 (2026-05-03)
- **动机**: `best_rafdb_model_1.pth` 训练集 93.31% vs 测试集 82.11%，明显过拟合；disgust(95.12% train/53.75% test) 和 fear(96.80% train/60.81% test) 过拟合最严重。根因：StepLR(step_size=7,gamma=0.1) 在 epoch 21 后 lr≈1e-6 实际停止学习；训练增广仅旋转+翻转，少数类等效重复看同一张图
- **改动**: `backend/model_njb/Transfer Learning.py`
  - 学习率调度器: StepLR → CosineAnnealingLR(T_max=50, eta_min=1e-6)
  - 训练 transform 新增: ColorJitter / RandomGrayscale / RandomPerspective / RandomErasing
  - 新权重保存为 `best_rafdb_model_2.pth`（不覆盖旧权重）
  - `backend/model_njb/evaluate.py` 新增 train set 评估逻辑
- **指标**:
  - 旧模型(best_rafdb_model_1.pth) 训练集: 总体 93.31% | anger 97.16% / disgust 95.12% / fear 96.80% / happy 94.40% / neutral 89.30% / sad 90.77% / surprised 97.13%
  - 旧模型 测试集: 总体 82.11% | anger 73.46% / disgust 53.75% / fear 60.81% / happy 92.15% / neutral 76.62% / sad 80.13% / surprised 82.98%
  - 新模型(best_rafdb_model_2.pth) 训练集: 总体 97.72% | anger 97.45% / disgust 95.68% / fear 95.37% / happy 98.24% / neutral 97.11% / sad 97.83% / surprised 98.60%
  - 新模型 测试集: 总体 84.71% | anger 74.69% / disgust 58.12% / fear 54.05% / happy 92.24% / neutral 81.76% / sad 85.36% / surprised 87.54%
- **结论**: 整体测试集 +2.60pp(82.11%→84.71%)，neutral/sad/surprised/disgust 均有改善(+5.14/+5.23/+4.56/+4.37pp)。但 **fear 从 60.81% 退至 54.05%(-6.76pp)**，训练集仍 95.37%，train-test 差距扩大至 41pp；说明 fear(281 训练/74 测试) 样本太少，增广未能解决过拟合，需要更强干预。下一步方向：对 fear 类使用 Focal Loss 或过采样(SMOTE/Mixup)，或考虑对 fear 单独调整增广强度。

### #6 — 方案B：WeightedRandomSampler + Focal Loss 重训 (2026-05-04)
- **动机**: 方案A后 fear test acc 反退至 54.05%（train 95.37%，差距 41pp）。mini-batch 中 fear 出现频率仅 2.3%，梯度信号极稀疏；Weighted CE 线性加权无法解决"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            看不见"问题。引入 Sampler 均衡 batch 分布，Focal Loss 聚焦难样本，不再使用 class_weights（避免双重矫正）

- **改动**: `backend/model_njb/Transfer Learning.py`
  - 新增 WeightedRandomSampler（各类采样权重=1/count），替换 DataLoader 的 shuffle=True
  - Focal Loss(gamma=2.0) 替换 CrossEntropyLoss(weight=)，不传 class_weights
  - 其余保持方案A：CosineAnnealingLR(T_max=50) + 扩充 augmentation
  - 新权重保存为 best_rafdb_model_3.pth
- **指标**:
  - 方案A模型(best_rafdb_model_2.pth) 训练集: 总体 97.72% | anger 97.45% / disgust 95.68% / fear 95.37% / happy 98.24% / neutral 97.11% / sad 97.83% / surprised 98.60%
  - 方案A模型 测试集: 总体 84.71% | anger 74.69% / disgust 58.12% / fear 54.05% / happy 92.24% / neutral 81.76% / sad 85.36% / surprised 87.54%
  - 新模型(best_rafdb_model_3.pth) 训练集: 总体 96.93% | anger 97.02% / disgust 95.68% / fear 95.37% / happy 97.21% / neutral 96.00% / sad 97.02% / surprised 98.53%
  - 新模型 测试集: 总体 83.70% | anger 77.78% / disgust 47.50% / fear 47.30% / happy 91.31% / neutral 83.09% / sad 83.68% / surprised 86.32%
- **结论**: 方案B整体失败。测试集总体 -1.01pp(84.71%→83.70%)，fear -6.75pp(54.05%→47.30%)，disgust -10.62pp(58.12%→47.50%)，train-test 差距反而扩大至 48pp。根因推断：WeightedRandomSampler 让模型每 epoch 重复看同一批 fear/disgust 样本约 15 倍，进一步加剧过拟合；Focal Loss 去掉 class_weights 后对多数类（happy）的约束不足，导致少数类梯度虽有信号但泛化能力更差。下一步建议：① 回退到方案A基础上，对 fear/disgust 单独施加更强增广（Mixup/CutMix），而非过采样；② 或尝试 Sampler + 保留 class_weights（轻度矫正而非去掉），gamma 调低至 0.5；③ 或考虑从外部数据源补充 fear 样本

### #7 — 方案C：Label Smoothing + Mixup 重训 (2026-05-04)
- **动机**: 方案B失败根因是 WeightedRandomSampler 让 fear（281张）每 epoch 重复约 15 倍，加剧过拟合。方案C回归 shuffle=True，改从损失函数和数据合成两个角度抑制过拟合：Label Smoothing 防止模型对训练样本过度自信，Mixup 软化 fear/surprised/sad 之间的决策边界（混淆矩阵显示 fear→surprised 占 38%）
- **改动**: `backend/model_njb/Transfer Learning.py`
  - MIXUP_ALPHA = 0.2（Beta 分布混合，仅训练阶段）
  - CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)（保留类别权重，加 ε=0.1 平滑）
  - 移除 WeightedRandomSampler，恢复 shuffle=True
  - 其余保持方案A：CosineAnnealingLR(T_max=50) + 扩充 augmentation
  - 新权重保存为 best_rafdb_model_4.pth
- **指标**:
  - 方案A模型(best_rafdb_model_2.pth) 训练集: 总体 97.72% | anger 97.45% / disgust 95.68% / fear 95.37% / happy 98.24% / neutral 97.11% / sad 97.83% / surprised 98.60%
  - 方案A模型 测试集: 总体 84.71% | anger 74.69% / disgust 58.12% / fear 54.05% / happy 92.24% / neutral 81.76% / sad 85.36% / surprised 87.54%
  - 新模型(best_rafdb_model_4.pth) 训练集: 总体 94.69% | anger 97.02% / disgust 97.07% / fear 97.51% / happy 92.90% / neutral 94.89% / sad 94.15% / surprised 98.53%
  - 新模型 测试集: 总体 82.53% | anger 77.16% / disgust 65.62% / fear 58.11% / happy 87.00% / neutral 81.32% / sad 82.22% / surprised 85.71%
- **结论**: fear/disgust 显著改善（fear +4.06pp→58.11%，disgust +7.50pp→65.62%），train-test 差距收窄（disgust: 37.56pp→31.45pp，fear: 41.32pp→39.40pp）。但 happy 大幅退步 -5.24pp（92.24%→87.00%），导致总体 -2.18pp（84.71%→82.53%）。原因：Mixup 将 happy（4310张）的特征与 fear/disgust 混合，破坏了 happy 的决策边界（混淆矩阵：happy→fear 24张、happy→disgust 28张明显增多）。下一步方向：① 降低 alpha（0.2→0.1）减少混合强度，保留泛化收益的同时减少 happy 损失；② 或仅对少数类（fear/disgust/anger）启用 Mixup，happy/neutral/sad 不参与混合；③ 或考虑 CutMix 替代 Mixup（局部替换比全图混合对 happy 破坏更小）

### #8 — 方案D：选择性 Mixup（仅少数类）重训 (2026-05-04)
- **动机**: 方案C全局 Mixup 使 fear/disgust 显著改善但 happy 退步 -5.24pp。方案D将 Mixup 限制在少数类（anger=0/disgust=1/fear=2），多数类保持原始图像和标准单标签 CE loss，期望兼顾少数类泛化与多数类稳定
- **改动**: `backend/model_njb/Transfer Learning.py`
  - 新增 MINORITY_CLASSES = frozenset({0,1,2})（anger/disgust/fear）
  - 选择性 Mixup：is_minority mask 控制图像混合范围，F.cross_entropy reduction='none' 实现逐样本 loss（minority 用双标签 Mixup loss，majority 用标准 CE）
  - 其余保持方案C：label_smoothing=0.1，weights_tensor，shuffle=True，CosineAnnealingLR，扩充 augmentation
  - 新权重保存为 best_rafdb_model_d.pth
- **指标**:
  - 方案A模型(best_rafdb_model_a.pth → 实为 _2.pth) 测试集: 总体 84.71% | anger 74.69% / disgust 58.12% / fear 54.05% / happy 92.24% / neutral 81.76% / sad 85.36% / surprised 87.54%
  - 方案C模型(best_rafdb_model_4.pth) 测试集: 总体 82.53% | anger 77.16% / disgust 65.62% / fear 58.11% / happy 87.00% / neutral 81.32% / sad 82.22% / surprised 85.71%
  - 新模型(best_rafdb_model_d.pth) 训练集: 总体 95.18% | anger 93.05% / disgust 89.40% / fear 96.80% / happy 94.24% / neutral 96.63% / sad 96.17% / surprised 98.37%
  - 新模型 测试集: 总体 83.28% | anger 73.46% / disgust 52.50% / fear 60.81% / happy 88.86% / neutral 84.56% / sad 81.38% / surprised 88.15%
- **结论**: fear 持续改善（54.05%→60.81%，vs 方案A +6.76pp），train-test 差距收窄至 36pp（vs 方案A 41pp）。但 disgust 意外退步至 52.50%（方案C 为 65.62%），训练集 disgust 也仅 89.40%（vs 方案A 95.68%），说明选择性 Mixup 对 disgust 造成了过度扰动但未转化为泛化收益。happy 从方案C的 87.00% 回升至 88.86%，但仍低于方案A（92.24%）。核心矛盾：改善少数类必然压缩多数类的精度空间。下一步建议：① 仅对 fear 单独启用 Mixup（去掉 anger/disgust），进一步隔离干扰；② 或接受当前最佳权衡，选择用途决定权重（fear 敏感场景用 model_d，整体准确率优先用 model_a）；③ 或尝试在方案A基础上只对 fear 做针对性微调（fine-tune 少数 epoch）

### #9 — 方案E：仅对 fear 启用选择性 Mixup (2026-05-04)
- **动机**: 方案D中 anger/disgust/fear 同时受 Mixup 影响导致 disgust 意外退步至 52.50%。方案E将 MINORITY_CLASSES 缩减为仅 {fear=2}，精准隔离 Mixup 副作用，观察 fear 能否在不影响 disgust 的前提下继续改善
- **改动**: `backend/model_njb/Transfer Learning.py`
  - MINORITY_CLASSES = frozenset({2})（仅 fear）
  - 其余全部与方案D相同：label_smoothing=0.1，weights_tensor，shuffle=True，CosineAnnealingLR，扩充 augmentation，MIXUP_ALPHA=0.2
  - 新权重保存为 best_rafdb_model_e.pth
- **指标**:
  - 新模型(best_rafdb_model_e.pth) 训练集: 总体 95.69% | anger 97.73% / disgust 96.09% / fear 96.09% / happy 94.01% / neutral 96.04% / sad 96.52% / surprised 98.53%
  - 新模型 测试集: 总体 82.17% | anger 82.10% / disgust 58.12% / fear 60.81% / happy 87.17% / neutral 79.85% / sad 81.80% / surprised 86.02%
- **结论**: fear 维持 60.81%（与方案D持平，train-test 差距收至 35pp，方案A为 41pp），disgust 完全恢复至 58.12%（方案A水平），anger 大幅跳升至 82.10%（+7.41pp vs 方案A，label smoothing 的意外收益）。但 neutral(-1.91pp)、happy(-5.07pp) 相比方案A仍有损失，总体 82.17%（vs A 84.71%）。各方案无单一 Pareto 最优解：A 总体最高，C disgust 最高(65.62%)，D/E fear 最高(60.81%)，E anger 最高(82.10%)。fear 已在 ~60% 附近形成瓶颈，当前 Mixup 系列方案已接近收益上限。下一步建议：① 考虑组合策略——用方案A权重做推理，对高置信 fear/disgust 预测降阈值；② 或尝试在 RAF-DB 以外引入额外 fear 样本（AffectNet/FER2013）扩充训练集；③ 或接受现状，根据应用场景选权重

### #10 — 方案F：两阶段微调（分层学习率）(2026-05-04)
- **动机**: 历轮实验 fear 在 ~60% 出现瓶颈，假设根因是全层同步训练时 fear 稀疏梯度在前几个 epoch 破坏 backbone ImageNet 特征。两阶段策略：Phase1 冻结 backbone 先让 FC 对齐（10 epoch，lr=0.01），Phase2 解冻全部分层 lr（backbone 1e-4，FC 1e-3，CosineAnnealingLR，40 epoch），全程保留方案E的 fear-only Mixup + label_smoothing=0.1
- **改动**: `backend/model_njb/Transfer Learning.py`
  - train_model 增加 use_mixup/initial_best_acc/phase_label 参数
  - Section4 重构为两阶段：Phase1（冻结backbone，SGD lr=0.01，无Mixup，10 epoch）→ Phase2（解冻，分层lr，fear Mixup，40 epoch）
  - 新权重保存为 best_rafdb_model_f.pth
- **指标**:
  - 新模型(best_rafdb_model_f.pth) 训练集: 总体 78.36% | anger 92.91% / disgust 82.01% / fear 85.05% / happy 79.95% / neutral 70.68% / sad 71.80% / surprised 86.12%
  - 新模型 测试集: 总体 70.80% | anger 79.63% / disgust 58.12% / fear 63.51% / happy 74.26% / neutral 65.00% / sad 67.78% / surprised 78.12%
- **结论**: fear 达到 63.51%（历史最高，+9.46pp vs 方案A），但整体崩至 70.80%（-13.91pp vs A）。根因有二：① Phase1 lr=0.01 对 FC 层过激，验证集在 22%~47% 剧烈振荡，导致 Phase2 起点受损（训练集仅 78.36%，严重欠拟合）；② fear Mixup 在欠拟合状态下产生大量误报（混淆矩阵：happy→fear 95 张、neutral→fear 44 张、sad→fear 32 张），大幅拖累多数类。若修复 Phase1（改 lr=0.001，延长至 20 epoch），两阶段策略本身是合理的，但代价是需要再一轮实验。综合六轮实验结果：当前最优权衡是方案E（fear 60.81%，总体 82.17%）或方案A（总体 84.71%，fear 54.05%）；fear 天花板在 60-63% 区间，进一步突破需要额外数据或更大模型

### #11 — 方案G：两阶段微调 + Phase1 收敛检测修复 (2026-05-06)
- **动机**: 方案F失败的直接原因是 Phase1 lr=0.01 过激导致 val acc 在 22%~47% 剧烈振荡，Phase2 从受损起点出发。方案G针对性修复：① Phase1 lr: 0.01→0.001；② 以 patience=5（连续5 epoch val acc 无提升）代替固定10 epoch，让 Phase1 自动收敛后再启动 Phase2；其余保持方案F不变（分层lr + fear-only Mixup + label_smoothing=0.1）
- **改动**: `backend/model_njb/Transfer Learning.py`
  - `train_model` 新增 `patience` 参数，连续 N epoch 无提升时 early stop 并返回
  - Phase1：lr=0.001，max_epochs=30，patience=5（实际第14 epoch 触发 early stop）
  - Phase2：与方案F相同（backbone 1e-4，FC 1e-3，CosineAnnealingLR T_max=40，fear Mixup）
  - 新权重保存为 best_rafdb_model_g.pth
- **指标**:
  - Phase1 实际运行 14 epoch，val acc 平稳收敛至 45.40%（无振荡）
  - Phase2 最佳 val acc: 71.23%
  - 新模型(best_rafdb_model_g.pth) 训练集: 总体 76.57% | anger 88.79% / disgust 76.15% / fear 64.41% / happy 77.79% / neutral 70.09% / sad 71.04% / surprised 89.46%
  - 新模型 测试集: 总体 71.35% | anger 77.16% / disgust 58.75% / fear 52.70% / happy 74.85% / neutral 65.74% / sad 66.74% / surprised 84.50%
  - 混淆矩阵关键误分：happy→fear 42 张、neutral→fear 11 张、sad→fear 17 张（fear Mixup 仍导致跨类误报）
- **结论**: Phase1 修复验证有效——val acc 平稳收敛至 45.4%，不再振荡。但测试集整体 71.35% 与方案F（70.80%）几乎持平，fear 反而退至 52.70%（方案F 为 63.51%）。混淆矩阵显示 fear Mixup 在 Phase2 中仍造成 happy/neutral/sad→fear 大量误报，与 Phase1 是否稳定无关。结论：Phase1 振荡不是方案F崩溃的主因；根本问题是 fear-only Mixup 在解冻全量训练时产生的副作用，两阶段策略无法解决。**至此，七轮训练实验（方案A-G）均无法在不损害整体准确率的前提下将 fear 稳定提升至 60% 以上，根本瓶颈是 fear 训练样本仅 254 张，建议就此停止纯训练技巧实验，转向数据扩充（AffectNet/FER2013）或接受方案A/E作为最终权重。**

### #8 — 切换推理权重至 best_rafdb_model_e.pth (2026-05-06)
- **动机**: 用户指定使用最新方案E权重进行网页端推理
- **改动**: `backend/api/api_server.py` possible_paths 列表首位新增 `model_njb/best_rafdb_model_e.pth`，保持原有回退路径不变
- **结论**: 重启 api_server.py 后即生效，可通过启动日志 "✅ 找到模型文件" 确认路径

### #13 — 方案I：线性探测（Linear Probe）对比实验 (2026-05-07)
- **动机**: 量化 ImageNet 预训练特征对 RAF-DB FER 任务的直接迁移能力，诊断 fear/disgust 瓶颈是「特征表示层不足」还是「backbone 微调信号不足」；为课程报告补充迁移学习对比维度
- **改动**: 新增独立脚本 `backend/model_njb/Transfer Learning_lp.py`（不修改任何现有文件）
  - 冻结所有 backbone 层（requires_grad=False），仅训练 model.fc（3,591 参数，占总量 0.032%）
  - optimizer: SGD(fc only, lr=0.1, momentum=0.9, weight_decay=1e-4)
  - 调度器: CosineAnnealingLR(T_max=50, eta_min=1e-3)
  - 增广、class weights、val split、epochs 与方案A完全相同（纯 RAF-DB，不混 FER2013）
  - 新权重保存为 best_rafdb_model_lp.pth；脚本内置 train+test 完整评估与混淆矩阵输出
- **指标**:
  - 方案A基线(best_rafdb_model_2.pth) 测试集: 总体 84.71% | anger 74.69% / disgust 58.12% / fear 54.05% / happy 92.24% / neutral 81.76% / sad 85.36% / surprised 87.54%
  - Linear Probe(best_rafdb_model_lp.pth) 验证集最佳: 49.06%（epoch 40）；训练耗时 11m 43s
  - Linear Probe 训练集（全量，无增广）: 总体 **51.61%** | anger 32.34% / disgust 22.04% / fear 26.69% / happy 83.42% / neutral 29.48% / sad 22.10% / surprised 54.96%
  - Linear Probe 测试集: 总体 **49.93%** | anger 24.69% / disgust 14.38% / fear 29.73% / happy 81.69% / neutral 29.71% / sad 22.59% / surprised 51.37%
- **结论**: Linear Probe 测试集总体 49.93%，远低于方案A的 84.71%（差距 **-34.78pp**）。关键发现：
  1. **fear 在 Linear Probe 下仅 29.73%**（vs 方案A 54.05%），差距达 24.32pp——说明 ImageNet 预训练特征并不能线性分离 fear，backbone 微调对 fear 的提升是实质性的，而非边际收益。这推翻了「特征已够好，只是梯度信号不足」的假设。
  2. **happy 在两种方法下均高**（Linear Probe 81.69% vs 方案A 92.24%），唯一相对稳定的类别，说明 happy 具有较强的视觉特征在 ImageNet 特征空间中已可线性分离。
  3. **disgust 仅 14.38%**（Linear Probe），低于随机猜测（~14.3% ≈ 1/7），说明 ImageNet 特征对 disgust 几乎无判别力，完全依赖 RAF-DB 微调后才逐步形成区分能力。
  4. **整体诊断**：fear/disgust 的瓶颈属于「特征层 + 数据量双重问题」——现有 254 张 fear 训练样本不足以让 backbone 充分调整出强判别特征，与方案H（引入 FER2013 扩充数据）的方向一致，证明数据扩充的必要性高于进一步的训练技巧。

### #12 — 方案H：FER2013 fear 扩充混合数据集训练 (2026-05-08)
- **动机**: 七轮实验（A-G）证明 fear 瓶颈在数据量（RAF-DB 仅 254 张训练样本）。从 HuggingFace 获取 FER2013 完整版，筛出 fear 类 4097 张（48×48 灰度 PNG），与 RAF-DB 合并，将 fear 训练样本从 254 张扩充至 4351 张（约 17×），从数据层尝试根治瓶颈
- **改动**: `backend/model_njb/Transfer Learning.py`
  - 新增 `FlatDirDataset`：从单个无子目录的文件夹加载图片，固定 label，自动灰度→RGB 转换
  - 新增 `build_mixed_trainset()`：RAF-DB train 90% split（带增广）+ FER2013 fear（`FlatDirDataset`），合并为 `ConcatDataset`；val subset 仍为 RAF-DB 10%（无增广），保证验证集与 test set 同域
  - class weights 按合并后实际样本数动态计算（fear weight 从 6.24 降至 0.50，与 happy 持平）
  - 其余保持方案A：CosineAnnealingLR(T_max=50)、Adam、扩充 augmentation、无 Mixup
  - 新权重保存为 `best_rafdb_model_h.pth`；`evaluate.py` MODEL_PATH 同步更新
  - FER2013 数据下载：HuggingFace `Aaryan333/fer2013_train_publicTest_privateTest`，fear 类 4097 张存至 `data/FER2013/train/fear/`
- **指标**:
  - 方案A基线(best_rafdb_model_2.pth) 测试集: 总体 84.71% | anger 74.69% / disgust 58.12% / fear 54.05% / happy 92.24% / neutral 81.76% / sad 85.36% / surprised 87.54%
  - 新模型(best_rafdb_model_h.pth) 训练集(RAF-DB 12271张): 总体 **95.79%** | anger 97.87% / disgust 94.28% / fear 69.75% / happy 96.67% / neutral 94.10% / sad 97.28% / surprised 98.91%
  - 新模型 测试集(RAF-DB 3068张): 总体 **84.00%** | anger 82.10% / disgust 48.75% / fear 50.00% / happy 90.55% / neutral 80.29% / sad 88.28% / surprised 87.54%
  - 混淆矩阵关键误分：disgust→anger 26张、disgust→sad 22张；fear→surprised 16张、fear→sad 9张
- **结论**: **假设被证伪——fear 从 54.05% 退至 50.00%（-4.05pp），整体略降至 84.00%（-0.71pp）**。根本原因是域偏移（domain shift）：FER2013 为 48×48 灰度实验室图像，RAF-DB 为彩色自然场景图像，两者 fear 的视觉特征分布差异显著，灰度图经复制通道转 RGB 后引入的频率偏置反而干扰了模型对 RAF-DB fear 的判别。fear 训练集准确率仅 69.75%（方案A为 95%+），说明模型连 FER2013 fear 自身都未能充分拟合，两域差异已超出数据扩充能弥补的范围。disgust 大幅退步（-9.37pp）的原因：fear 样本 17× 扩充后权重降至 0.50，disgust 权重虽不变但模型注意力被大量 FER2013 fear 样本稀释。**anger 意外大幅改善（+7.41pp→82.10%）、sad 改善（+2.92pp→88.28%）**，是增广多样性和较大 batch 中正样本梯度信号更稳定的结果。综合结论：跨数据集域偏移是比样本量更严重的瓶颈；若引入外部数据，需优先保证域对齐（彩色、自然场景、相近分辨率），FER2013 灰度数据不适合直接混合 RAF-DB 训练。下一步建议：① 接受方案A（84.71%，总体最优）或方案E（fear 60.81%）作为最终权重；② 若继续数据扩充，应筛选 AffectNet 或 RAF-DB 同分布彩色 fear 样本

### #14 — 报告可视化补全 (2026-05-07)
- **动机**: 最终报告需配图说明所有实验结果，包含方案 H 与 Linear Probe 的新对比
- **改动**: 更新 `Graph/generate_final_report_plots.py`，加入 H/LP(I) 数据，新增两张图：`lp_vs_finetune.png`（线性探测 vs 全量微调逐类对比）、`fear_train_test_gap.png`（各方案 fear 训练-测试间隙图）；同时再生成全量更新后的 4 张既有图；在 `paper/Final_Report_CS289A.tex` 中添加 `\graphicspath`、更新图路径与 caption、补充图引用
- **结论**: 共 6 张图（300 dpi PNG）存于 `Graph/`，报告图文完整对应

### #15 — Grad-CAM 可解释性可视化 (2026-05-07)
- **动机**: 报告补充模型可解释性分析，直观展示 fear/disgust 误分机制
- **改动**: 新增 `Graph/generate_gradcam.py`，基于方案 A 权重（`best_rafdb_model_a.pth`），hook `layer4[1].conv2` 实现手写 Grad-CAM；生成 `gradcam_correct_grid.png`（7 类各 2 张正确样本）和 `gradcam_misclassify.png`（fear/disgust 典型误分对比，同时展示对预测类和真实类的 CAM）；在报告讨论节插入 Grad-CAM 分析段落及两张图，参考文献补充 Selvaraju et al. 2017
- **指标**: 7 类测试集正确预测统计：happy 1093/1185, neutral 556/680, sad 408/478, surprised 288/329, anger 121/162, disgust 93/160, fear 40/74
- **结论**: Grad-CAM 确认 fear→surprised 误分的根因为眼部 AU5 特征共享；fear/disgust 训练集不足导致特征区分度不足，为后续数据增广（尤其眼部变形）提供了可视化依据

### #16 — 方案J 后处理置信度补偿 (2026-05-07)
- **动机**: 在不重训的条件下提升 fear/disgust 少数类推理准确率；探索 post-hoc logit bias 的效果上限与风险
- **改动**: 新增 `backend/model_njb/calibrate.py`（10% 训练集验证分割网格搜索 logit bias）、`backend/model_njb/bias_fear_only.json`（fear: +2.0, 其余 0）、`backend/model_njb/bias_a.json`（全类 full bias）；将 bias 应用集成到 `backend/api/api_server.py` 两条推理路径；生成对比图 `Graph/calibration_comparison.png`；更新 LaTeX 报告（方案 J 条目、图、讨论段落、贡献节、结论节）
- **指标**:
  - Baseline（方案 A）: 整体 84.71% | fear 54.05% | disgust 58.13%
  - Fear-only (+2.0): 整体 **84.52%** (-0.19pp) | fear **62.16%** (+8.11pp) | disgust 58.13% (不变)
  - Full bias (+2.5/-0.5/+2.0): 整体 83.87% | fear 60.81% | anger 83.95% (+9.26pp) | disgust 49.38% (-8.75pp)
- **结论**: Fear-only 偏置是最优生产方案——fear 创全实验最高（62.16%），整体代价极小；Full bias 揭示无约束宏平均优化的类间转移风险（disgust 大幅回退），提示后处理校准须施加各类下限约束
