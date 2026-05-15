# 项目约定

本仓库为 ResNet18 + YOLOv8-face + MTCNN 实时表情识别系统，仅做核心推理与训练，不再包含 LLM 对话、健康建议、数据库等模块。

---

## 必须遵守

### 1. 每次改动追加 EXPERIMENT_LOG.md

只要触及以下任一项，**任务结束前必须**在 `EXPERIMENT_LOG.md` 末尾追加一条 entry：

- 模型结构、权重、训练脚本（`backend/model_njb/Transfer Learning.py` 等）
- 推理管线（`backend/api/api_server.py`、`backend/api/yolo_face_detector.py`、`backend/models/emotion_model.py`）
- 依赖（`backend/requirements.txt`）
- API 端点行为
- 前端推理交互逻辑（`frontend/examples/realtime_emotion.html`）
- 数据集准备/格式

**模板**（保持简洁，一两行也行）：
```markdown
### #N — 标题 (YYYY-MM-DD)
- **动机**: 为什么改
- **改动**: 改了什么（关键文件）
- **指标**: 训练/推理结果（如适用，可省略）
- **结论**: 下一步或验证状态
```

N 自增（看上一条最大编号 +1）。日期用今日实际日期。

**不需要追加日志**的情况：纯文档修改（README/CLAUDE.md）、bug 修复但行为不变、格式化、重命名变量。

### 2. 训练脚本不得随意改动

`backend/model_njb/Transfer Learning.py` 是训练入口，除非用户明确要求否则不要改。需要改 `data_dir`（第 15 行）时优先提示用户而非直接改。

### 3. 不要重新引入已删除的模块

以下功能已**永久移除**，任何任务都不要把它们加回来：
- LangGraph / LangChain / 多 Agent 对话
- MySQL / 数据库存储
- 阿里云 DashScope / OpenAI LLM 调用
- 健康建议规则引擎
- 复杂摄像头监控（camera_monitor.py 系列）
- ResNet50（只用 ResNet18）

### 4. 前端只保留实用功能

`realtime_emotion.html` 当前展示：人脸框、各情绪概率条、推理速度、置信度阈值、FPS 控制。新功能优先讨论后再加，不要主动塞 LLM 分析、聊天框等重型 UI。

---

## 当前状态速查

- 推理权重：`backend/model_njb/best_rafdb_model_1.pth`
- YOLO 权重：`backend/yolov8n-face.pt`（6.1 MB）
- 数据集：`data/RAF-DB/{train(12271), test(3068)}/`
- 启动命令：见 `README_cn.md`「快速开始」或 `README.md`「Quick start」
- 已知问题：远距离小脸（~40×40 px）经 Resize(256) 上采样后 ResNet18 输出近均匀概率，待用模糊增广重训改善
