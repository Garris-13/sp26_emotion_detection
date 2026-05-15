"""
在 RAF-DB test set 上评估 best_rafdb_model_1.pth 的分类准确率。
用法（从项目根目录执行）:
    python "backend/model_njb/evaluate.py"
"""
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # Emotion_detection/
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)                    # backend/

for p in (BACKEND_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from models.emotion_model import load_model

# ── 路径 ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(SCRIPT_DIR, 'best_rafdb_model_h.pth')
TEST_DIR     = os.path.join(REPO_ROOT, 'data', 'RAF-DB', 'test')
TRAIN_DIR    = os.path.join(REPO_ROOT, 'data', 'RAF-DB', 'train')

assert os.path.isfile(MODEL_PATH), f"找不到权重文件: {MODEL_PATH}"
assert os.path.isdir(TEST_DIR),    f"找不到测试集目录: {TEST_DIR}"
assert os.path.isdir(TRAIN_DIR),   f"找不到训练集目录: {TRAIN_DIR}"

# ── 推理用的 transform（与训练 val_test 一致）────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── 加载 ──────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

print(f"加载权重: {MODEL_PATH}  ({os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB)")
model = load_model(MODEL_PATH, num_classes=7, device=device)
model.eval()
print("模型加载成功\n")

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# ImageFolder 按目录名字母序分配索引（测试集用命名目录）
test_class_names = test_dataset.classes   # ['anger','disgust','fear','happy','neutral','sad','surprised']

# best_rafdb_model_1.pth 训练时目录名已是英文类名，ImageFolder 字母序与测试集完全一致:
#   [0:anger, 1:disgust, 2:fear, 3:happy, 4:neutral, 5:sad, 6:surprised]
# pred_remap 因此变为恒等映射，实际不做任何重映射。
MODEL_CLASS_NAMES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# 建立 模型输出索引 → 当前测试集类索引 的映射
test_class_idx = {name: i for i, name in enumerate(test_class_names)}
pred_remap = {i: test_class_idx[name] for i, name in enumerate(MODEL_CLASS_NAMES)}

n_classes = len(test_class_names)
print(f"测试集: {len(test_dataset)} 张  ×  {n_classes} 类")
print(f"测试集类别 (字母序): {test_class_names}")
print(f"模型训练时类别顺序: {MODEL_CLASS_NAMES}")
print(f"\n预测索引重映射: { {MODEL_CLASS_NAMES[k]: test_class_names[v] for k,v in pred_remap.items()} }\n")

# ── 推理 ──────────────────────────────────────────────────────────────────────
class_correct = [0] * n_classes
class_total   = [0] * n_classes
all_preds     = []
all_labels    = []

with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, raw_preds = torch.max(outputs, 1)

        # 将模型输出索引重映射到当前测试集索引后再对比
        remapped = raw_preds.cpu().apply_(lambda p: pred_remap[p])

        for p, t in zip(remapped.tolist(), labels.cpu().tolist()):
            class_total[t]   += 1
            class_correct[t] += int(p == t)

        all_preds.extend(remapped.tolist())
        all_labels.extend(labels.cpu().tolist())

# ── 结果 ──────────────────────────────────────────────────────────────────────
total   = sum(class_total)
correct = sum(class_correct)
overall = 100.0 * correct / total

print("=" * 52)
print(f"  总体准确率 (Overall Accuracy): {overall:.2f}%")
print(f"  ({correct}/{total})")
print("=" * 52)
print(f"\n{'类别':<12}{'样本数':>6}  {'正确':>6}  {'准确率':>8}")
print("-" * 40)
for i, name in enumerate(test_class_names):
    acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] else 0.0
    print(f"{name:<12}{class_total[i]:>6}  {class_correct[i]:>6}  {acc:>7.2f}%")
print("-" * 40)

# ── 混淆矩阵 ─────────────────────────────────────────────────────────────────
cm = [[0]*n_classes for _ in range(n_classes)]
for p, t in zip(all_preds, all_labels):
    cm[t][p] += 1

print("\n混淆矩阵 (行=真实, 列=预测):")
header = f"{'':12}" + "".join(f"{c[:4]:>6}" for c in test_class_names)
print(header)
for i, name in enumerate(test_class_names):
    row = f"{name:<12}" + "".join(f"{cm[i][j]:>6}" for j in range(n_classes))
    print(row)

# ── Train set 评估（无增广，与 val_test transform 相同）────────────────────────
print("\n\n" + "=" * 52)
print("  训练集评估 (Train Set, 无增广)")
print("=" * 52)

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

train_class_names = train_dataset.classes
train_class_idx   = {name: i for i, name in enumerate(train_class_names)}
train_pred_remap  = {i: train_class_idx[name] for i, name in enumerate(MODEL_CLASS_NAMES)}

n_train_classes    = len(train_class_names)
tr_class_correct   = [0] * n_train_classes
tr_class_total     = [0] * n_train_classes

print(f"训练集: {len(train_dataset)} 张  ×  {n_train_classes} 类")

with torch.no_grad():
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, raw_preds = torch.max(outputs, 1)
        remapped = raw_preds.cpu().apply_(lambda p: train_pred_remap[p])
        for p, t in zip(remapped.tolist(), labels.cpu().tolist()):
            tr_class_total[t]   += 1
            tr_class_correct[t] += int(p == t)

tr_total   = sum(tr_class_total)
tr_correct = sum(tr_class_correct)
tr_overall = 100.0 * tr_correct / tr_total

print("=" * 52)
print(f"  训练集总体准确率: {tr_overall:.2f}%")
print(f"  ({tr_correct}/{tr_total})")
print("=" * 52)
print(f"\n{'类别':<12}{'样本数':>6}  {'正确':>6}  {'准确率':>8}")
print("-" * 40)
for i, name in enumerate(train_class_names):
    acc = 100.0 * tr_class_correct[i] / tr_class_total[i] if tr_class_total[i] else 0.0
    print(f"{name:<12}{tr_class_total[i]:>6}  {tr_class_correct[i]:>6}  {acc:>7.2f}%")
print("-" * 40)
