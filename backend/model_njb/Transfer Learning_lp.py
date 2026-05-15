"""
方案 I：线性探测（Linear Probe）对比实验
- 冻结 ResNet18 全部 backbone，只训练最后一层 FC（3584 参数）
- 纯 RAF-DB（不混 FER2013），与方案A完全可比
- 优化器：SGD(lr=0.1, momentum=0.9)，CosineAnnealingLR(eta_min=1e-3)
- 训练结束后自动对 train set + test set 做完整评估并输出混淆矩阵
- 输出权重：best_rafdb_model_lp.pth

用法（从项目根目录执行）：
    python "backend/model_njb/Transfer Learning_lp.py"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import os
import copy

# ==========================================
# 1. 配置参数
# ==========================================
data_dir     = '/home/wumenglin/Emotion_detection/data/RAF-DB'
batch_size   = 32
num_epochs   = 50
val_split    = 0.1
random_seed  = 42
device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'best_rafdb_model_lp.pth')

print(f"使用设备: {device}")
print(f"数据目录: {data_dir}")
print(f"检查点路径: {CHECKPOINT_PATH}")

for _split in ('train', 'test'):
    _d = os.path.join(data_dir, _split)
    assert os.path.isdir(_d), f"未找到目录: {_d}"

# ==========================================
# 2. 数据预处理（与方案A完全相同）
# ==========================================
data_transforms = {
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
    'val_test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

print("正在准备数据集...")

full_raf_aug   = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                      transform=data_transforms['train'])
full_raf_clean = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                      transform=data_transforms['val_test'])
class_names = full_raf_aug.classes  # ['anger','disgust','fear','happy','neutral','sad','surprised']
n_classes   = len(class_names)

n_total = len(full_raf_aug)
indices = list(range(n_total))
np.random.seed(random_seed)
np.random.shuffle(indices)
split      = int(np.floor(val_split * n_total))
train_idx  = indices[split:]
val_idx    = indices[:split]

train_dataset = Subset(full_raf_aug,   train_idx)
val_dataset   = Subset(full_raf_clean, val_idx)
test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                     transform=data_transforms['val_test'])

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4),
    'valid': DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4),
    'test':  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4),
}
dataset_sizes = {
    'train': len(train_dataset),
    'valid': len(val_dataset),
    'test':  len(test_dataset),
}

# 按 train split 计算类别权重（inverse frequency，与方案A一致）
from collections import Counter
train_labels  = [full_raf_aug.targets[i] for i in train_idx]
train_counts  = Counter(train_labels)
n_train       = len(train_idx)
class_weights = [n_train / (n_classes * train_counts[i]) for i in range(n_classes)]

print(f"数据准备完成:")
print(f"  训练集: {dataset_sizes['train']} 张")
print(f"  验证集: {dataset_sizes['valid']} 张")
print(f"  测试集: {dataset_sizes['test']} 张")
print(f"\n[类别分布（train split）]")
for i, name in enumerate(class_names):
    print(f"  [{i}] {name:<12}  count={train_counts[i]:>5}  weight={class_weights[i]:.4f}")

weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion      = nn.CrossEntropyLoss(weight=weights_tensor)

# ==========================================
# 3. 模型初始化：冻结 backbone，只开放 FC
# ==========================================
model_ft = models.resnet18(pretrained=True)

# 冻结所有参数（backbone）
for param in model_ft.parameters():
    param.requires_grad = False

# 替换 FC（新 FC 默认 requires_grad=True）
num_ftrs     = model_ft.fc.in_features
model_ft.fc  = nn.Linear(num_ftrs, n_classes)
model_ft     = model_ft.to(device)

trainable_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model_ft.parameters())
print(f"\n[Linear Probe] 可训练参数: {trainable_params:,} / 总参数: {total_params:,}")
print(f"  （仅 FC 层，约占 {100*trainable_params/total_params:.3f}%）\n")

# 只向 optimizer 传入 FC 参数
optimizer_ft = optim.SGD(
    model_ft.fc.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
)
# eta_min 设为 1e-3（不退到接近 0，凸优化不需要极小 lr）
scheduler_ft = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=num_epochs, eta_min=1e-3)

# ==========================================
# 4. 训练函数
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since          = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc       = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()

            running_loss     = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs       = model(inputs)
                    _, preds      = torch.max(outputs, 1)
                    loss          = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc       = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f'==> 发现新最佳模型，已保存 (Acc: {best_acc:.4f})')

    elapsed = time.time() - since
    print(f'\n训练完成，耗时: {elapsed // 60:.0f}m {elapsed % 60:.0f}s')
    print(f'验证集最佳准确率: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 5. 执行训练
# ==========================================
model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler_ft, num_epochs=num_epochs)

# ==========================================
# 6. 完整评估函数（输出逐类准确率 + 混淆矩阵）
# ==========================================
def evaluate(model, loader, split_name, class_names):
    model.eval()
    n = len(class_names)
    class_correct = [0] * n
    class_total   = [0] * n
    cm            = [[0] * n for _ in range(n)]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs   = model(inputs)
            _, preds  = torch.max(outputs, 1)
            for p, t in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                class_total[t]   += 1
                class_correct[t] += int(p == t)
                cm[t][p]         += 1

    total   = sum(class_total)
    correct = sum(class_correct)
    overall = 100.0 * correct / total

    print("\n" + "=" * 56)
    print(f"  {split_name} 评估结果（{total} 张）")
    print("=" * 56)
    print(f"  总体准确率: {overall:.2f}%  ({correct}/{total})")
    print("-" * 56)
    print(f"{'类别':<14}{'样本数':>6}  {'正确':>6}  {'准确率':>8}")
    print("-" * 40)
    for i, name in enumerate(class_names):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] else 0.0
        print(f"{name:<14}{class_total[i]:>6}  {class_correct[i]:>6}  {acc:>7.2f}%")
    print("-" * 40)

    print(f"\n混淆矩阵（行=真实，列=预测）:")
    header = f"{'':14}" + "".join(f"{c[:5]:>7}" for c in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = f"{name:<14}" + "".join(f"{cm[i][j]:>7}" for j in range(n))
        print(row)

    return overall, class_correct, class_total

# ==========================================
# 7. 加载最佳权重并运行 train + test 评估
# ==========================================
print("\n\n" + "=" * 56)
print("  加载最佳权重，开始完整评估")
print("=" * 56)
model_ft.load_state_dict(torch.load(CHECKPOINT_PATH))

# 7a. Train set（使用 val_test transform，无增广）
train_eval_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'train'),
    transform=data_transforms['val_test']
)
train_eval_loader = DataLoader(
    train_eval_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)
evaluate(model_ft, train_eval_loader, "Train Set（全量 RAF-DB train，无增广）", class_names)

# 7b. Test set
test_eval_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)
evaluate(model_ft, test_eval_loader, "Test Set（RAF-DB test）", class_names)

print("\n\n[方案A基线对照]")
print("  方案A测试集: 总体 84.71% | anger 74.69% / disgust 58.12% / fear 54.05%")
print("               happy 92.24% / neutral 81.76% / sad 85.36% / surprised 87.54%")
print("\n评估完成。请将以上结果填入 EXPERIMENT_LOG.md #13 条目。")
