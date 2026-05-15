import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import time
import os
import copy
from collections import Counter

# ==========================================
# 1. 配置参数 (根据你的实际情况修改)
# ==========================================
# RAF-DB 数据集根目录, 内部结构 (已经组织成 ImageFolder 约定):
#   data_dir/
#     train/{anger,disgust,fear,happy,neutral,sad,surprised}/*.jpg     12271 张
#     test/ {anger,disgust,fear,happy,neutral,sad,surprised}/*.jpg      3068 张
#     train_labels.csv / test_labels.csv  ← Kaggle 原始 1-7 数字标签映射, 此脚本不需要
data_dir = '/home/wumenglin/Emotion_detection/data/RAF-DB'
FER2013_FEAR_DIR = os.path.join(os.path.dirname(data_dir), 'FER2013', 'train', 'fear')
batch_size = 32
num_epochs = 50           # 训练轮数
learning_rate = 0.001     # 初始学习率
val_split = 0.1           # 划出 10% 的训练集作为验证集
random_seed = 42          # 固定随机种子，保证每次划分一致
MIXUP_ALPHA = 0.2         # Mixup Beta 分布参数，0 表示关闭 Mixup
MINORITY_CLASSES = frozenset({2})  # 仅 fear=2（ImageFolder 字母序：anger=0,disgust=1,fear=2）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 检查点路径锚定到脚本所在目录, 避免 cwd 不同导致找不到/覆盖错文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'best_rafdb_model_h.pth')

print(f"使用设备: {device}")
print(f"数据目录: {data_dir}")
print(f"检查点路径: {CHECKPOINT_PATH}")

# 数据目录存在性检查 (路径写错就快速失败, 不要等到 ImageFolder 抛 FileNotFoundError)
for _split in ('train', 'test'):
    _split_dir = os.path.join(data_dir, _split)
    assert os.path.isdir(_split_dir), (
        f"未找到 {_split} 目录: {_split_dir}\n"
        f"请确认 data_dir 是否指向 RAF-DB 根目录 (内部需含 train/ 和 test/ 两个子目录)。"
    )

assert os.path.isdir(FER2013_FEAR_DIR), (
    f"未找到 FER2013 fear 目录: {FER2013_FEAR_DIR}\n"
    f"请先运行: kaggle datasets download -d msambare/fer2013 -p data/FER2013 --unzip"
)

# ==========================================
# 2. 数据预处理与加载 (核心修改部分)
# ==========================================

class FlatDirDataset(torch.utils.data.Dataset):
    """从单个目录直接加载图片（无子目录结构），所有样本使用固定 label。
    支持灰度→RGB 转换，用于加载 FER2013 fear 目录。
    """
    EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

    def __init__(self, directory, label, transform=None):
        self.label = label
        self.transform = transform
        self.paths = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in self.EXTS
        ])
        if len(self.paths) == 0:
            raise RuntimeError(f"FlatDirDataset: 在 {directory} 中未找到图片")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert('RGB')  # 灰度→3通道
        if self.transform:
            img = self.transform(img)
        return img, self.label


def build_mixed_trainset(raf_train_dir, fer_fear_dir, train_transform, val_transform,
                         val_split=0.1, seed=42):
    """合并 RAF-DB train（90% split）与 FER2013 fear 类。
    - val subset 仅使用 RAF-DB，保持验证集与 test set 同域
    - FER2013 fear 图像灰度→RGB，标签强制映射为 fear=2（字母序）
    返回: combined_train, val_subset, class_counts, fer_count, class_names
    """
    FEAR_IDX = 2  # 字母序: anger=0, disgust=1, fear=2

    full_raf_aug   = datasets.ImageFolder(raf_train_dir, transform=train_transform)
    full_raf_clean = datasets.ImageFolder(raf_train_dir, transform=val_transform)

    n_total = len(full_raf_aug)
    indices = list(range(n_total))
    split   = int(np.floor(val_split * n_total))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    raf_train_subset = Subset(full_raf_aug,   train_idx)
    raf_val_subset   = Subset(full_raf_clean, val_idx)

    fer_fear = FlatDirDataset(
        fer_fear_dir,
        label=FEAR_IDX,
        transform=train_transform
    )

    combined_train = ConcatDataset([raf_train_subset, fer_fear])

    raf_labels = [full_raf_aug.targets[i] for i in train_idx]
    counts = Counter(raf_labels)
    counts[FEAR_IDX] += len(fer_fear)
    n_classes    = len(full_raf_aug.classes)
    class_counts = [counts[i] for i in range(n_classes)]

    return combined_train, raf_val_subset, class_counts, len(fer_fear), full_raf_aug.classes

# 定义数据增强和转换
data_transforms = {
    # 训练集：需要数据增强
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
    # 验证集和测试集：不需要增强，只需标准化
    'val_test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("正在准备数据集...")

# A. 加载 Test 集 (直接读取 test 文件夹)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), 
                                    transform=data_transforms['val_test'])

# B. 加载 Train 集并拆分（混合 RAF-DB + FER2013 fear）
train_dataset, val_dataset, _train_class_counts, _len_fer_fear, class_names = build_mixed_trainset(
    raf_train_dir   = os.path.join(data_dir, 'train'),
    fer_fear_dir    = FER2013_FEAR_DIR,
    train_transform = data_transforms['train'],
    val_transform   = data_transforms['val_test'],
    val_split       = val_split,
    seed            = random_seed
)

# 创建 DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4),
    'valid': DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4),
    'test':  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)
}

dataset_sizes = {
    'train': len(train_dataset),
    'valid': len(val_dataset),
    'test':  len(test_dataset)
}

print(f"数据准备完成:")
print(f"- 训练集 (Train): {dataset_sizes['train']} 张 (RAF-DB + FER2013 fear {_len_fer_fear} 张)")
print(f"- 验证集 (Valid): {dataset_sizes['valid']} 张 (仅 RAF-DB)")
print(f"- 测试集 (Test) : {dataset_sizes['test']} 张 (仅 RAF-DB)")
print(f"- 类别: {class_names}")

# ==========================================
# 3. 定义训练函数
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=25,
                use_mixup=True, initial_best_acc=0.0, phase_label="", patience=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = initial_best_acc
    no_improve_count = 0

    for epoch in range(num_epochs):
        prefix = f"[{phase_label}] " if phase_label else ""
        print(f'\n{prefix}Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个 epoch 包含训练和验证两个阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 选择性 Mixup（仅对少数类样本生效）
                labels_b = labels
                lam = 1.0
                is_minority = torch.zeros(inputs.size(0), device=device)
                if phase == 'train' and use_mixup and MIXUP_ALPHA > 0:
                    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                    rand_idx = torch.randperm(inputs.size(0), device=device)
                    labels_b = labels[rand_idx]
                    is_minority = torch.tensor(
                        [1.0 if l.item() in MINORITY_CLASSES else 0.0 for l in labels],
                        device=device
                    )
                    mask = is_minority.view(-1, 1, 1, 1)
                    inputs = mask * (lam * inputs + (1 - lam) * inputs[rand_idx]) + (1 - mask) * inputs

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train' and use_mixup and MIXUP_ALPHA > 0:
                        ce_a = F.cross_entropy(outputs, labels, weight=weights_tensor,
                                               label_smoothing=0.1, reduction='none')
                        ce_b = F.cross_entropy(outputs, labels_b, weight=weights_tensor,
                                               label_smoothing=0.1, reduction='none')
                        loss = (is_minority * (lam * ce_a + (1 - lam) * ce_b)
                                + (1 - is_minority) * ce_a).mean()
                    else:
                        loss = criterion(outputs, labels)

                    # 只有训练阶段才反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制模型 (只保存验证集准确率最高的模型)
            if phase == 'valid':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), CHECKPOINT_PATH)
                    print(f'==> 发现新最佳模型，已保存 (Acc: {best_acc:.4f})')
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if patience is not None and no_improve_count >= patience:
                        time_elapsed = time.time() - since
                        print(f'\n{prefix}Early stop: 连续 {patience} epoch 无提升，Phase 收敛')
                        print(f'{prefix}耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                        print(f'{prefix}验证集最佳准确率: {best_acc:.4f}')
                        model.load_state_dict(best_model_wts)
                        return model, best_acc

    time_elapsed = time.time() - since
    prefix = f"[{phase_label}] " if phase_label else ""
    print(f'\n{prefix}训练完成，耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'{prefix}验证集最佳准确率: {best_acc:.4f}')

    # 加载最佳权重
    model.load_state_dict(best_model_wts)
    return model, best_acc

# ==========================================
# 4. 初始化模型并开始训练（单阶段，方案A设置 + 混合数据集）
# ==========================================
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 7)
model_ft = model_ft.to(device)

# 按合并后实际样本数计算类别权重（inverse frequency）
_total_samples = sum(_train_class_counts)
_n_classes = len(class_names)
class_weights = [_total_samples / (_n_classes * _train_class_counts[i]) for i in range(_n_classes)]

print("\n[混合训练集 class 分布]")
for _i, _name in enumerate(class_names):
    print(f"  [{_i}] {_name:<12}  count={_train_class_counts[_i]:>5}  weight={class_weights[_i]:.4f}")
print(f"  合计: {_total_samples} 张（RAF-DB: {_total_samples - _len_fer_fear}，FER2013 fear: {_len_fer_fear}）\n")

weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
scheduler_ft = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=num_epochs, eta_min=1e-6)

model_ft, _ = train_model(
    model_ft, criterion, optimizer_ft, scheduler_ft,
    num_epochs=num_epochs, use_mixup=False, initial_best_acc=0.0
)

# ==========================================
# 5. 最终测试 (使用 Test 集)
# ==========================================
print("\n" + "="*20)
print("正在使用 Test 集进行最终评估...")
print("="*20)

# 加载刚才训练保存的最佳权重 (确保是最佳状态)
model_ft.load_state_dict(torch.load(CHECKPOINT_PATH))
model_ft.eval()

correct = 0
total = 0
class_correct = list(0. for i in range(7))
class_total = list(0. for i in range(7))

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print(f'\n最终测试集准确率 (Test Accuracy): {100 * correct / total:.2f}%')
print("-" * 30)
for i in range(7):
    if class_total[i] > 0:
        print(f'{class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
