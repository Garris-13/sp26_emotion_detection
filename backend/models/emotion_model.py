"""
表情识别模型定义
用于部署和推理
"""

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    TORCH_AVAILABLE = True
except Exception as e:
    # 允许在无法加载 PyTorch 的环境下导入此模块（例如缺少 DLL）
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    models = None
    print(f"⚠️ backend/models/emotion_model.py: PyTorch 导入失败，库不可用: {e}")


if TORCH_AVAILABLE:
    class EmotionRecognitionModel(nn.Module):
        """表情识别模型（基于 ResNet18）"""

        def __init__(self, num_classes=7, model_name='resnet18', pretrained=False):
            super(EmotionRecognitionModel, self).__init__()

            self.model_name = 'resnet18'

            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.backbone(x)
else:
    # 定义占位类，以便在没有 torch 的环境下导入模块不会崩溃
    class EmotionRecognitionModel(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('PyTorch 不可用，不能创建 EmotionRecognitionModel')


def load_model(model_path, model_name='resnet18', num_classes=7, device='cpu'):
    """加载训练好的模型，如果 PyTorch 不可用则抛出异常"""
    if not TORCH_AVAILABLE:
        raise RuntimeError('PyTorch 不可用，无法加载模型')

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # 判断权重是否带 backbone. 前缀
    sample_key = next(iter(state_dict))
    has_backbone_prefix = sample_key.startswith('backbone.')

    if has_backbone_prefix:
        # 权重与 EmotionRecognitionModel 包装层匹配，直接加载
        model = EmotionRecognitionModel(num_classes=num_classes, model_name=model_name, pretrained=False)
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"❌ 加载模型权重失败（带前缀模式）: {e}")
            raise e
    else:
        # 权重是裸 ResNet18，不含 backbone. 前缀（如用 Transfer Learning.py 直接训练保存的）
        # 直接加载进 torchvision ResNet18，再套上包装层
        import torchvision.models as tv_models
        backbone = tv_models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(in_features, num_classes)
        try:
            backbone.load_state_dict(state_dict)
        except Exception as e:
            print(f"❌ 加载裸 ResNet 权重失败: {e}")
            raise e
        # 包进 EmotionRecognitionModel 以统一接口
        model = EmotionRecognitionModel(num_classes=num_classes, model_name=model_name, pretrained=False)
        model.backbone = backbone

    model.to(device)
    model.eval()
    return model