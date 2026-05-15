"""
表情识别 REST API
基于 ResNet18（情绪分类）+ YOLOv8-face（实时人脸检测）+ MTCNN（单图人脸检测）
"""

import sys
import os

# ================ 路径设置 ================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
for path in (PROJECT_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

print("=" * 70)
print("🚀 表情识别 API")
print("=" * 70)
print(f"📁 项目根目录: {PROJECT_ROOT}")
print(f"📁 仓库根目录: {REPO_ROOT}")

# ================ 第三方库导入（含降级保护） ================
try:
    import cv2
    CV2_AVAILABLE = True
    print(f"✅ OpenCV 版本: {cv2.__version__}")
except ImportError as e:
    CV2_AVAILABLE = False
    print(f"❌ OpenCV 不可用: {e}")

from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except Exception as e:
    torch = None
    transforms = None
    TORCH_AVAILABLE = False
    print(f"⚠️ PyTorch 导入失败，进入降级模式: {e}")

try:
    from facenet import MTCNN
    FACENET_AVAILABLE = True
    FACENET_SOURCE = 'facenet'
except Exception as e1:
    try:
        from facenet_pytorch import MTCNN
        FACENET_AVAILABLE = True
        FACENET_SOURCE = 'facenet_pytorch'
    except Exception as e2:
        MTCNN = None
        FACENET_AVAILABLE = False
        FACENET_SOURCE = None
        print(f"⚠️ facenet MTCNN 不可用: {e1}")
        print(f"⚠️ facenet_pytorch MTCNN 也不可用: {e2}")

from PIL import Image
import io
import base64
import numpy as np
from datetime import datetime

# ================ 模型导入 ================
try:
    from models.emotion_model import load_model, EmotionRecognitionModel
    MODEL_IMPORT_SUCCESS = True
    print("✅ 成功导入表情识别模型模块")
except ImportError as e:
    MODEL_IMPORT_SUCCESS = False
    print(f"⚠️  导入模型模块失败: {e}")

# ================ Flask 应用 ================
app = Flask(__name__)
CORS(app)

# ================ 全局状态 ================
model = None
device = None
transform = None
single_image_face_detector = None
single_image_face_cascade = None
single_image_face_method = 'none'
yolo_face_detector = None

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
EMOTION_ZH = {
    'surprised': '惊讶',
    'fear': '恐惧',
    'disgust': '厌恶',
    'happy': '快乐',
    'sad': '悲伤',
    'anger': '愤怒',
    'neutral': '平静',
}

# ── 后处理 logit bias（置信度补偿） ──────────────────────────────────────────
# 加载 bias_fear_only.json（仅 fear +2.0），对少数类进行后验校准。
# 若文件不存在则 bias 全为 0（无校准）。
_BIAS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'model_njb', 'bias_fear_only.json'
)
if os.path.isfile(_BIAS_PATH):
    import json as _json
    _bias_cfg = _json.load(open(_BIAS_PATH))
    LOGIT_BIAS = [_bias_cfg.get(lbl, 0.0) for lbl in EMOTION_LABELS]
    print(f"✅ 加载 logit bias: {dict(zip(EMOTION_LABELS, LOGIT_BIAS))}")
else:
    LOGIT_BIAS = [0.0] * len(EMOTION_LABELS)
    print("ℹ️  未找到 bias 文件，logit bias 全为 0")


# ================ 初始化函数 ================
def initialize_model():
    """加载 ResNet18 情绪识别模型权重，并构建预处理 pipeline"""
    global model, device, transform
    print("\n" + "=" * 70)
    print("🤖 初始化表情识别模型")
    print("=" * 70)

    if TORCH_AVAILABLE and torch is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 使用设备: {device}")
    else:
        device = None
        print("⚠️ PyTorch 不可用，模型推理将禁用（降级模式）")

    try:
        possible_paths = [
            os.path.join(PROJECT_ROOT, 'model_njb', 'best_rafdb_model_e.pth'),
            os.path.join(PROJECT_ROOT, 'model_njb', 'best_rafdb_model_1.pth'),
            os.path.join(REPO_ROOT, 'best_model.pth'),
            os.path.join(PROJECT_ROOT, 'best_model.pth'),
            'best_model.pth',
        ]
        model_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if model_path is None:
            print("⚠️  未找到 best_model.pth，创建虚拟模型")
            if MODEL_IMPORT_SUCCESS and TORCH_AVAILABLE:
                model = EmotionRecognitionModel(num_classes=7, model_name='resnet18', pretrained=False)
                model.to(device)
                model.eval()
                print("✅ 虚拟模型创建成功（输出未训练）")
            else:
                model = None
        else:
            print(f"✅ 找到模型文件: {model_path} ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")
            if MODEL_IMPORT_SUCCESS:
                model = load_model(model_path, model_name='resnet18', num_classes=7, device=device)
                print("✅ 模型加载成功")
            else:
                model = None
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        model = None

    # 预处理 pipeline（与 backend/model_njb/Transfer Learning.py 一致）
    if transforms is not None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = lambda img: img
    print("✅ 图像预处理器初始化完成")

    initialize_single_image_face_detector()
    initialize_yolo_face_detector()
    return model is not None


def initialize_yolo_face_detector():
    """初始化 YOLOv8-face 实时人脸检测器"""
    global yolo_face_detector
    weights_candidates = [
        os.path.join(PROJECT_ROOT, 'yolov8n-face.pt'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8n-face.pt'),
        'yolov8n-face.pt',
    ]
    weights_path = next((os.path.abspath(p) for p in weights_candidates if os.path.isfile(p)), None)

    if weights_path is None:
        print("⚠️  未找到 yolov8n-face.pt，实时人脸检测不可用")
        yolo_face_detector = None
        return

    try:
        from api.yolo_face_detector import YOLOFaceDetector
        yolo_face_detector = YOLOFaceDetector(weights_path, device='cpu')
        print(f"✅ YOLOv8-face 初始化成功: {weights_path}")
    except Exception as e:
        print(f"⚠️  YOLOv8-face 初始化失败: {e}")
        yolo_face_detector = None


def initialize_single_image_face_detector():
    """单图人脸检测器（facenet MTCNN 优先，OpenCV Haar 回退）"""
    global single_image_face_detector, single_image_face_cascade, single_image_face_method

    single_image_face_detector = None
    single_image_face_cascade = None
    single_image_face_method = 'none'

    if FACENET_AVAILABLE and TORCH_AVAILABLE and MTCNN is not None:
        try:
            detector_device = device if device is not None else 'cpu'
            single_image_face_detector = MTCNN(keep_all=True, device=detector_device)
            single_image_face_method = 'facenet_mtcnn'
            print(f"✅ 单图人脸检测器已启用: {single_image_face_method}")
            return
        except Exception as e:
            print(f"⚠️ MTCNN 初始化失败: {e}")

    if CV2_AVAILABLE:
        try:
            single_image_face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if single_image_face_cascade is not None and not single_image_face_cascade.empty():
                single_image_face_method = 'opencv_haar'
                print(f"✅ 单图人脸检测器已启用: {single_image_face_method}")
            else:
                single_image_face_cascade = None
        except Exception as e:
            print(f"⚠️ OpenCV Haar 初始化失败: {e}")


# ================ 推理函数 ================
def detect_face_for_single_image(image):
    """单图人脸检测，返回 (face_pil, bbox_dict, method)"""
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame_h, frame_w = img_cv2.shape[:2]

    if single_image_face_detector is not None:
        try:
            rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            boxes, probs = single_image_face_detector.detect(rgb)
            if boxes is not None and len(boxes) > 0:
                valid_boxes = []
                for idx, box in enumerate(boxes):
                    if box is None:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1 = max(0, min(frame_w - 1, x1))
                    y1 = max(0, min(frame_h - 1, y1))
                    x2 = max(0, min(frame_w, x2))
                    y2 = max(0, min(frame_h, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    score = float(probs[idx]) if probs is not None and idx < len(probs) and probs[idx] is not None else 0.0
                    valid_boxes.append((score, (x2 - x1) * (y2 - y1), x1, y1, x2, y2))

                if valid_boxes:
                    _score, _area, x1, y1, x2, y2 = max(valid_boxes, key=lambda item: (item[0], item[1]))
                    pad = int(max(x2 - x1, y2 - y1) * 0.12)
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(frame_w, x2 + pad)
                    y2 = min(frame_h, y2 + pad)
                    bbox = {
                        'x': int(x1), 'y': int(y1),
                        'width': int(x2 - x1), 'height': int(y2 - y1),
                        'confidence': float(_score),
                    }
                    return image.crop((x1, y1, x2, y2)), bbox, 'facenet_mtcnn'
        except Exception as e:
            print(f"⚠️ MTCNN 检测失败，回退 OpenCV: {e}")

    if single_image_face_cascade is not None:
        try:
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            faces = single_image_face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                pad = int(max(fw, fh) * 0.12)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame_w, x + fw + pad)
                y2 = min(frame_h, y + fh + pad)
                bbox = {
                    'x': int(x1), 'y': int(y1),
                    'width': int(x2 - x1), 'height': int(y2 - y1),
                    'confidence': 1.0,
                }
                return image.crop((x1, y1, x2, y2)), bbox, 'opencv_haar'
        except Exception as e:
            print(f"⚠️ OpenCV Haar 检测失败: {e}")

    return image, None, single_image_face_method


def predict_emotion(image):
    """单张图片情绪识别（MTCNN + ResNet18）"""
    if model is None or transforms is None or not TORCH_AVAILABLE:
        face_img, face_bbox, face_method = detect_face_for_single_image(image)
        return 'neutral', 0.0, {emo: (1.0 / len(EMOTION_LABELS)) for emo in EMOTION_LABELS}, face_bbox, face_method

    try:
        face_img, face_bbox, face_method = detect_face_for_single_image(image)
        img_tensor = transform(face_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            # 后处理 logit bias 补偿（少数类置信度校准）
            bias_t = torch.tensor(LOGIT_BIAS, dtype=outputs.dtype, device=outputs.device)
            outputs = outputs + bias_t
            probs = torch.softmax(outputs, dim=1)[0]
            idx = torch.argmax(probs).item()
        return (
            EMOTION_LABELS[idx],
            float(probs[idx]),
            {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))},
            face_bbox,
            face_method,
        )
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return 'neutral', 0.0, {}, None, single_image_face_method


def predict_realtime_emotion(image_pil):
    """实时多人脸情绪识别（YOLOv8-face + ResNet18）"""
    if model is None or not TORCH_AVAILABLE or yolo_face_detector is None:
        emotion, confidence, probabilities, face_bbox, _ = predict_emotion(image_pil)
        return [{
            'bbox': [face_bbox['x'], face_bbox['y'],
                     face_bbox['x'] + face_bbox['width'],
                     face_bbox['y'] + face_bbox['height']] if face_bbox else None,
            'emotion': emotion,
            'emotion_zh': EMOTION_ZH.get(emotion, emotion),
            'confidence': confidence,
            'all_probs': probabilities,
        }]

    frame_rgb = np.array(image_pil)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    faces_info = yolo_face_detector.detect(frame_bgr, conf=0.45)
    if not faces_info:
        return []

    results = []
    for face in faces_info:
        bbox = face['bbox']
        face_pil = yolo_face_detector.crop_face(frame_bgr, bbox, pad=0.25)

        try:
            img_tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                # 后处理 logit bias 补偿（少数类置信度校准）
                bias_t = torch.tensor(LOGIT_BIAS, dtype=outputs.dtype, device=outputs.device)
                outputs = outputs + bias_t
                probs = torch.softmax(outputs, dim=1)[0]
                idx = int(torch.argmax(probs).item())
            emotion = EMOTION_LABELS[idx]
            confidence = float(probs[idx])
            all_probs = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
        except Exception as e:
            print(f"⚠️  情绪推理失败: {e}")
            emotion, confidence, all_probs = 'neutral', 0.0, {}

        results.append({
            'bbox': bbox,
            'emotion': emotion,
            'emotion_zh': EMOTION_ZH.get(emotion, emotion),
            'confidence': round(confidence, 4),
            'all_probs': all_probs,
            'yolo_conf': face['conf'],
        })

    return results


# ================ 路由 ================
@app.route('/')
def home():
    return jsonify({
        'success': True,
        'message': '表情识别 API（ResNet18 + YOLOv8-face + MTCNN）',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'GET /': 'API 信息',
            'GET /health': '健康检查',
            'GET /emotions': '支持的情绪标签',
            'POST /predict': '单张图像情绪识别（MTCNN + ResNet18）',
            'POST /predict_realtime': '实时多人脸情绪识别（YOLOv8-face + ResNet18）',
        },
        'model_loaded': model is not None,
        'yolo_available': yolo_face_detector is not None,
        'single_image_face_method': single_image_face_method,
        'device': str(device) if device else 'unknown',
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'yolo_available': yolo_face_detector is not None,
        'device': str(device) if device else 'unknown',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
    })


@app.route('/emotions', methods=['GET'])
def get_emotions():
    return jsonify({
        'success': True,
        'emotions': [{'en': emo, 'zh': EMOTION_ZH.get(emo, emo)} for emo in EMOTION_LABELS],
        'count': len(EMOTION_LABELS),
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/predict', methods=['POST'])
def predict():
    """单张图像情绪识别。支持 multipart/form-data 或 application/json (base64)。"""
    try:
        if 'image' in request.files:
            image_bytes = request.files['image'].read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif request.is_json and 'image' in request.json:
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            return jsonify({
                'success': False,
                'error': '请提供图像文件（multipart image）或 base64（json image）',
            }), 400

        emotion, confidence, probabilities, face_bbox, face_detection_method = predict_emotion(image)

        return jsonify({
            'success': True,
            'emotion': emotion,
            'emotion_zh': EMOTION_ZH.get(emotion, emotion),
            'confidence': float(confidence),
            'probabilities': probabilities,
            'face_detected': face_bbox is not None,
            'face_bbox': face_bbox,
            'face_detection_method': face_detection_method,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        print(f"❌ /predict 错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }), 500


@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    """实时多人脸情绪识别。支持 multipart/form-data 或 application/json (base64)。"""
    try:
        if 'image' in request.files:
            image_bytes = request.files['image'].read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif request.is_json and 'image' in request.json:
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            return jsonify({
                'success': False,
                'error': '请提供图像文件（multipart image）或 base64（json image）',
            }), 400

        faces = predict_realtime_emotion(image)
        return jsonify({
            'success': True,
            'faces': faces,
            'face_count': len(faces),
            'yolo_available': yolo_face_detector is not None,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        print(f"❌ /predict_realtime 错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }), 500


# ================ 启动 ================
if __name__ == '__main__':
    initialize_model()
    print("\n" + "=" * 70)
    print("🌐 API 服务器启动中...")
    print("=" * 70)
    print(f"📌 访问地址: http://0.0.0.0:7860")
    print(f"📌 健康检查: http://0.0.0.0:7860/health")
    print(f"📌 情绪列表: http://0.0.0.0:7860/emotions")
    print("=" * 70)
    app.run(host='0.0.0.0', port=7860, debug=False, use_reloader=False)
