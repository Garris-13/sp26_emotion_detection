# 表情识别 API 详细文档

## API 概述

表情识别 API 提供了基于深度学习的表情识别服务，支持识别6种基本表情。

**Base URL**: `http://localhost:5000`

**支持的表情**:
- Anger (愤怒)
- Disgust (厌恶)
- Fear (恐惧)
- Happy (快乐)
- Sad (悲伤)
- Surprised (惊讶)

---

## 端点详情

### 1. GET / - API 信息

获取 API 的基本信息和可用端点列表。

**请求示例**:
```bash
curl http://localhost:5000/
```

**响应示例**:
```json
{
    "message": "表情识别 API",
    "version": "1.0.0",
    "endpoints": {
        "/": "API 信息",
        "/predict": "POST - 预测表情",
        "/health": "GET - 健康检查",
        "/emotions": "GET - 获取支持的表情列表"
    }
}
```

---

### 2. GET /health - 健康检查

检查 API 服务器和模型的运行状态。

**请求示例**:
```bash
curl http://localhost:5000/health
```

**响应示例**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda:0"
}
```

**响应字段**:
- `status`: 服务器状态 ("healthy" 或 "unhealthy")
- `model_loaded`: 模型是否已加载 (true/false)
- `device`: 运行设备 ("cuda:0" 或 "cpu")

---

### 3. GET /emotions - 获取表情列表

获取 API 支持识别的所有表情类别。

**请求示例**:
```bash
curl http://localhost:5000/emotions
```

**响应示例**:
```json
{
    "emotions": ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprised"],
    "count": 6
}
```

---

### 4. POST /predict - 单图预测

识别单张图像中的表情。

#### 请求方式 1: 文件上传

**Content-Type**: `multipart/form-data`

**参数**:
- `image` (file, required): 图像文件

**请求示例 (cURL)**:
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/image.jpg"
```

**请求示例 (Python)**:
```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()
```

**请求示例 (JavaScript)**:
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

#### 请求方式 2: Base64 编码

**Content-Type**: `application/json`

**参数**:
- `image` (string, required): Base64 编码的图像数据

**请求示例 (Python)**:
```python
import requests
import base64

with open('image.jpg', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

headers = {'Content-Type': 'application/json'}
data = {'image': img_base64}
response = requests.post('http://localhost:5000/predict', 
                        json=data, headers=headers)
result = response.json()
```

**请求示例 (cURL)**:
```bash
IMAGE_BASE64=$(base64 -w 0 image.jpg)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_BASE64\"}"
```

#### 响应格式

**成功响应 (200 OK)**:
```json
{
    "success": true,
    "emotion": "Happy",
    "confidence": 0.8745,
    "probabilities": {
        "Anger": 0.0234,
        "Disgust": 0.0123,
        "Fear": 0.0189,
        "Happy": 0.8745,
        "Sad": 0.0456,
        "Surprised": 0.0253
    }
}
```

**错误响应 (400/500)**:
```json
{
    "success": false,
    "error": "错误描述信息"
}
```

**响应字段说明**:
- `success` (boolean): 请求是否成功
- `emotion` (string): 预测的表情类别
- `confidence` (float): 预测置信度 (0-1之间)
- `probabilities` (object): 所有类别的概率分布
- `error` (string): 错误信息（仅在失败时）

---

### 5. POST /predict_batch - 批量预测

批量识别多张图像中的表情。

**Content-Type**: `multipart/form-data`

**参数**:
- `images` (files, required): 多个图像文件

**请求示例 (cURL)**:
```bash
curl -X POST http://localhost:5000/predict_batch \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg"
```

**请求示例 (Python)**:
```python
import requests

files = [
    ('images', open('image1.jpg', 'rb')),
    ('images', open('image2.jpg', 'rb')),
    ('images', open('image3.jpg', 'rb'))
]

response = requests.post('http://localhost:5000/predict_batch', files=files)
result = response.json()

# 关闭文件
for _, f in files:
    f.close()
```

**响应格式**:
```json
{
    "success": true,
    "count": 3,
    "results": [
        {
            "emotion": "Happy",
            "confidence": 0.8745,
            "probabilities": { ... }
        },
        {
            "emotion": "Sad",
            "confidence": 0.7234,
            "probabilities": { ... }
        },
        {
            "emotion": "Anger",
            "confidence": 0.6891,
            "probabilities": { ... }
        }
    ]
}
```

---

## 错误代码

| HTTP 状态码 | 说明 |
|------------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

---

## 性能指标

### 延迟
- **单图预测**: ~50-100ms (GPU) / ~200-500ms (CPU)
- **批量预测**: ~100-300ms (GPU) / ~500-1500ms (CPU)

### 吞吐量
- **GPU**: ~20-30 图像/秒
- **CPU**: ~5-10 图像/秒

### 准确率
- **验证集准确率**: 72.38%
- **各类别F1分数**: 0.60-0.89

---

## 最佳实践

### 1. 图像质量
- **推荐分辨率**: 至少 224x224 像素
- **格式**: JPEG, PNG, BMP
- **文件大小**: < 5MB
- **人脸要求**: 清晰可见，正面或接近正面

### 2. 批量处理
- 对于多张图像，使用 `/predict_batch` 端点
- 建议每批不超过 10 张图像

### 3. 错误处理
```python
result = client.predict_from_file('image.jpg')
if result.get('success'):
    emotion = result['emotion']
    confidence = result['confidence']
    print(f"表情: {emotion}, 置信度: {confidence:.2%}")
else:
    print(f"预测失败: {result.get('error')}")
```

### 4. 置信度阈值
- 建议设置置信度阈值 (如 0.6)
- 低于阈值的结果可能不可靠

```python
threshold = 0.6
if result['confidence'] < threshold:
    print("置信度过低，结果可能不准确")
```

---

## 限制说明

### 1. 输入限制
- 单个图像文件大小: < 10MB
- 批量预测: 每次最多 20 张图像
- 支持的图像格式: JPEG, PNG, BMP, GIF

### 2. 识别限制
- 仅支持 6 种基本表情
- 需要清晰的人脸图像
- 最佳效果: 正面人脸，良好光照

### 3. 性能限制
- 并发请求: 取决于服务器配置
- 速率限制: 默认无限制（可配置）

---

## 安全建议

### 1. 生产环境部署
- 使用 HTTPS
- 添加 API 密钥认证
- 限制请求速率
- 添加请求日志

### 2. 输入验证
- 验证文件类型
- 限制文件大小
- 检查图像内容

### 3. 资源保护
- 设置超时时间
- 限制并发请求
- 监控资源使用

---

## 集成示例

### Flask 应用集成
```python
from flask import Flask, request
import requests

app = Flask(__name__)
API_URL = 'http://localhost:5000'

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    
    files = {'image': request.files['image']}
    response = requests.post(f'{API_URL}/predict', files=files)
    
    return response.json()
```

### Django 应用集成
```python
from django.views import View
from django.http import JsonResponse
import requests

class EmotionAnalysisView(View):
    API_URL = 'http://localhost:5000'
    
    def post(self, request):
        image = request.FILES.get('image')
        if not image:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        files = {'image': image}
        response = requests.post(f'{self.API_URL}/predict', files=files)
        
        return JsonResponse(response.json())
```

### React 前端集成
```javascript
import React, { useState } from 'react';

function EmotionRecognition() {
    const [result, setResult] = useState(null);
    
    const handleFileChange = async (event) => {
        const file = event.target.files[0];
        const formData = new FormData();
        formData.append('image', file);
        
        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Error:', error);
        }
    };
    
    return (
        <div>
            <input type="file" onChange={handleFileChange} />
            {result && result.success && (
                <div>
                    <h3>表情: {result.emotion}</h3>
                    <p>置信度: {(result.confidence * 100).toFixed(2)}%</p>
                </div>
            )}
        </div>
    );
}
```

---

## 版本历史

### v1.0.0 (2025-10-23)
- 初始版本发布
- 支持 6 种表情识别
- 提供单图和批量预测接口

---

## 技术支持

如有问题或建议，请联系技术支持团队。
