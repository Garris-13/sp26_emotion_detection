"""
测试 OpenCV 安装和摄像头功能
"""

import sys
import os

print("=" * 60)
print("🔍 OpenCV 安装测试")
print("=" * 60)

# 1. 检查 Python 环境
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")
print(f"当前目录: {os.getcwd()}")

# 2. 尝试导入 OpenCV
try:
    import cv2

    print(f"✅ OpenCV 版本: {cv2.__version__}")
    print(f"OpenCV 路径: {cv2.__file__}")

    # 3. 检查摄像头功能
    print("\n📷 测试摄像头连接...")

    # 尝试不同的后端
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Media Foundation (Windows)"),
        (cv2.CAP_ANY, "Auto"),
    ]

    for backend_code, backend_name in backends:
        print(f"\n尝试 {backend_name}...")

        for camera_index in range(4):  # 测试0-3号摄像头
            try:
                # 组合后端和摄像头索引
                cap = cv2.VideoCapture(camera_index + backend_code)

                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"  ✅ 摄像头 {camera_index}: {width}x{height}")

                        # 保存测试图像
                        test_dir = "data/camera_test"
                        os.makedirs(test_dir, exist_ok=True)
                        test_path = os.path.join(test_dir, f"cam{camera_index}_{backend_name}.jpg")
                        cv2.imwrite(test_path, frame)
                        print(f"    测试图像: {test_path}")

                        # 显示预览（短暂显示）
                        cv2.imshow(f'Camera {camera_index}', frame)
                        cv2.waitKey(300)
                        cv2.destroyAllWindows()
                    else:
                        print(f"  ⚠️  摄像头 {camera_index}: 已打开但无法读取")
                else:
                    print(f"  ❌ 摄像头 {camera_index}: 不可用")

                cap.release()

            except Exception as e:
                print(f"  ❌ 摄像头 {camera_index} 测试失败: {str(e)[:50]}")

    print("\n" + "=" * 60)
    print("✅ OpenCV 功能正常")

except ImportError as e:
    print(f"❌ 无法导入 OpenCV: {e}")
    print("\n💡 解决方案:")
    print("  1. 确认在正确的虚拟环境中: .venv1")
    print("  2. 尝试重新安装: pip uninstall opencv-python && pip install opencv-python")
    print("  3. 尝试安装 headless 版本: pip install opencv-python-headless")
    print("  4. 检查 Python 路径是否与 pip 安装路径一致")

except Exception as e:
    print(f"❌ OpenCV 测试异常: {e}")
    import traceback

    traceback.print_exc()

print("=" * 60)
input("按 Enter 键退出...")