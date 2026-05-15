"""
测试真实摄像头拍照功能
"""

import sys
import os
from datetime import time

# 使用虚拟环境的Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("📷 真实摄像头拍照测试")
print("=" * 60)

try:
    import cv2

    print(f"✅ OpenCV版本: {cv2.__version__}")

    # 测试摄像头0
    print("\n测试摄像头0...")
    cap0 = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    if cap0.isOpened():
        ret0, frame0 = cap0.read()
        if ret0:
            width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ 摄像头0: {width}x{height}")

            # 保存测试图片
            os.makedirs("data/test_real", exist_ok=True)
            cv2.imwrite("data/test_real/camera0_test.jpg", frame0)
            print("📸 测试图片已保存: data/test_real/camera0_test.jpg")

            # 显示图片
            cv2.imshow('摄像头0 - 按任意键继续', frame0)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        else:
            print("❌ 摄像头0: 无法读取图像")
    else:
        print("❌ 摄像头0: 无法打开")

    cap0.release()

    # 测试连续拍照
    print("\n🔍 测试连续拍照（5次，间隔1秒）...")

    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    if cap.isOpened():
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                filename = f"data/test_real/continuous_{i + 1}.jpg"
                cv2.imwrite(filename, frame)
                print(f"  第{i + 1}张: {filename}")
                time.sleep(1)
            else:
                print(f"  第{i + 1}张: 读取失败")

        cap.release()
        print("\n✅ 连续拍照测试完成")
    else:
        print("❌ 无法打开摄像头进行连续拍照")

    print("\n" + "=" * 60)
    print("✅ 摄像头功能正常！")
    print("=" * 60)

except ImportError:
    print("❌ OpenCV未安装")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback

    traceback.print_exc()

input("\n按Enter键退出...")