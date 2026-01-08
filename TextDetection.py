import keyboard
import pyautogui
from PIL import Image
import time
import os
from paddleocr import TextDetection

model = TextDetection(model_name="PP-OCRv5_server_det")

def capture_full_screen():
    """
    全屏截图
    
    返回:
        PIL Image对象
    """
    screenshot = pyautogui.screenshot()
    return screenshot


def on_f4_pressed():
    """
    F4键按下时的处理函数
    """
    print("\n=== F4 按下，开始处理 ===")
    
    # 全屏截图
    screenshot = capture_full_screen()
    
    # 生成带时间戳的文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_path = f"./output/screenshot_{timestamp}.png"
    
    # 确保output目录存在
    os.makedirs("./output", exist_ok=True)
    
    # 保存截图
    screenshot.save(screenshot_path)
    print(f"截图已保存: {screenshot_path}")
    
    # 文本检测
    print("正在进行文本检测...")
    output = model.predict(screenshot_path, batch_size=1)
    
    # 保存检测结果
    for i, res in enumerate(output):
        res.print()
        
        # 保存带文本区域的图片
        res.save_to_img(save_path="./output/")
        
        # 保存JSON结果
        json_path = f"./output/detection_{timestamp}.json"
        res.save_to_json(save_path=json_path)
        print(f"检测结果已保存到: {json_path}")


def main():
    """
    主函数，监听F4快捷键
    """
    print("程序已启动，按F4键进行全屏截图并检测文本区域...")
    print("按ESC键退出程序")
    
    try:
        keyboard.add_hotkey('f4', on_f4_pressed)
        keyboard.wait('esc')
    except KeyboardInterrupt:
        print("\n程序已退出")


if __name__ == "__main__":
    main()
