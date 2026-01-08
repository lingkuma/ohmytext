import keyboard
import pyautogui
from PIL import Image
import io
import os
import pyperclip
import google.generativeai as genai
from dotenv import load_dotenv
import threading
import queue

load_dotenv()

def capture_screenshot_around_mouse(width=1400, height=800):
    mouse_x, mouse_y = pyautogui.position()
    
    left = mouse_x - width // 2
    top = mouse_y - height // 2
    
    screen_width, screen_height = pyautogui.size()
    
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + width > screen_width:
        left = screen_width - width
    if top + height > screen_height:
        top = screen_height - height
    
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    
    return screenshot, (left, top)


def ocr_with_gemini(image, api_key, timeout=30):
    genai.configure(api_key=api_key)
    
    #model = genai.GenerativeModel('gemini-2.0-flash')
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    #prompt = "OCR图中德语，不同的句子（图中一般都是不同的对话框或区域）之间用换行隔开，每个句子最后整理成无换行的文本。忽略其他非德语文本。"
    prompt = """
    获取图片中心主要德语文本段落的OCR，主要是中心（或距离鼠标最近）的那个文本框，注意区分不同的文本框。
    只需要返回中心的或距离鼠标最近的文本框，不需要处理其他文本框，最后整理成无换行的文本，并添加适当标点符号。;
    
    """
#忽略其他所有文本，忽略其他非德语文本。
    result_queue = queue.Queue()
    
    def ocr_worker():
        try:
            response = model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": img_byte_arr}
            ])
            result_queue.put(("success", response.text.strip()))
        except Exception as e:
            result_queue.put(("error", str(e)))
    
    thread = threading.Thread(target=ocr_worker)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        print(f"OCR识别超时（{timeout}秒），已取消请求")
        return None
    
    try:
        status, result = result_queue.get_nowait()
        if status == "success":
            return result
        else:
            print(f"Gemini OCR 错误: {result}")
            return None
    except queue.Empty:
        print("OCR识别失败：未获取到结果")
        return None


def on_f4_pressed():
    print("\n=== F4 按下，开始处理 ===")
    
    mouse_x, mouse_y = pyautogui.position()
    print(f"鼠标位置: ({mouse_x}, {mouse_y})")
    
    screenshot, offset = capture_screenshot_around_mouse(1400, 800)
    print(f"截图偏移: {offset}")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("错误: 未找到 GEMINI_API_KEY 环境变量")
        return
    
    timeout_str = os.getenv('OCR_TIMEOUT', '30')
    try:
        timeout = int(timeout_str)
    except ValueError:
        timeout = 30
    
    print(f"正在进行OCR识别（超时设置: {timeout}秒）...")
    ocr_text = ocr_with_gemini(screenshot, api_key, timeout)
    
    if ocr_text:
        print(f"\n=== 识别结果 ===")
        print(f"文本: {ocr_text}")
        
        pyperclip.copy(ocr_text)
        print(f"\n文本已写入剪切板")
    else:
        print("\nOCR识别失败")


def main():
    print("程序已启动，按F4键进行截图识别...")
    print("按Ctrl+C退出程序")
    
    try:
        keyboard.add_hotkey('f4', on_f4_pressed)
        keyboard.wait('esc')
    except KeyboardInterrupt:
        print("\n程序已退出")


if __name__ == "__main__":
    main()
