import keyboard
import pyautogui
from PIL import Image, ImageDraw
import time
import os
import numpy as np
import pyperclip
import base64
import requests
import json
import io
from paddleocr import TextDetection
import google.generativeai as genai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from openai import OpenAI

load_dotenv()

model = TextDetection(model_name="PP-OCRv5_server_det")

save_debug_images = False
STATUS_BAR_HEIGHT = 71

# 是否发送裁剪后的截图到网页
SEND_CROPPED_SCREENSHOT = True
# 是否开启全局AI修正
ENABLE_AI_CORRECTION = False
# 开启全局AI修正后，激活的最小文本长度
AI_MIN_TEXT_LENGTH = 10

# 是否开启智能AI修正，修正距离鼠标最新的SMART_AI_SELECTION_COUNT个文本框
SMART_AI_SELECTION_MODE = True
# 智能AI修正模式下，选择的最近的文本框数量
SMART_AI_SELECTION_COUNT = 5

# AI相关配置
# 配置Gemini API密钥
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# OCR超时时间
OCR_TIMEOUT = int(os.getenv('OCR_TIMEOUT', 10))
# AI超时时间
AI_TIMEOUT = 14
# AI校验prompt
AI_CORRECTION_PROMPT = """下面收到的文本是用户通过OCR获取的德语句子，请按照一下要求，进行验证和清理：
1. ocr可能会丢失öü，或将ß识别成B，请将错误的德语单词纠正；
2. 格式的换行，请保持换行符，比如第一行是用户名，第二行是用户的推文正文
3. 句子的换行，请不要换行，比如第二行是推文正文，虽然ocr视觉上是多行的，但是你识别后就不用换行，
4. 可能含有不属于句子内的干扰单词、符号、网名等不是德语单词的拉丁单词，请你删除之后返回完整的德语句子。 
5. 记得只返回清理后的句子，不许说其他废话
原始文本：{text}"""

# AI提供商，可选值：gemini, openai
# AI 服务提供商，支持 gemini / openai，默认优先使用 gemini
AI_PROVIDER = os.getenv('AI_PROVIDER', 'gemini')
# OpenAI 接口的基础 URL，默认官方地址，可替换为第三方代理
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
# OpenAI API 密钥，用于调用 GPT 系列模型
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# 指定使用的 OpenAI 模型，默认 gpt-4o-mini 以平衡速度与质量
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
# 生成文本时的采样温度，值越小结果越确定，默认 0.7 保持适度多样性
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))

ai_correction_count = 0
ai_correction_completed = 0
ai_correction_lock = threading.Lock()


def image_to_base64(image_path_or_pil):
    if isinstance(image_path_or_pil, str):
        with open(image_path_or_pil, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        img_byte_arr = io.BytesIO()
        image_path_or_pil.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')


def call_luna_ocr_api(image_path_or_pil):
    api_url = 'http://127.0.0.1:2333/api/ocr'
    
    try:
        base64_image = image_to_base64(image_path_or_pil)
        
        payload = {
            'image': base64_image
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(api_url, data=json.dumps(payload), headers=headers, timeout=OCR_TIMEOUT)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f'API Error: Status Code {response.status_code}')
            return None
        
    except requests.exceptions.Timeout:
        print(f'API Timeout: 请求超时（{OCR_TIMEOUT}秒）')
        return None
    except requests.exceptions.RequestException as e:
        print(f'Request Error: {e}')
        return None
    except Exception as e:
        print(f'Error: {e}')
        return None


def init_gemini():
    """
    初始化Gemini AI模型
    
    返回:
        Gemini模型实例或None（如果初始化失败）
    """
    if not ENABLE_AI_CORRECTION and not SMART_AI_SELECTION_MODE:
        return None
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("Gemini AI模型初始化成功")
        return model
    except Exception as e:
        print(f"Gemini AI初始化失败: {e}")
        return None


def init_openai():
    """
    初始化OpenAI客户端
    
    返回:
        OpenAI客户端实例或None（如果初始化失败）
    """
    if not ENABLE_AI_CORRECTION and not SMART_AI_SELECTION_MODE:
        return None
    
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        print(f"OpenAI客户端初始化成功 (base_url: {OPENAI_BASE_URL}, model: {OPENAI_MODEL}, temperature: {OPENAI_TEMPERATURE})")
        return client
    except Exception as e:
        print(f"OpenAI初始化失败: {e}")
        return None


def correct_text_with_openai(text, openai_client, prompt=None, force_correction=False):
    """
    使用OpenAI校验和修正OCR识别的文本
    
    参数:
        text: OCR识别的原始文本
        openai_client: OpenAI客户端实例
        prompt: 自定义prompt，如果为None则使用默认的AI_CORRECTION_PROMPT
        force_correction: 是否强制进行AI纠正（用于智能AI选择模式）
    
    返回:
        校正后的文本，如果校验失败则返回原始文本
    """
    if not force_correction and (not ENABLE_AI_CORRECTION or not openai_client):
        return text
    
    if len(text.strip()) < AI_MIN_TEXT_LENGTH:
        return text
    
    if not text.strip():
        return text
    
    try:
        if prompt is None:
            prompt = AI_CORRECTION_PROMPT.format(text=text)
        elif '{text}' in prompt:
            prompt = prompt.format(text=text)
        
        print(f"发送给OpenAI的Prompt: {prompt}")
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=OPENAI_TEMPERATURE,
            timeout=AI_TIMEOUT
        )
        
        print(f"OpenAI原始响应: {response}")
        if response and response.choices and len(response.choices) > 0:
            corrected_text = response.choices[0].message.content
            print(f"OpenAI返回的文本: '{corrected_text}'")
            
            if corrected_text and corrected_text != text:
                print(f"OpenAI校验: '{text}' -> '{corrected_text}'")
                return corrected_text
            else:
                print(f"OpenAI校验: 文本无需修正 '{text}'")
                return text
        else:
            print(f"OpenAI校验: 未返回有效结果")
            return text
            
    except Exception as e:
        print(f"OpenAI校验失败: {e}")
        return text


gemini_model = None
openai_client = None

if AI_PROVIDER.lower() == 'openai':
    openai_client = init_openai()
    if openai_client:
        ai_model = openai_client
        print(f"已选择OpenAI作为AI提供商")
    else:
        print("OpenAI初始化失败，尝试使用Gemini...")
        gemini_model = init_gemini()
        ai_model = gemini_model
else:
    gemini_model = init_gemini()
    if gemini_model:
        ai_model = gemini_model
        print(f"已选择Gemini作为AI提供商")
    else:
        print("Gemini初始化失败，尝试使用OpenAI...")
        openai_client = init_openai()
        ai_model = openai_client


def detect_columns(items, x_thresh=10):
    """
    基于 x 坐标聚类检测列
    
    参数:
        items: 文本块列表，每个包含 box [x1, y1, x2, y2]
        x_thresh: 列聚类阈值，单位像素
    
    返回:
        列列表，每列包含该列的所有文本块
    """
    items_sorted = sorted(items, key=lambda x: x["box"][0])
    
    columns = []
    for it in items_sorted:
        placed = False
        for col in columns:
            if abs(it["box"][0] - col["x1_mean"]) < x_thresh:
                col["items"].append(it)
                col["x1_mean"] = np.mean([p["box"][0] for p in col["items"]])
                placed = True
                break
        if not placed:
            columns.append({"x1_mean": it["box"][0], "items": [it]})
    
    columns.sort(key=lambda x: x["x1_mean"])
    
    for col in columns:
        col["items"].sort(key=lambda x: x["box"][1])
    
    print("\n=== 列检测结果 ===")
    for i, col in enumerate(columns):
        print(f"\n第 {i+1} 列:")
        print(f"  平均 x1: {col['x1_mean']:.2f}")
        print(f"  文本块数量: {len(col['items'])}")
        for item in col["items"]:
            print(f"    - 坐标: x1={item['box'][0]:.1f}, y1={item['box'][1]:.1f}, x2={item['box'][2]:.1f}, y2={item['box'][3]:.1f}")
    
    return columns


def sort_ocr_results(boxes, y_thresh=15, x_thresh=50):
    """
    boxes: np.ndarray shape [N, 4]  => [x_min, y_min, x_max, y_max]
    y_thresh: 行分组阈值，单位像素，适当调大调小
    x_thresh: 列分组阈值，单位像素，用于检测多列布局
    """
    items = []
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        items.append({
            "box": [x1, y1, x2, y2],
            "cx": cx,
            "cy": cy,
            "y1": y1
        })

    columns = detect_columns(items, x_thresh=x_thresh)
    
    all_merged_paragraphs = []
    for col in columns:
        col_items = col["items"]
        
        col_items.sort(key=lambda x: x["y1"])

        lines = []
        for it in col_items:
            placed = False
            for line in lines:
                if abs(it["cy"] - line["cy_mean"]) < y_thresh:
                    line["items"].append(it)
                    line["cy_mean"] = np.mean([p["cy"] for p in line["items"]])
                    placed = True
                    break
            if not placed:
                lines.append({"cy_mean": it["cy"], "items": [it]})

        for line in lines:
            line["items"].sort(key=lambda x: x["cx"])

        lines.sort(key=lambda x: x["cy_mean"])

        col_sorted_items = []
        for line in lines:
            col_sorted_items.extend(line["items"])
        
        col_merged_paragraphs = merge_paragraphs(col_sorted_items, gap_coefficient=1.2)
        all_merged_paragraphs.extend(col_merged_paragraphs)

    return all_merged_paragraphs


def merge_paragraphs(sorted_items, gap_coefficient=1.2, height_consistency_check=True, horizontal_alignment_check=True):
    """
    智能段落合并算法，基于相对几何关系和统计特征
    
    参数:
        sorted_items: 已排序的OCR结果列表（按阅读顺序）
        gap_coefficient: 间距系数，默认1.2，表示间距小于平均行高的1.2倍时合并
        height_consistency_check: 是否启用字体高度一致性检查
        horizontal_alignment_check: 是否启用水平对齐检查
    
    返回:
        合并后的段落列表，每个段落包含:
        - box: 合并后的外接矩形 [x1, y1, x2, y2]
        - children: 构成该段落的原始文本块列表
    """
    if not sorted_items:
        return []
    
    for item in sorted_items:
        x1, y1, x2, y2 = item["box"]
        item["height"] = y2 - y1
        item["width"] = x2 - x1
        item["center_x"] = (x1 + x2) / 2
        item["center_y"] = (y1 + y2) / 2
    
    results = []
    first_item = sorted_items[0]
    current_block = {
        "box": first_item["box"].copy(),
        "children": [first_item],
        "height": first_item["height"],
        "width": first_item["width"]
    }
    
    for i in range(1, len(sorted_items)):
        item = sorted_items[i]
        
        current_y2 = current_block["box"][3]
        item_y1 = item["box"][1]
        gap = item_y1 - current_y2
        
        avg_height = (current_block["height"] + item["height"]) / 2
        
        should_merge = False
        
        if gap < avg_height * gap_coefficient:
            horizontal_overlap = True
            if horizontal_alignment_check:
                current_x1, _, current_x2, _ = current_block["box"]
                item_x1, _, item_x2, _ = item["box"]
                
                overlap = max(0, min(current_x2, item_x2) - max(current_x1, item_x1))
                min_width = min(current_block["width"], item["width"])
                
                if min_width > 0 and overlap / min_width < 0.2:
                    horizontal_overlap = False
            
            height_consistent = True
            if height_consistency_check:
                height_diff = abs(current_block["height"] - item["height"])
                max_height = max(current_block["height"], item["height"])
                if max_height > 0 and height_diff / max_height > 0.5:
                    height_consistent = False
            
            if horizontal_overlap and height_consistent:
                should_merge = True
        
        if should_merge:
            current_box = current_block["box"]
            item_box = item["box"]
            
            new_x1 = min(current_box[0], item_box[0])
            new_y1 = min(current_box[1], item_box[1])
            new_x2 = max(current_box[2], item_box[2])
            new_y2 = max(current_box[3], item_box[3])
            
            current_block["box"] = [new_x1, new_y1, new_x2, new_y2]
            current_block["children"].append(item)
            current_block["height"] = max(current_block["height"], item["height"])
            current_block["width"] = new_x2 - new_x1
        else:
            results.append(current_block)
            current_block = {
                "box": item["box"].copy(),
                "children": [item],
                "height": item["height"],
                "width": item["width"]
            }
    
    results.append(current_block)
    
    return results


def polys_to_boxes(dt_polys):
    """
    将多边形坐标转换为矩形框坐标
    
    参数:
        dt_polys: 多边形坐标数组，形状为 [N, 4, 2]
    
    返回:
        boxes: 矩形框坐标数组，形状为 [N, 4]，每个为 [x1, y1, x2, y2]
    """
    boxes = []
    for poly in dt_polys:
        x_coords = poly[:, 0]
        y_coords = poly[:, 1]
        x1 = int(x_coords.min())
        y1 = int(y_coords.min())
        x2 = int(x_coords.max())
        y2 = int(y_coords.max())
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes)


def draw_merged_paragraphs(image_path, merged_paragraphs, output_path):
    """
    在图片上绘制合并后的段落方框
    
    参数:
        image_path: 原始图片路径
        merged_paragraphs: 合并后的段落列表
        output_path: 输出图片路径
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    colors = [
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 蓝色
        (255, 255, 0),    # 黄色
        (255, 0, 255),    # 洋红色
        (0, 255, 255),    # 青色
        (255, 128, 0),    # 橙色
        (128, 0, 255),    # 紫色
        (0, 128, 255),    # 天蓝色
        (255, 0, 128),    # 粉红色
    ]
    
    for i, para in enumerate(merged_paragraphs):
        x1, y1, x2, y2 = para["box"]
        color = colors[i % len(colors)]
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        print(f"段落 {i+1}:")
        print(f"  坐标: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        print(f"  颜色: {color}")
    
    image.save(output_path)
    print(f"\n合并后的段落图片已保存到: {output_path}")


def draw_ocr_text_on_merged_image(image_path, merged_paragraphs, output_path):
    """
    在合并后的段落图片上绘制OCR识别的文本（红色字体）
    
    参数:
        image_path: 原始图片路径
        merged_paragraphs: 合并后的段落列表（包含OCR识别的文本）
        output_path: 输出图片路径
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    colors = [
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 蓝色
        (255, 255, 0),    # 黄色
        (255, 0, 255),    # 洋红色
        (0, 255, 255),    # 青色
        (255, 128, 0),    # 橙色
        (128, 0, 255),    # 紫色
        (0, 128, 255),    # 天蓝色
        (255, 0, 128),    # 粉红色
    ]
    
    print("\n" + "=" * 80)
    print("在合并后的段落图片上绘制OCR文本...")
    print("=" * 80)
    
    for i, para in enumerate(merged_paragraphs):
        x1, y1, x2, y2 = para["box"]
        color = colors[i % len(colors)]
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        if 'text' in para and para['text']:
            text = para['text']
            
            try:
                from PIL import ImageFont
                
                font_size = max(12, int((y2 - y1) * 0.8))
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                box_width = x2 - x1
                box_height = y2 - y1
                
                if text_width > box_width:
                    font_size = int(font_size * (box_width / text_width))
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                
                text_x = x1 + 5
                text_y = y1 + (box_height - text_height) // 2
                
                draw.text((text_x, text_y), text, fill=(255, 0, 0), font=font)
                
                print(f"段落 {i+1}: 绘制文本 '{text}'")
            except Exception as e:
                print(f"段落 {i+1}: 绘制文本失败 - {e}")
                draw.text((x1 + 5, y1 + 5), para['text'], fill=(255, 0, 0))
        else:
            print(f"段落 {i+1}: 无OCR文本")
    
    image.save(output_path)
    print(f"\n带OCR文本的合并段落图片已保存到: {output_path}")


def send_ocr_to_server(merged_paragraphs):
    """
    将OCR数据发送到Node.js服务器

    参数:
        merged_paragraphs: 合并后的段落列表（包含OCR识别的文本）
    """
    text_data = []

    for para in merged_paragraphs:
        x1, y1, x2, y2 = para["box"]

        adjusted_y = y1 - STATUS_BAR_HEIGHT

        text = para.get('text', '')
        
        children = para.get('children', [])
        is_merged = len(children) > 1
        merged_lines = len(children)

        text_data.append({
            'x': x1,
            'y': adjusted_y,
            'width': x2 - x1,
            'height': y2 - y1,
            'text': text,
            'is_merged': is_merged,
            'merged_lines': merged_lines
        })

    api_url = 'http://127.0.0.1:2334/api/update-ocr'

    try:
        print(f'[DEBUG] 准备发送OCR数据到: {api_url}')
        print(f'[DEBUG] 数据条数: {len(text_data)}')
        print(f'[DEBUG] 请求方法: POST')
        
        response = requests.post(api_url, json=text_data, headers={'Content-Type': 'application/json'})
        
        print(f'[DEBUG] 响应状态码: {response.status_code}')
        print(f'[DEBUG] 响应内容: {response.text[:200] if response.text else "无内容"}')

        if response.status_code == 200:
            result = response.json()
            print(f"OCR数据已成功发送到服务器，共 {len(text_data)} 个文本区域")
            return True
        else:
            print(f"发送OCR数据失败: HTTP {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"发送OCR数据时出错: {e}")
        return False


def send_ai_correction_to_server(para):
    """
    将AI纠错后的单个段落数据发送到Node.js服务器

    参数:
        para: 包含AI纠错文本的段落字典
    """
    x1, y1, x2, y2 = para["box"]

    adjusted_y = y1 - STATUS_BAR_HEIGHT

    text = para.get('corrected_text', para.get('text', ''))
    original_text = para.get('text', '')
    
    x1, y1, x2, y2 = para["box"]
    para_info = f"段落[({int(x1)},{int(y1)})]"
    
    children = para.get('children', [])
    is_merged = len(children) > 1
    merged_lines = len(children)

    text_data = {
        'x': x1,
        'y': adjusted_y,
        'width': x2 - x1,
        'height': y2 - y1,
        'text': text,
        'is_corrected': True,
        'is_merged': is_merged,
        'merged_lines': merged_lines
    }

    api_url = 'http://127.0.0.1:2334/api/update-ocr-correction'

    try:
        response = requests.post(api_url, json=text_data, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            result = response.json()
            if text != original_text:
                print(f"{para_info} 发送AI纠错数据: '{original_text}' -> '{text}'")
            else:
                print(f"{para_info} 发送AI纠错数据: '{text}' (无修正)")
            return True
        else:
            print(f"{para_info} 发送AI纠错数据失败: HTTP {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"{para_info} 发送AI纠错数据时出错: {e}")
        return False


def send_cropped_screenshot_to_server(screenshot):
    """
    将裁剪后的截图发送到Node.js服务器作为背景图片

    参数:
        screenshot: PIL Image 对象（全屏截图）
    """
    if not SEND_CROPPED_SCREENSHOT:
        return False

    width, height = screenshot.size

    cropped_screenshot = screenshot.crop((0, STATUS_BAR_HEIGHT, width, height))

    try:
        img_byte_arr = io.BytesIO()
        cropped_screenshot.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        payload = {
            'image': base64_image
        }

        headers = {
            'Content-Type': 'application/json'
        }

        api_url = 'http://127.0.0.1:2334/api/update-screenshot'

        print(f'[DEBUG] 准备发送截图到: {api_url}')
        print(f'[DEBUG] 图片数据长度: {len(base64_image)}')
        print(f'[DEBUG] 请求方法: POST')

        response = requests.post(api_url, data=json.dumps(payload), headers=headers, timeout=10)
        
        print(f'[DEBUG] 响应状态码: {response.status_code}')
        print(f'[DEBUG] 响应内容: {response.text[:200] if response.text else "无内容"}')

        if response.status_code == 200:
            result = response.json()
            print(f"裁剪后的截图已成功发送到服务器")
            return True
        else:
            print(f"发送裁剪后的截图失败: HTTP {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print(f'发送裁剪后的截图超时')
        return False
    except requests.exceptions.RequestException as e:
        print(f'发送裁剪后的截图时出错: {e}')
        return False
    except Exception as e:
        print(f'发送裁剪后的截图时发生错误: {e}')
        return False


def capture_full_screen():
    """
    全屏截图
    
    返回:
        PIL Image对象
    """
    screenshot = pyautogui.screenshot()
    return screenshot


def recognize_merged_paragraphs_concurrent(image_path_or_pil, merged_paragraphs):
    """
    对合并后的段落进行并发 OCR 识别（不包含AI校验）
    
    参数:
        image_path_or_pil: 原始图片路径或 PIL Image 对象
        merged_paragraphs: 合并后的段落列表
    
    返回:
        识别后的段落列表，每个段落添加了 'text' 字段
    """
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil)
    else:
        image = image_path_or_pil
    
    print("\n" + "=" * 80)
    print("开始对合并后的段落进行并发 OCR 识别...")
    print("=" * 80)
    
    def recognize_single_paragraph(para, image):
        x1, y1, x2, y2 = para["box"]
        cropped_image = image.crop((x1, y1, x2, y2))
        
        try:
            result = call_luna_ocr_api(cropped_image)
            
            if result and 'text' in result:
                text = result['text']
                para['text'] = text
            else:
                para['text'] = ""
        except Exception as e:
            print(f"段落识别失败: {e}")
            para['text'] = ""
        
        return para
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_para = {
            executor.submit(recognize_single_paragraph, para.copy(), image): para
            for para in merged_paragraphs
        }
        
        for future in as_completed(future_to_para):
            para = future_to_para[future]
            try:
                result_para = future.result()
                for i, original_para in enumerate(merged_paragraphs):
                    if original_para["box"] == result_para["box"]:
                        merged_paragraphs[i] = result_para
                        break
            except Exception as e:
                print(f"段落处理异常: {e}")
    
    print(f"\n并发OCR识别完成，共处理 {len(merged_paragraphs)} 个段落")
    
    print("\n" + "=" * 80)
    print("OCR识别结果详情：")
    print("=" * 80)
    for i, para in enumerate(merged_paragraphs):
        text = para.get('text', '')
        print(f"段落 {i+1}: '{text}' (长度: {len(text)})")
    print("=" * 80 + "\n")
    
    return merged_paragraphs


def correct_text_with_ai_async(para, ai_model, callback, force_correction=False):
    """
    异步使用AI校验和修正OCR识别的文本
    
    参数:
        para: 包含OCR文本的段落字典
        ai_model: Gemini AI模型实例或OpenAI客户端实例
        callback: 完成后的回调函数，接收修正后的段落
        force_correction: 是否强制进行AI纠正（用于智能AI选择模式）
    """
    global ai_correction_count, ai_correction_completed
    
    def _correct():
        global ai_correction_completed
        
        text = para.get('text', '')
        text_stripped = text.strip()
        text_length = len(text_stripped)
        
        x1, y1, x2, y2 = para["box"]
        para_info = f"段落[({int(x1)},{int(y1)})]"
        
        if not force_correction and (not ENABLE_AI_CORRECTION or not ai_model):
            print(f"{para_info} AI校验已禁用，跳过: '{text}'")
            with ai_correction_lock:
                ai_correction_completed += 1
                if ai_correction_completed >= ai_correction_count:
                    print("\n" + "=" * 80)
                    print("全屏OCR识别完成！")
                    print("OCR数据已发送到服务器，浏览器页面会自动更新")
                    print("AI纠错已全部完成，已修正的文本会显示为绿色")
                    print("按F4键可以重新识别并更新页面")
                    print("=" * 80 + "\n")
            return
        
        if not text_stripped:
            print(f"{para_info} 文本为空，跳过AI校验")
            with ai_correction_lock:
                ai_correction_completed += 1
                if ai_correction_completed >= ai_correction_count:
                    print("\n" + "=" * 80)
                    print("全屏OCR识别完成！")
                    print("OCR数据已发送到服务器，浏览器页面会自动更新")
                    print("AI纠错已全部完成，已修正的文本会显示为绿色")
                    print("按F4键可以重新识别并更新页面")
                    print("=" * 80 + "\n")
            return
        
        if text_length < AI_MIN_TEXT_LENGTH:
            print(f"{para_info} 文本长度({text_length}) < 最小长度({AI_MIN_TEXT_LENGTH})，跳过AI校验: '{text}'")
            with ai_correction_lock:
                ai_correction_completed += 1
                if ai_correction_completed >= ai_correction_count:
                    print("\n" + "=" * 80)
                    print("全屏OCR识别完成！")
                    print("OCR数据已发送到服务器，浏览器页面会自动更新")
                    print("AI纠错已全部完成，已修正的文本会显示为绿色")
                    print("按F4键可以重新识别并更新页面")
                    print("=" * 80 + "\n")
            return
        
        print(f"{para_info} 文本长度({text_length}) >= 最小长度({AI_MIN_TEXT_LENGTH})，开始AI校验: '{text}'")
        print(f"{para_info} AI_PROVIDER: {AI_PROVIDER}, openai_client: {openai_client is not None}, ai_model: {ai_model is not None}")
        
        try:
            if AI_PROVIDER.lower() == 'openai' and openai_client:
                print(f"{para_info} 使用OpenAI进行校验")
                corrected_text = correct_text_with_openai(text, openai_client, force_correction=force_correction)
            else:
                print(f"{para_info} 使用Gemini进行校验 (AI_PROVIDER={AI_PROVIDER}, openai_client={openai_client is not None})")
                prompt = AI_CORRECTION_PROMPT.format(text=text)
                print(f"{para_info} 发送给Gemini的Prompt: {prompt}")
                response = ai_model.generate_content(prompt, request_options={'timeout': AI_TIMEOUT})
                
                print(f"{para_info} Gemini原始响应: {response}")
                if response and hasattr(response, 'text'):
                    print(f"{para_info} Gemini返回的文本: '{response.text}'")
                    corrected_text = response.text
                else:
                    print(f"{para_info} Gemini未返回有效文本")
                    corrected_text = text
            
            if corrected_text and corrected_text != text:
                print(f"{para_info} AI校验成功: '{text}' -> '{corrected_text}'")
                para['corrected_text'] = corrected_text
                callback(para)
            else:
                print(f"{para_info} AI校验: 文本无需修正 '{text}'")
                para['corrected_text'] = text
                callback(para)
                
        except Exception as e:
            print(f"{para_info} AI校验失败: {e}，不发送到服务器")
        
        with ai_correction_lock:
            ai_correction_completed += 1
            if ai_correction_completed >= ai_correction_count:
                print("\n" + "=" * 80)
                print("全屏OCR识别完成！")
                print("OCR数据已发送到服务器，浏览器页面会自动更新")
                print("AI纠错已全部完成，已修正的文本会显示为绿色")
                print("按F4键可以重新识别并更新页面")
                print("=" * 80 + "\n")
    
    with ai_correction_lock:
        ai_correction_count += 1
    
    thread = threading.Thread(target=_correct)
    thread.daemon = True
    thread.start()


def recognize_merged_paragraphs(image_path_or_pil, merged_paragraphs, ai_model=None):
    """
    对合并后的段落进行 OCR 识别，并使用AI校验结果（已废弃，使用并发版本）
    
    参数:
        image_path_or_pil: 原始图片路径或 PIL Image 对象
        merged_paragraphs: 合并后的段落列表
        ai_model: Gemini AI模型实例（可选）
    
    返回:
        识别后的段落列表，每个段落添加了 'text' 字段
    """
    return recognize_merged_paragraphs_concurrent(image_path_or_pil, merged_paragraphs)


def find_paragraph_under_mouse(merged_paragraphs, mouse_x, mouse_y):
    """
    找到鼠标下或最近的文本段落
    
    参数:
        merged_paragraphs: 合并后的段落列表
        mouse_x: 鼠标 x 坐标
        mouse_y: 鼠标 y 坐标
    
    返回:
        鼠标下的段落或最近的段落，如果没有则返回 None
    """
    print(f"\n鼠标位置: ({mouse_x}, {mouse_y})")
    
    for para in merged_paragraphs:
        x1, y1, x2, y2 = para["box"]
        if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
            print(f"鼠标在段落内: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
            return para
    
    print("鼠标不在任何段落内，寻找最近的段落...")
    
    min_distance = float('inf')
    nearest_para = None
    
    for para in merged_paragraphs:
        x1, y1, x2, y2 = para["box"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        distance = ((mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            nearest_para = para
    
    if nearest_para:
        x1, y1, x2, y2 = nearest_para["box"]
        print(f"最近的段落: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, 距离={min_distance:.1f}")
    
    return nearest_para


def find_nearest_paragraphs(merged_paragraphs, mouse_x, mouse_y, count):
    """
    找到鼠标最近的N个文本段落（基于中心点距离）
    
    参数:
        merged_paragraphs: 合并后的段落列表
        mouse_x: 鼠标 x 坐标
        mouse_y: 鼠标 y 坐标
        count: 要选择的段落数量
    
    返回:
        最近的N个段落列表，按距离排序
    """
    print(f"\n鼠标位置: ({mouse_x}, {mouse_y})")
    print(f"智能选择模式: 选择最近的 {count} 个段落")
    
    para_distances = []
    
    for para in merged_paragraphs:
        x1, y1, x2, y2 = para["box"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        distance = ((mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2) ** 0.5
        
        para_distances.append({
            'para': para,
            'distance': distance,
            'center_x': center_x,
            'center_y': center_y
        })
    
    para_distances.sort(key=lambda x: x['distance'])
    
    selected_paras = [item['para'] for item in para_distances[:count]]
    
    print(f"\n已选择 {len(selected_paras)} 个段落:")
    for i, item in enumerate(para_distances[:count]):
        para = item['para']
        x1, y1, x2, y2 = para["box"]
        print(f"  {i+1}. 段落[({int(x1)},{int(y1)})] 距离={item['distance']:.1f}px")
    
    return selected_paras


def on_f4_pressed():
    global ai_correction_count, ai_correction_completed
    
    ai_correction_count = 0
    ai_correction_completed = 0
    
    print("\n=== F4 按下，开始全屏OCR识别 ===")

    screenshot = capture_full_screen()

    print("正在进行文本检测...")
    screenshot_np = np.array(screenshot)
    output = model.predict(screenshot_np, batch_size=1)

    for i, res in enumerate(output):
        res.print()

        if save_debug_images:
            os.makedirs("./output", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            res.save_to_img(save_path="./output/")
            json_path = f"./output/detection_{timestamp}.json"
            res.save_to_json(save_path=json_path)
            print(f"检测结果已保存到: {json_path}")

        if 'dt_polys' in res:
            dt_polys = res['dt_polys']

            boxes = polys_to_boxes(dt_polys)

            print(f"\n检测到 {len(boxes)} 个文本块")

            merged_paragraphs = sort_ocr_results(boxes, y_thresh=18)

            print("\n" + "=" * 80)
            print(f"合并完成！共得到 {len(merged_paragraphs)} 个段落:")
            print("=" * 80)

            for i, para in enumerate(merged_paragraphs):
                print(f"\n【段落 {i+1}】")
                print(f"坐标: x1={para['box'][0]:.1f}, y1={para['box'][1]:.1f}, x2={para['box'][2]:.1f}, y2={para['box'][3]:.1f}")
                print(f"包含 {len(para['children'])} 个文本块")

            if save_debug_images:
                os.makedirs("./output", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"./output/screenshot_{timestamp}.png"
                screenshot.save(screenshot_path)
                merged_image_path = f"./output/merged_{timestamp}.png"
                draw_merged_paragraphs(screenshot_path, merged_paragraphs, merged_image_path)

            merged_paragraphs = recognize_merged_paragraphs_concurrent(screenshot, merged_paragraphs)

            send_cropped_screenshot_to_server(screenshot)
            send_ocr_to_server(merged_paragraphs)

            selected_paras = None
            
            if SMART_AI_SELECTION_MODE and not ENABLE_AI_CORRECTION:
                mouse_x, mouse_y = pyautogui.position()
                selected_paras = find_nearest_paragraphs(merged_paragraphs, mouse_x, mouse_y, SMART_AI_SELECTION_COUNT)
                
                print("\n" + "=" * 80)
                print("智能AI选择模式已启用")
                print("OCR识别完成！数据已发送到服务器")
                print("开始AI纠错（仅对选中的段落进行后台异步处理）...")
                print("=" * 80)
                
                for para in selected_paras:
                    correct_text_with_ai_async(para.copy(), ai_model, send_ai_correction_to_server, force_correction=True)
            else:
                print("\n" + "=" * 80)
                print("OCR识别完成！数据已发送到服务器")
                print("开始AI纠错（后台异步处理）...")
                print("=" * 80)

                for para in merged_paragraphs:
                    correct_text_with_ai_async(para.copy(), ai_model, send_ai_correction_to_server)
        else:
            print("\n警告: 检测结果不包含 dt_polys 信息，无法应用分列合并算法")


def main():
    print("程序已启动，按F4键进行全屏截图并检测文本区域...")
    print("按Ctrl+C退出程序")
    
    try:
        keyboard.add_hotkey('f4', on_f4_pressed)
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n程序已退出")


if __name__ == "__main__":
    main()
