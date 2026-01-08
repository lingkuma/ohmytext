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
import webbrowser
import shutil

model = TextDetection(model_name="PP-OCRv5_server_det")

save_debug_images = False
STATUS_BAR_HEIGHT = 48


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
        
        response = requests.post(api_url, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f'API Error: Status Code {response.status_code}')
            return None
        
    except requests.exceptions.RequestException as e:
        print(f'Request Error: {e}')
        return None
    except Exception as e:
        print(f'Error: {e}')
        return None


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


def generate_html_page(merged_paragraphs, output_path):
    """
    生成HTML页面，显示OCR识别的文本
    
    参数:
        merged_paragraphs: 合并后的段落列表（包含OCR识别的文本）
        output_path: 输出HTML文件路径
    """
    text_data = []
    
    for para in merged_paragraphs:
        x1, y1, x2, y2 = para["box"]
        
        adjusted_y = y1 - STATUS_BAR_HEIGHT
        
        text = para.get('text', '')
        
        text_data.append({
            'x': x1,
            'y': adjusted_y,
            'width': x2 - x1,
            'height': y2 - y1,
            'text': text
        })
    
    html_template = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Text Display</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            background-color: #ffffff;
            position: relative;
        }

        .text-box {
            position: absolute;
            background-color: #ffffff;
            font-size: 16px;
            line-height: 1.5;
            color: #000000;
            overflow-y: auto;
            overflow-x: hidden;
            word-wrap: break-word;
            word-break: break-all;
            white-space: pre-wrap;
            padding: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .text-box::-webkit-scrollbar {
            width: 8px;
        }

        .text-box::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        .text-box::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }

        .text-box::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
</head>
<body>
    <div id="text-container"></div>

    <script>
        const textData = ''' + json.dumps(text_data, ensure_ascii=False) + ''';
        
        function renderTextBoxes() {
            const container = document.getElementById('text-container');
            container.innerHTML = '';
            
            textData.forEach((item, index) => {
                const box = document.createElement('div');
                box.className = 'text-box';
                box.style.left = item.x + 'px';
                box.style.top = item.y + 'px';
                box.style.width = item.width + 'px';
                box.style.height = item.height + 'px';
                box.textContent = item.text;
                box.id = `text-box-${index}`;
                container.appendChild(box);
            });
        }

        document.addEventListener('DOMContentLoaded', renderTextBoxes);
    </script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"\nHTML页面已生成: {output_path}")
    print(f"共生成 {len(text_data)} 个文本区域")


def capture_full_screen():
    """
    全屏截图
    
    返回:
        PIL Image对象
    """
    screenshot = pyautogui.screenshot()
    return screenshot


def recognize_merged_paragraphs(image_path, merged_paragraphs):
    """
    对合并后的段落进行 OCR 识别
    
    参数:
        image_path: 原始图片路径
        merged_paragraphs: 合并后的段落列表
    
    返回:
        识别后的段落列表，每个段落添加了 'text' 字段
    """
    image = Image.open(image_path)
    
    print("\n" + "=" * 80)
    print("开始对合并后的段落进行 OCR 识别...")
    print("=" * 80)
    
    for i, para in enumerate(merged_paragraphs):
        x1, y1, x2, y2 = para["box"]
        
        cropped_image = image.crop((x1, y1, x2, y2))
        
        print(f"\n【段落 {i+1}】")
        print(f"裁切区域: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
        print(f"裁切尺寸: {cropped_image.size[0]}x{cropped_image.size[1]}")
        
        try:
            result = call_luna_ocr_api(cropped_image)
            
            if result and 'text' in result:
                text = result['text']
                para['text'] = text
                print(f"识别结果: {text}")
            else:
                print(f"识别失败: API返回结果为空或格式错误")
                para['text'] = ""
                
        except Exception as e:
            print(f"识别失败: {e}")
            import traceback
            traceback.print_exc()
            para['text'] = ""
    
    return merged_paragraphs


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


def on_f4_pressed():
    print("\n=== F4 按下，开始全屏OCR识别并生成网页 ===")
    
    screenshot = capture_full_screen()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_path = f"./output/screenshot_{timestamp}.png"
    
    os.makedirs("./output", exist_ok=True)
    screenshot.save(screenshot_path)
    print(f"截图已保存: {screenshot_path}")
    
    print("正在进行文本检测...")
    screenshot_np = np.array(screenshot)
    output = model.predict(screenshot_np, batch_size=1)
    
    for i, res in enumerate(output):
        res.print()
        
        if save_debug_images:
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
                merged_image_path = f"./output/merged_{timestamp}.png"
                draw_merged_paragraphs(screenshot_path, merged_paragraphs, merged_image_path)
            
            merged_paragraphs = recognize_merged_paragraphs(screenshot_path, merged_paragraphs)
            
            html_output_path = f"./output/ocr_display_{timestamp}.html"
            generate_html_page(merged_paragraphs, html_output_path)
            
            print("\n" + "=" * 80)
            print("全屏OCR识别完成！")
            print(f"HTML页面已生成: {html_output_path}")
            print("您可以使用透明化工具在浏览器中打开此页面查看OCR结果")
            print("=" * 80)
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
