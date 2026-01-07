import numpy as np
import keyboard
import pyautogui
from PIL import Image
import time
import pyperclip

def detect_columns(items, x_thresh=10):
    """
    基于 x 坐标聚类检测列
    
    参数:
        items: 文本块列表，每个包含 box [x1, y1, x2, y2]
        x_thresh: 列聚类阈值，单位像素
    
    返回:
        列列表，每列包含该列的所有文本块
    """
    # 先按 x1 排序
    items_sorted = sorted(items, key=lambda x: x["box"][0])
    
    # 列聚类：把 x 接近的归为一列
    columns = []
    for it in items_sorted:
        placed = False
        for col in columns:
            # 用当前列的平均 x1 来判断是否同一列
            if abs(it["box"][0] - col["x1_mean"]) < x_thresh:
                col["items"].append(it)
                # 更新均值
                col["x1_mean"] = np.mean([p["box"][0] for p in col["items"]])
                placed = True
                break
        if not placed:
            columns.append({"x1_mean": it["box"][0], "items": [it]})
    
    # 按列的平均 x 坐标排序（从左到右）
    columns.sort(key=lambda x: x["x1_mean"])
    
    # 对每列内的文本块按y1排序（从上到下）
    for col in columns:
        col["items"].sort(key=lambda x: x["box"][1])
    
    print("\n=== 列检测结果 ===")
    for i, col in enumerate(columns):
        print(f"\n第 {i+1} 列:")
        print(f"  平均 x1: {col['x1_mean']:.2f}")
        print(f"  文本块数量: {len(col['items'])}")
        print(f"  文本块内容:")
        for item in col["items"]:
            print(f"    - 文本: {item['text']}")
            print(f"      坐标: x1={item['box'][0]:.1f}, y1={item['box'][1]:.1f}, x2={item['box'][2]:.1f}, y2={item['box'][3]:.1f}")
    
    return columns


def sort_ocr_results(texts, scores, boxes, y_thresh=15, x_thresh=50):
    """
    texts: list[str]
    scores: np.ndarray or list[float]
    boxes: np.ndarray shape [N, 4]  => [x_min, y_min, x_max, y_max] (PaddleOCR 3.x rec_boxes)
    y_thresh: 行分组阈值，单位像素，适当调大调小
    x_thresh: 列分组阈值，单位像素，用于检测多列布局
    """
    items = []
    for i in range(len(texts)):
        t = texts[i]
        s = float(scores[i])
        x1, y1, x2, y2 = boxes[i].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        items.append({
            "text": t,
            "score": s,
            "box": [x1, y1, x2, y2],
            "cx": cx,
            "cy": cy,
            "y1": y1
        })

    # 第一步：检测列
    columns = detect_columns(items, x_thresh=x_thresh)
    
    # 第二步：对每列单独进行行聚类、排序和合并
    all_merged_paragraphs = []
    for col in columns:
        col_items = col["items"]
        
        # 先按 y1 排一下，方便行聚类
        col_items.sort(key=lambda x: x["y1"])

        # 行聚类：把 y 接近的归为一行
        lines = []
        for it in col_items:
            placed = False
            for line in lines:
                # 用当前行的平均 cy 来判断是否同一行
                if abs(it["cy"] - line["cy_mean"]) < y_thresh:
                    line["items"].append(it)
                    # 更新均值
                    line["cy_mean"] = np.mean([p["cy"] for p in line["items"]])
                    placed = True
                    break
            if not placed:
                lines.append({"cy_mean": it["cy"], "items": [it]})

        # 行内按 x 排序
        for line in lines:
            line["items"].sort(key=lambda x: x["cx"])

        # 行按 y 排序
        lines.sort(key=lambda x: x["cy_mean"])

        # 展平该列的结果
        col_sorted_items = []
        for line in lines:
            col_sorted_items.extend(line["items"])
        
        # 对该列进行段落合并
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
        - text: 完整的段落文本
        - box: 合并后的外接矩形 [x1, y1, x2, y2]
        - children: 构成该段落的原始文本块列表
    """
    if not sorted_items:
        return []
    
    # 第一步：计算每个item的几何特征
    for item in sorted_items:
        x1, y1, x2, y2 = item["box"]
        item["height"] = y2 - y1
        item["width"] = x2 - x1
        item["center_x"] = (x1 + x2) / 2
        item["center_y"] = (y1 + y2) / 2
    
    # 初始化结果列表和当前块
    results = []
    first_item = sorted_items[0]
    current_block = {
        "text": first_item["text"],
        "box": first_item["box"].copy(),
        "children": [first_item],
        "height": first_item["height"],
        "width": first_item["width"]
    }
    
    # 遍历剩余的items
    for i in range(1, len(sorted_items)):
        item = sorted_items[i]
        
        # 计算垂直间距（Gap）
        current_y2 = current_block["box"][3]
        item_y1 = item["box"][1]
        gap = item_y1 - current_y2
        
        # 计算平均行高
        avg_height = (current_block["height"] + item["height"]) / 2
        
        # 第二步：双重合并校验逻辑
        should_merge = False
        
        # 1. 垂直亲密度检查
        if gap < avg_height * gap_coefficient:
            # 2. 水平对齐检查（防止多栏布局被错误合并）
            horizontal_overlap = True
            if horizontal_alignment_check:
                current_x1, _, current_x2, _ = current_block["box"]
                item_x1, _, item_x2, _ = item["box"]
                
                # 计算水平重叠
                overlap = max(0, min(current_x2, item_x2) - max(current_x1, item_x1))
                min_width = min(current_block["width"], item["width"])
                
                # 如果重叠率小于20%，认为不在同一列
                if min_width > 0 and overlap / min_width < 0.2:
                    horizontal_overlap = False
            
            # 3. 字体高度一致性检查（可选）
            height_consistent = True
            if height_consistency_check:
                height_diff = abs(current_block["height"] - item["height"])
                max_height = max(current_block["height"], item["height"])
                if max_height > 0 and height_diff / max_height > 0.5:
                    height_consistent = False
            
            # 如果所有检查都通过，则合并
            if horizontal_overlap and height_consistent:
                should_merge = True
        
        if should_merge:
            # 执行合并操作
            # 坐标合并（生成大矩形）
            current_box = current_block["box"]
            item_box = item["box"]
            
            new_x1 = min(current_box[0], item_box[0])
            new_y1 = min(current_box[1], item_box[1])
            new_x2 = max(current_box[2], item_box[2])
            new_y2 = max(current_box[3], item_box[3])
            
            # 文本合并（添加空格）
            current_text = current_block["text"]
            item_text = item["text"]
            
            # 判断是否需要添加空格
            if current_text and item_text:
                # 如果当前文本以非空格结尾，且新文本以非空格开头，则添加空格
                if not current_text[-1].isspace() and not item_text[0].isspace():
                    new_text = current_text + " " + item_text
                else:
                    new_text = current_text + item_text
            else:
                new_text = current_text + item_text
            
            # 更新当前块
            current_block["text"] = new_text
            current_block["box"] = [new_x1, new_y1, new_x2, new_y2]
            current_block["children"].append(item)
            current_block["height"] = max(current_block["height"], item["height"])
            current_block["width"] = new_x2 - new_x1
        else:
            # 将当前块存入结果列表
            results.append(current_block)
            # 开启新的段落块
            current_block = {
                "text": item["text"],
                "box": item["box"].copy(),
                "children": [item],
                "height": item["height"],
                "width": item["width"]
            }
    
    # 将最后一个块存入结果列表
    results.append(current_block)
    
    return results


def capture_screenshot_around_mouse(width=1400, height=800):
    """
    以鼠标为中心截图
    
    参数:
        width: 截图宽度
        height: 截图高度
    
    返回:
        PIL Image对象
    """
    # 获取鼠标位置
    mouse_x, mouse_y = pyautogui.position()
    
    # 计算截图区域的左上角坐标
    left = mouse_x - width // 2
    top = mouse_y - height // 2
    
    # 获取屏幕尺寸
    screen_width, screen_height = pyautogui.size()
    
    # 确保截图区域在屏幕范围内
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + width > screen_width:
        left = screen_width - width
    if top + height > screen_height:
        top = screen_height - height
    
    # 截图
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    
    return screenshot, (left, top)


def find_largest_paragraph_at_mouse(mouse_x, mouse_y, merged_paragraphs, screenshot_offset):
    """
    找到鼠标位置下最大的段落矩形
    
    参数:
        mouse_x: 鼠标屏幕X坐标
        mouse_y: 鼠标屏幕Y坐标
        merged_paragraphs: 合并后的段落列表
        screenshot_offset: 截图偏移量 (left, top)
    
    返回:
        最大的段落，如果没有则返回None
    """
    offset_left, offset_top = screenshot_offset
    
    # 将鼠标屏幕坐标转换为截图内的相对坐标
    relative_x = mouse_x - offset_left
    relative_y = mouse_y - offset_top
    
    # 找到所有包含鼠标位置的段落
    matching_paragraphs = []
    for para in merged_paragraphs:
        x1, y1, x2, y2 = para["box"]
        if x1 <= relative_x <= x2 and y1 <= relative_y <= y2:
            matching_paragraphs.append(para)
    
    if not matching_paragraphs:
        return None
    
    # 找到面积最大的段落
    largest_para = max(matching_paragraphs, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))
    
    return largest_para


def process_screenshot(screenshot_path):
    """
    处理截图，执行OCR并返回合并后的段落
    
    参数:
        screenshot_path: 截图文件路径
    
    返回:
        合并后的段落列表
    """
    result = ocr.predict(screenshot_path)
    
    for r in result:
        texts = r["rec_texts"]
        scores = r["rec_scores"]
        boxes = r["rec_boxes"]
        
        merged_paragraphs = sort_ocr_results(texts, scores, boxes, y_thresh=18)
        
        print("\n" + "=" * 80)
        print(f"合并完成！共得到 {len(merged_paragraphs)} 个段落:")
        print("=" * 80)
        
        for i, para in enumerate(merged_paragraphs):
            print(f"\n【段落 {i+1}】")
            print(f"文本: {para['text']}")
            print(f"坐标: x1={para['box'][0]:.1f}, y1={para['box'][1]:.1f}, x2={para['box'][2]:.1f}, y2={para['box'][3]:.1f}")
            print(f"包含 {len(para['children'])} 个文本块")
            print(f"高度: {para['height']:.1f}, 宽度: {para['width']:.1f}")
        
        return merged_paragraphs
    
    return []


def on_f4_pressed():
    """
    F4键按下时的处理函数
    """
    print("\n=== F4 按下，开始处理 ===")
    
    # 获取鼠标位置
    mouse_x, mouse_y = pyautogui.position()
    print(f"鼠标位置: ({mouse_x}, {mouse_y})")
    
    # 截图
    screenshot, offset = capture_screenshot_around_mouse(1400, 800)
    screenshot_path = "./temp_screenshot.png"
    screenshot.save(screenshot_path)
    print(f"截图已保存: {screenshot_path}")
    print(f"截图偏移: {offset}")
    
    # OCR处理
    print("正在进行OCR识别...")
    merged_paragraphs = process_screenshot(screenshot_path)
    
    if not merged_paragraphs:
        print("未识别到任何文本")
        return
    
    # 找到鼠标下最大的段落
    largest_para = find_largest_paragraph_at_mouse(mouse_x, mouse_y, merged_paragraphs, offset)
    
    if largest_para:
        print("\n=== 识别结果 ===")
        print(f"文本: {largest_para['text']}")
        print(f"坐标: {largest_para['box']}")
        print(f"包含 {len(largest_para['children'])} 个文本块")
        
        pyperclip.copy(largest_para['text'])
        print(f"\n文本已写入剪切板")
    else:
        print("\n鼠标位置下未找到文本段落")


def main():
    """
    主函数，监听F4快捷键
    """
    print("程序已启动，按F4键进行截图识别...")
    print("按Ctrl+C退出程序")
    
    try:
        keyboard.add_hotkey('f4', on_f4_pressed)
        keyboard.wait('esc')
    except KeyboardInterrupt:
        print("\n程序已退出")


# ====== OCR 初始化 ======
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="de",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
)

# 运行主程序
if __name__ == "__main__":
    main()
