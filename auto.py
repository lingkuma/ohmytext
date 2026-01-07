import numpy as np

def sort_ocr_results(texts, scores, boxes, y_thresh=15):
    """
    texts: list[str]
    scores: np.ndarray or list[float]
    boxes: np.ndarray shape [N, 4]  => [x_min, y_min, x_max, y_max] (PaddleOCR 3.x rec_boxes)
    y_thresh: 行分组阈值，单位像素，适当调大调小
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

    # 先按 y1 排一下，方便行聚类
    items.sort(key=lambda x: x["y1"])

    # 行聚类：把 y 接近的归为一行
    lines = []
    for it in items:
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

    # 展平
    sorted_items = []
    for line in lines:
        sorted_items.extend(line["items"])

    return sorted_items


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


# ====== 你原来的 OCR 代码 ======
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="de",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,   # 建议开着
)

result = ocr.predict("./test_img/test1.png")

for r in result:
    texts = r["rec_texts"]
    scores = r["rec_scores"]
    boxes = r["rec_boxes"]  # [N,4]

    sorted_items = sort_ocr_results(texts, scores, boxes, y_thresh=18)
    
    # 使用智能段落合并算法
    merged_paragraphs = merge_paragraphs(sorted_items, gap_coefficient=1.2)

    # 打印按阅读顺序排好的结果
    for it in sorted_items:
        if it["text"].strip():
            print(f'{it["text"]}  ({it["score"]:.3f})')
    
    print("\n=== 合并后的段落 ===")
    for para in merged_paragraphs:
        print(f"段落: {para['text']}")
        print(f"坐标: {para['box']}")
        print(f"包含 {len(para['children'])} 个文本块")
        print()
