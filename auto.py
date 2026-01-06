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

    # 打印按阅读顺序排好的结果
    for it in sorted_items:
        if it["text"].strip():
            print(f'{it["text"]}  ({it["score"]:.3f})')
