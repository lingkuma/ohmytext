from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="de")
result = ocr.ocr("./test_img/test1.png")     # 或者 result = ocr.predict("img.png")

for res in result:
    res.print()
    res.save_to_img("result")
