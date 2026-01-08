from paddleocr import TextRecognition
#model = TextRecognition(model_name="latin_PP-OCRv5_mobile_rec")
model = TextRecognition(model_name="PP-OCRv5_server_rec")
output = model.predict(input="./test/TEST4.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")