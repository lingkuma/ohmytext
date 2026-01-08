# OCR网页显示功能使用说明

## 功能概述
此功能可以在网页中显示OCR识别的文本，文本位置与原屏幕中的文本位置完全对应。

## 使用步骤

### 1. 运行程序
```bash
python TextDetection_ocr_Cover.py
```

### 2. 准备工作
- 将游戏窗口化运行
- 确保Windows状态栏在顶部（高度48像素）
- 准备好透明化工具（如TransparentWindows等）

### 3. 使用流程
1. 在游戏中，按 **F4** 键触发全屏OCR识别
2. 程序会自动：
   - 全屏截图
   - 检测所有文本区域
   - 对每个文本区域进行OCR识别
   - 生成HTML网页文件（保存在 `./output/ocr_display_时间戳.html`）
3. 使用透明化工具将浏览器窗口透明化
4. 在浏览器中打开生成的HTML文件
5. 将透明化的浏览器窗口覆盖在游戏窗口上
6. 文本会在相同位置显示，可以使用浏览器插件进行翻译、查词等操作

## 配置选项

### 状态栏高度
如果您的Windows状态栏高度不是48像素，请修改 `TextDetection_ocr_Cover.py` 中的：
```python
STATUS_BAR_HEIGHT = 48  # 修改为实际的状态栏高度
```

### 调试模式
如需查看中间结果图片，修改：
```python
save_debug_images = True
```

## 技术特点
- 文本背景统一为白色
- 字体大小固定为16px
- 支持文本自动换行
- 支持滚动查看长文本
- 坐标自动校正（考虑状态栏偏移）

## 输出文件
- `./output/screenshot_时间戳.png` - 全屏截图
- `./output/ocr_display_时间戳.html` - 生成的网页文件
- `./output/merged_时间戳.png` - 合并后的文本区域图片（调试模式）
- `./output/detection_时间戳.json` - 检测结果JSON（调试模式）

## 注意事项
- 确保OCR API服务正在运行（http://127.0.0.1:2333/api/ocr）
- 游戏窗口需要与浏览器窗口大小一致
- 建议使用Chrome或Edge浏览器以获得最佳兼容性
- 可以设置浏览器为"应用模式"（PWA）以获得更好的体验