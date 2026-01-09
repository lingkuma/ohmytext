# 游戏文本OCR透明显示系统使用说明

## 功能介绍

这个系统可以自动识别游戏中的文本区域，并通过WebSocket实时在浏览器中以相同位置显示OCR识别的文本。配合透明化工具，可以在游戏上方直接显示可交互的文本，方便使用浏览器的翻译插件进行翻译。

## 系统架构

- **Python OCR客户端**：负责截图、文本检测、OCR识别，并将数据发送到Node.js服务器
- **Node.js服务器**：提供HTTP API和WebSocket服务，实时推送OCR数据到浏览器
- **前端页面**：通过WebSocket接收OCR数据，动态更新显示

## 系统要求

- Windows操作系统
- Python 3.x
- Node.js 14.x 或更高版本
- PaddleOCR文本检测模型
- Luna OCR API服务（运行在 http://127.0.0.1:2333）
- 透明化工具（如TransparentWindows等）

## 安装步骤

### 1. 安装Python依赖

```bash
pip install keyboard pyautogui pillow numpy pyperclip base64 requests paddleocr
```

### 2. 安装Node.js依赖

进入 `ocr-server` 目录：

```bash
cd ocr-server
npm install
```

## 使用方法

### 1. 启动Node.js服务器

在 `ocr-server` 目录下运行：

```bash
npm start
```

服务器会启动在：
- HTTP: http://localhost:3000
- WebSocket: ws://localhost:3000

### 2. 启动OCR识别程序

运行 `TextDetection_ocr_Cover.py`：

```bash
python TextDetection_ocr_Cover.py
```

程序启动后会显示：
```
程序已启动，按F4键进行全屏截图并检测文本区域...
按Ctrl+C退出程序
```

### 3. 打开浏览器页面

在浏览器中打开：http://localhost:3000

页面右上角会显示连接状态：
- 绿色：已连接到服务器
- 红色：连接断开

### 4. 准备游戏环境

1. 将游戏设置为窗口化模式
2. 确保游戏窗口位置固定
3. 调整游戏窗口大小，使其适合屏幕

### 5. 设置透明化

1. 使用透明化工具将浏览器窗口设置为透明
2. 将透明化后的浏览器窗口安装为应用（PWA）
3. 这样浏览器视图就只有Windows顶栏+网页内容

### 6. 对齐位置

1. 将游戏窗口和透明化后的浏览器窗口重叠
2. 由于系统已自动减去48像素的状态栏高度，文本应该自动对齐
3. 如果有偏差，可以调整 `STATUS_BAR_HEIGHT` 常量（默认48）

### 7. 使用流程

1. 在游戏中显示需要识别的文本
2. 按 `F4` 键进行全屏OCR识别
3. 程序会自动：
   - 截取全屏截图
   - 检测文本区域
   - 进行OCR识别
   - 将OCR数据发送到服务器
   - 浏览器页面会实时更新显示新的OCR结果
4. 可以使用浏览器插件进行翻译、查词等操作

## 功能特性

### 实时更新

- 使用WebSocket实现实时数据推送
- OCR识别完成后立即更新浏览器显示
- 无需手动刷新页面

### 自动重连

- 浏览器与服务器断开连接后会自动重连
- 连接状态实时显示在页面右上角

### 文本显示

- 字体大小：16px
- 背景颜色：白色
- 文本颜色：黑色
- 自动换行：支持
- 滚动：内容超出区域时自动显示滚动条

### 坐标系统

- 系统自动处理48像素的状态栏高度
- 文本框位置与游戏中的文本区域精确对应
- 支持多列文本检测和合并

## 配置参数

### STATUS_BAR_HEIGHT

状态栏高度，默认48像素。如果你的系统状态栏高度不同，可以修改此值：

```python
STATUS_BAR_HEIGHT = 48
```

### save_debug_images

是否保存调试图片，默认False。设置为True可以保存中间结果：

```python
save_debug_images = True
```

### y_thresh

文本行合并阈值，默认18像素。调整此值可以改变文本合并的敏感度：

```python
merged_paragraphs = sort_ocr_results(boxes, y_thresh=18)
```

### 服务器端口

默认端口为3000。如需修改，编辑 `ocr-server/server.js`：

```javascript
const PORT = 3000;
```

## 输出文件

只有在 `save_debug_images = True` 时，程序才会在 `./output/` 目录下生成以下文件：

- `screenshot_YYYYMMDD_HHMMSS.png` - 全屏截图
- `merged_YYYYMMDD_HHMMSS.png` - 合并后的文本区域标注图
- `detection_YYYYMMDD_HHMMSS.json` - 检测结果JSON文件

## API接口

### POST /api/update-ocr

更新OCR数据

**请求体**：
```json
[
  {
    "x": 100,
    "y": 200,
    "width": 300,
    "height": 50,
    "text": "识别的文本"
  }
]
```

**响应**：
```json
{
  "success": true,
  "message": "OCR数据已更新"
}
```

### GET /api/ocr-data

获取最新的OCR数据

**响应**：
```json
[
  {
    "x": 100,
    "y": 200,
    "width": 300,
    "height": 50,
    "text": "识别的文本"
  }
]
```

### WebSocket消息

**服务器推送**：
```json
{
  "type": "update",
  "data": [
    {
      "x": 100,
      "y": 200,
      "width": 300,
      "height": 50,
      "text": "识别的文本"
    }
  ]
}
```

## 常见问题

### Q: 文本位置不对齐怎么办？

A: 检查以下几点：
1. 确认游戏是窗口化模式
2. 确认浏览器已安装为PWA应用
3. 调整 `STATUS_BAR_HEIGHT` 值以匹配你的系统状态栏高度
4. 确保游戏窗口和浏览器窗口完全重叠

### Q: OCR识别不准确怎么办？

A: 尝试以下方法：
1. 确保游戏文本清晰可见
2. 检查Luna OCR API服务是否正常运行
3. 调整游戏分辨率和文本大小
4. 启用调试模式查看检测结果

### Q: 浏览器显示"未连接"怎么办？

A: 检查以下几点：
1. 确认Node.js服务器正在运行
2. 确认服务器端口（默认3000）没有被占用
3. 检查浏览器控制台是否有错误信息
4. 尝试刷新页面

### Q: 文本框太小显示不全怎么办？

A: 文本框会自动显示滚动条，可以滚动查看完整内容。如果需要调整大小，可以修改 `ocr-server/public/index.html` 中的样式。

### Q: 如何停止服务器？

A: 在运行服务器的终端按 `Ctrl+C` 停止Node.js服务器。

## 快捷键

- `F4` - 执行全屏OCR识别并更新浏览器显示
- `Ctrl+C` - 退出Python OCR程序

## 技术细节

### 文本检测

使用PaddleOCR的PP-OCRv5_server_det模型进行文本检测，支持多列文本检测和智能合并。

### 文本识别

使用Luna OCR API进行文本识别，API地址：http://127.0.0.1:2333/api/ocr

### 段落合并

使用智能算法合并相邻的文本块，基于：
- 垂直间距
- 水平对齐
- 字体高度一致性

### 实时通信

使用WebSocket实现服务器到浏览器的实时数据推送，延迟极低。

## 注意事项

1. 确保Luna OCR API服务在运行
2. 确保Node.js服务器在运行
3. 首次使用建议启用调试模式查看检测结果
4. 游戏窗口大小和位置应该保持固定
5. 浏览器窗口需要设置为透明并安装为PWA
6. 系统状态栏高度需要正确设置

## 扩展功能

### 添加自定义样式

修改 `ocr-server/public/index.html` 中的样式，可以自定义文本框的样式：

```css
background-color: #ffffff;  /* 背景颜色 */
font-size: 16px;            /* 字体大小 */
color: #000000;             /* 文本颜色 */
```

### 添加快捷翻译

在浏览器中安装翻译插件（如Google翻译、沉浸式翻译等），可以一键翻译所有文本。

### 添加语音朗读

在浏览器中安装语音朗读插件，可以朗读识别的文本。

### 多客户端支持

服务器支持多个浏览器客户端同时连接，所有客户端会同步显示OCR结果。

## 文件结构

```
ohmytext/
├── TextDetection_ocr_Cover.py      # Python OCR客户端
├── ocr-server/
│   ├── package.json                # Node.js项目配置
│   ├── server.js                   # Node.js服务器
│   └── public/
│       └── index.html              # 前端页面
└── README_OCR_Overlay.md           # 使用说明文档
```

## 许可证

本项目仅供学习和个人使用。

## 联系方式

如有问题或建议，欢迎反馈。
