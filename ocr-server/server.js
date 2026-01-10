const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const cors = require('cors');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url} - ${req.ip}`);
  next();
});

let latestOCRData = null;
let latestScreenshot = null;

wss.on('connection', (ws) => {
  console.log('客户端已连接');

  if (latestOCRData) {
    ws.send(JSON.stringify({ type: 'update', data: latestOCRData }));
  }

  ws.on('close', () => {
    console.log('客户端已断开');
  });
});

app.post('/api/update-ocr', (req, res) => {
  console.log('[DEBUG] /api/update-ocr 接收到请求');
  console.log('[DEBUG] 请求体类型:', typeof req.body);
  console.log('[DEBUG] 请求体是否为数组:', Array.isArray(req.body));
  
  latestOCRData = req.body;
  console.log('收到OCR数据更新，文本区域数量:', latestOCRData.length);

  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ type: 'update', data: latestOCRData }));
    }
  });

  res.json({ success: true, message: 'OCR数据已更新' });
});

app.post('/api/update-ocr-correction', (req, res) => {
  const correctionData = req.body;
  console.log('收到AI纠错数据更新:', correctionData);

  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ type: 'correction', data: correctionData }));
    }
  });

  res.json({ success: true, message: 'AI纠错数据已更新' });
});

app.get('/api/ocr-data', (req, res) => {
  res.json(latestOCRData || []);
});

app.post('/api/update-screenshot', (req, res) => {
  console.log('[DEBUG] /api/update-screenshot 接收到请求');
  console.log('[DEBUG] 请求体包含image字段:', 'image' in req.body);
  console.log('[DEBUG] image数据长度:', req.body.image ? req.body.image.length : 0);
  
  const imageData = req.body.image;
  
  if (!imageData) {
    console.log('[DEBUG] 未接收到图片数据');
    return res.status(400).json({ success: false, message: '未接收到图片数据' });
  }
  
  latestScreenshot = imageData;
  console.log('收到截图更新');
  
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ type: 'screenshot', data: imageData }));
    }
  });
  
  res.json({ success: true, message: '截图已更新' });
});

app.get('/api/screenshot', (req, res) => {
  res.json({ image: latestScreenshot });
});

app.use((req, res) => {
  console.log('[DEBUG] 404 - 未找到路由:', req.method, req.url);
  res.status(404).json({ error: 'Not Found', path: req.url });
});

const PORT = 2334;
server.listen(PORT, () => {
  console.log(`OCR服务器运行在 http://localhost:${PORT}`);
  console.log(`WebSocket服务运行在 ws://localhost:${PORT}`);
});
