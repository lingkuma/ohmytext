const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const cors = require('cors');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

let latestOCRData = null;

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

const PORT = 8080;
server.listen(PORT, () => {
  console.log(`OCR服务器运行在 http://localhost:${PORT}`);
  console.log(`WebSocket服务运行在 ws://localhost:${PORT}`);
});
