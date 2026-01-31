# LangChain AI Chat Project

整合 Gradio Web 介面與 LINE Messaging API 的多功能對話機器人。

## 🚀 快速開始

### 1. 環境設定
```bash
# 建立並啟動虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```

### 2. 設定 .env 檔案
在根目錄建立 `.env` 檔案並填入金鑰：
```dotenv
GROQ_API_KEY=你的_Groq_Key
GOOGLE_API_KEY=你的_Gemini_Key
LINE_CHANNEL_ACCESS_TOKEN=你的_LINE_Token
LINE_CHANNEL_SECRET=你的_LINE_Secret
```

### 3. 執行應用程式

#### 網頁版 (Gradio)
```bash
python app.py
```
預設訪問 `http://localhost:7860`。

#### LINE Bot 版 (FastAPI)
```bash
# 啟動伺服器
python line_bot.py

# 使用 ngrok 轉發 (Port 8000)
ngrok http 8000
```
Webhook URL 請填寫：`https://你的ngrok網址/callback`

## 📁 檔案說明
- `app.py`: Gradio 網頁對話介面 (支援動態模型切換)。
- `line_bot.py`: 基於 FastAPI 的 LINE Webhook 伺服器。
- `DEVELOPMENT_LOG.md`: 詳細開發歷程紀錄。
- `.env`: API 金鑰管理 (不應上傳至 Git)。