import os
import sys
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 載入環境變數
load_dotenv()

# --- LINE Bot 設定 ---
channel_access_token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
channel_secret = os.environ.get("LINE_CHANNEL_SECRET")

if not channel_access_token or not channel_secret:
    print("請在 .env 中設定 LINE_CHANNEL_ACCESS_TOKEN 和 LINE_CHANNEL_SECRET")
    sys.exit(1)

configuration = Configuration(access_token=channel_access_token)
handler = WebhookHandler(channel_secret)

# --- LangChain 設定 ---
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個在 LINE 上的親切助理。請用繁體中文回答。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# 簡單的記憶體存儲 (Key 是 LINE 使用者 ID)
store = {}

def get_session_history(user_id: str) -> InMemoryChatMessageHistory:
    if user_id not in store:
        store[user_id] = InMemoryChatMessageHistory()
    return store[user_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# --- FastAPI 應用 ---
app = FastAPI()

@app.post("/callback")
async def callback(request: Request):
    # 獲取 X-Line-Signature 標記預防偽造
    signature = request.headers.get("X-Line-Signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing Signature")

    # 獲取請求內容
    body = await request.body()
    body_str = body.decode("utf-8")

    try:
        handler.handle(body_str, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid Signature")

    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    user_id = event.source.user_id
    user_message = event.message.text

    # 透過 LangChain 獲取回應
    try:
        response = chain_with_history.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": user_id}}
        )
        ai_reply = response.content
    except Exception as e:
        ai_reply = f"抱歉，發生了點錯誤：{str(e)}"

    # 傳回訊息給 LINE
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=ai_reply)]
            )
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
