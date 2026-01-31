import os
import gradio as gr
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 載入 .env 檔案中的環境變數
load_dotenv()

def get_groq_models():
    """
    從 Groq API 獲取可用模型清單
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return ["openai/gpt-oss-120b"] # 預設模型
    
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            models_data = response.json()
            # 取得模型 ID 並以字母序排序
            model_ids = [model['id'] for model in models_data['data']]
            return sorted(model_ids)
        else:
            print(f"無法獲取模型: {response.status_code}")
            return ["openai/gpt-oss-120b"]
    except Exception as e:
        print(f"獲取模型時發生錯誤: {e}")
        return ["openai/gpt-oss-120b"]

# 記憶體存儲
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def chat_response(message, history, model_name):
    """
    處理用戶訊息並返回 AI 回應
    """
    try:
        # 依照選擇的模型動態初始化 LLM
        llm = ChatGroq(
            model=model_name,
            temperature=0.7,
            max_tokens=1000
        )

        # 建立提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # 建立鏈並加上歷史記錄功能
        chain = prompt | llm
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        print(f"使用模型: {model_name} | 用戶訊息: {message}")
        
        response = chain_with_history.invoke(
            {"input": message},
            config={"configurable": {"session_id": "default"}}
        )
        
        # Gradio 5.0+ 使用字典格式
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response.content})
        
        return "", history
    except Exception as e:
        error_msg = f"發生錯誤: {str(e)}"
        print(error_msg)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history

def clear_history():
    """
    清除對話歷史
    """
    if "default" in store:
        store["default"] = InMemoryChatMessageHistory()
    return []

# 取得可用模型
available_models = get_groq_models()

# 創建 Gradio 介面
with gr.Blocks(title="LangChain + Gradio 對話應用", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 LangChain + Groq 對話應用")
    
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("與 AI 進行對話！請記得設定您的 Groq API 金鑰。")
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=available_models,
                value=available_models[0] if available_models else "openai/gpt-oss-120b",
                label="選擇 Groq 模型",
                interactive=True
            )

    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        container=True,
        # type="messages" # 明確指定使用訊息格式
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="輸入您的訊息...",
            show_label=False,
            container=False,
            scale=7
        )
        submit_btn = gr.Button("送出", scale=1, variant="primary")
        clear_btn = gr.Button("清除對話", scale=1)

    gr.Markdown("---")
    gr.Markdown("**使用說明：**")
    gr.Markdown("- 在右上方下拉選單選擇您想使用的 Groq 模型")
    gr.Markdown("- 在下方輸入框輸入您的問題或訊息")
    gr.Markdown("- 點擊「送出」按鈕或按 Enter 發送訊息")
    gr.Markdown("- 點擊「清除對話」按鈕來重新開始對話")
    gr.Markdown("- 請確保已設定有效的 Groq API 金鑰")

    # 設定事件處理
    msg.submit(chat_response, [msg, chatbot, model_selector], [msg, chatbot])
    submit_btn.click(chat_response, [msg, chatbot, model_selector], [msg, chatbot])
    clear_btn.click(clear_history, outputs=chatbot)

if __name__ == "__main__":
    # 啟動應用程式
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )