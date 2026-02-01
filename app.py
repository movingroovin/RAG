import os
import shutil
import gradio as gr
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

# 載入 .env 檔案中的環境變數
load_dotenv()

# 初始化本地嵌入模型 (FastEmbed)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 專案管理 路徑
PROJECTS_DIR = "./projects"
os.makedirs(PROJECTS_DIR, exist_ok=True)

def get_project_paths(project_name):
    if not project_name:
        return None, None
    project_path = os.path.join(PROJECTS_DIR, project_name)
    upload_dir = os.path.join(project_path, "upload")
    db_path = os.path.join(project_path, "chroma_db")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(db_path, exist_ok=True)
    return upload_dir, db_path

def list_projects():
    if not os.path.exists(PROJECTS_DIR):
        return []
    return sorted([d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))])

def list_indexed_files(project_name):
    """
    從指定專案的向量資料庫中獲取已索引的文件列表
    """
    if not project_name:
        return "請先選擇或建立專案", []
    
    try:
        _, db_path = get_project_paths(project_name)
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        data = vectorstore.get()
        if not data or not data['metadatas']:
            return "目前專案知識庫為空", []
        
        sources = set()
        for meta in data['metadatas']:
            if 'source' in meta:
                sources.add(meta['source'])
        
        if not sources:
            return "目前專案中無文件來源", []
        
        display_text = "\n".join([f"📄 {os.path.basename(s)}" for s in sorted(list(sources))])
        return display_text, sorted(list(sources))
    except Exception as e:
        return f"無法讀取清單: {str(e)}", []

def delete_file(file_path, project_name):
    """
    從指定專案的向量資料庫中刪除指定文件
    """
    if not project_name:
        return "請先選擇專案", "請先選擇專案", []
    if not file_path:
        return "請先選擇要刪除的文件", *list_indexed_files(project_name)
    
    try:
        _, db_path = get_project_paths(project_name)
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        vectorstore.delete(where={"source": file_path})
        
        if os.path.exists(file_path):
            os.remove(file_path)
        
        filename = os.path.basename(file_path)
        status = f"已成功從專案「{project_name}」中刪除文件：{filename}"
        
        display_text, file_list = list_indexed_files(project_name)
        return status, display_text, gr.update(choices=file_list, value=None)
    except Exception as e:
        return f"刪除失敗: {str(e)}", *list_indexed_files(project_name)

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

def process_files(files, project_name):
    """
    處理上傳的文件，儲存至專案目錄下的 upload 資料夾，並加入該專案的向量資料庫
    """
    if not project_name:
        return "請先選擇或建立專案", "", gr.update()
    if not files:
        return "未選擇任何檔案", *list_indexed_files(project_name)
    
    upload_dir, db_path = get_project_paths(project_name)
    documents = []
    
    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join(upload_dir, filename)
        shutil.copy(file.name, dest_path)
        
        if dest_path.endswith('.pdf'):
            loader = PyPDFLoader(dest_path)
            documents.extend(loader.load())
        elif dest_path.endswith('.txt') or dest_path.endswith('.md'):
            loader = TextLoader(dest_path)
            documents.extend(loader.load())
    
    if not documents:
        return "沒有找到可讀取的內容", *list_indexed_files(project_name)

    # 切分文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # 初始化並加入該專案的向量資料庫
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    vectorstore.add_documents(documents=splits)
    
    status = f"成功處理 {len(files)} 個檔案，並已加入專案「{project_name}」知識庫。"
    display_text, file_list = list_indexed_files(project_name)
    return status, display_text, gr.update(choices=file_list)

def chat_response(message, history, model_name, use_rag, project_name):
    """
    處理用戶訊息並返回 AI 回應
    """
    try:
        if use_rag and not project_name:
            error_msg = "請先選擇專案以使用 RAG 功能"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history

        # 依照選擇的模型動態初始化 LLM
        llm = ChatGroq(
            model=model_name,
            temperature=0.7,
            max_tokens=1000
        )

        if use_rag:
            # RAG 模式下的提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一個專業的助手。請僅根據下方提供的【上下文內容】來回答問題。如果問題的答案不在內容中，請直接回答：「抱歉，根據目前的知識庫內容，我無法回答這個問題。」，不要嘗試利用您原有的知識來回答。若回答是中文，用繁體中文\n\n上下文內容：\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # 建立針對該專案的檢索器
            _, db_path = get_project_paths(project_name)
            vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # 檢索相關內容
            docs = retriever.invoke(message)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 組成鏈
            chain = prompt | llm
            chain_with_history = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="history"
            )
            
            response = chain_with_history.invoke(
                {"input": message, "context": context},
                config={"configurable": {"session_id": "default"}}
            )
        else:
            # 普通對話模式
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])

            chain = prompt | llm
            chain_with_history = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="history"
            )

            response = chain_with_history.invoke(
                {"input": message},
                config={"configurable": {"session_id": "default"}}
            )
        
        print(f"使用模型: {model_name} | RAG: {use_rag} | 用戶訊息: {message}")
        
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

def create_project(new_name):
    if not new_name:
        return "專案名稱不能為空", gr.update(), ""
    
    existing_projects = list_projects()
    if new_name in existing_projects:
        return f"專案「{new_name}」已存在", gr.update(), ""
    
    # 建立專案目錄
    get_project_paths(new_name)
    updated_projects = list_projects()
    return f"成功建立專案：{new_name}", gr.update(choices=updated_projects, value=new_name), ""

def on_project_change(project_name):
    display_text, file_list = list_indexed_files(project_name)
    return display_text, gr.update(choices=file_list, value=None), f"已切換至專案：{project_name}" if project_name else "請選擇專案"

# 取得可用模型和初始專案清單
available_models = get_groq_models()
available_projects = list_projects()

# 設定預設模型邏輯
default_model = "openai/gpt-oss-120b"
if available_models and default_model not in available_models:
    default_model = available_models[0]

# 創建 Gradio 介面
with gr.Blocks(title="LangChain + Gradio RAG 應用", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 LangChain + Groq 多專案 RAG 應用")
    
    with gr.Sidebar():
        gr.Markdown("## 📁 專案管理")
        project_selector = gr.Dropdown(
            label="選擇現有專案",
            choices=available_projects,
            value=available_projects[0] if available_projects else None,
            interactive=True
        )
        
        with gr.Row():
            new_project_input = gr.Textbox(label="新專案名稱", placeholder="輸入專案名稱...", scale=2)
            create_project_btn = gr.Button("新增", scale=1)
        
        gr.Markdown("---")
        gr.Markdown("## 📚 知識庫設定")
        rag_toggle = gr.Checkbox(label="啟用 RAG 功能", value=False)
        
        with gr.Tab("上傳文件"):
            file_upload = gr.File(
                label="上傳新文件",
                file_types=[".pdf", ".txt", ".md"],
                file_count="multiple"
            )
            process_btn = gr.Button("更新知識庫", variant="primary")
        
        with gr.Tab("管理文件"):
            file_to_delete = gr.Dropdown(
                label="選擇要刪除的文件",
                choices=[],
                interactive=True
            )
            delete_btn = gr.Button("刪除選定文件", variant="stop")
            
        upload_status = gr.Textbox(label="狀態訊息", interactive=False)
        
        gr.Markdown("### 📂 目前專案文件")
        indexed_files_display = gr.Markdown("請先選擇專案")
        
        gr.Markdown("---")
        gr.Markdown("### 模型設定")
        model_selector = gr.Dropdown(
            choices=available_models,
            value=default_model,
            label="選擇 Groq 模型",
            interactive=True
        )

    with gr.Column():
        chatbot = gr.Chatbot(
            height=500,
            show_label=False,
            container=True,
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
    gr.Markdown("1. **選擇專案**：從左側下拉選單選擇現有專案，或輸入名稱並點選「新增」來建立新專案。")
    gr.Markdown("2. **上傳文件**：在該專案下上傳 PDF/TXT/MD 檔，並點擊「更新知識庫」。")
    gr.Markdown("3. **開啟 RAG**：勾選「啟用 RAG 功能」即可開始針對該專案內容進行問答。")

    # 初始化顯示
    demo.load(
        on_project_change,
        inputs=[project_selector],
        outputs=[indexed_files_display, file_to_delete, upload_status]
    )

    # 設定事件處理
    project_selector.change(
        on_project_change,
        inputs=[project_selector],
        outputs=[indexed_files_display, file_to_delete, upload_status]
    )

    create_project_btn.click(
        create_project,
        inputs=[new_project_input],
        outputs=[upload_status, project_selector, new_project_input]
    )

    process_btn.click(
        process_files, 
        inputs=[file_upload, project_selector], 
        outputs=[upload_status, indexed_files_display, file_to_delete]
    )

    delete_btn.click(
        delete_file,
        inputs=[file_to_delete, project_selector],
        outputs=[upload_status, indexed_files_display, file_to_delete]
    )
    
    msg.submit(chat_response, [msg, chatbot, model_selector, rag_toggle, project_selector], [msg, chatbot])
    submit_btn.click(chat_response, [msg, chatbot, model_selector, rag_toggle, project_selector], [msg, chatbot])
    clear_btn.click(clear_history, outputs=chatbot)

if __name__ == "__main__":
    # 啟動應用程式
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )