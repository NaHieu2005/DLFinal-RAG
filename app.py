import streamlit as st
from ui.sidebar import sidebar
from ui.chat_interface import file_upload_screen, processing_screen, chat_screen
from core.document_processor import process_uploaded_files
from core.embedding_handler import get_embedding_model, get_or_create_vector_store, generate_session_id
from core.llm_handler import get_llm_instance, get_qa_retrieval_chain
from core.chat_history import save_chat_history, load_chat_history

st.set_page_config(page_title="Chatbot Tài Liệu RAG", layout="wide")
st.title("💬 Chatbot Hỏi Đáp Tài Liệu (RAG với Llama 3)")
def local_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# --- Sidebar ---
with st.sidebar:
    new_chat, selected_session = sidebar()

# --- Session State ---
def reset_to_upload():
    st.session_state.state = "upload"
    st.session_state.uploaded_files = None
    st.session_state.processing = False
    st.session_state.vector_store = None
    st.session_state.session_id = None
    st.session_state.file_names = None
    st.session_state.messages = []
    st.session_state.bot_answering = False

def reset_to_chat():
    st.session_state.state = "chatting"
    st.session_state.processing = False
    st.session_state.bot_answering = False

if "state" not in st.session_state:
    st.session_state.state = "upload"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "file_names" not in st.session_state:
    st.session_state.file_names = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "bot_answering" not in st.session_state:
    st.session_state.bot_answering = False

# --- Xử lý New Chat hoặc chọn chat cũ ---
if new_chat:
    reset_to_upload()
    st.rerun()

if selected_session:
    st.session_state.session_id = selected_session
    st.session_state.messages = load_chat_history(selected_session)
    embedding_model = get_embedding_model()
    from core.embedding_handler import load_vector_store
    # Tải vector store cho session đã chọn
    loaded_vs = load_vector_store(selected_session, embedding_model)
    if loaded_vs:
        st.session_state.vector_store = loaded_vs
        st.session_state.file_names = None # File names không cần thiết khi tải session cũ
        st.session_state.state = "chatting"
        st.session_state.bot_answering = False
    else:
        st.error(f"Không thể tải cơ sở tri thức cho session '{selected_session}'. Có thể đã bị xóa hoặc lỗi. Vui lòng tạo chat mới.")
        reset_to_upload() # Reset về màn hình upload nếu không tải được vector store
    st.rerun()

# --- Giao diện chính ---
if st.session_state.state == "upload":
    valid_files, error_files, start_clicked, _ = file_upload_screen(st.session_state.uploaded_files)
    if valid_files:
        st.session_state.uploaded_files = valid_files
    else:
        st.session_state.uploaded_files = None
    if start_clicked and st.session_state.uploaded_files:
        session_id = generate_session_id([f.name for f in st.session_state.uploaded_files])
        st.session_state.session_id = session_id
        st.session_state.state = "processing"
        st.rerun()

elif st.session_state.state == "processing":
    stop_clicked = processing_screen(st.session_state.uploaded_files)
    if stop_clicked:
        reset_to_upload()
        st.rerun()
    else:
        if not st.session_state.vector_store:
            documents, file_names = process_uploaded_files(st.session_state.uploaded_files)
            if documents:
                embedding_model = get_embedding_model()
                # Sử dụng st.session_state.session_id đã được tạo ở bước upload
                # Bỏ tham số save, để dùng default True trong get_or_create_vector_store
                vector_store, vs_id_saved = get_or_create_vector_store(
                    st.session_state.session_id, 
                    documents, 
                    embedding_model
                )
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.file_names = file_names
                    st.session_state.messages = [{"role": "assistant", "content": "Tài liệu đã sẵn sàng! Bạn hãy đặt câu hỏi."}]
                    # Không cần save_vector_store ở đây nữa vì get_or_create_vector_store đã xử lý
                    reset_to_chat()
                    st.rerun()
                else:
                    st.error("Không thể tạo cơ sở tri thức.")
                    reset_to_upload()
                    st.rerun()
            else:
                st.error("Không xử lý được tài liệu.")
                reset_to_upload()
                st.rerun()

elif st.session_state.state == "chatting":
    prompt, send_clicked, stop_clicked = chat_screen(st.session_state.messages, st.session_state.bot_answering)
    if stop_clicked and st.session_state.bot_answering:
        st.session_state.bot_answering = False
        st.session_state.messages.append({"role": "assistant", "content": ":warning: Trả lời đã bị dừng bởi người dùng."})
        save_chat_history(st.session_state.session_id, st.session_state.messages)
        st.rerun()
    if send_clicked and not st.session_state.bot_answering and prompt.strip():
        st.session_state.messages.append({"role": "user", "content": prompt.strip()})
        st.session_state.bot_answering = True
        embedding_model = get_embedding_model()
        save_chat_history(st.session_state.session_id, st.session_state.messages)
        st.rerun()
    if st.session_state.bot_answering:
        embedding_model = get_embedding_model()
        vector_store = st.session_state.vector_store
        if not vector_store:
            from core.embedding_handler import load_vector_store
            print(f"[App] Thử tải lại vector store cho session: {st.session_state.session_id}")
            vector_store = load_vector_store(st.session_state.session_id, embedding_model)
            if vector_store:
                st.session_state.vector_store = vector_store
                print(f"[App] Tải lại vector store thành công.")
            else:
                st.error("Lỗi nghiêm trọng: Không thể tải cơ sở tri thức cho phiên làm việc này. Vui lòng thử tạo phiên mới từ đầu.")
                st.session_state.bot_answering = False # Đảm bảo dừng bot
                reset_to_upload() # Đưa về màn hình upload
                st.rerun() # Áp dụng thay đổi state và tải lại giao diện

        if not st.session_state.vector_store and st.session_state.state == "chatting":
             # Điều kiện st.session_state.state == "chatting" để tránh reset nếu đã ở upload
            st.error("Không thể tiếp tục phiên chat do thiếu cơ sở tri thức. Vui lòng bắt đầu lại.")
            reset_to_upload()
            st.rerun()

        # Chỉ tiếp tục nếu vector_store tồn tại và đang ở state chatting
        if st.session_state.vector_store and st.session_state.state == "chatting":
            llm = get_llm_instance()
            qa_chain = get_qa_retrieval_chain(llm, st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}))
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Bot đang suy nghĩ...")
                try:
                    last_user_msg = None
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "user":
                            last_user_msg = msg["content"]
                            break
                    response = qa_chain.invoke({"query": last_user_msg})
                    answer = response.get("result", "Xin lỗi, tôi không tìm thấy câu trả lời.")
                    sources = response.get("source_documents", [])
                    sources_list = []
                    for src in sources:
                        sources_list.append({
                            "source": src.metadata.get("source", "N/A"),
                            "chunk_id": src.metadata.get("chunk_id", "N/A"),
                            "content": src.page_content.replace("\n", " ")
                        })
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_list})
                    save_chat_history(st.session_state.session_id, st.session_state.messages)
                    st.session_state.bot_answering = False
                    st.rerun()
                except Exception as e:
                    error_message = f"Đã xảy ra lỗi: {e}"
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    save_chat_history(st.session_state.session_id, st.session_state.messages)
                    st.session_state.bot_answering = False
                    st.rerun()