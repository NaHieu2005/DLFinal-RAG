import streamlit as st
from ui.sidebar import sidebar
from ui.chat_interface import file_upload_screen, processing_screen, chat_screen
from core.document_processor import process_uploaded_files
from core.embedding_handler import (
    get_embedding_model, 
    get_or_create_vector_store, 
    generate_session_id,
    recreate_retriever_from_saved
)
from core.llm_handler import get_llm_instance, get_qa_retrieval_chain, get_reranker
from core.chat_history import save_chat_history, load_chat_history, list_chat_sessions
from config import CHAT_HISTORIES_DIR, VECTOR_STORES_DIR
import os
import shutil
import uuid
import json
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

st.set_page_config(page_title="Chatbot Tài Liệu RAG", layout="wide")

# Thêm hàm trực tiếp tìm kiếm tài liệu khi retriever thất bại
def direct_vector_search(question, embedding_model, vs_id, top_k=10):
    """
    Tìm kiếm trực tiếp từ vector store khi retriever thông thường thất bại.
    Trả về list các Document.
    """
    if not embedding_model or not vs_id:
        print("[app] Không thể thực hiện tìm kiếm trực tiếp - thiếu model hoặc vector store ID")
        return []
    
    try:
        # Tìm đường dẫn đến vector store
        vs_path = os.path.join(VECTOR_STORES_DIR, vs_id)
        if not os.path.exists(vs_path):
            print(f"[app] Không tìm thấy vector store tại {vs_path}")
            return []
            
        # Tải FAISS vector store trực tiếp
        try:
            print(f"[app] Đang tải FAISS vector store từ {vs_path}...")
            vector_store = FAISS.load_local(vs_path, embedding_model, allow_dangerous_deserialization=True)
            
            # Thực hiện tìm kiếm
            print(f"[app] Thực hiện tìm kiếm trực tiếp với k={top_k}...")
            docs_with_score = vector_store.similarity_search_with_score(question, k=top_k)
            
            # Lọc kết quả có điểm số tốt
            docs = [doc for doc, score in docs_with_score]
            print(f"[app] Tìm thấy {len(docs)} kết quả trong tìm kiếm trực tiếp")
            
            # Thêm thông tin vào metadata
            for doc in docs:
                doc.metadata["direct_search"] = True
                
            return docs
            
        except Exception as e:
            print(f"[app] Lỗi khi tải vector store: {e}")
            return []
    except Exception as e:
        print(f"[app] Lỗi trong direct_vector_search: {e}")
        return []

def local_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# --- Sidebar ---
with st.sidebar:
    new_chat, selected_session_id = sidebar()

# --- Session State ---
def reset_to_upload():
    keys_to_reset = [
        "uploaded_files", "vector_store", "retriever", "session_id", 
        "file_names", "messages", "current_session_display_name"
    ]
    for key in keys_to_reset:
        st.session_state[key] = None
    st.session_state.state = "upload"
    st.session_state.processing = False
    st.session_state.bot_answering = False
    st.session_state.messages = [] # Đảm bảo messages là list rỗng
    clear_memory() # Gọi hàm mới để giải phóng bộ nhớ

# Hàm mới để giải phóng bộ nhớ
def clear_memory():
    """Giải phóng bộ nhớ bằng cách xóa các đối tượng lớn khỏi session_state"""
    import gc
    
    # Hủy bỏ các đối tượng lớn
    if "retriever" in st.session_state:
        st.session_state.retriever = None
    if "vector_store" in st.session_state:
        st.session_state.vector_store = None
        
    # Buộc garbage collector thu hồi bộ nhớ
    gc.collect()
    print("[app] Đã giải phóng bộ nhớ không cần thiết")

# Khởi tạo session_state nếu chưa có
default_states = {
    "state": "upload",
    "uploaded_files": None,
    "processing": False,
    "vector_store": None,
    "retriever": None,  # Thêm trường mới cho retriever nâng cao
    "session_id": None,
    "file_names": None,
    "messages": [],
    "bot_answering": False,
    "current_session_display_name": None,
    "stop_action_requested": False
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Xử lý New Chat hoặc chọn chat cũ ---
if new_chat:
    reset_to_upload()
    st.rerun()

if selected_session_id:
    if st.session_state.session_id != selected_session_id or st.session_state.state != "chatting":
        st.session_state.session_id = selected_session_id
        messages, display_name = load_chat_history(selected_session_id)
        st.session_state.messages = messages
        st.session_state.current_session_display_name = display_name
        
        embedding_model = get_embedding_model()
        if embedding_model:
            # Sử dụng hàm mới để tái tạo retriever từ dữ liệu đã lưu
            retriever = recreate_retriever_from_saved(selected_session_id, embedding_model)
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.vector_store = None  # Không cần lưu vector_store riêng
                st.session_state.file_names = None 
                st.session_state.state = "chatting"
                st.session_state.processing = False # Đảm bảo reset processing flag
                st.session_state.bot_answering = False # Đảm bảo reset bot_answering flag
            else:
                st.error(f"Không thể tải cơ sở tri thức cho session '{st.session_state.current_session_display_name}'. Có thể đã bị xóa hoặc lỗi. Vui lòng tạo chat mới.")
                reset_to_upload() 
        else:
            st.error("Lỗi nghiêm trọng: Không thể khởi tạo embedding model khi tải session.")
            reset_to_upload()
    st.rerun()

# --- Giao diện chính ---
if st.session_state.state == "upload":
    st.title("💬 Chatbot Hỏi Đáp Tài Liệu (RAG với Llama 3)")
    st.markdown("#### Tải lên tài liệu của bạn để bắt đầu")
    valid_files, error_files, start_clicked, _ = file_upload_screen(st.session_state.uploaded_files)
    if valid_files:
        st.session_state.uploaded_files = valid_files
    else:
        st.session_state.uploaded_files = None
    
    if error_files:
        st.warning("Một số file không hợp lệ và sẽ bị bỏ qua:")
        for fname, reason in error_files.items():
            st.write(f"- {fname}: {reason}")

    if start_clicked and st.session_state.uploaded_files:
        # Tạo session_id mới cho chat mới
        new_session_id = generate_session_id([f.name for f in st.session_state.uploaded_files])
        st.session_state.session_id = new_session_id
        # Đặt display_name ban đầu bằng session_id (hoặc có thể tùy chỉnh sau)
        st.session_state.current_session_display_name = new_session_id 
        st.session_state.file_names = [f.name for f in st.session_state.uploaded_files] # Lưu tên file
        st.session_state.state = "processing"
        st.session_state.stop_action_requested = False # Reset cờ dừng khi bắt đầu xử lý mới
        st.session_state.bot_answering = False # Đảm bảo bot_answering là false
        st.rerun()

elif st.session_state.state == "processing":
    st.title(f"⚙️ Đang xử lý: {st.session_state.current_session_display_name}")
    if not st.session_state.uploaded_files:
        st.warning("Không có file nào để xử lý. Vui lòng quay lại và tải lên.")
        if st.button("Quay lại trang Upload"):
            reset_to_upload()
            st.rerun()
    else:
        stop_processing_clicked = processing_screen(st.session_state.uploaded_files)
        if stop_processing_clicked:
            st.warning("Đã dừng quá trình xử lý tài liệu.")
            reset_to_upload() # reset_to_upload đã bao gồm việc xóa session_id, etc.
            st.rerun()
        else:
            if not st.session_state.vector_store and not st.session_state.retriever:  # Chỉ xử lý nếu chưa có vector_store hoặc retriever
                parent_chunks, child_chunks = process_uploaded_files(st.session_state.uploaded_files)
                
                if parent_chunks and child_chunks:
                    embedding_model = get_embedding_model()
                    if embedding_model:
                        # Truyền cả parent_chunks và child_chunks để xử lý nâng cao
                        retriever, vs_id_saved = get_or_create_vector_store(
                            st.session_state.session_id, 
                            (parent_chunks, child_chunks),  # Truyền tuple gồm cả parent và child chunks
                            embedding_model
                        )
                        
                        if retriever:
                            # Lưu lại retriever và chuyển thành main retriever để dùng sau này
                            st.session_state.retriever = retriever
                            
                            # Có thể cũng lưu vector_store nếu cần thiết
                            # st.session_state.vector_store = vector_store
                            
                            # Khởi tạo tin nhắn chào mừng đầu tiên
                            st.session_state.messages = [{"role": "assistant", "content": f"Tài liệu cho '{st.session_state.current_session_display_name}' đã sẵn sàng! Bạn hãy đặt câu hỏi."}]
                            save_chat_history(
                                st.session_state.session_id, 
                                st.session_state.messages, 
                                display_name_to_set=st.session_state.current_session_display_name
                            )
                            st.session_state.state = "chatting" # Chuyển sang chatting
                            st.session_state.processing = False
                            st.session_state.bot_answering = False
                            st.session_state.stop_action_requested = False # Đảm bảo reset cờ dừng
                            st.rerun()
                        else:
                            st.error("Không thể tạo cơ sở tri thức.")
                            reset_to_upload()
                            st.rerun()
                    else:
                        st.error("Lỗi nghiêm trọng: Không thể khởi tạo embedding model khi xử lý tài liệu.")
                        reset_to_upload()
                        st.rerun()
                else:
                    st.error("Không xử lý được tài liệu. Vui lòng kiểm tra định dạng file và thử lại.")
                    # Không reset_to_upload() ngay, cho phép người dùng thấy lỗi và có thể quay lại
                    if st.button("Thử lại với file khác"):
                        reset_to_upload()
                        st.rerun()

elif st.session_state.state == "chatting":
    if not st.session_state.session_id or not st.session_state.current_session_display_name:
        st.warning("Không có session nào được chọn hoặc session bị lỗi. Vui lòng tạo chat mới hoặc chọn từ lịch sử.")
        if st.button("Bắt đầu Chat Mới"):
            reset_to_upload()
            st.rerun()
        st.stop()

    st.title(f"💬 {st.session_state.current_session_display_name}")

    # Khu vực quản lý session (đổi tên, xóa)
    with st.expander("Tùy chọn Session", expanded=False):
        new_name = st.text_input(
            "Đổi tên Session:", 
            value=st.session_state.current_session_display_name,
            key=f"rename_input_{st.session_state.session_id}"
        )
        if st.button("Lưu tên mới", key=f"save_rename_btn_{st.session_state.session_id}"):
            if new_name.strip() and new_name.strip() != st.session_state.current_session_display_name:
                save_chat_history(
                    st.session_state.session_id, 
                    st.session_state.messages, 
                    display_name_to_set=new_name.strip()
                )
                st.session_state.current_session_display_name = new_name.strip()
                st.success(f"Đã đổi tên session thành: {new_name.strip()}")
                st.rerun()
            elif not new_name.strip():
                st.warning("Tên hiển thị không được để trống.")
            else:
                st.info("Tên mới giống với tên hiện tại.")

        st.markdown("---")
        st.markdown("<h5 style='color: red;'>Xóa Session này</h5>", unsafe_allow_html=True)
        confirm_delete_text = f"Tôi chắc chắn muốn xóa session '{st.session_state.current_session_display_name}' và tất cả dữ liệu liên quan."
        confirm_delete = st.checkbox(confirm_delete_text, key=f"confirm_delete_cb_{st.session_state.session_id}")
        
        if st.button("XÁC NHẬN XÓA", type="primary", disabled=not confirm_delete, key=f"confirm_delete_btn_{st.session_state.session_id}"):
            if confirm_delete:
                session_id_to_delete = st.session_state.session_id
                display_name_deleted = st.session_state.current_session_display_name
                
                history_file_path = os.path.join(CHAT_HISTORIES_DIR, f"{session_id_to_delete}.json")
                vector_store_path = os.path.join(VECTOR_STORES_DIR, session_id_to_delete)
                
                deleted_files = False
                try:
                    if os.path.exists(history_file_path):
                        os.remove(history_file_path)
                        deleted_files = True
                    if os.path.exists(vector_store_path):
                        shutil.rmtree(vector_store_path)
                        deleted_files = True
                    
                    if deleted_files:
                        st.success(f"Đã xóa thành công session: {display_name_deleted} (ID: {session_id_to_delete})")
                    else:
                        st.warning(f"Không tìm thấy file nào để xóa cho session: {display_name_deleted}. Có thể đã được xóa trước đó.")
                    
                    reset_to_upload()
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi khi xóa session '{display_name_deleted}': {e}")
            else:
                st.warning("Vui lòng xác nhận trước khi xóa.")

    # Hiển thị lịch sử chat và placeholder cho "Bot đang suy nghĩ..."
    st.markdown("<div class='chat-history-area'>", unsafe_allow_html=True)
    for idx, message in enumerate(st.session_state.messages):
        # Đảm bảo tin nhắn chào mừng đầu tiên có sources
        if idx == 0 and message["role"] == "assistant" and "sources" not in message:
            message["sources"] = [{
                "source": "Tin nhắn đầu tiên",
                "chunk_id": "initial-message",
                "content": "Đây là tin nhắn chào mừng, không có nguồn tham khảo cụ thể."
            }]
                
        # Debug print cho mỗi message
        print(f"\n=== DEBUG MESSAGE {idx} ===")
        print(f"Role: {message.get('role')}")
        print(f"Has sources: {'sources' in message}")
        if 'sources' in message:
            print(f"Sources length: {len(message['sources'])}")
            if len(message['sources']) > 0:
                print(f"First source: {message['sources'][0]}")
        print(f"=== END DEBUG MESSAGE {idx} ===\n")
        
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Debug print để kiểm tra message
            print(f"\n=== DEBUG UI MESSAGE ===")
            print(f"Message role: {message.get('role')}")
            print(f"Message keys: {message.keys()}")
            print(f"'sources' in message: {'sources' in message}")
            if 'sources' in message:
                print(f"message['sources'] type: {type(message['sources'])}")
                print(f"message['sources'] length: {len(message['sources']) if message['sources'] else 0}")
                print(f"message['sources'] is empty or None: {not message['sources']}")
            print(f"=== END DEBUG UI MESSAGE ===\n")
            
            # Luôn hiển thị phần nguồn cho tin nhắn của assistant
            if message["role"] == "assistant":
                st.markdown("**NGUỒN THAM KHẢO:**")
                
                # Đảm bảo luôn có nguồn, thêm nếu không có
                if "sources" not in message or message["sources"] is None:
                    message["sources"] = [{
                        "source": "Tin nhắn hệ thống",
                        "chunk_id": "system-message",
                        "content": "Không có nguồn tham khảo cụ thể cho tin nhắn này."
                    }]
                    
                elif not message["sources"] or len(message["sources"]) == 0:
                    message["sources"] = [{
                        "source": "Kết quả tìm kiếm",
                        "chunk_id": "auto-generated",
                        "content": "Hệ thống không tìm thấy nguồn tham khảo cụ thể cho câu hỏi này. Câu trả lời được tổng hợp từ kiến thức có sẵn."
                    }]
                
                # Hiển thị thông tin về số lượng nguồn
                st.info(f"Có {len(message['sources'])} nguồn được tìm thấy.")
                
                # Hiển thị các nguồn
                    for i, source in enumerate(message["sources"]):
                    try:
                        source_name = source.get('source', 'N/A')
                        chunk_id = source.get('chunk_id', 'N/A')
                        content = source.get('content', 'N/A')
                        
                        st.markdown(f"**Nguồn {i+1}:** {source_name} - Chunk ID: {chunk_id}")
                        st.code(content[:150] + "..." if len(content) > 150 else content)
                    except Exception as e:
                        st.error(f"Lỗi khi hiển thị nguồn #{i+1}: {e}")
                        st.text(f"Dữ liệu nguồn: {source}")

    # Simplified: Display "Bot đang suy nghĩ..." directly if bot is answering
    if st.session_state.bot_answering:
        with st.chat_message("assistant"):
            st.markdown("▌ Bot đang suy nghĩ...")
        
    st.markdown("</div>", unsafe_allow_html=True) # Đóng div chat-history-area

    # Gọi chat_screen để lấy input và các nút điều khiển
    prompt, send_triggered, stop_button_clicked_in_ui = chat_screen(
        st.session_state.messages, 
        st.session_state.bot_answering
    )

    # Ưu tiên xử lý yêu cầu dừng nếu có
    if st.session_state.get('stop_action_requested', False):
        if st.session_state.bot_answering: 
            st.session_state.bot_answering = False
            # Placeholder sẽ được tự động xóa ở lần rerun tiếp theo bởi khối logic ở trên
            # (khi bot_answering là False và placeholder tồn tại)
            
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != ":warning: Trả lời đã bị dừng bởi người dùng.":
                st.session_state.messages.append({"role": "assistant", "content": ":warning: Trả lời đã bị dừng bởi người dùng."})
                save_chat_history(st.session_state.session_id, st.session_state.messages, st.session_state.current_session_display_name)
            
            st.session_state.stop_action_requested = False 
            st.rerun()
        else:
            st.session_state.stop_action_requested = False
            # Không cần rerun nếu không có gì thay đổi

    elif send_triggered and not st.session_state.bot_answering and prompt.strip():
        st.session_state.messages.append({"role": "user", "content": prompt.strip()})
        st.session_state.bot_answering = True
        st.session_state.stop_action_requested = False 
        save_chat_history(st.session_state.session_id, st.session_state.messages, st.session_state.current_session_display_name)
        # Placeholder sẽ được tạo ở lần rerun tiếp theo bởi khối logic ở trên
        st.rerun()

    elif st.session_state.bot_answering:
        # Placeholder đã được hiển thị bởi khối logic ở trên trước khi chat_screen được gọi.
        # Giờ chỉ tập trung vào việc lấy câu trả lời.

        if not st.session_state.retriever and not st.session_state.vector_store:
            st.warning("Đang thử tải lại cơ sở tri thức...")
            embedding_model = get_embedding_model()
            if embedding_model:
                # Sử dụng hàm mới để tái tạo retriever từ dữ liệu đã lưu
                retriever = recreate_retriever_from_saved(st.session_state.session_id, embedding_model)
                if retriever:
                    st.session_state.retriever = retriever
                    st.rerun() 
                else:
                    st.error("Lỗi nghiêm trọng: Không thể tải cơ sở tri thức cho phiên làm việc này. Vui lòng thử tạo phiên mới từ đầu.")
                    st.session_state.bot_answering = False # Dừng bot nếu không tải được VS
                    # Placeholder sẽ tự động xóa ở rerun tiếp theo
                    reset_to_upload()
                    st.rerun()
            else:
                st.error("Lỗi nghiêm trọng: Không thể khởi tạo embedding model để tải lại vector store.")
                st.session_state.bot_answering = False # Dừng bot
                reset_to_upload()
                st.rerun()

        # Ưu tiên sử dụng retriever nâng cao (đã được khởi tạo với parent-child)
        retriever_to_use = st.session_state.retriever
            
        llm = get_llm_instance()
        qa_chain = get_qa_retrieval_chain(llm, retriever_to_use)
            
            response_content = ""
            sources_list = []
            try:
                last_user_msg_content = ""
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "user":
                        last_user_msg_content = msg["content"]
                        break
                
                if not last_user_msg_content:
                    st.warning("Không tìm thấy câu hỏi từ người dùng để xử lý.")
                    st.session_state.bot_answering = False
                    st.session_state.stop_action_requested = False 
                    # Placeholder sẽ tự động xóa ở rerun tiếp theo
                    st.rerun()
                else:
                # Cập nhật: Sử dụng phương thức mới để gọi qa_chain
                try:
                    # Thử sử dụng phương thức invoke của LangChain mới
                    from langchain_core.runnables.config import RunnableConfig
                    response = qa_chain.invoke(
                        {"query": last_user_msg_content},
                        config=RunnableConfig(run_name="QA Query")
                    )
                except Exception as e1:
                    print(f"[app] Lỗi khi sử dụng phương thức invoke: {e1}")
                    # Fallback sang phương thức cũ nếu cần
                    try:
                        response = qa_chain({"query": last_user_msg_content})
                    except Exception as e2:
                        print(f"[app] Lỗi nghiêm trọng cả hai phương thức: {e2}")
                        response = {
                            "result": f"Có lỗi khi xử lý câu hỏi: {str(e2)}",
                            "source_documents": []
                        }
                
                # Debug print để kiểm tra dữ liệu trả về từ QA chain
                print("\n\n=== DEBUG QA RESPONSE ===")
                print(f"Response type: {type(response)}")
                print(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
                print(f"Has source_documents: {'source_documents' in response if isinstance(response, dict) else False}")
                if isinstance(response, dict) and 'source_documents' in response:
                    print(f"Number of source documents: {len(response['source_documents'])}")
                    for i, doc in enumerate(response['source_documents']):
                        print(f"Document {i+1}:")
                        print(f"  Type: {type(doc)}")
                        print(f"  Metadata: {doc.metadata}")
                        print(f"  Page content length: {len(doc.page_content)}")
                        print(f"  Content preview: {doc.page_content[:50]}...")
                print("=== END DEBUG QA RESPONSE ===\n\n")
                
                # Trích xuất kết quả từ QA chain
                if isinstance(response, dict):
                    response_content = response.get("result", "")
                    raw_sources = response.get("source_documents", [])
                    
                    # Luôn đảm bảo có ít nhất một nguồn để hiển thị
                    sources_list = []
                    if not raw_sources or len(raw_sources) == 0:
                        print("[app] Warning: source_documents rỗng, thử tìm kiếm trực tiếp...")
                        
                        # Sử dụng tìm kiếm trực tiếp nếu không có kết quả từ retriever
                        embedding_model = get_embedding_model()
                        if embedding_model and st.session_state.session_id:
                            direct_sources = direct_vector_search(last_user_msg_content, embedding_model, st.session_state.session_id, top_k=10)
                            
                            if direct_sources and len(direct_sources) > 0:
                                print(f"[app] Tìm thấy {len(direct_sources)} nguồn từ tìm kiếm trực tiếp")
                                raw_sources = direct_sources
                                # Xử lý các nguồn tìm thấy
                                for src in raw_sources:
                                    try:
                                        source_item = {
                                            "source": src.metadata.get("source", "Tìm kiếm trực tiếp") if hasattr(src, "metadata") else "Tìm kiếm trực tiếp",
                                            "chunk_id": src.metadata.get("chunk_id", "direct-search") if hasattr(src, "metadata") else "direct-search",
                                            "content": src.page_content.replace("\\n", " ") if hasattr(src, "page_content") else "No content"
                                        }
                                        sources_list.append(source_item)
                                        print(f"[app] Đã thêm nguồn trực tiếp: {source_item['source']}")
                                    except Exception as e:
                                        print(f"[app] Lỗi khi xử lý nguồn trực tiếp: {e}")
                        
                        # Nếu vẫn không có nguồn nào, tạo nguồn mặc định
                        if not sources_list:
                            print("[app] Không thể tìm thấy nguồn, tạo nguồn mặc định")
                            sources_list = [{
                                "source": "Kết quả tổng hợp",
                                "chunk_id": "generated",
                                "content": "Không tìm thấy nguồn tham khảo cụ thể. Câu trả lời được tổng hợp từ kiến thức chung."
                            }]
                    else:
                        # Xử lý nguồn thường
                        for src in raw_sources:
                            try:
                                source_item = {
                                    "source": src.metadata.get("source", "N/A") if hasattr(src, "metadata") else "Unknown",
                                    "chunk_id": src.metadata.get("chunk_id", "N/A") if hasattr(src, "metadata") else "unknown",
                                    "content": src.page_content.replace("\\n", " ") if hasattr(src, "page_content") else "No content"
                                }
                                sources_list.append(source_item)
                                print(f"[app] Đã thêm nguồn: {source_item['source']}")
                            except Exception as e:
                                print(f"[app] Lỗi khi xử lý nguồn: {e}")
                                # Thêm nguồn lỗi để có thông tin debug
                        sources_list.append({
                                    "source": "Lỗi khi xử lý",
                                    "chunk_id": "error",
                                    "content": f"Đã xảy ra lỗi: {str(e)}"
                                })
                    
                    # Đảm bảo luôn có ít nhất một nguồn
                    if not sources_list:
                        sources_list = [{
                            "source": "Không có nguồn",
                            "chunk_id": "empty",
                            "content": "Không thể lấy thông tin nguồn từ câu trả lời."
                        }]
                    
                    # Debug print để kiểm tra sources_list
                    print("\n\n=== DEBUG SOURCES LIST ===")
                    print(f"Number of sources after conversion: {len(sources_list)}")
                    if len(sources_list) > 0:
                        print(f"First source: {sources_list[0]}")
                    print("=== END DEBUG SOURCES LIST ===\n\n")
                else:
                    # Phòng trường hợp không phải dict
                    response_content = str(response)
                    sources_list = [{
                        "source": "Lỗi định dạng",
                        "chunk_id": "format-error",
                        "content": "Kết quả trả về không đúng định dạng. Vui lòng liên hệ quản trị viên."
                    }]
            except Exception as e:
                response_content = f"Đã xảy ra lỗi khi xử lý yêu cầu: {e}"
            
        # Debug print để kiểm tra message trước khi thêm vào st.session_state.messages
        print("\n\n=== DEBUG FINAL MESSAGE ===")
        print(f"Response content length: {len(response_content)}")
        print(f"Sources list length: {len(sources_list)}")
        message_to_append = {"role": "assistant", "content": response_content, "sources": sources_list}
        print(f"Message to append has sources: {'sources' in message_to_append}")
        print(f"Message sources length: {len(message_to_append['sources'])}")
        print("=== END DEBUG FINAL MESSAGE ===\n\n")
        
        st.session_state.messages.append(message_to_append)
            save_chat_history(st.session_state.session_id, st.session_state.messages, st.session_state.current_session_display_name)
            st.session_state.bot_answering = False
            st.rerun()

    # Mới thêm: Kiểm tra và đảm bảo tất cả tin nhắn đều có nguồn (để lưu đúng khi save_chat_history)
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "assistant" and ("sources" not in msg or not msg["sources"]):
            print(f"[app] Fix missing sources in message #{i}")
            # Tin nhắn assistant không có nguồn, thêm nguồn mặc định
            msg["sources"] = [{
                "source": "Hệ thống",
                "chunk_id": "auto-fixed",
                "content": "Không có nguồn tham khảo cụ thể. Đã tự động thêm."
            }]

else:
    st.error("Trạng thái không xác định. Đang reset về trang chủ.")
    reset_to_upload()
    st.rerun()