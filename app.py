import streamlit as st
from ui.sidebar import sidebar
from ui.chat_interface import file_upload_screen, processing_screen, chat_screen
from core.document_processor import process_uploaded_files
from core.embedding_handler import get_embedding_model, get_or_create_vector_store, generate_session_id, recreate_retriever_from_saved
from core.llm_handler import get_llm_instance, get_qa_retrieval_chain, get_reranker
from core.chat_history import save_chat_history, load_chat_history
import os
import shutil
from config import CHAT_HISTORIES_DIR, VECTOR_STORES_DIR
import uuid

st.set_page_config(page_title="Chatbot Tài Liệu RAG", layout="wide")

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
        "uploaded_files", "vector_store", "session_id", 
        "file_names", "messages", "current_session_display_name"
    ]
    for key in keys_to_reset:
        st.session_state[key] = None
    st.session_state.state = "upload"
    st.session_state.processing = False
    st.session_state.bot_answering = False
    st.session_state.messages = [] # Đảm bảo messages là list rỗng

    # Đảm bảo reset retriever
    st.session_state.retriever = None
    
    # Không xóa file_uploader state để giữ lại file đã upload
    # Streamlit sẽ tự quản lý uploader state
    
    print("[app] Đã reset toàn bộ session state về trạng thái upload")

# Hàm reset_to_chat đã được loại bỏ vì không còn cần thiết

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
    print("[app] Người dùng đã nhấn New Chat, đang reset...")
    reset_to_upload()
    # Giữ lại file đã upload khi nhấn New Chat
    # Không tạo giá trị ngẫu nhiên để uploader key không thay đổi
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
    valid_files, error_files, start_clicked = file_upload_screen(st.session_state.uploaded_files)
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
        print("[app] Trạng thái processing nhưng không có uploaded_files")
        if st.button("Quay lại trang Upload"):
            reset_to_upload()
            st.rerun()
    else:
        print(f"[app] Đang xử lý {len(st.session_state.uploaded_files)} file")
        stop_processing_clicked = processing_screen(st.session_state.uploaded_files)
        if stop_processing_clicked:
            st.warning("Đã dừng quá trình xử lý tài liệu.")
            reset_to_upload() # reset_to_upload đã bao gồm việc xóa session_id, etc.
            st.rerun()
        else:
            if not st.session_state.vector_store and not st.session_state.retriever:  # Chỉ xử lý nếu chưa có vector_store hoặc retriever
                print("[app] Bắt đầu xử lý tài liệu...")
                parent_chunks, child_chunks = process_uploaded_files(st.session_state.uploaded_files)
                
                if parent_chunks and child_chunks:
                    print(f"[app] Đã tạo {len(parent_chunks)} parent chunks và {len(child_chunks)} child chunks")
                    embedding_model = get_embedding_model()
                    if embedding_model:
                        print("[app] Đã khởi tạo embedding model, đang tạo vector store...")
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("Xem nguồn tham khảo"):
                    for i, source in enumerate(message["sources"]):
                        st.caption(f"Nguồn {i+1} (Từ: {source.get('source', 'N/A')}, Chunk ID: {source.get('chunk_id', 'N/A')})")
                        content_preview = source.get('content', '')[:300] + "..." if source.get('content') else "N/A"
                        st.markdown(content_preview)

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
                response = qa_chain.invoke({"query": last_user_msg_content})
                response_content = response.get("result", "Xin lỗi, tôi không tìm thấy câu trả lời.")
                sources = response.get("source_documents", [])
                for src in sources:
                    sources_list.append({
                        "source": src.metadata.get("source", "N/A"),
                        "chunk_id": src.metadata.get("chunk_id", "N/A"),
                        "content": src.page_content.replace("\\n", " ")
                    })
        except Exception as e:
            response_content = f"Đã xảy ra lỗi khi xử lý yêu cầu: {e}"
        
        # Tin nhắn "Bot đang suy nghĩ..." đã được hiển thị và sẽ tự động biến mất
        # khi rerun với bot_answering = False
        
        st.session_state.messages.append({"role": "assistant", "content": response_content, "sources": sources_list})
        save_chat_history(st.session_state.session_id, st.session_state.messages, st.session_state.current_session_display_name)
        st.session_state.bot_answering = False
        st.rerun()
else:
    st.error("Trạng thái không xác định. Đang reset về trang chủ.")
    reset_to_upload()
    st.rerun()