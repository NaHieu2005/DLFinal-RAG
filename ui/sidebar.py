import streamlit as st
from core.chat_history import list_chat_sessions, load_chat_history, save_chat_history
from core.embedding_handler import VECTOR_STORES_DIR
import os

def sidebar():
    st.header("📁 Quản lý Chat")
    new_chat = st.button("🆕 New Chat")
    st.markdown("---")
    st.subheader("💬 Lịch sử Chat")
    sessions = list_chat_sessions()
    selected_session = None
    if sessions:
        for session_id in sessions:
            col1, col2 = st.columns([8,1])
            with col1:
                if st.button(f"🗂️ {session_id}", key=f"chat_{session_id}"):
                    selected_session = session_id
            with col2:
                menu_clicked = st.button("☰", key=f"menu_{session_id}")
                if menu_clicked:
                    action = st.radio(f"Tùy chọn cho {session_id}", ["Đổi tên", "Xóa"], key=f"action_{session_id}")
                    if action == "Đổi tên":
                        new_name = st.text_input(f"Tên mới cho {session_id}", key=f"rename_{session_id}")
                        if st.button("Xác nhận đổi tên", key=f"confirm_rename_{session_id}"):
                            # Đổi tên file history và vector store
                            old_history = os.path.join("data/chat_histories", f"{session_id}.json")
                            new_history = os.path.join("data/chat_histories", f"{new_name}.json")
                            if os.path.exists(old_history):
                                os.rename(old_history, new_history)
                            old_vector = os.path.join(VECTOR_STORES_DIR, session_id)
                            new_vector = os.path.join(VECTOR_STORES_DIR, new_name)
                            if os.path.exists(old_vector):
                                os.rename(old_vector, new_vector)
                            st.success(f"Đã đổi tên thành {new_name}")
                            st.experimental_rerun()
                    elif action == "Xóa":
                        if st.button("Xác nhận xóa", key=f"confirm_delete_{session_id}"):
                            # Xóa file history và vector store
                            history_path = os.path.join("data/chat_histories", f"{session_id}.json")
                            vector_path = os.path.join(VECTOR_STORES_DIR, session_id)
                            if os.path.exists(history_path):
                                os.remove(history_path)
                            if os.path.exists(vector_path):
                                import shutil
                                shutil.rmtree(vector_path)
                            st.success(f"Đã xóa {session_id}")
                            st.experimental_rerun()
    else:
        st.info("Chưa có lịch sử chat nào.")
    return new_chat, selected_session 