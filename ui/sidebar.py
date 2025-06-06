import streamlit as st
from core.chat_history import list_chat_sessions

def sidebar():
    st.header("📁 Quản lý Chat")
    new_chat = st.button("🆕 New Chat")
    st.markdown("---")
    st.subheader("💬 Lịch sử Chat")
    sessions = list_chat_sessions()
    selected_session = None
    if sessions:
        # Sắp xếp session theo display_name hoặc session_id nếu cần
        # sessions.sort(key=lambda item: item[1]) # Ví dụ sắp xếp theo display_name
        
        for item in sessions: # item có thể là session_id hoặc (session_id, display_name)
            session_id_to_use = ""
            display_name_to_show = ""

            if isinstance(item, tuple) and len(item) == 2:
                session_id_to_use = item[0]
                display_name_to_show = item[1]
            elif isinstance(item, str): # Trường hợp list_chat_sessions chưa được cập nhật
                session_id_to_use = item
                display_name_to_show = item
            else:
                continue # Bỏ qua nếu định dạng không đúng

            if st.button(f"🗂️ {display_name_to_show}", key=f"chat_{session_id_to_use}", help=f"ID: {session_id_to_use}", use_container_width=True):
                selected_session = session_id_to_use
    else:
        st.info("Chưa có lịch sử chat nào.")
    return new_chat, selected_session 