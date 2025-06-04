import streamlit as st

def file_upload_screen(uploaded_files=None):
    """
    Giao diện chỉ cho upload file và nút Bắt đầu.
    """
    st.markdown("### 📄 Tải lên tài liệu (.txt, .pdf)")
    files = st.file_uploader(
        "Chọn một hoặc nhiều file:",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key="file_uploader",
        disabled=False
    )
    valid_files = []
    error_files = {}
    if files:
        for f in files:
            if f.size == 0:
                error_files[f.name] = "File rỗng"
            else:
                valid_files.append(f)

    start_clicked = st.button("🚀 Bắt đầu", disabled=not valid_files, use_container_width=True)
    return valid_files, error_files, start_clicked, False

def processing_screen(uploaded_files):
    """
    Giao diện khi đang xử lý file: chỉ hiện danh sách file và nút Dừng.
    """
    st.markdown("### ⏳ Đang xử lý tài liệu...")
    if uploaded_files:
        st.write("Các file đã upload:")
        for f in uploaded_files:
            st.write(f"- {f.name}")
    else:
        st.warning("Không có file nào được cung cấp để xử lý.")
    stop_clicked = st.button("⏹️ Dừng xử lý", use_container_width=True)
    return stop_clicked

def chat_screen(messages, bot_answering):
    """
    Giao diện chat: thanh nhập liệu cố định.
    - Khi bot trả lời: Input bị vô hiệu hóa, nút "Dừng".
    - Khi bot không trả lời: Form với input và nút "Gửi".
    Trả về: prompt mới, send_triggered (từ form submit), stop_button_clicked_in_ui
    """
    prompt_value = ""
    send_triggered = False
    stop_button_clicked_in_ui = False

    # Khởi tạo các giá trị session_state nếu chưa có
    if 'chat_input_value' not in st.session_state:
        st.session_state.chat_input_value = ""
    if 'stop_action_requested' not in st.session_state: # Đã có ở app.py nhưng để chắc chắn
        st.session_state.stop_action_requested = False

    # Container cho thanh input cố định
    st.markdown("<div class='fixed-chat-input-bar'>", unsafe_allow_html=True)

    if bot_answering:
        # Bot đang trả lời: Hiển thị input bị vô hiệu hóa và nút Dừng (không có form)
        col_text, col_stop = st.columns([0.85, 0.15]) # Tỷ lệ cho input và nút dừng
        with col_text:
            st.text_input(
                "chat_input_disabled", # key
                value="Bot đang trả lời...",
                disabled=True,
                label_visibility="collapsed",
                placeholder="Bot đang trả lời..."
            )
        with col_stop:
            if st.button("⏹️", key="chat_stop_button_active", use_container_width=True, help="Dừng tạo câu trả lời"):
                st.session_state.stop_action_requested = True
                stop_button_clicked_in_ui = True # Để app.py biết nút này vừa được nhấn (nếu cần)
    else:
        # Bot không trả lời: Hiển thị form với input và nút Gửi
        with st.form(key="chat_form_active"):
            col_input, col_send = st.columns([0.85, 0.15]) # Tỷ lệ cho input và nút gửi
            with col_input:
                current_prompt = st.text_input(
                    "chat_input_active", # key
                    value=st.session_state.chat_input_value,
                    label_visibility="collapsed",
                    placeholder="Bạn hỏi gì đi...",
                    key="chat_text_input_active_main" # Đổi key để tránh xung đột nếu có
                )
            with col_send:
                if st.form_submit_button("➤", use_container_width=True, help="Gửi câu hỏi"):
                    prompt_value = current_prompt
                    st.session_state.chat_input_value = "" # Xóa input sau khi gửi
                    send_triggered = True
            
            # Nếu form không được submit trong lần render này, giữ lại giá trị input
            # Điều này quan trọng nếu có rerun mà không phải do submit form (ví dụ: click nút khác)
            if not send_triggered:
                st.session_state.chat_input_value = current_prompt

    st.markdown("</div>", unsafe_allow_html=True) # Kết thúc fixed-chat-input-bar
    
    return prompt_value, send_triggered, stop_button_clicked_in_ui

# (Code for file uploader, chat display, user input will go here) 