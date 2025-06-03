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
    error_files = []
    if files:
        for f in files:
            if f.size == 0:
                error_files.append((f.name, "File rỗng"))
            elif not (f.name.lower().endswith(".txt") or f.name.lower().endswith(".pdf")):
                error_files.append((f.name, "Định dạng không hỗ trợ"))
            else:
                valid_files.append(f)
    if error_files:
        st.warning("Một số file không hợp lệ sẽ bị bỏ qua:")
        for fname, reason in error_files:
            st.write(f"- {fname}: {reason}")
    col1, col2 = st.columns([1,1])
    start_clicked = col1.button("🚀 Bắt đầu", disabled=not valid_files)
    stop_clicked = col2.button("⏹️ Dừng", disabled=True)
    return valid_files, error_files, start_clicked, stop_clicked

def processing_screen(uploaded_files):
    """
    Giao diện khi đang xử lý file: chỉ hiện danh sách file và nút Dừng.
    """
    st.markdown("### ⏳ Đang xử lý tài liệu...")
    st.write("Các file đã upload:")
    for f in uploaded_files:
        st.write(f"- {f.name}")
    stop_clicked = st.button("⏹️ Dừng")
    return stop_clicked

def chat_screen(messages, bot_answering):
    """
    Giao diện chat: hiển thị lịch sử chat, nhập câu hỏi, hoặc nút Dừng khi bot đang trả lời.
    Trả về: prompt mới, send_clicked, stop_clicked
    """
    st.markdown("---")
    st.markdown("### 💬 Lịch sử trò chuyện")

    # Khu vực lịch sử chat, cần padding ở dưới để không bị che bởi thanh input cố định
    st.markdown("<div class='chat-history-area'>", unsafe_allow_html=True)
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Xem nguồn tham khảo"):
                    for i, source in enumerate(message["sources"]):
                        st.caption(f"Nguồn {i+1} (Từ: {source['source']}, Chunk ID: {source['chunk_id']})")
                        st.markdown(source['content'][:300] + "...")
    st.markdown("</div>", unsafe_allow_html=True) # Kết thúc chat-history-area
    
    prompt = None
    send_clicked = False
    stop_clicked = False

    # Thanh input cố định ở dưới bằng position:fixed
    st.markdown("<div class='fixed-chat-input-bar'>", unsafe_allow_html=True)
    col1, col2 = st.columns([9, 1]) 
    with col1:
        prompt = st.text_input(
            "Nhập câu hỏi...",
            key="custom_chat_input", 
            disabled=bot_answering,
            label_visibility="collapsed"
        )
    with col2:
        if bot_answering:
            stop_clicked = st.button("⏹️", key="stop_btn_chat", use_container_width=True)
        else:
            send_clicked = st.button("➤", key="send_btn_chat", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True) # Kết thúc fixed-chat-input-bar
    
    return prompt, send_clicked, stop_clicked

# (Code for file uploader, chat display, user input will go here) 