import json
import os
import streamlit as st
from config import CHAT_HISTORIES_DIR

def save_chat_history(session_id, messages):
    """Lưu lịch sử chat vào file JSON theo session_id."""
    import json
    import os
    file_path = os.path.join(CHAT_HISTORIES_DIR, f"{session_id}.json")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Lỗi khi lưu lịch sử chat: {e}")
        return False

def load_chat_history(session_id):
    """Tải lịch sử chat từ file JSON theo session_id."""
    import json
    import os
    file_path = os.path.join(CHAT_HISTORIES_DIR, f"{session_id}.json")
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        return messages
    except Exception as e:
        st.error(f"Lỗi khi tải lịch sử chat: {e}")
        return []

def list_chat_sessions():
    """Liệt kê các session_id đã lưu (dựa trên file trong chat_histories)."""
    import os
    sessions = []
    for fname in os.listdir(CHAT_HISTORIES_DIR):
        if fname.endswith(".json"):
            sessions.append(fname[:-5]) # Bỏ .json
    return sessions
