import json
import os
import streamlit as st
from config import CHAT_HISTORIES_DIR

def save_chat_history(session_id, messages_list, display_name_to_set=None):
    """Lưu lịch sử chat và tên hiển thị vào file JSON theo session_id."""
    file_path = os.path.join(CHAT_HISTORIES_DIR, f"{session_id}.json")
    
    current_display_name = session_id 
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "display_name" in data:
                    current_display_name = data["display_name"]
        except Exception:
            pass 
            
    final_display_name = display_name_to_set if display_name_to_set is not None else current_display_name
    chat_data_to_save = {
        "display_name": final_display_name,
        "messages": messages_list
    }
    
    try:
        os.makedirs(CHAT_HISTORIES_DIR, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chat_data_to_save, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Lỗi khi lưu lịch sử chat cho '{session_id}': {e}")
        return False

def load_chat_history(session_id):
    """Tải lịch sử chat và tên hiển thị từ file JSON theo session_id.
       Trả về: (list_messages, display_name)
    """
    file_path = os.path.join(CHAT_HISTORIES_DIR, f"{session_id}.json")
    
    if not os.path.exists(file_path):
        return [], session_id 
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            messages = data.get("messages", [])
            display_name = data.get("display_name", session_id)
            return messages, display_name
        elif isinstance(data, list):
            return data, session_id
        else:
            st.warning(f"Định dạng file lịch sử chat không đúng cho '{session_id}'.")
            return [], session_id
            
    except Exception as e:
        st.error(f"Lỗi khi tải lịch sử chat cho '{session_id}': {e}")
        return [], session_id

def list_chat_sessions():
    """Liệt kê các session đã lưu, trả về list của (session_id, display_name).
       Sắp xếp theo display_name.
    """
    sessions_info = []
    if not os.path.exists(CHAT_HISTORIES_DIR):
        os.makedirs(CHAT_HISTORIES_DIR)
        return []

    for fname in os.listdir(CHAT_HISTORIES_DIR):
        if fname.endswith(".json"):
            session_id = fname[:-5]
            file_path = os.path.join(CHAT_HISTORIES_DIR, fname)
            display_name = session_id
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "display_name" in data:
                        display_name = data["display_name"]
            except Exception:
                pass
            sessions_info.append((session_id, display_name))
            
    sessions_info.sort(key=lambda item: item[1].lower())
    return sessions_info
