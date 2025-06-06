import json
import os
import streamlit as st
from config import CHAT_HISTORIES_DIR

def save_chat_history(session_id, messages_list, display_name_to_set=None):
    """Lưu lịch sử chat và tên hiển thị vào file JSON theo session_id."""
    file_path = os.path.join(CHAT_HISTORIES_DIR, f"{session_id}.json")
    
    current_display_name = session_id # Mặc định nếu file mới hoặc không có display_name cũ
    
    # Nếu file đã tồn tại, thử đọc display_name hiện tại
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "display_name" in data:
                    current_display_name = data["display_name"]
        except Exception:
            pass # Bỏ qua nếu không đọc được file cũ hoặc không phải JSON
            
    # Ưu tiên display_name mới nếu được cung cấp, nếu không dùng cái hiện tại (hoặc session_id)
    final_display_name = display_name_to_set if display_name_to_set is not None else current_display_name
    
    chat_data_to_save = {
        "display_name": final_display_name,
        "messages": messages_list
    }
    
    try:
        os.makedirs(CHAT_HISTORIES_DIR, exist_ok=True) # Đảm bảo thư mục tồn tại
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chat_data_to_save, f, ensure_ascii=False, indent=2)
        # st.toast(f"Lịch sử chat '{final_display_name}' đã được lưu.") # Có thể thêm thông báo ngầm
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
        # Trả về messages rỗng và session_id làm display_name mặc định nếu file không tồn tại
        return [], session_id 
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            messages = data.get("messages", [])
            display_name = data.get("display_name", session_id) # Mặc định là session_id nếu không có
            return messages, display_name
        elif isinstance(data, list): # Xử lý trường hợp file cũ chỉ chứa list messages
            return data, session_id
        else: # Trường hợp không nhận dạng được
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
            session_id = fname[:-5] # Bỏ .json
            file_path = os.path.join(CHAT_HISTORIES_DIR, fname)
            display_name = session_id # Mặc định
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "display_name" in data:
                        display_name = data["display_name"]
            except Exception:
                pass # Nếu lỗi đọc file hoặc JSON, vẫn dùng session_id làm display_name
            sessions_info.append((session_id, display_name))
            
    # Sắp xếp theo display_name (phần tử thứ 2 của tuple), không phân biệt chữ hoa thường
    sessions_info.sort(key=lambda item: item[1].lower())
    return sessions_info
