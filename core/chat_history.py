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
    
    # Debug: Kiểm tra nguồn trước khi lưu
    for idx, msg in enumerate(messages_list):
        if msg.get("role") == "assistant" and "sources" in msg:
            print(f"\n=== DEBUG SAVE SOURCES MSG #{idx} ===")
            print(f"Number of sources before saving: {len(msg['sources'])}")
            if len(msg['sources']) > 0:
                print(f"First source before saving: {msg['sources'][0]}")
                
            # Đảm bảo rằng mọi nguồn đều ở định dạng dictionary
            if len(msg['sources']) > 0:
                for i, src in enumerate(msg['sources']):
                    if not isinstance(src, dict):
                        print(f"[chat_history] Warning: nguồn #{i} không phải dictionary, đang chuyển đổi")
                        # Nếu nguồn không phải dictionary, chuyển đổi hoặc thay thế
                        if hasattr(src, 'metadata') and hasattr(src, 'page_content'):
                            # Document object
                            msg['sources'][i] = {
                                "source": src.metadata.get("source", "N/A"),
                                "chunk_id": src.metadata.get("chunk_id", "N/A"),
                                "content": src.page_content
                            }
                        else:
                            # Không nhận dạng được
                            msg['sources'][i] = {
                                "source": "Không xác định",
                                "chunk_id": f"unknown-{i}",
                                "content": str(src)
                            }
            print("=== END DEBUG SAVE SOURCES ===\n")

    chat_data_to_save = {
        "display_name": final_display_name,
        "messages": messages_list
    }
    
    # Hỗ trợ cho JSON serialization đặc biệt
    def json_serializer(obj):
        # Xử lý các loại đối tượng đặc biệt
        if hasattr(obj, 'metadata') and hasattr(obj, 'page_content'):
            # Document object
            return {
                "source": obj.metadata.get("source", "N/A"),
                "chunk_id": obj.metadata.get("chunk_id", "N/A"),
                "content": obj.page_content
            }
        # Fallback
        return str(obj)
    
    try:
        os.makedirs(CHAT_HISTORIES_DIR, exist_ok=True) # Đảm bảo thư mục tồn tại
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chat_data_to_save, f, ensure_ascii=False, indent=2, default=json_serializer)
        print(f"[chat_history] Đã lưu lịch sử chat '{final_display_name}' thành công.")
        return True
    except Exception as e:
        print(f"[chat_history] Lỗi khi lưu lịch sử chat: {e}")
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
            
            # Đảm bảo rằng mỗi tin nhắn assistant có trường sources, ngay cả khi trống
            for msg in messages:
                if msg.get("role") == "assistant" and "sources" not in msg:
                    msg["sources"] = []
                elif msg.get("role") == "assistant" and not isinstance(msg.get("sources"), list):
                    # Nếu sources không phải là list, chuyển đổi thành list rỗng
                    print(f"[chat_history] Warning: sources không phải list, đang sửa")
                    msg["sources"] = []
            
            # Debug: Kiểm tra nguồn sau khi tải
            for idx, msg in enumerate(messages):
                if msg.get("role") == "assistant" and "sources" in msg:
                    print(f"\n=== DEBUG LOAD SOURCES MSG #{idx} ===")
                    print(f"Number of sources after loading: {len(msg['sources'])}")
                    if len(msg['sources']) > 0:
                        print(f"First source after loading: {msg['sources'][0]}")
                    print("=== END DEBUG LOAD SOURCES ===\n")
                    
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
