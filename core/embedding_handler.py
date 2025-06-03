from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import os
import hashlib
from config import embedding_model_name, VECTOR_STORES_DIR
import time

# Cache để không phải tạo lại embedding model mỗi lần
@st.cache_resource
def get_embedding_model():
    try:
        print("[embedding_handler] Bắt đầu khởi tạo HuggingFaceEmbeddings...")
        return HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'} # Sửa thành 'cpu' để tránh lỗi CUDA
        )
    except Exception as e:
        print(f"[embedding_handler] Lỗi khi tải embedding model: {e}")
        st.error(f"Lỗi khi tải embedding model: {e}")
        return None

def generate_session_id(file_names):
    base = '+'.join([f.split('.')[0] for f in file_names])
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return f"{base}_{timestamp}"

# Cache để không phải tạo lại vector store mỗi lần tương tác cho cùng một bộ tài liệu
# Lưu ý: _documents và _embedding_model_name giờ đây nên được thay thế bằng một ID định danh cho bộ tài liệu đó
# Tuy nhiên, vì chúng ta sẽ lưu và tải vector store, việc cache get_vector_store có thể không còn quá quan trọng
# nếu việc tải từ disk đủ nhanh. Hiện tại vẫn giữ cache cho việc tạo mới.
@st.cache_resource(show_spinner="Đang tạo cơ sở tri thức từ tài liệu...")
def create_vector_store_from_documents(_documents, _embedding_model_instance):
    """Tạo FAISS vector store từ các document đã chia chunk."""
    print("[embedding_handler] Bắt đầu tạo vector store từ documents...")
    if _documents and _embedding_model_instance:
        try:
            vector_store_instance = FAISS.from_documents(_documents, _embedding_model_instance)
            print("[embedding_handler] Đã tạo xong vector store.")
            return vector_store_instance
        except Exception as e:
            print(f"[embedding_handler] Lỗi khi tạo vector store: {e}")
            st.error(f"Lỗi khi tạo vector store: {e}")
            return None
    print("[embedding_handler] Không có documents hoặc embedding model instance.")
    return None

def save_vector_store(vector_store_instance, vs_id):
    """Lưu FAISS vector store vào disk."""
    if vector_store_instance and vs_id:
        save_path = os.path.join(VECTOR_STORES_DIR, vs_id)
        try:
            # FAISS.save_local cần một thư mục, không phải file
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            vector_store_instance.save_local(save_path)
            print(f"[embedding_handler] Đã lưu vector store vào: {save_path}")
            st.success(f"Cơ sở tri thức đã được lưu vào: {save_path}")
            return save_path
        except Exception as e:
            print(f"[embedding_handler] Lỗi khi lưu vector store: {e}")
            st.error(f"Lỗi khi lưu vector store: {e}")
            return None
    print("[embedding_handler] Không có vector store instance hoặc vs_id để lưu.")
    return None

def load_vector_store(vs_id, _embedding_model_instance):
    """Tải FAISS vector store từ disk."""
    load_path = os.path.join(VECTOR_STORES_DIR, vs_id)
    if os.path.exists(load_path) and _embedding_model_instance:
        try:
            print(f"[embedding_handler] Đang tải vector store từ: {load_path}")
            return FAISS.load_local(load_path, _embedding_model_instance, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"[embedding_handler] Lỗi khi tải vector store từ '{load_path}': {e}")
            st.error(f"Lỗi khi tải vector store từ '{load_path}': {e}. Có thể cần phải tạo lại.")
            return None
    print(f"[embedding_handler] Không tìm thấy vector store tại: {load_path}")
    return None

def get_or_create_vector_store(p_session_id, documents, embedding_model_instance, save=True):
    """
    Lấy vector store đã có hoặc tạo mới và lưu lại.
    Sử dụng p_session_id được cung cấp để định danh vector store.
    """
    if not p_session_id:
        print("[embedding_handler] p_session_id không được cung cấp cho get_or_create_vector_store.")
        st.error("Lỗi hệ thống: Không có ID cho vector store.")
        return None, None

    vs_id = p_session_id # Sử dụng session_id được truyền vào làm vs_id
    print(f"[embedding_handler] Đang lấy hoặc tạo vector store với id: {vs_id}")
    vector_store = load_vector_store(vs_id, embedding_model_instance)

    if vector_store:
        print(f"[embedding_handler] Đã tải vector store có sẵn cho ID: {vs_id}")
        return vector_store, vs_id
    
    if documents:
        print(f"[embedding_handler] Đang tạo vector store mới cho ID: {vs_id}...")
        vector_store = create_vector_store_from_documents(documents, embedding_model_instance)
        if vector_store:
            if save: # Chỉ lưu nếu save là True
                save_path = save_vector_store(vector_store, vs_id)
                if save_path:
                    print(f"[embedding_handler] Đã tạo và lưu vector store mới tại: {save_path}")
                else:
                    print(f"[embedding_handler] Tạo vector store thành công nhưng không lưu được cho ID: {vs_id}")
                    st.error(f"Tạo vector store thành công nhưng không lưu được cho ID: {vs_id}")
                    # Trả về vector_store nhưng không có vs_id_saved để báo hiệu lưu thất bại? Hoặc trả về None?
                    # Hiện tại, nếu save thất bại, vẫn trả về vector_store và vs_id
            else:
                print(f"[embedding_handler] Đã tạo vector store mới cho ID: {vs_id} (không lưu).")
            return vector_store, vs_id
        else:
            print(f"[embedding_handler] Không thể tạo cơ sở tri thức cho ID: {vs_id}.")
            st.error(f"Không thể tạo cơ sở tri thức cho ID: {vs_id}.")
            return None, None
    
    print(f"[embedding_handler] Không có tài liệu để xử lý (cho ID: {vs_id}) hoặc không thể tải/tạo cơ sở tri thức.")
    st.warning(f"Không có tài liệu để xử lý (cho ID: {vs_id}) hoặc không thể tải/tạo cơ sở tri thức.")
    return None, None

# (Code for get_vector_store, save_vector_store, load_vector_store will go here) 