from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
import hashlib
import pickle
from config import embedding_model_name, VECTOR_STORES_DIR, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
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
def create_vector_store_from_documents(_documents, _embedding_model_instance):
    """Tạo FAISS vector store từ các document đã chia chunk."""
    print("[embedding_handler] Bắt đầu tạo vector store từ documents (không cache resource)...")
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

def save_parent_chunks(parent_chunks, vs_id):
    """Lưu parent chunks vào disk dưới dạng file pickle."""
    if parent_chunks and vs_id:
        directory = os.path.join(VECTOR_STORES_DIR, vs_id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        save_path = os.path.join(directory, "parent_chunks.pkl")
        try:
            with open(save_path, "wb") as f:
                pickle.dump(parent_chunks, f)
            print(f"[embedding_handler] Đã lưu {len(parent_chunks)} parent chunks vào: {save_path}")
            return save_path
        except Exception as e:
            print(f"[embedding_handler] Lỗi khi lưu parent chunks: {e}")
            st.error(f"Lỗi khi lưu parent chunks: {e}")
            return None
    print("[embedding_handler] Không có parent chunks hoặc vs_id để lưu.")
    return None

def load_parent_chunks(vs_id):
    """Tải parent chunks từ disk."""
    load_path = os.path.join(VECTOR_STORES_DIR, vs_id, "parent_chunks.pkl")
    if os.path.exists(load_path):
        try:
            with open(load_path, "rb") as f:
                parent_chunks = pickle.load(f)
            print(f"[embedding_handler] Đã tải {len(parent_chunks)} parent chunks từ: {load_path}")
            return parent_chunks
        except Exception as e:
            print(f"[embedding_handler] Lỗi khi tải parent chunks từ '{load_path}': {e}")
            st.error(f"Lỗi khi tải parent chunks từ '{load_path}': {e}")
            return None
    print(f"[embedding_handler] Không tìm thấy parent chunks tại: {load_path}")
    return None

def create_advanced_retriever(parent_chunks, child_chunks, embedding_model_instance):
    """
    Tạo một hybrid retriever kết hợp BM25 và FAISS với khả năng truy xuất parent document.
    """
    print("[embedding_handler] Bắt đầu khởi tạo advanced retriever...")
    
    # Khởi tạo store cho parent documents
    docstore = InMemoryStore()
    
    # Khởi tạo BM25Retriever cho child chunks
    bm25_retriever = BM25Retriever.from_documents(child_chunks)
    bm25_retriever.k = 5  # Số lượng kết quả trả về
    
    # Khởi tạo FAISS vector store cho child chunks
    vectorstore = FAISS.from_documents(child_chunks, embedding_model_instance)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Hybrid retriever kết hợp BM25 và FAISS với trọng số ngang nhau
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    
    # Tạo child_splitter là bắt buộc cho ParentDocumentRetriever
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    
    # Khởi tạo ParentDocumentRetriever
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,  # Sử dụng vectorstore cho child chunks
        docstore=docstore,  # Lưu trữ parent chunks
        child_splitter=child_splitter,  # Bắt buộc phải có
        search_kwargs={"k": 5},
        child_ids_key="parent_id"  # Khóa liên kết giữa child và parent
    )
    
    # Thêm parent documents vào docstore
    for p_doc in parent_chunks:
        if "parent_id" in p_doc.metadata:
            docstore.mset([(p_doc.metadata["parent_id"], p_doc)])
    
    print(f"[embedding_handler] Đã khởi tạo advanced retriever với {len(parent_chunks)} parent chunks và {len(child_chunks)} child chunks.")
    
    return parent_retriever, vectorstore

def recreate_retriever_from_saved(vs_id, embedding_model_instance):
    """
    Tái tạo ParentDocumentRetriever từ vector store và parent chunks đã lưu.
    Sử dụng cho tải lại session.
    """
    print(f"[embedding_handler] Đang tái tạo advanced retriever cho session: {vs_id}")
    
    # Tải vector store đã lưu
    vectorstore = load_vector_store(vs_id, embedding_model_instance)
    if not vectorstore:
        print(f"[embedding_handler] Không thể tải vector store cho session: {vs_id}")
        return None
        
    # Tải parent chunks đã lưu
    parent_chunks = load_parent_chunks(vs_id)
    if not parent_chunks:
        print(f"[embedding_handler] Không thể tải parent chunks cho session: {vs_id}")
        print(f"[embedding_handler] Sẽ sử dụng retriever đơn giản thay thế.")
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Khởi tạo docstore và điền parent documents
    docstore = InMemoryStore()
    for p_doc in parent_chunks:
        if "parent_id" in p_doc.metadata:
            docstore.mset([(p_doc.metadata["parent_id"], p_doc)])
    
    # Tạo child_splitter là bắt buộc cho ParentDocumentRetriever
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    
    # Tạo ParentDocumentRetriever
    try:
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            search_kwargs={"k": 5},
            child_ids_key="parent_id"
        )
        print(f"[embedding_handler] Đã tái tạo thành công ParentDocumentRetriever với {len(parent_chunks)} parent chunks")
        return parent_retriever
    except Exception as e:
        print(f"[embedding_handler] Lỗi khi tái tạo ParentDocumentRetriever: {e}")
        print(f"[embedding_handler] Sẽ sử dụng retriever đơn giản thay thế.")
        return vectorstore.as_retriever(search_kwargs={"k": 5})

def get_or_create_vector_store(p_session_id, documents_info, embedding_model_instance, save=True):
    """
    Lấy vector store đã có hoặc tạo mới và lưu lại.
    Tham số documents_info có thể là tuple (parent_chunks, child_chunks) hoặc documents truyền thống.
    """
    if not p_session_id:
        print("[embedding_handler] p_session_id không được cung cấp cho get_or_create_vector_store.")
        st.error("Lỗi hệ thống: Không có ID cho vector store.")
        return None, None

    vs_id = p_session_id  # Sử dụng session_id được truyền vào làm vs_id
    print(f"[embedding_handler] Đang lấy hoặc tạo vector store với id: {vs_id}")
    
    # Kiểm tra xem có retriever để tái tạo không
    retriever = recreate_retriever_from_saved(vs_id, embedding_model_instance)
    if retriever:
        print(f"[embedding_handler] Đã tái tạo retriever cho ID: {vs_id}")
        return retriever, vs_id
    
    # Kiểm tra định dạng của documents_info
    if isinstance(documents_info, tuple) and len(documents_info) == 2:
        parent_chunks, child_chunks = documents_info
        if parent_chunks and child_chunks:
            print(f"[embedding_handler] Đang tạo advanced retriever mới cho ID: {vs_id}...")
            parent_retriever, vector_store = create_advanced_retriever(
                parent_chunks, child_chunks, embedding_model_instance
            )
            
            if vector_store and save:
                # Lưu cả vector store và parent chunks
                save_path = save_vector_store(vector_store, vs_id)
                parent_path = save_parent_chunks(parent_chunks, vs_id)
                if save_path and parent_path:
                    print(f"[embedding_handler] Đã lưu đầy đủ vector store và parent chunks tại: {save_path}")
                else:
                    print(f"[embedding_handler] Có lỗi khi lưu vector store hoặc parent chunks cho ID: {vs_id}")
            
            # Trả về parent_retriever để sử dụng cho Parent Document Retrieval
            return parent_retriever, vs_id
    else:
        # Xử lý trường hợp documents truyền thống (để tương thích ngược)
        documents = documents_info
        if documents:
            print(f"[embedding_handler] Đang tạo vector store thông thường cho ID: {vs_id}...")
            vector_store = FAISS.from_documents(documents, embedding_model_instance)
            if vector_store and save:
                save_vector_store(vector_store, vs_id)
            return vector_store, vs_id
    
    print(f"[embedding_handler] Không thể tạo vector store cho ID: {vs_id}")
    return None, None

# (Code for get_vector_store, save_vector_store, load_vector_store will go here) 