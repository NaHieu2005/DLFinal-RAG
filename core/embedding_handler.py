from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
import pickle
from config import embedding_model_name, VECTOR_STORES_DIR, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, EMBEDDING_DEVICE
import time

# Cache để không phải tạo lại embedding model mỗi lần
@st.cache_resource
def get_embedding_model():
    try:
        print(f"[embedding_handler] Bắt đầu khởi tạo HuggingFaceEmbeddings với device={EMBEDDING_DEVICE}...")
        return HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': EMBEDDING_DEVICE}
        )
    except Exception as e:
        print(f"[embedding_handler] Lỗi khi tải embedding model: {e}")
        st.error(f"Lỗi khi tải embedding model: {e}")
        return None

def generate_session_id(file_names):
    base = '+'.join([f.split('.')[0] for f in file_names])
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return f"{base}_{timestamp}"

# Hàm mới cho phép tạo embeddings theo batch để tăng tốc 
@st.cache_data(ttl=3600)  # Cache kết quả embedding với thời hạn 1 giờ
def create_embeddings_in_batches(texts, embedding_model, batch_size=32):
    """Tạo embeddings theo batch để tăng hiệu suất."""
    print(f"[embedding_handler] Tạo embeddings cho {len(texts)} văn bản theo batch_size={batch_size}")
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embedding_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        
        # Cập nhật tiến độ
        if i % (batch_size * 10) == 0 or i + batch_size >= len(texts):
            current_batch = (i + batch_size) // batch_size
            print(f"[embedding_handler] Đang tạo embeddings: {current_batch}/{total_batches} batches xong ({(current_batch/total_batches)*100:.1f}%)")
    
    return all_embeddings

# Cache để không phải tạo lại vector store mỗi lần tương tác cho cùng một bộ tài liệu
# Lưu ý: _documents và _embedding_model_name giờ đây nên được thay thế bằng một ID định danh cho bộ tài liệu đó
# Tuy nhiên, vì chúng ta sẽ lưu và tải vector store, việc cache get_vector_store có thể không còn quá quan trọng
# nếu việc tải từ disk đủ nhanh. Hiện tại vẫn giữ cache cho việc tạo mới.
def create_vector_store_from_documents(_documents, _embedding_model_instance):
    """Tạo FAISS vector store từ các document đã chia chunk."""
    print("[embedding_handler] Bắt đầu tạo vector store từ documents (không cache resource)...")
    if _documents and _embedding_model_instance:
        try:
            # Tách texts và metadatas
            texts = [doc.page_content for doc in _documents]
            metadatas = [doc.metadata for doc in _documents]
            
            # Tạo embeddings theo batch
            embeddings = create_embeddings_in_batches(texts, _embedding_model_instance)
            
            # Tạo vector store từ embeddings đã tính toán trước
            vector_store_instance = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                embedding=_embedding_model_instance,
                metadatas=metadatas
            )
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
    bm25_retriever.k = 10  # Tăng số lượng kết quả trả về
    
    # Khởi tạo FAISS vector store cho child chunks
    vectorstore = FAISS.from_documents(child_chunks, embedding_model_instance)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Tăng k lên 10
    
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
    
    # Thêm parent documents vào docstore TRƯỚC khi khởi tạo ParentDocumentRetriever
    for p_doc in parent_chunks:
        if "parent_id" in p_doc.metadata:
            docstore.mset([(p_doc.metadata["parent_id"], p_doc)])
    
    # Khởi tạo ParentDocumentRetriever
    try:
        # Tăng search_kwargs để tìm thấy nhiều kết quả hơn
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,  # Sử dụng vectorstore cho child chunks
            docstore=docstore,  # Lưu trữ parent chunks
            child_splitter=child_splitter,  # Bắt buộc phải có
            search_kwargs={"k": 15},  # Tăng k để đảm bảo tìm thấy kết quả
            child_ids_key="parent_id"  # Khóa liên kết giữa child và parent
        )
        
        # Patch phương thức invoke nếu cần
        parent_retriever = patch_retriever_invoke(parent_retriever)
        
        # Fallback nếu ParentDocumentRetriever gặp vấn đề
        print(f"[embedding_handler] Đã khởi tạo advanced retriever với {len(parent_chunks)} parent chunks và {len(child_chunks)} child chunks.")
        return parent_retriever, vectorstore
    except Exception as e:
        print(f"[embedding_handler] Lỗi khi tạo ParentDocumentRetriever: {e}")
        print(f"[embedding_handler] Sử dụng simple retriever thay thế")
        # Trả về FAISS retriever đơn giản nếu gặp lỗi
        return faiss_retriever, vectorstore

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
        # Log chi tiết để dễ debug
        vs_path = os.path.join(VECTOR_STORES_DIR, vs_id)
        parent_path = os.path.join(VECTOR_STORES_DIR, vs_id, "parent_chunks.pkl")
        print(f"[embedding_handler] Kiểm tra VS path: {os.path.exists(vs_path)}, Parent path: {os.path.exists(parent_path)}")
        
        # Fallback to simple retriever
        simple_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        return patch_retriever_invoke(simple_retriever)
    
    # Khởi tạo docstore và điền parent documents
    docstore = InMemoryStore()
    valid_parents = 0
    
    for p_doc in parent_chunks:
        if "parent_id" in p_doc.metadata:
            docstore.mset([(p_doc.metadata["parent_id"], p_doc)])
            valid_parents += 1
        else:
            print(f"[embedding_handler] Cảnh báo: Tìm thấy parent chunk không có parent_id trong metadata")
    
    if valid_parents == 0:
        print(f"[embedding_handler] Không có parent chunk nào có parent_id hợp lệ. Sẽ sử dụng retriever đơn giản.")
        simple_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        return patch_retriever_invoke(simple_retriever)
    
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
        
        # Patch phương thức invoke nếu cần
        parent_retriever = patch_retriever_invoke(parent_retriever)
        
        print(f"[embedding_handler] Đã tái tạo thành công ParentDocumentRetriever với {valid_parents}/{len(parent_chunks)} parent chunks")
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
    if documents_info is None:
        print(f"[embedding_handler] documents_info là None, không thể tạo vector store mới cho ID: {vs_id}")
        return None, None

    # Xử lý định dạng parent-child
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
            print(f"[embedding_handler] parent_chunks hoặc child_chunks rỗng cho ID: {vs_id}")
            return None, None
    # Xử lý định dạng tài liệu truyền thống
    elif isinstance(documents_info, list):
        documents = documents_info
        if documents:
            print(f"[embedding_handler] Đang tạo vector store thông thường cho ID: {vs_id}...")
            vector_store = create_vector_store_from_documents(documents, embedding_model_instance)
            if vector_store and save:
                save_vector_store(vector_store, vs_id)
            return vector_store.as_retriever(), vs_id
        else:
            print(f"[embedding_handler] documents là list rỗng cho ID: {vs_id}")
            return None, None
    else:
        print(f"[embedding_handler] documents_info không phải tuple (parent_chunks, child_chunks) hoặc list documents. Kiểu dữ liệu: {type(documents_info)}")
        return None, None

# Patch function that can be reused
def patch_retriever_invoke(retriever):
    """Thêm phương thức invoke cho retriever nếu chưa có."""
    if not hasattr(retriever, 'invoke') or callable(getattr(retriever, 'invoke', None)) is False:
        print("[embedding_handler] Patch phương thức invoke cho retriever")
        
        def patched_invoke(self, query, *args, **kwargs):
            if hasattr(self, 'get_relevant_documents'):
                return self.get_relevant_documents(query)
            else:
                print("[embedding_handler] Lỗi: Không tìm thấy phương thức get_relevant_documents")
                return []
                
        import types
        retriever.invoke = types.MethodType(patched_invoke, retriever)
    
    return retriever

# (Code for get_vector_store, save_vector_store, load_vector_store will go here) 