from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain_core.runnables import RunnableLambda
import streamlit as st
import os
import pickle
import time
from config import embedding_model_name, VECTOR_STORES_DIR, device

@st.cache_resource
def get_embedding_model():
    """Khởi tạo và cache embedding model."""
    try:
        print("[embedding_handler] Bắt đầu khởi tạo HuggingFaceEmbeddings...")
        return HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device}
        )
    except Exception as e:
        print(f"[embedding_handler] Lỗi khi tải embedding model: {e}")
        st.error(f"Lỗi khi tải embedding model: {e}")
        return None

def generate_session_id(file_names):
    """Tạo session ID dựa trên tên file và timestamp."""
    base = '+'.join(sorted([f.split('.')[0] for f in file_names]))
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return f"{base}_{timestamp}"

def _save_pickle_data(data, vs_id, file_name):
    """Hàm chung để lưu dữ liệu dạng pickle."""
    if data and vs_id:
        directory = os.path.join(VECTOR_STORES_DIR, vs_id)
        os.makedirs(directory, exist_ok=True)
        save_path = os.path.join(directory, file_name)
        try:
            with open(save_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[embedding_handler] Đã lưu {len(data)} items vào: {save_path}")
            return True
        except Exception as e:
            print(f"[embedding_handler] Lỗi khi lưu '{file_name}': {e}")
            st.error(f"Lỗi khi lưu '{file_name}': {e}")
    return False

def _load_pickle_data(vs_id, file_name):
    """Hàm chung để tải dữ liệu dạng pickle."""
    load_path = os.path.join(VECTOR_STORES_DIR, vs_id, file_name)
    if os.path.exists(load_path):
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            print(f"[embedding_handler] Đã tải {len(data)} items từ: {load_path}")
            return data
        except Exception as e:
            print(f"[embedding_handler] Lỗi khi tải '{file_name}': {e}")
            st.error(f"Lỗi khi tải '{file_name}': {e}")
    return None

def save_vector_store(vector_store_instance, vs_id):
    """Lưu FAISS vector store vào disk."""
    save_path = os.path.join(VECTOR_STORES_DIR, vs_id)
    try:
        os.makedirs(save_path, exist_ok=True)
        vector_store_instance.save_local(save_path)
        print(f"[embedding_handler] Đã lưu vector store vào: {save_path}")
        return True
    except Exception as e:
        print(f"[embedding_handler] Lỗi khi lưu vector store: {e}")
        st.error(f"Lỗi khi lưu vector store: {e}")
    return False

def load_vector_store(vs_id, _embedding_model_instance):
    """Tải FAISS vector store từ disk."""
    load_path = os.path.join(VECTOR_STORES_DIR, vs_id)
    if os.path.exists(load_path):
        try:
            print(f"[embedding_handler] Đang tải vector store từ: {load_path}")
            return FAISS.load_local(load_path, _embedding_model_instance, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"[embedding_handler] Lỗi khi tải vector store từ '{load_path}': {e}")
    return None
    
def _get_parent_chunks_from_child_docs(child_docs, docstore):
    """
    Hàm helper của LCEL: Lấy các parent chunks từ các child docs đã được truy xuất.
    """
    parent_ids = {doc.metadata.get("parent_id") for doc in child_docs if "parent_id" in doc.metadata}
    if not parent_ids:
        return []
    
    retrieved_parents = docstore.mget(list(parent_ids))
    return [doc for doc in retrieved_parents if doc is not None]

def _build_lcel_retriever_chain(parent_chunks, child_chunks, embedding_model_instance, vectorstore=None):
    """
    Hàm lõi để xây dựng pipeline retriever theo kiến trúc của `test.py`.
    Hàm này được dùng cho cả việc tạo mới và tái tạo.
    """
    print("[embedding_handler] Bắt đầu xây dựng chuỗi retriever LCEL...")
    
    docstore = InMemoryStore()
    docstore.mset([(p_doc.metadata["parent_id"], p_doc) for p_doc in parent_chunks])
    print(f"[embedding_handler] Đã tạo docstore với {len(parent_chunks)} parent chunks.")

    bm25_retriever = BM25Retriever.from_documents(child_chunks)
    bm25_retriever.k = 10
    print("[embedding_handler] Đã khởi tạo BM25 Retriever.")

    if vectorstore is None:
        print("[embedding_handler] Đang tạo FAISS vectorstore mới từ child_chunks...")
        vectorstore = FAISS.from_documents(child_chunks, embedding_model_instance)
    else:
        print("[embedding_handler] Đang sử dụng FAISS vectorstore đã được tải.")
        
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    print("[embedding_handler] Đã khởi tạo FAISS Retriever.")
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    print("[embedding_handler] Đã khởi tạo Ensemble Retriever.")

    final_retriever_chain = ensemble_retriever | RunnableLambda(
        lambda docs: _get_parent_chunks_from_child_docs(docs, docstore)
    )
    print("[embedding_handler] Đã xây dựng thành công chuỗi retriever LCEL.")
    
    return final_retriever_chain, vectorstore


def recreate_retriever_from_saved(vs_id, embedding_model_instance):
    """
    Tái tạo Hybrid Parent Document Retriever từ các thành phần đã lưu.
    Luồng logic này giờ đây nhất quán với luồng tạo mới.
    """
    print(f"[embedding_handler] Đang tái tạo retriever cho session: {vs_id}")
    
    vectorstore = load_vector_store(vs_id, embedding_model_instance)
    parent_chunks = _load_pickle_data(vs_id, "parent_chunks.pkl")
    child_chunks = _load_pickle_data(vs_id, "child_chunks.pkl")

    if not all([vectorstore, parent_chunks, child_chunks]):
        print(f"[embedding_handler] Lỗi: Thiếu thành phần để tái tạo retriever cho session {vs_id}.")
        missing = []
        if not vectorstore: missing.append("Vector Store (FAISS)")
        if not parent_chunks: missing.append("Parent Chunks")
        if not child_chunks: missing.append("Child Chunks (for BM25)")
        print(f"[embedding_handler] Các thành phần bị thiếu: {', '.join(missing)}")
        return None

    final_retriever, _ = _build_lcel_retriever_chain(
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        embedding_model_instance=embedding_model_instance,
        vectorstore=vectorstore
    )
    
    return final_retriever


def get_or_create_vector_store(p_session_id, documents_info, embedding_model_instance):
    """
    Lấy retriever đã có hoặc tạo mới và lưu lại.
    Đây là hàm giao tiếp chính với `app.py`.
    """
    if not p_session_id:
        st.error("Lỗi hệ thống: Không có ID cho vector store.")
        return None, None

    vs_id = p_session_id
    print(f"[embedding_handler] Đang xử lý retriever cho ID: {vs_id}")
    if not isinstance(documents_info, tuple) or len(documents_info) != 2:
        st.error("Dữ liệu đầu vào không hợp lệ. Cần (parent_chunks, child_chunks).")
        return None, None

    parent_chunks, child_chunks = documents_info
    if not parent_chunks or not child_chunks:
        st.error("Dữ liệu đầu vào không hợp lệ: thiếu parent_chunks hoặc child_chunks.")
        return None, None
            
    print(f"[embedding_handler] Đang tạo retriever mới cho ID: {vs_id}...")
    
    new_retriever, new_vectorstore = _build_lcel_retriever_chain(
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        embedding_model_instance=embedding_model_instance
    )
        
    if new_retriever and new_vectorstore:
        print(f"[embedding_handler] Bắt đầu lưu các thành phần cho session: {vs_id}")
        vs_saved = save_vector_store(new_vectorstore, vs_id)
        parents_saved = _save_pickle_data(parent_chunks, vs_id, "parent_chunks.pkl")
        children_saved = _save_pickle_data(child_chunks, vs_id, "child_chunks.pkl")
        
        if all([vs_saved, parents_saved, children_saved]):
            print(f"[embedding_handler] Đã lưu thành công toàn bộ tài nguyên cho session: {vs_id}")
        else:
            print(f"[embedding_handler] CẢNH BÁO: Có lỗi xảy ra khi lưu một hoặc nhiều tài nguyên.")
        
        return new_retriever, vs_id
    
    print("[embedding_handler] Không thể tạo retriever mới.")
    return None, None