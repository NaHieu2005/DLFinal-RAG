import os
import re
import PyPDF2
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st
from config import PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
import docx

def process_uploaded_files(uploaded_files):
    """
    Đọc các file được tải lên, làm sạch, và chia thành các parent và child chunks.
    Trả về một tuple: (parent_chunks, child_chunks)
    """
    print("[document_processor] Bắt đầu xử lý và chia chunk parent/child...")
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    raw_docs_with_source = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            text = ""
            try:
                file_name = uploaded_file.name
                file_extension = os.path.splitext(file_name)[1].lower()

                if file_extension == ".txt":
                    text = str(uploaded_file.read(), "utf-8")
                elif file_extension == ".pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
                elif file_extension in [".doc", ".docx"]:
                    doc = docx.Document(uploaded_file)
                    text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
                else:
                    st.error(f"Định dạng file '{file_name}' không được hỗ trợ.")
                    continue

                if not text.strip():
                    st.warning(f"File '{file_name}' không có nội dung.")
                    continue
                
                # Làm sạch văn bản
                text = re.sub(r'\s+', ' ', text).strip()
                raw_docs_with_source.append(Document(page_content=text, metadata={"source": file_name}))

            except Exception as e:
                st.error(f"Lỗi khi đọc file '{uploaded_file.name}': {e}")
    
    if not raw_docs_with_source:
        print("[document_processor] Không có văn bản nào được trích xuất.")
        return None, None
        
    # --- LOGIC CỦA PARENT DOCUMENT ---
    # 1. Chia thành các parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, 
        chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    parent_chunks = parent_splitter.split_documents(raw_docs_with_source)
    print(f"[document_processor] Đã chia thành {len(parent_chunks)} parent chunks.")

    # 2. Gán ID duy nhất cho mỗi parent chunk để liên kết
    id_key = "parent_id"
    for i, p_chunk in enumerate(parent_chunks):
        p_chunk.metadata[id_key] = str(uuid.uuid4())
        p_chunk.metadata["chunk_id"] = i  # Thêm chunk_id để giữ khả năng tương thích với code cũ

    # 3. Chia parent chunks thành các child chunks
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, 
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    
    child_chunks = []
    for p_chunk in parent_chunks:
        # Tách nội dung của parent chunk
        _child_docs_content = child_splitter.split_text(p_chunk.page_content)
        
        # Gán metadata và tạo Document cho child chunk
        for j, _child_content in enumerate(_child_docs_content):
            child_metadata = p_chunk.metadata.copy()
            child_metadata["child_chunk_id"] = j  # ID riêng cho child để phân biệt
            child_doc = Document(page_content=_child_content, metadata=child_metadata)
            child_chunks.append(child_doc)

    print(f"[document_processor] Đã chia thành {len(child_chunks)} child chunks.")
    
    # Hàm giờ trả về parent và child chunks
    return parent_chunks, child_chunks

# Hàm sắp xếp lại tài liệu cho LLM
def reorder_documents(docs):
    """
    Sắp xếp lại tài liệu để tăng hiệu quả của LLM bằng cách đặt một số tài liệu ưu tiên lên đầu.
    Thông thường, LLM đọc và nhớ tốt hơn những nội dung ở đầu và cuối context.
    """
    print(f"[document_processor] Đang sắp xếp lại {len(docs)} tài liệu tham khảo để tối ưu...")
    if not docs or len(docs) == 0:
        return ""
    
    reordered_docs = []
    if len(docs) > 1:
        # Giữ tài liệu có điểm cao nhất đầu tiên
        reordered_docs.append(docs[0])
        
        # Thêm các tài liệu ở giữa
        if len(docs) > 2:
            reordered_docs.extend(docs[2:])
        
        # Kết thúc với tài liệu có điểm thứ hai - vị trí dễ nhớ
        if len(docs) > 1:
            reordered_docs.append(docs[1]) 
    else:
        reordered_docs = docs
    
    # Nối các tài liệu với nhau
    context_text = "\n\n---\n\n".join([doc.page_content for doc in reordered_docs])
    return context_text

# (Code for process_uploaded_files will go here) 