import os
import re
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st
from config import chunk_size, chunk_overlap # Import chunk configurations

def process_uploaded_files(uploaded_files): # Renamed to plural and takes a list
    """Đọc, làm sạch và chia chunk các tài liệu được tải lên."""
    print("[document_processor] Bắt đầu xử lý uploaded_files...")
    all_documents = []
    if not isinstance(uploaded_files, list): # Ensure it's a list for single file uploads
        uploaded_files = [uploaded_files]

    processed_file_names = []

    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            text = ""
            try:
                file_name = uploaded_file.name
                print(f"[document_processor] Đang xử lý file: {file_name}")
                processed_file_names.append(file_name)
                file_extension = os.path.splitext(file_name)[1].lower()

                if file_extension == ".txt":
                    text = str(uploaded_file.read(), "utf-8")
                elif file_extension == ".pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text()
                else:
                    print(f"[document_processor] Định dạng file '{file_name}' không được hỗ trợ.")
                    st.error(f"Định dạng file '{file_name}' không được hỗ trợ. Vui lòng tải file .txt hoặc .pdf.")
                    continue # Skip to the next file

                if not text.strip():
                    print(f"[document_processor] File '{file_name}' không có nội dung hoặc không thể đọc được.")
                    st.warning(f"File '{file_name}' không có nội dung hoặc không thể đọc được.")
                    continue

                # Làm sạch văn bản
                text = re.sub(r'\s+', ' ', text).strip()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, # Use from config
                    chunk_overlap=chunk_overlap, # Use from config
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                # create_documents expects a list of texts, so pass [text]
                documents = text_splitter.create_documents([text])

                for i, doc in enumerate(documents):
                    doc.metadata["source"] = file_name
                    doc.metadata["chunk_id"] = i
                all_documents.extend(documents)
                print(f"[document_processor] Đã xử lý xong file: {file_name}, số chunk: {len(documents)}")

            except Exception as e:
                print(f"[document_processor] Lỗi khi xử lý file '{uploaded_file.name}': {e}")
                st.error(f"Lỗi khi xử lý file '{uploaded_file.name}': {e}")
                # Optionally return partial results or None
    
    if not all_documents: # If no documents were successfully processed
        print("[document_processor] Không có tài liệu nào được xử lý thành công.")
        return None, None
        
    print(f"[document_processor] Hoàn thành xử lý tất cả files. Tổng số chunk: {len(all_documents)}")
    return all_documents, processed_file_names

# (Code for process_uploaded_files will go here) 