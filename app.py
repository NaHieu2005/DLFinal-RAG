import streamlit as st
import os
from dotenv import load_dotenv
import pickle # Nếu bạn lưu document objects
import re

# Import các lớp LangChain cần thiết
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Thư viện đọc PDF (ví dụ)
import PyPDF2 # Hoặc pdfplumber

# Tải biến môi trường (nếu có, ví dụ cho API key nếu bạn chuyển sang LLM API khác)
# load_dotenv() # Hiện tại không cần cho Ollama cục bộ

# --- Cấu hình cố định (có thể đưa ra ngoài hoặc làm cấu hình người dùng) ---
embedding_model_name = "bkai-foundation-models/vietnamese-bi-encoder"
ollama_model_name = "llama3:8b-instruct-q4_0"

# --- Hàm xử lý tài liệu ---
def process_uploaded_file(uploaded_file):
    """Đọc, làm sạch và chia chunk tài liệu được tải lên."""
    text = ""
    if uploaded_file is not None:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == ".txt":
                text = str(uploaded_file.read(), "utf-8")
            elif file_extension == ".pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            else:
                st.error("Định dạng file không được hỗ trợ. Vui lòng tải file .txt hoặc .pdf.")
                return None
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
            return None

        # Làm sạch văn bản (bạn có thể thêm các bước làm sạch chi tiết hơn ở đây)
        text = re.sub(r'\s+', ' ', text).strip() # Ví dụ làm sạch cơ bản

        # Chia chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        # chunks_text = text_splitter.split_text(text)
        # documents = [Document(page_content=t) for t in chunks_text] # Tạo Document objects đơn giản
        documents = text_splitter.create_documents([text]) # Cách này tốt hơn
        for i, doc in enumerate(documents): # Thêm metadata cơ bản
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["chunk_id"] = i
        return documents
    return None

# --- Hàm tạo hoặc tải Vector Store ---
# Cache để không phải tạo lại embedding và vector store mỗi lần tương tác
@st.cache_resource(show_spinner="Đang xử lý tài liệu và tạo cơ sở tri thức...")
def get_vector_store(_documents, _embedding_model_name): # Thêm _ trước embedding_model_name để Streamlit cache đúng
    """Tạo FAISS vector store từ các document đã chia chunk."""
    if _documents:
        try:
            embeddings_model_instance = HuggingFaceEmbeddings(
                model_name=_embedding_model_name,
                model_kwargs={'device': 'cuda'}
            )
            vector_store_instance = FAISS.from_documents(_documents, embeddings_model_instance)
            return vector_store_instance
        except Exception as e:
            st.error(f"Lỗi khi tạo vector store: {e}")
            return None
    return None

# --- Hàm khởi tạo LLM và QA Chain ---
# Cache để không phải load lại LLM mỗi lần
@st.cache_resource(show_spinner="Đang khởi tạo mô hình AI...")
def get_qa_chain(_ollama_model_name): # Thêm _ trước ollama_model_name
    """Khởi tạo LLM và RetrievalQA chain."""
    try:
        llm_instance = ChatOllama(
            model=_ollama_model_name,
            temperature=0.2,
        )

        prompt_template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Bạn là một trợ lý AI hữu ích, chuyên trả lời các câu hỏi dựa trên nội dung tài liệu được cung cấp.
Hãy sử dụng các đoạn thông tin sau đây để trả lời câu hỏi của người dùng.
Nếu bạn không biết câu trả lời dựa trên thông tin được cung cấp, hãy nói rằng bạn không biết hoặc thông tin không có trong tài liệu. Đừng cố bịa ra câu trả lời.
Luôn trả lời bằng tiếng Việt một cách rõ ràng và mạch lạc.

Thông tin tham khảo:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>
Câu hỏi: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Câu trả lời hữu ích:"""

        PROMPT = PromptTemplate(
            template=prompt_template_str, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}

        # Retriever sẽ được gán sau khi vector_store được tạo
        # Nên chúng ta không tạo qa_chain hoàn chỉnh ở đây ngay
        return llm_instance, chain_type_kwargs
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo LLM: {e}")
        return None, None


# --- Giao diện Streamlit ---
st.set_page_config(page_title="Chatbot Tài Liệu RAG", layout="wide")
st.title("💬 Chatbot Hỏi Đáp Tài Liệu (RAG với Llama 3)")
st.markdown("Tải lên tài liệu của bạn ( .txt hoặc .pdf) và đặt câu hỏi về nội dung đó.")

# --- Thanh bên (Sidebar) cho việc tải file ---
with st.sidebar:
    st.header("📁 Tải Tài Liệu")
    uploaded_file = st.file_uploader("Chọn một file .txt hoặc .pdf", type=["txt", "pdf"])

    if uploaded_file:
        st.success(f"Đã tải lên file: {uploaded_file.name}")
    else:
        st.info("Vui lòng tải lên một tài liệu để bắt đầu.")

# --- Xử lý và khởi tạo ---
documents = None
vector_store = None
llm, chain_type_kwargs = None, None

if uploaded_file:
    documents = process_uploaded_file(uploaded_file)
    if documents:
        vector_store = get_vector_store(documents, embedding_model_name) # Truyền tên model vào
        if vector_store:
            llm, chain_type_kwargs = get_qa_chain(ollama_model_name) # Truyền tên model vào

# --- Khởi tạo lịch sử chat trong session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Hãy tải tài liệu lên và đặt câu hỏi cho tôi nhé."}]

# Hiển thị các tin nhắn đã có
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Xem nguồn tham khảo"):
                for i, source in enumerate(message["sources"]):
                    st.caption(f"Nguồn {i+1} (Từ: {source.metadata.get('source', 'N/A')}, Chunk ID: {source.metadata.get('chunk_id', 'N/A')})")
                    st.markdown(source.page_content.replace("\n", " ")[:300] + "...")


# --- Nhận input từ người dùng ---
if prompt := st.chat_input("Câu hỏi của bạn về tài liệu..."):
    if not uploaded_file or not vector_store or not llm:
        st.warning("Vui lòng tải lên và xử lý tài liệu trước khi đặt câu hỏi.")
    else:
        # Thêm tin nhắn của người dùng vào lịch sử chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Tạo QA chain với retriever đã được cập nhật
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Bot đang suy nghĩ...")
            try:
                response = qa_chain.invoke({"query": prompt})
                answer = response.get("result", "Xin lỗi, tôi không tìm thấy câu trả lời.")
                if hasattr(answer, 'content'): # Nếu kết quả là AIMessage
                    answer = answer.content

                message_placeholder.markdown(answer)
                # Lưu nguồn tham khảo
                sources = response.get("source_documents", [])
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

                # Hiển thị nguồn ngay dưới câu trả lời (tùy chọn)
                if sources:
                    with st.expander("Xem nguồn tham khảo cho câu trả lời này"):
                        for i, source in enumerate(sources):
                            st.caption(f"Nguồn {i+1} (Từ: {source.metadata.get('source', 'N/A')}, Chunk ID: {source.metadata.get('chunk_id', 'N/A')})")
                            st.markdown(source.page_content.replace("\n", " ")[:300] + "...")


            except Exception as e:
                error_message = f"Đã xảy ra lỗi: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})