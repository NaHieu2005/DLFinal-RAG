import streamlit as st
import os
import pickle 
import re
import uuid 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.storage import InMemoryStore
from langchain_core.runnables import RunnableLambda 

# Thư viện đọc PDF
import PyPDF2

COHERE_API_KEY = "LSsVgNhEJmAWepOBmRNcSwXABh18VtmMXWYGLpD2" 

# --- Cấu hình cố định ---
embedding_model_name = "bkai-foundation-models/vietnamese-bi-encoder"
ollama_model_name = "llama3:8b-instruct-q4_0"

# --- Hàm xử lý tài liệu ---
def process_uploaded_file(uploaded_file):
    """Đọc file và trả về một list chứa Document object thô."""
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
        documents = [Document(page_content=text, metadata={"source": uploaded_file.name})]
        return documents
    return None

# --- Hàm khởi tạo LLM ---
@st.cache_resource(show_spinner="Đang khởi tạo mô hình AI...")
def get_llm_and_prompt(_ollama_model_name):
    """Khởi tạo LLM và Prompt Template."""
    try:
        llm_instance = ChatOllama(model=_ollama_model_name, temperature=0.2)
        prompt_template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Bạn là một trợ lý AI hữu ích, chuyên trả lời các câu hỏi dựa trên nội dung tài liệu được cung cấp.
Hãy sử dụng các đoạn thông tin sau đây để trả lời câu hỏi của người dùng.
Nếu bạn không biết câu trả lời dựa trên thông tin được cung cấp, hãy nói rằng bạn không biết hoặc thông tin không có trong tài liệu. Đừng cố bịa ra câu trả lời.
Luôn trả lời bằng tiếng Việt một cách rõ ràng và mạch lạc.

Thông tin tham khảo:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>
Câu hỏi: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Câu trả lời hữu ích:"""
        PROMPT = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        return llm_instance, chain_type_kwargs
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo LLM: {e}")
        return None, None

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Chatbot Tài Liệu RAG", layout="wide")
st.title("💬 Chatbot Hỏi Đáp Tài Liệu Nâng Cao")
st.markdown("Tải lên tài liệu của bạn ( .txt hoặc .pdf) và đặt câu hỏi về nội dung đó.")

# --- Thanh bên (Sidebar) ---
with st.sidebar:
    st.header("📁 Tải Tài Liệu")
    uploaded_file = st.file_uploader("Chọn một file .txt hoặc .pdf", type=["txt", "pdf"])
    if uploaded_file:
        st.success(f"Đã tải lên file: {uploaded_file.name}")
    else:
        st.info("Vui lòng tải lên một tài liệu để bắt đầu.")

# --- Xử lý và khởi tạo ---
# Khối này được tái cấu trúc hoàn toàn để tích hợp tất cả các kỹ thuật
@st.cache_resource(show_spinner="Đang xử lý tài liệu và xây dựng pipeline RAG...")
def build_full_rag_pipeline(_raw_documents):
    # --- 1: LOGIC CỦA PARENT DOCUMENT ---
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    parent_chunks = parent_splitter.split_documents(_raw_documents)
    
    docstore = InMemoryStore()
    parent_chunk_ids = [str(uuid.uuid4()) for _ in parent_chunks]
    docstore.mset(list(zip(parent_chunk_ids, parent_chunks)))

    child_chunks = []
    for i, p_chunk in enumerate(parent_chunks):
        _child_chunks = child_splitter.split_documents([p_chunk])
        for c_chunk in _child_chunks:
            c_chunk.metadata["parent_id"] = parent_chunk_ids[i]
        child_chunks.extend(_child_chunks)

    # --- 2: LOGIC CỦA HYBRID SEARCH (trên chunk con) ---
    embeddings_model_instance = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cuda'})
    vectorstore = FAISS.from_documents(child_chunks, embeddings_model_instance)
    
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever.from_documents(child_chunks)
    bm25_retriever.k = 10
    
    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

    # --- 3: KẾT HỢP PARENT RETRIEVAL VÀ HYBRID SEARCH ---
    def get_parent_chunks(child_docs):
        parent_ids = {doc.metadata["parent_id"] for doc in child_docs}
        return [doc for doc in docstore.mget(list(parent_ids)) if doc is not None]

    parent_retriever_chain = hybrid_retriever | RunnableLambda(get_parent_chunks)

    # --- 4: LOGIC CỦA RE-RANKING (trên chunk cha) ---
    reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-multilingual-v3.0", top_n=3)
    final_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=parent_retriever_chain)
    
    return final_retriever

retriever = None
llm, chain_type_kwargs = None, None

if uploaded_file:
    raw_documents = process_uploaded_file(uploaded_file)
    if raw_documents:
        retriever = build_full_rag_pipeline(raw_documents)
        llm, chain_type_kwargs = get_llm_and_prompt(ollama_model_name)

# --- Khởi tạo và hiển thị lịch sử chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Hãy tải tài liệu lên và đặt câu hỏi cho tôi nhé."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Xem nguồn tham khảo"):
                for i, source in enumerate(message["sources"]):
                    st.caption(f"Nguồn {i+1} (Từ: {source.metadata.get('source', 'N/A')})")
                    st.markdown(source.page_content.replace("\n", " "))

# --- Nhận input từ người dùng ---
if prompt := st.chat_input("Câu hỏi của bạn về tài liệu..."):
    if not uploaded_file or not retriever or not llm:
        st.warning("Vui lòng tải lên và xử lý tài liệu trước khi đặt câu hỏi.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever, 
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Bot đang tìm kiếm và suy luận...")
            try:
                response = qa_chain.invoke({"query": prompt})
                answer = response.get("result", "Xin lỗi, tôi không tìm thấy câu trả lời.")
                message_placeholder.markdown(answer)
                
                sources = response.get("source_documents", [])
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

                if sources:
                    with st.expander("Xem nguồn tham khảo cho câu trả lời này"):
                        for i, source in enumerate(sources):
                            st.caption(f"Nguồn {i+1} (Từ: {source.metadata.get('source', 'N/A')})")
                            st.markdown(source.page_content.replace("\n", " "))

            except Exception as e:
                error_message = f"Đã xảy ra lỗi: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})