from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from config import ollama_model_name

# Cache để không phải load lại LLM mỗi lần
@st.cache_resource(show_spinner="Đang khởi tạo mô hình AI...")
def get_llm_instance():
    """Khởi tạo LLM instance."""
    try:
        print("[llm_handler] Bắt đầu khởi tạo ChatOllama...")
        return ChatOllama(
            model=ollama_model_name,
            temperature=0.2,
            device="cpu"
        )
    except Exception as e:
        print(f"[llm_handler] Lỗi khi khởi tạo LLM: {e}")
        st.error(f"Lỗi khi khởi tạo LLM: {e}")
        return None

def get_qa_retrieval_chain(llm_instance, vector_store_retriever):
    """Khởi tạo RetrievalQA chain với LLM và retriever đã có."""
    if not llm_instance or not vector_store_retriever:
        print("[llm_handler] LLM instance hoặc vector store retriever không hợp lệ.")
        st.error("LLM instance hoặc vector store retriever không hợp lệ.")
        return None
    
    print("[llm_handler] Bắt đầu khởi tạo RetrievalQA chain...")
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

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_instance,
            chain_type="stuff", # Consider making this configurable if needed
            retriever=vector_store_retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        print("[llm_handler] Đã khởi tạo xong RetrievalQA chain.")
        return qa_chain
    except Exception as e:
        print(f"[llm_handler] Lỗi khi tạo QA chain: {e}")
        st.error(f"Lỗi khi tạo QA chain: {e}")
        return None

# (Code for get_qa_chain will go here) 