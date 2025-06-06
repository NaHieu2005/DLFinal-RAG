from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
import streamlit as st
from config import ollama_model_name, COHERE_API_KEY

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

# Cache để không phải load lại Cohere Reranker mỗi lần
@st.cache_resource(show_spinner="Đang khởi tạo mô hình Reranker...")
def get_reranker():
    """Khởi tạo Cohere Reranker."""
    if not COHERE_API_KEY:
        print("[llm_handler] COHERE_API_KEY không được cung cấp. Không thể khởi tạo reranker.")
        return None
    try:
        print("[llm_handler] Bắt đầu khởi tạo CohereRerank...")
        return CohereRerank(
            cohere_api_key=COHERE_API_KEY,
            model="rerank-multilingual-v3.0",
            top_n=3  # Chỉ giữ 3 kết quả tốt nhất
        )
    except Exception as e:
        print(f"[llm_handler] Lỗi khi khởi tạo Cohere Reranker: {e}")
        st.error(f"Lỗi khi khởi tạo Cohere Reranker: {e}")
        return None

# Hàm sắp xếp lại tài liệu để tối ưu thứ tự nhập vào LLM
def reorder_documents(docs):
    """
    Sắp xếp lại tài liệu để tăng hiệu quả của LLM bằng cách đặt một số tài liệu ưu tiên lên đầu.
    Thông thường, LLM đọc và nhớ tốt hơn những nội dung ở đầu và cuối context.
    """
    print(f"[llm_handler] Đang sắp xếp lại {len(docs)} tài liệu tham khảo để tối ưu...")
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

def get_qa_retrieval_chain(llm_instance, retriever):
    """Khởi tạo RetrievalQA chain với LLM và retriever đã có."""
    if not llm_instance or not retriever:
        print("[llm_handler] LLM instance hoặc retriever không hợp lệ.")
        st.error("LLM instance hoặc retriever không hợp lệ.")
        return None
    
    print("[llm_handler] Bắt đầu khởi tạo RetrievalQA chain...")
    
    # Nâng cấp prompt template để cải thiện chất lượng câu trả lời
    prompt_template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Bạn là một trợ lý AI chuyên nghiệp và thông thái. Vai trò của bạn là trả lời câu hỏi của người dùng một cách chính xác và súc tích, chỉ dựa vào nội dung trong "Thông tin tham khảo" được cung cấp.

**Quy trình làm việc của bạn như sau:**
1.  **Suy nghĩ nội bộ (không hiển thị phần này):** Đọc kỹ câu hỏi và toàn bộ "Thông tin tham khảo". Trong đầu, hãy xác định và gạch chân những câu, những dữ kiện từ tài liệu có liên quan trực tiếp đến câu hỏi.
2.  **Tổng hợp và trả lời (chỉ hiển thị phần này):** Dựa trên những thông tin liên quan đã xác định ở bước 1, hãy soạn một câu trả lời hoàn chỉnh, mạch lạc và tự nhiên cho người dùng.
    - Đi thẳng vào vấn đề, không dùng các cụm từ mở đầu như "Dựa trên tài liệu..." hay "Theo thông tin được cung cấp...".
    - Trình bày câu trả lời như một chuyên gia đang giải thích vấn đề.
    - Nếu không tìm thấy thông tin liên quan trong tài liệu, chỉ cần trả lời rằng: "Thông tin này không có trong tài liệu được cung cấp."
    - Luôn trả lời bằng tiếng Việt.

Thông tin tham khảo:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>
Câu hỏi: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Câu trả lời hữu ích:"""

    try:
        # Khởi tạo prompt
        PROMPT = PromptTemplate(
            template=prompt_template_str, input_variables=["context", "question"]
        )
        
        # Thêm reranking nếu COHERE_API_KEY được cung cấp
        reranker = get_reranker()
        if reranker:
            print("[llm_handler] Sử dụng Cohere Reranker để cải thiện kết quả...")
            try:
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=retriever
                )
                retriever = compression_retriever
            except Exception as e:
                print(f"[llm_handler] Lỗi khi khởi tạo contextual compression: {e}")
                # Giữ nguyên retriever nếu lỗi
        
        # Khởi tạo chain tiêu chuẩn
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_instance,
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("[llm_handler] Đã khởi tạo xong RetrievalQA chain.")
        return qa_chain
    except Exception as e:
        print(f"[llm_handler] Lỗi khi tạo QA chain: {e}")
        st.error(f"Lỗi khi tạo QA chain: {e}")
        return None

# (Code for get_qa_chain will go here) 