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
    
    # Quay lại prompt template cũ đã được chứng minh hoạt động tốt
    prompt_template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Bạn là một trợ lý AI hữu ích, chuyên trả lời các câu hỏi dựa trên nội dung tài liệu được cung cấp.
Hãy sử dụng các đoạn thông tin sau đây để trả lời câu hỏi của người dùng.
Nếu bạn không biết câu trả lời dựa trên thông tin được cung cấp, hãy nói rằng bạn không biết hoặc thông tin không có trong tài liệu. Đừng cố bịa ra câu trả lời.
Luôn trả lời bằng tiếng Việt một cách rõ ràng và mạch lạc.

Thông tin tham khảo:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>
Câu hỏi: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Câu trả lời hữu ích:"""

    try:
        # Khởi tạo prompt
        PROMPT = PromptTemplate(
            template=prompt_template_str, input_variables=["context", "question"]
        )
        
        # Tăng k cho retriever để đảm bảo luôn có kết quả
        if hasattr(retriever, 'search_kwargs'):
            # Đảm bảo k đủ lớn
            retriever.search_kwargs["k"] = max(retriever.search_kwargs.get("k", 5), 8)
            # Giảm score_threshold nếu có
            if "score_threshold" in retriever.search_kwargs:
                retriever.search_kwargs["score_threshold"] = min(retriever.search_kwargs["score_threshold"], 0.5)
        
        # LƯU Ý: KHÔNG sử dụng Cohere reranker vì gây ra lỗi
        # Bỏ đoạn sau để tránh lỗi
        """
        # Thêm reranking nếu COHERE_API_KEY được cung cấp
        reranker = get_reranker()
        if reranker:
            print("[llm_handler] Sử dụng Cohere Reranker để cải thiện kết quả...")
            try:
                # Đảm bảo top_n đủ lớn để có kết quả
                if hasattr(reranker, 'top_n'):
                    reranker.top_n = max(reranker.top_n, 5)  # Đảm bảo ít nhất 5 kết quả
                
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=retriever
                )
                retriever = compression_retriever
            except Exception as e:
                print(f"[llm_handler] Lỗi khi khởi tạo contextual compression: {e}")
                # Giữ nguyên retriever nếu lỗi
        """
        
        # Kiểm tra xem retriever có tìm thấy kết quả không
        print("[llm_handler] Kiểm tra retriever có tìm thấy kết quả...")
        try:
            test_results = retriever.get_relevant_documents("test query")
            print(f"[llm_handler] Retriever tìm thấy {len(test_results)} kết quả trong kiểm tra")
        except Exception as e:
            print(f"[llm_handler] Lỗi khi test retriever: {e}")
        
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