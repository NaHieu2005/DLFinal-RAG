import os

# API Keys
COHERE_API_KEY = "LSsVgNhEJmAWepOBmRNcSwXABh18VtmMXWYGLpD2" # Thêm API key của bạn ở đây

# Cấu hình model
embedding_model_name = "bkai-foundation-models/vietnamese-bi-encoder"
ollama_model_name = "llama3:8b-instruct-q4_0"

# Cấu hình chunking
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 100

# Paths for storing data
DATA_DIR = "data"
UPLOADED_FILES_DIR = os.path.join(DATA_DIR, "uploaded_files")
VECTOR_STORES_DIR = os.path.join(DATA_DIR, "vector_stores")
CHAT_HISTORIES_DIR = os.path.join(DATA_DIR, "chat_histories")

# Ensure directories exist
os.makedirs(UPLOADED_FILES_DIR, exist_ok=True)
os.makedirs(VECTOR_STORES_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORIES_DIR, exist_ok=True) 