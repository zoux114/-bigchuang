"""
配置模块 - 集中管理所有配置项
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(Path(__file__).parent / ".env")

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
MODELS_DIR = PROJECT_ROOT / "models"

# 确保目录存在
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== Embedding 模型配置 ====================
EMBEDDING_CONFIG = {
    # 可选: shibing624/text2vec-base-chinese, BAAI/bge-small-zh-v1.5
    "model_name": os.getenv("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"),
    "device": os.getenv("EMBEDDING_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"),
    "max_seq_length": 256,
    "batch_size": 32,
    # 本地模型缓存目录
    "cache_dir": str(MODELS_DIR),
}

# ==================== LLM API 配置 ====================
LLM_CONFIG = {
    "api_key": os.getenv("LLM_API_KEY", ""),
    "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    "model": os.getenv("LLM_MODEL", "deepseek-chat"),
    "temperature": 0.7,
    "max_tokens": 2048,
}

# ==================== 向量数据库配置 ====================
VECTOR_STORE_CONFIG = {
    "db_path": str(VECTOR_DB_DIR),
    "index_name": "regulations",
    # FAISS 索引类型: Flat, IVF, HNSW
    "index_type": "Flat",
}

# ==================== 文本分块配置 ====================
CHUNKING_CONFIG = {
    # 分块大小 (字符数)
    "chunk_size": 500,
    # 分块重叠
    "chunk_overlap": 50,
    # 是否按句子分割
    "split_by_sentence": True,
    # 规章制度章节识别正则
    "section_patterns": [
        r"第[一二三四五六七八九十百]+章",  # 第一章
        r"第[一二三四五六七八九十百]+节",  # 第一节
        r"第[一二三四五六七八九十百]+条",  # 第一条
        r"第\d+章",
        r"第\d+节",
        r"第\d+条",
    ],
}

# ==================== 检索配置 ====================
RETRIEVAL_CONFIG = {
    # 返回的最相关文档数量
    "top_k": 5,
    # 相似度阈值 (0-1)
    "similarity_threshold": 0.5,
}

# ==================== Prompt 模板 ====================
PROMPT_TEMPLATE = """你是一个专业的规章制度助手。请根据以下规章制度内容回答用户问题。

相关规章制度条文：
{context}

用户问题：{question}

请根据上述规章制度内容，准确、专业地回答用户问题。如果规章制度中没有相关内容，请明确告知。
回答时请引用具体的条款来源（如"根据第X条规定"）。

回答："""

# ==================== ETL 配置 ====================
ETL_CONFIG = {
    # 支持的文档格式
    "supported_formats": [".pdf", ".docx", ".txt"],
    # 是否保留原文格式
    "preserve_formatting": True,
    # 处理后的文件编码
    "output_encoding": "utf-8",
}
