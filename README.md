# 规章制度智能问答系统

基于 RAG (Retrieval-Augmented Generation) 架构的规章制度垂直领域智能问答系统。

## 项目特点

- **本地化 Embedding**: 使用 PyTorch 加载开源中文向量模型，无需外部 API
- **智能文档处理**: 自动识别 PDF/Docx/TXT，保留规章制度层级结构
- **增量索引**: 只处理新增或修改的文件，避免重复计算
- **灵活配置**: 支持 OpenAI 兼容 API (DeepSeek、智谱等)

## 技术栈

| 模块 | 技术选型 |
|------|----------|
| Embedding | PyTorch + text2vec-base-chinese / bge-small-zh-v1.5 |
| 向量数据库 | FAISS (轻量级本地存储) |
| LLM | OpenAI 兼容 API (可配置) |
| 文档处理 | python-docx, PyPDF2, pdfplumber |

## 项目结构

```
.
├── config/
│   ├── settings.py          # 配置文件
│   └── .env.example         # 环境变量模板
├── data/
│   ├── raw/                 # 原始文档 (PDF/Docx/TXT)
│   ├── processed/           # 处理后的 Markdown 文件
│   └── vector_db/           # FAISS 向量数据库
├── models/                  # 本地模型缓存目录
├── src/
│   ├── etl/
│   │   ├── __init__.py
│   │   └── document_processor.py  # 文档处理 ETL
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py      # 向量嵌入模块
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py  # 向量数据库操作
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py        # LLM API 客户端
│   └── utils/
│       ├── __init__.py
│       └── chunker.py       # 文本分块工具
├── ingest.py                # 增量索引脚本
├── query.py                 # 智能问答接口
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API

```bash
# 复制配置模板
cp config/.env.example config/.env

# 编辑配置文件，填入你的 API Key
# LLM_API_KEY=your_api_key_here
# LLM_BASE_URL=https://api.deepseek.com/v1  # 或其他兼容 API
```

### 3. 准备文档

将规章制度文档放入 `data/raw/` 目录，支持格式：
- PDF (.pdf)
- Word (.docx)
- 纯文本 (.txt)

### 4. 构建索引

```bash
# 处理文档并构建向量索引
python ingest.py

# 仅处理文档 (转换为 Markdown)
python ingest.py --etl-only

# 仅构建索引 (已处理过的文档)
python ingest.py --index-only
```

### 5. 智能问答

```bash
# 命令行问答
python query.py

# 指定问题
python query.py "公司请假制度的具体规定是什么？"
```

## 核心模块说明

### ETL 文档处理 (`src/etl/`)

- 监控 `data/raw/` 目录，自动识别文档类型
- 保留规章制度的层级结构 (章节、条款)
- 输出结构化 Markdown 到 `data/processed/`

### Embedding 模块 (`src/embedding/`)

- 本地加载 `shibing624/text2vec-base-chinese` 模型
- 支持批量向量化，提升处理效率
- 自动 GPU 加速 (如果可用)

### 向量检索 (`src/retrieval/`)

- 使用 FAISS 进行高效相似度搜索
- 支持增量添加向量
- 元数据存储，关联原文位置

### LLM 接口 (`src/llm/`)

- OpenAI SDK 兼容，支持多种 API 提供商
- 自动拼接检索上下文和用户问题
- 可自定义 Prompt 模板

## 配置说明

编辑 `config/settings.py` 或设置环境变量：

```python
# Embedding 模型配置
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
EMBEDDING_DEVICE = "cuda"  # 或 "cpu"

# LLM API 配置
LLM_API_KEY = "your_api_key"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"

# 检索配置
TOP_K = 5  # 检索返回的文档数量
CHUNK_SIZE = 500  # 文本分块大小
CHUNK_OVERLAP = 50  # 分块重叠字符数
```

## 示例用法

```python
from query import RAGQueryEngine

# 初始化问答引擎
engine = RAGQueryEngine()

# 提问
response = engine.query("年假申请需要提前多少天？")
print(response)

# 获取引用来源
sources = engine.get_sources()
for source in sources:
    print(f"- {source['file']}: {source['section']}")
```

## 注意事项

1. 首次运行会自动下载 Embedding 模型 (~400MB)
2. 建议使用 GPU 加速向量化过程
3. PDF 文档如有扫描件，需先进行 OCR 处理
4. 增量索引基于文件修改时间判断

## License

MIT License
