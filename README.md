# 规章制度智能问答系统

基于 RAG (Retrieval-Augmented Generation) 架构的规章制度垂直领域智能问答系统。

## 项目特点

- **本地化 Embedding**: 使用 PyTorch 加载开源中文向量模型，无需外部 API
- **智能文档处理**: 自动识别 PDF/Docx/TXT，保留规章制度层级结构
- **增量索引**: 只处理新增或修改的文件，避免重复计算
- **灵活配置**: 支持 OpenAI 兼容 API (讯飞星火、DeepSeek、智谱等)

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

## 🚀 一键安装

### 最简单的方式

**Windows**:
```bash
# 双击运行或在 PowerShell 中执行
python install.py
# 或运行批处理脚本
install.bat
```

**Linux/macOS**:
```bash
python install.py
```

一键安装程序会自动：
- ✓ 检查 Python 版本和可用模块
- ✓ 检测 CPU 运行环境
- ✓ 创建必要的目录结构
- ✓ 安装所有 Python 和 Node.js 依赖
- ✓ 创建配置文件模板
- ✓ 构建前端（如果已安装 Node.js）

详细安装说明请查看 [INSTALL.md](INSTALL.md)

### 系统诊断

如果遇到问题，运行诊断工具：

```bash
python doctor.py
```

会检查：
- Python 版本和模块
- CPU 运行配置
- 目录和文件完整性
- 配置文件
- 系统资源（内存、磁盘）

---

## 快速开始

### 1. 一键安装

详见上方 **一键安装** 部分

### 2. 配置 API

编辑 `config/.env` 文件：

```bash
# LLM API 配置（必需）
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://spark-api-open.xf-yun.com/v2/
LLM_MODEL=spark-x
```

支持的 LLM 服务：
- 讯飞星火（默认）
- OpenAI
- DeepSeek
- 智谱 (zhipu)
- 其他 OpenAI 兼容 API

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
- 默认 CPU 模式运行

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
EMBEDDING_DEVICE = "cpu"

# LLM API 配置
LLM_API_KEY = "your_api_password_or_AK:SK"
LLM_BASE_URL = "https://spark-api-open.xf-yun.com/v2/"
LLM_MODEL = "spark-x"

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
2. 默认 CPU 模式，便于跨环境稳定部署
3. PDF 文档如有扫描件，需先进行 OCR 处理
4. 增量索引基于文件修改时间判断

## Docker 部署 (推荐线上)

适用于“所有人访问网站，不吃本地用户算力”的部署方式。

### 1. 准备环境变量

确保 `config/.env` 已配置可用的 LLM 参数，例如：

```env
LLM_API_KEY=your_api_password_or_AK:SK
LLM_BASE_URL=https://spark-api-open.xf-yun.com/v2/
LLM_MODEL=spark-x
VECTOR_DB_DIR=data/vector_precomputed
EMBEDDING_DEVICE=cpu
```

线上建议固定 `EMBEDDING_DEVICE=cpu`，保证环境一致性。

### 1.1 预先向量化（推荐）

可在部署前提前完成向量分块和索引构建，减少线上实时算力消耗。

```bash
# 首次或文档大规模更新时
python ingest.py --force

# 仅文档有少量变动时，走增量索引
python ingest.py
```

默认向量目录可通过 `VECTOR_DB_DIR` 单独配置，例如 `data/vector_precomputed`。
部署时只需携带该目录即可直接查询，无需重复全量向量化。

### 2. 构建并启动

```bash
docker compose up -d --build
```

启动后访问：`http://<服务器IP>:8000`

### 3. 查看日志

```bash
docker compose logs -f rag-backend
```

### 4. 停止服务

```bash
docker compose down
```

### 5. 更新代码后重启

```bash
docker compose up -d --build
```

## License

MIT License
