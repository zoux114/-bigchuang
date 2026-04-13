# 一键安装指南

这个项目提供了自动化的安装程序，可以检查系统配置并自动安装所有依赖。

## 系统要求

- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 或更高版本
- **内存**: 至少 8GB（推荐 16GB）
- **存储**: 至少 20GB 空闲空间（用于模型和向量数据库）
- **网络**: 良好的网络连接（用于下载模型）

### 可选

- **GPU**: 支持 CUDA 11.8+ 的 NVIDIA GPU（可选，用于加速 Embedding）
- **Node.js**: 12.20+（仅为构建前端需要）

## 快速安装

### Windows

双击运行安装脚本：

```bash
install.bat
```

或在 PowerShell 中运行：

```powershell
python install.py
```

### macOS / Linux

在终端中运行：

```bash
python install.py
```

## 详细的安装步骤

### 步骤 1: 检查前置条件

#### 安装 Python 3.8+

- **Windows**: 从 [python.org](https://www.python.org/downloads/) 下载安装程序
  - 安装时 **必须** 勾选 "Add Python to PATH"
  - 建议安装 Python 3.11 LTS

- **macOS**: 使用 Homebrew
  ```bash
  brew install python@3.11
  ```

- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt-get install python3.11 python3.11-venv python3-pip
  ```

#### 验证 Python 安装

```bash
python --version
python -m pip --version
```

### 步骤 2: 克隆/下载项目

```bash
# 如果使用 Git
git clone <repository-url>
cd 终极代码

# 或直接下载并解压项目
```

### 步骤 3: 运行安装程序

#### 方式 A: 使用批处理脚本（Windows）

```bash
install.bat
```

#### 方式 B: 使用 Python 脚本（所有平台）

```bash
python install.py
```

#### 方式 C: 手动安装（如遇到问题）

```bash
# 创建虚拟环境（可选但推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装前端依赖（可选）
cd frontend
npm install
npm run build
cd ..
```

### 步骤 4: 配置环境变量

编辑 `config/.env` 文件：

```env
# LLM API 配置（必需）
LLM_API_KEY=your_actual_api_key
LLM_BASE_URL=https://spark-api-open.xf-yun.com/v2/
LLM_MODEL=spark-x

# 如果使用其他 LLM 服务，修改对应的配置
# OpenAI:
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4

# DeepSeek:
# LLM_BASE_URL=https://api.deepseek.com
# LLM_MODEL=deepseek-chat

# 其他配置
EMBEDDING_MODEL=shibing624/text2vec-base-chinese
EMBEDDING_DEVICE=auto  # 或 'cuda' / 'cpu'
VECTOR_DB_DIR=data/vector_db
LOG_LEVEL=INFO
```

### 步骤 5: 验证安装

```bash
python -c "import torch, transformers, sentence_transformers, faiss, fastapi; print('✓ 所有模块已安装')"
```

## 安装内容

安装程序将自动执行以下操作：

### 1. 环境检查
- ✓ Python 版本验证
- ✓ GPU 可用性检查
- ✓ 依赖模块扫描

### 2. 创建目录结构
```
data/
├── raw/              # 原始文档存放
├── processed/        # 处理后的文档
└── vector_db/        # 向量数据库
models/               # 模型缓存目录
```

### 3. 安装依赖

#### Python 依赖
- **深度学习**: PyTorch 2.0+, Transformers 4.30+
- **向量化**: Sentence-Transformers, FAISS
- **文档处理**: PyPDF2, python-docx, pdfplumber
- **Web 框架**: FastAPI, Uvicorn
- **其他**: dotenv, pyyaml, tqdm

#### Node.js 依赖（可选）
- React 19+
- Vite 8+
- Tailwind CSS 3+

### 4. CPU 运行说明（推荐）

默认使用 CPU 模式运行，安装脚本会直接安装通用依赖，无需 CUDA。

## 常见问题

### Q: 安装后仍有模块缺失？

**A**: 手动安装缺失的模块

```bash
pip install torch transformers sentence-transformers faiss-cpu
```

### Q: 运行较慢怎么办？

**A**: 可以适当降低批大小，或减少一次处理的文档规模。

### Q: 前端无法访问？

**A**: 确认前端已构建

```bash
cd frontend
npm install
npm run build
cd ..
```

### Q: 模型下载失败？

**A**: 可能是网络问题，重试即可

```bash
# 第一次运行脚本时会尝试下载模型
python ingest.py
```

如果网络确实有问题，可以手动下载并放入 `models/` 目录。

### Q: 内存不足？

**A**: 降低 Embedding 的 batch_size，编辑 `config/settings.py`

```python
EMBEDDING_CONFIG = {
    ...
    "batch_size": 8,  # 改小这个值
    ...
}
```

## 下一步

安装完成后，执行以下操作：

### 1. 准备数据

将 PDF 或 DOCX 文件放入 `data/raw/` 目录

### 2. 初始化向量数据库

```bash
python ingest.py
```

这会：
- 扫描 `data/raw/` 中的文件
- 处理文档（提取文本、保留结构）
- 生成向量嵌入
- 存储到 FAISS 数据库

### 3. 启动 API 服务器

```bash
python api_server.py
```

默认在 `http://localhost:8001` 运行

### 4. 打开 Web UI

访问 `http://localhost:8000` 或 `http://localhost:8001/ui`

## 高级配置

### 使用不同的 Embedding 模型

编辑 `config/.env`:

```env
# 轻量级模型（推荐用于 CPU）
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# 更好的效果（需要更多 GPU 内存）
EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
```

### 使用 Docker 部署（可选）

```bash
docker compose up -d
```

需要已安装 Docker 和 Docker Compose。

### 性能调优

编辑 `config/settings.py` 中的 `CHUNKING_CONFIG` 和 `EMBEDDING_CONFIG`，根据硬件情况调整。

## 卸载

### Windows

```bash
# 删除虚拟环境（如果创建了）
rmdir /s venv

# 删除项目目录即可
```

### macOS / Linux

```bash
# 删除虚拟环境
rm -rf venv

# 删除项目目录即可
```

## 获取帮助

如遇到问题：

1. 查看安装日志输出
2. 检查 `config/.env` 配置
3. 运行 `python install.py` 重新检查环境
4. 查看项目 README 获取更多信息

## 许可证

见 LICENSE 文件
