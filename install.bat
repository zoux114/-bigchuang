@echo off
REM ========================================
REM   RAG 系统一键安装程序 (Windows)
REM ========================================

setlocal enableextensions enabledelayedexpansion

cls
echo.
echo ========================================
echo   RAG 系统一键安装程序
echo ========================================
echo.

REM 获取脚本所在目录
cd /d "%~dp0"
set PROJECT_ROOT=%cd%

REM ========== 检查 Python ==========
echo [1/7] 检查 Python...
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    echo 下载: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [成功] %PYTHON_VERSION%

REM ========== 创建目录 ==========
echo.
echo [2/7] 创建目录结构...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\vector_db" mkdir "data\vector_db"
if not exist "models" mkdir "models"
echo [成功] 目录创建完成

REM ========== 创建 .env 文件 ==========
echo.
echo [3/7] 检查环境变量文件...
if exist "config\.env" (
    echo [成功] .env 已存在
) else (
    echo [创建] 生成 config\.env 文件...
    (
        echo # ==================== LLM 配置 ====================
        echo # 支持的 API: OpenAI、讯飞星火、DeepSeek、智谱等
        echo LLM_API_KEY=your_api_key_here
        echo LLM_BASE_URL=https://spark-api-open.xf-yun.com/v2/
        echo LLM_MODEL=spark-x
        echo.
        echo # ==================== Embedding 模型 ====================
        echo # 可选: shibing624/text2vec-base-chinese, BAAI/bge-small-zh-v1.5
        echo EMBEDDING_MODEL=shibing624/text2vec-base-chinese
        echo EMBEDDING_DEVICE=auto
        echo.
        echo # ==================== 向量数据库 ====================
        echo VECTOR_DB_DIR=data/vector_db
        echo.
        echo # ==================== 其他配置 ====================
        echo # 日志级别: DEBUG, INFO, WARNING, ERROR
        echo LOG_LEVEL=INFO
    ) > "config\.env"
    echo [成功] .env 文件已创建，请编辑并配置 API 密钥
)

REM ========== 安装 Python 依赖 ==========
echo.
echo [4/7] 安装 Python 依赖...
python -m pip install --upgrade pip setuptools wheel >nul 2>nul
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [警告] Python 依赖安装可能出现问题，请查看上述输出
)

REM ========== 检查 GPU ==========
echo.
echo [5/7] 检查 GPU 支持...
python -c "import torch; print('GPU 可用:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [提示] PyTorch 未安装或无 GPU
) else (
    echo [成功] GPU 检查完成
)

REM ========== 检查 Node.js 和前端 ==========
echo.
echo [6/7] 检查 Node.js 和前端...
where node >nul 2>nul
if errorlevel 1 (
    echo [警告] Node.js 未安装，跳过前端设置
    echo 若要启用前端，请访问: https://nodejs.org/
    goto :skip_frontend
)

for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo [成功] %NODE_VERSION% 已安装

if not exist "frontend\dist\index.html" (
    echo [构建] 构建前端...
    call npm --prefix frontend install
    if errorlevel 1 (
        echo [警告] 前端依赖安装失败
        goto :skip_frontend
    )
    
    call npm --prefix frontend run build
    if errorlevel 1 (
        echo [警告] 前端构建失败
        goto :skip_frontend
    )
    echo [成功] 前端构建完成
)

:skip_frontend

REM ========== 验证安装 ==========
echo.
echo [7/7] 验证安装...
python -c "import torch, transformers, sentence_transformers, faiss, fastapi; print('[成功] 所有关键模块已安装')" 2>nul
if errorlevel 1 (
    echo [警告] 某些模块可能未安装
)

REM ========== 显示总结 ==========
echo.
echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 后续步骤:
echo   1. 编辑 config\.env，配置 LLM API 密钥
echo   2. 将数据文件放入 data\raw\ 目录
echo   3. 运行: python ingest.py (初始化向量数据库)
echo   4. 运行: python api_server.py (启动服务)
echo   5. 访问: http://localhost:8000 (前端)
echo.
pause
exit /b 0
