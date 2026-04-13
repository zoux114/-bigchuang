@echo off
REM ========================================
REM   RAG 系统快速启动菜单
REM ========================================

setlocal enableextensions enabledelayedexpansion

:menu
cls
echo.
echo ========================================
echo   RAG 系统快速启动菜单
echo ========================================
echo.
echo 选择要执行的操作:
echo.
echo   1. 一键安装 (首次使用必选)
echo   2. 诊断系统
echo   3. 初始化向量数据库
echo   4. 启动 API 服务器
echo   5. 启动前端开发服务器
echo   6. 启动完整应用 (API + Web)
echo   7. 打开项目文件夹
echo   8. 编辑环境配置 (.env)
echo   9. 退出
echo.
set /p choice="请输入选项 (1-9): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto doctor
if "%choice%"=="3" goto ingest
if "%choice%"=="4" goto api
if "%choice%"=="5" goto frontend
if "%choice%"=="6" goto full
if "%choice%"=="7" goto explorer
if "%choice%"=="8" goto config
if "%choice%"=="9" exit /b 0

echo 无效选项，请重新输入
timeout /t 2 >nul
goto menu

:install
echo.
echo [执行] 一键安装程序...
echo.
python install.py
pause
goto menu

:doctor
echo.
echo [执行] 系统诊断...
echo.
python doctor.py
pause
goto menu

:ingest
echo.
echo [执行] 初始化向量数据库...
echo.
python ingest.py
pause
goto menu

:api
echo.
echo [启动] API 服务器...
echo.
python api_server.py
pause
goto menu

:frontend
echo.
echo [启动] 前端开发服务器...
echo.
cd frontend
npm run dev
cd ..
pause
goto menu

:full
echo.
echo [启动] 完整应用 (API + Web)...
echo.
echo 提示: 此操作需要两个终端窗口
echo   1. 启动 API 服务器: http://localhost:8001
echo   2. 启动前端开发: http://localhost:5173
echo.
pause

REM 启动 API 服务器（新窗口）
start "RAG API Server" cmd /k "python api_server.py"

REM 等待一秒
timeout /t 1 >nul

REM 启动前端开发服务器（新窗口）
start "RAG Frontend Dev" cmd /k "cd frontend && npm run dev"

echo.
echo 两个服务已在单独的窗口中启动
echo   - API: http://localhost:8001
echo   - Frontend: http://localhost:5173
echo.
pause
goto menu

:explorer
echo.
echo [打开] 项目文件夹...
echo.
start explorer "%cd%"
timeout /t 1 >nul
goto menu

:config
echo.
echo [编辑] 环境配置文件...
echo.
if not exist "config\.env" (
    echo .env 文件不存在，请先运行安装程序
    pause
    goto menu
)
notepad "config\.env"
goto menu
