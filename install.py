#!/usr/bin/env python3
"""
一键安装程序 - RAG 系统自动配置与安装
检查系统环境、安装依赖、配置模型、初始化数据库
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import json

class Colors:
    """终端颜色代码"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class Installer:
    """RAG系统安装器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.errors = []
        self.warnings = []
        self.success_count = 0
        
    def print_header(self, text: str):
        """打印标题"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text:^60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")
    
    def print_success(self, text: str):
        """打印成功信息"""
        print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")
        self.success_count += 1
    
    def print_error(self, text: str):
        """打印错误信息"""
        print(f"{Colors.RED}✗ {text}{Colors.RESET}")
        self.errors.append(text)
    
    def print_warning(self, text: str):
        """打印警告信息"""
        print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")
        self.warnings.append(text)
    
    def print_info(self, text: str):
        """打印信息"""
        print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")
    
    def run_command(self, cmd: List[str], description: str = None, check: bool = True) -> Tuple[bool, str]:
        """
        执行命令
        
        Args:
            cmd: 命令列表
            description: 命令描述
            check: 是否检查返回码
            
        Returns:
            (成功/失败, 输出)
        """
        try:
            if description:
                self.print_info(f"执行: {description}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                if check:
                    return False, result.stderr or result.stdout
                else:
                    return True, result.stdout
            
            return True, result.stdout
        except Exception as e:
            return False, str(e)
    
    def check_python_version(self) -> bool:
        """检查 Python 版本"""
        self.print_info(f"检查 Python 版本...")
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.print_error(f"Python 版本过低: {version_str}，需要 3.8+")
            return False
        
        self.print_success(f"Python 版本: {version_str}")
        return True
    
    def check_node_and_npm(self) -> bool:
        """检查 Node.js 和 npm"""
        self.print_info("检查 Node.js 和 npm...")
        
        # 检查 Node.js
        success, output = self.run_command(["node", "--version"], check=False)
        if not success:
            self.print_warning("未找到 Node.js 安装。前端构建将被跳过。")
            self.print_info("若要启用前端，请访问: https://nodejs.org/")
            return False
        
        node_version = output.strip()
        self.print_success(f"Node.js 版本: {node_version}")
        
        # 检查 npm
        success, output = self.run_command(["npm", "--version"], check=False)
        if not success:
            self.print_warning("未找到 npm。前端构建将被跳过。")
            return False
        
        npm_version = output.strip()
        self.print_success(f"npm 版本: {npm_version}")
        return True
    
    def check_torch_gpu(self) -> bool:
        """检查 GPU 支持（PyTorch）"""
        self.print_info("检查 GPU 支持...")
        
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            
            if has_cuda:
                cuda_version = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(0)
                self.print_success(f"GPU 可用: {gpu_name} (CUDA {cuda_version})")
                return True
            else:
                self.print_warning("GPU 不可用，将使用 CPU 进行 Embedding")
                return False
        except ImportError:
            return False
    
    def create_directories(self) -> bool:
        """创建必要的目录"""
        self.print_info("创建任务目录...")
        
        dirs = [
            self.project_root / "data" / "raw",
            self.project_root / "data" / "processed",
            self.project_root / "data" / "vector_db",
            self.project_root / "models",
        ]
        
        try:
            for dir_path in dirs:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.print_success(f"目录就绪: {dir_path.relative_to(self.project_root)}")
            return True
        except Exception as e:
            self.print_error(f"创建目录失败: {e}")
            return False
    
    def create_env_file(self) -> bool:
        """创建 .env 文件（如果不存在）"""
        self.print_info("检查环境变量文件...")
        
        env_file = self.project_root / "config" / ".env"
        env_example = self.project_root / "config" / ".env.example"
        
        if env_file.exists():
            self.print_success(".env 文件已存在")
            return True
        
        # 创建默认 .env 文件
        default_env = """# ==================== LLM 配置 ====================
# 支持的 API: OpenAI、讯飞星火、DeepSeek、智谱等
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://spark-api-open.xf-yun.com/v2/
LLM_MODEL=spark-x

# ==================== Embedding 模型 ====================
# 可选: shibing624/text2vec-base-chinese, BAAI/bge-small-zh-v1.5
EMBEDDING_MODEL=shibing624/text2vec-base-chinese
EMBEDDING_DEVICE=auto

# ==================== 向量数据库 ====================
VECTOR_DB_DIR=data/vector_db

# ==================== 其他配置 ====================
# 日志级别: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
"""
        
        try:
            env_file.write_text(default_env, encoding='utf-8')
            self.print_success(f".env 文件已创建: {env_file.relative_to(self.project_root)}")
            self.print_warning("请编辑 .env 文件，配置 LLM API 密钥和其他参数")
            return True
        except Exception as e:
            self.print_error(f"创建 .env 文件失败: {e}")
            return False
    
    def install_python_dependencies(self) -> bool:
        """安装 Python 依赖"""
        self.print_info("安装 Python 依赖...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            self.print_error(f"未找到 requirements.txt: {requirements_file}")
            return False
        
        # 检查是否使用 GPU 轮子
        has_gpu = self.check_torch_gpu()
        
        # 安装基础依赖
        success, output = self.run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            description="pip install -r requirements.txt"
        )
        
        if not success:
            self.print_error(f"安装依赖失败: {output}")
            return False
        
        self.print_success("Python 依赖安装完成")
        
        # 根据 GPU 情况安装 PyTorch
        if has_gpu:
            self.print_info("安装 GPU 版本 PyTorch...")
            # 如果项目根目录有 wheels 文件夹，使用本地 wheels
            wheels_dir = self.project_root / "wheels"
            if wheels_dir.exists():
                self.print_info(f"从本地 wheels 目录安装: {wheels_dir}")
                torch_whl = list(wheels_dir.glob("torch*"))
                if torch_whl:
                    success, _ = self.run_command(
                        [sys.executable, "-m", "pip", "install"] + [str(w) for w in torch_whl],
                        description="安装本地 PyTorch wheels"
                    )
                    if success:
                        self.print_success("PyTorch GPU 版本安装成功")
        
        return True
    
    def install_node_dependencies(self) -> bool:
        """安装 Node.js 依赖（前端）"""
        self.print_info("安装前端依赖...")
        
        if not self.check_node_and_npm():
            return False
        
        frontend_dir = self.project_root / "frontend"
        if not frontend_dir.exists():
            self.print_warning(f"前端目录不存在: {frontend_dir}")
            return False
        
        success, output = self.run_command(
            ["npm", "install"],
            description=f"npm install (in {frontend_dir})"
        )
        
        if not success:
            self.print_error(f"npm install 失败: {output}")
            return False
        
        self.print_success("前端依赖安装完成")
        return True
    
    def build_frontend(self) -> bool:
        """构建前端"""
        self.print_info("构建前端...")
        
        if not self.check_node_and_npm():
            return False
        
        frontend_dir = self.project_root / "frontend"
        dist_dir = frontend_dir / "dist"
        
        # 如果已经构建过，跳过
        if (dist_dir / "index.html").exists():
            self.print_success("前端已构建")
            return True
        
        success, output = self.run_command(
            ["npm", "run", "build"],
            description="npm run build (前端)"
        )
        
        if not success:
            self.print_error(f"前端构建失败: {output}")
            return False
        
        self.print_success("前端构建成功")
        return True
    
    def validate_installation(self) -> bool:
        """验证安装"""
        self.print_info("验证安装...")
        
        checks = []
        
        # 检查关键模块
        try:
            import torch
            checks.append(("PyTorch", True))
        except ImportError:
            checks.append(("PyTorch", False))
        
        try:
            import transformers
            checks.append(("Transformers", True))
        except ImportError:
            checks.append(("Transformers", False))
        
        try:
            import sentence_transformers
            checks.append(("Sentence-Transformers", True))
        except ImportError:
            checks.append(("Sentence-Transformers", False))
        
        try:
            import faiss
            checks.append(("FAISS", True))
        except ImportError:
            checks.append(("FAISS", False))
        
        try:
            import fastapi
            checks.append(("FastAPI", True))
        except ImportError:
            checks.append(("FastAPI", False))
        
        all_ok = True
        for module, available in checks:
            if available:
                self.print_success(f"模块 {module} 可用")
            else:
                self.print_warning(f"模块 {module} 未安装")
                all_ok = False
        
        return all_ok
    
    def show_next_steps(self):
        """显示后续步骤"""
        self.print_header("后续步骤")
        
        print(f"{Colors.CYAN}1. 编辑配置文件:{Colors.RESET}")
        print(f"   编辑 config/.env，配置 LLM API 密钥和其他参数\n")
        
        print(f"{Colors.CYAN}2. 准备数据:{Colors.RESET}")
        print(f"   将 PDF 或 DOCX 文件放入 data/raw/ 目录\n")
        
        print(f"{Colors.CYAN}3. 初始化向量数据库:{Colors.RESET}")
        print(f"   运行: python ingest.py\n")
        
        print(f"{Colors.CYAN}4. 启动服务器:{Colors.RESET}")
        print(f"   运行: python api_server.py\n")
        
        print(f"{Colors.CYAN}5. 打开前端:{Colors.RESET}")
        print(f"   访问: http://localhost:8000\n")
    
    def install(self):
        """执行完整的安装流程"""
        self.print_header("RAG 系统一键安装程序")
        
        print(f"{Colors.BOLD}系统信息:{Colors.RESET}")
        print(f"Python: {sys.version}")
        print(f"平台: {sys.platform}")
        print(f"项目路径: {self.project_root}\n")
        
        # 执行检查和安装
        steps = [
            ("检查 Python 版本", self.check_python_version),
            ("创建目录结构", self.create_directories),
            ("创建环境变量文件", self.create_env_file),
            ("安装 Python 依赖", self.install_python_dependencies),
            ("安装前端依赖", self.install_node_dependencies),
            ("构建前端", self.build_frontend),
            ("验证安装", self.validate_installation),
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            self.print_header(step_name)
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    self.print_warning(f"{step_name} 未完成")
            except Exception as e:
                self.print_error(f"{step_name} 出错: {e}")
                failed_steps.append(step_name)
        
        # 显示总结
        self.print_header("安装总结")
        
        print(f"{Colors.GREEN}✓ 成功的检查: {self.success_count}{Colors.RESET}\n")
        
        if self.warnings:
            print(f"{Colors.YELLOW}⚠ 警告 ({len(self.warnings)}):  {Colors.RESET}")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()
        
        if self.errors:
            print(f"{Colors.RED}✗ 错误 ({len(self.errors)}):  {Colors.RESET}")
            for error in self.errors:
                print(f"  - {error}")
            print()
        
        if failed_steps:
            print(f"{Colors.YELLOW}未完成的步骤:{Colors.RESET}")
            for step in failed_steps:
                print(f"  - {step}")
            print()
            print(f"{Colors.YELLOW}请查看上述错误并重新运行此脚本{Colors.RESET}\n")
        else:
            print(f"{Colors.GREEN}安装成功！{Colors.RESET}\n")
            self.show_next_steps()


def main():
    """主入口"""
    try:
        installer = Installer()
        installer.install()
        
        if installer.errors:
            sys.exit(1)
        else:
            sys.exit(0)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}安装被用户中断{Colors.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}安装程序出错: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
