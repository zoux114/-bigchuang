#!/usr/bin/env python3
"""
系统诊断工具 - 检查 RAG 系统的环境和配置
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess


class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class Diagnostics:
    """系统诊断"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {}
    
    def print_header(self, text: str):
        """打印标题"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")
    
    def check_ok(self, name: str, details: str = ""):
        """成功检查"""
        print(f"{Colors.GREEN}✓{Colors.RESET} {name}", end="")
        if details:
            print(f" ({Colors.BLUE}{details}{Colors.RESET})")
        else:
            print()
        self.results[name] = "OK"
    
    def check_error(self, name: str, details: str = ""):
        """错误检查"""
        print(f"{Colors.RED}✗{Colors.RESET} {name}", end="")
        if details:
            print(f" - {Colors.RED}{details}{Colors.RESET}")
        else:
            print()
        self.results[name] = "ERROR"
    
    def check_warning(self, name: str, details: str = ""):
        """警告检查"""
        print(f"{Colors.YELLOW}⚠{Colors.RESET} {name}", end="")
        if details:
            print(f" - {Colors.YELLOW}{details}{Colors.RESET}")
        else:
            print()
        self.results[name] = "WARNING"
    
    def check_python(self) -> bool:
        """检查 Python"""
        self.print_header("Python 环境")
        
        # Python 版本
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major >= 3 and version.minor >= 8:
            self.check_ok("Python", version_str)
        else:
            self.check_error("Python", f"{version_str}（需要 3.8+）")
            return False
        
        # Python 路径
        print(f"   路径: {sys.executable}")
        
        # pip 版本
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                pip_version = result.stdout.strip().split()[1]
                self.check_ok("pip", pip_version)
            else:
                self.check_warning("pip", "版本检查失败")
        except Exception as e:
            self.check_warning("pip", str(e))
        
        return True
    
    def check_modules(self) -> bool:
        """检查 Python 模块"""
        self.print_header("Python 模块")
        
        modules = {
            "torch": "PyTorch",
            "transformers": "Hugging Face Transformers",
            "sentence_transformers": "Sentence-Transformers",
            "faiss": "FAISS",
            "fastapi": "FastAPI",
            "uvicorn": "Uvicorn",
            "dotenv": "python-dotenv",
            "pyyaml": "PyYAML",
            "tqdm": "tqdm",
            "PyPDF2": "PyPDF2",
            "pdfplumber": "pdfplumber",
            "docx": "python-docx",
        }
        
        all_ok = True
        for module_name, display_name in modules.items():
            try:
                mod = __import__(module_name)
                version = getattr(mod, "__version__", "unknown")
                if version != "unknown":
                    self.check_ok(display_name, f"v{version}")
                else:
                    self.check_ok(display_name, "安装")
            except ImportError:
                self.check_error(display_name, "未安装")
                all_ok = False
        
        return all_ok
    
    def check_directories(self) -> bool:
        """检查目录结构"""
        self.print_header("目录结构")
        
        dirs = [
            "data/raw" ,
            "data/processed",
            "data/vector_db",
            "models",
            "src",
            "config",
            "frontend",
        ]
        
        all_ok = True
        for dir_name in dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                self.check_ok(f"目录 {dir_name}", "存在")
            else:
                self.check_warning(f"目录 {dir_name}", "缺失（可能在使用中创建）")
                all_ok = all_ok and (dir_name in ["data/processed", "data/vector_db"])
        
        return all_ok
    
    def check_files(self) -> bool:
        """检查关键文件"""
        self.print_header("关键文件")
        
        files = [
            ("requirements.txt", "Python 依赖文件"),
            ("config/settings.py", "配置文件"),
            ("api_server.py", "API 服务器"),
            ("ingest.py", "数据导入脚本"),
            ("query.py", "查询脚本"),
            ("frontend/package.json", "前端依赖文件"),
        ]
        
        all_ok = True
        for file_name, display_name in files:
            file_path = self.project_root / file_name
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                self.check_ok(display_name, f"{size_kb:.1f} KB")
            else:
                self.check_error(display_name, "缺失")
                all_ok = False
        
        return all_ok
    
    def check_config(self) -> bool:
        """检查配置"""
        self.print_header("配置检查")
        
        env_file = self.project_root / "config" / ".env"
        
        if not env_file.exists():
            self.check_warning(".env 文件", "不存在（使用默认配置）")
            return True
        
        self.check_ok(".env 文件", "存在")
        
        # 检查关键配置
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "LLM_API_KEY" in content and "your_api_key" not in content:
                self.check_ok("LLM_API_KEY", "已配置")
            else:
                self.check_warning("LLM_API_KEY", "未配置（需要手动设置）")
            
            if "EMBEDDING_MODEL" in content:
                self.check_ok("EMBEDDING_MODEL", "已配置")
            else:
                self.check_warning("EMBEDDING_MODEL", "使用默认值")
            
            return True
        except Exception as e:
            self.check_error(".env 文件", str(e))
            return False
    
    def check_node(self) -> bool:
        """检查 Node.js"""
        self.print_header("前端环境")
        
        # Node.js
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                node_version = result.stdout.strip()
                self.check_ok("Node.js", node_version)
            else:
                self.check_warning("Node.js", "未安装（前端无法构建）")
                return False
        except FileNotFoundError:
            self.check_warning("Node.js", "未安装（前端无法构建）")
            return False
        except Exception as e:
            self.check_error("Node.js", str(e))
            return False
        
        # npm
        try:
            result = subprocess.run(
                ["npm", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                npm_version = result.stdout.strip()
                self.check_ok("npm", npm_version)
            else:
                self.check_warning("npm", "版本检查失败")
        except Exception as e:
            self.check_warning("npm", str(e))
        
        # 前端构建状态
        dist_dir = self.project_root / "frontend" / "dist"
        if (dist_dir / "index.html").exists():
            self.check_ok("前端构建", "已构建")
        else:
            self.check_warning("前端构建", "未构建")
        
        return True
    
    def check_disk_space(self) -> bool:
        """检查磁盘空间"""
        self.print_header("系统资源")
        
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.project_root)
            free_gb = free / 1024 / 1024 / 1024
            total_gb = total / 1024 / 1024 / 1024
            
            if free_gb > 20:
                self.check_ok("硬盘空间", f"{free_gb:.2f} GB 可用（共 {total_gb:.2f} GB）")
            elif free_gb > 10:
                self.check_warning("硬盘空间", f"{free_gb:.2f} GB 可用（建议至少 20 GB）")
            else:
                self.check_error("硬盘空间", f"{free_gb:.2f} GB 可用（不足）")
            
            return True
        except Exception as e:
            self.check_warning("硬盘空间", str(e))
            return True
    
    def check_memory(self) -> bool:
        """检查内存"""
        try:
            import psutil
            
            mem = psutil.virtual_memory()
            total_gb = mem.total / 1024 / 1024 / 1024
            available_gb = mem.available / 1024 / 1024 / 1024
            
            if total_gb >= 16:
                self.check_ok("内存", f"{total_gb:.2f} GB（{available_gb:.2f} GB 可用）")
            elif total_gb >= 8:
                self.check_warning("内存", f"{total_gb:.2f} GB（建议至少 16 GB）")
            else:
                self.check_error("内存", f"{total_gb:.2f} GB（不足，建议 8 GB+）")
            
            return True
        except ImportError:
            print(f"{Colors.BLUE}ℹ{Colors.RESET} 内存检查需要 psutil，忽略")
            return True
        except Exception as e:
            self.check_warning("内存", str(e))
            return True
    
    def print_summary(self):
        """打印总结"""
        self.print_header("诊断总结")
        
        ok_count = sum(1 for v in self.results.values() if v == "OK")
        error_count = sum(1 for v in self.results.values() if v == "ERROR")
        warning_count = sum(1 for v in self.results.values() if v == "WARNING")
        
        print(f"总检查项: {len(self.results)}")
        print(f"{Colors.GREEN}✓ 成功: {ok_count}{Colors.RESET}")
        print(f"{Colors.YELLOW}⚠ 警告: {warning_count}{Colors.RESET}")
        print(f"{Colors.RED}✗ 错误: {error_count}{Colors.RESET}")
        print()
        
        if error_count > 0:
            print(f"{Colors.RED}存在问题！请查看上方错误信息。{Colors.RESET}")
            print(f"运行 {Colors.CYAN}python install.py{Colors.RESET} 重新安装。")
        elif warning_count > 0:
            print(f"{Colors.YELLOW}系统可用，但存在一些警告。{Colors.RESET}")
            print(f"建议查看上方警告信息。")
        else:
            print(f"{Colors.GREEN}系统配置正常！{Colors.RESET}")
        print()
    
    def run(self):
        """运行诊断"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}RAG 系统诊断工具{Colors.RESET}")
        print(f"项目路径: {self.project_root}\n")
        
        self.check_python()
        self.check_modules()
        self.check_directories()
        self.check_files()
        self.check_config()
        self.check_node()
        self.check_disk_space()
        self.check_memory()
        
        self.print_summary()


def main():
    """主入口"""
    diag = Diagnostics()
    diag.run()


if __name__ == "__main__":
    main()
