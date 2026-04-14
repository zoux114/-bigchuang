#!/usr/bin/env python3
"""
RAG 系统快速启动菜单
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class QuickStart:
    """快速启动程序"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.is_windows = platform.system() == "Windows"
    
    def clear_screen(self):
        """清屏"""
        os.system("cls" if self.is_windows else "clear")
    
    def print_menu(self):
        """显示菜单"""
        print("\n" + "=" * 50)
        print("   RAG 系统快速启动菜单".center(50))
        print("=" * 50 + "\n")
        
        print("选择要执行的操作:\n")
        print("  1. 一键安装 (首次使用必选)")
        print("  2. 诊断系统")
        print("  3. 初始化向量数据库")
        print("  4. 启动 API 服务器")
        print("  5. 启动前端开发服务器")
        print("  6. 启动完整应用 (API + Web)")
        print("  7. 打开项目文件夹")
        print("  8. 编辑环境配置 (.env)")
        print("  9. 查看帮助文档")
        print("  0. 退出\n")
    
    def run_command(self, cmd, description=""):
        """运行命令"""
        if description:
            print(f"\n[执行] {description}...\n")
        
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"错误: {e}")
    
    def install(self):
        """一键安装"""
        self.clear_screen()
        self.run_command(
            [sys.executable, "install.py"],
            "一键安装程序"
        )
        input("\n按 Enter 继续...")
    
    def doctor(self):
        """诊断系统"""
        self.clear_screen()
        self.run_command(
            [sys.executable, "doctor.py"],
            "系统诊断"
        )
        input("\n按 Enter 继续...")
    
    def ingest(self):
        """初始化数据库"""
        self.clear_screen()
        self.run_command(
            [sys.executable, "ingest.py"],
            "初始化向量数据库"
        )
        input("\n按 Enter 继续...")
    
    def start_api(self):
        """启动 API 服务器"""
        self.clear_screen()
        print("\n[启动] API 服务器...")
        print("访问地址: http://localhost:8001")
        print("按 Ctrl+C 停止服务\n")
        
        self.run_command([sys.executable, "api_server.py"])
    
    def start_frontend(self):
        """启动前端开发服务器"""
        self.clear_screen()
        print("\n[启动] 前端开发服务器...")
        print("访问地址: http://localhost:5173")
        print("按 Ctrl+C 停止服务\n")
        
        if self.is_windows:
            cmd = "npm run dev"
            subprocess.run(
                f"cd frontend && {cmd}",
                shell=True,
                check=False
            )
        else:
            subprocess.run(
                "cd frontend && npm run dev",
                shell=True,
                check=False
            )
    
    def start_full(self):
        """启动完整应用"""
        self.clear_screen()
        print("\n[启动] 完整应用 (API + Frontend)...\n")
        print("提示: 此操作会启动两个进程")
        print("  1. API 服务器: http://localhost:8001")
        print("  2. 前端开发: http://localhost:5173\n")
        
        # 启动 API 服务器
        print("[启动] API 服务器...")
        api_process = subprocess.Popen(
            [sys.executable, "api_server.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 给 API 时间启动
        import time
        time.sleep(2)
        
        # 启动前端
        print("[启动] 前端开发服务器...")
        if self.is_windows:
            subprocess.Popen("cd frontend && npm run dev", shell=True)
        else:
            subprocess.Popen("cd frontend && npm run dev", shell=True)
        
        print("\n两个服务已启动在后台")
        print("  - API: http://localhost:8001")
        print("  - Frontend: http://localhost:5173\n")
        print("按 Enter 返回菜单...")
        input()
    
    def open_project(self):
        """打开项目文件夹"""
        print("\n[打开] 项目文件夹...\n")
        
        if self.is_windows:
            os.startfile(str(self.project_root))
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(self.project_root)], check=False)
        else:  # Linux
            subprocess.run(["xdg-open", str(self.project_root)], check=False)
        
        input("\n按 Enter 继续...")
    
    def edit_config(self):
        """编辑配置文件"""
        env_file = self.project_root / "config" / ".env"
        
        if not env_file.exists():
            print("\n[错误] .env 文件不存在")
            print("请先运行一键安装程序\n")
            input("按 Enter 继续...")
            return
        
        print(f"\n[编辑] {env_file}\n")
        
        if self.is_windows:
            os.startfile(str(env_file))
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", "-a", "TextEdit", str(env_file)], check=False)
        else:  # Linux
            # 尝试用 nano 或 vi
            subprocess.run(["nano", str(env_file)], check=False)
        
        input("\n按 Enter 继续...")
    
    def show_help(self):
        """显示帮助"""
        self.clear_screen()
        print("""
================== RAG 系统帮助 ==================

【首次使用流程】
  1. 运行选项 1: 一键安装
  2. 编辑配置 (选项 8): 填入 LLM API 密钥
  3. 准备数据: 将文档放入 data/raw/
  4. 运行选项 3: 初始化向量数据库
  5. 运行选项 4 或 6: 启动服务

【支持的 LLM 服务】
  - 讯飞星火（默认）
  - OpenAI
  - DeepSeek
  - 智谱
  - 其他 OpenAI 兼容 API

【文档格式】
  支持: PDF, DOCX, TXT
  位置: data/raw/

【访问方式】
  API: http://localhost:8001
  Web: http://localhost:8000 或 5173

【常见问题】
  Q: 模块缺失?
  A: 运行选项 1 (一键安装) 或 2 (诊断系统)

  Q: 前端无法访问?
  A: 确保已安装 Node.js，运行选项 5 构建

    Q: 运行速度较慢?
    A: 运行选项 2 诊断，确认模型与索引配置

【文档】
  - 详细安装: INSTALL.md
  - API 文档: README.md

================================================
        """)
        input("按 Enter 返回菜单...")
    
    def run(self):
        """运行菜单循环"""
        while True:
            self.clear_screen()
            self.print_menu()
            
            choice = input("请输入选项 (0-9): ").strip()
            
            if choice == "1":
                self.install()
            elif choice == "2":
                self.doctor()
            elif choice == "3":
                self.ingest()
            elif choice == "4":
                self.start_api()
            elif choice == "5":
                self.start_frontend()
            elif choice == "6":
                self.start_full()
            elif choice == "7":
                self.open_project()
            elif choice == "8":
                self.edit_config()
            elif choice == "9":
                self.show_help()
            elif choice == "0":
                print("\n再见！\n")
                break
            else:
                print("\n无效选项，请重新输入")
                input("按 Enter 继续...")


def main():
    """主入口"""
    try:
        app = QuickStart()
        app.run()
    except KeyboardInterrupt:
        print("\n\n程序被中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
