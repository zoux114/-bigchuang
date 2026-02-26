"""
规章制度智能问答系统 - Web 界面
基于 Gradio 构建
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from query import RAGQueryEngine


class ChatInterface:
    """聊天界面管理器"""

    def __init__(self):
        self.engine = None

    def initialize(self):
        """延迟初始化引擎"""
        if self.engine is None:
            self.engine = RAGQueryEngine(
                show_sources=True,
                use_hybrid_search=True,
                use_rerank=True,
            )
        return self.engine

    def chat(self, message: str) -> str:
        """处理聊天消息"""
        if not message.strip():
            return "请输入您的问题。"

        try:
            engine = self.initialize()
            answer = engine.query(message)

            # 获取引用来源
            sources = engine.get_sources()
            if sources:
                sources_text = "\n\n---\n\n**📚 引用来源：**\n"
                for i, source in enumerate(sources, 1):
                    sources_text += f"\n{i}. **{source['source']}** - {source['section']} (相关度: {source['score']:.4f})"
                answer += sources_text

            return answer

        except Exception as e:
            return f"❌ 发生错误: {str(e)}"


def create_interface():
    """创建 Gradio 界面"""

    chat_interface = ChatInterface()

    with gr.Blocks(title="规章制度智能问答") as demo:

        # 标题
        gr.Markdown(
            """
            # 🎓 规章制度智能问答系统

            基于合肥工业大学大创项目规章制度文档，提供智能问答服务。

            **支持的问题类型：**
            - 📋 项目申报条件和流程
            - 👥 团队规模和成员要求
            - ⏰ 时间节点和进度安排
            - 📄 材料提交要求
            - 🏆 成果产出要求
            """
        )

        # 使用简单的文本输入输出，兼容所有版本
        with gr.Row():
            with gr.Column(scale=4):
                input_text = gr.Textbox(
                    label="问题",
                    placeholder="请输入您的问题，例如：大创项目需要几个人？",
                    lines=2,
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("发送", variant="primary")

        output_text = gr.Textbox(
            label="回答",
            lines=15,
        )

        # 示例问题
        gr.Examples(
            examples=[
                "大创项目需要几个人？",
                "申报大创项目需要什么条件？",
                "国家级项目和省级项目有什么区别？",
                "大创项目的截止时间是什么时候？",
                "指导教师有什么要求？",
            ],
            inputs=input_text,
        )

        # 处理提交
        def respond(message):
            if not message.strip():
                return "请输入您的问题。"
            return chat_interface.chat(message)

        submit_btn.click(respond, inputs=[input_text], outputs=[output_text])
        input_text.submit(respond, inputs=[input_text], outputs=[output_text])

        # 底部说明
        gr.Markdown(
            """
            ---
            💡 **使用提示：**
            - 系统会自动检索最相关的规章制度内容
            - 回答中会标注引用来源，方便查证
            - 如需更详细的信息，请查阅原始文档

            ⚙️ **技术架构：** RAG (检索增强生成) + 混合检索 (Dense + BM25) + 重排序
            """
        )

    return demo


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="规章制度智能问答 Web 界面")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务端口 (默认: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公网分享链接",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("规章制度智能问答系统")
    print("=" * 60)
    print(f"服务地址: http://localhost:{args.port}")
    if args.share:
        print("公网链接: 将在启动后显示")
    print("=" * 60 + "\n")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
