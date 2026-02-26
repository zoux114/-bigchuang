"""
智能问答接口
用户输入问题 -> 转化为向量 -> 检索相关条文 -> 拼接 Prompt -> 调用 LLM API
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    RETRIEVAL_CONFIG,
    PROMPT_TEMPLATE,
)
from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore, Document
from src.llm.client import LLMClient


class RAGQueryEngine:
    """RAG 问答引擎"""

    def __init__(
        self,
        top_k: Optional[int] = None,
        show_sources: bool = True,
    ):
        """
        初始化问答引擎

        Args:
            top_k: 检索返回的文档数量
            show_sources: 是否显示引用来源
        """
        self.top_k = top_k or RETRIEVAL_CONFIG["top_k"]
        self.show_sources = show_sources
        self._last_sources: List[Tuple[Document, float]] = []

        # 延迟初始化组件
        self._embedder = None
        self._vector_store = None
        self._llm_client = None

    def _init_components(self):
        """延迟初始化组件"""
        if self._embedder is None:
            print("正在初始化...")
            self._embedder = Embedder()
            self._vector_store = VectorStore()

            if self._vector_store.is_empty:
                print("警告: 向量索引为空，请先运行 python ingest.py 构建索引")

            try:
                self._llm_client = LLMClient()
            except ValueError as e:
                print(f"警告: {e}")
                self._llm_client = None

            print("初始化完成！")

    def query(
        self,
        question: str,
        return_context: bool = False,
    ) -> str:
        """
        执行问答

        Args:
            question: 用户问题
            return_context: 是否返回上下文

        Returns:
            回答文本 (如果 return_context=True，返回 (回答, 上下文))
        """
        self._init_components()

        # Step 1: 将问题转化为向量
        query_embedding = self._embedder.embed_single(question)

        # Step 2: 检索最相关的文档
        results = self._vector_store.search(
            query_embedding,
            top_k=self.top_k,
        )

        self._last_sources = results

        if not results:
            no_result_msg = "抱歉，我在规章制度中没有找到与您问题相关的内容。请尝试换一种问法，或者确认该内容是否在现有的规章制度文档中。"
            if return_context:
                return no_result_msg, ""
            return no_result_msg

        # Step 3: 构建上下文
        context_parts = []
        for doc, score in results:
            source_info = f"[{doc.metadata.get('source', '未知')} - {doc.metadata.get('section', '未知')}]"
            context_parts.append(f"{source_info}\n{doc.content}")

        context = "\n\n---\n\n".join(context_parts)

        # Step 4: 调用 LLM 生成回答
        if self._llm_client:
            try:
                answer = self._llm_client.rag_query(
                    question=question,
                    context=context,
                    prompt_template=PROMPT_TEMPLATE,
                )
            except Exception as e:
                answer = f"LLM 调用失败: {str(e)}\n\n以下是检索到的相关内容:\n{context}"
        else:
            # 如果 LLM 不可用，直接返回检索结果
            answer = f"LLM 未配置，以下是检索到的相关内容:\n\n{context}"

        if return_context:
            return answer, context
        return answer

    def get_sources(self) -> List[Dict]:
        """
        获取最近一次查询的引用来源

        Returns:
            来源列表 [{"content": ..., "source": ..., "section": ..., "score": ...}]
        """
        return [
            {
                "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "source": doc.metadata.get("source", "未知"),
                "section": doc.metadata.get("section", "未知"),
                "score": round(score, 4),
            }
            for doc, score in self._last_sources
        ]

    def interactive_mode(self):
        """交互式问答模式"""
        self._init_components()

        print("\n" + "=" * 60)
        print("规章制度智能问答系统")
        print("输入问题进行查询，输入 'quit' 或 'exit' 退出")
        print("=" * 60 + "\n")

        while True:
            try:
                question = input("\n请输入您的问题: ").strip()

                if not question:
                    continue

                if question.lower() in ["quit", "exit", "q"]:
                    print("再见！")
                    break

                # 执行查询
                print("\n正在查询...")
                answer = self.query(question)

                # 输出回答
                print("\n" + "-" * 40)
                print("回答:")
                print("-" * 40)
                print(answer)

                # 输出来源
                if self.show_sources:
                    sources = self.get_sources()
                    if sources:
                        print("\n" + "-" * 40)
                        print("引用来源:")
                        print("-" * 40)
                        for i, source in enumerate(sources, 1):
                            print(f"{i}. {source['source']} - {source['section']} (相似度: {source['score']})")

            except KeyboardInterrupt:
                print("\n\n已中断，再见！")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="规章制度智能问答")
    parser.add_argument(
        "question",
        nargs="?",
        help="要查询的问题 (不提供则进入交互模式)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_CONFIG["top_k"],
        help=f"检索返回的文档数量 (默认: {RETRIEVAL_CONFIG['top_k']})",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="不显示引用来源",
    )

    args = parser.parse_args()

    engine = RAGQueryEngine(
        top_k=args.top_k,
        show_sources=not args.no_sources,
    )

    if args.question:
        # 单次查询模式
        answer = engine.query(args.question)
        print(answer)

        if not args.no_sources:
            sources = engine.get_sources()
            if sources:
                print("\n---")
                print("引用来源:")
                for i, source in enumerate(sources, 1):
                    print(f"{i}. {source['source']} - {source['section']} (相似度: {source['score']})")
    else:
        # 交互模式
        engine.interactive_mode()


if __name__ == "__main__":
    main()
