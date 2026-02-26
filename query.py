"""
智能问答接口 - 优化版
基于 RAG 论文 (Lewis et al., 2020) 的优化策略：
1. Hybrid Search: BM25 + Dense Retrieval
2. Re-ranking: 多因素重排序
3. Dynamic top-k: 根据查询复杂度调整
4. Multi-document fusion: 多文档融合生成
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    RETRIEVAL_CONFIG,
    HYBRID_SEARCH_CONFIG,
    RERANK_CONFIG,
    PROMPT_TEMPLATE,
    VECTOR_DB_DIR,
)
from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore, Document
from src.retrieval.hybrid_search import HybridSearcher, SearchResult
from src.retrieval.reranker import ReRanker, RerankedResult, QueryAnalyzer
from src.llm.client import LLMClient


class RAGQueryEngine:
    """
    RAG 问答引擎 - 优化版

    检索流程:
    1. 查询分析 -> 识别问题类型和复杂度
    2. 混合检索 -> BM25 + Dense Retrieval
    3. 重排序 -> 多因素综合评分
    4. 生成回答 -> LLM 多文档融合
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        show_sources: bool = True,
        use_hybrid_search: bool = True,
        use_rerank: bool = True,
    ):
        """
        初始化问答引擎

        Args:
            top_k: 检索返回的文档数量
            show_sources: 是否显示引用来源
            use_hybrid_search: 是否使用混合检索
            use_rerank: 是否使用重排序
        """
        self.top_k = top_k or RETRIEVAL_CONFIG["top_k"]
        self.show_sources = show_sources
        self.use_hybrid_search = use_hybrid_search and HYBRID_SEARCH_CONFIG["enabled"]
        self.use_rerank = use_rerank and RERANK_CONFIG["enabled"]

        self._last_sources: List[Tuple[Document, float]] = []
        self._last_search_info: Dict = {}

        # 延迟初始化组件
        self._embedder = None
        self._vector_store = None
        self._llm_client = None
        self._hybrid_searcher = None
        self._reranker = None
        self._query_analyzer = None

    def _init_components(self):
        """延迟初始化组件"""
        if self._embedder is not None:
            return

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

        # 初始化混合检索器
        if self.use_hybrid_search:
            self._hybrid_searcher = HybridSearcher(
                dense_weight=HYBRID_SEARCH_CONFIG["dense_weight"],
                bm25_weight=HYBRID_SEARCH_CONFIG["bm25_weight"],
                use_rrf=HYBRID_SEARCH_CONFIG["use_rrf"],
                rrf_k=HYBRID_SEARCH_CONFIG["rrf_k"],
            )
            # 尝试加载 BM25 索引
            bm25_path = VECTOR_DB_DIR / "bm25_index.pkl"
            if not self._hybrid_searcher.load_bm25_index(bm25_path):
                # 如果没有 BM25 索引，从向量存储构建
                if self._vector_store._documents:
                    print("构建 BM25 索引...")
                    self._hybrid_searcher.build_bm25_index(self._vector_store._documents)
                    self._hybrid_searcher.save_bm25_index(bm25_path)

        # 初始化重排序器
        if self.use_rerank:
            self._reranker = ReRanker(
                similarity_weight=RERANK_CONFIG["similarity_weight"],
                coverage_weight=RERANK_CONFIG["coverage_weight"],
                structure_weight=RERANK_CONFIG["structure_weight"],
                freshness_weight=RERANK_CONFIG["freshness_weight"],
            )

        # 初始化查询分析器
        self._query_analyzer = QueryAnalyzer()

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

        # Step 1: 查询分析
        query_info = self._query_analyzer.analyze(question)
        self._last_search_info = query_info

        # 动态调整 top_k
        dynamic_top_k = int(query_info["suggested_top_k"])
        candidate_pool_size = RETRIEVAL_CONFIG["candidate_pool_size"]

        print(f"\n[查询分析] 类型: {query_info['type']}, 复杂度: {query_info['complexity']}, 建议top_k: {dynamic_top_k}")

        # Step 2: 向量化查询
        query_embedding = self._embedder.embed_single(question)

        # Step 3: 检索
        if self.use_hybrid_search and self._hybrid_searcher:
            # 混合检索
            dense_results = self._vector_store.search(
                query_embedding,
                top_k=candidate_pool_size,
            )
            search_results = self._hybrid_searcher.search(
                query=question,
                query_embedding=query_embedding,
                dense_results=dense_results,
                top_k=candidate_pool_size,
            )
            # 转换为统一格式
            results = [(r.document, r.score) for r in search_results]
            print(f"[混合检索] 召回 {len(results)} 个候选文档")
        else:
            # 纯 Dense 检索
            results = self._vector_store.search(
                query_embedding,
                top_k=candidate_pool_size,
            )
            print(f"[Dense检索] 召回 {len(results)} 个候选文档")

        if not results:
            no_result_msg = "抱歉，我在规章制度中没有找到与您问题相关的内容。请尝试换一种问法，或者确认该内容是否在现有的规章制度文档中。"
            if return_context:
                return no_result_msg, ""
            return no_result_msg

        # Step 4: 重排序
        if self.use_rerank and self._reranker:
            reranked = self._reranker.rerank(
                query=question,
                results=results,
                top_k=dynamic_top_k,
            )
            results = [(r.document, r.final_score) for r in reranked]
            print(f"[重排序] 精选出 {len(results)} 个文档")
        else:
            results = results[:dynamic_top_k]

        self._last_sources = results

        # Step 5: 构建上下文 (多文档融合)
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            source_info = f"[文档{i}] {doc.metadata.get('source', '未知')} - {doc.metadata.get('section', '未知')}"
            context_parts.append(f"{source_info}\n{doc.content}")

        context = "\n\n---\n\n".join(context_parts)

        # Step 6: 调用 LLM 生成回答
        if self._llm_client:
            try:
                # 使用增强的 Prompt
                enhanced_prompt = self._build_enhanced_prompt(question, context, query_info)
                answer = self._llm_client.rag_query(
                    question=question,
                    context=context,
                    prompt_template=enhanced_prompt,
                )
            except Exception as e:
                answer = f"LLM 调用失败: {str(e)}\n\n以下是检索到的相关内容:\n{context}"
        else:
            # 如果 LLM 不可用，直接返回检索结果
            answer = f"LLM 未配置，以下是检索到的相关内容:\n\n{context}"

        if return_context:
            return answer, context
        return answer

    def _build_enhanced_prompt(
        self,
        question: str,
        context: str,
        query_info: Dict,
    ) -> str:
        """构建增强的 Prompt (基于查询类型)"""
        q_type = query_info["type"]

        # 根据问题类型调整 Prompt
        type_instructions = {
            "definition": "请给出清晰的定义和解释。",
            "procedure": "请按步骤说明具体流程。",
            "quantity": "请给出具体的数字或时间要求。",
            "condition": "请列出所有适用条件和要求。",
            "comparison": "请对比说明两者的区别。",
            "list": "请列出所有相关内容。",
            "general": "",
        }

        instruction = type_instructions.get(q_type, "")

        return f"""你是一个专业的规章制度助手。请根据以下规章制度内容回答用户问题。

{instruction}

相关规章制度条文：
{context}

用户问题：{question}

请根据上述规章制度内容，准确、专业地回答用户问题。
要求：
1. 如果规章制度中没有相关内容，请明确告知
2. 回答时请引用具体的条款来源（如"根据第X条规定"）
3. 如果涉及多个文档，请综合说明
4. 回答要准确、简洁、有条理

回答："""

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

    def get_search_info(self) -> Dict:
        """获取最近一次查询的分析信息"""
        return self._last_search_info

    def interactive_mode(self):
        """交互式问答模式"""
        self._init_components()

        print("\n" + "=" * 60)
        print("规章制度智能问答系统 (优化版)")
        print("特性: 混合检索 + 重排序 + 动态 top-k")
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
                            print(f"{i}. {source['source']} - {source['section']} (分数: {source['score']})")

            except KeyboardInterrupt:
                print("\n\n已中断，再见！")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")

    def rebuild_bm25_index(self):
        """重建 BM25 索引"""
        self._init_components()

        if self._hybrid_searcher and self._vector_store._documents:
            print("重建 BM25 索引...")
            self._hybrid_searcher.build_bm25_index(self._vector_store._documents)
            bm25_path = VECTOR_DB_DIR / "bm25_index.pkl"
            self._hybrid_searcher.save_bm25_index(bm25_path)
            print("BM25 索引已保存")


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="规章制度智能问答 (优化版)")
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
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="禁用混合检索 (仅使用 Dense Retrieval)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="禁用重排序",
    )
    parser.add_argument(
        "--rebuild-bm25",
        action="store_true",
        help="重建 BM25 索引",
    )

    args = parser.parse_args()

    engine = RAGQueryEngine(
        top_k=args.top_k,
        show_sources=not args.no_sources,
        use_hybrid_search=not args.no_hybrid,
        use_rerank=not args.no_rerank,
    )

    if args.rebuild_bm25:
        engine.rebuild_bm25_index()
        return

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
                    print(f"{i}. {source['source']} - {source['section']} (分数: {source['score']})")
    else:
        # 交互模式
        engine.interactive_mode()


if __name__ == "__main__":
    main()
