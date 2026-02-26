"""
重排序模块
对初步检索结果进行精细排序
基于 RAG 论文的优化策略
"""
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .vector_store import Document


@dataclass
class RerankedResult:
    """重排序结果"""
    document: Document
    initial_score: float
    rerank_score: float
    final_score: float
    factors: Dict[str, float]


class ReRanker:
    """
    重排序器 - 多因素综合评分

    基于以下因素重排序：
    1. 原始相似度分数
    2. 查询词覆盖率
    3. 文档结构相关性 (章节标题匹配)
    4. 文档新鲜度 (可选)
    """

    def __init__(
        self,
        similarity_weight: float = 0.5,
        coverage_weight: float = 0.2,
        structure_weight: float = 0.2,
        freshness_weight: float = 0.1,
    ):
        """
        初始化重排序器

        Args:
            similarity_weight: 原始相似度权重
            coverage_weight: 查询词覆盖率权重
            structure_weight: 结构相关性权重
            freshness_weight: 新鲜度权重
        """
        self.weights = {
            "similarity": similarity_weight,
            "coverage": coverage_weight,
            "structure": structure_weight,
            "freshness": freshness_weight,
        }

    def rerank(
        self,
        query: str,
        results: List[Tuple[Document, float]],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """
        重排序检索结果

        Args:
            query: 用户查询
            results: 原始检索结果 [(Document, score), ...]
            top_k: 返回数量

        Returns:
            重排序后的结果
        """
        if not results:
            return []

        # 提取查询关键词
        query_keywords = self._extract_keywords(query)

        reranked = []
        for doc, initial_score in results:
            factors = {}

            # 1. 相似度分数 (归一化到 0-1)
            factors["similarity"] = self._normalize_score(initial_score, results)

            # 2. 查询词覆盖率
            factors["coverage"] = self._calculate_coverage(query_keywords, doc.content)

            # 3. 结构相关性
            factors["structure"] = self._calculate_structure_relevance(query_keywords, doc)

            # 4. 新鲜度 (如果有时间信息)
            factors["freshness"] = self._calculate_freshness(doc)

            # 计算最终分数
            final_score = sum(
                factors[key] * self.weights[key]
                for key in self.weights
            )

            reranked.append(RerankedResult(
                document=doc,
                initial_score=initial_score,
                rerank_score=sum(factors.values()) / len(factors),
                final_score=final_score,
                factors=factors,
            ))

        # 按最终分数排序
        reranked.sort(key=lambda x: x.final_score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 中文分词 - 简单实现
        keywords = []

        # 提取中文词组
        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        for phrase in chinese:
            # 单字
            keywords.extend(list(phrase))
            # 双字组合
            if len(phrase) >= 2:
                keywords.extend([phrase[i:i+2] for i in range(len(phrase)-1)])

        # 提取数字和英文
        keywords.extend(re.findall(r'[a-zA-Z]+|\d+', text.lower()))

        return list(set(keywords))

    def _normalize_score(
        self,
        score: float,
        all_results: List[Tuple[Document, float]],
    ) -> float:
        """归一化分数到 0-1"""
        if not all_results:
            return 0.0

        scores = [s for _, s in all_results]
        max_score = max(scores)
        min_score = min(scores)

        if max_score == min_score:
            return 1.0

        return (score - min_score) / (max_score - min_score)

    def _calculate_coverage(self, query_keywords: List[str], content: str) -> float:
        """计算查询词覆盖率"""
        if not query_keywords:
            return 0.0

        content_lower = content.lower()
        matched = sum(1 for kw in query_keywords if kw.lower() in content_lower)

        return matched / len(query_keywords)

    def _calculate_structure_relevance(
        self,
        query_keywords: List[str],
        doc: Document,
    ) -> float:
        """计算结构相关性 - 章节标题匹配"""
        section = doc.metadata.get("section", "")
        if not section:
            return 0.5  # 没有章节信息时返回中性分数

        section_lower = section.lower()
        matched = sum(1 for kw in query_keywords if kw.lower() in section_lower)

        # 如果章节标题匹配多个关键词，给予高分
        if matched > 0:
            return min(1.0, 0.5 + matched * 0.25)

        return 0.3  # 不匹配时返回较低分数

    def _calculate_freshness(self, doc: Document) -> float:
        """计算新鲜度 (暂时返回中性分数)"""
        # 可以根据文档的 processed_at 时间来计算
        # 这里暂时返回 0.5 (中性)
        return 0.5


class QueryAnalyzer:
    """
    查询分析器 - 分析查询类型和复杂度
    用于动态调整检索策略
    """

    # 问题类型模式
    QUESTION_PATTERNS = {
        "definition": [r"什么是", r"定义", r"是指", r"意思是"],
        "procedure": [r"如何", r"怎么", r"步骤", r"流程", r"方法"],
        "quantity": [r"多少", r"几天", r"几次", r"多长"],
        "condition": [r"条件", r"要求", r"资格", r"符合"],
        "comparison": [r"区别", r"对比", r"不同", r"差异"],
        "list": [r"有哪些", r"包括", r"种类", r"分类"],
    }

    # 复杂度指标
    COMPLEXITY_INDICATORS = {
        "high": [r"详细", r"全面", r"所有", r"完整"],
        "medium": [r"具体", r"相关", r"主要"],
        "simple": [r"简要", r"概括", r"简单"],
    }

    def analyze(self, query: str) -> Dict:
        """
        分析查询

        Returns:
            {
                "type": 问题类型,
                "complexity": 复杂度 (1-3),
                "suggested_top_k": 建议的检索数量,
                "keywords": 关键词列表,
            }
        """
        result = {
            "type": self._detect_type(query),
            "complexity": self._detect_complexity(query),
            "suggested_top_k": self._suggest_top_k(query),
            "keywords": self._extract_keywords(query),
        }

        return result

    def _detect_type(self, query: str) -> str:
        """检测问题类型"""
        for q_type, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return q_type
        return "general"

    def _detect_complexity(self, query: str) -> int:
        """检测复杂度 (1=简单, 2=中等, 3=复杂)"""
        for level, patterns in self.COMPLEXITY_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return 3 if level == "high" else (2 if level == "medium" else 1)

        # 根据查询长度估计
        if len(query) > 50:
            return 3
        elif len(query) > 20:
            return 2
        return 1

    def _suggest_top_k(self, query: str) -> int:
        """建议检索数量"""
        complexity = self._detect_complexity(query)
        q_type = self._detect_type(query)

        # 根据类型和复杂度调整
        base = {
            "definition": 3,
            "quantity": 2,
            "procedure": 5,
            "condition": 4,
            "comparison": 6,
            "list": 8,
            "general": 5,
        }

        return base.get(q_type, 5) * (0.7 + 0.3 * complexity)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []

        # 中文
        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        for phrase in chinese:
            keywords.extend(list(phrase))
            if len(phrase) >= 2:
                keywords.extend([phrase[i:i+2] for i in range(len(phrase)-1)])

        # 英文和数字
        keywords.extend(re.findall(r'[a-zA-Z]+|\d+', text.lower()))

        return list(set(keywords))
