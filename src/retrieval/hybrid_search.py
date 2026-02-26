"""
混合检索模块
结合 BM25 (关键词匹配) 和 Dense Retrieval (语义检索)
基于 RAG 论文 (Lewis et al., 2020) 的优化策略
"""
import re
import math
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from config.settings import VECTOR_STORE_CONFIG, RETRIEVAL_CONFIG
from .vector_store import Document


@dataclass
class SearchResult:
    """检索结果"""
    document: Document
    score: float
    source: str  # "dense", "bm25", "hybrid"


class BM25Index:
    """
    BM25 索引 - 基于 TF-IDF 的关键词检索
    论文发现：BM25 + Dense Retrieval 结合效果更好
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # 索引数据
        self._documents: List[Document] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0

        # 逆向索引
        self._inverted_index: Dict[str, List[int]] = {}
        self._idf: Dict[str, float] = {}
        self._doc_freq: Dict[str, int] = {}

    def build_index(self, documents: List[Document]):
        """构建 BM25 索引"""
        self._documents = documents
        self._doc_tokens = []
        self._doc_lengths = []
        self._inverted_index = {}
        self._doc_freq = {}

        # 分词并构建索引
        for doc_id, doc in enumerate(documents):
            tokens = self._tokenize(doc.content)
            self._doc_tokens.append(tokens)
            self._doc_lengths.append(len(tokens))

            # 更新逆向索引
            token_set = set(tokens)
            for token in token_set:
                if token not in self._inverted_index:
                    self._inverted_index[token] = []
                self._inverted_index[token].append(doc_id)

                # 文档频率
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

        # 计算平均文档长度
        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 1

        # 计算 IDF
        total_docs = len(documents)
        for token, df in self._doc_freq.items():
            self._idf[token] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

    def _tokenize(self, text: str) -> List[str]:
        """中文分词 - 简单的字符级分词 + 关键词提取"""
        # 移除标点和特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)

        # 提取中文词组和数字
        tokens = []

        # 提取中文 (按字符)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        for phrase in chinese_chars:
            # 单字
            tokens.extend(list(phrase))
            # 双字组合
            if len(phrase) >= 2:
                tokens.extend([phrase[i:i+2] for i in range(len(phrase)-1)])
            # 三字组合 (适用于专有名词)
            if len(phrase) >= 3:
                tokens.extend([phrase[i:i+3] for i in range(len(phrase)-2)])

        # 提取数字和英文
        tokens.extend(re.findall(r'[a-zA-Z]+|\d+', text.lower()))

        return tokens

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """BM25 搜索"""
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        scores = {}

        for token in query_tokens:
            if token not in self._inverted_index:
                continue

            idf = self._idf.get(token, 0)

            for doc_id in self._inverted_index[token]:
                doc_tokens = self._doc_tokens[doc_id]
                tf = doc_tokens.count(token)
                doc_length = self._doc_lengths[doc_id]

                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length)
                score = idf * numerator / denominator

                scores[doc_id] = scores.get(doc_id, 0) + score

        # 排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [(self._documents[doc_id], score) for doc_id, score in ranked]

    def save(self, path: Path):
        """保存索引"""
        data = {
            "documents": [(doc.content, doc.metadata) for doc in self._documents],
            "doc_tokens": self._doc_tokens,
            "doc_lengths": self._doc_lengths,
            "avg_doc_length": self._avg_doc_length,
            "inverted_index": self._inverted_index,
            "idf": self._idf,
            "doc_freq": self._doc_freq,
            "k1": self.k1,
            "b": self.b,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Path):
        """加载索引"""
        if not path.exists():
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._documents = [
            Document(id=i, content=content, metadata=meta)
            for i, (content, meta) in enumerate(data["documents"])
        ]
        self._doc_tokens = data["doc_tokens"]
        self._doc_lengths = data["doc_lengths"]
        self._avg_doc_length = data["avg_doc_length"]
        self._inverted_index = data["inverted_index"]
        self._idf = data["idf"]
        self._doc_freq = data["doc_freq"]
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)

        return True


class HybridSearcher:
    """
    混合检索器 - 结合 Dense Retrieval 和 BM25
    论文发现：混合检索在多个任务上表现更好
    """

    def __init__(
        self,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        use_rrf: bool = True,
        rrf_k: int = 60,
    ):
        """
        初始化混合检索器

        Args:
            dense_weight: Dense Retrieval 权重
            bm25_weight: BM25 权重
            use_rrf: 是否使用 Reciprocal Rank Fusion
            rrf_k: RRF 参数
        """
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k

        self._bm25_index: Optional[BM25Index] = None

    def build_bm25_index(self, documents: List[Document]):
        """构建 BM25 索引"""
        self._bm25_index = BM25Index()
        self._bm25_index.build_index(documents)

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        dense_results: List[Tuple[Document, float]],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        混合搜索

        Args:
            query: 查询文本
            query_embedding: 查询向量
            dense_results: Dense Retrieval 结果
            top_k: 返回数量

        Returns:
            混合检索结果
        """
        import numpy as np

        if self._bm25_index is None:
            # 只有 Dense 结果
            return [
                SearchResult(document=doc, score=score, source="dense")
                for doc, score in dense_results[:top_k]
            ]

        # BM25 检索
        bm25_results = self._bm25_index.search(query, top_k=top_k * 2)

        if self.use_rrf:
            # Reciprocal Rank Fusion
            return self._rrf_fusion(dense_results, bm25_results, top_k)
        else:
            # 加权分数融合
            return self._weighted_fusion(dense_results, bm25_results, top_k)

    def _rrf_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion - 基于排名的融合
        RRF(d) = Σ 1/(k + rank(d))
        """
        rrf_scores: Dict[int, float] = {}  # doc_id -> score
        doc_map: Dict[int, Document] = {}  # doc_id -> Document

        # Dense 排名
        for rank, (doc, score) in enumerate(dense_results, 1):
            doc_id = doc.id
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (self.rrf_k + rank)

        # BM25 排名
        for rank, (doc, score) in enumerate(bm25_results, 1):
            doc_id = doc.id
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (self.rrf_k + rank)

        # 排序
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            SearchResult(document=doc_map[doc_id], score=score, source="hybrid")
            for doc_id, score in ranked
        ]

    def _weighted_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[SearchResult]:
        """加权分数融合"""
        import numpy as np

        # 归一化分数
        def normalize(scores):
            if not scores:
                return {}
            max_score = max(s for _, s in scores)
            min_score = min(s for _, s in scores)
            range_score = max_score - min_score if max_score != min_score else 1
            return {doc.id: (score - min_score) / range_score for doc, score in scores}

        dense_norm = normalize(dense_results)
        bm25_norm = normalize(bm25_results)

        # 合并分数
        all_docs: Dict[int, Document] = {}
        for doc, _ in dense_results + bm25_results:
            all_docs[doc.id] = doc

        fused_scores = {}
        for doc_id, doc in all_docs.items():
            d_score = dense_norm.get(doc_id, 0) * self.dense_weight
            b_score = bm25_norm.get(doc_id, 0) * self.bm25_weight
            fused_scores[doc_id] = d_score + b_score

        # 排序
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            SearchResult(document=all_docs[doc_id], score=score, source="hybrid")
            for doc_id, score in ranked
        ]

    def save_bm25_index(self, path: Path):
        """保存 BM25 索引"""
        if self._bm25_index:
            self._bm25_index.save(path)

    def load_bm25_index(self, path) -> bool:
        """加载 BM25 索引"""
        from pathlib import Path
        path = Path(path) if not isinstance(path, Path) else path

        self._bm25_index = BM25Index()
        return self._bm25_index.load(path)


# 导入 numpy (用于类型提示)
import numpy as np
