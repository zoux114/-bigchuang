"""
向量数据库模块
使用 FAISS 进行高效向量检索
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    import faiss
except ImportError:
    raise ImportError("请安装 faiss: pip install faiss-cpu 或 faiss-gpu")

from config.settings import VECTOR_STORE_CONFIG, RETRIEVAL_CONFIG


@dataclass
class Document:
    """文档数据结构"""
    id: int
    content: str
    metadata: Dict  # 包含 source, section, chunk_index 等
    embedding: Optional[np.ndarray] = None


class VectorStore:
    """FAISS 向量存储"""

    def __init__(self, index_name: Optional[str] = None):
        """
        初始化向量存储

        Args:
            index_name: 索引名称
        """
        self.db_path = Path(VECTOR_STORE_CONFIG["db_path"])
        self.index_name = index_name or VECTOR_STORE_CONFIG["index_name"]

        # FAISS 索引
        self._index: Optional[faiss.Index] = None
        # 文档元数据存储
        self._documents: List[Document] = []
        # 文件哈希记录 (用于增量更新)
        self._file_hashes: Dict[str, str] = {}

        # 加载现有索引
        self._load()

    def _load(self):
        """加载现有索引"""
        index_path = self.db_path / f"{self.index_name}.index"
        docs_path = self.db_path / f"{self.index_name}.docs"
        hashes_path = self.db_path / f"{self.index_name}.hashes"

        if index_path.exists():
            try:
                # 使用 deserialize_index 避免中文路径问题
                with open(str(index_path), "rb") as f:
                    index_bytes = f.read()
                # 将 bytes 转换为 numpy 数组
                index_array = np.frombuffer(index_bytes, dtype=np.uint8)
                self._index = faiss.deserialize_index(index_array)
                print(f"加载索引: {index_path} (共 {self._index.ntotal} 个向量)")
            except Exception as e:
                print(f"加载索引失败: {e}")
                self._index = None

        if docs_path.exists():
            with open(docs_path, "rb") as f:
                self._documents = pickle.load(f)

        if hashes_path.exists():
            with open(hashes_path, "r", encoding="utf-8") as f:
                self._file_hashes = json.load(f)

    def save(self):
        """保存索引和元数据"""
        self.db_path.mkdir(parents=True, exist_ok=True)

        index_path = self.db_path / f"{self.index_name}.index"
        docs_path = self.db_path / f"{self.index_name}.docs"
        hashes_path = self.db_path / f"{self.index_name}.hashes"

        if self._index is not None:
            try:
                # 使用 serialize_index 避免中文路径问题
                index_bytes = faiss.serialize_index(self._index)
                with open(str(index_path), "wb") as f:
                    f.write(index_bytes)
            except Exception as e:
                print(f"保存索引失败: {e}")
                raise

        with open(docs_path, "wb") as f:
            pickle.dump(self._documents, f)

        with open(hashes_path, "w", encoding="utf-8") as f:
            json.dump(self._file_hashes, f, ensure_ascii=False, indent=2)

        print(f"索引已保存: {index_path}")

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
    ):
        """
        添加文档和向量到索引

        Args:
            documents: 文档列表
            embeddings: 对应的向量数组
        """
        if len(documents) != len(embeddings):
            raise ValueError("文档数量和向量数量不匹配")

        if len(documents) == 0:
            return

        # 确保向量是 float32 类型
        embeddings = embeddings.astype("float32")

        # 初始化索引 (如果需要)
        if self._index is None:
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)  # 内积相似度 (配合归一化向量)
            print(f"创建新索引，维度: {dimension}")

        # 分配 ID
        start_id = len(self._documents)
        for i, doc in enumerate(documents):
            doc.id = start_id + i

        # 添加到索引
        self._index.add(embeddings)

        # 添加到文档列表
        self._documents.extend(documents)

        print(f"添加 {len(documents)} 个文档，总文档数: {len(self._documents)}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """
        搜索相似文档

        Args:
            query_embedding: 查询向量
            top_k: 返回的文档数量
            threshold: 相似度阈值

        Returns:
            (文档, 相似度) 列表
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        top_k = top_k or RETRIEVAL_CONFIG["top_k"]
        threshold = threshold or RETRIEVAL_CONFIG["similarity_threshold"]

        # 确保向量格式正确
        query_embedding = query_embedding.astype("float32").reshape(1, -1)

        # 搜索
        scores, indices = self._index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                results.append((self._documents[idx], float(score)))

        return results

    def remove_by_source(self, source_file: str):
        """删除指定源文件的所有文档"""
        # FAISS 不支持删除，需要重建索引
        remaining_docs = [
            doc for doc in self._documents
            if doc.metadata.get("source") != source_file
        ]

        if len(remaining_docs) == len(self._documents):
            return  # 没有需要删除的

        # 保存剩余文档
        old_docs = self._documents
        self._documents = []
        self._index = None

        # 返回需要重新索引的文档
        return remaining_docs

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """获取文件的记录哈希"""
        return self._file_hashes.get(file_path)

    def set_file_hash(self, file_path: str, file_hash: str):
        """设置文件哈希"""
        self._file_hashes[file_path] = file_hash

    def clear(self):
        """清空索引"""
        self._index = None
        self._documents = []
        self._file_hashes = {}

    @property
    def total_documents(self) -> int:
        """总文档数"""
        return len(self._documents)

    @property
    def is_empty(self) -> bool:
        """索引是否为空"""
        return self._index is None or self._index.ntotal == 0


def test_vector_store():
    """测试向量存储"""
    store = VectorStore("test_index")

    # 模拟文档
    docs = [
        Document(
            id=0,
            content="这是第一条规章制度",
            metadata={"source": "test.md", "section": "第一章"},
        ),
        Document(
            id=1,
            content="这是第二条规章制度",
            metadata={"source": "test.md", "section": "第二章"},
        ),
    ]

    # 模拟向量
    embeddings = np.random.randn(2, 768).astype("float32")

    store.add_documents(docs, embeddings)
    store.save()

    # 测试搜索
    query = np.random.randn(768).astype("float32")
    results = store.search(query, top_k=2)
    print(f"搜索结果: {len(results)} 条")


if __name__ == "__main__":
    test_vector_store()
