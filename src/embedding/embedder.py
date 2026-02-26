"""
向量嵌入模块
使用 PyTorch 加载本地 Embedding 模型
"""
import torch
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from config.settings import EMBEDDING_CONFIG


class Embedder:
    """文本向量化器"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        初始化 Embedding 模型

        Args:
            model_name: 模型名称，默认使用配置文件中的设置
            device: 计算设备 (cuda/cpu)
        """
        self.model_name = model_name or EMBEDDING_CONFIG["model_name"]
        self.device = device or EMBEDDING_CONFIG["device"]
        self.batch_size = EMBEDDING_CONFIG["batch_size"]
        self.max_seq_length = EMBEDDING_CONFIG["max_seq_length"]

        # 延迟加载模型
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        print(f"正在加载 Embedding 模型: {self.model_name}")
        print(f"使用设备: {self.device}")

        try:
            # 优先使用 sentence-transformers
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=EMBEDDING_CONFIG["cache_dir"],
                device=self.device,
            )
            self._model.max_seq_length = self.max_seq_length
            self._is_sentence_transformer = True
            print("模型加载完成 (sentence-transformers)")

        except ImportError:
            # 回退到 transformers
            from transformers import AutoTokenizer, AutoModel

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=EMBEDDING_CONFIG["cache_dir"],
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=EMBEDDING_CONFIG["cache_dir"],
            ).to(self.device)
            self._model.eval()
            self._is_sentence_transformer = False
            print("模型加载完成 (transformers)")

    def embed(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        将文本列表转换为向量

        Args:
            texts: 文本列表
            show_progress: 是否显示进度条

        Returns:
            向量数组，shape=(len(texts), embedding_dim)
        """
        self._load_model()

        if not texts:
            return np.array([])

        if self._is_sentence_transformer:
            return self._embed_with_sentence_transformer(texts, show_progress)
        else:
            return self._embed_with_transformers(texts, show_progress)

    def _embed_with_sentence_transformer(
        self, texts: List[str], show_progress: bool
    ) -> np.ndarray:
        """使用 sentence-transformers 编码"""
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 归一化便于余弦相似度计算
        )
        return embeddings

    def _embed_with_transformers(
        self, texts: List[str], show_progress: bool
    ) -> np.ndarray:
        """使用原生 transformers 编码"""
        all_embeddings = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + self.batch_size]

                # Tokenize
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self.device)

                # 获取模型输出
                outputs = self._model(**encoded)

                # 使用 mean pooling
                attention_mask = encoded["attention_mask"]
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state, attention_mask
                )

                # 归一化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean Pooling - 计算词向量的加权平均"""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_single(self, text: str) -> np.ndarray:
        """将单个文本转换为向量"""
        return self.embed([text], show_progress=False)[0]

    @property
    def embedding_dim(self) -> int:
        """获取向量维度"""
        self._load_model()
        if self._is_sentence_transformer:
            return self._model.get_sentence_embedding_dimension()
        else:
            return self._model.config.hidden_size


def test_embedder():
    """测试 Embedder"""
    embedder = Embedder()

    # 测试文本
    texts = [
        "第一章 总则",
        "第一条 为规范公司管理，特制定本制度。",
        "第二条 本制度适用于公司全体员工。",
    ]

    embeddings = embedder.embed(texts)
    print(f"向量形状: {embeddings.shape}")
    print(f"向量维度: {embedder.embedding_dim}")

    # 测试相似度
    from numpy.linalg import norm

    cos_sim = np.dot(embeddings[1], embeddings[2]) / (
        norm(embeddings[1]) * norm(embeddings[2])
    )
    print(f"文本2和文本3的余弦相似度: {cos_sim:.4f}")


if __name__ == "__main__":
    test_embedder()
