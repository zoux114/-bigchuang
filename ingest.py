"""
增量索引脚本
读取 Markdown 文件 -> 语义分段 -> 生成向量 -> 存入 FAISS
"""
import os
import sys
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from config.settings import (
    PROCESSED_DATA_DIR,
    VECTOR_STORE_CONFIG,
)
from src.etl.document_processor import DocumentProcessor
from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore, Document
from src.utils.chunker import TextChunker


class IngestPipeline:
    """增量索引流水线"""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.embedder = Embedder()
        self.chunker = TextChunker()
        self.vector_store = VectorStore()

    def run(
        self,
        etl_only: bool = False,
        index_only: bool = False,
        force: bool = False,
    ):
        """
        运行索引流水线

        Args:
            etl_only: 仅运行 ETL (文档处理)
            index_only: 仅运行索引 (跳过 ETL)
            force: 强制重新处理所有文件
        """
        print("=" * 60)
        print("规章制度智能问答系统 - 增量索引")
        print("=" * 60)

        # Step 1: ETL - 处理原始文档
        if not index_only:
            print("\n[Step 1/2] 处理原始文档...")
            etl_results = self.processor.process_all(force=force)
            self._print_results("ETL", etl_results)
        else:
            print("\n[Step 1/2] 跳过 ETL (--index-only)")

        if etl_only:
            print("\n已完成 ETL 处理，跳过索引构建 (--etl-only)")
            return

        # Step 2: 索引 - 构建向量数据库
        print("\n[Step 2/2] 构建向量索引...")
        index_results = self._build_index(force=force)
        self._print_results("索引", index_results)

        # 保存索引
        self.vector_store.save()

        print("\n" + "=" * 60)
        print(f"索引完成！共 {self.vector_store.total_documents} 个文档片段")
        print("=" * 60)

    def _build_index(self, force: bool = False) -> Dict[str, str]:
        """
        构建向量索引

        Args:
            force: 强制重新索引所有文件

        Returns:
            处理结果字典
        """
        results = {}
        files_to_process = []

        # 扫描处理后的 Markdown 文件
        for md_file in PROCESSED_DATA_DIR.glob("*.md"):
            file_hash = self._compute_file_hash(md_file)
            stored_hash = self.vector_store.get_file_hash(str(md_file))

            if force or stored_hash is None or stored_hash != file_hash:
                files_to_process.append((md_file, file_hash))
                results[md_file.name] = "pending"
            else:
                results[md_file.name] = "skipped (unchanged)"

        if not files_to_process:
            print("没有需要处理的文件")
            return results

        print(f"发现 {len(files_to_process)} 个文件需要处理")

        # 处理每个文件
        all_chunks = []
        all_embeddings = []

        for md_file, file_hash in tqdm(files_to_process, desc="处理文件"):
            try:
                # 读取文件内容
                content = md_file.read_text(encoding="utf-8")

                # 移除 YAML 头部
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        content = parts[2].strip()

                # 分块
                chunks = self.chunker.chunk_markdown(content, md_file.name)

                if chunks:
                    # 生成向量
                    texts = [chunk.content for chunk in chunks]
                    embeddings = self.embedder.embed(texts, show_progress=False)

                    all_chunks.extend(chunks)
                    all_embeddings.append(embeddings)

                    # 更新文件哈希
                    self.vector_store.set_file_hash(str(md_file), file_hash)
                    results[md_file.name] = f"success ({len(chunks)} chunks)"
                else:
                    results[md_file.name] = "skipped (empty)"

            except Exception as e:
                results[md_file.name] = f"failed: {str(e)}"

        # 批量添加到向量存储
        if all_chunks:
            import numpy as np
            all_embeddings = np.vstack(all_embeddings)

            documents = [
                Document(
                    id=0,  # 会被 VectorStore 自动分配
                    content=chunk.content,
                    metadata=chunk.metadata,
                )
                for chunk in all_chunks
            ]

            self.vector_store.add_documents(documents, all_embeddings)

        return results

    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        content = file_path.read_bytes()
        return hashlib.md5(content).hexdigest()

    def _print_results(self, stage: str, results: Dict[str, str]):
        """打印处理结果"""
        success = sum(1 for v in results.values() if v.startswith("success"))
        skipped = sum(1 for v in results.values() if v.startswith("skipped"))
        failed = sum(1 for v in results.values() if v.startswith("failed"))

        print(f"\n{stage} 结果统计:")
        print(f"  成功: {success}")
        print(f"  跳过: {skipped}")
        print(f"  失败: {failed}")

        if failed > 0:
            print("\n失败详情:")
            for file_name, status in results.items():
                if status.startswith("failed"):
                    print(f"  {file_name}: {status}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="增量索引脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ingest.py                    # 运行完整流水线 (增量)
  python ingest.py --force            # 强制重新处理所有文件
  python ingest.py --etl-only         # 仅处理文档，不构建索引
  python ingest.py --index-only       # 仅构建索引，跳过文档处理
        """,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理所有文件",
    )
    parser.add_argument(
        "--etl-only",
        action="store_true",
        help="仅运行 ETL (文档处理)",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="仅运行索引构建",
    )

    args = parser.parse_args()

    pipeline = IngestPipeline()
    pipeline.run(
        etl_only=args.etl_only,
        index_only=args.index_only,
        force=args.force,
    )


if __name__ == "__main__":
    main()
