"""
文本分块工具
针对规章制度文档的智能分块
"""
import re
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config.settings import CHUNKING_CONFIG


@dataclass
class Chunk:
    """文本分块"""
    content: str
    metadata: Dict
    hash: str


class TextChunker:
    """文本分块器 - 针对规章制度优化"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or CHUNKING_CONFIG["chunk_size"]
        self.chunk_overlap = chunk_overlap or CHUNKING_CONFIG["chunk_overlap"]
        self.section_patterns = CHUNKING_CONFIG["section_patterns"]

    def chunk_markdown(
        self,
        content: str,
        source: str,
    ) -> List[Chunk]:
        """
        将 Markdown 内容分块

        Args:
            content: Markdown 文本
            source: 源文件名

        Returns:
            分块列表
        """
        # 首先按章节分割
        sections = self._split_by_sections(content)

        chunks = []
        chunk_index = 0

        for section_title, section_content in sections:
            # 对每个章节进行进一步分块
            section_chunks = self._chunk_section(
                section_content,
                self.chunk_size,
            )

            for chunk_content in section_chunks:
                chunk = Chunk(
                    content=chunk_content.strip(),
                    metadata={
                        "source": source,
                        "section": section_title,
                        "chunk_index": chunk_index,
                    },
                    hash=self._compute_hash(chunk_content),
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _split_by_sections(self, content: str) -> List[Tuple[str, str]]:
        """
        按章节分割文档

        Returns:
            [(章节标题, 章节内容), ...]
        """
        lines = content.split("\n")
        sections = []
        current_title = "文档开头"
        current_content = []

        for line in lines:
            # 检测章节标题
            is_section = False
            for pattern in self.section_patterns:
                if re.match(pattern, line.strip()):
                    # 保存之前的章节
                    if current_content:
                        sections.append((current_title, "\n".join(current_content)))
                    current_title = line.strip()
                    current_content = []
                    is_section = True
                    break

            if not is_section:
                current_content.append(line)

        # 保存最后一个章节
        if current_content:
            sections.append((current_title, "\n".join(current_content)))

        return sections

    def _chunk_section(
        self,
        content: str,
        max_size: int,
    ) -> List[str]:
        """
        对章节内容进行分块

        Args:
            content: 章节内容
            max_size: 最大分块大小

        Returns:
            分块内容列表
        """
        if len(content) <= max_size:
            return [content]

        # 按段落分割
        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果单个段落超过最大大小，需要进一步分割
            if len(para) > max_size:
                # 先保存当前累积的内容
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []

                # 按句子分割大段落
                sentences = self._split_sentences(para)
                temp_chunk = []

                for sentence in sentences:
                    if len("\n".join(temp_chunk)) + len(sentence) > max_size:
                        if temp_chunk:
                            chunks.append("\n".join(temp_chunk))
                        temp_chunk = [sentence]
                    else:
                        temp_chunk.append(sentence)

                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                # 检查添加这个段落是否会超过大小限制
                test_chunk = current_chunk + [para]
                if len("\n\n".join(test_chunk)) > max_size:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
                else:
                    current_chunk.append(para)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # 添加重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """按句子分割文本"""
        # 中文句子分割
        pattern = r'(?<=[。！？；\n])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加分块重叠"""
        overlapped = []

        for i, chunk in enumerate(chunks):
            if i > 0:
                # 从上一个块的末尾取一部分作为当前块的开头
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:]
                chunk = overlap_text + "\n" + chunk

            overlapped.append(chunk)

        return overlapped

    def _compute_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()


def test_chunker():
    """测试分块器"""
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)

    sample_content = """
# 公司规章制度

## 第一章 总则

第一条 为规范公司管理，根据国家相关法律法规，特制定本制度。

第二条 本制度适用于公司全体员工，包括正式员工、试用期员工和实习生。

## 第二章 考勤管理

第三条 员工应按时上下班，不得迟到早退。

第四条 工作时间为周一至周五，上午9:00至下午6:00，午休时间为12:00至13:00。
"""

    chunks = chunker.chunk_markdown(sample_content, "test.md")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Section: {chunk.metadata['section']}")
        print(f"Content: {chunk.content[:100]}...")
        print(f"Hash: {chunk.hash}")


if __name__ == "__main__":
    test_chunker()
