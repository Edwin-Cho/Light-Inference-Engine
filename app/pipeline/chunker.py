from __future__ import annotations

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def chunk_text(text: str, metadata: dict) -> List[dict]:
    """
    텍스트를 Recursive Character Text Splitter로 분할 후,
    각 청크에 메타데이터를 바인딩하여 반환.

    Returns:
        List of dicts: [{"text": str, "metadata": dict}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)

    return [
        {"text": chunk, "metadata": {**metadata, "chunk_index": i}}
        for i, chunk in enumerate(chunks)
        if chunk.strip()
    ]
