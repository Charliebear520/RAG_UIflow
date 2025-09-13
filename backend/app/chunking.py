"""
文檔分割模組
"""

import re
from typing import List, Dict, Any


class ChunkingStrategy:
    """分割策略基類"""
    
    def chunk(self, text: str, **kwargs) -> List[str]:
        """分割文本"""
        raise NotImplementedError


class FixedSizeChunking(ChunkingStrategy):
    """固定大小分割策略"""
    
    def chunk(self, text: str, chunk_size: int = 500, overlap_ratio: float = 0.1, **kwargs) -> List[str]:
        """固定大小分割"""
        overlap = int(chunk_size * overlap_ratio)
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
            
        return chunks


class HierarchicalChunking(ChunkingStrategy):
    """層次化分割策略"""
    
    def chunk(self, text: str, max_chunk_size: int = 500, **kwargs) -> List[str]:
        """層次化分割"""
        # 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 如果當前段落加上現有chunk超過最大大小，先保存現有chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # 添加最後一個chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks


class SemanticChunking(ChunkingStrategy):
    """語義分割策略"""
    
    def chunk(self, text: str, max_chunk_size: int = 500, **kwargs) -> List[str]:
        """語義分割"""
        # 按句子分割
        sentences = re.split(r'[。！？]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 如果當前句子加上現有chunk超過最大大小，先保存現有chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += "。" + sentence
                else:
                    current_chunk = sentence
        
        # 添加最後一個chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks


class AdaptiveChunking(ChunkingStrategy):
    """自適應分割策略"""
    
    def chunk(self, text: str, base_chunk_size: int = 500, **kwargs) -> List[str]:
        """自適應分割"""
        # 先嘗試語義分割
        semantic_chunker = SemanticChunking()
        chunks = semantic_chunker.chunk(text, max_chunk_size=base_chunk_size)
        
        # 如果某些chunk太大，再進行固定大小分割
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > base_chunk_size * 1.5:
                # 對過大的chunk進行固定大小分割
                fixed_chunker = FixedSizeChunking()
                sub_chunks = fixed_chunker.chunk(chunk, chunk_size=base_chunk_size, overlap_ratio=0.1)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
                
        return final_chunks


def get_chunking_strategy(strategy_name: str) -> ChunkingStrategy:
    """獲取分割策略"""
    strategies = {
        "fixed_size": FixedSizeChunking(),
        "hierarchical": HierarchicalChunking(),
        "semantic": SemanticChunking(),
        "adaptive": AdaptiveChunking(),
    }
    
    return strategies.get(strategy_name, FixedSizeChunking())


def chunk_text(text: str, strategy: str = "fixed_size", **kwargs) -> List[str]:
    """分割文本"""
    chunker = get_chunking_strategy(strategy)
    return chunker.chunk(text, **kwargs)
