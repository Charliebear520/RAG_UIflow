"""
BM25關鍵字索引模組
"""

import os
import pickle
import jieba
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    BM25_AVAILABLE = False


@dataclass
class BM25IndexInfo:
    """BM25索引信息"""
    total_documents: int
    vocabulary_size: int
    avg_doc_length: float
    k1: float
    b: float
    metadata: Dict[str, Any]


class BM25KeywordIndex:
    """BM25關鍵字索引類"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1  # 詞頻飽和參數
        self.b = b    # 長度正規化參數
        self.bm25_index = None
        self.index_info: Optional[BM25IndexInfo] = None
        self.chunk_ids: List[str] = []
        self.chunk_doc_ids: List[str] = []
        self.chunks_flat: List[str] = []
        
        # 多層次BM25索引
        self.multi_level_bm25_indices: Dict[str, Any] = {}
        self.multi_level_chunk_ids: Dict[str, List[str]] = {}
        self.multi_level_chunk_doc_ids: Dict[str, List[str]] = {}
        self.multi_level_chunks_flat: Dict[str, List[str]] = {}
        self.multi_level_index_info: Dict[str, BM25IndexInfo] = {}
        
        # 持久化設置
        self.data_dir = "data"
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """確保數據目錄存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"✅ 創建BM25數據目錄: {self.data_dir}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本預處理和分詞"""
        if not text:
            return []
        
        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 使用jieba分詞
        if jieba:
            words = jieba.lcut(text)
            # 過濾停用詞和短詞
            words = [word for word in words if len(word) > 1 and word.strip()]
        else:
            # 簡單的空格分割
            words = text.split()
        
        return words
    
    def _preprocess_documents(self, documents: List[str]) -> List[List[str]]:
        """批量預處理文檔"""
        processed_docs = []
        for doc in documents:
            processed_docs.append(self._preprocess_text(doc))
        return processed_docs
    
    def build_index(self, documents: List[str], chunk_ids: List[str], 
                   chunk_doc_ids: List[str]) -> None:
        """構建BM25索引"""
        if not BM25_AVAILABLE:
            raise RuntimeError("BM25 not available. Please install rank-bm25.")
        
        if not documents:
            return
        
        # 預處理文檔
        processed_docs = self._preprocess_documents(documents)
        
        # 創建BM25索引
        self.bm25_index = BM25Okapi(processed_docs, k1=self.k1, b=self.b)
        
        # 存儲元數據
        self.chunks_flat = documents
        self.chunk_ids = chunk_ids
        self.chunk_doc_ids = chunk_doc_ids
        
        # 計算索引信息
        vocab_size = len(self.bm25_index.idf) if hasattr(self.bm25_index, 'idf') else 0
        avg_doc_length = sum(len(doc) for doc in processed_docs) / len(processed_docs) if processed_docs else 0
        
        self.index_info = BM25IndexInfo(
            total_documents=len(documents),
            vocabulary_size=vocab_size,
            avg_doc_length=avg_doc_length,
            k1=self.k1,
            b=self.b,
            metadata={"created_at": None}
        )
        
        print(f"✅ 構建BM25索引: {len(documents)} 個文檔, 詞彙量: {vocab_size}")
    
    def build_multi_level_index(self, level_name: str, documents: List[str], 
                               chunk_ids: List[str], chunk_doc_ids: List[str]) -> None:
        """構建多層次BM25索引"""
        if not BM25_AVAILABLE:
            raise RuntimeError("BM25 not available. Please install rank-bm25.")
        
        if not documents:
            return
        
        # 預處理文檔
        processed_docs = self._preprocess_documents(documents)
        
        # 創建BM25索引
        bm25_index = BM25Okapi(processed_docs, k1=self.k1, b=self.b)
        
        # 存儲索引和元數據
        self.multi_level_bm25_indices[level_name] = bm25_index
        self.multi_level_chunks_flat[level_name] = documents
        self.multi_level_chunk_ids[level_name] = chunk_ids
        self.multi_level_chunk_doc_ids[level_name] = chunk_doc_ids
        
        # 計算索引信息
        vocab_size = len(bm25_index.idf) if hasattr(bm25_index, 'idf') else 0
        avg_doc_length = sum(len(doc) for doc in processed_docs) / len(processed_docs) if processed_docs else 0
        
        self.multi_level_index_info[level_name] = BM25IndexInfo(
            total_documents=len(documents),
            vocabulary_size=vocab_size,
            avg_doc_length=avg_doc_length,
            k1=self.k1,
            b=self.b,
            metadata={"level": level_name}
        )
        
        print(f"✅ 構建層次 '{level_name}' BM25索引: {len(documents)} 個文檔")
    
    def search(self, query: str, k: int = 10) -> Tuple[List[int], List[float]]:
        """搜索最相關的文檔"""
        if self.bm25_index is None:
            return [], []
        
        if not query:
            return [], []
        
        # 預處理查詢
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return [], []
        
        # 計算BM25分數
        scores = self.bm25_index.get_scores(query_tokens)
        
        # 獲取前k個結果
        if len(scores) == 0:
            return [], []
        
        # 排序並獲取索引
        scored_indices = [(i, score) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        top_k = scored_indices[:k]
        indices = [idx for idx, _ in top_k]
        scores_list = [score for _, score in top_k]
        
        return indices, scores_list
    
    def search_multi_level(self, level_name: str, query: str, k: int = 10) -> Tuple[List[int], List[float]]:
        """搜索指定層次的最相關文檔"""
        if level_name not in self.multi_level_bm25_indices:
            return [], []
        
        bm25_index = self.multi_level_bm25_indices[level_name]
        
        if not query:
            return [], []
        
        # 預處理查詢
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return [], []
        
        # 計算BM25分數
        scores = bm25_index.get_scores(query_tokens)
        
        # 獲取前k個結果
        if len(scores) == 0:
            return [], []
        
        # 排序並獲取索引
        scored_indices = [(i, score) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        top_k = scored_indices[:k]
        indices = [idx for idx, _ in top_k]
        scores_list = [score for _, score in top_k]
        
        return indices, scores_list
    
    def get_chunk_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """根據索引獲取chunk信息"""
        if index < 0 or index >= len(self.chunk_ids):
            return None
        
        return {
            "chunk_id": self.chunk_ids[index],
            "doc_id": self.chunk_doc_ids[index],
            "content": self.chunks_flat[index]
        }
    
    def get_multi_level_chunk_by_index(self, level_name: str, index: int) -> Optional[Dict[str, Any]]:
        """根據索引獲取多層次chunk信息"""
        if level_name not in self.multi_level_chunk_ids:
            return None
        
        chunk_ids = self.multi_level_chunk_ids[level_name]
        if index < 0 or index >= len(chunk_ids):
            return None
        
        return {
            "chunk_id": chunk_ids[index],
            "doc_id": self.multi_level_chunk_doc_ids[level_name][index],
            "content": self.multi_level_chunks_flat[level_name][index]
        }
    
    def get_available_levels(self) -> List[str]:
        """獲取可用的層次"""
        return list(self.multi_level_bm25_indices.keys())
    
    def has_index(self) -> bool:
        """檢查是否有索引"""
        return self.bm25_index is not None
    
    def has_multi_level_index(self) -> bool:
        """檢查是否有多層次索引"""
        return len(self.multi_level_bm25_indices) > 0
    
    def save_data(self) -> None:
        """保存BM25索引和元數據"""
        try:
            # 保存標準索引
            if self.bm25_index is not None:
                with open(os.path.join(self.data_dir, "bm25_index.pkl"), "wb") as f:
                    pickle.dump(self.bm25_index, f)
            
            # 保存多層次索引
            for level_name, bm25_index in self.multi_level_bm25_indices.items():
                with open(os.path.join(self.data_dir, f"bm25_index_{level_name}.pkl"), "wb") as f:
                    pickle.dump(bm25_index, f)
            
            # 保存元數據
            metadata = {
                "index_info": self.index_info.__dict__ if self.index_info else None,
                "chunk_ids": self.chunk_ids,
                "chunk_doc_ids": self.chunk_doc_ids,
                "chunks_flat": self.chunks_flat,
                "k1": self.k1,
                "b": self.b,
                "multi_level_index_info": {k: v.__dict__ for k, v in self.multi_level_index_info.items()},
                "multi_level_chunk_ids": self.multi_level_chunk_ids,
                "multi_level_chunk_doc_ids": self.multi_level_chunk_doc_ids,
                "multi_level_chunks_flat": self.multi_level_chunks_flat
            }
            
            with open(os.path.join(self.data_dir, "bm25_metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
            
            print(f"✅ BM25數據已保存到 {self.data_dir}")
            
        except Exception as e:
            print(f"❌ 保存BM25數據失敗: {e}")
    
    def load_data(self) -> None:
        """載入BM25索引和元數據"""
        try:
            # 載入元數據
            metadata_file = os.path.join(self.data_dir, "bm25_metadata.pkl")
            if not os.path.exists(metadata_file):
                print(f"📁 BM25元數據文件不存在: {metadata_file}")
                return
            
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            
            # 恢復標準索引
            index_file = os.path.join(self.data_dir, "bm25_index.pkl")
            if os.path.exists(index_file):
                with open(index_file, "rb") as f:
                    self.bm25_index = pickle.load(f)
                self.index_info = BM25IndexInfo(**metadata["index_info"]) if metadata["index_info"] else None
            
            # 恢復多層次索引
            for level_name in metadata.get("multi_level_index_info", {}).keys():
                level_index_file = os.path.join(self.data_dir, f"bm25_index_{level_name}.pkl")
                if os.path.exists(level_index_file):
                    with open(level_index_file, "rb") as f:
                        self.multi_level_bm25_indices[level_name] = pickle.load(f)
            
            # 恢復元數據
            self.chunk_ids = metadata.get("chunk_ids", [])
            self.chunk_doc_ids = metadata.get("chunk_doc_ids", [])
            self.chunks_flat = metadata.get("chunks_flat", [])
            self.k1 = metadata.get("k1", self.k1)
            self.b = metadata.get("b", self.b)
            
            # 恢復多層次元數據
            self.multi_level_index_info = {
                k: BM25IndexInfo(**v) for k, v in metadata.get("multi_level_index_info", {}).items()
            }
            self.multi_level_chunk_ids = metadata.get("multi_level_chunk_ids", {})
            self.multi_level_chunk_doc_ids = metadata.get("multi_level_chunk_doc_ids", {})
            self.multi_level_chunks_flat = metadata.get("multi_level_chunks_flat", {})
            
            print(f"✅ BM25數據已從 {self.data_dir} 載入")
            print(f"   📄 標準文檔: {self.index_info.total_documents if self.index_info else 0} 個")
            print(f"   🏗️ 多層次文檔: {len(self.multi_level_bm25_indices)} 個層次")
            
        except Exception as e:
            print(f"❌ 載入BM25數據失敗: {e}")
    
    def reset_index(self) -> None:
        """重置所有索引"""
        self.bm25_index = None
        self.index_info = None
        self.chunk_ids = []
        self.chunk_doc_ids = []
        self.chunks_flat = []
        
        self.multi_level_bm25_indices = {}
        self.multi_level_chunk_ids = {}
        self.multi_level_chunk_doc_ids = {}
        self.multi_level_chunks_flat = {}
        self.multi_level_index_info = {}
        
        print("🗑️ BM25索引數據已重置")
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        stats = {
            "bm25_available": BM25_AVAILABLE,
            "standard_index": {
                "has_index": self.bm25_index is not None,
                "total_documents": self.index_info.total_documents if self.index_info else 0,
                "vocabulary_size": self.index_info.vocabulary_size if self.index_info else 0,
                "avg_doc_length": self.index_info.avg_doc_length if self.index_info else 0,
                "k1": self.k1,
                "b": self.b
            },
            "multi_level_indices": {
                level: {
                    "total_documents": info.total_documents,
                    "vocabulary_size": info.vocabulary_size,
                    "avg_doc_length": info.avg_doc_length,
                    "k1": info.k1,
                    "b": info.b
                }
                for level, info in self.multi_level_index_info.items()
            }
        }
        return stats
