"""
FAISS向量存儲模組
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import faiss

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False


@dataclass
class FAISSIndexInfo:
    """FAISS索引信息"""
    index_type: str
    dimension: int
    total_vectors: int
    is_trained: bool
    metadata: Dict[str, Any]


class FAISSVectorStore:
    """FAISS向量存儲類"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.index_info: Optional[FAISSIndexInfo] = None
        self.chunk_ids: List[str] = []
        self.chunk_doc_ids: List[str] = []
        self.chunks_flat: List[str] = []
        self.enhanced_metadata: Dict[str, Dict] = {}
        
        # 多層次embedding存儲
        self.multi_level_indices: Dict[str, Any] = {}
        self.multi_level_chunk_ids: Dict[str, List[str]] = {}
        self.multi_level_chunk_doc_ids: Dict[str, List[str]] = {}
        self.multi_level_chunks_flat: Dict[str, List[str]] = {}
        self.multi_level_enhanced_metadata: Dict[str, Dict[str, Dict]] = {}
        self.multi_level_index_info: Dict[str, FAISSIndexInfo] = {}
        
        # 持久化設置
        self.data_dir = "data"
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """確保數據目錄存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"✅ 創建FAISS數據目錄: {self.data_dir}")
    
    def create_index(self, dimension: int, index_type: str = "flat") -> None:
        """創建FAISS索引"""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available. Please install faiss-cpu or faiss-gpu.")
        
        self.dimension = dimension
        
        if index_type == "flat":
            # 使用FlatL2索引（精確搜索）
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        elif index_type == "ivf":
            # 使用IVF索引（近似搜索，適合大規模數據）
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.index_info = FAISSIndexInfo(
            index_type=index_type,
            dimension=dimension,
            total_vectors=0,
            is_trained=index_type != "ivf",
            metadata={"created_at": None}
        )
        
        print(f"✅ 創建FAISS索引: {index_type}, 維度: {dimension}")
    
    def add_vectors(self, vectors: List[List[float]], chunk_ids: List[str], 
                   chunk_doc_ids: List[str], chunks_flat: List[str]) -> None:
        """添加向量到索引"""
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        if not vectors:
            return
        
        # 轉換為numpy數組
        vectors_array = np.array(vectors, dtype=np.float32)
        
        # 正規化向量（用於cosine similarity）
        faiss.normalize_L2(vectors_array)
        
        # 如果使用IVF索引且未訓練，需要先訓練
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("🔧 訓練IVF索引...")
            self.index.train(vectors_array)
            self.index_info.is_trained = True
        
        # 添加向量到索引
        self.index.add(vectors_array)
        
        # 更新元數據
        self.chunk_ids.extend(chunk_ids)
        self.chunk_doc_ids.extend(chunk_doc_ids)
        self.chunks_flat.extend(chunks_flat)
        self.index_info.total_vectors = len(self.chunk_ids)
        
        print(f"✅ 添加 {len(vectors)} 個向量到FAISS索引，總計: {self.index_info.total_vectors}")
    
    def add_multi_level_vectors(self, level_name: str, vectors: List[List[float]], 
                               chunk_ids: List[str], chunk_doc_ids: List[str], 
                               chunks_flat: List[str]) -> None:
        """添加多層次向量到索引"""
        if not vectors:
            return
        
        # 為該層次創建索引
        dimension = len(vectors[0])
        level_index = faiss.IndexFlatIP(dimension)
        
        # 轉換為numpy數組並正規化
        vectors_array = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_array)
        
        # 添加向量
        level_index.add(vectors_array)
        
        # 存儲索引和元數據
        self.multi_level_indices[level_name] = level_index
        self.multi_level_chunk_ids[level_name] = chunk_ids
        self.multi_level_chunk_doc_ids[level_name] = chunk_doc_ids
        self.multi_level_chunks_flat[level_name] = chunks_flat
        
        # 創建索引信息
        self.multi_level_index_info[level_name] = FAISSIndexInfo(
            index_type="flat",
            dimension=dimension,
            total_vectors=len(chunk_ids),
            is_trained=True,
            metadata={"level": level_name}
        )
        
        print(f"✅ 添加 {len(vectors)} 個向量到層次 '{level_name}' FAISS索引")
    
    def search(self, query_vector: List[float], k: int = 10) -> Tuple[List[int], List[float]]:
        """搜索最相似的向量"""
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        if not query_vector:
            return [], []
        
        # 轉換查詢向量為numpy數組
        query_array = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # 執行搜索 - 使用index.ntotal而不是index_info.total_vectors（更可靠）
        total_vectors = self.index.ntotal if hasattr(self.index, 'ntotal') else (
            self.index_info.total_vectors if self.index_info else 0
        )
        scores, indices = self.index.search(query_array, min(k, total_vectors))
        
        # 轉換結果
        result_indices = [int(idx) for idx in indices[0] if idx >= 0]
        result_scores = [float(score) for score in scores[0][:len(result_indices)]]
        
        return result_indices, result_scores
    
    def search_multi_level(self, level_name: str, query_vector: List[float], k: int = 10) -> Tuple[List[int], List[float]]:
        """搜索指定層次的最相似向量"""
        if level_name not in self.multi_level_indices:
            return [], []
        
        index = self.multi_level_indices[level_name]
        index_info = self.multi_level_index_info[level_name]
        
        if not query_vector:
            return [], []
        
        # 轉換查詢向量
        query_array = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # 執行搜索
        scores, indices = index.search(query_array, min(k, index_info.total_vectors))
        
        # 轉換結果
        result_indices = [int(idx) for idx in indices[0] if idx >= 0]
        result_scores = [float(score) for score in scores[0][:len(result_indices)]]
        
        return result_indices, result_scores
    
    def get_chunk_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """根據索引獲取chunk信息"""
        if index < 0 or index >= len(self.chunk_ids):
            return None
        
        return {
            "chunk_id": self.chunk_ids[index],
            "doc_id": self.chunk_doc_ids[index],
            "content": self.chunks_flat[index],
            "enhanced_metadata": self.enhanced_metadata.get(self.chunk_ids[index], {})
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
            "content": self.multi_level_chunks_flat[level_name][index],
            "enhanced_metadata": self.multi_level_enhanced_metadata.get(level_name, {}).get(chunk_ids[index], {})
        }
    
    def set_enhanced_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> None:
        """設置增強metadata"""
        self.enhanced_metadata[chunk_id] = metadata
    
    def set_multi_level_enhanced_metadata(self, level_name: str, chunk_id: str, metadata: Dict[str, Any]) -> None:
        """設置多層次增強metadata"""
        if level_name not in self.multi_level_enhanced_metadata:
            self.multi_level_enhanced_metadata[level_name] = {}
        self.multi_level_enhanced_metadata[level_name][chunk_id] = metadata
    
    def get_available_levels(self) -> List[str]:
        """獲取可用的層次"""
        return list(self.multi_level_indices.keys())
    
    def has_vectors(self) -> bool:
        """檢查是否有向量"""
        return self.index is not None and self.index_info is not None and self.index_info.total_vectors > 0
    
    def has_multi_level_vectors(self) -> bool:
        """檢查是否有多層次向量"""
        return len(self.multi_level_indices) > 0
    
    def save_data(self) -> None:
        """保存FAISS索引和元數據"""
        try:
            # 保存標準索引
            if self.index is not None:
                faiss.write_index(self.index, os.path.join(self.data_dir, "faiss_index.bin"))
            
            # 保存多層次索引
            for level_name, index in self.multi_level_indices.items():
                faiss.write_index(index, os.path.join(self.data_dir, f"faiss_index_{level_name}.bin"))
            
            # 保存元數據
            metadata = {
                "index_info": self.index_info.__dict__ if self.index_info else None,
                "chunk_ids": self.chunk_ids,
                "chunk_doc_ids": self.chunk_doc_ids,
                "chunks_flat": self.chunks_flat,
                "enhanced_metadata": self.enhanced_metadata,
                "multi_level_index_info": {k: v.__dict__ for k, v in self.multi_level_index_info.items()},
                "multi_level_chunk_ids": self.multi_level_chunk_ids,
                "multi_level_chunk_doc_ids": self.multi_level_chunk_doc_ids,
                "multi_level_chunks_flat": self.multi_level_chunks_flat,
                "multi_level_enhanced_metadata": self.multi_level_enhanced_metadata
            }
            
            with open(os.path.join(self.data_dir, "faiss_metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
            
            print(f"✅ FAISS數據已保存到 {self.data_dir}")
            
        except Exception as e:
            print(f"❌ 保存FAISS數據失敗: {e}")
    
    def load_data(self) -> None:
        """載入FAISS索引和元數據"""
        try:
            # 載入元數據
            metadata_file = os.path.join(self.data_dir, "faiss_metadata.pkl")
            if not os.path.exists(metadata_file):
                print(f"📁 FAISS元數據文件不存在: {metadata_file}")
                return
            
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            
            # 恢復標準索引
            index_file = os.path.join(self.data_dir, "faiss_index.bin")
            if os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
                self.index_info = FAISSIndexInfo(**metadata["index_info"]) if metadata["index_info"] else None
            
            # 恢復多層次索引
            for level_name in metadata.get("multi_level_index_info", {}).keys():
                level_index_file = os.path.join(self.data_dir, f"faiss_index_{level_name}.bin")
                if os.path.exists(level_index_file):
                    self.multi_level_indices[level_name] = faiss.read_index(level_index_file)
            
            # 恢復元數據
            self.chunk_ids = metadata.get("chunk_ids", [])
            self.chunk_doc_ids = metadata.get("chunk_doc_ids", [])
            self.chunks_flat = metadata.get("chunks_flat", [])
            self.enhanced_metadata = metadata.get("enhanced_metadata", {})
            
            # 恢復多層次元數據
            self.multi_level_index_info = {
                k: FAISSIndexInfo(**v) for k, v in metadata.get("multi_level_index_info", {}).items()
            }
            self.multi_level_chunk_ids = metadata.get("multi_level_chunk_ids", {})
            self.multi_level_chunk_doc_ids = metadata.get("multi_level_chunk_doc_ids", {})
            self.multi_level_chunks_flat = metadata.get("multi_level_chunks_flat", {})
            self.multi_level_enhanced_metadata = metadata.get("multi_level_enhanced_metadata", {})
            
            print(f"✅ FAISS數據已從 {self.data_dir} 載入")
            print(f"   📄 標準向量: {self.index_info.total_vectors if self.index_info else 0} 個")
            print(f"   🏗️ 多層次向量: {len(self.multi_level_indices)} 個層次")
            
        except Exception as e:
            print(f"❌ 載入FAISS數據失敗: {e}")
    
    def reset_vectors(self) -> None:
        """重置所有向量"""
        self.index = None
        self.index_info = None
        self.chunk_ids = []
        self.chunk_doc_ids = []
        self.chunks_flat = []
        self.enhanced_metadata = {}
        
        self.multi_level_indices = {}
        self.multi_level_chunk_ids = {}
        self.multi_level_chunk_doc_ids = {}
        self.multi_level_chunks_flat = {}
        self.multi_level_enhanced_metadata = {}
        self.multi_level_index_info = {}
        
        print("🗑️ FAISS向量數據已重置")
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        stats = {
            "faiss_available": FAISS_AVAILABLE,
            "standard_index": {
                "has_index": self.index is not None,
                "total_vectors": self.index_info.total_vectors if self.index_info else 0,
                "dimension": self.index_info.dimension if self.index_info else 0,
                "index_type": self.index_info.index_type if self.index_info else None
            },
            "multi_level_indices": {
                level: {
                    "total_vectors": info.total_vectors,
                    "dimension": info.dimension,
                    "index_type": info.index_type
                }
                for level, info in self.multi_level_index_info.items()
            }
        }
        return stats
