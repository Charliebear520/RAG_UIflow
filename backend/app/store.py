"""
數據存儲模組
"""

import os
import json
import pickle
from typing import Dict, Optional, List, Any
from .models import DocRecord, EvaluationTask, ECUAnnotation


class InMemoryStore:
    """內存數據存儲"""
    
    def __init__(self) -> None:
        self.docs: Dict[str, DocRecord] = {}
        self.embeddings = None  # matrix for chunks (numpy array or list)
        self.chunk_doc_ids: List[str] = []
        self.chunks_flat: List[str] = []
        self.evaluation_tasks: Dict[str, EvaluationTask] = {}
        
        # 多層次embedding存儲（當前激活的實驗組）
        self.multi_level_embeddings: Dict[str, Dict[str, Any]] = {}
        self.multi_level_chunk_doc_ids: Dict[str, List[str]] = {}
        self.multi_level_chunks_flat: Dict[str, List[str]] = {}
        self.multi_level_metadata: Dict[str, Dict[str, Any]] = {}  # 存儲模型信息等元數據
        
        # 按實驗組分別存儲embedding（支持多個實驗組同時存在）
        # 結構: {group_id: {multi_level_embeddings: {...}, multi_level_chunk_doc_ids: {...}, ...}}
        self.experimental_group_embeddings: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced metadata存儲（在分塊階段生成）
        self.enhanced_metadata: Dict[str, Dict[str, Any]] = {}  # chunk_id -> enhanced_metadata
        
        # E/C/U標註存儲
        self.annotations: Dict[str, ECUAnnotation] = {}  # annotation_id -> annotation
        
        # 演示資料管理
        self.demo_data_deleted = False  # 標記演示資料是否已被刪除
        
        # 持久化設置
        self.data_dir = "data"
        self.ensure_data_dir()
        
        # 啟動時自動載入數據
        self.load_data()

    def reset_embeddings(self):
        """清除向量/索引狀態，以便重新計算嵌入"""
        self.embeddings = None
        self.chunk_doc_ids = []
        self.chunks_flat = []
        
        # 清除多層次embedding
        self.multi_level_embeddings = {}
        self.multi_level_chunk_doc_ids = {}
        self.multi_level_chunks_flat = {}
        self.multi_level_metadata = {}
        
        # 清除enhanced metadata
        self.enhanced_metadata = {}
        
        # 清除標註數據（可選，根據需要）
        # self.annotations = {}

    def add_doc(self, doc_record: DocRecord):
        """添加文檔記錄"""
        self.docs[doc_record.id] = doc_record
        self.reset_embeddings()

    def get_doc(self, doc_id: str) -> Optional[DocRecord]:
        """獲取文檔記錄"""
        return self.docs.get(doc_id)

    def list_docs(self) -> List[DocRecord]:
        """列出所有文檔記錄"""
        return list(self.docs.values())

    def delete_doc(self, doc_id: str) -> bool:
        """刪除文檔記錄"""
        if doc_id in self.docs:
            del self.docs[doc_id]
            self.reset_embeddings()
            return True
        return False

    def add_evaluation_task(self, task: EvaluationTask):
        """添加評估任務"""
        self.evaluation_tasks[task.id] = task
    
    def ensure_data_dir(self):
        """確保數據目錄存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"✅ 創建數據目錄: {self.data_dir}")
    
    def save_data(self):
        """保存所有數據到文件"""
        try:
            # 準備要保存的數據
            data_to_save = {
                "docs": {},
                "embeddings": self.embeddings,
                "chunk_doc_ids": self.chunk_doc_ids,
                "chunks_flat": self.chunks_flat,
                "multi_level_embeddings": self.multi_level_embeddings,
                "multi_level_chunk_doc_ids": self.multi_level_chunk_doc_ids,
                "multi_level_chunks_flat": self.multi_level_chunks_flat,
                "multi_level_metadata": self.multi_level_metadata,
                "experimental_group_embeddings": self.experimental_group_embeddings,  # 新增：按實驗組存儲
                "enhanced_metadata": self.enhanced_metadata,
                "demo_data_deleted": self.demo_data_deleted
            }
            
            # 轉換DocRecord對象為可序列化的字典
            for doc_id, doc in self.docs.items():
                data_to_save["docs"][doc_id] = {
                    "id": doc.id,
                    "filename": doc.filename,
                    "text": doc.text,
                    "chunks": doc.chunks,
                    "chunk_size": doc.chunk_size,
                    "overlap": doc.overlap,
                    "json_data": doc.json_data,
                    "structured_chunks": doc.structured_chunks,
                    "generated_questions": doc.generated_questions,
                    "multi_level_chunks": doc.multi_level_chunks if hasattr(doc, 'multi_level_chunks') else None,
                    "chunking_strategy": doc.chunking_strategy if hasattr(doc, 'chunking_strategy') else None
                }
            
            # 保存到pickle文件
            with open(os.path.join(self.data_dir, "store_data.pkl"), "wb") as f:
                pickle.dump(data_to_save, f)
            
            print(f"✅ 數據已保存到 {self.data_dir}/store_data.pkl")
            
        except Exception as e:
            print(f"❌ 保存數據失敗: {e}")
    
    def load_data(self):
        """從文件載入數據"""
        try:
            data_file = os.path.join(self.data_dir, "store_data.pkl")
            if not os.path.exists(data_file):
                print(f"📁 數據文件不存在，使用空數據: {data_file}")
                return
            
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            
            # 恢復docs
            self.docs = {}
            for doc_id, doc_data in data.get("docs", {}).items():
                self.docs[doc_id] = DocRecord(
                    id=doc_data["id"],
                    filename=doc_data["filename"],
                    text=doc_data["text"],
                    chunks=doc_data["chunks"],
                    chunk_size=doc_data["chunk_size"],
                    overlap=doc_data["overlap"],
                    json_data=doc_data.get("json_data"),
                    structured_chunks=doc_data.get("structured_chunks"),
                    generated_questions=doc_data.get("generated_questions"),
                    multi_level_chunks=doc_data.get("multi_level_chunks"),
                    chunking_strategy=doc_data.get("chunking_strategy")
                )
            
            # 恢復其他數據
            self.embeddings = data.get("embeddings")
            self.chunk_doc_ids = data.get("chunk_doc_ids", [])
            self.chunks_flat = data.get("chunks_flat", [])
            self.multi_level_embeddings = data.get("multi_level_embeddings", {})
            self.multi_level_chunk_doc_ids = data.get("multi_level_chunk_doc_ids", {})
            self.multi_level_chunks_flat = data.get("multi_level_chunks_flat", {})
            self.multi_level_metadata = data.get("multi_level_metadata", {})
            self.experimental_group_embeddings = data.get("experimental_group_embeddings", {})  # 新增：載入實驗組數據
            self.enhanced_metadata = data.get("enhanced_metadata", {})
            self.demo_data_deleted = data.get("demo_data_deleted", False)
            
            print(f"✅ 數據已從 {data_file} 載入")
            print(f"   📄 文檔數量: {len(self.docs)}")
            print(f"   🔢 標準embedding: {'有' if self.embeddings is not None else '無'}")
            print(f"   🏗️ 多層次embedding: {len(self.multi_level_embeddings)} 個層次")
            
        except Exception as e:
            print(f"❌ 載入數據失敗: {e}")
    
    def clear_all_data(self):
        """清除所有數據並保存"""
        self.docs = {}
        self.embeddings = None
        self.chunk_doc_ids = []
        self.chunks_flat = []
        self.multi_level_embeddings = {}
        self.multi_level_chunk_doc_ids = {}
        self.multi_level_chunks_flat = {}
        self.multi_level_metadata = {}
        self.demo_data_deleted = False
        self.save_data()
        print("🗑️ 所有數據已清除")

    def get_evaluation_task(self, task_id: str) -> Optional[EvaluationTask]:
        """獲取評估任務"""
        return self.evaluation_tasks.get(task_id)

    def update_evaluation_task(self, task_id: str, **kwargs):
        """更新評估任務"""
        if task_id in self.evaluation_tasks:
            task = self.evaluation_tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)

    def list_evaluation_tasks(self) -> List[EvaluationTask]:
        """列出所有評估任務"""
        return list(self.evaluation_tasks.values())
    
    def set_multi_level_embeddings(self, level_name: str, embeddings: Any, chunks: List[str], doc_ids: List[str], metadata: Dict[str, Any] = None):
        """設置多層次embedding"""
        self.multi_level_embeddings[level_name] = embeddings
        self.multi_level_chunks_flat[level_name] = chunks
        self.multi_level_chunk_doc_ids[level_name] = doc_ids
        if metadata:
            self.multi_level_metadata[level_name] = metadata
    
    def get_multi_level_embeddings(self, level_name: str) -> Optional[Dict[str, Any]]:
        """獲取指定層次的embedding"""
        if level_name in self.multi_level_embeddings:
            return {
                'embeddings': self.multi_level_embeddings[level_name],
                'chunks': self.multi_level_chunks_flat.get(level_name, []),
                'doc_ids': self.multi_level_chunk_doc_ids.get(level_name, []),
                'metadata': self.multi_level_metadata.get(level_name, {})
            }
        return None
    
    def has_multi_level_embeddings(self) -> bool:
        """檢查是否有可用的多層次embedding"""
        return len(self.multi_level_embeddings) > 0
    
    def get_available_levels(self) -> List[str]:
        """獲取可用的embedding層次"""
        return list(self.multi_level_embeddings.keys())
    
    def save_experimental_group_embeddings(self, group_id: str):
        """保存當前激活的實驗組embedding到實驗組存儲中"""
        self.experimental_group_embeddings[group_id] = {
            "multi_level_embeddings": self.multi_level_embeddings.copy(),
            "multi_level_chunk_doc_ids": self.multi_level_chunk_doc_ids.copy(),
            "multi_level_chunks_flat": self.multi_level_chunks_flat.copy(),
            "multi_level_metadata": self.multi_level_metadata.copy()
        }
        print(f"✅ 已保存實驗組 {group_id} 的embedding數據")
    
    def load_experimental_group_embeddings(self, group_id: str) -> bool:
        """從實驗組存儲中加載指定實驗組的embedding"""
        if group_id not in self.experimental_group_embeddings:
            return False
        
        group_data = self.experimental_group_embeddings[group_id]
        self.multi_level_embeddings = group_data.get("multi_level_embeddings", {}).copy()
        self.multi_level_chunk_doc_ids = group_data.get("multi_level_chunk_doc_ids", {}).copy()
        self.multi_level_chunks_flat = group_data.get("multi_level_chunks_flat", {}).copy()
        self.multi_level_metadata = group_data.get("multi_level_metadata", {}).copy()
        print(f"✅ 已加載實驗組 {group_id} 的embedding數據")
        return True
    
    def list_experimental_groups(self) -> List[str]:
        """列出所有已保存的實驗組ID"""
        return list(self.experimental_group_embeddings.keys())
    
    # E/C/U標註管理方法
    def save_annotation(self, annotation: ECUAnnotation):
        """保存標註"""
        self.annotations[annotation.annotation_id] = annotation
    
    def get_annotations_for_query(self, query: str) -> List[ECUAnnotation]:
        """獲取特定查詢的所有標註"""
        return [a for a in self.annotations.values() if a.query == query]
    
    def get_all_annotations(self) -> List[ECUAnnotation]:
        """獲取所有標註"""
        return list(self.annotations.values())
    
    def delete_annotations_for_query(self, query: str):
        """刪除特定查詢的所有標註"""
        to_delete = [annotation_id for annotation_id, annotation in self.annotations.items() 
                    if annotation.query == query]
        for annotation_id in to_delete:
            del self.annotations[annotation_id]


# store實例在main.py中創建，避免重複定義
