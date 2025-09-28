"""
數據存儲模組
"""

from typing import Dict, Optional, List
from .models import DocRecord, EvaluationTask


class InMemoryStore:
    """內存數據存儲"""
    
    def __init__(self) -> None:
        self.docs: Dict[str, DocRecord] = {}
        self.embeddings = None  # matrix for chunks (numpy array or list)
        self.chunk_doc_ids: List[str] = []
        self.chunks_flat: List[str] = []
        self.evaluation_tasks: Dict[str, EvaluationTask] = {}

    def reset_embeddings(self):
        """清除向量/索引狀態，以便重新計算嵌入"""
        self.embeddings = None
        self.chunk_doc_ids = []
        self.chunks_flat = []

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
