"""
數據模型定義
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


@dataclass
class DocRecord:
    """文檔記錄"""
    id: str
    filename: str
    text: str
    chunks: List[str]
    chunk_size: int
    overlap: int
    json_data: Optional[Dict[str, Any]] = None
    structured_chunks: Optional[List[Dict[str, Any]]] = None
    generated_questions: Optional[List[str]] = None


@dataclass
class EvaluationTask:
    """評估任務"""
    id: str
    doc_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    configs: List[Dict[str, Any]]
    results: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """評估指標"""
    precision_omega: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    chunk_count: int
    avg_chunk_length: float
    length_variance: float


@dataclass
class EvaluationResult:
    """評估結果"""
    config: Dict[str, Any]
    metrics: EvaluationMetrics


# Pydantic 模型
class ChunkConfig(BaseModel):
    """分塊配置"""
    chunk_size: int = 500
    overlap: int = 50  # 重疊大小（字符數）
    overlap_ratio: float = 0.1
    strategy: str = "fixed_size"  # "fixed_size", "hierarchical", "structured_hierarchical", "semantic", "adaptive"
    chunk_by: str = "article"  # 對於structured_hierarchical策略: "chapter", "section", "article"
    
    # 策略特定參數
    preserve_structure: bool = True  # RCTS層次分割
    level_depth: int = 3  # 層次分割
    similarity_threshold: float = 0.6  # 語義分割
    semantic_threshold: float = 0.7  # LLM語義分割
    switch_threshold: float = 0.5  # 混合分割
    min_chunk_size: int = 200  # 層次分割
    context_window: int = 100  # 語義分割
    step_size: int = 250  # 滑動視窗
    secondary_size: int = 400  # 混合分割


class EvaluationRequest(BaseModel):
    """評估請求"""
    doc_id: str
    # Adjusted defaults to focus the sweep and reduce grid size while keeping coverage
    chunk_sizes: List[int] = [300, 600, 900]
    overlap_ratios: List[float] = [0.0, 0.1]
    question_types: List[str] = [
        "案例應用",
        "情境分析",
        "實務處理",
        "法律後果",
        "合規判斷",
    ]
    # Increase questions to reduce variance when evaluation mode is default-on
    num_questions: int = 20


class GenerateQuestionsRequest(BaseModel):
    """生成問題請求"""
    doc_id: str
    question_types: List[str] = ["案例應用", "情境分析", "實務處理", "法律後果", "合規判斷"]
    num_questions: int = 10
    difficulty_levels: List[str] = ["基礎", "進階", "應用"]


class MetadataOptions(BaseModel):
    """元數據選項"""
    include_page_numbers: bool = True
    include_section_headers: bool = True
    include_footnotes: bool = False
    include_tables: bool = True
    include_figures: bool = False
    preserve_formatting: bool = True
    extract_metadata: bool = True
