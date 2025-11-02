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
    multi_level_chunks: Optional[Dict[str, List[Dict[str, Any]]]] = None  # 存儲多層次chunks
    chunking_strategy: Optional[str] = None


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
    test_queries: List[str]
    retrieval_results: Dict[str, List[Dict]]  # query -> results
    timestamp: Any  # datetime object


# Pydantic 模型
class ChunkConfig(BaseModel):
    """分塊配置"""
    doc_id: str  # 文檔ID
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
    window_size: int = 500  # 滑動視窗
    boundary_aware: bool = True  # 滑動視窗
    min_chunk_size_sw: int = 100  # 滑動視窗專用
    max_chunk_size_sw: int = 1000  # 滑動視窗專用
    preserve_sentences: bool = True  # 滑動視窗
    secondary_size: int = 400  # 混合分割


class MultiLevelFusionRequest(BaseModel):
    """多層次融合請求"""
    query: str
    k: int = 10
    fusion_strategy: str = "weighted_sum"  # weighted_sum, reciprocal_rank, comb_sum, comb_anz, comb_mnz
    level_weights: Dict[str, float] = None  # 層次權重
    similarity_threshold: float = 0.0
    max_results: int = 10
    normalize_scores: bool = True


class EvaluationRequest(BaseModel):
    """評估請求"""
    doc_id: str
    # Adjusted defaults to focus the sweep and reduce grid size while keeping coverage
    chunk_sizes: List[int] = [300, 500, 800]
    overlap_ratios: List[float] = [0.0, 0.1, 0.2]
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


class ECUAnnotation(BaseModel):
    """E/C/U標註記錄"""
    annotation_id: str
    query: str
    chunk_content: str
    chunk_index: int
    level: str  # 層次名稱
    doc_id: str
    relevance_label: str  # 'E', 'C', 'U'
    annotator: str  # 標註者標識
    timestamp: str
    notes: Optional[str] = None


class GranularityComparisonRequest(BaseModel):
    """粒度對比實驗請求"""
    queries: List[Dict[str, Any]]  # 包含query和gold_standard
    k: int = 10
    combinations_to_test: List[str]  # 要測試的組合keys


class AnnotationBatchRequest(BaseModel):
    """批量標註請求"""
    query: str
    results: List[Dict[str, Any]]
    annotations: Dict[str, str]  # {chunk_index: label}


class MetadataOptions(BaseModel):
    """元數據選項"""
    include_page_numbers: bool = True
    include_section_headers: bool = True
    include_footnotes: bool = False
    include_tables: bool = True
    include_figures: bool = False
    preserve_formatting: bool = True
    extract_metadata: bool = True
    # 簡化版本 - 移除不必要的metadata
    include_id: bool = True
