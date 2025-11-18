from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass
import re
from typing import List, Optional, Dict, Any, Tuple, Set
import json
from datetime import datetime
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .models import DocRecord, ChunkConfig, MetadataOptions, MultiLevelFusionRequest, ECUAnnotation, GranularityComparisonRequest, AnnotationBatchRequest
from .hybrid_search import hybrid_rank, HybridConfig
from .store import InMemoryStore
from .faiss_store import FAISSVectorStore
from .bm25_index import BM25KeywordIndex
from .metadata_enhancer import MetadataEnhancer
from .enhanced_hybrid_rag import EnhancedHybridRAG, EnhancedHybridConfig
from .query_classifier import query_classifier, get_query_analysis
from .result_fusion import MultiLevelResultFusion, FusionConfig, fuse_multi_level_results
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None  # type: ignore
    BM25_AVAILABLE = False
from dotenv import load_dotenv
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PdfReader = None
    PYPDF2_AVAILABLE = False
import pdfplumber
try:
    import jieba  # type: ignore
    import jieba.analyse  # type: ignore
    jieba.initialize()
except ImportError:
    jieba = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
    GEMINI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore
    GEMINI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Embedding 維度配置
# Gemini: 支援 128-3072，建議 768/1536/3072
# BGE-M3: 固定 1024 或 3072（取決於配置）
EMBEDDING_DIMENSION = 3072  # 🎯 統一配置：改這裡就能改全部

load_dotenv()


def get_env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GEMINI_EMBEDDING = True  # ✅ 使用 Gemini Embedding（已優化速率限制）
USE_GEMINI_COMPLETION = True  # LLM推理使用Gemini
USE_BGE_M3_EMBEDDING = False  # ❌ BGE-M3在Mac上太慢，已禁用
DEFAULT_THINKING_MODEL = os.getenv("GOOGLE_THINKING_MODEL", "gemini-2.5-flash")
DEFAULT_SUMMARY_MODEL = os.getenv("GOOGLE_SUMMARY_MODEL", "gemini-2.5-flash")
CHAPTER_SUMMARY_PATH = os.path.join("data", "chapter_summaries.json")
chapter_summary_cache: Dict[str, Any] = {}
chapter_summary_loaded = False
_multi_level_chunker = None

# 調試信息
print(f"🔧 Embedding 配置:")
print(f"   USE_GEMINI_EMBEDDING: {USE_GEMINI_EMBEDDING}")
print(f"   GOOGLE_API_KEY: {'已設置' if GOOGLE_API_KEY else '未設置'}")
print(f"   GEMINI_API_KEY: {'已設置' if os.getenv('GEMINI_API_KEY') else '未設置'}")
print(f"   USE_BGE_M3_EMBEDDING: {USE_BGE_M3_EMBEDDING}")
print(f"   GOOGLE_EMBEDDING_MODEL: {os.getenv('GOOGLE_EMBEDDING_MODEL', 'gemini-embedding-001')}")
print(f"   USE_GEMINI_COMPLETION: {USE_GEMINI_COMPLETION}")

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None


# DocRecord 已從 models 導入








from .store import InMemoryStore
store = InMemoryStore()

# 新增：FAISS向量存儲
faiss_store = FAISSVectorStore()

# 新增：BM25關鍵字索引
bm25_index = BM25KeywordIndex()

# 新增：Metadata增強器
metadata_enhancer = MetadataEnhancer()

# 新增：增強版HybridRAG
enhanced_hybrid_rag = EnhancedHybridRAG(faiss_store, bm25_index, metadata_enhancer)

# 初始化時載入已保存的數據
try:
    faiss_store.load_data()
    bm25_index.load_data()
    print("✅ 已載入FAISS和BM25數據")
except Exception as e:
    print(f"⚠️ 載入FAISS和BM25數據失敗: {e}")





app = FastAPI(title="RAG Visualizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # During dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加驗證錯誤處理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """處理請求驗證錯誤，提供更清晰的錯誤信息"""
    errors = exc.errors()
    error_details = []
    for error in errors:
        error_details.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    return JSONResponse(
        status_code=400,
        content={
            "error": "請求驗證失敗",
            "details": error_details,
            "raw_errors": errors
        }
    )

# 暫時停用routes.py的包含，避免循環導入問題
from .routes import router
app.include_router(router, prefix="/api")


class ChunkRequest(BaseModel):
    doc_id: str
    chunk_size: int = 500
    overlap: int = 50
    strategy: str = "fixed_size"
    use_json_structure: bool = False
    
    # 策略特定參數
    hierarchical_params: Optional[Dict[str, Any]] = None
    adaptive_params: Optional[Dict[str, Any]] = None
    hybrid_params: Optional[Dict[str, Any]] = None
    semantic_params: Optional[Dict[str, Any]] = None
    rcts_hierarchical_params: Optional[Dict[str, Any]] = None
    structured_hierarchical_params: Optional[Dict[str, Any]] = None


class EmbedRequest(BaseModel):
    doc_ids: Optional[List[str]] = None  # if None, embed all
    enable_metadata_enhancement: bool = True  # 是否啟用metadata增強


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5


class GenerateRequest(BaseModel):
    query: str
    top_k: int = 5


# MetadataOptions 已移至 models.py


# 評測相關的數據模型
# ChunkConfig 已移至 models.py


class EvaluationMetrics(BaseModel):
    precision_omega: float  # PrecisionΩ - 最大準確率
    precision_at_k: Dict[int, float]  # k -> precision score
    recall_at_k: Dict[int, float]  # k -> recall score
    chunk_count: int
    avg_chunk_length: float
    length_variance: float


class EvaluationResult(BaseModel):
    config: Dict[str, Any]  # 改為字典以支援動態參數
    metrics: EvaluationMetrics
    test_queries: List[str]
    retrieval_results: Dict[str, List[Dict]]  # query -> results
    timestamp: datetime


@dataclass
class EvaluationTask:
    id: str
    doc_id: str
    configs: List[ChunkConfig]
    test_queries: List[str]
    k_values: List[int]
    status: str  # "pending", "running", "completed", "failed"
    results: List[EvaluationResult]
    created_at: datetime
    progress: float = 0.0  # 新增：進度 0.0 to 1.0
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    strategy: str = "fixed_size"  # 新增：分割策略


class EvaluationStore:
    def __init__(self) -> None:
        self.tasks: Dict[str, EvaluationTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    def create_task(self, doc_id: str, configs: List[ChunkConfig], 
                   test_queries: List[str], k_values: List[int], 
                   strategy: str = "fixed_size") -> str:
        task_id = str(uuid.uuid4())
        task = EvaluationTask(
            id=task_id,
            doc_id=doc_id,
            configs=configs,
            test_queries=test_queries,
            k_values=k_values,
            strategy=strategy,
            status="pending",
            results=[],
            created_at=datetime.now()
        )
        self.tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, status: str, 
                          results: Optional[List[EvaluationResult]] = None,
                          error_message: Optional[str] = None,
                          progress: Optional[float] = None):
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if results is not None:
                self.tasks[task_id].results = results
            if error_message is not None:
                self.tasks[task_id].error_message = error_message
            if progress is not None:
                self.tasks[task_id].progress = progress
            if status == "completed":
                self.tasks[task_id].completed_at = datetime.now()
                self.tasks[task_id].progress = 1.0


# 創建評估存儲實例
eval_store = EvaluationStore()


class FixedSizeEvaluationRequest(BaseModel):
    doc_id: str
    chunk_sizes: List[int] = [300, 500, 800]
    overlap_ratios: List[float] = [0.0, 0.1, 0.2]
    strategy: str = "fixed_size"  # 新增：分割策略
    test_queries: List[str] = [
        "著作權的定義是什麼？",
        "什麼情況下可以合理使用他人作品？",
        "侵犯著作權的法律後果是什麼？",
        "著作權的保護期限是多久？",
        "如何申請著作權登記？"
    ]
    k_values: List[int] = [1, 3, 5, 10]
    
    # 策略特定參數選項 - 預設包含所有排列組合
    chunk_by_options: List[str] = ["article", "item", "section", "chapter"]  # 結構化層次分割選項
    preserve_structure_options: List[bool] = [True, False]  # RCTS層次分割選項
    level_depth_options: List[int] = [2, 3, 4]  # 層次分割選項
    similarity_threshold_options: List[float] = [0.5, 0.6, 0.7]  # 語義分割選項
    semantic_threshold_options: List[float] = [0.6, 0.7, 0.8]  # LLM語義分割選項
    switch_threshold_options: List[float] = [0.3, 0.5, 0.7]  # 混合分割選項
    min_chunk_size_options: List[int] = [100, 200, 300]  # 層次分割選項
    context_window_options: List[int] = [50, 100, 150]  # 語義分割選項
    step_size_options: List[int] = [200, 250, 300]  # 滑動視窗選項
    window_size_options: List[int] = [400, 500, 600, 800]  # 滑動視窗選項
    boundary_aware_options: List[bool] = [True, False]  # 滑動視窗選項
    preserve_sentences_options: List[bool] = [True, False]  # 滑動視窗選項
    min_chunk_size_options_sw: List[int] = [50, 100, 150]  # 滑動視窗專用選項
    max_chunk_size_options_sw: List[int] = [800, 1000, 1200]  # 滑動視窗專用選項
    secondary_size_options: List[int] = [300, 400, 500]  # 混合分割選項  # 用於計算recall@K


class GenerateQuestionsRequest(BaseModel):
    doc_id: str
    num_questions: int = 10
    question_types: List[str] = ["案例應用", "情境分析", "實務處理", "法律後果", "合規判斷"]  # 問題類型
    difficulty_levels: List[str] = ["基礎", "進階", "應用"]  # 難度等級


class GeneratedQuestion(BaseModel):
    question: str
    references: List[str]  # 相關法規條文
    question_type: str
    difficulty: str
    keywords: List[str]
    estimated_tokens: int


class QuestionGenerationResult(BaseModel):
    doc_id: str
    total_questions: int
    questions: List[GeneratedQuestion]
    generation_time: float
    timestamp: datetime


def generate_unique_id(law_name: str, chapter: str, section: str, article: str, item: Optional[str] = None) -> str:
    """生成id"""
    # 清理法規名稱
    law_clean = re.sub(r'[^\w]', '', law_name.lower())
    law_clean = re.sub(r'法規名稱|法|條例', '', law_clean)
    
    # 提取章節
    chapter_num = re.search(r'第([一二三四五六七八九十百千0-9]+)章', chapter)
    chapter_num = chapter_num.group(1) if chapter_num else "0"
    
    # 提取節
    section_num = re.search(r'第([一二三四五六七八九十百千0-9]+)節', section)
    section_num = section_num.group(1) if section_num else "0"
    
    # 提取條文
    article_num = re.search(r'第([一二三四五六七八九十百千0-9]+)條', article)
    article_num = article_num.group(1) if article_num else "0"
    
    # 組合ID
    parts = [law_clean, f"ch{chapter_num}", f"sec{section_num}", f"art{article_num}"]
    if item:
        parts.append(f"item{item}")
    
    return "-".join(parts)


def extract_keywords_with_gemini(text: str, top_k: int = 5) -> List[str]:
    """使用Gemini模型提取關鍵詞"""
    if not GEMINI_AVAILABLE:
        return extract_keywords_fallback(text, top_k)
    
    try:
        # 優先使用 GOOGLE_API_KEY，如果沒有則使用 GEMINI_API_KEY
        api_key = GOOGLE_API_KEY or os.getenv('GEMINI_API_KEY')
        if not api_key:
            return extract_keywords_fallback(text, top_k)
        
        # Configure API key using getattr to avoid static export issues
        cfg = getattr(genai, "configure", None)
        if callable(cfg):
            cfg(api_key=api_key)  # type: ignore[misc]
        ModelCls = getattr(genai, "GenerativeModel", None)
        if ModelCls is None:
            return extract_keywords_fallback(text, top_k)
        model = ModelCls('gemini-2.0-flash')
        
        prompt = f"""
        請從以下法律條文內容中提取{top_k}個最重要的關鍵詞。
        關鍵詞應該是法律術語、重要概念或核心內容。
        請只返回關鍵詞，用逗號分隔，不要其他解釋。
        
        條文內容：
        {text}
        """
        
        response = model.generate_content(prompt)
        keywords_text = response.text.strip()
        
        # 解析關鍵詞
        keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
        return keywords[:top_k]
        
    except Exception as e:
        print(f"Gemini關鍵詞提取失敗: {e}")
        return extract_keywords_fallback(text, top_k)


def extract_keywords_fallback(text: str, top_k: int = 5) -> List[str]:
    """備用關鍵詞提取方法"""
    if jieba is None:
        # 如果jieba不可用，使用簡單的正則表達式
        words = re.findall(r'[\u4e00-\u9fff]+', text)
        return list(set(words))[:top_k]
    
    try:
        # 使用jieba提取關鍵詞；部分版本型別為 List[Tuple[str, float]] | List[str]
        from typing import cast, List as _List
        kws = jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)  # type: ignore[call-arg]
        keywords = cast(_List[str], list(kws))
        return keywords[:top_k] if keywords else []
    except:
        # 如果jieba失敗，使用簡單的正則表達式
        words = re.findall(r'[\u4e00-\u9fff]+', text)
        return list(set(words))[:top_k]


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """提取關鍵詞 - 優先使用Gemini，備用jieba"""
    return extract_keywords_with_gemini(text, top_k)


def extract_cross_references(text: str) -> List[str]:
    """提取交叉引用"""
    references = []
    
    # 匹配「第X條」
    article_refs = re.findall(r'第[一二三四五六七八九十百千0-9]+條', text)
    references.extend(article_refs)
    
    # 匹配「第X項」
    item_refs = re.findall(r'第[一二三四五六七八九十百千0-9]+項', text)
    references.extend(item_refs)
    
    # 匹配「第X款」
    clause_refs = re.findall(r'第[一二三四五六七八九十百千0-9]+款', text)
    references.extend(clause_refs)
    
    # 匹配「前項」「前條」「準用之」等
    if re.search(r'前項|前條|準用之|依.*規定|比照.*辦理|適用.*規定', text):
        references.append("internal_reference")
    
    # 匹配「本法」「本條例」等自引用
    if re.search(r'本法|本條例|本規則|本辦法', text):
        references.append("self_reference")
    
    # 匹配「其他法律」「相關法規」等外部引用
    if re.search(r'其他法律|相關法規|其他法規|其他條例', text):
        references.append("external_reference")
    
    return list(set(references))


def preprocess_text(text: str) -> List[str]:
    """
    文本預處理：分詞、去停用詞、清理
    """
    if not text:
        return []
    
    # 使用jieba分詞
    if jieba:
        words = jieba.lcut(text)
    else:
        # 簡單的字符級分詞作為備選
        words = list(text)
    
    # 中文停用詞列表（法律文檔專用，較少過濾）
    stop_words = {
        '的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '上', '也', '很', '到', '要', '去', '會', '著', '沒有', '看', '好', '自己', '這', '那', '它', '他', '她', '我們', '你們', '他們', '她們', '它們', '什麼', '怎麼', '為什麼', '哪裡', '什麼時候', '多少', '幾個', '一些', '所有', '每個', '任何', '如果', '因為', '所以', '但是', '然後', '或者', '而且', '雖然', '不過', '只是', '就是', '還是', '已經', '正在', '將要', '可以', '應該', '必須', '需要', '想要', '希望', '喜歡', '不喜歡', '知道', '不知道', '明白', '不明白', '記得', '忘記', '開始', '結束', '繼續', '停止', '完成', '做', '做過', '正在做', '將要做', '被', '把', '給', '對', '向', '從', '到', '在', '於', '為', '以', '用', '通過', '根據', '按照', '依照', '關於', '對於', '至於', '除了', '包括', '以及', '與', '或', '但', '然而', '因此', '於是', '然後', '接著', '最後', '首先', '其次', '再次', '另外', '此外', '並且', '同時', '一起', '分別', '各自', '共同', '單獨', '獨立', '相關', '無關', '重要', '不重要', '主要', '次要', '基本', '根本', '核心', '關鍵', '必要', '不必要', '可能', '不可能', '一定', '不一定', '肯定', '不肯定', '確定', '不確定', '清楚', '不清楚', '明確', '不明確', '具體', '不具體', '詳細', '不詳細', '簡單', '複雜', '容易', '困難', '方便', '不方便', '快速', '慢速', '高效', '低效', '有效', '無效', '成功', '失敗', '正確', '錯誤', '對', '錯', '好', '壞', '優', '劣', '高', '低', '大', '小', '多', '少', '長', '短', '寬', '窄', '厚', '薄', '深', '淺', '新', '舊', '年輕', '老', '早', '晚', '快', '慢', '熱', '冷', '暖', '涼', '乾', '濕', '亮', '暗', '明', '清', '濁', '靜', '動', '安', '危', '平', '陡', '直', '彎', '圓', '方', '尖', '鈍', '軟', '硬', '輕', '重', '強', '弱', '緊', '鬆', '密', '疏', '滿', '空', '實', '虛', '真', '假', '正', '負', '加', '減', '乘', '除', '等於', '不等於', '大於', '小於', '大於等於', '小於等於', '和', '差', '積', '商', '餘', '倍', '分', '比', '率', '比例', '百分', '千分', '萬分', '億分', '兆分', '京分', '垓分', '秭分', '穰分', '溝分', '澗分', '正分', '載分', '極分', '恆河沙分', '阿僧祇分', '那由他分', '不可思議分', '無量大數分'
    }
    
    # 過濾停用詞和短詞
    filtered_words = []
    for word in words:
        word = word.strip()
        if len(word) > 1 and word not in stop_words and not word.isdigit():
            filtered_words.append(word)
    
    return filtered_words


def calculate_tfidf_importance(texts: List[str], target_text: str) -> float:
    """
    使用TF-IDF計算文本重要性
    """
    if not texts or not target_text:
        return 1.0
    
    try:
        # 預處理所有文本
        processed_texts = [' '.join(preprocess_text(text)) for text in texts]
        processed_target = ' '.join(preprocess_text(target_text))
        
        if not processed_target:
            return 1.0
        
        # 計算TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # 擬合所有文本
        all_texts = processed_texts + [processed_target]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # 獲取目標文本的TF-IDF向量（避免稀疏矩陣的切片警告）
        target_vector = tfidf_matrix.getrow(tfidf_matrix.shape[0] - 1)
        
        # 計算與其他文本的平均相似度
        similarities = cosine_similarity(target_vector, tfidf_matrix[:-1])
        avg_similarity = similarities.mean()
        
        # 計算TF-IDF分數（詞頻-逆文檔頻率）
        tfidf_scores = target_vector.toarray()[0]
        tfidf_sum = tfidf_scores.sum()
        
        # 綜合評分：TF-IDF分數 + 相似度
        importance = (tfidf_sum * 0.7 + avg_similarity * 0.3) * 10
        
        # 標準化到1-5範圍
        importance = max(0.1, min(5.0, importance))
        
        return round(importance, 2)
        
    except Exception as e:
        print(f"TF-IDF計算錯誤: {e}")
        return 1.0


def calculate_bm25_importance(texts: List[str], target_text: str) -> float:
    """
    使用BM25計算文本重要性
    """
    if not texts or not target_text or not BM25_AVAILABLE:
        return 1.0
    
    try:
        # 預處理所有文本
        processed_texts = [preprocess_text(text) for text in texts]
        processed_target = preprocess_text(target_text)
        
        if not processed_target:
            return 1.0
        
        # 初始化BM25
        bm25 = BM25Okapi(processed_texts)
        
        # 計算BM25分數
        scores = bm25.get_scores(processed_target)
        
        if len(scores) == 0:
            return 1.0
        
        # 計算平均分數
        avg_score = scores.mean()
        
        # 計算最高分數
        max_score = scores.max()
        
        # 綜合評分：平均分數 + 最高分數
        importance = (avg_score * 0.6 + max_score * 0.4) * 2
        
        # 標準化到1-5範圍
        importance = max(0.1, min(5.0, importance))
        
        return round(importance, 2)
        
    except Exception as e:
        print(f"BM25計算錯誤: {e}")
        return 1.0


def calculate_importance(chapter: str, section: str, article: str, content: str = "", all_articles: List[Dict] = None) -> float:
    """
    計算重要性權重 - 使用TF-IDF和BM25動態計算
    
    參數:
    - chapter: 章節名稱
    - section: 節名稱  
    - article: 條文名稱
    - content: 條文內容
    - all_articles: 所有條文列表，用於計算相對重要性
    """
    # 基礎權重
    base_weight = 1.0
    
    # 如果沒有內容或所有條文，使用靜態權重
    if not content or not all_articles:
        return calculate_static_importance(chapter, section, article)
    
    try:
        # 準備所有條文的文本
        all_texts = []
        for art in all_articles:
            text = f"{art.get('article', '')} {art.get('content', '')}"
            if text.strip():
                all_texts.append(text)
        
        if len(all_texts) < 2:
            return calculate_static_importance(chapter, section, article)
        
        # 目標文本
        target_text = f"{article} {content}"
        
        # 計算TF-IDF重要性
        tfidf_importance = calculate_tfidf_importance(all_texts, target_text)
        
        # 計算BM25重要性
        bm25_importance = calculate_bm25_importance(all_texts, target_text)
        
        # 綜合評分：TF-IDF 60% + BM25 40%
        dynamic_weight = tfidf_importance * 0.6 + bm25_importance * 0.4
        
        # 結合靜態權重（30%）和動態權重（70%）
        final_weight = base_weight * 0.3 + dynamic_weight * 0.7
        
        return round(final_weight, 2)
        
    except Exception as e:
        print(f"動態重要性計算錯誤: {e}")
        return calculate_static_importance(chapter, section, article)


def calculate_static_importance(chapter: str, section: str, article: str) -> float:
    """
    靜態重要性權重計算（備用方法）
    """
    weight = 1.0
    
    # 總則章節權重更高 (基礎性條文)
    if "總則" in chapter or "第一章" in chapter or "通則" in chapter:
        weight *= 1.5
    
    # 定義性條文權重更高 (核心概念)
    if "定義" in article or "用詞" in article or "釋義" in article:
        weight *= 1.3
    
    # 罰則章節權重較高 (法律後果)
    if "罰則" in chapter or "罰" in chapter or "處罰" in chapter:
        weight *= 1.2
    
    # 施行細則權重較低 (程序性條文)
    if "施行" in chapter or "程序" in chapter or "流程" in chapter:
        weight *= 0.8
    
    # 附則權重最低 (補充性條文)
    if "附則" in chapter or "附" in chapter:
        weight *= 0.7
    
    # 通則節權重較高
    if "通則" in section:
        weight *= 1.2
    
    return round(weight, 2)






def get_text_position_in_document(full_text: str, target_text: str) -> Dict[str, Any]:
    """獲取文本在文檔中的位置信息"""
    if not target_text.strip():
        return {"start": 0, "end": 0, "found": False}
    
    # 清理文本內容
    clean_target = re.sub(r'\s+', ' ', target_text.strip())
    clean_full = re.sub(r'\s+', ' ', full_text)
    
    start_idx = clean_full.find(clean_target)
    if start_idx != -1:
        return {
            "start": start_idx,
            "end": start_idx + len(clean_target),
            "found": True,
            "confidence": 1.0
        }
    
    # 嘗試部分匹配
    if len(clean_target) > 10:
        partial = clean_target[:15]
        start_idx = clean_full.find(partial)
        if start_idx != -1:
            return {
                "start": start_idx,
                "end": start_idx + len(clean_target),
                "found": True,
                "confidence": 0.7,
                "note": "partial_match"
            }
    
    return {"start": 0, "end": 0, "found": False, "confidence": 0.0}






@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    doc_id = str(uuid.uuid4())
    
    # 檢查文件類型並相應處理
    if file.filename and file.filename.lower().endswith('.pdf'):
        # 處理PDF文件
        try:
            import io
            if pdfplumber:
                # 使用pdfplumber解析PDF
                pdf_file = io.BytesIO(content)
                text = ""
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                # 備用方案：使用PyPDF2
                pdf_file = io.BytesIO(content)
                if PYPDF2_AVAILABLE:
                    pdf_reader = PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                else:
                    # 如果沒有PDF解析庫，返回錯誤
                    return JSONResponse(
                        status_code=400, 
                        content={"error": "PDF parsing libraries not available. Please install pdfplumber or PyPDF2."}
                    )
        except Exception as e:
            return JSONResponse(
                status_code=400, 
                content={"error": f"Failed to parse PDF: {str(e)}"}
            )
    else:
        # 處理文本文件
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = str(content)
    
    # 清理文本
    text = text.strip()
    if not text:
        return JSONResponse(
            status_code=400, 
            content={"error": "No text content found in the file"}
        )
    
    doc_record = DocRecord(
        id=doc_id,
        filename=file.filename,
        text=text,
        json_data=None,  # 初始為None，後續通過/update-json端點更新
        chunks=[],
        chunk_size=0,
        overlap=0,
    )
    store.add_doc(doc_record)
    
    # 自動保存數據
    store.save_data()
    
    return {"doc_id": doc_id, "filename": file.filename, "num_chars": len(text)}


@app.post("/api/update-json")
async def update_json(request: dict):
    """更新文檔的JSON結構化數據"""
    doc_id = request.get("doc_id")
    json_data = request.get("json_data")
    
    if not doc_id or not json_data:
        return JSONResponse(
            status_code=400,
            content={"error": "doc_id and json_data are required"}
        )
    
    if doc_id not in store.docs:
        return JSONResponse(
            status_code=404,
            content={"error": "Document not found"}
        )
    
    # 更新文檔的JSON數據
    store.docs[doc_id].json_data = json_data
    
    # 重置相關狀態，因為JSON數據改變可能影響chunking
    store.docs[doc_id].chunks = []
    store.docs[doc_id].chunk_size = 0
    store.docs[doc_id].overlap = 0
    store.reset_embeddings()
    
    return {"success": True, "message": "JSON data updated successfully"}


def sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """固定大小滑動窗口分割"""
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def hierarchical_chunks(text: str, max_chunk_size: int, min_chunk_size: int, overlap: int, level_depth: int) -> List[str]:
    """層次化分割策略"""
    if max_chunk_size <= 0:
        return [text]
    
    # 首先按段落分割
    paragraphs = text.split('\n\n')
    chunks = []
    
    for para in paragraphs:
        if len(para) <= max_chunk_size:
            chunks.append(para)
        else:
            # 如果段落太長，按句子分割
            sentences = para.split('。')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) <= max_chunk_size:
                    current_chunk += sentence + "。"
                else:
                    if current_chunk and len(current_chunk) >= min_chunk_size:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + "。"
            
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
    
    # 應用重疊
    if overlap > 0:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            overlapped_chunks.append(chunk)
            if i < len(chunks) - 1 and len(chunk) > overlap:
                # 添加重疊部分
                overlap_text = chunk[-overlap:]
                next_chunk = chunks[i + 1]
                if len(next_chunk) > overlap:
                    overlapped_chunks.append(overlap_text + next_chunk[overlap:])
        return overlapped_chunks
    
    return chunks


def adaptive_chunks(text: str, target_size: int, tolerance: int, overlap: int, semantic_threshold: float) -> List[str]:
    """自適應分割策略"""
    if target_size <= 0:
        return [text]
    
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        # 嘗試找到最佳分割點
        end = min(n, start + target_size)
        
        # 如果接近目標大小，尋找語義邊界
        if end - start >= target_size - tolerance:
            # 尋找句號、段落等語義邊界
            for i in range(end, max(start + target_size - tolerance, start), -1):
                if i < n and text[i] in ['。', '\n', '！', '？']:
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        if end == n:
            break
        
        # 計算下一個chunk的起始位置（考慮重疊）
        start = max(start + 1, end - overlap)
    
    return chunks


def hybrid_chunks(text: str, primary_size: int, secondary_size: int, overlap: int, switch_threshold: float) -> List[str]:
    """混合分割策略"""
    if primary_size <= 0:
        return [text]
    
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        # 決定使用主要大小還是次要大小
        remaining_text = text[start:]
        avg_sentence_length = len(remaining_text) / max(1, remaining_text.count('。'))
        
        if avg_sentence_length > primary_size * switch_threshold:
            chunk_size = secondary_size
        else:
            chunk_size = primary_size
        
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        if end == n:
            break
        
        start = max(start + 1, end - overlap)
    
    return chunks


def semantic_chunks(text: str, target_size: int, similarity_threshold: float, overlap: int, context_window: int) -> List[str]:
    """語義分割策略"""
    if target_size <= 0:
        return [text]
    
    # 簡化實現：按句子分割，然後合併相似的句子
    sentences = text.split('。')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence = sentence.strip() + "。"
        
        # 如果當前chunk加上新句子不超過目標大小
        if len(current_chunk + sentence) <= target_size:
            current_chunk += sentence
        else:
            # 保存當前chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 開始新chunk
            current_chunk = sentence
    
    # 添加最後一個chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def json_structured_chunks(json_data: Dict[str, Any], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """
    基於JSON結構的智能分割
    保留法律文檔的結構化信息
    支持單一法律文檔和多法律文檔格式
    """
    if not json_data or chunk_size <= 0:
        return []
    
    chunks = []
    
    # 檢查是否為多法律文檔格式
    if "laws" in json_data:
        # 多法律文檔格式
        laws = json_data.get("laws", [])
        for law in laws:
            law_chunks = process_single_law(law, chunk_size, overlap)
            chunks.extend(law_chunks)
    else:
        # 單一法律文檔格式
        law_chunks = process_single_law(json_data, chunk_size, overlap)
        chunks.extend(law_chunks)
    
    return chunks


def process_single_law(law_data: Dict[str, Any], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """
    處理單一法律文檔
    """
    chunks = []
    law_name = law_data.get("law_name", "未命名法規")
    
    def create_chunk(content: str, metadata: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
        """創建包含metadata的chunk"""
        return {
            "chunk_id": chunk_id,
            "content": content,
            "metadata": {
                "id": metadata.get("id", ""),
                "spans": metadata.get("spans", {}),
                "page_range": metadata.get("page_range", {})
            }
        }
    
    def process_article(article: Dict[str, Any], chapter: str, section: str) -> List[Dict[str, Any]]:
        """處理單個條文"""
        article_chunks = []
        article_title = article.get("article", "")
        article_content = article.get("content", "")
        items = article.get("items", [])
        
        # 處理條文主體
        if article_content:
            # 如果條文內容較短，直接作為一個chunk
            if len(article_content) <= chunk_size:
                metadata = {
                    "id": article.get("metadata", {}).get("id", ""),
                    "spans": article.get("metadata", {}).get("spans", {}),
                    "page_range": article.get("metadata", {}).get("page_range", {})
                }
                chunk_id = f"{article_title}_main"
                article_chunks.append(create_chunk(article_content, metadata, chunk_id))
            else:
                # 條文內容較長，需要分割
                text_chunks = sliding_window_chunks(article_content, chunk_size, overlap)
                for i, chunk_text in enumerate(text_chunks):
                    metadata = {
                        "id": article.get("metadata", {}).get("id", ""),
                        "spans": article.get("metadata", {}).get("spans", {}),
                        "page_range": article.get("metadata", {}).get("page_range", {})
                    }
                    chunk_id = f"{article_title}_part_{i+1}"
                    article_chunks.append(create_chunk(chunk_text, metadata, chunk_id))
        
        # 處理條文項目 - 支援新結構 (paragraphs) 和舊結構 (items)
        paragraphs = article.get("paragraphs", [])
        items = article.get("items", [])
        
        # 使用 paragraphs 如果存在，否則使用 items
        items_to_process = paragraphs if paragraphs else items
        
        for item in items_to_process:
            # 支援新結構的鍵名
            item_title = item.get("paragraph", item.get("item", ""))
            item_content = item.get("content", "")
            
            # 處理項目主體
            if item_content:
                if len(item_content) <= chunk_size:
                    metadata = {
                        "id": item.get("metadata", {}).get("id", ""),
                        "spans": item.get("metadata", {}).get("spans", {}),
                        "page_range": item.get("metadata", {}).get("page_range", {})
                    }
                    chunk_id = f"{article_title}_{item_title}_main"
                    article_chunks.append(create_chunk(item_content, metadata, chunk_id))
                else:
                    # 項目內容較長，需要分割
                    text_chunks = sliding_window_chunks(item_content, chunk_size, overlap)
                    for i, chunk_text in enumerate(text_chunks):
                        metadata = {
                            "id": item.get("metadata", {}).get("id", ""),
                            "spans": item.get("metadata", {}).get("spans", {}),
                            "page_range": item.get("metadata", {}).get("page_range", {})
                        }
                        chunk_id = f"{article_title}_{item_title}_part_{i+1}"
                        article_chunks.append(create_chunk(chunk_text, metadata, chunk_id))
            
            # 處理子項目 - 支援新結構 (subparagraphs) 和舊結構 (sub_items)
            subparagraphs = item.get("subparagraphs", [])
            sub_items = item.get("sub_items", [])
            
            # 使用 subparagraphs 如果存在，否則使用 sub_items
            sub_items_to_process = subparagraphs if subparagraphs else sub_items
            
            for sub_item in sub_items_to_process:
                # 支援新結構的鍵名
                sub_item_title = sub_item.get("subparagraph", sub_item.get("sub_item", ""))
                sub_item_content = sub_item.get("content", "")
                
                if sub_item_content:
                    if len(sub_item_content) <= chunk_size:
                        metadata = {
                            "id": sub_item.get("metadata", {}).get("id", ""),
                            "spans": sub_item.get("metadata", {}).get("spans", {}),
                            "page_range": sub_item.get("metadata", {}).get("page_range", {})
                        }
                        chunk_id = f"{article_title}_{item_title}_{sub_item_title}"
                        article_chunks.append(create_chunk(sub_item_content, metadata, chunk_id))
                    else:
                        # 子項目內容較長，需要分割
                        text_chunks = sliding_window_chunks(sub_item_content, chunk_size, overlap)
                        for i, chunk_text in enumerate(text_chunks):
                            metadata = {
                                "id": sub_item.get("metadata", {}).get("id", ""),
                                "spans": sub_item.get("metadata", {}).get("spans", {}),
                                "page_range": sub_item.get("metadata", {}).get("page_range", {})
                            }
                            chunk_id = f"{article_title}_{item_title}_{sub_item_title}_part_{i+1}"
                            article_chunks.append(create_chunk(chunk_text, metadata, chunk_id))
                
                # 處理第三層項目 (items)
                third_level_items = sub_item.get("items", [])
                for third_item in third_level_items:
                    third_item_title = third_item.get("item", "")
                    third_item_content = third_item.get("content", "")
                    
                    if third_item_content:
                        if len(third_item_content) <= chunk_size:
                            metadata = {
                                "id": third_item.get("metadata", {}).get("id", ""),
                                "spans": third_item.get("metadata", {}).get("spans", {}),
                                "page_range": third_item.get("metadata", {}).get("page_range", {})
                            }
                            chunk_id = f"{article_title}_{item_title}_{sub_item_title}_{third_item_title}"
                            article_chunks.append(create_chunk(third_item_content, metadata, chunk_id))
                        else:
                            # 第三層項目內容較長，需要分割
                            text_chunks = sliding_window_chunks(third_item_content, chunk_size, overlap)
                            for i, chunk_text in enumerate(text_chunks):
                                metadata = {
                                    "id": third_item.get("metadata", {}).get("id", ""),
                                    "spans": third_item.get("metadata", {}).get("spans", {}),
                                    "page_range": third_item.get("metadata", {}).get("page_range", {})
                                }
                                chunk_id = f"{article_title}_{item_title}_{sub_item_title}_{third_item_title}_part_{i+1}"
                                article_chunks.append(create_chunk(chunk_text, metadata, chunk_id))
        
        return article_chunks
    
    # 遍歷所有章節
    chapters = law_data.get("chapters", [])
    for chapter in chapters:
        chapter_title = chapter.get("chapter", "")
        sections = chapter.get("sections", [])
        
        for section in sections:
            section_title = section.get("section", "")
            articles = section.get("articles", [])
            
            for article in articles:
                article_chunks = process_article(article, chapter_title, section_title)
                chunks.extend(article_chunks)
    
    return chunks


# 評測相關函數
def calculate_precision_at_k(retrieved_chunks: List[str], query: str, k: int) -> float:
    """
    計算Precision@K - 檢索出來的tokens中，有多少是真正相關的
    """
    if not retrieved_chunks or k <= 0:
        return 0.0
    
    # 取前k個結果
    top_k_chunks = retrieved_chunks[:k]
    
    # 改進的關鍵詞匹配方法 - 使用字符級匹配
    query_chars = set(query.replace(' ', '').replace('？', '').replace('！', '').replace('，', '').replace('。', ''))
    if not query_chars:
        return 0.0
    
    relevant_count = 0
    for chunk in top_k_chunks:
        chunk_chars = set(chunk.replace(' ', '').replace('，', '').replace('。', '').replace('；', '').replace('：', ''))
        # 如果查詢中的字符有50%以上出現在chunk中，認為相關
        overlap_chars = query_chars & chunk_chars
        if len(overlap_chars) >= len(query_chars) * 0.5:
            relevant_count += 1
    
    return relevant_count / len(top_k_chunks)


def calculate_precision_omega(retrieved_chunks: List[str], query: str) -> float:
    """
    計算PrecisionΩ - 假設Recall是滿分，最大的準確率是多少
    """
    if not retrieved_chunks:
        return 0.0
    
    # 改進的關鍵詞匹配方法 - 使用字符級匹配
    query_chars = set(query.replace(' ', '').replace('？', '').replace('！', '').replace('，', '').replace('。', ''))
    if not query_chars:
        return 0.0
    
    relevant_count = 0
    for chunk in retrieved_chunks:
        chunk_chars = set(chunk.replace(' ', '').replace('，', '').replace('。', '').replace('；', '').replace('：', ''))
        # 如果查詢中的字符有30%以上出現在chunk中，認為相關
        overlap_chars = query_chars & chunk_chars
        if len(overlap_chars) >= len(query_chars) * 0.3:
            relevant_count += 1
    
    return relevant_count / len(retrieved_chunks)


def calculate_recall_at_k(retrieved_chunks: List[str], query: str, k: int, 
                         ground_truth_chunks: List[str] = None) -> float:
    """
    計算Recall@K - 在前K個檢索結果中命中相關chunk的比例
    """
    if not retrieved_chunks or k <= 0:
        return 0.0
    
    # 取前k個結果
    top_k_chunks = retrieved_chunks[:k]
    
    # 如果沒有ground truth，使用關鍵詞匹配作為近似
    if ground_truth_chunks is None:
        # 改進的關鍵詞匹配方法 - 使用字符級匹配
        query_chars = set(query.replace(' ', '').replace('？', '').replace('！', '').replace('，', '').replace('。', ''))
        if not query_chars:
            return 0.0
        
        # 首先計算總相關文檔數量（需要從所有chunks中計算，不只是top_k）
        # 但由於我們沒有訪問所有chunks，我們需要一個近似方法
        # 這裡我們假設總相關文檔數量等於檢索到的相關文檔數量（這是一個近似）
        retrieved_relevant_count = 0
        for chunk in top_k_chunks:
            chunk_chars = set(chunk.replace(' ', '').replace('，', '').replace('。', '').replace('；', '').replace('：', ''))
            # 如果查詢中的字符有50%以上出現在chunk中，認為相關
            overlap_chars = query_chars & chunk_chars
            if len(overlap_chars) >= len(query_chars) * 0.5:
                retrieved_relevant_count += 1
        
        # 由於無法準確計算總相關文檔數量，我們使用一個保守的估計
        # 假設總相關文檔數量至少等於檢索到的相關文檔數量
        total_relevant_estimate = max(retrieved_relevant_count, 1)
        
        return retrieved_relevant_count / total_relevant_estimate
    
    # 使用ground truth計算 - 這裡ground_truth_chunks實際上是所有chunks
    # 首先計算所有chunks中相關的數量
    query_chars = set(query.replace(' ', '').replace('？', '').replace('！', '').replace('，', '').replace('。', ''))
    if not query_chars:
        return 0.0
    
    total_relevant_count = 0
    for chunk in ground_truth_chunks:
        chunk_chars = set(chunk.replace(' ', '').replace('，', '').replace('。', '').replace('；', '').replace('：', ''))
        overlap_chars = query_chars & chunk_chars
        if len(overlap_chars) >= len(query_chars) * 0.3:
            total_relevant_count += 1
    
    # 計算檢索到的相關chunks數量
    retrieved_relevant_count = 0
    for chunk in top_k_chunks:
        chunk_chars = set(chunk.replace(' ', '').replace('，', '').replace('。', '').replace('；', '').replace('：', ''))
        overlap_chars = query_chars & chunk_chars
        if len(overlap_chars) >= len(query_chars) * 0.3:
            retrieved_relevant_count += 1
    
    return retrieved_relevant_count / total_relevant_count if total_relevant_count > 0 else 0


def calculate_faithfulness(chunks: List[str]) -> float:
    """
    計算忠實度 - 評估chunk是否保持完整語義
    基於句子完整性、段落邊界等
    """
    if not chunks:
        return 0.0
    
    total_score = 0.0
    
    for chunk in chunks:
        score = 1.0
        
        # 檢查句子完整性
        sentences = re.split(r'[。！？]', chunk)
        incomplete_sentences = sum(1 for s in sentences if s.strip() and not s.endswith(('。', '！', '？')))
        if len(sentences) > 1:
            score *= (1.0 - incomplete_sentences / len(sentences))
        
        # 檢查段落完整性
        if chunk.startswith(('第', '條', '項', '款')) and not chunk.endswith(('。', '！', '？')):
            score *= 0.8
        
        total_score += score
    
    return total_score / len(chunks)


def calculate_fragmentation_score(chunks: List[str], original_text: str) -> float:
    """
    計算碎片化程度 - 評估文本被分割的細碎程度
    返回值越高表示碎片化越嚴重
    """
    if not chunks or not original_text:
        return 0.0
    
    # 計算平均chunk長度相對於原文的比例
    avg_chunk_length = sum(len(chunk) for chunk in chunks) / len(chunks)
    length_ratio = avg_chunk_length / len(original_text)
    
    # 計算chunk數量
    chunk_count_ratio = len(chunks) / (len(original_text) / 500)  # 以500字符為基準
    
    # 綜合評分
    fragmentation = (1.0 - length_ratio) * 0.6 + chunk_count_ratio * 0.4
    
    return min(1.0, max(0.0, fragmentation))


def generate_questions_with_gemini(text_content: str, num_questions: int, 
                                 question_types: List[str], difficulty_levels: List[str]) -> List[GeneratedQuestion]:
    """
    使用Gemini生成繁體中文法律考古題
    參考ihower文章的做法，從文本中隨機選擇內容生成問題
    """
    if not GEMINI_AVAILABLE:
        return generate_questions_fallback(text_content, num_questions)
    
    try:
        # 優先使用 GOOGLE_API_KEY，如果沒有則使用 GEMINI_API_KEY
        api_key = GOOGLE_API_KEY or os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("警告：GOOGLE_API_KEY 和 GEMINI_API_KEY 都未設置，使用備用方法")
            return generate_questions_fallback(text_content, num_questions)
        
        cfg = getattr(genai, "configure", None)
        if callable(cfg):
            cfg(api_key=api_key)  # type: ignore[misc]
        ModelCls = getattr(genai, "GenerativeModel", None)
        if ModelCls is None:
            print("警告：無法獲取 GenerativeModel 類，使用備用方法")
            return generate_questions_fallback(text_content, num_questions)
        model = ModelCls('gemini-2.0-flash-exp')
        
        # 從文本中隨機選擇4000 tokens的內容（模擬ihower的做法）
        import random
        text_chunks = text_content.split('\n')
        random.shuffle(text_chunks)
        
        # 選擇足夠的內容來生成問題
        selected_content = ""
        current_tokens = 0
        max_tokens = 4000
        
        for chunk in text_chunks:
            if current_tokens + len(chunk) > max_tokens:
                break
            selected_content += chunk + "\n"
            current_tokens += len(chunk)
        
        if not selected_content.strip():
            selected_content = text_content[:2000]  # 備用方案
        
        prompt = f"""
你是一位專業的法律教育專家，請根據以下法律文本內容，生成{num_questions}道繁體中文考古題。

重要要求：
1. 所有問題必須使用繁體中文（台灣用法）
2. 問題類型應包含：{', '.join(question_types)}，隨機分配但確保多樣性
3. 難度等級應包含：{', '.join(difficulty_levels)}，隨機分配，基礎問題聚焦單一概念，進階問題涉及多概念，應用問題模擬實務場景
4. 每道題目都要標明相關的法規條文
5. 問題應該基於文本中的具體內容，不是泛泛而談
6. 問題應該有明確的答案，可以在文本中找到依據

核心設計原則：
7. 重點：避免純粹的條文背誦題目，改為實際生活案例應用題
8. 問題應該設計成情境式案例，讓學生思考如何在實際生活中應用法律概念
9. 使用「如果...那麼...」或「當...時...」的情境設定
10. 提供具體的生活場景（如：網路使用、創作分享、商業活動等）
11. 詢問「應該如何處理」、「是否符合法律規定」、「會產生什麼後果」等
12. 避免直接問「第X條規定什麼」這類背誦題

文本內容：
{selected_content}

請以JSON格式返回結果，格式如下：
{{
  "questions": [
    {{
      "question": "問題內容",
      "references": ["第X條", "第Y條第Z項"],
      "question_type": "案例應用/情境分析/實務處理/法律後果/合規判斷",
      "difficulty": "基礎/進階/應用",
      "keywords": ["關鍵詞1", "關鍵詞2"],
      "estimated_tokens": 估算的token數量
    }}
  ]
}}

請確保生成的問題都是實際生活案例應用題，避免條文背誦，讓學生能夠思考如何在真實情境中應用法律知識。
"""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # 解析JSON響應
        try:
            # 清理響應文本，移除可能的markdown格式
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            import json
            result = json.loads(response_text)
            
            questions = []
            for q_data in result.get('questions', []):
                question = GeneratedQuestion(
                    question=q_data.get('question', ''),
                    references=q_data.get('references', []),
                    question_type=q_data.get('question_type', ''),
                    difficulty=q_data.get('difficulty', ''),
                    keywords=q_data.get('keywords', []),
                    estimated_tokens=q_data.get('estimated_tokens', 0)
                )
                questions.append(question)
            
            return questions[:num_questions]  # 確保不超過請求數量
            
        except json.JSONDecodeError as e:
            print(f"JSON解析錯誤: {e}")
            print(f"響應內容: {response_text[:500]}...")  # 只顯示前500字符
            return generate_questions_fallback(text_content, num_questions)
        
    except Exception as e:
        print(f"Gemini問題生成失敗: {e}")
        return generate_questions_fallback(text_content, num_questions)


def generate_questions_fallback(text_content: str, num_questions: int) -> List[GeneratedQuestion]:
    """
    備用問題生成方法
    """
    questions = []
    
    # 簡單的正則表達式提取法條
    import re
    articles = re.findall(r'第[一二三四五六七八九十百千0-9]+條[^。]*。', text_content)
    
    print(f"備用方法：從文本中找到 {len(articles)} 個法條")
    
    # 生成基礎問題
    question_templates = [
        ("{article}的定義是什麼？", "定義", "基礎"),
        ("{article}的適用條件為何？", "條件", "基礎"),
        ("違反{article}的法律後果是什麼？", "後果", "進階"),
        ("{article}的申請程序為何？", "程序", "進階"),
        ("{article}的保護期限是多久？", "期限", "基礎"),
    ]
    
    if articles:
        # 如果有法條，基於法條生成問題
        for i in range(min(num_questions, len(articles))):
            article = articles[i % len(articles)]
            template, q_type, difficulty = question_templates[i % len(question_templates)]
            
            # 提取條文號碼
            article_match = re.search(r'第([一二三四五六七八九十百千0-9]+)條', article)
            article_num = article_match.group(1) if article_match else str(i+1)
            
            question = GeneratedQuestion(
                question=template.format(article=f"第{article_num}條"),
                references=[f"第{article_num}條"],
                question_type=q_type,
                difficulty=difficulty,
                keywords=extract_keywords(article, 3),
                estimated_tokens=len(article) + 50
            )
            questions.append(question)
    else:
        # 如果沒有找到法條，生成通用問題
        print("警告：沒有找到法條，生成通用問題")
        generic_questions = [
            "請說明本法律文檔的主要內容和目的？",
            "本法律文檔適用於哪些情況？",
            "違反本法律規定會產生什麼後果？",
            "如何申請本法律規定的相關權利？",
            "本法律規定的保護期限是多久？"
        ]
        
        for i in range(min(num_questions, len(generic_questions))):
            question = GeneratedQuestion(
                question=generic_questions[i],
                references=["相關法條"],
                question_type="基礎概念",
                difficulty="基礎",
                keywords=extract_keywords(text_content[:200], 3),
                estimated_tokens=100
            )
            questions.append(question)
    
    print(f"備用方法生成了 {len(questions)} 個問題")
    return questions


def evaluate_chunk_config(doc: DocRecord, config: ChunkConfig, 
                         test_queries: List[str], k_values: List[int], 
                         strategy: str = "fixed_size") -> EvaluationResult:
    """
    評估單個chunk配置
    """
    # 根據策略生成chunks，傳遞策略特定參數
    if strategy == "fixed_size":
        chunks = sliding_window_chunks(doc.text, config.chunk_size, config.overlap)
    elif strategy == "hierarchical":
        from .chunking import chunk_text
        chunks = chunk_text(doc.text, strategy="hierarchical", 
                           max_chunk_size=config.chunk_size, 
                           overlap_ratio=config.overlap_ratio,
                           min_chunk_size=config.min_chunk_size,
                           level_depth=config.level_depth)
    elif strategy == "rcts_hierarchical":
        from .chunking import chunk_text
        chunks = chunk_text(doc.text, strategy="rcts_hierarchical", 
                           max_chunk_size=config.chunk_size, 
                           overlap_ratio=config.overlap_ratio,
                           preserve_structure=config.preserve_structure)
    elif strategy == "structured_hierarchical":
        from .chunking import chunk_text
        chunks = chunk_text(doc.text, strategy="structured_hierarchical", 
                           json_data=doc.json_data, 
                           max_chunk_size=config.chunk_size, 
                           overlap_ratio=config.overlap_ratio,
                           chunk_by=config.chunk_by)
    elif strategy == "semantic":
        from .chunking import chunk_text
        chunks = chunk_text(doc.text, strategy="semantic", 
                           max_chunk_size=config.chunk_size, 
                           similarity_threshold=config.similarity_threshold,
                           context_window=config.context_window,
                           overlap_ratio=config.overlap_ratio)
    elif strategy == "sliding_window":
        from .chunking import chunk_text
        chunks = chunk_text(doc.text, strategy="sliding_window", 
                           window_size=config.window_size, 
                           step_size=config.step_size,
                           overlap_ratio=config.overlap_ratio,
                           boundary_aware=config.boundary_aware,
                           min_chunk_size_sw=config.min_chunk_size_sw,
                           max_chunk_size_sw=config.max_chunk_size_sw,
                           preserve_sentences=config.preserve_sentences)
    elif strategy == "llm_semantic":
        from .chunking import chunk_text
        chunks = chunk_text(doc.text, strategy="llm_semantic", 
                           max_chunk_size=config.chunk_size, 
                           semantic_threshold=config.semantic_threshold,
                           context_window=config.context_window,
                           overlap_ratio=config.overlap_ratio)
    elif strategy == "hybrid":
        from .chunking import chunk_text
        chunks = chunk_text(doc.text, strategy="hybrid", 
                           primary_size=config.chunk_size, 
                           secondary_size=config.secondary_size,
                           switch_threshold=config.switch_threshold,
                           overlap_ratio=config.overlap_ratio)
    else:
        # 默認使用固定大小分塊
        chunks = sliding_window_chunks(doc.text, config.chunk_size, config.overlap)

    # 計算基本統計
    chunk_count = len(chunks)
    avg_chunk_length = sum(len(c) for c in chunks) / chunk_count if chunk_count else 0.0
    lengths = [len(c) for c in chunks]
    length_variance = (
        sum((l - avg_chunk_length) ** 2 for l in lengths) / chunk_count if chunk_count else 0.0
    )

    # 使用TF-IDF為每個查詢做檢索打分（中文用自定義分詞）
    def to_tokens(s: str) -> str:
        toks = preprocess_text(s)
        return " ".join(toks) if toks else s

    processed_chunks = [to_tokens(c) for c in chunks]
    # 若文檔過短，避免vectorizer報錯
    if not processed_chunks:
        processed_chunks = [""]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^\s]+",
        max_features=5000,
        min_df=1,
        max_df=0.98,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(processed_chunks)

    retrieval_results: Dict[str, List[Dict]] = {}
    precision_at_k_scores: Dict[int, List[float]] = {k: [] for k in k_values}
    recall_at_k_scores: Dict[int, List[float]] = {k: [] for k in k_values}
    precision_omega_scores: List[float] = []

    def compute_pr(retrieved_indices: List[int], relevant_set: set[int], k: int) -> Tuple[float, float]:
        if k <= 0:
            return 0.0, 0.0
        topk = retrieved_indices[:k]
        hit = sum(1 for i in topk if i in relevant_set)
        precision = hit / k
        recall = hit / max(1, len(relevant_set))
        return precision, recall

    max_k = max(k_values) if k_values else 10

    for query in test_queries:
        q = to_tokens(query)
        if not q.strip():
            q = query or ""

        q_vec = vectorizer.transform([q])
        # 余弦相似度
        sims = cosine_similarity(q_vec, X).ravel()
        ranked_idx = sims.argsort()[::-1].tolist()

        # 定義相關集：分數達到最佳分數的某一比例閾值（例如0.7）且>0
        best = float(sims[ranked_idx[0]]) if ranked_idx else 0.0
        threshold = best * 0.7 if best > 0 else 0.0
        relevant_set = {i for i, s in enumerate(sims) if s >= threshold and s > 0}
        # 防止空集合導致recall無意義，若全部為0分，則認為沒有相關文檔
        # 若只有極少數非零，至少保留top1為相關
        if best > 0 and not relevant_set:
            relevant_set = {ranked_idx[0]}

        # 保存前max_k個檢索結果供審查
        retrieval_results[query] = [
            {
                "chunk_index": i,
                "score": float(sims[i]),
                "content": (chunks[i][:200] + "...") if len(chunks[i]) > 200 else chunks[i],
            }
            for i in ranked_idx[:max_k]
        ]

        # 指標計算
        for k in k_values:
            p, r = compute_pr(ranked_idx, relevant_set, k)
            precision_at_k_scores[k].append(p)
            recall_at_k_scores[k].append(r)

        # PrecisionΩ: 理想情況下（最優排序）在k=max_k時可達到的精度
        # = min(|R|, max_k) / max_k
        precision_omega_scores.append(
            min(len(relevant_set), max_k) / max_k if max_k > 0 else 0.0
        )

    # 聚合平均
    avg_precision_omega = sum(precision_omega_scores) / len(precision_omega_scores) if precision_omega_scores else 0.0
    avg_precision_at_k = {k: (sum(v) / len(v) if v else 0.0) for k, v in precision_at_k_scores.items()}
    avg_recall_at_k = {k: (sum(v) / len(v) if v else 0.0) for k, v in recall_at_k_scores.items()}

    metrics = EvaluationMetrics(
        precision_omega=avg_precision_omega,
        precision_at_k=avg_precision_at_k,
        recall_at_k=avg_recall_at_k,
        chunk_count=chunk_count,
        avg_chunk_length=avg_chunk_length,
        length_variance=length_variance,
    )

    # 創建詳細的配置信息，包含所有策略特定參數
    detailed_config = {
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "overlap_ratio": config.overlap_ratio,
        "strategy": strategy,
    }
    
    # 根據策略添加特定參數
    if strategy == "structured_hierarchical":
        detailed_config["chunk_by"] = config.chunk_by
    elif strategy == "rcts_hierarchical":
        detailed_config["preserve_structure"] = config.preserve_structure
    elif strategy == "hierarchical":
        detailed_config["level_depth"] = config.level_depth
        detailed_config["min_chunk_size"] = config.min_chunk_size
    elif strategy == "semantic":
        detailed_config["similarity_threshold"] = config.similarity_threshold
        detailed_config["context_window"] = config.context_window
    elif strategy == "llm_semantic":
        detailed_config["semantic_threshold"] = config.semantic_threshold
        detailed_config["context_window"] = config.context_window
    elif strategy == "sliding_window":
        detailed_config["window_size"] = config.window_size
        detailed_config["step_size"] = config.step_size
        detailed_config["boundary_aware"] = config.boundary_aware
        detailed_config["preserve_sentences"] = config.preserve_sentences
        detailed_config["min_chunk_size_sw"] = config.min_chunk_size_sw
        detailed_config["max_chunk_size_sw"] = config.max_chunk_size_sw
    elif strategy == "hybrid":
        detailed_config["switch_threshold"] = config.switch_threshold
        detailed_config["secondary_size"] = config.secondary_size

    return EvaluationResult(
        config=detailed_config,
        metrics=metrics,
        test_queries=test_queries,
        retrieval_results=retrieval_results,
        timestamp=datetime.now(),
    )


@app.post("/api/chunk")
def chunk(req: ChunkRequest):
    doc = store.docs.get(req.doc_id)
    if not doc:
        return JSONResponse(status_code=404, content={"error": "doc not found"})
    
    # 導入新的chunking模組
    from .chunking import chunk_text
    
    # 根據不同策略進行分塊
    strategy = req.strategy
    use_json_structure = req.use_json_structure
    
    # 如果啟用JSON結構化分割且有JSON數據，優先使用JSON結構化分割
    if use_json_structure and doc.json_data:
        structured_chunks = json_structured_chunks(doc.json_data, req.chunk_size, req.overlap)
        # 提取純文本chunks用於後續處理
        chunks = [chunk["content"] for chunk in structured_chunks]
        # 存儲結構化chunks到文檔中
        doc.structured_chunks = structured_chunks
    else:
        # 使用新的chunking模組
        chunk_kwargs = {
            "chunk_size": req.chunk_size,
            "overlap": req.overlap,
        }
        
        # 根據策略添加特定參數
        if strategy == 'hierarchical' and req.hierarchical_params:
            chunk_kwargs.update({
                "max_chunk_size": req.chunk_size,
                "min_chunk_size": req.hierarchical_params.get('min_chunk_size', req.chunk_size // 2),
                "overlap_ratio": req.overlap / req.chunk_size if req.chunk_size > 0 else 0.1,
                "level_depth": req.hierarchical_params.get('level_depth', 2)
            })
        elif strategy == 'rcts_hierarchical' and req.rcts_hierarchical_params:
            chunk_kwargs.update({
                "max_chunk_size": req.chunk_size,
                "overlap_ratio": req.rcts_hierarchical_params.get('overlap_ratio', 0.1),
                "preserve_structure": req.rcts_hierarchical_params.get('preserve_structure', True)
            })
        elif strategy == 'structured_hierarchical' and req.structured_hierarchical_params:
            chunk_kwargs.update({
                "max_chunk_size": req.chunk_size,
                "overlap_ratio": req.structured_hierarchical_params.get('overlap_ratio', 0.1),
                "chunk_by": req.structured_hierarchical_params.get('chunk_by', 'article')
            })
        elif strategy == 'adaptive' and req.adaptive_params:
            chunk_kwargs.update({
                "target_size": req.chunk_size,
                "tolerance": req.adaptive_params.get('tolerance', req.chunk_size // 10),
                "semantic_threshold": req.adaptive_params.get('semantic_threshold', 0.7)
            })
        elif strategy == 'hybrid' and req.hybrid_params:
            chunk_kwargs.update({
                "primary_size": req.chunk_size,
                "secondary_size": req.hybrid_params.get('secondary_size', req.chunk_size // 2),
                "switch_threshold": req.hybrid_params.get('switch_threshold', 0.8)
            })
        elif strategy == 'semantic' and req.semantic_params:
            chunk_kwargs.update({
                "target_size": req.chunk_size,
                "similarity_threshold": req.semantic_params.get('similarity_threshold', 0.6),
                "context_window": req.semantic_params.get('context_window', 100)
            })
        
        # 使用新的chunking模組
        chunks = chunk_text(doc.text, strategy=strategy, json_data=doc.json_data, **chunk_kwargs)
        
        # 清空結構化chunks
        doc.structured_chunks = []
    
    # 計算詳細指標
    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunks else 0
    length_variance = 0
    if len(chunk_lengths) > 1:
        variance = sum((length - avg_length) ** 2 for length in chunk_lengths) / len(chunk_lengths)
        length_variance = variance / avg_length if avg_length > 0 else 0
    
    doc.chunks = chunks
    doc.chunk_size = req.chunk_size
    doc.overlap = req.overlap
    # invalidates embeddings for safety
    store.reset_embeddings()
    
    return {
        "doc_id": doc.id, 
        "num_chunks": len(chunks), 
        "chunk_size": req.chunk_size, 
        "overlap": req.overlap,
        "strategy": strategy,
        "sample": chunks[:3],  # 前3個chunks作為預覽
        "all_chunks": chunks,  # 所有chunks
        "metrics": {
            "avg_length": round(avg_length, 2),
            "length_variance": round(length_variance, 3),
            "min_length": min(chunk_lengths) if chunks else 0,
            "max_length": max(chunk_lengths) if chunks else 0,
            "overlap_rate": req.overlap / req.chunk_size if req.chunk_size > 0 else 0
        }
    }


def clean_text_for_gemini(text: str) -> str:
    """清理文本以符合Gemini API要求
    
    注意：Gemini API的payload限制是36000字节，需要按字节数而不是字符数来限制文本大小
    """
    import re
    
    # 移除控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 移除全角特殊字符（但保留中文常用标点）
    # 移除【】、〖〗、『』等全角方括号和特殊符号
    text = re.sub(r'[【】〖〗『』［］〔〕｛｝〈〉《》「」『』]', '', text)
    
    # 处理法律文档的特殊格式
    # 移除过多的换行符，但保留段落结构
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # 移除多余的空格
    text = re.sub(r' +', ' ', text)
    
    # 移除行首行尾空格
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    
    # 按字节数限制文本大小（Gemini API限制是36000字节）
    # 设置安全限制为30000字节，留出余量给JSON payload的其他部分
    MAX_BYTES = 30000
    text_bytes = text.encode('utf-8')
    
    if len(text_bytes) > MAX_BYTES:
        # 截断到安全大小
        truncated_bytes = text_bytes[:MAX_BYTES]
        # 尝试在UTF-8字符边界截断（避免截断多字节字符）
        while truncated_bytes and (truncated_bytes[-1] & 0xC0) == 0x80:
            truncated_bytes = truncated_bytes[:-1]
        
        text = truncated_bytes.decode('utf-8', errors='ignore')
        
        # 尝试在合适的位置截断（如段落边界），保持文本完整性
        # 在最后20%的范围内寻找段落边界
        search_start = int(len(text) * 0.8)
        last_paragraph = text.rfind('\n\n', search_start)
        if last_paragraph > int(len(text) * 0.7):  # 如果找到的段落边界不太远
            text = text[:last_paragraph]
        else:
            # 如果找不到段落边界，尝试在句子边界截断
            last_sentence = max(
                text.rfind('。', search_start),
                text.rfind('！', search_start),
                text.rfind('？', search_start),
                text.rfind('.', search_start),
                text.rfind('!', search_start),
                text.rfind('?', search_start)
            )
            if last_sentence > int(len(text) * 0.7):
                text = text[:last_sentence + 1]
    
    return text.strip()


async def embed_gemini(texts: List[str]) -> List[List[float]]:
    """Call Google Generative API (Gemini) embeddings endpoint using API key.

    Note: This uses the REST endpoint pattern with the API key in query params.
    """
    if not httpx:
        raise RuntimeError("httpx not available")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    # Gemini embedding model: gemini-embedding-001 (維度可配置: 128-3072)
    model = "gemini-embedding-001"
    # 使用正確的 API 端點格式
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
    headers = {
        "x-goog-api-key": GOOGLE_API_KEY,
        "Content-Type": "application/json"
    }
    out: List[List[float]] = []
    total_texts = len(texts)
    
    # 記錄開始時間
    start_time = time.time()
    print(f"🔧 開始Gemini embedding處理，共 {total_texts} 個文本")
    
    async with httpx.AsyncClient(timeout=60) as client:
        # 逐個處理文本（Gemini API 需要單個請求）
        for i, text in enumerate(texts):
            try:
                # 使用專門的文本清理函數
                original_text = text
                text = clean_text_for_gemini(text)
                
                # 如果文本為空或過短，跳過
                if len(text.strip()) < 10:
                    print(f"⚠️ 文本過短或為空，跳過處理")
                    import numpy as np
                    fallback_vector = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32).tolist()
                    out.append(fallback_vector)
                    continue
                
                payload = {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": text}]},
                    "output_dimensionality": EMBEDDING_DIMENSION  # 使用全局配置的維度
                }
                
                r = await client.post(url, headers=headers, json=payload)
                
                if r.status_code == 400:
                    print(f"❌ Gemini API 400錯誤，嘗試處理...")
                    # 尝试读取错误详情
                    try:
                        error_data = r.json()
                        error_message = error_data.get('error', {}).get('message', '')
                        print(f"❌ API錯誤詳情: {error_data}")
                        
                        # 检查是否是payload大小超限错误
                        if 'payload size exceeds' in error_message.lower() or '36000 bytes' in error_message:
                            print(f"⚠️ 檢測到payload大小超限，嘗試截斷文本...")
                            # 获取当前文本的字节大小
                            current_bytes = len(text.encode('utf-8'))
                            print(f"📊 當前文本大小: {current_bytes} 字節")
                            
                            # 更激进的截断：减少到25000字节（留更多余量）
                            MAX_BYTES_RETRY = 25000
                            text_bytes = text.encode('utf-8')
                            
                            if len(text_bytes) > MAX_BYTES_RETRY:
                                truncated_bytes = text_bytes[:MAX_BYTES_RETRY]
                                # 在UTF-8字符边界截断
                                while truncated_bytes and (truncated_bytes[-1] & 0xC0) == 0x80:
                                    truncated_bytes = truncated_bytes[:-1]
                                
                                text = truncated_bytes.decode('utf-8', errors='ignore')
                                
                                # 尝试在段落边界截断
                                search_start = int(len(text) * 0.8)
                                last_paragraph = text.rfind('\n\n', search_start)
                                if last_paragraph > int(len(text) * 0.6):
                                    text = text[:last_paragraph]
                                
                                print(f"📊 截斷後文本大小: {len(text.encode('utf-8'))} 字節")
                                
                                # 重新构建payload并重试
                                payload["content"]["parts"][0]["text"] = text
                                r = await client.post(url, headers=headers, json=payload)
                                
                                if r.status_code == 400:
                                    print(f"⚠️ 截斷後仍失敗，使用fallback向量")
                                    # 使用fallback向量而不是抛出异常
                                    import numpy as np
                                    fallback_vector = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32).tolist()
                                    out.append(fallback_vector)
                                    continue
                                else:
                                    # 截断后重试成功，继续处理响应
                                    print(f"✅ 文本截斷後成功處理")
                            else:
                                print(f"⚠️ 文本大小未超限但仍返回400錯誤，使用fallback向量")
                                import numpy as np
                                fallback_vector = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32).tolist()
                                out.append(fallback_vector)
                                continue
                        else:
                            # 其他类型的400错误，尝试更激进的文本清理
                            cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u3001\u3002\u300a\u300b\u300c\u300d\u300e\u300f\u2018\u2019\u201c\u201d]', ' ', text)
                            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                            
                            if len(cleaned_text) > 10:
                                payload["content"]["parts"][0]["text"] = cleaned_text
                                r = await client.post(url, headers=headers, json=payload)
                                
                                if r.status_code == 400:
                                    print(f"⚠️ 清理後仍失敗，使用fallback向量")
                                    import numpy as np
                                    fallback_vector = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32).tolist()
                                    out.append(fallback_vector)
                                    continue
                                else:
                                    # 清理后重试成功，继续处理响应
                                    print(f"✅ 文本清理後成功處理")
                            else:
                                print(f"⚠️ 清理後文本過短，使用fallback向量")
                                import numpy as np
                                fallback_vector = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32).tolist()
                                out.append(fallback_vector)
                                continue
                    except Exception as parse_error:
                        print(f"❌ 解析錯誤詳情失敗: {parse_error}")
                        print(f"❌ API錯誤響應: {r.text[:200]}")
                        # 使用fallback向量
                        import numpy as np
                        fallback_vector = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32).tolist()
                        out.append(fallback_vector)
                        continue
                
                r.raise_for_status()
                data = r.json()
                
                # 調試：打印完整的API響應結構
                if i == 0:  # 只在第一次打印
                    print(f"📋 Gemini API響應結構: {list(data.keys())}")
                    if "embedding" in data:
                        print(f"📋 Embedding結構: {list(data['embedding'].keys())}")
                
                # 根據官方文檔，響應格式是 {"embedding": {"values": [...]}}
                embedding_values = data.get("embedding", {}).get("values", [])
                
                if not embedding_values:
                    print(f"❌ 獲取到的embedding為空，使用fallback向量")
                    print(f"❌ 完整響應: {data}")
                    import numpy as np
                    fallback_vector = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32).tolist()
                    out.append(fallback_vector)
                else:
                    # 調試：打印實際返回的維度
                    actual_dimension = len(embedding_values)
                    if i == 0:  # 只在第一次打印
                        print(f"✅ Gemini返回的向量維度: {actual_dimension}")
                    if actual_dimension != EMBEDDING_DIMENSION:
                        print(f"⚠️ 警告：Gemini返回的向量維度為 {actual_dimension}，與配置的{EMBEDDING_DIMENSION}不同")
                        print(f"⚠️ 這可能會導致與之前存儲的embedding維度不匹配")
                    out.append(embedding_values)
                
                # 顯示進度
                progress = ((i + 1) / total_texts) * 100
                print(f"📊 Gemini embedding進度: {i + 1}/{total_texts} ({progress:.1f}%)")
                
            except Exception as e:
                print(f"❌ 處理第{i+1}個文本時出錯: {e}")
                print(f"❌ 錯誤文本前100字符: {text[:100] if 'text' in locals() else 'N/A'}")
                
                # 嘗試獲取更詳細的錯誤信息
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    try:
                        error_detail = e.response.json()
                        print(f"❌ API錯誤詳情: {error_detail}")
                    except:
                        print(f"❌ API錯誤響應: {e.response.text[:200]}")
                
                # 使用隨機向量作為fallback
                import numpy as np
                fallback_vector = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32).tolist()
                out.append(fallback_vector)
                continue
    
    # 計算總時間
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_text = total_time / total_texts if total_texts > 0 else 0
    
    print(f"✅ Gemini embedding完成，共處理 {len(out)} 個向量")
    print(f"⏱️  Embedding總花費時間: {total_time:.2f} 秒 (平均每個文本: {avg_time_per_text:.3f} 秒)")
    return out


def embed_bge_m3(texts: List[str]) -> List[List[float]]:
    """使用 BGE-M3 模型進行 embedding"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("sentence-transformers not available")
    
    try:
        total_texts = len(texts)
        
        # 記錄開始時間
        start_time = time.time()
        print(f"🔧 開始BGE-M3 embedding處理，共 {total_texts} 個文本")
        
        # 載入 BGE-M3 模型
        model = SentenceTransformer('BAAI/bge-m3')
        
        # 批量處理文本
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        
        # 轉換為列表格式
        result = embeddings.tolist()
        
        # 計算總時間
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_text = total_time / total_texts if total_texts > 0 else 0
        
        print(f"✅ BGE-M3 embedding完成，共處理 {len(result)} 個向量")
        print(f"⏱️  Embedding總花費時間: {total_time:.2f} 秒 (平均每個文本: {avg_time_per_text:.3f} 秒)")
        return result
        
    except Exception as e:
        raise RuntimeError(f"BGE-M3 embedding failed: {e}")


@app.post("/api/embed")
async def embed(req: EmbedRequest):
    print(f"🔍 Embed函数被调用，请求: {req}")
    
    # 收集選定文檔的chunks
    # 如果沒有指定doc_ids，只選擇使用structured_hierarchical策略的最近文檔
    requested_doc_ids = req.doc_ids
    if requested_doc_ids:
        # 即使指定了doc_ids，也要按文件名去重，避免重複embedding同名文檔
        candidates = []
        for doc_id in requested_doc_ids:
            doc = store.docs.get(doc_id)
            if doc:
                candidates.append((doc_id, doc))
        
        if not candidates:
            return JSONResponse(
                status_code=400,
                content={"error": "指定的文檔ID不存在"}
            )
        
        # 按文件名去重，只選擇每個文件名的第一個文檔（或者chunks最多的）
        filename_to_doc = {}  # {filename: (doc_id, doc, chunk_count)}
        for doc_id, doc in candidates:
            # 優先使用structured_chunks計算chunk數量
            chunk_count = len(doc.structured_chunks) if doc.structured_chunks else (len(doc.chunks) if doc.chunks else 0)
            if doc.filename not in filename_to_doc:
                filename_to_doc[doc.filename] = (doc_id, doc, chunk_count)
            else:
                existing_count = filename_to_doc[doc.filename][2]
                if chunk_count > existing_count:
                    print(f"🔄 發現更新的文檔 {doc.filename}: {chunk_count} > {existing_count} chunks")
                    filename_to_doc[doc.filename] = (doc_id, doc, chunk_count)
                else:
                    print(f"⚠️ 跳過重複文檔 {doc.filename} (doc_id: {doc_id})，已選擇chunks更多的版本")
        
        selected = [doc_id for doc_id, _, _ in filename_to_doc.values()]
        if len(selected) < len(requested_doc_ids):
            print(f"⚠️ 去重後，從 {len(requested_doc_ids)} 個指定的文檔中選擇了 {len(selected)} 個文檔")
    else:
        # 只選擇使用structured_hierarchical或multi_level_structured策略的文檔
        candidates = [
            (doc_id, doc) for doc_id, doc in store.docs.items()
            if doc and getattr(doc, 'chunking_strategy', None) in ['structured_hierarchical', 'multi_level_structured']
        ]
        
        if not candidates:
            return JSONResponse(
                status_code=400,
                content={"error": "沒有找到使用structured_hierarchical策略的文檔。請先進行多層級結構化分塊。"}
            )
        
        # 按文件名去重，只選擇每個文件名的第一個文檔（或者chunks最多的）
        filename_to_doc = {}  # {filename: (doc_id, doc, chunk_count)}
        for doc_id, doc in candidates:
            # 優先使用structured_chunks計算chunk數量
            chunk_count = len(doc.structured_chunks) if doc.structured_chunks else (len(doc.chunks) if doc.chunks else 0)
            if doc.filename not in filename_to_doc:
                filename_to_doc[doc.filename] = (doc_id, doc, chunk_count)
            else:
                existing_count = filename_to_doc[doc.filename][2]
                if chunk_count > existing_count:
                    print(f"🔄 發現更新的文檔 {doc.filename}: {chunk_count} > {existing_count} chunks")
                    filename_to_doc[doc.filename] = (doc_id, doc, chunk_count)
        
        selected = [doc_id for doc_id, _, _ in filename_to_doc.values()]
        print(f"🔍 未指定doc_ids，自動選擇 {len(selected)} 個使用structured_hierarchical策略的文檔（已去重）: {[store.docs[d].filename for d in selected]}")
    
    all_chunks: List[str] = []
    chunk_doc_ids: List[str] = []
    chunk_ids: List[str] = []
    
    # 優先使用structured_chunks，如果沒有才使用doc.chunks
    for doc_id in selected:
        doc = store.docs.get(doc_id)
        if not doc:
            continue
        
        # 優先使用structured_chunks（實際顯示的428個chunks）
        if doc.structured_chunks:
            print(f"✅ 使用文檔 {doc.filename} 的structured_chunks（{len(doc.structured_chunks)}個chunks）")
            for i, chunk_data in enumerate(doc.structured_chunks):
                if isinstance(chunk_data, dict):
                    content = chunk_data.get('content', '')
                else:
                    content = str(chunk_data)
                
                if content:
                    all_chunks.append(content)
                    chunk_doc_ids.append(doc.id)
                    chunk_id = chunk_data.get('chunk_id', '') if isinstance(chunk_data, dict) else f"{doc.id}_{i}"
                    chunk_ids.append(chunk_id if chunk_id else f"{doc.id}_{i}")
        elif doc.chunks:
            # 回退到舊的doc.chunks
            print(f"⚠️ 文檔 {doc.filename} 沒有structured_chunks，使用doc.chunks（{len(doc.chunks)}個chunks）")
            all_chunks.extend(doc.chunks)
            chunk_doc_ids.extend([doc.id] * len(doc.chunks))
            # 生成chunk_id
            for i in range(len(doc.chunks)):
                chunk_ids.append(f"{doc.id}_{i}")

    if not all_chunks:
        return JSONResponse(status_code=400, content={"error": "no chunks to embed"})
    
    # 打印統計信息
    print(f"📊 Embedding統計: 將為 {len(selected)} 個文檔進行embedding，共 {len(all_chunks)} 個chunks")
    for doc_id in selected:
        doc = store.docs.get(doc_id)
        if doc:
            chunk_count = len([c for c in chunk_doc_ids if c == doc_id])
            print(f"   文檔 {doc.filename}: {chunk_count} 個chunks")

    # 調試信息
    print(f"🔍 Embedding 調試信息:")
    print(f"   USE_GEMINI_EMBEDDING: {USE_GEMINI_EMBEDDING}")
    print(f"   GOOGLE_API_KEY: {'已設置' if GOOGLE_API_KEY else '未設置'}")
    print(f"   USE_BGE_M3_EMBEDDING: {USE_BGE_M3_EMBEDDING}")
    print(f"   SENTENCE_TRANSFORMERS_AVAILABLE: {SENTENCE_TRANSFORMERS_AVAILABLE}")
    print(f"🎯 實驗組A統一使用 {EMBEDDING_DIMENSION} 維索引")
    print(f"📊 當前EMBEDDING_DIMENSION配置: {EMBEDDING_DIMENSION}")
    
    # 嘗試使用 Gemini embedding（主要選項）
    if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
        try:
            vectors = await embed_gemini(all_chunks)
            # 使用實際向量維度，如果為空則使用全局配置
            dimension = len(vectors[0]) if vectors and len(vectors) > 0 else EMBEDDING_DIMENSION
            print(f"📊 檢測到embedding維度: {dimension} (配置: {EMBEDDING_DIMENSION})")
            
            # 驗證維度一致性
            if dimension != EMBEDDING_DIMENSION:
                print(f"⚠️ 警告：實際embedding維度({dimension})與配置({EMBEDDING_DIMENSION})不同")
            
            # 清除舊索引（如果維度不同），確保一致性
            if faiss_store.has_vectors() and faiss_store.dimension != dimension:
                print(f"⚠️ 檢測到舊索引維度({faiss_store.dimension})與新embedding維度({dimension})不匹配，清除舊索引")
                faiss_store.reset_vectors()
                # 同時清除BM25索引以保持一致性
                bm25_index.reset_index()
            
            # 創建FAISS索引
            faiss_store.create_index(dimension, "flat")
            faiss_store.add_vectors(vectors, chunk_ids, chunk_doc_ids, all_chunks)
            
            # 構建BM25索引
            bm25_index.build_index(all_chunks, chunk_ids, chunk_doc_ids)
            
            # 檢查是否已有enhanced metadata（在分塊階段生成）
            enhanced_metadata = {}
            if hasattr(store, 'enhanced_metadata') and store.enhanced_metadata:
                print("📋 使用已存在的enhanced metadata...")
                enhanced_metadata = store.enhanced_metadata
                
                # 設置增強metadata到FAISS存儲
                for chunk_id, metadata in enhanced_metadata.items():
                    faiss_store.set_enhanced_metadata(chunk_id, metadata)
            else:
                print("⚠️ 未找到enhanced metadata，HybridRAG將使用基礎metadata")
            
            # 保持原有store的兼容性
            store.embeddings = vectors
            store.chunk_doc_ids = chunk_doc_ids
            store.chunks_flat = all_chunks
            
            # 自動保存數據
            store.save_data()
            faiss_store.save_data()
            bm25_index.save_data()
            
            print(f"✅ 完成embedding: FAISS索引({len(vectors)}向量), BM25索引({len(all_chunks)}文檔), 增強metadata({len(enhanced_metadata)}條)")
            
            return {
                "provider": "gemini", 
                "model": "gemini-embedding-001",
                "num_vectors": len(vectors),
                "dimension": dimension,
                "enhanced_metadata_count": len(enhanced_metadata),
                "faiss_available": True,
                "bm25_available": True
            }
        except Exception as e:
            print(f"Gemini embedding failed: {e}")
            # 如果 Gemini 失敗，嘗試 BGE-M3
    
    # 嘗試使用 BGE-M3 embedding（備用選項）
    if USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            vectors = embed_bge_m3(all_chunks)
            dimension = len(vectors[0]) if vectors else 1024
            
            # 創建FAISS索引
            faiss_store.create_index(dimension, "flat")
            faiss_store.add_vectors(vectors, chunk_ids, chunk_doc_ids, all_chunks)
            
            # 構建BM25索引
            bm25_index.build_index(all_chunks, chunk_ids, chunk_doc_ids)
            
            # 檢查是否已有enhanced metadata（在分塊階段生成）
            enhanced_metadata = {}
            if hasattr(store, 'enhanced_metadata') and store.enhanced_metadata:
                print("📋 使用已存在的enhanced metadata...")
                enhanced_metadata = store.enhanced_metadata
                
                # 設置增強metadata到FAISS存儲
                for chunk_id, metadata in enhanced_metadata.items():
                    faiss_store.set_enhanced_metadata(chunk_id, metadata)
            else:
                print("⚠️ 未找到enhanced metadata，HybridRAG將使用基礎metadata")
            
            # 保持原有store的兼容性
            store.embeddings = vectors
            store.chunk_doc_ids = chunk_doc_ids
            store.chunks_flat = all_chunks
            
            # 自動保存數據
            store.save_data()
            faiss_store.save_data()
            bm25_index.save_data()
            
            print(f"✅ 完成embedding: FAISS索引({len(vectors)}向量), BM25索引({len(all_chunks)}文檔), 增強metadata({len(enhanced_metadata)}條)")
            
            return {
                "provider": "bge-m3", 
                "model": "BAAI/bge-m3",
                "num_vectors": len(vectors),
                "dimension": dimension,
                "enhanced_metadata_count": len(enhanced_metadata),
                "faiss_available": True,
                "bm25_available": True
            }
        except Exception as e:
            print(f"BGE-M3 embedding failed: {e}")
    
    # 沒有可用的 embedding 方法
    return JSONResponse(
        status_code=500, 
        content={
            "error": "No embedding method available. Please configure Gemini API key or BGE-M3 model."
        }
    )


def convert_structured_to_multi_level(structured_chunks):
    """將結構化分塊轉換為論文中的六個粒度級別格式，確保上下文連貫性"""
    # 論文中的六個層次
    six_level_chunks = {
        'document': [],                    # 1. 文件層級 (Document Level)
        'document_component': [],          # 2. 文件組成部分層級 (Document Component Level)
        'basic_unit_hierarchy': [],        # 3. 基本單位層次結構層級 (Basic Unit Hierarchy Level)
        'basic_unit': [],                  # 4. 基本單位層級 (Basic Unit Level)
        'basic_unit_component': [],        # 5. 基本單位組成部分層級 (Basic Unit Component Level)
        'enumeration': []                  # 6. 列舉層級 (Enumeration Level)
    }
    
    for chunk in structured_chunks:
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        # 優先使用metadata中的level信息（多層次結構化分塊會設置這個）
        if 'level' in metadata:
            chunk_by = metadata['level']
        else:
            chunk_by = metadata.get('chunk_by', 'article')
        
        # 將大寫的level名稱轉換為小寫（兼容MultiLevelStructuredChunking生成的格式）
        chunk_by = chunk_by.lower()
        
        # 跳過 Law 層級的 chunk（不包含在分塊和 embedding 中）
        # 因為它包含所有章節內容，過於龐大且不適合檢索
        if chunk_by == 'law' or metadata.get('level_en', '').lower() == 'law':
            continue
        
        # 根據chunk_by和內容特徵分類到對應層次
        level_name, semantic_features = classify_chunk_to_level(content, metadata, chunk_by)
        
        # 處理上下文連貫性：為列舉元素添加父級上下文
        final_content = content
        if level_name == 'enumeration' and chunk_by == 'item':
            # 檢查是否已經包含父級內容（通過檢查是否包含條文主文）
            if not has_parent_context(content, metadata):
                # 嘗試從其他chunks中找到父級條文內容
                parent_content = find_parent_article_content(structured_chunks, metadata)
                if parent_content:
                    final_content = f"{parent_content}\n{content}"
                    # 更新語義特徵以反映上下文連貫性
                    semantic_features['has_parent_context'] = True
                    semantic_features['parent_content_length'] = len(parent_content)
        
        if level_name in six_level_chunks:
            # 保留原始的 chunk_id（如果存在）
            original_chunk_id = chunk.get('chunk_id') or metadata.get('chunk_id')
            
            # 如果沒有 chunk_id，嘗試生成一個（使用 MultiLevelStructuredChunking 的邏輯）
            if not original_chunk_id:
                try:
                    from .chunking import MultiLevelStructuredChunking
                    chunker = MultiLevelStructuredChunking()
                    # 構建完整的 metadata 用於生成 chunk_id
                    full_metadata = {
                        **metadata,
                        'level': chunk_by,  # 使用原始的 level/chunk_by
                        'level_en': chunk_by.capitalize() if chunk_by else 'Article',
                    }
                    original_chunk_id = chunker._generate_provision_id(full_metadata)
                except Exception as e:
                    print(f"⚠️ 生成 chunk_id 失敗: {e}，使用後備 ID")
                    # 後備方案：使用索引生成 ID
                    original_chunk_id = f"{level_name}_{len(six_level_chunks[level_name])}"
            
            six_level_chunks[level_name].append({
                'content': final_content,
                'original_content': content,  # 保留原始內容
                'chunk_id': original_chunk_id,  # 保留或生成 chunk_id
                'metadata': {
                    **metadata,
                    'semantic_level': level_name,
                    'semantic_features': semantic_features,
                    'target_queries': get_target_queries_for_level(level_name),
                    'has_context_consistency': level_name == 'enumeration' and final_content != content
                }
            })
    
    return six_level_chunks


def ensure_multi_level_chunks(doc: DocRecord) -> Optional[Dict[str, Any]]:
    """確保文檔具備multi_level_chunks資料，必要時由structured_chunks轉換"""
    try:
        if getattr(doc, "multi_level_chunks", None):
            return doc.multi_level_chunks
        if getattr(doc, "structured_chunks", None):
            converted = convert_structured_to_multi_level(doc.structured_chunks)
            doc.multi_level_chunks = converted
            store.docs[doc.id] = doc
            store.save_data()
            return converted
    except Exception as e:
        print(f"⚠️ ensure_multi_level_chunks失敗: {e}")
    return None


CHUNK_METADATA_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def _normalize_label(label: Optional[str]) -> str:
    if not label:
        return ""
    return re.sub(r"\s+", "", label).lower()


def _hash_content_snippet(content: str) -> str:
    snippet = (content or "")[:200].encode("utf-8", "ignore")
    return hashlib.md5(snippet).hexdigest()


def _get_chunk_metadata_by_content(doc_id: str, level_name: str, chunk_content: str) -> Dict[str, Any]:
    cache_key = (doc_id, level_name, _hash_content_snippet(chunk_content))
    if cache_key in CHUNK_METADATA_CACHE:
        return CHUNK_METADATA_CACHE[cache_key]
    
    doc = store.get_doc(doc_id)
    if not doc:
        CHUNK_METADATA_CACHE[cache_key] = {}
        return {}
    
    multi_chunks = ensure_multi_level_chunks(doc)
    if not multi_chunks or level_name not in multi_chunks:
        CHUNK_METADATA_CACHE[cache_key] = {}
        return {}
    
    target_snippet = chunk_content[:200]
    for chunk in multi_chunks[level_name]:
        if not isinstance(chunk, dict):
            continue
        content = chunk.get("content", "")
        if content == chunk_content or content[:200] == target_snippet:
            metadata = (chunk.get("metadata") or {}).copy()
            metadata["chunk_id"] = chunk.get("chunk_id")
            metadata["doc_id"] = doc_id
            metadata["level"] = level_name
            CHUNK_METADATA_CACHE[cache_key] = metadata
            return metadata
    
    CHUNK_METADATA_CACHE[cache_key] = {}
    return {}


def _build_chapter_section_catalog(
    max_docs: int = 5,
    max_chapters_per_doc: int = 8,
    max_sections_per_chapter: int = 3,
    max_total_entries: int = 24,
) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    docs_processed = 0
    
    for doc in store.docs.values():
        multi_chunks = ensure_multi_level_chunks(doc)
        if not multi_chunks:
            continue
        
        chapters = multi_chunks.get("document_component") or []
        sections = multi_chunks.get("basic_unit_hierarchy") or []
        
        sections_by_chapter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for section_chunk in sections:
            if not isinstance(section_chunk, dict):
                continue
            meta = section_chunk.get("metadata") or {}
            section_title = (meta.get("section") or "").strip()
            if not section_title or section_title in {"未分類節", "未分類"}:
                continue
            chapter_title = (meta.get("chapter") or "").strip()
            sections_by_chapter[chapter_title].append({
                "section_title": section_title,
                "section_key": _normalize_label(section_title),
                "chunk_id": section_chunk.get("chunk_id"),
                "summary": (section_chunk.get("content") or "")[:360]
            })
        
        chapter_count = 0
        for chapter_chunk in chapters:
            if chapter_count >= max_chapters_per_doc or len(catalog) >= max_total_entries:
                break
            if not isinstance(chapter_chunk, dict):
                continue
            meta = chapter_chunk.get("metadata") or {}
            chapter_title = (meta.get("chapter") or "").strip() or f"{doc.filename}-Chapter-{chapter_count+1}"
            entry = {
                "doc_id": doc.id,
                "law_name": meta.get("law_name") or doc.filename,
                "chapter_title": chapter_title,
                "chapter_key": _normalize_label(chapter_title),
                "chunk_id": chapter_chunk.get("chunk_id"),
                "summary": (chapter_chunk.get("content") or "")[:600],
                "sections": sections_by_chapter.get(chapter_title, [])[:max_sections_per_chapter]
            }
            catalog.append(entry)
            chapter_count += 1
        
        docs_processed += 1
        if docs_processed >= max_docs or len(catalog) >= max_total_entries:
            break
    
    return catalog


def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return None


def _safe_parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        snippet = _extract_json_block(text)
        if snippet:
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return {}
    return {}


async def _llm_select_relevant_chapters(
    query: str,
    catalog: List[Dict[str, Any]],
    max_selected_chapters: int = 3
) -> Dict[str, Any]:
    if not catalog:
        return {}
    
    llm_payload = []
    for entry in catalog:
        llm_payload.append({
            "chunk_id": entry.get("chunk_id"),
            "chapter_title": entry.get("chapter_title"),
            "law_name": entry.get("law_name"),
            "summary": entry.get("routing_summary") or entry.get("summary"),
            "sections": [
                {
                    "chunk_id": section.get("chunk_id"),
                    "section_title": section.get("section_title"),
                    "summary": section.get("routing_summary") or section.get("summary"),
                }
                for section in entry.get("sections", [])
            ]
        })
    catalog_text = json.dumps(llm_payload, ensure_ascii=False, indent=2)
    instruction = (
        "你是法律檢索專家。請閱讀查詢與章節摘要，挑選最相關的章節，"
        "必要時列出特別關鍵的節。只輸出JSON，格式為：\n"
        "{{\n"
        '  "chapters": [\n'
        "    {{\n"
        '      "chapter_title": "...",\n'
        '      "chapter_key": "...",\n'
        '      "reason": "...",\n'
        '      "sections": [\n'
        '        {{"section_title": "...", "section_key": "...", "reason": "..."}}\n'
        "      ]\n"
        "    }}\n"
        "  ],\n"
        '  "thinking": "..." \n'
        "}}\n"
        "章節數量不超過 {max_selected_chapters}。"
    ).format(max_selected_chapters=max_selected_chapters)
    
    prompt = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": f"查詢: {query}\n\n章節/節摘要:\n{catalog_text}"
        }
    ]
    
    try:
        response = await gemini_chat(prompt, model=DEFAULT_THINKING_MODEL)
    except Exception as e:
        error_msg = f"LLM 章節選擇失敗: {str(e)}"
        print(f"⚠️ {error_msg}")
        raise RuntimeError(error_msg)
    
    parsed = _safe_parse_json(response)
    if not parsed or not parsed.get("chapters"):
        error_msg = "LLM 返回結果無效或缺少章節信息"
        print(f"⚠️ {error_msg}")
        print(f"   LLM 原始響應（前500字符）: {response[:500] if response else 'None'}")
        print(f"   解析後的結果: {parsed}")
        raise ValueError(error_msg)
    
    # 驗證 chapters 是否為空列表
    chapters = parsed.get("chapters", [])
    if not isinstance(chapters, list) or len(chapters) == 0:
        error_msg = f"LLM 返回的 chapters 為空或格式錯誤（期望非空列表，實際: {type(chapters).__name__}）"
        print(f"⚠️ {error_msg}")
        print(f"   LLM 原始響應（前500字符）: {response[:500] if response else 'None'}")
        raise ValueError(error_msg)
    
    return parsed


def _fallback_catalog_selection(catalog: List[Dict[str, Any]], top_n: int = 3) -> Dict[str, Any]:
    selected = []
    for entry in catalog[:top_n]:
        selected.append({
            "chapter_title": entry.get("chapter_title"),
            "chapter_key": entry.get("chapter_key"),
            "chunk_id": entry.get("chunk_id"),
            "sections": entry.get("sections", [])[:1],
            "reason": "依預設順序選取（LLM未提供結果）"
        })
    return {
        "chapters": selected,
        "thinking": "LLM選擇失敗，採用預設章節",
        "_fallback": True
    }


def _prepare_selection_sets(llm_result: Dict[str, Any], fallback_catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not llm_result or not llm_result.get("chapters"):
        raise ValueError("LLM 結果無效或缺少章節信息，無法處理")
    
    chapter_keys = set()
    chapter_titles = set()
    section_keys = set()
    section_titles = set()
    chapter_chunk_ids = set()
    section_chunk_ids = set()
    structured_details = []
    
    for chapter in llm_result.get("chapters", []):
        chapter_title = chapter.get("chapter_title") or ""
        chapter_key = chapter.get("chapter_key") or _normalize_label(chapter_title)
        chunk_id = chapter.get("chunk_id") or ""
        chapter_titles.add(chapter_title.strip())
        if chapter_key:
            chapter_keys.add(chapter_key)
        if chunk_id:
            chapter_chunk_ids.add(chunk_id)
        detail = {
            "chapter_title": chapter_title,
            "chapter_key": chapter_key,
            "chunk_id": chunk_id,
            "reason": chapter.get("reason", ""),
            "sections": []
        }
        for section in chapter.get("sections", []):
            section_title = section.get("section_title") or ""
            section_key = section.get("section_key") or _normalize_label(section_title)
            sec_chunk_id = section.get("chunk_id") or ""
            section_titles.add(section_title.strip())
            if section_key:
                section_keys.add(section_key)
            if sec_chunk_id:
                section_chunk_ids.add(sec_chunk_id)
            detail["sections"].append({
                "section_title": section_title,
                "section_key": section_key,
                "chunk_id": sec_chunk_id,
                "reason": section.get("reason", "")
            })
        structured_details.append(detail)
    
    return {
        "chapter_titles": chapter_titles,
        "chapter_keys": chapter_keys,
        "section_titles": section_titles,
        "section_keys": section_keys,
        "chapter_chunk_ids": chapter_chunk_ids,
        "section_chunk_ids": section_chunk_ids,
        "details": structured_details,
        "thinking": llm_result.get("thinking", ""),
        "fallback_used": bool(llm_result.get("_fallback"))
    }


def _chunk_matches_selection(metadata: Dict[str, Any], selection: Dict[str, Any]) -> bool:
    chapter = _normalize_label(metadata.get("chapter"))
    section = _normalize_label(metadata.get("section"))
    chunk_id = (metadata.get("chunk_id") or "").strip()
    if selection.get("chapter_chunk_ids") or selection.get("section_chunk_ids"):
        if chunk_id and (
            chunk_id in selection.get("chapter_chunk_ids", set())
            or chunk_id in selection.get("section_chunk_ids", set())
        ):
            return True
    if not selection["chapter_keys"] and not selection["section_keys"]:
        return True
    if chapter and (chapter in selection["chapter_keys"] or metadata.get("chapter", "").strip() in selection["chapter_titles"]):
        return True
    if section and (section in selection["section_keys"] or metadata.get("section", "").strip() in selection["section_titles"]):
        return True
    return False


def _load_chapter_summary_cache() -> Dict[str, Any]:
    global chapter_summary_cache, chapter_summary_loaded
    if chapter_summary_loaded:
        return chapter_summary_cache
    try:
        with open(CHAPTER_SUMMARY_PATH, "r", encoding="utf-8") as f:
            chapter_summary_cache = json.load(f)
    except FileNotFoundError:
        chapter_summary_cache = {"entries": {}}
    except Exception as e:
        print(f"⚠️ 無法讀取章節摘要檔: {e}")
        chapter_summary_cache = {"entries": {}}
    if "entries" not in chapter_summary_cache:
        chapter_summary_cache["entries"] = {}
    chapter_summary_loaded = True
    return chapter_summary_cache


def _save_chapter_summary_cache():
    cache = _load_chapter_summary_cache()
    os.makedirs(os.path.dirname(CHAPTER_SUMMARY_PATH), exist_ok=True)
    try:
        with open(CHAPTER_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 無法儲存章節摘要檔: {e}")


def _get_chapter_summary_entries(doc_id: Optional[str] = None, level: Optional[str] = None) -> List[Dict[str, Any]]:
    cache = _load_chapter_summary_cache()
    entries = list(cache.get("entries", {}).values())
    if doc_id:
        entries = [e for e in entries if e.get("doc_id") == doc_id]
    if level:
        entries = [e for e in entries if e.get("level") == level]
    return entries


def _get_summary_entry_map(doc_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    entries = _get_chapter_summary_entries(doc_id)
    entry_map: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        chunk_id = entry.get("chunk_id")
        if chunk_id:
            entry_map[chunk_id] = entry
    return entry_map


def _get_multi_level_chunker():
    global _multi_level_chunker
    if _multi_level_chunker is None:
        from .chunking import MultiLevelStructuredChunking
        _multi_level_chunker = MultiLevelStructuredChunking()
    return _multi_level_chunker


def _ensure_chunk_id_for_metadata(chunk_data: Dict[str, Any], metadata: Dict[str, Any], fallback_level: str) -> str:
    chunk_id = chunk_data.get("chunk_id") or metadata.get("chunk_id")
    if chunk_id:
        return chunk_id
    try:
        chunker = _get_multi_level_chunker()
        chunk_id = chunker._generate_provision_id(metadata)
    except Exception:
        chunk_id = f"{fallback_level}_{metadata.get('chunk_index', len(metadata))}"
    chunk_data["chunk_id"] = chunk_id
    return chunk_id


async def _summarize_chunk_content(title: str, level: str, law_name: str, content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""
    excerpt = text[:4000]
    system_prompt = (
        "你是一名臺灣法律編輯，負責為章或節生成高度濃縮的摘要。"
        "摘要需使用繁體中文，重點描述該章節規範的主題、適用對象或範圍。"
        "請避免列舉條號，只需2~3句話。"
    )
    user_prompt = (
        f"法規：{law_name}\n"
        f"層級：{level}\n"
        f"標題：{title}\n"
        f"內容：\n{excerpt}\n\n"
        "請輸出2~3句簡潔摘要。"
    )
    try:
        response = await gemini_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=DEFAULT_SUMMARY_MODEL,
        )
        return response.strip()
    except Exception as e:
        print(f"⚠️ 摘要生成失敗: {e}")
        return excerpt[:240]


async def _generate_routing_summary(provision_id: str, chapter_title: str, full_text: str) -> str:
    text = (full_text or "").strip()
    if not text:
        return ""
    prompt_text = (
        "# 角色：法律資訊學專家 (Legal Informatics Expert)\n\n"
        "您是一位專精於法律資訊學與法條結構分析的專家。"
        "請閱讀台灣《著作權法》的單一「章」或「節」的完整條文內容，"
        "並為其生成一個「路由最佳化摘要」（Routing-Optimized Summary）。\n\n"
        "## 摘要黃金準則\n"
        "1. 職能導向：說明該章/節的法律職能（定義、權利限制、罰則、免責等）。\n"
        "2. 關鍵字豐富：包含最重要、最獨特的法律概念與行為動詞。\n"
        "3. 條文索引：指明涵蓋的主要條文編號。\n"
        "4. 精簡客觀：避免評論，維持2~3句精煉描述。\n\n"
        "## 輸出格式\n"
        "請嚴格輸出 JSON：\n"
        "{\n"
        '  "provision_id": "<章節ID>",\n'
        '  "chapter_title": "<章節標題>",\n'
        '  "summary_for_routing": "<路由最佳化摘要>"\n'
        "}\n"
        "勿加入多餘說明。"
    )
    user_prompt = (
        f"[法規章節 ID]:\n{provision_id}\n\n"
        f"[法規章節標題]:\n{chapter_title}\n\n"
        "[法規章節完整內文]:\n"
        f"{text[:6000]}"
    )
    try:
        response = await gemini_chat(
            [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": user_prompt},
            ],
            model=DEFAULT_SUMMARY_MODEL,
        )
        parsed = _safe_parse_json(response)
        summary = ""
        if isinstance(parsed, dict):
            summary = parsed.get("summary_for_routing") or ""
        if not summary and isinstance(response, str):
            summary = response.strip()
        return summary or text[:360]
    except Exception as e:
        print(f"⚠️ 路由摘要生成失敗: {e}")
        return text[:360]


async def ensure_chapter_summaries(
    doc_id: Optional[str] = None,
    include_sections: bool = True,
    max_items: Optional[int] = None,
    target_chunk_ids: Optional[Set[str]] = None
) -> Dict[str, Any]:
    docs: List[DocRecord] = []
    if doc_id:
        doc = store.get_doc(doc_id)
        if doc:
            docs.append(doc)
    else:
        docs = list(store.docs.values())
    
    cache = _load_chapter_summary_cache()
    target_chunk_ids = set(target_chunk_ids) if target_chunk_ids else None
    total_new_documents = 0
    total_new_chapters = 0
    total_new_sections = 0
    total_processed = 0
    max_items = max_items if isinstance(max_items, int) and max_items > 0 else None
    
    for doc in docs:
        multi_chunks = ensure_multi_level_chunks(doc)
        if not multi_chunks:
            continue
        
        law_name = doc.filename
        doc_level_chunks = multi_chunks.get("document") or []
        book_chapters = multi_chunks.get("document_component") or []
        sections = multi_chunks.get("basic_unit_hierarchy") or []
        chapter_map: Dict[str, Dict[str, Any]] = {}
        
        for doc_chunk in doc_level_chunks:
            if max_items and total_processed >= max_items:
                break
            if not isinstance(doc_chunk, dict):
                continue
            metadata = (doc_chunk.get("metadata") or {}).copy()
            title = metadata.get("title") or metadata.get("law_name") or doc.filename
            chunk_id = _ensure_chunk_id_for_metadata(doc_chunk, metadata, "document")
            metadata["chunk_id"] = chunk_id
            if target_chunk_ids and chunk_id not in target_chunk_ids:
                continue
            entry_key = f"{doc.id}:{chunk_id}"
            content = doc_chunk.get("content", "")
            content_hash = hashlib.md5(content.encode("utf-8", "ignore")).hexdigest()
            existing = cache["entries"].get(entry_key)
            if existing and existing.get("content_hash") == content_hash:
                continue
            
            summary = await _summarize_chunk_content(title, "法規", law_name, content)
            cache["entries"][entry_key] = {
                "chunk_id": chunk_id,
                "doc_id": doc.id,
                "doc_name": doc.filename,
                "law_name": metadata.get("law_name") or law_name,
                "title": title,
                "level": "document",
                "document_key": _normalize_label(title),
                "summary": summary,
                "routing_summary": summary,
                "content_hash": content_hash,
                "word_count": len(content),
                "last_updated": datetime.now().isoformat(),
            }
            total_new_documents += 1
            total_processed += 1
            if max_items and total_processed >= max_items:
                break
        
        if max_items and total_processed >= max_items:
            break
        
        for chapter_chunk in book_chapters:
            if not isinstance(chapter_chunk, dict):
                continue
            metadata = (chapter_chunk.get("metadata") or {}).copy()
            title = metadata.get("chapter") or metadata.get("title") or metadata.get("level") or "章節"
            chunk_id = _ensure_chunk_id_for_metadata(chapter_chunk, metadata, "chapter")
            chapter_key = _normalize_label(title) or chunk_id.lower()
            metadata["chunk_id"] = chunk_id
            chapter_map[chapter_key] = {"chunk_id": chunk_id, "metadata": metadata}
            
            if target_chunk_ids and chunk_id not in target_chunk_ids:
                continue
            if max_items and total_new_chapters >= max_items:
                break
            
            entry_key = f"{doc.id}:{chunk_id}"
            content = chapter_chunk.get("content", "")
            content_hash = hashlib.md5(content.encode("utf-8", "ignore")).hexdigest()
            existing = cache["entries"].get(entry_key)
            if existing and existing.get("content_hash") == content_hash:
                continue
            
            summary = await _summarize_chunk_content(title, "章", law_name, content)
            routing_summary = await _generate_routing_summary(chunk_id, title, content)
            cache["entries"][entry_key] = {
                "chunk_id": chunk_id,
                "doc_id": doc.id,
                "doc_name": doc.filename,
                "law_name": metadata.get("law_name") or law_name,
                "title": title,
                "level": "chapter",
                "chapter_key": chapter_key,
                "summary": summary,
                "routing_summary": routing_summary,
                "content_hash": content_hash,
                "word_count": len(content),
                "last_updated": datetime.now().isoformat(),
                "section_count": 0,
            }
            total_new_chapters += 1
            total_processed += 1
            if max_items and total_processed >= max_items:
                break
        
        if (include_sections or target_chunk_ids) and (not max_items or total_processed < max_items):
            for section_chunk in sections:
                if max_items and total_processed >= max_items:
                    break
                if not isinstance(section_chunk, dict):
                    continue
                metadata = (section_chunk.get("metadata") or {}).copy()
                section_title = metadata.get("section") or metadata.get("title") or "節"
                if section_title in {"未分類節", "未分類"}:
                    continue
                chunk_id = _ensure_chunk_id_for_metadata(section_chunk, metadata, "section")
                chapter_key = _normalize_label(metadata.get("chapter") or "")
                parent_chapter = chapter_map.get(chapter_key)
                parent_chunk_id = parent_chapter["chunk_id"] if parent_chapter else None
                if target_chunk_ids:
                    if chunk_id not in target_chunk_ids:
                        continue
                elif not include_sections:
                    continue
                
                entry_key = f"{doc.id}:{chunk_id}"
                content = section_chunk.get("content", "")
                content_hash = hashlib.md5(content.encode("utf-8", "ignore")).hexdigest()
                existing = cache["entries"].get(entry_key)
                if existing and existing.get("content_hash") == content_hash:
                    continue
                
                summary = await _summarize_chunk_content(section_title, "節", law_name, content)
                cache["entries"][entry_key] = {
                    "chunk_id": chunk_id,
                    "doc_id": doc.id,
                    "doc_name": doc.filename,
                    "law_name": metadata.get("law_name") or law_name,
                    "title": section_title,
                    "level": "section",
                    "chapter_key": chapter_key,
                    "section_key": _normalize_label(section_title),
                    "parent_chapter_id": parent_chunk_id,
                    "parent_chapter_key": chapter_key,
                    "summary": summary,
                    "routing_summary": summary,
                    "content_hash": content_hash,
                    "word_count": len(content),
                    "last_updated": datetime.now().isoformat(),
                }
                total_new_sections += 1
                total_processed += 1
        if max_items and total_processed >= max_items:
            break
    
    if total_new_documents or total_new_chapters or total_new_sections:
        _save_chapter_summary_cache()
    
    return {
        "documents_updated": total_new_documents,
        "chapters_updated": total_new_chapters,
        "sections_updated": total_new_sections,
        "total_entries": len(_get_chapter_summary_entries()),
        "doc_id": doc_id,
    }


def _reset_chapter_summaries(doc_id: Optional[str] = None) -> Dict[str, Any]:
    cache = _load_chapter_summary_cache()
    entries = cache.get("entries", {})
    if not isinstance(entries, dict):
        cache["entries"] = {}
        entries = cache["entries"]
    if doc_id:
        targets = [key for key, value in entries.items() if value.get("doc_id") == doc_id]
        for key in targets:
            entries.pop(key, None)
        removed = len(targets)
    else:
        removed = len(entries)
        cache["entries"] = {}
    _save_chapter_summary_cache()
    return {
        "doc_id": doc_id,
        "removed_entries": removed,
        "total_entries": len(_get_chapter_summary_entries())
    }


def _build_selection_from_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    chapter_titles = filters.get("chapter_titles") or []
    section_titles = filters.get("section_titles") or []
    chapter_keys = {_normalize_label(title) for title in chapter_titles if title}
    section_keys = {_normalize_label(title) for title in section_titles if title}
    return {
        "chapter_titles": set(chapter_titles),
        "chapter_keys": set(filter(None, chapter_keys)),
        "section_titles": set(section_titles),
        "section_keys": set(filter(None, section_keys)),
        "chapter_chunk_ids": set(filters.get("chapter_chunk_ids") or []),
        "section_chunk_ids": set(filters.get("section_chunk_ids") or []),
    }


async def _build_chapter_section_catalog(
    doc_id: Optional[str] = None,
    limit_chapters: Optional[int] = None,
    max_sections_per_chapter: Optional[int] = 3,
) -> List[Dict[str, Any]]:
    entries = _get_chapter_summary_entries(doc_id)
    if not entries:
        return []
    
    chapters = [e for e in entries if e.get("level") == "chapter"]
    sections = [e for e in entries if e.get("level") == "section"]
    sections_by_parent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for section in sections:
        parent_id = section.get("parent_chapter_id") or section.get("parent_chapter_key")
        if parent_id:
            sections_by_parent[parent_id].append(section)
    
    catalog: List[Dict[str, Any]] = []
    chapters_sorted = sorted(
        chapters,
        key=lambda c: (
            c.get("law_name") or "",
            c.get("chapter_key") or c.get("title") or "",
        ),
    )
    
    for chapter in chapters_sorted:
        chapter_entry = {
            "doc_id": chapter.get("doc_id"),
            "law_name": chapter.get("law_name"),
            "chapter_title": chapter.get("title"),
            "chapter_key": chapter.get("chapter_key"),
            "chunk_id": chapter.get("chunk_id"),
            "summary": chapter.get("routing_summary") or chapter.get("summary"),
            "routing_summary": chapter.get("routing_summary") or chapter.get("summary"),
            "last_updated": chapter.get("last_updated"),
            "section_count": 0,
            "sections": [],
        }
        parent_id = chapter.get("chunk_id") or chapter.get("chapter_key")
        section_list = sections_by_parent.get(parent_id, [])
        chapter_entry["section_count"] = len(section_list)
        section_list = sorted(section_list, key=lambda s: s.get("title") or "")
        if max_sections_per_chapter:
            section_list = section_list[:max_sections_per_chapter]
        chapter_entry["sections"] = [
            {
                "chunk_id": section.get("chunk_id"),
                "section_title": section.get("title"),
                "section_key": section.get("section_key"),
                    "summary": section.get("routing_summary") or section.get("summary"),
                    "routing_summary": section.get("routing_summary") or section.get("summary"),
                "last_updated": section.get("last_updated"),
            }
            for section in section_list
        ]
        catalog.append(chapter_entry)
        if limit_chapters and len(catalog) >= limit_chapters:
            break
    
    return catalog


GROUP_E_LEVELS = ["basic_unit", "basic_unit_component", "enumeration"]
GROUP_E_LEVEL_WEIGHTS = {
    "basic_unit": 1.0,
    "basic_unit_component": 0.9,
    "enumeration": 0.85
}


async def _embed_query_vector(query: str) -> Optional[List[float]]:
    if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
        vectors = await embed_gemini([query])
        return vectors[0]
    if USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE:
        return embed_bge_m3([query])[0]
    return None


def _prepare_candidate_indices(level_name: str, selection: Dict[str, Any]):
    level_data = store.get_multi_level_embeddings(level_name)
    if not level_data:
        return None
    vectors = level_data["embeddings"]
    chunks = level_data["chunks"]
    doc_ids = level_data["doc_ids"]
    
    candidate_indices = []
    candidate_metadata = []
    candidate_chunks = []
    
    for idx, (doc_id, chunk_content) in enumerate(zip(doc_ids, chunks)):
        metadata = _get_chunk_metadata_by_content(doc_id, level_name, chunk_content)
        if not metadata:
            continue
        if _chunk_matches_selection(metadata, selection):
            candidate_indices.append(idx)
            candidate_metadata.append(metadata)
            candidate_chunks.append(chunk_content)
    
    if not candidate_indices:
        return None
    
    return {
        "vectors": np.array(vectors)[candidate_indices],
        "metadata": candidate_metadata,
        "chunks": candidate_chunks,
        "doc_ids": [doc_ids[idx] for idx in candidate_indices],
        "original_indices": candidate_indices
    }


async def _retrieve_level_with_selection(
    query_vector: List[float],
    level_name: str,
    selection: Dict[str, Any],
    k: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_pack = _prepare_candidate_indices(level_name, selection)
    if not candidate_pack:
        return [], {"candidates": 0, "selected": 0}
    
    vectors = candidate_pack["vectors"]
    if vectors.size == 0:
        return [], {"candidates": 0, "selected": 0}
    
    sims = cosine_similarity([query_vector], vectors)[0]
    order = np.argsort(sims)[::-1][:k]
    weight = GROUP_E_LEVEL_WEIGHTS.get(level_name, 1.0)
    
    results = []
    for rank_idx in order:
        metadata = (candidate_pack["metadata"][rank_idx] or {}).copy()
        chunk_id = metadata.get("chunk_id") or candidate_pack["doc_ids"][rank_idx]
        result = {
            "content": candidate_pack["chunks"][rank_idx],
            "similarity": float(sims[rank_idx]) * weight,
            "raw_similarity": float(sims[rank_idx]),
            "doc_id": candidate_pack["doc_ids"][rank_idx],
            "chunk_id": chunk_id,
            "level": level_name,
            "metadata": metadata
        }
        results.append(result)
    
    contribution = {
        "candidates": len(candidate_pack["metadata"]),
        "selected": len(results),
        "weight": weight
    }
    return results, contribution


async def _run_experimental_group_e_pipeline(query: str, k: int, doc_id: Optional[str] = None) -> Dict[str, Any]:
    # 強制使用已知的正確 doc_id（與 chapter_summaries.json 一致）
    KNOWN_GROUP_E_DOC_ID = "71c23286-d3fb-4982-a2e2-33a931687c5d"
    if not doc_id or doc_id != KNOWN_GROUP_E_DOC_ID:
        original_doc_id = doc_id
        print(
            f"[_run_experimental_group_e_pipeline] ⚠️ 強制覆寫 doc_id={original_doc_id!r} "
            f"為 {KNOWN_GROUP_E_DOC_ID}（確保與 chapter_summaries.json 一致）"
        )
        doc_id = KNOWN_GROUP_E_DOC_ID
    
    catalog = await _build_chapter_section_catalog(
        doc_id=doc_id,
        limit_chapters=24,
        max_sections_per_chapter=3
    )
    if not catalog:
        raise HTTPException(status_code=400, detail=f"無法為 doc_id={doc_id} 構建章節目錄，請確認該文檔是否有章節摘要")
    
    # 嘗試使用 LLM 選擇章節，失敗時使用 fallback
    llm_raw = None
    fallback_used = False
    try:
        llm_raw = await _llm_select_relevant_chapters(query, catalog)
        selection = _prepare_selection_sets(llm_raw, catalog)
    except (RuntimeError, ValueError) as e:
        error_msg = str(e)
        print(f"⚠️ LLM 章節選擇失敗，使用 fallback 機制: {error_msg}")
        # 使用 fallback 機制：選擇前3個章節
        llm_raw = _fallback_catalog_selection(catalog, top_n=3)
        selection = _prepare_selection_sets(llm_raw, catalog)
        fallback_used = True
    
    query_vector = await _embed_query_vector(query)
    if query_vector is None:
        raise HTTPException(status_code=500, detail="無法生成查詢向量")
    
    combined_results = []
    level_contributions = {}
    
    for level_name in GROUP_E_LEVELS:
        level_results, contribution = await _retrieve_level_with_selection(query_vector, level_name, selection, k)
        if level_results:
            combined_results.extend(level_results)
        level_contributions[level_name] = contribution
    
    if not combined_results:
        # 如果選擇過於嚴苛，放寬限制重新檢索
        relaxed_selection = {
            "chapter_titles": set(),
            "chapter_keys": set(),
            "section_titles": set(),
            "section_keys": set()
        }
        fallback_used = True  # 標記使用了完全放寬的 fallback
        for level_name in GROUP_E_LEVELS:
            level_results, contribution = await _retrieve_level_with_selection(query_vector, level_name, relaxed_selection, k)
            if level_results:
                combined_results.extend(level_results)
                level_contributions[level_name] = contribution
    
    combined_results.sort(key=lambda x: x["similarity"], reverse=True)
    fused_results = combined_results[:k]
    
    return {
        "group_info": GRANULARITY_COMBINATIONS.get("group_e"),
        "llm_stage": {
            "selection_details": selection.get("details", []),
            "thinking": selection.get("thinking", ""),
            "fallback_used": fallback_used or selection.get("fallback_used", False),
            "raw_response": llm_raw
        },
        "level_contributions": level_contributions,
        "fused_results": fused_results,
        "total_results": len(fused_results)
    }

_CN_NUM_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
    "千": 1000,
}


LEVEL_ALIAS = {
    "law": "law",
    "document": "law",
    "chapter": "chapter",
    "document_component": "chapter",
    "section": "section",
    "basic_unit_hierarchy": "section",
    "article": "article",
    "basic_unit": "article",
    "paragraph": "paragraph",
    "basic_unit_component": "paragraph",
    "subparagraph": "subparagraph",
    "item": "item",
    "enumeration": "item",
}

LEVEL_SEQUENCE = ["chapter", "section", "article", "paragraph", "subparagraph", "item"]


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", str(text).replace("　", "").strip())


def _cn_to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = _normalize_text(value)
    if not text:
        return None
    if text.isdigit():
        try:
            return int(text)
        except Exception:
            return None
    total = 0
    section = 0
    number = 0
    found = False
    for ch in text:
        if ch in _CN_NUM_MAP:
            v = _CN_NUM_MAP[ch]
            if v < 10:
                number = number * 10 + v
                found = True
            else:
                if number == 0:
                    number = 1
                section += number * v
                number = 0
                found = True
        elif ch.isdigit():
            number = number * 10 + int(ch)
            found = True
        elif ch == "-":
            # 支援 3-1 格式
            section += number
            number = 0
        else:
            continue
    total += section + number
    return total if found else None


def _cn_to_int_str(value: Optional[str]) -> Optional[str]:
    result = _cn_to_int(value)
    return str(result) if result is not None else None


def _extract_level_info(label: Optional[str], level: str) -> Optional[Dict[str, Any]]:
    if not label:
        return None
    original_text = str(label).strip()
    if not original_text:
        return None
    normalized = _normalize_text(original_text)
    info: Dict[str, Any] = {
        "text": original_text,
        "number": None,
        "key": normalized or original_text.strip(),
    }

    if level == "chapter":
        # 匹配「第X章」或「第X章之一」格式
        match = re.search(r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*章(?:[之\-]([0-9一二兩三四五六七八九十百千〇零]+))?", normalized)
        if match:
            main = _cn_to_int_str(match.group(1))
            suffix = _cn_to_int_str(match.group(2)) if match.group(2) else None
            if main:
                info["number"] = main
                if suffix:
                    info["suffix"] = suffix
                    info["canonical"] = f"{main}-{suffix}"
                    info["key"] = info["canonical"]
                else:
                    info["key"] = main
    elif level == "section":
        match = re.search(r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*節", normalized)
        if match:
            number = _cn_to_int_str(match.group(1))
            if number:
                info["number"] = number
                info["key"] = number
    elif level == "article":
        # 優先匹配「第X-Y條」格式（用「-」代表「之」）
        match = re.search(
            r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*[-]\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*條",
            normalized,
        )
        if match:
            main = _cn_to_int_str(match.group(1))
            suffix = _cn_to_int_str(match.group(2))
            if main and suffix:
                info["number"] = main
                info["suffix"] = suffix
                info["canonical"] = f"{main}-{suffix}"
                info["key"] = info["canonical"]
        else:
            # 如果沒有匹配到「第X-Y條」格式，嘗試匹配「第X條之Y」格式
            match = re.search(
                r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*條(?:之\s*([0-9一二兩三四五六七八九十百千〇零]+))?",
                normalized,
            )
            if match:
                main = _cn_to_int_str(match.group(1))
                suffix = _cn_to_int_str(match.group(2)) if match.group(2) else None
                if main:
                    info["number"] = main
                    info["suffix"] = suffix
                    info["canonical"] = f"{main}-{suffix}" if suffix else main
                    info["key"] = info["canonical"]
    elif level == "paragraph":
        match = re.search(r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*項", normalized)
        if match:
            number = _cn_to_int_str(match.group(1))
            if number:
                info["number"] = number
                info["key"] = number
        else:
            stripped = re.sub(r"[（()）)．、.]", "", normalized)
            number = _cn_to_int_str(stripped)
            if number:
                info["number"] = number
                info["key"] = number
    elif level == "subparagraph":
        match = re.search(r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*款", normalized)
        if match:
            number = _cn_to_int_str(match.group(1))
            if number:
                info["number"] = number
                info["key"] = number
        else:
            stripped = re.sub(r"[（()）)．、.]", "", normalized)
            number = _cn_to_int_str(stripped)
            if number:
                info["number"] = number
                info["key"] = number
    elif level == "item":
        match = re.search(r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*目", normalized)
        if match:
            number = _cn_to_int_str(match.group(1))
            if number:
                info["number"] = number
                info["key"] = number
        else:
            stripped = re.sub(r"[（()）)．、.]", "", normalized)
            number = _cn_to_int_str(stripped)
            if number:
                info["number"] = number
                info["key"] = number
    else:
        # fallback already set
        pass

    return info


def build_path_label(path: Dict[str, Any]) -> str:
    parts = []
    law = path.get("law")
    if law and law.get("text"):
        parts.append(law["text"])
    for key in LEVEL_SEQUENCE:
        entry = path.get(key)
        if entry and entry.get("text"):
            parts.append(entry["text"])
    return " > ".join(parts)


def _build_chunk_path(metadata: Dict[str, Any], doc_name: str) -> Dict[str, Any]:
    path: Dict[str, Any] = {}

    law_name = metadata.get("law_name") or metadata.get("category") or doc_name
    if isinstance(law_name, str):
        law_name = law_name.replace("法規名稱：", "").strip()
    law_key = _normalize_text(law_name)
    path["law"] = {
        "text": law_name,
        "key": law_key or law_name,
    }

    chapter_info = _extract_level_info(metadata.get("chapter"), "chapter")
    if chapter_info:
        path["chapter"] = chapter_info

    section_info = _extract_level_info(metadata.get("section"), "section")
    if section_info:
        path["section"] = section_info

    article_info = _extract_level_info(metadata.get("article"), "article")
    if article_info:
        path["article"] = article_info

    paragraph_source = metadata.get("paragraph")
    if not paragraph_source and isinstance(metadata.get("items"), list) and metadata["items"]:
        paragraph_source = metadata["items"][0]
    paragraph_info = _extract_level_info(paragraph_source, "paragraph")
    if paragraph_info:
        path["paragraph"] = paragraph_info

    subparagraph_info = _extract_level_info(metadata.get("subparagraph"), "subparagraph")
    if subparagraph_info:
        path["subparagraph"] = subparagraph_info

    item_info = _extract_level_info(metadata.get("item"), "item")
    if item_info:
        path["item"] = item_info

    return path


def _resolve_level(metadata: Dict[str, Any], fallback: str) -> str:
    level_en = metadata.get("level_en") or metadata.get("level")
    if isinstance(level_en, str):
        normalized = level_en.lower()
        return LEVEL_ALIAS.get(normalized, normalized)
    return LEVEL_ALIAS.get(fallback.lower(), fallback.lower())


def _build_chunk_records(doc: DocRecord) -> List[Dict[str, Any]]:
    multi_level = ensure_multi_level_chunks(doc)
    if not multi_level:
        return []

    records: List[Dict[str, Any]] = []
    for level_name, chunk_list in multi_level.items():
        if not isinstance(chunk_list, list):
            continue
        for idx, chunk_data in enumerate(chunk_list):
            if not isinstance(chunk_data, dict):
                continue
            content = chunk_data.get("content") or chunk_data.get("original_content") or ""
            metadata = chunk_data.get("metadata", {}) or {}
            chunk_index = metadata.get("chunk_index", idx)
            level = _resolve_level(metadata, level_name)
            chunk_id = (
                chunk_data.get("chunk_id")
                or metadata.get("chunk_id")
                or f"{doc.id}_{level}_{chunk_index}"
            )
            path = _build_chunk_path(metadata, doc.filename)
            record = {
                "doc_id": doc.id,
                "doc_name": doc.filename,
                "level": level,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "content": content,
                "metadata": metadata,
                "path": path,
            }
            records.append(record)
    return records


def _get_or_create_child(parent: Dict[str, Any], level: str, key: str, label: str, parent_path_label: str) -> Dict[str, Any]:
    for child in parent.setdefault("children", []):
        if child.get("level") == level and child.get("key") == key:
            return child
    node = {
        "level": level,
        "key": key,
        "label": label,
        "path_label": f"{parent_path_label} > {label}" if parent_path_label else label,
        "children": [],
        "chunk": None,
    }
    node["doc_id"] = parent.get("doc_id")
    node["doc_name"] = parent.get("doc_name")
    parent["children"].append(node)
    return node


def build_hierarchy_tree_for_doc(doc: DocRecord) -> Dict[str, Any]:
    records = _build_chunk_records(doc)
    doc_entry = {
        "doc_id": doc.id,
        "doc_name": doc.filename,
        "law_nodes": [],
    }
    if not records:
        return doc_entry

    law_nodes: Dict[str, Dict[str, Any]] = {}
    records_sorted = sorted(records, key=lambda r: r.get("chunk_index", 0))

    for record in records_sorted:
        path = record["path"]
        law = path.get("law", {})
        law_key = law.get("key") or law.get("text") or doc.filename
        law_label = law.get("text") or doc.filename
        law_node = law_nodes.get(law_key)
        if not law_node:
            law_node = {
                "level": "law",
                "key": law_key,
                "label": law_label,
                "path_label": law_label,
                "children": [],
                "chunk": None,
            }
            law_node["doc_id"] = doc.id
            law_node["doc_name"] = doc.filename
            law_nodes[law_key] = law_node

        level_nodes = {"law": law_node}
        current_parent = law_node
        for level_name in LEVEL_SEQUENCE:
            entry = path.get(level_name)
            if entry and entry.get("key"):
                child = _get_or_create_child(
                    current_parent,
                    level_name,
                    entry["key"],
                    entry.get("text", entry["key"]),
                    current_parent.get("path_label", ""),
                )
                level_nodes[level_name] = child
                current_parent = child

        target_level = record["level"]
        target_node = level_nodes.get(target_level, current_parent)
        target_node["chunk"] = {
            "chunk_id": record["chunk_id"],
            "level": record["level"],
            "content": record["content"],
            "content_length": len(record["content"]),
            "metadata": record["metadata"],
            "strategy": record["metadata"].get("strategy"),
            "path_label": build_path_label(record["path"]),
        }
        target_node["path"] = record["path"]
        target_node["doc_id"] = record["doc_id"]
        target_node["doc_name"] = record["doc_name"]

    doc_entry["law_nodes"] = list(law_nodes.values())
    return doc_entry


def _parse_citation_query(query: str) -> Dict[str, Any]:
    original = query or ""
    normalized = _normalize_text(original)
    filters: Dict[str, Any] = {
        "raw": original,
    }

    # 檢查是否為 ID 搜尋（例如：Art_11, Chap_4_1, Art_87_P1_C8_I1）
    id_match = re.match(r'^(Art|Chap|Sec|Law)_([A-Za-z0-9_]+)$', query.strip())
    if id_match:
        filters["chunk_id"] = query.strip()
        return filters

    # 匹配「第X章」或「第X章之一」格式
    chapter_match = re.search(r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*章(?:[之\-]([0-9一二兩三四五六七八九十百千〇零]+))?", normalized)
    if chapter_match:
        main = _cn_to_int_str(chapter_match.group(1))
        suffix = _cn_to_int_str(chapter_match.group(2)) if chapter_match.group(2) else None
        if main:
            filters["chapter"] = main
            if suffix:
                filters["chapter_suffix"] = suffix
                filters["chapter_canonical"] = f"{main}-{suffix}"

    section_match = re.search(r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*節", normalized)
    if section_match:
        filters["section"] = _cn_to_int_str(section_match.group(1))

    # 優先匹配「第X-Y條」格式（用「-」代表「之」）
    article_match = re.search(
        r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*[-]\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*條",
        normalized,
    )
    if article_match:
        main = _cn_to_int_str(article_match.group(1))
        suffix = _cn_to_int_str(article_match.group(2))
        if main and suffix:
            filters["article"] = main
            filters["article_suffix"] = suffix
            filters["article_canonical"] = f"{main}-{suffix}"
    else:
        # 如果沒有匹配到「第X-Y條」格式，嘗試匹配「第X條之Y」格式
        article_match = re.search(
            r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*條(?:之\s*([0-9一二兩三四五六七八九十百千〇零]+))?",
            normalized,
        )
        if article_match:
            main = _cn_to_int_str(article_match.group(1))
            suffix = _cn_to_int_str(article_match.group(2)) if article_match.group(2) else None
            if main:
                filters["article"] = main
                filters["article_suffix"] = suffix
                filters["article_canonical"] = f"{main}-{suffix}" if suffix else main

    paragraph_match = re.search(r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*項", normalized)
    if paragraph_match:
        filters["paragraph"] = _cn_to_int_str(paragraph_match.group(1))

    subparagraph_match = re.search(r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*款", normalized)
    if subparagraph_match:
        filters["subparagraph"] = _cn_to_int_str(subparagraph_match.group(1))

    item_match = re.search(r"第\s*([0-9一二兩三四五六七八九十百千〇零]+)\s*目", normalized)
    if item_match:
        filters["item"] = _cn_to_int_str(item_match.group(1))

    trimmed = normalized
    for pattern in [
        r"第\s*[0-9一二兩三四五六七八九十百千〇零]+\s*章",
        r"第\s*[0-9一二兩三四五六七八九十百千〇零]+\s*節",
        r"第\s*[0-9一二兩三四五六七八九十百千〇零]+\s*條(?:之\s*[0-9一二兩三四五六七八九十百千〇零]+)?",
        r"第\s*[0-9一二兩三四五六七八九十百千〇零]+\s*項",
        r"第\s*[0-9一二兩三四五六七八九十百千〇零]+\s*款",
        r"第\s*[0-9一二兩三四五六七八九十百千〇零]+\s*目",
    ]:
        trimmed = re.sub(pattern, "", trimmed)
    law_candidate = trimmed.strip()
    if law_candidate:
        filters["law_key"] = _normalize_text(law_candidate)
        filters["law_text"] = law_candidate

    return filters


def _record_matches_filters(record: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    # 如果提供了 chunk_id 過濾器，直接匹配 chunk_id
    chunk_id_filter = filters.get("chunk_id")
    if chunk_id_filter:
        record_chunk_id = record.get("chunk_id", "")
        # 支持完全匹配或部分匹配（例如：Art_11 可以匹配 Art_11_P1）
        if record_chunk_id == chunk_id_filter or record_chunk_id.startswith(chunk_id_filter + "_"):
            return True
        return False
    
    path = record.get("path", {})
    law_filter = filters.get("law_key")
    if law_filter:
        law_entry = path.get("law")
        if not law_entry:
            return False
        if _normalize_text(law_entry.get("text")) != law_filter:
            return False

    chapter_filter = filters.get("chapter")
    if chapter_filter:
        chapter_entry = path.get("chapter")
        if not chapter_entry:
            return False
        # 檢查是否有章節後綴（「之一」）
        chapter_canonical = filters.get("chapter_canonical")
        if chapter_canonical:
            # 如果有 canonical 格式（例如：4-1），需要匹配 canonical
            entry_canonical = chapter_entry.get("canonical") or chapter_entry.get("key")
            if entry_canonical != chapter_canonical:
                return False
        else:
            # 如果沒有後綴，只匹配主編號
            if chapter_entry.get("number") != chapter_filter:
                return False

    section_filter = filters.get("section")
    if section_filter:
        section_entry = path.get("section")
        if not section_entry or section_entry.get("number") != section_filter:
            return False

    article_filter = filters.get("article_canonical")
    if article_filter:
        article_entry = path.get("article")
        if not article_entry:
            return False
        canonical = article_entry.get("canonical") or article_entry.get("key")
        if canonical != article_filter:
            return False

    paragraph_filter = filters.get("paragraph")
    if paragraph_filter:
        paragraph_entry = path.get("paragraph")
        if not paragraph_entry or paragraph_entry.get("number") != paragraph_filter:
            return False

    subparagraph_filter = filters.get("subparagraph")
    if subparagraph_filter:
        sub_entry = path.get("subparagraph")
        if not sub_entry or sub_entry.get("number") != subparagraph_filter:
            return False

    item_filter = filters.get("item")
    if item_filter:
        item_entry = path.get("item")
        if not item_entry or item_entry.get("number") != item_filter:
            return False

    return True


def classify_chunk_to_level(content: str, metadata: dict, chunk_by: str) -> tuple:
    """根據內容和元數據將chunk分類到合適的層次 - 對應論文中的六個粒度級別"""
    import re
    
    # 根據論文定義的六個粒度級別映射
    level_mapping = {
        # 1) law_name → document
        'law': 'document',
        # 2) chapter → document_component
        'chapter': 'document_component',
        # 3) section → basic_unit_hierarchy
        'section': 'basic_unit_hierarchy',
        # 4) article → basic_unit
        'article': 'basic_unit',
        # 5) paragraph/項 → basic_unit_component
        'paragraph': 'basic_unit_component',
        # 6) subparagraph/款 → enumeration；item/目 → enumeration
        'subparagraph': 'enumeration',
        'item': 'enumeration'
    }
    
    # 首先根據chunk_by確定基本層次
    base_level = level_mapping.get(chunk_by, 'basic_unit')
    
    # 基於內容特徵進行語義分析
    semantic_features = analyze_chunk_semantics(content)
    
    # 根據語義特徵和內容長度進行精細調整
    # 以你指定的固定映射為主；只保留少量合理化（例如 article 的定義性長文可歸到 basic_unit_component）
    if chunk_by == 'article':
        level = 'basic_unit' if not (semantic_features['is_definition'] and len(content) > 200) else 'basic_unit_component'
    elif chunk_by in ('paragraph',):
        # 項（paragraph）固定為 basic_unit_component
        level = 'basic_unit_component'
    elif chunk_by in ('subparagraph', 'item'):
        # 款/目固定為 enumeration（注意：此處的 item 代表「目」）
        level = 'enumeration'
    elif chunk_by == 'chapter':
        level = 'document_component'
    elif chunk_by == 'section':
        level = 'basic_unit_hierarchy'
    elif chunk_by == 'law':
        level = 'document'
    else:
        level = base_level
    
    return level, semantic_features


def analyze_chunk_semantics(content: str) -> dict:
    """分析chunk的語義特徵"""
    import re
    
    features = {
        'is_definition': False,
        'is_procedural': False,
        'is_enumeration': False,
        'is_normative': False,
        'has_article_reference': False,
        'concept_density': 0.0,
        'legal_keywords': []
    }
    
    content_lower = content.lower()
    
    # 檢查定義性內容
    definition_patterns = [
        r'本法所稱.*?是指',
        r'.*?指.*?者',
        r'.*?為.*?者',
        r'定義.*?為',
        r'所謂.*?係指'
    ]
    for pattern in definition_patterns:
        if re.search(pattern, content):
            features['is_definition'] = True
            break
    
    # 檢查程序性內容
    procedural_patterns = [
        r'應.*?申請',
        r'得.*?辦理',
        r'依.*?程序',
        r'如何.*?',
        r'程序.*?',
        r'流程.*?'
    ]
    for pattern in procedural_patterns:
        if re.search(pattern, content):
            features['is_procedural'] = True
            break
    
    # 檢查列舉性內容
    enumeration_patterns = [
        r'[（(]\d+[）)]',
        r'[一二三四五六七八九十]+[、．]',
        r'\d+[、．]',
        r'第.*?項',
        r'第.*?款'
    ]
    for pattern in enumeration_patterns:
        if re.search(pattern, content):
            features['is_enumeration'] = True
            break
    
    # 檢查規範性內容
    normative_patterns = [
        r'應.*?',
        r'得.*?',
        r'不得.*?',
        r'禁止.*?',
        r'規定.*?'
    ]
    for pattern in normative_patterns:
        if re.search(pattern, content):
            features['is_normative'] = True
            break
    
    # 檢查法條引用
    if re.search(r'第\s*\d+\s*條', content):
        features['has_article_reference'] = True
    
    # 計算概念密度
    legal_keywords = ['本法', '條文', '規定', '權利', '義務', '申請', '辦理', '程序', '定義', '範圍', '責任', '權力', '職權', '職責', '法律', '法規', '條例']
    keyword_count = sum(1 for keyword in legal_keywords if keyword in content)
    features['concept_density'] = keyword_count / max(len(content.split()), 1)
    features['legal_keywords'] = [kw for kw in legal_keywords if kw in content]
    
    return features


def get_target_queries_for_level(level_name: str) -> list:
    """根據層次返回目標查詢關鍵詞"""
    query_mapping = {
        'document': ['整部', '全文', '整個', '全部'],
        'document_component': ['章', '部分', '編', '篇'],
        'basic_unit_hierarchy': ['節', '標題', '章節'],
        'basic_unit': ['第.*條', '條文', '法條'],
        'basic_unit_component': ['段落', '主文', '內容', '定義'],
        'enumeration': ['項', '目', '款', '子項']
    }
    return query_mapping.get(level_name, ['第.*條'])


def has_parent_context(content: str, metadata: dict) -> bool:
    """檢查內容是否已經包含父級上下文"""
    import re
    
    # 檢查是否包含條文主文的特徵
    article_main_patterns = [
        r'本法.*?定義',
        r'本法.*?規定',
        r'本法.*?用詞',
        r'應.*?申請',
        r'得.*?辦理',
        r'依.*?程序'
    ]
    
    # 如果內容長度較短且不包含條文主文特徵，可能缺少父級上下文
    if len(content) < 200:
        for pattern in article_main_patterns:
            if re.search(pattern, content):
                return True
        return False
    
    return True


def find_parent_article_content(structured_chunks: list, current_metadata: dict) -> str:
    """從結構化chunks中找到父級條文內容"""
    current_article = current_metadata.get('article', '')
    current_chapter = current_metadata.get('chapter', '')
    current_section = current_metadata.get('section', '')
    
    # 查找對應的條文chunk
    for chunk in structured_chunks:
        chunk_metadata = chunk.get('metadata', {})
        chunk_by = chunk_metadata.get('chunk_by', '')
        
        # 找到對應的條文chunk
        if (chunk_by == 'article' and 
            chunk_metadata.get('article', '') == current_article and
            chunk_metadata.get('chapter', '') == current_chapter and
            chunk_metadata.get('section', '') == current_section):
            
            content = chunk.get('content', '')
            # 提取條文主文部分（排除項目內容）
            lines = content.split('\n')
            main_content_lines = []
            
            for line in lines:
                line = line.strip()
                # 如果遇到項目標記，停止提取主文
                if re.match(r'^[一二三四五六七八九十]+[、．]', line) or re.match(r'^\d+[、．]', line):
                    break
                # 包含条文标题和主文内容，但排除结构信息
                if line and not line.startswith('【') and not line.startswith('章') and not line.startswith('節'):
                    main_content_lines.append(line)
            
            return '\n'.join(main_content_lines)
    
    return ""


@app.post("/api/generate-enhanced-metadata")
async def generate_enhanced_metadata(req: Dict[str, Any]):
    """在分塊階段生成enhanced metadata - 專門用於HybridRAG"""
    print(f"🔧 生成enhanced metadata請求: {req}")
    
    try:
        # 獲取所有chunks
        all_chunks = []
        chunk_ids = []
        chunk_doc_ids = []
        
        for doc_id, doc in store.docs.items():
            if doc.structured_chunks:
                for chunk in doc.structured_chunks:
                    all_chunks.append(chunk.get("content", ""))
                    chunk_ids.append(chunk.get("chunk_id", f"{doc_id}_{len(chunk_ids)}"))
                    chunk_doc_ids.append(doc_id)
        
        if not all_chunks:
            return {"error": "沒有找到可用的chunks"}
        
        # 準備chunks數據
        chunks_data = [
            {
                "chunk_id": chunk_ids[i],
                "content": all_chunks[i],
                "metadata": {}
            }
            for i in range(len(all_chunks))
        ]
        
        # 批量增強metadata
        print(f"🔧 開始為 {len(chunks_data)} 個chunks生成enhanced metadata...")
        enhanced_metadata = metadata_enhancer.enhance_metadata_batch(chunks_data)
        
        # 保存到store
        store.enhanced_metadata = enhanced_metadata
        store.save_data()
        
        # 統計信息
        article_level_count = sum(1 for meta in enhanced_metadata.values() if meta.get("is_article_level", False))
        chapter_section_count = sum(1 for meta in enhanced_metadata.values() if meta.get("is_chapter_section_level", False))
        inherited_count = sum(1 for meta in enhanced_metadata.values() if meta.get("inherited_from"))
        
        return {
            "success": True,
            "message": "Enhanced metadata生成完成",
            "stats": {
                "total_chunks": len(chunks_data),
                "enhanced_metadata_count": len(enhanced_metadata),
                "article_level_chunks": article_level_count,
                "chapter_section_chunks": chapter_section_count,
                "inherited_chunks": inherited_count
            }
        }
        
    except Exception as e:
        print(f"❌ 生成enhanced metadata失敗: {e}")
        return {"error": f"生成enhanced metadata失敗: {str(e)}"}

@app.get("/api/enhanced-metadata-stats")
async def get_enhanced_metadata_stats():
    """獲取enhanced metadata統計信息"""
    try:
        if not hasattr(store, 'enhanced_metadata') or not store.enhanced_metadata:
            return {
                "enhanced_metadata_count": 0,
                "message": "尚未生成enhanced metadata"
            }
        
        enhanced_metadata = store.enhanced_metadata
        article_level_count = sum(1 for meta in enhanced_metadata.values() if meta.get("is_article_level", False))
        chapter_section_count = sum(1 for meta in enhanced_metadata.values() if meta.get("is_chapter_section_level", False))
        inherited_count = sum(1 for meta in enhanced_metadata.values() if meta.get("inherited_from"))
        
        return {
            "enhanced_metadata_count": len(enhanced_metadata),
            "article_level_chunks": article_level_count,
            "chapter_section_chunks": chapter_section_count,
            "inherited_chunks": inherited_count,
            "enhancement_levels": {
                "full": sum(1 for meta in enhanced_metadata.values() if meta.get("enhancement_level") == "full"),
                "medium": sum(1 for meta in enhanced_metadata.values() if meta.get("enhancement_level") == "medium"),
                "lightweight": sum(1 for meta in enhanced_metadata.values() if meta.get("enhancement_level") == "lightweight"),
                "none": sum(1 for meta in enhanced_metadata.values() if meta.get("enhancement_level") == "none"),
            }
        }
    except Exception as e:
        print(f"❌ 獲取enhanced metadata統計失敗: {e}")
        return {"error": f"獲取統計失敗: {str(e)}"}

@app.get("/api/chunking-hierarchy-stats")
async def get_chunking_hierarchy_stats(doc_id: Optional[str] = None):
    """獲取分塊結果的法律層級統計信息 - 統計實際顯示的分塊列表（428個分塊）"""
    try:
        # 獲取所有文檔的多層級分塊數據
        hierarchy_stats = {
            'document': 0,                    # 章級 (文件層級)
            'document_component': 0,          # 節級 (文件組成部分層級) 
            'basic_unit_hierarchy': 0,        # 條級 (基本單位層次結構層級)
            'basic_unit': 0,                  # 項級 (基本單位層級)
            'basic_unit_component': 0,        # 款級 (基本單位組成部分層級)
            'enumeration': 0                  # 目級 (列舉層級)
        }
        
        total_chunks = 0
        
        # 層級映射：將level_en或chunk_by映射到六層分類
        # 注意：前端顯示標籤為 Chapter->章, Section->節, Article->條, Paragraph->項, Subparagraph->款, Item->目
        def map_level_to_hierarchy(level_en: str = None, chunk_by: str = None) -> str:
            """將level_en或chunk_by映射到六層分類"""
            # 優先使用level_en（如果存在）
            if level_en:
                level_en_lower = level_en.lower()
                if level_en_lower == 'law':
                    return 'document'  # 章級 (文件層級)
                elif level_en_lower == 'chapter':
                    return 'document'  # 章級 (文件層級)
                elif level_en_lower == 'section':
                    return 'document_component'  # 節級 (文件組成部分層級)
                elif level_en_lower == 'article':
                    return 'basic_unit_hierarchy'  # 條級 (基本單位層次結構層級)
                elif level_en_lower == 'paragraph':
                    return 'basic_unit'  # 項級 (基本單位層級)
                elif level_en_lower == 'subparagraph':
                    return 'basic_unit_component'  # 款級 (基本單位組成部分層級)
                elif level_en_lower == 'item':
                    return 'enumeration'  # 目級 (列舉層級)
            
            # 如果沒有level_en，使用chunk_by
            if chunk_by:
                chunk_by_lower = chunk_by.lower()
                if chunk_by_lower == 'law':
                    return 'document'  # 章級 (文件層級)
                elif chunk_by_lower == 'chapter':
                    return 'document'  # 章級 (文件層級)
                elif chunk_by_lower == 'section':
                    return 'document_component'  # 節級 (文件組成部分層級)
                elif chunk_by_lower == 'article':
                    return 'basic_unit_hierarchy'  # 條級 (基本單位層次結構層級)
                elif chunk_by_lower == 'paragraph':
                    return 'basic_unit'  # 項級 (基本單位層級)
                elif chunk_by_lower == 'subparagraph':
                    return 'basic_unit_component'  # 款級 (基本單位組成部分層級)
                elif chunk_by_lower == 'item':
                    return 'enumeration'  # 目級 (列舉層級)
            
            # 默認歸類到項級（basic_unit）
            return 'basic_unit'
        
        # 確定要處理的文檔列表
        if doc_id:
            # 如果指定了 doc_id，只處理該文檔
            doc = store.docs.get(doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="找不到指定文檔")
            docs_to_process = [(doc_id, doc)]
        else:
            # 否則，只統計使用structured_hierarchical策略的文檔（避免統計所有文檔導致數字過大）
            # 按文件名去重，只統計每個文件名的第一個符合條件的文檔（避免重複統計）
            # 如果有同名文檔，優先選擇有structured_chunks且chunks數量最多的
            filename_to_doc = {}  # {filename: (doc_id, doc, chunk_count)}
            
            # 第一次遍歷：找出每個文件名的最佳文檔（有structured_chunks且chunks最多的）
            for current_doc_id, doc in store.docs.items():
                # 只統計structured_hierarchical策略的文檔
                chunking_strategy = getattr(doc, 'chunking_strategy', None)
                if chunking_strategy not in ['structured_hierarchical', 'multi_level_structured']:
                    continue
                
                # 優先統計structured_chunks（實際顯示的428個chunks）
                if doc.structured_chunks:
                    chunk_count = len(doc.structured_chunks)
                    
                    # 如果這個文件名還沒有記錄，或者這個文檔有更多的chunks，則更新
                    if doc.filename not in filename_to_doc:
                        filename_to_doc[doc.filename] = (current_doc_id, doc, chunk_count)
                    else:
                        existing_count = filename_to_doc[doc.filename][2]
                        if chunk_count > existing_count:
                            print(f"🔄 發現更新的文檔 {doc.filename}: {chunk_count} > {existing_count} chunks")
                            filename_to_doc[doc.filename] = (current_doc_id, doc, chunk_count)
            
            docs_to_process = [(doc_id, doc) for doc_id, doc, _ in filename_to_doc.values()]
        
        # 遍歷要處理的文檔
        for current_doc_id, doc in docs_to_process:
            doc_chunk_count = 0
            # 優先統計structured_chunks（實際顯示的分塊）
            if doc.structured_chunks:
                for chunk in doc.structured_chunks:
                    metadata = chunk.get('metadata', {})
                    level_en = metadata.get('level_en') or metadata.get('level')
                    chunk_by = metadata.get('chunk_by')
                    
                    # 映射到六層分類
                    hierarchy_level = map_level_to_hierarchy(level_en, chunk_by)
                    
                    if hierarchy_level in hierarchy_stats:
                        hierarchy_stats[hierarchy_level] += 1
                        total_chunks += 1
                        doc_chunk_count += 1
            
            chunking_strategy = getattr(doc, 'chunking_strategy', None)
            print(f"📊 統計文檔 {doc.filename} (策略: {chunking_strategy}, doc_id: {current_doc_id}): {doc_chunk_count} 個分塊")
        
        # 添加中文層級名稱映射
        level_names = {
            'document': '章級 (文件層級)',
            'document_component': '節級 (文件組成部分層級)',
            'basic_unit_hierarchy': '條級 (基本單位層次結構層級)', 
            'basic_unit': '項級 (基本單位層級)',
            'basic_unit_component': '款級 (基本單位組成部分層級)',
            'enumeration': '目級 (列舉層級)'
        }
        
        print(f"📊 總統計結果: 總分塊數={total_chunks}, 各層級統計={hierarchy_stats}")
        
        return {
            "total_chunks": total_chunks,
            "hierarchy_stats": hierarchy_stats,
            "level_names": level_names,
            "has_multi_level_chunks": any(count > 0 for count in hierarchy_stats.values())
        }
        
    except Exception as e:
        print(f"❌ 獲取分塊層級統計失敗: {e}")
        return {"error": f"獲取統計失敗: {str(e)}"}

@app.get("/api/chunks-by-hierarchy/{level_name}")
async def get_chunks_by_hierarchy(level_name: str, doc_id: Optional[str] = None):
    """根據法律層級獲取chunks列表"""
    try:
        chunks_by_level = []
        
        # 確定要處理的文檔列表
        if doc_id:
            # 如果指定了 doc_id，只處理該文檔
            doc = store.docs.get(doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="找不到指定文檔")
            docs_to_process = [(doc_id, doc)]
        else:
            # 否則處理所有文檔
            docs_to_process = list(store.docs.items())
        
        # 層級映射：將 level_en 映射到六層分類
        def map_level_to_hierarchy(level_en: str = None, chunk_by: str = None) -> Optional[str]:
            """將level_en或chunk_by映射到六層分類"""
            if level_en:
                level_en_lower = level_en.lower()
                if level_en_lower in ['law', 'chapter']:
                    return 'document'
                elif level_en_lower == 'section':
                    return 'document_component'
                elif level_en_lower == 'article':
                    return 'basic_unit_hierarchy'
                elif level_en_lower == 'paragraph':
                    return 'basic_unit'
                elif level_en_lower == 'subparagraph':
                    return 'basic_unit_component'
                elif level_en_lower == 'item':
                    return 'enumeration'
            
            if chunk_by:
                chunk_by_lower = chunk_by.lower()
                if chunk_by_lower in ['law', 'chapter']:
                    return 'document'
                elif chunk_by_lower == 'section':
                    return 'document_component'
                elif chunk_by_lower == 'article':
                    return 'basic_unit_hierarchy'
                elif chunk_by_lower == 'paragraph':
                    return 'basic_unit'
                elif chunk_by_lower == 'subparagraph':
                    return 'basic_unit_component'
                elif chunk_by_lower == 'item':
                    return 'enumeration'
            
            return None
        
        # 遍歷要處理的文檔
        for current_doc_id, doc in docs_to_process:
            # 優先使用 structured_chunks（這是實際顯示的分塊）
            if doc.structured_chunks:
                for i, chunk in enumerate(doc.structured_chunks):
                    metadata = chunk.get('metadata', {})
                    level_en = metadata.get('level_en') or metadata.get('level')
                    chunk_by = metadata.get('chunk_by')
                    
                    # 映射到六層分類
                    mapped_level = map_level_to_hierarchy(level_en, chunk_by)
                    
                    # 檢查是否匹配指定的層級
                    if mapped_level == level_name:
                        # 使用 chunk 中的 chunk_id（如果存在），否則生成
                        chunk_id = chunk.get('chunk_id') or metadata.get('chunk_id') or f"{current_doc_id}_structured_{i}"
                        
                        chunk_info = {
                            'chunk_id': chunk_id,
                            'doc_id': current_doc_id,
                            'doc_name': doc.filename,
                            'level': level_name,
                            'content': chunk.get('content', ''),
                            'metadata': metadata,
                            'span': chunk.get('span', {}),
                            'chunk_index': metadata.get('chunk_index', i)
                        }
                        chunks_by_level.append(chunk_info)
            # 如果沒有 structured_chunks，嘗試使用 multi_level_chunks
            elif doc.multi_level_chunks and isinstance(doc.multi_level_chunks, dict):
                # 從多層級chunks中獲取指定層級的chunks
                if level_name in doc.multi_level_chunks:
                    chunks = doc.multi_level_chunks[level_name]
                    if chunks:
                        for i, chunk_data in enumerate(chunks):
                            # 使用 chunk 中的 chunk_id（如果存在）
                            chunk_id = chunk_data.get('chunk_id') or f"{current_doc_id}_{level_name}_{i}"
                            
                            chunk_info = {
                                'chunk_id': chunk_id,
                                'doc_id': current_doc_id,
                                'doc_name': doc.filename,
                                'level': level_name,
                                'content': chunk_data.get('content', ''),
                                'metadata': chunk_data.get('metadata', {}),
                                'span': chunk_data.get('span', {}),
                                'chunk_index': chunk_data.get('metadata', {}).get('chunk_index', i)
                            }
                            chunks_by_level.append(chunk_info)
        
        return {
            "level_name": level_name,
            "chunks": chunks_by_level,
            "total_count": len(chunks_by_level)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 獲取層級chunks失敗: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"獲取chunks失敗: {str(e)}"}

@app.get("/api/chunk-hierarchy-tree")
async def get_chunk_hierarchy_tree(doc_id: Optional[str] = None):
    """返回文檔分塊的層級樹狀結構"""
    try:
        if doc_id:
            doc = store.docs.get(doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="找不到指定文檔")
            documents = [build_hierarchy_tree_for_doc(doc)]
        else:
            documents = [build_hierarchy_tree_for_doc(doc) for doc in store.docs.values()]

        documents = [doc_tree for doc_tree in documents if doc_tree.get("law_nodes")]

        return {
            "documents": documents,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 生成層級樹失敗: {e}")
        raise HTTPException(status_code=500, detail=f"無法生成層級樹：{e}")


@app.get("/api/chunk-search")
async def search_chunk_by_citation(query: str, doc_id: Optional[str] = None):
    """根據法條文字（例如：第3條第1項第19款第3目）搜尋對應chunk"""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="請提供搜尋關鍵字")

    try:
        filters = _parse_citation_query(query)
        if doc_id:
            doc = store.docs.get(doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="找不到指定文檔")
            records = _build_chunk_records(doc)
        else:
            records: List[Dict[str, Any]] = []
            for doc in store.docs.values():
                records.extend(_build_chunk_records(doc))

        matched_records = [
            record for record in records if _record_matches_filters(record, filters)
        ]

        level_priority = {
            "law": 0,
            "chapter": 1,
            "section": 2,
            "article": 3,
            "paragraph": 4,
            "subparagraph": 5,
            "item": 6,
        }

        matched_records.sort(
            key=lambda r: (
                level_priority.get(r.get("level"), 99),
                r.get("chunk_index", 0),
            )
        )

        results = []
        for record in matched_records:
            path = record.get("path", {})
            results.append(
                {
                    "doc_id": record.get("doc_id"),
                    "doc_name": record.get("doc_name"),
                    "level": record.get("level"),
                    "chunk_id": record.get("chunk_id"),
                    "chunk_index": record.get("chunk_index"),
                    "path": path,
                    "path_label": build_path_label(path),
                    "content": record.get("content", ""),
                    "content_length": len(record.get("content", "") or ""),
                    "metadata": record.get("metadata", {}),
                }
            )

        return {
            "query": query,
            "normalized_filters": {
                key: value
                for key, value in filters.items()
                if key not in {"raw"}
            },
            "result_count": len(results),
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 法條搜尋失敗: {e}")
        raise HTTPException(status_code=500, detail=f"搜尋失敗：{e}")


@app.get("/api/enhanced-metadata-list")
async def get_enhanced_metadata_list():
    """獲取enhanced metadata列表"""
    try:
        if not hasattr(store, 'enhanced_metadata') or not store.enhanced_metadata:
            return {"enhanced_metadata": {}}
        
        return {"enhanced_metadata": store.enhanced_metadata}
    except Exception as e:
        print(f"❌ 獲取enhanced metadata列表失敗: {e}")
        return {"error": f"獲取列表失敗: {str(e)}"}

@app.post("/api/update-enhanced-metadata")
async def update_enhanced_metadata(req: Dict[str, Any]):
    """更新特定chunk的enhanced metadata"""
    try:
        chunk_id = req.get("chunk_id")
        enhanced_metadata = req.get("enhanced_metadata")
        
        if not chunk_id or not enhanced_metadata:
            return {"error": "缺少必要參數"}
        
        # 更新store中的enhanced metadata
        if not hasattr(store, 'enhanced_metadata'):
            store.enhanced_metadata = {}
        
        store.enhanced_metadata[chunk_id] = enhanced_metadata
        store.save_data()
        
        # 同時更新FAISS存儲中的metadata
        if faiss_store.has_vectors():
            faiss_store.set_enhanced_metadata(chunk_id, enhanced_metadata)
            faiss_store.save_data()
        
        return {"success": True, "message": "Enhanced metadata更新成功"}
    except Exception as e:
        print(f"❌ 更新enhanced metadata失敗: {e}")
        return {"error": f"更新失敗: {str(e)}"}

@app.post("/api/multi-level-embed-fast")
async def multi_level_embed_fast(req: Dict[str, Any]):
    """快速多層次embedding - 不進行metadata增強，專門用於多層次融合檢索"""
    print(f"🚀 快速多層次embedding請求: {req}")
    
    # 設置為不進行metadata增強
    req["enable_metadata_enhancement"] = False
    
    # 調用標準的多層次embedding
    return await multi_level_embed(req)

@app.post("/api/multi-level-embed")
async def multi_level_embed(req: Dict[str, Any]):
    """多層次embedding端點 - 為論文中的六個粒度級別創建獨立的embedding"""
    print(f"🔍 多层级Embedding函数被调用，请求: {req}")
    print(f"🔍 配置检查:")
    print(f"   USE_GEMINI_EMBEDDING: {USE_GEMINI_EMBEDDING}")
    print(f"   GOOGLE_API_KEY: {'已設置' if GOOGLE_API_KEY else '未設置'}")
    print(f"   USE_BGE_M3_EMBEDDING: {USE_BGE_M3_EMBEDDING}")
    # 收集選定文檔的多層次chunks
    # 如果沒有指定doc_ids，只選擇使用structured_hierarchical策略的最近文檔
    requested_doc_ids = req.get("doc_ids")
    if requested_doc_ids:
        # 即使指定了doc_ids，也要按文件名去重，避免重複embedding同名文檔
        candidates = []
        for doc_id in requested_doc_ids:
            doc = store.docs.get(doc_id)
            if doc:
                candidates.append((doc_id, doc))
        
        if not candidates:
            return JSONResponse(
                status_code=400,
                content={"error": "指定的文檔ID不存在"}
            )
        
        # 按文件名去重，只選擇每個文件名的第一個文檔（或者chunks最多的）
        filename_to_doc = {}  # {filename: (doc_id, doc, chunk_count)}
        for doc_id, doc in candidates:
            chunk_count = len(doc.structured_chunks) if doc.structured_chunks else 0
            if doc.filename not in filename_to_doc:
                filename_to_doc[doc.filename] = (doc_id, doc, chunk_count)
            else:
                existing_count = filename_to_doc[doc.filename][2]
                if chunk_count > existing_count:
                    print(f"🔄 發現更新的文檔 {doc.filename}: {chunk_count} > {existing_count} chunks")
                    filename_to_doc[doc.filename] = (doc_id, doc, chunk_count)
                else:
                    print(f"⚠️ 跳過重複文檔 {doc.filename} (doc_id: {doc_id})，已選擇chunks更多的版本")
        
        selected = [doc_id for doc_id, _, _ in filename_to_doc.values()]
        if len(selected) < len(requested_doc_ids):
            print(f"⚠️ 去重後，從 {len(requested_doc_ids)} 個指定的文檔中選擇了 {len(selected)} 個文檔")
    else:
        # 只選擇使用structured_hierarchical或multi_level_structured策略的文檔
        candidates = [
            (doc_id, doc) for doc_id, doc in store.docs.items()
            if doc and getattr(doc, 'chunking_strategy', None) in ['structured_hierarchical', 'multi_level_structured']
        ]
        
        if not candidates:
            return JSONResponse(
                status_code=400,
                content={"error": "沒有找到使用structured_hierarchical策略的文檔。請先進行多層級結構化分塊。"}
            )
        
        # 按文件名去重，只選擇每個文件名的第一個文檔（或者chunks最多的）
        filename_to_doc = {}  # {filename: (doc_id, doc, chunk_count)}
        for doc_id, doc in candidates:
            chunk_count = len(doc.structured_chunks) if doc.structured_chunks else 0
            if doc.filename not in filename_to_doc:
                filename_to_doc[doc.filename] = (doc_id, doc, chunk_count)
            else:
                existing_count = filename_to_doc[doc.filename][2]
                if chunk_count > existing_count:
                    print(f"🔄 發現更新的文檔 {doc.filename}: {chunk_count} > {existing_count} chunks")
                    filename_to_doc[doc.filename] = (doc_id, doc, chunk_count)
        
        selected = [doc_id for doc_id, _, _ in filename_to_doc.values()]
        print(f"🔍 未指定doc_ids，自動選擇 {len(selected)} 個使用structured_hierarchical策略的文檔（已去重）: {[store.docs[d].filename for d in selected]}")
    
    experimental_groups = req.get("experimental_groups", [])  # 新增：實驗組選擇
    group_e_filters_raw = req.get("group_e_filters")
    group_e_filters_by_doc: Dict[str, Dict[str, Any]] = {}
    if isinstance(group_e_filters_raw, list):
        for filt in group_e_filters_raw:
            if isinstance(filt, dict):
                filt_doc_id = filt.get("doc_id")
                if filt_doc_id:
                    group_e_filters_by_doc[filt_doc_id] = _build_selection_from_filters(filt)
    elif isinstance(group_e_filters_raw, dict):
        filt_doc_id = group_e_filters_raw.get("doc_id")
        if filt_doc_id:
            group_e_filters_by_doc[filt_doc_id] = _build_selection_from_filters(group_e_filters_raw)
    group_e_active = "group_e" in experimental_groups and bool(group_e_filters_by_doc)
    all_multi_level_chunks = {}
    
    for doc_id in selected:
        doc = store.docs.get(doc_id)
        if not doc:
            continue
            
        # 優先使用已有的multi_level_chunks
        if doc and hasattr(doc, 'multi_level_chunks') and doc.multi_level_chunks:
            all_multi_level_chunks[doc_id] = doc.multi_level_chunks
            print(f"✅ 使用文檔 {doc.filename} 已有的multi_level_chunks")
        # 優先使用已有的structured_chunks，而不是重新從JSON生成
        elif doc and hasattr(doc, 'structured_chunks') and doc.structured_chunks:
            print(f"🔄 從structured_chunks轉換為multi_level_chunks，文檔: {doc.filename}")
            try:
                # 直接從structured_chunks轉換，而不是重新從JSON生成
                converted_chunks = convert_structured_to_multi_level(doc.structured_chunks)
                all_multi_level_chunks[doc_id] = converted_chunks
                # 保存到文檔
                doc.multi_level_chunks = converted_chunks
                store.add_doc(doc)
                store.save_data()
                print(f"✅ 成功轉換 {doc.filename} 的structured_chunks為multi_level_chunks")
            except Exception as e:
                print(f"⚠️ 從structured_chunks轉換失敗: {e}")
                # 如果轉換失敗，才回退到從JSON生成
                if hasattr(doc, 'json_data') and doc.json_data:
                    experimental_group = experimental_groups[0] if experimental_groups else 'group_d'
                    print(f"🔄 回退：基於JSON生成六個粒度級別格式，文檔: {doc.filename}，實驗組: {experimental_group}")
                    try:
                        from .chunking import MultiLevelStructuredChunking
                        ml_chunker = MultiLevelStructuredChunking()
                        raw_multi_level_list = ml_chunker.chunk_with_span(
                            doc.text, 
                            json_data=doc.json_data,
                            experimental_group=experimental_group
                        )
                        converted_chunks = convert_structured_to_multi_level(raw_multi_level_list)
                        all_multi_level_chunks[doc_id] = converted_chunks
                        doc.multi_level_chunks = converted_chunks
                        store.add_doc(doc)
                        store.save_data()
                    except Exception as e2:
                        print(f"❌ 基於JSON生成也失敗: {e2}")
        # 最後才考慮從JSON生成（通常不應該走到這裡）
        elif doc and hasattr(doc, 'json_data') and doc.json_data:
            experimental_group = experimental_groups[0] if experimental_groups else 'group_d'
            print(f"⚠️ 警告：文檔 {doc.filename} 沒有structured_chunks，將從JSON重新生成（可能產生不一致的結果）")
            try:
                from .chunking import MultiLevelStructuredChunking
                ml_chunker = MultiLevelStructuredChunking()
                raw_multi_level_list = ml_chunker.chunk_with_span(
                    doc.text, 
                    json_data=doc.json_data,
                    experimental_group=experimental_group
                )
                converted_chunks = convert_structured_to_multi_level(raw_multi_level_list)
                all_multi_level_chunks[doc_id] = converted_chunks
                doc.multi_level_chunks = converted_chunks
                doc.chunking_strategy = "structured_to_multi_level"
                store.add_doc(doc)
                store.save_data()
            except Exception as e:
                print(f"❌ 基於JSON生成多層級失敗: {e}")
    
    if not all_multi_level_chunks:
        return JSONResponse(
            status_code=400, 
            content={"error": "No multi-level chunks available. Please run structured hierarchical chunking or multi-level semantic chunking first."}
        )
    
    # 打印每個文檔的multi_level_chunks統計
    print(f"📊 收集到的multi_level_chunks統計:")
    for doc_id, multi_chunks in all_multi_level_chunks.items():
        doc = store.docs.get(doc_id)
        doc_name = doc.filename if doc else doc_id
        total_chunks = sum(len(chunks) for chunks in multi_chunks.values() if isinstance(chunks, list))
        level_counts = {level: len(chunks) for level, chunks in multi_chunks.items() if isinstance(chunks, list)}
        print(f"   文檔 {doc_name}: 總計 {total_chunks} 個chunks, 各層級: {level_counts}")
    
    # 論文中的六個層次
    six_levels = [
        'document',                    # 1. 文件層級
        'document_component',          # 2. 文件組成部分層級
        'basic_unit_hierarchy',        # 3. 基本單位層次結構層級
        'basic_unit',                  # 4. 基本單位層級
        'basic_unit_component',        # 5. 基本單位組成部分層級
        'enumeration'                  # 6. 列舉層級
    ]
    
    # 如果指定了實驗組，只處理相關層次
    if experimental_groups:
        print(f"🎯 收到實驗組選擇: {experimental_groups}")
        # 收集所有需要的層次
        required_levels = set()
        for group_key in experimental_groups:
            if group_key in GRANULARITY_COMBINATIONS:
                group_levels = GRANULARITY_COMBINATIONS[group_key]["levels"]
                print(f"   📋 {group_key}: {GRANULARITY_COMBINATIONS[group_key]['name']} -> 層次: {group_levels}")
                required_levels.update(group_levels)
            else:
                print(f"   ⚠️ 未知的實驗組: {group_key}")
        
        # 只處理需要的層次
        original_levels = six_levels.copy()
        six_levels = [level for level in six_levels if level in required_levels]
        print(f"🎯 實驗組模式：從 {len(original_levels)} 個層次中選擇 {len(six_levels)} 個層次")
        print(f"   原始層次: {original_levels}")
        print(f"   選中層次: {six_levels}")
        print(f"   跳過層次: {[level for level in original_levels if level not in required_levels]}")
    
    # 為每個層次創建獨立的embedding
    level_results = {}
    total_vectors = 0
    total_levels = len(six_levels)
    completed_levels = 0
    
    print(f"🚀 開始多層次embedding處理，共 {total_levels} 個層次")
    print(f"🎯 所有實驗組（A、B、C、D）統一使用 {EMBEDDING_DIMENSION} 維索引")
    print(f"📊 當前EMBEDDING_DIMENSION配置: {EMBEDDING_DIMENSION}")
    
    for level_idx, level_name in enumerate(six_levels):
        level_chunks = []
        level_doc_ids = []
        
        # 收集該層次的所有chunks
        for doc_id, multi_chunks in all_multi_level_chunks.items():
            if group_e_active and group_e_filters_by_doc and doc_id not in group_e_filters_by_doc:
                continue
            selection_filter = group_e_filters_by_doc.get(doc_id) if group_e_active else None
            if level_name in multi_chunks:
                for chunk_data in multi_chunks[level_name]:
                    if not (isinstance(chunk_data, dict) and 'content' in chunk_data):
                        continue
                    if selection_filter and level_name in GROUP_E_LEVELS:
                        metadata = (chunk_data.get("metadata") or {}).copy()
                        chunk_id = chunk_data.get("chunk_id")
                        if chunk_id and not metadata.get("chunk_id"):
                            metadata["chunk_id"] = chunk_id
                        if not _chunk_matches_selection(metadata, selection_filter):
                            continue
                    level_chunks.append(chunk_data['content'])
                    level_doc_ids.append(doc_id)
        
        if not level_chunks:
            print(f"⚠️ 層次 '{level_name}' 沒有可用的chunks")
            completed_levels += 1
            progress = (completed_levels / total_levels) * 100
            print(f"📊 進度: {completed_levels}/{total_levels} ({progress:.1f}%)")
            continue
        
        print(f"🔍 開始為層次 '{level_name}' 創建embedding，共 {len(level_chunks)} 個chunks")
        
            # 為該層次創建embedding
        try:
            print(f"⏳ 正在處理層次 '{level_name}' 的embedding...")
            if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
                vectors = await embed_gemini(level_chunks)
                provider = "gemini"
                model = "gemini-embedding-001"
            elif USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE:
                vectors = embed_bge_m3(level_chunks)
                provider = "bge-m3"
                model = "BAAI/bge-m3"
            else:
                print(f"❌ 層次 '{level_name}' embedding失敗：沒有可用的embedding方法")
                completed_levels += 1
                progress = (completed_levels / total_levels) * 100
                print(f"📊 進度: {completed_levels}/{total_levels} ({progress:.1f}%)")
                continue
            
            # 驗證向量維度
            dimension = len(vectors[0]) if vectors and len(vectors) > 0 else EMBEDDING_DIMENSION
            print(f"📊 層次 '{level_name}' embedding維度: {dimension} (配置: {EMBEDDING_DIMENSION})")
            
            # 驗證維度一致性（應該都是3072）
            if dimension != EMBEDDING_DIMENSION:
                print(f"⚠️ 警告：層次 '{level_name}' 的embedding維度({dimension})與配置({EMBEDDING_DIMENSION})不同")
                print(f"⚠️ 強制使用配置的維度 {EMBEDDING_DIMENSION}，請檢查embedding配置")
            
            # 檢查並清除舊的多層次索引（如果維度不匹配）
            if level_name in faiss_store.multi_level_index_info:
                old_dimension = faiss_store.multi_level_index_info[level_name].dimension
                if old_dimension != dimension:
                    print(f"⚠️ 檢測到層次 '{level_name}' 舊索引維度({old_dimension})與新embedding維度({dimension})不匹配，清除舊索引")
                    # 清除該層次的舊索引
                    if level_name in faiss_store.multi_level_indices:
                        del faiss_store.multi_level_indices[level_name]
                    if level_name in faiss_store.multi_level_index_info:
                        del faiss_store.multi_level_index_info[level_name]
                    if level_name in faiss_store.multi_level_chunk_ids:
                        faiss_store.multi_level_chunk_ids[level_name] = []
                    if level_name in faiss_store.multi_level_chunk_doc_ids:
                        faiss_store.multi_level_chunk_doc_ids[level_name] = []
                    if level_name in faiss_store.multi_level_chunks_flat:
                        faiss_store.multi_level_chunks_flat[level_name] = []
            
            # 存儲該層次的embedding和元數據
            metadata = {
                "provider": provider,
                "model": model,
                "dimension": dimension
            }
            store.set_multi_level_embeddings(level_name, vectors, level_chunks, level_doc_ids, metadata)
            
            # 新增：存儲到FAISS和BM25
            # 確保chunk_id包含層次信息，避免跨層次重複
            level_chunk_ids = [f"{level_name}_{doc_id}_{i}" for i, doc_id in enumerate(level_doc_ids)]
            faiss_store.add_multi_level_vectors(level_name, vectors, level_chunk_ids, level_doc_ids, level_chunks)
            bm25_index.build_multi_level_index(level_name, level_chunks, level_chunk_ids, level_doc_ids)
            
            # 新增：批量增強該層次的metadata（可選）
            level_enhanced_metadata = {}
            if req.get("enable_metadata_enhancement", True):
                print(f"🔧 開始增強層次 '{level_name}' 的metadata...")
                level_chunks_data = [
                    {
                        "chunk_id": level_chunk_ids[i],
                        "content": level_chunks[i],
                        "metadata": {}
                    }
                    for i in range(len(level_chunks))
                ]
                level_enhanced_metadata = metadata_enhancer.enhance_metadata_batch(level_chunks_data)
                
                # 設置增強metadata到FAISS存儲
                for chunk_id, enhanced_metadata in level_enhanced_metadata.items():
                    faiss_store.set_multi_level_enhanced_metadata(level_name, chunk_id, enhanced_metadata)
            else:
                print(f"⚠️ 跳過層次 '{level_name}' 的metadata增強")
            
            level_results[level_name] = {
                "provider": provider,
                "model": model,
                "num_vectors": len(vectors),
                "dimension": len(vectors[0]) if vectors else 0,
                "num_chunks": len(level_chunks),
                "level_description": get_level_description(level_name)
            }
            
            total_vectors += len(vectors)
            completed_levels += 1
            progress = (completed_levels / total_levels) * 100
            print(f"✅ 層次 '{level_name}' embedding完成：{len(vectors)} 個向量")
            print(f"📊 進度: {completed_levels}/{total_levels} ({progress:.1f}%)")
            
        except Exception as e:
            print(f"❌ 層次 '{level_name}' embedding失敗：{e}")
            # 使用隨機向量作為fallback
            try:
                import numpy as np
                fallback_vectors = np.random.randn(len(level_chunks), EMBEDDING_DIMENSION).astype(np.float32).tolist()
                metadata = {
                    "provider": "fallback_random",
                    "model": f"random_{EMBEDDING_DIMENSION}d",
                    "dimension": EMBEDDING_DIMENSION
                }
                store.set_multi_level_embeddings(level_name, fallback_vectors, level_chunks, level_doc_ids, metadata)
                
                level_results[level_name] = {
                    "provider": "fallback_random",
                    "model": f"random_{EMBEDDING_DIMENSION}d",
                    "num_vectors": len(fallback_vectors),
                    "dimension": EMBEDDING_DIMENSION,
                    "num_chunks": len(level_chunks),
                    "level_description": get_level_description(level_name),
                    "error": f"Original embedding failed, using fallback: {str(e)}"
                }
                
                total_vectors += len(fallback_vectors)
                print(f"⚠️ 層次 '{level_name}' 使用fallback向量：{len(fallback_vectors)} 個")
                
            except Exception as fallback_error:
                print(f"❌ 層次 '{level_name}' fallback也失敗：{fallback_error}")
                level_results[level_name] = {
                    "error": f"Both original and fallback failed: {str(e)} | {str(fallback_error)}",
                    "num_chunks": len(level_chunks),
                    "level_description": get_level_description(level_name)
                }
            
            completed_levels += 1
            progress = (completed_levels / total_levels) * 100
            print(f"📊 進度: {completed_levels}/{total_levels} ({progress:.1f}%)")
    
    print(f"🎉 多層次embedding處理完成！總共處理了 {total_vectors} 個向量，成功完成 {completed_levels}/{total_levels} 個層次")
    
    # 確保多層次embedding狀態正確設置
    print(f"🎉 多層次embedding完成，保存的層次: {list(store.multi_level_embeddings.keys())}")
    print(f"🔍 store.has_multi_level_embeddings(): {store.has_multi_level_embeddings()}")
    print(f"🔍 可用層次: {store.get_available_levels()}")
    
    # 如果指定了實驗組，保存到實驗組存儲中（支持多個實驗組同時存在）
    if experimental_groups and len(experimental_groups) == 1:
        group_id = experimental_groups[0]
        print(f"💾 準備保存實驗組 {group_id} 的embedding數據...")
        print(f"   當前multi_level_embeddings層次: {list(store.multi_level_embeddings.keys())}")
        print(f"   當前已保存的實驗組: {store.list_experimental_groups()}")
        
        store.save_experimental_group_embeddings(group_id)
        
        # 驗證保存是否成功
        if group_id in store.experimental_group_embeddings:
            saved_levels = list(store.experimental_group_embeddings[group_id].get("multi_level_embeddings", {}).keys())
            print(f"✅ 已將實驗組 {group_id} 的embedding保存到獨立存儲，包含層次: {saved_levels}")
        else:
            print(f"❌ 警告：實驗組 {group_id} 保存後未在存儲中找到！")
    
    # 自動保存多層次embedding數據
    store.save_data()
    faiss_store.save_data()
    bm25_index.save_data()
    
    # 如果這是A組（僅basic_unit），也創建標準embedding以保持兼容性
    if experimental_groups and len(experimental_groups) == 1 and experimental_groups[0] == "group_a":
        if "basic_unit" in store.multi_level_embeddings:
            # 使用 get_multi_level_embeddings 方法獲取數據（返回dict格式）
            basic_unit_data = store.get_multi_level_embeddings("basic_unit")
            if basic_unit_data:
                store.embeddings = basic_unit_data.get('embeddings', [])
                store.chunk_doc_ids = basic_unit_data.get('doc_ids', [])
                store.chunks_flat = basic_unit_data.get('chunks', [])
                print(f"🔄 A組：同步創建標準embedding，向量數量: {len(store.embeddings) if store.embeddings else 0}")
                store.save_data()
            else:
                # 如果 get_multi_level_embeddings 返回 None，直接從存儲中獲取
                store.embeddings = store.multi_level_embeddings.get("basic_unit", [])
                store.chunk_doc_ids = store.multi_level_chunk_doc_ids.get("basic_unit", [])
                store.chunks_flat = store.multi_level_chunks_flat.get("basic_unit", [])
                print(f"🔄 A組：同步創建標準embedding（直接獲取），向量數量: {len(store.embeddings) if store.embeddings else 0}")
                store.save_data()
    
    if not level_results:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to create embeddings for any level"}
        )
    
    # 如果指定了實驗組，計算各組的embedding狀態
    group_results = {}
    if experimental_groups:
        for group_key in experimental_groups:
            if group_key in GRANULARITY_COMBINATIONS:
                combination = GRANULARITY_COMBINATIONS[group_key]
                group_levels = combination["levels"]
                
                group_results[group_key] = {
                    "name": combination["name"],
                    "levels": group_levels,
                    "embedding_status": {},
                    "total_chunks": 0
                }
                
                for level in group_levels:
                    if level in level_results:
                        group_results[group_key]["embedding_status"][level] = "completed"
                        group_results[group_key]["total_chunks"] += level_results[level]["num_chunks"]
                    else:
                        group_results[group_key]["embedding_status"][level] = "missing"

    return {
        "message": "Six-level embeddings created successfully",
        "total_vectors": total_vectors,
        "levels": level_results,
        "available_levels": list(level_results.keys()),
        "level_descriptions": {
            level: get_level_description(level) for level in six_levels
        },
        "experimental_groups": group_results if experimental_groups else None,
        "faiss_available": True,
        "bm25_available": True,
        "enhanced_metadata_available": True
    }


def get_level_description(level_name: str) -> str:
    """獲取層次描述"""
    descriptions = {
        'document': '文件層級 (Document Level) - 整個法律文檔',
        'document_component': '文件組成部分層級 (Document Component Level) - 文檔的主要組成部分',
        'basic_unit_hierarchy': '基本單位層次結構層級 (Basic Unit Hierarchy Level) - 書籍、標題、章節',
        'basic_unit': '基本單位層級 (Basic Unit Level) - 文章/條文 (article)',
        'basic_unit_component': '基本單位組成部分層級 (Basic Unit Component Level) - 強制性主文或段落',
        'enumeration': '列舉層級 (Enumeration Level) - 項目、子項'
    }
    return descriptions.get(level_name, f"未知層次: {level_name}")


def generate_hierarchical_description_from_metadata(doc_id: str, metadata: Dict[str, Any], content: str, store) -> str:
    """
    從metadata直接生成層級描述（用於多層級檢索）
    
    Args:
        doc_id: 文檔ID
        metadata: chunk的metadata
        content: chunk內容
        store: 存儲實例
    
    Returns:
        層級描述字符串
    """
    try:
        # 獲取文檔信息
        doc = store.get_doc(doc_id)
        if not doc:
            return f"doc={doc_id}"
        
        # 獲取文檔名稱（去除.json後綴）
        doc_name = doc.filename
        if doc_name.endswith('.json'):
            doc_name = doc_name[:-5]
        
        # 優先從metadata中獲取法律名稱
        law_name = ""
        if metadata.get('id'):
            law_name = extract_law_name_from_metadata_id(metadata['id'])
        
        if not law_name and metadata.get('law_name'):
            law_name = metadata['law_name'].replace('法規名稱：', '')
        
        if not law_name:
            law_name = extract_law_name_from_content(content)
        
        if not law_name:
            law_name = doc_name
        
        # 構建層級描述
        hierarchy_parts = [law_name]
        
        # 添加章節信息
        if metadata.get('chapter'):
            chapter = metadata['chapter']
            if chapter != "未分類章" and chapter != "未分類節":
                if not chapter.startswith('第') and not chapter.startswith('章'):
                    chapter = f"第{chapter}章"
                hierarchy_parts.append(chapter)
        
        # 添加節信息
        if metadata.get('section'):
            section = metadata['section']
            if section != "未分類節":
                if not section.startswith('第') and not section.startswith('節'):
                    section = f"第{section}節"
                hierarchy_parts.append(section)
        
        # 從內容中提取正確的條文號碼，優先使用內容中的信息
        article_number = extract_article_number_from_content(content)
        if article_number:
            hierarchy_parts.append(article_number)
        elif metadata.get('article'):
            article = metadata['article']
            if not article.startswith('第') and not article.startswith('條'):
                article = f"第{article}條"
            hierarchy_parts.append(article)
        
        # 添加項信息
        if metadata.get('items') and len(metadata['items']) > 0:
            items = metadata['items']
            if len(items) == 1:
                hierarchy_parts.append(f"第{items[0]}項")
            else:
                hierarchy_parts.append(f"第{items[0]}-{items[-1]}項")
        
        return ' '.join(hierarchy_parts)
        
    except Exception as e:
        print(f"❌ 從metadata生成層級描述失敗: {e}")
        return f"doc={doc_id}"


def generate_hierarchical_description(doc_id: str, level: str, chunk_index: int, store) -> str:
    """
    生成層級描述，例如：著作權法 第三章 第一節 第11條
    
    Args:
        doc_id: 文檔ID
        level: 層級名稱
        chunk_index: chunk索引
        store: 存儲實例
    
    Returns:
        層級描述字符串
    """
    try:
        # 獲取文檔信息
        doc = store.get_doc(doc_id)
        if not doc:
            return f"doc={doc_id}"
        
        # 獲取文檔名稱（去除.json後綴）
        doc_name = doc.filename
        if doc_name.endswith('.json'):
            doc_name = doc_name[:-5]
        
        # 如果是多層次embedding，嘗試從原始文檔的structured_chunks中獲取層級信息
        if hasattr(doc, 'structured_chunks') and doc.structured_chunks and chunk_index < len(doc.structured_chunks):
            structured_chunk = doc.structured_chunks[chunk_index]
            metadata = structured_chunk.get('metadata', {})
            content = structured_chunk.get('content', '')
            
            # 優先從metadata中獲取法律名稱，這是最可靠的來源
            law_name = ""
            if metadata.get('id'):
                # 從metadata id中提取法律名稱
                law_name = extract_law_name_from_metadata_id(metadata['id'])
            
            # 如果metadata中有law_name字段，也嘗試使用它
            if not law_name and metadata.get('law_name'):
                law_name = metadata['law_name']
                # 清理可能存在的"法規名稱："前綴
                law_name = law_name.replace('法規名稱：', '')
            
            # 如果metadata中沒有，再嘗試從內容中提取
            if not law_name:
                law_name = extract_law_name_from_content(content)
            
            # 最後使用文檔名稱
            if not law_name:
                law_name = doc_name
            
            # 構建層級描述
            hierarchy_parts = [law_name]
            
            # 添加章節信息
            if metadata.get('chapter'):
                chapter = metadata['chapter']
                # 清理章節格式
                if chapter != "未分類節":
                    if not chapter.startswith('第') and not chapter.startswith('章'):
                        chapter = f"第{chapter}章"
                    hierarchy_parts.append(chapter)
            
            # 添加節信息
            if metadata.get('section'):
                section = metadata['section']
                # 清理節格式
                if section != "未分類節":
                    if not section.startswith('第') and not section.startswith('節'):
                        section = f"第{section}節"
                    hierarchy_parts.append(section)
            
            # 從內容中提取正確的條文號碼，優先使用內容中的信息
            article_number = extract_article_number_from_content(content)
            if article_number:
                hierarchy_parts.append(article_number)
            elif metadata.get('article'):
                article = metadata['article']
                if not article.startswith('第') and not article.startswith('條'):
                    article = f"第{article}條"
                hierarchy_parts.append(article)
            
            # 添加項信息
            if metadata.get('items') and len(metadata['items']) > 0:
                items = metadata['items']
                if len(items) == 1:
                    hierarchy_parts.append(f"第{items[0]}項")
                else:
                    hierarchy_parts.append(f"第{items[0]}-{items[-1]}項")
            
            return ' '.join(hierarchy_parts)
        
        # 如果沒有結構化信息，根據層級名稱生成基本描述
        level_descriptions = {
            'document': f"{doc_name} 全文",
            'document_component': f"{doc_name} 章節",
            'basic_unit_hierarchy': f"{doc_name} 條文層次",
            'basic_unit': f"{doc_name} 條文",
            'basic_unit_component': f"{doc_name} 條文組成",
            'enumeration': f"{doc_name} 列舉項"
        }
        
        return level_descriptions.get(level, f"{doc_name} {level}")
        
    except Exception as e:
        print(f"❌ 生成層級描述失敗: {e}")
        return f"doc={doc_id}"


def extract_law_name_from_content(content: str) -> str:
    """從內容中提取法律名稱"""
    import re
    
    # 匹配【法律名稱】格式
    law_pattern = r'【([^】]+)】'
    match = re.search(law_pattern, content)
    if match:
        return match.group(1)
    
    # 如果沒有找到【】格式，嘗試其他模式
    law_patterns = [
        r'^([^第章節條項]+法)',
        r'([^第章節條項]+法)',
    ]
    
    for pattern in law_patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1).strip()
    
    return ""


def extract_law_name_from_metadata_id(metadata_id: str) -> str:
    """從metadata ID中提取法律名稱"""
    import re
    
    # 嘗試兩種格式：
    # 1. 原始格式：法規名稱：商標法_0_第一章_總則_未分類節_第1條
    # 2. 清理後格式：商標法_0_第一章_總則_未分類節_第1條
    
    # 首先嘗試原始格式
    law_pattern = r'法規名稱：([^_]+)'
    match = re.search(law_pattern, metadata_id)
    if match:
        return match.group(1)
    
    # 如果沒有找到，嘗試清理後的格式（第一個部分就是法規名稱）
    parts = metadata_id.split('_')
    if parts and parts[0]:
        # 檢查第一個部分是否包含"法"字，確認它是法規名稱
        first_part = parts[0].strip()
        if '法' in first_part or '條例' in first_part:
            return first_part
    
    return ""


def extract_article_number_from_content(content: str) -> str:
    """從內容中提取條文號碼"""
    import re
    
    # 按行分割內容，尋找條文號碼
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 匹配各種條文號碼格式
        article_patterns = [
            r'第(\d+條)',           # 第43條
            r'第([一二三四五六七八九十百千]+)條',  # 第四十三條
            r'第(\d+-\d+條)',      # 第43-1條
            r'第(\d+[之-]\d+條)',   # 第43之1條
        ]
        
        for pattern in article_patterns:
            match = re.search(pattern, line)
            if match:
                # 檢查這行是否看起來像條文標題（通常比較簡短，不包含太多內容）
                if len(line) < 50:  # 條文標題通常比較短
                    return f"第{match.group(1)}"
    
    # 如果沒有找到簡短的條文標題，嘗試在整個內容中找第一個條文號碼
    article_patterns = [
        r'第(\d+條)',
        r'第([一二三四五六七八九十百千]+)條',
        r'第(\d+-\d+條)',
        r'第(\d+[之-]\d+條)',
    ]
    
    for pattern in article_patterns:
        match = re.search(pattern, content)
        if match:
            return f"第{match.group(1)}"
    
    return ""


def rank_with_dense_vectors(query: str, k: int):
    """使用密集向量進行相似度計算（支持 Gemini 和 BGE-M3，優先使用FAISS，支持多層級索引）"""
    import numpy as np
    
    # 生成查詢向量
    query_vector = None
    if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
        try:
            query_vector = asyncio_run(embed_gemini([query]))[0]
        except Exception as e:
            print(f"Gemini query embedding failed: {e}")
            if USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE:
                query_vector = embed_bge_m3([query])[0]
            else:
                raise RuntimeError("Both Gemini and BGE-M3 query embedding failed")
    elif USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE:
        query_vector = embed_bge_m3([query])[0]
    else:
        raise RuntimeError("No dense embedding method available")
    
    if query_vector is None:
        raise RuntimeError("Failed to generate query vector")
    
    # 優先使用多層級FAISS索引
    if faiss_store.has_multi_level_vectors():
        try:
            available_levels = faiss_store.get_available_levels()
            if available_levels:
                # 計算每個層級在合併列表中的起始偏移量（與generate函數中的合併邏輯保持一致）
                level_offsets = {}
                current_offset = 0
                for level_name in available_levels:
                    level_offsets[level_name] = current_offset
                    level_chunks_count = len(faiss_store.multi_level_chunks_flat.get(level_name, []))
                    current_offset += level_chunks_count
                
                # 合併所有層級的搜索結果
                all_indices = []
                all_scores = []
                
                for level_name in available_levels:
                    level_indices, level_scores = faiss_store.search_multi_level(level_name, query_vector, k * 2)
                    # 將層級內索引轉換為合併列表中的全局索引
                    level_offset = level_offsets[level_name]
                    adjusted_indices = [idx + level_offset for idx in level_indices]
                    all_indices.extend(adjusted_indices)
                    all_scores.extend(level_scores)
                
                # 按分數排序並取前k個
                if all_scores:
                    sorted_pairs = sorted(zip(all_indices, all_scores), key=lambda x: x[1], reverse=True)
                    top_k_pairs = sorted_pairs[:k]
                    result_indices = [idx for idx, _ in top_k_pairs]
                    result_scores = [score for _, score in top_k_pairs]
                    print(f"✅ 多層級向量檢索完成，返回 {len(result_indices)} 個結果")
                    return result_indices, result_scores
        except Exception as e:
            print(f"多層級FAISS搜索失敗，嘗試標準FAISS: {e}")
    
    # 使用標準FAISS索引
    if faiss_store.has_vectors():
        try:
            # FAISS搜索
            indices, scores = faiss_store.search(query_vector, k)
            return indices, scores
            
        except Exception as e:
            print(f"FAISS search failed, falling back to NumPy: {e}")
    
    # 回退到NumPy（保持向後兼容）
    if store.embeddings is None:
        raise ValueError("No embeddings available")
    if isinstance(store.embeddings, list):
        vecs = np.array(store.embeddings, dtype=float)
    else:
        vecs = np.array(store.embeddings, dtype=float)  # type: ignore[assignment]
    
    # 根據當前配置選擇查詢向量化方法
    if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
        try:
            qvec = np.array(asyncio_run(embed_gemini([query]))[0], dtype=float)
        except Exception as e:
            print(f"Gemini query embedding failed: {e}")
            # 如果 Gemini 失敗，嘗試 BGE-M3
            if USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    qvec = np.array(embed_bge_m3([query])[0], dtype=float)
                except Exception as e2:
                    print(f"BGE-M3 query embedding failed: {e2}")
                    raise RuntimeError("Both Gemini and BGE-M3 query embedding failed")
            else:
                raise RuntimeError("Gemini query embedding failed and BGE-M3 not available")
    elif USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            qvec = np.array(embed_bge_m3([query])[0], dtype=float)
        except Exception as e:
            print(f"BGE-M3 query embedding failed: {e}")
            raise RuntimeError("BGE-M3 query embedding failed")
    else:
        raise RuntimeError("No dense embedding method available")
    
    # normalize
    vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    q_norm = qvec / (np.linalg.norm(qvec) + 1e-8)
    sims = vecs_norm @ q_norm
    idxs = np.argsort(-sims)[:k]
    return idxs.tolist(), sims[idxs].tolist()


def calculate_query_qa_similarity(query: str, qa_query: str) -> float:
    """計算查詢與QA的相似度"""
    import re
    
    query_lower = query.lower().strip()
    qa_lower = qa_query.lower().strip()
    
    # 方法1: 完全匹配
    if query_lower == qa_lower:
        return 1.0
    
    # 方法2: 包含匹配
    if query_lower in qa_lower or qa_lower in query_lower:
        return 0.9
    
    # 方法3: 法條號碼匹配
    query_article_match = re.search(r'第\s*(\d+(?:之\d+)?)\s*條', query_lower)
    qa_article_match = re.search(r'第\s*(\d+(?:之\d+)?)\s*條', qa_lower)
    
    if query_article_match and qa_article_match:
        query_article = query_article_match.group(1)
        qa_article = qa_article_match.group(1)
        if query_article == qa_article:
            return 0.8
    
    # 方法4: 法律名稱匹配
    law_names = ["著作權法", "商標法", "專利法", "民法", "刑法"]
    query_laws = [law for law in law_names if law in query_lower]
    qa_laws = [law for law in law_names if law in qa_lower]
    
    if query_laws and qa_laws and any(law in qa_laws for law in query_laws):
        return 0.7
    
    # 方法5: 關鍵詞重疊
    query_words = set(re.findall(r'\w+', query_lower))
    qa_words = set(re.findall(r'\w+', qa_lower))
    
    if query_words and qa_words:
        overlap = len(query_words.intersection(qa_words))
        union = len(query_words.union(qa_words))
        jaccard_similarity = overlap / union if union > 0 else 0
        return jaccard_similarity * 0.6  # 降低權重
    
    return 0.0


def expand_query_with_legal_domain(query: str) -> Dict[str, Any]:
    """使用領域專屬詞庫進行查詢擴展"""
    
    # 如果有法律推理引擎，優先使用
    if legal_reasoning_engine:
        try:
            analysis = legal_reasoning_engine.analyze_query(query)
            expanded_query = legal_reasoning_engine.get_expanded_query(query)
            
            return {
                "original_query": query,
                "expanded_query": expanded_query,
                "expansion_ratio": len(expanded_query.split()) / len(query.split()),
                "domain_matches": analysis["concept_mappings"],
                "detected_domains": ["copyright"] if analysis["detected_concepts"] else [],
                "applicable_articles": analysis["applicable_articles"],
                "reasoning_explanation": analysis["reasoning_explanation"],
                "confidence_scores": analysis["confidence_scores"],
                "reasoning_engine_used": True
            }
        except Exception as e:
            print(f"⚠️ 法律推理引擎執行失敗: {e}")
            # 回退到原有方法
    
    # 領域專屬詞庫 - 法律概念映射
    legal_domain_dict = {
        # 著作權法專屬詞彙
        "copyright": {
                   "核心概念": {
                       "重製": ["複製", "抄襲", "盜版", "翻印", "影印", "掃描", "下載", "保存", "直接複製"],
                       "改作": ["改寫", "改編", "修改", "衍生", "創作", "重新創作", "二次創作", "用自己的語氣", "改寫成自己的語氣", "改寫成", "語氣", "翻譯", "譯", "轉譯", "中譯", "英譯", "日譯"],
                       "散布": ["分享", "傳播", "發布", "上傳", "轉載", "轉發", "傳送"],
                "公開傳輸": ["上網", "網路傳播", "線上分享", "串流", "直播"],
                "公開演出": ["表演", "演奏", "演唱", "播放", "放映"],
                "公開展示": ["展覽", "展示", "陳列", "展出"],
                "出租": ["租借", "租賃", "出借"],
                "侵害": ["違反", "侵犯", "損害", "違法", "不法"],
                       "合理使用": ["引用", "評論", "教學", "研究", "報導", "學術"],
                       "教育用途": ["課堂", "學校", "教育", "教學", "授課", "學生", "播放", "影片", "youtube", "影片"],
                       "著作財產權": ["版權", "財產權", "經濟權利"],
                       "著作人格權": ["署名權", "完整性權", "名譽權"],
                "公開發表": ["發表", "出版", "公開", "發布"],
                "創作": ["製作", "產生", "完成", "寫作", "繪製"],
                "著作": ["作品", "創作", "藝術品", "文學", "音樂", "美術", "攝影"],
                "著作人": ["作者", "創作者", "藝術家", "作家"],
            },
            "法律條文": {
                "第3條": ["定義", "概念", "解釋", "說明", "何謂"],
                "第10條": ["著作權取得", "完成時", "享有", "產生"],
                "第22條": ["重製權", "複製權"],
                "第26條": ["公開演出權"],
                "第26-1條": ["公開傳輸權"],
                "第28條": ["改作權", "衍生著作", "翻譯", "譯", "轉譯", "中譯", "英譯", "日譯", "改作", "改編", "修改", "衍生", "創作", "重新創作", "二次創作"],
                "第28-1條": ["散布權"],
                "第29條": ["出租權"],
                       "第44條": ["司法程序", "重製"],
                       "第46條": ["學校", "授課", "教學", "重製"],
                       "第47條": ["教育", "學校", "公開播送", "公開傳輸"],
                       "第65條": ["合理使用", "例外", "限制"],
                       "第87條": ["視為侵害", "禁止行為"],
                       "第91條": ["重製罪", "刑罰", "罰金"],
            }
        },
        
        # 商標法專屬詞彙
        "trademark": {
            "核心概念": {
                "商標": ["標誌", "標識", "品牌", "商號", "logo", "標記"],
                "註冊": ["申請", "登記", "核准", "取得"],
                "仿冒": ["假冒", "偽造", "仿製", "山寨", "盜用"],
                "混淆": ["相似", "近似", "誤認", "混同"],
                "使用": ["使用", "經營", "銷售", "廣告"],
                "專用權": ["獨占", "排他", "專有"],
                "侵害": ["侵權", "違反", "損害"],
            },
            "法律條文": {
                "第2條": ["定義", "商標", "服務標章"],
                "第5條": ["註冊", "申請", "核准"],
                "第29條": ["近似", "混淆", "類似"],
                "第68條": ["侵害", "侵權", "禁止"],
                "第95條": ["刑罰", "仿冒罪"],
            }
        },
        
        # 專利法專屬詞彙
        "patent": {
            "核心概念": {
                "發明": ["創新", "技術", "改良", "設計"],
                "專利": ["專利權", "獨占權"],
                "新穎性": ["新", "未公開", "首創"],
                "進步性": ["非顯而易見", "技術進步"],
                "產業利用性": ["實用", "可行", "製造"],
                "申請": ["提出", "提交", "申報"],
                "核准": ["通過", "授權", "公告"],
            }
        }
    }
    
    # 查詢擴展邏輯
    expanded_terms = set()
    # 改進查詢分割，處理中文和標點符號
    import re
    # 移除標點符號，然後分割
    cleaned_query = re.sub(r'[，。？！、；：？]', ' ', query.lower())
    # 使用空格和標點符號分割
    original_terms = set(re.split(r'[\s，。？！、；：？]+', cleaned_query))
    # 移除空字符串
    original_terms = {term for term in original_terms if term.strip()}
    domain_matches = []
    
    # 1. 識別查詢領域
    detected_domains = []
    if any(term in query for term in ["著作權", "版權", "著作", "創作", "重製", "改作", "課堂", "教育", "教學", "學校", "播放", "影片", "youtube", "授權"]):
        detected_domains.append("copyright")
    if any(term in query for term in ["商標", "品牌", "標誌", "仿冒"]):
        detected_domains.append("trademark")
    if any(term in query for term in ["專利", "發明", "技術", "創新"]):
        detected_domains.append("patent")
    
    # 2. 查詢擴展
    for domain in detected_domains:
        if domain in legal_domain_dict:
            domain_data = legal_domain_dict[domain]
            
            # 核心概念擴展
            for legal_concept, synonyms in domain_data["核心概念"].items():
                # 直接檢查查詢中是否包含同義詞
                for synonym in synonyms:
                    if synonym in query:
                        expanded_terms.update(synonyms)
                        expanded_terms.add(legal_concept)
                        domain_matches.append(f"{synonym}→{legal_concept}")
                # 也檢查查詢中是否包含概念本身
                if legal_concept in query:
                    expanded_terms.update(synonyms)
                    expanded_terms.add(legal_concept)
                    domain_matches.append(f"查詢→{legal_concept}")
            
            # 法律條文擴展
            for article, keywords in domain_data["法律條文"].items():
                for term in original_terms:
                    if term in keywords:
                        expanded_terms.update(keywords)
                        expanded_terms.add(article)
                        domain_matches.append(f"{term}→{article}")
                # 也檢查查詢中是否包含條文關鍵字
                for keyword in keywords:
                    if keyword in query:
                        expanded_terms.update(keywords)
                        expanded_terms.add(article)
                        domain_matches.append(f"{keyword}→{article}")
    
    # 3. 生成擴展查詢
    expanded_query_terms = list(original_terms.union(expanded_terms))
    expanded_query = " ".join(expanded_query_terms)
    
    return {
        "original_query": query,
        "expanded_query": expanded_query,
        "detected_domains": detected_domains,
        "expanded_terms": list(expanded_terms),
        "domain_matches": domain_matches,
        "expansion_ratio": len(expanded_terms) / len(original_terms) if original_terms else 0
    }


def detect_content_hierarchy(content: str) -> str:
    """基於內容分析檢測層次級別"""
    import re
    
    # 檢測法條級別（包含"第X條"）
    if re.search(r'第\s*\d+\s*條', content):
        return "article"
    
    # 檢測節級別（包含"第X節"或"第X章"）
    if re.search(r'第\s*\d+\s*[節章]', content):
        return "section"
    
    # 檢測章級別（包含"第X章"或"總則"、"附則"等）
    if re.search(r'第\s*\d+\s*章|總則|附則', content):
        return "chapter"
    
    # 檢測是否為具體法律條文內容
    if re.search(r'條文|規定|權利|義務|禁止|處罰', content):
        return "article"
    
    # 默認為一般內容
    return "general"


def calculate_hierarchical_relevance(query: str, result: Dict) -> Dict[str, Any]:
    """計算層次化相關性分數 - 基於論文的Aboutness概念和內容分析"""
    content = result.get("content", "")
    metadata = result.get("metadata", {})
    
    # 基於內容分析檢測層次級別（備用方案）
    content_hierarchy = detect_content_hierarchy(content)
    
    # 提取層次信息（優先使用metadata，備用內容分析）
    hierarchy_level = "article"  # 默認層級
    if metadata and metadata.get("article"):
        hierarchy_level = "article"
    elif metadata and metadata.get("section"):
        hierarchy_level = "section"
    elif metadata and metadata.get("chapter"):
        hierarchy_level = "chapter"
    else:
        # 使用內容分析結果
        hierarchy_level = content_hierarchy
    
    # Aboutness分析 - 識別文本的主要主題
    aboutness_score = 0.0
    aboutness_keywords = []
    
    # 法律概念aboutness
    legal_concepts = ["著作權", "版權", "侵權", "重製", "改作", "散布", "合理使用", "授權", "商標", "專利"]
    for concept in legal_concepts:
        if concept in content:
            aboutness_score += 1.0
            aboutness_keywords.append(concept)
    
    # 結構層級權重（基於論文的多層次方法）
    hierarchy_weights = {
        "article": 1.0,    # 法條級別 - 最高精度
        "section": 0.8,    # 節級別 - 中等精度
        "chapter": 0.6,    # 章級別 - 較低精度但廣度更大
        "general": 0.4     # 一般內容 - 最低權重
    }
    
    hierarchy_weight = hierarchy_weights.get(hierarchy_level, 1.0)
    
    return {
        "aboutness_score": aboutness_score,
        "aboutness_keywords": aboutness_keywords,
        "hierarchy_level": hierarchy_level,
        "hierarchy_weight": hierarchy_weight,
        "content_hierarchy": content_hierarchy  # 內容分析的結果
    }


def calculate_retrieval_metrics(query: str, results: List[Dict], k: int) -> Dict[str, float]:
    """計算檢索指標 P@K 和 R@K - 整合查詢擴展、智能相關性判斷和多層次檢索"""
    try:
        print(f"🔍 開始計算檢索指標，查詢: '{query}', k={k}")
        
        if not results:
            print("❌ 沒有檢索結果")
            return {"p_at_k": 0.0, "r_at_k": 0.0, "note": "No retrieval results"}
        
        # 1. 查詢擴展處理
        query_expansion = expand_query_with_legal_domain(query)
        expanded_query = query_expansion["expanded_query"]
        detected_domains = query_expansion["detected_domains"]
        domain_matches = query_expansion["domain_matches"]
        
        print(f"🔍 查詢擴展: 原查詢='{query}'")
        print(f"🔍 擴展查詢: '{expanded_query}'")
        print(f"🔍 檢測領域: {detected_domains}")
        print(f"🔍 領域映射: {domain_matches[:5]}...")  # 只顯示前5個
        
        # 基於查詢內容和檢索結果計算相關性
        relevant_chunks = []
        query_lower = query.lower()
        expanded_query_lower = expanded_query.lower()
        
        # 提取查詢中的關鍵信息
        import re
        
        # 提取法條號碼
        article_patterns = [
            r'第\s*(\d+)\s*條',
            r'條\s*(\d+)',
            r'article\s*(\d+)',
        ]
        
        article_numbers = []
        for pattern in article_patterns:
            matches = re.findall(pattern, query)
            article_numbers.extend([int(m) for m in matches])
        
        # 提取法律名稱
        law_keywords = []
        law_patterns = ['著作權法', '商標法', '專利法', '民法', '刑法']
        for law in law_patterns:
            if law in query:
                law_keywords.append(law)
        
        # 檢測查詢類型
        has_explicit_article = len(article_numbers) > 0
        query_type = "explicit_article" if has_explicit_article else "semantic_query"
        
        print(f"📋 查詢分析: 類型={query_type}, 法條號碼={article_numbers}, 法律關鍵字={law_keywords}")
        
        # 判斷每個檢索結果的相關性（整合查詢擴展和多層次檢索）
        for i, result in enumerate(results):
            content = result.get("content", "")
            content_lower = content.lower()
            
            relevance_score = 0
            relevance_reasons = []
            
            # 計算層次化相關性（基於論文的多層次方法）
            hierarchical_analysis = calculate_hierarchical_relevance(query, result)
            aboutness_score = hierarchical_analysis["aboutness_score"]
            hierarchy_weight = hierarchical_analysis["hierarchy_weight"]
            hierarchy_level = hierarchical_analysis["hierarchy_level"]
            
            # 層次化相關性加分
            if aboutness_score > 0:
                relevance_score += aboutness_score * hierarchy_weight * 0.5  # 適度權重
                relevance_reasons.append(f"層次化aboutness({hierarchy_level}):{aboutness_score:.1f}")
            
            # 1. 法條號碼匹配（權重最高，僅適用於明確法條查詢）
            if has_explicit_article:
                for article_num in article_numbers:
                    if f'第{article_num}條' in content or f'第 {article_num} 條' in content:
                        relevance_score += 4  # 提高權重
                        relevance_reasons.append(f"精確匹配法條{article_num}")
                        break
            
            # 2. 法律名稱匹配
            for law in law_keywords:
                if law in content:
                    relevance_score += 2
                    relevance_reasons.append(f"匹配法律{law}")
            
            # 3. 查詢擴展匹配（新增）
            expanded_words = set(expanded_query_lower.split())
            content_words = set(content_lower.split())
            expanded_matches = expanded_words.intersection(content_words)
            
            if len(expanded_matches) > 0:
                # 計算擴展匹配的權重
                expansion_weight = min(len(expanded_matches) * 0.8, 3.0)  # 最多3分
                relevance_score += expansion_weight
                relevance_reasons.append(f"擴展匹配{len(expanded_matches)}個詞:{list(expanded_matches)[:3]}")
            
            # 4. 領域專屬概念匹配（新增）- 改進邏輯
            for domain_match in domain_matches[:5]:  # 限制匹配數量
                concept = domain_match.split("→")[-1]
                if concept in content:
                    # 對於"改作"概念，給予更高權重
                    if concept == "改作":
                        relevance_score += 2.5  # 高權重
                        relevance_reasons.append(f"核心概念:{concept}")
                    else:
                        relevance_score += 1.5
                        relevance_reasons.append(f"領域概念:{concept}")
            
            # 5. 直接概念匹配（新增）
            if "改作" in content and ("改寫" in query_lower or "語氣" in query_lower):
                relevance_score += 3.0  # 最高權重
                relevance_reasons.append("直接概念匹配:改作")
            
            # 6. 原始查詢關鍵詞匹配
            query_words = set(query_lower.split())
            original_matches = query_words.intersection(content_words)
            
            if len(original_matches) >= 1:
                relevance_score += len(original_matches) * 0.3  # 較低權重，避免重複計算
                relevance_reasons.append(f"原始匹配{len(original_matches)}個詞:{list(original_matches)}")
            
            # 7. 相似度分數（調整閾值）
            if 'score' in result:
                if result['score'] > 0.3:  # 進一步降低閾值
                    relevance_score += result['score'] * 1.5  # 適度權重
                    relevance_reasons.append(f"相似度{result['score']:.2f}")
            
            # 8. 語義查詢的特殊加分（更嚴格的條件）
            if not has_explicit_article and relevance_score > 1.0:  # 只有當基礎分數夠高時才加分
                relevance_score += 0.3  # 進一步降低額外加分
                relevance_reasons.append("語義查詢加分")
            
            # 動態閾值：更寬鬆的標準以識別相關內容
            if has_explicit_article:
                base_threshold = 3.0  # 明確法條查詢的閾值
            else:
                base_threshold = 1.5  # 語義查詢的閾值，確保能識別相關內容
            
            # 如果有查詢擴展，適度降低閾值
            if query_expansion["expansion_ratio"] > 2.0:  # 當有顯著擴展時
                base_threshold *= 0.7  # 更積極的調整
            
            if relevance_score >= base_threshold:
                relevant_chunks.append(i)
                print(f"   ✅ Chunk {i+1} 相關 (分數:{relevance_score:.1f}): {relevance_reasons} - {content[:50]}...")
            else:
                print(f"   ❌ Chunk {i+1} 不相關 (分數:{relevance_score:.1f}): {content[:50]}...")
        
        print(f"📊 找到 {len(relevant_chunks)} 個相關chunks: {relevant_chunks}")
        
        # 計算P@K和R@K
        top_k_results = results[:k]
        relevant_in_top_k = 0
        
        for i, result in enumerate(top_k_results):
            if i in relevant_chunks:
                relevant_in_top_k += 1
        
        p_at_k = relevant_in_top_k / k if k > 0 else 0.0
        r_at_k = relevant_in_top_k / len(relevant_chunks) if relevant_chunks else 0.0
        
        print(f"📈 評測結果: P@{k}={p_at_k:.3f}, R@{k}={r_at_k:.3f}")
        
        return {
            "p_at_k": p_at_k,
            "r_at_k": r_at_k,
            "relevant_chunks_count": len(relevant_chunks),
            "relevant_chunks_indices": relevant_chunks,
            "query_analysis": {
                "query_type": query_type,
                "article_numbers": article_numbers,
                "law_keywords": law_keywords,
                "total_results": len(results),
                "threshold_used": base_threshold,
                "expansion_ratio": query_expansion["expansion_ratio"]
            },
            "query_expansion": {
                "original_query": query,
                "expanded_query": expanded_query,
                "detected_domains": detected_domains,
                "expansion_ratio": query_expansion["expansion_ratio"],
                "domain_matches": domain_matches[:10]  # 限制返回數量
            },
            "note": f"智能分析({query_type}+查詢擴展)，找到{len(relevant_chunks)}個相關結果"
        }
        
    except Exception as e:
        print(f"❌ 計算檢索指標時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return {"p_at_k": 0.0, "r_at_k": 0.0, "error": str(e)}


def load_qa_data() -> List[Dict]:
    """載入 QA 數據"""
    try:
        import json
        import os
        
        # 嘗試載入不同的 QA 文件（按優先級排序）
        qa_files = [
            "QA/qa_gold.json",  # 優先使用qa_gold.json
            "QA/copyright.json",
            "QA/copyright_p.json",
            "QA/copyright_n.json"
        ]
        
        # 獲取項目根目錄
        current_dir = os.path.dirname(__file__)
        project_root = os.path.join(current_dir, "..", "..")
        project_root = os.path.abspath(project_root)
        
        print(f"🔍 正在載入QA數據，項目根目錄: {project_root}")
        
        for qa_file in qa_files:
            qa_path = os.path.join(project_root, qa_file)
            print(f"   嘗試載入: {qa_path}")
            
            if os.path.exists(qa_path):
                try:
                    with open(qa_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            print(f"✅ 成功載入 {qa_file}，共 {len(data)} 條QA數據")
                            return data
                        else:
                            print(f"⚠️  {qa_file} 格式不正確或為空")
                except Exception as e:
                    print(f"❌ 載入 {qa_file} 失敗: {e}")
            else:
                print(f"❌ 文件不存在: {qa_path}")
        
        print("❌ 所有QA文件都無法載入")
        return []
    except Exception as e:
        print(f"❌ 載入 QA 數據失敗: {e}")
        return []


def is_article_match(chunk_content: str, article_number: int, article_suffix: int = None) -> bool:
    """檢查chunk內容是否包含指定的法條號碼"""
    import re
    
    if not chunk_content or article_number is None:
        return False
    
    # 標準化文本
    content = chunk_content.replace(" ", "").replace("　", "")
    
    if article_suffix is not None:
        # 第10條之1 或 第10-1條 格式
        patterns = [
            rf"第\s*{article_number}\s*條\s*(?:之|-)\s*{article_suffix}",
            rf"第\s*{article_number}\s*條\s*之\s*{article_suffix}",
            rf"第\s*{article_number}\s*條\s*-\s*{article_suffix}",
            rf"第\s*{article_number}\s*條之{article_suffix}",
            rf"第\s*{article_number}\s*條-{article_suffix}"
        ]
    else:
        # 第3條 格式（不包含之或-）
        patterns = [
            rf"第\s*{article_number}\s*條(?![\d之-])",
            rf"第\s*{article_number}\s*條$",
            rf"第\s*{article_number}\s*條[^0-9之-]"
        ]
    
    for pattern in patterns:
        if re.search(pattern, content):
            return True
    
    return False


def is_law_match(chunk_content: str, law_name: str) -> bool:
    """檢查chunk內容是否包含指定的法律名稱"""
    if not law_name or not chunk_content:
        return True  # 如果沒有指定法律名稱，不進行匹配
    
    # 法律名稱變體映射
    law_variants = {
        "著作權法": ["著作權法", "著作權", "版權法", "版權"],
        "商標法": ["商標法", "商標"],
        "專利法": ["專利法", "專利"],
        "民法": ["民法", "民事"],
        "刑法": ["刑法", "刑事"]
    }
    
    variants = law_variants.get(law_name, [law_name])
    content_lower = chunk_content.lower()
    
    # 如果chunk中包含任何法律名稱變體，則匹配
    if any(variant in content_lower for variant in variants):
        return True
    
    # 如果chunk中沒有明確的法律名稱，但包含法條號碼，也認為匹配
    # 這是因為法條內容本身可能不包含法律名稱
    import re
    if re.search(r'第\s*\d+\s*條', content_lower):
        return True
    
    return False


def is_relevant_chunk(chunk_content: str, gold_info: Dict[str, Any]) -> bool:
    """判斷chunk是否與gold標準相關"""
    if not chunk_content or not gold_info:
        return False
    
    # 法條號碼匹配（必須）
    article_number = gold_info.get("article_number")
    article_suffix = gold_info.get("article_suffix")
    
    if article_number is None:
        return False  # 沒有法條號碼，無法判斷相關性
    
    article_match = is_article_match(chunk_content, article_number, article_suffix)
    if not article_match:
        return False
    
    # 法律名稱匹配（加分項）
    law_name = gold_info.get("law", "")
    law_match = is_law_match(chunk_content, law_name)
    
    return law_match


def extract_articles_from_text(text: str) -> List[str]:
    """從文本中提取法條信息"""
    import re
    
    articles = []
    
    # 匹配 "第X條" 模式
    patterns = [
        r"第(\d+)條",
        r"第(\d+)條之(\d+)",
        r"第(\d+)-(\d+)條"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                if len(match) == 2:
                    if match[1]:  # 有之N
                        articles.append(f"第{match[0]}條之{match[1]}")
                    else:  # 範圍
                        articles.append(f"第{match[0]}-{match[1]}條")
                else:
                    articles.append(f"第{match[0]}條")
            else:
                articles.append(f"第{match}條")
    
    return list(set(articles))  # 去重


def asyncio_run(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # create task and wait
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return asyncio.run(coro)


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest):
    if store.embeddings is None:
        return JSONResponse(status_code=400, content={"error": "run /embed first"})
    
    # 計算相似度並排序（只使用密集向量）
    idxs, sims = rank_with_dense_vectors(req.query, req.k)

    # Use the same order as built in /embed
    chunks_flat = store.chunks_flat
    mapping_doc_ids = store.chunk_doc_ids

    results = []
    for rank, (i, score) in enumerate(zip(idxs, sims), start=1):
        if i < 0 or i >= len(chunks_flat):
            continue
        
        # 獲取文檔信息
        doc_id = mapping_doc_ids[i]
        doc = store.docs.get(doc_id)
        
        # 基本結果
        hierarchical_desc = generate_hierarchical_description(doc_id, "standard", i, store)
        
        result = {
            "rank": rank,
            "score": float(score),
            "doc_id": doc_id,
            "chunk_index": i,
            "content": chunks_flat[i][:2000],
            "hierarchical_description": hierarchical_desc,  # 新增層級描述
        }
        
        # 如果有結構化chunks，添加metadata
        if doc and hasattr(doc, 'structured_chunks') and doc.structured_chunks and i < len(doc.structured_chunks):
            structured_chunk = doc.structured_chunks[i]
            result["metadata"] = structured_chunk.get("metadata", {})
            result["chunk_id"] = structured_chunk.get("chunk_id", "")
            
            # 添加法律結構信息
            metadata = structured_chunk.get("metadata", {})
            result["legal_structure"] = {
                "id": metadata.get("id", ""),
                "spans": metadata.get("spans", {}),
                "page_range": metadata.get("page_range", {})
            }
        
        results.append(result)
    
    # 計算 P@K 和 R@K（如果有 QA 數據）
    metrics = calculate_retrieval_metrics(req.query, results, req.k)
    
    # 判斷 embedding provider 和 model（不再支持 TF-IDF）
    if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
        embedding_provider = "gemini"
        embedding_model = "gemini-embedding-001"
    elif USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE:
        embedding_provider = "bge-m3"
        embedding_model = "BAAI/bge-m3"
    else:
        embedding_provider = "unknown"
        embedding_model = "unknown"

    return {
        "query": req.query, 
        "k": req.k, 
        "results": results,
        "metrics": metrics,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model
    }


@app.post("/api/hierarchical-retrieve")
def hierarchical_retrieve(req: RetrieveRequest):
    """多層次檢索：基於論文的多層次嵌入檢索方法"""
    if store.embeddings is None:
        return JSONResponse(status_code=400, content={"error": "run /embed first"})
    
    # 獲取所有 chunks 和 metadata
    chunks_flat = store.chunks_flat
    mapping_doc_ids = store.chunk_doc_ids
    
    if not chunks_flat:
        return JSONResponse(status_code=400, content={"error": "no chunks available"})
    
    # 構建層次化節點
    hierarchical_nodes = []
    for i, (chunk, doc_id) in enumerate(zip(chunks_flat, mapping_doc_ids)):
        doc = store.docs.get(doc_id)
        metadata = {}
        
        # 提取層次化metadata
        if doc and hasattr(doc, 'structured_chunks') and doc.structured_chunks and i < len(doc.structured_chunks):
            structured_chunk = doc.structured_chunks[i]
            metadata = structured_chunk.get("metadata", {})
            
            # 確定層次級別
            hierarchy_level = "article"  # 默認
            if metadata.get("article"):
                hierarchy_level = "article"
            elif metadata.get("section"):
                hierarchy_level = "section"
            elif metadata.get("chapter"):
                hierarchy_level = "chapter"
            
            metadata["hierarchy_level"] = hierarchy_level
        
        hierarchical_nodes.append({
            "content": chunk,
            "metadata": metadata,
            "doc_id": doc_id,
            "chunk_index": i
        })
    
    # 多層次檢索邏輯 - 改進版
    # 1. 先按層次分組（基於內容分析）
    article_nodes = []
    section_nodes = []
    chapter_nodes = []
    general_nodes = []
    
    for node in hierarchical_nodes:
        content = node.get("content", "")
        hierarchy_level = detect_content_hierarchy(content)
        
        if hierarchy_level == "article":
            article_nodes.append(node)
        elif hierarchy_level == "section":
            section_nodes.append(node)
        elif hierarchy_level == "chapter":
            chapter_nodes.append(node)
        else:
            general_nodes.append(node)
    
    print(f"🔍 層次分組: 法條{len(article_nodes)}個, 節{len(section_nodes)}個, 章{len(chapter_nodes)}個, 一般{len(general_nodes)}個")
    
    # 2. 對每個層次進行檢索
    all_results = []
    
    # 法條級別檢索（最高精度）
    if article_nodes:
        article_indices = [i for i, node in enumerate(hierarchical_nodes) if node in article_nodes]
        if article_indices:
            article_idxs, article_sims = rank_with_dense_vectors(req.query, k=min(len(article_indices), req.k * 2))
            for idx, sim in zip(article_idxs, article_sims):
                if idx in article_indices:
                    node = hierarchical_nodes[idx]
                    all_results.append({
                        "rank": len(all_results) + 1,
                        "score": float(sim),
                        "doc_id": node["doc_id"],
                        "chunk_index": idx,
                        "content": node["content"][:2000],
                        "metadata": node["metadata"],
                        "hierarchy_level": "article",
                        "hierarchy_weight": 1.0
                    })
    
    # 節級別檢索（中等精度）
    if section_nodes and len(all_results) < req.k:
        section_indices = [i for i, node in enumerate(hierarchical_nodes) if node in section_nodes]
        if section_indices:
            section_idxs, section_sims = rank_with_dense_vectors(req.query, k=min(len(section_indices), req.k))
            for idx, sim in zip(section_idxs, section_sims):
                if idx in section_indices and len(all_results) < req.k:
                    node = hierarchical_nodes[idx]
                    all_results.append({
                        "rank": len(all_results) + 1,
                        "score": float(sim) * 0.8,  # 節級別權重
                        "doc_id": node["doc_id"],
                        "chunk_index": idx,
                        "content": node["content"][:2000],
                        "metadata": node["metadata"],
                        "hierarchy_level": "section",
                        "hierarchy_weight": 0.8
                    })
    
    # 章級別檢索（較低精度但廣度更大）
    if chapter_nodes and len(all_results) < req.k:
        chapter_indices = [i for i, node in enumerate(hierarchical_nodes) if node in chapter_nodes]
        if chapter_indices:
            chapter_idxs, chapter_sims = rank_with_dense_vectors(req.query, k=min(len(chapter_indices), req.k))
            for idx, sim in zip(chapter_idxs, chapter_sims):
                if idx in chapter_indices and len(all_results) < req.k:
                    node = hierarchical_nodes[idx]
                    all_results.append({
                        "rank": len(all_results) + 1,
                        "score": float(sim) * 0.6,  # 章級別權重
                        "doc_id": node["doc_id"],
                        "chunk_index": idx,
                        "content": node["content"][:2000],
                        "metadata": node["metadata"],
                        "hierarchy_level": "chapter",
                        "hierarchy_weight": 0.6
                    })
    
    # 取前k個結果
    results = all_results[:req.k]
    
    # 計算多層次檢索指標
    metrics = calculate_retrieval_metrics(req.query, results, req.k)
    
    # 添加多層次檢索特定信息
    hierarchy_stats = {
        "article_results": len([r for r in results if r.get("hierarchy_level") == "article"]),
        "section_results": len([r for r in results if r.get("hierarchy_level") == "section"]),
        "chapter_results": len([r for r in results if r.get("hierarchy_level") == "chapter"])
    }
    
    metrics["hierarchical_analysis"] = hierarchy_stats
    metrics["note"] = f"多層次檢索: 法條{hierarchy_stats['article_results']}個, 節{hierarchy_stats['section_results']}個, 章{hierarchy_stats['chapter_results']}個"
    
    # 判斷 embedding provider 和 model
    embedding_provider = "gemini"
    embedding_model = "text-embedding-004"
    
    return {
        "results": results,
        "metrics": metrics,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model
    }


@app.post("/api/multi-level-retrieve")
async def multi_level_retrieve(req: Dict[str, Any]):
    """多層次檢索：基於查詢分類的智能層次選擇檢索，支持實驗組限制"""
    query = req.get("query")
    k = req.get("k", 10)
    experimental_groups = req.get("experimental_groups", [])
    
    if not query:
        return JSONResponse(status_code=400, content={"error": "Query is required"})
    
    # 檢查是否有可用的多層次embedding
    if not store.has_multi_level_embeddings():
        return JSONResponse(
            status_code=400, 
            content={"error": "Multi-level embeddings not available. Please run /api/multi-level-embed first."}
        )
    
    # 如果指定了實驗組，限制可用的層次
    available_levels = store.get_available_levels()
    if experimental_groups:
        print(f"🎯 實驗組限制檢索: {experimental_groups}")
        # 收集實驗組需要的層次
        required_levels = set()
        for group_key in experimental_groups:
            if group_key in GRANULARITY_COMBINATIONS:
                required_levels.update(GRANULARITY_COMBINATIONS[group_key]["levels"])
        
        # 只使用實驗組需要的層次
        available_levels = [level for level in available_levels if level in required_levels]
        print(f"🎯 實驗組可用層次: {available_levels}")
    
    # 分析查詢並分類
    query_analysis = get_query_analysis(query)
    recommended_level = query_analysis['recommended_level']
    query_type = query_analysis['query_type']
    confidence = query_analysis['confidence']
    
    print(f"🔍 查詢分析：類型={query_type}, 置信度={confidence:.3f}, 推薦層次={recommended_level}")
    print(f"📊 可用層次: {available_levels}")
    
    # 檢查推薦層次是否可用，如果不可用則選擇最佳可用層次
    if recommended_level not in available_levels:
        # 按優先級選擇可用的層次
        fallback_levels = ['basic_unit', 'basic_unit_component', 'enumeration', 'basic_unit_hierarchy', 'document_component', 'document']
        for fallback_level in fallback_levels:
            if fallback_level in available_levels:
                recommended_level = fallback_level
                print(f"⚠️  推薦層次 {query_analysis['recommended_level']} 不可用，使用備選層次: {recommended_level}")
                break
    
    # 獲取推薦層次的embedding
    level_data = store.get_multi_level_embeddings(recommended_level)
    if not level_data:
        return JSONResponse(
            status_code=400, 
            content={"error": f"No embeddings available for level: {recommended_level}. Available levels: {available_levels}"}
        )
    
    vectors = level_data['embeddings']
    chunks = level_data['chunks']
    doc_ids = level_data['doc_ids']
    metadata = level_data.get('metadata', {})
    
    print(f"📊 使用層次 '{recommended_level}' 進行檢索，共 {len(chunks)} 個chunks")
    
    # 執行檢索
    try:
        import numpy as np
        
        # 檢測存儲的embedding模型信息
        embedding_provider = metadata.get('provider')
        embedding_dimension = metadata.get('dimension')
        
        if embedding_provider:
            print(f"🔍 檢測到存儲的embedding提供者: {embedding_provider}, 維度: {embedding_dimension}")
        
        # 根據存儲的embedding模型選擇查詢向量化方法
        query_vector = None
        if embedding_provider == 'gemini' or (not embedding_provider and USE_GEMINI_EMBEDDING and GOOGLE_API_KEY):
            query_vector = (await embed_gemini([req.query]))[0]
            print(f"✅ 使用Gemini生成查詢向量，維度: {len(query_vector)}")
        elif embedding_provider == 'bge-m3' or (not embedding_provider and USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE):
            query_vector = embed_bge_m3([req.query])[0]
            print(f"✅ 使用BGE-M3生成查詢向量，維度: {len(query_vector)}")
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "No embedding method available for query"}
            )
        
        # 驗證維度匹配
        if embedding_dimension and len(query_vector) != embedding_dimension:
            print(f"⚠️ 警告：查詢向量維度({len(query_vector)})與存儲向量維度({embedding_dimension})不匹配")
            return JSONResponse(
                status_code=500,
                content={"error": f"Dimension mismatch: query vector has {len(query_vector)} dimensions but stored embeddings have {embedding_dimension} dimensions. Please re-run /api/multi-level-embed with the current embedding provider."}
            )
        
        # 計算相似度
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector)
        
        similarities = cosine_similarity([query_vector], vectors)[0]
        
        # 獲取top-k結果
        top_indices = np.argsort(similarities)[::-1][:req.k]
        
        results = []
        for i, idx in enumerate(top_indices):
            doc_id = doc_ids[idx]
            doc = store.get_doc(doc_id)
            
            # 生成層級描述
            hierarchical_desc = generate_hierarchical_description(doc_id, recommended_level, idx, store)
            
            result = {
                "rank": i + 1,
                "content": chunks[idx],
                "similarity": float(similarities[idx]),
                "doc_id": doc_id,
                "doc_name": doc.filename if doc else "Unknown",
                "chunk_index": idx,
                "hierarchical_description": hierarchical_desc,  # 新增層級描述
                "metadata": {
                    "level": recommended_level,
                    "query_type": query_type,
                    "confidence": confidence
                }
            }
            results.append(result)
        
        # 計算檢索指標
        metrics = {
            "total_chunks_searched": len(chunks),
            "query_type": query_type,
            "recommended_level": recommended_level,
            "classification_confidence": confidence,
            "embedding_provider": "gemini" if USE_GEMINI_EMBEDDING else "bge-m3",
            "embedding_model": "text-embedding-004" if USE_GEMINI_EMBEDDING else "BAAI/bge-m3"
        }
        
        return {
            "results": results,
            "metrics": metrics,
            "query_analysis": query_analysis
        }
        
    except Exception as e:
        print(f"❌ 多層次檢索錯誤: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Multi-level retrieval failed: {str(e)}"}
        )


@app.post("/api/query-analysis")
def analyze_query(req: RetrieveRequest):
    """查詢分析端點：分析查詢類型並推薦檢索策略"""
    query_analysis = get_query_analysis(req.query)
    
    # 檢查可用的embedding層次
    available_levels = store.get_available_levels()
    has_multi_level = store.has_multi_level_embeddings()
    
    # 生成檢索建議
    retrieval_suggestions = {
        "recommended_method": "multi-level" if has_multi_level else "standard",
        "recommended_level": query_analysis['recommended_level'],
        "available_levels": available_levels,
        "alternative_levels": [level for level in available_levels if level != query_analysis['recommended_level']]
    }
    
    return {
        "query_analysis": query_analysis,
        "retrieval_suggestions": retrieval_suggestions,
        "system_status": {
            "has_multi_level_embeddings": has_multi_level,
            "has_standard_embeddings": store.embeddings is not None
        }
    }


@app.post("/api/multi-level-fusion-retrieve")
async def multi_level_fusion_retrieve(req: MultiLevelFusionRequest):
    """多層次融合檢索：從所有層次檢索並融合結果"""
    # 檢查是否有可用的多層次embedding
    if not store.has_multi_level_embeddings():
        return JSONResponse(
            status_code=400, 
            content={"error": "Multi-level embeddings not available. Please run /api/multi-level-embed first."}
        )
    
    # 分析查詢
    query_analysis = get_query_analysis(req.query)
    available_levels = store.get_available_levels()
    
    print(f"🔍 多層次融合檢索：查詢類型={query_analysis['query_type']}, 可用層次={available_levels}")
    
    # 如果沒有可用的層次，返回錯誤
    if not available_levels:
        return JSONResponse(
            status_code=400,
            content={"error": "No multi-level embeddings available. Please run /api/multi-level-embed first."}
        )
    
    # 從所有可用層次檢索
    level_results = {}
    total_chunks_searched = 0
    
    try:
        import numpy as np
        
        # 檢測第一個可用層次的模型信息，確保使用相同的模型
        first_level = available_levels[0] if available_levels else None
        embedding_provider = None
        embedding_dimension = None
        
        if first_level:
            first_level_data = store.get_multi_level_embeddings(first_level)
            if first_level_data and 'metadata' in first_level_data:
                embedding_provider = first_level_data['metadata'].get('provider')
                embedding_dimension = first_level_data['metadata'].get('dimension')
                print(f"🔍 檢測到存儲的embedding提供者: {embedding_provider}, 維度: {embedding_dimension}")
        
        # 根據存儲的embedding模型選擇查詢向量化方法
        query_vector = None
        if embedding_provider == 'gemini' or (not embedding_provider and USE_GEMINI_EMBEDDING and GOOGLE_API_KEY):
            query_vector = (await embed_gemini([req.query]))[0]
            print(f"✅ 使用Gemini生成查詢向量，維度: {len(query_vector)}")
        elif embedding_provider == 'bge-m3' or (not embedding_provider and USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE):
            query_vector = embed_bge_m3([req.query])[0]
            print(f"✅ 使用BGE-M3生成查詢向量，維度: {len(query_vector)}")
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "No embedding method available for query"}
            )
        
        # 驗證維度匹配
        if embedding_dimension and len(query_vector) != embedding_dimension:
            print(f"⚠️ 警告：查詢向量維度({len(query_vector)})與存儲向量維度({embedding_dimension})不匹配")
            return JSONResponse(
                status_code=500,
                content={"error": f"Dimension mismatch: query vector has {len(query_vector)} dimensions but stored embeddings have {embedding_dimension} dimensions. Please re-run /api/multi-level-embed with the current embedding provider."}
            )
        
        # 對每個層次進行檢索
        for level_name in available_levels:
            level_data = store.get_multi_level_embeddings(level_name)
            if not level_data:
                continue
            
            vectors = level_data['embeddings']
            chunks = level_data['chunks']
            doc_ids = level_data['doc_ids']
            
            print(f"📊 檢索層次 '{level_name}'：{len(chunks)} 個chunks")
            total_chunks_searched += len(chunks)
            
            # 計算相似度
            if isinstance(vectors, list):
                vectors = np.array(vectors)
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector)
            
            similarities = cosine_similarity([query_vector], vectors)[0]
            
            # 獲取top-k結果
            top_indices = np.argsort(similarities)[::-1][:req.k]
            
            level_results[level_name] = []
            for i, idx in enumerate(top_indices):
                doc_id = doc_ids[idx]
                doc = store.get_doc(doc_id)
                
                # 生成層級描述
                hierarchical_desc = generate_hierarchical_description(doc_id, level_name, idx, store)
                
                result = {
                    "rank": int(i + 1),
                    "content": chunks[idx],
                    "similarity": float(similarities[idx]),
                    "doc_id": doc_id,
                    "doc_name": doc.filename if doc else "Unknown",
                    "chunk_index": int(idx),
                    "hierarchical_description": hierarchical_desc,  # 新增層級描述
                    "metadata": {
                        "level": level_name,
                        "query_type": query_analysis['query_type'],
                        "confidence": query_analysis['confidence']
                    }
                }
                level_results[level_name].append(result)
        
        if not level_results:
            return JSONResponse(
                status_code=400,
                content={"error": "No results found from any level"}
            )
        
        # 創建融合配置
        fusion_config = FusionConfig(
            strategy=req.fusion_strategy,
            level_weights=req.level_weights,
            similarity_threshold=req.similarity_threshold,
            max_results=req.max_results,
            normalize_scores=req.normalize_scores
        )
        
        # 執行結果融合
        print(f"🔄 執行結果融合：策略={req.fusion_strategy}")
        fused_results = fuse_multi_level_results(level_results, fusion_config)
        
        # 計算融合指標
        fusion_metrics = {
            "total_chunks_searched": total_chunks_searched,
            "levels_searched": list(level_results.keys()),
            "fusion_strategy": req.fusion_strategy,
            "level_weights": req.level_weights or fusion_config.level_weights,
            "similarity_threshold": req.similarity_threshold,
            "max_results": req.max_results,
            "query_type": query_analysis['query_type'],
            "classification_confidence": query_analysis['confidence'],
            "embedding_provider": "gemini" if USE_GEMINI_EMBEDDING else "bge-m3",
            "embedding_model": "text-embedding-004" if USE_GEMINI_EMBEDDING else "BAAI/bge-m3"
        }
        
        # 統計各層次的貢獻
        level_contributions = {}
        for level, results in level_results.items():
            level_contributions[level] = {
                "num_results": len(results),
                "avg_similarity": sum(r['similarity'] for r in results) / len(results) if results else 0,
                "max_similarity": max(r['similarity'] for r in results) if results else 0
            }
        
        fusion_metrics["level_contributions"] = level_contributions
        
        return {
            "results": fused_results,
            "metrics": fusion_metrics,
            "query_analysis": query_analysis,
            "level_results": level_results  # 包含原始各層次結果
        }
        
    except Exception as e:
        print(f"❌ 多層次融合檢索錯誤: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Multi-level fusion retrieval failed: {str(e)}"}
        )


@app.post("/api/hybrid-retrieve")
def hybrid_retrieve(req: RetrieveRequest):
    """HybridRAG 檢索：結合向量相似度和法律結構規則"""
    if store.embeddings is None:
        return JSONResponse(status_code=400, content={"error": "run /embed first"})
    
    # 獲取所有 chunks 和 metadata
    chunks_flat = store.chunks_flat
    mapping_doc_ids = store.chunk_doc_ids
    
    if not chunks_flat:
        return JSONResponse(status_code=400, content={"error": "no chunks available"})
    
    # 構建 nodes 格式供 hybrid_rank 使用
    nodes = []
    for i, (chunk, doc_id) in enumerate(zip(chunks_flat, mapping_doc_ids)):
        doc = store.docs.get(doc_id)
        metadata = {}
        
        # 如果有結構化chunks，提取metadata
        if doc and hasattr(doc, 'structured_chunks') and doc.structured_chunks and i < len(doc.structured_chunks):
            structured_chunk = doc.structured_chunks[i]
            metadata = structured_chunk.get("metadata", {})
        
        nodes.append({
            "content": chunk,
            "metadata": metadata,
            "doc_id": doc_id,
            "chunk_index": i
        })
    
    # 先用密集向量得到每個節點的向量分數
    # 我們對所有節點進行相似度計算，然後只取前 k 的結果做 Hybrid 排序
    dense_top_k = min(len(nodes), max(req.k * 4, req.k))
    all_vec_idxs, all_vec_sims = rank_with_dense_vectors(req.query, k=len(nodes))
    # 映射出節點順序對應的分數，初始化為0
    node_vector_scores = [0.0] * len(nodes)
    for rank_idx, node_idx in enumerate(all_vec_idxs):
        node_vector_scores[node_idx] = float(all_vec_sims[rank_idx])

    # 取向量分數最高的前 dense_top_k 節點作為 Hybrid 候選
    top_vec_pairs = sorted(
        [(i, s) for i, s in enumerate(node_vector_scores)], key=lambda x: x[1], reverse=True
    )[:dense_top_k]
    candidate_nodes = [nodes[i] for i, _ in top_vec_pairs]
    candidate_scores = [s for _, s in top_vec_pairs]

    # 使用 hybrid_rank 進行檢索（向量分數 + metadata 加分）
    config = HybridConfig(
        alpha=0.8,  # 向量相似度權重
        w_law_match=0.15,  # 法名對齊權重
        w_article_match=0.15,  # 條號對齊權重
        w_keyword_hit=0.05,  # 術語命中權重
        max_bonus=0.4  # 最大加分
    )

    hybrid_results = hybrid_rank(
        req.query, candidate_nodes, k=req.k, config=config, vector_scores=candidate_scores
    )
    
    # 轉換為標準格式
    results = []
    for rank, item in enumerate(hybrid_results, start=1):
        # 生成層級描述
        hierarchical_desc = generate_hierarchical_description(item["doc_id"], "hybrid", item["chunk_index"], store)
        
        result = {
            "rank": rank,
            "score": item["score"],
            "vector_score": item["vector_score"],
            "bonus": item["bonus"],
            "doc_id": item["doc_id"],
            "chunk_index": item["chunk_index"],
            "content": item["content"][:2000],
            "metadata": item["metadata"],
            "hierarchical_description": hierarchical_desc,  # 新增層級描述
        }
        
        # 添加法律結構信息
        if item["metadata"]:
            result["legal_structure"] = {
                "id": item["metadata"].get("id", ""),
                "category": item["metadata"].get("category", ""),
                "article_label": item["metadata"].get("article_label", ""),
                "article_number": item["metadata"].get("article_number"),
                "article_suffix": item["metadata"].get("article_suffix"),
                "spans": item["metadata"].get("spans", {}),
                "page_range": item["metadata"].get("page_range", {})
            }
        
        results.append(result)
    
    # 計算 P@K 和 R@K（如果有 QA 數據）
    metrics = calculate_retrieval_metrics(req.query, results, req.k)
    
    # 判斷 embedding provider 和 model（不再支持 TF-IDF）
    if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
        embedding_provider = "gemini"
        embedding_model = "gemini-embedding-001"
    elif USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE:
        embedding_provider = "bge-m3"
        embedding_model = "BAAI/bge-m3"
    else:
        embedding_provider = "unknown"
        embedding_model = "unknown"
    
    return {
        "query": req.query, 
        "k": req.k, 
        "results": results,
        "method": "hybrid_rag",
        "metrics": metrics,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "config": {
            "alpha": config.alpha,
            "w_law_match": config.w_law_match,
            "w_article_match": config.w_article_match,
            "w_keyword_hit": config.w_keyword_hit,
            "max_bonus": config.max_bonus
        }
    }


@app.post("/api/enhanced-hybrid-retrieve")
def enhanced_hybrid_retrieve(req: RetrieveRequest):
    """使用增強版HybridRAG進行檢索"""
    print(f"🚀 增強版HybridRAG檢索請求: {req.query}, k={req.k}")
    
    # 檢查是否有FAISS和BM25索引（標準或多層次）
    faiss_available = faiss_store.has_vectors() or faiss_store.has_multi_level_vectors()
    bm25_available = bm25_index.has_index() or bm25_index.has_multi_level_index()
    print(f"📊 索引狀態: FAISS={faiss_available}, BM25={bm25_available}")
    
    # 如果索引不可用，嘗試自動重新加載
    if not faiss_available or not bm25_available:
        print("⚠️ 索引不完整，嘗試自動重新加載...")
        try:
            # 嘗試從磁盤加載索引
            if not faiss_available:
                print("🔄 嘗試重新加載FAISS索引...")
                faiss_store.load_data()
                faiss_available = faiss_store.has_vectors() or faiss_store.has_multi_level_vectors()
                print(f"   FAISS加載結果: {faiss_available}")
            
            if not bm25_available:
                print("🔄 嘗試重新加載BM25索引...")
                bm25_index.load_data()
                bm25_available = bm25_index.has_index() or bm25_index.has_multi_level_index()
                print(f"   BM25加載結果: {bm25_available}")
            
            # 如果加載失敗，嘗試從store重建（標準索引）
            if (not faiss_available or not bm25_available) and (store.embeddings is not None and store.chunks_flat):
                print("⚠️ 索引文件不存在，嘗試從store重建標準索引...")
                vectors = store.embeddings
                chunks = store.chunks_flat
                chunk_ids = [f"{doc_id}_{i}" for i, doc_id in enumerate(store.chunk_doc_ids)]
                
                # 重建FAISS索引
                if not faiss_available and vectors:
                    print("🔧 重建FAISS索引...")
                    dimension = len(vectors[0]) if vectors else EMBEDDING_DIMENSION
                    faiss_store.create_index(dimension, "flat")
                    faiss_store.add_vectors(vectors, chunk_ids, store.chunk_doc_ids, chunks)
                    
                    # 恢復enhanced metadata
                    if hasattr(store, 'enhanced_metadata') and store.enhanced_metadata:
                        for chunk_id, metadata in store.enhanced_metadata.items():
                            faiss_store.set_enhanced_metadata(chunk_id, metadata)
                    
                    faiss_store.save_data()
                    faiss_available = faiss_store.has_vectors() or faiss_store.has_multi_level_vectors()
                    print(f"   ✅ FAISS索引已重建: {faiss_available}")
                
                # 重建BM25索引
                if not bm25_available and chunks:
                    print("🔧 重建BM25索引...")
                    bm25_index.build_index(chunks, chunk_ids, store.chunk_doc_ids)
                    bm25_index.save_data()
                    bm25_available = bm25_index.has_index() or bm25_index.has_multi_level_index()
                    print(f"   ✅ BM25索引已重建: {bm25_available}")
            
        except Exception as e:
            print(f"⚠️ 自動重新加載索引失敗: {e}")
    
    # 再次檢查索引狀態
    if not faiss_available and not bm25_available:
        return JSONResponse(
            status_code=400,
            content={
                "error": "No enhanced indices available. Please run /api/embed or /api/multi-level-embed first.",
                "faiss_available": faiss_available,
                "bm25_available": bm25_available,
                "suggestion": "請先執行 /api/embed 或 /api/multi-level-embed 來創建索引，或從Upload頁面選擇已存在的Embedding資料庫"
            }
        )
    
    try:
        # 配置增強版HybridRAG
        config = EnhancedHybridConfig(
            vector_weight=0.6,
            bm25_weight=0.25,
            metadata_weight=0.15,
            w_law_match=0.15,
            w_article_match=0.15,
            w_concept_match=0.1,
            w_keyword_hit=0.05,
            w_domain_match=0.05,
            w_title_match=0.1,
            w_category_match=0.05,
            max_bonus=0.4,
            title_boost_factor=1.5,
            category_boost_factor=1.3,
            # Metadata向下繼承配置
            enable_inheritance_strategy=True,
            metadata_match_threshold=0.3,
            inheritance_bonus=0.1,
            inheritance_boost_factor=1.2
        )
        
        # 執行增強版HybridRAG檢索
        enhanced_results = enhanced_hybrid_rag.retrieve(req.query, req.k, config)
        
        # 生成層級描述
        for result in enhanced_results:
            if 'doc_id' in result:
                doc_id = result.get('doc_id', 'unknown')
                level = 'basic_unit'  # 默認層級
                chunk_index = result.get('chunk_index', 0)
                result['hierarchical_description'] = generate_hierarchical_description(
                    doc_id, level, chunk_index, store
                )
        
        print(f"✅ 增強版HybridRAG檢索完成，返回 {len(enhanced_results)} 個結果")
        
        return {
            "results": enhanced_results,
            "query": req.query,
            "final_results": len(enhanced_results),
            "config": config.__dict__,
            "retrieval_stats": enhanced_hybrid_rag.get_retrieval_stats()
        }
        
    except Exception as e:
        print(f"❌ 增強版HybridRAG檢索失敗: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Enhanced HybridRAG retrieval failed: {str(e)}"}
        )


@app.post("/api/hybrid-rrf-retrieve")
async def hybrid_rrf_retrieve(req: RetrieveRequest):
    """HybridRAG(RRF)檢索：純RRF融合向量+BM25，不考慮Metadata加分"""
    # 記錄檢索開始時間
    retrieval_start_time = time.time()
    print(f"🔄 HybridRAG(RRF)檢索請求: {req.query}, k={req.k}")
    
    # 驗證查詢
    if not req.query or not req.query.strip():
        error_msg = "Query cannot be empty"
        print(f"❌ 驗證失敗: {error_msg}")
        return JSONResponse(
            status_code=400,
            content={"error": error_msg}
        )
    
    if req.k <= 0:
        error_msg = f"k must be greater than 0, got {req.k}"
        print(f"❌ 驗證失敗: {error_msg}")
        return JSONResponse(
            status_code=400,
            content={"error": error_msg}
        )
    
    # 檢查是否有FAISS和BM25索引（標準或多層次）
    faiss_available = faiss_store.has_vectors() or faiss_store.has_multi_level_vectors()
    bm25_available = bm25_index.has_index() or bm25_index.has_multi_level_index()
    print(f"📊 索引狀態: FAISS={faiss_available}, BM25={bm25_available}")
    
    # 如果索引不可用，嘗試自動重新加載
    if not faiss_available or not bm25_available:
        print("⚠️ 索引不完整，嘗試自動重新加載...")
        try:
            # 嘗試從磁盤加載索引
            if not faiss_available:
                print("🔄 嘗試重新加載FAISS索引...")
                faiss_store.load_data()
                faiss_available = faiss_store.has_vectors() or faiss_store.has_multi_level_vectors()
                print(f"   FAISS加載結果: {faiss_available}")
            
            if not bm25_available:
                print("🔄 嘗試重新加載BM25索引...")
                bm25_index.load_data()
                bm25_available = bm25_index.has_index() or bm25_index.has_multi_level_index()
                print(f"   BM25加載結果: {bm25_available}")
            
            # 如果加載失敗，嘗試從store重建（標準索引）
            if (not faiss_available or not bm25_available) and (store.embeddings is not None and store.chunks_flat):
                print("⚠️ 索引文件不存在，嘗試從store重建標準索引...")
                vectors = store.embeddings
                chunks = store.chunks_flat
                chunk_ids = [f"{doc_id}_{i}" for i, doc_id in enumerate(store.chunk_doc_ids)]
                
                # 重建FAISS索引
                if not faiss_available and vectors:
                    print("🔧 重建FAISS索引...")
                    dimension = len(vectors[0]) if vectors else EMBEDDING_DIMENSION
                    faiss_store.create_index(dimension, "flat")
                    faiss_store.add_vectors(vectors, chunk_ids, store.chunk_doc_ids, chunks)
                    
                    # 恢復enhanced metadata
                    if hasattr(store, 'enhanced_metadata') and store.enhanced_metadata:
                        for chunk_id, metadata in store.enhanced_metadata.items():
                            faiss_store.set_enhanced_metadata(chunk_id, metadata)
                    
                    faiss_store.save_data()
                    faiss_available = faiss_store.has_vectors() or faiss_store.has_multi_level_vectors()
                    print(f"   ✅ FAISS索引已重建: {faiss_available}")
                
                # 重建BM25索引
                if not bm25_available and chunks:
                    print("🔧 重建BM25索引...")
                    bm25_index.build_index(chunks, chunk_ids, store.chunk_doc_ids)
                    bm25_index.save_data()
                    bm25_available = bm25_index.has_index() or bm25_index.has_multi_level_index()
                    print(f"   ✅ BM25索引已重建: {bm25_available}")
            
        except Exception as e:
            print(f"⚠️ 自動重新加載索引失敗: {e}")
    
    # 再次檢查索引狀態
    if not faiss_available and not bm25_available:
        error_msg = "No enhanced indices available. Please run /api/embed or /api/multi-level-embed first."
        print(f"❌ 驗證失敗: {error_msg}")
        return JSONResponse(
            status_code=400,
            content={
                "error": error_msg,
                "faiss_available": faiss_available,
                "bm25_available": bm25_available,
                "suggestion": "請先執行 /api/embed 或 /api/multi-level-embed 來創建索引，或從Upload頁面選擇已存在的Embedding資料庫"
            }
        )
    
    try:
        # 1. 向量檢索 - 生成查詢向量
        print("📊 執行向量檢索...")
        # 記錄查詢 embedding 開始時間
        query_embedding_start = time.time()
        query_vector = None
        query_embedding_time = 0.0
        if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
            try:
                query_vector = (await embed_gemini([req.query]))[0]
                # 驗證向量維度
                if not query_vector or len(query_vector) != EMBEDDING_DIMENSION:
                    raise ValueError(f"Query vector dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(query_vector) if query_vector else 0}")
                query_embedding_time = time.time() - query_embedding_start
                print(f"✅ 使用Gemini生成查詢向量，維度: {len(query_vector)}")
                print(f"⏱️  查詢 Embedding 花費時間: {query_embedding_time:.3f} 秒")
            except Exception as e:
                query_embedding_time = time.time() - query_embedding_start
                print(f"❌ Gemini query embedding failed: {e}")
                print(f"⏱️  查詢 Embedding 失敗前花費時間: {query_embedding_time:.3f} 秒")
                # 全部使用Gemini，不使用BGE-M3 fallback
                raise RuntimeError(
                    f"Gemini embedding failed: {str(e)}. "
                    f"請檢查：1) GOOGLE_API_KEY是否正確 2) 網絡連接是否正常 3) 查詢文本是否包含無法處理的特殊字符"
                )
        else:
            raise RuntimeError(
                f"Gemini embedding未啟用或API key未設置。"
                f"USE_GEMINI_EMBEDDING={USE_GEMINI_EMBEDDING}, GOOGLE_API_KEY={'已設置' if GOOGLE_API_KEY else '未設置'}"
            )
        
        # 驗證query_vector
        if not query_vector:
            raise RuntimeError("Failed to generate query vector")
        
        # 檢測是標準索引還是多層次索引
        use_standard_index = faiss_store.has_vectors()
        use_multi_level_index = faiss_store.has_multi_level_vectors()
        
        # 優先使用多層次索引（如果存在），因為這是實驗組B、C、D使用的索引
        # 如果標準索引維度不匹配，自動清除並使用多層次索引
        if use_standard_index:
            expected_dim = faiss_store.dimension
            query_dim = len(query_vector)
            if query_dim != expected_dim:
                print(f"⚠️ 檢測到標準索引維度({expected_dim})與查詢向量維度({query_dim})不匹配")
                if use_multi_level_index:
                    print(f"💡 自動清除舊的標準索引，改用多層次索引（實驗組B/C/D）")
                    faiss_store.reset_vectors()
                    bm25_index.reset_index()
                    use_standard_index = False
                else:
                    print(f"❌ 維度不匹配: 查詢向量維度={query_dim}, FAISS索引維度={expected_dim}")
                    print(f"💡 解決方案: 請重新運行 /api/embed 或 /api/multi-level-embed 以統一維度")
                    raise ValueError(
                        f"Query vector dimension ({query_dim}) does not match FAISS index dimension ({expected_dim}). "
                        f"Please re-run /api/embed or /api/multi-level-embed to regenerate embeddings with the same dimension. "
                        f"Current EMBEDDING_DIMENSION setting: {EMBEDDING_DIMENSION}"
                    )
        
        all_candidates = {}
        
        # 1. 向量檢索（支持標準和多層次）
        # 記錄向量檢索開始時間
        vector_retrieval_start = time.time()
        vector_retrieval_time = 0.0
        print("📊 執行向量檢索...")
        # 優先使用多層次索引（如果存在），因為這是實驗組B、C、D使用的索引
        # 只有在沒有多層次索引時才使用標準索引
        if use_multi_level_index:
            # 多層次索引檢索：檢索所有層次並合併（實驗組B、C、D）
            print(f"✅ 使用多層次索引進行檢索（實驗組B/C/D）")
            available_levels = faiss_store.get_available_levels()
            print(f"🔍 多層次索引可用層次: {available_levels}")
            
            # 根據可用層次推斷實驗組並過濾層次
            filtered_levels = available_levels.copy()
            level_set = set(available_levels)
            
            print(f"📊 開始推斷實驗組，可用層次: {available_levels}, level_set: {level_set}")
            
            # 優先檢測實驗組D：包含所有6個層次（document, document_component, basic_unit_hierarchy, basic_unit, basic_unit_component, enumeration）
            expected_d_levels = {"document", "document_component", "basic_unit_hierarchy", "basic_unit", "basic_unit_component", "enumeration"}
            if level_set == expected_d_levels or len(level_set) == 6:
                filtered_levels = available_levels
                print(f"🎯 推斷為實驗組D（完整多層次ML-RAG），檢索所有層次: {filtered_levels}")
            # 推斷實驗組A：只有basic_unit
            elif level_set == {"basic_unit"}:
                filtered_levels = ["basic_unit"]
                print(f"🎯 推斷為實驗組A（僅條文層），只檢索: {filtered_levels}")
            # 推斷實驗組B：basic_unit_hierarchy + basic_unit
            elif level_set == {"basic_unit_hierarchy", "basic_unit"}:
                filtered_levels = ["basic_unit_hierarchy", "basic_unit"]
                print(f"🎯 推斷為實驗組B（條文+章節結構），只檢索: {filtered_levels}")
            # 推斷實驗組C：basic_unit + basic_unit_component + enumeration（不包含basic_unit_hierarchy、document、document_component）
            elif "basic_unit_component" in level_set and "enumeration" in level_set and "basic_unit" in level_set:
                # 實驗組C的特徵：必須沒有basic_unit_hierarchy、document、document_component
                if "basic_unit_hierarchy" not in level_set and "document" not in level_set and "document_component" not in level_set:
                    filtered_levels = ["basic_unit", "basic_unit_component", "enumeration"]
                    print(f"🎯 推斷為實驗組C（條文+細節層次），只檢索: {filtered_levels}")
                else:
                    # 如果包含不應該有的層次，過濾掉
                    filtered_levels = [l for l in available_levels if l in ["basic_unit", "basic_unit_component", "enumeration"]]
                    removed_levels = [l for l in level_set if l not in ["basic_unit", "basic_unit_component", "enumeration"]]
                    print(f"🎯 檢測到實驗組C特徵，但包含不應有的層次")
                    print(f"   ❌ 過濾掉不應有的層次: {removed_levels}")
                    print(f"   ✅ 實驗組C只檢索: {filtered_levels}")
            # 其他情況：可能是實驗組B（如果有basic_unit_hierarchy和basic_unit）
            elif "basic_unit_hierarchy" in level_set and "basic_unit" in level_set:
                # 實驗組B，過濾掉其他層次
                filtered_levels = ["basic_unit_hierarchy", "basic_unit"]
                removed_levels = [l for l in level_set if l not in ["basic_unit_hierarchy", "basic_unit"]]
                if removed_levels:
                    print(f"🎯 推斷為實驗組B，過濾掉其他層次: {removed_levels}")
                print(f"   ✅ 實驗組B只檢索: {filtered_levels}")
            else:
                # 無法確定實驗組，使用所有可用層次
                print(f"⚠️ 無法推斷實驗組，使用所有可用層次: {filtered_levels}")
            
            for level_name in filtered_levels:
                try:
                    level_indices, level_scores = faiss_store.search_multi_level(level_name, query_vector, req.k * 10)
                    print(f"   ✅ 層次 '{level_name}' 返回 {len(level_indices)} 個候選")
                    
                    # 為該層次的結果分配rank
                    for rank, (idx, score) in enumerate(zip(level_indices, level_scores), start=1):
                        chunk_info = faiss_store.get_multi_level_chunk_by_index(level_name, idx)
                        if chunk_info and 'chunk_id' in chunk_info:
                            chunk_id = chunk_info['chunk_id']
                            # 跨層次可能會有相同chunk_id，使用第一個層次的排名
                            if chunk_id not in all_candidates:
                                # 從multi_level_chunks中獲取原始metadata（如果可用）
                                doc_id = chunk_info.get('doc_id', 'unknown')
                                content = chunk_info.get('content', '')
                                enhanced_metadata = chunk_info.get('enhanced_metadata', {})
                                
                                # 嘗試從doc的multi_level_chunks中找到對應的chunk以獲取原始metadata
                                original_metadata = {}
                                doc = store.docs.get(doc_id) if doc_id != 'unknown' else None
                                if doc and hasattr(doc, 'multi_level_chunks') and doc.multi_level_chunks:
                                    if level_name in doc.multi_level_chunks:
                                        doc_level_chunks = doc.multi_level_chunks[level_name]
                                        # 通過content精確匹配找到對應的chunk（最可靠的方法）
                                        # 因為multi_level_chunks中的content和檢索返回的content應該完全一致
                                        matched = False
                                        for chunk_data in doc_level_chunks:
                                            chunk_content = chunk_data.get('content', '')
                                            # 精確匹配或前200字符匹配（考慮可能的微小差異）
                                            if chunk_content == content or (
                                                len(chunk_content) > 100 and 
                                                len(content) > 100 and
                                                chunk_content[:200] == content[:200]
                                            ):
                                                original_metadata = chunk_data.get('metadata', {})
                                                matched = True
                                                print(f"📋 chunk_id {chunk_id}: 通過content匹配找到metadata - 章:{original_metadata.get('chapter', '')}, 節:{original_metadata.get('section', '')}, 條:{original_metadata.get('article', '')}")
                                                break
                                        
                                        # 如果仍然沒找到，嘗試通過chunk_id中的索引推算
                                        if not matched:
                                            try:
                                                import re
                                                match = re.search(r'_(\d+)$', chunk_id)
                                                if match:
                                                    global_idx = int(match.group(1))
                                                    # 統計在level_chunks中，屬於當前doc_id的chunks數量（到global_idx為止）
                                                    # 通過store.multi_level_chunk_doc_ids來統計
                                                    if level_name in store.multi_level_chunk_doc_ids:
                                                        level_doc_ids = store.multi_level_chunk_doc_ids[level_name]
                                                        # 統計前global_idx個chunks中屬於當前doc_id的數量
                                                        doc_chunk_count = sum(1 for i in range(min(global_idx + 1, len(level_doc_ids))) if level_doc_ids[i] == doc_id)
                                                        relative_idx = doc_chunk_count - 1  # 減1因為當前chunk是第doc_chunk_count個
                                                        if 0 <= relative_idx < len(doc_level_chunks):
                                                            original_metadata = doc_level_chunks[relative_idx].get('metadata', {})
                                                            print(f"📋 chunk_id {chunk_id}: 通過索引推算找到metadata - 章:{original_metadata.get('chapter', '')}, 節:{original_metadata.get('section', '')}, 條:{original_metadata.get('article', '')}")
                                            except (ValueError, IndexError, AttributeError) as e:
                                                print(f"⚠️ 解析chunk_id失敗: {chunk_id}, 錯誤: {e}")
                                
                                all_candidates[chunk_id] = {
                                    'chunk_id': chunk_id,
                                    'doc_id': doc_id,
                                    'content': content,
                                    'enhanced_metadata': enhanced_metadata,
                                    'original_metadata': original_metadata,  # 保存原始metadata
                                    'chunk_index': idx,
                                    'level': level_name,
                                    'vector_rank': rank,
                                    'vector_score': float(score),
                                    'bm25_rank': None,
                                    'bm25_score': 0.0
                                }
                except Exception as e:
                    print(f"   ⚠️ 層次 '{level_name}' 檢索失敗: {e}")
        elif use_standard_index:
            # 標準索引檢索（實驗組A）
            print(f"✅ 使用標準索引進行檢索（實驗組A）")
            vector_indices, vector_scores = faiss_store.search(query_vector, req.k * 10)
            print(f"✅ 標準向量檢索返回 {len(vector_indices)} 個候選")
            
            # 為向量結果分配rank
            for rank, (idx, score) in enumerate(zip(vector_indices, vector_scores), start=1):
                chunk_info = faiss_store.get_chunk_by_index(idx)
                if chunk_info and 'chunk_id' in chunk_info:
                    chunk_id = chunk_info['chunk_id']
                    all_candidates[chunk_id] = {
                        'chunk_id': chunk_id,
                        'doc_id': chunk_info.get('doc_id', 'unknown'),
                        'content': chunk_info.get('content', ''),
                        'enhanced_metadata': chunk_info.get('enhanced_metadata', {}),
                        'chunk_index': idx,
                        'level': 'standard',
                        'vector_rank': rank,
                        'vector_score': float(score),
                        'bm25_rank': None,
                        'bm25_score': 0.0
                    }
        else:
            print("⚠️ FAISS索引不可用，跳過向量檢索")
        
        # 記錄向量檢索結束時間
        vector_retrieval_time = time.time() - vector_retrieval_start
        print(f"⏱️  向量檢索花費時間: {vector_retrieval_time:.3f} 秒")
        
        # 2. BM25檢索（支持標準和多層次）
        # 記錄BM25檢索開始時間
        bm25_retrieval_start = time.time()
        bm25_retrieval_time = 0.0
        print("📊 執行BM25檢索...")
        # 優先使用多層次索引（如果存在）
        if bm25_index.has_multi_level_index():
            # 多層次BM25檢索：檢索所有層次並合併（實驗組B、C、D）
            print(f"✅ 使用多層次BM25索引進行檢索（實驗組B/C/D）")
            available_bm25_levels = bm25_index.get_available_levels()
            print(f"🔍 多層次BM25索引可用層次: {available_bm25_levels}")
            
            # 使用與向量檢索相同的過濾邏輯
            # 優先檢測實驗組C：包含basic_unit_component和enumeration，但不應該包含basic_unit_hierarchy
            bm25_levels = available_bm25_levels.copy()
            bm25_level_set = set(available_bm25_levels)
            
            # 使用與向量檢索完全相同的推斷邏輯
            if "basic_unit_component" in bm25_level_set and "enumeration" in bm25_level_set:
                # 實驗組C：必須過濾掉basic_unit_hierarchy和其他不應該有的層次
                bm25_levels = [l for l in available_bm25_levels if l in ["basic_unit", "basic_unit_component", "enumeration"]]
                if "basic_unit_hierarchy" in bm25_level_set or "document" in bm25_level_set or "document_component" in bm25_level_set:
                    removed_levels = [l for l in bm25_level_set if l not in ["basic_unit", "basic_unit_component", "enumeration"]]
                    print(f"🎯 BM25檢索：檢測到實驗組C特徵，過濾掉不應有的層次: {removed_levels}")
                    print(f"   ✅ BM25只檢索: {bm25_levels}")
            elif bm25_level_set == {"basic_unit"}:
                bm25_levels = ["basic_unit"]
            elif bm25_level_set == {"basic_unit_hierarchy", "basic_unit"}:
                bm25_levels = ["basic_unit_hierarchy", "basic_unit"]
            elif len(bm25_level_set) == 6 or ("document" in bm25_level_set and len(bm25_level_set) >= 4):
                bm25_levels = available_bm25_levels
            elif "basic_unit_hierarchy" in bm25_level_set and "basic_unit" in bm25_level_set:
                bm25_levels = ["basic_unit_hierarchy", "basic_unit"]
            
            for level_name in bm25_levels:
                try:
                    level_indices, level_scores = bm25_index.search_multi_level(level_name, req.query, req.k * 10)
                    print(f"   ✅ 層次 '{level_name}' BM25返回 {len(level_indices)} 個候選")
                    
                    # 為該層次的結果分配rank並合併
                    for rank, (idx, score) in enumerate(zip(level_indices, level_scores), start=1):
                        chunk_info = bm25_index.get_multi_level_chunk_by_index(level_name, idx)
                        if chunk_info and 'chunk_id' in chunk_info:
                            chunk_id = chunk_info['chunk_id']
                            if chunk_id in all_candidates:
                                all_candidates[chunk_id]['bm25_rank'] = rank
                                all_candidates[chunk_id]['bm25_score'] = float(score)
                            else:
                                all_candidates[chunk_id] = {
                                    'chunk_id': chunk_id,
                                    'doc_id': chunk_info.get('doc_id', 'unknown'),
                                    'content': chunk_info.get('content', ''),
                                    'enhanced_metadata': {},
                                    'chunk_index': idx,
                                    'level': level_name,
                                    'vector_rank': None,
                                    'vector_score': 0.0,
                                    'bm25_rank': rank,
                                    'bm25_score': float(score)
                                }
                except Exception as e:
                    print(f"   ⚠️ 層次 '{level_name}' BM25檢索失敗: {e}")
        elif bm25_index.has_index():
            # 標準BM25檢索
            bm25_indices, bm25_scores = bm25_index.search(req.query, req.k * 10)
            print(f"✅ 標準BM25檢索返回 {len(bm25_indices)} 個候選")
            
            # 為BM25結果分配rank並合併
            for rank, (idx, score) in enumerate(zip(bm25_indices, bm25_scores), start=1):
                chunk_info = bm25_index.get_chunk_by_index(idx)
                if chunk_info and 'chunk_id' in chunk_info:
                    chunk_id = chunk_info['chunk_id']
                    if chunk_id in all_candidates:
                        all_candidates[chunk_id]['bm25_rank'] = rank
                        all_candidates[chunk_id]['bm25_score'] = float(score)
                    else:
                        all_candidates[chunk_id] = {
                            'chunk_id': chunk_id,
                            'doc_id': chunk_info.get('doc_id', 'unknown'),
                            'content': chunk_info.get('content', ''),
                            'enhanced_metadata': {},
                            'chunk_index': idx,
                            'level': 'standard',
                            'vector_rank': None,
                            'vector_score': 0.0,
                            'bm25_rank': rank,
                            'bm25_score': float(score)
                        }
        else:
            print("⚠️ BM25索引不可用，跳過BM25檢索")
        
        # 記錄BM25檢索結束時間
        bm25_retrieval_time = time.time() - bm25_retrieval_start
        print(f"⏱️  BM25檢索花費時間: {bm25_retrieval_time:.3f} 秒")
        
        # 3. RRF融合 - 計算RRF分數：1 / (60 + rank)
        fusion_start = time.time()
        fusion_time = 0.0
        k_rrf = 60
        for chunk_id, candidate in all_candidates.items():
            rrf_score = 0.0
            
            # 向量排名分數
            if candidate['vector_rank'] is not None:
                rrf_score += 1.0 / (k_rrf + candidate['vector_rank'])
            
            # BM25排名分數
            if candidate['bm25_rank'] is not None:
                rrf_score += 1.0 / (k_rrf + candidate['bm25_rank'])
            
            candidate['rrf_score'] = rrf_score
            candidate['hybrid_score'] = rrf_score
            
            # 添加分數分解
            candidate['score_breakdown'] = {
                'vector_rank': candidate['vector_rank'],
                'bm25_rank': candidate['bm25_rank'],
                'rrf_score': rrf_score
            }
        
        # 檢查是否有候選結果
        if not all_candidates:
            print("⚠️ 沒有找到任何候選結果")
            return {
                "results": [],
                "query": req.query,
                "final_results": 0,
                "fusion_method": "RRF",
                "k_rrf": 60,
                "warning": "No candidates found from vector or BM25 search"
            }
        
        # 按RRF分數排序
        final_results = sorted(all_candidates.values(), key=lambda x: x['rrf_score'], reverse=True)
        final_results = final_results[:req.k]
        
        # 生成層級描述
        for result in final_results:
            if 'doc_id' in result:
                doc_id = result.get('doc_id', 'unknown')
                level = result.get('level', 'basic_unit')  # 使用實際層級
                content = result.get('content', '')
                original_metadata = result.get('original_metadata', {})
                
                # 對於多層級檢索，優先使用content和original_metadata來生成描述
                if original_metadata:
                    # 從original_metadata生成描述
                    hierarchical_desc = generate_hierarchical_description_from_metadata(
                        doc_id, original_metadata, content, store
                    )
                    result['hierarchical_description'] = hierarchical_desc
                else:
                    # 回退到舊的方法（標準索引）
                    chunk_index = result.get('chunk_index', 0)
                    result['hierarchical_description'] = generate_hierarchical_description(
                        doc_id, level, chunk_index, store
                    )
        
        # 記錄融合時間
        fusion_time = time.time() - fusion_start
        print(f"⏱️  RRF融合花費時間: {fusion_time:.3f} 秒")
        
        # 計算總時間
        total_retrieval_time = time.time() - retrieval_start_time
        
        print(f"✅ HybridRAG(RRF)檢索完成，返回 {len(final_results)} 個結果")
        print(f"{'='*60}")
        print(f"⏱️  檢索總花費時間: {total_retrieval_time:.3f} 秒")
        print(f"    - 查詢 Embedding: {query_embedding_time:.3f} 秒")
        print(f"    - 向量檢索: {vector_retrieval_time:.3f} 秒")
        print(f"    - BM25檢索: {bm25_retrieval_time:.3f} 秒")
        print(f"    - RRF融合: {fusion_time:.3f} 秒")
        print(f"{'='*60}")
        
        return {
            "results": final_results,
            "query": req.query,
            "final_results": len(final_results),
            "fusion_method": "RRF",
            "k_rrf": k_rrf,
            "timing": {
                "total_time": round(total_retrieval_time, 3),
                "query_embedding_time": round(query_embedding_time, 3),
                "vector_retrieval_time": round(vector_retrieval_time, 3),
                "bm25_retrieval_time": round(bm25_retrieval_time, 3),
                "fusion_time": round(fusion_time, 3)
            }
        }
        
    except Exception as e:
        print(f"❌ HybridRAG(RRF)檢索失敗: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"HybridRAG(RRF) retrieval failed: {str(e)}"}
        )


@app.post("/api/enhanced-multi-level-hybrid-retrieve")
def enhanced_multi_level_hybrid_retrieve(req: MultiLevelFusionRequest):
    """使用增強版HybridRAG進行多層次檢索"""
    print(f"🚀 增強版多層次HybridRAG檢索請求: {req.query}, k={req.k}")
    
    # 檢查是否有多層次索引
    if not faiss_store.has_multi_level_vectors() and not bm25_index.has_multi_level_index():
        return JSONResponse(
            status_code=400,
            content={"error": "No multi-level indices available. Please run /api/multi-level-embed first."}
        )
    
    try:
        # 配置增強版HybridRAG
        config = EnhancedHybridConfig(
            vector_weight=0.6,
            bm25_weight=0.25,
            metadata_weight=0.15,
            w_law_match=0.15,
            w_article_match=0.15,
            w_concept_match=0.1,
            w_keyword_hit=0.05,
            w_domain_match=0.05,
            w_title_match=0.1,
            w_category_match=0.05,
            max_bonus=0.4,
            title_boost_factor=1.5,
            category_boost_factor=1.3,
            # Metadata向下繼承配置
            enable_inheritance_strategy=True,
            metadata_match_threshold=0.3,
            inheritance_bonus=0.1,
            inheritance_boost_factor=1.2
        )
        
        # 執行多層次檢索
        level_results = {}
        available_levels = faiss_store.get_available_levels()
        
        for level_name in available_levels:
            try:
                level_results[level_name] = enhanced_hybrid_rag.retrieve_multi_level(
                    req.query, level_name, req.k, config
                )
                print(f"✅ 層次 '{level_name}' 檢索完成，返回 {len(level_results[level_name])} 個結果")
            except Exception as e:
                print(f"⚠️ 層次 '{level_name}' 檢索失敗: {e}")
                level_results[level_name] = []
        
        # 使用融合策略合併結果
        fusion_config = FusionConfig(
            strategy=req.fusion_strategy,
            level_weights=req.level_weights,
            similarity_threshold=req.similarity_threshold,
            max_results=req.max_results,
            normalize_scores=req.normalize_scores
        )
        
        # 轉換為融合器期望的格式
        formatted_level_results = {}
        for level_name, results in level_results.items():
            formatted_level_results[level_name] = []
            for result in results:
                formatted_result = {
                    "content": result.get("content", ""),
                    "similarity": result.get("hybrid_score", 0.0),
                    "metadata": result.get("enhanced_metadata", {}),
                    "hierarchical_description": result.get("hierarchical_description", "")
                }
                formatted_level_results[level_name].append(formatted_result)
        
        # 執行融合
        fusion = MultiLevelResultFusion(fusion_config)
        fused_results = fusion.fuse_results(formatted_level_results)
        
        print(f"✅ 增強版多層次HybridRAG檢索完成，融合後返回 {len(fused_results)} 個結果")
        
        return {
            "results": fused_results,
            "query": req.query,
            "level_results": {k: len(v) for k, v in level_results.items()},
            "final_results": len(fused_results),
            "fusion_config": fusion_config.__dict__,
            "retrieval_stats": enhanced_hybrid_rag.get_retrieval_stats()
        }
        
    except Exception as e:
        print(f"❌ 增強版多層次HybridRAG檢索失敗: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Enhanced multi-level HybridRAG retrieval failed: {str(e)}"}
        )


def clean_markdown_from_answer(answer: str) -> str:
    """清理答案中的 Markdown 格式，转换为纯文本"""
    import re
    
    if not answer:
        return answer
    
    # 移除 Markdown 标题符号 (# ## ###)
    answer = re.sub(r'^#{1,6}\s+', '', answer, flags=re.MULTILINE)
    
    # 移除粗体 (**text** 或 __text__)
    answer = re.sub(r'\*\*(.+?)\*\*', r'\1', answer)
    answer = re.sub(r'__(.+?)__', r'\1', answer)
    
    # 移除斜体 (*text* 或 _text_)
    answer = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', answer)
    answer = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'\1', answer)
    
    # 移除代码块 (```code```)
    answer = re.sub(r'```[\s\S]*?```', '', answer)
    
    # 移除行内代码 (`code`)
    answer = re.sub(r'`([^`]+)`', r'\1', answer)
    
    # 移除 Markdown 列表符号，保留内容
    # 处理 - * + 开头的列表项
    answer = re.sub(r'^[\s]*[-*+]\s+', '', answer, flags=re.MULTILINE)
    
    # 处理数字列表，保留数字但移除 Markdown 格式
    # 将 1. 2. 3. 转换为 1. 2. 3. (已经是纯文本格式)
    
    # 移除多余的空白行
    answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
    
    # 清理行首行尾空格
    answer = '\n'.join(line.strip() for line in answer.split('\n'))
    
    return answer.strip()


async def gemini_chat(messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
    if not httpx:
        raise RuntimeError("httpx not available")
    
    # 優先使用 GOOGLE_API_KEY，如果沒有則使用 GEMINI_API_KEY
    api_key = GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
    
    model_name = model or os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.0-flash")
    # Use Generative Language API: models/{model}:generateContent
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    # Convert messages to Gemini format
    contents = []
    for message in messages:
        contents.append({
            "parts": [{"text": message.get("content", "")}],
            "role": "user" if message.get("role") == "user" else "model"
        })
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096,  # 增加token限制，避免答案被截断
            "topP": 0.95,
            "topK": 40
        }
    }
    
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=120) as client:  # 增加超时时间
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        # Extract response from new format
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            
            # 检查是否因为token限制而截断
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "MAX_TOKENS":
                print("⚠️ 警告：答案可能因token限制而被截断")
            
            if "content" in candidate and "parts" in candidate["content"]:
                answer_text = candidate["content"]["parts"][0].get("text", "").strip()
                
                # 如果答案被截断，添加提示
                if finish_reason == "MAX_TOKENS" and answer_text:
                    answer_text += "\n\n（注意：答案可能因长度限制而被截断，如需完整答案请重新提问或分段询问）"
                
                return answer_text
        return "No response generated"


def simple_extractive_answer(query: str, contexts: List[str]) -> str:
    """針對中英文改進的極簡抽取式回答：
    - 支援中文斷句（。！？；）與換行
    - 分詞同時考慮英文/數字詞與中文單字
    - 若無明顯重疊，回退輸出前幾句最前面的內容
    """
    import re
    from collections import Counter

    # 1) 斷句（同時支援中英標點與換行）
    def split_sentences(text: str) -> List[str]:
        # 保留原文片段，避免過度切碎
        # 先按換行拆，再按中文/英文句末標點細分
        parts: List[str] = []
        for seg in re.split(r"[\n\r]+", text):
            seg = seg.strip()
            if not seg:
                continue
            parts.extend([s.strip() for s in re.split(r"(?<=[。！？!?；;])\s+", seg) if s.strip()])
        return parts

    # 2) 簡單分詞：英文/數字詞 + 中文單字
    def tokenize(text: str) -> List[str]:
        text_norm = text.lower()
        en = re.findall(r"[a-z0-9_]+", text_norm)
        zh = re.findall(r"[\u4e00-\u9fff]", text_norm)
        return en + zh

    q_tokens = set(tokenize(query))
    if not q_tokens:
        q_tokens = set(query.lower())  # 退化為字符集合

    # 3) 聚合所有上下文的句子
    sents: List[str] = []
    for ctx in contexts:
        sents.extend(split_sentences(ctx))

    # 4) 計分：重疊 token 數量 + 輕度長度平衡
    counts = Counter()
    for s in sents:
        t = tokenize(s)
        if not t:
            continue
        overlap = len(set(t) & q_tokens)
        if overlap > 0:
            # 輕度鼓勵較完整句子
            counts[s] = overlap + min(len(s) / 200.0, 1.0)

    # 5) 回傳：有匹配則取前5句，否則回退取最前面內容
    if counts:
        best = [s for s, _ in counts.most_common(5)]
        return " \n".join(best)

    # 回退：取前兩段的前兩句
    fallback: List[str] = []
    for ctx in contexts[:2]:
        ss = split_sentences(ctx)
        fallback.extend(ss[:2])
        if len(fallback) >= 4:
            break
    if fallback:
        return " \n".join(fallback[:4])
    return "No relevant answer found in context."


@app.post("/api/generate")
def generate(req: GenerateRequest):
    try:
        # 驗證請求參數
        if not req.query or not req.query.strip():
            return JSONResponse(status_code=400, content={"error": "query 不能為空"})
        
        if req.top_k <= 0:
            return JSONResponse(status_code=400, content={"error": "top_k 必須大於 0"})
        
        # 使用 HybridRAG（向量檢索 + metadata 關鍵字加分）取得生成上下文
        # 檢查標準索引和多層次索引
        faiss_standard_has_vectors = faiss_store.has_vectors()
        faiss_multi_level_has_vectors = faiss_store.has_multi_level_vectors()
        faiss_standard_has_chunks = faiss_standard_has_vectors and len(faiss_store.chunks_flat) > 0
        store_has_embeddings = store.embeddings is not None
        store_has_chunks = store.chunks_flat and len(store.chunks_flat) > 0
        
        # 檢查多層次索引是否有chunks（需要檢查所有層次）
        faiss_multi_level_has_chunks = False
        multi_level_chunks_count = 0
        if faiss_multi_level_has_vectors:
            available_levels = faiss_store.get_available_levels()
            for level in available_levels:
                level_chunks = faiss_store.multi_level_chunks_flat.get(level, [])
                multi_level_chunks_count += len(level_chunks)
            faiss_multi_level_has_chunks = multi_level_chunks_count > 0
        
        # 調試信息
        print(f"🔍 Generate 檢查狀態:")
        print(f"   FAISS標準索引 has_vectors: {faiss_standard_has_vectors}")
        print(f"   FAISS標準索引 has_chunks: {faiss_standard_has_chunks} (chunks數量: {len(faiss_store.chunks_flat) if faiss_standard_has_vectors else 0})")
        print(f"   FAISS多層次索引 has_vectors: {faiss_multi_level_has_vectors}")
        print(f"   FAISS多層次索引 has_chunks: {faiss_multi_level_has_chunks} (總chunks數量: {multi_level_chunks_count})")
        if faiss_multi_level_has_vectors:
            available_levels = faiss_store.get_available_levels()
            print(f"   多層次索引可用層次: {available_levels}")
        print(f"   Store has_embeddings: {store_has_embeddings}")
        print(f"   Store has_chunks: {store_has_chunks} (chunks數量: {len(store.chunks_flat) if store.chunks_flat else 0})")
        
        # 檢查是否有可用的數據（優先多層次索引，然後標準索引，最後store）
        use_multi_level = faiss_multi_level_has_chunks
        use_standard = not use_multi_level and faiss_standard_has_chunks
        use_store = not use_multi_level and not use_standard and store_has_chunks
        
        if not use_multi_level and not use_standard and not use_store:
            error_msg = "沒有可用的 embeddings 和 chunks"
            if faiss_multi_level_has_vectors and not faiss_multi_level_has_chunks:
                error_msg = "FAISS 多層次索引有向量但缺少 chunks，請重新運行 /api/multi-level-embed"
            elif faiss_standard_has_vectors and not faiss_standard_has_chunks:
                error_msg = "FAISS 標準索引有向量但缺少 chunks，請重新運行 /api/embed"
            elif store_has_embeddings and not store_has_chunks:
                error_msg = "Store 有 embeddings 但缺少 chunks，請重新運行 /api/embed"
            else:
                error_msg = "沒有可用的 embeddings 和 chunks，請先運行 /api/embed 或 /api/multi-level-embed"
            
            return JSONResponse(status_code=400, content={"error": error_msg})

        # 構建 nodes（與 /api/hybrid-retrieve 保持一致）
        # 優先使用多層次索引，然後標準索引，最後store
        if use_multi_level:
            # 使用多層次索引：合併所有層次的chunks
            available_levels = faiss_store.get_available_levels()
            chunks_flat = []
            mapping_doc_ids = []
            for level in available_levels:
                level_chunks = faiss_store.multi_level_chunks_flat.get(level, [])
                level_doc_ids = faiss_store.multi_level_chunk_doc_ids.get(level, [])
                chunks_flat.extend(level_chunks)
                mapping_doc_ids.extend(level_doc_ids)
            print(f"✅ 使用 FAISS 多層次索引數據: {len(chunks_flat)} chunks (來自 {len(available_levels)} 個層次)")
        elif use_standard:
            chunks_flat = faiss_store.chunks_flat
            mapping_doc_ids = faiss_store.chunk_doc_ids
            print(f"✅ 使用 FAISS 標準索引數據: {len(chunks_flat)} chunks")
        else:
            chunks_flat = store.chunks_flat
            mapping_doc_ids = store.chunk_doc_ids
            print(f"✅ 使用 Store 數據: {len(chunks_flat)} chunks")
            
        if not chunks_flat or len(chunks_flat) == 0:
            return JSONResponse(status_code=400, content={"error": "no chunks available"})

        nodes = []
        for i, (chunk, doc_id) in enumerate(zip(chunks_flat, mapping_doc_ids)):
            doc = store.docs.get(doc_id)
            metadata = {}
            
            # 優先使用 FAISS 的 enhanced_metadata
            if use_multi_level:
                # 多層次索引：嘗試從所有層次中查找enhanced_metadata
                # 由於chunks已合併，需要遍歷所有層次查找對應的chunk
                metadata = {}
                found_metadata = False
                available_levels = faiss_store.get_available_levels()
                
                # 嘗試從每個層次查找對應索引的chunk
                for level in available_levels:
                    level_chunks = faiss_store.multi_level_chunks_flat.get(level, [])
                    level_doc_ids = faiss_store.multi_level_chunk_doc_ids.get(level, [])
                    
                    # 查找當前doc_id在該層次中的索引位置
                    for level_idx, level_doc_id in enumerate(level_doc_ids):
                        if level_doc_id == doc_id and level_idx < len(level_chunks):
                            # 檢查是否是同一個chunk（通過內容匹配）
                            if level_chunks[level_idx] == chunk:
                                chunk_info = faiss_store.get_multi_level_chunk_by_index(level, level_idx)
                                if chunk_info and chunk_info.get("enhanced_metadata"):
                                    metadata = chunk_info["enhanced_metadata"]
                                    found_metadata = True
                                    break
                    
                    if found_metadata:
                        break
                
                # 如果沒有找到enhanced_metadata，從doc的structured_chunks獲取
                if not found_metadata and doc and hasattr(doc, 'structured_chunks') and doc.structured_chunks:
                    # 嘗試通過內容匹配找到對應的structured_chunk
                    for structured_chunk in doc.structured_chunks:
                        if structured_chunk.get("content") == chunk:
                            metadata = structured_chunk.get("metadata", {})
                            found_metadata = True
                            break
            elif use_standard:
                chunk_info = faiss_store.get_chunk_by_index(i)
                if chunk_info and chunk_info.get("enhanced_metadata"):
                    metadata = chunk_info["enhanced_metadata"]
                # 如果 FAISS 沒有 enhanced_metadata，嘗試從 doc 的 structured_chunks 獲取
                elif doc and hasattr(doc, 'structured_chunks') and doc.structured_chunks and i < len(doc.structured_chunks):
                    structured_chunk = doc.structured_chunks[i]
                    metadata = structured_chunk.get("metadata", {})
            else:
                # 使用 store 的數據時，從 doc 的 structured_chunks 獲取
                if doc and hasattr(doc, 'structured_chunks') and doc.structured_chunks and i < len(doc.structured_chunks):
                    structured_chunk = doc.structured_chunks[i]
                    metadata = structured_chunk.get("metadata", {})
            
            nodes.append({
                "content": chunk,
                "metadata": metadata,
                "doc_id": doc_id,
                "chunk_index": i
            })

        # 先用密集向量計算所有節點的相似度，取前 N 做 Hybrid 候選
        dense_top_k = min(len(nodes), max(req.top_k * 4, req.top_k))
        all_vec_idxs, all_vec_sims = rank_with_dense_vectors(req.query, k=len(nodes))
        node_vector_scores = [0.0] * len(nodes)
        for rank_idx, node_idx in enumerate(all_vec_idxs):
            node_vector_scores[node_idx] = float(all_vec_sims[rank_idx])
        top_vec_pairs = sorted(
            [(i, s) for i, s in enumerate(node_vector_scores)], key=lambda x: x[1], reverse=True
        )[:dense_top_k]
        candidate_nodes = [nodes[i] for i, _ in top_vec_pairs]
        candidate_scores = [s for _, s in top_vec_pairs]

        config = HybridConfig(
            alpha=0.8,
            w_law_match=0.15,
            w_article_match=0.15,
            w_keyword_hit=0.05,
            max_bonus=0.4,
        )
        hybrid_results = hybrid_rank(req.query, candidate_nodes, k=req.top_k, config=config, vector_scores=candidate_scores)

        # 生成使用的結果
        results = []
        for rank, item in enumerate(hybrid_results, start=1):
            result = {
                "rank": rank,
                "score": item.get("score"),
                "vector_score": item.get("vector_score"),
                "bonus": item.get("bonus"),
                "doc_id": item.get("doc_id"),
                "chunk_index": item.get("chunk_index"),
                "content": item.get("content"),
            }
            md = (item.get("metadata") or {})
            if md:
                result["legal_structure"] = {
                    "id": md.get("id", ""),
                    "category": md.get("category", ""),
                    "article_label": md.get("article_label", ""),
                    "article_number": md.get("article_number"),
                    "article_suffix": md.get("article_suffix"),
                    "spans": md.get("spans", {}),
                    "page_range": md.get("page_range", {}),
                }
            results.append(result)
        contexts = [item["content"] for item in results]

        # 構建結構化上下文信息
        structured_context = []
        legal_references = []
        
        for item in results:
            context_text = item["content"]
            
            # 如果有法律結構信息，添加到上下文中
            if "legal_structure" in item:
                legal_info = item["legal_structure"]
                law_name = legal_info.get("law_name", "")
                article = legal_info.get("article", "")
                item_ref = legal_info.get("item", "")
                sub_item = legal_info.get("sub_item", "")
                chunk_type = legal_info.get("chunk_type", "")
                
                # 構建法律引用
                legal_ref = f"{law_name}"
                if article:
                    legal_ref += f" {article}"
                if item_ref:
                    legal_ref += f" {item_ref}"
                if sub_item:
                    legal_ref += f" {sub_item}"
                
                if legal_ref not in legal_references:
                    legal_references.append(legal_ref)
                
                # 添加結構化上下文
                structured_context.append(f"[{legal_ref}] {context_text}")
            else:
                structured_context.append(context_text)

        reasoning_steps = [
            {"type": "plan", "text": "Read query, identify entities and constraints."},
            {"type": "gather", "text": f"Collect top-{req.top_k} chunks as context."},
            {"type": "analyze", "text": f"Analyze legal structure: {', '.join(legal_references[:3])}."},
            {"type": "synthesize", "text": "Synthesize answer grounded in retrieved text with legal references."},
        ]

        if USE_GEMINI_COMPLETION:
            # 構建包含法律結構信息的prompt
            system_prompt = """你是一個專業的法律助手。請基於提供的法律文檔內容回答問題。

重要要求：
1. 只使用提供的上下文內容回答問題
2. 如果答案涉及具體法律條文，請引用相關的法規名稱和條文號碼
3. 如果信息不足，請明確說明你不知道
4. 回答要準確、專業，符合法律文檔的表述方式
5. 使用純文本格式回答，不要使用 Markdown 格式（如 **粗體**、*斜體*、# 標題等）
6. 使用簡單的列表格式（如：1. 2. 3. 或 • • •），不要使用 Markdown 列表語法
7. 確保答案完整，不要在中途截斷
8. 如果答案較長，請確保所有要點都完整表達"""

            user_content = f"問題: {req.query}\n\n"
            
            if legal_references:
                user_content += f"相關法規: {', '.join(legal_references)}\n\n"
            
            user_content += "法律文檔內容:\n" + "\n---\n".join(structured_context)
            
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            
            try:
                answer = asyncio_run(gemini_chat(prompt))
                # 清理可能的 Markdown 格式
                answer = clean_markdown_from_answer(answer)
            except Exception as e:
                answer = f"Gemini調用失敗: {e}. 回退到提取式回答。\n" + simple_extractive_answer(req.query, contexts)
        else:
            answer = simple_extractive_answer(req.query, contexts)

        return {
            "query": req.query,
            "answer": answer,
            "contexts": results,
            "legal_references": legal_references,
            "steps": reasoning_steps,
        }
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"❌ Generate 錯誤: {error_msg}")
        print(f"❌ 錯誤堆疊: {traceback_str}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"生成答案時發生錯誤: {error_msg}",
                "details": traceback_str
            }
        )


def merge_law_documents(law_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    整合多個法律文檔成一個統一的JSON結構
    完全按照單個PDF轉換的方式來構建結構
    
    參數:
    - law_documents: 多個法律文檔的列表
    
    返回:
    - 整合後的法律文檔，格式為 {"laws": [...]}
    """
    if not law_documents:
        return {"laws": []}
    
    merged_laws = []
    
    for doc in law_documents:
        if not doc or "law_name" not in doc:
            continue
            
        # 直接使用原始文檔結構，確保完全一致
        merged_law = {
            "law_name": doc["law_name"],
            "chapters": []
        }
        
        # 處理章節
        chapters = doc.get("chapters", [])
        for chapter in chapters:
            merged_chapter = {
                "chapter": chapter.get("chapter", ""),
                "chapter_no": chapter.get("chapter_no", ""),
                "type_en": chapter.get("type_en", "Chapter"),
                "sections": []
            }
            
            # 處理節
            sections = chapter.get("sections", [])
            for section in sections:
                merged_section = {
                    "section": section.get("section", ""),
                    "section_no": section.get("section_no", ""),
                    "type_en": section.get("type_en", "Section"),
                    "articles": []
                }
                
                # 處理條文
                articles = section.get("articles", [])
                for article in articles:
                    # 按照單個PDF轉換的方式構建條文結構
                    merged_article = {
                        "article": article.get("article", ""),
                        "article_no": article.get("article_no", ""),
                        "type_en": article.get("type_en", "Article"),
                        "content": article.get("content", ""),
                        "paragraphs": [],
                        "metadata": article.get("metadata", {})
                    }
                    
                    # 處理段落 - 支援新結構 (paragraphs) 和舊結構 (items)
                    paragraphs = article.get("paragraphs", [])
                    items = article.get("items", [])
                    
                    # 使用 paragraphs 如果存在，否則使用 items
                    items_to_process = paragraphs if paragraphs else items
                    
                    for item in items_to_process:
                        # 按照單個PDF轉換的方式構建段落結構
                        merged_paragraph = {
                            "paragraph": item.get("paragraph", item.get("item", "")),
                            "paragraph_no": item.get("paragraph_no", ""),
                            "type_en": item.get("type_en", "Paragraph"),
                            "content": item.get("content", ""),
                            "subparagraphs": [],
                            "metadata": item.get("metadata", {})
                        }
                        
                        # 處理子段落 - 支援新結構 (subparagraphs) 和舊結構 (sub_items)
                        subparagraphs = item.get("subparagraphs", [])
                        sub_items = item.get("sub_items", [])
                        
                        # 使用 subparagraphs 如果存在，否則使用 sub_items
                        sub_items_to_process = subparagraphs if subparagraphs else sub_items
                        
                        for sub_item in sub_items_to_process:
                            # 按照單個PDF轉換的方式構建子段落結構
                            merged_subparagraph = {
                                "subparagraph": sub_item.get("subparagraph", sub_item.get("sub_item", "")),
                                "subparagraph_no": sub_item.get("subparagraph_no", ""),
                                "type_en": sub_item.get("type_en", "Subparagraph"),
                                "content": sub_item.get("content", ""),
                                "items": [],
                                "metadata": sub_item.get("metadata", {})
                            }
                            
                            # 處理第三層項目 (items/目)
                            third_level_items = sub_item.get("items", [])
                            for third_item in third_level_items:
                                merged_third_item = {
                                    "item": third_item.get("item", ""),
                                    "item_no": third_item.get("item_no", ""),
                                    "type_en": third_item.get("type_en", "Item"),
                                    "content": third_item.get("content", ""),
                                    "metadata": third_item.get("metadata", {})
                                }
                                merged_subparagraph["items"].append(merged_third_item)
                            
                            merged_paragraph["subparagraphs"].append(merged_subparagraph)
                        
                        merged_article["paragraphs"].append(merged_paragraph)
                    
                    merged_section["articles"].append(merged_article)
                
                merged_chapter["sections"].append(merged_section)
            
            merged_law["chapters"].append(merged_chapter)
        
        merged_laws.append(merged_law)
    
    return {"laws": merged_laws}


def clean_legal_amendments_and_effective_status(text: str) -> str:
    """
    清理法規中的修正日期和生效狀態信息
    
    移除模式：
    1. 修正日期：民國 XXX 年 XX 月 XX 日
    2. 生效狀態：※本法規部分或全部條文尚未生效
    3. 相關的施行日期說明
    4. 法規名稱行（避免產生未分類章節）
    """
    import re
    
    lines = text.split('\n')
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 跳過修正日期行
        if re.match(r'^修正日期：民國\s*\d+\s*年\s*\d+\s*月\s*\d+\s*日', line):
            i += 1
            continue
            
        # 跳過生效狀態行
        if '※本法規部分或全部條文尚未生效' in line or '生效狀態：' in line:
            i += 1
            continue
            
        # 跳過法規名稱行（避免產生未分類章節）
        if re.match(r'^法規名稱：', line):
            i += 1
            continue
            
        # 跳過施行日期說明段落（通常以數字開頭的列表項）
        if re.match(r'^\d+\.', line) and any(keyword in line for keyword in ['修正', '施行', '生效', '民國', '年', '月', '日']):
            # 檢查後續行是否也是施行日期說明（包括縮進行）
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                # 如果是空行，跳過
                if not next_line:
                    j += 1
                    continue
                # 如果是以空格/全形空格開頭的縮進行，或者是數字開頭的列表項，或者是包含關鍵詞的行
                if (next_line.startswith((' ', '　', '\t')) or 
                    re.match(r'^\d+\.', next_line) or
                    any(keyword in next_line for keyword in ['修正', '施行', '生效', '民國', '年', '月', '日', '政院', '條文', '增訂', '刪除'])):
                    j += 1
                    continue
                else:
                    break
            i = j
            continue
            
        # 跳過以數字開頭且包含修正/施行關鍵詞的連續行
        if re.match(r'^\d+\.', line) and any(keyword in line for keyword in ['修正', '施行', '生效']):
            # 檢查是否為修正條文說明
            if any(keyword in line for keyword in ['條文', '增訂', '刪除']):
                i += 1
                continue
        
        # 跳過包含施行日期相關關鍵詞的孤立行
        if (any(keyword in line for keyword in ['政院', '施行日期', '定之', '修正之第']) and 
            not any(keyword in line for keyword in ['第', '條', '章', '節'])):
            i += 1
            continue
        
        # 保留其他行
        cleaned_lines.append(lines[i])
        i += 1
    
    # 進一步清理：移除開頭的空白行，確保從真正的章節開始
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    
    return '\n'.join(cleaned_lines)


def convert_pdf_structured(file_content: bytes, filename: str, options: MetadataOptions) -> Dict[str, Any]:
    """將PDF轉換為結構化格式"""
    import time
    start_time = time.time()
    
    try:
        # Read PDF content safely; skip pages with no text
        try:
            reader = PdfReader(io.BytesIO(file_content))
        except Exception as e:
            raise Exception(f"無法讀取PDF文件: {str(e)}")
        
        # 批量提取文本，顯示進度
        texts = []
        total_pages = len(reader.pages)
        print(f"總頁數: {total_pages}")
        
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
                texts.append(t)
            except Exception:
                texts.append("")
            
            # 每處理10頁顯示進度
            if (i + 1) % 10 == 0 or (i + 1) == total_pages:
                print(f"已處理 {i + 1}/{total_pages} 頁")
        
        if not any(texts):  # No text extracted
            raise Exception("PDF文件中没有找到可提取的文本内容")
            
        full_text = "\n".join(texts)
        print(f"文本提取完成，總長度: {len(full_text)} 字符")
        
        # 清理修正日期和生效狀態信息
        cleaned_text = clean_legal_amendments_and_effective_status(full_text)
        print(f"清理完成，清理後長度: {len(cleaned_text)} 字符")
        
        # 使用清理後的文本進行後續處理
        full_text = cleaned_text

        def normalize_digits(s: str) -> str:
            # Convert fullwidth digits to ASCII for simpler matching
            fw = "０１２３４５６７８９"
            hw = "0123456789"
            return s.translate(str.maketrans(fw, hw))

        # Determine law name: 從原始文本中提取法規名稱，但使用清理後的文本進行結構化
        original_text = "\n".join(texts)  # 原始未清理的文本
        original_lines = [normalize_digits((ln or "").strip()) for ln in original_text.splitlines()]
        law_name = None
        for ln in original_lines:
            if not ln:
                continue
            if any(key in ln for key in ["法", "條例", "法規", "法律"]):
                law_name = ln
                break
        if not law_name:
            base = os.path.splitext(filename or "document")[0]
            law_name = base or "未命名法規"

        # 修改章節正則表達式，支持「第X章之一」格式
        chapter_re = re.compile(r"^第\s*([一二三四五六七八九十百千0-9]+)\s*章(?:[之\-]([一二三四五六七八九十百千0-9]+))?[\u3000\s]*(.*)$")
        section_re = re.compile(r"^第\s*([一二三四五六七八九十百千0-9]+)\s*節[\u3000\s]*(.*)$")
        # 識別「節下的款」（在節和條之間的層級，用來細分節）
        # 格式：第X款（中文數字），例如「第 一 款」、「第 二 款」
        subsection_re = re.compile(r"^第\s*([一二三四五六七八九十百千]+)\s*款[\u3000\s]*(.*)$")
        # 修改條文正則表達式，只匹配阿拉伯數字的條文號碼（避免將中文數字的條文引用誤識別為新條文）
        # 例如：「第四十二條」不應該被識別為新條文，而是條文內容的一部分
        article_re = re.compile(r"^第\s*([0-9]+)(?:[之\-]([0-9]+))?\s*條[\u3000\s]*(.*)$")

        def parse_item_line(ln: str):
            # Match common item markers like 「一、」「1.」「（一）」「(1)」「1）」 etc.
            ln = ln.lstrip()
            # （一） or (1)
            m = re.match(r"^[（(]([0-9０-９一二三四五六七八九十]+)[）)]\s*(.*)$", ln)
            if m:
                return m.group(1), m.group(2), "parentheses"
            # 一、 二、 十、 style (Chinese numerals with punctuation)
            m = re.match(r"^([一二三四五六七八九十]+)[、．.）)]\s*(.*)$", ln)
            if m:
                return m.group(1), m.group(2), "chinese_with_punct"
            # 1. 1、 1) styles (Arabic numbers with punctuation)
            m = re.match(r"^([0-9０-９]+)[、．.）)]\s*(.*)$", ln)
            if m:
                return m.group(1), m.group(2), "arabic_with_punct"
            # 1 2 3 styles (Arabic numbers followed by space, common in ROC legal documents)
            m = re.match(r"^([0-9０-９]+)\s+(.*)$", ln)
            if m:
                return m.group(1), m.group(2), "arabic_space"
            # 一 二 三 styles (Chinese numerals followed by space, sub-items)
            m = re.match(r"^([一二三四五六七八九十]+)\s+(.*)$", ln)
            if m:
                return m.group(1), m.group(2), "chinese_space"
            return None, None, None

        structure: Dict[str, Any] = {"law_name": law_name, "chapters": []}
        current_chapter: Optional[Dict[str, Any]] = None
        current_section: Optional[Dict[str, Any]] = None
        current_subsection: Optional[Dict[str, Any]] = None  # 節下的款（在節和條之間的層級）
        current_article: Optional[Dict[str, Any]] = None
        # 依據台灣法律層次：條 → 項(Paragraph) → 款(Subparagraph) → 目(Item)
        current_paragraph: Optional[Dict[str, Any]] = None
        current_subparagraph: Optional[Dict[str, Any]] = None
        current_item_lvl3: Optional[Dict[str, Any]] = None

        def ensure_chapter():
            nonlocal current_chapter
            if current_chapter is None:
                current_chapter = {"chapter": "未分類章", "sections": []}
                structure["chapters"].append(current_chapter)

        def ensure_section():
            nonlocal current_section
            ensure_chapter()
            if current_section is None:
                current_section = {"section": "未分類節", "subsections": [], "articles": []}
                current_chapter["sections"].append(current_section)
        
        def ensure_subsection():
            """確保有節下的款（如果節下沒有款，則直接使用節）"""
            nonlocal current_subsection
            ensure_section()
            if current_subsection is None:
                # 如果節下沒有款，則創建一個默認的款（用於直接包含條文的情況）
                # 但這裡我們不創建，而是讓條文直接歸屬於節
                pass

        # 使用清理後的文本進行結構化解析
        lines = [normalize_digits((ln or "").strip()) for ln in full_text.splitlines()]
        
        for raw in lines:
            ln = raw.strip()
            if not ln:
                continue

            # Headings
            m = chapter_re.match(ln)
            if m:
                num_raw = m.group(1)
                suffix_raw = m.group(2)  # 「之一」的後綴
                title_suffix = m.group(3).strip() if m.group(3) else ""  # 標題後綴（如「總則」）
                
                # 構建章節標題
                if suffix_raw:
                    # 有「之一」後綴：第四章之一
                    title = f"第{num_raw}章之{suffix_raw}" + (f" {title_suffix}" if title_suffix else "")
                else:
                    # 沒有「之一」後綴：第四章
                    title = f"第{num_raw}章" + (f" {title_suffix}" if title_suffix else "")
                
                current_chapter = {"chapter": title, "chapter_no": normalize_digits(num_raw), "type_en": "Chapter", "sections": []}
                if suffix_raw:
                    current_chapter["chapter_suffix"] = normalize_digits(suffix_raw)
                structure["chapters"].append(current_chapter)
                current_section = None
                current_subsection = None  # 重置節下的款
                current_article = None
                current_paragraph = None
                current_subparagraph = None
                current_item_lvl3 = None
                continue

            m = section_re.match(ln)
            if m:
                ensure_chapter()
                num_raw = m.group(1)
                title = f"第{num_raw}節" + (f" {m.group(2).strip()}" if m.group(2) else "")
                current_section = {"section": title, "section_no": normalize_digits(num_raw), "type_en": "Section", "subsections": [], "articles": []}
                current_chapter["sections"].append(current_section)
                current_subsection = None  # 重置節下的款
                current_article = None
                current_paragraph = None
                current_subparagraph = None
                current_item_lvl3 = None
                continue

            # 識別「節下的款」（在節和條之間的層級）
            m = subsection_re.match(ln)
            if m:
                ensure_section()
                num_raw = m.group(1)  # 中文數字
                title_suffix = m.group(2).strip() if m.group(2) else ""  # 標題後綴（如「著作財產權之種類」）
                title = f"第{num_raw}款" + (f" {title_suffix}" if title_suffix else "")
                
                # 創建節下的款
                current_subsection = {
                    "subsection": title,
                    "subsection_no": normalize_digits(num_raw),
                    "type_en": "Subsection",
                    "articles": []
                }
                current_section["subsections"].append(current_subsection)
                current_article = None
                current_paragraph = None
                current_subparagraph = None
                current_item_lvl3 = None
                continue

            m = article_re.match(ln)
            if m:
                ensure_section()
                num_raw = m.group(1)  # 主編號
                suffix_raw = m.group(2)  # 「之一」或「-1」的後綴
                rest = m.group(3).strip() if m.group(3) else ""  # 條文內容
                
                # 構建條文標題
                if suffix_raw:
                    # 有「之一」或「-1」後綴：第4-1條 或 第4條之1
                    # 統一使用「之」格式存儲
                    title = f"第{num_raw}條之{suffix_raw}"
                else:
                    # 沒有後綴：第4條
                    title = f"第{num_raw}條"
                
                # 建立條文，新增 paragraphs 清單並保留相容的 items 欄位
                current_article = {"article": title, "article_no": normalize_digits(num_raw), "type_en": "Article", "content": rest, "paragraphs": []}
                if suffix_raw:
                    current_article["article_suffix"] = normalize_digits(suffix_raw)
                # 相容舊欄位（將指向同一個列表）
                current_article["items"] = current_article["paragraphs"]
                
                # 將條文歸屬於「節下的款」或直接歸屬於「節」
                if current_subsection is not None:
                    # 如果有節下的款，則歸屬於款
                    current_subsection["articles"].append(current_article)
                else:
                    # 如果沒有節下的款，則直接歸屬於節
                    current_section["articles"].append(current_article)
                
                current_paragraph = None
                current_subparagraph = None
                current_item_lvl3 = None
                continue

            # 條文內層級解析：項(阿拉伯數字) → 款(中文數字) → 目（括號中文數字）
            if current_article is not None:
                num, content, item_type = parse_item_line(ln)
                if num is not None:
                    num = normalize_digits(num)
                    # 1) 項 Paragraph: 阿拉伯數字（含 1. 1、 1) 或數字+空白）
                    if item_type in ("arabic_with_punct", "arabic_space"):
                        current_paragraph = {"paragraph": str(num), "paragraph_no": str(num), "type_en": "Paragraph", "content": content or "", "subparagraphs": []}
                        # 相容欄位
                        current_paragraph["sub_items"] = current_paragraph["subparagraphs"]
                        current_article["paragraphs"].append(current_paragraph)
                        current_item_lvl3 = None
                        current_subparagraph = None
                    # 2) 款 Subparagraph: 中文數字（含 一、 或 中文數字+空白）
                    elif item_type in ("chinese_with_punct", "chinese_space") and current_paragraph is not None:
                        if "subparagraphs" not in current_paragraph:
                            current_paragraph["subparagraphs"] = []
                            current_paragraph["sub_items"] = current_paragraph["subparagraphs"]
                        current_subparagraph = {"subparagraph": str(num), "subparagraph_no": str(num), "type_en": "Subparagraph", "content": content or "", "items": []}
                        # 第三級相容鍵名
                        current_subparagraph["sub_sub_items"] = current_subparagraph["items"]
                        current_paragraph["subparagraphs"].append(current_subparagraph)
                        current_item_lvl3 = None
                    # 3) 目 Item: 括號中文或數字（（一）、(1)）出現在款內
                    elif item_type == "parentheses" and current_subparagraph is not None:
                        if "items" not in current_subparagraph:
                            current_subparagraph["items"] = []
                            current_subparagraph["sub_sub_items"] = current_subparagraph["items"]
                        current_item_lvl3 = {"item": str(num), "item_no": str(num), "type_en": "Item", "content": content or ""}
                        current_subparagraph["items"].append(current_item_lvl3)
                    else:
                        # 若無法判別層級，視為當前最深層的續行文字
                        pass
                else:
                    # 續行文字：附加到最深層（目 → 款 → 項 → 條）
                    if current_item_lvl3 is not None:
                        sep = "\n" if current_item_lvl3.get("content") else ""
                        current_item_lvl3["content"] = f"{current_item_lvl3.get('content','')}{sep}{ln}"
                    elif current_subparagraph is not None:
                        sep = "\n" if current_subparagraph.get("content") else ""
                        current_subparagraph["content"] = f"{current_subparagraph.get('content','')}{sep}{ln}"
                    elif current_paragraph is not None:
                        sep = "\n" if current_paragraph.get("content") else ""
                        current_paragraph["content"] = f"{current_paragraph.get('content','')}{sep}{ln}"
                    else:
                        # accumulate into article content
                        if "content" not in current_article or current_article["content"] is None:
                            current_article["content"] = ln
                        else:
                            current_article["content"] = (current_article["content"] + "\n" + ln).strip()
                continue

            # If no article yet, but we have text, place it under a default article
            if current_section is not None and current_article is None:
                current_article = {"article": "未標示條文", "content": ln, "paragraphs": []}
                current_article["items"] = current_article["paragraphs"]
                current_section["articles"].append(current_article)
                current_paragraph = None
                current_subparagraph = None
                current_item_lvl3 = None
            elif current_article is None:
                ensure_section()
                current_article = {"article": "未標示條文", "content": ln, "paragraphs": []}
                current_article["items"] = current_article["paragraphs"]
                current_section["articles"].append(current_article)
                current_paragraph = None
                current_subparagraph = None
                current_item_lvl3 = None
            else:
                # fallback append
                current_article["content"] = (current_article.get("content", "") + "\n" + ln).strip()

        # 優化版本的metadata添加
        def add_metadata_to_structure_optimized(structure, options, full_text):
            """優化版本的metadata添加，大幅提升性能"""
            print("開始添加metadata...")
            metadata_start = time.time()
            
            # 預計算所有條文（避免重複計算）
            all_articles = []
            for chapter in structure["chapters"]:
                for section in chapter["sections"]:
                    for article in section["articles"]:
                        all_articles.append({
                            "article": article["article"],
                            "content": article["content"],
                            "chapter": chapter["chapter"],
                            "section": section["section"]
                        })
            
            print(f"找到 {len(all_articles)} 個條文")
            
            # 批量處理metadata（如果啟用）
            if options.include_id:
                print("批量處理metadata...")
            
            processed_count = 0
            for chapter in structure["chapters"]:
                chapter_name = chapter["chapter"]
                for section in chapter["sections"]:
                    section_name = section["section"]
                    for article in section["articles"]:
                        article_name = article["article"]
                        
                        # 簡化的metadata處理
                        article_metadata = {}
                        if options.include_id:
                            article_metadata["id"] = f"{structure['law_name']}_{chapter_name}_{section_name}_{article_name}".replace(" ", "_")
                        
                        article["metadata"] = article_metadata
                        
                        # 為項目添加簡化metadata - 支援新結構 (paragraphs) 和舊結構 (items)
                        paragraphs = article.get("paragraphs", [])
                        items = article.get("items", [])
                        items_to_process = paragraphs if paragraphs else items
                        
                        for item in items_to_process:
                            # 支援新結構的鍵名
                            item_name = item.get("paragraph", item.get("item", ""))
                            item_metadata = {}
                            if options.include_id:
                                item_metadata["id"] = f"{structure['law_name']}_{chapter_name}_{section_name}_{article_name}_{item_name}".replace(" ", "_")
                            
                            item["metadata"] = item_metadata
                            
                            # 為子項目添加簡化metadata - 支援新結構 (subparagraphs) 和舊結構 (sub_items)
                            subparagraphs = item.get("subparagraphs", [])
                            sub_items = item.get("sub_items", [])
                            sub_items_to_process = subparagraphs if subparagraphs else sub_items
                            
                            for sub_item in sub_items_to_process:
                                # 支援新結構的鍵名
                                sub_item_name = sub_item.get("subparagraph", sub_item.get("sub_item", ""))
                                sub_item_metadata = {}
                                if options.include_id:
                                    sub_item_metadata["id"] = f"{structure['law_name']}_{chapter_name}_{section_name}_{article_name}_{item_name}_{sub_item_name}".replace(" ", "_")
                                
                                sub_item["metadata"] = sub_item_metadata
                                
                                # 處理第三層項目 (items)
                                third_level_items = sub_item.get("items", [])
                                for third_item in third_level_items:
                                    third_item_name = third_item.get("item", "")
                                    third_item_metadata = {}
                                    if options.include_id:
                                        third_item_metadata["id"] = f"{structure['law_name']}_{chapter_name}_{section_name}_{article_name}_{item_name}_{sub_item_name}_{third_item_name}".replace(" ", "_")
                                    
                                    third_item["metadata"] = third_item_metadata
                        
                        processed_count += 1
                        if processed_count % 10 == 0:
                            print(f"已處理 {processed_count} 個條文")
            
            metadata_time = time.time() - metadata_start
            print(f"Metadata處理完成，耗時: {metadata_time:.2f}秒")
        
        # 添加metadata（使用優化版本）
        if options.include_id:
            add_metadata_to_structure_optimized(structure, options, full_text)
        else:
            print("跳過metadata處理（未啟用）")

        total_time = time.time() - start_time
        print(f"總轉換時間: {total_time:.2f}秒")
        
        return {
            "text": full_text,
            "metadata": structure,
            "processing_time": total_time,
            "success": True
        }
        
    except Exception as e:
        return {
            "text": "",
            "metadata": {"error": str(e)},
            "processing_time": time.time() - start_time,
            "success": False,
            "error": str(e)
        }


# 異步任務存儲
conversion_tasks = {}

# PDF緩存存儲 (基於文件內容哈希)
pdf_cache = {}

# 清理舊任務的後台任務
async def cleanup_old_tasks():
    """清理超過1小時的舊任務"""
    while True:
        try:
            current_time = time.time()
            expired_tasks = []
            
            for task_id, task in conversion_tasks.items():
                if current_time - task["created_at"] > 3600:  # 1小時
                    expired_tasks.append(task_id)
            
            for task_id in expired_tasks:
                del conversion_tasks[task_id]
                print(f"清理過期任務: {task_id}")
            
            # 每5分鐘清理一次
            await asyncio.sleep(300)
        except Exception as e:
            print(f"清理任務時發生錯誤: {e}")
            await asyncio.sleep(60)  # 出錯時等待1分鐘再重試

# 清理任務將在應用啟動時啟動
@app.on_event("startup")
async def startup_event():
    """應用啟動時的事件"""
    import asyncio
    asyncio.create_task(cleanup_old_tasks())

@app.post("/api/convert")
async def convert(file: UploadFile = File(...), metadata_options: str = Form("{}")):
    """啟動PDF轉換任務"""
    try:
        # Parse metadata options
        try:
            metadata_config = json.loads(metadata_options)
            options = MetadataOptions(**metadata_config)
        except:
            options = MetadataOptions()  # 使用默認選項
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            return JSONResponse(
                status_code=400, 
                content={"error": "只支持PDF文件格式", "detail": "Invalid file type"}
            )
        
        # Reset file pointer to beginning
        await file.seek(0)
        
        # 生成任務ID
        task_id = f"convert_{int(time.time() * 1000)}_{hash(file.filename) % 10000}"
        
        # 讀取文件內容
        file_content = await file.read()
        
        # 檢查緩存（基於文件內容哈希）
        import hashlib
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # 檢查是否已緩存
        cache_key = f"{file_hash}_{json.dumps(options.__dict__, sort_keys=True)}"
        if cache_key in pdf_cache:
            cached_result = pdf_cache[cache_key]
            print(f"使用緩存的PDF轉換結果: {file.filename}")
            
            # 生成新的doc_id
            doc_id = f"doc_{int(time.time() * 1000)}_{hash(file.filename) % 10000}"
            
            # 將文檔存儲到store中
            store.docs[doc_id] = DocRecord(
                id=doc_id,
                filename=file.filename,
                text=cached_result["text"],
                json_data=cached_result["metadata"],
                chunks=[],
                chunk_size=0,
                overlap=0,
            )
            
            return {
                "doc_id": doc_id,
                "filename": file.filename,
                "text_length": cached_result["text_length"],
                "metadata": cached_result["metadata"],
                "processing_time": 0.1,  # 緩存命中，幾乎瞬間完成
                "cached": True
            }
        
        # 創建任務
        conversion_tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "filename": file.filename,
            "created_at": time.time(),
            "result": None,
            "error": None,
            "file_hash": file_hash,
            "cache_key": cache_key
        }
        
        # 啟動後台任務
        import asyncio
        asyncio.create_task(process_pdf_conversion(task_id, file_content, options))
        
        return {
            "task_id": task_id,
            "status": "pending",
            "message": "PDF轉換任務已啟動，請使用task_id查詢進度"
        }
        
    except Exception as e:
        print(f"Convert endpoint error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": "啟動PDF轉換任務失敗", "detail": str(e)}
        )


async def process_pdf_conversion(task_id: str, file_content: bytes, options: MetadataOptions):
    """後台處理PDF轉換"""
    import time
    start_time = time.time()
    
    try:
        # 更新任務狀態
        conversion_tasks[task_id]["status"] = "processing"
        conversion_tasks[task_id]["progress"] = 10
        
        print(f"開始轉換PDF: {conversion_tasks[task_id]['filename']}")
        
        # 直接調用convert_pdf_structured函數
        conversion_tasks[task_id]["progress"] = 20
        result = convert_pdf_structured(file_content, conversion_tasks[task_id]['filename'], options)
        
        if not result["success"]:
            conversion_tasks[task_id]["status"] = "failed"
            conversion_tasks[task_id]["error"] = result.get("error", "PDF轉換失敗")
            return
        
        # 提取結果
        full_text = result["text"]
        structure = result["metadata"]
        total_time = result["processing_time"]
        
        conversion_tasks[task_id]["progress"] = 80
        print(f"PDF轉換完成，文本長度: {len(full_text)} 字符")
        
        # 生成文檔ID
        doc_id = f"doc_{int(time.time() * 1000)}_{hash(conversion_tasks[task_id]['filename']) % 10000}"
        
        # 將文檔存儲到store中
        store.docs[doc_id] = DocRecord(
            id=doc_id,
            filename=conversion_tasks[task_id]['filename'],
            text=full_text,
            json_data=structure,
            chunks=[],
            chunk_size=0,
            overlap=0,
        )
        
        # 重置嵌入狀態
        store.reset_embeddings()
        store.save_data()
        
        # 保存到緩存
        cache_data = {
            "text": full_text,
            "metadata": structure,
            "text_length": len(full_text),
            "processing_time": total_time
        }
        pdf_cache[conversion_tasks[task_id]["cache_key"]] = cache_data
        
        # 限制緩存大小（最多保存100個轉換結果）
        if len(pdf_cache) > 100:
            # 刪除最舊的緩存項
            oldest_key = next(iter(pdf_cache))
            del pdf_cache[oldest_key]
        
        # 更新任務狀態為完成
        conversion_tasks[task_id]["status"] = "completed"
        conversion_tasks[task_id]["progress"] = 100
        conversion_tasks[task_id]["result"] = {
            "doc_id": doc_id,
            "filename": conversion_tasks[task_id]['filename'],
            "text_length": len(full_text),
            "metadata": structure,
            "processing_time": total_time
        }
        
    except Exception as e:
        # 更新任務狀態為失敗
        conversion_tasks[task_id]["status"] = "failed"
        conversion_tasks[task_id]["error"] = str(e)
        print(f"PDF轉換失敗: {str(e)}")


@app.get("/api/convert/status/{task_id}")
async def get_convert_status(task_id: str):
    """查詢PDF轉換任務狀態"""
    if task_id not in conversion_tasks:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    task = conversion_tasks[task_id]
    
    # 清理超過1小時的舊任務
    if time.time() - task["created_at"] > 3600:
        del conversion_tasks[task_id]
        raise HTTPException(status_code=404, detail="任務已過期")
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "filename": task["filename"],
        "result": task.get("result"),
        "error": task.get("error")
    }


def run_evaluation_task(task_id: str):
    """
    在後台運行評測任務
    """
    task = eval_store.get_task(task_id)
    if not task:
        return
    
    try:
        eval_store.update_task_status(task_id, "running")
        
        doc = store.docs.get(task.doc_id)
        if not doc:
            eval_store.update_task_status(task_id, "failed", error_message="Document not found")
            return
        
        results = []
        total_configs = len(task.configs)
        
        for i, config in enumerate(task.configs):
            result = evaluate_chunk_config(doc, config, task.test_queries, task.k_values, task.strategy)
            results.append(result)
            
            # 更新進度
            progress = (i + 1) / total_configs
            eval_store.update_task_status(task_id, "running", progress=progress)
        
        eval_store.update_task_status(task_id, "completed", results=results)
        
    except Exception as e:
        eval_store.update_task_status(task_id, "failed", error_message=str(e))


@app.post("/api/evaluate/fixed-size")
def start_fixed_size_evaluation(req: FixedSizeEvaluationRequest, background_tasks: BackgroundTasks):
    """
    開始固定大小分割策略評測
    """
    doc = store.docs.get(req.doc_id)
    if not doc:
        return JSONResponse(status_code=404, content={"error": "Document not found"})
    
    # 檢查是否已有生成的問題
    if not hasattr(doc, 'generated_questions') or not doc.generated_questions:
        return JSONResponse(
            status_code=400, 
            content={"error": "請先使用「生成問題」功能為文檔生成測試問題，然後再進行評測"}
        )
    
    # 使用文檔中存儲的問題而不是預設問題
    req.test_queries = doc.generated_questions
    
    # 生成所有配置組合，包括策略特定參數
    configs = []
    for chunk_size in req.chunk_sizes:
        for overlap_ratio in req.overlap_ratios:
            overlap = int(chunk_size * overlap_ratio)
            
            # 根據策略生成不同的參數組合
            if req.strategy == "structured_hierarchical":
                for chunk_by in req.chunk_by_options:
                    config = ChunkConfig(
                        chunk_size=chunk_size,
                        overlap=overlap,
                        overlap_ratio=overlap_ratio,
                        chunk_by=chunk_by
                    )
                    configs.append(config)
            elif req.strategy == "rcts_hierarchical":
                for preserve_structure in req.preserve_structure_options:
                    config = ChunkConfig(
                        chunk_size=chunk_size,
                        overlap=overlap,
                        overlap_ratio=overlap_ratio,
                        preserve_structure=preserve_structure,
                        chunk_by="article"  # 默認值
                    )
                    configs.append(config)
            elif req.strategy == "hierarchical":
                for level_depth in req.level_depth_options:
                    for min_chunk_size in req.min_chunk_size_options:
                        config = ChunkConfig(
                            chunk_size=chunk_size,
                            overlap=overlap,
                            overlap_ratio=overlap_ratio,
                            level_depth=level_depth,
                            min_chunk_size=min_chunk_size,
                            chunk_by="article"  # 默認值
                        )
                        configs.append(config)
            elif req.strategy == "semantic":
                for similarity_threshold in req.similarity_threshold_options:
                    for context_window in req.context_window_options:
                        config = ChunkConfig(
                            chunk_size=chunk_size,
                            overlap=overlap,
                            overlap_ratio=overlap_ratio,
                            similarity_threshold=similarity_threshold,
                            context_window=context_window,
                            chunk_by="article"  # 默認值
                        )
                        configs.append(config)
            elif req.strategy == "llm_semantic":
                for semantic_threshold in req.semantic_threshold_options:
                    for context_window in req.context_window_options:
                        config = ChunkConfig(
                            chunk_size=chunk_size,
                            overlap=overlap,
                            overlap_ratio=overlap_ratio,
                            semantic_threshold=semantic_threshold,
                            context_window=context_window,
                            chunk_by="article"  # 默認值
                        )
                        configs.append(config)
            elif req.strategy == "sliding_window":
                for window_size in req.window_size_options:
                    for step_size in req.step_size_options:
                        for boundary_aware in req.boundary_aware_options:
                            for preserve_sentences in req.preserve_sentences_options:
                                for min_chunk_size_sw in req.min_chunk_size_options_sw:
                                    for max_chunk_size_sw in req.max_chunk_size_options_sw:
                                        config = ChunkConfig(
                                            chunk_size=window_size,  # 使用window_size作為chunk_size
                                            overlap=overlap,
                                            overlap_ratio=overlap_ratio,
                                            strategy="sliding_window",
                                            step_size=step_size,
                                            window_size=window_size,
                                            boundary_aware=boundary_aware,
                                            preserve_sentences=preserve_sentences,
                                            min_chunk_size_sw=min_chunk_size_sw,
                                            max_chunk_size_sw=max_chunk_size_sw,
                                            chunk_by="article"  # 默認值
                                        )
                                        configs.append(config)
            elif req.strategy == "hybrid":
                for switch_threshold in req.switch_threshold_options:
                    for secondary_size in req.secondary_size_options:
                        config = ChunkConfig(
                            chunk_size=chunk_size,
                            overlap=overlap,
                            overlap_ratio=overlap_ratio,
                            switch_threshold=switch_threshold,
                            secondary_size=secondary_size,
                            chunk_by="article"  # 默認值
                        )
                        configs.append(config)
            else:
                # 默認配置（fixed_size等）
                config = ChunkConfig(
                    chunk_size=chunk_size,
                    overlap=overlap,
                    overlap_ratio=overlap_ratio,
                    chunk_by="article"  # 默認值
                )
                configs.append(config)
    
    # 獲取分割策略（從請求中獲取，默認為fixed_size）
    strategy = getattr(req, 'strategy', 'fixed_size')
    
    # 創建評測任務
    task_id = eval_store.create_task(
        doc_id=req.doc_id,
        configs=configs,
        test_queries=req.test_queries,
        k_values=req.k_values,
        strategy=strategy
    )
    
    # 在後台運行評測
    background_tasks.add_task(run_evaluation_task, task_id)
    
    return {
        "task_id": task_id,
        "status": "started",
        "total_configs": len(configs),
        "message": "評測任務已開始，請使用task_id查詢進度"
    }


@app.get("/api/evaluate/status/{task_id}")
def get_evaluation_status(task_id: str):
    """
    獲取評測任務狀態
    """
    task = eval_store.get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    
    total_configs = len(task.configs)
    completed_configs = int(task.progress * total_configs) if task.progress > 0 else 0
    
    return {
        "task_id": task_id,
        "status": task.status,
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "error_message": task.error_message,
        "total_configs": total_configs,
        "completed_configs": completed_configs,
        "progress": task.progress
    }


@app.get("/api/evaluate/results/{task_id}")
def get_evaluation_results(task_id: str):
    """
    獲取評測結果
    """
    task = eval_store.get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    
    if task.status != "completed":
        return JSONResponse(status_code=400, content={"error": "Task not completed yet"})
    
    # 轉換結果為可序列化的格式
    results = []
    for result in task.results:
        result_dict = {
            "config": result.config,  # 現在 config 已經是字典了
            "metrics": {
                "precision_omega": result.metrics.precision_omega,
                "precision_at_k": result.metrics.precision_at_k,
                "recall_at_k": result.metrics.recall_at_k,
                "chunk_count": result.metrics.chunk_count,
                "avg_chunk_length": result.metrics.avg_chunk_length,
                "length_variance": result.metrics.length_variance
            },
            "test_queries": result.test_queries,
            "retrieval_results": result.retrieval_results,
            "timestamp": result.timestamp.isoformat()
        }
        results.append(result_dict)
    
    return {
        "task_id": task_id,
        "status": task.status,
        "results": results,
        "summary": {
            "total_configs": len(results),
            "best_precision_omega": max(r["metrics"]["precision_omega"] for r in results),
            "best_precision_at_5": max(r["metrics"]["precision_at_k"].get(5, 0) for r in results),
            "best_recall_at_5": max(r["metrics"]["recall_at_k"].get(5, 0) for r in results),
            "avg_chunk_count": sum(r["metrics"]["chunk_count"] for r in results) / len(results),
            "avg_chunk_length": sum(r["metrics"]["avg_chunk_length"] for r in results) / len(results)
        }
    }


@app.get("/api/evaluate/comparison/{task_id}")
def get_evaluation_comparison(task_id: str):
    """
    獲取評測結果對比分析
    """
    task = eval_store.get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    
    if task.status != "completed":
        return JSONResponse(status_code=400, content={"error": "Task not completed yet"})
    
    # 生成對比分析
    comparison = {
        "chunk_size_analysis": {},
        "overlap_analysis": {},
        "strategy_specific_analysis": {},
        "recommendations": []
    }
    
    # 按chunk size分組分析
    chunk_size_groups = {}
    for result in task.results:
        size = result.config["chunk_size"]
        if size not in chunk_size_groups:
            chunk_size_groups[size] = []
        chunk_size_groups[size].append(result)
    
    for size, results in chunk_size_groups.items():
        avg_metrics = {
            "precision_omega": sum(r.metrics.precision_omega for r in results) / len(results),
            "precision_at_k": {
                "1": sum(r.metrics.precision_at_k.get(1, 0) for r in results) / len(results),
                "3": sum(r.metrics.precision_at_k.get(3, 0) for r in results) / len(results),
                "5": sum(r.metrics.precision_at_k.get(5, 0) for r in results) / len(results),
                "10": sum(r.metrics.precision_at_k.get(10, 0) for r in results) / len(results)
            },
            "recall_at_k": {
                "1": sum(r.metrics.recall_at_k.get(1, 0) for r in results) / len(results),
                "3": sum(r.metrics.recall_at_k.get(3, 0) for r in results) / len(results),
                "5": sum(r.metrics.recall_at_k.get(5, 0) for r in results) / len(results),
                "10": sum(r.metrics.recall_at_k.get(10, 0) for r in results) / len(results)
            },
            "avg_chunk_count": sum(r.metrics.chunk_count for r in results) / len(results),
            "avg_chunk_length": sum(r.metrics.avg_chunk_length for r in results) / len(results),
            "length_variance": sum(r.metrics.length_variance for r in results) / len(results)
        }
        comparison["chunk_size_analysis"][size] = avg_metrics
    
    # 按overlap ratio分組分析
    overlap_groups = {}
    for result in task.results:
        ratio = result.config["overlap_ratio"]
        if ratio not in overlap_groups:
            overlap_groups[ratio] = []
        overlap_groups[ratio].append(result)
    
    for ratio, results in overlap_groups.items():
        avg_metrics = {
            "precision_omega": sum(r.metrics.precision_omega for r in results) / len(results),
            "precision_at_k": {
                "1": sum(r.metrics.precision_at_k.get(1, 0) for r in results) / len(results),
                "3": sum(r.metrics.precision_at_k.get(3, 0) for r in results) / len(results),
                "5": sum(r.metrics.precision_at_k.get(5, 0) for r in results) / len(results),
                "10": sum(r.metrics.precision_at_k.get(10, 0) for r in results) / len(results)
            },
            "recall_at_k": {
                "1": sum(r.metrics.recall_at_k.get(1, 0) for r in results) / len(results),
                "3": sum(r.metrics.recall_at_k.get(3, 0) for r in results) / len(results),
                "5": sum(r.metrics.recall_at_k.get(5, 0) for r in results) / len(results),
                "10": sum(r.metrics.recall_at_k.get(10, 0) for r in results) / len(results)
            },
            "avg_chunk_count": sum(r.metrics.chunk_count for r in results) / len(results),
            "avg_chunk_length": sum(r.metrics.avg_chunk_length for r in results) / len(results),
            "length_variance": sum(r.metrics.length_variance for r in results) / len(results)
        }
        comparison["overlap_analysis"][ratio] = avg_metrics
    
    # 按策略特定參數分組分析
    if task.results:
        strategy = task.results[0].config.get("strategy", "fixed_size")
        
        if strategy == "structured_hierarchical":
            # 按分割單位分組
            chunk_by_groups = {}
            for result in task.results:
                chunk_by = result.config.get("chunk_by", "article")
                if chunk_by not in chunk_by_groups:
                    chunk_by_groups[chunk_by] = []
                chunk_by_groups[chunk_by].append(result)
            
            for chunk_by, results in chunk_by_groups.items():
                avg_metrics = {
                    "precision_omega": sum(r.metrics.precision_omega for r in results) / len(results),
                    "precision_at_k": {
                        "1": sum(r.metrics.precision_at_k.get(1, 0) for r in results) / len(results),
                        "3": sum(r.metrics.precision_at_k.get(3, 0) for r in results) / len(results),
                        "5": sum(r.metrics.precision_at_k.get(5, 0) for r in results) / len(results),
                        "10": sum(r.metrics.precision_at_k.get(10, 0) for r in results) / len(results)
                    },
                    "recall_at_k": {
                        "1": sum(r.metrics.recall_at_k.get(1, 0) for r in results) / len(results),
                        "3": sum(r.metrics.recall_at_k.get(3, 0) for r in results) / len(results),
                        "5": sum(r.metrics.recall_at_k.get(5, 0) for r in results) / len(results),
                        "10": sum(r.metrics.recall_at_k.get(10, 0) for r in results) / len(results)
                    },
                    "avg_chunk_count": sum(r.metrics.chunk_count for r in results) / len(results),
                    "avg_chunk_length": sum(r.metrics.avg_chunk_length for r in results) / len(results),
                    "length_variance": sum(r.metrics.length_variance for r in results) / len(results)
                }
                comparison["strategy_specific_analysis"][f"chunk_by_{chunk_by}"] = avg_metrics
        
        elif strategy == "rcts_hierarchical":
            # 按保持結構分組
            preserve_groups = {}
            for result in task.results:
                preserve = result.config.get("preserve_structure", True)
                key = "preserve_structure" if preserve else "no_preserve_structure"
                if key not in preserve_groups:
                    preserve_groups[key] = []
                preserve_groups[key].append(result)
            
            for key, results in preserve_groups.items():
                avg_metrics = {
                    "precision_omega": sum(r.metrics.precision_omega for r in results) / len(results),
                    "precision_at_k": {
                        "1": sum(r.metrics.precision_at_k.get(1, 0) for r in results) / len(results),
                        "3": sum(r.metrics.precision_at_k.get(3, 0) for r in results) / len(results),
                        "5": sum(r.metrics.precision_at_k.get(5, 0) for r in results) / len(results),
                        "10": sum(r.metrics.precision_at_k.get(10, 0) for r in results) / len(results)
                    },
                    "recall_at_k": {
                        "1": sum(r.metrics.recall_at_k.get(1, 0) for r in results) / len(results),
                        "3": sum(r.metrics.recall_at_k.get(3, 0) for r in results) / len(results),
                        "5": sum(r.metrics.recall_at_k.get(5, 0) for r in results) / len(results),
                        "10": sum(r.metrics.recall_at_k.get(10, 0) for r in results) / len(results)
                    },
                    "avg_chunk_count": sum(r.metrics.chunk_count for r in results) / len(results),
                    "avg_chunk_length": sum(r.metrics.avg_chunk_length for r in results) / len(results),
                    "length_variance": sum(r.metrics.length_variance for r in results) / len(results)
                }
                comparison["strategy_specific_analysis"][key] = avg_metrics
        
        elif strategy == "hierarchical":
            # 按層次深度分組
            level_groups = {}
            for result in task.results:
                level = result.config.get("level_depth", 3)
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(result)
            
            for level, results in level_groups.items():
                avg_metrics = {
                    "precision_omega": sum(r.metrics.precision_omega for r in results) / len(results),
                    "precision_at_k": {
                        "1": sum(r.metrics.precision_at_k.get(1, 0) for r in results) / len(results),
                        "3": sum(r.metrics.precision_at_k.get(3, 0) for r in results) / len(results),
                        "5": sum(r.metrics.precision_at_k.get(5, 0) for r in results) / len(results),
                        "10": sum(r.metrics.precision_at_k.get(10, 0) for r in results) / len(results)
                    },
                    "recall_at_k": {
                        "1": sum(r.metrics.recall_at_k.get(1, 0) for r in results) / len(results),
                        "3": sum(r.metrics.recall_at_k.get(3, 0) for r in results) / len(results),
                        "5": sum(r.metrics.recall_at_k.get(5, 0) for r in results) / len(results),
                        "10": sum(r.metrics.recall_at_k.get(10, 0) for r in results) / len(results)
                    },
                    "avg_chunk_count": sum(r.metrics.chunk_count for r in results) / len(results),
                    "avg_chunk_length": sum(r.metrics.avg_chunk_length for r in results) / len(results),
                    "length_variance": sum(r.metrics.length_variance for r in results) / len(results)
                }
                comparison["strategy_specific_analysis"][f"level_depth_{level}"] = avg_metrics
    
    # 生成推薦
    best_overall = max(task.results, key=lambda r: (
        r.metrics.precision_omega * 0.4 + 
        r.metrics.precision_at_k.get(5, 0) * 0.3 + 
        r.metrics.recall_at_k.get(5, 0) * 0.3
    ))
    
    # 生成詳細的推薦配置
    config_parts = []
    config_parts.append(f"chunk_size={best_overall.config['chunk_size']}")
    config_parts.append(f"overlap_ratio={best_overall.config['overlap_ratio']}")
    
    # 添加策略特定參數
    strategy = best_overall.config.get("strategy", "fixed_size")
    if strategy == "structured_hierarchical":
        chunk_by = best_overall.config.get("chunk_by", "article")
        chunk_by_label = {"article": "按條文分割", "item": "按項分割", "section": "按節分割", "chapter": "按章分割"}.get(chunk_by, chunk_by)
        config_parts.append(f"chunk_by={chunk_by}({chunk_by_label})")
    elif strategy == "rcts_hierarchical":
        preserve = best_overall.config.get("preserve_structure", True)
        config_parts.append(f"preserve_structure={preserve}({'保持結構' if preserve else '不保持結構'})")
    elif strategy == "hierarchical":
        level = best_overall.config.get("level_depth", 3)
        min_size = best_overall.config.get("min_chunk_size", 200)
        config_parts.append(f"level_depth={level}")
        config_parts.append(f"min_chunk_size={min_size}")
    elif strategy == "semantic":
        threshold = best_overall.config.get("similarity_threshold", 0.6)
        context = best_overall.config.get("context_window", 100)
        config_parts.append(f"similarity_threshold={threshold}")
        config_parts.append(f"context_window={context}")
    elif strategy == "llm_semantic":
        threshold = best_overall.config.get("semantic_threshold", 0.7)
        context = best_overall.config.get("context_window", 100)
        config_parts.append(f"semantic_threshold={threshold}")
        config_parts.append(f"context_window={context}")
    elif strategy == "sliding_window":
        step = best_overall.config.get("step_size", 250)
        config_parts.append(f"step_size={step}")
    elif strategy == "hybrid":
        switch = best_overall.config.get("switch_threshold", 0.5)
        secondary = best_overall.config.get("secondary_size", 400)
        config_parts.append(f"switch_threshold={switch}")
        config_parts.append(f"secondary_size={secondary}")
    
    comparison["recommendations"] = [
        f"最佳配置：{', '.join(config_parts)}",
        f"該配置的precision omega: {best_overall.metrics.precision_omega:.3f}",
        f"該配置的precision@5: {best_overall.metrics.precision_at_k.get(5, 0):.3f}",
        f"該配置的recall@5: {best_overall.metrics.recall_at_k.get(5, 0):.3f}",
        f"該配置的chunk count: {best_overall.metrics.chunk_count}",
        f"該配置的平均chunk長度: {best_overall.metrics.avg_chunk_length:.1f}"
    ]
    
    return comparison


@app.post("/api/generate-questions")
def generate_questions(req: GenerateQuestionsRequest):
    """
    生成繁體中文法律考古題從法律文本中生成問題
    """
    doc = store.docs.get(req.doc_id)
    if not doc:
        return JSONResponse(status_code=404, content={"error": "Document not found"})
    
    start_time = time.time()
    
    try:
        # 使用Gemini生成問題
        questions = generate_questions_with_gemini(
            doc.text, 
            req.num_questions, 
            req.question_types, 
            req.difficulty_levels
        )
        
        generation_time = time.time() - start_time
        
        # 將生成的問題存儲到文檔記錄中
        question_texts = [q.question for q in questions]
        doc.generated_questions = question_texts
        store.docs[req.doc_id] = doc  # 更新文檔記錄
        
        # 檢查是否生成了問題
        if not questions:
            print("警告：沒有生成任何問題")
            return JSONResponse(
                status_code=400,
                content={"error": "無法從文檔中生成問題，請檢查文檔內容是否包含法律條文"}
            )
        
        result = QuestionGenerationResult(
            doc_id=req.doc_id,
            total_questions=len(questions),
            questions=questions,
            generation_time=generation_time,
            timestamp=datetime.now()
        )
        
        response_data = {
            "success": True,
            "result": {
                "doc_id": result.doc_id,
                "total_questions": result.total_questions,
                "generation_time": result.generation_time,
                "timestamp": result.timestamp.isoformat(),
                "questions": [
                    {
                        "question": q.question,
                        "references": q.references,
                        "question_type": q.question_type,
                        "difficulty": q.difficulty,
                        "keywords": q.keywords,
                        "estimated_tokens": q.estimated_tokens
                    }
                    for q in result.questions
                ]
            }
        }
        
        print(f"返回響應數據: success={response_data['success']}, questions_count={len(response_data['result']['questions'])}")
        return response_data
        
    except Exception as e:
        print(f"問題生成異常: {str(e)}")  # 添加日誌
        return JSONResponse(
            status_code=500, 
            content={"error": f"問題生成失敗: {str(e)}"}
        )


@app.get("/docs/schema")
def schema():
    # Minimal shape for frontend wiring/testing
    return {
        "upload": {"POST": {"multipart": True}},
        "chunk": {"POST": {"json": {"doc_id": "str", "chunk_size": "int", "overlap": "int"}}},
        "embed": {"POST": {"json": {"doc_ids": "List[str]|None"}}},
        "retrieve": {"POST": {"json": {"query": "str", "k": "int"}}},
        "generate": {"POST": {"json": {"query": "str", "top_k": "int"}}},
        "evaluate/fixed-size": {"POST": {"json": "FixedSizeEvaluationRequest"}},
        "evaluate/status/{task_id}": {"GET": {}},
        "evaluate/results/{task_id}": {"GET": {}},
        "evaluate/comparison/{task_id}": {"GET": {}},
        "generate-questions": {"POST": {"json": "GenerateQuestionsRequest"}},
        # 新增的增強版API端點
        "legal-semantic-chunk": {"POST": {"json": "ChunkConfig"}},
        "multi-level-semantic-chunk": {"POST": {"json": "ChunkConfig"}},
        "build-concept-graph": {"POST": {}},
        "concept-graph-retrieve": {"POST": {"json": "RetrieveRequest"}},
        "adaptive-retrieve": {"POST": {"json": "RetrieveRequest"}},
        "strategy-performance": {"GET": {}},
        "concept-graph-info": {"GET": {}},
    }


# ============================================================================
# 新增的增強版功能 - 法律語義檢索改進
# ============================================================================

# 導入新的模組
try:
    from .legal_semantic_chunking import LegalSemanticIntegrityChunking, MultiLevelSemanticChunking
    from .legal_concept_graph import LegalConceptGraph, LegalConceptGraphRetrieval
    from .adaptive_legal_rag import AdaptiveLegalRAG, QueryAnalyzer
    from .legal_reasoning_engine import legal_reasoning_engine
    from .intelligent_legal_concept_extractor import intelligent_extractor
    from .dynamic_concept_learning import dynamic_learning_system
    
    # 初始化增強版組件
    legal_semantic_chunker = LegalSemanticIntegrityChunking()
    multi_level_chunker = MultiLevelSemanticChunking()
    concept_graph = LegalConceptGraph()
    concept_graph_retrieval = None
    adaptive_rag = AdaptiveLegalRAG()
    
    print("✅ 增強版功能模組載入成功")
    
except ImportError as e:
    print(f"⚠️  增強版功能模組載入失敗: {e}")
    print("   請確保所有新增文件都存在")
    legal_semantic_chunker = None
    multi_level_chunker = None
    concept_graph = None
    concept_graph_retrieval = None
    adaptive_rag = None
    legal_reasoning_engine = None
    intelligent_extractor = None
    dynamic_learning_system = None


@app.post("/api/legal-semantic-chunk")
def legal_semantic_chunk(req: ChunkConfig):
    """法律語義完整性分塊"""
    if not legal_semantic_chunker:
        return JSONResponse(status_code=503, content={"error": "法律語義分塊功能未啟用"})
    
    try:
        doc = store.get_doc(req.doc_id)
        if not doc:
            return JSONResponse(status_code=404, content={"error": f"文檔 {req.doc_id} 不存在"})
        
        print(f"🔍 開始法律語義完整性分塊，文檔: {doc.filename}")
        
        # 使用法律語義完整性分塊
        chunks_with_span = legal_semantic_chunker.chunk(
            doc.text,
            max_chunk_size=req.chunk_size,
            overlap_ratio=req.overlap_ratio,
            preserve_concepts=True
        )
        
        # 提取純文本chunks
        chunks = [chunk["content"] for chunk in chunks_with_span]
        
        # 更新文檔記錄
        doc.chunks = chunks
        doc.chunk_size = req.chunk_size
        doc.overlap = int(req.chunk_size * req.overlap_ratio)
        doc.structured_chunks = chunks_with_span
        doc.chunking_strategy = "legal_semantic_integrity"
        store.add_doc(doc)
        
        store.reset_embeddings()
        store.save_data()
        
        # 計算統計信息
        chunk_lengths = [len(chunk) for chunk in chunks] if chunks else []
        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        min_length = min(chunk_lengths) if chunk_lengths else 0
        max_length = max(chunk_lengths) if chunk_lengths else 0
        
        if chunk_lengths:
            variance = sum((length - avg_chunk_length) ** 2 for length in chunk_lengths) / len(chunk_lengths)
        else:
            variance = 0
        
        # 計算概念完整性統計
        concept_stats = _calculate_concept_statistics(chunks_with_span)
        
        return {
            "doc_id": req.doc_id,
            "chunk_count": len(chunks),
            "avg_chunk_length": avg_chunk_length,
            "min_chunk_length": min_length,
            "max_chunk_length": max_length,
            "length_variance": variance,
            "strategy": "legal_semantic_integrity",
            "config": req.dict(),
            "chunks_with_span": chunks_with_span,
            "concept_statistics": concept_stats
        }
        
    except Exception as e:
        print(f"❌ 法律語義分塊錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"分塊錯誤: {str(e)}"})


@app.post("/api/multi-level-semantic-chunk")
def multi_level_semantic_chunk(req: ChunkConfig):
    """多層次語義分塊"""
    if not multi_level_chunker:
        return JSONResponse(status_code=503, content={"error": "多層次語義分塊功能未啟用"})
    
    try:
        doc = store.get_doc(req.doc_id)
        if not doc:
            return JSONResponse(status_code=404, content={"error": f"文檔 {req.doc_id} 不存在"})
        
        print(f"🔍 開始多層次語義分塊，文檔: {doc.filename}")
        
        # 使用多層次語義分塊
        multi_level_chunks = multi_level_chunker.chunk(
            doc.text,
            max_chunk_size=req.chunk_size,
            overlap_ratio=req.overlap_ratio
        )
        
        # 保存多層次分塊結果
        doc.multi_level_chunks = multi_level_chunks
        doc.chunking_strategy = "multi_level_semantic"
        store.add_doc(doc)
        
        store.reset_embeddings()
        store.save_data()
        
        # 計算各層次統計
        level_statistics = {}
        for level_name, level_chunks in multi_level_chunks.items():
            chunk_lengths = [len(chunk["content"]) for chunk in level_chunks]
            level_statistics[level_name] = {
                "chunk_count": len(level_chunks),
                "avg_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                "min_length": min(chunk_lengths) if chunk_lengths else 0,
                "max_length": max(chunk_lengths) if chunk_lengths else 0
            }
        
        return {
            "doc_id": req.doc_id,
            "strategy": "multi_level_semantic",
            "config": req.dict(),
            "multi_level_chunks": multi_level_chunks,
            "level_statistics": level_statistics
        }
        
    except Exception as e:
        print(f"❌ 多層次語義分塊錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"分塊錯誤: {str(e)}"})


@app.post("/api/build-concept-graph")
def build_concept_graph():
    """構建法律概念圖"""
    if not concept_graph:
        return JSONResponse(status_code=503, content={"error": "概念圖功能未啟用"})
    
    try:
        print("🔨 開始構建法律概念圖...")
        
        # 獲取所有文檔
        docs = store.list_docs()
        if not docs:
            return JSONResponse(status_code=400, content={"error": "沒有文檔可用"})
        
        # 準備文檔數據
        documents = []
        for doc in docs:
            if doc.chunks:
                for i, chunk in enumerate(doc.chunks):
                    documents.append({
                        'content': chunk,
                        'doc_id': doc.id,
                        'chunk_index': i,
                        'filename': doc.filename
                    })
        
        if not documents:
            return JSONResponse(status_code=400, content={"error": "沒有可用的文檔內容"})
        
        # 構建概念圖
        concept_graph.build_graph(documents)
        
        # 初始化概念圖檢索
        global concept_graph_retrieval
        concept_graph_retrieval = LegalConceptGraphRetrieval(concept_graph)
        
        # 註冊到自適應RAG
        if adaptive_rag:
            adaptive_rag.register_strategy('concept_graph', concept_graph_retrieval)
        
        # 獲取概念圖統計
        graph_stats = {
            'node_count': concept_graph.graph.number_of_nodes(),
            'edge_count': concept_graph.graph.number_of_edges(),
            'concept_count': len(concept_graph.concepts),
            'relation_count': len(concept_graph.relations)
        }
        
        print(f"✅ 概念圖構建完成: {graph_stats}")
        
        return {
            "status": "success",
            "message": "概念圖構建完成",
            "statistics": graph_stats
        }
        
    except Exception as e:
        print(f"❌ 概念圖構建錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"構建錯誤: {str(e)}"})


@app.post("/api/concept-graph-retrieve")
def concept_graph_retrieve(req: RetrieveRequest):
    """概念圖檢索"""
    if not concept_graph_retrieval:
        return JSONResponse(status_code=400, content={"error": "概念圖未構建，請先調用 /api/build-concept-graph"})
    
    try:
        print(f"🔍 開始概念圖檢索，查詢: '{req.query}'")
        
        # 執行概念圖檢索
        results = concept_graph_retrieval.retrieve(req.query, req.k)
        
        # 計算檢索指標
        metrics = calculate_retrieval_metrics(req.query, results, req.k)
        
        # 添加概念圖特定信息
        metrics["concept_graph_analysis"] = {
            "reasoning_paths_used": len(set(r.get('reasoning_path', []) for r in results)),
            "concept_matches": len([r for r in results if r.get('concept_based', False)]),
            "avg_reasoning_score": sum(r.get('reasoning_score', 0) for r in results) / len(results) if results else 0
        }
        
        metrics["note"] = f"概念圖檢索: 使用{metrics['concept_graph_analysis']['reasoning_paths_used']}條推理路徑"
        
        return {
            "results": results,
            "metrics": metrics,
            "embedding_provider": "concept_graph",
            "embedding_model": "legal_concept_reasoning"
        }
        
    except Exception as e:
        print(f"❌ 概念圖檢索錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"檢索錯誤: {str(e)}"})


@app.post("/api/adaptive-retrieve")
def adaptive_retrieve(req: RetrieveRequest):
    """自適應檢索"""
    if not adaptive_rag:
        return JSONResponse(status_code=503, content={"error": "自適應檢索功能未啟用"})
    
    try:
        print(f"🚀 開始自適應檢索，查詢: '{req.query}'")
        
        # 確保檢索策略已註冊
        if not adaptive_rag.retrieval_strategies:
            _register_default_strategies()
        
        # 執行自適應檢索
        results = adaptive_rag.retrieve(req.query, req.k)
        
        # 計算檢索指標
        metrics = calculate_retrieval_metrics(req.query, results, req.k)
        
        # 添加自適應檢索特定信息
        if results:
            first_result = results[0]
            contributing_strategies = first_result.get('contributing_strategies', [])
            strategy_count = first_result.get('strategy_count', 0)
            
            metrics["adaptive_analysis"] = {
                "strategies_used": list(set(contributing_strategies)),
                "strategy_count": strategy_count,
                "fusion_performed": first_result.get('metadata', {}).get('adaptive_fusion', False),
                "avg_fused_score": sum(r.get('fused_score', 0) for r in results) / len(results)
            }
            
            metrics["note"] = f"自適應檢索: 融合{strategy_count}個策略"
        
        return {
            "results": results,
            "metrics": metrics,
            "embedding_provider": "adaptive_rag",
            "embedding_model": "multi_strategy_fusion"
        }
        
    except Exception as e:
        print(f"❌ 自適應檢索錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"檢索錯誤: {str(e)}"})


@app.get("/api/strategy-performance")
def get_strategy_performance():
    """獲取策略性能統計"""
    if not adaptive_rag:
        return JSONResponse(status_code=503, content={"error": "自適應檢索功能未啟用"})
    
    try:
        performance = adaptive_rag.performance_monitor.get_strategy_performance()
        
        return {
            "strategy_performance": performance,
            "total_retrievals": len(adaptive_rag.performance_monitor.retrieval_history)
        }
        
    except Exception as e:
        print(f"❌ 獲取策略性能錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"獲取性能錯誤: {str(e)}"})


@app.get("/api/concept-graph-info")
def get_concept_graph_info():
    """獲取概念圖信息"""
    if not concept_graph:
        return JSONResponse(status_code=503, content={"error": "概念圖功能未啟用"})
    
    try:
        # 獲取概念列表
        concepts_info = []
        for concept_id, concept in concept_graph.concepts.items():
            concepts_info.append({
                "concept_id": concept_id,
                "concept_name": concept.concept_name,
                "content": concept.content[:200] + "..." if len(concept.content) > 200 else concept.content,
                "importance_score": concept.importance_score,
                "frequency": concept.frequency
            })
        
        # 獲取關係列表
        relations_info = []
        for relation in concept_graph.relations:
            relations_info.append({
                "source": relation.source_concept,
                "target": relation.target_concept,
                "relation_type": relation.relation_type,
                "confidence": relation.confidence
            })
        
        # 獲取圖統計
        graph_stats = {
            "node_count": concept_graph.graph.number_of_nodes(),
            "edge_count": concept_graph.graph.number_of_edges(),
            "concept_count": len(concept_graph.concepts),
            "relation_count": len(concept_graph.relations)
        }
        
        # 獲取度中心性最高的概念（前10個）
        if concept_graph.graph.number_of_nodes() > 0:
            import networkx as nx
            centrality = nx.degree_centrality(concept_graph.graph)
            top_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            top_concepts_info = [
                {
                    "concept_id": concept_id,
                    "concept_name": concept_graph.concepts[concept_id].concept_name,
                    "centrality": centrality_score
                }
                for concept_id, centrality_score in top_concepts
            ]
        else:
            top_concepts_info = []
        
        return {
            "graph_statistics": graph_stats,
            "top_concepts": top_concepts_info,
            "concepts": concepts_info[:20],  # 只返回前20個概念
            "relations": relations_info[:20],  # 只返回前20個關係
            "total_concepts": len(concepts_info),
            "total_relations": len(relations_info)
        }
        
    except Exception as e:
        print(f"❌ 獲取概念圖信息錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"獲取概念圖信息錯誤: {str(e)}"})


@app.post("/api/legal-reasoning")
def analyze_legal_query(request: Dict[str, str]):
    """法律推理分析"""
    if not legal_reasoning_engine:
        return JSONResponse(status_code=503, content={"error": "法律推理引擎未啟用"})
    
    try:
        query = request.get("query", "")
        if not query:
            return JSONResponse(status_code=400, content={"error": "查詢不能為空"})
        
        print(f"🔍 開始法律推理分析，查詢: '{query}'")
        
        # 執行推理分析
        analysis = legal_reasoning_engine.analyze_query(query)
        
        return {
            "analysis_result": analysis,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ 法律推理分析錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"推理分析錯誤: {str(e)}"})


@app.post("/api/extract-legal-concepts")
def extract_legal_concepts():
    """智能提取法律概念"""
    if not intelligent_extractor:
        return JSONResponse(status_code=503, content={"error": "智能概念提取器未啟用"})
    
    try:
        print("🔍 開始智能法律概念提取...")
        
        # 獲取所有文檔
        docs = store.list_docs()
        if not docs:
            return JSONResponse(status_code=400, content={"error": "沒有文檔可用"})
        
        # 準備文檔數據
        documents = []
        for doc in docs:
            if hasattr(doc, 'structured_chunks') and doc.structured_chunks:
                documents.append({
                    'filename': doc.filename,
                    'structured_chunks': doc.structured_chunks
                })
        
        if not documents:
            return JSONResponse(status_code=400, content={"error": "沒有結構化分塊數據"})
        
        # 執行概念提取
        extraction_result = intelligent_extractor.extract_concepts_from_documents(documents)
        
        # 保存提取結果到全局變量
        global extracted_legal_concepts
        extracted_legal_concepts = extraction_result
        
        return {
            "extraction_result": extraction_result,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ 概念提取錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"概念提取錯誤: {str(e)}"})


@app.post("/api/learn-from-feedback")
def learn_from_feedback(request: Dict[str, Any]):
    """從用戶反饋中學習"""
    if not dynamic_learning_system:
        return JSONResponse(status_code=503, content={"error": "動態學習系統未啟用"})
    
    try:
        query = request.get("query", "")
        retrieved_results = request.get("retrieved_results", [])
        user_feedback = request.get("user_feedback", {})
        
        if not query:
            return JSONResponse(status_code=400, content={"error": "查詢不能為空"})
        
        print(f"🧠 開始從反饋中學習: '{query}'")
        
        # 執行學習
        learning_result = dynamic_learning_system.learn_from_query_feedback(
            query, retrieved_results, user_feedback
        )
        
        return {
            "learning_result": learning_result,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ 學習錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"學習錯誤: {str(e)}"})


@app.get("/api/learning-statistics")
def get_learning_statistics():
    """獲取學習統計"""
    if not dynamic_learning_system:
        return JSONResponse(status_code=503, content={"error": "動態學習系統未啟用"})
    
    try:
        statistics = dynamic_learning_system.get_learning_statistics()
        
        return {
            "statistics": statistics,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ 獲取學習統計錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"獲取學習統計錯誤: {str(e)}"})


@app.post("/api/enhanced-query-expansion")
def enhanced_query_expansion(request: Dict[str, str]):
    """增強查詢擴展"""
    if not dynamic_learning_system:
        return JSONResponse(status_code=503, content={"error": "動態學習系統未啟用"})
    
    try:
        query = request.get("query", "")
        if not query:
            return JSONResponse(status_code=400, content={"error": "查詢不能為空"})
        
        print(f"🔍 開始增強查詢擴展: '{query}'")
        
        # 執行增強查詢擴展
        expansion_result = dynamic_learning_system.generate_enhanced_query_expansion(query)
        
        return {
            "expansion_result": expansion_result,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ 增強查詢擴展錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"增強查詢擴展錯誤: {str(e)}"})


def _calculate_concept_statistics(chunks_with_span: List[Dict[str, Any]]) -> Dict[str, Any]:
    """計算概念統計信息"""
    stats = {
        "total_chunks": len(chunks_with_span),
        "concept_chunks": 0,
        "definition_chunks": 0,
        "exception_chunks": 0,
        "condition_chunks": 0,
        "avg_importance_score": 0.0,
        "concept_density": 0.0
    }
    
    total_importance = 0.0
    total_concept_count = 0
    
    for chunk in chunks_with_span:
        metadata = chunk.get("metadata", {})
        semantic_features = metadata.get("semantic_features", {})
        
        concept_count = semantic_features.get("concept_count", 0)
        importance_score = semantic_features.get("importance_score", 0.0)
        
        if concept_count > 0:
            stats["concept_chunks"] += 1
            total_importance += importance_score
            total_concept_count += concept_count
        
        if semantic_features.get("has_definition", False):
            stats["definition_chunks"] += 1
        
        if semantic_features.get("has_exception", False):
            stats["exception_chunks"] += 1
        
        if semantic_features.get("has_condition", False):
            stats["condition_chunks"] += 1
    
    if stats["concept_chunks"] > 0:
        stats["avg_importance_score"] = total_importance / stats["concept_chunks"]
    
    if len(chunks_with_span) > 0:
        stats["concept_density"] = total_concept_count / len(chunks_with_span)
    
    return stats


def _register_default_strategies():
    """註冊默認檢索策略"""
    if not adaptive_rag:
        return
        
    # 註冊向量檢索
    adaptive_rag.register_strategy('vector_search', {
        'retrieve': lambda query, **kwargs: retrieve_original(query, kwargs.get('k', 5))
    })
    
    # 註冊HybridRAG
    adaptive_rag.register_strategy('hybrid_rag', {
        'retrieve': lambda query, **kwargs: hybrid_retrieve_original(query, kwargs.get('k', 5))
    })
    
    # 註冊多層次檢索
    adaptive_rag.register_strategy('hierarchical', {
        'retrieve': lambda query, **kwargs: []  # 暫時返回空列表
    })


def retrieve_original(query: str, k: int):
    """原始向量檢索"""
    # 這裡調用原有的檢索邏輯
    pass


async def hybrid_retrieve_original(query: str, k: int):
    """原始HybridRAG檢索"""
    # 暫時返回空列表，HybridRAG功能待實現
    return []


@app.get("/api/debug-experimental-groups")
async def debug_experimental_groups():
    """調試接口：查看實驗組存儲狀態"""
    return {
        "saved_groups": store.list_experimental_groups(),
        "experimental_group_embeddings_keys": list(store.experimental_group_embeddings.keys()),
        "current_multi_level_embeddings_keys": list(store.multi_level_embeddings.keys()),
        "experimental_group_details": {
            group_id: {
                "levels": list(group_data.get("multi_level_embeddings", {}).keys()),
                "has_data": len(group_data.get("multi_level_embeddings", {})) > 0
            }
            for group_id, group_data in store.experimental_group_embeddings.items()
        }
    }


@app.get("/api/embedding-databases")
async def list_embedding_databases():
    """列出所有可用的embedding資料庫"""
    databases = []
    print(f"🔍 API調用開始: has_multi_level_embeddings={store.has_multi_level_embeddings()}")
    if store.has_multi_level_embeddings():
        print(f"🔍 可用層次: {store.get_available_levels()}")
    
    # 強制測試多層次embedding
    print(f"🔍 強制檢查: multi_level_embeddings keys = {list(store.multi_level_embeddings.keys())}")
    
    # 手動檢查並創建測試數據
    if len(store.multi_level_embeddings) == 0:
        print("🔍 沒有多層次embedding，跳過合併邏輯")
    else:
        print(f"🔍 找到多層次embedding: {list(store.multi_level_embeddings.keys())}")
    
    # 為了演示目的，如果沒有embedding資料且沒有被標記為已刪除，則創建一些測試embedding資料
    # 移除自動創建演示embedding資料庫的邏輯
    # 現在只有在用戶實際創建embedding時才會顯示資料庫
    
    # 檢查標準embedding
    if store.embeddings is not None and store.chunks_flat:
        # 獲取相關文檔信息
        doc_info = {}
        for doc_id in set(store.chunk_doc_ids):
            doc = store.get_doc(doc_id)
            if doc:
                doc_info[doc_id] = {
                    "filename": doc.filename,
                    "json_data": doc.json_data is not None
                }
        
        databases.append({
            "id": "standard_embedding",
            "type": "standard",
            "name": "標準Embedding",
            "provider": "gemini",  # 從配置推斷
            "model": "gemini-embedding-001",
            "num_vectors": len(store.embeddings),
            "dimension": len(store.embeddings[0]) if store.embeddings else 0,
            "chunking_strategy": "basic",  # 需要從配置推斷
            "documents": list(doc_info.values()),
            "created_at": datetime.now().isoformat()
        })
    
    # 檢查多層次embedding - 按實驗組分別列出
    print(f"🔍 檢查多層次embedding: has_multi_level_embeddings={store.has_multi_level_embeddings()}")
    
    # 首先列出所有已保存的實驗組
    saved_groups = store.list_experimental_groups()
    print(f"🔍 已保存的實驗組: {saved_groups}")
    
    for group_id in saved_groups:
        if group_id not in GRANULARITY_COMBINATIONS:
            continue
        
        # 臨時加載該實驗組的數據以獲取信息
        group_data = store.experimental_group_embeddings.get(group_id, {})
        group_multi_level_embeddings = group_data.get("multi_level_embeddings", {})
        
        if not group_multi_level_embeddings:
            continue
        
        # 收集該實驗組的信息
        all_doc_info = {}
        total_vectors = 0
        providers = set()
        models = set()
        dimensions = set()
        levels_info = []
        
        for level in group_multi_level_embeddings.keys():
            level_data = {
                'embeddings': group_multi_level_embeddings.get(level, []),
                'chunks': group_data.get("multi_level_chunks_flat", {}).get(level, []),
                'doc_ids': group_data.get("multi_level_chunk_doc_ids", {}).get(level, []),
                'metadata': group_data.get("multi_level_metadata", {}).get(level, {})
            }
            
            # 收集文檔信息
            for doc_id in set(level_data.get('doc_ids', [])):
                doc = store.get_doc(doc_id)
                if doc:
                    all_doc_info[doc_id] = {
                        "filename": doc.filename,
                        "json_data": doc.json_data is not None
                    }
            
            # 統計信息
            level_vectors = len(level_data.get('embeddings', []))
            total_vectors += level_vectors
            providers.add(level_data.get('metadata', {}).get('provider', 'unknown'))
            models.add(level_data.get('metadata', {}).get('model', 'unknown'))
            dimensions.add(level_data.get('metadata', {}).get('dimension', 0))
            
            levels_info.append({
                "level": level,
                "description": get_level_description(level),
                "num_vectors": level_vectors
            })
        
        if total_vectors > 0:
            # 根據層次組合確定實驗組名稱
            level_names = [level["level"] for level in levels_info]
            combination = GRANULARITY_COMBINATIONS[group_id]
            group_name = combination["name"]
            
            databases.append({
                "id": f"experimental_group_{group_id}",
                "type": "multi_level",
                "name": f"實驗組Embedding - {group_name}",
                "provider": list(providers)[0] if providers else "unknown",
                "model": list(models)[0] if models else "unknown",
                "num_vectors": total_vectors,
                "dimension": list(dimensions)[0] if dimensions else 0,
                "chunking_strategy": "hierarchical",
                "documents": list(all_doc_info.values()),
                "levels": levels_info,
                "experimental_group": group_name,
                "experimental_group_id": group_id,
                "created_at": datetime.now().isoformat()
            })
    
    # 也檢查當前激活的多層次embedding（如果沒有保存到實驗組存儲中）
    # 如果檢測到有multi_level_embeddings但沒有對應的experimental_group_embeddings，嘗試自動遷移
    if store.has_multi_level_embeddings():
        available_levels = store.get_available_levels()
        
        # 收集所有層次的信息
        all_doc_info = {}
        total_vectors = 0
        providers = set()
        models = set()
        dimensions = set()
        levels_info = []
        
        for level in available_levels:
            level_data = store.get_multi_level_embeddings(level)
            if level_data:
                # 收集文檔信息
                for doc_id in set(level_data.get('doc_ids', [])):
                    doc = store.get_doc(doc_id)
                    if doc:
                        all_doc_info[doc_id] = {
                            "filename": doc.filename,
                            "json_data": doc.json_data is not None
                        }
                
                # 統計信息
                level_vectors = len(level_data.get('embeddings', []))
                total_vectors += level_vectors
                providers.add(level_data.get('metadata', {}).get('provider', 'unknown'))
                models.add(level_data.get('metadata', {}).get('model', 'unknown'))
                dimensions.add(level_data.get('metadata', {}).get('dimension', 0))
                
                levels_info.append({
                    "level": level,
                    "description": get_level_description(level),
                    "num_vectors": level_vectors
                })
        
        if total_vectors > 0:
            # 根據層次組合確定實驗組
            level_names = [level["level"] for level in levels_info]
            level_names_set = set(level_names)
            group_name = "未知實驗組"
            group_id = None
            
            # 修復實驗組識別邏輯
            if level_names_set == {"basic_unit"}:
                group_name = "A組：僅條文層 (Baseline)"
                group_id = "group_a"
            elif level_names_set == {"document_component", "basic_unit_hierarchy", "basic_unit"}:
                group_name = "B組：條文+章節結構"
                group_id = "group_b"
            elif level_names_set == {"basic_unit", "basic_unit_component", "enumeration"}:
                group_name = "C組：條文+細節層次"
                group_id = "group_c"
            elif level_names_set == {"document_component", "basic_unit_hierarchy", "basic_unit", "basic_unit_component", "enumeration"}:
                group_name = "D組：完整多層次ML-RAG"
                group_id = "group_d"
            
            # 如果識別出實驗組但沒有保存到實驗組存儲中，自動遷移
            if group_id and group_id not in saved_groups:
                print(f"🔄 檢測到未保存的實驗組 {group_id}，自動遷移到實驗組存儲...")
                store.save_experimental_group_embeddings(group_id)
                store.save_data()
                print(f"✅ 已自動遷移實驗組 {group_id} 到實驗組存儲")
                saved_groups.append(group_id)  # 更新列表，避免重複顯示
            
            # 檢查是否已經在實驗組存儲中（避免重複顯示）
            if group_id and group_id in saved_groups:
                # 已經在實驗組列表中顯示，跳過
                pass
            else:
                databases.append({
                    "id": "multi_level_combined" if not group_id else f"experimental_group_{group_id}",
                    "type": "multi_level",
                    "name": f"實驗組Embedding - {group_name}",
                    "provider": list(providers)[0] if providers else "unknown",
                    "model": list(models)[0] if models else "unknown",
                    "num_vectors": total_vectors,
                    "dimension": list(dimensions)[0] if dimensions else 0,
                    "chunking_strategy": "hierarchical",
                    "documents": list(all_doc_info.values()),
                    "levels": levels_info,
                    "experimental_group": group_name,
                    "experimental_group_id": group_id,
                    "created_at": datetime.now().isoformat()
                })
    
    return databases


@app.post("/api/embedding-databases/{database_id}/activate")
async def activate_embedding_database(database_id: str):
    """激活指定的embedding資料庫，加載對應的FAISS和BM25索引"""
    try:
        print(f"🔄 激活embedding資料庫: {database_id}")
        
        if database_id == "standard_embedding":
            # 檢查標準embedding是否存在
            if store.embeddings is None or not store.chunks_flat:
                return JSONResponse(
                    status_code=404,
                    content={"error": "標準embedding資料不存在，請先執行embedding"}
                )
            
            # 重新加載FAISS和BM25索引
            print("📊 重新加載FAISS和BM25索引...")
            faiss_store.load_data()
            bm25_index.load_data()
            
            # 驗證索引是否成功加載
            faiss_loaded = faiss_store.has_vectors()
            bm25_loaded = bm25_index.has_index()
            print(f"📊 索引加載狀態: FAISS={faiss_loaded}, BM25={bm25_loaded}")
            
            # 如果任一索引未加載，嘗試從store重建
            if not faiss_loaded or not bm25_loaded:
                print("⚠️ 部分或全部索引未找到，嘗試從store重建索引...")
                vectors = store.embeddings
                chunks = store.chunks_flat
                
                if not vectors or not chunks:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "error": "無法重建索引：store中沒有embedding數據",
                            "faiss_available": faiss_loaded,
                            "bm25_available": bm25_loaded
                        }
                    )
                
                chunk_ids = [f"{doc_id}_{i}" for i, doc_id in enumerate(store.chunk_doc_ids)]
                
                # 重建FAISS索引（如果不存在）
                if not faiss_loaded:
                    print("🔄 重建FAISS索引...")
                    dimension = len(vectors[0]) if vectors else EMBEDDING_DIMENSION
                    faiss_store.create_index(dimension, "flat")
                    faiss_store.add_vectors(vectors, chunk_ids, store.chunk_doc_ids, chunks)
                    print(f"✅ FAISS索引已重建: {len(vectors)} 個向量")
                
                # 重建BM25索引（如果不存在）
                if not bm25_loaded:
                    print("🔄 重建BM25索引...")
                    bm25_index.build_index(chunks, chunk_ids, store.chunk_doc_ids)
                    print(f"✅ BM25索引已重建: {len(chunks)} 個文檔")
                
                # 如果有enhanced metadata，也需要恢復
                if hasattr(store, 'enhanced_metadata') and store.enhanced_metadata:
                    for chunk_id, metadata in store.enhanced_metadata.items():
                        faiss_store.set_enhanced_metadata(chunk_id, metadata)
                    print(f"✅ 已恢復 {len(store.enhanced_metadata)} 個enhanced metadata")
                
                # 保存索引
                faiss_store.save_data()
                bm25_index.save_data()
                print("✅ 索引已保存到磁盤")
                
                # 再次驗證
                faiss_loaded = faiss_store.has_vectors()
                bm25_loaded = bm25_index.has_index()
                print(f"📊 重建後索引狀態: FAISS={faiss_loaded}, BM25={bm25_loaded}")
            
            print(f"✅ 標準embedding資料庫已激活")
            return {
                "message": "標準embedding資料庫已激活",
                "database_id": database_id,
                "faiss_available": faiss_store.has_vectors(),
                "bm25_available": bm25_index.has_index(),
                "num_vectors": len(store.embeddings) if store.embeddings else 0,
                "success": True
            }
            
        elif database_id.startswith("experimental_group_"):
            # 激活特定實驗組的embedding
            # 處理可能的重複前綴（如 experimental_group_group_b）
            group_id = database_id.replace("experimental_group_", "")
            # 如果還有前綴，再次移除
            if group_id.startswith("group_"):
                # 已經是正確格式，不需要再處理
                pass
            print(f"🔄 激活實驗組: {group_id} (原始database_id: {database_id})")
            
            if group_id not in GRANULARITY_COMBINATIONS:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"未知的實驗組: {group_id}"}
                )
            
            # 檢查實驗組數據是否存在
            if group_id not in store.experimental_group_embeddings:
                print(f"⚠️ 實驗組 {group_id} 不在存儲中，已保存的實驗組: {store.list_experimental_groups()}")
                return JSONResponse(
                    status_code=404,
                    content={"error": f"實驗組 {group_id} 的embedding資料不存在，請先執行該實驗組的embedding。已保存的實驗組: {', '.join(store.list_experimental_groups())}"}
                )
            
            # 從實驗組存儲中加載數據
            if not store.load_experimental_group_embeddings(group_id):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"實驗組 {group_id} 的embedding資料加載失敗"}
                )
            
            # 重建FAISS和BM25索引
            print("📊 重建實驗組的FAISS和BM25索引...")
            
            # 清除現有索引
            faiss_store.reset_vectors()
            bm25_index.reset_index()
            
            # 為每個層次重建索引
            available_levels = store.get_available_levels()
            for level_name in available_levels:
                level_data = store.get_multi_level_embeddings(level_name)
                if not level_data:
                    continue
                
                vectors = level_data.get('embeddings', [])
                chunks = level_data.get('chunks', [])
                doc_ids = level_data.get('doc_ids', [])
                
                if not vectors or not chunks:
                    continue
                
                # 重建FAISS索引
                level_chunk_ids = [f"{level_name}_{doc_id}_{i}" for i, doc_id in enumerate(doc_ids)]
                faiss_store.add_multi_level_vectors(level_name, vectors, level_chunk_ids, doc_ids, chunks)
                
                # 重建BM25索引
                bm25_index.build_multi_level_index(level_name, chunks, level_chunk_ids, doc_ids)
                
                print(f"✅ 層次 '{level_name}' 索引已重建: {len(vectors)} 個向量")
            
            # 保存索引
            faiss_store.save_data()
            bm25_index.save_data()
            
            combination = GRANULARITY_COMBINATIONS[group_id]
            print(f"✅ 實驗組 {group_id} ({combination['name']}) 已激活")
            return {
                "message": f"實驗組 {group_id} 已激活",
                "database_id": database_id,
                "experimental_group_id": group_id,
                "experimental_group_name": combination['name'],
                "faiss_available": faiss_store.has_multi_level_vectors(),
                "bm25_available": bm25_index.has_multi_level_index(),
                "available_levels": available_levels,
                "success": True
            }
            
        elif database_id == "multi_level_combined":
            # 檢查多層次embedding是否存在
            if not store.has_multi_level_embeddings():
                return JSONResponse(
                    status_code=404,
                    content={"error": "多層次embedding資料不存在，請先執行multi-level-embed"}
                )
            
            # 重新加載FAISS和BM25索引
            print("📊 重新加載多層次FAISS和BM25索引...")
            faiss_store.load_data()
            bm25_index.load_data()
            
            # 驗證多層次索引是否成功加載
            available_levels = faiss_store.get_available_levels()
            if not available_levels:
                # 如果加載失敗，嘗試從store重建索引
                print("⚠️ 多層次索引未找到，嘗試從store重建索引...")
                available_levels = store.get_available_levels()
                
                for level_name in available_levels:
                    level_data = store.get_multi_level_embeddings(level_name)
                    if level_data:
                        vectors = level_data.get('embeddings', [])
                        chunks = level_data.get('chunks', [])
                        doc_ids = level_data.get('doc_ids', [])
                        chunk_ids = [f"{doc_id}_{i}" for i, doc_id in enumerate(doc_ids)]
                        
                        if vectors and chunks:
                            faiss_store.add_multi_level_vectors(level_name, vectors, chunk_ids, doc_ids, chunks)
                            bm25_index.build_multi_level_index(level_name, chunks, chunk_ids, doc_ids)
                
                # 保存索引
                faiss_store.save_data()
                bm25_index.save_data()
                print("✅ 已從store重建多層次索引並保存")
            
            print(f"✅ 多層次embedding資料庫已激活，可用層次: {available_levels}")
            return {
                "message": "多層次embedding資料庫已激活",
                "database_id": database_id,
                "faiss_available": faiss_store.has_multi_level_vectors(),
                "bm25_available": bm25_index.has_multi_level_index(),
                "available_levels": available_levels,
                "success": True
            }
        else:
            return JSONResponse(
                status_code=404,
                content={"error": f"未知的embedding資料庫ID: {database_id}"}
            )
            
    except Exception as e:
        print(f"❌ 激活embedding資料庫失敗: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"激活embedding資料庫失敗: {str(e)}"}
        )


@app.delete("/api/embedding-databases/{database_id}")
async def delete_embedding_database(database_id: str):
    """刪除指定的embedding資料庫"""
    try:
        if database_id == "standard_embedding":
            # 刪除標準embedding
            if store.embeddings is not None:
                store.reset_embeddings()
                store.save_data()
                # 標記演示資料已被刪除，防止重新創建
                store.demo_data_deleted = True
                print(f"✅ 已刪除標準embedding資料庫")
                return {"message": "標準embedding資料庫已刪除", "success": True}
            else:
                return JSONResponse(
                    status_code=404, 
                    content={"error": "標準embedding資料庫不存在"}
                )
        elif database_id == "multi_level_combined":
            # 刪除整個多層次embedding資料庫
            if store.has_multi_level_embeddings() or faiss_store.has_multi_level_vectors() or bm25_index.has_multi_level_index():
                # 清除所有多層次embedding數據
                store.multi_level_embeddings = {}
                store.multi_level_chunk_doc_ids = {}
                store.multi_level_chunks_flat = {}
                store.multi_level_metadata = {}
                store.save_data()
                
                # 清除FAISS和BM25多層次索引
                faiss_store.reset_vectors()
                bm25_index.reset_index()
                
                # 刪除磁盤上的索引文件
                import os
                data_dir = "data"
                for level_name in ["document", "document_component", "basic_unit_hierarchy", "basic_unit", "basic_unit_component", "enumeration"]:
                    faiss_file = os.path.join(data_dir, f"faiss_index_{level_name}.bin")
                    bm25_file = os.path.join(data_dir, f"bm25_index_{level_name}.pkl")
                    if os.path.exists(faiss_file):
                        os.remove(faiss_file)
                        print(f"🗑️ 刪除FAISS文件: {faiss_file}")
                    if os.path.exists(bm25_file):
                        os.remove(bm25_file)
                        print(f"🗑️ 刪除BM25文件: {bm25_file}")
                
                # 重新保存空的metadata
                faiss_store.save_data()
                bm25_index.save_data()
                
                print(f"✅ 已刪除整個多層次embedding資料庫（包括磁盤文件）")
                return {"message": "多層次embedding資料庫已刪除（包括磁盤文件）", "success": True}
            else:
                return JSONResponse(
                    status_code=404, 
                    content={"error": "多層次embedding資料庫不存在"}
                )
        elif database_id.startswith("multi_level_"):
            # 刪除特定層次的多層次embedding（保留向後兼容性）
            level_name = database_id.replace("multi_level_", "")
            if store.has_multi_level_embeddings():
                available_levels = store.get_available_levels()
                if level_name in available_levels:
                    # 刪除特定層次
                    if level_name in store.multi_level_embeddings:
                        del store.multi_level_embeddings[level_name]
                    if level_name in store.multi_level_chunk_doc_ids:
                        del store.multi_level_chunk_doc_ids[level_name]
                    if level_name in store.multi_level_chunks_flat:
                        del store.multi_level_chunks_flat[level_name]
                    if level_name in store.multi_level_metadata:
                        del store.multi_level_metadata[level_name]
                    
                    print(f"✅ 已刪除多層次embedding層次: {level_name}")
                    return {"message": f"多層次embedding層次 '{level_name}' 已刪除", "success": True}
                else:
                    return JSONResponse(
                        status_code=404, 
                        content={"error": f"多層次embedding層次 '{level_name}' 不存在"}
                    )
            else:
                    return JSONResponse(
                        status_code=404, 
                        content={"error": "多層次embedding資料庫不存在"}
                    )
        elif database_id.startswith("experimental_group_"):
            # 刪除特定實驗組的embedding
            # 處理可能的重複前綴
            group_id = database_id.replace("experimental_group_", "")
            print(f"🗑️ 刪除實驗組: {group_id} (原始database_id: {database_id})")
            
            if group_id not in GRANULARITY_COMBINATIONS:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"未知的實驗組: {group_id}"}
                )
            
            if group_id not in store.experimental_group_embeddings:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"實驗組 {group_id} 的embedding資料不存在"}
                )
            
            # 刪除實驗組數據
            del store.experimental_group_embeddings[group_id]
            store.save_data()
            
            # 如果當前激活的是這個實驗組，清除當前激活的數據
            current_levels = store.get_available_levels()
            if current_levels:
                # 檢查當前激活的層次是否屬於這個實驗組
                combination = GRANULARITY_COMBINATIONS[group_id]
                group_levels = set(combination["levels"])
                current_levels_set = set(current_levels)
                
                # 如果當前激活的層次完全匹配這個實驗組，清除當前激活的數據
                if current_levels_set == group_levels:
                    store.multi_level_embeddings = {}
                    store.multi_level_chunk_doc_ids = {}
                    store.multi_level_chunks_flat = {}
                    store.multi_level_metadata = {}
                    store.save_data()
                    
                    # 清除FAISS和BM25索引
                    faiss_store.reset_vectors()
                    bm25_index.reset_index()
                    faiss_store.save_data()
                    bm25_index.save_data()
            
            combination = GRANULARITY_COMBINATIONS[group_id]
            print(f"✅ 已刪除實驗組 {group_id} ({combination['name']}) 的embedding資料")
            return {
                "message": f"實驗組 {group_id} ({combination['name']}) 的embedding資料已刪除",
                "success": True
            }
        else:
            return JSONResponse(
                status_code=400, 
                content={"error": f"不支持的embedding資料庫類型: {database_id}"}
            )
    except Exception as e:
        print(f"❌ 刪除embedding資料庫失敗: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"刪除embedding資料庫失敗: {str(e)}"}
        )


@app.post("/api/reset-demo-data")
async def reset_demo_data():
    """重置演示資料狀態，用於測試目的"""
    try:
        store.demo_data_deleted = False
        store.reset_embeddings()
        store.save_data()
        print("✅ 已重置演示資料狀態")
        return {"message": "演示資料狀態已重置", "success": True}
    except Exception as e:
        print(f"❌ 重置演示資料失敗: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"重置演示資料失敗: {str(e)}"}
        )


@app.post("/api/clear-all-data")
async def clear_all_data():
    """清除所有數據（用於測試）"""
    try:
        store.clear_all_data()
        print("🗑️ 所有數據已清除")
        return {"message": "All data cleared successfully", "success": True}
    except Exception as e:
        print(f"❌ 清除數據失敗: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"清除數據失敗: {str(e)}"}
        )


@app.get("/api/debug-store")
async def debug_store():
    """調試store狀態"""
    return {
        "has_standard_embeddings": store.embeddings is not None,
        "has_multi_level_embeddings": store.has_multi_level_embeddings(),
        "available_levels": store.get_available_levels(),
        "multi_level_embeddings_keys": list(store.multi_level_embeddings.keys()),
        "multi_level_embeddings_details": {
            level: {
                "num_vectors": len(store.multi_level_embeddings.get(level, {}).get('embeddings', [])),
                "metadata": store.multi_level_embeddings.get(level, {}).get('metadata', {})
            }
            for level in store.get_available_levels()
        },
        "demo_data_deleted": getattr(store, 'demo_data_deleted', False),
        "docs_count": len(store.docs)
    }


@app.get("/api/docs")
async def list_docs():
    """列出所有文檔"""
    docs = store.list_docs()
    return [{"id": d.id, "filename": d.filename, "num_chars": len(d.text)} for d in docs]


# 定義粒度組合配置 - 對應論文的六個層次
GRANULARITY_COMBINATIONS = {
    # A組：僅層次 4 (基本單元層 - 條文)
    "group_a": {
        "name": "A組：僅條文層 (Baseline)",
        "description": "傳統平面法的表現 - 僅使用基本單元層（條文）",
        "levels": ["basic_unit"],
        "research_purpose": "基線對照組，評估傳統平面檢索的表現"
    },
    
    # B組：層次 2 + 3 + 4 (文件組件層 + 基本單元層級層 + 基本單元層)
    "group_b": {
        "name": "B組：條文+章節結構",
        "description": "基本單元層 + 基本單元層級層（章、節、編）",
        "levels": ["document_component", "basic_unit_hierarchy", "basic_unit"],
        "research_purpose": "評估結構分組（如：《商標法》的「章、節」）的嵌入是否能更好地捕捉廣泛主題(aboutness)"
    },
    
    # C組：層次 4 + 5 + 6 (基本單元層 + 基本單元組成層 + 列舉層)
    "group_c": {
        "name": "C組：條文+細節層次",
        "description": "基本單元層 + 基本單元組成層（項）+ 列舉層（款、目）",
        "levels": ["basic_unit", "basic_unit_component", "enumeration"],
        "research_purpose": "評估細節化層次對於處理臺灣法律中常見的列舉式規定（如：《商標法》第30條的15款不得註冊情形）所帶來的精確度增益"
    },
    
    # D組：層次 2 + 3 + 4 + 5 + 6 (章、節、條文、項、款、目)
    "group_d": {
        "name": "D組：完整多層次ML-RAG",
        "description": "章、節、條文、項、款、目層級",
        "levels": ["document_component", "basic_unit_hierarchy", 
                   "basic_unit", "basic_unit_component", "enumeration"],
        "research_purpose": "作為最佳效能的對比組，評估完整多層次方法的綜合表現"
    },
    
    # E組：LLM章節導向 + 細節檢索
    "group_e": {
        "name": "E組：LLM章節導向 + 細節RAG",
        "description": "Gemini-2.5 Flash Thinking 選章→章節內C組細節檢索",
        "levels": ["document_component", "basic_unit_hierarchy",
                   "basic_unit", "basic_unit_component", "enumeration"],
        "research_purpose": "評估先由LLM鎖定章節，再於章節內進行細粒度檢索的效益",
        "strategy": "llm_guided"
    },
    
    # 額外的對比組合，用於更細緻的分析
    "document_only": {
        "name": "僅文件層",
        "description": "僅使用文件層級embedding",
        "levels": ["document"],
        "research_purpose": "評估最高層級結構的獨立貢獻"
    },
    
    "structure_only": {
        "name": "僅結構層",
        "description": "僅使用結構層次（文件、文件組件、基本單元層級）",
        "levels": ["document", "document_component", "basic_unit_hierarchy"],
        "research_purpose": "評估純結構層次的貢獻，不包含具體內容"
    },
    
    "content_only": {
        "name": "僅內容層",
        "description": "僅使用內容層次（條文、項、款目）",
        "levels": ["basic_unit", "basic_unit_component", "enumeration"],
        "research_purpose": "評估純內容層次的貢獻，不包含高層結構"
    }
}


@app.get("/api/granularity-combinations")
def get_granularity_combinations():
    """獲取可用的粒度組合配置"""
    return {"combinations": GRANULARITY_COMBINATIONS}


@app.post("/api/test-experimental-groups")
async def test_experimental_groups(req: Dict[str, Any]):
    """測試實驗組層次選擇邏輯"""
    experimental_groups = req.get("experimental_groups", [])
    
    if not experimental_groups:
        return {"message": "請提供experimental_groups參數"}
    
    # 模擬實驗組選擇邏輯
    six_levels = [
        'document', 'document_component', 'basic_unit_hierarchy', 
        'basic_unit', 'basic_unit_component', 'enumeration'
    ]
    
    print(f"🧪 測試實驗組選擇: {experimental_groups}")
    
    # 收集所有需要的層次
    required_levels = set()
    group_details = {}
    
    for group_key in experimental_groups:
        if group_key in GRANULARITY_COMBINATIONS:
            group_info = GRANULARITY_COMBINATIONS[group_key]
            group_levels = group_info["levels"]
            required_levels.update(group_levels)
            
            group_details[group_key] = {
                "name": group_info["name"],
                "description": group_info["description"],
                "levels": group_levels,
                "research_purpose": group_info["research_purpose"]
            }
        else:
            group_details[group_key] = {"error": "未知的實驗組"}
    
    # 確定要處理的層次
    selected_levels = [level for level in six_levels if level in required_levels]
    skipped_levels = [level for level in six_levels if level not in required_levels]
    
    return {
        "experimental_groups": experimental_groups,
        "group_details": group_details,
        "all_levels": six_levels,
        "selected_levels": selected_levels,
        "skipped_levels": skipped_levels,
        "total_selected": len(selected_levels),
        "total_skipped": len(skipped_levels)
    }


@app.post("/api/granularity-comparison-retrieve")
async def granularity_comparison_retrieve(req: Dict[str, Any]):
    """
    使用指定粒度組合進行檢索
    req = {query, k, granularity_combination}
    """
    query = req.get("query")
    k = req.get("k", 10)
    combination_key = req.get("granularity_combination", "full_ml")
    
    # 獲取層次組合配置
    combination = GRANULARITY_COMBINATIONS.get(combination_key)
    if not combination:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown combination: {combination_key}"}
        )
    
    selected_levels = combination["levels"]
    
    # 生成查詢向量
    if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
        query_vector = (await embed_gemini([query]))[0]
    elif USE_BGE_M3_EMBEDDING:
        query_vector = embed_bge_m3([query])[0]
    else:
        return JSONResponse(status_code=400, content={"error": "No embedding method available"})
    
    # 從選定的層次中檢索並融合結果
    all_results = []
    level_contributions = {}
    
    for level_name in selected_levels:
        level_data = store.get_multi_level_embeddings(level_name)
        if not level_data:
            continue
        
        vectors = np.array(level_data['embeddings'])
        chunks = level_data['chunks']
        doc_ids = level_data['doc_ids']
        
        # 計算相似度
        similarities = cosine_similarity([query_vector], vectors)[0]
        
        # 獲取該層次的top-k結果
        top_indices = np.argsort(similarities)[::-1][:k]
        
        level_results = []
        for idx in top_indices:
            result = {
                "content": chunks[idx],
                "similarity": float(similarities[idx]),
                "level": level_name,
                "doc_id": doc_ids[idx],
                "chunk_index": int(idx)
            }
            level_results.append(result)
            all_results.append(result)
        
        level_contributions[level_name] = {
            "results": level_results,
            "total_chunks": len(chunks),
            "avg_similarity": float(np.mean([r["similarity"] for r in level_results]))
        }
    
    # 融合結果（按相似度排序）
    fused_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)
    
    return {
        "query": query,
        "combination": combination,
        "level_contributions": level_contributions,
        "fused_results": fused_results[:k],
        "total_results": len(all_results)
    }


@app.post("/api/annotations/save")
async def save_annotations(req: AnnotationBatchRequest):
    """保存E/C/U標註"""
    saved_annotations = []
    
    for idx_str, label in req.annotations.items():
        idx = int(idx_str)
        if idx >= len(req.results):
            continue
            
        result = req.results[idx]
        annotation = ECUAnnotation(
            annotation_id=str(uuid.uuid4()),
            query=req.query,
            chunk_content=result["content"],
            chunk_index=idx,
            level=result.get("level", "unknown"),
            doc_id=result.get("doc_id", ""),
            relevance_label=label,
            annotator="user",
            timestamp=datetime.now().isoformat()
        )
        store.save_annotation(annotation)
        saved_annotations.append(annotation)
    
    return {"saved": len(saved_annotations), "annotations": saved_annotations}


@app.get("/api/annotations/stats")
def get_annotation_stats(query: Optional[str] = None):
    """獲取標註統計"""
    if query:
        annotations = store.get_annotations_for_query(query)
    else:
        annotations = store.get_all_annotations()
    
    stats = {
        "total": len(annotations),
        "by_label": {
            "E": sum(1 for a in annotations if a.relevance_label == 'E'),
            "C": sum(1 for a in annotations if a.relevance_label == 'C'),
            "U": sum(1 for a in annotations if a.relevance_label == 'U')
        },
        "by_level": {}
    }
    
    for annotation in annotations:
        level = annotation.level
        if level not in stats["by_level"]:
            stats["by_level"][level] = {"E": 0, "C": 0, "U": 0}
        stats["by_level"][level][annotation.relevance_label] += 1
    
    return stats


@app.get("/api/annotations/query/{query}")
def get_annotations_for_query(query: str):
    """獲取特定查詢的所有標註"""
    annotations = store.get_annotations_for_query(query)
    return {"query": query, "annotations": annotations}


@app.delete("/api/annotations/query/{query}")
def delete_annotations_for_query(query: str):
    """刪除特定查詢的所有標註"""
    store.delete_annotations_for_query(query)
    return {"message": f"Deleted annotations for query: {query}"}


def calculate_ecu_metrics(annotations: List[ECUAnnotation], k_values: List[int]) -> Dict:
    """基於標註計算E/C/U指標"""
    metrics = {}
    
    # 按相似度排序（假設有similarity字段）
    sorted_annotations = sorted(annotations, key=lambda x: getattr(x, 'similarity', 0), reverse=True)
    
    for k in k_values:
        top_k = sorted_annotations[:k]
        e_count = sum(1 for a in top_k if a.relevance_label == 'E')
        c_count = sum(1 for a in top_k if a.relevance_label == 'C')
        u_count = sum(1 for a in top_k if a.relevance_label == 'U')
        
        metrics[f"E@{k}"] = (e_count / k) * 100 if k > 0 else 0
        metrics[f"C@{k}"] = (c_count / k) * 100 if k > 0 else 0
        metrics[f"U@{k}"] = (u_count / k) * 100 if k > 0 else 0
        metrics[f"E+C@{k}"] = ((e_count + c_count) / k) * 100 if k > 0 else 0
    
    return metrics


@app.get("/api/chapters/catalog")
async def get_chapter_catalog(
    doc_id: Optional[str] = None,
    limit: Optional[int] = None,
    max_sections: Optional[int] = 5
):
    if max_sections is not None and max_sections <= 0:
        max_sections = None
    catalog = await _build_chapter_section_catalog(
        doc_id=doc_id,
        limit_chapters=limit,
        max_sections_per_chapter=max_sections,
    )
    entries = _get_chapter_summary_entries(doc_id)
    document_entries = [
        e for e in entries if e.get("level") == "document"
    ]
    documents = sorted(
        [
            {
                "doc_id": e.get("doc_id"),
                "law_name": e.get("law_name"),
                "title": e.get("title"),
                "chunk_id": e.get("chunk_id"),
                "summary": e.get("routing_summary") or e.get("summary"),
                "last_updated": e.get("last_updated"),
            }
            for e in document_entries
        ],
        key=lambda d: (d.get("law_name") or "", d.get("title") or ""),
    )
    doc_ids = sorted({e.get("doc_id") for e in entries if e.get("doc_id")})
    stats = {
        "total_entries": len(entries),
        "total_documents": len(document_entries),
        "total_chapters": sum(1 for e in entries if e.get("level") == "chapter"),
        "total_sections": sum(1 for e in entries if e.get("level") == "section"),
        "doc_ids": doc_ids,
        "last_updated": max((e.get("last_updated") for e in entries if e.get("last_updated")), default=None),
    }
    return {"catalog": catalog, "documents": documents, "stats": stats}


@app.post("/api/chapters/summarize")
async def summarize_chapters(req: Dict[str, Any]):
    doc_id = req.get("doc_id")
    include_sections = req.get("include_sections", True)
    max_items = req.get("max_items")
    target_chunk_ids = req.get("target_chunk_ids")
    if target_chunk_ids and not isinstance(target_chunk_ids, (list, set, tuple)):
        return JSONResponse(status_code=400, content={"error": "target_chunk_ids must be a list"})
    summary_stats = await ensure_chapter_summaries(
        doc_id=doc_id,
        include_sections=include_sections,
        max_items=max_items,
        target_chunk_ids=set(target_chunk_ids) if target_chunk_ids else None,
    )
    catalog = await _build_chapter_section_catalog(doc_id=doc_id)
    summary_stats["catalog"] = catalog
    return summary_stats


@app.delete("/api/chapters/summarize")
async def delete_chapter_summaries(doc_id: Optional[str] = None):
    result = _reset_chapter_summaries(doc_id=doc_id)
    return result


@app.post("/api/group-e/route")
async def route_group_e_chapters(req: Dict[str, Any]):
    doc_id = req.get("doc_id")
    query = req.get("query")
    max_chapters = req.get("max_chapters", 24)
    max_sections = req.get("max_sections", 3)
    if not doc_id or not query:
        return JSONResponse(status_code=400, content={"error": "doc_id and query are required"})
    catalog = await _build_chapter_section_catalog(
        doc_id=doc_id,
        limit_chapters=max_chapters,
        max_sections_per_chapter=max_sections,
    )
    if not catalog:
        return JSONResponse(
            status_code=400,
            content={"error": "找不到章節摘要，請先執行章/節摘要生成"}
        )
    try:
        llm_raw = await _llm_select_relevant_chapters(query, catalog)
        selection = _prepare_selection_sets(llm_raw, catalog)
    except (RuntimeError, ValueError) as e:
        error_msg = str(e)
        print(f"❌ 實驗組E章節路由失敗: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "章節路由失敗",
                "message": error_msg,
                "detail": "LLM 無法完成章節選擇，請檢查網絡連接、API 配置或重新嘗試"
            }
        )
    selection_payload = {
        "chapter_chunk_ids": list(selection.get("chapter_chunk_ids", set())),
        "section_chunk_ids": list(selection.get("section_chunk_ids", set())),
        "chapter_titles": list(selection.get("chapter_titles", set())),
        "section_titles": list(selection.get("section_titles", set())),
        "chapter_keys": list(selection.get("chapter_keys", set())),
        "section_keys": list(selection.get("section_keys", set())),
    }
    return {
        "doc_id": doc_id,
        "query": query,
        "llm_stage": {
            "selection_details": selection.get("details", []),
            "thinking": selection.get("thinking", ""),
            "fallback_used": selection.get("fallback_used", False),
            "raw_response": llm_raw,
        },
        "selection": selection_payload,
    }


@app.get("/api/chapters/summary-status")
async def get_chapter_summary_status(doc_id: Optional[str] = None):
    if not doc_id:
        return JSONResponse(status_code=400, content={"error": "doc_id is required"})
    doc = store.get_doc(doc_id)
    if not doc:
        return JSONResponse(status_code=404, content={"error": "Document not found"})
    multi_chunks = ensure_multi_level_chunks(doc)
    if not multi_chunks:
        return {
            "doc_id": doc_id,
            "doc_name": doc.filename,
            "document_chunks": [],
            "chapters": [],
            "orphan_sections": [],
            "stats": {
                "total_chapters": 0,
                "total_sections": 0,
                "summarized_chapters": 0,
                "summarized_sections": 0,
            },
        }
    
    summary_map = _get_summary_entry_map(doc_id)
    document_status = []
    for doc_chunk in multi_chunks.get("document") or []:
        if not isinstance(doc_chunk, dict):
            continue
        metadata = (doc_chunk.get("metadata") or {}).copy()
        title = metadata.get("title") or metadata.get("law_name") or doc.filename
        chunk_id = _ensure_chunk_id_for_metadata(doc_chunk, metadata, "document")
        entry = summary_map.get(chunk_id) or {}
        document_status.append({
            "chunk_id": chunk_id,
            "title": title,
            "has_summary": bool(entry),
            "summary": entry.get("routing_summary") or entry.get("summary") or "",
            "last_updated": entry.get("last_updated"),
            "level": "document",
        })
    
    chapters_raw = multi_chunks.get("document_component") or []
    chapter_entries: List[Dict[str, Any]] = []
    chapter_index_by_id: Dict[str, Dict[str, Any]] = {}
    chapter_index_by_key: Dict[str, Dict[str, Any]] = {}
    summarized_chapters = 0
    
    for chapter_chunk in chapters_raw:
        if not isinstance(chapter_chunk, dict):
            continue
        metadata = (chapter_chunk.get("metadata") or {}).copy()
        title = metadata.get("chapter") or metadata.get("title") or metadata.get("level") or "章節"
        chunk_id = _ensure_chunk_id_for_metadata(chapter_chunk, metadata, "chapter")
        chapter_key = _normalize_label(title) or chunk_id.lower()
        entry = summary_map.get(chunk_id) or {}
        has_summary = bool(entry)
        if has_summary:
            summarized_chapters += 1
        chapter_entry = {
            "chunk_id": chunk_id,
            "title": title,
            "has_summary": has_summary,
            "summary": entry.get("routing_summary") or entry.get("summary") or "",
            "last_updated": entry.get("last_updated"),
            "sections": [],
            "level": "chapter",
        }
        chapter_entries.append(chapter_entry)
        chapter_index_by_id[chunk_id] = chapter_entry
        chapter_index_by_key[chapter_key] = chapter_entry
    
    sections_raw = multi_chunks.get("basic_unit_hierarchy") or []
    total_sections = 0
    summarized_sections = 0
    orphan_sections: List[Dict[str, Any]] = []
    
    for section_chunk in sections_raw:
        if not isinstance(section_chunk, dict):
            continue
        metadata = (section_chunk.get("metadata") or {}).copy()
        section_title = metadata.get("section") or metadata.get("title") or "節"
        chunk_id = _ensure_chunk_id_for_metadata(section_chunk, metadata, "section")
        chapter_label = metadata.get("chapter") or ""
        parent_chunk_id = metadata.get("parent_chunk_id") or metadata.get("parent_id")
        entry = summary_map.get(chunk_id) or {}
        has_summary = bool(entry)
        total_sections += 1
        if has_summary:
            summarized_sections += 1
        section_entry = {
            "chunk_id": chunk_id,
            "title": section_title,
            "chapter_title": chapter_label,
            "has_summary": has_summary,
            "summary": entry.get("routing_summary") or entry.get("summary") or "",
            "last_updated": entry.get("last_updated"),
            "level": "section",
        }
        parent = None
        if parent_chunk_id and parent_chunk_id in chapter_index_by_id:
            parent = chapter_index_by_id[parent_chunk_id]
        else:
            normalized_parent = _normalize_label(chapter_label)
            if normalized_parent and normalized_parent in chapter_index_by_key:
                parent = chapter_index_by_key[normalized_parent]
        if parent:
            parent["sections"].append(section_entry)
        else:
            orphan_sections.append(section_entry)
    
    stats = {
        "total_chapters": len(chapter_entries),
        "total_sections": total_sections,
        "summarized_chapters": summarized_chapters,
        "summarized_sections": summarized_sections,
    }
    
    return {
        "doc_id": doc_id,
        "doc_name": doc.filename,
        "document_chunks": document_status,
        "chapters": chapter_entries,
        "orphan_sections": orphan_sections,
        "stats": stats,
    }


@app.post("/api/experimental-groups-generate-embeddings")
async def experimental_groups_generate_embeddings(req: Dict[str, Any]):
    """
    為不同實驗組生成對應層次的embedding
    req = {
        "doc_id": str,
        "groups_to_embed": List[str]  # ["group_a", "group_b", "group_c", "group_d", "group_e"]
    }
    """
    doc_id = req.get("doc_id")
    groups_to_embed = req.get("groups_to_embed", ["group_a", "group_b", "group_c", "group_d", "group_e"])
    
    if not doc_id:
        return JSONResponse(status_code=400, content={"error": "Document ID is required"})
    
    doc = store.get_doc(doc_id)
    if not doc:
        return JSONResponse(status_code=404, content={"error": "Document not found"})
    
    results = {}
    
    for group_key in groups_to_embed:
        if group_key not in GRANULARITY_COMBINATIONS:
            continue
            
        combination = GRANULARITY_COMBINATIONS[group_key]
        selected_levels = combination["levels"]
        
        # 為該實驗組生成embedding
        group_results = {
            "group_info": combination,
            "levels_processed": [],
            "total_chunks": 0,
            "embedding_status": "processing"
        }
        
        try:
            # 獲取該組需要的層次數據
            for level_name in selected_levels:
                # 檢查是否已有該層次的embedding
                existing_data = store.get_multi_level_embeddings(level_name)
                if existing_data and len(existing_data['embeddings']) > 0:
                    group_results["levels_processed"].append({
                        "level": level_name,
                        "status": "existing",
                        "chunk_count": len(existing_data['chunks'])
                    })
                    group_results["total_chunks"] += len(existing_data['chunks'])
                else:
                    # 需要生成該層次的embedding
                    group_results["levels_processed"].append({
                        "level": level_name,
                        "status": "missing",
                        "chunk_count": 0
                    })
            
            results[group_key] = group_results
            
        except Exception as e:
            results[group_key] = {
                "group_info": combination,
                "error": str(e),
                "embedding_status": "error"
            }
    
    return {
        "doc_id": doc_id,
        "groups_processed": list(results.keys()),
        "results": results,
        "message": "請先為需要的層次生成embedding，然後再進行實驗組對比"
    }


@app.post("/api/experimental-groups-batch-retrieve")
async def experimental_groups_batch_retrieve(req: Dict[str, Any]):
    """
    批量檢索不同實驗組的結果，用於對比實驗
    注意：需要先為各實驗組生成對應的embedding
    req = {
        "query": str,
        "k": int,
        "groups_to_test": List[str]  # ["group_a", "group_b", "group_c", "group_d", "group_e"]
    }
    """
    query = req.get("query")
    k = req.get("k", 10)
    groups_to_test = req.get("groups_to_test", ["group_a", "group_b", "group_c", "group_d", "group_e"])
    doc_id = req.get("doc_id")

    # 臨時策略（為了讓當前 RQ 實驗穩定可跑）：
    # 只要要測 group_e，就強制使用目前 chapter_summaries.json 對應的 doc_id，
    # 避免前端或其他客戶端傳入沒有章節摘要的 doc_id。
    KNOWN_GROUP_E_DOC_ID = "71c23286-d3fb-4982-a2e2-33a931687c5d"
    if "group_e" in groups_to_test:
        original_doc_id = doc_id
        if doc_id != KNOWN_GROUP_E_DOC_ID:
            print(
                f"[experimental-groups-batch-retrieve] ⚠️ 強制覆寫傳入的 doc_id={original_doc_id!r} "
                f"為已知可用的 doc_id={KNOWN_GROUP_E_DOC_ID}（group_e 專用，確保與 chapter_summaries.json 一致）"
            )
        doc_id = KNOWN_GROUP_E_DOC_ID  # 無論如何都使用正確的 doc_id
    
    if not query:
        return JSONResponse(status_code=400, content={"error": "Query is required"})
    
    # 檢查各實驗組是否有對應的embedding
    missing_embeddings = []
    for group_key in groups_to_test:
        if group_key not in GRANULARITY_COMBINATIONS:
            continue
        combination = GRANULARITY_COMBINATIONS[group_key]
        for level_name in combination["levels"]:
            level_data = store.get_multi_level_embeddings(level_name)
            if not level_data or len(level_data['embeddings']) == 0:
                missing_embeddings.append(f"{group_key}: {level_name}")
    
    if missing_embeddings:
        return JSONResponse(
            status_code=400, 
            content={
                "error": "Missing embeddings for experimental groups",
                "missing": missing_embeddings,
                "message": "請先為這些層次生成embedding：\n" + "\n".join(missing_embeddings)
            }
        )
    
    # 生成查詢向量
    if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
        query_vector = (await embed_gemini([query]))[0]
    elif USE_BGE_M3_EMBEDDING:
        query_vector = embed_bge_m3([query])[0]
    else:
        return JSONResponse(status_code=400, content={"error": "No embedding method available"})
    
    results = {}
    
    for group_key in groups_to_test:
        if group_key not in GRANULARITY_COMBINATIONS:
            continue
        
        if group_key == "group_e":
            try:
                results[group_key] = await _run_experimental_group_e_pipeline(query, k, doc_id=doc_id)
            except Exception as e:
                print(f"⚠️ 實驗組E檢索失敗: {e}")
                results[group_key] = {
                    "group_info": GRANULARITY_COMBINATIONS[group_key],
                    "error": str(e),
                    "fused_results": [],
                    "total_results": 0
                }
            continue
            
        combination = GRANULARITY_COMBINATIONS[group_key]
        selected_levels = combination["levels"]
        
        # 從選定的層次中檢索並融合結果
        all_results = []
        level_contributions = {}
        
        for level_name in selected_levels:
            level_data = store.get_multi_level_embeddings(level_name)
            if not level_data:
                continue
            
            vectors = np.array(level_data['embeddings'])
            chunks = level_data['chunks']
            doc_ids = level_data['doc_ids']
            
            # 計算相似度
            similarities = cosine_similarity([query_vector], vectors)[0]
            
            # 獲取該層次的top-k結果
            top_indices = np.argsort(similarities)[::-1][:k]
            
            level_results = []
            for idx in top_indices:
                doc_id = doc_ids[idx]
                doc = store.get_doc(doc_id)
                
                # 嘗試獲取chunk_id
                chunk_id = None
                metadata = {}
                
                # 方法1：從multi_level_chunks中獲取
                if doc and doc.multi_level_chunks and level_name in doc.multi_level_chunks:
                    level_chunks = doc.multi_level_chunks[level_name]
                    # 通過content匹配找到對應的chunk
                    chunk_content = chunks[idx]
                    for chunk_data in level_chunks:
                        if isinstance(chunk_data, dict):
                            chunk_content_data = chunk_data.get('content', '')
                            # 精確匹配或前200字符匹配
                            if chunk_content_data == chunk_content or (
                                len(chunk_content_data) > 100 and 
                                len(chunk_content) > 100 and
                                chunk_content_data[:200] == chunk_content[:200]
                            ):
                                chunk_id = chunk_data.get('chunk_id') or chunk_data.get('metadata', {}).get('chunk_id')
                                metadata = chunk_data.get('metadata', {})
                                break
                
                # 方法2：從faiss_store或bm25_index獲取
                if not chunk_id:
                    try:
                        chunk_info = faiss_store.get_multi_level_chunk_by_index(level_name, idx)
                        if chunk_info:
                            chunk_id = chunk_info.get('chunk_id')
                            if 'enhanced_metadata' in chunk_info:
                                metadata = chunk_info.get('enhanced_metadata', {})
                    except:
                        pass
                
                # 方法3：從bm25_index獲取
                if not chunk_id:
                    try:
                        chunk_info = bm25_index.get_multi_level_chunk_by_index(level_name, idx)
                        if chunk_info:
                            chunk_id = chunk_info.get('chunk_id')
                    except:
                        pass
                
                # 方法4：生成默認chunk_id
                if not chunk_id:
                    chunk_id = f"{doc_id}_{level_name}_{idx}"
                
                result = {
                    "content": chunks[idx],
                    "similarity": float(similarities[idx]),
                    "level": level_name,
                    "doc_id": doc_id,
                    "chunk_index": int(idx),
                    "chunk_id": chunk_id,
                    "metadata": metadata
                }
                level_results.append(result)
                all_results.append(result)
            
            level_contributions[level_name] = {
                "results": level_results,
                "total_chunks": len(chunks),
                "avg_similarity": float(np.mean([r["similarity"] for r in level_results])) if level_results else 0
            }
        
        # 融合結果（按相似度排序）
        fused_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)
        
        results[group_key] = {
            "group_info": combination,
            "level_contributions": level_contributions,
            "fused_results": fused_results[:k],
            "total_results": len(all_results)
        }
    
    return {
        "query": query,
        "k": k,
        "groups_tested": list(results.keys()),
        "results": results
    }


@app.get("/api/granularity-comparison-report")
def generate_comparison_report():
    """生成粒度對比報告"""
    all_annotations = store.get_all_annotations()
    
    if not all_annotations:
        return {"message": "No annotations available for comparison"}
    
    # 按查詢和實驗組分組
    query_group_data = {}
    for annotation in all_annotations:
        # 從annotation中提取實驗組信息（需要在前端標註時記錄）
        query = annotation.query
        group_info = getattr(annotation, 'experimental_group', 'unknown')
        
        if query not in query_group_data:
            query_group_data[query] = {}
        if group_info not in query_group_data[query]:
            query_group_data[query][group_info] = []
        
        query_group_data[query][group_info].append(annotation)
    
    # 計算各查詢各組的指標
    report = {
        "total_queries": len(query_group_data),
        "total_annotations": len(all_annotations),
        "experimental_groups": ["group_a", "group_b", "group_c", "group_d", "group_e"],
        "per_query_results": {},
        "group_comparison": {},
        "marginal_benefit_analysis": {}
    }
    
    k_values = [1, 3, 5, 10]
    
    # 計算各查詢的結果
    for query, group_annotations in query_group_data.items():
        report["per_query_results"][query] = {}
        
        for group, annotations in group_annotations.items():
            metrics = calculate_ecu_metrics(annotations, k_values)
            report["per_query_results"][query][group] = {
                "total_annotations": len(annotations),
                "metrics": metrics,
                "label_distribution": {
                    "E": sum(1 for a in annotations if a.relevance_label == 'E'),
                    "C": sum(1 for a in annotations if a.relevance_label == 'C'),
                    "U": sum(1 for a in annotations if a.relevance_label == 'U')
                }
            }
    
    # 計算各實驗組的聚合指標
    for group in report["experimental_groups"]:
        group_metrics = []
        for query_data in report["per_query_results"].values():
            if group in query_data:
                group_metrics.append(query_data[group]["metrics"])
        
        if group_metrics:
            report["group_comparison"][group] = {}
            for k in k_values:
                report["group_comparison"][group][f"avg_E@{k}"] = np.mean([m[f"E@{k}"] for m in group_metrics])
                report["group_comparison"][group][f"avg_C@{k}"] = np.mean([m[f"C@{k}"] for m in group_metrics])
                report["group_comparison"][group][f"avg_U@{k}"] = np.mean([m[f"U@{k}"] for m in group_metrics])
                report["group_comparison"][group][f"avg_E+C@{k}"] = np.mean([m[f"E+C@{k}"] for m in group_metrics])
    
    # 計算邊際效益分析
    if "group_a" in report["group_comparison"]:
        baseline = report["group_comparison"]["group_a"]
        for group in ["group_b", "group_c", "group_d", "group_e"]:
            if group in report["group_comparison"]:
                comparison = report["group_comparison"][group]
                report["marginal_benefit_analysis"][f"{group}_vs_group_a"] = {}
                
                for k in k_values:
                    report["marginal_benefit_analysis"][f"{group}_vs_group_a"][f"E@{k}_improvement"] = (
                        comparison[f"avg_E@{k}"] - baseline[f"avg_E@{k}"]
                    )
                    report["marginal_benefit_analysis"][f"{group}_vs_group_a"][f"E+C@{k}_improvement"] = (
                        comparison[f"avg_E+C@{k}"] - baseline[f"avg_E+C@{k}"]
                    )
                    report["marginal_benefit_analysis"][f"{group}_vs_group_a"][f"U@{k}_reduction"] = (
                        baseline[f"avg_U@{k}"] - comparison[f"avg_U@{k}"]
                    )
    
    return report


# ============================================
