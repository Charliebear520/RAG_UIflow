from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass
import re
from typing import List, Optional, Dict, Any, Tuple
import json
from datetime import datetime
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import ChunkConfig, MetadataOptions
from .hybrid_search import hybrid_rank, HybridConfig
from .store import InMemoryStore
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

load_dotenv()


def get_env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GEMINI_EMBEDDING = get_env_bool("USE_GEMINI_EMBEDDING", True)  # 默認使用 Gemini
USE_GEMINI_COMPLETION = get_env_bool("USE_GEMINI_COMPLETION", False)
USE_BGE_M3_EMBEDDING = get_env_bool("USE_BGE_M3_EMBEDDING", False)  # BGE-M3 備用選項

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


# ---- Simple in-memory store (demo only) ----
@dataclass
class DocRecord:
    id: str
    filename: str
    text: str
    chunks: List[str]
    chunk_size: int
    overlap: int
    json_data: Optional[Dict[str, Any]] = None  # 存儲結構化JSON數據
    structured_chunks: Optional[List[Dict[str, Any]]] = None  # 存儲結構化chunks
    generated_questions: Optional[List[str]] = None  # 存儲生成的問題








store = InMemoryStore()


app = FastAPI(title="RAG Visualizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # During dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        model = ModelCls('gemini-2.0-flash-exp')
        
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




def extract_spans_with_pdfplumber(pdf_file, text_content: str, full_text: str = "") -> List[Dict[str, Any]]:
    """使用pdfplumber提取文字片段範圍"""
    spans = []
    
    try:
        # 重置文件指針
        pdf_file.seek(0)
        
        with pdfplumber.open(pdf_file) as pdf:
            # 首先在整個文檔中查找內容
            all_text = ""
            page_texts = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                page_texts.append(page_text)
                all_text += page_text + "\n"
            
            # 在完整文本中查找內容
            if text_content.strip():
                # 清理文本內容，去除多餘空白
                clean_content = re.sub(r'\s+', ' ', text_content.strip())
                clean_all_text = re.sub(r'\s+', ' ', all_text)
                
                start_idx = clean_all_text.find(clean_content)
                if start_idx != -1:
                    end_idx = start_idx + len(clean_content)
                    
                    # 計算在哪個頁面
                    page_num = 1
                    current_pos = 0
                    for i, page_text in enumerate(page_texts):
                        clean_page_text = re.sub(r'\s+', ' ', page_text)
                        page_len = len(clean_page_text)
                        
                        if current_pos <= start_idx < current_pos + page_len:
                            page_num = i + 1
                            # 計算在該頁面內的相對位置
                            page_start = start_idx - current_pos
                            page_end = page_start + len(clean_content)
                            
                            spans.append({
                                "start_char": start_idx,
                                "end_char": end_idx,
                                "page_start_char": page_start,
                                "page_end_char": page_end,
                                "text": clean_content[:100] + "..." if len(clean_content) > 100 else clean_content,
                                "page": page_num,
                                "confidence": 1.0
                            })
                            break
                        current_pos += page_len + 1  # +1 for newline
                
                # 如果沒找到完整匹配，嘗試部分匹配
                if not spans and len(clean_content) > 10:
                    # 嘗試匹配前20個字符
                    partial_content = clean_content[:20]
                    start_idx = clean_all_text.find(partial_content)
                    if start_idx != -1:
                        end_idx = start_idx + len(clean_content)
                        
                        # 計算頁面位置
                        page_num = 1
                        current_pos = 0
                        for i, page_text in enumerate(page_texts):
                            clean_page_text = re.sub(r'\s+', ' ', page_text)
                            page_len = len(clean_page_text)
                            
                            if current_pos <= start_idx < current_pos + page_len:
                                page_num = i + 1
                                page_start = start_idx - current_pos
                                page_end = page_start + len(clean_content)
                                
                                spans.append({
                                    "start_char": start_idx,
                                    "end_char": end_idx,
                                    "page_start_char": page_start,
                                    "page_end_char": page_end,
                                    "text": clean_content[:100] + "..." if len(clean_content) > 100 else clean_content,
                                    "page": page_num,
                                    "confidence": 0.8,
                                    "note": "partial_match"
                                })
                                break
                            current_pos += page_len + 1
            
            # 如果還是沒找到，使用關鍵詞匹配
            if not spans and text_content.strip():
                keywords = re.findall(r'[\u4e00-\u9fff]+', text_content)
                if keywords:
                    # 找到包含最多關鍵詞的頁面
                    best_page = 1
                    best_score = 0
                    
                    for page_num, page_text in enumerate(page_texts, 1):
                        clean_page_text = re.sub(r'\s+', ' ', page_text)
                        score = sum(1 for keyword in keywords if keyword in clean_page_text)
                        if score > best_score:
                            best_score = score
                            best_page = page_num
                    
                    if best_score > 0:
                        spans.append({
                            "start_char": 0,
                            "end_char": len(text_content),
                            "page_start_char": 0,
                            "page_end_char": len(text_content),
                            "text": text_content[:100] + "..." if len(text_content) > 100 else text_content,
                            "page": best_page,
                            "confidence": 0.5,
                            "note": "keyword_match",
                            "matched_keywords": [kw for kw in keywords if kw in page_texts[best_page-1]]
                        })
                        
    except Exception as e:
        print(f"Error extracting spans: {e}")
    
    return spans


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


def get_page_range_for_text(pdf_file, target_text: str) -> Dict[str, int]:
    """獲取文本在PDF中的頁碼範圍"""
    try:
        # 重置文件指针
        pdf_file.seek(0)
        
        with pdfplumber.open(pdf_file) as pdf:
            start_page = None
            end_page = None
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                clean_page_text = re.sub(r'\s+', ' ', page_text)
                clean_target = re.sub(r'\s+', ' ', target_text.strip())
                
                if clean_target in clean_page_text:
                    if start_page is None:
                        start_page = page_num
                    end_page = page_num
                elif len(clean_target) > 10:
                    # 嘗試部分匹配
                    partial = clean_target[:20]
                    if partial in clean_page_text:
                        if start_page is None:
                            start_page = page_num
                        end_page = page_num
            
            if start_page is not None:
                return {"start": start_page, "end": end_page or start_page}
            else:
                return {"start": 1, "end": 1}  # 默認值
                
    except Exception as e:
        print(f"Error getting page range: {e}")
        return {"start": 1, "end": 1}  # 默认值




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
    
    store.docs[doc_id] = DocRecord(
        id=doc_id,
        filename=file.filename,
        text=text,
        json_data=None,  # 初始為None，後續通過/update-json端點更新
        chunks=[],
        chunk_size=0,
        overlap=0,
    )
    # When uploading new docs, prior embeddings are invalid
    store.reset_embeddings()
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
        
        # 處理條文項目
        for item in items:
            item_title = item.get("item", "")
            item_content = item.get("content", "")
            sub_items = item.get("sub_items", [])
            
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
            
            # 處理子項目
            for sub_item in sub_items:
                sub_item_title = sub_item.get("sub_item", "")
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


async def embed_gemini(texts: List[str]) -> List[List[float]]:
    """Call Google Generative API (Gemini) embeddings endpoint using API key.

    Note: This uses the REST endpoint pattern with the API key in query params.
    """
    if not httpx:
        raise RuntimeError("httpx not available")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    model = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
    # 使用正確的 API 端點格式
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
    headers = {
        "x-goog-api-key": GOOGLE_API_KEY,
        "Content-Type": "application/json"
    }
    out: List[List[float]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        # 逐個處理文本（Gemini API 需要單個請求）
        for text in texts:
            payload = {
                "model": f"models/{model}",
                "content": {"parts": [{"text": text}]}
            }
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            # 根據官方文檔，響應格式是 {"embedding": {"values": [...]}}
            embedding_values = data.get("embedding", {}).get("values", [])
            out.append(embedding_values)
    return out


def embed_bge_m3(texts: List[str]) -> List[List[float]]:
    """使用 BGE-M3 模型進行 embedding"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("sentence-transformers not available")
    
    try:
        # 載入 BGE-M3 模型
        model = SentenceTransformer('BAAI/bge-m3')
        
        # 批量處理文本
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        
        # 轉換為列表格式
        return embeddings.tolist()
        
    except Exception as e:
        raise RuntimeError(f"BGE-M3 embedding failed: {e}")


@app.post("/api/embed")
async def embed(req: EmbedRequest):
    # gather chunks across selected docs
    selected = req.doc_ids or list(store.docs.keys())
    all_chunks: List[str] = []
    chunk_doc_ids: List[str] = []
    for d in selected:
        doc = store.docs.get(d)
        if doc and doc.chunks:
            all_chunks.extend(doc.chunks)
            chunk_doc_ids.extend([doc.id] * len(doc.chunks))

    if not all_chunks:
        return JSONResponse(status_code=400, content={"error": "no chunks to embed"})

    # 嘗試使用 Gemini embedding（主要選項）
    if USE_GEMINI_EMBEDDING and GOOGLE_API_KEY:
        try:
            vectors = await embed_gemini(all_chunks)
            store.embeddings = vectors
            store.chunk_doc_ids = chunk_doc_ids
            store.chunks_flat = all_chunks
            return {
                "provider": "gemini", 
                "model": "text-embedding-004",
                "num_vectors": len(vectors),
                "dimension": len(vectors[0]) if vectors else 0
            }
        except Exception as e:
            print(f"Gemini embedding failed: {e}")
            # 如果 Gemini 失敗，嘗試 BGE-M3
    
    # 嘗試使用 BGE-M3 embedding（備用選項）
    if USE_BGE_M3_EMBEDDING and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            vectors = embed_bge_m3(all_chunks)
            store.embeddings = vectors
            store.chunk_doc_ids = chunk_doc_ids
            store.chunks_flat = all_chunks
            return {
                "provider": "bge-m3", 
                "model": "BAAI/bge-m3",
                "num_vectors": len(vectors),
                "dimension": len(vectors[0]) if vectors else 0
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


def rank_with_dense_vectors(query: str, k: int):
    """使用密集向量進行相似度計算（支持 Gemini 和 BGE-M3）"""
    import numpy as np
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


def calculate_retrieval_metrics(query: str, results: List[Dict], k: int) -> Dict[str, float]:
    """計算檢索指標 P@K 和 R@K"""
    try:
        # 嘗試從 QA 數據中獲取相關文檔
        qa_data = load_qa_data()
        if not qa_data:
            return {"p_at_k": 0.0, "r_at_k": 0.0, "note": "No QA data available"}
        
        # 找到與查詢最匹配的 QA 項目
        best_match = None
        best_similarity = 0.0
        
        for qa_item in qa_data:
            # 改進的文本相似度匹配
            qa_query = qa_item.get("query", "").lower()
            query_lower = query.lower()
            
            # 方法1: 直接包含匹配
            if query_lower in qa_query or qa_query in query_lower:
                similarity = 1.0
            else:
                # 方法2: 提取法條號碼進行匹配
                import re
                query_article = re.search(r'第(\d+(?:之\d+)?)條', query_lower)
                qa_article = re.search(r'第(\d+(?:之\d+)?)條', qa_query)
                
                if query_article and qa_article:
                    if query_article.group(1) == qa_article.group(1):
                        similarity = 0.8
                    else:
                        similarity = 0.0
                else:
                    # 方法3: 詞彙重疊度
                    query_words = set(query_lower.split())
                    qa_words = set(qa_query.split())
                    overlap = len(query_words.intersection(qa_words))
                    similarity = overlap / max(len(query_words), len(qa_words), 1)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = qa_item
        
        if not best_match or best_similarity < 0.3:  # 相似度閾值
            return {"p_at_k": 0.0, "r_at_k": 0.0, "note": "No matching QA found"}
        
        # 從 gold 字段中提取相關的法條信息
        gold = best_match.get("gold", {})
        if gold:
            # 從 gold 字段構建法條信息
            law = gold.get("law", "")
            article_number = gold.get("article_number")
            article_suffix = gold.get("article_suffix")
            
            if law and article_number:
                article_text = f"第{article_number}條"
                if article_suffix:
                    article_text += f"之{article_suffix}"
                relevant_articles = [article_text]
            else:
                # 如果沒有 gold 信息，嘗試從查詢中提取
                relevant_articles = extract_articles_from_text(best_match.get("query", ""))
        else:
            # 如果沒有 gold 字段，嘗試從查詢中提取
            relevant_articles = extract_articles_from_text(best_match.get("query", ""))
        
        if not relevant_articles:
            return {"p_at_k": 0.0, "r_at_k": 0.0, "note": "No relevant articles found"}
        
        # 計算 P@K 和 R@K
        relevant_count = 0
        for result in results[:k]:
            content = result.get("content", "").lower()
            # 檢查是否包含相關法條
            for article in relevant_articles:
                # 標準化法條格式進行匹配
                article_normalized = article.lower().replace(" ", "")
                content_normalized = content.replace(" ", "")
                if article_normalized in content_normalized:
                    relevant_count += 1
                    break
        
        p_at_k = relevant_count / k if k > 0 else 0.0
        r_at_k = relevant_count / len(relevant_articles) if relevant_articles else 0.0
        
        return {
            "p_at_k": p_at_k,
            "r_at_k": r_at_k,
            "relevant_articles": relevant_articles,
            "qa_similarity": best_similarity,
            "matched_qa": best_match.get("query", "")[:100] + "..."
        }
        
    except Exception as e:
        return {"p_at_k": 0.0, "r_at_k": 0.0, "error": str(e)}


def load_qa_data() -> List[Dict]:
    """載入 QA 數據"""
    try:
        import json
        import os
        
        # 嘗試載入不同的 QA 文件
        qa_files = [
            "QA/copyright_p.json",
            "QA/copyright_n.json", 
            "QA/copyright.json",
            "QA/qa_gold.json"
        ]
        
        for qa_file in qa_files:
            qa_path = os.path.join(os.path.dirname(__file__), "..", "..", qa_file)
            if os.path.exists(qa_path):
                with open(qa_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        return data
        
        return []
    except Exception as e:
        print(f"載入 QA 數據失敗: {e}")
        return []


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
        result = {
            "rank": rank,
            "score": float(score),
            "doc_id": doc_id,
            "chunk_index": i,
            "content": chunks_flat[i][:2000],
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
        embedding_model = "text-embedding-004"
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
        result = {
            "rank": rank,
            "score": item["score"],
            "vector_score": item["vector_score"],
            "bonus": item["bonus"],
            "doc_id": item["doc_id"],
            "chunk_index": item["chunk_index"],
            "content": item["content"][:2000],
            "metadata": item["metadata"]
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
        embedding_model = "text-embedding-004"
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


async def gemini_chat(messages: List[Dict[str, str]]) -> str:
    if not httpx:
        raise RuntimeError("httpx not available")
    
    # 優先使用 GOOGLE_API_KEY，如果沒有則使用 GEMINI_API_KEY
    api_key = GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
    
    model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5-flash")
    # Use Generative Language API: models/{model}:generateContent
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
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
            "maxOutputTokens": 2048
        }
    }
    
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        # Extract response from new format
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0].get("text", "").strip()
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
    # 使用 HybridRAG（向量檢索 + metadata 關鍵字加分）取得生成上下文
    if store.embeddings is None:
        return JSONResponse(status_code=400, content={"error": "run /embed first"})

    # 構建 nodes（與 /api/hybrid-retrieve 保持一致）
    chunks_flat = store.chunks_flat
    mapping_doc_ids = store.chunk_doc_ids
    if not chunks_flat:
        return JSONResponse(status_code=400, content={"error": "no chunks available"})

    nodes = []
    for i, (chunk, doc_id) in enumerate(zip(chunks_flat, mapping_doc_ids)):
        doc = store.docs.get(doc_id)
        metadata = {}
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
4. 回答要準確、專業，符合法律文檔的表述方式"""

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


def merge_law_documents(law_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    整合多個法律文檔成一個統一的JSON結構
    
    參數:
    - law_documents: 多個法律文檔的列表
    
    返回:
    - 整合後的法律文檔，格式為 {"laws": [...]}
    """
    if not law_documents:
        return {"laws": []}
    
    # 確保每個法律文檔都有唯一的ID前綴
    merged_laws = []
    global_id_counter = 0
    
    for doc in law_documents:
        if not doc or "law_name" not in doc:
            continue
            
        law_name = doc["law_name"]
        law_prefix = f"{law_name}_{global_id_counter}"
        
        # 創建新的法律文檔結構
        merged_law = {
            "law_name": law_name,
            "chapters": []
        }
        
        # 處理章節
        chapters = doc.get("chapters", [])
        for chapter in chapters:
            chapter_name = chapter.get("chapter", "")
            merged_chapter = {
                "chapter": chapter_name,
                "sections": []
            }
            
            # 處理節
            sections = chapter.get("sections", [])
            for section in sections:
                section_name = section.get("section", "")
                merged_section = {
                    "section": section_name,
                    "articles": []
                }
                
                # 處理條文
                articles = section.get("articles", [])
                for article in articles:
                    article_name = article.get("article", "")
                    merged_article = {
                        "article": article_name,
                        "content": article.get("content", ""),
                        "items": []
                    }
                    
                    # 處理項目
                    items = article.get("items", [])
                    for item in items:
                        item_name = item.get("item", "")
                        merged_item = {
                            "item": item_name,
                            "content": item.get("content", ""),
                            "sub_items": []
                        }
                        
                        # 處理子項目
                        sub_items = item.get("sub_items", [])
                        for sub_item in sub_items:
                            sub_item_name = sub_item.get("sub_item", "")
                            merged_sub_item = {
                                "sub_item": sub_item_name,
                                "content": sub_item.get("content", ""),
                                "metadata": {
                                    "id": f"{law_prefix}_{chapter_name}_{section_name}_{article_name}_{item_name}_{sub_item_name}".replace(" ", "_"),
                                    "spans": sub_item.get("metadata", {}).get("spans", {}),
                                    "page_range": sub_item.get("metadata", {}).get("page_range", {})
                                }
                            }
                            merged_item["sub_items"].append(merged_sub_item)
                        
                        # 為項目添加metadata
                        merged_item["metadata"] = {
                            "id": f"{law_prefix}_{chapter_name}_{section_name}_{article_name}_{item_name}".replace(" ", "_"),
                            "spans": item.get("metadata", {}).get("spans", {}),
                            "page_range": item.get("metadata", {}).get("page_range", {})
                        }
                        merged_article["items"].append(merged_item)
                    
                    # 為條文添加metadata
                    merged_article["metadata"] = {
                        "id": f"{law_prefix}_{chapter_name}_{section_name}_{article_name}".replace(" ", "_"),
                        "spans": article.get("metadata", {}).get("spans", {}),
                        "page_range": article.get("metadata", {}).get("page_range", {})
                    }
                    merged_section["articles"].append(merged_article)
                
                merged_chapter["sections"].append(merged_section)
            
            merged_law["chapters"].append(merged_chapter)
        
        merged_laws.append(merged_law)
        global_id_counter += 1
    
    return {"laws": merged_laws}


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

        def normalize_digits(s: str) -> str:
            # Convert fullwidth digits to ASCII for simpler matching
            fw = "０１２３４５６７８９"
            hw = "0123456789"
            return s.translate(str.maketrans(fw, hw))

        # Determine law name: first non-empty line containing a legal keyword, else filename
        lines = [normalize_digits((ln or "").strip()) for ln in full_text.splitlines()]
        law_name = None
        for ln in lines:
            if not ln:
                continue
            if any(key in ln for key in ["法", "條例", "法規", "法律"]):
                law_name = ln
                break
        if not law_name:
            base = os.path.splitext(filename or "document")[0]
            law_name = base or "未命名法規"

        chapter_re = re.compile(r"^第\s*([一二三四五六七八九十百千0-9]+)\s*章[\u3000\s]*(.*)$")
        section_re = re.compile(r"^第\s*([一二三四五六七八九十百千0-9]+)\s*節[\u3000\s]*(.*)$")
        article_re = re.compile(r"^第\s*([一二三四五六七八九十百千0-9]+(?:之[一二三四五六七八九十0-9]+)?)\s*條[\u3000\s]*(.*)$")

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
        current_article: Optional[Dict[str, Any]] = None
        current_item: Optional[Dict[str, Any]] = None
        current_sub_item: Optional[Dict[str, Any]] = None

        def ensure_chapter():
            nonlocal current_chapter
            if current_chapter is None:
                current_chapter = {"chapter": "未分類章", "sections": []}
                structure["chapters"].append(current_chapter)

        def ensure_section():
            nonlocal current_section
            ensure_chapter()
            if current_section is None:
                current_section = {"section": "未分類節", "articles": []}
                current_chapter["sections"].append(current_section)

        for raw in lines:
            ln = raw.strip()
            if not ln:
                continue

            # Headings
            m = chapter_re.match(ln)
            if m:
                title = f"第{m.group(1)}章" + (f" {m.group(2).strip()}" if m.group(2) else "")
                current_chapter = {"chapter": title, "sections": []}
                structure["chapters"].append(current_chapter)
                current_section = None
                current_article = None
                current_item = None
                current_sub_item = None
                continue

            m = section_re.match(ln)
            if m:
                ensure_chapter()
                title = f"第{m.group(1)}節" + (f" {m.group(2).strip()}" if m.group(2) else "")
                current_section = {"section": title, "articles": []}
                current_chapter["sections"].append(current_section)
                current_article = None
                current_item = None
                current_sub_item = None
                current_sub_item = None
                continue

            m = article_re.match(ln)
            if m:
                ensure_section()
                title = f"第{m.group(1)}條"
                rest = m.group(2).strip() if m.group(2) else ""
                current_article = {"article": title, "content": rest, "items": []}
                current_section["articles"].append(current_article)
                current_item = None
                current_sub_item = None
                current_sub_item = None
                continue

            # Items within article
            if current_article is not None:
                num, content, item_type = parse_item_line(ln)
                if num is not None:
                    num = normalize_digits(num)
                    
                    # Determine if this is a sub-item (Chinese numerals after Arabic numerals)
                    if (item_type == "chinese_space" or item_type == "chinese_with_punct") and current_item is not None:
                        # This is a sub-item of the current item
                        if "sub_items" not in current_item:
                            current_item["sub_items"] = []
                        sub_item = {"item": str(num), "content": content or "", "sub_items": []}
                        current_item["sub_items"].append(sub_item)
                        current_sub_item = sub_item
                    else:
                        # This is a main item
                        current_item = {"item": str(num), "content": content or "", "sub_items": []}
                        current_article["items"].append(current_item)
                        current_sub_item = None
                else:
                    # continuation line
                    if current_sub_item is not None:
                        # Append to current sub-item
                        sep = "\n" if current_sub_item["content"] else ""
                        current_sub_item["content"] = f"{current_sub_item['content']}{sep}{ln}"
                    elif current_item is not None:
                        # Check if we have sub-items and append to the last sub-item
                        if "sub_items" in current_item and current_item["sub_items"]:
                            last_sub_item = current_item["sub_items"][-1]
                            sep = "\n" if last_sub_item["content"] else ""
                            last_sub_item["content"] = f"{last_sub_item['content']}{sep}{ln}"
                        else:
                            sep = "\n" if current_item["content"] else ""
                            current_item["content"] = f"{current_item['content']}{sep}{ln}"
                    else:
                        # accumulate into article content
                        if "content" not in current_article or current_article["content"] is None:
                            current_article["content"] = ln
                        else:
                            current_article["content"] = (current_article["content"] + "\n" + ln).strip()
                continue

            # If no article yet, but we have text, place it under a default article
            if current_section is not None and current_article is None:
                current_article = {"article": "未標示條文", "content": ln, "items": []}
                current_section["articles"].append(current_article)
                current_item = None
                current_sub_item = None
            elif current_article is None:
                ensure_section()
                current_article = {"article": "未標示條文", "content": ln, "items": []}
                current_section["articles"].append(current_article)
                current_item = None
                current_sub_item = None
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
            if options.include_id or options.include_page_range or options.include_spans:
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
                        if options.include_page_range:
                            article_metadata["page_range"] = {"start": 1, "end": 1}  # 簡化的頁面範圍
                        if options.include_spans:
                            article_metadata["spans"] = {"start": 0, "end": len(article["content"])}
                        if options.include_page_range:
                            # 簡化的頁碼範圍（基於文本位置估算）
                            article_metadata["page_range"] = {"start": 1, "end": 1}  # 簡化版本
                        if options.include_spans:
                            # 簡化的文本定位
                            start_pos = full_text.find(article["content"][:50])  # 使用前50字符定位
                            if start_pos >= 0:
                                article_metadata["spans"] = [{
                                    "start_char": start_pos,
                                    "end_char": start_pos + len(article["content"]),
                                    "text": article["content"][:100] + "..." if len(article["content"]) > 100 else article["content"],
                                    "page": 1,
                                    "confidence": 0.8,
                                    "found": True
                                }]
                            else:
                                article_metadata["spans"] = []
                        
                        article["metadata"] = article_metadata
                        
                        # 為項目添加簡化metadata
                        for item in article["items"]:
                            item_metadata = {}
                            if options.include_id:
                                item_metadata["id"] = f"{structure['law_name']}_{chapter_name}_{section_name}_{article_name}_{item['item']}".replace(" ", "_")
                            if options.include_page_range:
                                item_metadata["page_range"] = {"start": 1, "end": 1}  # 簡化的頁面範圍
                            if options.include_spans:
                                item_metadata["spans"] = {"start": 0, "end": len(item["content"])}
                            
                            item["metadata"] = item_metadata
                            
                            # 為子項目添加簡化metadata
                            for sub_item in item["sub_items"]:
                                sub_item_metadata = {}
                                if options.include_id:
                                    sub_item_metadata["id"] = f"{structure['law_name']}_{chapter_name}_{section_name}_{article_name}_{item['item']}_{sub_item['item']}".replace(" ", "_")
                                if options.include_page_range:
                                    sub_item_metadata["page_range"] = {"start": 1, "end": 1}  # 簡化的頁面範圍
                                if options.include_spans:
                                    sub_item_metadata["spans"] = {"start": 0, "end": len(sub_item["content"])}
                                
                                sub_item["metadata"] = sub_item_metadata
                        
                        processed_count += 1
                        if processed_count % 10 == 0:
                            print(f"已處理 {processed_count} 個條文")
            
            metadata_time = time.time() - metadata_start
            print(f"Metadata處理完成，耗時: {metadata_time:.2f}秒")
        
        # 添加metadata（使用優化版本）
        if any([options.include_id, options.include_page_range, options.include_spans]):
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
    }
