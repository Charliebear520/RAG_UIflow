from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass
import re
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pdfplumber
try:
    import jieba
    import jieba.analyse
    jieba.initialize()
except ImportError:
    jieba = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

load_dotenv()


def get_env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GEMINI_EMBEDDING = get_env_bool("USE_GEMINI_EMBEDDING", False)
USE_GEMINI_COMPLETION = get_env_bool("USE_GEMINI_COMPLETION", False)

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


class InMemoryStore:
    def __init__(self) -> None:
        self.docs: Dict[str, DocRecord] = {}
        self.tfidf: Optional[TfidfVectorizer] = None
        self.embeddings = None  # matrix for chunks (numpy array or list)
        self.chunk_doc_ids: List[str] = []
        self.chunks_flat: List[str] = []

    def reset_embeddings(self):
        """Clear vector/index state so embeddings can be recomputed."""
        self.tfidf = None
        self.embeddings = None
        self.chunk_doc_ids = []
        self.chunks_flat = []


store = InMemoryStore()


app = FastAPI(title="RAG Visualizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # During dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChunkRequest(BaseModel):
    doc_id: str
    chunk_size: int = 500
    overlap: int = 50


class EmbedRequest(BaseModel):
    doc_ids: Optional[List[str]] = None  # if None, embed all


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5


class GenerateRequest(BaseModel):
    query: str
    top_k: int = 5


class MetadataOptions(BaseModel):
    include_id: bool = True
    include_page_range: bool = True
    include_keywords: bool = True
    include_cross_references: bool = True
    include_importance: bool = True
    include_length: bool = True
    include_extracted_entities: bool = False 
    include_spans: bool = True


def generate_unique_id(law_name: str, chapter: str, section: str, article: str, item: str = None) -> str:
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
        # 配置Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return extract_keywords_fallback(text, top_k)
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
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
        # 使用jieba提取關鍵詞
        keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
        return keywords
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
        
        # 獲取目標文本的TF-IDF向量
        target_vector = tfidf_matrix[-1]
        
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
    if not texts or not target_text:
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
    # For simplicity, treat all as text; for PDFs, a proper parser should be used
    content = await file.read()
    try:
        text = content.decode("utf-8", errors="ignore")
    except Exception:
        text = str(content)
    doc_id = str(uuid.uuid4())
    store.docs[doc_id] = DocRecord(
        id=doc_id,
        filename=file.filename,
        text=text,
        chunks=[],
        chunk_size=0,
        overlap=0,
    )
    # When uploading new docs, prior embeddings are invalid
    store.reset_embeddings()
    return {"doc_id": doc_id, "filename": file.filename, "num_chars": len(text)}


def sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
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


@app.post("/chunk")
def chunk(req: ChunkRequest):
    doc = store.docs.get(req.doc_id)
    if not doc:
        return JSONResponse(status_code=404, content={"error": "doc not found"})
    chunks = sliding_window_chunks(doc.text, req.chunk_size, req.overlap)
    doc.chunks = chunks
    doc.chunk_size = req.chunk_size
    doc.overlap = req.overlap
    # invalidates embeddings for safety
    store.reset_embeddings()
    return {"doc_id": doc.id, "num_chunks": len(chunks), "chunk_size": req.chunk_size, "overlap": req.overlap, "sample": chunks[:3]}


async def embed_gemini(texts: List[str]) -> List[List[float]]:
    """Call Google Generative API (Gemini) embeddings endpoint using API key.

    Note: This uses the REST endpoint pattern with the API key in query params.
    """
    if not httpx:
        raise RuntimeError("httpx not available")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    model = os.getenv("GOOGLE_EMBEDDING_MODEL", "embed-gecko-001")
    # endpoint pattern: models/{model}:embed
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:embed?key={GOOGLE_API_KEY}"
    out: List[List[float]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            payload = {"input": batch}
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            # expected shape: { data: [ { embedding: [...] }, ... ] }
            out.extend([d.get("embedding") for d in data.get("data", [])])
    return out


@app.post("/embed")
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

    if USE_GEMINI_EMBEDDING:
        vectors = await embed_gemini(all_chunks)
        # keep as numpy matrix-like list of lists; for cosine sim we'll rely on numpy if available
        store.tfidf = None
        store.embeddings = vectors
        store.chunk_doc_ids = chunk_doc_ids
        store.chunks_flat = all_chunks
        return {"provider": "gemini", "num_vectors": len(vectors)}
    else:
        # TF-IDF fallback (per-chunk bag-of-words)
        vectorizer = TfidfVectorizer(max_features=4096, stop_words="english")
        X = vectorizer.fit_transform(all_chunks)
        store.tfidf = vectorizer
        store.embeddings = X
        store.chunk_doc_ids = chunk_doc_ids
        store.chunks_flat = all_chunks
        return {"provider": "tfidf", "num_vectors": X.shape[0], "num_features": X.shape[1]}


def rank_with_tfidf(query: str, k: int):
    assert store.tfidf is not None
    q = store.tfidf.transform([query])
    sims = cosine_similarity(q, store.embeddings).ravel()  # type: ignore[arg-type]
    idxs = sims.argsort()[::-1][:k]
    return idxs, sims[idxs]


def rank_with_gemini(query: str, k: int):
    # cosine similarity on dense vectors list
    import numpy as np
    vecs = np.array(store.embeddings, dtype=float)  # type: ignore[assignment]
    qvec = np.array(asyncio_run(embed_gemini([query]))[0], dtype=float)
    # normalize
    vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    q_norm = qvec / (np.linalg.norm(qvec) + 1e-8)
    sims = vecs_norm @ q_norm
    idxs = np.argsort(-sims)[:k]
    return idxs.tolist(), sims[idxs].tolist()


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


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    if store.embeddings is None:
        return JSONResponse(status_code=400, content={"error": "run /embed first"})
    if store.tfidf is not None:
        idxs, sims = rank_with_tfidf(req.query, req.k)
    else:
        idxs, sims = rank_with_gemini(req.query, req.k)

    # Use the same order as built in /embed
    chunks_flat = store.chunks_flat
    mapping_doc_ids = store.chunk_doc_ids

    results = []
    for rank, (i, score) in enumerate(zip(idxs, sims), start=1):
        if i < 0 or i >= len(chunks_flat):
            continue
        results.append({
            "rank": rank,
            "score": float(score),
            "doc_id": mapping_doc_ids[i],
            "chunk_index": i,
            "content": chunks_flat[i][:2000],
        })
    return {"query": req.query, "k": req.k, "results": results}


async def gemini_chat(messages: List[Dict[str, str]]) -> str:
    if not httpx:
        raise RuntimeError("httpx not available")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5")
    # Use Generative Language API: models/{model}:generateText
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateText?key={GOOGLE_API_KEY}"
    # Flatten messages into a single prompt
    prompt = "".join([f"{m.get('role','user')}: {m.get('content','')}\n" for m in messages])
    payload = {"prompt": {"text": prompt}, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        # Expect generatedText in response
        if "candidates" in data:
            return data["candidates"][0].get("output", "").strip()
        return data.get("output", "").strip()


def simple_extractive_answer(query: str, contexts: List[str]) -> str:
    # very simple heuristic: return the highest-overlap sentences as an extractive summary
    import re
    from collections import Counter
    q_terms = [t.lower() for t in re.findall(r"\w+", query)]
    counts = Counter()
    sents: List[str] = []
    for ctx in contexts:
        sents.extend(re.split(r"(?<=[.!?])\s+", ctx))
    for s in sents:
        tokens = [t.lower() for t in re.findall(r"\w+", s)]
        overlap = len(set(tokens) & set(q_terms))
        if overlap:
            counts[s] = overlap
    best = [s for s, _ in counts.most_common(5)]
    return " \n".join(best) if best else "No relevant answer found in context."


@app.post("/generate")
def generate(req: GenerateRequest):
    # retrieve first
    r = retrieve(RetrieveRequest(query=req.query, k=req.top_k))
    if isinstance(r, JSONResponse):
        return r
    results = r["results"]
    contexts = [item["content"] for item in results]

    reasoning_steps = [
        {"type": "plan", "text": "Read query, identify entities and constraints."},
        {"type": "gather", "text": f"Collect top-{req.top_k} chunks as context."},
        {"type": "synthesize", "text": "Synthesize answer grounded in retrieved text."},
    ]

    if USE_GEMINI_COMPLETION:
        prompt = [
            {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the provided context. If missing, say you don't know."},
            {"role": "user", "content": f"Query: {req.query}\n\nContext:\n" + "\n---\n".join(contexts)},
        ]
        try:
            answer = asyncio_run(gemini_chat(prompt))
        except Exception as e:
            answer = f"Gemini call failed: {e}. Falling back to extractive answer.\n" + simple_extractive_answer(req.query, contexts)
    else:
        answer = simple_extractive_answer(req.query, contexts)

    return {
        "query": req.query,
        "answer": answer,
        "contexts": results,
        "steps": reasoning_steps,
    }


@app.post("/convert")
async def convert(file: UploadFile = File(...), metadata_options: str = Form("{}")):
    try:
        # Parse metadata options
        try:
            metadata_config = json.loads(metadata_options)
            options = MetadataOptions(**metadata_config)
        except:
            options = MetadataOptions()  # 使用默認選項
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支持PDF文件格式")
        
        # Reset file pointer to beginning
        await file.seek(0)
        
        # Read PDF content safely; skip pages with no text
        try:
            reader = PdfReader(file.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"無法讀取PDF文件: {str(e)}")
        
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)
        
        if not any(texts):  # No text extracted
            raise HTTPException(status_code=400, detail="PDF文件中没有找到可提取的文本内容")
            
        full_text = "\n".join(texts)

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
            base = os.path.splitext(file.filename or "document")[0]
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

        # 添加metadata到結構中
        def add_metadata_to_structure(structure, options, full_text):
            """為結構添加metadata"""
            
            # 首先收集所有條文，用於動態重要性計算
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
            
            for chapter in structure["chapters"]:
                chapter_name = chapter["chapter"]
                for section in chapter["sections"]:
                    section_name = section["section"]
                    for article in section["articles"]:
                        article_name = article["article"]
                        
                        # 為條文添加metadata
                        article_metadata = {}
                        if options.include_id:
                            article_metadata["id"] = generate_unique_id(
                                structure["law_name"], chapter_name, section_name, article_name
                            )
                        if options.include_page_range:
                            # 重置文件指针以獲取頁碼範圍
                            file.seek(0)
                            article_metadata["page_range"] = get_page_range_for_text(file.file, article["content"])
                        if options.include_keywords:
                            article_metadata["keywords"] = extract_keywords(article["content"])
                        if options.include_cross_references:
                            article_metadata["cross_references"] = extract_cross_references(article["content"])
                        if options.include_importance:
                            # 使用動態重要性計算，傳入所有條文
                            article_metadata["importance"] = calculate_importance(
                                chapter_name, section_name, article_name, 
                                article["content"], all_articles
                            )
                        if options.include_length:
                            article_metadata["length"] = len(article["content"])
                        if options.include_spans:
                            # 使用簡單的文本定位
                            position = get_text_position_in_document(full_text, article["content"])
                            article_metadata["spans"] = [{
                                "start_char": position["start"],
                                "end_char": position["end"],
                                "text": article["content"][:100] + "..." if len(article["content"]) > 100 else article["content"],
                                "page": 1,  # 簡化版本，假設在第1頁
                                "confidence": position.get("confidence", 0.5),
                                "found": position.get("found", False)
                            }] if position.get("found", False) else []
                        
                        article["metadata"] = article_metadata
                        
                        # 為項目添加metadata
                        for item in article["items"]:
                            item_metadata = {}
                            if options.include_id:
                                item_metadata["id"] = generate_unique_id(
                                    structure["law_name"], chapter_name, section_name, article_name, item["item"]
                                )
                            if options.include_keywords:
                                item_metadata["keywords"] = extract_keywords(item["content"])
                            if options.include_cross_references:
                                item_metadata["cross_references"] = extract_cross_references(item["content"])
                            if options.include_importance:
                                # 項目使用條文的重要性，但權重較低
                                base_importance = calculate_importance(
                                    chapter_name, section_name, article_name, 
                                    article["content"], all_articles
                                )
                                item_metadata["importance"] = round(base_importance * 0.8, 2)
                            if options.include_length:
                                item_metadata["length"] = len(item["content"])
                            if options.include_spans:
                                position = get_text_position_in_document(full_text, item["content"])
                                item_metadata["spans"] = [{
                                    "start_char": position["start"],
                                    "end_char": position["end"],
                                    "text": item["content"][:100] + "..." if len(item["content"]) > 100 else item["content"],
                                    "page": 1,
                                    "confidence": position.get("confidence", 0.5),
                                    "found": position.get("found", False)
                                }] if position.get("found", False) else []
                            
                            item["metadata"] = item_metadata
                            
                            # 為子項目添加metadata
                            for sub_item in item["sub_items"]:
                                sub_item_metadata = {}
                                if options.include_id:
                                    sub_item_metadata["id"] = generate_unique_id(
                                        structure["law_name"], chapter_name, section_name, article_name, 
                                        f"{item['item']}-{sub_item['item']}"
                                    )
                                if options.include_keywords:
                                    sub_item_metadata["keywords"] = extract_keywords(sub_item["content"])
                                if options.include_cross_references:
                                    sub_item_metadata["cross_references"] = extract_cross_references(sub_item["content"])
                                if options.include_importance:
                                    # 子項目使用條文的重要性，但權重更低
                                    base_importance = calculate_importance(
                                        chapter_name, section_name, article_name, 
                                        article["content"], all_articles
                                    )
                                    sub_item_metadata["importance"] = round(base_importance * 0.6, 2)
                                if options.include_length:
                                    sub_item_metadata["length"] = len(sub_item["content"])
                                if options.include_spans:
                                    position = get_text_position_in_document(full_text, sub_item["content"])
                                    sub_item_metadata["spans"] = [{
                                        "start_char": position["start"],
                                        "end_char": position["end"],
                                        "text": sub_item["content"][:100] + "..." if len(sub_item["content"]) > 100 else sub_item["content"],
                                        "page": 1,
                                        "confidence": position.get("confidence", 0.5),
                                        "found": position.get("found", False)
                                    }] if position.get("found", False) else []
                                
                                sub_item["metadata"] = sub_item_metadata
            
            return structure
        
        # 添加metadata
        structure = add_metadata_to_structure(structure, options, full_text)

        return structure
    except HTTPException:
        # Re-raise HTTPExceptions with their original status codes
        raise
    except Exception as e:
        # For other unexpected errors, return 500
        raise HTTPException(status_code=500, detail=f"處理PDF時發生意外錯誤: {str(e)}")


@app.get("/docs/schema")
def schema():
    # Minimal shape for frontend wiring/testing
    return {
        "upload": {"POST": {"multipart": True}},
        "chunk": {"POST": {"json": {"doc_id": "str", "chunk_size": "int", "overlap": "int"}}},
        "embed": {"POST": {"json": {"doc_ids": "List[str]|None"}}},
        "retrieve": {"POST": {"json": {"query": "str", "k": "int"}}},
        "generate": {"POST": {"json": {"query": "str", "top_k": "int"}}},
    }
