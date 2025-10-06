"""
API路由模組
"""

import uuid
import json
from datetime import datetime
from typing import List, Dict, Any
import re
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import (
    DocRecord, EvaluationTask, ChunkConfig, EvaluationRequest, 
    GenerateQuestionsRequest
)
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from .store import InMemoryStore
from .pdf_processor import convert_pdf_to_text, convert_pdf_fallback
from .chunking import chunk_text
from .evaluation import evaluate_chunk_config
from .qa_converter import convert_qa_set_with_law_data
# from .question_generator import generate_questions  # 使用main.py中的函數

# 創建路由器
router = APIRouter()

# 使用main.py中的store實例
from .main import store

# 簡單的任務狀態存儲
task_status_store = {}

# 批量分塊任務存儲
chunking_task_store = {}


class EvaluationMetrics(BaseModel):
    precision_omega: float  # PrecisionΩ - 最大準確率
    precision_at_k: Dict[int, float]  # k -> precision score
    recall_at_k: Dict[int, float]  # k -> recall score
    chunk_count: int
    avg_chunk_length: float
    length_variance: float
# === Helper: 構建文字來源自 JSON（確保以當前 JSON 為準） ===
def build_text_from_json(json_data: Dict[str, Any]) -> str:
    """將法條 JSON 結構序列化為線性文字，供非結構化策略（如 fixed_size）使用。

    注意：此處僅串接可見內容欄位，確保使用者在上傳頁面刪除的內容不再出現在分塊中。
    """
    if not json_data or not isinstance(json_data, dict):
        return ""

    parts: list[str] = []
    for law in (json_data.get("laws") or []):
        law_name = law.get("law_name") or ""
        if law_name:
            parts.append(str(law_name))
        for chapter in (law.get("chapters") or []):
            chapter_title = chapter.get("chapter") or ""
            if chapter_title:
                parts.append(str(chapter_title))
            for section in (chapter.get("sections") or []):
                section_title = section.get("section") or ""
                if section_title:
                    parts.append(str(section_title))
                for article in (section.get("articles") or []):
                    # 條文標題與內容
                    article_title = article.get("article") or ""
                    if article_title:
                        parts.append(str(article_title))
                    article_content = article.get("content") or ""
                    if article_content:
                        parts.append(str(article_content))

                    # 新鍵 paragraphs 或舊鍵 items
                    paragraphs = article.get("paragraphs") or []
                    items = article.get("items") or []
                    items_to_process = paragraphs if paragraphs else items
                    for item in items_to_process:
                        item_title = item.get("paragraph") or item.get("item") or ""
                        if item_title:
                            parts.append(str(item_title))
                        item_content = item.get("content") or ""
                        if item_content:
                            parts.append(str(item_content))

                        # 新鍵 subparagraphs 或舊鍵 sub_items
                        subparagraphs = item.get("subparagraphs") or []
                        old_sub_items = item.get("sub_items") or []
                        sub_items_to_process = subparagraphs if subparagraphs else old_sub_items
                        for sub in sub_items_to_process:
                            sub_title = sub.get("subparagraph") or sub.get("sub_item") or ""
                            if sub_title:
                                parts.append(str(sub_title))
                            sub_content = sub.get("content") or ""
                            if sub_content:
                                parts.append(str(sub_content))

                            # 第三層 items（目）
                            third_items = sub.get("items") or []
                            for t in third_items:
                                third_title = t.get("item") or ""
                                if third_title:
                                    parts.append(str(third_title))
                                third_content = t.get("content") or ""
                                if third_content:
                                    parts.append(str(third_content))

    # 使用換行連接，利於之後的固定大小/滑動窗口等策略邊界感知
    return "\n\n".join([p for p in parts if isinstance(p, str) and p.strip()])


# === Normalization helpers (add minimal keys for evaluation/alignment) ===
def _nfkc(s: str) -> str:
    try:
        import unicodedata
        return unicodedata.normalize('NFKC', s or '')
    except Exception:
        return s or ''


def _to_int_cn_num_local(s: Any) -> int | None:  # type: ignore[valid-type]
    if s is None:
        return None
    t = _nfkc(str(s))
    if t.isdigit():
        try:
            return int(t)
        except Exception:
            return None
    mapping = {'零':0,'〇':0,'一':1,'二':2,'兩':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'百':100,'千':1000}
    total, section, num = 0, 0, 0
    found = False
    for ch in t:
        if ch in mapping:
            v = mapping[ch]
            if v < 10:
                num = v
                found = True
            else:
                if num == 0:
                    num = 1
                section += num * v
                num = 0
                found = True
    total += section + num
    return total if found else None


def _parse_article_label(label: str) -> tuple[int | None, int | None]:
    t = _nfkc(label)
    m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)條之([0-9一二兩三四五六七八九十百千〇零]+)", t)
    if m:
        return (_to_int_cn_num_local(m.group(1)), _to_int_cn_num_local(m.group(2)))
    m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)-([0-9一二兩三四五六七八九十百千〇零]+)條", t)
    if m:
        return (_to_int_cn_num_local(m.group(1)), _to_int_cn_num_local(m.group(2)))
    m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)條", t)
    if m:
        return (_to_int_cn_num_local(m.group(1)), None)
    return (None, None)


def normalize_corpus_metadata(corpus: Dict[str, Any]) -> Dict[str, Any]:
    try:
        def _strip_keys(obj: Any, keys: set[str]):  # type: ignore[valid-type]
            if isinstance(obj, dict):
                # 先刪除目標鍵
                for k in list(obj.keys()):
                    if k in keys:
                        del obj[k]
                # 遞迴處理子節點
                for v in list(obj.values()):
                    _strip_keys(v, keys)
            elif isinstance(obj, list):
                for v in obj:
                    _strip_keys(v, keys)

        def _filter_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
            allowed = {
                'id',
                'category',
                'article_label',
                'article_number',
                'article_suffix',
                'item_number',
                'clause_type',
                'clause_number',
                'clause_sub_number',
            }
            # 刪除 spans / global_span / page / page_range / note / matched_keywords / confidence / found
            for drop in ['spans', 'global_span', 'page', 'page_range', 'note', 'matched_keywords', 'confidence', 'found']:
                if drop in md:
                    md.pop(drop, None)
            # 僅保留必要鍵
            return {k: v for k, v in md.items() if k in allowed}

        for law in corpus.get('laws', []) or []:
            law_name = law.get('law_name') or ''
            clean_law = law_name.replace('法規名稱：', '')
            # 覆寫 law_name，去掉前綴
            law['law_name'] = clean_law
            # 構建新章節（過濾掉未分類章/未標示條文）
            new_chapters = []
            for chapter in (law.get('chapters', []) or []):
                new_sections = []
                for section in (chapter.get('sections', []) or []):
                    new_articles = []
                    for article in (section.get('articles', []) or []):
                        # 僅保留必要的文章層鍵
                        a_label = article.get('article') or ''
                        a_num, a_suf = _parse_article_label(a_label)
                        md = article.get('metadata') or {}
                        md['category'] = clean_law
                        md['article_label'] = a_label
                        md['article_number'] = a_num
                        md['article_suffix'] = a_suf
                        # 清理 id 前綴
                        if 'id' in md and isinstance(md['id'], str):
                            md['id'] = md['id'].replace('法規名稱：', '')
                        # 過濾為最小 metadata
                        article['metadata'] = _filter_metadata(md)
                        # 刪除文章層非必要鍵（如 spans 等）；保留 article/content/items/metadata
                        for drop_key in list(article.keys()):
                            if drop_key not in {'article', 'content', 'items', 'metadata'}:
                                article.pop(drop_key, None)

                        # 過濾掉 PDF 首頁描述類的假條文：article == '未標示條文' 或 article_number 為 None 且內容是法規描述
                        if article.get('article') == '未標示條文':
                            continue
                        if article['metadata'].get('article_number') is None and '法規名稱' in (article.get('content') or ''):
                            # 進一步防守：若是純描述也過濾
                            continue

                        for item in article.get('items', []) or []:
                            # 僅保留必要的項層鍵
                            md_i = item.get('metadata') or {}
                            md_i['category'] = clean_law
                            md_i['article_label'] = a_label
                            md_i['article_number'] = a_num
                            md_i['article_suffix'] = a_suf
                            md_i['item_number'] = _to_int_cn_num_local(item.get('item'))
                            if 'id' in md_i and isinstance(md_i['id'], str):
                                md_i['id'] = md_i['id'].replace('法規名稱：', '')
                            item['metadata'] = _filter_metadata(md_i)
                            # 刪除項層非必要鍵（保留 item/content/sub_items/metadata）
                            for drop_key in list(item.keys()):
                                if drop_key not in {'item', 'content', 'sub_items', 'metadata'}:
                                    item.pop(drop_key, None)
                            for sub in item.get('sub_items', []) or []:
                                # 僅保留必要的子項層鍵
                                md_s = sub.get('metadata') or {}
                                md_s['category'] = clean_law
                                md_s['article_label'] = a_label
                                md_s['article_number'] = a_num
                                md_s['article_suffix'] = a_suf
                                md_s['clause_type'] = '目'
                                md_s['clause_number'] = _to_int_cn_num_local(sub.get('sub_item'))
                                md_s['clause_sub_number'] = None
                                if 'id' in md_s and isinstance(md_s['id'], str):
                                    md_s['id'] = md_s['id'].replace('法規名稱：', '')
                                sub['metadata'] = _filter_metadata(md_s)
                                # 刪除子項層非必要鍵（保留 sub_item/content/metadata）
                                for drop_key in list(sub.keys()):
                                    if drop_key not in {'sub_item', 'content', 'metadata'}:
                                        sub.pop(drop_key, None)
                        # 保留處理後的條文
                        new_articles.append(article)

                    # 若該節存在有效條文才保留
                    if new_articles:
                        section['articles'] = new_articles
                        new_sections.append(section)
                # 若該章存在有效節才保留
                if new_sections:
                    chapter['sections'] = new_sections
                    new_chapters.append(chapter)
            # 覆寫章節
            law['chapters'] = new_chapters
        return corpus
    except Exception:
        return corpus


# === New: Evaluate with qa_gold (no mapping, regex-based relevance) ===
@router.post("/evaluate/gold")
async def evaluate_with_gold(payload: Dict[str, Any]):
    """
    以 qa_gold 與 chunking_results 計算 P@K / R@K / PrecisionΩ（簡化版）
    payload = {
      doc_id: str,
      qa_gold: [ { query, label, gold: { law, article_number, article_suffix, ... } }, ... ],
      chunking_results: [ { strategy, config: { chunk_size, overlap_ratio, ... }, chunks: [text], ... }, ... ],
      k_values: [int]
    }
    """
    try:
        doc_id = payload.get('doc_id')
        qa_gold = payload.get('qa_gold') or []
        chunking_results = payload.get('chunking_results') or []
        k_values = payload.get('k_values') or [1, 3, 5, 10]

        if not isinstance(qa_gold, list) or not isinstance(chunking_results, list):
            return JSONResponse(status_code=400, content={"error": "qa_gold 或 chunking_results 格式錯誤"})

        # 僅取正例（大小寫寬鬆）
        positives = [
            q for q in qa_gold
            if str(q.get('label', '')).strip().lower() == 'yes'
        ]
        if not positives:
            return {"results": [], "summary": {"message": "沒有正例可供評測"}}

        def is_chunk_relevant(chunk_content: str, gold_info: Dict[str, Any]) -> bool:
            """判斷chunk是否與gold標準相關（使用統一的評測邏輯）"""
            from .main import is_relevant_chunk
            return is_relevant_chunk(chunk_content, gold_info)

        # 顯示評測配置信息
        print(f"🔧 評測配置:")
        print(f"   文檔ID: {doc_id}")
        print(f"   QA數據數量: {len(qa_gold)}")
        print(f"   正例數量: {len(positives)}")
        print(f"   分塊策略數量: {len(chunking_results)}")
        print(f"   K值: {k_values}")
        
        results = []
        for cr in chunking_results:
            # 兼容不同字段命名：chunks / all_chunks / chunks_with_span
            chunks_raw = (
                cr.get('chunks')
                or cr.get('all_chunks')
                or cr.get('chunks_with_span')
                or []
            )
            chunks: list[str] = []
            if isinstance(chunks_raw, list):
                for c in chunks_raw:
                    if isinstance(c, str):
                        chunks.append(c)
                    elif isinstance(c, dict):
                        # 嘗試多個常見鍵
                        text = (
                            c.get('content')
                            or c.get('text')
                            or c.get('chunk')
                            or ''
                        )
                        if isinstance(text, str) and text:
                            chunks.append(text)
            # 若仍為空，保底為空列表
            # 使用與實際檢索相同的embedding策略
            print(f"🔄 開始為策略 '{cr.get('strategy')}' 進行embedding...")
            
            # 檢查是否有現有的embedding
            from .main import store
            current_embeddings = store.embeddings
            current_chunks = store.chunks_flat
            current_doc_ids = store.chunk_doc_ids
            
            # 如果chunks不同，需要重新計算embedding
            chunks_changed = (len(chunks) != len(current_chunks) or 
                            any(c1 != c2 for c1, c2 in zip(chunks, current_chunks)))
            
            if chunks_changed:
                print(f"📊 Chunks已改變，重新計算embedding...")
                # 臨時更新store中的chunks
                store.chunks_flat = chunks
                store.chunk_doc_ids = [doc_id] * len(chunks)  # 假設所有chunks都來自同一個doc
                
                # 重新計算embedding
                try:
                    import asyncio
                    from .main import embed_bge_m3, embed_gemini
                    import os
                    
                    # 根據配置選擇embedding方法
                    if os.getenv('USE_BGE_M3_EMBEDDING', 'False').lower() == 'true':
                        print(f"🔧 使用BGE-M3 embedding...")
                        embeddings = embed_bge_m3(chunks)
                    elif os.getenv('USE_GEMINI_EMBEDDING', 'False').lower() == 'true':
                        print(f"🔧 使用Gemini embedding...")
                        embeddings = asyncio.run(embed_gemini(chunks))
                    else:
                        raise Exception("沒有啟用任何embedding方法")
                    
                    import numpy as np
                    embeddings = np.array(embeddings)
                    store.embeddings = embeddings
                    print(f"✅ Embedding計算完成，維度: {embeddings.shape}")
                except Exception as e:
                    print(f"❌ Embedding計算失敗: {e}")
                    # 回退到TF-IDF
                    try:
                        vec = TfidfVectorizer()
                        mat = vec.fit_transform(chunks) if chunks else None
                        print(f"⚠️  回退到TF-IDF檢索")
                    except Exception:
                        vec, mat = None, None
                        print(f"❌ TF-IDF也失敗了")
            else:
                if current_embeddings is not None:
                    if hasattr(current_embeddings, 'shape'):
                        print(f"✅ 使用現有embedding，維度: {current_embeddings.shape}")
                    else:
                        print(f"✅ 使用現有embedding，類型: {type(current_embeddings)}")
                    embeddings = current_embeddings
                else:
                    print(f"⚠️  沒有現有embedding，跳過此策略")
                    continue

            # 逐題計算
            per_k_precisions: Dict[int, list] = {k: [] for k in k_values}
            per_k_recalls: Dict[int, list] = {k: [] for k in k_values}

            for q in positives:
                query = q.get('query', '')
                gold_info = q.get('gold', {}) or {}
                
                # 檢查是否有法條號碼
                if gold_info.get('article_number') is None:
                    continue  # 跳過沒有法條號碼的題目
                
                print(f"🔍 評測查詢: '{query[:50]}...'")
                print(f"📋 Gold標準: {gold_info}")

                # 使用統一的相關性判斷邏輯
                gold_unit_hits = set()
                for idx, text in enumerate(chunks):
                    if not isinstance(text, str):
                        continue
                    if is_chunk_relevant(text, gold_info):
                        gold_unit_hits.add(idx)
                        print(f"   ✅ Chunk {idx+1} 相關: {text[:50]}...")
                
                if not gold_unit_hits:
                    print(f"   ❌ 沒有找到相關chunks")
                    continue
                
                print(f"📊 找到 {len(gold_unit_hits)} 個相關chunks: {list(gold_unit_hits)}")

                # 使用與實際檢索相同的策略（HybridRAG或密集向量）
                retrieved_order = []
                
                if 'embeddings' in locals() and embeddings is not None:
                    print(f"🔍 使用HybridRAG檢索...")
                    try:
                        # 使用HybridRAG檢索邏輯
                        from .main import rank_with_dense_vectors, hybrid_rank
                        from .hybrid_search import HybridConfig
                        
                        # 構建nodes格式
                        nodes = []
                        for i, chunk in enumerate(chunks):
                            # 嘗試從chunking_results中獲取metadata
                            metadata = {}
                            if 'chunks_with_span' in cr and i < len(cr['chunks_with_span']):
                                chunk_data = cr['chunks_with_span'][i]
                                if isinstance(chunk_data, dict):
                                    metadata = chunk_data.get('metadata', {})
                            
                            nodes.append({
                                "content": chunk,
                                "metadata": metadata,
                                "doc_id": doc_id,
                                "chunk_index": i
                            })
                        
                        # 使用密集向量檢索
                        max_k = max(k_values)
                        dense_top_k = min(len(nodes), max_k * 4)
                        all_vec_idxs, all_vec_sims = rank_with_dense_vectors(query, k=len(nodes))
                        
                        # 映射向量分數
                        node_vector_scores = [0.0] * len(nodes)
                        for rank_idx, node_idx in enumerate(all_vec_idxs):
                            node_vector_scores[node_idx] = float(all_vec_sims[rank_idx])
                        
                        # 取前dense_top_k個候選
                        top_vec_pairs = sorted(
                            [(i, s) for i, s in enumerate(node_vector_scores)], 
                            key=lambda x: x[1], reverse=True
                        )[:dense_top_k]
                        
                        candidate_nodes = [nodes[i] for i, _ in top_vec_pairs]
                        candidate_scores = [s for _, s in top_vec_pairs]
                        
                        # 使用HybridRAG排序
                        config = HybridConfig(
                            alpha=0.8,
                            w_law_match=0.15,
                            w_article_match=0.15,
                            w_keyword_hit=0.05,
                            max_bonus=0.4
                        )
                        
                        hybrid_results = hybrid_rank(
                            query, candidate_nodes, k=max_k, config=config, vector_scores=candidate_scores
                        )
                        
                        # 提取檢索順序
                        retrieved_order = [result['chunk_index'] for result in hybrid_results]
                        print(f"🔍 HybridRAG檢索完成，檢索順序: {retrieved_order[:10]}...")
                        
                    except Exception as e:
                        print(f"⚠️  HybridRAG檢索失敗: {e}，回退到密集向量檢索")
                        try:
                            from .main import rank_with_dense_vectors
                            max_k = max(k_values)
                            idxs, sims = rank_with_dense_vectors(query, k=max_k)
                            retrieved_order = idxs
                            print(f"🔍 使用密集向量檢索，檢索順序: {retrieved_order[:10]}...")
                        except Exception as e2:
                            print(f"⚠️  密集向量檢索也失敗: {e2}，使用順序檢索")
                            retrieved_order = list(range(len(chunks)))
                
                elif 'vec' in locals() and vec is not None and 'mat' in locals() and mat is not None:
                    print(f"🔍 使用TF-IDF檢索...")
                    try:
                        qv = vec.transform([query])
                        sims = cosine_similarity(qv, mat).flatten()
                        order = sims.argsort()[::-1]
                        retrieved_order = list(order)
                        print(f"🔍 TF-IDF檢索完成，檢索順序: {retrieved_order[:10]}...")
                    except Exception as e:
                        print(f"⚠️  TF-IDF檢索失敗: {e}，使用順序檢索")
                        retrieved_order = list(range(len(chunks)))
                else:
                    print("⚠️  沒有可用的檢索方法，使用順序檢索")
                    retrieved_order = list(range(len(chunks)))

                for k in k_values:
                    topk = set(retrieved_order[:k])
                    # P@K: 命中比例
                    prec = (len(topk & gold_unit_hits) / k) if k > 0 else 0.0
                    # R@K: 覆蓋比例
                    rec = (len(topk & gold_unit_hits) / len(gold_unit_hits)) if gold_unit_hits else 0.0
                    per_k_precisions[k].append(prec)
                    per_k_recalls[k].append(rec)
                    print(f"   📈 P@{k}={prec:.3f}, R@{k}={rec:.3f}")

            # 恢復原始store狀態
            if chunks_changed:
                store.chunks_flat = current_chunks
                store.chunk_doc_ids = current_doc_ids
                store.embeddings = current_embeddings
                print(f"🔄 已恢復原始store狀態")

            # 匯總
            result_entry = {
                "strategy": cr.get('strategy'),
                "config": cr.get('config'),
                "chunk_count": cr.get('chunk_count'),
                "metrics": {
                    "precision_at_k": {k: (sum(v)/len(v) if v else 0.0) for k, v in per_k_precisions.items()},
                    "recall_at_k": {k: (sum(v)/len(v) if v else 0.0) for k, v in per_k_recalls.items()},
                }
            }
            results.append(result_entry)

        # 簡單摘要：以 P@5 排序
        best = None
        if results:
            best = max(results, key=lambda r: r["metrics"]["precision_at_k"].get(5, 0))

        return {
            "results": results,
            "summary": {
                "best_by_p_at_5": best
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


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


class QASetItem(BaseModel):
    query: str
    label: str
    answer: str
    snippets: Optional[List[Dict[str, Any]]] = []
    spans: Optional[List[Dict[str, Any]]] = []
    article_metadata: Optional[Dict[str, Any]] = None
    relevant_chunks: Optional[List[str]] = []  # 映射後的chunk IDs


class QASetUploadRequest(BaseModel):
    doc_id: str
    chunk_sizes: List[int] = [300, 500, 800]
    overlap_ratios: List[float] = [0.0, 0.1, 0.2]
    strategy: str = "fixed_size"
    k_values: List[int] = [1, 3, 5, 10]


class ChunkMappingResult(BaseModel):
    task_id: str
    status: str
    qa_set: List[QASetItem]
    chunk_configs: List[Dict[str, Any]]
    mapping_results: Dict[str, Dict[str, List[str]]]  # config_id -> question_id -> chunk_ids


class MultipleChunkingRequest(BaseModel):
    doc_id: str
    strategies: List[str] = ["fixed_size"]
    chunk_sizes: List[int] = [300, 500, 800]
    overlap_ratios: List[float] = [0.0, 0.1, 0.2]


class StrategyEvaluationRequest(BaseModel):
    doc_id: str
    chunking_results: List[Dict[str, Any]]
    qa_mapping_result: Dict[str, Any]
    test_queries: List[str]
    k_values: List[int] = [1, 3, 5, 10]


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
    secondary_size_options: List[int] = [300, 400, 500]  # 混合分割選項


# 創建評估存儲實例
eval_store = EvaluationStore()


def calculate_iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
    """計算兩個span的IoU (Intersection over Union)"""
    start1, end1 = span1
    start2, end2 = span2
    
    # 計算交集
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_length = max(0, intersection_end - intersection_start)
    
    # 計算聯集
    union_length = (end1 - start1) + (end2 - start2) - intersection_length
    
    if union_length == 0:
        return 0.0
    
    return intersection_length / union_length


def map_spans_to_chunks(qa_items: List[Dict], chunks_with_span: List[Dict[str, Any]], iou_threshold: float = 0.05, overlap_threshold: float = 0.3) -> List[Dict]:
    """
    將QA set中的spans映射到chunks（使用帶span信息的chunks）
    使用更智能的映射策略來提高成功率到90%-95%
    
    主要改進：
    1. 降低IoU閾值到0.05，因為小span與大chunk的IoU天然較低
    2. 使用多種匹配策略：IoU、重疊率、包含關係、鄰近匹配
    3. 對小span進行特殊處理
    
    參數:
    - qa_items: QA set項目列表
    - chunks_with_span: 帶span信息的chunk列表 [{"content": str, "span": {"start": int, "end": int}, "chunk_id": str, "metadata": dict}, ...]
    - iou_threshold: IoU閾值，默認為0.05（大幅降低）
    - overlap_threshold: 重疊閾值，默認為0.3（降低）
    
    返回:
    - 映射後的QA items，包含relevant_chunks字段
    """
    mapped_qa_items = []
    mapped_count = 0
    positive_qa_count = 0
    
    print(f"開始映射: {len(qa_items)} 個QA項目, {len(chunks_with_span)} 個chunks")
    print(f"IoU閾值: {iou_threshold}, 重疊閾值: {overlap_threshold}")
    
    for qa_item in qa_items:
        mapped_item = qa_item.copy()
        relevant_chunks = []
        
        # 只處理Label為"Yes"的QA項目
        if qa_item.get('label', '').lower() != 'yes':
            mapped_item['relevant_chunks'] = []
            mapped_qa_items.append(mapped_item)
            continue
        
        positive_qa_count += 1
        print(f"處理正例問題 {positive_qa_count}: '{qa_item.get('query', '')[:50]}...'")
        
        # 處理snippets字段（優先使用snippets，因為它更準確）
        if 'snippets' in qa_item and qa_item['snippets']:
            snippet = qa_item['snippets'][0]  # 使用第一個snippet
            if 'span' in snippet and len(snippet['span']) == 2:
                qa_span = snippet['span']
                qa_span_length = qa_span[1] - qa_span[0]
                print(f"  QA span: [{qa_span[0]}-{qa_span[1]}] (長度: {qa_span_length})")
                print(f"  檢查 {len(chunks_with_span)} 個chunks...")
                
                # 使用多種匹配策略
                for chunk_info in chunks_with_span:
                    chunk_span = chunk_info['span']
                    chunk_start = chunk_span['start']
                    chunk_end = chunk_span['end']
                    chunk_id = chunk_info['chunk_id']
                    chunk_length = chunk_end - chunk_start
                    
                    # 策略1: 計算IoU
                    iou = calculate_iou(qa_span, (chunk_start, chunk_end))
                    
                    # 策略2: 計算重疊率（chunk覆蓋qa_span的比例）
                    overlap_start = max(qa_span[0], chunk_start)
                    overlap_end = min(qa_span[1], chunk_end)
                    if overlap_end > overlap_start:
                        overlap_length = overlap_end - overlap_start
                        overlap_ratio = overlap_length / qa_span_length if qa_span_length > 0 else 0
                    else:
                        overlap_ratio = 0
                    
                    # 策略3: 檢查包含關係（chunk完全包含qa_span）
                    chunk_contains_qa = chunk_start <= qa_span[0] and chunk_end >= qa_span[1]
                    
                    # 策略4: 檢查鄰近關係（qa_span在chunk附近，距離小於chunk長度的10%）
                    if qa_span[0] > chunk_end:
                        distance_to_chunk = qa_span[0] - chunk_end
                    elif qa_span[1] < chunk_start:
                        distance_to_chunk = chunk_start - qa_span[1]
                    else:
                        distance_to_chunk = 0  # 有重疊
                    nearby_threshold = chunk_length * 0.1  # chunk長度的10%
                    is_nearby = distance_to_chunk <= nearby_threshold
                    
                    # 策略5: 對於特別小的span（<100字符），使用更寬鬆的條件
                    is_small_span = qa_span_length < 100
                    
                    # 綜合匹配條件 - 使用更寬鬆的條件
                    match_found = False
                    match_reason = ""
                    
                    if iou > iou_threshold:
                        match_found = True
                        match_reason = f"IoU匹配 (IoU: {iou:.3f})"
                    elif chunk_contains_qa:
                        match_found = True
                        match_reason = f"包含關係 (IoU: {iou:.3f})"
                    elif overlap_ratio > overlap_threshold:
                        match_found = True
                        match_reason = f"重疊率匹配 (重疊率: {overlap_ratio:.3f}, IoU: {iou:.3f})"
                    elif is_small_span and (overlap_ratio > 0.05 or iou > 0.005):
                        # 對小span使用更寬鬆的條件
                        match_found = True
                        match_reason = f"小span寬鬆匹配 (重疊率: {overlap_ratio:.3f}, IoU: {iou:.3f})"
                    elif is_nearby and iou > 0.005:
                        # 鄰近匹配
                        match_found = True
                        match_reason = f"鄰近匹配 (距離: {distance_to_chunk}, IoU: {iou:.3f})"
                    elif is_small_span and is_nearby:
                        # 小span且鄰近，使用最寬鬆的條件
                        match_found = True
                        match_reason = f"小span鄰近匹配 (距離: {distance_to_chunk}, IoU: {iou:.3f})"
                    
                    if match_found and chunk_id not in relevant_chunks:
                        relevant_chunks.append(chunk_id)
                        print(f"    找到匹配chunk: {chunk_id} - {match_reason}")
        
        # 如果沒有snippets，嘗試使用spans字段
        elif 'spans' in qa_item and qa_item['spans']:
            span = qa_item['spans'][0]  # 使用第一個span
            span_start = span.get('start_char', 0)
            span_end = span.get('end_char', span_start)
            qa_span = (span_start, span_end)
            qa_span_length = qa_span[1] - qa_span[0]
            print(f"  QA span: [{qa_span[0]}-{qa_span[1]}] (長度: {qa_span_length})")
            
            # 使用相同的多種匹配策略
            for chunk_info in chunks_with_span:
                chunk_span = chunk_info['span']
                chunk_start = chunk_span['start']
                chunk_end = chunk_span['end']
                chunk_id = chunk_info['chunk_id']
                chunk_length = chunk_end - chunk_start
                
                # 計算各種指標
                iou = calculate_iou(qa_span, (chunk_start, chunk_end))
                
                overlap_start = max(qa_span[0], chunk_start)
                overlap_end = min(qa_span[1], chunk_end)
                if overlap_end > overlap_start:
                    overlap_length = overlap_end - overlap_start
                    overlap_ratio = overlap_length / qa_span_length if qa_span_length > 0 else 0
                else:
                    overlap_ratio = 0
                
                chunk_contains_qa = chunk_start <= qa_span[0] and chunk_end >= qa_span[1]
                
                if qa_span[0] > chunk_end:
                    distance_to_chunk = qa_span[0] - chunk_end
                elif qa_span[1] < chunk_start:
                    distance_to_chunk = chunk_start - qa_span[1]
                else:
                    distance_to_chunk = 0  # 有重疊
                nearby_threshold = chunk_length * 0.1
                is_nearby = distance_to_chunk <= nearby_threshold
                is_small_span = qa_span_length < 100
                
                # 綜合匹配條件 - 使用更寬鬆的條件
                match_found = False
                match_reason = ""
                
                if iou > iou_threshold:
                    match_found = True
                    match_reason = f"IoU匹配 (IoU: {iou:.3f})"
                elif chunk_contains_qa:
                    match_found = True
                    match_reason = f"包含關係 (IoU: {iou:.3f})"
                elif overlap_ratio > overlap_threshold:
                    match_found = True
                    match_reason = f"重疊率匹配 (重疊率: {overlap_ratio:.3f}, IoU: {iou:.3f})"
                elif is_small_span and (overlap_ratio > 0.05 or iou > 0.005):
                    # 對小span使用更寬鬆的條件
                    match_found = True
                    match_reason = f"小span寬鬆匹配 (重疊率: {overlap_ratio:.3f}, IoU: {iou:.3f})"
                elif is_nearby and iou > 0.005:
                    # 鄰近匹配
                    match_found = True
                    match_reason = f"鄰近匹配 (距離: {distance_to_chunk}, IoU: {iou:.3f})"
                elif is_small_span and is_nearby:
                    # 小span且鄰近，使用最寬鬆的條件
                    match_found = True
                    match_reason = f"小span鄰近匹配 (距離: {distance_to_chunk}, IoU: {iou:.3f})"
                
                if match_found and chunk_id not in relevant_chunks:
                    relevant_chunks.append(chunk_id)
                    print(f"    找到匹配chunk: {chunk_id} - {match_reason}")
        
        # 統計映射成功的數量
        if relevant_chunks:
            mapped_count += 1
        else:
            print(f"  警告: 正例問題沒有找到相關chunks，嘗試fallback策略")
            
            # Fallback策略：找到距離最近的chunk
            min_distance = float('inf')
            closest_chunk_id = None
            
            if 'snippets' in qa_item and qa_item['snippets']:
                snippet = qa_item['snippets'][0]
                if 'span' in snippet and len(snippet['span']) == 2:
                    qa_span = snippet['span']
                    
                    for chunk_info in chunks_with_span:
                        chunk_span = chunk_info['span']
                        chunk_start = chunk_span['start']
                        chunk_end = chunk_span['end']
                        chunk_id = chunk_info['chunk_id']
                        
                        # 計算距離
                        if qa_span[0] > chunk_end:
                            distance = qa_span[0] - chunk_end
                        elif qa_span[1] < chunk_start:
                            distance = chunk_start - qa_span[1]
                        else:
                            distance = 0  # 有重疊
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_chunk_id = chunk_id
            
            # 策略2: 如果沒有span信息或距離太遠，使用內容匹配
            if not closest_chunk_id or min_distance >= 1000:
                print(f"    嘗試內容匹配策略")
                
                query = qa_item.get('query', '')
                answer = qa_item.get('answer', '')
                
                # 提取關鍵詞
                import re
                keywords = set()
                
                # 從問題和答案中提取法條號碼
                article_patterns = [
                    r"第(\d+)條",
                    r"第(\d+)條之(\d+)",
                    r"第(\d+)-(\d+)條"
                ]
                
                for pattern in article_patterns:
                    for text in [query, answer]:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            if isinstance(match, tuple):
                                keywords.add(f"第{match[0]}條")
                            else:
                                keywords.add(f"第{match}條")
                
                # 添加其他關鍵詞
                keywords.update(query.split()[:5])  # 前5個詞
                keywords.update(answer.split()[:5])  # 前5個詞
                
                print(f"    提取的關鍵詞: {list(keywords)[:10]}")
                
                # 使用TF-IDF進行內容匹配
                if keywords and chunks_with_span:
                    try:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.metrics.pairwise import cosine_similarity
                        
                        # 準備文本數據
                        texts = [query + " " + answer] + [chunk_info['content'] for chunk_info in chunks_with_span]
                        
                        vectorizer = TfidfVectorizer(
                            max_features=1000,
                            stop_words=None,
                            ngram_range=(1, 2)
                        )
                        
                        tfidf_matrix = vectorizer.fit_transform(texts)
                        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
                        
                        # 找到相似度超過閾值的chunks
                        similarity_threshold = 0.1  # 降低閾值
                        best_similarity = 0
                        best_chunk_id = None
                        
                        for i, similarity in enumerate(cosine_similarities[0]):
                            if similarity > similarity_threshold and similarity > best_similarity:
                                best_similarity = similarity
                                best_chunk_id = chunks_with_span[i]['chunk_id']
                        
                        if best_chunk_id:
                            relevant_chunks.append(best_chunk_id)
                            print(f"    內容匹配找到chunk: {best_chunk_id} (相似度: {best_similarity:.3f})")
                        
                    except Exception as e:
                        print(f"    TF-IDF匹配失敗: {e}")
                        
                        # 回退到簡單關鍵詞匹配
                        for chunk_info in chunks_with_span:
                            chunk_content = chunk_info['content']
                            chunk_id = chunk_info['chunk_id']
                            
                            # 檢查關鍵詞匹配
                            keyword_matches = sum(1 for keyword in keywords if keyword in chunk_content)
                            
                            if keyword_matches > 0:
                                relevant_chunks.append(chunk_id)
                                print(f"    關鍵詞匹配找到chunk: {chunk_id} (匹配數: {keyword_matches})")
                                break
            
            # 策略3: 如果還是沒有找到，使用最近的chunk
            if not relevant_chunks and closest_chunk_id and min_distance < 2000:  # 放寬距離限制
                relevant_chunks.append(closest_chunk_id)
                print(f"    使用最近chunk: {closest_chunk_id} (距離: {min_distance})")
            
            # 如果仍然沒有找到
            if not relevant_chunks:
                print(f"    該問題無法找到合適的chunk")
                if 'snippets' in qa_item and qa_item['snippets']:
                    print(f"    該問題有 {len(qa_item['snippets'])} 個snippets但未匹配到任何chunk")
                elif 'spans' in qa_item and qa_item['spans']:
                    print(f"    該問題有 {len(qa_item['spans'])} 個spans但未匹配到任何chunk")
                else:
                    print(f"    該問題沒有snippets或spans信息，且內容匹配失敗")
        
        mapped_item['relevant_chunks'] = relevant_chunks
        mapped_qa_items.append(mapped_item)
    
    # 計算映射成功率
    success_rate = mapped_count / positive_qa_count if positive_qa_count > 0 else 0
    print(f"映射完成: {mapped_count}/{positive_qa_count} 個正例問題有相關chunks")
    print(f"映射成功率: {success_rate*100:.2f}%")
    
    return mapped_qa_items


def map_spans_to_chunks_legacy(qa_items: List[Dict], chunks: List[str], chunk_size: int, overlap: int, strategy: str = "fixed_size") -> List[Dict]:
    """
    將QA set中的spans映射到chunks（舊版本，向後兼容）
    
    參數:
    - qa_items: QA set項目列表
    - chunks: 分塊後的文本列表
    - chunk_size: 分塊大小
    - overlap: 重疊大小
    - strategy: 分塊策略
    
    返回:
    - 映射後的QA items，包含relevant_chunks字段
    """
    # 計算每個chunk的字符範圍
    chunk_ranges = []
    current_pos = 0
    
    for i, chunk in enumerate(chunks):
        chunk_start = current_pos
        chunk_end = current_pos + len(chunk)
        chunk_ranges.append((chunk_start, chunk_end, f"chunk_{i+1:03d}"))
        
        # 計算下一個chunk的起始位置（考慮重疊）
        current_pos = chunk_end - overlap
    
    mapped_qa_items = []
    
    for qa_item in qa_items:
        mapped_item = qa_item.copy()
        relevant_chunks = []
        
        # 處理spans字段
        if 'spans' in qa_item and qa_item['spans']:
            for span in qa_item['spans']:
                span_start = span.get('start_char', 0)
                span_end = span.get('end_char', span_start)
                
                # 找到與此span有IoU > 0.5的chunks
                for chunk_start, chunk_end, chunk_id in chunk_ranges:
                    iou = calculate_iou((span_start, span_end), (chunk_start, chunk_end))
                    if iou > 0.5:
                        if chunk_id not in relevant_chunks:
                            relevant_chunks.append(chunk_id)
        
        # 處理snippets字段（如果存在）
        if 'snippets' in qa_item and qa_item['snippets']:
            for snippet in qa_item['snippets']:
                if 'span' in snippet:
                    span_start, span_end = snippet['span']
                    
                    # 找到與此span有IoU > 0.5的chunks
                    for chunk_start, chunk_end, chunk_id in chunk_ranges:
                        iou = calculate_iou((span_start, span_end), (chunk_start, chunk_end))
                        if iou > 0.5:
                            if chunk_id not in relevant_chunks:
                                relevant_chunks.append(chunk_id)
        
        # 如果沒有找到相關chunks，檢查是否為負例
        if not relevant_chunks and qa_item.get('label', '').lower() == 'no':
            # 負例不需要相關chunks
            pass
        elif not relevant_chunks:
            # 正例但沒有找到相關chunks，可能需要調整IoU閾值或檢查數據
            print(f"警告: 問題 '{qa_item.get('query', '')[:50]}...' 沒有找到相關chunks")
        
        mapped_item['relevant_chunks'] = relevant_chunks
        mapped_qa_items.append(mapped_item)
    
    return mapped_qa_items


def generate_text_from_merged_doc(merged_doc: Dict[str, Any]) -> str:
    """
    從合併的法律文檔JSON結構生成文本內容
    
    參數:
    - merged_doc: 合併後的法律文檔JSON結構
    
    返回:
    - 生成的文本內容
    """
    if not merged_doc or "laws" not in merged_doc:
        return ""
    
    text_parts = []
    
    for law in merged_doc["laws"]:
        law_name = law.get("law_name", "未命名法規")
        text_parts.append(f"=== {law_name} ===\n")
        
        chapters = law.get("chapters", [])
        for chapter in chapters:
            chapter_name = chapter.get("chapter", "")
            if chapter_name:
                text_parts.append(f"\n{chapter_name}\n")
            
            sections = chapter.get("sections", [])
            for section in sections:
                section_name = section.get("section", "")
                if section_name:
                    text_parts.append(f"\n{section_name}\n")
                
                articles = section.get("articles", [])
                for article in articles:
                    article_name = article.get("article", "")
                    article_content = article.get("content", "")
                    
                    if article_name:
                        text_parts.append(f"\n{article_name}")
                        if article_content:
                            text_parts.append(f" {article_content}")
                        text_parts.append("\n")
                    
                    # 處理項目 - 支援新結構 (paragraphs) 和舊結構 (items)
                    paragraphs = article.get("paragraphs", [])
                    items = article.get("items", [])
                    
                    # 使用 paragraphs 如果存在，否則使用 items
                    items_to_process = paragraphs if paragraphs else items
                    
                    for item in items_to_process:
                        # 支援新結構的鍵名
                        item_name = item.get("paragraph", item.get("item", ""))
                        item_content = item.get("content", "")
                        
                        if item_name and item_content:
                            text_parts.append(f"{item_name} {item_content}\n")
                        
                        # 處理子項目 - 支援新結構 (subparagraphs) 和舊結構 (sub_items)
                        subparagraphs = item.get("subparagraphs", [])
                        sub_items = item.get("sub_items", [])
                        
                        # 使用 subparagraphs 如果存在，否則使用 sub_items
                        sub_items_to_process = subparagraphs if subparagraphs else sub_items
                        
                        for sub_item in sub_items_to_process:
                            # 支援新結構的鍵名
                            sub_item_name = sub_item.get("subparagraph", sub_item.get("sub_item", ""))
                            sub_item_content = sub_item.get("content", "")
                            
                            if sub_item_name and sub_item_content:
                                text_parts.append(f"{sub_item_name} {sub_item_content}\n")
                            
                            # 處理第三層項目 (items)
                            third_level_items = sub_item.get("items", [])
                            for third_item in third_level_items:
                                third_item_name = third_item.get("item", "")
                                third_item_content = third_item.get("content", "")
                                
                                if third_item_name and third_item_content:
                                    text_parts.append(f"{third_item_name} {third_item_content}\n")
        
        text_parts.append("\n" + "="*50 + "\n")
    
    return "\n".join(text_parts)


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
            # 準備策略參數
            strategy_kwargs = {}
            if hasattr(config, 'strategy'):
                strategy = config.strategy
            else:
                strategy = "fixed_size"
            
            # 根據策略添加特定參數
            if strategy == "sliding_window" and hasattr(config, 'step_size'):
                strategy_kwargs['step_size'] = config.step_size
                if hasattr(config, 'window_size'):
                    strategy_kwargs['window_size'] = config.window_size
                if hasattr(config, 'boundary_aware'):
                    strategy_kwargs['boundary_aware'] = config.boundary_aware
                if hasattr(config, 'preserve_sentences'):
                    strategy_kwargs['preserve_sentences'] = config.preserve_sentences
                if hasattr(config, 'min_chunk_size_sw'):
                    strategy_kwargs['min_chunk_size_sw'] = config.min_chunk_size_sw
                if hasattr(config, 'max_chunk_size_sw'):
                    strategy_kwargs['max_chunk_size_sw'] = config.max_chunk_size_sw
            elif strategy == "hierarchical" and hasattr(config, 'level_depth'):
                strategy_kwargs['level_depth'] = config.level_depth
            elif strategy == "semantic" and hasattr(config, 'similarity_threshold'):
                strategy_kwargs['similarity_threshold'] = config.similarity_threshold
            elif strategy == "structured_hierarchical" and hasattr(config, 'chunk_by'):
                strategy_kwargs['chunk_by'] = config.chunk_by
            
            result = evaluate_chunk_config(
                doc.text, 
                task.test_queries, 
                config.chunk_size, 
                config.overlap_ratio,
                strategy=strategy,
                **strategy_kwargs
            )
            results.append(result)
            
            # 更新進度
            progress = (i + 1) / total_configs
            eval_store.update_task_status(task_id, "running", progress=progress)
        
        eval_store.update_task_status(task_id, "completed", results=results)
        
    except Exception as e:
        eval_store.update_task_status(task_id, "failed", error_message=str(e))


@router.get("/health")
async def health_check():
    """健康檢查"""
    return {"status": "healthy", "message": "RAG API 運行正常"}


@router.get("/docs")
async def list_documents():
    """列出所有文檔"""
    docs = store.list_docs()
    return {
        "docs": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "text_length": len(doc.text) if doc.text else 0,
                "has_json_data": bool(doc.json_data),
                "chunks_count": len(doc.chunks) if doc.chunks else 0,
                "chunk_size": doc.chunk_size,
                "overlap": doc.overlap,
                "chunking_strategy": getattr(doc, 'chunking_strategy', None),
                "structured_chunks": getattr(doc, 'structured_chunks', None)
            }
            for doc in docs
        ],
        "total": len(docs)
    }


# PDF 轉換路由
@router.post("/upload-json")
async def upload_json(file: UploadFile = File(...)):
    """上傳JSON文件"""
    try:
        # 驗證文件格式
        if not file.filename or not file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="只支持JSON格式文件")
        
        # 讀取文件內容
        file_content = await file.read()
        
        # 解析JSON內容
        try:
            json_data = json.loads(file_content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"JSON格式錯誤: {str(e)}")
        
        # 生成文檔ID
        doc_id = str(uuid.uuid4())
        
        # 從JSON結構生成文本內容
        # 如果是法條JSON格式（包含laws字段），使用專門的函數
        if isinstance(json_data, dict) and "laws" in json_data:
            text_content = generate_text_from_merged_doc(json_data)
        else:
            # 對於其他JSON格式，直接轉換為字符串
            text_content = json.dumps(json_data, ensure_ascii=False, indent=2)
        
        if not text_content or not text_content.strip():
            raise HTTPException(status_code=400, detail="JSON文件中沒有可用的文本內容")
        
        # 創建文檔記錄
        doc_record = DocRecord(
            id=doc_id,
            filename=file.filename,
            text=text_content,
            chunks=[],
            chunk_size=0,
            overlap=0,
            json_data=json_data
        )
        
        # 存儲文檔
        store.add_doc(doc_record)
        
        # 計算響應數據
        response_data = {
            "doc_id": doc_id,
            "filename": file.filename,
            "text_length": len(text_content),
            "metadata": json_data,  # 添加JSON數據到響應中
            "message": "JSON文件上傳成功"
        }
        
        # 如果是法條JSON，添加laws_count
        if isinstance(json_data, dict) and "laws" in json_data:
            response_data["laws_count"] = len(json_data.get("laws", []))
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/convert")
async def convert_pdf(
    file: UploadFile = File(...),
    metadata_options: str = Form("{}"),
    background_tasks: BackgroundTasks = None
):
    """轉換PDF文檔"""
    try:
        # 使用 main.py 中的結構化解析
        from .main import convert_pdf_structured, MetadataOptions
        
        # 解析元數據選項
        try:
            metadata_config = json.loads(metadata_options)
            options = MetadataOptions(**metadata_config)
        except:
            options = MetadataOptions()
        
        # 驗證文件類型
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支持PDF文件格式")
        
        # 讀取文件內容
        file_content = await file.read()
        
        # 轉換PDF為結構化格式
        result = convert_pdf_structured(file_content, file.filename, options)
        
        # 檢查轉換是否成功
        if not result.get("success", False):
            error_msg = result.get("error", "PDF轉換失敗")
            print(f"PDF轉換失敗: {error_msg}")
            raise HTTPException(status_code=400, detail=f"PDF轉換失敗: {error_msg}")
        
        # 檢查提取的文本是否為空
        extracted_text = result.get("text", "")
        if not extracted_text or not extracted_text.strip():
            error_msg = "PDF轉換成功但沒有提取到文本內容，可能是掃描版PDF或文本被加密"
            print(f"文本提取為空: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        print(f"PDF轉換成功，提取文本長度: {len(extracted_text)} 字符")
        
        # 生成文檔ID
        doc_id = str(uuid.uuid4())
        
        # 正規化 metadata（補齊評測最小必要鍵，不受前端勾選影響）
        try:
            normalized_metadata = normalize_corpus_metadata(result["metadata"]) if isinstance(result.get("metadata"), dict) else result.get("metadata")
        except Exception:
            normalized_metadata = result.get("metadata")

        # 創建文檔記錄
        doc_record = DocRecord(
            id=doc_id,
            filename=file.filename,
            text=extracted_text,
            chunks=[],
            chunk_size=0,
            overlap=0,
            json_data=normalized_metadata
        )
        
        # 存儲文檔
        store.add_doc(doc_record)
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "text_length": len(result["text"]),
            "metadata": normalized_metadata,
            "processing_time": result["processing_time"]
        }
        
    except HTTPException:
        # 重新拋出HTTPException，不要轉換為500錯誤
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 分塊路由
@router.post("/chunk")
async def chunk_document(request: ChunkConfig):
    """分塊文檔"""
    try:
        # 使用請求中的doc_id獲取文檔
        doc = store.get_doc(request.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"文檔 {request.doc_id} 不存在")
        
        # 添加調試信息
        print(f"文檔ID: {request.doc_id}")
        print(f"文檔文件名: {doc.filename}")
        print(f"文檔文本長度: {len(doc.text) if doc.text else 0}")
        print(f"文檔是否有JSON數據: {bool(doc.json_data)}")
        
        # 檢查文檔是否有文本內容
        if not doc.text or not doc.text.strip():
            # 提供更詳細的錯誤信息
            error_detail = f"文檔沒有文本內容，無法進行分塊。文檔ID: {request.doc_id}, 文件名: {doc.filename}"
            if doc.json_data:
                error_detail += f", JSON數據存在: {bool(doc.json_data)}"
            print(f"分塊錯誤: {error_detail}")
            raise HTTPException(status_code=400, detail=error_detail)
        
        # 準備分塊參數
        chunk_kwargs = {
            "chunk_size": request.chunk_size if request.strategy == "fixed_size" else request.chunk_size,
            "max_chunk_size": request.chunk_size,
            "overlap_ratio": request.overlap_ratio
        }
        
        # 如果是結構化層次分割，添加額外參數並驗證 JSON 結構
        if request.strategy == "structured_hierarchical":
            chunk_kwargs["chunk_by"] = request.chunk_by
            # 強化驗證：json_data 必須存在且為合法結構（包含 laws 或 chapters）
            if not doc.json_data or not isinstance(doc.json_data, dict) or not (
                (doc.json_data.get("laws") or []) or (doc.json_data.get("chapters") or [])
            ):
                return JSONResponse(status_code=400, content={
                    "error": "多層級結構化分割需要合法的 JSON（需包含 laws 陣列或 chapters 陣列）。請於上傳頁保存正確的 JSON 後再試。"
                })
        
        # 若為結構化層次分割但未提供已保存的 JSON，明確拒絕，避免回退到純文字導致層級遺失
        if request.strategy == "structured_hierarchical" and not doc.json_data:
            return JSONResponse(status_code=400, content={
                "error": "請先於上傳頁保存結構化JSON（update-json），再執行多層級結構化分割。當前文檔未檢測到 json_data。"
            })

        # 生成分塊 - 使用chunk_with_span來獲取結構化信息
        from .chunking import chunk_text_with_span

        # 當存在 json_data 時，對非 structured_hierarchical 策略也改用由 JSON 構建的文字來源
        effective_text = doc.text
        if doc.json_data and request.strategy != "structured_hierarchical":
            try:
                effective_text = build_text_from_json(doc.json_data)
            except Exception:
                # 保底：若構建失敗，仍回退到 doc.text
                effective_text = doc.text
        
        chunks_with_span = chunk_text_with_span(
            effective_text,
            strategy=request.strategy,
            json_data=doc.json_data,
            **chunk_kwargs
        )
        
        # 提取純文本chunks
        chunks = [chunk["content"] for chunk in chunks_with_span]
        
        # 更新文檔記錄
        doc.chunks = chunks
        doc.chunk_size = request.chunk_size
        doc.overlap = int(request.chunk_size * request.overlap_ratio)
        doc.structured_chunks = chunks_with_span  # 保存結構化分塊信息
        doc.chunking_strategy = request.strategy  # 保存分塊策略
        store.add_doc(doc)
        
        # 重置嵌入
        store.reset_embeddings()
        
        # 計算chunk統計信息
        chunk_lengths = [len(chunk) for chunk in chunks] if chunks else []
        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        min_length = min(chunk_lengths) if chunk_lengths else 0
        max_length = max(chunk_lengths) if chunk_lengths else 0
        
        # 計算長度方差
        if chunk_lengths:
            variance = sum((length - avg_chunk_length) ** 2 for length in chunk_lengths) / len(chunk_lengths)
        else:
            variance = 0
        
        # 計算重疊率
        overlap_rate = request.overlap_ratio if hasattr(request, 'overlap_ratio') else 0
        
        # 準備返回的chunks（前幾個作為sample）
        sample_chunks = chunks[:3] if chunks else []
        
        return {
            "chunks": chunks,
            "num_chunks": len(chunks),
            "chunk_count": len(chunks),
            "strategy": request.strategy,
            "chunk_size": request.chunk_size,
            "overlap_ratio": request.overlap_ratio,
            "chunk_by": request.chunk_by if request.strategy == "structured_hierarchical" else None,
            "avg_chunk_length": avg_chunk_length,
            "chunk_lengths": chunk_lengths,
            # 前端期望的數據結構
            "metrics": {
                "avg_length": avg_chunk_length,
                "length_variance": variance,
                "overlap_rate": overlap_rate,
                "min_length": min_length,
                "max_length": max_length,
            },
            "sample": sample_chunks,
            "all_chunks": chunks,
            "overlap": int(request.chunk_size * request.overlap_ratio)
        }
        
    except HTTPException:
        # 重新拋出HTTPException，不要轉換為500錯誤
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 問題生成路由
@router.post("/generate-questions")
async def generate_questions_endpoint(request: GenerateQuestionsRequest):
    """生成問題"""
    try:
        # 獲取文檔
        doc = store.get_doc(request.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文檔不存在")
        
        # 獲取main.py中的函數
        from .main import generate_questions_with_gemini
        
        questions = generate_questions_with_gemini(
            doc.text,
            request.num_questions,
            request.question_types,
            request.difficulty_levels
        )
        
        # 提取問題文本並存儲到文檔記錄中
        question_texts = [q.question for q in questions]
        doc.generated_questions = question_texts
        store.add_doc(doc)
        
        return {
            "success": True,
            "result": {
                "doc_id": request.doc_id,
                "questions": [
                    {
                        "question": q.question,
                        "references": q.references,
                        "question_type": q.question_type,
                        "difficulty": q.difficulty,
                        "keywords": q.keywords,
                        "estimated_tokens": q.estimated_tokens
                    }
                    for q in questions
                ],
                "total_generated": len(questions)
            }
        }
        
    except HTTPException:
        # 重新拋出HTTPException，不要轉換為500錯誤
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 固定大小評估路由
@router.post("/evaluate/fixed-size")
async def start_fixed_size_evaluation(req: FixedSizeEvaluationRequest, background_tasks: BackgroundTasks):
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


@router.get("/evaluate/status/{task_id}")
async def get_evaluation_status(task_id: str):
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
        "progress": task.progress  # 使用任務對象中的progress字段
    }


@router.get("/evaluate/results/{task_id}")
async def get_evaluation_results(task_id: str):
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
    
    # 計算摘要統計，處理空結果的情況
    summary = {
        "total_configs": len(results),
        "best_precision_omega": 0,
        "best_precision_at_5": 0,
        "best_recall_at_5": 0,
        "avg_chunk_count": 0,
        "avg_chunk_length": 0
    }
    
    if results:
        summary.update({
            "best_precision_omega": max(r["metrics"]["precision_omega"] for r in results),
            "best_precision_at_5": max(r["metrics"]["precision_at_k"].get(5, 0) for r in results),
            "best_recall_at_5": max(r["metrics"]["recall_at_k"].get(5, 0) for r in results),
            "avg_chunk_count": sum(r["metrics"]["chunk_count"] for r in results) / len(results),
            "avg_chunk_length": sum(r["metrics"]["avg_chunk_length"] for r in results) / len(results)
        })
    
    return {
        "task_id": task_id,
        "status": task.status,
        "results": results,
        "summary": summary
    }


@router.get("/evaluate/comparison/{task_id}")
async def get_evaluation_comparison(task_id: str):
    """
    獲取評測對比分析
    """
    task = eval_store.get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    
    if task.status != "completed":
        return JSONResponse(status_code=400, content={"error": "Task not completed"})
    
    # 分析結果
    chunk_size_analysis = {}
    overlap_analysis = {}
    strategy_specific_analysis = {}
    
    for result in task.results:
        config = result.config
        metrics = result.metrics
        
        # 分析chunk size
        chunk_size = config.get("chunk_size", 0)
        if chunk_size not in chunk_size_analysis:
            chunk_size_analysis[chunk_size] = {
                "precision_at_k": {},
                "recall_at_k": {},
                "precision_omega": 0,
                "count": 0
            }
        
        chunk_size_analysis[chunk_size]["count"] += 1
        chunk_size_analysis[chunk_size]["precision_omega"] = max(
            chunk_size_analysis[chunk_size]["precision_omega"],
            metrics.precision_omega
        )
        
        for k, precision in metrics.precision_at_k.items():
            if k not in chunk_size_analysis[chunk_size]["precision_at_k"]:
                chunk_size_analysis[chunk_size]["precision_at_k"][k] = 0
            chunk_size_analysis[chunk_size]["precision_at_k"][k] = max(
                chunk_size_analysis[chunk_size]["precision_at_k"][k],
                precision
            )
        
        for k, recall in metrics.recall_at_k.items():
            if k not in chunk_size_analysis[chunk_size]["recall_at_k"]:
                chunk_size_analysis[chunk_size]["recall_at_k"][k] = 0
            chunk_size_analysis[chunk_size]["recall_at_k"][k] = max(
                chunk_size_analysis[chunk_size]["recall_at_k"][k],
                recall
            )
        
        # 分析overlap ratio
        overlap_ratio = config.get("overlap_ratio", 0)
        if overlap_ratio not in overlap_analysis:
            overlap_analysis[overlap_ratio] = {
                "precision_at_k": {},
                "recall_at_k": {},
                "precision_omega": 0,
                "count": 0
            }
        
        overlap_analysis[overlap_ratio]["count"] += 1
        overlap_analysis[overlap_ratio]["precision_omega"] = max(
            overlap_analysis[overlap_ratio]["precision_omega"],
            metrics.precision_omega
        )
        
        for k, precision in metrics.precision_at_k.items():
            if k not in overlap_analysis[overlap_ratio]["precision_at_k"]:
                overlap_analysis[overlap_ratio]["precision_at_k"][k] = 0
            overlap_analysis[overlap_ratio]["precision_at_k"][k] = max(
                overlap_analysis[overlap_ratio]["precision_at_k"][k],
                precision
            )
        
        for k, recall in metrics.recall_at_k.items():
            if k not in overlap_analysis[overlap_ratio]["recall_at_k"]:
                overlap_analysis[overlap_ratio]["recall_at_k"][k] = 0
            overlap_analysis[overlap_ratio]["recall_at_k"][k] = max(
                overlap_analysis[overlap_ratio]["recall_at_k"][k],
                recall
            )
        
        # 分析策略特定參數
        if task.strategy == "structured_hierarchical":
            chunk_by = config.get("chunk_by", "article")
            param_key = f"chunk_by_{chunk_by}"
            if param_key not in strategy_specific_analysis:
                strategy_specific_analysis[param_key] = {
                    "precision_at_k": {},
                    "recall_at_k": {},
                    "precision_omega": 0,
                    "count": 0
                }
            
            strategy_specific_analysis[param_key]["count"] += 1
            strategy_specific_analysis[param_key]["precision_omega"] = max(
                strategy_specific_analysis[param_key]["precision_omega"],
                metrics.precision_omega
            )
            
            for k, precision in metrics.precision_at_k.items():
                if k not in strategy_specific_analysis[param_key]["precision_at_k"]:
                    strategy_specific_analysis[param_key]["precision_at_k"][k] = 0
                strategy_specific_analysis[param_key]["precision_at_k"][k] = max(
                    strategy_specific_analysis[param_key]["precision_at_k"][k],
                    precision
                )
            
            for k, recall in metrics.recall_at_k.items():
                if k not in strategy_specific_analysis[param_key]["recall_at_k"]:
                    strategy_specific_analysis[param_key]["recall_at_k"][k] = 0
                strategy_specific_analysis[param_key]["recall_at_k"][k] = max(
                    strategy_specific_analysis[param_key]["recall_at_k"][k],
                    recall
                )
    
    # 生成推薦
    recommendations = []
    if chunk_size_analysis:
        best_chunk_size = max(chunk_size_analysis.keys(), 
                            key=lambda x: chunk_size_analysis[x]["precision_omega"])
        recommendations.append(f"最佳分塊大小: {best_chunk_size} 字符")
    
    if overlap_analysis:
        best_overlap = max(overlap_analysis.keys(), 
                         key=lambda x: overlap_analysis[x]["precision_omega"])
        recommendations.append(f"最佳重疊比例: {best_overlap:.1%}")
    
    if strategy_specific_analysis:
        best_param = max(strategy_specific_analysis.keys(), 
                        key=lambda x: strategy_specific_analysis[x]["precision_omega"])
        recommendations.append(f"最佳策略參數: {best_param}")
    
    return {
        "task_id": task_id,
        "chunk_size_analysis": chunk_size_analysis,
        "overlap_analysis": overlap_analysis,
        "strategy_specific_analysis": strategy_specific_analysis,
        "recommendations": recommendations
    }


# 評估路由
@router.post("/evaluate")
async def evaluate_chunking(request: EvaluationRequest):
    """評估分塊策略"""
    try:
        # 獲取文檔
        doc = store.get_doc(request.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文檔不存在")
        
        # 檢查是否已有生成的問題
        if not hasattr(doc, 'generated_questions') or not doc.generated_questions:
            raise HTTPException(
                status_code=400, 
                detail="請先使用「生成問題」功能為文檔生成測試問題，然後再進行評測"
            )
        
        questions = doc.generated_questions
        
        # 創建評估任務
        task_id = str(uuid.uuid4())
        task = EvaluationTask(
            id=task_id,
            doc_id=request.doc_id,
            status="pending",
            progress=0.0,
            configs=[],
            created_at=datetime.now().isoformat()
        )
        
        # 生成配置組合
        configs = []
        for chunk_size in request.chunk_sizes:
            for overlap_ratio in request.overlap_ratios:
                configs.append({
                    "chunk_size": chunk_size,
                    "overlap_ratio": overlap_ratio
                })
        
        task.configs = configs
        store.add_evaluation_task(task)
        
        return {
            "task_id": task_id,
            "total_configs": len(configs),
            "questions_generated": len(questions)
        }
        
    except HTTPException:
        # 重新拋出HTTPException，不要轉換為500錯誤
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate/task/{task_id}")
async def get_evaluation_task(task_id: str):
    """獲取評估任務狀態"""
    task = store.get_evaluation_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    return {
        "task_id": task_id,
        "status": task.status,
        "progress": task.progress,
        "total_configs": len(task.configs),
        "completed_configs": len(task.results) if task.results else 0,
        "error_message": task.error_message
    }


@router.post("/evaluate/run/{task_id}")
async def run_evaluation(task_id: str, background_tasks: BackgroundTasks):
    """執行評估任務"""
    task = store.get_evaluation_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    if task.status != "pending":
        raise HTTPException(status_code=400, detail="任務已開始或完成")
    
    # 更新任務狀態
    store.update_evaluation_task(task_id, status="running", progress=0.0)
    
    # 在後台執行評估
    background_tasks.add_task(execute_evaluation_task, task_id)
    
    return {"message": "評估任務已開始"}


async def execute_evaluation_task(task_id: str):
    """執行評估任務的後台函數"""
    try:
        task = store.get_evaluation_task(task_id)
        if not task:
            return
        
        # 獲取文檔
        doc = store.get_doc(task.doc_id)
        if not doc:
            store.update_evaluation_task(task_id, status="failed", error_message="文檔不存在")
            return
        
        # 檢查是否已有生成的問題
        if not hasattr(doc, 'generated_questions') or not doc.generated_questions:
            store.update_evaluation_task(task_id, status="failed", error_message="請先生成問題再進行評測")
            return
        
        questions = doc.generated_questions
        results = []
        
        # 評估每個配置
        for i, config in enumerate(task.configs):
            try:
                result = evaluate_chunk_config(
                    doc.text,
                    questions,
                    config["chunk_size"],
                    config["overlap_ratio"]
                )
                
                # 轉換為字典格式
                result_dict = {
                    "config": result.config,
                    "metrics": {
                        "precision_omega": result.metrics.precision_omega,
                        "precision_at_k": result.metrics.precision_at_k,
                        "recall_at_k": result.metrics.recall_at_k,
                        "chunk_count": result.metrics.chunk_count,
                        "avg_chunk_length": result.metrics.avg_chunk_length,
                        "length_variance": result.metrics.length_variance
                    }
                }
                
                results.append(result_dict)
                
                # 更新進度
                progress = (i + 1) / len(task.configs)
                store.update_evaluation_task(task_id, progress=progress)
                
            except Exception as e:
                print(f"評估配置 {config} 時出錯: {e}")
                continue
        
        # 完成任務
        store.update_evaluation_task(
            task_id,
            status="completed",
            progress=1.0,
            results=results,
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        store.update_evaluation_task(task_id, status="failed", error_message=str(e))


@router.get("/evaluate/results/{task_id}")
async def get_evaluation_results(task_id: str):
    """獲取評估結果"""
    task = store.get_evaluation_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="任務未完成")
    
    # 計算最佳配置
    best_config = None
    best_score = -1
    
    for result in task.results:
        # 綜合評分
        score = (
            result["metrics"]["precision_omega"] * 0.4 +
            result["metrics"]["precision_at_k"].get(5, 0) * 0.3 +
            result["metrics"]["recall_at_k"].get(5, 0) * 0.3
        )
        
        if score > best_score:
            best_score = score
            best_config = result
    
    return {
        "task_id": task_id,
        "status": task.status,
        "results": task.results,
        "summary": {
            "best_config": best_config,
            "total_configs": len(task.results),
            "best_precision_omega": max(r["metrics"]["precision_omega"] for r in task.results),
            "best_precision_at_5": max(r["metrics"]["precision_at_k"].get(5, 0) for r in task.results),
            "best_recall_at_5": max(r["metrics"]["recall_at_k"].get(5, 0) for r in task.results)
        }
    }


@router.post("/convert-multiple")
async def convert_multiple_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    metadata_options: str = Form("{}")
):
    """轉換多個PDF文檔並整合成一個法律JSON"""
    try:
        # 解析元數據選項
        try:
            metadata_config = json.loads(metadata_options)
            from .main import MetadataOptions
            options = MetadataOptions(**metadata_config)
        except:
            from .main import MetadataOptions
            options = MetadataOptions()
        
        # 驗證文件類型
        for file in files:
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"文件 {file.filename} 不是PDF格式")
        
        # 創建任務ID
        task_id = str(uuid.uuid4())
        
        # 在後台處理轉換之前，先讀取所有文件內容
        file_contents = []
        for file in files:
            content = await file.read()
            file_contents.append({
                'content': content,
                'filename': file.filename or 'unknown.pdf'
            })
        
        background_tasks.add_task(process_multiple_pdf_conversion, task_id, file_contents, options)
        
        return {"task_id": task_id, "status": "processing"}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"多文件轉換失敗: {str(e)}"}
        )


@router.get("/convert-multiple/status/{task_id}")
async def get_multiple_convert_status(task_id: str):
    """獲取多文件轉換狀態"""
    try:
        if task_id not in task_status_store:
            return JSONResponse(
                status_code=404,
                content={"error": "任務不存在"}
            )
        
        status_info = task_status_store[task_id]
        return {
            "task_id": task_id,
            "status": status_info["status"],
            "progress": status_info.get("progress", 0.0),
            "result": status_info.get("result"),
            "error": status_info.get("error")
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"獲取狀態失敗: {str(e)}"}
        )


# QA Set上傳和映射相關的存儲
qa_mapping_store = {}

# 新增：QA映射請求模型
class QAMappingRequest(BaseModel):
    doc_id: str
    qa_set: List[Dict[str, Any]]
    chunking_results: List[Dict[str, Any]]  # 來自批量分塊的結果
    iou_threshold: Optional[float] = 0.5  # IoU閾值

@router.post("/upload-qa-set")
async def upload_qa_set(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    chunk_sizes: str = Form("[300, 500, 800]"),
    overlap_ratios: str = Form("[0.0, 0.1, 0.2]"),
    strategy: str = Form("fixed_size"),
    background_tasks: BackgroundTasks = None
):
    """上傳QA set並進行chunk映射"""
    try:
        # 驗證文檔是否存在
        doc = store.get_doc(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文檔不存在")
        
        # 驗證文件格式
        if not file.filename or not file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="只支持JSON格式的QA set文件")
        
        # 讀取並解析QA set文件
        file_content = await file.read()
        try:
            qa_set = json.loads(file_content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"JSON格式錯誤: {str(e)}")
        
        # 驗證QA set格式
        if not isinstance(qa_set, list):
            raise HTTPException(status_code=400, detail="QA set必須是一個列表")
        
        # 檢查是否有法條JSON數據用於轉換
        law_json_data = doc.json_data if hasattr(doc, 'json_data') and doc.json_data else None
        
        # 轉換QA set，補充缺失的span信息
        if law_json_data:
            print(f"開始轉換QA set，原始項目數: {len(qa_set)}")
            try:
                converted_qa_set, conversion_stats = convert_qa_set_with_law_data(qa_set, law_json_data)
                print(f"QA set轉換完成:")
                print(f"  - 總項目數: {conversion_stats['total_items']}")
                print(f"  - 有snippets的項目: {conversion_stats['items_with_snippets']}")
                print(f"  - 有有效span的項目: {conversion_stats['items_with_valid_spans']}")
                print(f"  - 有file_path的項目: {conversion_stats['items_with_file_path']}")
                print(f"  - snippet覆蓋率: {conversion_stats['snippet_coverage']:.2%}")
                print(f"  - span覆蓋率: {conversion_stats['span_coverage']:.2%}")
                print(f"  - file_path覆蓋率: {conversion_stats['file_path_coverage']:.2%}")
                qa_set = converted_qa_set
            except Exception as e:
                print(f"QA set轉換失敗: {e}")
                # 轉換失敗時繼續使用原始QA set
        else:
            print("沒有法條JSON數據，跳過QA set轉換")
        
        # 解析配置參數
        try:
            chunk_sizes_list = json.loads(chunk_sizes)
            overlap_ratios_list = json.loads(overlap_ratios)
        except json.JSONDecodeError:
            chunk_sizes_list = [300, 500, 800]
            overlap_ratios_list = [0.0, 0.1, 0.2]
        
        # 創建任務ID
        task_id = str(uuid.uuid4())
        
        # 準備返回的統計信息
        response_data = {
            "task_id": task_id,
            "status": "processing",
            "message": "QA set上傳成功，正在進行chunk映射...",
            "original_qa_set": qa_set
        }
        
        # 如果有轉換統計信息，添加到響應中
        if law_json_data:
            try:
                _, conversion_stats = convert_qa_set_with_law_data(qa_set, law_json_data)
                response_data["conversion_stats"] = conversion_stats
            except Exception as e:
                print(f"獲取轉換統計失敗: {e}")
        
        # 在後台處理映射
        background_tasks.add_task(
            process_qa_mapping,
            task_id,
            doc_id,
            qa_set,
            chunk_sizes_list,
            overlap_ratios_list,
            strategy
        )
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/map-qa-set")
async def map_qa_set(
    file: UploadFile = File(...),
    doc_id: str = Form(...)
):
    """直接映射QA set到法條JSON，不進行分塊處理"""
    try:
        # 驗證文檔是否存在
        doc = store.get_doc(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文檔不存在")
        
        # 驗證文件格式
        if not file.filename or not file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="只支持JSON格式的QA set文件")
        
        # 讀取並解析QA set文件
        file_content = await file.read()
        try:
            qa_set = json.loads(file_content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"JSON格式錯誤: {str(e)}")
        
        # 驗證QA set格式
        if not isinstance(qa_set, list):
            raise HTTPException(status_code=400, detail="QA set必須是一個列表")
        
        # 檢查是否有法條JSON數據用於轉換
        law_json_data = doc.json_data if hasattr(doc, 'json_data') and doc.json_data else None
        
        # 轉換QA set，補充缺失的span信息
        if law_json_data:
            print(f"開始映射QA set，原始項目數: {len(qa_set)}")
            try:
                converted_qa_set, conversion_stats = convert_qa_set_with_law_data(qa_set, law_json_data)
                print(f"QA set映射完成:")
                print(f"  - 總項目數: {conversion_stats['total_items']}")
                print(f"  - 有snippets的項目: {conversion_stats['items_with_snippets']}")
                print(f"  - 有有效span的項目: {conversion_stats['items_with_valid_spans']}")
                print(f"  - snippet覆蓋率: {conversion_stats['snippet_coverage']:.2%}")
                print(f"  - span覆蓋率: {conversion_stats['span_coverage']:.2%}")
                
                # 只返回標準格式的QA Set，不包含額外的metadata
                return converted_qa_set
            except Exception as e:
                print(f"QA set映射失敗: {e}")
                raise HTTPException(status_code=500, detail=f"QA set映射失敗: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="沒有法條JSON數據，無法進行映射")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qa-mapping/status/{task_id}")
async def get_qa_mapping_status(task_id: str):
    """獲取QA映射任務狀態"""
    if task_id not in qa_mapping_store:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    status_info = qa_mapping_store[task_id]
    return {
        "task_id": task_id,
        "status": status_info["status"],
        "progress": status_info.get("progress", 0.0),
        "result": status_info.get("result"),
        "error": status_info.get("error")
    }


@router.get("/qa-mapping/result/{task_id}")
async def get_qa_mapping_result(task_id: str):
    """獲取QA映射結果"""
    if task_id not in qa_mapping_store:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    status_info = qa_mapping_store[task_id]
    if status_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="任務未完成")
    
    return status_info["result"]


# 調試端點：測試QA映射請求
@router.post("/qa-mapping/debug")
async def debug_qa_mapping(req: dict):
    """調試QA映射請求"""
    try:
        print("調試QA映射請求:")
        print(f"- 請求類型: {type(req)}")
        print(f"- 請求鍵: {list(req.keys()) if isinstance(req, dict) else 'Not a dict'}")
        
        if isinstance(req, dict):
            print(f"- doc_id: {req.get('doc_id')}")
            print(f"- qa_set類型: {type(req.get('qa_set'))}")
            print(f"- qa_set長度: {len(req.get('qa_set', []))}")
            print(f"- chunking_results類型: {type(req.get('chunking_results'))}")
            print(f"- chunking_results長度: {len(req.get('chunking_results', []))}")
            print(f"- iou_threshold: {req.get('iou_threshold')}")
            
            # 檢查chunking_results結構
            chunking_results = req.get('chunking_results', [])
            if chunking_results:
                print(f"- 第一個chunking_result鍵: {list(chunking_results[0].keys()) if isinstance(chunking_results[0], dict) else 'Not a dict'}")
                if isinstance(chunking_results[0], dict):
                    print(f"- 是否有chunks_with_span: {'chunks_with_span' in chunking_results[0]}")
                    if 'chunks_with_span' in chunking_results[0]:
                        print(f"- chunks_with_span長度: {len(chunking_results[0]['chunks_with_span'])}")
        
        return {"status": "debug_success", "message": "調試信息已輸出到控制台"}
        
    except Exception as e:
        print(f"調試端點錯誤: {e}")
        return {"status": "debug_error", "error": str(e)}


# 新增：直接進行QA映射的API端點
@router.post("/qa-mapping/map")
async def map_qa_to_chunks(req: QAMappingRequest, background_tasks: BackgroundTasks):
    """
    將QA set映射到分塊結果
    
    這個API接收QA set和分塊結果，進行映射並返回結果
    """
    try:
        # 添加調試信息
        print(f"QA映射請求 - doc_id: {req.doc_id}")
        print(f"QA映射請求 - qa_set長度: {len(req.qa_set) if req.qa_set else 0}")
        print(f"QA映射請求 - chunking_results長度: {len(req.chunking_results) if req.chunking_results else 0}")
        print(f"QA映射請求 - iou_threshold: {req.iou_threshold}")
        
        # 驗證輸入數據
        if not req.qa_set:
            raise HTTPException(status_code=400, detail="QA set不能為空")
        
        if not req.chunking_results:
            raise HTTPException(status_code=400, detail="分塊結果不能為空")
        
        # 驗證文檔是否存在
        doc = store.get_doc(req.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文檔不存在")
        
        # 創建任務ID
        task_id = str(uuid.uuid4())
        
        # 在後台處理映射
        background_tasks.add_task(
            process_qa_mapping_with_chunks,
            task_id,
            req.doc_id,
            req.qa_set,
            req.chunking_results,
            req.iou_threshold
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "QA映射任務已開始"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"QA映射API錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"QA映射失敗: {str(e)}")


async def process_qa_mapping_with_chunks(
    task_id: str,
    doc_id: str,
    qa_set: List[Dict],
    chunking_results: List[Dict[str, Any]],
    iou_threshold: float
):
    """處理QA映射的後台任務（使用已生成的分塊結果）"""
    try:
        # 初始化任務狀態
        qa_mapping_store[task_id] = {
            "status": "processing",
            "progress": 0.0
        }
        
        # 獲取文檔
        doc = store.get_doc(doc_id)
        if not doc:
            qa_mapping_store[task_id] = {
                "status": "failed",
                "error": "文檔不存在"
            }
            return
        
        total_configs = len(chunking_results)
        mapping_results = {}
        
        # 為每個分塊配置進行映射
        for i, chunking_result in enumerate(chunking_results):
            try:
                # 獲取帶span信息的chunks
                chunks_with_span = chunking_result.get('chunks_with_span', [])
                
                if not chunks_with_span:
                    print(f"警告: 配置 {i} 沒有chunks_with_span信息，跳過")
                    continue
                
                # 映射spans到chunks
                mapped_qa_set = map_spans_to_chunks(
                    qa_set,
                    chunks_with_span,
                    iou_threshold,
                    0.3  # overlap_threshold (降低到0.3)
                )
                
                # 存儲映射結果
                config_id = f"config_{i+1:03d}_{chunking_result['strategy']}_{chunking_result['config']['chunk_size']}_{chunking_result['config']['overlap_ratio']}"
                mapping_results[config_id] = {
                    "config": chunking_result['config'],
                    "strategy": chunking_result['strategy'],
                    "chunks_with_span": chunks_with_span,
                    "mapped_qa_set": mapped_qa_set,
                    "chunk_count": chunking_result['chunk_count'],
                    "mapping_stats": _calculate_mapping_stats(mapped_qa_set)
                }
                
                # 更新進度
                progress = (i + 1) / total_configs
                qa_mapping_store[task_id]["progress"] = progress
                
            except Exception as e:
                print(f"配置 {i} 映射失敗: {e}")
                continue
        
        # 完成任務
        qa_mapping_store[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "result": {
                "task_id": task_id,
                "doc_id": doc_id,
                "original_qa_set": qa_set,
                "chunking_results": chunking_results,
                "mapping_results": mapping_results,
                "total_configs": len(chunking_results),
                "iou_threshold": iou_threshold,
                "mapping_summary": _generate_mapping_summary(mapping_results)
            }
        }
        
    except Exception as e:
        qa_mapping_store[task_id] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"QA映射失敗: {e}")


def evaluate_chunk_config_with_mapped_qa(text: str, mapped_qa_set: List[Dict], chunk_size: int, 
                                       overlap_ratio: float, strategy: str = "fixed_size", **kwargs) -> EvaluationResult:
    """
    使用映射的QA Set進行分塊配置評估
    """
    from .main import calculate_precision_at_k, calculate_recall_at_k, calculate_precision_omega
    
    # 生成分塊
    from .chunking import chunk_text
    chunks = chunk_text(text, strategy=strategy, chunk_size=chunk_size, overlap_ratio=overlap_ratio, **kwargs)
    
    # 準備評估數據
    k_values = [1, 3, 5, 10]
    precision_at_k_scores = {k: [] for k in k_values}
    recall_at_k_scores = {k: [] for k in k_values}
    precision_omega_scores = []
    
    # 只評估正例問題
    positive_qa_items = [item for item in mapped_qa_set if item.get('label', '').lower() == 'yes']
    
    if not positive_qa_items:
        # 如果沒有正例問題，返回零分
        metrics = EvaluationMetrics(
            precision_omega=0.0,
            precision_at_k={k: 0.0 for k in k_values},
            recall_at_k={k: 0.0 for k in k_values},
            chunk_count=len(chunks),
            avg_chunk_length=sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            length_variance=0.0
        )
        return EvaluationResult(
            config={"chunk_size": chunk_size, "overlap_ratio": overlap_ratio, "strategy": strategy, **kwargs},
            metrics=metrics,
            test_queries=[],
            retrieval_results={},
            timestamp=datetime.now()
        )
    
    # 為每個正例問題計算指標
    for qa_item in positive_qa_items:
        query = qa_item.get('query', '')
        ground_truth_chunks = qa_item.get('relevant_chunks', [])
        
        if not ground_truth_chunks:
            # 如果沒有ground truth chunks，跳過
            continue
        
        # 使用簡單的關鍵詞匹配來模擬檢索
        # 這裡可以改進為更複雜的檢索邏輯
        retrieved_chunks = []
        query_keywords = set(query.split())
        
        for i, chunk in enumerate(chunks):
            chunk_keywords = set(chunk.split())
            # 計算關鍵詞重疊度
            overlap = len(query_keywords & chunk_keywords)
            if overlap > 0:
                retrieved_chunks.append((i, chunk, overlap))
        
        # 按重疊度排序
        retrieved_chunks.sort(key=lambda x: x[2], reverse=True)
        retrieved_chunk_ids = [f"chunk_{i:03d}" for i, _, _ in retrieved_chunks]
        
        # 計算各種指標
        for k in k_values:
            precision = calculate_precision_at_k(retrieved_chunk_ids, query, k)
            recall = calculate_recall_at_k(retrieved_chunk_ids, query, k, ground_truth_chunks)
            precision_at_k_scores[k].append(precision)
            recall_at_k_scores[k].append(recall)
        
        precision_omega = calculate_precision_omega(retrieved_chunk_ids, query)
        precision_omega_scores.append(precision_omega)
    
    # 計算平均指標
    avg_precision_at_k = {k: sum(scores) / len(scores) if scores else 0.0 for k, scores in precision_at_k_scores.items()}
    avg_recall_at_k = {k: sum(scores) / len(scores) if scores else 0.0 for k, scores in recall_at_k_scores.items()}
    avg_precision_omega = sum(precision_omega_scores) / len(precision_omega_scores) if precision_omega_scores else 0.0
    
    metrics = EvaluationMetrics(
        precision_omega=avg_precision_omega,
        precision_at_k=avg_precision_at_k,
        recall_at_k=avg_recall_at_k,
        chunk_count=len(chunks),
        avg_chunk_length=sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
        length_variance=0.0  # 簡化計算
    )
    
    return EvaluationResult(
        config={"chunk_size": chunk_size, "overlap_ratio": overlap_ratio, "strategy": strategy, **kwargs},
        metrics=metrics,
        test_queries=[item.get('query', '') for item in positive_qa_items],
        retrieval_results={},
        timestamp=datetime.now()
    )


def _calculate_mapping_stats(mapped_qa_set: List[Dict]) -> Dict[str, Any]:
    """計算映射統計信息"""
    total_questions = len(mapped_qa_set)
    questions_with_chunks = sum(1 for item in mapped_qa_set if item.get('relevant_chunks'))
    positive_questions = sum(1 for item in mapped_qa_set if item.get('label', '').lower() == 'yes')
    negative_questions = total_questions - positive_questions
    
    # 計算映射成功率：有映射的正例問題數 / 總正例問題數
    positive_questions_with_chunks = sum(1 for item in mapped_qa_set 
                                       if item.get('label', '').lower() == 'yes' and item.get('relevant_chunks'))
    mapping_success_rate = (positive_questions_with_chunks / positive_questions * 100) if positive_questions > 0 else 0
    
    return {
        "total_questions": total_questions,
        "questions_with_chunks": questions_with_chunks,
        "mapping_coverage": questions_with_chunks / total_questions if total_questions > 0 else 0,
        "mapping_success_rate": mapping_success_rate,  # 新增：映射成功率（百分比）
        "positive_questions": positive_questions,
        "negative_questions": negative_questions,
        "avg_chunks_per_question": sum(len(item.get('relevant_chunks', [])) for item in mapped_qa_set) / total_questions if total_questions > 0 else 0
    }


def _generate_mapping_summary(mapping_results: Dict[str, Dict]) -> Dict[str, Any]:
    """生成映射摘要"""
    if not mapping_results:
        return {}
    
    best_coverage = 0
    best_config = None
    
    for config_id, result in mapping_results.items():
        coverage = result['mapping_stats']['mapping_coverage']
        if coverage > best_coverage:
            best_coverage = coverage
            best_config = config_id
    
    return {
        "best_config": best_config,
        "best_coverage": best_coverage,
        "total_configs": len(mapping_results),
        "avg_coverage": sum(result['mapping_stats']['mapping_coverage'] for result in mapping_results.values()) / len(mapping_results)
    }


async def process_qa_mapping(
    task_id: str,
    doc_id: str,
    qa_set: List[Dict],
    chunk_sizes: List[int],
    overlap_ratios: List[float],
    strategy: str
):
    """處理QA映射的後台任務"""
    try:
        # 初始化任務狀態
        qa_mapping_store[task_id] = {
            "status": "processing",
            "progress": 0.0
        }
        
        # 獲取文檔
        doc = store.get_doc(doc_id)
        if not doc:
            qa_mapping_store[task_id] = {
                "status": "failed",
                "error": "文檔不存在"
            }
            return
        
        # 生成所有配置組合
        configs = []
        for chunk_size in chunk_sizes:
            for overlap_ratio in overlap_ratios:
                overlap = int(chunk_size * overlap_ratio)
                configs.append({
                    "chunk_size": chunk_size,
                    "overlap_ratio": overlap_ratio,
                    "overlap": overlap,
                    "strategy": strategy
                })
        
        total_configs = len(configs)
        mapping_results = {}
        
        # 為每個配置進行映射
        for i, config in enumerate(configs):
            try:
                # 生成分塊
                chunks = chunk_text(
                    doc.text,
                    strategy=strategy,
                    chunk_size=config["chunk_size"],
                    overlap_ratio=config["overlap_ratio"]
                )
                
                # 映射spans到chunks
                mapped_qa_set = map_spans_to_chunks(
                    qa_set,
                    chunks,
                    config["chunk_size"],
                    config["overlap"],
                    strategy
                )
                
                # 存儲映射結果
                config_id = f"config_{i+1:03d}_{config['chunk_size']}_{config['overlap_ratio']}"
                mapping_results[config_id] = {
                    "config": config,
                    "chunks": chunks,
                    "mapped_qa_set": mapped_qa_set,
                    "chunk_count": len(chunks)
                }
                
                # 更新進度
                progress = (i + 1) / total_configs
                qa_mapping_store[task_id]["progress"] = progress
                
            except Exception as e:
                print(f"配置 {config} 映射失敗: {e}")
                continue
        
        # 完成任務
        qa_mapping_store[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "result": {
                "task_id": task_id,
                "doc_id": doc_id,
                "original_qa_set": qa_set,
                "configs": configs,
                "mapping_results": mapping_results,
                "total_configs": len(configs)
            }
        }
        
    except Exception as e:
        qa_mapping_store[task_id] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"QA映射失敗: {e}")


async def process_multiple_chunking(
    task_id: str,
    doc_id: str,
    strategies: List[str],
    chunk_sizes: List[int],
    overlap_ratios: List[float]
):
    """處理批量分塊的後台任務"""
    try:
        # 初始化任務狀態
        chunking_task_store[task_id] = {
            "status": "processing",
            "progress": 0.0,
            "results": []
        }
        
        # 獲取文檔
        doc = store.get_doc(doc_id)
        if not doc:
            chunking_task_store[task_id] = {
                "status": "failed",
                "error": "文檔不存在"
            }
            return
        
        # 生成所有配置組合
        total_combinations = len(strategies) * len(chunk_sizes) * len(overlap_ratios)
        completed_combinations = 0
        results = []
        
        for strategy in strategies:
            for chunk_size in chunk_sizes:
                for overlap_ratio in overlap_ratios:
                    try:
                        # 準備分塊參數
                        chunk_kwargs = {
                            "chunk_size": chunk_size,
                            "max_chunk_size": chunk_size,
                            "overlap_ratio": overlap_ratio
                        }
                        
                        # 根據策略添加特定參數
                        if strategy == "structured_hierarchical":
                            chunk_kwargs["chunk_by"] = "article"
                        elif strategy == "rcts_hierarchical":
                            chunk_kwargs["preserve_structure"] = True
                        elif strategy == "hierarchical":
                            chunk_kwargs["level_depth"] = 3
                            chunk_kwargs["min_chunk_size"] = 200
                        elif strategy == "semantic":
                            chunk_kwargs["similarity_threshold"] = 0.6
                            chunk_kwargs["context_window"] = 100
                        elif strategy == "sliding_window":
                            chunk_kwargs["window_size"] = chunk_size
                            chunk_kwargs["step_size"] = int(chunk_size * 0.5)
                            chunk_kwargs["boundary_aware"] = True
                            chunk_kwargs["preserve_sentences"] = True
                            chunk_kwargs["min_chunk_size_sw"] = 100
                            chunk_kwargs["max_chunk_size_sw"] = chunk_size * 2
                        
                        # 若為結構化層次分割但未提供已保存的 JSON，明確跳過，避免回退到純文字
                        if strategy == "structured_hierarchical":
                            if not doc.json_data or not isinstance(doc.json_data, dict) or not (
                                (doc.json_data.get("laws") or []) or (doc.json_data.get("chapters") or [])
                            ):
                                raise Exception("structured_hierarchical 需要合法的 JSON（包含 laws 或 chapters）。")

                        # 生成分塊（帶span信息）
                        from .chunking import chunk_text_with_span
                        effective_text = doc.text
                        if doc.json_data and strategy != "structured_hierarchical":
                            try:
                                effective_text = build_text_from_json(doc.json_data)
                            except Exception:
                                effective_text = doc.text
                        chunks_with_span = chunk_text_with_span(
                            effective_text,
                            strategy=strategy,
                            json_data=doc.json_data,
                            **chunk_kwargs
                        )
                        
                        # 提取chunks文本（向後兼容）
                        chunks = [chunk_info['content'] for chunk_info in chunks_with_span]
                        
                        # 計算統計信息
                        chunk_lengths = [len(chunk) for chunk in chunks] if chunks else []
                        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
                        min_length = min(chunk_lengths) if chunk_lengths else 0
                        max_length = max(chunk_lengths) if chunk_lengths else 0
                        
                        # 計算長度方差
                        if chunk_lengths:
                            variance = sum((length - avg_chunk_length) ** 2 for length in chunk_lengths) / len(chunk_lengths)
                        else:
                            variance = 0
                        
                        # 創建結果
                        result = {
                            "strategy": strategy,
                            "config": {
                                "chunk_size": chunk_size,
                                "overlap_ratio": overlap_ratio,
                                "strategy": strategy,
                                **chunk_kwargs
                            },
                            "chunks": chunks,
                            "all_chunks": chunks,
                            "chunks_with_span": chunks_with_span,  # 新增：帶span信息的chunks
                            "chunk_count": len(chunks),
                            "metrics": {
                                "avg_length": avg_chunk_length,
                                "length_variance": variance,
                                "overlap_rate": overlap_ratio,
                                "min_length": min_length,
                                "max_length": max_length,
                            },
                            "sample_chunks": chunks[:3] if chunks else [],
                            "timestamp": datetime.now().isoformat(),
                        }
                        
                        results.append(result)
                        
                        # 更新進度
                        completed_combinations += 1
                        progress = completed_combinations / total_combinations
                        chunking_task_store[task_id]["progress"] = progress
                        
                    except Exception as e:
                        print(f"分塊組合 {strategy}-{chunk_size}-{overlap_ratio} 失敗: {e}")
                        continue
        
        # 完成任務
        chunking_task_store[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "results": results
        }
        
        # 將第一個結果同步到文檔中，以便後續的 embed 操作
        if results:
            # 選擇第一個結果作為主要分塊結果
            primary_result = results[0]
            doc = store.get_doc(doc_id)
            if doc:
                # 更新文檔的分塊信息
                doc.chunks = primary_result.get("chunks", [])
                doc.chunk_size = primary_result.get("config", {}).get("chunk_size", 500)
                doc.overlap = int(doc.chunk_size * primary_result.get("config", {}).get("overlap_ratio", 0.1))
                
                # 如果有結構化chunks，也保存
                if "chunks_with_span" in primary_result:
                    doc.structured_chunks = primary_result["chunks_with_span"]
                
                # 保存文檔
                store.add_doc(doc)
                print(f"已將批量分塊結果同步到文檔 {doc_id}，分塊數量: {len(doc.chunks)}")
        
    except Exception as e:
        chunking_task_store[task_id] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"批量分塊失敗: {e}")


async def process_strategy_evaluation(
    task_id: str,
    doc_id: str,
    chunking_results: List[Dict[str, Any]],
    qa_mapping_result: Dict[str, Any],
    test_queries: List[str],
    k_values: List[int]
):
    """處理策略評估的後台任務"""
    try:
        # 初始化任務狀態
        eval_store.update_task_status(task_id, "running", progress=0.0)
        
        # 獲取文檔
        doc = store.get_doc(doc_id)
        if not doc:
            eval_store.update_task_status(task_id, "failed", error_message="文檔不存在")
            return
        
        results = []
        total_configs = len(chunking_results)
        
        for i, chunking_result in enumerate(chunking_results):
            try:
                # 獲取對應的映射結果
                config_id = f"config_{i+1:03d}_{chunking_result['strategy']}_{chunking_result['config']['chunk_size']}_{chunking_result['config']['overlap_ratio']}"
                mapped_qa_set = qa_mapping_result.get("mapping_results", {}).get(config_id, {}).get("mapped_qa_set", [])
                
                # 使用映射的QA Set進行評估
                result = evaluate_chunk_config_with_mapped_qa(
                    doc.text,
                    mapped_qa_set,
                    chunking_result["config"]["chunk_size"],
                    chunking_result["config"]["overlap_ratio"],
                    strategy=chunking_result["strategy"],
                    **{k: v for k, v in chunking_result["config"].items() if k not in ["chunk_size", "overlap_ratio", "strategy"]}
                )
                
                # 添加分塊結果信息
                result.config.update({
                    "strategy": chunking_result["strategy"],
                    "chunk_count": chunking_result["chunk_count"],
                    "avg_chunk_length": chunking_result["metrics"]["avg_length"]
                })
                
                results.append(result)
                
                # 更新進度
                progress = (i + 1) / total_configs
                eval_store.update_task_status(task_id, "running", progress=progress)
                
            except Exception as e:
                print(f"評估分塊結果 {i} 時出錯: {e}")
                continue
        
        # 完成任務
        eval_store.update_task_status(task_id, "completed", results=results)
        
    except Exception as e:
        eval_store.update_task_status(task_id, "failed", error_message=str(e))
        print(f"策略評估失敗: {e}")


async def process_multiple_pdf_conversion(task_id: str, file_contents: List[Dict], options):
    """處理多個PDF轉換的後台任務"""
    try:
        # 初始化任務狀態
        task_status_store[task_id] = {
            "status": "processing",
            "progress": 0.0
        }
        
        from .main import convert_pdf_structured, merge_law_documents
        
        # 轉換每個PDF文件
        law_documents = []
        all_texts = []  # 收集所有文本內容
        total_files = len(file_contents)
        
        for i, file_info in enumerate(file_contents):
            # 更新進度
            progress = (i / total_files) * 0.8  # 80%用於轉換
            task_status_store[task_id]["progress"] = progress
            
            # 使用預先讀取的文件內容
            file_content = file_info['content']
            filename = file_info['filename']
            conversion_result = convert_pdf_structured(file_content, filename, options)
            
            # 檢查轉換是否成功
            if not conversion_result.get("success", False):
                print(f"PDF轉換失敗: {filename}")
                continue
            
            # 檢查文本內容
            extracted_text = conversion_result.get("text", "")
            if not extracted_text or not extracted_text.strip():
                print(f"PDF轉換成功但沒有提取到文本內容: {filename}")
                continue
                
            # 收集文本內容和metadata
            all_texts.append(extracted_text)
            law_doc = conversion_result["metadata"]
            law_documents.append(law_doc)
        
        # 更新進度到90%
        task_status_store[task_id]["progress"] = 0.9
        
        # 整合多個法律文檔
        merged_doc = merge_law_documents(law_documents)
        # 正規化合併後的 metadata（補齊必要鍵）
        merged_doc = normalize_corpus_metadata(merged_doc)
        
        # 合併所有文本內容
        if all_texts:
            merged_text = "\n\n" + "="*80 + "\n\n".join(all_texts)
        else:
            # 如果沒有文本內容，從JSON結構生成
            merged_text = generate_text_from_merged_doc(merged_doc)
        
        print(f"多文件合併完成，生成文本長度: {len(merged_text)} 字符")
        
        # 創建文檔記錄
        doc_id = str(uuid.uuid4())
        doc_record = DocRecord(
            id=doc_id,
            filename=f"merged_{len(file_contents)}_laws",
            text=merged_text,  # 合併後的文本
            chunks=[],  # 將在後續步驟中生成
            chunk_size=0,
            overlap=0,
            json_data=merged_doc,
            structured_chunks=None,
            generated_questions=None
        )
        
        # 存儲文檔記錄
        store.docs[doc_id] = doc_record
        
        # 更新任務狀態為完成
        task_status_store[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "result": {
                "doc_id": doc_id,
                "metadata": merged_doc
            }
        }
        
    except Exception as e:
        # 更新任務狀態為失敗
        task_status_store[task_id] = {
            "status": "failed",
            "progress": 0.0,
            "error": str(e)
        }
        print(f"多文件轉換失敗: {e}")


# 批量分塊路由
@router.post("/chunk/multiple")
async def start_multiple_chunking(req: MultipleChunkingRequest, background_tasks: BackgroundTasks):
    """開始批量分塊任務"""
    try:
        # 驗證文檔是否存在
        doc = store.get_doc(req.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文檔不存在")
        
        # 創建任務ID
        task_id = str(uuid.uuid4())
        
        # 在後台處理批量分塊
        background_tasks.add_task(
            process_multiple_chunking,
            task_id,
            req.doc_id,
            req.strategies,
            req.chunk_sizes,
            req.overlap_ratios
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "total_combinations": len(req.strategies) * len(req.chunk_sizes) * len(req.overlap_ratios),
            "message": "批量分塊任務已開始"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunk/status/{task_id}")
async def get_chunking_status(task_id: str):
    """獲取批量分塊任務狀態"""
    if task_id not in chunking_task_store:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    status_info = chunking_task_store[task_id]
    return {
        "task_id": task_id,
        "status": status_info["status"],
        "progress": status_info.get("progress", 0.0),
        "error": status_info.get("error")
    }


@router.get("/chunk/results/{task_id}")
async def get_chunking_results(task_id: str):
    """獲取批量分塊結果"""
    if task_id not in chunking_task_store:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    status_info = chunking_task_store[task_id]
    if status_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="任務未完成")
    
    return {
        "task_id": task_id,
        "status": status_info["status"],
        "results": status_info["results"]
    }


# 策略評估路由
@router.post("/evaluate/strategy")
async def start_strategy_evaluation(req: StrategyEvaluationRequest, background_tasks: BackgroundTasks):
    """開始策略評估任務"""
    try:
        # 驗證文檔是否存在
        doc = store.get_doc(req.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文檔不存在")
        
        # 創建評估任務
        task_id = eval_store.create_task(
            doc_id=req.doc_id,
            configs=[],  # 不需要預先生成配置
            test_queries=req.test_queries,
            k_values=req.k_values,
            strategy="strategy_evaluation"
        )
        
        # 在後台處理策略評估
        background_tasks.add_task(
            process_strategy_evaluation,
            task_id,
            req.doc_id,
            req.chunking_results,
            req.qa_mapping_result,
            req.test_queries,
            req.k_values
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "total_configs": len(req.chunking_results),
            "message": "策略評估任務已開始"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
