"""
API路由模組
"""

import uuid
import json
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .models import (
    DocRecord, EvaluationTask, ChunkConfig, EvaluationRequest, 
    GenerateQuestionsRequest
)
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from .store import InMemoryStore
from .pdf_processor import convert_pdf_to_text, convert_pdf_fallback
from .chunking import chunk_text
from .evaluation import evaluate_chunk_config
# from .question_generator import generate_questions  # 使用main.py中的函數

# 創建路由器
router = APIRouter()

# 使用main.py中的store實例
from .main import store

# 簡單的任務狀態存儲
task_status_store = {}


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
                    
                    # 處理項目
                    items = article.get("items", [])
                    for item in items:
                        item_name = item.get("item", "")
                        item_content = item.get("content", "")
                        
                        if item_name and item_content:
                            text_parts.append(f"{item_name} {item_content}\n")
                        
                        # 處理子項目
                        sub_items = item.get("sub_items", [])
                        for sub_item in sub_items:
                            sub_item_name = sub_item.get("sub_item", "")
                            sub_item_content = sub_item.get("content", "")
                            
                            if sub_item_name and sub_item_content:
                                text_parts.append(f"{sub_item_name} {sub_item_content}\n")
        
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


# PDF 轉換路由
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
        
        # 創建文檔記錄
        doc_record = DocRecord(
            id=doc_id,
            filename=file.filename,
            text=extracted_text,
            chunks=[],
            chunk_size=0,
            overlap=0,
            json_data=result["metadata"]
        )
        
        # 存儲文檔
        store.add_doc(doc_record)
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "text_length": len(result["text"]),
            "metadata": result["metadata"],
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
        
        # 如果是結構化層次分割，添加額外參數
        if request.strategy == "structured_hierarchical":
            chunk_kwargs["chunk_by"] = request.chunk_by
        
        # 生成分塊
        chunks = chunk_text(
            doc.text,
            strategy=request.strategy,
            json_data=doc.json_data,
            **chunk_kwargs
        )
        
        # 更新文檔記錄
        doc.chunks = chunks
        doc.chunk_size = request.chunk_size
        doc.overlap = int(request.chunk_size * request.overlap_ratio)
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
