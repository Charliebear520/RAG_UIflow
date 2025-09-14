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
from .store import InMemoryStore
from .pdf_processor import convert_pdf_to_text, convert_pdf_fallback
from .chunking import chunk_text
from .evaluation import evaluate_chunk_config
# from .question_generator import generate_questions  # 使用main.py中的函數

# 創建路由器
router = APIRouter()

# 創建store實例
store = InMemoryStore()


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
        
        # 生成文檔ID
        doc_id = str(uuid.uuid4())
        
        # 創建文檔記錄
        doc_record = DocRecord(
            id=doc_id,
            filename=file.filename,
            text=result["text"],
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 分塊路由
@router.post("/chunk")
async def chunk_document(request: ChunkConfig):
    """分塊文檔"""
    try:
        # 這裡需要從請求中獲取doc_id，暫時使用第一個文檔
        docs = store.list_docs()
        if not docs:
            raise HTTPException(status_code=404, detail="沒有找到文檔")
        
        doc = docs[0]  # 使用第一個文檔
        
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
        
        return {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "strategy": request.strategy,
            "chunk_size": request.chunk_size,
            "overlap_ratio": request.overlap_ratio,
            "chunk_by": request.chunk_by if request.strategy == "structured_hierarchical" else None,
            "avg_chunk_length": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            "chunk_lengths": [len(chunk) for chunk in chunks]
        }
        
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
