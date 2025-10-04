"""
增強版主程序 - 整合新的法律語義檢索功能
"""

from .main import *  # 導入原有功能
from .legal_semantic_chunking import LegalSemanticIntegrityChunking, MultiLevelSemanticChunking
from .legal_concept_graph import LegalConceptGraph, LegalConceptGraphRetrieval
from .adaptive_legal_rag import AdaptiveLegalRAG, QueryAnalyzer
from typing import List, Dict, Any
import json


# 初始化增強版組件
legal_semantic_chunker = LegalSemanticIntegrityChunking()
multi_level_chunker = MultiLevelSemanticChunking()
concept_graph = LegalConceptGraph()
concept_graph_retrieval = None
adaptive_rag = AdaptiveLegalRAG()


@app.post("/api/legal-semantic-chunk")
def legal_semantic_chunk(req: ChunkConfig):
    """法律語義完整性分塊"""
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
        
        # 構建概念圖
        concept_graph.build_graph(documents)
        
        # 初始化概念圖檢索
        global concept_graph_retrieval
        concept_graph_retrieval = LegalConceptGraphRetrieval(concept_graph)
        
        # 註冊到自適應RAG
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
    try:
        performance = adaptive_rag.performance_monitor.get_strategy_performance()
        
        return {
            "strategy_performance": performance,
            "total_retrievals": len(adaptive_rag.performance_monitor.retrieval_history)
        }
        
    except Exception as e:
        print(f"❌ 獲取策略性能錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": f"獲取性能錯誤: {str(e)}"})


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
        'retrieve': lambda query, **kwargs: hierarchical_retrieve_original(query, kwargs.get('k', 5))
    })


def retrieve_original(query: str, k: int):
    """原始向量檢索"""
    # 這裡調用原有的檢索邏輯
    pass


def hybrid_retrieve_original(query: str, k: int):
    """原始HybridRAG檢索"""
    # 這裡調用原有的HybridRAG邏輯
    pass


def hierarchical_retrieve_original(query: str, k: int):
    """原始多層次檢索"""
    # 這裡調用原有的多層次檢索邏輯
    pass
