"""
評估模組
"""

import re
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import EvaluationMetrics, EvaluationResult
from .chunking import chunk_text


def calculate_precision_omega(questions: List[str], chunks: List[str], k: int = 5) -> float:
    """計算 Precision Omega (理想情況下可達到的最大精度)"""
    if not questions or not chunks:
        return 0.0
    
    total_precision = 0.0
    valid_questions = 0
    
    for question in questions:
        question_chars = set(question.lower())
        relevant_chunks = []
        
        # 計算每個chunk與問題的相關性
        for chunk in chunks:
            chunk_chars = set(chunk.lower())
            overlap_chars = question_chars.intersection(chunk_chars)
            
            # 如果字符重疊度達到50%，認為是相關的
            if len(overlap_chars) >= len(question_chars) * 0.5:
                relevant_chunks.append(chunk)
        
        if relevant_chunks:
            # 取前k個相關chunk的精度
            precision = min(len(relevant_chunks), k) / k
            total_precision += precision
            valid_questions += 1
    
    return total_precision / valid_questions if valid_questions > 0 else 0.0


def calculate_precision_at_k(questions: List[str], chunks: List[str], k: int = 5) -> float:
    """計算 Precision@K"""
    if not questions or not chunks:
        return 0.0
    
    total_precision = 0.0
    valid_questions = 0
    
    # 使用TF-IDF進行檢索
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    
    for question in questions:
        question_vector = vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
        
        # 獲取前k個最相似的chunk
        top_k_indices = similarities.argsort()[-k:][::-1]
        top_k_chunks = [chunks[i] for i in top_k_indices]
        
        # 計算相關性
        question_chars = set(question.lower())
        relevant_count = 0
        
        for chunk in top_k_chunks:
            chunk_chars = set(chunk.lower())
            overlap_chars = question_chars.intersection(chunk_chars)
            
            # 如果字符重疊度達到50%，認為是相關的
            if len(overlap_chars) >= len(question_chars) * 0.5:
                relevant_count += 1
        
        precision = relevant_count / k
        total_precision += precision
        valid_questions += 1
    
    return total_precision / valid_questions if valid_questions > 0 else 0.0


def calculate_recall_at_k(questions: List[str], chunks: List[str], ground_truth_chunks: List[str], k: int = 5) -> float:
    """計算 Recall@K"""
    if not questions or not chunks:
        return 0.0
    
    total_recall = 0.0
    valid_questions = 0
    
    # 使用TF-IDF進行檢索
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    
    for question in questions:
        question_vector = vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
        
        # 獲取前k個最相似的chunk
        top_k_indices = similarities.argsort()[-k:][::-1]
        top_k_chunks = [chunks[i] for i in top_k_indices]
        
        # 計算相關性
        question_chars = set(question.lower())
        relevant_chunks = []
        
        # 找出所有相關的chunk
        for chunk in ground_truth_chunks:
            chunk_chars = set(chunk.lower())
            overlap_chars = question_chars.intersection(chunk_chars)
            
            if len(overlap_chars) >= len(question_chars) * 0.5:
                relevant_chunks.append(chunk)
        
        if relevant_chunks:
            # 計算召回率
            retrieved_relevant_count = 0
            for chunk in top_k_chunks:
                if chunk in relevant_chunks:
                    retrieved_relevant_count += 1
            
            recall = retrieved_relevant_count / len(relevant_chunks)
            total_recall += recall
            valid_questions += 1
    
    return total_recall / valid_questions if valid_questions > 0 else 0.0


def calculate_chunk_metrics(chunks: List[str]) -> Dict[str, float]:
    """計算分塊指標"""
    if not chunks:
        return {
            "chunk_count": 0,
            "avg_chunk_length": 0.0,
            "length_variance": 0.0
        }
    
    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths)
    
    # 計算方差
    variance = sum((length - avg_length) ** 2 for length in chunk_lengths) / len(chunk_lengths)
    
    return {
        "chunk_count": len(chunks),
        "avg_chunk_length": avg_length,
        "length_variance": variance
    }


def evaluate_chunk_config(text: str, questions: List[str], chunk_size: int, overlap_ratio: float) -> EvaluationResult:
    """評估單個分塊配置"""
    # 生成分塊
    chunks = chunk_text(text, strategy="fixed_size", chunk_size=chunk_size, overlap_ratio=overlap_ratio)
    
    # 計算指標
    chunk_metrics = calculate_chunk_metrics(chunks)
    
    # 計算不同k值的precision和recall
    precision_at_k = {}
    recall_at_k = {}
    
    for k in [1, 3, 5, 10]:
        precision_at_k[k] = calculate_precision_at_k(questions, chunks, k)
        recall_at_k[k] = calculate_recall_at_k(questions, chunks, chunks, k)
    
    # 計算precision omega
    precision_omega = calculate_precision_omega(questions, chunks)
    
    # 創建評估指標
    metrics = EvaluationMetrics(
        precision_omega=precision_omega,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        chunk_count=chunk_metrics["chunk_count"],
        avg_chunk_length=chunk_metrics["avg_chunk_length"],
        length_variance=chunk_metrics["length_variance"]
    )
    
    # 創建配置
    config = {
        "chunk_size": chunk_size,
        "overlap_ratio": overlap_ratio
    }
    
    return EvaluationResult(config=config, metrics=metrics)
