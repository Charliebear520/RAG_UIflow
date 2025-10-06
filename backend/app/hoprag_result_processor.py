"""
HopRAG結果處理器模組
包含RelevanceFilter、ResultRanker
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .hoprag_config import HopRAGConfig, DEFAULT_CONFIG
from .hoprag_graph_builder import LegalNode

@dataclass
class RetrievalResult:
    """檢索結果數據結構"""
    node_id: str
    content: str
    contextualized_text: str
    law_name: str
    article_number: str
    item_number: Optional[str] = None
    node_type: str = ""
    hop_level: int = 0
    hop_source: str = "base_retrieval"
    similarity_score: float = 0.0
    relevance_score: float = 0.0
    rank: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RelevanceFilter:
    """相關性過濾器"""
    
    def __init__(self, config: HopRAGConfig = DEFAULT_CONFIG):
        self.config = config
        
    def filter_results(self, results: List[RetrievalResult], 
                      query: str, min_score: float = 0.3) -> List[RetrievalResult]:
        """過濾相關性較低的結果"""
        print(f"🔍 過濾檢索結果，最小分數: {min_score}")
        
        filtered_results = []
        for result in results:
            # 計算綜合相關性分數
            relevance_score = self._calculate_relevance_score(result, query)
            result.relevance_score = relevance_score
            
            if relevance_score >= min_score:
                filtered_results.append(result)
        
        print(f"✅ 過濾完成，{len(results)} -> {len(filtered_results)} 個結果")
        return filtered_results
    
    def _calculate_relevance_score(self, result: RetrievalResult, query: str) -> float:
        """計算綜合相關性分數"""
        base_score = result.similarity_score
        
        # 根據跳躍層次調整分數
        hop_penalty = self.config.hop_weight_decay ** result.hop_level
        
        # 根據節點類型調整分數
        type_bonus = self._get_type_bonus(result.node_type)
        
        # 綜合分數
        relevance_score = base_score * hop_penalty * type_bonus
        
        return min(relevance_score, 1.0)  # 限制在[0,1]範圍內
    
    def _get_type_bonus(self, node_type: str) -> float:
        """根據節點類型獲取分數加成"""
        if node_type == "article":
            return 1.0
        elif node_type == "item":
            return 0.9
        else:
            return 0.8
    
    def filter_by_hop_level(self, results: List[RetrievalResult], 
                           max_hop_level: int) -> List[RetrievalResult]:
        """根據跳躍層次過濾結果"""
        filtered_results = [
            result for result in results 
            if result.hop_level <= max_hop_level
        ]
        
        print(f"🔍 按跳躍層次過濾 (max_hop={max_hop_level}): {len(results)} -> {len(filtered_results)} 個結果")
        return filtered_results
    
    def filter_by_node_type(self, results: List[RetrievalResult], 
                           allowed_types: List[str]) -> List[RetrievalResult]:
        """根據節點類型過濾結果"""
        filtered_results = [
            result for result in results 
            if result.node_type in allowed_types
        ]
        
        print(f"🔍 按節點類型過濾 {allowed_types}: {len(results)} -> {len(filtered_results)} 個結果")
        return filtered_results

class ResultRanker:
    """結果排序器"""
    
    def __init__(self, config: HopRAGConfig = DEFAULT_CONFIG):
        self.config = config
        
    def rank_results(self, results: List[RetrievalResult], 
                    query: str, strategy: str = None) -> List[RetrievalResult]:
        """對結果進行排序"""
        if strategy is None:
            strategy = self.config.result_merge_strategy
        
        print(f"📊 使用策略 '{strategy}' 對 {len(results)} 個結果進行排序")
        
        if strategy == "weighted_merge":
            return self._weighted_rank(results, query)
        elif strategy == "simple_merge":
            return self._simple_rank(results, query)
        elif strategy == "hop_aware_rank":
            return self._hop_aware_rank(results, query)
        else:
            return self._default_rank(results, query)
    
    def _weighted_rank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """加權排序策略"""
        for result in results:
            # 計算綜合分數
            weighted_score = self._calculate_weighted_score(result, query)
            result.relevance_score = weighted_score
        
        # 按分數排序
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        
        # 重新分配排名
        for i, result in enumerate(sorted_results):
            result.rank = i + 1
        
        return sorted_results
    
    def _simple_rank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """簡單排序策略"""
        # 按相似度分數排序
        sorted_results = sorted(results, key=lambda x: x.similarity_score, reverse=True)
        
        # 重新分配排名
        for i, result in enumerate(sorted_results):
            result.rank = i + 1
        
        return sorted_results
    
    def _hop_aware_rank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """跳躍感知排序策略"""
        # 按跳躍層次分組
        hop_groups = {}
        for result in results:
            hop_level = result.hop_level
            if hop_level not in hop_groups:
                hop_groups[hop_level] = []
            hop_groups[hop_level].append(result)
        
        # 對每組內部排序
        ranked_groups = {}
        for hop_level, group in hop_groups.items():
            ranked_groups[hop_level] = sorted(group, key=lambda x: x.similarity_score, reverse=True)
        
        # 合併排序結果（優先級：低跳躍層次 > 高分數）
        sorted_results = []
        current_rank = 1
        
        for hop_level in sorted(ranked_groups.keys()):
            group = ranked_groups[hop_level]
            for result in group:
                result.rank = current_rank
                sorted_results.append(result)
                current_rank += 1
        
        return sorted_results
    
    def _default_rank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """默認排序策略"""
        return self._weighted_rank(results, query)
    
    def _calculate_weighted_score(self, result: RetrievalResult, query: str) -> float:
        """計算加權分數"""
        base_score = result.similarity_score
        
        # 跳躍層次權重衰減
        hop_weight = self.config.hop_weight_decay ** result.hop_level
        
        # 節點類型權重
        type_weight = self._get_node_type_weight(result.node_type)
        
        # 內容長度權重（適中長度的內容可能更有價值）
        length_weight = self._get_content_length_weight(len(result.content))
        
        # 綜合加權分數
        weighted_score = base_score * hop_weight * type_weight * length_weight
        
        return min(weighted_score, 1.0)
    
    def _get_node_type_weight(self, node_type: str) -> float:
        """獲取節點類型權重"""
        weights = {
            "article": 1.0,
            "item": 0.9,
            "": 0.8
        }
        return weights.get(node_type, 0.8)
    
    def _get_content_length_weight(self, content_length: int) -> float:
        """獲取內容長度權重"""
        # 理想長度範圍：100-500字符
        if 100 <= content_length <= 500:
            return 1.0
        elif 50 <= content_length < 100:
            return 0.9
        elif 500 < content_length <= 1000:
            return 0.8
        else:
            return 0.7

class ResultProcessor:
    """結果處理器主類"""
    
    def __init__(self, config: HopRAGConfig = DEFAULT_CONFIG):
        self.config = config
        
        # 初始化組件
        self.relevance_filter = RelevanceFilter(config)
        self.result_ranker = ResultRanker(config)
        
    def process_results(self, base_results: List[Dict[str, Any]], 
                       hop_results: Dict[int, List[str]], 
                       nodes: Dict[str, LegalNode],
                       query: str, k: int) -> List[Dict[str, Any]]:
        """處理和合併結果"""
        print(f"🔄 處理檢索結果，目標數量: {k}")
        
        # Step 1: 轉換為RetrievalResult對象
        retrieval_results = self._convert_to_retrieval_results(
            base_results, hop_results, nodes
        )
        
        # Step 2: 過濾相關性較低的結果
        filtered_results = self.relevance_filter.filter_results(
            retrieval_results, query, min_score=0.3
        )
        
        # Step 3: 排序結果
        ranked_results = self.result_ranker.rank_results(
            filtered_results, query
        )
        
        # Step 4: 限制結果數量
        final_results = ranked_results[:k]
        
        # Step 5: 轉換回字典格式
        processed_results = self._convert_to_dict_results(final_results)
        
        print(f"✅ 結果處理完成，返回 {len(processed_results)} 個結果")
        return processed_results
    
    def _convert_to_retrieval_results(self, base_results: List[Dict[str, Any]], 
                                    hop_results: Dict[int, List[str]], 
                                    nodes: Dict[str, LegalNode]) -> List[RetrievalResult]:
        """轉換為RetrievalResult對象"""
        retrieval_results = []
        
        # 處理基礎結果
        for result in base_results:
            node_id = result.get('node_id') or result.get('id')
            if node_id and node_id in nodes:
                node = nodes[node_id]
                
                retrieval_result = RetrievalResult(
                    node_id=node_id,
                    content=node.content,
                    contextualized_text=node.contextualized_text,
                    law_name=node.law_name,
                    article_number=node.article_number,
                    item_number=node.item_number,
                    node_type=node.node_type.value,
                    hop_level=0,
                    hop_source="base_retrieval",
                    similarity_score=result.get('similarity_score', 0.0),
                    metadata=node.metadata
                )
                retrieval_results.append(retrieval_result)
        
        # 處理HopRAG結果
        for hop_level, node_ids in hop_results.items():
            if hop_level == 0:  # 跳過基礎結果
                continue
                
            for node_id in node_ids:
                if node_id in nodes:
                    node = nodes[node_id]
                    
                    retrieval_result = RetrievalResult(
                        node_id=node_id,
                        content=node.content,
                        contextualized_text=node.contextualized_text,
                        law_name=node.law_name,
                        article_number=node.article_number,
                        item_number=node.item_number,
                        node_type=node.node_type.value,
                        hop_level=hop_level,
                        hop_source="hoprag_traversal",
                        similarity_score=0.0,  # HopRAG結果沒有直接的相似度分數
                        metadata=node.metadata
                    )
                    retrieval_results.append(retrieval_result)
        
        return retrieval_results
    
    def _convert_to_dict_results(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """轉換為字典格式結果"""
        dict_results = []
        
        for result in results:
            dict_result = {
                'node_id': result.node_id,
                'content': result.content,
                'contextualized_text': result.contextualized_text,
                'law_name': result.law_name,
                'article_number': result.article_number,
                'item_number': result.item_number,
                'node_type': result.node_type,
                'hop_level': result.hop_level,
                'hop_source': result.hop_source,
                'similarity_score': result.similarity_score,
                'relevance_score': result.relevance_score,
                'rank': result.rank,
                'metadata': result.metadata
            }
            dict_results.append(dict_result)
        
        return dict_results
    
    def get_processing_stats(self, original_count: int, final_count: int) -> Dict[str, Any]:
        """獲取處理統計信息"""
        return {
            "original_count": original_count,
            "final_count": final_count,
            "filter_ratio": final_count / original_count if original_count > 0 else 0,
            "processing_strategy": self.config.result_merge_strategy,
            "hop_weight_decay": self.config.hop_weight_decay
        }
