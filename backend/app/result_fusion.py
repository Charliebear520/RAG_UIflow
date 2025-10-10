"""
多層次結果融合模組
用於將不同層次的檢索結果進行智能融合，提供更全面的檢索結果
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class FusionStrategy(Enum):
    """融合策略枚舉"""
    WEIGHTED_SUM = "weighted_sum"          # 加權求和
    RECIPROCAL_RANK = "reciprocal_rank"    # 倒數排名融合
    COMB_SUM = "comb_sum"                  # CombSUM融合
    COMB_ANZ = "comb_anz"                  # CombANZ融合
    COMB_MNZ = "comb_mnz"                  # CombMNZ融合


@dataclass
class RetrievalResult:
    """檢索結果數據類"""
    content: str
    similarity: float
    doc_id: str
    doc_name: str
    chunk_index: int
    level: str
    rank: int
    metadata: Dict[str, Any]
    hierarchical_description: str = ""  # 新增層級描述字段


@dataclass
class FusionConfig:
    """融合配置"""
    strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM
    level_weights: Dict[str, float] = None
    similarity_threshold: float = 0.0
    max_results: int = 10
    normalize_scores: bool = True
    
    def __post_init__(self):
        if self.level_weights is None:
            # 默認權重：對應論文中的六個粒度級別
            self.level_weights = {
                'document': 0.3,                    # 文件層級 - 權重較低
                'document_component': 0.5,          # 文件組成部分層級
                'basic_unit_hierarchy': 0.7,        # 基本單位層次結構層級
                'basic_unit': 1.0,                  # 基本單位層級 - 最高權重
                'basic_unit_component': 0.9,        # 基本單位組成部分層級
                'enumeration': 0.8                  # 列舉層級
            }


class MultiLevelResultFusion:
    """多層次結果融合器"""
    
    def __init__(self, config: FusionConfig = None):
        self.config = config or FusionConfig()
    
    def fuse_results(self, level_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        融合多層次檢索結果
        
        Args:
            level_results: 各層次的檢索結果 {level_name: [results]}
            
        Returns:
            List[Dict]: 融合後的結果列表
        """
        if not level_results:
            return []
        
        # 將結果轉換為RetrievalResult對象
        all_results = []
        for level, results in level_results.items():
            for result in results:
                retrieval_result = RetrievalResult(
                    content=result['content'],
                    similarity=result['similarity'],
                    doc_id=result['doc_id'],
                    doc_name=result['doc_name'],
                    chunk_index=result['chunk_index'],
                    level=level,
                    rank=result['rank'],
                    metadata=result.get('metadata', {}),
                    hierarchical_description=result.get('hierarchical_description', '')  # 添加層級描述
                )
                all_results.append(retrieval_result)
        
        # 根據策略進行融合
        if self.config.strategy == FusionStrategy.WEIGHTED_SUM:
            fused_results = self._weighted_sum_fusion(all_results)
        elif self.config.strategy == FusionStrategy.RECIPROCAL_RANK:
            fused_results = self._reciprocal_rank_fusion(all_results)
        elif self.config.strategy == FusionStrategy.COMB_SUM:
            fused_results = self._comb_sum_fusion(all_results)
        elif self.config.strategy == FusionStrategy.COMB_ANZ:
            fused_results = self._comb_anz_fusion(all_results)
        elif self.config.strategy == FusionStrategy.COMB_MNZ:
            fused_results = self._comb_mnz_fusion(all_results)
        else:
            fused_results = self._weighted_sum_fusion(all_results)
        
        # 過濾和排序結果
        filtered_results = self._filter_and_rank_results(fused_results)
        
        # 轉換回字典格式
        return [self._result_to_dict(result) for result in filtered_results]
    
    def _weighted_sum_fusion(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """加權求和融合"""
        # 按內容分組
        content_groups = {}
        for result in results:
            content_key = self._get_content_key(result)
            if content_key not in content_groups:
                content_groups[content_key] = []
            content_groups[content_key].append(result)
        
        fused_results = []
        for content_key, group in content_groups.items():
            # 計算加權分數
            weighted_score = 0.0
            total_weight = 0.0
            level_scores = {}
            
            for result in group:
                weight = self.config.level_weights.get(result.level, 1.0)
                weighted_score += result.similarity * weight
                total_weight += weight
                level_scores[result.level] = result.similarity
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
                
                # 選擇最佳結果作為代表
                best_result = max(group, key=lambda x: x.similarity)
                best_result.similarity = final_score
                best_result.metadata['level_scores'] = level_scores
                best_result.metadata['fusion_method'] = 'weighted_sum'
                
                fused_results.append(best_result)
        
        return fused_results
    
    def _reciprocal_rank_fusion(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """倒數排名融合"""
        # 按內容分組
        content_groups = {}
        for result in results:
            content_key = self._get_content_key(result)
            if content_key not in content_groups:
                content_groups[content_key] = []
            content_groups[content_key].append(result)
        
        fused_results = []
        for content_key, group in content_groups.items():
            # 計算倒數排名分數
            reciprocal_score = 0.0
            level_scores = {}
            
            for result in group:
                # 倒數排名：1/rank
                reciprocal_rank = 1.0 / result.rank
                weight = self.config.level_weights.get(result.level, 1.0)
                reciprocal_score += reciprocal_rank * weight
                level_scores[result.level] = reciprocal_rank
            
            # 選擇最佳結果作為代表
            best_result = max(group, key=lambda x: x.similarity)
            best_result.similarity = reciprocal_score
            best_result.metadata['level_scores'] = level_scores
            best_result.metadata['fusion_method'] = 'reciprocal_rank'
            
            fused_results.append(best_result)
        
        return fused_results
    
    def _comb_sum_fusion(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """CombSUM融合：簡單求和"""
        return self._weighted_sum_fusion(results)
    
    def _comb_anz_fusion(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """CombANZ融合：平均分數"""
        # 按內容分組
        content_groups = {}
        for result in results:
            content_key = self._get_content_key(result)
            if content_key not in content_groups:
                content_groups[content_key] = []
            content_groups[content_key].append(result)
        
        fused_results = []
        for content_key, group in content_groups.items():
            # 計算平均分數
            avg_score = sum(result.similarity for result in group) / len(group)
            level_scores = {result.level: result.similarity for result in group}
            
            # 選擇最佳結果作為代表
            best_result = max(group, key=lambda x: x.similarity)
            best_result.similarity = avg_score
            best_result.metadata['level_scores'] = level_scores
            best_result.metadata['fusion_method'] = 'comb_anz'
            
            fused_results.append(best_result)
        
        return fused_results
    
    def _comb_mnz_fusion(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """CombMNZ融合：分數乘以出現次數"""
        # 按內容分組
        content_groups = {}
        for result in results:
            content_key = self._get_content_key(result)
            if content_key not in content_groups:
                content_groups[content_key] = []
            content_groups[content_key].append(result)
        
        fused_results = []
        for content_key, group in content_groups.items():
            # 計算CombMNZ分數：分數和 × 出現次數
            score_sum = sum(result.similarity for result in group)
            occurrence_count = len(group)
            mnz_score = score_sum * occurrence_count
            
            level_scores = {result.level: result.similarity for result in group}
            
            # 選擇最佳結果作為代表
            best_result = max(group, key=lambda x: x.similarity)
            best_result.similarity = mnz_score
            best_result.metadata['level_scores'] = level_scores
            best_result.metadata['fusion_method'] = 'comb_mnz'
            best_result.metadata['occurrence_count'] = occurrence_count
            
            fused_results.append(best_result)
        
        return fused_results
    
    def _get_content_key(self, result: RetrievalResult) -> str:
        """生成內容的唯一鍵"""
        # 使用內容的hash作為唯一標識
        import hashlib
        content_hash = hashlib.md5(result.content.encode('utf-8')).hexdigest()
        return f"{result.doc_id}_{content_hash}"
    
    def _filter_and_rank_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """過濾和排序結果"""
        # 過濾低分數結果
        filtered_results = [
            result for result in results 
            if result.similarity >= self.config.similarity_threshold
        ]
        
        # 按分數排序
        filtered_results.sort(key=lambda x: x.similarity, reverse=True)
        
        # 限制結果數量
        if self.config.max_results > 0:
            filtered_results = filtered_results[:self.config.max_results]
        
        # 重新分配排名
        for i, result in enumerate(filtered_results):
            result.rank = i + 1
        
        return filtered_results
    
    def _result_to_dict(self, result: RetrievalResult) -> Dict[str, Any]:
        """將RetrievalResult轉換為字典"""
        result_dict = {
            "rank": int(result.rank),
            "content": result.content,
            "similarity": float(result.similarity),
            "doc_id": result.doc_id,
            "doc_name": result.doc_name,
            "chunk_index": int(result.chunk_index),
            "level": result.level,
            "metadata": result.metadata
        }
        
        # 如果有hierarchical_description字段，添加到結果中
        if hasattr(result, 'hierarchical_description'):
            result_dict["hierarchical_description"] = result.hierarchical_description
        
        return result_dict


def create_fusion_config(
    strategy: str = "weighted_sum",
    level_weights: Dict[str, float] = None,
    similarity_threshold: float = 0.0,
    max_results: int = 10
) -> FusionConfig:
    """
    創建融合配置的便捷函數
    
    Args:
        strategy: 融合策略
        level_weights: 層次權重
        similarity_threshold: 相似度閾值
        max_results: 最大結果數量
        
    Returns:
        FusionConfig: 融合配置對象
    """
    return FusionConfig(
        strategy=FusionStrategy(strategy),
        level_weights=level_weights,
        similarity_threshold=similarity_threshold,
        max_results=max_results
    )


def fuse_multi_level_results(
    level_results: Dict[str, List[Dict[str, Any]]],
    config: FusionConfig = None
) -> List[Dict[str, Any]]:
    """
    便捷函數：融合多層次檢索結果
    
    Args:
        level_results: 各層次的檢索結果
        config: 融合配置
        
    Returns:
        List[Dict]: 融合後的結果
    """
    fusion = MultiLevelResultFusion(config)
    return fusion.fuse_results(level_results)
