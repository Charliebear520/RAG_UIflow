"""
增強版HybridRAG模組
深度融合向量檢索、BM25關鍵字檢索和metadata增強
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re

try:
    import jieba  # type: ignore
    import jieba.analyse  # type: ignore
    jieba.initialize()
except ImportError:
    jieba = None  # type: ignore

from .faiss_store import FAISSVectorStore
from .bm25_index import BM25KeywordIndex
from .metadata_enhancer import MetadataEnhancer


@dataclass
class EnhancedHybridConfig:
    """增強版HybridRAG配置"""
    # 向量檢索權重
    vector_weight: float = 0.6
    
    # BM25關鍵字檢索權重
    bm25_weight: float = 0.25
    
    # Metadata加分權重
    metadata_weight: float = 0.15
    
    # 具體的metadata加分權重
    w_law_match: float = 0.15
    w_article_match: float = 0.15
    w_concept_match: float = 0.1
    w_keyword_hit: float = 0.05
    w_domain_match: float = 0.05
    w_title_match: float = 0.1
    w_category_match: float = 0.05
    
    # 加分上限
    max_bonus: float = 0.4
    
    # 標題匹配配置
    title_boost_factor: float = 1.5
    category_boost_factor: float = 1.3
    
    # Metadata向下繼承配置
    enable_inheritance_strategy: bool = True
    metadata_match_threshold: float = 0.3
    inheritance_bonus: float = 0.1
    inheritance_boost_factor: float = 1.2


class EnhancedHybridRAG:
    """增強版HybridRAG"""
    
    def __init__(self, faiss_store: FAISSVectorStore, bm25_index: BM25KeywordIndex, 
                 metadata_enhancer: MetadataEnhancer):
        self.faiss_store = faiss_store
        self.bm25_index = bm25_index
        self.metadata_enhancer = metadata_enhancer
        
        # 法律同義詞字典
        self.legal_synonyms = {
            '公開傳輸': ['公開傳輸', '數位傳輸', '網路傳輸', '線上傳輸', '上線提供', 'public transmission'],
            '公開播送': ['公開播送', '廣播', '播送', 'broadcast'],
            '重製': ['重製', '複製', '拷貝', '複本製作', 'reproduction'],
            '散布': ['散布', '發行', '流通', 'distribution'],
            '改作': ['改作', '改編', '翻案', '衍生創作', 'derivative'],
            '引用': ['引用', '節錄', '摘錄', '引用他人著作'],
            '合理使用': ['合理使用', '公平使用', 'fair use'],
            '商標權': ['商標權', '商標專用權', '商標使用權'],
            '著作權': ['著作權', '版權', 'copyright'],
            '專利權': ['專利權', '專利', 'patent'],
            '侵害': ['侵害', '侵犯', '違反', '損害', '違法', '不法'],
            '處罰': ['處罰', '制裁', '懲罰', 'penalty']
        }
    
    def retrieve(self, query: str, k: int = 10, config: Optional[EnhancedHybridConfig] = None) -> List[Dict[str, Any]]:
        """執行增強版HybridRAG檢索 - 支持metadata向下繼承"""
        if not config:
            config = EnhancedHybridConfig()
        
        # 1. Metadata關鍵字匹配（實現向下繼承策略）
        metadata_matched_articles = []
        if config.enable_inheritance_strategy:
            metadata_matched_articles = self._metadata_keyword_match(query)
            print(f"🔍 Metadata關鍵字匹配到 {len(metadata_matched_articles)} 個條層級")
        
        # 2. 向量檢索
        vector_results = self._vector_retrieve(query, k * 3)  # 獲取更多候選
        
        # 3. BM25關鍵字檢索
        bm25_results = self._bm25_retrieve(query, k * 3)
        
        # 4. 合併候選結果
        candidate_nodes = self._merge_candidates(vector_results, bm25_results, k * 2)
        
        # 5. 應用metadata向下繼承策略
        if config.enable_inheritance_strategy and metadata_matched_articles:
            candidate_nodes = self._apply_inheritance_strategy(candidate_nodes, metadata_matched_articles, query, config)
        
        # 6. 計算綜合分數
        final_results = self._calculate_hybrid_scores(query, candidate_nodes, config)
        
        # 7. 排序並返回前k個結果
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return final_results[:k]
    
    def retrieve_multi_level(self, query: str, level_name: str, k: int = 10, 
                           config: Optional[EnhancedHybridConfig] = None) -> List[Dict[str, Any]]:
        """執行多層次增強版HybridRAG檢索"""
        if not config:
            config = EnhancedHybridConfig()
        
        # 1. 多層次向量檢索
        vector_results = self._multi_level_vector_retrieve(query, level_name, k * 3)
        
        # 2. 多層次BM25檢索
        bm25_results = self._multi_level_bm25_retrieve(query, level_name, k * 3)
        
        # 3. 合併候選結果
        candidate_nodes = self._merge_candidates(vector_results, bm25_results, k * 2)
        
        # 4. 計算綜合分數
        final_results = self._calculate_hybrid_scores(query, candidate_nodes, config)
        
        # 5. 排序並返回前k個結果
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return final_results[:k]
    
    def _vector_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """向量檢索"""
        if not self.faiss_store.has_vectors():
            return []
        
        # 生成查詢向量（這裡需要調用embedding函數）
        query_vector = self._get_query_vector(query)
        if not query_vector:
            return []
        
        # FAISS搜索
        indices, scores = self.faiss_store.search(query_vector, k)
        
        results = []
        for idx, score in zip(indices, scores):
            chunk_info = self.faiss_store.get_chunk_by_index(idx)
            if chunk_info:
                results.append({
                    'chunk_id': chunk_info['chunk_id'],
                    'doc_id': chunk_info['doc_id'],
                    'content': chunk_info['content'],
                    'enhanced_metadata': chunk_info['enhanced_metadata'],
                    'vector_score': score,
                    'bm25_score': 0.0,
                    'chunk_index': idx
                })
        
        return results
    
    def _bm25_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """BM25關鍵字檢索"""
        if not self.bm25_index.has_index():
            return []
        
        # BM25搜索
        indices, scores = self.bm25_index.search(query, k)
        
        results = []
        for idx, score in zip(indices, scores):
            chunk_info = self.bm25_index.get_chunk_by_index(idx)
            if chunk_info:
                results.append({
                    'chunk_id': chunk_info['chunk_id'],
                    'doc_id': chunk_info['doc_id'],
                    'content': chunk_info['content'],
                    'enhanced_metadata': {},  # BM25索引不包含enhanced_metadata
                    'vector_score': 0.0,
                    'bm25_score': score,
                    'chunk_index': idx
                })
        
        return results
    
    def _multi_level_vector_retrieve(self, query: str, level_name: str, k: int) -> List[Dict[str, Any]]:
        """多層次向量檢索"""
        if level_name not in self.faiss_store.get_available_levels():
            return []
        
        # 生成查詢向量
        query_vector = self._get_query_vector(query)
        if not query_vector:
            return []
        
        # 多層次FAISS搜索
        indices, scores = self.faiss_store.search_multi_level(level_name, query_vector, k)
        
        results = []
        for idx, score in zip(indices, scores):
            chunk_info = self.faiss_store.get_multi_level_chunk_by_index(level_name, idx)
            if chunk_info:
                results.append({
                    'chunk_id': chunk_info['chunk_id'],
                    'doc_id': chunk_info['doc_id'],
                    'content': chunk_info['content'],
                    'enhanced_metadata': chunk_info['enhanced_metadata'],
                    'vector_score': score,
                    'bm25_score': 0.0,
                    'chunk_index': idx,
                    'level': level_name
                })
        
        return results
    
    def _multi_level_bm25_retrieve(self, query: str, level_name: str, k: int) -> List[Dict[str, Any]]:
        """多層次BM25檢索"""
        if level_name not in self.bm25_index.get_available_levels():
            return []
        
        # 多層次BM25搜索
        indices, scores = self.bm25_index.search_multi_level(level_name, query, k)
        
        results = []
        for idx, score in zip(indices, scores):
            chunk_info = self.bm25_index.get_multi_level_chunk_by_index(level_name, idx)
            if chunk_info:
                results.append({
                    'chunk_id': chunk_info['chunk_id'],
                    'doc_id': chunk_info['doc_id'],
                    'content': chunk_info['content'],
                    'enhanced_metadata': {},
                    'vector_score': 0.0,
                    'bm25_score': score,
                    'chunk_index': idx,
                    'level': level_name
                })
        
        return results
    
    def _merge_candidates(self, vector_results: List[Dict[str, Any]], 
                         bm25_results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """合併候選結果"""
        # 使用chunk_id作為唯一標識符
        merged = {}
        
        # 添加向量檢索結果
        for result in vector_results:
            chunk_id = result['chunk_id']
            merged[chunk_id] = result.copy()
        
        # 合併BM25檢索結果
        for result in bm25_results:
            chunk_id = result['chunk_id']
            if chunk_id in merged:
                # 合併分數
                merged[chunk_id]['bm25_score'] = result['bm25_score']
                if not merged[chunk_id]['enhanced_metadata']:
                    merged[chunk_id]['enhanced_metadata'] = result['enhanced_metadata']
            else:
                merged[chunk_id] = result.copy()
        
        # 轉換為列表並排序
        candidate_list = list(merged.values())
        
        # 按綜合分數排序（向量分數 + BM25分數）
        candidate_list.sort(key=lambda x: x['vector_score'] + x['bm25_score'], reverse=True)
        
        return candidate_list[:k]
    
    def _calculate_hybrid_scores(self, query: str, candidates: List[Dict[str, Any]], 
                                config: EnhancedHybridConfig) -> List[Dict[str, Any]]:
        """計算HybridRAG綜合分數"""
        query_features = self._extract_query_features(query)
        
        for candidate in candidates:
            # 1. 標準化向量和BM25分數
            vector_score = self._normalize_score(candidate['vector_score'], 'vector')
            bm25_score = self._normalize_score(candidate['bm25_score'], 'bm25')
            
            # 2. 計算metadata加分
            metadata_bonus = self._calculate_metadata_bonus(
                candidate['enhanced_metadata'], query_features, config
            )
            
            # 3. 計算標題專門處理加分
            title_bonus = self._calculate_title_bonus(
                candidate['enhanced_metadata'], query, config
            )
            
            # 4. 計算綜合分數
            hybrid_score = (
                config.vector_weight * vector_score +
                config.bm25_weight * bm25_score +
                config.metadata_weight * metadata_bonus +
                title_bonus
            )
            
            # 5. 添加詳細分數分解
            candidate['hybrid_score'] = hybrid_score
            candidate['score_breakdown'] = {
                'vector_score': vector_score,
                'bm25_score': bm25_score,
                'metadata_bonus': metadata_bonus,
                'title_bonus': title_bonus,
                'final_score': hybrid_score
            }
        
        return candidates
    
    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """提取查詢特徵"""
        query_lower = query.lower()
        
        # 提取法名
        law = ''
        if '著作權法' in query:
            law = '著作權法'
        elif '商標法' in query:
            law = '商標法'
        elif '專利法' in query:
            law = '專利法'
        
        # 提取條號
        article_number, article_suffix = self._extract_article_number(query)
        
        # 提取法律概念
        concepts = []
        for canonical, variants in self.legal_synonyms.items():
            if any(v in query for v in variants):
                concepts.append(canonical)
        
        # 提取查詢意圖
        intent_tags = []
        if any(word in query for word in ["什麼是", "定義", "權利", "什麼權"]):
            intent_tags.append("權利查詢")
        if any(word in query for word in ["必須", "應", "不得", "禁止", "義務"]):
            intent_tags.append("義務查詢")
        if any(word in query for word in ["例外", "除外", "但", "惟", "不適用"]):
            intent_tags.append("例外查詢")
        if any(word in query for word in ["處罰", "違反", "後果", "責任", "賠償"]):
            intent_tags.append("後果查詢")
        
        return {
            'law': law,
            'article_number': article_number,
            'article_suffix': article_suffix,
            'concepts': concepts,
            'intent_tags': intent_tags,
            'query_text': query
        }
    
    def _extract_article_number(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """提取條號"""
        # 第X條之Y
        match = re.search(r"第(\d+)條之(\d+)", text)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        # 第X條
        match = re.search(r"第(\d+)條", text)
        if match:
            return int(match.group(1)), None
        
        return None, None
    
    def _calculate_metadata_bonus(self, metadata: Dict[str, Any], 
                                 query_features: Dict[str, Any], 
                                 config: EnhancedHybridConfig) -> float:
        """計算metadata加分"""
        bonus = 0.0
        
        # 1. 法名匹配加分
        if query_features['law'] and metadata.get('category') == query_features['law']:
            bonus += config.w_law_match
        
        # 2. 條號匹配加分
        if query_features['article_number'] and metadata.get('article_number') == query_features['article_number']:
            article_bonus = config.w_article_match
            if query_features['article_suffix'] and metadata.get('article_suffix') != query_features['article_suffix']:
                article_bonus *= 0.5
            bonus += article_bonus
        
        # 3. 法律概念匹配加分
        legal_concepts = metadata.get('legal_concepts', [])
        for concept in legal_concepts:
            concept_name = concept.get('concept_name', '')
            synonyms = concept.get('synonyms', [])
            importance = concept.get('importance_score', 0.5)
            
            if concept_name in query_features['query_text']:
                bonus += config.w_concept_match * importance
            elif any(syn in query_features['query_text'] for syn in synonyms):
                bonus += config.w_concept_match * importance * 0.7
        
        # 4. 語義關鍵詞匹配加分
        semantic_keywords = metadata.get('semantic_keywords', {})
        keyword_weights = semantic_keywords.get('keyword_weights', {})
        for keyword, weight in keyword_weights.items():
            if keyword in query_features['query_text']:
                bonus += config.w_keyword_hit * weight
        
        # 5. 法律領域匹配加分
        legal_domain = metadata.get('legal_domain', {})
        domain_name = legal_domain.get('legal_domain', '')
        if domain_name in query_features['query_text']:
            bonus += config.w_domain_match
        
        # 6. 查詢意圖匹配加分
        intent_tags = metadata.get('query_intent_tags', [])
        if any(intent in intent_tags for intent in query_features['intent_tags']):
            bonus += config.w_keyword_hit * 0.5
        
        return min(bonus, config.max_bonus)
    
    def _calculate_title_bonus(self, metadata: Dict[str, Any], query: str, 
                              config: EnhancedHybridConfig) -> float:
        """計算標題專門處理加分"""
        bonus = 0.0
        
        # 1. 條文標題匹配
        article_label = metadata.get('article_label', '')
        if article_label and article_label in query:
            bonus += config.w_title_match * config.title_boost_factor
        
        # 2. 章節標題匹配
        chapter = metadata.get('chapter', '')
        if chapter and any(word in query for word in chapter.split()):
            bonus += config.w_title_match * 0.5
        
        # 3. 分類匹配
        category = metadata.get('category', '')
        if category and category in query:
            bonus += config.w_category_match * config.category_boost_factor
        
        # 4. 法律名稱匹配（標題層面）
        law_name = metadata.get('law_name', '')
        if law_name and law_name in query:
            bonus += config.w_category_match * 0.7
        
        return min(bonus, config.max_bonus)
    
    def _normalize_score(self, score: float, score_type: str) -> float:
        """標準化分數"""
        if score_type == 'vector':
            # 向量相似度分數通常在0-1之間
            return max(0.0, min(1.0, score))
        elif score_type == 'bm25':
            # BM25分數需要標準化，通常使用sigmoid函數
            import math
            return 1.0 / (1.0 + math.exp(-score))
        else:
            return max(0.0, min(1.0, score))
    
    def _get_query_vector(self, query: str) -> Optional[List[float]]:
        """獲取查詢向量"""
        # 這裡需要調用embedding函數
        # 實際實現時需要調用embed_gemini或embed_bge_m3
        # 暫時返回None，需要在main.py中實現具體的embedding調用
        return None
    
    def _metadata_keyword_match(self, query: str) -> List[str]:
        """通過metadata關鍵字匹配找到相關的條層級"""
        matched_articles = []
        
        # 獲取所有條層級的metadata
        article_metadata_map = self.metadata_enhancer.get_article_metadata_map()
        
        # 提取查詢關鍵詞
        query_keywords = self._extract_query_keywords(query)
        
        for article_id, metadata in article_metadata_map.items():
            match_score = self._calculate_metadata_match_score(query_keywords, metadata)
            
            # 如果匹配分數超過閾值，則認為匹配
            if match_score > 0.3:  # 使用配置中的閾值
                matched_articles.append(article_id)
                print(f"📋 條層級 {article_id} 匹配分數: {match_score:.3f}")
        
        return matched_articles
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """提取查詢中的關鍵詞"""
        keywords = []
        
        # 使用jieba分詞
        if jieba:
            words = jieba.analyse.extract_tags(query, topK=10, withWeight=False)
            keywords.extend(words)
        
        # 添加法律同義詞匹配
        query_lower = query.lower()
        for canonical, variants in self.legal_synonyms.items():
            if any(variant in query_lower for variant in variants):
                keywords.append(canonical)
        
        return keywords
    
    def _calculate_metadata_match_score(self, query_keywords: List[str], metadata: Dict[str, Any]) -> float:
        """計算查詢關鍵詞與metadata的匹配分數"""
        total_score = 0.0
        matched_fields = 0
        
        # 1. 檢查法律概念匹配
        legal_concepts = metadata.get("legal_concepts", [])
        for concept in legal_concepts:
            concept_name = concept.get("concept_name", "")
            concept_synonyms = concept.get("synonyms", [])
            importance = concept.get("importance_score", 0.5)
            
            if any(kw in concept_name for kw in query_keywords):
                total_score += importance * 0.3
                matched_fields += 1
            elif any(kw in syn for kw in query_keywords for syn in concept_synonyms):
                total_score += importance * 0.2
                matched_fields += 1
        
        # 2. 檢查語義關鍵詞匹配
        semantic_keywords = metadata.get("semantic_keywords", {})
        keyword_weights = semantic_keywords.get("keyword_weights", {})
        
        for keyword, weight in keyword_weights.items():
            if any(kw in keyword for kw in query_keywords):
                total_score += weight * 0.2
                matched_fields += 1
        
        # 3. 檢查法律領域匹配
        legal_domain = metadata.get("legal_domain", {})
        domain_name = legal_domain.get("legal_domain", "")
        
        if any(kw in domain_name for kw in query_keywords):
            total_score += 0.2
            matched_fields += 1
        
        # 4. 檢查查詢意圖匹配
        query_intent_tags = metadata.get("query_intent_tags", [])
        for intent_tag in query_intent_tags:
            if any(kw in intent_tag for kw in query_keywords):
                total_score += 0.1
                matched_fields += 1
        
        # 正規化分數
        if matched_fields > 0:
            return min(total_score, 1.0)
        
        return 0.0
    
    def _apply_inheritance_strategy(self, candidate_nodes: List[Dict[str, Any]], 
                                   matched_articles: List[str], query: str, config: EnhancedHybridConfig) -> List[Dict[str, Any]]:
        """應用metadata向下繼承策略"""
        # 獲取繼承關係映射
        inheritance_hierarchy = self.metadata_enhancer.get_inheritance_hierarchy()
        
        # 找到匹配條層級的所有子chunks
        inherited_candidates = []
        
        for article_id in matched_articles:
            # 找到該條層級的所有子chunks
            child_chunks = []
            for child_chunk_id, parent_article_id in inheritance_hierarchy.items():
                if parent_article_id == article_id:
                    child_chunks.append(child_chunk_id)
            
            print(f"📋 條層級 {article_id} 有 {len(child_chunks)} 個子chunks")
            
            # 為每個子chunk創建候選節點
            for child_chunk_id in child_chunks:
                # 從FAISS或BM25獲取chunk信息
                chunk_info = self._get_chunk_info_by_id(child_chunk_id)
                if chunk_info:
                    # 添加繼承標記和額外加分
                    chunk_info["inherited_from"] = article_id
                    chunk_info["inheritance_bonus"] = config.inheritance_bonus  # 使用配置中的繼承加分
                    chunk_info["inheritance_boost_factor"] = config.inheritance_boost_factor
                    chunk_info["metadata_match_reason"] = f"繼承自條層級 {article_id}"
                    
                    inherited_candidates.append(chunk_info)
        
        # 合併原有候選和繼承候選
        all_candidates = candidate_nodes + inherited_candidates
        
        print(f"🔄 應用繼承策略：原有候選 {len(candidate_nodes)} + 繼承候選 {len(inherited_candidates)} = 總計 {len(all_candidates)}")
        
        return all_candidates
    
    def _get_chunk_info_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """根據chunk_id獲取chunk信息"""
        # 嘗試從FAISS獲取
        if self.faiss_store.has_vectors():
            # 這裡需要實現根據chunk_id查找的邏輯
            # 暫時返回None，實際實現時需要維護chunk_id到index的映射
            pass
        
        # 嘗試從BM25獲取
        if self.bm25_index.has_index():
            # 同樣需要實現chunk_id查找邏輯
            pass
        
        return None
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """獲取檢索統計信息"""
        stats = {
            "faiss_stats": self.faiss_store.get_stats(),
            "bm25_stats": self.bm25_index.get_stats(),
            "metadata_stats": self.metadata_enhancer.get_enhancement_stats()
        }
        
        # 添加繼承相關統計
        article_metadata_map = self.metadata_enhancer.get_article_metadata_map()
        inheritance_hierarchy = self.metadata_enhancer.get_inheritance_hierarchy()
        
        stats["inheritance_stats"] = {
            "total_articles": len(article_metadata_map),
            "total_inheritance_relations": len(inheritance_hierarchy),
            "avg_children_per_article": len(inheritance_hierarchy) / max(len(article_metadata_map), 1)
        }
        
        return stats
