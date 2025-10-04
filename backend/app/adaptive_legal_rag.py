"""
自適應法律RAG系統
根據查詢特徵動態選擇最優的檢索策略組合
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


class QueryType(Enum):
    """查詢類型枚舉"""
    CONCEPTUAL = "conceptual"      # 概念性查詢
    NORMATIVE = "normative"        # 規範性查詢
    PROCEDURAL = "procedural"      # 程序性查詢
    EXPLICIT_ARTICLE = "explicit_article"  # 明確法條查詢
    COMPARATIVE = "comparative"    # 比較性查詢
    CASE_BASED = "case_based"      # 案例性查詢


@dataclass
class QueryAnalysis:
    """查詢分析結果"""
    query_type: QueryType
    confidence: float
    key_concepts: List[str]
    legal_articles: List[str]
    complexity_score: float
    semantic_features: Dict[str, Any]
    recommended_strategies: List[str]


@dataclass
class RetrievalStrategy:
    """檢索策略配置"""
    name: str
    weight: float
    parameters: Dict[str, Any]
    applicable_query_types: List[QueryType]
    performance_history: List[float]


class QueryAnalyzer:
    """查詢分析器"""
    
    def __init__(self):
        # 查詢類型識別模式
        self.query_patterns = {
            QueryType.CONCEPTUAL: [
                r'什麼是.*?',
                r'.*?的定義',
                r'.*?是指.*?',
                r'.*?概念.*?',
                r'何謂.*?'
            ],
            QueryType.NORMATIVE: [
                r'第\s*\d+\s*條',
                r'.*?規定.*?',
                r'.*?禁止.*?',
                r'.*?應.*?',
                r'.*?不得.*?'
            ],
            QueryType.PROCEDURAL: [
                r'如何.*?',
                r'.*?程序.*?',
                r'.*?申請.*?',
                r'.*?辦理.*?',
                r'.*?流程.*?'
            ],
            QueryType.EXPLICIT_ARTICLE: [
                r'第\s*\d+\s*條.*?',
                r'article\s*\d+',
                r'條文\s*\d+'
            ],
            QueryType.COMPARATIVE: [
                r'.*?與.*?的差別',
                r'.*?比較.*?',
                r'.*?區別.*?',
                r'.*?差異.*?'
            ],
            QueryType.CASE_BASED: [
                r'.*?案例.*?',
                r'.*?情形.*?',
                r'.*?情況.*?',
                r'.*?例子.*?'
            ]
        }
        
        # 法律關鍵詞
        self.legal_keywords = {
            'copyright': ['著作權', '版權', '重製', '改作', '散布', '合理使用'],
            'trademark': ['商標', '註冊', '仿冒', '混淆', '專用權'],
            'patent': ['專利', '發明', '新穎性', '進步性', '產業利用性'],
            'civil': ['民法', '契約', '損害賠償', '侵權'],
            'criminal': ['刑法', '犯罪', '刑罰', '罰金']
        }
        
        # 複雜度指標
        self.complexity_indicators = {
            'high': ['比較', '分析', '評估', '綜合', '複雜', '多重'],
            'medium': ['關係', '影響', '適用', '範圍'],
            'low': ['定義', '什麼', '如何', '是否']
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """分析查詢"""
        print(f"🔍 開始分析查詢: '{query}'")
        
        # 1. 識別查詢類型
        query_type, confidence = self._identify_query_type(query)
        print(f"🔍 識別查詢類型: {query_type.value} (置信度: {confidence:.3f})")
        
        # 2. 提取關鍵概念
        key_concepts = self._extract_key_concepts(query)
        print(f"🔍 提取關鍵概念: {key_concepts}")
        
        # 3. 提取法律條文
        legal_articles = self._extract_legal_articles(query)
        print(f"🔍 提取法律條文: {legal_articles}")
        
        # 4. 計算複雜度
        complexity_score = self._calculate_complexity(query)
        print(f"🔍 計算複雜度: {complexity_score:.3f}")
        
        # 5. 提取語義特徵
        semantic_features = self._extract_semantic_features(query)
        print(f"🔍 提取語義特徵: {len(semantic_features)} 個特徵")
        
        # 6. 推薦策略
        recommended_strategies = self._recommend_strategies(
            query_type, complexity_score, key_concepts, legal_articles
        )
        print(f"🔍 推薦策略: {recommended_strategies}")
        
        return QueryAnalysis(
            query_type=query_type,
            confidence=confidence,
            key_concepts=key_concepts,
            legal_articles=legal_articles,
            complexity_score=complexity_score,
            semantic_features=semantic_features,
            recommended_strategies=recommended_strategies
        )
    
    def _identify_query_type(self, query: str) -> Tuple[QueryType, float]:
        """識別查詢類型"""
        type_scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                score += len(matches) * 0.5
            
            # 特殊處理明確法條查詢
            if query_type == QueryType.EXPLICIT_ARTICLE:
                article_matches = re.findall(r'第\s*\d+\s*條', query)
                score += len(article_matches) * 1.0
            
            type_scores[query_type] = score
        
        # 找到最高分數的類型
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            max_score = best_type[1]
            confidence = min(1.0, max_score / 2.0)  # 歸一化到0-1
            return best_type[0], confidence
        
        return QueryType.CONCEPTUAL, 0.5  # 默認類型
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """提取關鍵概念"""
        concepts = []
        
        # 從法律關鍵詞中提取
        for domain, keywords in self.legal_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    concepts.append(keyword)
        
        # 從查詢中提取法律相關詞彙
        legal_terms = re.findall(r'[著作商專民刑][權法]|[權法]|[重製改作散布授權合理使用]', query)
        concepts.extend(legal_terms)
        
        return list(set(concepts))
    
    def _extract_legal_articles(self, query: str) -> List[str]:
        """提取法律條文"""
        articles = []
        
        # 提取第X條格式
        article_matches = re.findall(r'第\s*(\d+)\s*條', query)
        articles.extend([f"第{article}條" for article in article_matches])
        
        # 提取article X格式
        article_en_matches = re.findall(r'article\s*(\d+)', query, re.IGNORECASE)
        articles.extend([f"Article {article}" for article in article_en_matches])
        
        return articles
    
    def _calculate_complexity(self, query: str) -> float:
        """計算查詢複雜度"""
        complexity_score = 0.0
        
        # 基於複雜度指標
        for level, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in query:
                    if level == 'high':
                        complexity_score += 0.8
                    elif level == 'medium':
                        complexity_score += 0.5
                    else:
                        complexity_score += 0.2
        
        # 基於查詢長度
        length_factor = min(1.0, len(query) / 100)
        complexity_score += length_factor * 0.3
        
        # 基於關鍵概念數量
        concept_count = len(self._extract_key_concepts(query))
        concept_factor = min(1.0, concept_count / 5)
        complexity_score += concept_factor * 0.2
        
        return min(1.0, complexity_score)
    
    def _extract_semantic_features(self, query: str) -> Dict[str, Any]:
        """提取語義特徵"""
        features = {
            'length': len(query),
            'word_count': len(query.split()),
            'has_question_mark': '?' in query or '？' in query,
            'has_comparison': any(word in query for word in ['比較', '與', '差別', '區別']),
            'has_condition': any(word in query for word in ['如果', '當', '情形', '情況']),
            'has_negation': any(word in query for word in ['不', '非', '無', '禁止']),
            'domain_keywords': [],
            'sentence_structure': self._analyze_sentence_structure(query)
        }
        
        # 識別領域關鍵詞
        for domain, keywords in self.legal_keywords.items():
            domain_matches = [kw for kw in keywords if kw in query]
            if domain_matches:
                features['domain_keywords'].append({
                    'domain': domain,
                    'keywords': domain_matches
                })
        
        return features
    
    def _analyze_sentence_structure(self, query: str) -> str:
        """分析句子結構"""
        if '？' in query or '?' in query:
            return 'question'
        elif '，' in query or ',' in query:
            return 'compound'
        elif any(word in query for word in ['和', '與', '及']):
            return 'conjunction'
        else:
            return 'simple'
    
    def _recommend_strategies(self, query_type: QueryType, complexity_score: float,
                            key_concepts: List[str], legal_articles: List[str]) -> List[str]:
        """推薦檢索策略"""
        strategies = []
        
        # 基於查詢類型推薦策略
        if query_type == QueryType.CONCEPTUAL:
            strategies.extend(['concept_graph', 'semantic_search', 'multi_modal'])
        elif query_type == QueryType.NORMATIVE:
            strategies.extend(['hybrid_rag', 'structured_search', 'legal_semantic'])
        elif query_type == QueryType.PROCEDURAL:
            strategies.extend(['hierarchical', 'sequential_search'])
        elif query_type == QueryType.EXPLICIT_ARTICLE:
            strategies.extend(['exact_match', 'structured_search'])
        elif query_type == QueryType.COMPARATIVE:
            strategies.extend(['multi_modal', 'concept_graph', 'comparative_analysis'])
        elif query_type == QueryType.CASE_BASED:
            strategies.extend(['semantic_search', 'case_matching'])
        
        # 基於複雜度調整策略
        if complexity_score > 0.7:
            strategies.extend(['multi_strategy', 'adaptive_fusion'])
        
        # 基於關鍵概念調整
        if len(key_concepts) > 3:
            strategies.append('concept_expansion')
        
        # 基於法律條文調整
        if legal_articles:
            strategies.extend(['article_focused', 'legal_structure'])
        
        return list(set(strategies))


class StrategySelector:
    """策略選擇器"""
    
    def __init__(self):
        self.strategies = {
            'vector_search': RetrievalStrategy(
                name='vector_search',
                weight=1.0,
                parameters={'k': 5, 'similarity_threshold': 0.7},
                applicable_query_types=[QueryType.CONCEPTUAL, QueryType.CASE_BASED],
                performance_history=[]
            ),
            'hybrid_rag': RetrievalStrategy(
                name='hybrid_rag',
                weight=1.2,
                parameters={'alpha': 0.7, 'legal_boost': 0.3},
                applicable_query_types=[QueryType.NORMATIVE, QueryType.EXPLICIT_ARTICLE],
                performance_history=[]
            ),
            'concept_graph': RetrievalStrategy(
                name='concept_graph',
                weight=1.5,
                parameters={'reasoning_weight': 0.8},
                applicable_query_types=[QueryType.CONCEPTUAL, QueryType.COMPARATIVE],
                performance_history=[]
            ),
            'multi_modal': RetrievalStrategy(
                name='multi_modal',
                weight=1.3,
                parameters={'text_weight': 0.6, 'structure_weight': 0.4},
                applicable_query_types=[QueryType.COMPARATIVE, QueryType.PROCEDURAL],
                performance_history=[]
            ),
            'hierarchical': RetrievalStrategy(
                name='hierarchical',
                weight=1.1,
                parameters={'level_weights': [1.0, 0.8, 0.6]},
                applicable_query_types=[QueryType.PROCEDURAL, QueryType.NORMATIVE],
                performance_history=[]
            ),
            'semantic_search': RetrievalStrategy(
                name='semantic_search',
                weight=1.0,
                parameters={'semantic_threshold': 0.6},
                applicable_query_types=[QueryType.CONCEPTUAL, QueryType.CASE_BASED],
                performance_history=[]
            )
        }
    
    def select_strategies(self, query_analysis: QueryAnalysis, 
                         available_strategies: Dict[str, Any]) -> Dict[str, Any]:
        """選擇檢索策略"""
        print(f"🎯 開始策略選擇，推薦策略: {query_analysis.recommended_strategies}")
        
        selected_strategies = {}
        
        # 1. 基於推薦策略選擇
        for strategy_name in query_analysis.recommended_strategies:
            if strategy_name in available_strategies:
                strategy = self.strategies.get(strategy_name)
                if strategy and self._is_strategy_applicable(strategy, query_analysis):
                    selected_strategies[strategy_name] = {
                        'strategy': available_strategies[strategy_name],
                        'weight': strategy.weight,
                        'parameters': strategy.parameters.copy()
                    }
        
        # 2. 確保至少有一個策略
        if not selected_strategies:
            # 默認選擇策略
            default_strategy = 'hybrid_rag' if 'hybrid_rag' in available_strategies else 'vector_search'
            selected_strategies[default_strategy] = {
                'strategy': available_strategies[default_strategy],
                'weight': 1.0,
                'parameters': {}
            }
        
        # 3. 根據查詢複雜度調整策略數量
        if query_analysis.complexity_score > 0.7 and len(selected_strategies) == 1:
            # 為複雜查詢添加額外策略
            additional_strategy = self._select_additional_strategy(query_analysis, available_strategies)
            if additional_strategy:
                selected_strategies.update(additional_strategy)
        
        print(f"🎯 最終選擇策略: {list(selected_strategies.keys())}")
        
        return selected_strategies
    
    def _is_strategy_applicable(self, strategy: RetrievalStrategy, 
                               query_analysis: QueryAnalysis) -> bool:
        """判斷策略是否適用"""
        return query_analysis.query_type in strategy.applicable_query_types
    
    def _select_additional_strategy(self, query_analysis: QueryAnalysis,
                                   available_strategies: Dict[str, Any]) -> Dict[str, Any]:
        """為複雜查詢選擇額外策略"""
        additional = {}
        
        # 基於查詢類型選擇互補策略
        if query_analysis.query_type == QueryType.CONCEPTUAL:
            if 'semantic_search' in available_strategies:
                additional['semantic_search'] = {
                    'strategy': available_strategies['semantic_search'],
                    'weight': 0.8,
                    'parameters': {}
                }
        elif query_analysis.query_type == QueryType.NORMATIVE:
            if 'hierarchical' in available_strategies:
                additional['hierarchical'] = {
                    'strategy': available_strategies['hierarchical'],
                    'weight': 0.9,
                    'parameters': {}
                }
        
        return additional


class AdaptiveLegalRAG:
    """自適應法律RAG系統"""
    
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.strategy_selector = StrategySelector()
        
        # 檢索策略實例
        self.retrieval_strategies = {}
        
        # 性能監控
        self.performance_monitor = PerformanceMonitor()
    
    def register_strategy(self, name: str, strategy_instance: Any) -> None:
        """註冊檢索策略"""
        self.retrieval_strategies[name] = strategy_instance
        print(f"✅ 註冊策略: {name}")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """自適應檢索"""
        print(f"🚀 開始自適應檢索，查詢: '{query}'")
        
        # 1. 查詢分析
        query_analysis = self.query_analyzer.analyze_query(query)
        
        # 2. 策略選擇
        selected_strategies = self.strategy_selector.select_strategies(
            query_analysis, self.retrieval_strategies
        )
        
        # 3. 多策略檢索
        strategy_results = {}
        for strategy_name, strategy_config in selected_strategies.items():
            try:
                strategy_instance = strategy_config['strategy']
                parameters = strategy_config['parameters']
                parameters['k'] = k
                
                print(f"🔍 執行策略: {strategy_name}")
                results = strategy_instance.retrieve(query, **parameters)
                strategy_results[strategy_name] = {
                    'results': results,
                    'weight': strategy_config['weight']
                }
            except Exception as e:
                print(f"❌ 策略 {strategy_name} 執行失敗: {e}")
                continue
        
        # 4. 結果融合
        if strategy_results:
            fused_results = self._fuse_results(strategy_results, query_analysis)
            
            # 5. 性能監控
            self.performance_monitor.record_retrieval(query, query_analysis, fused_results)
            
            return fused_results
        else:
            print("❌ 沒有可用的檢索策略")
            return []
    
    def _fuse_results(self, strategy_results: Dict[str, Dict[str, Any]], 
                     query_analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """融合多策略結果"""
        print(f"🔀 開始結果融合，策略數量: {len(strategy_results)}")
        
        # 收集所有結果
        all_results = []
        result_scores = {}
        
        for strategy_name, strategy_data in strategy_results.items():
            results = strategy_data['results']
            weight = strategy_data['weight']
            
            for result in results:
                # 計算加權分數
                base_score = result.get('score', 0.0)
                weighted_score = base_score * weight
                
                # 根據查詢類型調整分數
                adjusted_score = self._adjust_score_by_query_type(
                    weighted_score, result, query_analysis
                )
                
                result_id = self._generate_result_id(result)
                if result_id not in result_scores:
                    result_scores[result_id] = {
                        'result': result,
                        'scores': [],
                        'strategies': []
                    }
                
                result_scores[result_id]['scores'].append(adjusted_score)
                result_scores[result_id]['strategies'].append(strategy_name)
        
        # 融合分數
        fused_results = []
        for result_id, data in result_scores.items():
            result = data['result']
            scores = data['scores']
            strategies = data['strategies']
            
            # 計算融合分數（加權平均）
            if scores:
                fused_score = sum(scores) / len(scores)
                
                # 添加融合元數據
                result['fused_score'] = fused_score
                result['contributing_strategies'] = strategies
                result['strategy_count'] = len(strategies)
                result['metadata']['adaptive_fusion'] = True
                
                fused_results.append(result)
        
        # 按融合分數排序
        fused_results.sort(key=lambda x: x['fused_score'], reverse=True)
        
        print(f"🔀 融合完成，結果數量: {len(fused_results)}")
        
        return fused_results
    
    def _adjust_score_by_query_type(self, score: float, result: Dict[str, Any], 
                                   query_analysis: QueryAnalysis) -> float:
        """根據查詢類型調整分數"""
        adjusted_score = score
        
        # 基於查詢類型調整
        if query_analysis.query_type == QueryType.CONCEPTUAL:
            # 概念性查詢：重視概念匹配
            if result.get('metadata', {}).get('concept_based', False):
                adjusted_score *= 1.2
        
        elif query_analysis.query_type == QueryType.NORMATIVE:
            # 規範性查詢：重視法條匹配
            if any(article in result.get('content', '') for article in query_analysis.legal_articles):
                adjusted_score *= 1.3
        
        elif query_analysis.query_type == QueryType.EXPLICIT_ARTICLE:
            # 明確法條查詢：重視精確匹配
            if any(article in result.get('content', '') for article in query_analysis.legal_articles):
                adjusted_score *= 1.5
        
        # 基於複雜度調整
        if query_analysis.complexity_score > 0.7:
            # 複雜查詢：重視多策略支持
            if result.get('strategy_count', 1) > 1:
                adjusted_score *= 1.1
        
        return adjusted_score
    
    def _generate_result_id(self, result: Dict[str, Any]) -> str:
        """生成結果ID"""
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        
        # 基於內容和元數據生成ID
        id_parts = [
            content[:50],  # 內容前50字符
            metadata.get('strategy', 'unknown'),
            str(metadata.get('chunk_index', 0))
        ]
        
        return '_'.join(id_parts)


class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self):
        self.retrieval_history = []
        self.strategy_performance = {}
    
    def record_retrieval(self, query: str, query_analysis: QueryAnalysis, 
                        results: List[Dict[str, Any]]) -> None:
        """記錄檢索性能"""
        record = {
            'query': query,
            'query_type': query_analysis.query_type.value,
            'complexity': query_analysis.complexity_score,
            'result_count': len(results),
            'timestamp': self._get_timestamp()
        }
        
        self.retrieval_history.append(record)
        
        # 更新策略性能
        for result in results:
            strategies = result.get('contributing_strategies', [])
            for strategy in strategies:
                if strategy not in self.strategy_performance:
                    self.strategy_performance[strategy] = []
                
                self.strategy_performance[strategy].append(result.get('fused_score', 0.0))
    
    def get_strategy_performance(self) -> Dict[str, float]:
        """獲取策略性能統計"""
        performance = {}
        
        for strategy, scores in self.strategy_performance.items():
            if scores:
                performance[strategy] = {
                    'avg_score': sum(scores) / len(scores),
                    'usage_count': len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores)
                }
        
        return performance
    
    def _get_timestamp(self) -> str:
        """獲取時間戳"""
        import datetime
        return datetime.datetime.now().isoformat()
