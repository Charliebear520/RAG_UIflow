"""
法律概念圖檢索
構建法律概念圖，基於概念關係進行檢索
"""

import re
import networkx as nx
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class LegalConcept:
    """法律概念數據結構"""
    concept_id: str
    concept_name: str
    concept_type: str  # 'right', 'obligation', 'exception', 'condition', 'definition'
    content: str
    related_articles: List[str]
    importance_score: float
    semantic_embedding: Optional[np.ndarray] = None


@dataclass
class ConceptRelation:
    """概念關係數據結構"""
    source_concept: str
    target_concept: str
    relation_type: str  # 'implies', 'conflicts', 'includes', 'excludes', 'requires'
    confidence: float
    evidence: str


class LegalConceptGraph:
    """法律概念圖"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concepts: Dict[str, LegalConcept] = {}
        self.relations: List[ConceptRelation] = []
        
        # 法律概念模板
        self.concept_templates = {
            'copyright_rights': {
                'patterns': [
                    r'著作人專有.*?權利',
                    r'著作權人.*?權利',
                    r'專有.*?權利'
                ],
                'type': 'right',
                'importance': 1.0
            },
            'trademark_rights': {
                'patterns': [
                    r'商標權人.*?權利',
                    r'商標.*?專用權',
                    r'註冊商標.*?權利'
                ],
                'type': 'right',
                'importance': 1.0
            },
            'exceptions': {
                'patterns': [
                    r'但.*?不在此限',
                    r'除外.*?規定',
                    r'不適用.*?情形'
                ],
                'type': 'exception',
                'importance': 0.8
            },
            'conditions': {
                'patterns': [
                    r'有下列情形之一',
                    r'如.*?時',
                    r'於.*?情形'
                ],
                'type': 'condition',
                'importance': 0.7
            },
            'obligations': {
                'patterns': [
                    r'應.*?申請',
                    r'必須.*?辦理',
                    r'不得.*?行為'
                ],
                'type': 'obligation',
                'importance': 0.9
            }
        }
        
        # 關係識別模式
        self.relation_patterns = {
            'implies': [
                r'因此.*?應',
                r'所以.*?必須',
                r'故.*?得'
            ],
            'conflicts': [
                r'但.*?不得',
                r'不在此限',
                r'除外'
            ],
            'includes': [
                r'包括.*?情形',
                r'包含.*?項目',
                r'涵蓋.*?範圍'
            ],
            'excludes': [
                r'不包括.*?情形',
                r'排除.*?項目',
                r'不涵蓋.*?範圍'
            ],
            'requires': [
                r'應.*?申請',
                r'必須.*?具備',
                r'需要.*?條件'
            ]
        }
    
    def build_graph(self, documents: List[Dict[str, Any]]) -> None:
        """構建法律概念圖"""
        print("🔨 開始構建法律概念圖...")
        
        # 1. 提取概念
        concepts = self._extract_concepts_from_documents(documents)
        print(f"🔍 提取到 {len(concepts)} 個法律概念")
        
        # 2. 建立概念關係
        relations = self._establish_concept_relations(documents, concepts)
        print(f"🔍 建立 {len(relations)} 個概念關係")
        
        # 3. 構建圖結構
        self._build_graph_structure(concepts, relations)
        print(f"🔍 構建完成，圖包含 {self.graph.number_of_nodes()} 個節點，{self.graph.number_of_edges()} 條邊")
        
        # 4. 計算概念嵌入
        self._compute_concept_embeddings()
        print("🔍 完成概念嵌入計算")
    
    def _extract_concepts_from_documents(self, documents: List[Dict[str, Any]]) -> List[LegalConcept]:
        """從文檔中提取法律概念"""
        concepts = []
        concept_counter = Counter()
        
        for doc in documents:
            content = doc.get('content', '')
            
            for concept_name, concept_config in self.concept_templates.items():
                for pattern in concept_config['patterns']:
                    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                    
                    for match in matches:
                        concept_id = f"{concept_name}_{len(concepts)}"
                        
                        # 提取相關法條
                        related_articles = self._extract_related_articles(content, match.start(), match.end())
                        
                        concept = LegalConcept(
                            concept_id=concept_id,
                            concept_name=concept_name,
                            concept_type=concept_config['type'],
                            content=match.group().strip(),
                            related_articles=related_articles,
                            importance_score=concept_config['importance']
                        )
                        
                        concepts.append(concept)
                        concept_counter[concept_name] += 1
        
        # 去重並合併相似概念
        unique_concepts = self._deduplicate_concepts(concepts)
        
        # 更新概念字典
        for concept in unique_concepts:
            self.concepts[concept.concept_id] = concept
        
        return unique_concepts
    
    def _extract_related_articles(self, content: str, start_pos: int, end_pos: int) -> List[str]:
        """提取相關法條"""
        # 在概念前後一定範圍內查找法條引用
        search_start = max(0, start_pos - 200)
        search_end = min(len(content), end_pos + 200)
        search_text = content[search_start:search_end]
        
        article_pattern = r'第\s*(\d+)\s*條'
        articles = re.findall(article_pattern, search_text)
        return [f"第{article}條" for article in articles]
    
    def _deduplicate_concepts(self, concepts: List[LegalConcept]) -> List[LegalConcept]:
        """去重並合併相似概念"""
        unique_concepts = []
        seen_contents = set()
        
        for concept in concepts:
            content_key = concept.content.lower().strip()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_concepts.append(concept)
            else:
                # 合併到已存在的概念
                for existing in unique_concepts:
                    if existing.content.lower().strip() == content_key:
                        existing.related_articles.extend(concept.related_articles)
                        existing.related_articles = list(set(existing.related_articles))
                        break
        
        return unique_concepts
    
    def _establish_concept_relations(self, documents: List[Dict[str, Any]], 
                                   concepts: List[LegalConcept]) -> List[ConceptRelation]:
        """建立概念關係"""
        relations = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            for relation_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                    
                    for match in matches:
                        # 在匹配的上下文附近查找相關概念
                        context_start = max(0, match.start() - 300)
                        context_end = min(len(content), match.end() + 300)
                        context = content[context_start:context_end]
                        
                        # 找到上下文中的概念
                        context_concepts = self._find_concepts_in_context(context, concepts)
                        
                        if len(context_concepts) >= 2:
                            # 建立關係
                            for i in range(len(context_concepts) - 1):
                                relation = ConceptRelation(
                                    source_concept=context_concepts[i].concept_id,
                                    target_concept=context_concepts[i + 1].concept_id,
                                    relation_type=relation_type,
                                    confidence=self._calculate_relation_confidence(match.group(), context),
                                    evidence=match.group()
                                )
                                relations.append(relation)
        
        # 去重並合併相似關係
        unique_relations = self._deduplicate_relations(relations)
        
        self.relations = unique_relations
        return unique_relations
    
    def _find_concepts_in_context(self, context: str, concepts: List[LegalConcept]) -> List[LegalConcept]:
        """在上下文中查找概念"""
        found_concepts = []
        
        for concept in concepts:
            if concept.content in context:
                found_concepts.append(concept)
        
        return found_concepts
    
    def _calculate_relation_confidence(self, evidence: str, context: str) -> float:
        """計算關係置信度"""
        # 基於證據的長度和上下文相關性
        evidence_length = len(evidence)
        context_relevance = len(set(context.split()) & set(evidence.split())) / len(set(context.split()))
        
        confidence = min(1.0, evidence_length / 50 + context_relevance * 0.5)
        return confidence
    
    def _deduplicate_relations(self, relations: List[ConceptRelation]) -> List[ConceptRelation]:
        """去重並合併相似關係"""
        unique_relations = []
        seen_relations = set()
        
        for relation in relations:
            relation_key = (relation.source_concept, relation.target_concept, relation.relation_type)
            if relation_key not in seen_relations:
                seen_relations.add(relation_key)
                unique_relations.append(relation)
            else:
                # 合併到已存在的關係
                for existing in unique_relations:
                    if (existing.source_concept == relation.source_concept and
                        existing.target_concept == relation.target_concept and
                        existing.relation_type == relation.relation_type):
                        # 取較高的置信度
                        existing.confidence = max(existing.confidence, relation.confidence)
                        break
        
        return unique_relations
    
    def _build_graph_structure(self, concepts: List[LegalConcept], 
                             relations: List[ConceptRelation]) -> None:
        """構建圖結構"""
        # 添加節點
        for concept in concepts:
            self.graph.add_node(
                concept.concept_id,
                concept_name=concept.concept_name,
                concept_type=concept.concept_type,
                content=concept.content,
                related_articles=concept.related_articles,
                importance_score=concept.importance_score
            )
        
        # 添加邊
        for relation in relations:
            if (relation.source_concept in self.graph.nodes and 
                relation.target_concept in self.graph.nodes):
                self.graph.add_edge(
                    relation.source_concept,
                    relation.target_concept,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence,
                    evidence=relation.evidence
                )
    
    def _compute_concept_embeddings(self) -> None:
        """計算概念嵌入"""
        # 準備概念文本
        concept_texts = []
        concept_ids = []
        
        for concept_id, concept in self.concepts.items():
            # 組合概念內容和相關法條
            combined_text = f"{concept.content} {' '.join(concept.related_articles)}"
            concept_texts.append(combined_text)
            concept_ids.append(concept_id)
        
        # 使用TF-IDF計算嵌入
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        embeddings = vectorizer.fit_transform(concept_texts).toarray()
        
        # 更新概念嵌入
        for i, concept_id in enumerate(concept_ids):
            self.concepts[concept_id].semantic_embedding = embeddings[i]


class LegalConceptGraphRetrieval:
    """法律概念圖檢索"""
    
    def __init__(self, concept_graph: LegalConceptGraph):
        self.concept_graph = concept_graph
        self.graph = concept_graph.graph
        self.concepts = concept_graph.concepts
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """基於概念圖的檢索"""
        print(f"🔍 開始概念圖檢索，查詢: '{query}'")
        
        # 1. 查詢概念化
        query_concepts = self._conceptualize_query(query)
        print(f"🔍 查詢概念化結果: {[c.concept_name for c in query_concepts]}")
        
        # 2. 概念圖推理
        reasoning_paths = self._graph_reasoning(query_concepts)
        print(f"🔍 推理路徑數量: {len(reasoning_paths)}")
        
        # 3. 基於推理路徑檢索
        results = self._retrieve_by_reasoning_paths(reasoning_paths, k)
        print(f"🔍 檢索結果數量: {len(results)}")
        
        return results
    
    def _conceptualize_query(self, query: str) -> List[LegalConcept]:
        """查詢概念化"""
        query_concepts = []
        
        # 直接匹配概念內容
        for concept_id, concept in self.concepts.items():
            if self._is_concept_relevant(query, concept):
                query_concepts.append(concept)
        
        # 基於概念名稱匹配
        query_lower = query.lower()
        for concept_id, concept in self.concepts.items():
            if concept.concept_name in query_lower or any(
                keyword in query_lower for keyword in concept.concept_name.split('_')
            ):
                if concept not in query_concepts:
                    query_concepts.append(concept)
        
        return query_concepts
    
    def _is_concept_relevant(self, query: str, concept: LegalConcept) -> bool:
        """判斷概念是否與查詢相關"""
        query_lower = query.lower()
        concept_content_lower = concept.content.lower()
        
        # 直接字符串匹配
        if any(word in concept_content_lower for word in query_lower.split()):
            return True
        
        # 法律關鍵詞匹配
        legal_keywords = ['著作權', '授權', '翻譯', '出版', '權利', '法律', '條文', '法規']
        query_keywords = [kw for kw in legal_keywords if kw in query_lower]
        concept_keywords = [kw for kw in legal_keywords if kw in concept_content_lower]
        
        if query_keywords and concept_keywords:
            overlap = set(query_keywords) & set(concept_keywords)
            if overlap:
                return True
        
        # 詞彙重疊度計算（放寬閾值）
        query_words = set(query_lower.split())
        concept_words = set(concept_content_lower.split())
        overlap = len(query_words & concept_words)
        relevance_score = overlap / max(len(query_words), 1)
        
        return relevance_score > 0.1  # 降低閾值
    
    def _graph_reasoning(self, query_concepts: List[LegalConcept]) -> List[List[str]]:
        """概念圖推理"""
        reasoning_paths = []
        
        for concept in query_concepts:
            # 從查詢概念開始的推理路徑
            paths = self._find_reasoning_paths(concept.concept_id, max_depth=3)
            reasoning_paths.extend(paths)
        
        # 去重並排序
        unique_paths = []
        seen_paths = set()
        
        for path in reasoning_paths:
            path_key = tuple(path)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_paths.append(path)
        
        # 按路徑長度和重要性排序
        unique_paths.sort(key=lambda x: (len(x), self._calculate_path_importance(x)), reverse=True)
        
        return unique_paths[:10]  # 返回前10個最相關的路徑
    
    def _find_reasoning_paths(self, start_concept: str, max_depth: int = 3) -> List[List[str]]:
        """找到推理路徑"""
        paths = []
        
        def dfs(current_concept: str, current_path: List[str], depth: int):
            if depth > max_depth:
                return
            
            current_path.append(current_concept)
            
            if len(current_path) > 1:  # 至少包含兩個概念
                paths.append(current_path.copy())
            
            # 遍歷鄰居節點
            for neighbor in self.graph.neighbors(current_concept):
                if neighbor not in current_path:  # 避免循環
                    dfs(neighbor, current_path, depth + 1)
            
            current_path.pop()
        
        dfs(start_concept, [], 0)
        return paths
    
    def _calculate_path_importance(self, path: List[str]) -> float:
        """計算路徑重要性"""
        total_importance = 0.0
        
        for concept_id in path:
            if concept_id in self.concepts:
                total_importance += self.concepts[concept_id].importance_score
        
        # 考慮路徑長度
        length_penalty = 1.0 / len(path) if len(path) > 0 else 0.0
        
        return total_importance * length_penalty
    
    def _retrieve_by_reasoning_paths(self, reasoning_paths: List[List[str]], 
                                   k: int) -> List[Dict[str, Any]]:
        """基於推理路徑檢索"""
        results = []
        
        for path in reasoning_paths:
            if len(results) >= k:
                break
            
            # 收集路徑中所有概念的相關內容
            path_results = self._collect_path_results(path)
            
            for result in path_results:
                if len(results) >= k:
                    break
                
                # 計算推理分數
                reasoning_score = self._calculate_reasoning_score(path, result)
                result['reasoning_score'] = reasoning_score
                result['reasoning_path'] = path
                
                results.append(result)
        
        # 按推理分數排序
        results.sort(key=lambda x: x['reasoning_score'], reverse=True)
        
        return results[:k]
    
    def _collect_path_results(self, path: List[str]) -> List[Dict[str, Any]]:
        """收集路徑結果"""
        results = []
        
        for concept_id in path:
            if concept_id in self.concepts:
                concept = self.concepts[concept_id]
                
                result = {
                    'content': concept.content,
                    'concept_id': concept_id,
                    'concept_name': concept.concept_name,
                    'concept_type': concept.concept_type,
                    'related_articles': concept.related_articles,
                    'importance_score': concept.importance_score,
                    'metadata': {
                        'strategy': 'concept_graph_retrieval',
                        'concept_based': True
                    }
                }
                
                results.append(result)
        
        return results
    
    def _calculate_reasoning_score(self, path: List[str], result: Dict[str, Any]) -> float:
        """計算推理分數"""
        # 基礎分數：概念重要性
        base_score = result.get('importance_score', 0.0)
        
        # 路徑分數：路徑的連貫性和長度
        path_score = self._calculate_path_coherence(path)
        
        # 組合分數
        reasoning_score = base_score * 0.6 + path_score * 0.4
        
        return reasoning_score
    
    def _calculate_path_coherence(self, path: List[str]) -> float:
        """計算路徑連貫性"""
        if len(path) < 2:
            return 0.0
        
        coherence_score = 0.0
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            if self.graph.has_edge(source, target):
                edge_data = self.graph[source][target]
                confidence = edge_data.get('confidence', 0.5)
                coherence_score += confidence
        
        # 平均連貫性
        avg_coherence = coherence_score / max(len(path) - 1, 1)
        
        return avg_coherence
