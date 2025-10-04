"""
動態概念學習系統 - 自動學習新的法律概念和關係
"""

import json
import re
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import jieba


@dataclass
class LearningPattern:
    """學習模式"""
    pattern_id: str
    pattern_type: str  # 概念提取、關係識別、推理規則等
    pattern_regex: str
    confidence_threshold: float
    success_count: int = 0
    failure_count: int = 0
    last_updated: str = ""


@dataclass
class LearnedConcept:
    """學習到的概念"""
    concept_id: str
    concept_name: str
    concept_type: str
    legal_domain: str
    source_queries: List[str]  # 來源查詢
    related_articles: List[str]
    confidence: float
    learning_method: str  # 自動學習、人工驗證等
    created_at: str
    updated_at: str


@dataclass
class LearnedRelation:
    """學習到的關係"""
    relation_id: str
    source_concept: str
    target_concept: str
    relation_type: str
    confidence: float
    evidence_queries: List[str]
    learning_method: str
    created_at: str


class DynamicConceptLearningSystem:
    """動態概念學習系統"""
    
    def __init__(self):
        self.learning_patterns = self._initialize_learning_patterns()
        self.learned_concepts = {}
        self.learned_relations = []
        self.query_history = []
        self.feedback_history = []
        self.concept_similarity_matrix = None
        
    def _initialize_learning_patterns(self) -> List[LearningPattern]:
        """初始化學習模式"""
        return [
            # 概念提取模式
            LearningPattern(
                pattern_id="concept_extraction_1",
                pattern_type="概念提取",
                pattern_regex=r"([^，。；：！？]+)(?:屬於|是|為)([^，。；：！？]*)(?:的一種|類型|概念)",
                confidence_threshold=0.7
            ),
            LearningPattern(
                pattern_id="concept_extraction_2",
                pattern_type="概念提取",
                pattern_regex=r"([^，。；：！？]+)(?:包括|含|涵蓋)([^，。；：！？]*)",
                confidence_threshold=0.6
            ),
            
            # 關係識別模式
            LearningPattern(
                pattern_id="relation_identification_1",
                pattern_type="關係識別",
                pattern_regex=r"([^，。；：！？]+)(?:需要|必須|應)([^，。；：！？]*)",
                confidence_threshold=0.8
            ),
            LearningPattern(
                pattern_id="relation_identification_2",
                pattern_type="關係識別",
                pattern_regex=r"([^，。；：！？]+)(?:適用於|適用)([^，。；：！？]*)",
                confidence_threshold=0.7
            ),
            
            # 推理規則模式
            LearningPattern(
                pattern_id="reasoning_rule_1",
                pattern_type="推理規則",
                pattern_regex=r"([^，。；：！？]+)(?:等同於|相當於|等同)([^，。；：！？]*)",
                confidence_threshold=0.9
            )
        ]
    
    def learn_from_query_feedback(self, query: str, retrieved_results: List[Dict], 
                                user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """從查詢反饋中學習"""
        learning_result = {
            "new_concepts": [],
            "new_relations": [],
            "updated_patterns": [],
            "learning_insights": []
        }
        
        print(f"🧠 開始從查詢反饋中學習: '{query}'")
        
        # 記錄查詢歷史
        self.query_history.append({
            "query": query,
            "results": retrieved_results,
            "feedback": user_feedback,
            "timestamp": self._get_current_timestamp()
        })
        
        # 分析用戶反饋
        if user_feedback.get("relevant_results"):
            self._learn_from_relevant_results(query, user_feedback["relevant_results"], learning_result)
        
        if user_feedback.get("missing_concepts"):
            self._learn_missing_concepts(query, user_feedback["missing_concepts"], learning_result)
        
        if user_feedback.get("concept_mappings"):
            self._learn_concept_mappings(query, user_feedback["concept_mappings"], learning_result)
        
        # 更新學習模式
        self._update_learning_patterns(query, retrieved_results, user_feedback, learning_result)
        
        # 生成學習洞察
        self._generate_learning_insights(learning_result)
        
        print(f"✅ 學習完成: {len(learning_result['new_concepts'])}個新概念, {len(learning_result['new_relations'])}個新關係")
        
        return learning_result
    
    def _learn_from_relevant_results(self, query: str, relevant_results: List[Dict], 
                                   learning_result: Dict[str, Any]):
        """從相關結果中學習"""
        for result in relevant_results:
            content = result.get('content', '')
            
            # 提取新的法律概念
            new_concepts = self._extract_concepts_from_content(content, query)
            learning_result["new_concepts"].extend(new_concepts)
            
            # 提取概念關係
            new_relations = self._extract_relations_from_content(content, query)
            learning_result["new_relations"].extend(new_relations)
    
    def _learn_missing_concepts(self, query: str, missing_concepts: List[str], 
                              learning_result: Dict[str, Any]):
        """學習缺失的概念"""
        for concept_name in missing_concepts:
            # 分析概念與查詢的關係
            concept_info = self._analyze_missing_concept(query, concept_name)
            
            if concept_info:
                learned_concept = LearnedConcept(
                    concept_id=f"learned_{concept_name}_{len(self.learned_concepts)}",
                    concept_name=concept_name,
                    concept_type=concept_info.get("type", "未知"),
                    legal_domain=concept_info.get("domain", "其他"),
                    source_queries=[query],
                    related_articles=concept_info.get("articles", []),
                    confidence=concept_info.get("confidence", 0.5),
                    learning_method="缺失概念學習",
                    created_at=self._get_current_timestamp(),
                    updated_at=self._get_current_timestamp()
                )
                
                self.learned_concepts[learned_concept.concept_id] = learned_concept
                learning_result["new_concepts"].append(asdict(learned_concept))
    
    def _learn_concept_mappings(self, query: str, concept_mappings: List[Dict], 
                              learning_result: Dict[str, Any]):
        """學習概念映射"""
        for mapping in concept_mappings:
            source_term = mapping.get("source")
            target_concept = mapping.get("target")
            confidence = mapping.get("confidence", 0.8)
            
            if source_term and target_concept:
                # 創建新的推理規則
                new_relation = LearnedRelation(
                    relation_id=f"mapping_{source_term}_{target_concept}_{len(self.learned_relations)}",
                    source_concept=source_term,
                    target_concept=target_concept,
                    relation_type="概念映射",
                    confidence=confidence,
                    evidence_queries=[query],
                    learning_method="用戶反饋學習",
                    created_at=self._get_current_timestamp()
                )
                
                self.learned_relations.append(new_relation)
                learning_result["new_relations"].append(asdict(new_relation))
    
    def _extract_concepts_from_content(self, content: str, query: str) -> List[Dict[str, Any]]:
        """從內容中提取概念"""
        concepts = []
        
        # 使用學習模式提取概念
        for pattern in self.learning_patterns:
            if pattern.pattern_type == "概念提取":
                matches = re.finditer(pattern.pattern_regex, content)
                
                for match in matches:
                    concept_name = match.group(1).strip()
                    concept_description = match.group(2).strip() if len(match.groups()) > 1 else ""
                    
                    if concept_name and len(concept_name) > 1:
                        concept = {
                            "concept_name": concept_name,
                            "concept_description": concept_description,
                            "source_content": content[:200],
                            "source_query": query,
                            "learning_pattern": pattern.pattern_id,
                            "confidence": pattern.confidence_threshold
                        }
                        concepts.append(concept)
        
        return concepts
    
    def _extract_relations_from_content(self, content: str, query: str) -> List[Dict[str, Any]]:
        """從內容中提取關係"""
        relations = []
        
        for pattern in self.learning_patterns:
            if pattern.pattern_type == "關係識別":
                matches = re.finditer(pattern.pattern_regex, content)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        source = match.group(1).strip()
                        target = match.group(2).strip()
                        
                        if source and target:
                            relation = {
                                "source_concept": source,
                                "target_concept": target,
                                "relation_type": "學習關係",
                                "source_content": content[:200],
                                "source_query": query,
                                "learning_pattern": pattern.pattern_id,
                                "confidence": pattern.confidence_threshold
                            }
                            relations.append(relation)
        
        return relations
    
    def _analyze_missing_concept(self, query: str, concept_name: str) -> Dict[str, Any]:
        """分析缺失的概念"""
        analysis = {
            "type": "未知",
            "domain": "其他",
            "articles": [],
            "confidence": 0.5
        }
        
        # 基於查詢上下文推斷概念類型
        query_lower = query.lower()
        concept_lower = concept_name.lower()
        
        # 判斷法律領域
        if any(keyword in query_lower for keyword in ["著作權", "版權", "重製", "改作"]):
            analysis["domain"] = "著作權"
        elif any(keyword in query_lower for keyword in ["商標", "註冊", "使用"]):
            analysis["domain"] = "商標權"
        elif any(keyword in query_lower for keyword in ["專利", "發明", "技術"]):
            analysis["domain"] = "專利權"
        
        # 判斷概念類型
        if any(keyword in concept_lower for keyword in ["權", "權利"]):
            analysis["type"] = "權利"
        elif any(keyword in concept_lower for keyword in ["義務", "責任", "應", "必須"]):
            analysis["type"] = "義務"
        elif any(keyword in concept_lower for keyword in ["例外", "但", "除外"]):
            analysis["type"] = "例外"
        
        # 提取可能的條文
        article_matches = re.findall(r'第(\d+(?:-\d+)?)條', query)
        analysis["articles"] = [f"第{art}條" for art in article_matches]
        
        return analysis
    
    def _update_learning_patterns(self, query: str, results: List[Dict], 
                                feedback: Dict[str, Any], learning_result: Dict[str, Any]):
        """更新學習模式"""
        updated_patterns = []
        
        # 分析成功的模式
        if feedback.get("successful_patterns"):
            for pattern_id in feedback["successful_patterns"]:
                pattern = next((p for p in self.learning_patterns if p.pattern_id == pattern_id), None)
                if pattern:
                    pattern.success_count += 1
                    pattern.last_updated = self._get_current_timestamp()
                    updated_patterns.append({
                        "pattern_id": pattern_id,
                        "action": "success",
                        "new_confidence": min(1.0, pattern.confidence_threshold + 0.05)
                    })
        
        # 分析失敗的模式
        if feedback.get("failed_patterns"):
            for pattern_id in feedback["failed_patterns"]:
                pattern = next((p for p in self.learning_patterns if p.pattern_id == pattern_id), None)
                if pattern:
                    pattern.failure_count += 1
                    pattern.last_updated = self._get_current_timestamp()
                    updated_patterns.append({
                        "pattern_id": pattern_id,
                        "action": "failure",
                        "new_confidence": max(0.1, pattern.confidence_threshold - 0.05)
                    })
        
        learning_result["updated_patterns"] = updated_patterns
    
    def _generate_learning_insights(self, learning_result: Dict[str, Any]):
        """生成學習洞察"""
        insights = []
        
        # 分析新概念的特徵
        if learning_result["new_concepts"]:
            concepts_by_domain = defaultdict(int)
            concepts_by_type = defaultdict(int)
            
            for concept in learning_result["new_concepts"]:
                domain = concept.get("legal_domain", "其他")
                concept_type = concept.get("concept_type", "未知")
                concepts_by_domain[domain] += 1
                concepts_by_type[concept_type] += 1
            
            insights.append({
                "type": "概念分布",
                "message": f"新學習到{len(learning_result['new_concepts'])}個概念",
                "details": {
                    "domain_distribution": dict(concepts_by_domain),
                    "type_distribution": dict(concepts_by_type)
                }
            })
        
        # 分析學習模式的效果
        if learning_result["updated_patterns"]:
            successful_updates = len([p for p in learning_result["updated_patterns"] if p["action"] == "success"])
            failed_updates = len([p for p in learning_result["updated_patterns"] if p["action"] == "failure"])
            
            insights.append({
                "type": "模式效果",
                "message": f"更新了{len(learning_result['updated_patterns'])}個學習模式",
                "details": {
                    "successful_updates": successful_updates,
                    "failed_updates": failed_updates,
                    "success_rate": successful_updates / (successful_updates + failed_updates) if (successful_updates + failed_updates) > 0 else 0
                }
            })
        
        learning_result["learning_insights"] = insights
    
    def generate_enhanced_query_expansion(self, query: str) -> Dict[str, Any]:
        """生成增強的查詢擴展"""
        expansion = {
            "original_query": query,
            "expanded_query": query,
            "learned_concepts": [],
            "learned_relations": [],
            "confidence_boost": 0.0
        }
        
        # 查找學習到的概念
        query_lower = query.lower()
        for concept_id, concept in self.learned_concepts.items():
            if concept.concept_name.lower() in query_lower:
                expansion["learned_concepts"].append(asdict(concept))
                expansion["expanded_query"] += f" {concept.concept_name}"
                expansion["confidence_boost"] += concept.confidence * 0.1
        
        # 查找學習到的關係
        for relation in self.learned_relations:
            if relation.source_concept.lower() in query_lower:
                expansion["learned_relations"].append(asdict(relation))
                expansion["expanded_query"] += f" {relation.target_concept}"
                expansion["confidence_boost"] += relation.confidence * 0.05
        
        return expansion
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """獲取學習統計"""
        return {
            "total_learned_concepts": len(self.learned_concepts),
            "total_learned_relations": len(self.learned_relations),
            "total_queries_processed": len(self.query_history),
            "learning_patterns": {
                "total_patterns": len(self.learning_patterns),
                "successful_patterns": len([p for p in self.learning_patterns if p.success_count > p.failure_count]),
                "pattern_success_rate": self._calculate_pattern_success_rate()
            },
            "concept_distribution": self._get_concept_distribution(),
            "recent_learning_activity": self._get_recent_learning_activity()
        }
    
    def _calculate_pattern_success_rate(self) -> float:
        """計算模式成功率"""
        if not self.learning_patterns:
            return 0.0
        
        total_success = sum(p.success_count for p in self.learning_patterns)
        total_attempts = sum(p.success_count + p.failure_count for p in self.learning_patterns)
        
        return total_success / total_attempts if total_attempts > 0 else 0.0
    
    def _get_concept_distribution(self) -> Dict[str, int]:
        """獲取概念分布"""
        domain_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for concept in self.learned_concepts.values():
            domain_counts[concept.legal_domain] += 1
            type_counts[concept.concept_type] += 1
        
        return {
            "by_domain": dict(domain_counts),
            "by_type": dict(type_counts)
        }
    
    def _get_recent_learning_activity(self) -> List[Dict[str, Any]]:
        """獲取最近的學習活動"""
        recent_queries = self.query_history[-10:] if self.query_history else []
        
        activity = []
        for query_info in recent_queries:
            activity.append({
                "query": query_info["query"],
                "timestamp": query_info["timestamp"],
                "has_feedback": bool(query_info.get("feedback")),
                "result_count": len(query_info.get("results", []))
            })
        
        return activity
    
    def _get_current_timestamp(self) -> str:
        """獲取當前時間戳"""
        import datetime
        return datetime.datetime.now().isoformat()


# 全局學習系統實例
dynamic_learning_system = DynamicConceptLearningSystem()
