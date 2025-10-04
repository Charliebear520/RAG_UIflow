"""
法律概念推理引擎 - 專門處理法律語義推理和概念映射
"""

from typing import Dict, List, Any, Tuple
import re
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class LegalConcept:
    """法律概念"""
    concept_id: str
    concept_name: str
    legal_terms: List[str]  # 法律術語
    everyday_terms: List[str]  # 日常用語
    related_concepts: List[str]  # 相關概念
    applicable_articles: List[str]  # 適用條文
    reasoning_rules: List[str]  # 推理規則


@dataclass
class ReasoningRule:
    """推理規則"""
    rule_id: str
    condition_patterns: List[str]  # 條件模式
    conclusion_concept: str  # 結論概念
    confidence: float  # 置信度
    applicable_articles: List[str]  # 適用條文


class LegalReasoningEngine:
    """法律推理引擎"""
    
    def __init__(self):
        self.concepts = {}
        self.reasoning_rules = []
        self._initialize_legal_concepts()
        self._initialize_reasoning_rules()
    
    def _initialize_legal_concepts(self):
        """初始化法律概念庫"""
        self.concepts = {
            "改作": LegalConcept(
                concept_id="derivative_work",
                concept_name="改作",
                legal_terms=["改作", "衍生著作", "改作權"],
                everyday_terms=["翻譯", "譯", "轉譯", "中譯", "英譯", "日譯", "改寫", "改編", "修改", "衍生", "創作", "重新創作", "二次創作", "用自己的語氣"],
                related_concepts=["著作財產權", "授權", "侵害"],
                applicable_articles=["第28條"],
                reasoning_rules=["translation_to_derivative", "rewrite_to_derivative"]
            ),
            "重製": LegalConcept(
                concept_id="reproduction",
                concept_name="重製",
                legal_terms=["重製", "重製權"],
                everyday_terms=["複製", "抄襲", "盜版", "翻印", "影印", "掃描", "下載", "保存", "直接複製"],
                related_concepts=["著作財產權", "侵害"],
                applicable_articles=["第22條"],
                reasoning_rules=["copy_to_reproduction"]
            ),
            "散布": LegalConcept(
                concept_id="distribution",
                concept_name="散布",
                legal_terms=["散布", "散布權"],
                everyday_terms=["分享", "傳播", "發布", "上傳", "轉載", "轉發", "傳送", "出版"],
                related_concepts=["著作財產權", "授權"],
                applicable_articles=["第28-1條"],
                reasoning_rules=["publish_to_distribution"]
            ),
            "公開傳輸": LegalConcept(
                concept_id="public_transmission",
                concept_name="公開傳輸",
                legal_terms=["公開傳輸", "公開傳輸權"],
                everyday_terms=["上網", "網路傳播", "線上分享", "串流", "直播"],
                related_concepts=["著作財產權", "授權"],
                applicable_articles=["第26-1條"],
                reasoning_rules=["online_to_transmission"]
            ),
            "合理使用": LegalConcept(
                concept_id="fair_use",
                concept_name="合理使用",
                legal_terms=["合理使用", "著作權限制"],
                everyday_terms=["引用", "評論", "教學", "研究", "報導", "學術", "教育用途"],
                related_concepts=["著作財產權", "限制"],
                applicable_articles=["第44條", "第46條", "第47條", "第65條"],
                reasoning_rules=["education_to_fair_use"]
            )
        }
    
    def _initialize_reasoning_rules(self):
        """初始化推理規則"""
        self.reasoning_rules = [
            ReasoningRule(
                rule_id="translation_to_derivative",
                condition_patterns=["翻譯", "譯", "轉譯", "中譯", "英譯", "日譯"],
                conclusion_concept="改作",
                confidence=0.95,
                applicable_articles=["第28條"]
            ),
            ReasoningRule(
                rule_id="rewrite_to_derivative",
                condition_patterns=["改寫", "改編", "修改", "衍生", "創作", "重新創作", "二次創作", "用自己的語氣"],
                conclusion_concept="改作",
                confidence=0.9,
                applicable_articles=["第28條"]
            ),
            ReasoningRule(
                rule_id="publish_to_distribution",
                condition_patterns=["出版", "發布", "公開發表", "發行"],
                conclusion_concept="散布",
                confidence=0.85,
                applicable_articles=["第28-1條"]
            ),
            ReasoningRule(
                rule_id="education_to_fair_use",
                condition_patterns=["教學", "教育", "課堂", "學校", "學生", "授課"],
                conclusion_concept="合理使用",
                confidence=0.8,
                applicable_articles=["第46條", "第47條"]
            ),
            ReasoningRule(
                rule_id="online_to_transmission",
                condition_patterns=["上網", "網路", "線上", "串流", "直播"],
                conclusion_concept="公開傳輸",
                confidence=0.9,
                applicable_articles=["第26-1條"]
            )
        ]
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查詢並進行法律推理"""
        analysis_result = {
            "original_query": query,
            "detected_concepts": [],
            "reasoning_paths": [],
            "concept_mappings": [],
            "applicable_articles": [],
            "confidence_scores": {},
            "reasoning_explanation": []
        }
        
        query_lower = query.lower()
        
        # 1. 檢測法律概念
        detected_concepts = self._detect_legal_concepts(query_lower)
        analysis_result["detected_concepts"] = detected_concepts
        
        # 2. 應用推理規則
        reasoning_paths = self._apply_reasoning_rules(query_lower, detected_concepts)
        analysis_result["reasoning_paths"] = reasoning_paths
        
        # 3. 生成概念映射
        concept_mappings = self._generate_concept_mappings(query_lower, detected_concepts, reasoning_paths)
        analysis_result["concept_mappings"] = concept_mappings
        
        # 4. 提取適用條文
        applicable_articles = self._extract_applicable_articles(detected_concepts, reasoning_paths)
        analysis_result["applicable_articles"] = applicable_articles
        
        # 5. 計算置信度
        confidence_scores = self._calculate_confidence_scores(reasoning_paths)
        analysis_result["confidence_scores"] = confidence_scores
        
        # 6. 生成推理解釋
        reasoning_explanation = self._generate_reasoning_explanation(detected_concepts, reasoning_paths, concept_mappings)
        analysis_result["reasoning_explanation"] = reasoning_explanation
        
        return analysis_result
    
    def _detect_legal_concepts(self, query: str) -> List[Dict[str, Any]]:
        """檢測查詢中的法律概念"""
        detected = []
        
        for concept_id, concept in self.concepts.items():
            # 檢查法律術語
            for term in concept.legal_terms:
                if term in query:
                    detected.append({
                        "concept_id": concept_id,
                        "concept_name": concept.concept_name,
                        "matched_term": term,
                        "match_type": "legal_term",
                        "confidence": 1.0
                    })
            
            # 檢查日常用語
            for term in concept.everyday_terms:
                if term in query:
                    detected.append({
                        "concept_id": concept_id,
                        "concept_name": concept.concept_name,
                        "matched_term": term,
                        "match_type": "everyday_term",
                        "confidence": 0.8
                    })
        
        return detected
    
    def _apply_reasoning_rules(self, query: str, detected_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """應用推理規則"""
        reasoning_paths = []
        
        for rule in self.reasoning_rules:
            # 檢查條件模式
            matched_patterns = []
            for pattern in rule.condition_patterns:
                if pattern in query:
                    matched_patterns.append(pattern)
            
            if matched_patterns:
                # 檢查是否已經檢測到結論概念
                conclusion_already_detected = any(
                    concept["concept_id"] == rule.conclusion_concept 
                    for concept in detected_concepts
                )
                
                if not conclusion_already_detected:
                    reasoning_paths.append({
                        "rule_id": rule.rule_id,
                        "matched_patterns": matched_patterns,
                        "conclusion_concept": rule.conclusion_concept,
                        "conclusion_name": self.concepts[rule.conclusion_concept].concept_name,
                        "confidence": rule.confidence,
                        "applicable_articles": rule.applicable_articles,
                        "reasoning_type": "concept_inference"
                    })
        
        return reasoning_paths
    
    def _generate_concept_mappings(self, query: str, detected_concepts: List[Dict[str, Any]], reasoning_paths: List[Dict[str, Any]]) -> List[str]:
        """生成概念映射"""
        mappings = []
        
        # 直接概念映射
        for concept in detected_concepts:
            if concept["match_type"] == "everyday_term":
                mappings.append(f"{concept['matched_term']}→{concept['concept_name']}")
        
        # 推理概念映射
        for path in reasoning_paths:
            for pattern in path["matched_patterns"]:
                mappings.append(f"{pattern}→{path['conclusion_name']}")
        
        return list(set(mappings))  # 去重
    
    def _extract_applicable_articles(self, detected_concepts: List[Dict[str, Any]], reasoning_paths: List[Dict[str, Any]]) -> List[str]:
        """提取適用條文"""
        articles = []
        
        # 從直接檢測的概念中提取
        for concept in detected_concepts:
            concept_obj = self.concepts.get(concept["concept_id"])
            if concept_obj:
                articles.extend(concept_obj.applicable_articles)
        
        # 從推理路徑中提取
        for path in reasoning_paths:
            articles.extend(path["applicable_articles"])
        
        return list(set(articles))  # 去重
    
    def _calculate_confidence_scores(self, reasoning_paths: List[Dict[str, Any]]) -> Dict[str, float]:
        """計算置信度分數"""
        scores = {}
        
        for path in reasoning_paths:
            concept = path["conclusion_concept"]
            if concept not in scores:
                scores[concept] = path["confidence"]
            else:
                # 取最高置信度
                scores[concept] = max(scores[concept], path["confidence"])
        
        return scores
    
    def _generate_reasoning_explanation(self, detected_concepts: List[Dict[str, Any]], reasoning_paths: List[Dict[str, Any]], concept_mappings: List[str]) -> List[str]:
        """生成推理解釋"""
        explanations = []
        
        # 直接概念檢測解釋
        for concept in detected_concepts:
            if concept["match_type"] == "everyday_term":
                explanations.append(f"檢測到日常用語「{concept['matched_term']}」對應法律概念「{concept['concept_name']}」")
        
        # 推理路徑解釋
        for path in reasoning_paths:
            patterns_str = "、".join(path["matched_patterns"])
            explanations.append(f"根據推理規則：查詢中的「{patterns_str}」在法律上屬於「{path['conclusion_name']}」行為")
            
            if path["applicable_articles"]:
                articles_str = "、".join(path["applicable_articles"])
                explanations.append(f"「{path['conclusion_name']}」適用於{articles_str}")
        
        return explanations
    
    def get_expanded_query(self, query: str) -> str:
        """獲取擴展後的查詢"""
        analysis = self.analyze_query(query)
        
        expanded_terms = [query]
        
        # 添加檢測到的概念
        for concept in analysis["detected_concepts"]:
            expanded_terms.append(concept["concept_name"])
        
        # 添加推理得出的概念
        for path in analysis["reasoning_paths"]:
            expanded_terms.append(path["conclusion_name"])
        
        # 添加適用條文
        expanded_terms.extend(analysis["applicable_articles"])
        
        return " ".join(expanded_terms)


# 全局推理引擎實例
legal_reasoning_engine = LegalReasoningEngine()
