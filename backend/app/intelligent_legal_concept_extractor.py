"""
智能法律概念提取器 - 自動從法律文檔中提取概念和關係
"""

import re
import json
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@dataclass
class LegalConceptPattern:
    """法律概念模式"""
    pattern_type: str  # 概念類型：定義、權利、義務、例外等
    pattern_regex: str  # 正則表達式模式
    concept_category: str  # 概念分類：著作權、商標權等
    importance_weight: float  # 重要性權重


@dataclass
class ExtractedConcept:
    """提取的法律概念"""
    concept_id: str
    concept_name: str
    concept_type: str  # 定義、權利、義務、例外、條件等
    source_text: str
    context: str
    related_articles: List[str]
    legal_domain: str
    confidence: float
    synonyms: List[str]
    related_concepts: List[str]


@dataclass
class ConceptRelation:
    """概念關係"""
    source_concept: str
    target_concept: str
    relation_type: str  # 包含、適用於、前提、例外等
    confidence: float
    evidence: str


class IntelligentLegalConceptExtractor:
    """智能法律概念提取器"""
    
    def __init__(self):
        self.concept_patterns = self._initialize_concept_patterns()
        self.legal_domains = self._initialize_legal_domains()
        self.extracted_concepts = {}
        self.concept_relations = []
        self.article_concepts = defaultdict(list)  # 條文 -> 概念映射
        
    def _initialize_concept_patterns(self) -> List[LegalConceptPattern]:
        """初始化法律概念模式"""
        return [
            # 權利定義模式
            LegalConceptPattern(
                pattern_type="權利定義",
                pattern_regex=r"([^。，；：]+)(?:專有|享有|具有)([^。，；：]*)權利",
                concept_category="權利",
                importance_weight=0.9
            ),
            LegalConceptPattern(
                pattern_type="權利定義2",
                pattern_regex=r"([^。，；：]+)權[利]?(?:是指|係指|為)([^。，；：]*)",
                concept_category="權利",
                importance_weight=0.9
            ),
            
            # 義務定義模式
            LegalConceptPattern(
                pattern_type="義務定義",
                pattern_regex=r"([^。，；：]+)(?:應|必須|得|可以)([^。，；：]*)",
                concept_category="義務",
                importance_weight=0.8
            ),
            
            # 例外條件模式
            LegalConceptPattern(
                pattern_type="例外條件",
                pattern_regex=r"(?:但|惟|例外|除外)([^。，；：]*)",
                concept_category="例外",
                importance_weight=0.7
            ),
            
            # 法律後果模式
            LegalConceptPattern(
                pattern_type="法律後果",
                pattern_regex=r"(?:違反|侵害|侵犯)([^。，；：]*)(?:者|時)([^。，；：]*)",
                concept_category="後果",
                importance_weight=0.8
            ),
            
            # 適用條件模式
            LegalConceptPattern(
                pattern_type="適用條件",
                pattern_regex=r"(?:於|在)([^。，；：]*)(?:情形|情況|條件)([^。，；：]*)",
                concept_category="條件",
                importance_weight=0.6
            )
        ]
    
    def _initialize_legal_domains(self) -> Dict[str, List[str]]:
        """初始化法律領域關鍵詞"""
        return {
            "著作權": [
                "著作", "著作權", "版權", "重製", "改作", "散布", "公開傳輸", 
                "公開演出", "公開展示", "出租", "衍生著作", "編輯著作"
            ],
            "商標權": [
                "商標", "商標權", "註冊", "使用", "近似", "混淆", "著名商標"
            ],
            "專利權": [
                "專利", "發明", "新型", "新式樣", "技術", "創新", "實施"
            ]
        }
    
    def extract_concepts_from_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """從文檔中提取法律概念"""
        print("🔍 開始智能法律概念提取...")
        
        all_concepts = []
        all_relations = []
        
        for doc in documents:
            print(f"📄 處理文檔: {doc.get('filename', 'Unknown')}")
            
            # 按條文分組處理
            if 'structured_chunks' in doc and doc['structured_chunks']:
                for chunk in doc['structured_chunks']:
                    concepts = self._extract_concepts_from_chunk(chunk, doc)
                    all_concepts.extend(concepts)
                    
                    # 提取概念關係
                    relations = self._extract_concept_relations(chunk, concepts)
                    all_relations.extend(relations)
        
        # 概念去重和合併
        merged_concepts = self._merge_duplicate_concepts(all_concepts)
        
        # 建立概念關係網絡
        concept_network = self._build_concept_network(merged_concepts, all_relations)
        
        # 建立條文-概念映射
        article_concept_mapping = self._build_article_concept_mapping(merged_concepts)
        
        # 生成概念推理規則
        reasoning_rules = self._generate_reasoning_rules(merged_concepts, all_relations)
        
        result = {
            "extracted_concepts": merged_concepts,
            "concept_relations": all_relations,
            "concept_network": concept_network,
            "article_concept_mapping": article_concept_mapping,
            "reasoning_rules": reasoning_rules,
            "statistics": {
                "total_concepts": len(merged_concepts),
                "total_relations": len(all_relations),
                "concept_types": self._get_concept_type_statistics(merged_concepts),
                "legal_domains": self._get_domain_statistics(merged_concepts)
            }
        }
        
        print(f"✅ 概念提取完成: {len(merged_concepts)}個概念, {len(all_relations)}個關係")
        return result
    
    def _extract_concepts_from_chunk(self, chunk: Dict[str, Any], doc: Dict[str, Any]) -> List[ExtractedConcept]:
        """從分塊中提取概念"""
        concepts = []
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        # 提取條文信息
        article_info = self._extract_article_info(content, metadata)
        
        # 應用概念模式
        for pattern in self.concept_patterns:
            matches = re.finditer(pattern.pattern_regex, content, re.MULTILINE)
            
            for match in matches:
                concept = self._create_concept_from_match(
                    match, pattern, content, article_info, doc
                )
                if concept:
                    concepts.append(concept)
        
        # 提取法律關鍵詞
        legal_keywords = self._extract_legal_keywords(content)
        for keyword, domain in legal_keywords:
            concept = ExtractedConcept(
                concept_id=f"keyword_{keyword}_{len(concepts)}",
                concept_name=keyword,
                concept_type="關鍵詞",
                source_text=content,
                context=self._get_context_around_keyword(content, keyword),
                related_articles=article_info.get('articles', []),
                legal_domain=domain,
                confidence=0.6,
                synonyms=[],
                related_concepts=[]
            )
            concepts.append(concept)
        
        return concepts
    
    def _extract_concept_relations(self, chunk: Dict[str, Any], concepts: List[ExtractedConcept]) -> List[ConceptRelation]:
        """提取概念關係"""
        relations = []
        content = chunk.get('content', '')
        
        # 權利-義務關係
        rights_obligations = self._extract_rights_obligations_relations(content, concepts)
        relations.extend(rights_obligations)
        
        # 包含關係
        inclusion_relations = self._extract_inclusion_relations(content, concepts)
        relations.extend(inclusion_relations)
        
        # 條件關係
        condition_relations = self._extract_condition_relations(content, concepts)
        relations.extend(condition_relations)
        
        return relations
    
    def _create_concept_from_match(self, match, pattern: LegalConceptPattern, 
                                 content: str, article_info: Dict, doc: Dict) -> ExtractedConcept:
        """從匹配結果創建概念"""
        try:
            # 提取概念名稱
            concept_name = match.group(1).strip() if match.groups() else match.group(0).strip()
            
            # 清理概念名稱
            concept_name = self._clean_concept_name(concept_name)
            
            if not concept_name or len(concept_name) < 2:
                return None
            
            # 生成概念ID
            concept_id = f"{pattern.concept_category}_{concept_name}_{hash(concept_name) % 10000}"
            
            # 獲取上下文
            context = self._get_context_around_match(content, match)
            
            # 識別法律領域
            legal_domain = self._identify_legal_domain(content)
            
            # 提取同義詞
            synonyms = self._extract_synonyms(concept_name, content)
            
            concept = ExtractedConcept(
                concept_id=concept_id,
                concept_name=concept_name,
                concept_type=pattern.pattern_type,
                source_text=match.group(0),
                context=context,
                related_articles=article_info.get('articles', []),
                legal_domain=legal_domain,
                confidence=pattern.importance_weight,
                synonyms=synonyms,
                related_concepts=[]
            )
            
            return concept
            
        except Exception as e:
            print(f"⚠️ 創建概念失敗: {e}")
            return None
    
    def _extract_article_info(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """提取條文信息"""
        articles = []
        
        # 從metadata提取
        if metadata.get('article'):
            articles.append(f"第{metadata['article']}條")
        
        # 從內容提取
        article_matches = re.findall(r'第(\d+(?:-\d+)?)條', content)
        articles.extend([f"第{art}條" for art in article_matches])
        
        return {
            "articles": list(set(articles)),
            "chapter": metadata.get('chapter', ''),
            "section": metadata.get('section', '')
        }
    
    def _clean_concept_name(self, name: str) -> str:
        """清理概念名稱"""
        # 移除多餘的標點符號和空格
        name = re.sub(r'[，。；：！？\s]+', '', name)
        
        # 移除常見的無意義詞
        meaningless_words = ['的', '在', '於', '為', '是', '有', '及', '與', '或', '但', '惟']
        for word in meaningless_words:
            name = name.replace(word, '')
        
        return name.strip()
    
    def _get_context_around_match(self, content: str, match) -> str:
        """獲取匹配項周圍的上下文"""
        start = max(0, match.start() - 50)
        end = min(len(content), match.end() + 50)
        return content[start:end]
    
    def _get_context_around_keyword(self, content: str, keyword: str) -> str:
        """獲取關鍵詞周圍的上下文"""
        idx = content.find(keyword)
        if idx == -1:
            return content[:100]
        
        start = max(0, idx - 50)
        end = min(len(content), idx + len(keyword) + 50)
        return content[start:end]
    
    def _identify_legal_domain(self, content: str) -> str:
        """識別法律領域"""
        content_lower = content.lower()
        
        for domain, keywords in self.legal_domains.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return domain
        
        return "其他"
    
    def _extract_synonyms(self, concept_name: str, content: str) -> List[str]:
        """提取同義詞"""
        synonyms = []
        
        # 常見的同義詞模式
        synonym_patterns = [
            r'([^，。；：]*)(?:亦稱|又稱|別稱|俗稱)([^，。；：]*)',
            r'([^，。；：]*)(?:包括|含|涵蓋)([^，。；：]*)',
        ]
        
        for pattern in synonym_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if concept_name in match.group(0):
                    # 提取可能的同義詞
                    potential_synonyms = [group.strip() for group in match.groups() if group.strip()]
                    synonyms.extend(potential_synonyms)
        
        return list(set(synonyms))
    
    def _extract_legal_keywords(self, content: str) -> List[Tuple[str, str]]:
        """提取法律關鍵詞"""
        keywords = []
        content_lower = content.lower()
        
        for domain, domain_keywords in self.legal_domains.items():
            for keyword in domain_keywords:
                if keyword in content_lower:
                    keywords.append((keyword, domain))
        
        return keywords
    
    def _extract_rights_obligations_relations(self, content: str, concepts: List[ExtractedConcept]) -> List[ConceptRelation]:
        """提取權利-義務關係"""
        relations = []
        
        # 權利-義務關係模式
        patterns = [
            r'([^，。；：]+)專有([^，。；：]+)權利',
            r'([^，。；：]+)享有([^，。；：]+)權利',
            r'([^，。；：]+)應([^，。；：]+)',
            r'([^，。；：]+)必須([^，。；：]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if len(match.groups()) >= 2:
                    source = match.group(1).strip()
                    target = match.group(2).strip()
                    
                    if source and target:
                        relation = ConceptRelation(
                            source_concept=source,
                            target_concept=target,
                            relation_type="權利義務",
                            confidence=0.8,
                            evidence=match.group(0)
                        )
                        relations.append(relation)
        
        return relations
    
    def _extract_inclusion_relations(self, content: str, concepts: List[ExtractedConcept]) -> List[ConceptRelation]:
        """提取包含關係"""
        relations = []
        
        # 包含關係模式
        patterns = [
            r'([^，。；：]+)包括([^，。；：]+)',
            r'([^，。；：]+)含([^，。；：]+)',
            r'([^，。；：]+)涵蓋([^，。；：]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if len(match.groups()) >= 2:
                    source = match.group(1).strip()
                    target = match.group(2).strip()
                    
                    if source and target:
                        relation = ConceptRelation(
                            source_concept=source,
                            target_concept=target,
                            relation_type="包含",
                            confidence=0.7,
                            evidence=match.group(0)
                        )
                        relations.append(relation)
        
        return relations
    
    def _extract_condition_relations(self, content: str, concepts: List[ExtractedConcept]) -> List[ConceptRelation]:
        """提取條件關係"""
        relations = []
        
        # 條件關係模式
        patterns = [
            r'([^，。；：]+)時([^，。；：]+)',
            r'([^，。；：]+)情形([^，。；：]+)',
            r'([^，。；：]+)條件([^，。；：]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if len(match.groups()) >= 2:
                    source = match.group(1).strip()
                    target = match.group(2).strip()
                    
                    if source and target:
                        relation = ConceptRelation(
                            source_concept=source,
                            target_concept=target,
                            relation_type="條件",
                            confidence=0.6,
                            evidence=match.group(0)
                        )
                        relations.append(relation)
        
        return relations
    
    def _merge_duplicate_concepts(self, concepts: List[ExtractedConcept]) -> List[ExtractedConcept]:
        """合併重複概念"""
        concept_groups = defaultdict(list)
        
        # 按概念名稱分組
        for concept in concepts:
            key = concept.concept_name.lower().strip()
            concept_groups[key].append(concept)
        
        merged_concepts = []
        
        for group in concept_groups.values():
            if len(group) == 1:
                merged_concepts.append(group[0])
            else:
                # 合併多個相同概念
                merged = self._merge_concept_group(group)
                merged_concepts.append(merged)
        
        return merged_concepts
    
    def _merge_concept_group(self, concepts: List[ExtractedConcept]) -> ExtractedConcept:
        """合併概念組"""
        # 選擇置信度最高的作為主概念
        main_concept = max(concepts, key=lambda x: x.confidence)
        
        # 合併其他屬性
        all_synonyms = []
        all_articles = []
        all_contexts = []
        
        for concept in concepts:
            all_synonyms.extend(concept.synonyms)
            all_articles.extend(concept.related_articles)
            all_contexts.append(concept.context)
        
        main_concept.synonyms = list(set(all_synonyms))
        main_concept.related_articles = list(set(all_articles))
        main_concept.context = " | ".join(all_contexts[:3])  # 保留前3個上下文
        
        return main_concept
    
    def _build_concept_network(self, concepts: List[ExtractedConcept], 
                             relations: List[ConceptRelation]) -> Dict[str, Any]:
        """建立概念網絡"""
        network = {
            "nodes": [],
            "edges": [],
            "statistics": {}
        }
        
        # 添加節點
        for concept in concepts:
            network["nodes"].append({
                "id": concept.concept_id,
                "name": concept.concept_name,
                "type": concept.concept_type,
                "domain": concept.legal_domain,
                "confidence": concept.confidence,
                "article_count": len(concept.related_articles)
            })
        
        # 添加邊
        for relation in relations:
            network["edges"].append({
                "source": relation.source_concept,
                "target": relation.target_concept,
                "type": relation.relation_type,
                "confidence": relation.confidence,
                "evidence": relation.evidence
            })
        
        # 統計信息
        network["statistics"] = {
            "total_nodes": len(network["nodes"]),
            "total_edges": len(network["edges"]),
            "avg_confidence": np.mean([node["confidence"] for node in network["nodes"]]),
            "domain_distribution": self._get_domain_distribution(network["nodes"])
        }
        
        return network
    
    def _build_article_concept_mapping(self, concepts: List[ExtractedConcept]) -> Dict[str, List[str]]:
        """建立條文-概念映射"""
        mapping = defaultdict(list)
        
        for concept in concepts:
            for article in concept.related_articles:
                mapping[article].append({
                    "concept_name": concept.concept_name,
                    "concept_type": concept.concept_type,
                    "confidence": concept.confidence,
                    "context": concept.context[:100]
                })
        
        return dict(mapping)
    
    def _generate_reasoning_rules(self, concepts: List[ExtractedConcept], 
                                relations: List[ConceptRelation]) -> List[Dict[str, Any]]:
        """生成推理規則"""
        rules = []
        
        # 基於概念關係生成推理規則
        for relation in relations:
            rule = {
                "rule_id": f"rule_{len(rules)}",
                "condition": relation.source_concept,
                "conclusion": relation.target_concept,
                "relation_type": relation.relation_type,
                "confidence": relation.confidence,
                "evidence": relation.evidence
            }
            rules.append(rule)
        
        # 基於概念類型生成推理規則
        concept_types = defaultdict(list)
        for concept in concepts:
            concept_types[concept.concept_type].append(concept)
        
        for concept_type, type_concepts in concept_types.items():
            if len(type_concepts) > 1:
                rule = {
                    "rule_id": f"type_rule_{concept_type}",
                    "condition": f"查詢包含{concept_type}相關詞彙",
                    "conclusion": f"相關於{concept_type}概念",
                    "relation_type": "類型推理",
                    "confidence": 0.7,
                    "evidence": f"基於{len(type_concepts)}個{concept_type}概念"
                }
                rules.append(rule)
        
        return rules
    
    def _get_concept_type_statistics(self, concepts: List[ExtractedConcept]) -> Dict[str, int]:
        """獲取概念類型統計"""
        type_counts = defaultdict(int)
        for concept in concepts:
            type_counts[concept.concept_type] += 1
        return dict(type_counts)
    
    def _get_domain_statistics(self, concepts: List[ExtractedConcept]) -> Dict[str, int]:
        """獲取領域統計"""
        domain_counts = defaultdict(int)
        for concept in concepts:
            domain_counts[concept.legal_domain] += 1
        return dict(domain_counts)
    
    def _get_domain_distribution(self, nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        """獲取領域分布"""
        domain_counts = defaultdict(int)
        for node in nodes:
            domain_counts[node["domain"]] += 1
        return dict(domain_counts)


# 全局提取器實例
intelligent_extractor = IntelligentLegalConceptExtractor()
