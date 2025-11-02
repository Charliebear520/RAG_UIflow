"""
Metadata增強模組
"""

import re
import hashlib
import jieba
import jieba.analyse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import json


@dataclass
class LegalConcept:
    """法律概念"""
    concept_name: str
    concept_type: str
    legal_domain: str
    importance_score: float
    synonyms: List[str]
    confidence: float


@dataclass
class LegalRelation:
    """法律關係"""
    relation_type: str
    subject: str
    object: str
    relation: str
    confidence: float


class MetadataEnhancer:
    """Metadata增強器 - 專注於「條」層級的metadata生成和向下繼承"""
    
    def __init__(self):
        self.legal_domains = self._initialize_legal_domains()
        self.legal_concept_patterns = self._initialize_concept_patterns()
        self.article_type_patterns = self._initialize_article_type_patterns()
        self.legal_synonyms = self._initialize_legal_synonyms()
        
        # 緩存機制
        self.concept_cache = {}
        self.metadata_cache = {}
        
        # 條層級metadata存儲（用於向下繼承）
        self.article_metadata_map = {}  # {article_id: enhanced_metadata}
        self.inheritance_hierarchy = {}  # {child_chunk_id: parent_article_id}
    
    def _initialize_legal_domains(self) -> Dict[str, List[str]]:
        """初始化法律領域關鍵詞"""
        return {
            "著作權法": ["著作權", "著作", "重製", "改作", "散布", "公開傳輸", "公開播送", "公開演出", "公開展示", "出租", "合理使用"],
            "商標法": ["商標", "商標權", "註冊", "使用", "侵權", "混淆", "證明標章", "團體標章", "團體商標"],
            "專利法": ["專利", "發明", "新型", "設計", "申請", "審查", "專利權", "實施", "授權"],
            "智慧財產權法": ["智慧財產權", "智慧財產", "IP", "知識產權"],
            "民法": ["契約", "債務", "債權", "物權", "所有權", "占有", "侵權行為", "損害賠償"],
            "刑法": ["犯罪", "刑罰", "有期徒刑", "罰金", "沒收", "緩刑"],
            "行政法": ["行政處分", "行政救濟", "訴願", "行政訴訟", "國家賠償"]
        }
    
    def _initialize_concept_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化法律概念模式"""
        return {
            "權利定義": {
                "patterns": [
                    r"([^。，；：]+)(?:專有|享有|具有)([^。，；：]*)權利",
                    r"([^。，；：]+)權[利]?(?:是指|係指|為)([^。，；：]*)",
                    r"([^。，；：]+)(?:專有|享有)([^。，；：]*)權"
                ],
                "importance_weight": 0.9
            },
            "義務規定": {
                "patterns": [
                    r"([^。，；：]+)(?:應|必須|得|可以)([^。，；：]*)",
                    r"([^。，；：]+)(?:不得|不可|禁止)([^。，；：]*)"
                ],
                "importance_weight": 0.8
            },
            "例外條件": {
                "patterns": [
                    r"(?:但|惟|例外|除外)([^。，；：]*)",
                    r"(?:不在此限|不適用)([^。，；：]*)"
                ],
                "importance_weight": 0.7
            },
            "法律後果": {
                "patterns": [
                    r"(?:違反|侵害|侵犯)([^。，；：]*)(?:者|時)([^。，；：]*)",
                    r"(?:處|科)([^。，；：]*)(?:罰|刑)"
                ],
                "importance_weight": 0.8
            }
        }
    
    def _initialize_article_type_patterns(self) -> Dict[str, List[str]]:
        """初始化條文類型模式"""
        return {
            "權利定義": ["專有", "享有", "具有", "權利", "是指", "係指", "為"],
            "義務規定": ["應", "必須", "得", "可以", "不得", "不可", "禁止"],
            "例外條件": ["但", "惟", "例外", "除外", "不在此限", "不適用"],
            "法律後果": ["違反", "侵害", "侵犯", "處", "科", "罰", "刑", "責任"],
            "立法目的": ["為", "為保障", "為維護", "為促進", "為保護", "制定本法"],
            "適用條件": ["適用於", "適用", "於", "在", "當", "如"],
            "程序規定": ["申請", "審查", "核准", "登記", "註冊", "公告"],
            "定義條文": ["稱", "指", "謂", "係指", "是指", "為", "包括"]
        }
    
    def _initialize_legal_synonyms(self) -> Dict[str, List[str]]:
        """初始化法律同義詞"""
        return {
            "公開傳輸": ["公開傳輸", "數位傳輸", "網路傳輸", "線上傳輸", "上線提供", "public transmission"],
            "公開播送": ["公開播送", "廣播", "播送", "broadcast"],
            "重製": ["重製", "複製", "拷貝", "複本製作", "reproduction"],
            "散布": ["散布", "發行", "流通", "distribution"],
            "改作": ["改作", "改編", "翻案", "衍生創作", "derivative"],
            "引用": ["引用", "節錄", "摘錄", "引用他人著作"],
            "合理使用": ["合理使用", "公平使用", "fair use"],
            "商標權": ["商標權", "商標專用權", "商標使用權"],
            "著作權": ["著作權", "版權", "copyright"],
            "專利權": ["專利權", "專利", "patent"],
            "侵害": ["侵害", "侵犯", "違反", "損害", "違法", "不法"],
            "處罰": ["處罰", "制裁", "懲罰", "penalty"],
            "權利": ["權利", "權益", "right", "entitlement"],
            "義務": ["義務", "責任", "duty", "obligation"]
        }
    
    def enhance_metadata_batch(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """批量增強metadata - 專注於「條」層級並實現向下繼承"""
        enhanced_metadata = {}
        
        # 去重處理
        unique_chunks = self._deduplicate_chunks(chunks)
        print(f"📊 原始chunks: {len(chunks)}, 去重後: {len(unique_chunks)}")
        
        # 第一步：識別不同層級的chunks
        article_chunks = []
        chapter_section_chunks = []
        other_chunks = []
        
        for chunk in unique_chunks:
            if self._is_article_level_chunk(chunk):
                article_chunks.append(chunk)
            elif self._is_chapter_section_level_chunk(chunk):
                chapter_section_chunks.append(chunk)
            else:
                other_chunks.append(chunk)
        
        print(f"📋 識別出「條」層級chunks: {len(article_chunks)}個")
        print(f"📋 識別出「章、節」層級chunks: {len(chapter_section_chunks)}個")
        print(f"📋 其他層級chunks: {len(other_chunks)}個")
        
        # 第二步：為「條」層級生成metadata
        article_metadata_results = {}
        for i, chunk in enumerate(article_chunks):
            chunk_id = chunk.get("chunk_id", f"article_chunk_{i}")
            content = chunk.get("content", "")
            original_metadata = chunk.get("metadata", {})
            
            # 檢查緩存
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.metadata_cache:
                enhanced = self.metadata_cache[content_hash]
            else:
                # 增強metadata（專注於條層級）
                enhanced = self._enhance_article_level_chunk(content, original_metadata)
                # 緩存結果
                self.metadata_cache[content_hash] = enhanced
            
            article_metadata_results[chunk_id] = enhanced
            
            # 存儲到條層級metadata映射
            article_id = self._extract_article_id(chunk)
            if article_id:
                self.article_metadata_map[article_id] = enhanced
            
            if (i + 1) % 10 == 0:
                print(f"📈 已處理「條」層級 {i + 1}/{len(article_chunks)} 個chunks")
        
        # 第三步：為「章、節」層級進行中等強度metadata增強
        chapter_section_metadata_results = {}
        for i, chunk in enumerate(chapter_section_chunks):
            chunk_id = chunk.get("chunk_id", f"chapter_section_chunk_{i}")
            content = chunk.get("content", "")
            original_metadata = chunk.get("metadata", {})
            
            # 檢查緩存
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.metadata_cache:
                enhanced = self.metadata_cache[content_hash]
            else:
                # 進行中等強度metadata增強
                enhanced = self._enhance_chapter_section_chunk(content, original_metadata)
                # 緩存結果
                self.metadata_cache[content_hash] = enhanced
            
            chapter_section_metadata_results[chunk_id] = enhanced
            
            if (i + 1) % 20 == 0:
                print(f"📈 已處理「章、節」層級 {i + 1}/{len(chapter_section_chunks)} 個chunks")
        
        # 第四步：為其他層級chunks實現向下繼承
        inherited_metadata_results = {}
        for i, chunk in enumerate(other_chunks):
            chunk_id = chunk.get("chunk_id", f"other_chunk_{i}")
            content = chunk.get("content", "")
            original_metadata = chunk.get("metadata", {})
            
            # 查找父級「條」的metadata
            parent_article_id = self._find_parent_article_id(chunk)
            inherited_metadata = None
            
            if parent_article_id and parent_article_id in self.article_metadata_map:
                inherited_metadata = self.article_metadata_map[parent_article_id].copy()
                # 標記這是繼承的metadata
                inherited_metadata["inherited_from"] = parent_article_id
                inherited_metadata["inheritance_type"] = "downward_inheritance"
                
                # 建立繼承關係映射
                self.inheritance_hierarchy[chunk_id] = parent_article_id
                
                print(f"🔄 {chunk_id} 繼承了 {parent_article_id} 的metadata")
            
            # 如果沒有找到繼承的metadata，則進行輕量級增強
            if not inherited_metadata:
                inherited_metadata = self._enhance_lightweight_chunk(content, original_metadata)
            
            inherited_metadata_results[chunk_id] = inherited_metadata
            
            if (i + 1) % 100 == 0:
                print(f"📈 已處理其他層級 {i + 1}/{len(other_chunks)} 個chunks")
        
        # 第五步：合併結果
        enhanced_metadata.update(article_metadata_results)
        enhanced_metadata.update(chapter_section_metadata_results)
        enhanced_metadata.update(inherited_metadata_results)
        
        # 映射回原始chunks
        final_results = self._map_enhanced_to_original(chunks, enhanced_metadata)
        
        print(f"✅ 完成metadata增強：條層級({len(article_metadata_results)}) + 章節層級({len(chapter_section_metadata_results)}) + 繼承層級({len(inherited_metadata_results)})")
        
        return final_results
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重：相同內容只保留一個"""
        content_hash = {}
        unique_chunks = []
        
        for chunk in chunks:
            content = chunk.get("content", "")
            content_md5 = hashlib.md5(content.encode()).hexdigest()
            
            if content_md5 not in content_hash:
                content_hash[content_md5] = chunk
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _map_enhanced_to_original(self, original_chunks: List[Dict[str, Any]], 
                                 enhanced_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """將增強metadata映射回原始chunks"""
        final_results = {}
        
        for chunk in original_chunks:
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")
            
            # 找到對應的增強metadata
            content_hash = hashlib.md5(content.encode()).hexdigest()
            for enhanced_id, enhanced_data in enhanced_metadata.items():
                if enhanced_id in content_hash or content in enhanced_data.get("content", ""):
                    final_results[chunk_id] = enhanced_data
                    break
        
        return final_results
    
    def _enhance_single_chunk(self, content: str, original_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """增強單個chunk的metadata"""
        # 清理原始metadata
        cleaned_metadata = self._clean_metadata(original_metadata)
        
        # 提取各種增強信息
        enhanced = {
            **cleaned_metadata,
            "legal_concepts": self._extract_legal_concepts(content),
            "semantic_keywords": self._extract_semantic_keywords(content),
            "article_type": self._classify_article_type(content),
            "legal_domain": self._classify_legal_domain(content),
            "legal_relations": self._extract_legal_relations(content),
            "query_intent_tags": self._extract_query_intent_tags(content),
            "semantic_similarity": self._precompute_semantic_similarity(content)
        }
        
        return enhanced
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清理metadata，移除不必要的字段"""
        # 要移除的字段
        remove_fields = ["spans", "page_range", "chunk_index", "length", "chunk_by", "strategy"]
        
        # 保留的字段
        keep_fields = [
            "id", "category", "article_label", "article_number", "article_suffix",
            "law_name", "chapter", "section", "article", "item", "level"
        ]
        
        cleaned = {}
        for key, value in metadata.items():
            if key in keep_fields and key not in remove_fields:
                cleaned[key] = value
        
        return cleaned
    
    def _extract_legal_concepts(self, content: str) -> List[Dict[str, Any]]:
        """提取法律概念"""
        concepts = []
        
        # 使用模式匹配提取概念
        for concept_type, config in self.legal_concept_patterns.items():
            for pattern in config["patterns"]:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    concept_name = match.group(1).strip()
                    if concept_name:
                        # 識別法律領域
                        legal_domain = self._identify_legal_domain_from_content(content)
                        
                        # 提取同義詞
                        synonyms = self._extract_synonyms_for_concept(concept_name, content)
                        
                        concept = {
                            "concept_name": concept_name,
                            "concept_type": concept_type,
                            "legal_domain": legal_domain,
                            "importance_score": config["importance_weight"],
                            "synonyms": synonyms,
                            "confidence": 0.8
                        }
                        concepts.append(concept)
        
        # 使用關鍵詞提取補充概念
        keyword_concepts = self._extract_keyword_concepts(content)
        concepts.extend(keyword_concepts)
        
        # 去重
        concepts = self._deduplicate_concepts(concepts)
        
        return concepts
    
    def _extract_keyword_concepts(self, content: str) -> List[Dict[str, Any]]:
        """使用關鍵詞提取補充法律概念"""
        concepts = []
        
        # 使用jieba提取關鍵詞
        if jieba:
            keywords = jieba.analyse.extract_tags(content, topK=10, withWeight=True)
        else:
            # 簡單的詞頻統計
            words = re.findall(r'[\u4e00-\u9fff]+', content)
            word_freq = Counter(words)
            keywords = [(word, freq/len(words)) for word, freq in word_freq.most_common(10)]
        
        for keyword, weight in keywords:
            if weight > 0.1:  # 只保留重要關鍵詞
                legal_domain = self._identify_legal_domain_from_keyword(keyword)
                if legal_domain != "其他":
                    concept = {
                        "concept_name": keyword,
                        "concept_type": "關鍵詞",
                        "legal_domain": legal_domain,
                        "importance_score": weight,
                        "synonyms": self.legal_synonyms.get(keyword, []),
                        "confidence": 0.6
                    }
                    concepts.append(concept)
        
        return concepts
    
    def _extract_semantic_keywords(self, content: str) -> Dict[str, Any]:
        """提取語義關鍵詞"""
        # 使用jieba分詞和TF-IDF
        if jieba:
            words = jieba.analyse.extract_tags(content, topK=20, withWeight=True)
        else:
            words = []
        
        # 分類關鍵詞
        legal_terms = ["權利", "義務", "禁止", "處罰", "規定", "適用", "違反", "侵害", "著作權", "商標", "專利"]
        domain_terms = ["著作權", "商標", "專利", "智慧財產權"]
        action_terms = ["保障", "維護", "促進", "制定", "保護", "限制"]
        
        primary_keywords = []
        secondary_keywords = []
        domain_keywords = []
        action_keywords = []
        keyword_weights = {}
        
        for word, weight in words:
            keyword_weights[word] = weight
            if word in legal_terms:
                primary_keywords.append(word)
            elif word in domain_terms:
                domain_keywords.append(word)
            elif word in action_terms:
                action_keywords.append(word)
            else:
                secondary_keywords.append(word)
        
        return {
            "primary_keywords": primary_keywords,
            "secondary_keywords": secondary_keywords,
            "domain_keywords": domain_keywords,
            "action_keywords": action_keywords,
            "keyword_weights": keyword_weights
        }
    
    def _classify_article_type(self, content: str) -> Dict[str, Any]:
        """分類條文類型"""
        detected_type = "一般規定"
        confidence = 0.5
        
        for article_type, patterns in self.article_type_patterns.items():
            match_count = sum(1 for pattern in patterns if pattern in content)
            if match_count > 0:
                type_confidence = min(0.9, match_count * 0.2 + 0.3)
                if type_confidence > confidence:
                    detected_type = article_type
                    confidence = type_confidence
        
        return {
            "article_type": detected_type,
            "article_purpose": f"定義{detected_type}的相關規定",
            "legal_function": "權利保護" if "權利" in detected_type else "義務規範",
            "scope": "一般適用",
            "severity": "重要" if detected_type in ["權利定義", "法律後果"] else "一般",
            "confidence": confidence
        }
    
    def _classify_legal_domain(self, content: str) -> Dict[str, Any]:
        """分類法律領域"""
        domain_scores = {}
        
        for domain, keywords in self.legal_domains.items():
            score = sum(1 for keyword in keywords if keyword in content)
            domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            max_score = best_domain[1]
            confidence = min(1.0, max_score / 5.0)
            
            return {
                "legal_domain": best_domain[0],
                "sub_domain": best_domain[0],
                "domain_keywords": [kw for kw in self.legal_domains[best_domain[0]] if kw in content],
                "related_domains": [d for d, s in domain_scores.items() if d != best_domain[0] and s > 0],
                "domain_confidence": confidence
            }
        
        return {
            "legal_domain": "其他",
            "sub_domain": "其他",
            "domain_keywords": [],
            "related_domains": [],
            "domain_confidence": 0.1
        }
    
    def _extract_legal_relations(self, content: str) -> List[Dict[str, Any]]:
        """提取法律關係"""
        relations = []
        
        # 權利義務關係
        rights_pattern = r"([^。，；：]+)(?:享有|具有|專有)([^。，；：]*)權利"
        obligations_pattern = r"([^。，；：]+)(?:應|必須)([^。，；：]*)"
        
        for pattern in [rights_pattern, obligations_pattern]:
            matches = re.finditer(pattern, content)
            for match in matches:
                subject = match.group(1).strip()
                relation = "享有" if "享有" in pattern else "應"
                object = match.group(2).strip()
                
                if subject and object:
                    relation_obj = {
                        "relation_type": "權利義務",
                        "subject": subject,
                        "object": object,
                        "relation": relation,
                        "confidence": 0.7
                    }
                    relations.append(relation_obj)
        
        return relations
    
    def _extract_query_intent_tags(self, content: str) -> List[str]:
        """提取查詢意圖標籤"""
        intent_tags = []
        
        intent_patterns = {
            "權利查詢": ["什麼是", "定義", "權利", "什麼權"],
            "義務查詢": ["必須", "應", "不得", "禁止", "義務"],
            "例外查詢": ["例外", "除外", "但", "惟", "不適用"],
            "後果查詢": ["處罰", "違反", "後果", "責任", "賠償"],
            "適用查詢": ["適用", "條件", "情況", "何時"]
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in content for pattern in patterns):
                intent_tags.append(intent)
        
        return intent_tags
    
    def _precompute_semantic_similarity(self, content: str) -> Dict[str, Any]:
        """預計算語義相似度"""
        # 提取常見查詢模式
        common_queries = []
        
        # 基於內容生成可能的查詢
        if "權利" in content:
            common_queries.extend(["什麼是權利", "權利的定義", "權利範圍"])
        if "義務" in content:
            common_queries.extend(["什麼是義務", "義務的規定", "義務內容"])
        if "處罰" in content or "違反" in content:
            common_queries.extend(["違反後果", "處罰規定", "法律責任"])
        
        return {
            "common_queries": common_queries[:5],  # 限制數量
            "similar_articles": [],  # 需要後續計算
            "semantic_cluster": "自動聚類結果"
        }
    
    def _identify_legal_domain_from_content(self, content: str) -> str:
        """從內容識別法律領域"""
        for domain, keywords in self.legal_domains.items():
            for keyword in keywords:
                if keyword in content:
                    return domain
        return "其他"
    
    def _identify_legal_domain_from_keyword(self, keyword: str) -> str:
        """從關鍵詞識別法律領域"""
        for domain, keywords in self.legal_domains.items():
            if keyword in keywords:
                return domain
        return "其他"
    
    def _extract_synonyms_for_concept(self, concept_name: str, content: str) -> List[str]:
        """為概念提取同義詞"""
        synonyms = []
        
        # 從同義詞字典獲取
        if concept_name in self.legal_synonyms:
            synonyms.extend(self.legal_synonyms[concept_name])
        
        # 從內容中提取同義詞模式
        synonym_patterns = [
            r'([^，。；：]*)(?:亦稱|又稱|別稱|俗稱)([^，。；：]*)',
            r'([^，。；：]*)(?:包括|含|涵蓋)([^，。；：]*)',
        ]
        
        for pattern in synonym_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if concept_name in match.group(0):
                    potential_synonyms = [group.strip() for group in match.groups() if group.strip()]
                    synonyms.extend(potential_synonyms)
        
        return list(set(synonyms))
    
    def _deduplicate_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重法律概念"""
        unique_concepts = []
        seen_names = set()
        
        for concept in concepts:
            concept_name = concept["concept_name"]
            if concept_name not in seen_names:
                seen_names.add(concept_name)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def _is_article_level_chunk(self, chunk: Dict[str, Any]) -> bool:
        """判斷是否為「條」層級的chunk"""
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        
        # 方法1：檢查metadata中的層級信息
        if metadata:
            level = metadata.get("level", "")
            chunk_by = metadata.get("chunk_by", "")
            if level == "basic_unit" or chunk_by == "article":
                return True
        
        # 方法2：檢查內容中是否包含條號模式
        article_patterns = [
            r"第[一二三四五六七八九十\d]+條",
            r"第\d+條",
            r"條文",
            r"規定"
        ]
        
        for pattern in article_patterns:
            if re.search(pattern, content):
                return True
        
        # 方法3：檢查chunk_id是否包含條層級標識
        chunk_id = chunk.get("chunk_id", "")
        if "article" in chunk_id.lower() or "條" in chunk_id:
            return True
        
        return False
    
    def _is_chapter_section_level_chunk(self, chunk: Dict[str, Any]) -> bool:
        """判斷是否為「章、節」層級的chunk"""
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        
        # 方法1：檢查metadata中的層級信息
        if metadata:
            level = metadata.get("level", "")
            chunk_by = metadata.get("chunk_by", "")
            if level in ["document_component", "basic_unit_hierarchy"] or chunk_by in ["chapter", "section"]:
                return True
        
        # 方法2：檢查內容中是否包含章節模式
        chapter_section_patterns = [
            r"第[一二三四五六七八九十\d]+章",
            r"第[一二三四五六七八九十\d]+節",
            r"章\s*[：:]",
            r"節\s*[：:]",
            r"總則|分則|附則"
        ]
        
        for pattern in chapter_section_patterns:
            if re.search(pattern, content):
                return True
        
        # 方法3：檢查chunk_id是否包含章節層級標識
        chunk_id = chunk.get("chunk_id", "")
        if any(keyword in chunk_id.lower() for keyword in ["chapter", "section", "章", "節"]):
            return True
        
        return False
    
    def _extract_article_id(self, chunk: Dict[str, Any]) -> Optional[str]:
        """從chunk中提取條ID"""
        metadata = chunk.get("metadata", {})
        content = chunk.get("content", "")
        
        # 從metadata提取
        if metadata:
            law_name = metadata.get("category", "")
            article_number = metadata.get("article_number")
            if law_name and article_number:
                return f"{law_name}_第{article_number}條"
        
        # 從內容提取
        match = re.search(r"第(\d+)條", content)
        if match:
            article_num = match.group(1)
            # 嘗試從內容中提取法名
            law_match = re.search(r"([^。，；：]+法)", content)
            if law_match:
                law_name = law_match.group(1)
                return f"{law_name}_第{article_num}條"
        
        return None
    
    def _find_parent_article_id(self, chunk: Dict[str, Any]) -> Optional[str]:
        """為非條層級的chunk查找父級條ID"""
        metadata = chunk.get("metadata", {})
        content = chunk.get("content", "")
        
        # 從metadata查找
        if metadata:
            law_name = metadata.get("category", "")
            article_number = metadata.get("article_number")
            if law_name and article_number:
                return f"{law_name}_第{article_number}條"
        
        # 從內容中查找條號
        # 查找最近的條號（向上搜索）
        lines = content.split('\n')
        for line in lines:
            match = re.search(r"第(\d+)條", line)
            if match:
                article_num = match.group(1)
                # 嘗試從上下文提取法名
                law_match = re.search(r"([^。，；：]+法)", content)
                if law_match:
                    law_name = law_match.group(1)
                    return f"{law_name}_第{article_num}條"
        
        return None
    
    def _enhance_article_level_chunk(self, content: str, original_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """為「條」層級chunk進行完整metadata增強"""
        # 清理原始metadata
        cleaned_metadata = self._clean_metadata(original_metadata)
        
        # 提取各種增強信息
        enhanced = {
            **cleaned_metadata,
            "legal_concepts": self._extract_legal_concepts(content),
            "semantic_keywords": self._extract_semantic_keywords(content),
            "article_type": self._classify_article_type(content),
            "legal_domain": self._classify_legal_domain(content),
            "legal_relations": self._extract_legal_relations(content),
            "query_intent_tags": self._extract_query_intent_tags(content),
            "semantic_similarity": self._precompute_semantic_similarity(content),
            "enhancement_level": "full",  # 標記為完整增強
            "is_article_level": True
        }
        
        return enhanced
    
    def _enhance_chapter_section_chunk(self, content: str, original_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """為「章、節」層級chunk進行中等強度metadata增強"""
        # 清理原始metadata
        cleaned_metadata = self._clean_metadata(original_metadata)
        
        # 進行中等強度的metadata增強
        enhanced = {
            **cleaned_metadata,
            "semantic_keywords": self._extract_chapter_section_keywords(content),
            "legal_domain": self._classify_legal_domain(content),
            "chapter_section_type": self._classify_chapter_section_type(content),
            "legal_concepts": self._extract_chapter_section_concepts(content),
            "scope_keywords": self._extract_scope_keywords(content),
            "enhancement_level": "medium",  # 標記為中等強度增強
            "is_article_level": False,
            "is_chapter_section_level": True
        }
        
        return enhanced
    
    def _enhance_lightweight_chunk(self, content: str, original_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """為非條層級chunk進行輕量級metadata增強"""
        # 清理原始metadata
        cleaned_metadata = self._clean_metadata(original_metadata)
        
        # 只進行基本的關鍵詞提取
        enhanced = {
            **cleaned_metadata,
            "semantic_keywords": self._extract_basic_keywords(content),
            "legal_domain": self._classify_basic_legal_domain(content),
            "enhancement_level": "lightweight",  # 標記為輕量級增強
            "is_article_level": False,
            "is_chapter_section_level": False
        }
        
        return enhanced
    
    def _extract_basic_keywords(self, content: str) -> Dict[str, Any]:
        """提取基本關鍵詞（輕量級版本）"""
        # 使用jieba分詞
        if jieba:
            words = jieba.analyse.extract_tags(content, topK=10, withWeight=True)
        else:
            words = []
        
        keyword_weights = {word: weight for word, weight in words}
        
        return {
            "primary_keywords": [word for word, weight in words if weight > 0.1],
            "keyword_weights": keyword_weights
        }
    
    def _extract_chapter_section_keywords(self, content: str) -> Dict[str, Any]:
        """提取章、節層級的關鍵詞"""
        # 使用jieba分詞
        if jieba:
            words = jieba.analyse.extract_tags(content, topK=15, withWeight=True)
        else:
            words = []
        
        # 分類關鍵詞
        structural_keywords = ["章", "節", "總則", "分則", "附則", "規定", "原則"]
        legal_terms = ["權利", "義務", "責任", "處罰", "程序", "適用"]
        scope_keywords = ["範圍", "定義", "原則", "一般", "特殊", "例外"]
        
        primary_keywords = []
        structural_keywords_list = []
        legal_terms_list = []
        keyword_weights = {}
        
        for word, weight in words:
            keyword_weights[word] = weight
            
            if word in structural_keywords:
                structural_keywords_list.append(word)
            elif word in legal_terms:
                legal_terms_list.append(word)
            elif weight > 0.1:
                primary_keywords.append(word)
        
        return {
            "primary_keywords": primary_keywords,
            "structural_keywords": structural_keywords_list,
            "legal_terms": legal_terms_list,
            "keyword_weights": keyword_weights
        }
    
    def _classify_chapter_section_type(self, content: str) -> Dict[str, Any]:
        """分類章、節類型"""
        chapter_type = "一般章節"
        confidence = 0.5
        
        # 檢查特殊章節類型
        if "總則" in content or "一般" in content:
            chapter_type = "總則性章節"
            confidence = 0.9
        elif "分則" in content or "特別" in content:
            chapter_type = "分則性章節"
            confidence = 0.9
        elif "附則" in content or "附" in content:
            chapter_type = "附則性章節"
            confidence = 0.9
        elif "罰則" in content or "處罰" in content:
            chapter_type = "罰則性章節"
            confidence = 0.8
        elif "程序" in content or "手續" in content:
            chapter_type = "程序性章節"
            confidence = 0.8
        
        return {
            "chapter_section_type": chapter_type,
            "type_description": f"定義{chapter_type}的相關規定",
            "confidence": confidence
        }
    
    def _extract_chapter_section_concepts(self, content: str) -> List[Dict[str, Any]]:
        """提取章、節層級的法律概念（簡化版）"""
        concepts = []
        
        # 提取結構性概念
        structural_concepts = {
            "總則": {"importance": 0.8, "type": "結構性概念"},
            "分則": {"importance": 0.8, "type": "結構性概念"},
            "附則": {"importance": 0.7, "type": "結構性概念"},
            "罰則": {"importance": 0.9, "type": "結構性概念"},
            "程序": {"importance": 0.7, "type": "程序性概念"}
        }
        
        for concept, info in structural_concepts.items():
            if concept in content:
                concepts.append({
                    "concept_name": concept,
                    "concept_type": info["type"],
                    "legal_domain": "程序法",
                    "importance_score": info["importance"],
                    "synonyms": [],
                    "confidence": 0.8
                })
        
        return concepts
    
    def _extract_scope_keywords(self, content: str) -> Dict[str, Any]:
        """提取範圍關鍵詞"""
        scope_patterns = {
            "適用範圍": ["適用", "範圍", "適用於"],
            "定義範圍": ["定義", "指", "謂"],
            "例外範圍": ["例外", "除外", "不適用"],
            "程序範圍": ["程序", "手續", "方式"]
        }
        
        detected_scopes = []
        for scope_type, keywords in scope_patterns.items():
            if any(keyword in content for keyword in keywords):
                detected_scopes.append(scope_type)
        
        return {
            "scope_types": detected_scopes,
            "scope_description": "定義適用範圍和限制條件"
        }
    
    def _classify_basic_legal_domain(self, content: str) -> Dict[str, Any]:
        """基本法律領域分類（輕量級版本）"""
        for domain, keywords in self.legal_domains.items():
            if any(keyword in content for keyword in keywords):
                return {
                    "legal_domain": domain,
                    "confidence": 0.7
                }
        
        return {
            "legal_domain": "其他",
            "confidence": 0.1
        }
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """獲取增強統計信息"""
        return {
            "cache_size": len(self.metadata_cache),
            "legal_domains": len(self.legal_domains),
            "concept_patterns": len(self.legal_concept_patterns),
            "article_type_patterns": len(self.article_type_patterns),
            "legal_synonyms": len(self.legal_synonyms),
            "article_metadata_count": len(self.article_metadata_map),
            "inheritance_relations": len(self.inheritance_hierarchy)
        }
    
    def get_article_metadata_map(self) -> Dict[str, Any]:
        """獲取條層級metadata映射"""
        return self.article_metadata_map.copy()
    
    def get_inheritance_hierarchy(self) -> Dict[str, str]:
        """獲取繼承關係映射"""
        return self.inheritance_hierarchy.copy()
