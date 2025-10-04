"""
法律語義完整性分塊策略
基於法律概念的完整性和語義邊界進行分塊
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class LegalConcept:
    """法律概念數據結構"""
    concept_type: str  # 'definition', 'right', 'exception', 'condition', 'procedure'
    content: str
    start_pos: int
    end_pos: int
    importance_score: float
    related_articles: List[str]
    semantic_boundaries: List[int]


class LegalSemanticIntegrityChunking:
    """法律語義完整性分塊策略"""
    
    def __init__(self):
        # 法律概念完整性規則
        self.legal_concept_patterns = {
            'definition': [
                r'本法所稱.*?是指.*?(?=第|$|本法|前項|後項)',
                r'.*?指.*?者.*?(?=第|$|本法|前項|後項)',
                r'.*?為.*?者.*?(?=第|$|本法|前項|後項)'
            ],
            'right_definition': [
                r'著作人專有.*?權利.*?(?=第|$|本法|前項|後項)',
                r'商標權人.*?權利.*?(?=第|$|本法|前項|後項)',
                r'.*?專有.*?權利.*?(?=第|$|本法|前項|後項)'
            ],
            'exception_clause': [
                r'但.*?(?=第|$|本法|前項|後項)',
                r'不在此限.*?(?=第|$|本法|前項|後項)',
                r'除外.*?(?=第|$|本法|前項|後項)'
            ],
            'conditional_clause': [
                r'有下列情形之一.*?(?=第|$|本法|前項|後項)',
                r'如.*?時.*?(?=第|$|本法|前項|後項)',
                r'於.*?情形.*?(?=第|$|本法|前項|後項)'
            ],
            'procedural_clause': [
                r'應.*?申請.*?(?=第|$|本法|前項|後項)',
                r'得.*?辦理.*?(?=第|$|本法|前項|後項)',
                r'依.*?程序.*?(?=第|$|本法|前項|後項)'
            ]
        }
        
        # 語義邊界標記
        self.semantic_boundaries = {
            'concept_shift': ['本法', '前項', '後項', '但', '因此', '所以'],
            'logical_connector': ['因此', '所以', '但是', '然而', '另外', '此外'],
            'legal_structure': ['第.*條', '第.*章', '第.*節', '第.*項'],
            'scope_marker': ['在.*範圍內', '除.*外', '不適用於']
        }
        
        # 法律概念重要性權重
        self.concept_importance_weights = {
            'definition': 1.0,      # 定義最重要
            'right_definition': 0.9, # 權利定義
            'exception_clause': 0.8, # 例外條款
            'conditional_clause': 0.7, # 條件條款
            'procedural_clause': 0.6   # 程序條款
        }
    
    def chunk(self, text: str, max_chunk_size: int = 1000, 
              overlap_ratio: float = 0.1, preserve_concepts: bool = True,
              **kwargs) -> List[Dict[str, Any]]:
        """
        基於法律語義完整性的分塊
        
        Args:
            text: 輸入文本
            max_chunk_size: 最大分塊大小
            overlap_ratio: 重疊比例
            preserve_concepts: 是否保持概念完整性
            
        Returns:
            List[Dict]: 包含content、metadata、concepts的chunk列表
        """
        
        # 1. 識別法律概念邊界
        legal_concepts = self._identify_legal_concepts(text)
        print(f"🔍 識別到 {len(legal_concepts)} 個法律概念")
        
        # 2. 計算語義邊界
        semantic_boundaries = self._calculate_semantic_boundaries(text, legal_concepts)
        print(f"🔍 計算出 {len(semantic_boundaries)} 個語義邊界")
        
        # 3. 基於語義連貫性分塊
        semantic_chunks = self._create_semantic_chunks(
            text, semantic_boundaries, max_chunk_size, overlap_ratio
        )
        print(f"🔍 創建 {len(semantic_chunks)} 個語義分塊")
        
        # 4. 確保概念完整性
        if preserve_concepts:
            integrity_chunks = self._ensure_concept_integrity(
                semantic_chunks, legal_concepts
            )
            print(f"🔍 確保概念完整性後得到 {len(integrity_chunks)} 個分塊")
            return integrity_chunks
        
        return semantic_chunks
    
    def _identify_legal_concepts(self, text: str) -> List[LegalConcept]:
        """識別法律概念"""
        concepts = []
        
        for concept_type, patterns in self.legal_concept_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
                
                for match in matches:
                    concept = LegalConcept(
                        concept_type=concept_type,
                        content=match.group().strip(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        importance_score=self.concept_importance_weights.get(concept_type, 0.5),
                        related_articles=self._extract_related_articles(match.group()),
                        semantic_boundaries=self._find_semantic_boundaries(match.group())
                    )
                    concepts.append(concept)
        
        # 按重要性排序
        concepts.sort(key=lambda x: x.importance_score, reverse=True)
        return concepts
    
    def _extract_related_articles(self, content: str) -> List[str]:
        """提取相關法條"""
        article_pattern = r'第\s*(\d+)\s*條'
        articles = re.findall(article_pattern, content)
        return [f"第{article}條" for article in articles]
    
    def _find_semantic_boundaries(self, content: str) -> List[int]:
        """找到語義邊界"""
        boundaries = []
        
        for boundary_type, markers in self.semantic_boundaries.items():
            for marker in markers:
                if isinstance(marker, str):
                    # 字符串標記
                    pos = content.find(marker)
                    if pos != -1:
                        boundaries.append(pos)
                else:
                    # 正則表達式標記
                    pattern = re.compile(marker)
                    for match in pattern.finditer(content):
                        boundaries.append(match.start())
        
        return sorted(set(boundaries))
    
    def _calculate_semantic_boundaries(self, text: str, 
                                     concepts: List[LegalConcept]) -> List[int]:
        """計算語義邊界"""
        boundaries = set()
        
        # 添加概念邊界
        for concept in concepts:
            boundaries.add(concept.start_pos)
            boundaries.add(concept.end_pos)
            
            # 添加概念內的語義邊界
            for boundary in concept.semantic_boundaries:
                boundaries.add(concept.start_pos + boundary)
        
        # 添加結構邊界（法條、章節）
        structure_pattern = r'第\s*\d+\s*[條章節項]'
        for match in re.finditer(structure_pattern, text):
            boundaries.add(match.start())
        
        # 添加句子邊界
        sentence_pattern = r'[。！？]'
        for match in re.finditer(sentence_pattern, text):
            boundaries.add(match.end())
        
        return sorted(list(boundaries))
    
    def _create_semantic_chunks(self, text: str, boundaries: List[int],
                               max_chunk_size: int, overlap_ratio: float) -> List[Dict[str, Any]]:
        """創建語義分塊"""
        chunks = []
        overlap_size = int(max_chunk_size * overlap_ratio)
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # 找到最適合的結束位置
            end = self._find_optimal_end_position(
                text, start, max_chunk_size, boundaries
            )
            
            if end <= start:
                # 如果找不到合適的結束位置，強制結束
                end = min(start + max_chunk_size, len(text))
            
            chunk_content = text[start:end]
            
            # 分析chunk的語義特徵
            semantic_features = self._analyze_chunk_semantics(chunk_content)
            
            chunk = {
                "content": chunk_content,
                "span": {"start": start, "end": end},
                "metadata": {
                    "strategy": "legal_semantic_integrity",
                    "chunk_index": chunk_index,
                    "length": len(chunk_content),
                    "semantic_features": semantic_features,
                    "concept_density": semantic_features.get("concept_count", 0) / max(len(chunk_content), 1),
                    "importance_score": semantic_features.get("importance_score", 0.0)
                }
            }
            
            chunks.append(chunk)
            chunk_index += 1
            
            # 計算下一個chunk的開始位置（考慮重疊）
            start = max(start + max_chunk_size - overlap_size, end - overlap_size)
            
            if start >= len(text):
                break
        
        return chunks
    
    def _find_optimal_end_position(self, text: str, start: int, 
                                  max_chunk_size: int, boundaries: List[int]) -> int:
        """找到最優的結束位置"""
        ideal_end = start + max_chunk_size
        
        # 找到最接近理想結束位置的語義邊界
        optimal_boundary = None
        min_distance = float('inf')
        
        for boundary in boundaries:
            if start < boundary <= ideal_end:
                distance = abs(boundary - ideal_end)
                if distance < min_distance:
                    min_distance = distance
                    optimal_boundary = boundary
        
        if optimal_boundary:
            return optimal_boundary
        
        # 如果沒有找到合適的邊界，返回理想結束位置
        return min(ideal_end, len(text))
    
    def _analyze_chunk_semantics(self, chunk_content: str) -> Dict[str, Any]:
        """分析chunk的語義特徵"""
        features = {
            "concept_count": 0,
            "importance_score": 0.0,
            "concept_types": [],
            "has_definition": False,
            "has_exception": False,
            "has_condition": False,
            "article_references": []
        }
        
        # 檢查各種概念類型
        for concept_type, patterns in self.legal_concept_patterns.items():
            for pattern in patterns:
                if re.search(pattern, chunk_content, re.MULTILINE | re.DOTALL):
                    features["concept_count"] += 1
                    features["concept_types"].append(concept_type)
                    features["importance_score"] += self.concept_importance_weights.get(concept_type, 0.5)
                    
                    if concept_type == "definition":
                        features["has_definition"] = True
                    elif concept_type == "exception_clause":
                        features["has_exception"] = True
                    elif concept_type == "conditional_clause":
                        features["has_condition"] = True
        
        # 提取法條引用
        article_pattern = r'第\s*(\d+)\s*條'
        articles = re.findall(article_pattern, chunk_content)
        features["article_references"] = [f"第{article}條" for article in articles]
        
        return features
    
    def _ensure_concept_integrity(self, chunks: List[Dict[str, Any]], 
                                 concepts: List[LegalConcept]) -> List[Dict[str, Any]]:
        """確保概念完整性"""
        integrity_chunks = []
        
        for chunk in chunks:
            chunk_start = chunk["span"]["start"]
            chunk_end = chunk["span"]["end"]
            
            # 檢查是否有概念被分割
            incomplete_concepts = []
            for concept in concepts:
                concept_start = concept.start_pos
                concept_end = concept.end_pos
                
                # 如果概念跨越chunk邊界
                if (concept_start < chunk_end and concept_end > chunk_start and
                    not (chunk_start <= concept_start and concept_end <= chunk_end)):
                    incomplete_concepts.append(concept)
            
            # 如果有不完整的概念，調整chunk邊界
            if incomplete_concepts:
                adjusted_chunk = self._adjust_chunk_for_concepts(
                    chunk, incomplete_concepts
                )
                integrity_chunks.append(adjusted_chunk)
            else:
                integrity_chunks.append(chunk)
        
        return integrity_chunks
    
    def _adjust_chunk_for_concepts(self, chunk: Dict[str, Any], 
                                  incomplete_concepts: List[LegalConcept]) -> Dict[str, Any]:
        """調整chunk以包含完整概念"""
        original_start = chunk["span"]["start"]
        original_end = chunk["span"]["end"]
        
        # 找到需要包含的所有概念的邊界
        min_start = original_start
        max_end = original_end
        
        for concept in incomplete_concepts:
            min_start = min(min_start, concept.start_pos)
            max_end = max(max_end, concept.end_pos)
        
        # 更新chunk內容和元數據
        adjusted_content = chunk["content"]  # 這裡需要從原始文本重新提取
        
        chunk["span"]["start"] = min_start
        chunk["span"]["end"] = max_end
        chunk["metadata"]["adjusted_for_concepts"] = True
        chunk["metadata"]["incomplete_concept_count"] = len(incomplete_concepts)
        
        return chunk


class MultiLevelSemanticChunking:
    """多層次語義分塊策略 - 對應論文中的六個粒度級別"""
    
    def __init__(self):
        # 論文中的六個層次配置
        self.semantic_levels = {
            'document': {
                'granularity': 'document',
                'size_range': (2000, 10000),
                'target_queries': ['整部', '全文', '整個', '全部'],
                'priority_concepts': ['complete_document'],
                'description': '文件層級 (Document Level) - 整個法律文檔'
            },
            'document_component': {
                'granularity': 'document_component',
                'size_range': (1000, 3000),
                'target_queries': ['章', '部分', '編', '篇'],
                'priority_concepts': ['chapter', 'part', 'section'],
                'description': '文件組成部分層級 (Document Component Level) - 文檔的主要組成部分'
            },
            'basic_unit_hierarchy': {
                'granularity': 'basic_unit_hierarchy',
                'size_range': (500, 1500),
                'target_queries': ['節', '標題', '章節'],
                'priority_concepts': ['section', 'title', 'subsection'],
                'description': '基本單位層次結構層級 (Basic Unit Hierarchy Level) - 書籍、標題、章節'
            },
            'basic_unit': {
                'granularity': 'basic_unit',
                'size_range': (200, 800),
                'target_queries': ['第.*條', '條文', '法條'],
                'priority_concepts': ['article', 'clause', 'provision'],
                'description': '基本單位層級 (Basic Unit Level) - 文章/條文 (article)'
            },
            'basic_unit_component': {
                'granularity': 'basic_unit_component',
                'size_range': (100, 400),
                'target_queries': ['段落', '主文', '內容', '定義'],
                'priority_concepts': ['definition', 'paragraph', 'content'],
                'description': '基本單位組成部分層級 (Basic Unit Component Level) - 強制性主文或段落'
            },
            'enumeration': {
                'granularity': 'enumeration',
                'size_range': (50, 200),
                'target_queries': ['項', '目', '款', '子項'],
                'priority_concepts': ['item', 'subitem', 'enumeration'],
                'description': '列舉層級 (Enumeration Level) - 項目、子項'
            }
        }
    
    def chunk(self, text: str, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """創建多層次語義分塊"""
        multi_level_chunks = {}
        
        # 使用法律語義完整性分塊作為基礎
        base_chunker = LegalSemanticIntegrityChunking()
        
        for level_name, level_config in self.semantic_levels.items():
            # 根據層次配置調整參數
            level_kwargs = kwargs.copy()
            level_kwargs['max_chunk_size'] = level_config['size_range'][1]
            level_kwargs['min_chunk_size'] = level_config['size_range'][0]
            
            # 創建該層次的分塊
            level_chunks = base_chunker.chunk(text, **level_kwargs)
            
            # 根據層次特性過濾和調整分塊
            filtered_chunks = self._filter_chunks_for_level(
                level_chunks, level_config
            )
            
            multi_level_chunks[level_name] = filtered_chunks
        
        return multi_level_chunks
    
    def _filter_chunks_for_level(self, chunks: List[Dict[str, Any]], 
                                level_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根據層次特性過濾分塊"""
        filtered_chunks = []
        
        for chunk in chunks:
            semantic_features = chunk["metadata"].get("semantic_features", {})
            concept_types = semantic_features.get("concept_types", [])
            
            # 檢查是否包含該層次優先的概念類型
            has_priority_concept = any(
                concept_type in concept_types 
                for concept_type in level_config['priority_concepts']
            )
            
            # 檢查chunk大小是否在範圍內
            chunk_size = len(chunk["content"])
            size_in_range = (
                level_config['size_range'][0] <= chunk_size <= level_config['size_range'][1]
            )
            
            if has_priority_concept or size_in_range:
                # 添加層次特定的元數據
                chunk["metadata"]["semantic_level"] = level_config['granularity']
                chunk["metadata"]["target_queries"] = level_config['target_queries']
                filtered_chunks.append(chunk)
        
        return filtered_chunks
