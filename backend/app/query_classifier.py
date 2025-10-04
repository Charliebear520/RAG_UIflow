"""
查詢分類器模組
用於根據查詢內容自動識別查詢類型，並選擇合適的embedding層次
"""

import re
from typing import Dict, List, Tuple
from enum import Enum


class QueryType(Enum):
    """查詢類型枚舉 - 對應論文中的六個粒度級別"""
    DOCUMENT = "document"                    # 文件層級查詢
    DOCUMENT_COMPONENT = "document_component"          # 文件組成部分層級查詢
    BASIC_UNIT_HIERARCHY = "basic_unit_hierarchy"        # 基本單位層次結構層級查詢
    BASIC_UNIT = "basic_unit"                  # 基本單位層級查詢
    BASIC_UNIT_COMPONENT = "basic_unit_component"        # 基本單位組成部分層級查詢
    ENUMERATION = "enumeration"                  # 列舉層級查詢
    MIXED = "mixed"                           # 混合型查詢


class LegalQueryClassifier:
    """法律查詢分類器"""
    
    def __init__(self):
        """初始化分類器，定義各類查詢的關鍵詞模式 - 對應論文中的六個粒度級別"""
        
        # 文件層級查詢關鍵詞 - 詢問整個法律文檔
        self.document_patterns = [
            r'整部|全文|整個|全部|完整|全部內容',
            r'整個.*?法|整部.*?法|全文.*?法',
            r'法律.*?全文|法規.*?全文|條例.*?全文',
            r'完整.*?法律|完整.*?法規|完整.*?條例'
        ]
        
        # 文件組成部分層級查詢關鍵詞 - 詢問章、部分、編、篇
        self.document_component_patterns = [
            r'第.*?章|第.*?編|第.*?篇|第.*?部分',
            r'章.*?內容|編.*?內容|篇.*?內容|部分.*?內容',
            r'總則|附則|罰則|附錄',
            r'第一章|第二章|第三章|第四章|第五章'
        ]
        
        # 基本單位層次結構層級查詢關鍵詞 - 詢問節、標題、章節
        self.basic_unit_hierarchy_patterns = [
            r'第.*?節|第.*?標題|第.*?章節',
            r'節.*?內容|標題.*?內容|章節.*?內容',
            r'第一節|第二節|第三節|第四節|第五節'
        ]
        
        # 基本單位層級查詢關鍵詞 - 詢問條文、法條
        self.basic_unit_patterns = [
            r'第.*?條|第.*?法條|條文.*?',
            r'第\d+條|第\d+之\d+條',
            r'條文.*?內容|法條.*?內容',
            r'第.*?條.*?規定|第.*?條.*?內容'
        ]
        
        # 基本單位組成部分層級查詢關鍵詞 - 詢問段落、主文、定義
        self.basic_unit_component_patterns = [
            r'段落|主文|內容|定義',
            r'第.*?項.*?內容|第.*?款.*?內容',
            r'本法所稱|定義.*?為|所謂.*?係指',
            r'指.*?者|為.*?者|定義.*?為'
        ]
        
        # 列舉層級查詢關鍵詞 - 詢問項、目、款、子項
        self.enumeration_patterns = [
            r'第.*?項|第.*?目|第.*?款|第.*?子項',
            r'[（(]\d+[）)]|[一二三四五六七八九十]+[、．]',
            r'項.*?內容|目.*?內容|款.*?內容',
            r'第一項|第二項|第三項|第四項|第五項'
        ]
        
        # 編譯正則表達式
        self.document_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.document_patterns]
        self.document_component_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.document_component_patterns]
        self.basic_unit_hierarchy_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.basic_unit_hierarchy_patterns]
        self.basic_unit_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.basic_unit_patterns]
        self.basic_unit_component_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.basic_unit_component_patterns]
        self.enumeration_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.enumeration_patterns]
    
    def classify_query(self, query: str) -> Tuple[QueryType, float, Dict[str, float]]:
        """
        分類查詢並返回查詢類型、置信度和各類別得分 - 對應論文中的六個粒度級別
        
        Args:
            query: 用戶查詢文本
            
        Returns:
            Tuple[QueryType, float, Dict[str, float]]: (查詢類型, 置信度, 各類別得分)
        """
        query_lower = query.lower().strip()
        
        # 計算各類別的匹配得分
        document_score = self._calculate_score(query_lower, self.document_regex)
        document_component_score = self._calculate_score(query_lower, self.document_component_regex)
        basic_unit_hierarchy_score = self._calculate_score(query_lower, self.basic_unit_hierarchy_regex)
        basic_unit_score = self._calculate_score(query_lower, self.basic_unit_regex)
        basic_unit_component_score = self._calculate_score(query_lower, self.basic_unit_component_regex)
        enumeration_score = self._calculate_score(query_lower, self.enumeration_regex)
        
        scores = {
            'document': document_score,
            'document_component': document_component_score,
            'basic_unit_hierarchy': basic_unit_hierarchy_score,
            'basic_unit': basic_unit_score,
            'basic_unit_component': basic_unit_component_score,
            'enumeration': enumeration_score
        }
        
        # 確定最高得分的類別
        max_score = max(scores.values())
        max_category = max(scores, key=scores.get)
        
        # 計算置信度
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.0
        
        # 如果最高得分太低，可能是混合型查詢
        if max_score < 0.3 and total_score > 0.5:
            query_type = QueryType.MIXED
            confidence = 0.5  # 混合型查詢的置信度設為中等
        else:
            query_type = QueryType(max_category)
        
        return query_type, confidence, scores
    
    def _calculate_score(self, query: str, regex_patterns: List[re.Pattern]) -> float:
        """計算查詢與模式列表的匹配得分"""
        score = 0.0
        
        for pattern in regex_patterns:
            matches = pattern.findall(query)
            if matches:
                # 根據匹配數量增加得分
                score += len(matches) * 0.1
                # 根據匹配長度增加得分
                for match in matches:
                    if isinstance(match, str):
                        score += len(match) * 0.01
        
        return min(score, 1.0)  # 限制最大得分為1.0
    
    def get_recommended_level(self, query: str) -> str:
        """
        根據查詢類型推薦合適的embedding層次 - 對應論文中的六個粒度級別
        
        Args:
            query: 用戶查詢文本
            
        Returns:
            str: 推薦的embedding層次名稱
        """
        query_type, confidence, scores = self.classify_query(query)
        
        # 根據查詢類型映射到embedding層次
        level_mapping = {
            QueryType.DOCUMENT: 'document',
            QueryType.DOCUMENT_COMPONENT: 'document_component',
            QueryType.BASIC_UNIT_HIERARCHY: 'basic_unit_hierarchy',
            QueryType.BASIC_UNIT: 'basic_unit',
            QueryType.BASIC_UNIT_COMPONENT: 'basic_unit_component',
            QueryType.ENUMERATION: 'enumeration',
            QueryType.MIXED: 'basic_unit'  # 混合型查詢默認使用基本單位層
        }
        
        return level_mapping.get(query_type, 'basic_unit')
    
    def get_query_analysis(self, query: str) -> Dict[str, any]:
        """
        獲取查詢的詳細分析結果 - 對應論文中的六個粒度級別
        
        Args:
            query: 用戶查詢文本
            
        Returns:
            Dict: 包含查詢分析的詳細信息
        """
        query_type, confidence, scores = self.classify_query(query)
        recommended_level = self.get_recommended_level(query)
        
        return {
            'query': query,
            'query_type': query_type.value,
            'confidence': confidence,
            'scores': scores,
            'recommended_level': recommended_level,
            'analysis': {
                'is_document': scores['document'] > 0.3,
                'is_document_component': scores['document_component'] > 0.3,
                'is_basic_unit_hierarchy': scores['basic_unit_hierarchy'] > 0.3,
                'is_basic_unit': scores['basic_unit'] > 0.3,
                'is_basic_unit_component': scores['basic_unit_component'] > 0.3,
                'is_enumeration': scores['enumeration'] > 0.3,
                'is_mixed': query_type == QueryType.MIXED
            }
        }


# 全局分類器實例
query_classifier = LegalQueryClassifier()


def classify_legal_query(query: str) -> Tuple[QueryType, float, str]:
    """
    便捷函數：分類法律查詢並返回推薦層次
    
    Args:
        query: 用戶查詢文本
        
    Returns:
        Tuple[QueryType, float, str]: (查詢類型, 置信度, 推薦層次)
    """
    query_type, confidence, _ = query_classifier.classify_query(query)
    recommended_level = query_classifier.get_recommended_level(query)
    
    return query_type, confidence, recommended_level


def get_query_analysis(query: str) -> Dict[str, any]:
    """
    便捷函數：獲取查詢的詳細分析
    
    Args:
        query: 用戶查詢文本
        
    Returns:
        Dict: 查詢分析結果
    """
    return query_classifier.get_query_analysis(query)
