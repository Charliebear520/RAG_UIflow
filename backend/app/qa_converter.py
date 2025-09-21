"""
QA Set轉換器
用於將用戶上傳的QA set轉換為標準格式，自動補充缺失的span信息
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple


class QASetConverter:
    def __init__(self, law_json_data: Dict[str, Any]):
        """
        初始化QA set轉換器
        
        Args:
            law_json_data: 法條JSON數據，用於查找span信息
        """
        self.law_json_data = law_json_data
        self.law_spans = self._extract_law_spans()
        
    def _extract_law_spans(self) -> List[Dict[str, Any]]:
        """從法條JSON中提取所有span信息"""
        law_spans = []
        
        if "laws" in self.law_json_data:
            for law in self.law_json_data["laws"]:
                for chapter in law.get("chapters", []):
                    for section in chapter.get("sections", []):
                        for article in section.get("articles", []):
                            if "metadata" in article and "spans" in article["metadata"]:
                                for span in article["metadata"]["spans"]:
                                    law_spans.append({
                                        "start_char": span["start_char"],
                                        "end_char": span["end_char"],
                                        "text": span["text"],
                                        "article_id": article["metadata"]["id"],
                                        "article_name": article.get("article", ""),
                                        "chapter_name": chapter.get("chapter", ""),
                                        "section_name": section.get("section", ""),
                                        "file_path": "copyright.json"  # 默認文件路徑
                                    })
        return law_spans
    
    def convert_qa_set(self, qa_set: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        轉換QA set，補充缺失的span信息
        
        Args:
            qa_set: 原始QA set
            
        Returns:
            轉換後的QA set
        """
        converted_qa_set = []
        
        for qa_item in qa_set:
            converted_item = qa_item.copy()
            
            # 處理spans信息
            if "spans" in converted_item and converted_item["spans"]:
                # 有spans但可能缺少file_path
                converted_spans = []
                for span in converted_item["spans"]:
                    converted_span = span.copy()
                    if "file_path" not in converted_span:
                        converted_span["file_path"] = "copyright.json"
                    converted_spans.append(converted_span)
                converted_item["spans"] = converted_spans
            else:
                # 沒有spans，嘗試根據answer內容查找
                spans = self._find_spans_by_content(qa_item)
                if spans:
                    converted_item["spans"] = spans
                else:
                    # 如果還是找不到，設置為空數組
                    converted_item["spans"] = []
            
            # 確保有relevant_chunks字段（初始為空）
            if "relevant_chunks" not in converted_item:
                converted_item["relevant_chunks"] = []
                
            converted_qa_set.append(converted_item)
            
        return converted_qa_set
    
    def _find_spans_by_content(self, qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根據QA項目的內容查找對應的span信息
        
        Args:
            qa_item: QA項目
            
        Returns:
            找到的span信息列表
        """
        answer = qa_item.get("answer", "")
        query = qa_item.get("query", "")
        
        if not answer:
            return []
        
        # 方法1: 根據法條號碼查找
        article_spans = self._find_spans_by_article_number(query, answer)
        if article_spans:
            return article_spans
        
        # 方法2: 根據答案內容的關鍵詞匹配
        content_spans = self._find_spans_by_keywords(answer)
        if content_spans:
            return content_spans
            
        return []
    
    def _find_spans_by_article_number(self, query: str, answer: str) -> List[Dict[str, Any]]:
        """根據法條號碼查找span信息"""
        # 提取法條號碼
        article_patterns = [
            r"第(\d+)條",
            r"第(\d+)條之(\d+)",
            r"第(\d+)-(\d+)條"
        ]
        
        found_articles = []
        
        # 從query中提取法條號碼
        for pattern in article_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    article_num = match[0]
                else:
                    article_num = match
                found_articles.append(article_num)
        
        # 從answer中提取法條號碼
        for pattern in article_patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                if isinstance(match, tuple):
                    article_num = match[0]
                else:
                    article_num = match
                found_articles.append(article_num)
        
        # 根據法條號碼查找對應的spans
        result_spans = []
        for article_num in found_articles:
            for span in self.law_spans:
                # 更精確的匹配：檢查法條名稱是否包含該號碼
                if self._is_article_match(article_num, span["article_name"]):
                    # 只添加有效的span（start_char != end_char）
                    if span["start_char"] != span["end_char"]:
                        result_spans.append({
                            "start_char": span["start_char"],
                            "end_char": span["end_char"],
                            "text": span["text"],
                            "file_path": span["file_path"],
                            "page": 1,  # 默認頁碼
                            "confidence": 0.9,  # 法條號碼匹配的置信度較高
                            "found": True
                        })
        
        return result_spans
    
    def _is_article_match(self, article_num: str, article_name: str) -> bool:
        """檢查法條號碼是否匹配"""
        if not article_name:
            return False
        
        # 直接包含法條號碼
        if article_num in article_name:
            return True
        
        # 檢查是否為"第X條"格式
        if f"第{article_num}條" in article_name:
            return True
        
        return False
    
    def _find_spans_by_keywords(self, answer: str) -> List[Dict[str, Any]]:
        """根據關鍵詞匹配查找span信息"""
        # 提取答案中的關鍵詞
        keywords = self._extract_keywords(answer)
        
        if not keywords:
            return []
        
        # 在法條spans中查找包含關鍵詞的內容
        matching_spans = []
        for span in self.law_spans:
            span_text = span["text"]
            # 計算關鍵詞匹配度
            match_score = self._calculate_keyword_match(keywords, span_text)
            if match_score > 0.3:  # 匹配度閾值
                matching_spans.append({
                    "start_char": span["start_char"],
                    "end_char": span["end_char"],
                    "text": span["text"],
                    "file_path": span["file_path"],
                    "page": 1,
                    "confidence": match_score,
                    "found": True
                })
        
        # 按匹配度排序，返回前3個
        matching_spans.sort(key=lambda x: x["confidence"], reverse=True)
        return matching_spans[:3]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """從文本中提取關鍵詞"""
        # 移除標點符號和常見停用詞
        stop_words = {"的", "是", "在", "有", "和", "與", "或", "但", "而", "則", "即", "如", "若", "因", "為", "以", "及", "等", "之", "者", "所", "得", "可", "應", "須", "應", "得", "可", "須", "應", "得", "可", "須"}
        
        # 簡單的關鍵詞提取
        words = re.findall(r'[\u4e00-\u9fff]+', text)  # 提取中文字符
        keywords = [word for word in words if len(word) >= 2 and word not in stop_words]
        
        # 去重並限制數量
        return list(set(keywords))[:10]
    
    def _calculate_keyword_match(self, keywords: List[str], text: str) -> float:
        """計算關鍵詞匹配度"""
        if not keywords:
            return 0.0
        
        matches = 0
        for keyword in keywords:
            if keyword in text:
                matches += 1
        
        return matches / len(keywords)
    
    def validate_converted_qa_set(self, converted_qa_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        驗證轉換後的QA set
        
        Returns:
            驗證結果統計
        """
        total_items = len(converted_qa_set)
        items_with_spans = sum(1 for item in converted_qa_set if item.get("spans"))
        items_with_file_path = 0
        
        for item in converted_qa_set:
            if item.get("spans"):
                for span in item["spans"]:
                    if span.get("file_path"):
                        items_with_file_path += 1
                        break
        
        return {
            "total_items": total_items,
            "items_with_spans": items_with_spans,
            "items_with_file_path": items_with_file_path,
            "span_coverage": items_with_spans / total_items if total_items > 0 else 0,
            "file_path_coverage": items_with_file_path / total_items if total_items > 0 else 0
        }


def convert_qa_set_with_law_data(qa_set: List[Dict[str, Any]], law_json_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    使用法條JSON數據轉換QA set的便捷函數
    
    Args:
        qa_set: 原始QA set
        law_json_data: 法條JSON數據
        
    Returns:
        (轉換後的QA set, 驗證統計)
    """
    converter = QASetConverter(law_json_data)
    converted_qa_set = converter.convert_qa_set(qa_set)
    validation_stats = converter.validate_converted_qa_set(converted_qa_set)
    
    return converted_qa_set, validation_stats
