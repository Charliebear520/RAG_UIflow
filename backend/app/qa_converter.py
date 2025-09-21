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
                if law['law_name'] == '法規名稱：著作權法':
                    for chapter in law.get("chapters", []):
                        for section in chapter.get("sections", []):
                            for article in section.get("articles", []):
                                article_name = article.get("article", "")
                                
                                # 處理article層級的spans
                                if "metadata" in article and "spans" in article["metadata"]:
                                    for span in article["metadata"]["spans"]:
                                        if isinstance(span, dict) and 'start_char' in span and 'end_char' in span:
                                            law_spans.append({
                                                "start_char": span["start_char"],
                                                "end_char": span["end_char"],
                                                "text": span.get("text", article.get("content", "")),
                                                "article_id": article["metadata"]["id"],
                                                "article_name": article_name,
                                                "chapter_name": chapter.get("chapter", ""),
                                                "section_name": section.get("section", ""),
                                                "file_path": "copyright&tradmark.json"
                                            })
                                
                                # 處理items和sub_items
                                for item in article.get("items", []):
                                    item_name = item.get("item", "")
                                    
                                    # 處理item的sub_items
                                    for sub_item in item.get("sub_items", []):
                                        sub_item_content = sub_item.get("content", "")
                                        sub_item_metadata = sub_item.get("metadata", {})
                                        
                                        if "spans" in sub_item_metadata:
                                            spans = sub_item_metadata["spans"]
                                            if isinstance(spans, list):
                                                for span in spans:
                                                    if isinstance(span, dict) and 'start_char' in span and 'end_char' in span:
                                                        law_spans.append({
                                                            "start_char": span["start_char"],
                                                            "end_char": span["end_char"],
                                                            "text": span.get("text", sub_item_content),
                                                            "article_id": article["metadata"]["id"],
                                                            "article_name": article_name,
                                                            "item_name": item_name,
                                                            "sub_item_name": sub_item.get("sub_item", ""),
                                                            "chapter_name": chapter.get("chapter", ""),
                                                            "section_name": section.get("section", ""),
                                                            "file_path": "copyright&tradmark.json"
                                                        })
                                            elif isinstance(spans, dict):
                                                if 'start' in spans and 'end' in spans:
                                                    # 對於相對位置，我們需要計算絕對位置
                                                    # 這裡使用一個簡化的方法：基於內容長度估算
                                                    start_pos = len(sub_item_content) + spans['start']
                                                    end_pos = len(sub_item_content) + spans['end']
                                                    law_spans.append({
                                                        "start_char": start_pos,
                                                        "end_char": end_pos,
                                                        "text": sub_item_content,
                                                        "article_id": article["metadata"]["id"],
                                                        "article_name": article_name,
                                                        "item_name": item_name,
                                                        "sub_item_name": sub_item.get("sub_item", ""),
                                                        "chapter_name": chapter.get("chapter", ""),
                                                        "section_name": section.get("section", ""),
                                                        "file_path": "copyright&tradmark.json"
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
            # 只保留標準格式需要的字段
            converted_item = {
                "query": qa_item.get("query", ""),
                "label": qa_item.get("label", ""),
                "answer": qa_item.get("answer", "")
            }
            
            # 處理新的snippets格式
            if "snippets" in converted_item and converted_item["snippets"]:
                # 已有snippets，確保格式正確
                converted_item["snippets"] = self._normalize_snippets(converted_item["snippets"])
            elif "spans" in converted_item and converted_item["spans"]:
                # 舊格式spans，轉換為新格式snippets
                converted_item["snippets"] = self._convert_spans_to_snippets(converted_item["spans"])
                # 移除舊的spans字段
                if "spans" in converted_item:
                    del converted_item["spans"]
            else:
                # 沒有spans或snippets，嘗試根據answer內容查找
                snippets = self._find_snippets_by_content(qa_item)
                if snippets:
                    converted_item["snippets"] = snippets
                else:
                    # 如果精確匹配失敗，嘗試寬鬆匹配
                    snippets = self._find_snippets_by_keywords(qa_item)
                    if snippets:
                        converted_item["snippets"] = snippets
                    else:
                        # 如果還是找不到，設置為空數組
                        converted_item["snippets"] = []
            
            # 不需要relevant_chunks字段，這是Chunk頁面映射時才需要的
                
            converted_qa_set.append(converted_item)
            
        return converted_qa_set
    
    def _normalize_snippets(self, snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """標準化snippets格式"""
        normalized = []
        for snippet in snippets:
            normalized_snippet = {
                "file_path": snippet.get("file_path", "copyright&tradmark.json"),
                "span": snippet.get("span", [0, 0])
            }
            normalized.append(normalized_snippet)
        return normalized
    
    def _convert_spans_to_snippets(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """將舊格式的spans轉換為新格式的snippets"""
        snippets = []
        for span in spans:
            snippet = {
                "file_path": span.get("file_path", "copyright&tradmark.json"),
                "span": [span.get("start_char", 0), span.get("end_char", 0)]
            }
            snippets.append(snippet)
        return snippets
    
    def _find_snippets_by_content(self, qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根據QA項目的內容查找對應的snippets信息"""
        answer = qa_item.get("answer", "")
        query = qa_item.get("query", "")
        if not answer:
            return []
        
        # 策略1: 精確條號匹配
        article_numbers = self._extract_all_article_numbers(answer + " " + query)
        if article_numbers:
            matching_spans = self._find_spans_by_article_numbers(article_numbers, qa_item)
            if matching_spans:
                return matching_spans
        
        # 策略2: 寬鬆條號匹配（忽略項款細節）
        simplified_articles = self._extract_simplified_article_numbers(answer + " " + query)
        if simplified_articles:
            matching_spans = self._find_spans_by_simplified_articles(simplified_articles, qa_item)
            if matching_spans:
                return matching_spans
        
        # 策略3: 關鍵詞匹配
        matching_spans = self._find_spans_by_keywords_enhanced(qa_item)
        if matching_spans:
            return matching_spans
        
        # 策略4: 文本相似度匹配
        matching_spans = self._find_spans_by_text_similarity(qa_item)
        if matching_spans:
            return matching_spans
        
        # 策略5: 基於文本重疊度的精確匹配（參考LegalBenchRAG）
        matching_spans = self._find_spans_by_text_overlap(qa_item)
        if matching_spans:
            return matching_spans
        
        # 策略6: 基於句子級別的匹配（參考LegalBenchRAG）
        matching_spans = self._find_spans_by_sentence_matching(qa_item)
        if matching_spans:
            return matching_spans
        
        # 策略7: 基於語義內容的匹配（參考LegalBenchRAG）
        matching_spans = self._find_spans_by_semantic_content(qa_item)
        if matching_spans:
            return matching_spans
        
        return []
    
    def _find_snippets_by_keywords(self, qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根據關鍵詞匹配查找snippets（寬鬆匹配策略）"""
        answer = qa_item.get("answer", "")
        query = qa_item.get("query", "")
        
        if not answer:
            return []
        
        # 提取關鍵詞
        keywords = self._extract_keywords(answer + " " + query)
        if not keywords:
            return []
        
        # 在法條spans中查找包含關鍵詞的內容
        matching_spans = []
        for span_data in self.law_spans:
            # 檢查span是否有效
            start_char = span_data.get("start_char", 0)
            end_char = span_data.get("end_char", 0)
            if end_char <= start_char or start_char == 0:  # 無效的span
                continue
            
            span_text = span_data.get("text", "")
            if not span_text:
                continue
            
            # 計算關鍵詞匹配度
            match_score = self._calculate_keyword_match(keywords, span_text)
            if match_score > 0.3:  # 匹配度閾值
                snippet = {
                    "file_path": "copyright&tradmark.json",
                    "span": [start_char, end_char]
                }
                matching_spans.append((snippet, match_score))
        
        # 按匹配度排序，返回前3個
        matching_spans.sort(key=lambda x: x[1], reverse=True)
        return [snippet for snippet, _ in matching_spans[:3]]
    
    def _extract_all_article_numbers(self, text: str) -> List[str]:
        """從文本中提取所有可能的條號"""
        import re
        
        article_numbers = []
        
        # 提取基本條號（第X條）
        basic_matches = re.findall(r'第(\d+)條', text)
        for match in basic_matches:
            article_numbers.append(f"第{match}條")
        
        # 提取帶"之"的條號（第X條之Y）
        zhi_matches = re.findall(r'第(\d+)條之(\d+)', text)
        for match in zhi_matches:
            article_numbers.append(f"第{match[0]}條之{match[1]}")
        
        return list(set(article_numbers))  # 去重
    
    def _extract_simplified_article_numbers(self, text: str) -> List[str]:
        """提取簡化的條號（忽略項款細節）"""
        import re
        
        article_numbers = []
        
        # 提取基本條號（第X條）
        basic_matches = re.findall(r'第(\d+)條', text)
        for match in basic_matches:
            article_numbers.append(f"第{match}條")
        
        return list(set(article_numbers))  # 去重
    
    def _find_spans_by_article_numbers(self, article_numbers: List[str], qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根據條號查找spans"""
        answer = qa_item.get("answer", "")
        qa_structure = self._parse_qa_structure(answer)
        
        matching_spans = []
        for span_data in self.law_spans:
            if span_data["article_name"] in article_numbers:
                start_char = span_data.get("start_char", 0)
                end_char = span_data.get("end_char", 0)
                if end_char <= start_char or start_char == 0:
                    continue
                
                if self._is_structure_matching(qa_structure, span_data):
                    span_text = span_data.get("text", "")
                    if span_text and self._is_content_matching(answer, span_text):
                        snippet = {
                            "file_path": "copyright&tradmark.json",
                            "span": [start_char, end_char]
                        }
                        matching_spans.append(snippet)
        
        return matching_spans
    
    def _find_spans_by_simplified_articles(self, simplified_articles: List[str], qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根據簡化條號查找spans（忽略項款細節）"""
        answer = qa_item.get("answer", "")
        
        matching_spans = []
        for span_data in self.law_spans:
            # 檢查是否匹配簡化條號
            for article in simplified_articles:
                if article in span_data["article_name"]:
                    start_char = span_data.get("start_char", 0)
                    end_char = span_data.get("end_char", 0)
                    if end_char <= start_char or start_char == 0:
                        continue
                    
                    # 寬鬆的內容匹配
                    span_text = span_data.get("text", "")
                    if span_text and self._is_content_matching_loose(answer, span_text):
                        snippet = {
                            "file_path": "copyright&tradmark.json",
                            "span": [start_char, end_char]
                        }
                        matching_spans.append(snippet)
                    break
        
        return matching_spans
    
    def _find_spans_by_keywords_enhanced(self, qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """增強的關鍵詞匹配"""
        answer = qa_item.get("answer", "")
        query = qa_item.get("query", "")
        
        if not answer:
            return []
        
        # 提取關鍵詞
        keywords = self._extract_keywords_enhanced(answer + " " + query)
        if not keywords:
            return []
        
        # 在法條spans中查找包含關鍵詞的內容
        matching_spans = []
        for span_data in self.law_spans:
            start_char = span_data.get("start_char", 0)
            end_char = span_data.get("end_char", 0)
            if end_char <= start_char or start_char == 0:
                continue
            
            span_text = span_data.get("text", "")
            if not span_text:
                continue
            
            # 計算關鍵詞匹配度
            match_score = self._calculate_keyword_match_enhanced(keywords, span_text)
            if match_score > 0.3:  # 提高閾值，確保匹配質量
                snippet = {
                    "file_path": "copyright&tradmark.json",
                    "span": [start_char, end_char]
                }
                matching_spans.append((snippet, match_score))
        
        # 按匹配度排序，返回前3個
        matching_spans.sort(key=lambda x: x[1], reverse=True)
        return [snippet for snippet, _ in matching_spans[:3]]
    
    def _find_spans_by_text_similarity(self, qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基於文本相似度的匹配"""
        answer = qa_item.get("answer", "")
        query = qa_item.get("query", "")
        
        if not answer:
            return []
        
        # 使用簡單的文本相似度匹配
        target_text = answer + " " + query
        
        matching_spans = []
        for span_data in self.law_spans:
            start_char = span_data.get("start_char", 0)
            end_char = span_data.get("end_char", 0)
            if end_char <= start_char or start_char == 0:
                continue
            
            span_text = span_data.get("text", "")
            if not span_text:
                continue
            
            # 計算文本相似度
            similarity = self._calculate_text_similarity(target_text, span_text)
            if similarity > 0.4:  # 提高相似度閾值，確保匹配質量
                snippet = {
                    "file_path": "copyright&tradmark.json",
                    "span": [start_char, end_char]
                }
                matching_spans.append((snippet, similarity))
        
        # 按相似度排序，返回前2個
        matching_spans.sort(key=lambda x: x[1], reverse=True)
        return [snippet for snippet, _ in matching_spans[:2]]
    
    def _extract_keywords_enhanced(self, text: str) -> List[str]:
        """增強的關鍵詞提取"""
        import re
        
        # 移除標點符號和常見停用詞
        stop_words = {"的", "是", "在", "有", "和", "與", "或", "但", "而", "則", "即", "如", "若", "因", "為", "以", "及", "等", "之", "者", "所", "得", "可", "應", "須", "本法", "規定", "條文"}
        
        # 提取中文字符和數字
        words = re.findall(r'[\u4e00-\u9fff\d]+', text)
        keywords = [word for word in words if len(word) >= 2 and word not in stop_words]
        
        # 去重並限制數量
        return list(set(keywords))[:15]  # 增加關鍵詞數量
    
    def _calculate_keyword_match_enhanced(self, keywords: List[str], text: str) -> float:
        """增強的關鍵詞匹配度計算"""
        if not keywords:
            return 0.0
        
        matches = 0
        total_weight = 0
        
        for keyword in keywords:
            # 長關鍵詞權重更高
            weight = len(keyword)
            total_weight += weight
            
            if keyword in text:
                matches += weight
        
        return matches / total_weight if total_weight > 0 else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """計算文本相似度"""
        # 簡單的字符級相似度
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _is_content_matching_loose(self, answer: str, span_text: str) -> bool:
        """寬鬆的內容匹配"""
        if not span_text or span_text.strip() == "":
            return False
        
        # 提取答案中的關鍵詞
        answer_keywords = self._extract_keywords_enhanced(answer)
        span_keywords = self._extract_keywords_enhanced(span_text)
        
        if not answer_keywords or not span_keywords:
            return False
        
        # 計算匹配度，降低閾值
        matches = 0
        for keyword in answer_keywords:
            if keyword in span_keywords:
                matches += 1
        
        # 如果超過25%的關鍵詞匹配，認為是匹配的（提高閾值確保質量）
        return matches / len(answer_keywords) > 0.25
    
    
    def _parse_qa_structure(self, answer: str) -> Dict[str, Any]:
        """解析QA答案中的條款結構"""
        import re
        
        structure = {
            "article": None,
            "item": None,
            "sub_item": None
        }
        
        # 提取條號
        article_match = re.search(r'第(\d+)條', answer)
        if article_match:
            structure["article"] = f"第{article_match.group(1)}條"
        
        # 提取項號
        item_match = re.search(r'第(\d+)項', answer)
        if item_match:
            structure["item"] = item_match.group(1)  # 只保存數字
        
        # 提取款號
        sub_item_match = re.search(r'第(\d+)款', answer)
        if sub_item_match:
            structure["sub_item"] = sub_item_match.group(1)  # 只保存數字
        
        return structure
    
    def _is_structure_matching(self, qa_structure: Dict[str, Any], span_data: Dict[str, Any]) -> bool:
        """檢查條款結構是否匹配"""
        # 如果QA沒有指定項或款，則匹配所有項和款
        if qa_structure["item"] is None:
            return True
        
        # 檢查項匹配
        span_item = span_data.get("item_name", "")
        if qa_structure["item"] != span_item:
            return False
        
        # 如果QA指定了款，檢查款匹配
        if qa_structure["sub_item"] is not None:
            span_sub_item = span_data.get("sub_item_name", "")
            # 特殊處理：空字符串表示第1款
            if span_sub_item == "" and qa_structure["sub_item"] == "1":
                return True
            # 特殊處理：第3條的項款匹配邏輯
            elif qa_structure["article"] == "第3條" and span_item == "1":
                # 對於第3條第1項，根據內容匹配具體的款
                if qa_structure["sub_item"] == "2":  # 第2款：著作人
                    return "著作人" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "3":  # 第3款：著作權
                    return "著作權" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "4":  # 第4款：公開發表
                    return "公開發表" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "5":  # 第5款：發行
                    return "發行" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "6":  # 第6款：公開口述
                    return "公開口述" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "7":  # 第7款：公開播送
                    return "公開播送" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "8":  # 第8款：公開上映
                    return "公開上映" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "9":  # 第9款：公開演出
                    return "公開演出" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "10":  # 第10款：公開傳輸
                    return "公開傳輸" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "11":  # 第11款：公開展示
                    return "公開展示" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "12":  # 第12款：改作
                    return "改作" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "13":  # 第13款：編輯
                    return "編輯" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "14":  # 第14款：出租
                    return "出租" in span_data.get("text", "")
                elif qa_structure["sub_item"] == "15":  # 第15款：重製
                    return "重製" in span_data.get("text", "")
            elif qa_structure["sub_item"] != span_sub_item:
                return False
        
        return True
    
    def _is_content_matching(self, answer: str, span_text: str) -> bool:
        """檢查答案內容是否與span文本匹配"""
        # 如果span_text為空，不匹配
        if not span_text or span_text.strip() == "":
            return False
        
        # 如果span的span範圍無效，不匹配
        # 這裡我們需要從調用者那裡獲取span信息，但為了簡化，我們先跳過這個檢查
        
        # 提取答案中的關鍵詞
        answer_keywords = self._extract_keywords(answer)
        span_keywords = self._extract_keywords(span_text)
        
        # 計算匹配度
        if not answer_keywords or not span_keywords:
            return False
            
        matches = 0
        for keyword in answer_keywords:
            if keyword in span_keywords:
                matches += 1
        
        # 進一步降低匹配閾值，如果超過20%的關鍵詞匹配，認為是匹配的
        # 或者如果關鍵詞數量很少，只要有匹配就認為是匹配的
        if len(answer_keywords) <= 3:
            return matches > 0
        else:
            return matches / len(answer_keywords) > 0.2
    
    def _extract_keywords(self, text: str) -> List[str]:
        """從文本中提取關鍵詞"""
        import re
        
        # 移除標點符號和常見停用詞
        stop_words = {"的", "是", "在", "有", "和", "與", "或", "但", "而", "則", "即", "如", "若", "因", "為", "以", "及", "等", "之", "者", "所", "得", "可", "應", "須", "本法", "規定", "條文", "如下", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"}
        
        # 提取中文字符和數字
        words = re.findall(r'[\u4e00-\u9fff\d]+', text)
        keywords = [word for word in words if len(word) >= 2 and word not in stop_words]
        
        # 去重並限制數量
        return list(set(keywords))[:15]
    
    def _find_spans_by_text_overlap(self, qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基於文本重疊度的精確匹配（參考LegalBenchRAG方法）"""
        answer = qa_item.get("answer", "")
        query = qa_item.get("query", "")
        
        if not answer:
            return []
        
        # 提取答案中的核心內容（移除條號等格式信息）
        clean_answer = self._clean_answer_text(answer)
        if not clean_answer:
            return []
        
        matching_spans = []
        for span_data in self.law_spans:
            start_char = span_data.get("start_char", 0)
            end_char = span_data.get("end_char", 0)
            if end_char <= start_char or start_char == 0:
                continue
            
            span_text = span_data.get("text", "")
            if not span_text:
                continue
            
            # 計算文本重疊度
            overlap_score = self._calculate_text_overlap(clean_answer, span_text)
            if overlap_score > 0.4:  # 重疊度閾值
                snippet = {
                    "file_path": "copyright&tradmark.json",
                    "span": [start_char, end_char]
                }
                matching_spans.append((snippet, overlap_score))
        
        # 按重疊度排序，返回前2個
        matching_spans.sort(key=lambda x: x[1], reverse=True)
        return [snippet for snippet, _ in matching_spans[:2]]
    
    def _clean_answer_text(self, answer: str) -> str:
        """清理答案文本，移除條號等格式信息，保留核心內容"""
        import re
        
        # 移除條號格式
        cleaned = re.sub(r'第\d+條[之\d+]*[第\d+項]*[第\d+款]*[：:]?', '', answer)
        
        # 移除常見的格式詞
        format_words = ['本法', '規定', '如下', '一、', '二、', '三、', '四、', '五、', '六、', '七、', '八、', '九、', '十、']
        for word in format_words:
            cleaned = cleaned.replace(word, '')
        
        # 移除多餘的標點符號和空格
        cleaned = re.sub(r'[，。；：！？\s]+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """計算兩個文本的重疊度（基於字符級別的連續匹配）"""
        if not text1 or not text2:
            return 0.0
        
        # 將文本轉換為字符列表
        chars1 = list(text1)
        chars2 = list(text2)
        
        # 計算最長公共子序列
        max_overlap = 0
        for i in range(len(chars1)):
            for j in range(len(chars2)):
                overlap = 0
                k = 0
                while (i + k < len(chars1) and 
                       j + k < len(chars2) and 
                       chars1[i + k] == chars2[j + k]):
                    overlap += 1
                    k += 1
                max_overlap = max(max_overlap, overlap)
        
        # 計算重疊度比例
        return max_overlap / max(len(chars1), len(chars2))
    
    def _find_spans_by_sentence_matching(self, qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基於句子級別的匹配（參考LegalBenchRAG方法）"""
        answer = qa_item.get("answer", "")
        query = qa_item.get("query", "")
        
        if not answer:
            return []
        
        # 將答案分解為句子
        answer_sentences = self._split_into_sentences(answer)
        if not answer_sentences:
            return []
        
        matching_spans = []
        for span_data in self.law_spans:
            start_char = span_data.get("start_char", 0)
            end_char = span_data.get("end_char", 0)
            if end_char <= start_char or start_char == 0:
                continue
            
            span_text = span_data.get("text", "")
            if not span_text:
                continue
            
            # 將span文本分解為句子
            span_sentences = self._split_into_sentences(span_text)
            
            # 計算句子級別的匹配度
            sentence_match_score = self._calculate_sentence_match(answer_sentences, span_sentences)
            if sentence_match_score > 0.3:  # 句子匹配閾值
                snippet = {
                    "file_path": "copyright&tradmark.json",
                    "span": [start_char, end_char]
                }
                matching_spans.append((snippet, sentence_match_score))
        
        # 按匹配度排序，返回前2個
        matching_spans.sort(key=lambda x: x[1], reverse=True)
        return [snippet for snippet, _ in matching_spans[:2]]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """將文本分解為句子"""
        import re
        
        # 使用句號、問號、感嘆號等作為句子分隔符
        sentences = re.split(r'[。！？；]', text)
        
        # 清理句子，移除空白和短句
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:  # 只保留長度大於5的句子
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_sentence_match(self, sentences1: List[str], sentences2: List[str]) -> float:
        """計算兩個句子列表的匹配度"""
        if not sentences1 or not sentences2:
            return 0.0
        
        total_matches = 0
        total_sentences = len(sentences1)
        
        for sentence1 in sentences1:
            best_match = 0
            for sentence2 in sentences2:
                # 計算句子相似度
                similarity = self._calculate_sentence_similarity(sentence1, sentence2)
                best_match = max(best_match, similarity)
            
            if best_match > 0.5:  # 句子相似度閾值
                total_matches += best_match
        
        return total_matches / total_sentences if total_sentences > 0 else 0.0
    
    def _calculate_sentence_similarity(self, sentence1: str, sentence2: str) -> float:
        """計算兩個句子的相似度"""
        if not sentence1 or not sentence2:
            return 0.0
        
        # 提取關鍵詞
        keywords1 = set(self._extract_keywords(sentence1))
        keywords2 = set(self._extract_keywords(sentence2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # 計算Jaccard相似度
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def _find_spans_by_semantic_content(self, qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基於語義內容的匹配（參考LegalBenchRAG方法）"""
        answer = qa_item.get("answer", "")
        query = qa_item.get("query", "")
        
        if not answer:
            return []
        
        # 提取問題和答案中的法律概念
        legal_concepts = self._extract_legal_concepts(query, answer)
        if not legal_concepts:
            return []
        
        matching_spans = []
        for span_data in self.law_spans:
            start_char = span_data.get("start_char", 0)
            end_char = span_data.get("end_char", 0)
            if end_char <= start_char or start_char == 0:
                continue
            
            span_text = span_data.get("text", "")
            if not span_text:
                continue
            
            # 計算語義匹配度
            semantic_score = self._calculate_semantic_match(legal_concepts, span_text)
            if semantic_score > 0.3:  # 語義匹配閾值
                snippet = {
                    "file_path": "copyright&tradmark.json",
                    "span": [start_char, end_char]
                }
                matching_spans.append((snippet, semantic_score))
        
        # 按匹配度排序，返回前2個
        matching_spans.sort(key=lambda x: x[1], reverse=True)
        return [snippet for snippet, _ in matching_spans[:2]]
    
    def _extract_legal_concepts(self, query: str, answer: str) -> List[str]:
        """提取法律概念"""
        import re
        
        # 法律領域的關鍵概念詞彙
        legal_terms = [
            "著作權", "著作人", "著作", "重製", "公開", "演出", "播送", "上映", "傳輸", "展示",
            "改作", "編輯", "出租", "授權", "合理使用", "侵害", "損害賠償", "刑責", "罰金",
            "製版權", "集體管理", "調解", "爭議", "告訴", "自訴", "扣押", "查扣", "沒收",
            "專屬授權", "非專屬授權", "著作財產權", "著作人格權", "公開發表", "發行",
            "公開口述", "公開播送", "公開上映", "公開演出", "公開傳輸", "公開展示",
            "視覺障礙者", "學習障礙者", "聽覺障礙者", "學校教學", "司法程序", "時事報導",
            "廣播電視台", "錄音", "錄影", "考試題目", "非營利", "海關", "外國人"
        ]
        
        text = query + " " + answer
        found_concepts = []
        
        for term in legal_terms:
            if term in text:
                found_concepts.append(term)
        
        # 也提取一些常見的法律動作詞彙
        action_terms = [
            "享有", "專有", "得", "不得", "應", "須", "禁止", "允許", "授權", "重製",
            "散布", "公開", "利用", "侵害", "請求", "排除", "防止", "賠償", "處罰"
        ]
        
        for term in action_terms:
            if term in text:
                found_concepts.append(term)
        
        return list(set(found_concepts))  # 去重
    
    def _calculate_semantic_match(self, legal_concepts: List[str], span_text: str) -> float:
        """計算語義匹配度"""
        if not legal_concepts or not span_text:
            return 0.0
        
        matches = 0
        for concept in legal_concepts:
            if concept in span_text:
                matches += 1
        
        # 計算概念匹配比例
        concept_match_ratio = matches / len(legal_concepts)
        
        # 同時考慮文本長度因素
        text_length_factor = min(len(span_text) / 200, 1.0)  # 文本越長，匹配度可能越高
        
        # 綜合評分
        final_score = concept_match_ratio * 0.7 + text_length_factor * 0.3
        
        return final_score
    
    def _extract_article_number(self, text: str) -> Optional[str]:
        """從文本中提取條號"""
        patterns = [
            r'第(\d+)條之(\d+)',  # 先匹配更具體的模式
            r'第(\d+)條第(\d+)項第(\d+)款',
            r'第(\d+)-(\d+)條',
            r'第(\d+)條'  # 最後匹配一般模式
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 1:
                    return f"第{match.group(1)}條"
                elif len(match.groups()) == 2:
                    if '之' in match.group(0):
                        return f"第{match.group(1)}條之{match.group(2)}"
                    else:
                        return f"第{match.group(1)}-{match.group(2)}條"
                elif len(match.groups()) == 3:
                    return f"第{match.group(1)}條第{match.group(2)}項第{match.group(3)}款"
        
        return None
    
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
        items_with_snippets = sum(1 for item in converted_qa_set if item.get("snippets"))
        items_with_valid_spans = 0
        items_with_file_path = 0
        
        for item in converted_qa_set:
            if item.get("snippets"):
                for snippet in item["snippets"]:
                    if snippet.get("file_path"):
                        items_with_file_path += 1
                    if "span" in snippet and isinstance(snippet["span"], list) and len(snippet["span"]) == 2:
                        start, end = snippet["span"]
                        if end > start:  # 有效的span範圍
                            items_with_valid_spans += 1
                        break
        
        return {
            "total_items": total_items,
            "items_with_snippets": items_with_snippets,
            "items_with_valid_spans": items_with_valid_spans,
            "items_with_file_path": items_with_file_path,
            "snippet_coverage": items_with_snippets / total_items if total_items > 0 else 0,
            "span_coverage": items_with_valid_spans / total_items if total_items > 0 else 0,
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
