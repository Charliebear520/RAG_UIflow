#!/usr/bin/env python3
"""
QA Set映射工具
將原始QA set映射到法條JSON文件，生成包含snippets的新格式
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

class QAMapper:
    def __init__(self, qa_file_path: str, corpus_file_path: str):
        """初始化QA映射器
        
        Args:
            qa_file_path: 原始QA set文件路徑
            corpus_file_path: 法條JSON文件路徑
        """
        self.qa_file_path = qa_file_path
        self.corpus_file_path = corpus_file_path
        self.qa_data = []
        self.corpus_data = {}
        self.file_path = "copyright&tradmark.json"
        
    def load_data(self):
        """載入QA數據和法條數據"""
        # 載入QA set
        with open(self.qa_file_path, 'r', encoding='utf-8') as f:
            self.qa_data = json.load(f)
            
        # 載入法條數據
        with open(self.corpus_file_path, 'r', encoding='utf-8') as f:
            corpus_raw = json.load(f)
            
        # 建立法條索引
        self._build_corpus_index(corpus_raw)
        
    def _build_corpus_index(self, corpus_raw: Dict):
        """建立法條內容索引以便快速查找"""
        self.corpus_index = {}
        
        for law in corpus_raw.get('laws', []):
            if law['law_name'] == '法規名稱：著作權法':
                for chapter in law.get('chapters', []):
                    for section in chapter.get('sections', []):
                        for article in section.get('articles', []):
                            article_id = article['metadata']['id']
                            article_content = article['content']
                            
                            # 建立文章索引
                            self.corpus_index[article_id] = {
                                'content': article_content,
                                'metadata': article['metadata'],
                                'article_name': article['article']
                            }
                            
                            # 處理items
                            for item in article.get('items', []):
                                item_id = item['metadata']['id']
                                item_content = item['content']
                                
                                self.corpus_index[item_id] = {
                                    'content': item_content,
                                    'metadata': item['metadata'],
                                    'article_name': article['article'],
                                    'item_name': item['item']
                                }
    
    def _extract_article_number(self, answer: str) -> Optional[str]:
        """從答案中提取條號"""
        # 匹配各種條號格式
        patterns = [
            r'第(\d+)條',
            r'第(\d+)條之(\d+)',
            r'第(\d+)-(\d+)條',
            r'第(\d+)條第(\d+)項第(\d+)款'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer)
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
    
    def _find_matching_article(self, answer: str) -> Optional[Dict]:
        """根據答案內容找到對應的法條"""
        article_number = self._extract_article_number(answer)
        
        if not article_number:
            return None
            
        # 在所有索引中查找匹配的條號
        for article_id, article_data in self.corpus_index.items():
            if article_data.get('article_name') == article_number:
                return article_data
                
        return None
    
    def _get_span_from_metadata(self, article_data: Dict) -> List[int]:
        """從metadata中提取span信息"""
        metadata = article_data.get('metadata', {})
        spans = metadata.get('spans', [])
        
        if isinstance(spans, list) and spans:
            # 處理spans數組格式
            span = spans[0]
            start_char = span.get('start_char', 0)
            end_char = span.get('end_char', 0)
            # 如果span有效，返回它
            if end_char > start_char:
                return [start_char, end_char]
        elif isinstance(spans, dict):
            # 處理spans對象格式
            start = spans.get('start', 0)
            end = spans.get('end', 0)
            # 如果span有效，返回它
            if end > start:
                return [start, end]
        
        return None
    
    def _calculate_span_from_content(self, article_data: Dict, answer: str) -> Optional[List[int]]:
        """根據內容計算span"""
        content = article_data.get('content', '')
        if not content:
            return None
            
        # 嘗試在內容中查找答案的關鍵詞
        # 提取答案中的關鍵條號
        article_number = self._extract_article_number(answer)
        if article_number and article_number in content:
            # 找到條號在內容中的位置
            start_pos = content.find(article_number)
            if start_pos != -1:
                # 計算一個合理的結束位置
                end_pos = min(start_pos + len(answer) * 2, len(content))
                return [start_pos, end_pos]
        
        # 如果找不到具體位置，返回None表示無法確定span
        return None
    
    def map_qa_item(self, qa_item: Dict) -> Dict:
        """映射單個QA項目"""
        result = {
            'query': qa_item['query'],
            'label': qa_item['label'],
            'snippets': []
        }
        
        # 只有標籤為Yes的項目才需要添加snippets
        if qa_item['label'] == 'Yes' and 'answer' in qa_item:
            answer = qa_item['answer']
            
            # 查找對應的法條
            article_data = self._find_matching_article(answer)
            
            if article_data:
                # 獲取span信息
                span = self._get_span_from_metadata(article_data)
                
                # 如果metadata中沒有有效的span，嘗試計算
                if span is None:
                    span = self._calculate_span_from_content(article_data, answer)
                
                # 只有當有有效span時才添加snippets
                if span is not None:
                    result['snippets'] = [{
                        'file_path': self.file_path,
                        'span': span
                    }]
                
                result['answer'] = answer
            else:
                # 如果找不到對應的法條，保持原始格式
                result['answer'] = answer
        else:
            # 對於標籤為No的項目，不需要snippets
            pass
            
        return result
    
    def map_all(self) -> List[Dict]:
        """映射所有QA項目"""
        mapped_results = []
        
        for qa_item in self.qa_data:
            mapped_item = self.map_qa_item(qa_item)
            mapped_results.append(mapped_item)
            
        return mapped_results
    
    def save_mapped_results(self, output_path: str, results: List[Dict]):
        """保存映射結果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    """主函數"""
    # 文件路徑
    qa_file = '/Users/charliebear/Desktop/code/RAG/QA/qa_set.json'
    corpus_file = '/Users/charliebear/Desktop/code/RAG/corpus/copyright&tradmark.json'
    output_file = '/Users/charliebear/Desktop/code/RAG/QA/qa_set_mapped.json'
    
    # 創建映射器
    mapper = QAMapper(qa_file, corpus_file)
    
    # 載入數據
    print("載入數據...")
    mapper.load_data()
    
    # 執行映射
    print("執行映射...")
    results = mapper.map_all()
    
    # 保存結果
    print(f"保存結果到 {output_file}")
    mapper.save_mapped_results(output_file, results)
    
    # 統計信息
    total_items = len(results)
    yes_items = len([r for r in results if r['label'] == 'Yes'])
    no_items = len([r for r in results if r['label'] == 'No'])
    mapped_items = len([r for r in results if r['label'] == 'Yes' and r['snippets']])
    
    print(f"\n映射完成!")
    print(f"總項目數: {total_items}")
    print(f"Yes項目數: {yes_items}")
    print(f"No項目數: {no_items}")
    print(f"成功映射項目數: {mapped_items}")
    print(f"映射成功率: {mapped_items/yes_items*100:.1f}%" if yes_items > 0 else "映射成功率: 0%")

if __name__ == "__main__":
    main()
