#!/usr/bin/env python3
"""
QA Set映射工具
將原始QA set映射到法條JSON文件，生成包含snippets的新格式
"""

import json
import re
import argparse
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
    
    # 無映射模式下不再需要映射/片段輸出，保留索引與正規化工具供子類使用
    pass

class NoMappingGoldBuilder(QAMapper):
    """無映射模式：僅從 QA 中規則抽取 gold（法名與條號層級）。"""

    def _to_int_cn_num_local(self, s: str) -> Optional[int]:
        if s is None:
            return None
        s = str(s)
        import unicodedata
        s = unicodedata.normalize('NFKC', s)
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return None
        mapping = {'零':0,'〇':0,'一':1,'二':2,'兩':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'百':100,'千':1000}
        total, section, num = 0, 0, 0
        found = False
        for ch in s:
            if ch in mapping:
                v = mapping[ch]
                if v < 10:
                    num = v
                    found = True
                else:
                    if num == 0:
                        num = 1
                    section += num * v
                    num = 0
                    found = True
        total += section + num
        return total if found else None

    def _extract_law_name_from_text(self, text: str) -> str:
        if not text:
            return ''
        if '著作權法' in text:
            return '著作權法'
        if '商標法' in text:
            return '商標法'
        if '專利法' in text:
            return '專利法'
        return ''

    def _extract_hierarchy(self, text: str) -> Dict[str, Optional[int]]:
        info = {
            'article_number': None,
            'article_suffix': None,
            'item_number': None,
            'clause_type': None,
            'clause_number': None,
            'clause_sub_number': None,
        }
        if not text:
            return info
        import unicodedata
        t = unicodedata.normalize('NFKC', text)
        m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)條之([0-9一二兩三四五六七八九十百千〇零]+)", t)
        if m:
            info['article_number'] = self._to_int_cn_num_local(m.group(1))
            info['article_suffix'] = self._to_int_cn_num_local(m.group(2))
        else:
            m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)-([0-9一二兩三四五六七八九十百千〇零]+)條", t)
            if m:
                info['article_number'] = self._to_int_cn_num_local(m.group(1))
                info['article_suffix'] = self._to_int_cn_num_local(m.group(2))
            else:
                m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)條", t)
                if m:
                    info['article_number'] = self._to_int_cn_num_local(m.group(1))
        m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)項", t)
        if m:
            info['item_number'] = self._to_int_cn_num_local(m.group(1))
        m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)款", t)
        if m:
            info['clause_type'] = '款'
            info['clause_number'] = self._to_int_cn_num_local(m.group(1))
        m = re.search(r"[（(]([0-9一二兩三四五六七八九十〇零]+)[)）]", t)
        if m:
            info['clause_type'] = '目'
            info['clause_number'] = self._to_int_cn_num_local(m.group(1))
        m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)目之([0-9一二兩三四五六七八九十百千〇零]+)", t)
        if m:
            info['clause_type'] = '目之'
            info['clause_number'] = self._to_int_cn_num_local(m.group(1))
            info['clause_sub_number'] = self._to_int_cn_num_local(m.group(2))
        return info

    def build_gold_for_item(self, qa_item: Dict) -> Dict:
        query = qa_item.get('query', '')
        answer = ''
        if 'snippets' in qa_item and qa_item['snippets']:
            answer = qa_item['snippets'][0].get('answer', '')
        elif 'answer' in qa_item:
            answer = qa_item['answer']

        law = self._extract_law_name_from_text(query) or self._extract_law_name_from_text(answer)
        ans_h = self._extract_hierarchy(answer)
        qry_h = self._extract_hierarchy(query)

        keys = set(list(ans_h.keys()) + list(qry_h.keys()))
        merged = {k: (ans_h.get(k) if ans_h.get(k) is not None else qry_h.get(k)) for k in keys}

        gold = {
            'law': law,
            'article_number': merged.get('article_number'),
            'article_suffix': merged.get('article_suffix'),
            'item_number': merged.get('item_number'),
            'clause_type': merged.get('clause_type'),
            'clause_number': merged.get('clause_number'),
            'clause_sub_number': merged.get('clause_sub_number'),
        }

        result = {
            'query': query,
            'label': qa_item.get('label'),
            'gold': gold,
            'gold_source': 'answer_first_then_query'
        }
        # 可選：輸出 answer 以便人工抽查/誤差分析
        if getattr(self, 'include_answer', False) and answer:
            result['answer'] = answer
        return result

    def build_all(self) -> List[Dict]:
        results = []
        for qa_item in self.qa_data:
            results.append(self.build_gold_for_item(qa_item))
        return results

    def save_gold(self, output_path: str, results: List[Dict]):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    """主函數（無映射模式）"""
    parser = argparse.ArgumentParser(description='QA 無映射 gold 生成工具')
    parser.add_argument('--qa', default='/Users/charliebear/Desktop/code/RAG/QA/qa_set.json', help='QA set 檔案路徑')
    parser.add_argument('--corpus', default='/Users/charliebear/Desktop/code/RAG/corpus/copyright&tradmark.json', help='法律 JSON 檔案路徑（可選，用於正規化）')
    parser.add_argument('--out', default='/Users/charliebear/Desktop/code/RAG/QA/qa_gold.json', help='輸出檔案路徑')
    parser.add_argument('--include-answer', action='store_true', help='輸出時保留 answer 欄位（供人工抽查/分析）')
    parser.add_argument('--normalize-corpus', action='store_true', help='將法律JSON正規化並輸出（補齊評測最小必要欄位）')
    parser.add_argument('--corpus-out', default='/Users/charliebear/Desktop/code/RAG/corpus/corpus_normalized.json', help='正規化後法律JSON輸出檔案路徑')
    args = parser.parse_args()

    qa_file = args.qa
    corpus_file = args.corpus
    output_file = args.out

    # 若指定正規化，先執行法律 JSON 正規化後退出
    if args.normalize_corpus:
        def to_int_cn_num_local(s: str) -> Optional[int]:
            if s is None:
                return None
            s = str(s)
            import unicodedata
            s = unicodedata.normalize('NFKC', s)
            if s.isdigit():
                try:
                    return int(s)
                except Exception:
                    return None
            mapping = {'零':0,'〇':0,'一':1,'二':2,'兩':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'百':100,'千':1000}
            total, section, num = 0, 0, 0
            found = False
            for ch in s:
                if ch in mapping:
                    v = mapping[ch]
                    if v < 10:
                        num = v
                        found = True
                    else:
                        if num == 0:
                            num = 1
                        section += num * v
                        num = 0
                        found = True
            total += section + num
            return total if found else None

        def parse_article_label(label: str) -> Tuple[Optional[int], Optional[int]]:
            if not label:
                return (None, None)
            import unicodedata
            t = unicodedata.normalize('NFKC', label)
            m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)條之([0-9一二兩三四五六七八九十百千〇零]+)", t)
            if m:
                return (to_int_cn_num_local(m.group(1)), to_int_cn_num_local(m.group(2)))
            m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)-([0-9一二兩三四五六七八九十百千〇零]+)條", t)
            if m:
                return (to_int_cn_num_local(m.group(1)), to_int_cn_num_local(m.group(2)))
            m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)條", t)
            if m:
                return (to_int_cn_num_local(m.group(1)), None)
            return (None, None)

        def normalize_corpus(corpus: Dict) -> Dict:
            for law in corpus.get('laws', []):
                law_name = law.get('law_name') or ''
                clean_law = law_name.replace('法規名稱：', '')
                for chapter in law.get('chapters', []) or []:
                    for section in (chapter.get('sections', []) or []):
                        for article in (section.get('articles', []) or []):
                            a_label = article.get('article') or ''
                            a_num, a_suf = parse_article_label(a_label)
                            md = article.get('metadata') or {}
                            md['category'] = clean_law
                            md['article_label'] = a_label
                            md['article_number'] = a_num
                            md['article_suffix'] = a_suf
                            article['metadata'] = md
                            for item in (article.get('items', []) or []):
                                md_i = item.get('metadata') or {}
                                md_i['category'] = clean_law
                                md_i['article_label'] = a_label
                                md_i['article_number'] = a_num
                                md_i['article_suffix'] = a_suf
                                i_str = str(item.get('item') or '').strip()
                                md_i['item_number'] = to_int_cn_num_local(i_str)
                                item['metadata'] = md_i
                                for sub in (item.get('sub_items', []) or []):
                                    md_s = sub.get('metadata') or {}
                                    md_s['category'] = clean_law
                                    md_s['article_label'] = a_label
                                    md_s['article_number'] = a_num
                                    md_s['article_suffix'] = a_suf
                                    s_str = str(sub.get('sub_item') or '').strip()
                                    md_s['clause_type'] = '目'
                                    md_s['clause_number'] = to_int_cn_num_local(s_str)
                                    md_s['clause_sub_number'] = None
                                    sub['metadata'] = md_s
            return corpus

        print("讀取法律JSON...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_raw = json.load(f)
        print("正規化法律JSON欄位（最小必要鍵）...")
        normalized = normalize_corpus(corpus_raw)
        print(f"保存至 {args.corpus_out}")
        with open(args.corpus_out, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        print("完成法律JSON正規化。")
        return

    builder = NoMappingGoldBuilder(qa_file, corpus_file)
    builder.include_answer = args.include_answer
    print("載入數據...")
    builder.load_data()

    print("生成 gold（無映射模式）...")
    results = builder.build_all()

    print(f"保存結果到 {output_file}")
    builder.save_gold(output_file, results)

    total_items = len(results)
    yes_items = len([r for r in results if r.get('label') == 'Yes'])
    with_gold = len([r for r in results if r.get('gold', {}).get('article_number') is not None])
    print(f"\n完成! 總項目數: {total_items}")
    print(f"Yes項目數: {yes_items}")
    print(f"抽取出條號的項目數: {with_gold}")

if __name__ == "__main__":
    main()
