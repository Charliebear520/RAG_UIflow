"""
文檔分割模組
"""

import re
from typing import List, Dict, Any, Tuple, Optional

# 嘗試導入langchain，如果不可用則使用自定義實現
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    RecursiveCharacterTextSplitter = None
    LANGCHAIN_AVAILABLE = False


class ChunkingStrategy:
    """分割策略基類"""
    
    def chunk(self, text: str, **kwargs) -> List[str]:
        """分割文本"""
        raise NotImplementedError


class FixedSizeChunking(ChunkingStrategy):
    """固定大小分割策略"""
    
    def chunk(self, text: str, chunk_size: int = 500, overlap_ratio: float = 0.1, **kwargs) -> List[str]:
        """固定大小分割"""
        overlap = int(chunk_size * overlap_ratio)
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
            
        return chunks


class HierarchicalChunking(ChunkingStrategy):
    """層次化分割策略 - 基於法律文檔結構"""
    
    def __init__(self):
        """初始化正則表達式模式"""
        # 章節標記
        self.chapter_patterns = [
            r'^第\s*([一二三四五六七八九十百千0-9]+)\s*章[\u3000\s]*(.*)$',
            r'^第\s*([0-9]+)\s*章[\u3000\s]*(.*)$',
            # 無編號章節（如：總則、附則、罰則、附錄）
            r'^(總則|附則|罰則|附錄)[\u3000\s]*(.*)$'
        ]
        
        # 節標記
        self.section_patterns = [
            r'^第\s*([一二三四五六七八九十百千0-9]+)\s*節[\u3000\s]*(.*)$',
            r'^第\s*([0-9]+)\s*節[\u3000\s]*(.*)$'
        ]
        
        # 條文標記
        self.article_patterns = [
            r'^第\s*([一二三四五六七八九十百千0-9]+(?:之[一二三四五六七八九十0-9]+)?)\s*條[\u3000\s]*(.*)$',
            r'^第\s*([0-9]+(?:之[0-9]+)?)\s*條[\u3000\s]*(.*)$'
        ]
        
        # 項目標記
        self.item_patterns = [
            r'^[（(]([0-9０-９一二三四五六七八九十]+)[）)]\s*(.*)$',  # （一）、(1)
            r'^([一二三四五六七八九十]+)[、．\.）)]\s*(.*)$',  # 一、二、
            r'^([0-9０-９]+)[、．\.）)]\s*(.*)$',  # 1.2.3.
            r'^([•‧·–-])\s*(.*)$',  # • / ‧ / · / – / -
            r'^([0-9０-９]+)\s+(.*)$',  # 1 2 3
            r'^([一二三四五六七八九十]+)\s+(.*)$'  # 一 二 三
        ]
        
        # 編譯正則表達式
        self.chapter_re = [re.compile(pattern, re.MULTILINE) for pattern in self.chapter_patterns]
        self.section_re = [re.compile(pattern, re.MULTILINE) for pattern in self.section_patterns]
        self.article_re = [re.compile(pattern, re.MULTILINE) for pattern in self.article_patterns]
        self.item_re = [re.compile(pattern, re.MULTILINE) for pattern in self.item_patterns]
    
    def _normalize_digits(self, text: str) -> str:
        """標準化數字格式"""
        # 將全形數字轉換為半形
        digit_map = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9'
        }
        for full, half in digit_map.items():
            text = text.replace(full, half)
        return text
    
    def _detect_structure_level(self, line: str) -> Tuple[str, str | None, str]:
        """檢測行的結構層級"""
        line = self._normalize_digits(line.strip())
        
        # 檢查章
        for regex in self.chapter_re:
            match = regex.match(line)
            if match:
                return 'chapter', match.group(1), match.group(2) if len(match.groups()) > 1 else ""
        
        # 檢查節
        for regex in self.section_re:
            match = regex.match(line)
            if match:
                return 'section', match.group(1), match.group(2) if len(match.groups()) > 1 else ""
        
        # 檢查條
        for regex in self.article_re:
            match = regex.match(line)
            if match:
                return 'article', match.group(1), match.group(2) if len(match.groups()) > 1 else ""
        
        # 檢查項
        for regex in self.item_re:
            match = regex.match(line)
            if match:
                return 'item', match.group(1), match.group(2) if len(match.groups()) > 1 else ""
        
        return 'content', None, line
    
    def _create_chunk_metadata(self, level: str, number: str, content: str, 
                              chapter: str = "", section: str = "", article: str = "") -> dict:
        """創建chunk的元數據"""
        return {
            "level": level,
            "number": number,
            "content": content,
            "chapter": chapter,
            "section": section,
            "article": article,
            "length": len(content)
        }
    
    def chunk(
        self,
        text: str,
        max_chunk_size: int = 1000,
        overlap_ratio: float = 0.1,
        min_chunk_size: int = 200,
        force_new_chunk_on_article: bool = True,
        overlap_mode: str = "lines",  # "chars" | "lines"
        overlap_lines: int = 2,
        **kwargs,
    ) -> List[str]:
        """層次化分割

        Args:
            text: 原始文本
            max_chunk_size: 每個chunk最大長度（字符）
            overlap_ratio: 重疊比例（僅在 overlap_mode="chars" 時生效）
            min_chunk_size: 最小chunk長度，小於此值時嘗試與前一chunk合併
            force_new_chunk_on_article: 遇到新條文是否強制切分
            overlap_mode: 重疊策略，按字符或按行
            overlap_lines: overlap_mode="lines" 時，與前一chunk重疊的末尾行數
        """
        if not text.strip():
            return []
        
        lines = text.split('\n')
        chunks = []
        current_chunk_lines: List[str] = []
        current_metadata = {
            "chapter": "",
            "section": "",
            "article": "",
            "items": []
        }
        
        overlap_size = int(max_chunk_size * overlap_ratio)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 檢測結構層級
            level, number, content = self._detect_structure_level(line)
            
            # 更新當前元數據
            if level == 'chapter':
                # 支援無編號章節（如 總則/附則）
                if number and re.match(r'^[0-9一二三四五六七八九十百千]+$', number):
                    current_metadata["chapter"] = f"第{number}章 {content}".strip()
                else:
                    current_metadata["chapter"] = f"{number or ''} {content}".strip()
                current_metadata["section"] = ""
                current_metadata["article"] = ""
                current_metadata["items"] = []
            elif level == 'section':
                current_metadata["section"] = f"第{number}節 {content}"
                current_metadata["article"] = ""
                current_metadata["items"] = []
            elif level == 'article':
                current_metadata["article"] = f"第{number}條 {content}"
                current_metadata["items"] = []
            elif level == 'item':
                current_metadata["items"].append(f"{number}. {content}")
            
            # 決定是否開始新的chunk
            should_start_new_chunk = False
            
            # 如果是新的條文，強制開始新chunk
            if level == 'article' and current_chunk_lines:
                should_start_new_chunk = force_new_chunk_on_article
            # 如果超過最大大小，開始新chunk
            elif sum(len(l) + 1 for l in current_chunk_lines) + len(line) > max_chunk_size and current_chunk_lines:
                should_start_new_chunk = True
            
            if should_start_new_chunk:
                # 保存當前chunk；若太小且由條文邊界觸發，則嘗試與新內容合併避免碎片
                current_text = "\n".join(current_chunk_lines).strip()
                if current_text:
                    if len(current_text) < min_chunk_size and level == 'article':
                        current_chunk_lines.append(line)
                        continue
                    chunks.append(current_text)

                # 開始新chunk，添加重疊內容
                if overlap_mode == "lines" and current_chunk_lines:
                    tail = current_chunk_lines[-overlap_lines:] if overlap_lines > 0 else []
                    current_chunk_lines = [*tail, line]
                else:
                    if overlap_ratio > 0 and len(current_text) > overlap_size:
                        overlap_content = current_text[-overlap_size:].strip()
                        current_chunk_lines = ([overlap_content] if overlap_content else []) + [line]
                    else:
                        current_chunk_lines = [line]
            else:
                # 添加到當前chunk
                current_chunk_lines.append(line)
        
        # 添加最後一個chunk
        if current_chunk_lines:
            final_text = "\n".join(current_chunk_lines).strip()
            if final_text:
                if len(final_text) < min_chunk_size and chunks:
                    prev = chunks.pop()
                    chunks.append(prev + "\n" + final_text)
                else:
                    chunks.append(final_text)
        
        return chunks
    
    def chunk_with_span(self, text: str, max_chunk_size: int = 1000, overlap_ratio: float = 0.1, **kwargs) -> List[Dict[str, Any]]:
        """
        帶span信息的層次分割
        
        Returns:
            List[Dict]: 包含content、span、metadata的chunk列表
        """
        chunks = self.chunk(text, max_chunk_size, overlap_ratio, **kwargs)
        
        result = []
        current_pos = 0
        
        for i, chunk in enumerate(chunks):
            # 在原文中查找chunk位置
            start_pos = text.find(chunk, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(chunk)
            
            # 提取metadata
            level, number, content = self._detect_structure_level(chunk.split('\n')[0])
            
            result.append({
                "content": chunk,
                "span": {"start": start_pos, "end": end_pos},
                "chunk_id": f"hierarchical_chunk_{i}",
                "metadata": {
                    "level": level,
                    "number": number,
                    "content_preview": content,
                    "length": len(chunk),
                    "strategy": "hierarchical"
                }
            })
            
            current_pos = start_pos
        
        return result


class RCTSHierarchicalChunking(ChunkingStrategy):
    """基於RCTS的層次分割策略 - 結合層次結構識別和RCTS智能分割"""
    
    def __init__(self):
        """初始化RCTS和層次識別模式"""
        # 層次標記模式（繼承自HierarchicalChunking）
        self.chapter_patterns = [
            r'^第\s*([一二三四五六七八九十百千0-9]+)\s*章[\u3000\s]*(.*)$',
            r'^第\s*([0-9]+)\s*章[\u3000\s]*(.*)$',
            r'^(總則|附則|罰則|附錄)[\u3000\s]*(.*)$'
        ]
        
        self.article_patterns = [
            r'^第\s*([一二三四五六七八九十百千0-9]+(?:之[一二三四五六七八九十0-9]+)?)\s*條[\u3000\s]*(.*)$',
            r'^第\s*([0-9]+(?:之[0-9]+)?)\s*條[\u3000\s]*(.*)$'
        ]
        
        self.item_patterns = [
            r'^[（(]([0-9０-９一二三四五六七八九十]+)[）)]\s*(.*)$',
            r'^([一二三四五六七八九十]+)[、．\.）)]\s*(.*)$',
            r'^([0-9０-９]+)[、．\.）)]\s*(.*)$'
        ]
        
        # 編譯正則表達式
        self.chapter_re = [re.compile(pattern, re.MULTILINE) for pattern in self.chapter_patterns]
        self.article_re = [re.compile(pattern, re.MULTILINE) for pattern in self.article_patterns]
        self.item_re = [re.compile(pattern, re.MULTILINE) for pattern in self.item_patterns]
        
        # RCTS分割器 - 針對中文法律文檔優化
        if LANGCHAIN_AVAILABLE and RecursiveCharacterTextSplitter:
            self.rcts_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=[
                    "\n\n",  # 段落分隔
                    "\n",    # 行分隔
                    "。",    # 句號
                    "；",    # 分號
                    "，",    # 逗號
                    "、",    # 頓號
                    " ",     # 空格
                    ""       # 字符級別
                ],
                is_separator_regex=False
            )
            self.use_langchain = True
        else:
            # 使用自定義的RecursiveCharacterChunking作為替代
            self.rcts_splitter = RecursiveCharacterChunking()
            self.use_langchain = False
    
    def chunk(self, text: str, max_chunk_size: int = 1000, overlap_ratio: float = 0.1, 
              preserve_structure: bool = True, **kwargs) -> List[str]:
        """
        使用RCTS進行層次分割
        
        Args:
            text: 原始文本
            max_chunk_size: 最大chunk大小
            overlap_ratio: 重疊比例
            preserve_structure: 是否保持層次結構
        """
        if not preserve_structure:
            # 如果不需要保持結構，直接使用RCTS
            if self.use_langchain:
                self.rcts_splitter.chunk_size = max_chunk_size
                self.rcts_splitter.chunk_overlap = int(max_chunk_size * overlap_ratio)
                return self.rcts_splitter.split_text(text)
            else:
                # 使用自定義實現
                return self.rcts_splitter.chunk(text, max_chunk_size, overlap_ratio)
        
        # 保持層次結構的RCTS分割
        return self._hierarchical_rcts_split(text, max_chunk_size, overlap_ratio)
    
    def _hierarchical_rcts_split(self, text: str, max_chunk_size: int, overlap_ratio: float) -> List[str]:
        """結合層次結構的RCTS分割"""
        lines = text.split('\n')
        chunks = []
        current_chunk_lines = []
        current_structure = {"chapter": "", "article": "", "items": []}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_chunk_lines:
                    current_chunk_lines.append("")
                continue
            
            # 檢測結構層級
            level, number, content = self._detect_structure_level(line)
            
            # 更新當前結構
            if level == "chapter":
                current_structure["chapter"] = f"第{number}章 {content}"
                current_structure["article"] = ""
                current_structure["items"] = []
            elif level == "article":
                current_structure["article"] = f"第{number}條 {content}"
                current_structure["items"] = []
            elif level == "item":
                current_structure["items"].append(f"{number}. {content}")
            
            # 檢查是否需要分割
            current_text = "\n".join(current_chunk_lines)
            should_split = False
            
            # 在條文邊界強制分割
            if level == "article" and current_chunk_lines:
                should_split = True
            
            # 檢查大小限制
            elif len(current_text) + len(line) > max_chunk_size and current_chunk_lines:
                should_split = True
            
            if should_split:
                # 使用RCTS進一步分割過大的chunk
                if len(current_text) > max_chunk_size:
                    sub_chunks = self._rcts_split_with_structure(
                        current_text, max_chunk_size, overlap_ratio, current_structure
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(current_text)
                
                # 開始新chunk，添加重疊內容
                overlap_lines = self._get_overlap_lines(current_chunk_lines, overlap_ratio, max_chunk_size)
                current_chunk_lines = overlap_lines + [line]
            else:
                current_chunk_lines.append(line)
        
        # 處理最後一個chunk
        if current_chunk_lines:
            final_text = "\n".join(current_chunk_lines)
            if len(final_text) > max_chunk_size:
                sub_chunks = self._rcts_split_with_structure(
                    final_text, max_chunk_size, overlap_ratio, current_structure
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(final_text)
        
        return chunks
    
    def _detect_structure_level(self, line: str) -> Tuple[str, str, str]:
        """檢測行的結構層級"""
        line = self._normalize_digits(line)
        
        # 檢查章
        for regex in self.chapter_re:
            match = regex.match(line)
            if match:
                return 'chapter', match.group(1), match.group(2) if len(match.groups()) > 1 else ""
        
        # 檢查條
        for regex in self.article_re:
            match = regex.match(line)
            if match:
                return 'article', match.group(1), match.group(2) if len(match.groups()) > 1 else ""
        
        # 檢查項
        for regex in self.item_re:
            match = regex.match(line)
            if match:
                return 'item', match.group(1), match.group(2) if len(match.groups()) > 1 else ""
        
        return 'content', None, line
    
    def _normalize_digits(self, text: str) -> str:
        """標準化數字格式"""
        digit_map = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9'
        }
        for full, half in digit_map.items():
            text = text.replace(full, half)
        return text
    
    def _rcts_split_with_structure(self, text: str, max_size: int, overlap_ratio: float, structure: dict) -> List[str]:
        """使用RCTS分割並保持結構信息"""
        # 構建結構前綴
        structure_prefix = self._build_structure_prefix(structure)
        
        # 使用RCTS分割
        if self.use_langchain:
            self.rcts_splitter.chunk_size = max_size
            self.rcts_splitter.chunk_overlap = int(max_size * overlap_ratio)
            sub_chunks = self.rcts_splitter.split_text(text)
        else:
            # 使用自定義實現
            sub_chunks = self.rcts_splitter.chunk(text, max_size, overlap_ratio)
        
        # 為每個sub-chunk添加結構前綴
        result = []
        for i, chunk in enumerate(sub_chunks):
            if i == 0:
                # 第一個chunk包含完整結構信息
                result.append(structure_prefix + chunk)
            else:
                # 後續chunk只包含基本結構信息
                basic_prefix = f"【{structure.get('chapter', '')}】\n" if structure.get('chapter') else ""
                result.append(basic_prefix + chunk)
        
        return result
    
    def _build_structure_prefix(self, structure: dict) -> str:
        """構建結構前綴"""
        prefix_parts = []
        
        if structure.get('chapter'):
            prefix_parts.append(f"【{structure['chapter']}】")
        
        if structure.get('article'):
            prefix_parts.append(structure['article'])
        
        if structure.get('items'):
            prefix_parts.extend(structure['items'][:3])  # 只顯示前3個項目
        
        return "\n".join(prefix_parts) + "\n" if prefix_parts else ""
    
    def _get_overlap_lines(self, lines: List[str], overlap_ratio: float, max_size: int) -> List[str]:
        """獲取重疊行"""
        if not lines:
            return []
        
        overlap_size = int(max_size * overlap_ratio)
        overlap_lines = []
        current_size = 0
        
        # 從後往前添加行，直到達到重疊大小
        for line in reversed(lines):
            if current_size + len(line) > overlap_size:
                break
            overlap_lines.insert(0, line)
            current_size += len(line) + 1  # +1 for newline
        
        return overlap_lines


class StructuredHierarchicalChunking(ChunkingStrategy):
    """結構化層次分割策略 - 基於JSON結構數據"""
    
    def chunk(self, text: str, json_data: Dict[str, Any] | None = None, max_chunk_size: int = 1000, 
              overlap_ratio: float = 0.1, chunk_by: str = "article", **kwargs) -> List[str]:
        """
        基於JSON結構進行層次分割
        
        Args:
            text: 原始文本
            json_data: 結構化JSON數據
            max_chunk_size: 最大chunk大小
            overlap_ratio: 重疊比例
            chunk_by: 分割單位 ("article", "section", "chapter")
        """
        if json_data and "chapters" in json_data:
            return self._chunk_by_json_structure(json_data, max_chunk_size, overlap_ratio, chunk_by)
        else:
            # 如果沒有JSON結構，回退到普通層次分割
            hierarchical_chunker = HierarchicalChunking()
            return hierarchical_chunker.chunk(text, max_chunk_size, overlap_ratio)
    
    def _chunk_by_json_structure(self, json_data: dict, max_chunk_size: int, 
                                overlap_ratio: float, chunk_by: str) -> List[str]:
        """根據JSON結構進行分割"""
        chunks = []
        law_name = json_data.get("law_name", "未命名法規")
        
        for chapter_data in json_data.get("chapters", []):
            chapter_title = chapter_data.get("chapter", "")
            
            if chunk_by == "chapter":
                # 按章分割
                chapter_chunk = self._build_chapter_chunk(chapter_data, law_name)
                if len(chapter_chunk) > max_chunk_size:
                    # 如果章太大，進一步分割
                    sub_chunks = self._split_large_chunk(chapter_chunk, max_chunk_size, overlap_ratio)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chapter_chunk)
            
            elif chunk_by == "section":
                # 按節分割
                for section_data in chapter_data.get("sections", []):
                    section_chunk = self._build_section_chunk(section_data, law_name, chapter_title)
                    if len(section_chunk) > max_chunk_size:
                        sub_chunks = self._split_large_chunk(section_chunk, max_chunk_size, overlap_ratio)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(section_chunk)
            
            elif chunk_by == "article":
                # 按條文分割
                for section_data in chapter_data.get("sections", []):
                    section_title = section_data.get("section", "")
                    for article_data in section_data.get("articles", []):
                        article_chunk = self._build_article_chunk(article_data, law_name, chapter_title, section_title)
                        if len(article_chunk) > max_chunk_size:
                            sub_chunks = self._split_large_chunk(article_chunk, max_chunk_size, overlap_ratio)
                            chunks.extend(sub_chunks)
                        else:
                            chunks.append(article_chunk)
            
            elif chunk_by == "item":
                # 按項分割
                for section_data in chapter_data.get("sections", []):
                    section_title = section_data.get("section", "")
                    for article_data in section_data.get("articles", []):
                        article_title = article_data.get("article", "")
                        for item_data in article_data.get("items", []):
                            item_chunk = self._build_item_chunk(item_data, law_name, chapter_title, section_title, article_title)
                            if len(item_chunk) > max_chunk_size:
                                sub_chunks = self._split_large_chunk(item_chunk, max_chunk_size, overlap_ratio)
                                chunks.extend(sub_chunks)
                            else:
                                chunks.append(item_chunk)
        
        return chunks
    
    def _build_chapter_chunk(self, chapter_data: dict, law_name: str) -> str:
        """構建章級chunk"""
        chunk_parts = [f"【{law_name}】"]
        chunk_parts.append(chapter_data.get("chapter", ""))
        
        for section_data in chapter_data.get("sections", []):
            chunk_parts.append(section_data.get("section", ""))
            
            for article_data in section_data.get("articles", []):
                article_title = article_data.get("article", "")
                chunk_parts.append(article_title)
                
                # 添加條文內容
                if "content" in article_data:
                    chunk_parts.append(article_data["content"])
                
                # 添加項目
                for item_data in article_data.get("items", []):
                    chunk_parts.append(item_data.get("item", ""))
                    if "content" in item_data:
                        chunk_parts.append(item_data["content"])
        
        return "\n".join(filter(None, chunk_parts))
    
    def _build_section_chunk(self, section_data: dict, law_name: str, chapter_title: str) -> str:
        """構建節級chunk"""
        chunk_parts = [f"【{law_name}】", chapter_title, section_data.get("section", "")]
        
        for article_data in section_data.get("articles", []):
            article_title = article_data.get("article", "")
            chunk_parts.append(article_title)
            
            # 添加條文內容
            if "content" in article_data:
                chunk_parts.append(article_data["content"])
            
            # 添加項目
            for item_data in article_data.get("items", []):
                chunk_parts.append(item_data.get("item", ""))
                if "content" in item_data:
                    chunk_parts.append(item_data["content"])
        
        return "\n".join(filter(None, chunk_parts))
    
    def _build_article_chunk(self, article_data: dict, law_name: str, chapter_title: str, section_title: str) -> str:
        """構建條文級chunk"""
        chunk_parts = [
            f"【{law_name}】",
            chapter_title,
            section_title,
            article_data.get("article", "")
        ]
        
        # 添加條文內容
        if "content" in article_data:
            chunk_parts.append(article_data["content"])
        
        # 添加項目
        for item_data in article_data.get("items", []):
            chunk_parts.append(item_data.get("item", ""))
            if "content" in item_data:
                chunk_parts.append(item_data["content"])
        
        return "\n".join(filter(None, chunk_parts))
    
    def _build_item_chunk(self, item_data: dict, law_name: str, chapter_title: str, section_title: str, article_title: str) -> str:
        """構建項級chunk"""
        chunk_parts = [
            f"【{law_name}】",
            chapter_title,
            section_title,
            article_title,
            item_data.get("item", "")
        ]
        
        # 添加項內容
        if "content" in item_data:
            chunk_parts.append(item_data["content"])
        
        return "\n".join(filter(None, chunk_parts))
    
    def _split_large_chunk(self, chunk: str, max_size: int, overlap_ratio: float) -> List[str]:
        """分割過大的chunk，優先以段落切分，退化為行或句子切分"""
        if len(chunk) <= max_size:
            return [chunk]
        
        # 使用段落分割（空行作為界）
        paragraphs = [p for p in chunk.split('\n\n') if p.strip()]
        if len(paragraphs) <= 1:
            # 無空段落，改用單行
            paragraphs = [l for l in chunk.split('\n') if l.strip()]
        if len(paragraphs) <= 1:
            # 仍然只有一段，使用中文標點近似句子切分
            sentences = re.split(r'(?<=[。！？；;])', chunk)
            paragraphs = []
            buf = ""
            for s in sentences:
                if len(buf) + len(s) > max_size and buf:
                    paragraphs.append(buf)
                    buf = s
                else:
                    buf += s
            if buf:
                paragraphs.append(buf)
        chunks = []
        current_chunk = ""
        overlap_size = int(max_size * overlap_ratio)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果加上當前段落會超過大小限制
            if len(current_chunk) + len(paragraph) > max_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # 開始新chunk，添加重疊內容
                if overlap_ratio > 0 and len(current_chunk) > overlap_size:
                    overlap_content = current_chunk[-overlap_size:].strip()
                    current_chunk = overlap_content + "\n\n" + paragraph if overlap_content else paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # 添加最後一個chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class SemanticChunking(ChunkingStrategy):
    """語義分割策略"""
    
    def chunk(self, text: str, max_chunk_size: int = 500, **kwargs) -> List[str]:
        """語義分割"""
        # 按句子分割
        sentences = re.split(r'[。！？]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 如果當前句子加上現有chunk超過最大大小，先保存現有chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += "。" + sentence
                else:
                    current_chunk = sentence
        
        # 添加最後一個chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks


class SlidingWindowChunking(ChunkingStrategy):
    """滑動視窗分割策略"""
    
    def chunk(self, text: str, window_size: int = 500, step_size: int = 250, **kwargs) -> List[str]:
        """滑動視窗分割"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + window_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += step_size
            
        return chunks


class LLMAssistedSemanticChunking(ChunkingStrategy):
    """LLM輔助語義分割策略"""
    
    def chunk(self, text: str, max_chunk_size: int = 500, semantic_threshold: float = 0.7, **kwargs) -> List[str]:
        """LLM輔助語義分割"""
        # 首先按句子分割
        sentences = re.split(r'[。！？]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 如果當前句子加上現有chunk超過最大大小，先保存現有chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += "。" + sentence
                else:
                    current_chunk = sentence
        
        # 添加最後一個chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 這裡可以添加LLM輔助的語義分析邏輯
        # 目前先返回基本的句子級分割結果
        return chunks


class HybridChunking(ChunkingStrategy):
    """混合分割策略"""
    
    def chunk(self, text: str, primary_size: int = 600, secondary_size: int = 400, 
              switch_threshold: float = 0.5, overlap_ratio: float = 0.1, **kwargs) -> List[str]:
        """混合分割策略"""
        # 先嘗試層次分割
        hierarchical_chunker = HierarchicalChunking()
        hierarchical_chunks = hierarchical_chunker.chunk(text, max_chunk_size=primary_size, overlap_ratio=overlap_ratio)
        
        # 對過大的chunk使用固定大小分割
        final_chunks = []
        for chunk in hierarchical_chunks:
            if len(chunk) > primary_size * 1.5:
                # 使用固定大小分割處理過大的chunk
                fixed_chunker = FixedSizeChunking()
                sub_chunks = fixed_chunker.chunk(chunk, chunk_size=secondary_size, overlap_ratio=overlap_ratio)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks


class RecursiveCharacterChunking(ChunkingStrategy):
    """遞迴字符分割策略（RCTS 近似）

    以多層分隔符（段落/行/句子/逗號/空白）逐層遞迴切分，
    最終將片段合併為不超過 max_chunk_size 的塊，並加入重疊。
    """

    def chunk(
        self,
        text: str,
        max_chunk_size: int = 1000,
        overlap_ratio: float = 0.1,
        separators: list[str] | None = None,
        **kwargs,
    ) -> List[str]:
        if not text.strip():
            return []

        seps = separators or ["\n\n", "\n", "。", "，", " ", ""]
        pieces = self._recursive_split(text, seps, max_chunk_size)
        # 合併片段並加入重疊
        overlap = int(max_chunk_size * overlap_ratio)
        chunks: List[str] = []
        current = ""

        for seg in pieces:
            seg = seg.strip()
            if not seg:
                continue
            if not current:
                current = seg
                continue
            if len(current) + 1 + len(seg) <= max_chunk_size:
                current = f"{current}\n{seg}" if "\n" in seg else f"{current} {seg}".strip()
            else:
                chunks.append(current)
                prefix = current[-overlap:] if overlap > 0 else ""
                current = (prefix + ("\n" if "\n" in seg else " ") + seg).strip()
                # 若仍超長，直接窗口切分
                while len(current) > max_chunk_size:
                    chunk = current[:max_chunk_size]
                    chunks.append(chunk)
                    step = max_chunk_size - overlap if overlap > 0 else max_chunk_size
                    current = current[step:]

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _recursive_split(self, text: str, separators: list[str], max_size: int) -> List[str]:
        """按分隔符遞迴分割，直到片段不超過 max_size。
        最後一層 separators 包含空字串，代表強制按固定字元長度切分。
        """
        if len(text) <= max_size:
            return [text]
        if not separators:
            # 沒有分隔符，固定長度切分
            return [text[i : i + max_size] for i in range(0, len(text), max_size)]

        sep = separators[0]
        rest = separators[1:]

        if sep == "":
            # 最後兜底：固定長度
            return [text[i : i + max_size] for i in range(0, len(text), max_size)]

        parts = text.split(sep)
        if len(parts) == 1:
            # 無法用當前分隔符切分，遞迴到下一層
            return self._recursive_split(text, rest, max_size)

        results: List[str] = []
        buffer = ""
        for part in parts:
            candidate = (buffer + sep + part) if buffer else part
            if len(candidate) <= max_size:
                buffer = candidate
            else:
                if buffer:
                    # buffer 太大就遞迴再切
                    if len(buffer) > max_size:
                        results.extend(self._recursive_split(buffer, rest, max_size))
                    else:
                        results.append(buffer)
                buffer = part

        if buffer:
            if len(buffer) > max_size:
                results.extend(self._recursive_split(buffer, rest, max_size))
            else:
                results.append(buffer)

        return results


def get_chunking_strategy(strategy_name: str) -> ChunkingStrategy:
    """獲取分割策略"""
    strategies = {
        "fixed_size": FixedSizeChunking(),
        "hierarchical": HierarchicalChunking(),
        "rcts_hierarchical": RCTSHierarchicalChunking(),
        "structured_hierarchical": StructuredHierarchicalChunking(),
        "semantic": SemanticChunking(),
        "recursive": RecursiveCharacterChunking(),
        "sliding_window": SlidingWindowChunking(),
        "llm_semantic": LLMAssistedSemanticChunking(),
        "hybrid": HybridChunking(),
    }
    
    return strategies.get(strategy_name, FixedSizeChunking())


def chunk_text(text: str, strategy: str = "fixed_size", json_data: Dict[str, Any] | None = None, **kwargs) -> List[str]:
    """分割文本"""
    chunker = get_chunking_strategy(strategy)
    
    # 如果使用結構化層次分割，傳遞JSON數據
    if strategy == "structured_hierarchical" and json_data:
        return chunker.chunk(text, json_data=json_data, **kwargs)
    else:
        return chunker.chunk(text, **kwargs)
