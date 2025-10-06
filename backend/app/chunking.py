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
    
    def chunk_with_span(self, text: str, chunk_size: int = 500, overlap_ratio: float = 0.1, **kwargs) -> List[Dict[str, Any]]:
        """固定大小分割，返回帶span信息的結果"""
        overlap = int(chunk_size * overlap_ratio)
        result = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunk_id = f"fixed_size_chunk_{chunk_index+1:03d}"
                metadata = {
                    "strategy": "fixed_size",
                    "chunk_size": chunk_size,
                    "overlap_ratio": overlap_ratio,
                    "overlap": overlap,
                    "chunk_index": chunk_index,
                    "length": len(chunk)
                }
                
                result.append({
                    "content": chunk,
                    "span": {"start": start, "end": end},
                    "chunk_id": chunk_id,
                    "metadata": metadata
                })
                chunk_index += 1
            
            start = end - overlap
            
        return result


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
        # 直接使用_generate_chunk_spans來生成帶span信息的chunks
        chunks = self.chunk(text, max_chunk_size, overlap_ratio, **kwargs)
        return _generate_chunk_spans(chunks, text, "hierarchical", **kwargs)


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
            # 使用自定義的RCTS替代方案
            self.rcts_splitter = CustomRCTS()
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
    
    @staticmethod
    def _to_chinese_numeral(n: int) -> str:
        """將阿拉伯數字轉為中文數字（1-99）"""
        ones = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        if n <= 10:
            return ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"][n]
        if n < 20:
            return "十" + (ones[n - 10] if n > 10 else "")
        tens = ones[n // 10] + "十"
        return tens + (ones[n % 10] if n % 10 != 0 else "")

    @classmethod
    def _normalize_kuan_label(cls, original: str, index: int) -> str:
        """款：一、二、三（無括號）"""
        return cls._to_chinese_numeral(index)

    @classmethod
    def _normalize_mu_label(cls, original: str, index: int) -> str:
        """目：(一)、(二)、(三)（有括號）"""
        return f"({cls._to_chinese_numeral(index)})"

    @staticmethod
    def _is_deleted_text(text: str) -> bool:
        """判斷條文主文是否為（刪除）或等價表示（括號可為全形/半形）。"""
        if not isinstance(text, str):
            return False
        s = text.strip()
        if not s:
            return False
        # 精確匹配（刪除），允許全形/半形括號與空白
        import re as _re
        if _re.match(r"^[（(]\s*刪除\s*[）)]$", s):
            return True
        # 兼容少數資料僅保留關鍵詞
        return s == "刪除"

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
        if json_data and ("chapters" in json_data or "laws" in json_data):
            # 支援兩種形態：單一法規物件（包含 chapters）或 corpus（包含 laws）
            if "laws" in json_data:
                chunks: List[str] = []
                for law in json_data.get("laws", []) or []:
                    chunks.extend(self._chunk_by_json_structure(law, max_chunk_size, overlap_ratio, chunk_by))
                return chunks
            return self._chunk_by_json_structure(json_data, max_chunk_size, overlap_ratio, chunk_by)
        else:
            # 如果沒有JSON結構，回退到普通層次分割
            hierarchical_chunker = HierarchicalChunking()
            return hierarchical_chunker.chunk(text, max_chunk_size, overlap_ratio)
    
    def _chunk_by_json_structure(self, json_data: dict, max_chunk_size: int, 
                                overlap_ratio: float, chunk_by: str) -> List[str]:
        """根據JSON結構進行分割（新層級命名：law_name/chapter/section/article/paragraph/subparagraph/item）"""
        chunks: List[str] = []
        law_name = json_data.get("law_name", "未命名法規")

        # law 層
        if chunk_by == "law" or chunk_by == "law_name":
            law_chunk = self._build_law_chunk(json_data)
            if len(law_chunk) > max_chunk_size:
                chunks.extend(self._split_large_chunk(law_chunk, max_chunk_size, overlap_ratio))
            else:
                chunks.append(law_chunk)
            return chunks

        for chapter_data in json_data.get("chapters", []) or []:
            chapter_title = chapter_data.get("chapter", "")

            if chunk_by == "chapter":
                chapter_chunk = self._build_chapter_chunk(chapter_data, law_name)
                if len(chapter_chunk) > max_chunk_size:
                    chunks.extend(self._split_large_chunk(chapter_chunk, max_chunk_size, overlap_ratio))
                else:
                    chunks.append(chapter_chunk)
                continue

                for section_data in chapter_data.get("sections", []) or []:
                    section_title = section_data.get("section", "")
                    # 略過未分類層級，避免與章級chunk重複
                    if section_title in ("未分類節", "未分類"):
                        continue

                if chunk_by == "section":
                    # 未分類節不產生獨立節級chunk
                    if section_title not in ("未分類節", "未分類"):
                        section_chunk = self._build_section_chunk(section_data, law_name, chapter_title)
                        if len(section_chunk) > max_chunk_size:
                            chunks.extend(self._split_large_chunk(section_chunk, max_chunk_size, overlap_ratio))
                        else:
                            chunks.append(section_chunk)
                    continue

                for article_data in section_data.get("articles", []) or []:
                    article_title = article_data.get("article", "")

                    if chunk_by == "article":
                        # 刪除條不產生條級chunk
                        if self._is_deleted_text(article_data.get("content", "")):
                            continue
                        article_chunk = self._build_article_chunk(article_data, law_name, chapter_title, section_title)
                        if len(article_chunk) > max_chunk_size:
                            chunks.extend(self._split_large_chunk(article_chunk, max_chunk_size, overlap_ratio))
                        else:
                            chunks.append(article_chunk)
                        continue

                    # paragraph 層
                    paragraphs = article_data.get("paragraphs", [])
                    items = article_data.get("items", [])
                    items_to_process = paragraphs if paragraphs else items
                    for item_data in items_to_process:
                        item_title = item_data.get("paragraph", item_data.get("item", ""))

                        if chunk_by == "paragraph":
                            item_chunk = self._build_item_chunk(item_data, law_name, chapter_title, section_title, article_title)
                            if len(item_chunk) > max_chunk_size:
                                chunks.extend(self._split_large_chunk(item_chunk, max_chunk_size, overlap_ratio))
                            else:
                                chunks.append(item_chunk)
                            continue

                        # subparagraph 層
                        subparagraphs = item_data.get("subparagraphs", [])
                        old_sub_items = item_data.get("sub_items", [])
                        sub_items_to_process = subparagraphs if subparagraphs else old_sub_items
                        for sub_item_data in sub_items_to_process:
                            if chunk_by == "subparagraph":
                                sub_item_chunk = self._build_sub_item_chunk(sub_item_data, law_name, chapter_title, section_title, article_title, item_title)
                                if len(sub_item_chunk) > max_chunk_size:
                                    chunks.extend(self._split_large_chunk(sub_item_chunk, max_chunk_size, overlap_ratio))
                                else:
                                    chunks.append(sub_item_chunk)
                                continue

                            # 第三層 item 層
                            if chunk_by == "item":
                                third_items = sub_item_data.get("items", [])
                                for third in third_items:
                                    third_chunk = third.get("content", "") or ""
                                    if not third_chunk:
                                        continue
                                    if len(third_chunk) > max_chunk_size:
                                        chunks.extend(self._split_large_chunk(third_chunk, max_chunk_size, overlap_ratio))
                                    else:
                                        chunks.append(third_chunk)

        return chunks
    
    def _build_chapter_chunk(self, chapter_data: dict, law_name: str) -> str:
        """構建章級chunk"""
        chunk_parts = [f"【{law_name}】"]
        chunk_parts.append(chapter_data.get("chapter", ""))
        
        for section_data in chapter_data.get("sections", []):
            section_title = section_data.get("section", "")
            if section_title and section_title != "未分類節":
                chunk_parts.append(section_title)
            
            for article_data in section_data.get("articles", []):
                article_title = article_data.get("article", "")
                if article_title:
                    # 若條文主文為（刪除），整條跳過
                    if self._is_deleted_text(article_data.get("content", "")):
                        continue
                    chunk_parts.append(article_title)
                
                # 條文主文
                if "content" in article_data and not self._is_deleted_text(article_data["content"]):
                    chunk_parts.append(article_data["content"])
                
                # 支援新舊結構：paragraphs 或 items
                paragraphs = article_data.get("paragraphs", [])
                items = article_data.get("items", [])
                items_to_process = paragraphs if paragraphs else items
                for item in items_to_process:
                    # 項/段落
                    item_title = item.get("paragraph", item.get("item", ""))
                    if item_title:
                        chunk_parts.append(item_title)
                    if "content" in item:
                        chunk_parts.append(item["content"])
                    
                    # 支援新結構 subparagraphs 與舊結構 sub_items（需在 item 迴圈內處理）
                    subparagraphs = item.get("subparagraphs", [])
                    old_sub_items = item.get("sub_items", [])
                    sub_items_to_process = subparagraphs if subparagraphs else old_sub_items
                    # 款索引
                    kuan_idx = 1
                    for sub in sub_items_to_process:
                        # 款：用一、二、三 ... 表示
                        normalized_kuan = self._normalize_kuan_label(sub.get("subparagraph", sub.get("sub_item", "")), kuan_idx)
                        if normalized_kuan:
                            chunk_parts.append(normalized_kuan)
                        if "content" in sub:
                            chunk_parts.append(sub["content"])
                        
                        # 第三層 items（目下的項目）
                        third_items = sub.get("items", [])
                        mu_idx = 1
                        for t in third_items:
                            # 目：用 (一)、(二)、(三) ... 表示
                            normalized_mu = self._normalize_mu_label(t.get("item", ""), mu_idx)
                            if normalized_mu:
                                chunk_parts.append(normalized_mu)
                            if "content" in t:
                                chunk_parts.append(t["content"]) 
                            mu_idx += 1
                        kuan_idx += 1
        
        return "\n".join(filter(None, chunk_parts))
    
    def _build_section_chunk(self, section_data: dict, law_name: str, chapter_title: str) -> str:
        """構建節級chunk"""
        chunk_parts = [f"【{law_name}】", chapter_title, section_data.get("section", "")]
        
        for article_data in section_data.get("articles", []):
            article_title = article_data.get("article", "")
            if article_title:
                # 若條文主文為（刪除），整條跳過
                if self._is_deleted_text(article_data.get("content", "")):
                    continue
                chunk_parts.append(article_title)
            
            # 條文主文
            if "content" in article_data and not self._is_deleted_text(article_data["content"]):
                chunk_parts.append(article_data["content"])
            
            # 支援新舊結構：paragraphs 或 items
            paragraphs = article_data.get("paragraphs", [])
            items = article_data.get("items", [])
            items_to_process = paragraphs if paragraphs else items
            for item in items_to_process:
                item_title = item.get("paragraph", item.get("item", ""))
                if item_title:
                    chunk_parts.append(item_title)
                if "content" in item:
                    chunk_parts.append(item["content"])
                
                # 支援新結構 subparagraphs 與舊結構 sub_items（需在 item 迴圈內處理）
                subparagraphs = item.get("subparagraphs", [])
                old_sub_items = item.get("sub_items", [])
                sub_items_to_process = subparagraphs if subparagraphs else old_sub_items
                kuan_idx = 1
                for sub in sub_items_to_process:
                    normalized_kuan = self._normalize_kuan_label(sub.get("subparagraph", sub.get("sub_item", "")), kuan_idx)
                    if normalized_kuan:
                        chunk_parts.append(normalized_kuan)
                    if "content" in sub:
                        chunk_parts.append(sub["content"])
                    
                    # 第三層 items（目下的項目）
                    third_items = sub.get("items", [])
                    mu_idx = 1
                    for t in third_items:
                        normalized_mu = self._normalize_mu_label(t.get("item", ""), mu_idx)
                        if normalized_mu:
                            chunk_parts.append(normalized_mu)
                        if "content" in t:
                            chunk_parts.append(t["content"])
                        mu_idx += 1
                    kuan_idx += 1
        
        return "\n".join(filter(None, chunk_parts))
    
    def _build_article_chunk(self, article_data: dict, law_name: str, chapter_title: str, section_title: str) -> str:
        """構建條文級chunk"""
        chunk_parts = [
            f"【{law_name}】",
            chapter_title,
            section_title,
            article_data.get("article", "")
        ]
        
        # 添加條文內容（僅使用新結構）
        if "content" in article_data and not self._is_deleted_text(article_data["content"]):
            chunk_parts.append(article_data["content"])
        
        # 僅使用新結構 (paragraphs)
        paragraphs = article_data.get("paragraphs", [])
        for para in paragraphs:
            item_title = para.get("paragraph", "")
            if item_title:
                chunk_parts.append(item_title)
            if "content" in para:
                chunk_parts.append(para["content"])
        
        return "\n".join(filter(None, chunk_parts))
    
    def _build_law_chunk(self, law_data: dict) -> str:
        """構建法規級chunk - 整個法規"""
        law_name = law_data.get("law_name", "未命名法規")
        chunk_parts = [f"【{law_name}】"]
        
        # 添加所有章節內容
        for chapter_data in law_data.get("chapters", []):
            chapter_title = chapter_data.get("chapter", "")
            chunk_parts.append(chapter_title)
            
            for section_data in chapter_data.get("sections", []):
                section_title = section_data.get("section", "")
                if section_title and section_title != "未分類節":
                    chunk_parts.append(section_title)
                
                for article_data in section_data.get("articles", []):
                    article_title = article_data.get("article", "")
                    chunk_parts.append(article_title)
                    
                    # 添加條文內容
                    if "content" in article_data:
                        chunk_parts.append(article_data["content"])
                    
                    # 添加項目 - 支援新結構 (paragraphs) 和舊結構 (items)
                    paragraphs = article_data.get("paragraphs", [])
                    items = article_data.get("items", [])
                    items_to_process = paragraphs if paragraphs else items
                    
                    for item_data in items_to_process:
                        # 支援新結構的鍵名
                        item_title = item_data.get("paragraph", item_data.get("item", ""))
                        chunk_parts.append(item_title)
                        if "content" in item_data:
                            chunk_parts.append(item_data["content"])
                        
                        # 添加子項
                        for sub_item_data in item_data.get("sub_items", []):
                            if "content" in sub_item_data:
                                chunk_parts.append(sub_item_data["content"])
        
        return "\n".join(filter(None, chunk_parts))
    
    def _build_item_chunk(self, item_data: dict, law_name: str, chapter_title: str, section_title: str, article_title: str, article_content: str = "") -> str:
        """構建項級chunk，包含父級條文的上下文"""
        chunk_parts = [
            f"【{law_name}】",
            chapter_title,
            section_title,
            article_title
        ]
        
        # 添加父級條文的主文內容（上下文連貫性）
        if article_content:
            chunk_parts.append(article_content)
        
        # 添加項標題和內容 - 支援新結構的鍵名
        item_title = item_data.get("paragraph", item_data.get("item", ""))
        chunk_parts.append(item_title)
        if "content" in item_data:
            chunk_parts.append(item_data["content"])
        
        return "\n".join(filter(None, chunk_parts))
    
    def _build_sub_item_chunk(self, sub_item_data: dict, law_name: str, chapter_title: str, section_title: str, article_title: str, item_title: str, article_content: str = "") -> str:
        """構建款/目級chunk，包含完整的父級上下文"""
        chunk_parts = [
            f"【{law_name}】",
            chapter_title,
            section_title,
            article_title
        ]
        
        # 添加父級條文的主文內容（上下文連貫性）
        if article_content:
            chunk_parts.append(article_content)
        
        # 添加項標題
        chunk_parts.append(item_title)
        
        # 添加款/目內容
        if "content" in sub_item_data:
            chunk_parts.append(sub_item_data["content"])
        
        return "\n".join(filter(None, chunk_parts))
    
    def chunk_with_span(self, text: str, json_data: Dict[str, Any] | None = None, max_chunk_size: int = 1000, 
                       overlap_ratio: float = 0.1, chunk_by: str = "article", **kwargs) -> List[Dict[str, Any]]:
        """
        基於JSON結構進行層次分割，返回帶span信息的結果
        
        Returns:
            List[Dict]: 包含content、span、metadata的chunk列表
        """
        if json_data and ("laws" in json_data or "chapters" in json_data):
            # 同時支援多法規格式(laws)與單一法規格式(chapters)
            normalized_json = json_data if "laws" in json_data else {"laws": [json_data]}
            return self._chunk_by_json_structure_with_span(normalized_json, max_chunk_size, overlap_ratio, chunk_by)
        else:
            # 如果沒有JSON結構，回退到普通層次分割
            hierarchical_chunker = HierarchicalChunking()
            return hierarchical_chunker.chunk_with_span(text, max_chunk_size, overlap_ratio)
    
    def _chunk_by_json_structure_with_span(self, json_data: dict, max_chunk_size: int, 
                                         overlap_ratio: float, chunk_by: str) -> List[Dict[str, Any]]:
        """根據JSON結構進行分割，返回帶span信息的結果"""
        chunks_with_span = []
        
        # 處理所有法律，不只是第一個
        for law_data in json_data.get("laws", []):
            law_name = law_data.get("law_name", "未命名法規")
            
            if chunk_by == "law":
                # 按法規分割 - 整個法規
                law_chunk = self._build_law_chunk(law_data)
                if len(law_chunk) > max_chunk_size:
                    # 如果法規太大，進一步分割
                    sub_chunks = self._split_large_chunk(law_chunk, max_chunk_size, overlap_ratio)
                    for i, sub_chunk in enumerate(sub_chunks):
                        chunks_with_span.append({
                            "content": sub_chunk,
                            "span": {"start": 0, "end": len(sub_chunk)},
                            "metadata": {
                                "strategy": "structured_hierarchical",
                                "chunk_by": chunk_by,
                                "law_name": law_name,
                                "chapter": "",
                                "section": "",
                                "article": "",
                                "chunk_index": len(chunks_with_span),
                                "length": len(sub_chunk)
                            }
                        })
                else:
                    chunks_with_span.append({
                        "content": law_chunk,
                        "span": {"start": 0, "end": len(law_chunk)},
                        "metadata": {
                            "strategy": "structured_hierarchical",
                            "chunk_by": chunk_by,
                            "law_name": law_name,
                            "chapter": "",
                            "section": "",
                            "article": "",
                            "chunk_index": len(chunks_with_span),
                            "length": len(law_chunk)
                        }
                    })
                continue  # 跳過後續的章節處理
            
            for chapter_data in law_data.get("chapters", []):
                chapter_title = chapter_data.get("chapter", "")
                
                if chunk_by == "chapter":
                    # 按章分割
                    chapter_chunk = self._build_chapter_chunk(chapter_data, law_name)
                    if len(chapter_chunk) > max_chunk_size:
                        # 如果章太大，進一步分割
                        sub_chunks = self._split_large_chunk(chapter_chunk, max_chunk_size, overlap_ratio)
                        for i, sub_chunk in enumerate(sub_chunks):
                            chunks_with_span.append({
                                "content": sub_chunk,
                                "span": {"start": 0, "end": len(sub_chunk)},  # 簡化span計算
                                "metadata": {
                                    "strategy": "structured_hierarchical",
                                    "chunk_by": chunk_by,
                                    "law_name": law_name,
                                    "chapter": chapter_title,
                                    "section": "",
                                    "article": "",
                                    "chunk_index": len(chunks_with_span),
                                    "length": len(sub_chunk)
                                }
                            })
                    else:
                        chunks_with_span.append({
                            "content": chapter_chunk,
                            "span": {"start": 0, "end": len(chapter_chunk)},
                            "metadata": {
                                "strategy": "structured_hierarchical",
                                "chunk_by": chunk_by,
                                "law_name": law_name,
                                "chapter": chapter_title,
                                "section": "",
                                "article": "",
                                "chunk_index": len(chunks_with_span),
                                "length": len(chapter_chunk)
                            }
                        })
                
                elif chunk_by == "section":
                    # 按節分割
                    for section_data in chapter_data.get("sections", []):
                        section_title = section_data.get("section", "")
                        # 略過未分類層級，避免與章級chunk重複
                        if section_title in ("未分類節", "未分類"):
                            continue
                        section_chunk = self._build_section_chunk(section_data, law_name, chapter_title)
                        if len(section_chunk) > max_chunk_size:
                            sub_chunks = self._split_large_chunk(section_chunk, max_chunk_size, overlap_ratio)
                            for sub_chunk in sub_chunks:
                                chunks_with_span.append({
                                    "content": sub_chunk,
                                    "span": {"start": 0, "end": len(sub_chunk)},
                                    "metadata": {
                                        "strategy": "structured_hierarchical",
                                        "chunk_by": chunk_by,
                                        "law_name": law_name,
                                        "chapter": chapter_title,
                                        "section": section_title,
                                        "article": "",
                                        "chunk_index": len(chunks_with_span),
                                        "length": len(sub_chunk)
                                    }
                                })
                        else:
                            chunks_with_span.append({
                                "content": section_chunk,
                                "span": {"start": 0, "end": len(section_chunk)},
                                "metadata": {
                                    "strategy": "structured_hierarchical",
                                    "chunk_by": chunk_by,
                                    "law_name": law_name,
                                    "chapter": chapter_title,
                                    "section": section_title,
                                    "article": "",
                                    "chunk_index": len(chunks_with_span),
                                    "length": len(section_chunk)
                                }
                            })

                elif chunk_by == "article":
                    # 按條文分割
                    for section_data in chapter_data.get("sections", []):
                        section_title = section_data.get("section", "")
                        # 略過未分類層級，避免與章級chunk重複
                        if section_title in ("未分類節", "未分類"):
                            continue
                        for article_data in section_data.get("articles", []):
                            article_title = article_data.get("article", "")
                            # 若條文主文為（刪除），整條跳過
                            if self._is_deleted_text(article_data.get("content", "")):
                                continue
                            article_chunk = self._build_article_chunk(article_data, law_name, chapter_title, section_title)
                            if len(article_chunk) > max_chunk_size:
                                sub_chunks = self._split_large_chunk(article_chunk, max_chunk_size, overlap_ratio)
                                for sub_chunk in sub_chunks:
                                    chunks_with_span.append({
                                        "content": sub_chunk,
                                        "span": {"start": 0, "end": len(sub_chunk)},
                                        "metadata": {
                                            "strategy": "structured_hierarchical",
                                            "chunk_by": chunk_by,
                                            "law_name": law_name,
                                            "chapter": chapter_title,
                                            "section": section_title,
                                            "article": article_title,
                                            "chunk_index": len(chunks_with_span),
                                            "length": len(sub_chunk)
                                        }
                                    })
                            else:
                                chunks_with_span.append({
                                    "content": article_chunk,
                                    "span": {"start": 0, "end": len(article_chunk)},
                                    "metadata": {
                                        "strategy": "structured_hierarchical",
                                        "chunk_by": chunk_by,
                                        "law_name": law_name,
                                        "chapter": chapter_title,
                                        "section": section_title,
                                        "article": article_title,
                                        "chunk_index": len(chunks_with_span),
                                        "length": len(article_chunk)
                                    }
                                })
                
                elif chunk_by == "item":
                    # 按項分割
                    for section_data in chapter_data.get("sections", []):
                        section_title = section_data.get("section", "")
                        # 略過未分類層級，避免與章級chunk重複
                        if section_title in ("未分類節", "未分類"):
                            continue
                        for article_data in section_data.get("articles", []):
                            article_title = article_data.get("article", "")
                            # 若條文主文為（刪除），整條跳過
                            if self._is_deleted_text(article_data.get("content", "")):
                                continue
                            # 提取條文主文內容（用於上下文連貫性）
                            article_content = article_data.get("content", "")
                            paragraphs = article_data.get("paragraphs", [])
                            items = article_data.get("items", [])
                            items_to_process = paragraphs if paragraphs else items
                            
                            for item_data in items_to_process:
                                # 支援新結構的鍵名
                                item_title = item_data.get("paragraph", item_data.get("item", ""))
                                item_chunk = self._build_item_chunk(item_data, law_name, chapter_title, section_title, article_title, article_content)
                                if len(item_chunk) > max_chunk_size:
                                    sub_chunks = self._split_large_chunk(item_chunk, max_chunk_size, overlap_ratio)
                                    for sub_chunk in sub_chunks:
                                        chunks_with_span.append({
                                            "content": sub_chunk,
                                            "span": {"start": 0, "end": len(sub_chunk)},
                                            "metadata": {
                                                "strategy": "structured_hierarchical",
                                                "chunk_by": chunk_by,
                                                "law_name": law_name,
                                                "chapter": chapter_title,
                                                "section": section_title,
                                                "article": article_title,
                                                "item": item_title,
                                                "chunk_index": len(chunks_with_span),
                                                "length": len(sub_chunk)
                                            }
                                        })
                                else:
                                    chunks_with_span.append({
                                        "content": item_chunk,
                                        "span": {"start": 0, "end": len(item_chunk)},
                                        "metadata": {
                                            "strategy": "structured_hierarchical",
                                            "chunk_by": chunk_by,
                                            "law_name": law_name,
                                            "chapter": chapter_title,
                                            "section": section_title,
                                            "article": article_title,
                                            "item": item_title,
                                            "chunk_index": len(chunks_with_span),
                                            "length": len(item_chunk)
                                        }
                                    })
                
                elif chunk_by == "sub_item":
                    # 按款/目分割
                    for section_data in chapter_data.get("sections", []):
                        section_title = section_data.get("section", "")
                        # 略過未分類節，避免與章級chunk重複
                        if section_title == "未分類節":
                            continue
                        for article_data in section_data.get("articles", []):
                            article_title = article_data.get("article", "")
                            # 若條文主文為（刪除），整條跳過
                            if self._is_deleted_text(article_data.get("content", "")):
                                continue
                            # 提取條文主文內容（用於上下文連貫性）
                            article_content = article_data.get("content", "")
                            paragraphs = article_data.get("paragraphs", [])
                            items = article_data.get("items", [])
                            items_to_process = paragraphs if paragraphs else items
                            
                            for item_data in items_to_process:
                                # 支援新結構的鍵名
                                item_title = item_data.get("paragraph", item_data.get("item", ""))
                                # 支援新結構的子項目
                                subparagraphs = item_data.get("subparagraphs", [])
                                sub_items = item_data.get("sub_items", [])
                                sub_items_to_process = subparagraphs if subparagraphs else sub_items
                                
                                for sub_item_data in sub_items_to_process:
                                    sub_item_chunk = self._build_sub_item_chunk(sub_item_data, law_name, chapter_title, section_title, article_title, item_title, article_content)
                                    if len(sub_item_chunk) > max_chunk_size:
                                        sub_chunks = self._split_large_chunk(sub_item_chunk, max_chunk_size, overlap_ratio)
                                        for sub_chunk in sub_chunks:
                                            chunks_with_span.append({
                                                "content": sub_chunk,
                                                "span": {"start": 0, "end": len(sub_chunk)},
                                                "metadata": {
                                                    "strategy": "structured_hierarchical",
                                                    "chunk_by": chunk_by,
                                                    "law_name": law_name,
                                                    "chapter": chapter_title,
                                                    "section": section_title,
                                                    "article": article_title,
                                                    "item": item_title,
                                                    "sub_item": sub_item_data.get("sub_item", ""),
                                                    "chunk_index": len(chunks_with_span),
                                                    "length": len(sub_chunk)
                                                }
                                            })
                                    else:
                                        chunks_with_span.append({
                                            "content": sub_item_chunk,
                                            "span": {"start": 0, "end": len(sub_item_chunk)},
                                            "metadata": {
                                                "strategy": "structured_hierarchical",
                                                "chunk_by": chunk_by,
                                                "law_name": law_name,
                                                "chapter": chapter_title,
                                                "section": section_title,
                                                "article": article_title,
                                                "item": item_title,
                                                "sub_item": sub_item_data.get("sub_item", ""),
                                                "chunk_index": len(chunks_with_span),
                                                "length": len(sub_item_chunk)
                                            }
                                        })
        
        return chunks_with_span
    
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


class MultiLevelStructuredChunking(StructuredHierarchicalChunking):
    """多層級結構化分割：一次性生成所有六個粒度級別的分塊"""
    
    def chunk_with_span(self, text: str, json_data: Dict[str, Any] | None = None, **kwargs) -> List[Dict[str, Any]]:
        """
        多層級結構化分割，一次性生成所有六個粒度級別的分塊
        
        Returns:
            List[Dict]: 包含所有六個層次的chunk列表
        """
        if json_data and ("laws" in json_data or "chapters" in json_data):
            # 同時支援多法規格式(laws)與單一法規格式(chapters)
            normalized_json = json_data if "laws" in json_data else {"laws": [json_data]}
            return self._multi_level_chunk_by_json_structure(normalized_json)
        else:
            # 如果沒有JSON結構，回退到普通層次分割
            hierarchical_chunker = HierarchicalChunking()
            return hierarchical_chunker.chunk_with_span(text)
    
    def _multi_level_chunk_by_json_structure(self, json_data: dict) -> List[Dict[str, Any]]:
        """根據JSON結構進行多層級分割，返回按照 Chapter/Section/Article/Paragraph/Subparagraph/Item 六層的chunks"""
        all_chunks = []
        
        # 處理所有法律
        for law_data in json_data.get("laws", []):
            law_name = law_data.get("law_name", "未命名法規")
            
            # 額外：Law 層（保持向後相容）
            law_chunk = self._build_law_chunk(law_data)
            all_chunks.append({
                "content": law_chunk,
                "span": {"start": 0, "end": len(law_chunk)},
                "metadata": {
                    "strategy": "multi_level_structured",
                    "level": "Law",
                    "level_en": "Law",
                    "law_name": law_name,
                    "chapter": "",
                    "section": "",
                    "article": "",
                    "chunk_index": len(all_chunks),
                    "length": len(law_chunk)
                }
            })
            
            # 處理章節
            for chapter_data in law_data.get("chapters", []):
                chapter_title = chapter_data.get("chapter", "")
                
                # 1. 章 Chapter
                chapter_chunk = self._build_chapter_chunk(chapter_data, law_name)
                all_chunks.append({
                    "content": chapter_chunk,
                    "span": {"start": 0, "end": len(chapter_chunk)},
                    "metadata": {
                        "strategy": "multi_level_structured",
                        "level": "Chapter",
                        "level_en": "Chapter",
                        "law_name": law_name,
                        "chapter": chapter_title,
                        "section": "",
                        "article": "",
                        "chunk_index": len(all_chunks),
                        "length": len(chapter_chunk)
                    }
                })
                
                # 處理節
                for section_data in chapter_data.get("sections", []):
                    section_title = section_data.get("section", "")
                    
                # 2. 節 Section（未分類節/未分類 不產生節級chunk）
                    if section_title not in ("未分類節", "未分類"):
                        section_chunk = self._build_section_chunk(section_data, law_name, chapter_title)
                        all_chunks.append({
                            "content": section_chunk,
                            "span": {"start": 0, "end": len(section_chunk)},
                            "metadata": {
                                "strategy": "multi_level_structured",
                                "level": "Section",
                                "level_en": "Section",
                                "law_name": law_name,
                                "chapter": chapter_title,
                                "section": section_title,
                                "article": "",
                                "chunk_index": len(all_chunks),
                                "length": len(section_chunk)
                            }
                        })
                    
                    # 處理條文
                    for article_data in section_data.get("articles", []):
                        article_title = article_data.get("article", "")
                        article_content = article_data.get("content", "")
                        
                        # 3. 條 Article（內容為「（刪除）」時跳過）
                        if not self._is_deleted_text(article_content):
                            article_chunk = self._build_article_chunk(article_data, law_name, chapter_title, section_title)
                            all_chunks.append({
                                "content": article_chunk,
                                "span": {"start": 0, "end": len(article_chunk)},
                                "metadata": {
                                    "strategy": "multi_level_structured",
                                    "level": "Article",
                                    "level_en": "Article",
                                    "law_name": law_name,
                                    "chapter": chapter_title,
                                    "section": section_title,
                                    "article": article_title,
                                    "chunk_index": len(all_chunks),
                                    "length": len(article_chunk)
                                }
                            })
                        else:
                            # 刪除條不再生成後續層級
                            continue
                        
                        # 處理項 - 支援新結構 (paragraphs) 和舊結構 (items)
                        paragraphs = article_data.get("paragraphs", [])
                        for item_data in paragraphs:
                            # 4. 項 Paragraph（僅使用新結構）
                            item_title = item_data.get("paragraph", "")
                            item_chunk = self._build_item_chunk(item_data, law_name, chapter_title, section_title, article_title, article_content)
                            all_chunks.append({
                                "content": item_chunk,
                                "span": {"start": 0, "end": len(item_chunk)},
                                "metadata": {
                                    "strategy": "multi_level_structured",
                                    "level": "Paragraph",
                                    "level_en": "Paragraph",
                                    "law_name": law_name,
                                    "chapter": chapter_title,
                                    "section": section_title,
                                    "article": article_title,
                                    "paragraph": item_title,
                                    "chunk_index": len(all_chunks),
                                    "length": len(item_chunk)
                                }
                            })
                            
                            # 處理款/目（僅使用新結構 subparagraphs → items）
                            subparagraphs = item_data.get("subparagraphs", [])
                            for sub_item_data in subparagraphs:
                                # 5. 款 Subparagraph
                                sub_item_chunk = self._build_sub_item_chunk(sub_item_data, law_name, chapter_title, section_title, article_title, item_title, article_content)
                                subparagraph_name = sub_item_data.get("subparagraph", "")
                                all_chunks.append({
                                    "content": sub_item_chunk,
                                    "span": {"start": 0, "end": len(sub_item_chunk)},
                                    "metadata": {
                                        "strategy": "multi_level_structured",
                                        "level": "Subparagraph",
                                        "level_en": "Subparagraph",
                                        "law_name": law_name,
                                        "chapter": chapter_title,
                                        "section": section_title,
                                        "article": article_title,
                                        "paragraph": item_title,
                                        "subparagraph": subparagraph_name,
                                        "chunk_index": len(all_chunks),
                                        "length": len(sub_item_chunk)
                                    }
                                })

                                # 6. 目 Item（第三層枚舉）
                                third_items = sub_item_data.get("items", [])
                                for third in third_items:
                                    third_name = third.get("item", "")
                                    third_chunk = third.get("content", "") or ""
                                    if not third_chunk:
                                        continue
                                    all_chunks.append({
                                        "content": third_chunk,
                                        "span": {"start": 0, "end": len(third_chunk)},
                                        "metadata": {
                                            "strategy": "multi_level_structured",
                                            "level": "Item",
                                            "level_en": "Item",
                                            "law_name": law_name,
                                            "chapter": chapter_title,
                                            "section": section_title,
                                            "article": article_title,
                                            "paragraph": item_title,
                                            "subparagraph": subparagraph_name,
                                            "item": third_name,
                                            "chunk_index": len(all_chunks),
                                            "length": len(third_chunk)
                                        }
                                    })
        
        return all_chunks


class SemanticChunking(ChunkingStrategy):
    """語義分割策略"""
    
    def chunk(self, text: str, max_chunk_size: int = 500, similarity_threshold: float = 0.6, context_window: int = 100, **kwargs) -> List[str]:
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
    """增強版滑動視窗分割策略 - 支援智能邊界感知和自適應重疊"""
    
    def __init__(self):
        """初始化增強版滑動視窗分割器"""
        # 中文詞語邊界標記（按優先級排序）
        self.word_boundaries = [
            '。', '！', '？', '；', '：',  # 句號、感嘆號、問號、分號、冒號
            '，', '、', '．', '·',        # 逗號、頓號、句點、間隔號
            '\n', '\r\n', '\r',           # 換行符
            ' ', '\t',                    # 空格、制表符
            '（', '）', '(', ')',         # 括號
            '「', '」', '"', '"',         # 引號
            '【', '】', '[', ']',         # 方括號
            '《', '》', '<', '>',         # 書名號
        ]
        
        # 法律文檔結構標記（高優先級邊界）
        self.legal_boundaries = [
            r'^第\s*([一二三四五六七八九十百千0-9]+)\s*條',  # 條文標記
            r'^第\s*([一二三四五六七八九十百千0-9]+)\s*章',  # 章節標記
            r'^第\s*([一二三四五六七八九十百千0-9]+)\s*節',  # 節標記
            r'^[（(]([0-9０-９一二三四五六七八九十]+)[）)]',  # 項目標記
            r'^([一二三四五六七八九十]+)[、．\.）)]',        # 項目標記
        ]
        
        # 編譯邊界正則表達式
        self.boundary_pattern = re.compile('|'.join(re.escape(b) for b in self.word_boundaries))
        self.legal_patterns = [re.compile(pattern, re.MULTILINE) for pattern in self.legal_boundaries]
        
        # 語義連貫性關鍵詞
        self.coherence_keywords = [
            '因此', '所以', '但是', '然而', '另外', '此外', '同時', '並且',
            '如果', '當', '除非', '只要', '只有', '不論', '無論',
            '根據', '依據', '按照', '依照'
        ]
    
    def chunk(self, text: str, window_size: int = 500, step_size: int = 250, 
              overlap_ratio: float = 0.1, boundary_aware: bool = True, 
              min_chunk_size: int = 100, max_chunk_size: int = 1000,
              preserve_sentences: bool = True, adaptive_overlap: bool = True,
              legal_structure_aware: bool = True, **kwargs) -> List[str]:
        """
        增強版滑動視窗分割
        
        Args:
            text: 原始文本
            window_size: 視窗大小（字符數）
            step_size: 步長（字符數）
            overlap_ratio: 重疊比例（當step_size未指定時使用）
            boundary_aware: 是否啟用邊界感知
            min_chunk_size: 最小chunk大小
            max_chunk_size: 最大chunk大小
            preserve_sentences: 是否保持句子完整性
            adaptive_overlap: 是否啟用自適應重疊
            legal_structure_aware: 是否啟用法律結構感知
        """
        if not text.strip():
            return []
        
        # 預處理：分析文本結構
        if legal_structure_aware:
            structure_info = self._analyze_legal_structure(text)
        else:
            structure_info = None
        
        # 如果未指定step_size，根據overlap_ratio計算
        if step_size is None or step_size <= 0:
            step_size = int(window_size * (1 - overlap_ratio))
        
        # 確保step_size不超過window_size
        step_size = min(step_size, window_size)
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # 計算視窗結束位置
            end = min(start + window_size, text_length)
            
            # 提取基本chunk
            chunk = text[start:end]
            
            if boundary_aware:
                # 智能邊界感知調整
                chunk, adjusted_end = self._smart_boundary_adjustment(
                    chunk, text, start, end, preserve_sentences, legal_structure_aware, structure_info
                )
                end = adjusted_end
            
            # 檢查chunk大小
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            
            # 自適應步長調整
            if adaptive_overlap:
                step_size = self._calculate_adaptive_step_size(
                    text, start, end, window_size, overlap_ratio, structure_info
                )
            
            # 移動到下一個位置
            start += step_size
            
            # 避免無限循環
            if step_size <= 0:
                break
        
        # 後處理：合併過小的chunk
        chunks = self._merge_small_chunks(chunks, min_chunk_size)
        
        # 後處理：分割過大的chunk
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk, max_chunk_size, step_size)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _analyze_legal_structure(self, text: str) -> Dict[str, Any]:
        """分析法律文檔結構"""
        structure_info = {
            'articles': [],  # 條文位置
            'chapters': [],  # 章節位置
            'sections': [],  # 節位置
            'items': [],     # 項目標記位置
            'density': 0.0   # 結構密度
        }
        
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 檢測條文
            for pattern in self.legal_patterns:
                if pattern.match(line):
                    if '條' in line:
                        structure_info['articles'].append(i)
                    elif '章' in line:
                        structure_info['chapters'].append(i)
                    elif '節' in line:
                        structure_info['sections'].append(i)
                    elif any(marker in line for marker in ['（', '(', '一', '二', '三', '四', '五']):
                        structure_info['items'].append(i)
                    break
        
        # 計算結構密度
        total_structures = len(structure_info['articles']) + len(structure_info['chapters']) + len(structure_info['sections'])
        structure_info['density'] = total_structures / len(non_empty_lines) if non_empty_lines else 0.0
        
        return structure_info
    
    def _smart_boundary_adjustment(self, chunk: str, full_text: str, start: int, end: int,
                                 preserve_sentences: bool, legal_structure_aware: bool,
                                 structure_info: Dict[str, Any]) -> Tuple[str, int]:
        """智能邊界調整，優先考慮法律結構"""
        if not preserve_sentences:
            return chunk, end
        
        # 1. 優先檢查法律結構邊界
        if legal_structure_aware and structure_info:
            legal_boundary = self._find_legal_boundary(full_text, start, end, structure_info)
            if legal_boundary is not None:
                return full_text[start:legal_boundary], legal_boundary
        
        # 2. 檢查句子邊界
        sentence_endings = ['。', '！', '？', '；']
        for i in range(end - 1, start, -1):
            if full_text[i] in sentence_endings:
                return full_text[start:i + 1], i + 1
        
        # 3. 檢查語義連貫性
        coherence_boundary = self._find_coherence_boundary(full_text, start, end)
        if coherence_boundary is not None:
            return full_text[start:coherence_boundary], coherence_boundary
        
        # 4. 檢查其他邊界
        for i in range(end - 1, start, -1):
            if full_text[i] in self.word_boundaries:
                return full_text[start:i + 1], i + 1
        
        # 如果都沒找到合適邊界，返回原始chunk
        return chunk, end
    
    def _find_legal_boundary(self, text: str, start: int, end: int, structure_info: Dict[str, Any]) -> Optional[int]:
        """查找法律結構邊界"""
        # 在當前chunk範圍內查找最近的結構邊界
        chunk_text = text[start:end]
        lines = chunk_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 檢查是否為結構標記
            for pattern in self.legal_patterns:
                if pattern.match(line):
                    # 找到結構邊界，返回該行的結束位置
                    line_end = start + chunk_text.find(line) + len(line)
                    # 確保不超過原始end位置
                    return min(line_end, end)
        
        return None
    
    def _find_coherence_boundary(self, text: str, start: int, end: int) -> Optional[int]:
        """查找語義連貫性邊界"""
        chunk_text = text[start:end]
        
        # 查找語義連貫性關鍵詞
        for keyword in self.coherence_keywords:
            keyword_pos = chunk_text.rfind(keyword)
            if keyword_pos != -1:
                # 找到關鍵詞，在其後尋找合適的切分點
                search_start = start + keyword_pos + len(keyword)
                # 使用更高效的字符查找
                punctuation_chars = {'。', '！', '？', '；', '，', '、'}
                for i in range(search_start, end):
                    if text[i] in punctuation_chars:
                        return i + 1
        
        return None
    
    def _calculate_adaptive_step_size(self, text: str, start: int, end: int, 
                                    window_size: int, overlap_ratio: float,
                                    structure_info: Dict[str, Any]) -> int:
        """計算自適應步長"""
        base_step_size = int(window_size * (1 - overlap_ratio))
        
        # 根據結構密度調整步長
        if structure_info and structure_info['density'] > 0.1:
            # 高結構密度，減少步長以保持結構完整性
            return int(base_step_size * 0.8)
        elif structure_info and structure_info['density'] < 0.05:
            # 低結構密度，可以增加步長
            return int(base_step_size * 1.2)
        
        return base_step_size
    
    def _adjust_boundary(self, chunk: str, full_text: str, start: int, end: int, 
                        preserve_sentences: bool = True) -> str:
        """調整chunk邊界，避免在詞語中間切分（向後兼容）"""
        if not preserve_sentences:
            return chunk
        
        # 嘗試在句子邊界切分
        sentence_endings = ['。', '！', '？', '；']
        
        # 向前查找最近的句子結束符
        for i in range(end - 1, start, -1):
            if full_text[i] in sentence_endings:
                return full_text[start:i + 1]
        
        # 如果沒找到句子邊界，嘗試其他邊界
        for i in range(end - 1, start, -1):
            if full_text[i] in self.word_boundaries:
                return full_text[start:i + 1]
        
        # 如果都沒找到合適邊界，返回原始chunk
        return chunk
    
    def _merge_small_chunks(self, chunks: List[str], min_size: int) -> List[str]:
        """合併過小的chunk"""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        max_merge_size = min_size * 2
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            
            # 如果當前chunk太小，嘗試與下一個合併
            if len(current_chunk) < min_size and len(current_chunk + next_chunk) <= max_merge_size:
                current_chunk += next_chunk
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        # 添加最後一個chunk
        merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def _split_large_chunk(self, chunk: str, max_size: int, step_size: int) -> List[str]:
        """分割過大的chunk"""
        if len(chunk) <= max_size:
            return [chunk]
        
        sub_chunks = []
        start = 0
        
        while start < len(chunk):
            end = min(start + max_size, len(chunk))
            sub_chunk = chunk[start:end]
            
            # 嘗試在句子邊界切分
            if end < len(chunk):
                for i in range(end - 1, start, -1):
                    if sub_chunk[i] in ['。', '！', '？', '；']:
                        sub_chunk = sub_chunk[:i + 1]
                        break
            
            sub_chunks.append(sub_chunk)
            start += step_size
        
        return sub_chunks
    
    def chunk_with_span(self, text: str, window_size: int = 500, step_size: int = 250,
                       overlap_ratio: float = 0.1, **kwargs) -> List[Dict[str, Any]]:
        """
        帶span信息的滑動視窗分割
        
        Returns:
            List[Dict]: 包含content、span、metadata的chunk列表
        """
        chunks = self.chunk(text, window_size, step_size, overlap_ratio, **kwargs)
        
        result = []
        current_pos = 0
        
        for i, chunk in enumerate(chunks):
            # 在原文中查找chunk位置
            start_pos = text.find(chunk, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(chunk)
            
            result.append({
                "content": chunk,
                "span": {"start": start_pos, "end": end_pos},
                "chunk_id": f"sliding_window_chunk_{i}",
                "metadata": {
                    "window_size": window_size,
                    "step_size": step_size,
                    "overlap_ratio": overlap_ratio,
                    "length": len(chunk),
                    "strategy": "sliding_window",
                    "chunk_index": i
                }
            })
            
            current_pos = start_pos
        
        return result
    
    def chunk_with_overlap_control(self, text: str, window_size: int = 500, 
                                 overlap_ratio: float = 0.1, **kwargs) -> List[str]:
        """
        精確控制重疊的滑動視窗分割
        """
        if not text.strip():
            return []
        
        overlap_size = int(window_size * overlap_ratio)
        step_size = window_size - overlap_size
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + window_size, text_length)
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            start += step_size
            
            # 避免無限循環
            if step_size <= 0:
                break
        
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


class CustomRCTS:
    """自定義的 Recursive Character Text Splitter 替代方案
    
    當 langchain 不可用時，提供類似的遞迴字符分割功能
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=100, length_function=len, 
                 separators=None, is_separator_regex=False):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or [
            "\n\n",  # 段落分隔
            "\n",    # 行分隔
            "。",    # 句號
            "；",    # 分號
            "，",    # 逗號
            "、",    # 頓號
            " ",     # 空格
            ""       # 字符級別
        ]
        self.is_separator_regex = is_separator_regex
    
    def split_text(self, text: str) -> List[str]:
        """分割文本，模擬 langchain 的 RecursiveCharacterTextSplitter.split_text 方法"""
        if not text.strip():
            return []
        
        # 使用遞迴分割邏輯
        return self._recursive_split(text, self.separators, self.chunk_size, self.chunk_overlap)
    
    def _recursive_split(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """遞迴分割文本"""
        if self.length_function(text) <= chunk_size:
            return [text]
        
        if not separators:
            # 如果沒有分隔符，使用固定長度分割
            return self._split_by_length(text, chunk_size, chunk_overlap)
        
        # 嘗試使用第一個分隔符分割
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # 空字符串表示字符級別分割
            return self._split_by_length(text, chunk_size, chunk_overlap)
        
        # 分割文本
        splits = text.split(separator)
        
        if len(splits) == 1:
            # 無法用當前分隔符分割，嘗試下一個分隔符
            return self._recursive_split(text, remaining_separators, chunk_size, chunk_overlap)
        
        # 合併分割結果
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # 嘗試將當前分割添加到當前chunk
            test_chunk = current_chunk + separator + split if current_chunk else split
            
            if self.length_function(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                # 當前chunk已滿，保存它
                if current_chunk:
                    chunks.append(current_chunk)
                    # 添加重疊內容
                    overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_chunk = overlap_text + separator + split if overlap_text else split
                else:
                    # 如果單個split就超過chunk_size，遞迴分割它
                    sub_chunks = self._recursive_split(split, remaining_separators, chunk_size, chunk_overlap)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
        
        # 添加最後一個chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_length(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """按固定長度分割文本"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - chunk_overlap if chunk_overlap > 0 else end
            
        return chunks




def get_chunking_strategy(strategy_name: str) -> ChunkingStrategy:
    """獲取分割策略"""
    strategies = {
        "fixed_size": FixedSizeChunking(),
        "hierarchical": HierarchicalChunking(),
        "rcts_hierarchical": RCTSHierarchicalChunking(),
        "structured_hierarchical": MultiLevelStructuredChunking(),  # 使用新的多層級分塊
        "semantic": SemanticChunking(),
        "sliding_window": SlidingWindowChunking(),
        "llm_semantic": LLMAssistedSemanticChunking(),
        "hybrid": HybridChunking(),
    }
    
    return strategies.get(strategy_name, FixedSizeChunking())


def chunk_text(text: str, strategy: str = "fixed_size", json_data: Dict[str, Any] | None = None, **kwargs) -> List[str]:
    """分割文本"""
    chunker = get_chunking_strategy(strategy)
    
    # 參數映射：將通用的chunk_size映射到策略特定的參數名
    if strategy == "sliding_window" and "chunk_size" in kwargs:
        kwargs["window_size"] = kwargs.pop("chunk_size")
    
    # 如果使用結構化層次分割，傳遞JSON數據
    if strategy == "structured_hierarchical" and json_data:
        return chunker.chunk(text, json_data=json_data, **kwargs)
    else:
        return chunker.chunk(text, **kwargs)


def chunk_text_with_span(text: str, strategy: str = "fixed_size", json_data: Dict[str, Any] | None = None, **kwargs) -> List[Dict[str, Any]]:
    """
    分割文本並返回帶span信息的結果
    
    Returns:
        List[Dict]: 包含content、span、chunk_id、metadata的chunk列表
    """
    chunker = get_chunking_strategy(strategy)
    
    # 參數映射：將通用的chunk_size映射到策略特定的參數名
    if strategy == "sliding_window" and "chunk_size" in kwargs:
        kwargs["window_size"] = kwargs.pop("chunk_size")
    
    # 檢查策略是否支持span功能
    if hasattr(chunker, 'chunk_with_span'):
        # 如果使用結構化層次分割，傳遞JSON數據
        if strategy == "structured_hierarchical" and json_data:
            return chunker.chunk_with_span(text, json_data=json_data, **kwargs)
        else:
            return chunker.chunk_with_span(text, **kwargs)
    else:
        # 回退到基本分割，然後手動計算span
        chunks = chunk_text(text, strategy, json_data, **kwargs)
        return _generate_chunk_spans(chunks, text, strategy, json_data, **kwargs)


def _generate_chunk_spans(chunks: List[str], text: str, strategy: str, json_data: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
    """
    為chunks生成span信息，並添加對應的法條JSON span信息
    
    Args:
        chunks: 分塊後的文本列表
        text: 原始文本
        strategy: 分割策略
        json_data: 法條JSON數據
        **kwargs: 其他參數
    
    Returns:
        List[Dict]: 包含span信息的chunk列表
    """
    result = []
    current_pos = 0
    
    # 提取所有法條的span信息
    law_spans = []
    if json_data and "laws" in json_data:
        for law in json_data["laws"]:
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
                                    "section_name": section.get("section", "")
                                })
    
    for i, chunk in enumerate(chunks):
        # 對於hierarchical策略，使用更準確的span計算
        if strategy == "hierarchical":
            # 使用累積位置計算
            if i == 0:
                start_pos = 0
            else:
                # 基於前一個chunk的位置計算當前位置
                prev_chunk = chunks[i-1]
                prev_start = current_pos
                # 在文檔中查找當前chunk的位置，從前一個chunk結束位置開始
                start_pos = text.find(chunk, prev_start)
                if start_pos == -1:
                    # 如果找不到，嘗試在整個文檔中查找
                    start_pos = text.find(chunk)
                    if start_pos == -1:
                        # 如果還是找不到，使用累積位置
                        start_pos = current_pos
        else:
            # 對於其他策略，使用原來的邏輯
            start_pos = text.find(chunk, current_pos)
            if start_pos == -1:
                # 如果找不到，嘗試在整個文檔中查找
                start_pos = text.find(chunk)
                if start_pos == -1:
                    # 如果還是找不到，使用當前位置
                    start_pos = current_pos
        
        end_pos = start_pos + len(chunk)
        
        # 確保不超出文檔範圍
        if end_pos > len(text):
            end_pos = len(text)
        
        # 更新current_pos，但不要超過文檔長度
        current_pos = min(end_pos, len(text))
        
        # 生成chunk ID
        chunk_id = f"{strategy}_chunk_{i+1:03d}"
        
        # 找到與此chunk重疊的法條spans
        overlapping_law_spans = []
        for law_span in law_spans:
            law_start = law_span["start_char"]
            law_end = law_span["end_char"]
            
            # 計算重疊
            overlap_start = max(start_pos, law_start)
            overlap_end = min(end_pos, law_end)
            if overlap_start < overlap_end:
                overlap_ratio = (overlap_end - overlap_start) / (end_pos - start_pos)
                if overlap_ratio > 0.1:  # 至少10%重疊
                    overlapping_law_spans.append({
                        **law_span,
                        "overlap_ratio": overlap_ratio,
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end
                    })
        
        # 按重疊比例排序
        overlapping_law_spans.sort(key=lambda x: x["overlap_ratio"], reverse=True)
        
        # 生成metadata
        metadata = {
            "strategy": strategy,
            "chunk_index": i,
            "length": len(chunk),
            "overlapping_law_spans": overlapping_law_spans,
            **kwargs
        }
        
        result.append({
            "content": chunk,
            "span": {"start": start_pos, "end": end_pos},
            "chunk_id": chunk_id,
            "metadata": metadata
        })
        
        # 更新當前位置，考慮可能的重疊
        current_pos = max(current_pos, start_pos)
    
    return result
