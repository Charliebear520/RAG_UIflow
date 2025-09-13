"""
PDF處理模組
"""

import io
import time
from typing import Dict, Any, Optional
import pdfplumber
from PyPDF2 import PdfReader
from .models import MetadataOptions


def convert_pdf_to_text(file_content: bytes, options: MetadataOptions) -> Dict[str, Any]:
    """將PDF轉換為文本"""
    start_time = time.time()
    
    try:
        # 使用pdfplumber讀取PDF
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            text_content = ""
            metadata = {
                "page_count": len(pdf.pages),
                "pages": [],
                "total_characters": 0,
                "extraction_method": "pdfplumber"
            }
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_start = time.time()
                
                # 提取文本
                page_text = page.extract_text() or ""
                
                # 提取表格
                tables = []
                if options.include_tables:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:
                            tables.append(table)
                
                # 構建頁面信息
                page_info = {
                    "page_number": page_num,
                    "text": page_text,
                    "character_count": len(page_text),
                    "extraction_time": time.time() - page_start
                }
                
                if options.include_tables and tables:
                    page_info["tables"] = tables
                
                if options.include_page_numbers:
                    page_info["page_number"] = page_num
                
                metadata["pages"].append(page_info)
                text_content += page_text + "\n\n"
                metadata["total_characters"] += len(page_text)
            
            # 處理時間統計
            processing_time = time.time() - start_time
            
            return {
                "text": text_content.strip(),
                "metadata": metadata,
                "processing_time": processing_time,
                "success": True
            }
            
    except Exception as e:
        return {
            "text": "",
            "metadata": {"error": str(e)},
            "processing_time": time.time() - start_time,
            "success": False,
            "error": str(e)
        }


def convert_pdf_fallback(file_content: bytes, options: MetadataOptions) -> Dict[str, Any]:
    """PDF轉換備用方法（使用PyPDF2）"""
    start_time = time.time()
    
    try:
        reader = PdfReader(io.BytesIO(file_content))
        text_content = ""
        metadata = {
            "page_count": len(reader.pages),
            "pages": [],
            "total_characters": 0,
            "extraction_method": "PyPDF2_fallback"
        }
        
        for page_num, page in enumerate(reader.pages, 1):
            page_start = time.time()
            
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            
            page_info = {
                "page_number": page_num,
                "text": page_text,
                "character_count": len(page_text),
                "extraction_time": time.time() - page_start
            }
            
            if options.include_page_numbers:
                page_info["page_number"] = page_num
            
            metadata["pages"].append(page_info)
            text_content += page_text + "\n\n"
            metadata["total_characters"] += len(page_text)
        
        processing_time = time.time() - start_time
        
        return {
            "text": text_content.strip(),
            "metadata": metadata,
            "processing_time": processing_time,
            "success": True
        }
        
    except Exception as e:
        return {
            "text": "",
            "metadata": {"error": str(e)},
            "processing_time": time.time() - start_time,
            "success": False,
            "error": str(e)
        }


def extract_pdf_metadata(file_content: bytes) -> Dict[str, Any]:
    """提取PDF元數據"""
    try:
        reader = PdfReader(io.BytesIO(file_content))
        metadata = reader.metadata or {}
        
        return {
            "title": metadata.get("/Title", ""),
            "author": metadata.get("/Author", ""),
            "subject": metadata.get("/Subject", ""),
            "creator": metadata.get("/Creator", ""),
            "producer": metadata.get("/Producer", ""),
            "creation_date": metadata.get("/CreationDate", ""),
            "modification_date": metadata.get("/ModDate", ""),
            "page_count": len(reader.pages)
        }
    except Exception as e:
        return {"error": str(e)}
