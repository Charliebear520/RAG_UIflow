"""
HopRAG客戶端適配器
整合現有的LLM和Embedding系統
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
import numpy as np

class LLMClientAdapter:
    """LLM客戶端適配器 - 使用Gemini"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.use_gemini = False
        
        # 嘗試檢測Gemini
        self._detect_available_llm()
    
    def _detect_available_llm(self):
        """檢測可用的LLM服務"""
        try:
            # 檢查Gemini
            import google.generativeai as genai
            import os
            if os.getenv('GOOGLE_API_KEY'):
                self.use_gemini = True
                print("✅ 檢測到Gemini API可用")
        except ImportError:
            pass
    
    async def generate_async(self, prompt: str, max_retries: int = 3) -> str:
        """異步生成文本"""
        for attempt in range(max_retries):
            try:
                if self.use_gemini:
                    return await self._generate_with_gemini(prompt)
                else:
                    # 使用模擬響應
                    return self._generate_mock_response(prompt)
                    
            except Exception as e:
                print(f"❌ LLM生成失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # 最後一次嘗試失敗，返回模擬響應
                    return self._generate_mock_response(prompt)
                await asyncio.sleep(1)  # 等待1秒後重試
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """使用Gemini生成"""
        import google.generativeai as genai
        import os
        
        # 配置Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 生成響應
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model.generate_content(prompt)
        )
        
        return response.text
    
    
    def _generate_mock_response(self, prompt: str) -> str:
        """生成模擬響應（用於測試）"""
        if "內向問題" in prompt:
            return json.dumps({
                "incoming_questions": [
                    "此條文的主要內容是什麼？",
                    "此條文規定了哪些法律要件？",
                    "根據此條文，相關的定義是什麼？",
                    "此條文的適用範圍是什麼？",
                    "此條文規定了什麼法律後果？"
                ]
            }, ensure_ascii=False)
        
        elif "外向問題" in prompt:
            return json.dumps({
                "outgoing_questions": [
                    "違反此條文會有什麼法律後果？",
                    "如何申請此條文規定的權利？",
                    "此條文與其他法條有什麼關聯？",
                    "在什麼情況下適用此條文？",
                    "此條文的實務操作程序是什麼？"
                ]
            }, ensure_ascii=False)
        
        elif "Completely Irrelevant" in prompt or "Indirectly relevant" in prompt:
            return "Indirectly relevant"
        
        else:
            return "相關且必要"

class EmbeddingClientAdapter:
    """Embedding客戶端適配器"""
    
    def __init__(self, embedding_client=None):
        self.embedding_client = embedding_client
        self.use_gemini = False
        self.use_bge = False
        self.use_sentence_transformers = False
        
        # 嘗試檢測可用的Embedding服務
        self._detect_available_embedding()
    
    def _detect_available_embedding(self):
        """檢測可用的Embedding服務"""
        try:
            # 檢查Gemini Embedding
            import google.generativeai as genai
            import os
            if os.getenv('GOOGLE_API_KEY'):
                self.use_gemini = True
                print("✅ 檢測到Gemini Embedding API可用")
        except ImportError:
            pass
        
        try:
            # 檢查BGE-M3
            import sentence_transformers
            self.use_bge = True
            self.use_sentence_transformers = True
            print("✅ 檢測到Sentence Transformers可用")
        except ImportError:
            pass
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """同步編碼文本"""
        if self.use_gemini:
            return self._encode_with_gemini(texts)
        elif self.use_bge:
            return self._encode_with_bge(texts)
        else:
            return self._encode_mock(texts)
    
    async def encode_async(self, texts: List[str]) -> np.ndarray:
        """異步編碼文本"""
        if self.use_gemini:
            return await self._encode_with_gemini_async(texts)
        elif self.use_bge:
            return self._encode_with_bge(texts)
        else:
            return self._encode_mock(texts)
    
    async def _encode_with_gemini_async(self, texts: List[str]) -> np.ndarray:
        """使用Gemini異步編碼"""
        import google.generativeai as genai
        import os
        
        # 配置Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        embeddings = []
        for text in texts:
            try:
                # 使用Gemini Embedding API
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: genai.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type="retrieval_document"
                    )
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"❌ Gemini Embedding失敗: {e}")
                # 使用隨機向量作為fallback
                embeddings.append(np.random.randn(768).astype(np.float32))
        
        return np.array(embeddings)
    
    def _encode_with_gemini(self, texts: List[str]) -> np.ndarray:
        """使用Gemini同步編碼"""
        import google.generativeai as genai
        import os
        
        # 配置Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        embeddings = []
        for text in texts:
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"❌ Gemini Embedding失敗: {e}")
                # 使用隨機向量作為fallback
                embeddings.append(np.random.randn(768).astype(np.float32))
        
        return np.array(embeddings)
    
    def _encode_with_bge(self, texts: List[str]) -> np.ndarray:
        """使用BGE-M3編碼"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # 加載BGE-M3模型
            model = SentenceTransformer('BAAI/bge-m3')
            embeddings = model.encode(texts)
            
            return embeddings
            
        except Exception as e:
            print(f"❌ BGE-M3 Embedding失敗: {e}")
            # 使用隨機向量作為fallback
            return np.random.randn(len(texts), 1024).astype(np.float32)
    
    def _encode_mock(self, texts: List[str]) -> np.ndarray:
        """生成模擬embedding（用於測試）"""
        # 生成隨機向量作為模擬embedding
        return np.random.randn(len(texts), 768).astype(np.float32)

class HopRAGClientManager:
    """HopRAG客戶端管理器"""
    
    def __init__(self):
        self.llm_client = LLMClientAdapter()
        self.embedding_client = EmbeddingClientAdapter()
        
    def get_llm_client(self) -> LLMClientAdapter:
        """獲取LLM客戶端"""
        return self.llm_client
    
    def get_embedding_client(self) -> EmbeddingClientAdapter:
        """獲取Embedding客戶端"""
        return self.embedding_client
    
    def get_client_status(self) -> Dict[str, Any]:
        """獲取客戶端狀態"""
        return {
            "llm_status": {
                "gemini_available": self.llm_client.use_gemini,
                "model": "gemini-2.5-flash"
            },
            "embedding_status": {
                "gemini_available": self.embedding_client.use_gemini,
                "bge_available": self.embedding_client.use_bge,
                "sentence_transformers_available": self.embedding_client.use_sentence_transformers
            }
        }
