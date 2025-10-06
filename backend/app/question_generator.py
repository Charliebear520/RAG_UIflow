"""
問題生成模組
"""

import json
import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai


class QuestionGenerator:
    """問題生成器"""
    
    def __init__(self):
        # 優先使用 GOOGLE_API_KEY，如果沒有則使用 GEMINI_API_KEY
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None
    
    def is_available(self) -> bool:
        """檢查API是否可用"""
        return self.api_key is not None and self.model is not None
    
    def generate_questions(self, text: str, question_types: List[str], num_questions: int, difficulty_levels: List[str]) -> Dict[str, Any]:
        """生成問題"""
        if not self.is_available():
            return {
                "success": False,
                "error": "Gemini API 未配置或不可用",
                "questions": []
            }
        
        try:
            # 構建提示詞
            prompt = self._build_prompt(text, question_types, num_questions, difficulty_levels)
            
            # 生成問題
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # 解析響應
            questions_data = self._parse_response(response_text)
            
            return {
                "success": True,
                "questions": questions_data,
                "total_generated": len(questions_data)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "questions": []
            }
    
    def _build_prompt(self, text: str, question_types: List[str], num_questions: int, difficulty_levels: List[str]) -> str:
        """構建提示詞"""
        prompt = f"""你是一個專業的法律教育專家，請根據以下法律文本生成考古題。

## 核心設計原則
1. **實際應用導向**: 重點生成能夠測試實際法律應用能力的問題，而非單純的法條背誦
2. **情境化設計**: 創建真實的法律情境和案例，讓學習者能夠將法律知識應用到具體情況中
3. **避免機械記憶**: 不要生成「第幾條規定什麼」這類純背誦性問題，而是測試理解和應用能力
4. **實務導向**: 問題應該反映法律實務中的真實挑戰和決策需求

## 法律文本內容
{text[:3000]}...

## 問題生成要求
- 問題類型: {', '.join(question_types)}
- 難度等級: {', '.join(difficulty_levels)}
- 生成數量: {num_questions} 題

## 輸出格式
請以JSON格式輸出，包含以下結構：
```json
{{
  "questions": [
    {{
      "question": "問題內容",
      "question_type": "案例應用",
      "difficulty": "基礎",
      "related_articles": ["相關法條"],
      "keywords": ["關鍵詞1", "關鍵詞2"],
      "explanation": "解析說明"
    }}
  ]
}}
```

請確保問題具有實用性和挑戰性，能夠真正測試學習者對法律概念的理解和應用能力。"""

        return prompt
    
    def _parse_response(self, response_text: str) -> List[Dict[str, Any]]:
        """解析API響應"""
        try:
            # 嘗試直接解析JSON
            if response_text.startswith('```json'):
                # 移除markdown代碼塊標記
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                # 移除通用代碼塊標記
                response_text = response_text.replace('```', '').strip()
            
            # 解析JSON
            data = json.loads(response_text)
            
            if isinstance(data, dict) and 'questions' in data:
                return data['questions']
            elif isinstance(data, list):
                return data
            else:
                return []
                
        except json.JSONDecodeError:
            # 如果JSON解析失敗，嘗試提取問題
            return self._extract_questions_fallback(response_text)
    
    def _extract_questions_fallback(self, text: str) -> List[Dict[str, Any]]:
        """備用問題提取方法"""
        questions = []
        lines = text.split('\n')
        
        current_question = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 簡單的啟發式規則提取問題
            if '?' in line or '？' in line:
                if current_question:
                    questions.append(current_question)
                
                current_question = {
                    "question": line,
                    "question_type": "案例應用",
                    "difficulty": "基礎",
                    "related_articles": [],
                    "keywords": [],
                    "explanation": ""
                }
        
        if current_question:
            questions.append(current_question)
        
        return questions


def generate_questions(text: str, question_types: List[str], num_questions: int, difficulty_levels: List[str]) -> Dict[str, Any]:
    """生成問題的便捷函數"""
    generator = QuestionGenerator()
    return generator.generate_questions(text, question_types, num_questions, difficulty_levels)
