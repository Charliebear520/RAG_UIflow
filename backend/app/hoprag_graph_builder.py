"""
HopRAG圖構建器模組
包含PassageGraphBuilder、PseudoQueryGenerator、EdgeConnector
"""

import json
import numpy as np
import networkx as nx
import time
import re
import jieba
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import asyncio

from .hoprag_config import HopRAGConfig, NodeType, EdgeType, QueryType, DEFAULT_CONFIG

@dataclass
class PseudoQuery:
    """偽查詢數據結構"""
    query_id: str
    content: str
    query_type: QueryType
    embedding: Optional[np.ndarray] = None
    keywords: Optional[set] = None  # k: 關鍵詞集合
    similarity_threshold: float = 0.7
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = set()

@dataclass
class EdgeFeature:
    """邊特徵數據結構 - 按照論文要求"""
    query: str  # q_t*,j*^-
    combined_keywords: Set[str]  # k_t*,j*^- ∪ k_s,i^+
    embedding: np.ndarray  # v_t*,j*^-
    jaccard_score: float
    cosine_score: float
    mixed_score: float

@dataclass
class LegalNode:
    """法律節點數據結構"""
    node_id: str
    node_type: NodeType
    content: str
    contextualized_text: str
    law_name: str
    article_number: str
    item_number: Optional[str] = None
    parent_article_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    # 偽查詢
    incoming_questions: List[str] = None
    outgoing_questions: List[str] = None
    pseudo_queries: Dict[str, List[PseudoQuery]] = None
    
    # 圖結構相關
    outgoing_edges: List[str] = None
    incoming_edges: List[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.incoming_questions is None:
            self.incoming_questions = []
        if self.outgoing_questions is None:
            self.outgoing_questions = []
        if self.pseudo_queries is None:
            self.pseudo_queries = {"incoming": [], "outgoing": []}
        if self.outgoing_edges is None:
            self.outgoing_edges = []
        if self.incoming_edges is None:
            self.incoming_edges = []

class PseudoQueryGenerator:
    """偽查詢生成器"""
    
    def __init__(self, llm_client, config: HopRAGConfig = DEFAULT_CONFIG):
        self.llm_client = llm_client
        self.config = config
        
    async def generate_pseudo_queries_for_node(self, node: LegalNode) -> LegalNode:
        """為單個節點生成內向和外向問題"""
        print(f"🔍 為節點 {node.node_id} 生成偽查詢...")
        
        # 生成內向問題
        incoming_questions = await self._generate_incoming_questions(node)
        
        # 生成外向問題
        outgoing_questions = await self._generate_outgoing_questions(node)
        
        # 更新節點
        node.incoming_questions = incoming_questions
        node.outgoing_questions = outgoing_questions
        
        # 創建PseudoQuery對象
        node.pseudo_queries = {
            "incoming": [
                PseudoQuery(
                    query_id=f"{node.node_id}_in_{i}",
                    content=question,
                    query_type=QueryType.INCOMING,
                    keywords=self._extract_keywords_from_query(question)
                ) for i, question in enumerate(incoming_questions)
            ],
            "outgoing": [
                PseudoQuery(
                    query_id=f"{node.node_id}_out_{i}",
                    content=question,
                    query_type=QueryType.OUTGOING,
                    keywords=self._extract_keywords_from_query(question)
                ) for i, question in enumerate(outgoing_questions)
            ]
        }
        
        print(f"✅ 節點 {node.node_id} 偽查詢生成完成：{len(incoming_questions)}個內向，{len(outgoing_questions)}個外向")
        return node
    
    async def _generate_incoming_questions(self, node: LegalNode) -> List[str]:
        """生成內向問題 - 動態數量版本"""
        
        if self.config.use_dynamic_question_count:
            return await self._generate_dynamic_incoming_questions(node)
        else:
            return await self._generate_fixed_incoming_questions(node)
    
    async def _generate_dynamic_incoming_questions(self, node: LegalNode) -> List[str]:
        """動態生成內向問題 - 讓LLM決定適當數量"""
        
        prompt = f"""
您是一位法律專家，擅長提出問題並精通中文。您需要根據法律條文中的幾句連續句子生成問題。

任務：為以下法律條文生成問題，這些問題的答案必須直接來自該文本本身。

法律條文內容：
{node.contextualized_text}

要求：
1. 法律要素：每個問題必須包含特定的法律要素（條文號、適用範圍、法律後果、權利義務、構成要件等）或其他關鍵特徵，以減少歧義、澄清上下文並確保自包含性。
2. 省略/空白：您可以省略或留空句子的重要部分來形成問題，但不應對同一部分提出多個問題。不必為句子的每個部分都提出問題。
3. 與省略部分的連貫性：當詢問省略的部分時，非省略的信息應包含在問題中，只要保持連貫性。
4. 多樣性：不同的問題應關注句子中信息的不同方面，確保多樣性和代表性。
5. 覆蓋和標準化：所有問題結合起來應涵蓋所提供句子的所有關鍵點，措辭應標準化。
6. 客觀和詳細：問題應客觀、基於事實且注重細節（例如，詢問法律要件、適用條件、法律效果等）。答案必須僅來自所提供的句子。

數量要求：
- 最少生成 {self.config.min_incoming_questions} 個問題
- 最多生成 {self.config.max_incoming_questions} 個問題
- 請根據法律條文的複雜度和信息豐富程度，生成適當數量的問題來充分覆蓋所有重要信息
- 如果條文內容簡單，生成較少問題；如果條文內容複雜，生成較多問題

範例：
句子列表：["第八條：著作權人享有下列權利：一、重製權；二、公開播送權；三、公開傳輸權。"]
答案範例：
{{
    "Question List": [
        "第八條規定了著作權人的哪些權利？",
        "著作權人的重製權是什麼？",
        "著作權人的公開播送權是什麼？",
        "著作權人的公開傳輸權是什麼？",
        "第八條總共規定了幾項著作權？"
    ]
}}

請以JSON格式返回，格式如下：
{{
    "Question List": [
        "問題1",
        "問題2",
        "問題3"
    ]
}}

請確保JSON格式正確，避免不必要的轉義、換行或空格。您還應該特別注意確保，除了JSON和列表格式本身使用雙引號(")外，其他所有雙引號的實例都應替換為單引號。例如，使用'著作權法'而不是"著作權法"。
請確保JSON格式正確，不要包含任何其他文字。
"""
        
        try:
            response = await self.llm_client.generate_async(prompt)
            
            # 解析JSON響應
            if response.strip().startswith('{'):
                result = json.loads(response.strip())
                questions = result.get('Question List', [])
                
                # 驗證問題數量
                questions = self._validate_question_count(
                    questions, 
                    self.config.min_incoming_questions, 
                    self.config.max_incoming_questions,
                    "incoming"
                )
                
                # 驗證問題質量
                validated_questions = self._validate_questions(questions, "incoming")
                return validated_questions
            else:
                # 如果響應不是JSON格式，嘗試提取問題
                questions = self._extract_questions_from_text(response)
                questions = self._validate_question_count(
                    questions, 
                    self.config.min_incoming_questions, 
                    self.config.max_incoming_questions,
                    "incoming"
                )
                return questions
                
        except Exception as e:
            print(f"❌ 生成動態內向問題失敗: {e}")
            # 返回默認問題
            return self._generate_default_incoming_questions(node)
    
    async def _generate_fixed_incoming_questions(self, node: LegalNode) -> List[str]:
        """固定數量生成內向問題 - 向後兼容"""
        
        prompt = f"""
您是一位法律專家，擅長提出問題並精通中文。您需要根據法律條文中的幾句連續句子生成問題。

任務：為以下法律條文生成問題，這些問題的答案必須直接來自該文本本身。

法律條文內容：
{node.contextualized_text}

要求：
1. 法律要素：每個問題必須包含特定的法律要素（條文號、適用範圍、法律後果、權利義務、構成要件等）或其他關鍵特徵，以減少歧義、澄清上下文並確保自包含性。
2. 省略/空白：您可以省略或留空句子的重要部分來形成問題，但不應對同一部分提出多個問題。不必為句子的每個部分都提出問題。
3. 與省略部分的連貫性：當詢問省略的部分時，非省略的信息應包含在問題中，只要保持連貫性。
4. 多樣性：不同的問題應關注句子中信息的不同方面，確保多樣性和代表性。
5. 覆蓋和標準化：所有問題結合起來應涵蓋所提供句子的所有關鍵點，措辭應標準化。
6. 客觀和詳細：問題應客觀、基於事實且注重細節（例如，詢問法律要件、適用條件、法律效果等）。答案必須僅來自所提供的句子。

範例：
句子列表：["第八條：著作權人享有下列權利：一、重製權；二、公開播送權；三、公開傳輸權。"]

答案範例：
{{
    "Question List": [
        "第八條規定了著作權人的哪些權利？",
        "著作權人的重製權是什麼？",
        "著作權人的公開播送權是什麼？",
        "著作權人的公開傳輸權是什麼？",
        "第八條總共規定了幾項著作權？"
    ]
}}

請生成{self.config.max_pseudo_queries_per_node}個問題，嚴格遵循JSON格式，避免不必要的轉義、換行或空格。您還應該特別注意確保，除了JSON和列表格式本身使用雙引號(")外，其他所有雙引號的實例都應替換為單引號。例如，使用'著作權法'而不是"著作權法"。

請以JSON格式返回，格式如下：
{{
    "Question List": [
        "問題1",
        "問題2",
        "問題3",
        "問題4",
        "問題5"
    ]
}}

請確保JSON格式正確，不要包含任何其他文字。
"""
        
        try:
            response = await self.llm_client.generate_async(prompt)
            
            # 解析JSON響應
            if response.strip().startswith('{'):
                result = json.loads(response.strip())
                questions = result.get('Question List', [])
                
                # 驗證問題質量
                validated_questions = self._validate_questions(questions, "incoming")
                # 根據配置決定是否限制數量
                if self.config.use_dynamic_question_count:
                    # 動態模式：不限制數量，讓LLM決定
                    return validated_questions
                else:
                    # 固定模式：使用配置的數量限制
                    return validated_questions[:self.config.max_pseudo_queries_per_node]
            else:
                # 如果響應不是JSON格式，嘗試提取問題
                questions = self._extract_questions_from_text(response)
                # 根據配置決定是否限制數量
                if self.config.use_dynamic_question_count:
                    # 動態模式：不限制數量，讓LLM決定
                    return questions
                else:
                    # 固定模式：使用配置的數量限制
                    return questions[:self.config.max_pseudo_queries_per_node]
                
        except Exception as e:
            print(f"❌ 生成內向問題失敗: {e}")
            # 返回默認問題
            return self._generate_default_incoming_questions(node)
    
    async def _generate_outgoing_questions(self, node: LegalNode) -> List[str]:
        """生成外向問題 - 動態數量版本"""
        
        if self.config.use_dynamic_question_count:
            return await self._generate_dynamic_outgoing_questions(node)
        else:
            return await self._generate_fixed_outgoing_questions(node)
    
    async def _generate_dynamic_outgoing_questions(self, node: LegalNode) -> List[str]:
        """動態生成外向問題 - 讓LLM決定適當數量"""
        
        prompt = f"""
您是一位法律專家，擅長提出深刻問題並精通中文。您需要根據法律條文中的幾句連續句子生成後續問題。

任務：為以下法律條文生成後續問題，這些問題的答案不在給定的句子中找到。

後續問題定義：答案不在給定的句子中找到。答案可以從給定句子之前或之後的上下文、涵蓋相同法律事件的相關法條、或從給定句子中關鍵詞的邏輯、因果或時間延伸中推斷出來。

法律條文內容：
{node.contextualized_text}

要求：
1. 法律要素：每個問題必須包含特定的法律要素（條文號、適用範圍、法律後果、權利義務、構成要件等）或其他關鍵特徵，以減少歧義並確保問題的獨立性。
2. 多樣性和客觀性：不同的後續問題應關注這些句子所代表的整體法律事件的多樣化、客觀方面，確保多樣性和代表性。優先考慮客觀問題。
3. 因果和邏輯關係：根據給定的句子，生成涉及因果關係、並行關係、序列、進程、連接和其他邏輯方面的問題。可探索的領域包括但不限於：法律事件的背景、信息、原因、影響、意義、發展趨勢或相關個人的觀點。

數量要求：
- 最少生成 {self.config.min_outgoing_questions} 個問題
- 最多生成 {self.config.max_outgoing_questions} 個問題
- 請根據法律條文的複雜度和延伸性，生成適當數量的問題來充分探索邏輯連接
- 如果條文內容簡單，生成較少問題；如果條文內容複雜且有多個延伸點，生成較多問題

範例：
句子列表：["第八條：著作權人享有下列權利：一、重製權；二、公開播送權；三、公開傳輸權。"]
答案範例：
{{
    "Question List": [
        "著作權人如何行使重製權？",
        "重製權的保護期限是多久？",
        "違反重製權會有什麼法律後果？",
        "公開播送權與公開傳輸權有什麼區別？",
        "著作權人如何證明其權利受到侵害？"
    ]
}}

請以JSON格式返回，格式如下：
{{
    "Question List": [
        "問題1",
        "問題2",
        "問題3"
    ]
}}

請確保JSON格式正確，避免不必要的轉義、換行或空格。您還應該特別注意確保，除了JSON和列表格式本身使用雙引號(")外，其他所有雙引號的實例都應替換為單引號。例如，使用'著作權法'而不是"著作權法"。
請確保JSON格式正確，不要包含任何其他文字。
"""
        
        try:
            response = await self.llm_client.generate_async(prompt)
            
            # 解析JSON響應
            if response.strip().startswith('{'):
                result = json.loads(response.strip())
                questions = result.get('Question List', [])
                
                # 驗證問題數量
                questions = self._validate_question_count(
                    questions, 
                    self.config.min_outgoing_questions, 
                    self.config.max_outgoing_questions,
                    "outgoing"
                )
                
                # 驗證問題質量
                validated_questions = self._validate_questions(questions, "outgoing")
                return validated_questions
            else:
                # 如果響應不是JSON格式，嘗試提取問題
                questions = self._extract_questions_from_text(response)
                questions = self._validate_question_count(
                    questions, 
                    self.config.min_outgoing_questions, 
                    self.config.max_outgoing_questions,
                    "outgoing"
                )
                return questions
                
        except Exception as e:
            print(f"❌ 生成動態外向問題失敗: {e}")
            # 返回默認問題
            return self._generate_default_outgoing_questions(node)
    
    async def _generate_fixed_outgoing_questions(self, node: LegalNode) -> List[str]:
        """固定數量生成外向問題 - 向後兼容"""
        
        prompt = f"""
您是一位法律專家，擅長提出深刻問題並精通中文。您需要根據法律條文中的幾句連續句子生成後續問題。

任務：為以下法律條文生成後續問題，這些問題的答案不在給定的句子中找到。

後續問題定義：答案不在給定的句子中找到。答案可以從給定句子之前或之後的上下文、涵蓋相同法律事件的相關法條、或從給定句子中關鍵詞的邏輯、因果或時間延伸中推斷出來。

法律條文內容：
{node.contextualized_text}

要求：
1. 法律要素：每個問題必須包含特定的法律要素（條文號、適用範圍、法律後果、權利義務、構成要件等）或其他關鍵特徵，以減少歧義並確保問題的獨立性。
2. 多樣性和客觀性：不同的後續問題應關注這些句子所代表的整體法律事件的多樣化、客觀方面，確保多樣性和代表性。優先考慮客觀問題。
3. 因果和邏輯關係：根據給定的句子，生成涉及因果關係、並行關係、序列、進程、連接和其他邏輯方面的問題。可探索的領域包括但不限於：法律事件的背景、信息、原因、影響、意義、發展趨勢或相關個人的觀點。

範例：
句子列表：["第八條：著作權人享有下列權利：一、重製權；二、公開播送權；三、公開傳輸權。"]

答案範例：
{{
    "Question List": [
        "著作權人如何行使重製權？",
        "重製權的保護期限是多久？",
        "違反重製權會有什麼法律後果？",
        "公開播送權與公開傳輸權有什麼區別？",
        "著作權人如何證明其權利受到侵害？"
    ]
}}

請生成{self.config.max_pseudo_queries_per_node}個問題，嚴格遵循JSON格式，避免不必要的轉義、換行或空格。您還應該特別注意確保，除了JSON和列表格式本身使用雙引號(")外，其他所有雙引號的實例都應替換為單引號。例如，使用'著作權法'而不是"著作權法"。

請以JSON格式返回，格式如下：
{{
    "Question List": [
        "問題1",
        "問題2",
        "問題3",
        "問題4",
        "問題5"
    ]
}}

請確保JSON格式正確，不要包含任何其他文字。
"""
        
        try:
            response = await self.llm_client.generate_async(prompt)
            
            # 解析JSON響應
            if response.strip().startswith('{'):
                result = json.loads(response.strip())
                questions = result.get('Question List', [])
                
                # 驗證問題質量
                validated_questions = self._validate_questions(questions, "outgoing")
                # 根據配置決定是否限制數量
                if self.config.use_dynamic_question_count:
                    # 動態模式：不限制數量，讓LLM決定
                    return validated_questions
                else:
                    # 固定模式：使用配置的數量限制
                    return validated_questions[:self.config.max_pseudo_queries_per_node]
            else:
                # 如果響應不是JSON格式，嘗試提取問題
                questions = self._extract_questions_from_text(response)
                # 根據配置決定是否限制數量
                if self.config.use_dynamic_question_count:
                    # 動態模式：不限制數量，讓LLM決定
                    return questions
                else:
                    # 固定模式：使用配置的數量限制
                    return questions[:self.config.max_pseudo_queries_per_node]
                
        except Exception as e:
            print(f"❌ 生成外向問題失敗: {e}")
            # 返回默認問題
            return self._generate_default_outgoing_questions(node)
    
    def _validate_questions(self, questions: List[str], question_type: str) -> List[str]:
        """驗證問題質量"""
        validated = []
        
        for question in questions:
            if isinstance(question, str) and len(question.strip()) > 10:
                # 基本驗證：問題長度、包含問號等
                question = question.strip()
                if question.endswith('？') or question.endswith('?'):
                    validated.append(question)
                else:
                    # 如果沒有問號，添加一個
                    validated.append(question + '？')
        
        return validated
    
    def _validate_question_count(self, questions: List[str], min_count: int, max_count: int, question_type: str) -> List[str]:
        """驗證問題數量是否符合要求"""
        if not questions:
            print(f"⚠️ {question_type}問題為空，使用默認問題")
            return self._get_default_questions(question_type, min_count)
        
        question_count = len(questions)
        
        if question_count < min_count:
            print(f"⚠️ {question_type}問題數量不足（{question_count} < {min_count}），補充默認問題")
            # 補充默認問題到最少數量
            default_questions = self._get_default_questions(question_type, min_count - question_count)
            questions.extend(default_questions)
        elif question_count > max_count:
            print(f"⚠️ {question_type}問題數量過多（{question_count} > {max_count}），截取前{max_count}個")
            questions = questions[:max_count]
        
        print(f"✅ {question_type}問題數量驗證完成：{len(questions)}個問題")
        return questions
    
    def _get_default_questions(self, question_type: str, count: int) -> List[str]:
        """獲取默認問題"""
        if question_type == "incoming":
            return [
                f"此條文的主要內容是什麼？",
                f"此條文規定了哪些法律要件？",
                f"此條文的適用範圍是什麼？"
            ][:count]
        else:  # outgoing
            return [
                f"違反此條文會有什麼法律後果？",
                f"此條文與其他相關法條有什麼關係？",
                f"此條文在實務中如何應用？",
                f"此條文的立法目的是什麼？"
            ][:count]
    
    def _extract_questions_from_text(self, text: str) -> List[str]:
        """從文本中提取問題"""
        questions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if ('？' in line or '?' in line) and len(line) > 10:
                # 清理行內容
                line = line.replace('- ', '').replace('* ', '').replace('1. ', '').replace('2. ', '').replace('3. ', '')
                questions.append(line)
        
        # 根據配置決定是否限制數量
        if self.config.use_dynamic_question_count:
            # 動態模式：不限制數量，讓LLM決定
            return questions
        else:
            # 固定模式：使用配置的數量限制
            return questions[:self.config.max_pseudo_queries_per_node]
    
    def _generate_default_incoming_questions(self, node: LegalNode) -> List[str]:
        """生成默認內向問題"""
        article_num = node.article_number
        
        return [
            f"{article_num}的主要內容是什麼？",
            f"{article_num}規定了哪些法律要件？",
            f"根據{article_num}，相關的定義是什麼？",
            f"{article_num}的適用範圍是什麼？",
            f"{article_num}規定了什麼法律後果？"
        ][:self.config.max_pseudo_queries_per_node]
    
    def _generate_default_outgoing_questions(self, node: LegalNode) -> List[str]:
        """生成默認外向問題"""
        article_num = node.article_number
        
        return [
            f"違反{article_num}會有什麼法律後果？",
            f"如何申請{article_num}規定的權利？",
            f"{article_num}與其他法條有什麼關聯？",
            f"在什麼情況下適用{article_num}？",
            f"{article_num}的實務操作程序是什麼？"
        ][:self.config.max_pseudo_queries_per_node]
    
    def _extract_keywords_from_query(self, query_content: str) -> Set[str]:
        """從查詢內容中提取關鍵詞"""
        # 法律領域停用詞
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一個', '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有', '看', '好', '自己', '這',
            '什麼', '如何', '為什麼', '哪些', '什麼時候', '哪裡', '多少', '怎麼', '是否', '能否', '可以', '應該', '必須', '需要', '要求', '規定', '條文', '法律', '法條', '條款', '項目', '內容', '定義', '適用', '範圍', '後果', '程序', '權利', '義務', '責任', '處罰', '罰則', '違反', '申請', '實務', '操作', '關聯', '情況', '條件', '標準', '原則', '方法', '方式', '流程', '步驟', '過程', '結果', '效果', '影響', '意義', '作用', '功能', '特點', '性質', '類型', '種類', '分類', '區別', '差異', '相同', '不同', '類似', '相關', '無關', '重要', '主要', '基本', '核心', '關鍵', '必要', '充分', '有效', '無效', '合法', '非法', '正當', '不正當', '合理', '不合理', '適當', '不適當', '正確', '錯誤', '準確', '不準確', '完整', '不完整', '清楚', '不清楚', '明確', '不明確', '具體', '抽象', '詳細', '簡略', '全面', '片面', '客觀', '主觀', '公正', '不公正', '公平', '不公平', '平等', '不平等', '自由', '限制', '保護', '保障', '維護', '促進', '發展', '改善', '提高', '降低', '增加', '減少', '擴大', '縮小', '加強', '削弱', '完善', '改進', '優化', '調整', '修改', '變更', '更新', '升級', '降級', '提升', '下降', '上升', '下降', '增長', '下降', '上升', '下降', '提高', '降低', '增加', '減少', '擴大', '縮小', '加強', '削弱', '完善', '改進', '優化', '調整', '修改', '變更', '更新', '升級', '降級', '提升', '下降', '上升', '下降', '增長', '下降', '上升', '下降'
        }
        
        # 1. 分詞
        words = jieba.lcut(query_content)
        
        # 2. 過濾停用詞和短詞
        keywords = set()
        for word in words:
            word = word.strip()
            if (len(word) >= 2 and 
                word not in stop_words and 
                not re.match(r'^[0-9]+$', word) and
                not re.match(r'^[a-zA-Z]+$', word) and
                not re.match(r'^[^\u4e00-\u9fff]+$', word)):  # 只保留中文字符
                keywords.add(word)
        
        return keywords

class EdgeConnector:
    """邊連接器"""
    
    def __init__(self, config: HopRAGConfig = DEFAULT_CONFIG):
        self.config = config
    
    async def connect_edges(self, nodes: Dict[str, LegalNode], embedding_model) -> Dict[str, List[Dict[str, Any]]]:
        """連接節點邊 - 支持動態邊數限制"""
        print("🔗 開始邊匹配和連接...")
        
        # 計算動態邊數限制（基於論文O(n log n)要求）
        n = len(nodes)
        if self.config.use_dynamic_edge_limit:
            dynamic_limit = int(self.config.edge_limit_factor * n * np.log2(n + 1))
            print(f"📊 動態邊數限制: O(n log n) = {dynamic_limit} (n={n})")
        else:
            dynamic_limit = self.config.max_edges_per_node * n
            print(f"📊 固定邊數限制: {dynamic_limit}")
        
        # Step 1: 為所有偽查詢生成embedding
        await self._generate_pseudo_query_embeddings(nodes, embedding_model)
        
        # Step 2: 執行邊匹配算法
        edges = await self._perform_edge_matching(nodes)
        
        # Step 3: 應用邊數限制
        edges = self._apply_edge_limit(edges, dynamic_limit)
        
        # 統計混合檢索信息
        total_edges = sum(len(edge_list) for edge_list in edges.values())
        print(f"✅ 邊連接完成，共建立 {total_edges} 條邊")
        
        if self.config.use_hybrid_retrieval:
            print(f"🔗 混合檢索統計:")
            print(f"   - Jaccard權重: {self.config.jaccard_weight}")
            print(f"   - 餘弦權重: {self.config.cosine_weight}")
            print(f"   - 詞彙閾值: {self.config.lexical_threshold}")
            print(f"   - 語義閾值: {self.config.semantic_threshold}")
        
        return edges
    
    def _apply_edge_limit(self, edges: Dict[str, List[Dict[str, Any]]], total_limit: int) -> Dict[str, List[Dict[str, Any]]]:
        """應用邊數限制 - 基於論文O(n log n)要求"""
        current_total = sum(len(edge_list) for edge_list in edges.values())
        
        if current_total <= total_limit:
            print(f"✅ 邊數未超過限制: {current_total} <= {total_limit}")
            return edges
        
        print(f"⚠️ 邊數超過限制: {current_total} > {total_limit}，開始修剪...")
        
        # 收集所有邊並按相似度排序
        all_edges = []
        for node_id, edge_list in edges.items():
            for edge in edge_list:
                edge['source_node'] = node_id
                all_edges.append(edge)
        
        # 按相似度分數排序（降序）
        all_edges.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # 保留前total_limit條邊
        selected_edges = all_edges[:total_limit]
        
        # 重新組織邊字典
        limited_edges = {}
        for edge in selected_edges:
            source_node = edge.pop('source_node')
            if source_node not in limited_edges:
                limited_edges[source_node] = []
            limited_edges[source_node].append(edge)
        
        # 確保所有節點都有邊列表（即使為空）
        for node_id in edges.keys():
            if node_id not in limited_edges:
                limited_edges[node_id] = []
        
        print(f"✅ 邊數修剪完成: {current_total} -> {sum(len(edge_list) for edge_list in limited_edges.values())}")
        
        return limited_edges
    
    async def _generate_pseudo_query_embeddings(self, nodes: Dict[str, LegalNode], embedding_model):
        """為所有偽查詢生成embedding"""
        print("📊 開始生成偽查詢embedding向量...")
        
        all_queries = []
        query_mapping = {}
        
        # 收集所有偽查詢
        for node_id, node in nodes.items():
            for pseudo_query in node.pseudo_queries.get("outgoing", []):
                all_queries.append(pseudo_query.content)
                query_mapping[pseudo_query.content] = (node_id, "outgoing", pseudo_query)
                
            for pseudo_query in node.pseudo_queries.get("incoming", []):
                all_queries.append(pseudo_query.content)
                query_mapping[pseudo_query.content] = (node_id, "incoming", pseudo_query)
        
        if not all_queries:
            print("⚠️ 沒有偽查詢需要生成embedding")
            return
        
        print(f"📈 總共需要生成 {len(all_queries)} 個embedding向量")
        print(f"⏱️ 預計需要 1-2 分鐘")
        
        # 批量生成embedding
        try:
            start_time = time.time()
            if hasattr(embedding_model, 'encode_async'):
                embeddings = await embedding_model.encode_async(all_queries)
            else:
                embeddings = embedding_model.encode(all_queries)
            
            embedding_time = time.time() - start_time
            print(f"✅ Embedding生成完成！耗時: {embedding_time:.1f} 秒")
            
            # 將embedding分配回偽查詢
            for i, query_content in enumerate(all_queries):
                node_id, query_type, pseudo_query = query_mapping[query_content]
                pseudo_query.embedding = embeddings[i]
                
        except Exception as e:
            print(f"❌ 生成偽查詢embedding失敗: {e}")
    
    async def _perform_edge_matching(self, nodes: Dict[str, LegalNode]) -> Dict[str, List[Dict[str, Any]]]:
        """執行邊匹配算法"""
        print("🔗 執行邊匹配算法...")
        
        node_ids = list(nodes.keys())
        edges = {}
        
        for i, node_a in enumerate(node_ids):
            outgoing_queries_a = nodes[node_a].pseudo_queries.get("outgoing", [])
            edges[node_a] = []
            
            for j, node_b in enumerate(node_ids):
                if i == j:
                    continue
                    
                incoming_queries_b = nodes[node_b].pseudo_queries.get("incoming", [])
                
                # 計算最佳相似度
                best_similarity = 0
                best_outgoing_query = None
                best_incoming_query = None
                
                for out_query in outgoing_queries_a:
                    for in_query in incoming_queries_b:
                        if (out_query.embedding is not None and in_query.embedding is not None and
                            out_query.keywords and in_query.keywords):
                            
                            # 計算各種相似度
                            jaccard_sim = self._calculate_jaccard_similarity(out_query.keywords, in_query.keywords)
                            cosine_sim = self._calculate_cosine_similarity(out_query.embedding, in_query.embedding)
                            
                            # 混合檢索過濾：檢查詞彙和語義閾值
                            if self.config.use_hybrid_retrieval:
                                if (jaccard_sim < self.config.lexical_threshold or 
                                    cosine_sim < self.config.semantic_threshold):
                                    continue  # 跳過不滿足閾值的組合
                            
                            similarity = self._calculate_similarity(out_query, in_query)
                            
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_outgoing_query = out_query
                                best_incoming_query = in_query
                
                # 如果相似度超過閾值，建立邊
                if best_similarity >= self.config.similarity_threshold:
                    # 創建完整的邊特徵
                    edge_feature = self._create_edge_feature(best_outgoing_query, best_incoming_query)
                    
                    edge_attr = {
                        'to_node': node_b,
                        'pseudo_query': f"{best_outgoing_query.content} -> {best_incoming_query.content}",
                        'similarity_score': best_similarity,
                        'jaccard_score': edge_feature.jaccard_score,
                        'cosine_score': edge_feature.cosine_score,
                        'mixed_score': edge_feature.mixed_score,
                        'combined_keywords': list(edge_feature.combined_keywords),
                        'edge_query': edge_feature.query,
                        'edge_embedding': edge_feature.embedding.tolist() if edge_feature.embedding is not None else None,
                        'outgoing_query_id': best_outgoing_query.query_id,
                        'incoming_query_id': best_incoming_query.query_id,
                        'edge_type': self._determine_edge_type(node_a, node_b)
                    }
                    
                    edges[node_a].append(edge_attr)
                    
                    # 限制每個節點的最大出邊數量
                    if len(edges[node_a]) >= self.config.max_edges_per_node:
                        break
            
            if i % 10 == 0:
                print(f"  處理進度: {i+1}/{len(node_ids)} 節點")
        
        return edges
    
    def _calculate_similarity(self, query_a: PseudoQuery, query_b: PseudoQuery) -> float:
        """計算兩個偽查詢的混合相似度（Jaccard + 餘弦相似度）"""
        # 1. 計算Jaccard相似度（詞彙層面）
        jaccard_sim = self._calculate_jaccard_similarity(query_a.keywords, query_b.keywords)
        
        # 2. 計算餘弦相似度（語義層面）
        cosine_sim = self._calculate_cosine_similarity(query_a.embedding, query_b.embedding)
        
        # 3. 混合檢索：根據配置權重計算
        if self.config.use_hybrid_retrieval:
            # 使用配置的權重
            mixed_similarity = (jaccard_sim * self.config.jaccard_weight + 
                              cosine_sim * self.config.cosine_weight)
        else:
            # 使用論文公式：兩者平均值
            mixed_similarity = (jaccard_sim + cosine_sim) / 2
        
        return float(mixed_similarity)
    
    def _calculate_jaccard_similarity(self, keywords_a: Set[str], keywords_b: Set[str]) -> float:
        """計算Jaccard相似度"""
        if not keywords_a or not keywords_b:
            return 0.0
        
        intersection = len(keywords_a.intersection(keywords_b))
        union = len(keywords_a.union(keywords_b))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_cosine_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """計算餘弦相似度"""
        if embedding_a is None or embedding_b is None:
            return 0.0
        
        dot_product = np.dot(embedding_a, embedding_b)
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _create_edge_feature(self, outgoing_query: PseudoQuery, incoming_query: PseudoQuery) -> EdgeFeature:
        """創建邊特徵 - 按照論文要求 e_s,t* = (q_t*,j*^-, k_t*,j*^- ∪ k_s,i^+, v_t*,j*^-)"""
        
        # 合併關鍵詞集合：k_t*,j*^- ∪ k_s,i^+
        combined_keywords = incoming_query.keywords.union(outgoing_query.keywords)
        
        # 計算各種相似度分數
        jaccard_score = self._calculate_jaccard_similarity(
            outgoing_query.keywords, incoming_query.keywords
        )
        cosine_score = self._calculate_cosine_similarity(
            outgoing_query.embedding, incoming_query.embedding
        )
        mixed_score = (jaccard_score + cosine_score) / 2
        
        return EdgeFeature(
            query=incoming_query.content,  # q_t*,j*^-
            combined_keywords=combined_keywords,  # k_t*,j*^- ∪ k_s,i^+
            embedding=incoming_query.embedding,  # v_t*,j*^-
            jaccard_score=jaccard_score,
            cosine_score=cosine_score,
            mixed_score=mixed_score
        )
    
    def _determine_edge_type(self, from_node: str, to_node: str) -> EdgeType:
        """確定邊的類型"""
        # 這裡需要根據實際的節點類型來判斷
        # 暫時使用字符串類型
        if from_node.startswith('article_') and to_node.startswith('article_'):
            return EdgeType.ARTICLE_TO_ARTICLE
        elif from_node.startswith('article_') and to_node.startswith('item_'):
            return EdgeType.ARTICLE_TO_ITEM
        elif from_node.startswith('item_') and to_node.startswith('article_'):
            return EdgeType.ITEM_TO_ARTICLE
        else:
            return EdgeType.ITEM_TO_ITEM

class PassageGraphBuilder:
    """段落圖構建器"""
    
    def __init__(self, config: HopRAGConfig = DEFAULT_CONFIG):
        self.config = config
        self.pseudo_query_generator = None
        self.edge_connector = None
        self.embedding_model = None
        
    def set_components(self, pseudo_query_generator: PseudoQueryGenerator, 
                      edge_connector: EdgeConnector, embedding_model):
        """設置組件"""
        self.pseudo_query_generator = pseudo_query_generator
        self.edge_connector = edge_connector
        self.embedding_model = embedding_model
        
    async def build_graph(self, multi_level_chunks: Dict[str, Dict[str, List[Dict]]]) -> Tuple[Dict[str, LegalNode], Dict[str, List[Dict[str, Any]]]]:
        """構建完整的知識圖譜"""
        print("🏗️ 開始構建HopRAG知識圖譜...")
        
        # Step 1: 創建節點
        nodes = await self._create_nodes_from_chunks(multi_level_chunks)
        
        # Step 2: 生成偽查詢
        await self._generate_pseudo_queries(nodes)
        
        # Step 3: 構建圖邊
        edges = await self.edge_connector.connect_edges(nodes, self.embedding_model)
        
        print("✅ HopRAG知識圖譜構建完成！")
        return nodes, edges
    
    async def _create_nodes_from_chunks(self, multi_level_chunks: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, LegalNode]:
        """從chunks創建節點"""
        print("📝 創建節點...")
        
        nodes = {}
        
        for doc_id, levels in multi_level_chunks.items():
            # 處理條級節點 (basic_unit層次)
            if 'basic_unit' in levels:
                for chunk_idx, chunk in enumerate(levels['basic_unit']):
                    try:
                        # 確保chunk結構完整
                        if not isinstance(chunk, dict):
                            print(f"❌ Chunk {chunk_idx} 不是字典類型: {type(chunk)}")
                            continue
                        
                        # 確保metadata存在
                        if 'metadata' not in chunk:
                            chunk['metadata'] = {}
                        
                        # 確保content存在
                        if 'content' not in chunk:
                            print(f"❌ Chunk {chunk_idx} 沒有content欄位")
                            continue
                        
                        # 生成node_id，如果metadata中沒有id欄位
                        node_id = chunk['metadata'].get('id', f"{doc_id}_basic_unit_{chunk_idx}")
                        
                        article_node = LegalNode(
                            node_id=node_id,
                            node_type=NodeType.ARTICLE,
                            content=chunk['content'],
                            contextualized_text=chunk['content'],
                            law_name=chunk['metadata'].get('law_name', ''),
                            article_number=chunk['metadata'].get('article_label', ''),
                            metadata=chunk['metadata']
                        )
                        nodes[article_node.node_id] = article_node
                        
                    except Exception as e:
                        print(f"❌ 創建basic_unit節點失敗 (chunk {chunk_idx}): {e}")
                        continue
            
            # 處理項級節點 (basic_unit_component層次)
            if 'basic_unit_component' in levels:
                for chunk_idx, chunk in enumerate(levels['basic_unit_component']):
                    try:
                        # 確保chunk結構完整
                        if not isinstance(chunk, dict):
                            print(f"❌ Component Chunk {chunk_idx} 不是字典類型: {type(chunk)}")
                            continue
                        
                        # 確保metadata存在
                        if 'metadata' not in chunk:
                            chunk['metadata'] = {}
                        
                        # 確保content存在
                        if 'content' not in chunk:
                            print(f"❌ Component Chunk {chunk_idx} 沒有content欄位")
                            continue
                        
                        # 生成node_id，如果metadata中沒有id欄位
                        node_id = chunk['metadata'].get('id', f"{doc_id}_basic_unit_component_{chunk_idx}")
                        
                        item_node = LegalNode(
                            node_id=node_id,
                            node_type=NodeType.ITEM,
                            content=chunk['content'],
                            contextualized_text=chunk['content'],
                            law_name=chunk['metadata'].get('law_name', ''),
                            article_number=chunk['metadata'].get('article_label', ''),
                            item_number=chunk['metadata'].get('item_label', ''),
                            parent_article_id=chunk['metadata'].get('parent_article_id'),
                            metadata=chunk['metadata']
                        )
                        nodes[item_node.node_id] = item_node
                        
                    except Exception as e:
                        print(f"❌ 創建basic_unit_component節點失敗 (chunk {chunk_idx}): {e}")
                        continue
        
        print(f"✅ 節點創建完成，共 {len(nodes)} 個節點")
        return nodes
    
    async def _generate_pseudo_queries(self, nodes: Dict[str, LegalNode]):
        """為所有節點生成偽查詢"""
        print("🤖 開始生成偽查詢...")
        
        node_list = list(nodes.values())
        total_nodes = len(node_list)
        start_time = time.time()
        
        print(f"📊 總共需要處理 {total_nodes} 個節點")
        print(f"⏱️ 預計每個節點需要 2-3 秒（包含LLM調用）")
        print(f"🕐 預計總時間: {total_nodes * 2.5 / 60:.1f} 分鐘")
        print("=" * 60)
        
        for i, node in enumerate(node_list):
            try:
                node_start_time = time.time()
                await self.pseudo_query_generator.generate_pseudo_queries_for_node(node)
                node_time = time.time() - node_start_time
                
                # 計算進度和剩餘時間
                progress = (i + 1) / total_nodes * 100
                elapsed_time = time.time() - start_time
                avg_time_per_node = elapsed_time / (i + 1)
                remaining_nodes = total_nodes - (i + 1)
                estimated_remaining_time = remaining_nodes * avg_time_per_node
                
                # 每5個節點顯示一次進度
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"📈 進度: {i+1}/{total_nodes} ({progress:.1f}%) | "
                          f"節點: {node.node_id[:20]}... | "
                          f"耗時: {node_time:.1f}s | "
                          f"剩餘: {estimated_remaining_time/60:.1f}分鐘")
                
                # 每20個節點顯示詳細統計
                if (i + 1) % 20 == 0:
                    print(f"📊 統計: 平均 {avg_time_per_node:.1f}s/節點 | "
                          f"已用時: {elapsed_time/60:.1f}分鐘 | "
                          f"預計完成: {estimated_remaining_time/60:.1f}分鐘")
                    
            except Exception as e:
                print(f"❌ 節點 {node.node_id} 偽查詢生成失敗: {e}")
                continue
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"✅ 偽查詢生成完成！總耗時: {total_time/60:.1f} 分鐘")
        print(f"📊 平均每個節點: {total_time/total_nodes:.1f} 秒")
