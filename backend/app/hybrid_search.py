"""
Hybrid-RAG（向量檢索 + 規則關鍵字加分）

使用方法（模組級）：
  - 調用 hybrid_rank(query, nodes, k=10, config=..., vector_scores=...) 返回加權後的排序結果
  - nodes: List[Dict]，每個節點需含 content 與 metadata，建議 metadata 來自正規化後的法律JSON：
    {
      'content': '...',
      'metadata': {
        'id': '...',
        'category': '著作權法',
        'article_label': '第10條之1',
        'article_number': 10,
        'article_suffix': 1,
        'item_number': 1,              # 可選
        'clause_type': '目'|'款',        # 可選
        'clause_number': 1,            # 可選
      }
    }

策略：
  1) 先用「密集向量」在外部算好每個節點的 vector_scores（由主流程提供）
  2) 再用 metadata 關鍵字規則加分：法名對齊、條號對齊、法律術語/同義詞命中
  3) 最終分數 = alpha * vector_score + bonus（加分上限可配置）
"""

from typing import List, Dict, Any, Tuple
import re
from dataclasses import dataclass

# 不再在此模組內做內容相似度計算（改由主流程提供 vector_scores）


def _normalize_nfkc(text: str) -> str:
    import unicodedata
    return unicodedata.normalize('NFKC', text or '')


def _extract_article_tuple(label_or_text: str) -> Tuple[int | None, int | None]:
    t = _normalize_nfkc(label_or_text)
    m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)條之([0-9一二兩三四五六七八九十百千〇零]+)", t)
    if m:
        return (_to_int_cn(m.group(1)), _to_int_cn(m.group(2)))
    m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)-([0-9一二兩三四五六七八九十百千〇零]+)條", t)
    if m:
        return (_to_int_cn(m.group(1)), _to_int_cn(m.group(2)))
    m = re.search(r"第([0-9一二兩三四五六七八九十百千〇零]+)條", t)
    if m:
        return (_to_int_cn(m.group(1)), None)
    return (None, None)


def _to_int_cn(s: str | None) -> int | None:
    if s is None:
        return None
    s = _normalize_nfkc(str(s))
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


# 法律專用術語同義詞與正規化規則（可擴充）
LEGAL_SYNONYMS: Dict[str, List[str]] = {
    # canonical: variants
    '公開傳輸': ['公開傳輸', '數位傳輸', '網路傳輸', '線上傳輸', '上線提供', 'public transmission'],
    '公開播送': ['公開播送', '廣播', '播送', 'broadcast'],
    '重製': ['重製', '複製', '拷貝', '複本製作', 'reproduction'],
    '散布': ['散布', '發行', '流通', 'distribution'],
    '改作': ['改作', '改編', '翻案', '衍生創作', 'derivative'],
    '引用': ['引用', '節錄', '摘錄', '引用他人著作'],
    '合理使用': ['合理使用', '公平使用', 'fair use'],
}


@dataclass
class HybridConfig:
    alpha: float = 0.8                  # 內容相似度權重
    w_law_match: float = 0.15           # 法名對齊
    w_article_match: float = 0.15       # 條/之N 對齊
    w_keyword_hit: float = 0.05         # 每個術語同義命中加分
    max_bonus: float = 0.4              # 加分上限


def _extract_query_features(query: str) -> Dict[str, Any]:
    t = _normalize_nfkc(query)
    law = '著作權法' if '著作權法' in t else ('商標法' if '商標法' in t else ('專利法' if '專利法' in t else ''))
    art, suf = _extract_article_tuple(t)
    # 術語抽取：只要 query 包含同義任一詞，就記錄 canonical key
    terms = []
    for canonical, variants in LEGAL_SYNONYMS.items():
        if any(v in t for v in variants):
            terms.append(canonical)
    return {
        'law': law,
        'article_number': art,
        'article_suffix': suf,
        'terms': terms,
    }


def _metadata_bonus(md: Dict[str, Any], qf: Dict[str, Any], cfg: HybridConfig) -> float:
    bonus = 0.0
    # 法名
    if qf['law'] and (md.get('category') == qf['law'] or qf['law'] in (md.get('id') or '')):
        bonus += cfg.w_law_match
    # 條/之N
    a_num = md.get('article_number')
    a_suf = md.get('article_suffix')
    if qf['article_number'] is not None and a_num == qf['article_number']:
        # 條一致
        bonus += cfg.w_article_match * (1.0 if (qf['article_suffix'] is None or a_suf == qf['article_suffix']) else 0.5)
    # 術語命中（在 metadata.id 或 article_label）
    text_meta = _normalize_nfkc((md.get('id') or '') + ' ' + (md.get('article_label') or ''))
    for canonical in qf['terms']:
        variants = LEGAL_SYNONYMS.get(canonical, [])
        if any(v in text_meta for v in variants):
            bonus += cfg.w_keyword_hit
    
    # 新增：內容結構加分（即使沒有metadata也能工作）
    content = md.get('content', '')
    if content:
        import re
        # 法條結構加分
        if re.search(r'第\s*\d+\s*條', content):
            bonus += cfg.w_article_match * 0.3
        
        # 法律關鍵詞密度加分
        legal_terms = ['權利', '義務', '禁止', '處罰', '規定', '適用', '違反', '侵害', '著作權', '商標', '專利']
        term_count = sum(1 for term in legal_terms if term in content)
        if term_count > 0:
            bonus += min(term_count * cfg.w_keyword_hit * 0.2, cfg.max_bonus * 0.3)
        
        # 查詢關鍵詞在內容中的匹配
        for canonical in qf['terms']:
            if canonical in content:
                bonus += cfg.w_keyword_hit * 0.5
            # 檢查同義詞
            variants = LEGAL_SYNONYMS.get(canonical, [])
            for variant in variants:
                if variant in content:
                    bonus += cfg.w_keyword_hit * 0.3
    
    return min(bonus, cfg.max_bonus)


def hybrid_rank(
    query: str,
    nodes: List[Dict[str, Any]],
    k: int = 10,
    config: HybridConfig | None = None,
    vector_scores: List[float] | None = None,
) -> List[Dict[str, Any]]:
    """對 nodes 進行 Hybrid 排序並返回前 k 個節點（附加分數）
    nodes: [{'content': str, 'metadata': {...}}, ...]
    返回 [{'content':..., 'metadata':..., 'score': float, 'vector_score': float, 'bonus': float}, ...]
    """
    if not nodes:
        return []
    cfg = config or HybridConfig()

    # 1) 內容相似度：改用主流程提供的密集向量分數
    if vector_scores is None or len(vector_scores) != len(nodes):
        sims = [0.0] * len(nodes)
    else:
        sims = vector_scores

    # 2) metadata 規則加分
    qf = _extract_query_features(query)
    ranked: List[Tuple[int, float, float]] = []  # (idx, vector, bonus)
    for i, n in enumerate(nodes):
        md = n.get('metadata') or {}
        bonus = _metadata_bonus(md, qf, cfg)
        ranked.append((i, float(sims[i]), float(bonus)))

    # 3) 線性組合
    scored: List[Tuple[int, float, float, float]] = []  # (idx, final, vec, bonus)
    for i, v, b in ranked:
        final = cfg.alpha * v + b
        scored.append((i, final, v, b))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    results: List[Dict[str, Any]] = []
    for i, final, v, b in top:
        item = dict(nodes[i])
        item['score'] = final
        item['vector_score'] = v
        item['bonus'] = b
        results.append(item)
    return results


