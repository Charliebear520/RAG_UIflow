#!/usr/bin/env python3
"""
RQ4 評估腳本
評估實驗組E（LLM章節導向）相較於對照組D（完整多層次）和對照組C（條文+細節層次）的效益提升

使用方法：
    python evaluate_rq4.py
"""

import json
import requests
import sys
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import time

# API配置
API_BASE_URL = "http://localhost:8000/api"
GROUPS_TO_EVALUATE = ["group_c", "group_d", "group_e"]
GROUP_LABELS = {
    "group_c": "對照組2（C組：條文+細節層次）",
    "group_d": "對照組1（D組：完整多層次）",
    "group_e": "實驗組（E組：LLM章節導向）",
}

def load_ground_truth(file_path: str = "QA/ground_truth.json") -> List[Dict]:
    """載入ground truth數據"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_available_doc_id() -> str:
    """獲取可用的doc_id（用於group_e）
    
    為了確保與 chapter_summaries.json 完全一致，這裡固定使用已知的 doc_id。
    如需切換到其他法規，只需修改這個常數即可。
    """
    known_correct_doc_id = "71c23286-d3fb-4982-a2e2-33a931687c5d"
    print(f"   使用固定 doc_id（來自 chapter_summaries.json）: {known_correct_doc_id}")
    return known_correct_doc_id

def extract_chunk_identifiers(result: Dict) -> Set[str]:
    """
    從檢索結果中提取chunk標識符，用於匹配ground_truth
    
    支持的格式：
    - Art_11, Art_22, Art_28_1, Art_29_P1, Art_87_P1_C8 等
    - Chap_3, Chap_3_Sec_2, Chap_3_Sec_4 等
    """
    import re
    identifiers = set()
    
    # 方法1：從chunk_id直接提取（最優先）
    chunk_id = result.get("chunk_id", "")
    if chunk_id:
        # 檢查是否符合Art_或Chap_格式
        if chunk_id.startswith("Art_") or chunk_id.startswith("Chap_"):
            # 清理可能的額外字符
            clean_id = chunk_id.strip()
            identifiers.add(clean_id)
            # 如果是Art_格式，也添加基礎條文號（如Art_28_1 -> Art_28）
            if clean_id.startswith("Art_"):
                base_match = re.match(r'Art_(\d+)', clean_id)
                if base_match:
                    identifiers.add(f"Art_{base_match.group(1)}")
    
    # 方法2：從enhanced_metadata中提取（優先於普通metadata）
    enhanced_metadata = result.get("enhanced_metadata", {}) or {}
    metadata = result.get("metadata", {}) or {}
    
    # 合併metadata，enhanced_metadata優先
    all_metadata = {**metadata, **enhanced_metadata}
    
    # 從metadata中提取id字段（如果存在）
    metadata_id = all_metadata.get("id", "")
    if metadata_id and (metadata_id.startswith("Art_") or metadata_id.startswith("Chap_")):
        identifiers.add(metadata_id)
    
    # 提取章節信息
    chapter = all_metadata.get("chapter", "") or ""
    section = all_metadata.get("section", "") or ""
    article = all_metadata.get("article", "") or ""
    
    # 構建Chap_標識符
    if chapter:
        chapter_match = re.search(r'第\s*([0-9一二三四五六七八九十百千〇零]+)\s*章', chapter)
        if chapter_match:
            chapter_num = _cn_to_int_str(chapter_match.group(1))
            if chapter_num:
                if section:
                    section_match = re.search(r'第\s*([0-9一二三四五六七八九十百千〇零]+)\s*節', section)
                    if section_match:
                        section_num = _cn_to_int_str(section_match.group(1))
                        if section_num:
                            identifiers.add(f"Chap_{chapter_num}_Sec_{section_num}")
                identifiers.add(f"Chap_{chapter_num}")
    
    # 構建Art_標識符
    if article:
        # 優先匹配「第X-Y條」格式（用「-」代表「之」）
        art_match = re.search(
            r'第\s*([0-9一二三四五六七八九十百千〇零]+)\s*[-]\s*([0-9一二三四五六七八九十百千〇零]+)\s*條',
            article
        )
        if art_match:
            main_num = _cn_to_int_str(art_match.group(1))
            suffix_num = _cn_to_int_str(art_match.group(2))
            if main_num and suffix_num:
                identifiers.add(f"Art_{main_num}_{suffix_num}")
                identifiers.add(f"Art_{main_num}")  # 也添加基礎條文號
        else:
            # 匹配「第X條」或「第X條之Y」格式
            art_match = re.search(
                r'第\s*([0-9一二三四五六七八九十百千〇零]+)\s*條(?:之\s*([0-9一二三四五六七八九十百千〇零]+))?',
                article
            )
            if art_match:
                main_num = _cn_to_int_str(art_match.group(1))
                suffix_num = _cn_to_int_str(art_match.group(2)) if art_match.group(2) else None
                
                if main_num:
                    # 檢查是否有段落、項、款、目信息
                    paragraph = all_metadata.get("paragraph", "") or ""
                    subparagraph = all_metadata.get("subparagraph", "") or ""
                    item = all_metadata.get("item", "") or ""
                    
                    # 提取段落編號（項）
                    para_num = None
                    if paragraph:
                        para_match = re.search(r'第\s*([0-9一二三四五六七八九十百千〇零]+)\s*項', paragraph)
                        if para_match:
                            para_num = _cn_to_int_str(para_match.group(1))
                    
                    # 提取項編號（款）
                    subpara_num = None
                    if subparagraph:
                        subpara_match = re.search(r'第\s*([0-9一二三四五六七八九十百千〇零]+)\s*款', subparagraph)
                        if subpara_match:
                            subpara_num = _cn_to_int_str(subpara_match.group(1))
                    
                    # 提取目編號
                    item_num = None
                    if item:
                        item_match = re.search(r'第\s*([0-9一二三四五六七八九十百千〇零]+)\s*目', item)
                        if item_match:
                            item_num = _cn_to_int_str(item_match.group(1))
                    
                    # 構建完整的Art_標識符
                    if suffix_num:
                        art_id = f"Art_{main_num}_{suffix_num}"
                        identifiers.add(art_id)
                        identifiers.add(f"Art_{main_num}")  # 也添加基礎條文號
                    elif para_num:
                        art_id = f"Art_{main_num}_P{para_num}"
                        identifiers.add(art_id)
                        identifiers.add(f"Art_{main_num}")  # 也添加基礎條文號
                        if subpara_num:
                            art_id = f"Art_{main_num}_P{para_num}_C{subpara_num}"
                            identifiers.add(art_id)
                            if item_num:
                                art_id = f"Art_{main_num}_P{para_num}_C{subpara_num}_I{item_num}"
                                identifiers.add(art_id)
                    else:
                        identifiers.add(f"Art_{main_num}")
    
    # 方法3：從hierarchical_description中提取
    hierarchical_desc = result.get("hierarchical_description", "")
    if hierarchical_desc:
        # 匹配Art_或Chap_開頭的標識符
        matches = re.findall(r'(Art_|Chap_)[A-Za-z0-9_]+', hierarchical_desc)
        for match in matches:
            if match.startswith("Art_") or match.startswith("Chap_"):
                identifiers.add(match)
                # 如果是Art_格式，也添加基礎條文號
                if match.startswith("Art_"):
                    base_match = re.match(r'Art_(\d+)', match)
                    if base_match:
                        identifiers.add(f"Art_{base_match.group(1)}")
    
    # 方法4：從content中提取（作為備選，只提取基礎條文號）
    if not identifiers:  # 只有在前面方法都沒找到時才使用
        content = result.get("content", "")
        if content:
            # 匹配「第X條」並嘗試構建Art_標識符
            art_matches = re.findall(r'第\s*([0-9一二三四五六七八九十百千〇零]+)\s*條', content[:500])
            for match in art_matches[:3]:  # 最多取前3個匹配
                art_num = _cn_to_int_str(match)
                if art_num:
                    identifiers.add(f"Art_{art_num}")
    
    return identifiers

def _cn_to_int_str(cn_str: str) -> str:
    """將中文數字轉換為阿拉伯數字字符串"""
    if not cn_str:
        return ""
    
    # 如果已經是數字，直接返回
    if cn_str.isdigit():
        return cn_str
    
    # 中文數字映射
    cn_digit_map = {
        '零': '0', '〇': '0', '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        '十': '10', '百': '100', '千': '1000'
    }
    
    # 簡單轉換：如果包含中文數字，嘗試轉換
    result = ""
    for char in cn_str:
        if char in cn_digit_map:
            result += cn_digit_map[char]
        elif char.isdigit():
            result += char
    
    # 如果轉換失敗，嘗試直接匹配常見的中文數字
    cn_num_map = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
        '十一': '11', '十二': '12', '十三': '13', '十四': '14', '十五': '15',
        '十六': '16', '十七': '17', '十八': '18', '十九': '19', '二十': '20'
    }
    
    if cn_str in cn_num_map:
        return cn_num_map[cn_str]
    
    return result if result else cn_str

def retrieve_experimental_groups(query: str, k: int, groups: List[str], doc_id: str = None) -> Dict[str, Any]:
    """一次性檢索多個實驗組"""
    try:
        payload = {
            "query": query,
            "k": k,
            "groups_to_test": groups
        }
        # 如果包含group_e，必須提供doc_id
        if "group_e" in groups:
            if not doc_id:
                print(f"⚠️ group_e需要doc_id，但未提供，跳過group_e檢索")
                # 從groups中移除group_e
                groups = [g for g in groups if g != "group_e"]
                payload["groups_to_test"] = groups
            else:
                payload["doc_id"] = doc_id
        
        if not groups:
            return {}
        
        response = requests.post(
            f"{API_BASE_URL}/experimental-groups-batch-retrieve",
            json=payload,
            timeout=120  # 增加超時時間，因為group_e需要LLM調用
        )
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", {})
            
            # 檢查group_e的結果，如果有錯誤則打印詳細信息
            if "group_e" in groups and "group_e" in results:
                group_e_data = results.get("group_e", {})
                if group_e_data.get("error"):
                    print(f"    ⚠️ group_e檢索錯誤: {group_e_data.get('error')}")
                elif len(group_e_data.get("fused_results", [])) == 0:
                    # 檢查LLM階段是否成功
                    llm_stage = group_e_data.get("llm_stage", {})
                    if not llm_stage:
                        print(f"    ⚠️ group_e未返回LLM階段信息，可能LLM調用失敗")
                    elif llm_stage.get("fallback_used"):
                        print(f"    ⚠️ group_e使用了fallback模式（LLM選擇過於嚴苛）")
            
            return results
        else:
            error_text = response.text
            print(f"⚠️ 實驗組檢索失敗: {response.status_code}")
            print(f"   錯誤詳情: {error_text[:200]}")
            return {}
    except Exception as e:
        print(f"❌ 實驗組檢索異常: {e}")
        import traceback
        traceback.print_exc()
        return {}

def calculate_metrics(
    retrieved_results: List[Dict],
    ground_truth_e: List[str],
    ground_truth_c: List[str],
    k: int
) -> Dict[str, float]:
    """
    計算檢索指標
    
    Args:
        retrieved_results: 檢索結果列表
        ground_truth_e: 必要（Essential）標註列表
        ground_truth_c: 補充（Complementary）標註列表
        k: Top-K值
    
    Returns:
        包含各項指標的字典
    """
    # 提取Top-K結果的標識符
    top_k_results = retrieved_results[:k]
    retrieved_identifiers = set()
    for result in top_k_results:
        identifiers = extract_chunk_identifiers(result)
        retrieved_identifiers.update(identifiers)
    
    # 轉換ground_truth為集合
    gt_e_set = set(ground_truth_e)
    gt_c_set = set(ground_truth_c)
    
    # 計算嚴格指標（僅考慮E）
    strict_e_retrieved = retrieved_identifiers & gt_e_set
    strict_precision = len(strict_e_retrieved) / k if k > 0 else 0.0
    strict_recall = len(strict_e_retrieved) / len(gt_e_set) if len(gt_e_set) > 0 else 0.0
    strict_f1 = (
        2 * strict_precision * strict_recall / (strict_precision + strict_recall)
        if (strict_precision + strict_recall) > 0 else 0.0
    )
    
    # 計算寬鬆指標（考慮E或C，不去重）
    # 分別計算E和C的命中數量
    relaxed_e_retrieved = retrieved_identifiers & gt_e_set
    relaxed_c_retrieved = retrieved_identifiers & gt_c_set
    # 分子 = E命中數量 + C命中數量（不去重，允許重複計算）
    relaxed_ec_count = len(relaxed_e_retrieved) + len(relaxed_c_retrieved)
    # 分母 = E總數 + C總數（不去重）
    total_ec_count = len(gt_e_set) + len(gt_c_set)
    
    relaxed_precision = relaxed_ec_count / k if k > 0 else 0.0
    relaxed_recall = relaxed_ec_count / total_ec_count if total_ec_count > 0 else 0.0
    relaxed_f1 = (
        2 * relaxed_precision * relaxed_recall / (relaxed_precision + relaxed_recall)
        if (relaxed_precision + relaxed_recall) > 0 else 0.0
    )
    
    return {
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": strict_f1,
        "relaxed_precision": relaxed_precision,
        "relaxed_recall": relaxed_recall,
        "relaxed_f1": relaxed_f1,
        "retrieved_e_count": len(strict_e_retrieved),
        "retrieved_c_count": len(relaxed_c_retrieved),
        "retrieved_ec_count": relaxed_ec_count,  # E命中數量 + C命中數量（不去重）
        "total_e": len(gt_e_set),
        "total_c": len(gt_c_set),
        "total_ec": total_ec_count,  # E總數 + C總數（不去重）
        "retrieved_identifiers": list(retrieved_identifiers)
    }

def evaluate_rq4():
    """執行RQ4評估"""
    print("=" * 80)
    print("RQ4 評估：實驗組E（LLM章節導向）vs 對照組D（完整多層次）vs 對照組C（條文+細節層次）")
    print("=" * 80)
    
    # 載入ground truth
    print("\n📚 載入ground truth數據...")
    ground_truth_data = load_ground_truth()
    print(f"✅ 載入 {len(ground_truth_data)} 道題目")
    
    # 獲取doc_id（用於group_e）
    print("\n🔍 獲取doc_id（用於group_e檢索）...")
    doc_id = get_available_doc_id()
    print(f"✅ 使用doc_id: {doc_id}")
    
    # 檢查API連接
    print("\n🔌 檢查API連接...")
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api', '')}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API連接正常")
        else:
            print("⚠️ API連接異常，但繼續執行...")
    except Exception as e:
        print(f"❌ API連接失敗: {e}")
        print("請確保後端服務正在運行（http://localhost:8000）")
        sys.exit(1)
    
    # 存儲所有結果
    all_results = []
    k_values = [5, 10]
    
    # 逐題評估
    print("\n" + "=" * 80)
    print("開始逐題評估...")
    print("=" * 80)
    
    for idx, item in enumerate(ground_truth_data, 1):
        query_id = item["query_id"]
        query_text = item["query_text"]
        query_type = item["query_type"]
        ground_truth = item["ground_truth"]
        
        gt_e = ground_truth.get("E", [])
        gt_c = ground_truth.get("C", [])
        
        print(f"\n[{idx}/{len(ground_truth_data)}] {query_id} ({query_type})")
        print(f"查詢: {query_text[:60]}...")
        print(f"Ground Truth - E: {gt_e}, C: {gt_c}")
        
        # 檢索C/D/E組
        print("  🔍 檢索實驗組 (C/D/E)...")
        group_payloads = retrieve_experimental_groups(
            query_text,
            k=max(k_values),
            groups=GROUPS_TO_EVALUATE,
            doc_id=doc_id
        )
        for group in GROUPS_TO_EVALUATE:
            group_data = group_payloads.get(group, {})
            fused = group_data.get("fused_results", [])
            error = group_data.get("error")
            if error:
                print(f"    - {GROUP_LABELS[group]} 檢索失敗: {error}")
            else:
                print(f"    - {GROUP_LABELS[group]} 返回 {len(fused)} 個結果")
                # 如果是group_e，顯示LLM選擇的章節信息
                if group == "group_e" and group_data.get("llm_stage"):
                    llm_stage = group_data.get("llm_stage", {})
                    selection_details = llm_stage.get("selection_details", [])
                    if selection_details:
                        print(f"      LLM選擇了 {len(selection_details)} 個章節")
                        for detail in selection_details[:3]:  # 只顯示前3個
                            print(f"        - {detail.get('chapter_title', 'N/A')}")
                    else:
                        print(f"      ⚠️ LLM未選擇任何章節")
        
        # 計算各K值的指標
        item_result = {
            "query_id": query_id,
            "query_text": query_text,
            "query_type": query_type,
            "ground_truth": ground_truth,
        }
        for group in GROUPS_TO_EVALUATE:
            item_result[group] = {}
        
        for k in k_values:
            summaries = []
            for group in GROUPS_TO_EVALUATE:
                fused_results = group_payloads.get(group, {}).get("fused_results", [])
                metrics = calculate_metrics(fused_results, gt_e, gt_c, k)
                item_result[group][f"k_{k}"] = metrics
                summaries.append(
                    f"    {GROUP_LABELS[group]} - Strict P@{k}: {metrics['strict_precision']:.3f}, "
                    f"R@{k}: {metrics['strict_recall']:.3f}, F1@{k}: {metrics['strict_f1']:.3f}"
                )
                summaries.append(
                    f"                     Relaxed P@{k}: {metrics['relaxed_precision']:.3f}, "
                    f"R@{k}: {metrics['relaxed_recall']:.3f}, F1@{k}: {metrics['relaxed_f1']:.3f}"
                )
            if summaries:
                print(f"  📊 K={k} 指標:")
                for line in summaries:
                    print(line)
        
        all_results.append(item_result)
        
        # 添加延遲，避免API速率限制
        time.sleep(0.5)
    
    # 計算統計結果
    print("\n" + "=" * 80)
    print("計算統計結果...")
    print("=" * 80)
    
    # 按query_type分組
    type_results = defaultdict(list)
    for result in all_results:
        type_results[result["query_type"]].append(result)
    
    # 計算各類型平均分
    type_averages = {}
    for query_type, type_items in type_results.items():
        type_averages[query_type] = {}
        for group in GROUPS_TO_EVALUATE:
            type_averages[query_type][group] = {}
            for k in k_values:
                k_key = f"k_{k}"
                metrics_list = [
                    item[group][k_key]
                    for item in type_items
                    if group in item and k_key in item[group]
                ]
                if metrics_list:
                    type_averages[query_type][group][k_key] = {
                        "strict_precision": sum(m["strict_precision"] for m in metrics_list) / len(metrics_list),
                        "strict_recall": sum(m["strict_recall"] for m in metrics_list) / len(metrics_list),
                        "strict_f1": sum(m["strict_f1"] for m in metrics_list) / len(metrics_list),
                        "relaxed_precision": sum(m["relaxed_precision"] for m in metrics_list) / len(metrics_list),
                        "relaxed_recall": sum(m["relaxed_recall"] for m in metrics_list) / len(metrics_list),
                        "relaxed_f1": sum(m["relaxed_f1"] for m in metrics_list) / len(metrics_list),
                    }
    
    # 計算總體平均分
    overall_averages = {}
    for group in GROUPS_TO_EVALUATE:
        overall_averages[group] = {}
        for k in k_values:
            k_key = f"k_{k}"
            metrics_list = [
                item[group][k_key]
                for item in all_results
                if group in item and k_key in item[group]
            ]
            if metrics_list:
                overall_averages[group][k_key] = {
                    "strict_precision": sum(m["strict_precision"] for m in metrics_list) / len(metrics_list),
                    "strict_recall": sum(m["strict_recall"] for m in metrics_list) / len(metrics_list),
                    "strict_f1": sum(m["strict_f1"] for m in metrics_list) / len(metrics_list),
                    "relaxed_precision": sum(m["relaxed_precision"] for m in metrics_list) / len(metrics_list),
                    "relaxed_recall": sum(m["relaxed_recall"] for m in metrics_list) / len(metrics_list),
                    "relaxed_f1": sum(m["relaxed_f1"] for m in metrics_list) / len(metrics_list),
                }
    
    # 生成報告
    report = {
        "evaluation_info": {
            "total_queries": len(ground_truth_data),
            "k_values": k_values,
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "doc_id_used": doc_id
        },
        "per_query_results": all_results,
        "type_averages": type_averages,
        "overall_averages": overall_averages
    }
    
    # 保存報告
    output_file = f"RQ4_evaluation_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 評估報告已保存至: {output_file}")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("評估結果摘要")
    print("=" * 80)
    
    for k in k_values:
        k_key = f"k_{k}"
        print(f"\n📊 K={k} 總體平均指標:")
        for group in GROUPS_TO_EVALUATE:
            if k_key in overall_averages.get(group, {}):
                avg = overall_averages[group][k_key]
                print(f"\n  {GROUP_LABELS[group]}:")
                print(f"    嚴格指標 - P@{k}: {avg['strict_precision']:.4f}, R@{k}: {avg['strict_recall']:.4f}, F1@{k}: {avg['strict_f1']:.4f}")
                print(f"    寬鬆指標 - P@{k}: {avg['relaxed_precision']:.4f}, R@{k}: {avg['relaxed_recall']:.4f}, F1@{k}: {avg['relaxed_f1']:.4f}")
        
        # 與對照組比較
        baseline_c = overall_averages.get("group_c", {}).get(k_key)
        baseline_d = overall_averages.get("group_d", {}).get(k_key)
        experimental = overall_averages.get("group_e", {}).get(k_key)
        
        if baseline_c and experimental:
            strict_diff_c = experimental["strict_f1"] - baseline_c["strict_f1"]
            relaxed_diff_c = experimental["relaxed_f1"] - baseline_c["relaxed_f1"]
            print(f"\n  📈 提升幅度（{GROUP_LABELS['group_e']} vs {GROUP_LABELS['group_c']}）:")
            if baseline_c["strict_f1"] > 0:
                print(f"    嚴格F1@{k}提升: {strict_diff_c:+.4f} ({strict_diff_c / baseline_c['strict_f1'] * 100:+.2f}%)")
            else:
                print(f"    嚴格F1@{k}提升: N/A")
            if baseline_c["relaxed_f1"] > 0:
                print(f"    寬鬆F1@{k}提升: {relaxed_diff_c:+.4f} ({relaxed_diff_c / baseline_c['relaxed_f1'] * 100:+.2f}%)")
            else:
                print(f"    寬鬆F1@{k}提升: N/A")
        
        if baseline_d and experimental:
            strict_diff_d = experimental["strict_f1"] - baseline_d["strict_f1"]
            relaxed_diff_d = experimental["relaxed_f1"] - baseline_d["relaxed_f1"]
            print(f"\n  📈 提升幅度（{GROUP_LABELS['group_e']} vs {GROUP_LABELS['group_d']}）:")
            if baseline_d["strict_f1"] > 0:
                print(f"    嚴格F1@{k}提升: {strict_diff_d:+.4f} ({strict_diff_d / baseline_d['strict_f1'] * 100:+.2f}%)")
            else:
                print(f"    嚴格F1@{k}提升: N/A")
            if baseline_d["relaxed_f1"] > 0:
                print(f"    寬鬆F1@{k}提升: {relaxed_diff_d:+.4f} ({relaxed_diff_d / baseline_d['relaxed_f1'] * 100:+.2f}%)")
            else:
                print(f"    寬鬆F1@{k}提升: N/A")
    
    # 按類型打印
    print("\n" + "=" * 80)
    print("按查詢類型分組的平均指標")
    print("=" * 80)
    
    for query_type in ["基礎型", "細節型", "主題型"]:
        if query_type in type_averages:
            print(f"\n📊 {query_type} (共 {len(type_results[query_type])} 題):")
            for k in k_values:
                k_key = f"k_{k}"
                print(f"\n  K={k}:")
                for group in GROUPS_TO_EVALUATE:
                    avg = type_averages[query_type].get(group, {}).get(k_key)
                    if avg:
                        print(f"    {GROUP_LABELS[group]}:")
                        print(f"      嚴格 - P@{k}: {avg['strict_precision']:.4f}, R@{k}: {avg['strict_recall']:.4f}, F1@{k}: {avg['strict_f1']:.4f}")
                        print(f"      寬鬆 - P@{k}: {avg['relaxed_precision']:.4f}, R@{k}: {avg['relaxed_recall']:.4f}, F1@{k}: {avg['relaxed_f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("評估完成！")
    print("=" * 80)
    
    return report

if __name__ == "__main__":
    try:
        report = evaluate_rq4()
    except KeyboardInterrupt:
        print("\n\n⚠️ 評估被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 評估過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

