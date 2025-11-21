import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ==============================================================================
# ⚠️ 數據自動匯入
# 直接從最新的評估報告載入所有指標（Strict/Relaxed × K=5/10）
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent

REPORT_PATHS = {
    "RQ1": BASE_DIR / "RQ1_evaluation_report_20251120_154325.json",
    "RQ2_RQ3": BASE_DIR / "RQ2_RQ3_evaluation_report_20251119_161638.json",
    "RQ4": BASE_DIR / "RQ4_evaluation_report_20251119_162826.json",
}

METRIC_FIELDS = OrderedDict(
    [
        ("Strict F1@5", ("k_5", "strict_f1")),
        ("Strict F1@10", ("k_10", "strict_f1")),
        ("Relaxed F1@5", ("k_5", "relaxed_f1")),
        ("Relaxed F1@10", ("k_10", "relaxed_f1")),
    ]
)

RQ3_TYPE_ORDER = ["基礎型", "主題型", "細節型"]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def collect_metrics(per_query_results, group_mapping):
    metrics = {metric: {label: [] for label in group_mapping} for metric in METRIC_FIELDS}
    for item in per_query_results:
        for label, json_key in group_mapping.items():
            group = item.get(json_key)
            if not group:
                raise ValueError(f"缺少 {json_key} 的檢索結果（query_id={item['query_id']}）")
            for metric_name, (k_field, stat_field) in METRIC_FIELDS.items():
                metrics[metric_name][label].append(group[k_field][stat_field])
    return metrics


def build_type_buckets(per_query_results, group_key, metric_name):
    k_field, stat_field = METRIC_FIELDS[metric_name]
    buckets = defaultdict(list)
    for item in per_query_results:
        q_type = item["query_type"]
        group = item[group_key]
        buckets[q_type].append(group[k_field][stat_field])
    return buckets


rq1_json = load_json(REPORT_PATHS["RQ1"])
rq2_json = load_json(REPORT_PATHS["RQ2_RQ3"])
rq4_json = load_json(REPORT_PATHS["RQ4"])

data_RQ1 = collect_metrics(
    rq1_json["per_query_results"],
    {
        "對照組 A (僅條文層)": "group_a",
        "實驗組 D (完整多層次)": "group_d",
    },
)

data_RQ2 = collect_metrics(
    rq2_json["per_query_results"],
    {
        "對照組 B (宏觀結構)": "group_b",
        "實驗組 C (微觀細節)": "group_c",
    },
)

data_RQ4 = collect_metrics(
    rq4_json["per_query_results"],
    {
        "對照組 D (純向量)": "group_d",
        "實驗組 E (LLM導向)": "group_e",
    },
)

# ==============================================================================
# 統計檢定執行函式
# ==============================================================================

def run_test(name1, data1, name2, data2, metric_name):
    """執行成對樣本 Wilcoxon 符號等級檢定"""
    if len(data1) != 30 or len(data2) != 30:
        print(f"⚠️ 跳過：{name1} 或 {name2} 的數據數量不足 30 個。")
        return

    # 使用 'greater' 代表假設 data1 > data2
    stat, p_value = stats.wilcoxon(data1, data2, alternative='greater')
    
    print(f"📌 檢定目標：證明 {name1} > {name2} ({metric_name})")
    print(f"   - P-value: {p_value:.6f}")
    
    if p_value < 0.001:
        print("   - 結果: *** 極顯著差異 (p < 0.001) -> 實驗假設強力成立")
    elif p_value < 0.01:
        print("   - 結果: ** 非常顯著差異 (p < 0.01) -> 實驗假設成立")
    elif p_value < 0.05:
        print("   - 結果: * 顯著差異 (p < 0.05) -> 實驗假設成立")
    else:
        print("   - 結果: 無顯著差異 (p >= 0.05)")
    print("-" * 60)

# ==============================================================================
# 開始分析
# ==============================================================================

print("📊 碩士論文實驗數據統計檢定報告\n" + "=" * 60)

# --- RQ1: 階層式分塊 (D) vs 傳統條文 (A) ---
print("\n# --- RQ1: 階層式分塊 vs 傳統條文 ---")
for metric_name in METRIC_FIELDS:
    run_test(
        "實驗組 D (完整多層次)",
        data_RQ1[metric_name]["實驗組 D (完整多層次)"],
        "對照組 A (僅條文層)",
        data_RQ1[metric_name]["對照組 A (僅條文層)"],
        metric_name,
    )

# --- RQ2: 微觀細節 (C) vs 宏觀結構 (B) ---
print("\n# --- RQ2: 微觀細節 vs 宏觀結構 ---")
for metric_name in METRIC_FIELDS:
    run_test(
        "實驗組 C (微觀細節)",
        data_RQ2[metric_name]["實驗組 C (微觀細節)"],
        "對照組 B (宏觀結構)",
        data_RQ2[metric_name]["對照組 B (宏觀結構)"],
        metric_name,
    )

# --- RQ4: LLM 兩階段 (E) vs 純向量 (D) ---
print("\n# --- RQ4: LLM 兩階段推理 vs 純向量 ---")
for metric_name in METRIC_FIELDS:
    run_test(
        "實驗組 E (LLM導向)",
        data_RQ4[metric_name]["實驗組 E (LLM導向)"],
        "對照組 D (純向量)",
        data_RQ4[metric_name]["對照組 D (純向量)"],
        metric_name,
    )

# --- RQ3: 查詢意圖分析 (C組在不同題型表現) ---
print("\n# --- RQ3: 題型差異分析 (C 組) ---")
for metric_name in METRIC_FIELDS:
    c_scores = data_RQ2[metric_name]["實驗組 C (微觀細節)"]
    if len(c_scores) != 30:
        print(f"⚠️ 跳過：C 組 {metric_name} 的樣本不足 30 筆。")
        print("-" * 60)
        continue

    type_buckets = build_type_buckets(
        rq2_json["per_query_results"], "group_c", metric_name
    )
    ordered_buckets = [
        (label, type_buckets[label]) for label in RQ3_TYPE_ORDER if label in type_buckets
    ]

    if len(ordered_buckets) < 2:
        print(f"⚠️ 題型數量不足，無法對 {metric_name} 執行 Kruskal-Wallis 檢定。")
        print("-" * 60)
        continue

    stat, p_kw = stats.kruskal(*[vals for _, vals in ordered_buckets])
    print(f"📌 RQ3 分析 ({metric_name}): C 組在不同題型間的表現差異")
    print(f"   - P-value: {p_kw:.6f}")
    summary = ", ".join(
        f"{label} 平均 {np.mean(vals):.3f}" for label, vals in ordered_buckets
    )
    print(f"   - 類型平均: {summary}")
    if p_kw < 0.05:
        print("   - 結果: * 顯著差異 (不同題型的 F1 分數有明顯落差)")
    else:
        print("   - 結果: 無顯著差異")
    print("-" * 60)