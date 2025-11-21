import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import stats

# ==============================================================================
# 自動載入實驗結果，檢查常態性與變異數同質性，評估是否適合使用 ANOVA
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

EXPERIMENTS = [
    {
        "name": "RQ1 階層式 vs 傳統條文",
        "path": REPORT_PATHS["RQ1"],
        "groups": OrderedDict(
            [
                ("對照組 A (僅條文層)", "group_a"),
                ("實驗組 D (完整多層次)", "group_d"),
            ]
        ),
    },
    {
        "name": "RQ2 微觀細節 vs 宏觀結構",
        "path": REPORT_PATHS["RQ2_RQ3"],
        "groups": OrderedDict(
            [
                ("對照組 B (宏觀結構)", "group_b"),
                ("實驗組 C (微觀細節)", "group_c"),
            ]
        ),
    },
    {
        "name": "RQ4 LLM 兩階段 vs 純向量",
        "path": REPORT_PATHS["RQ4"],
        "groups": OrderedDict(
            [
                ("對照組 D (純向量)", "group_d"),
                ("實驗組 E (LLM 導向)", "group_e"),
            ]
        ),
    },
]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def collect_metric_arrays(per_query_results, groups):
    data = {metric: {label: [] for label in groups} for metric in METRIC_FIELDS}
    for item in per_query_results:
        for label, json_key in groups.items():
            group = item.get(json_key)
            if not group:
                raise ValueError(f"缺少 {json_key} 的檢索結果（query_id={item['query_id']}）")
            for metric_name, (k_field, stat_field) in METRIC_FIELDS.items():
                data[metric_name][label].append(group[k_field][stat_field])
    return data


def shapiro_test(label, scores):
    stat, p = stats.shapiro(scores)
    return {
        "label": label,
        "p_value": p,
        "is_normal": p >= 0.05,
    }


def levene_test(groups_scores):
    stat, p = stats.levene(*groups_scores)
    return {
        "p_value": p,
        "is_homogenous": p >= 0.05,
    }


def recommend_test(normal_ok, homogenous_ok, group_count):
    if normal_ok and homogenous_ok:
        if group_count == 2:
            return "✅ 符合常態與同質性，可採 Paired t-test 或單因子 ANOVA。"
        return "✅ 符合常態與同質性，可採單因子 ANOVA。"
    if normal_ok and not homogenous_ok:
        return "⚠️ 常態但變異數不同，建議使用 Welch's ANOVA / Welch t-test。"
    return "⚠️ 數據不常態，建議使用無母數檢定（如 Wilcoxon / Friedman）。"


def summarize_metric(exp_name, metric_name, metric_data):
    print(f"\n🔍 {exp_name} | 指標：{metric_name}")

    shapiro_results = []
    for label, scores in metric_data.items():
        result = shapiro_test(label, scores)
        shapiro_results.append(result)
        status = "✅ 常態" if result["is_normal"] else "❌ 不常態"
        print(f"   - Shapiro-Wilk | {label}: p = {result['p_value']:.5f} → {status}")

    normal_ok = all(r["is_normal"] for r in shapiro_results)

    groups_scores = list(metric_data.values())
    levene_result = levene_test(groups_scores)
    hom_status = "✅ 變異數相等" if levene_result["is_homogenous"] else "❌ 變異數不同"
    print(f"   - Levene Test: p = {levene_result['p_value']:.5f} → {hom_status}")

    recommendation = recommend_test(normal_ok, levene_result["is_homogenous"], len(metric_data))
    print(f"📝 建議：{recommendation}")

    if len(metric_data) == 2:
        labels = list(metric_data.keys())
        t_stat, p_t = stats.ttest_rel(metric_data[labels[0]], metric_data[labels[1]])
        w_stat, p_w = stats.wilcoxon(metric_data[labels[0]], metric_data[labels[1]])
        print(f"   - 參考 Paired t-test p = {p_t:.5f}")
        print(f"   - 參考 Wilcoxon p = {p_w:.5f}")

    means = ", ".join(f"{label} 平均 {np.mean(scores):.3f}" for label, scores in metric_data.items())
    print(f"   - 各組平均：{means}")
    print("-" * 60)


def main():
    print("📊 ANOVA 前提檢查報告\n" + "=" * 60)
    for exp in EXPERIMENTS:
        exp_json = load_json(exp["path"])
        metric_arrays = collect_metric_arrays(exp_json["per_query_results"], exp["groups"])
        print(f"\n# --- {exp['name']} ---")
        for metric_name, group_scores in metric_arrays.items():
            summarize_metric(exp["name"], metric_name, group_scores)


if __name__ == "__main__":
    main()