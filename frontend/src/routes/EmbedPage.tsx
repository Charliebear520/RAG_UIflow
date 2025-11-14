import React, { useState } from "react";
import { useRag } from "../lib/ragStore";

// 實驗組定義
const EXPERIMENTAL_GROUPS = {
  group_a: {
    name: "A組：僅條文層 (Baseline)",
    description: "僅使用基本單元層（條文）",
    levels: ["basic_unit"],
    research_purpose: "傳統平面法的表現，作為基線對照組",
  },
  group_b: {
    name: "B組：條文+章節結構",
    description: "基本單元層級層 + 基本單元層（章、節、編 + 條文）",
    levels: ["document_component", "basic_unit_hierarchy", "basic_unit"],
    research_purpose: "評估結構分組的嵌入是否能更好地捕捉廣泛主題",
  },
  group_c: {
    name: "C組：條文+細節層次",
    description: "基本單元層 + 基本單元組成層 + 列舉層（條文 + 項 + 款目）",
    levels: ["basic_unit", "basic_unit_component", "enumeration"],
    research_purpose: "評估細節化層次對列舉式規定的精確度增益",
  },
  group_d: {
    name: "D組：完整多層次ML-RAG",
    description: "章、節、條文、項、款、目層級",
    levels: [
      "document_component",
      "basic_unit_hierarchy",
      "basic_unit",
      "basic_unit_component",
      "enumeration",
    ],
    research_purpose: "作為最佳效能的對比組，評估完整多層次方法的綜合表現",
  },
};

type ExperimentalGroup = "group_a" | "group_b" | "group_c" | "group_d";

export function EmbedPage() {
  const {
    canEmbed,
    embed,
    multiLevelEmbed,
    embedProvider,
    embedModel,
    embedDimension,
  } = useRag();
  const [busy, setBusy] = useState(false);
  const [selectedGroup, setSelectedGroup] =
    useState<ExperimentalGroup>("group_a");

  const handleEmbed = async () => {
    setBusy(true);
    try {
      // 所有實驗組統一使用multiLevelEmbed，確保只處理指定層級的chunks
      await multiLevelEmbed([selectedGroup]);
    } catch (error) {
      console.error("Embedding failed:", error);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-4">Embedding 設置</h2>

        {!canEmbed ? (
          <div className="alert alert-warning">
            <i className="bi bi-exclamation-triangle me-2"></i>
            請先完成文檔上傳和分塊處理
          </div>
        ) : (
          <>
            {/* 實驗組選擇 */}
            <div className="mb-4">
              <h5>選擇實驗組</h5>
              <p className="text-muted small mb-3">
                請選擇要生成的實驗組embedding，用於後續的對比實驗：
              </p>
              <div className="row">
                {Object.entries(EXPERIMENTAL_GROUPS).map(([key, group]) => (
                  <div key={key} className="col-md-6 mb-3">
                    <div
                      className={`card ${
                        selectedGroup === key
                          ? "border-success bg-light"
                          : "border-light"
                      }`}
                    >
                      <div className="card-body">
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="radio"
                            name="experimentalGroup"
                            id={key}
                            value={key}
                            checked={selectedGroup === key}
                            onChange={(e) =>
                              setSelectedGroup(
                                e.target.value as ExperimentalGroup
                              )
                            }
                          />
                          <label className="form-check-label" htmlFor={key}>
                            <strong>{group.name}</strong>
                          </label>
                        </div>
                        <p className="card-text small text-muted mt-2">
                          {group.description}
                        </p>
                        <div className="small">
                          <strong>包含層次:</strong> {group.levels.join(", ")}
                        </div>
                        <div className="small text-muted">
                          <em>{group.research_purpose}</em>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* 選中實驗組的詳細信息 */}
              <div className="alert alert-info">
                <h6 className="alert-heading">
                  選中的實驗組：{EXPERIMENTAL_GROUPS[selectedGroup].name}
                </h6>
                <p className="mb-1">
                  <strong>描述：</strong>
                  {EXPERIMENTAL_GROUPS[selectedGroup].description}
                </p>
                <p className="mb-1">
                  <strong>包含層次：</strong>
                  {EXPERIMENTAL_GROUPS[selectedGroup].levels.join(", ")}
                </p>
                <p className="mb-0">
                  <strong>研究目的：</strong>
                  {EXPERIMENTAL_GROUPS[selectedGroup].research_purpose}
                </p>
              </div>
            </div>

            {/* 執行Embedding */}
            <div className="mb-4">
              <button
                className="btn btn-primary"
                onClick={handleEmbed}
                disabled={busy}
              >
                {busy ? (
                  <>
                    <span
                      className="spinner-border spinner-border-sm me-2"
                      role="status"
                      aria-hidden="true"
                    ></span>
                    生成 Embedding 中...
                  </>
                ) : (
                  <>
                    <i className="bi bi-play-circle me-2"></i>
                    生成 {EXPERIMENTAL_GROUPS[selectedGroup].name} Embedding
                  </>
                )}
              </button>
            </div>

            {/* 當前狀態顯示 */}
            {embedProvider && (
              <div className="alert alert-success">
                <h6 className="alert-heading">Embedding 狀態</h6>
                <div className="row">
                  <div className="col-md-4">
                    <strong>Provider:</strong> {embedProvider}
                  </div>
                  <div className="col-md-4">
                    <strong>Model:</strong> {embedModel}
                  </div>
                  <div className="col-md-4">
                    <strong>Dimension:</strong> {embedDimension}
                  </div>
                </div>
              </div>
            )}

            {/* 測試按鈕 */}
            <div className="mt-3">
              <button
                className="btn btn-sm btn-outline-info me-2"
                onClick={async () => {
                  try {
                    const response = await fetch(
                      "/api/test-experimental-groups",
                      {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          experimental_groups: [selectedGroup],
                        }),
                      }
                    );
                    const result = await response.json();
                    console.log("🧪 實驗組測試結果:", result);
                    alert(
                      `實驗組測試完成！\n選中層次: ${result.selected_levels.join(
                        ", "
                      )}\n跳過層次: ${result.skipped_levels.join(
                        ", "
                      )}\n\n詳細結果請查看控制台`
                    );
                  } catch (error) {
                    console.error("測試失敗:", error);
                    alert("測試失敗，請檢查控制台");
                  }
                }}
              >
                🧪 測試實驗組層次選擇
              </button>

              <button
                className="btn btn-sm btn-outline-warning"
                onClick={async () => {
                  try {
                    const response = await fetch("/api/debug-store");
                    const result = await response.json();
                    console.log("🔍 Store狀態:", result);
                    alert(
                      `Store狀態檢查完成！\n多層次embedding: ${
                        result.has_multi_level_embeddings ? "是" : "否"
                      }\n可用層次: ${result.available_levels.join(
                        ", "
                      )}\n\n詳細結果請查看控制台`
                    );
                  } catch (error) {
                    console.error("檢查失敗:", error);
                    alert("檢查失敗，請檢查控制台");
                  }
                }}
              >
                🔍 檢查Embedding狀態
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
