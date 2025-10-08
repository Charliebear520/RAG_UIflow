import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";
import { api } from "../lib/api";

export function RetrievePage() {
  const nav = useNavigate();
  const {
    canRetrieve,
    retrieve,
    hybridRetrieve,
    hierarchicalRetrieve,
    multiLevelRetrieve,
    multiLevelFusionRetrieve,
    hopragEnhancedRetrieve,
    retrieval,
    generate,
    answer,
    steps,
    legalReferences,
  } = useRag();
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [retrievalMethod, setRetrievalMethod] = useState("vector");

  // HopRAG管理狀態
  const [hopragStatus, setHopragStatus] = useState<any>(null);
  const [hopragConfig, setHopragConfig] = useState<any>(null);
  const [showHopragManager, setShowHopragManager] = useState(false);
  const [hopragLoading, setHopragLoading] = useState(false);
  const [hopragMessage, setHopragMessage] = useState<string>("");

  // Structured-HopRAG構建狀態
  const [structuredHopragBuilt, setStructuredHopragBuilt] = useState(false);
  const [buildingStructuredHoprag, setBuildingStructuredHoprag] =
    useState(false);
  const [structuredHopragRetrieval, setStructuredHopragRetrieval] =
    useState<any>(null);

  // HopRAG管理功能
  const fetchHopragStatus = async () => {
    try {
      const response = await api.getHopragStatus();
      setHopragStatus(response);
      return response;
    } catch (error) {
      console.error("獲取HopRAG狀態失敗:", error);
      setHopragMessage("獲取HopRAG狀態失敗");
      return null;
    }
  };

  const fetchHopragConfig = async () => {
    try {
      const response = await api.getHopragConfig();
      setHopragConfig(response.config);
      return response.config;
    } catch (error) {
      console.error("獲取HopRAG配置失敗:", error);
      return null;
    }
  };

  const buildHopragGraph = async () => {
    setHopragLoading(true);
    setHopragMessage("");
    try {
      const response = await api.buildHopragGraph();
      setHopragMessage(
        `HopRAG圖譜構建成功！節點數: ${response.statistics.total_nodes}, 邊數: ${response.statistics.total_edges}`
      );
      await fetchHopragStatus();
    } catch (error: any) {
      setHopragMessage(
        `HopRAG圖譜構建失敗: ${error.response?.data?.error || error.message}`
      );
    } finally {
      setHopragLoading(false);
    }
  };

  const updateHopragConfig = async () => {
    if (!hopragConfig) return;

    setHopragLoading(true);
    setHopragMessage("");
    try {
      await api.updateHopragConfig(hopragConfig);
      setHopragMessage("HopRAG配置更新成功！");
      await fetchHopragConfig();
    } catch (error: any) {
      setHopragMessage(
        `配置更新失敗: ${error.response?.data?.error || error.message}`
      );
    } finally {
      setHopragLoading(false);
    }
  };

  const resetHopragSystem = async () => {
    if (!confirm("確定要重置HopRAG系統嗎？這將清除所有圖數據。")) {
      return;
    }

    setHopragLoading(true);
    setHopragMessage("");
    try {
      await api.resetHopragSystem();
      setHopragMessage("HopRAG系統重置成功！");
      await fetchHopragStatus();
      await fetchHopragConfig();
    } catch (error: any) {
      setHopragMessage(
        `系統重置失敗: ${error.response?.data?.error || error.message}`
      );
    } finally {
      setHopragLoading(false);
    }
  };

  const buildStructuredHoprag = async () => {
    setBuildingStructuredHoprag(true);
    try {
      const response = await fetch("/api/build-structured-hoprag-graph", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await response.json();

      if (data.status === "success") {
        setStructuredHopragBuilt(true);
        alert(
          `Structured-HopRAG圖譜構建成功！\n節點: ${
            data.statistics.total_nodes || 0
          }\n邊: ${
            data.statistics.total_edges || 0
          }\n構建時間: ${data.build_time.toFixed(2)}秒`
        );
      } else {
        alert("Structured-HopRAG圖譜構建失敗");
      }
    } catch (error) {
      console.error("構建Structured-HopRAG圖譜錯誤:", error);
      alert("Structured-HopRAG圖譜構建錯誤");
    } finally {
      setBuildingStructuredHoprag(false);
    }
  };

  useEffect(() => {
    if (retrievalMethod === "hoprag") {
      fetchHopragStatus();
      fetchHopragConfig();
    }
  }, [retrievalMethod]);

  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-3">Retrieve</h2>

        {/* 檢索方法選擇 */}
        <div className="mb-3">
          <div className="form-check form-check-inline">
            <input
              className="form-check-input"
              type="radio"
              name="retrievalMethod"
              id="vectorOnly"
              value="vector"
              checked={retrievalMethod === "vector"}
              onChange={(e) => setRetrievalMethod(e.target.value)}
            />
            <label className="form-check-label" htmlFor="vectorOnly">
              標準檢索 (Standard Retrieval)
            </label>
          </div>
          <div className="form-check form-check-inline">
            <input
              className="form-check-input"
              type="radio"
              name="retrievalMethod"
              id="hybridRag"
              value="hybrid"
              checked={retrievalMethod === "hybrid"}
              onChange={(e) => setRetrievalMethod(e.target.value)}
            />
            <label className="form-check-label" htmlFor="hybridRag">
              HybridRAG (向量 + 法律規則)
            </label>
          </div>
          <div className="form-check form-check-inline">
            <input
              className="form-check-input"
              type="radio"
              name="retrievalMethod"
              id="hopragRag"
              value="hoprag"
              checked={retrievalMethod === "hoprag"}
              onChange={(e) => setRetrievalMethod(e.target.value)}
            />
            <label className="form-check-label" htmlFor="hopragRag">
              HopRAG (多跳推理檢索) 🧠
            </label>
          </div>
          {/* <div className="form-check form-check-inline">
            <input
              className="form-check-input"
              type="radio"
              name="retrievalMethod"
              id="multiLevelRag"
              value="multi_level"
              checked={retrievalMethod === "multi_level"}
              onChange={(e) => setRetrievalMethod(e.target.value)}
            />
            <label className="form-check-label" htmlFor="multiLevelRag">
              多層次檢索 (Multi-Layered Retrieval) 📚
            </label>
          </div> */}
          <div className="form-check form-check-inline">
            <input
              className="form-check-input"
              type="radio"
              name="retrievalMethod"
              id="multiLevelFusionRag"
              value="multi_level_fusion"
              checked={retrievalMethod === "multi_level_fusion"}
              onChange={(e) => setRetrievalMethod(e.target.value)}
            />
            <label className="form-check-label" htmlFor="multiLevelFusionRag">
              多層次融合檢索 (Multi-Layered Fusion) 🔄
            </label>
          </div>
          <div className="form-check form-check-inline">
            <input
              className="form-check-input"
              type="radio"
              name="retrievalMethod"
              id="structuredHoprag"
              value="structured_hoprag"
              checked={retrievalMethod === "structured_hoprag"}
              onChange={(e) => setRetrievalMethod(e.target.value)}
            />
            <label className="form-check-label" htmlFor="structuredHoprag">
              Structured-HopRAG 🚀
            </label>
          </div>
        </div>

        {/* Structured-HopRAG構建按鈕 */}
        {retrievalMethod === "structured_hoprag" && (
          <div className="alert alert-info mb-3">
            <div className="d-flex align-items-center gap-2 mb-2">
              <button
                className="btn btn-sm btn-outline-success"
                onClick={buildStructuredHoprag}
                disabled={buildingStructuredHoprag || structuredHopragBuilt}
              >
                {buildingStructuredHoprag
                  ? "構建中..."
                  : structuredHopragBuilt
                  ? "Structured-HopRAG已構建"
                  : "構建Structured-HopRAG"}
              </button>
              {structuredHopragBuilt && (
                <span className="badge bg-success">就緒 🚀</span>
              )}
            </div>
            <small className="text-muted">
              針對結構化法律文本優化的HopRAG系統：95%索引成本降低 +
              99.8%檢索速度提升
            </small>
          </div>
        )}

        {/* 方法說明 */}
        <div className="alert alert-info mb-3">
          <h6 className="mb-2">檢索方法說明</h6>
          <ul className="mb-0 small">
            <li>
              <strong>標準檢索</strong>：傳統的單一embedding檢索
            </li>
            <li>
              <strong>HybridRAG</strong>：結合向量檢索和法律規則的混合檢索
            </li>
            <li>
              <strong>HopRAG</strong>
              ：多跳推理檢索，通過邏輯連接發現間接相關的法律條文
            </li>
            <li>
              <strong>多層次檢索</strong>
              ：基於六個粒度級別（文件、章、節、條、項、款/目），智能選擇最合適的層次進行檢索
            </li>
            <li>
              <strong>多層次融合檢索</strong>
              ：從所有六個粒度級別檢索並融合結果，提供最全面的檢索效果
            </li>
          </ul>
        </div>

        {/* HopRAG使用指導 */}
        {retrievalMethod === "hoprag" && (
          <div
            className={`alert ${
              hopragStatus?.system_ready ? "alert-success" : "alert-warning"
            } mb-3`}
          >
            <h6 className="mb-2">🧠 HopRAG 使用說明</h6>
            <div className="row">
              <div className="col-md-8">
                <p className="mb-2">
                  <strong>HopRAG狀態：</strong>
                  {hopragStatus?.system_ready ? (
                    <span className="badge bg-success ms-2">
                      ✅ 系統就緒，可以進行檢索
                    </span>
                  ) : (
                    <span className="badge bg-warning ms-2">
                      ⚠️ 系統未就緒，需要構建圖譜
                    </span>
                  )}
                </p>
                {!hopragStatus?.system_ready && (
                  <div>
                    <p className="mb-2">
                      <strong>HopRAG需要先完成以下步驟：</strong>
                    </p>
                    <ol className="mb-2 small">
                      <li>確保已上傳法律文檔並完成分塊處理</li>
                      <li>執行多層次embedding生成</li>
                      <li>構建HopRAG圖譜（見下方管理面板）</li>
                      <li>等待圖譜構建完成後即可使用HopRAG檢索</li>
                    </ol>
                  </div>
                )}
                {hopragStatus?.system_ready &&
                  hopragStatus?.graph_statistics && (
                    <div className="mb-2">
                      <small className="text-muted">
                        📊 圖譜統計：{hopragStatus.graph_statistics.total_nodes}
                        個節點，{hopragStatus.graph_statistics.total_edges}條邊
                      </small>
                    </div>
                  )}
              </div>
              <div className="col-md-4 text-end">
                <button
                  className="btn btn-sm btn-outline-primary"
                  onClick={() => setShowHopragManager(!showHopragManager)}
                >
                  {showHopragManager ? "隱藏管理面板" : "顯示管理面板"}
                </button>
              </div>
            </div>
            <div className="mt-2">
              <small className="text-muted">
                💡 <strong>提示</strong>
                ：HopRAG通過構建法律概念圖譜，能夠發現間接相關的法律條文，
                提供更全面的檢索結果和更深入的法律推理。
              </small>
            </div>
          </div>
        )}

        {/* HopRAG管理面板 */}
        {retrievalMethod === "hoprag" && showHopragManager && (
          <div className="card mb-4">
            <div className="card-header">
              <h5 className="mb-0">HopRAG 系統管理</h5>
            </div>
            <div className="card-body">
              {hopragMessage && (
                <div
                  className={`alert ${
                    hopragMessage.includes("成功")
                      ? "alert-success"
                      : "alert-danger"
                  } alert-dismissible fade show mb-3`}
                  role="alert"
                >
                  {hopragMessage}
                  <button
                    type="button"
                    className="btn-close"
                    data-bs-dismiss="alert"
                    onClick={() => setHopragMessage("")}
                  ></button>
                </div>
              )}

              {/* 系統狀態 */}
              <div className="mb-4">
                <h6>系統狀態</h6>
                {hopragStatus ? (
                  <div className="row">
                    <div className="col-md-6">
                      <div className="card bg-light">
                        <div className="card-body">
                          <h6 className="card-title">圖譜狀態</h6>
                          <p className="mb-1">
                            <span
                              className={`badge ${
                                hopragStatus.graph_statistics?.graph_built
                                  ? "bg-success"
                                  : "bg-warning"
                              }`}
                            >
                              圖譜狀態:{" "}
                              {hopragStatus.graph_statistics?.graph_built
                                ? "已構建"
                                : "未構建"}
                            </span>
                          </p>
                          <p className="mb-1">
                            節點總數:{" "}
                            {hopragStatus.graph_statistics?.total_nodes || 0}
                          </p>
                          <p className="mb-1">
                            邊總數:{" "}
                            {hopragStatus.graph_statistics?.total_edges || 0}
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="col-md-6">
                      <div className="card bg-light">
                        <div className="card-body">
                          <h6 className="card-title">系統狀態</h6>
                          <p className="mb-1">
                            <span
                              className={`badge ${
                                hopragStatus.system_ready
                                  ? "bg-success"
                                  : "bg-warning"
                              }`}
                            >
                              {hopragStatus.system_ready
                                ? "✅ 系統就緒"
                                : "⚠️ 系統未就緒"}
                            </span>
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-muted">載入中...</div>
                )}
              </div>

              {/* 操作按鈕 */}
              <div className="mb-4">
                <h6>系統操作</h6>
                <div className="btn-group" role="group">
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={buildHopragGraph}
                    disabled={hopragLoading}
                  >
                    {hopragLoading ? "構建中..." : "構建HopRAG圖譜"}
                  </button>
                  <button
                    type="button"
                    className="btn btn-info"
                    onClick={fetchHopragStatus}
                    disabled={hopragLoading}
                  >
                    刷新狀態
                  </button>
                  <button
                    type="button"
                    className="btn btn-warning"
                    onClick={resetHopragSystem}
                    disabled={hopragLoading}
                  >
                    重置系統
                  </button>
                </div>
              </div>

              {/* 配置設置 */}
              <div>
                <h6>HopRAG 配置</h6>
                {hopragConfig ? (
                  <div className="row">
                    <div className="col-md-6">
                      <div className="mb-3">
                        <label className="form-label">相似度閾值</label>
                        <input
                          type="number"
                          className="form-control"
                          min="0"
                          max="1"
                          step="0.1"
                          value={hopragConfig.similarity_threshold}
                          onChange={(e) =>
                            setHopragConfig({
                              ...hopragConfig,
                              similarity_threshold: parseFloat(e.target.value),
                            })
                          }
                        />
                      </div>
                      <div className="mb-3">
                        <label className="form-label">最大跳躍數</label>
                        <input
                          type="number"
                          className="form-control"
                          min="1"
                          max="10"
                          value={hopragConfig.max_hops}
                          onChange={(e) =>
                            setHopragConfig({
                              ...hopragConfig,
                              max_hops: parseInt(e.target.value),
                            })
                          }
                        />
                      </div>
                    </div>
                    <div className="col-md-6">
                      <div className="mb-3">
                        <label className="form-label">每跳最大節點數</label>
                        <input
                          type="number"
                          className="form-control"
                          min="1"
                          max="50"
                          value={hopragConfig.top_k_per_hop}
                          onChange={(e) =>
                            setHopragConfig({
                              ...hopragConfig,
                              top_k_per_hop: parseInt(e.target.value),
                            })
                          }
                        />
                      </div>
                      <div className="mb-3">
                        <label className="form-label">基礎檢索策略</label>
                        <select
                          className="form-select"
                          value={hopragConfig.base_strategy}
                          onChange={(e) =>
                            setHopragConfig({
                              ...hopragConfig,
                              base_strategy: e.target.value,
                            })
                          }
                        >
                          <option value="multi_level">多層次檢索</option>
                          <option value="single_level">單層次檢索</option>
                          <option value="hybrid">混合檢索</option>
                        </select>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-muted">載入中...</div>
                )}
                <div className="mb-3">
                  <div className="form-check">
                    <input
                      className="form-check-input"
                      type="checkbox"
                      id="useHopRAG"
                      checked={hopragConfig?.use_hoprag || false}
                      onChange={(e) =>
                        setHopragConfig({
                          ...hopragConfig,
                          use_hoprag: e.target.checked,
                        })
                      }
                    />
                    <label className="form-check-label" htmlFor="useHopRAG">
                      啟用HopRAG增強
                    </label>
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-success"
                  onClick={updateHopragConfig}
                  disabled={hopragLoading}
                >
                  更新配置
                </button>
              </div>
            </div>
          </div>
        )}

        <form
          className="row gy-2 gx-2 align-items-end"
          onSubmit={async (e) => {
            e.preventDefault();
            if (!canRetrieve) return;
            setBusy(true);

            try {
              if (retrievalMethod === "hybrid") {
                await hybridRetrieve(query, k);
              } else if (retrievalMethod === "multi_level") {
                await multiLevelRetrieve(query, k);
              } else if (retrievalMethod === "multi_level_fusion") {
                await multiLevelFusionRetrieve(query, k);
              } else if (retrievalMethod === "hoprag") {
                // 檢查HopRAG圖譜是否已構建
                if (!hopragStatus?.system_ready) {
                  alert("請先構建HopRAG圖譜，或檢查HopRAG系統狀態");
                  return;
                }
                console.log("🚀 開始HopRAG檢索，查詢:", query, "k:", k);
                await hopragEnhancedRetrieve(query, k);
                console.log("✅ HopRAG檢索完成，當前retrieval狀態:", retrieval);
              } else if (retrievalMethod === "structured_hoprag") {
                if (!structuredHopragBuilt) {
                  alert("請先構建Structured-HopRAG圖譜");
                  return;
                }
                const response = await fetch(
                  "/api/structured-hoprag-retrieve",
                  {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query, k }),
                  }
                );
                const data = await response.json();
                setStructuredHopragRetrieval(data);
              } else {
                await retrieve(query, k);
              }
            } catch (error) {
              console.error("檢索錯誤:", error);
              alert("檢索失敗，請檢查控制台");
            } finally {
              setBusy(false);
            }
          }}
        >
          <div className="col-12 col-md-6">
            <label className="form-label">Query</label>
            <input
              className="form-control"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask something…"
            />
          </div>
          <div className="col-auto">
            <label className="form-label">Top-K</label>
            <select
              className="form-select"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
            >
              <option value={1}>1</option>
              <option value={3}>3</option>
              <option value={5}>5</option>
              <option value={10}>10</option>
            </select>
          </div>
          <div className="col-auto">
            <button
              disabled={
                busy ||
                (retrievalMethod === "hoprag"
                  ? !hopragStatus?.system_ready
                  : !canRetrieve)
              }
              className="btn btn-primary"
              type="submit"
            >
              {busy ? "Searching…" : "Search"}
            </button>
          </div>
          <div className="col-auto">
            <button
              type="button"
              className="btn btn-success"
              disabled={
                ((!retrieval || retrieval.length === 0) &&
                  (!structuredHopragRetrieval ||
                    !structuredHopragRetrieval.results ||
                    structuredHopragRetrieval.results.length === 0)) ||
                busy
              }
              onClick={async () => {
                setBusy(true);
                try {
                  await generate(query, k);
                } finally {
                  setBusy(false);
                }
              }}
            >
              {busy ? "Generating…" : "Generate Answer"}
            </button>
          </div>
        </form>

        {(retrieval || structuredHopragRetrieval) && (
          <div className="mt-3">
            <h3 className="h6">
              Top results{" "}
              {retrievalMethod === "hybrid" && (
                <span className="badge bg-primary">HybridRAG</span>
              )}
              {retrievalMethod === "hoprag" && (
                <span className="badge bg-warning">HopRAG 🧠</span>
              )}
              {retrievalMethod === "multi_level" && (
                <span className="badge bg-success">
                  Multi-Layered Retrieval 📚
                </span>
              )}
              {retrievalMethod === "multi_level_fusion" && (
                <span className="badge bg-warning">
                  Multi-Layered Fusion 🔄
                </span>
              )}
              {retrievalMethod === "structured_hoprag" && (
                <span className="badge bg-success">Structured-HopRAG 🚀</span>
              )}
              {retrieval &&
                retrieval.length > 0 &&
                retrieval[0].embedding_provider && (
                  <span className="badge bg-secondary ms-2">
                    {retrieval[0].embedding_provider}
                    {retrieval[0].embedding_model &&
                      ` (${retrieval[0].embedding_model})`}
                  </span>
                )}
            </h3>

            {/* 顯示檢索指標 */}
            {retrieval && retrieval.length > 0 && retrieval[0].metrics && (
              <div className="alert alert-info mb-3">
                <h6 className="mb-2">檢索指標</h6>
                <div className="row">
                  <div className="col-md-2">
                    <span className="badge bg-success me-2">
                      P@{k}: {retrieval[0].metrics.p_at_k?.toFixed(3) || "N/A"}
                    </span>
                  </div>
                  <div className="col-md-2">
                    <span className="badge bg-warning me-2">
                      R@{k}: {retrieval[0].metrics.r_at_k?.toFixed(3) || "N/A"}
                    </span>
                  </div>
                  <div className="col-md-2">
                    <span className="badge bg-secondary me-2">
                      相關: {retrieval[0].metrics.relevant_chunks_count || 0}
                    </span>
                  </div>
                  <div className="col-md-2">
                    <span className="badge bg-info me-2">
                      總數:{" "}
                      {retrieval[0].metrics.query_analysis?.total_results || 0}
                    </span>
                  </div>
                  <div className="col-md-4">
                    <span
                      className={`badge me-2 ${
                        retrieval[0].metrics.query_analysis?.query_type ===
                        "explicit_article"
                          ? "bg-primary"
                          : "bg-success"
                      }`}
                    >
                      {retrieval[0].metrics.query_analysis?.query_type ===
                      "explicit_article"
                        ? "明確法條查詢"
                        : "語義化查詢"}
                    </span>
                    <small className="text-muted">
                      閾值:{" "}
                      {retrieval[0].metrics.query_analysis?.threshold_used ||
                        "N/A"}
                    </small>
                  </div>
                </div>
                {retrieval[0].metrics.query_analysis?.article_numbers &&
                  retrieval[0].metrics.query_analysis.article_numbers.length >
                    0 && (
                    <div className="mt-2">
                      <small className="text-muted">
                        提取法條:{" "}
                        {retrieval[0].metrics.query_analysis.article_numbers.join(
                          ", "
                        )}
                      </small>
                    </div>
                  )}
                {retrieval[0].metrics.query_analysis?.law_keywords &&
                  retrieval[0].metrics.query_analysis.law_keywords.length >
                    0 && (
                    <div className="mt-1">
                      <small className="text-muted">
                        法律關鍵字:{" "}
                        {retrieval[0].metrics.query_analysis.law_keywords.join(
                          ", "
                        )}
                      </small>
                    </div>
                  )}
                {retrieval[0].metrics.query_expansion && (
                  <div className="mt-2">
                    <h6 className="mb-1">查詢擴展分析</h6>
                    <div className="row">
                      <div className="col-md-6">
                        <small className="text-muted">
                          <strong>檢測領域:</strong>{" "}
                          {retrieval[0].metrics.query_expansion.detected_domains?.join(
                            ", "
                          ) || "無"}
                        </small>
                      </div>
                      <div className="col-md-6">
                        <small className="text-muted">
                          <strong>擴展比例:</strong>{" "}
                          {retrieval[0].metrics.query_expansion.expansion_ratio?.toFixed(
                            2
                          ) || "0.00"}
                        </small>
                      </div>
                    </div>
                    {retrieval[0].metrics.query_expansion.domain_matches &&
                      retrieval[0].metrics.query_expansion.domain_matches
                        .length > 0 && (
                        <div className="mt-1">
                          <small className="text-muted">
                            <strong>概念映射:</strong>{" "}
                            {retrieval[0].metrics.query_expansion.domain_matches
                              .slice(0, 3)
                              .join(", ")}
                            {retrieval[0].metrics.query_expansion.domain_matches
                              .length > 3 && "..."}
                          </small>
                        </div>
                      )}
                  </div>
                )}
                {retrieval[0].metrics.hierarchical_analysis && (
                  <div className="mt-2">
                    <h6 className="mb-1">多層次檢索分析</h6>
                    <div className="row">
                      <div className="col-md-4">
                        <small className="text-muted">
                          <strong>法條級別:</strong>{" "}
                          {retrieval[0].metrics.hierarchical_analysis
                            .article_results || 0}{" "}
                          個
                        </small>
                      </div>
                      <div className="col-md-4">
                        <small className="text-muted">
                          <strong>節級別:</strong>{" "}
                          {retrieval[0].metrics.hierarchical_analysis
                            .section_results || 0}{" "}
                          個
                        </small>
                      </div>
                      <div className="col-md-4">
                        <small className="text-muted">
                          <strong>章級別:</strong>{" "}
                          {retrieval[0].metrics.hierarchical_analysis
                            .chapter_results || 0}{" "}
                          個
                        </small>
                      </div>
                    </div>
                  </div>
                )}
                {retrieval[0].metrics.note && (
                  <div className="mt-1">
                    <small className="text-success">
                      {retrieval[0].metrics.note}
                    </small>
                  </div>
                )}
              </div>
            )}
            <ol>
              {(() => {
                // 確定要顯示的結果數據
                let resultsToShow = null;
                if (
                  retrievalMethod === "structured_hoprag" &&
                  structuredHopragRetrieval?.results
                ) {
                  resultsToShow = structuredHopragRetrieval.results;
                } else if (retrieval) {
                  resultsToShow = retrieval;
                }

                console.log("🎯 結果顯示邏輯:", {
                  retrievalMethod,
                  hasRetrieval: !!retrieval,
                  retrievalLength: retrieval?.length,
                  hasStructuredHopragRetrieval:
                    !!structuredHopragRetrieval?.results,
                  structuredHopragLength:
                    structuredHopragRetrieval?.results?.length,
                  resultsToShow: resultsToShow?.length,
                });

                return resultsToShow?.map((r: any, index: number) => (
                  <li
                    key={r.node_id || `${r.doc_id}-${r.chunk_index || index}`}
                    className="mb-2"
                  >
                    <div className="small text-muted">
                      {retrievalMethod === "hybrid" ? (
                        <>
                          <span className="badge bg-success me-1">
                            總分: {r.score?.toFixed(3) || "N/A"}
                          </span>
                          <span className="badge bg-info me-1">
                            向量: {r.vector_score?.toFixed(3) || "N/A"}
                          </span>
                          <span className="badge bg-warning me-1">
                            規則: {r.bonus?.toFixed(3) || "N/A"}
                          </span>
                        </>
                      ) : retrievalMethod === "multi_level" ? (
                        <>
                          <span className="badge bg-success me-1">
                            相似度: {r.similarity?.toFixed(3) || "N/A"}
                          </span>
                          <span className="badge bg-primary me-1">
                            層次: {r.metadata?.level || "N/A"}
                          </span>
                          <span className="badge bg-info me-1">
                            查詢類型: {r.metadata?.query_type || "N/A"}
                          </span>
                          <span className="badge bg-warning me-1">
                            置信度:{" "}
                            {r.metadata?.confidence?.toFixed(2) || "N/A"}
                          </span>
                        </>
                      ) : retrievalMethod === "multi_level_fusion" ? (
                        <>
                          <span className="badge bg-success me-1">
                            融合分數: {r.similarity?.toFixed(3) || "N/A"}
                          </span>
                          <span className="badge bg-primary me-1">
                            排名: {r.rank || "N/A"}
                          </span>
                          {r.original_scores && (
                            <span className="badge bg-info me-1">
                              原始分數:{" "}
                              {Object.entries(r.original_scores)
                                .map(
                                  ([level, score]) =>
                                    `${level}:${
                                      typeof score === "number"
                                        ? score.toFixed(2)
                                        : score
                                    }`
                                )
                                .join(", ")}
                            </span>
                          )}
                        </>
                      ) : (
                        `score=${r.score?.toFixed(3) || "N/A"}`
                      )}
                      <span className="ms-2">
                        doc={r.doc_id} idx={r.chunk_index}
                      </span>
                    </div>
                    {r.legal_structure && (
                      <div className="mt-1 mb-2">
                        <span className="badge bg-primary me-1">
                          {r.legal_structure.law_name}
                        </span>
                        {r.legal_structure.article && (
                          <span className="badge bg-secondary me-1">
                            {r.legal_structure.article}
                          </span>
                        )}
                        {r.legal_structure.item && (
                          <span className="badge bg-info me-1">
                            {r.legal_structure.item}
                          </span>
                        )}
                        {r.legal_structure.sub_item && (
                          <span className="badge bg-warning me-1">
                            {r.legal_structure.sub_item}
                          </span>
                        )}
                        <span className="badge bg-light text-dark">
                          {r.legal_structure.chunk_type}
                        </span>
                      </div>
                    )}
                    <pre
                      className="bg-light p-2 rounded"
                      style={{ whiteSpace: "pre-wrap" }}
                    >
                      {r.content}
                    </pre>
                  </li>
                ));
              })()}
            </ol>
            {/* 內嵌生成結果（如已生成） */}
            {answer && (
              <div className="mt-4">
                <h4 className="h6 mb-2">Answer</h4>
                <div
                  className="alert alert-secondary"
                  style={{ whiteSpace: "pre-wrap" }}
                >
                  {answer}
                </div>
                {legalReferences && legalReferences.length > 0 && (
                  <div className="mb-3">
                    <h5 className="h6">Legal References</h5>
                    <div>
                      {legalReferences.map((ref, i) => (
                        <span
                          key={i}
                          className="badge bg-light text-dark me-1 mb-1"
                        >
                          {ref}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {steps && steps.length > 0 && (
                  <div className="mb-3">
                    <h5 className="h6">Reasoning steps</h5>
                    <ol className="mb-0">
                      {steps.map((s: any, i: number) => (
                        <li key={i} className="small">
                          <strong>{s.type}:</strong> {s.text}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
