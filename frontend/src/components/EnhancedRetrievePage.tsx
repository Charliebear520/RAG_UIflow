import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";

export function EnhancedRetrievePage() {
  const nav = useNavigate();
  const {
    canRetrieve,
    retrieve,
    hybridRetrieve,
    hierarchicalRetrieve,
    retrieval,
    generate,
    answer,
    steps,
    legalReferences,
  } = useRag();

  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [retrievalMethod, setRetrievalMethod] = useState("adaptive");

  // 新增的檢索方法
  const [conceptGraphRetrieval, setConceptGraphRetrieval] = useState(null);
  const [adaptiveRetrieval, setAdaptiveRetrieval] = useState(null);

  // 概念圖構建狀態
  const [conceptGraphBuilt, setConceptGraphBuilt] = useState(false);
  const [buildingGraph, setBuildingGraph] = useState(false);

  const buildConceptGraph = async () => {
    setBuildingGraph(true);
    try {
      const response = await fetch("/api/build-concept-graph", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await response.json();

      if (data.status === "success") {
        setConceptGraphBuilt(true);
        alert(
          `概念圖構建成功！\n節點: ${data.statistics.node_count}\n邊: ${data.statistics.edge_count}\n概念: ${data.statistics.concept_count}`
        );
      } else {
        alert("概念圖構建失敗");
      }
    } catch (error) {
      console.error("構建概念圖錯誤:", error);
      alert("概念圖構建錯誤");
    } finally {
      setBuildingGraph(false);
    }
  };

  const handleRetrieve = async () => {
    if (!canRetrieve) return;
    setBusy(true);

    try {
      let response;

      switch (retrievalMethod) {
        case "vector":
          await retrieve(query, k);
          break;
        case "hybrid":
          await hybridRetrieve(query, k);
          break;
        case "hierarchical":
          await hierarchicalRetrieve(query, k);
          break;
        case "concept_graph":
          if (!conceptGraphBuilt) {
            alert("請先構建概念圖");
            return;
          }
          response = await fetch("/api/concept-graph-retrieve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, k }),
          });
          const conceptData = await response.json();
          setConceptGraphRetrieval(conceptData);
          break;
        case "adaptive":
          response = await fetch("/api/adaptive-retrieve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, k }),
          });
          const adaptiveData = await response.json();
          setAdaptiveRetrieval(adaptiveData);
          break;
        default:
          await retrieve(query, k);
      }
    } catch (error) {
      console.error("檢索錯誤:", error);
    } finally {
      setBusy(false);
    }
  };

  const getCurrentResults = () => {
    switch (retrievalMethod) {
      case "concept_graph":
        return conceptGraphRetrieval;
      case "adaptive":
        return adaptiveRetrieval;
      default:
        return { results: retrieval };
    }
  };

  const currentResults = getCurrentResults();

  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-3">增強版檢索 (Enhanced Retrieve)</h2>

        {/* 概念圖構建 */}
        <div className="mb-3">
          <div className="d-flex align-items-center gap-2">
            <button
              className="btn btn-outline-primary"
              onClick={buildConceptGraph}
              disabled={buildingGraph || conceptGraphBuilt}
            >
              {buildingGraph
                ? "構建中..."
                : conceptGraphBuilt
                ? "概念圖已構建"
                : "構建概念圖"}
            </button>
            {conceptGraphBuilt && (
              <span className="badge bg-success">概念圖就緒</span>
            )}
          </div>
        </div>

        {/* 檢索方法選擇 */}
        <div className="mb-3">
          <h6>檢索方法:</h6>
          <div className="row">
            <div className="col-md-4">
              <div className="form-check">
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
                  向量檢索 (Vector)
                </label>
              </div>
            </div>
            <div className="col-md-4">
              <div className="form-check">
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
                  HybridRAG
                </label>
              </div>
            </div>
            <div className="col-md-4">
              <div className="form-check">
                <input
                  className="form-check-input"
                  type="radio"
                  name="retrievalMethod"
                  id="hierarchicalRag"
                  value="hierarchical"
                  checked={retrievalMethod === "hierarchical"}
                  onChange={(e) => setRetrievalMethod(e.target.value)}
                />
                <label className="form-check-label" htmlFor="hierarchicalRag">
                  多層次檢索
                </label>
              </div>
            </div>
            <div className="col-md-4">
              <div className="form-check">
                <input
                  className="form-check-input"
                  type="radio"
                  name="retrievalMethod"
                  id="conceptGraph"
                  value="concept_graph"
                  checked={retrievalMethod === "concept_graph"}
                  onChange={(e) => setRetrievalMethod(e.target.value)}
                />
                <label className="form-check-label" htmlFor="conceptGraph">
                  概念圖檢索
                </label>
              </div>
            </div>
            <div className="col-md-4">
              <div className="form-check">
                <input
                  className="form-check-input"
                  type="radio"
                  name="retrievalMethod"
                  id="adaptiveRag"
                  value="adaptive"
                  checked={retrievalMethod === "adaptive"}
                  onChange={(e) => setRetrievalMethod(e.target.value)}
                />
                <label className="form-check-label" htmlFor="adaptiveRag">
                  自適應檢索
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* 查詢表單 */}
        <form
          className="row gy-2 gx-2 align-items-end"
          onSubmit={async (e) => {
            e.preventDefault();
            await handleRetrieve();
          }}
        >
          <div className="col-12 col-md-6">
            <label className="form-label">查詢 (Query)</label>
            <input
              className="form-control"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="輸入法律問題..."
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
              disabled={!canRetrieve || busy}
              className="btn btn-primary"
              type="submit"
            >
              {busy ? "檢索中..." : "檢索"}
            </button>
          </div>
        </form>

        {/* 檢索結果 */}
        {currentResults && currentResults.results && (
          <div className="mt-3">
            <h3 className="h6">
              檢索結果{" "}
              <span
                className={`badge ${
                  retrievalMethod === "concept_graph"
                    ? "bg-info"
                    : retrievalMethod === "adaptive"
                    ? "bg-warning"
                    : retrievalMethod === "hybrid"
                    ? "bg-primary"
                    : retrievalMethod === "hierarchical"
                    ? "bg-success"
                    : "bg-secondary"
                }`}
              >
                {retrievalMethod === "concept_graph"
                  ? "概念圖檢索"
                  : retrievalMethod === "adaptive"
                  ? "自適應檢索"
                  : retrievalMethod === "hybrid"
                  ? "HybridRAG"
                  : retrievalMethod === "hierarchical"
                  ? "多層次檢索"
                  : "向量檢索"}
              </span>
            </h3>

            {/* 檢索指標 */}
            {currentResults.metrics && (
              <div className="alert alert-info mb-3">
                <h6 className="mb-2">檢索指標</h6>
                <div className="row">
                  <div className="col-md-2">
                    <span className="badge bg-success me-2">
                      P@{k}:{" "}
                      {currentResults.metrics.p_at_k?.toFixed(3) || "N/A"}
                    </span>
                  </div>
                  <div className="col-md-2">
                    <span className="badge bg-warning me-2">
                      R@{k}:{" "}
                      {currentResults.metrics.r_at_k?.toFixed(3) || "N/A"}
                    </span>
                  </div>
                  <div className="col-md-2">
                    <span className="badge bg-secondary me-2">
                      相關: {currentResults.metrics.relevant_chunks_count || 0}
                    </span>
                  </div>
                  <div className="col-md-6">
                    <span className="badge bg-info me-2">
                      {currentResults.metrics.note || "檢索完成"}
                    </span>
                  </div>
                </div>

                {/* 概念圖分析 */}
                {currentResults.metrics.concept_graph_analysis && (
                  <div className="mt-2">
                    <h6 className="mb-1">概念圖分析</h6>
                    <div className="row">
                      <div className="col-md-4">
                        <small className="text-muted">
                          推理路徑:{" "}
                          {
                            currentResults.metrics.concept_graph_analysis
                              .reasoning_paths_used
                          }{" "}
                          條
                        </small>
                      </div>
                      <div className="col-md-4">
                        <small className="text-muted">
                          概念匹配:{" "}
                          {
                            currentResults.metrics.concept_graph_analysis
                              .concept_matches
                          }{" "}
                          個
                        </small>
                      </div>
                      <div className="col-md-4">
                        <small className="text-muted">
                          平均推理分數:{" "}
                          {currentResults.metrics.concept_graph_analysis.avg_reasoning_score?.toFixed(
                            3
                          )}
                        </small>
                      </div>
                    </div>
                  </div>
                )}

                {/* 自適應分析 */}
                {currentResults.metrics.adaptive_analysis && (
                  <div className="mt-2">
                    <h6 className="mb-1">自適應分析</h6>
                    <div className="row">
                      <div className="col-md-6">
                        <small className="text-muted">
                          使用策略:{" "}
                          {currentResults.metrics.adaptive_analysis.strategies_used?.join(
                            ", "
                          )}
                        </small>
                      </div>
                      <div className="col-md-6">
                        <small className="text-muted">
                          融合分數:{" "}
                          {currentResults.metrics.adaptive_analysis.avg_fused_score?.toFixed(
                            3
                          )}
                        </small>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 結果列表 */}
            <ol>
              {currentResults.results.map((r: any, index: number) => (
                <li
                  key={`${r.doc_id || index}-${r.chunk_index || index}`}
                  className="mb-2"
                >
                  <div className="small text-muted">
                    {retrievalMethod === "concept_graph" ? (
                      <>
                        <span className="badge bg-info me-1">
                          推理分數: {r.reasoning_score?.toFixed(3) || "N/A"}
                        </span>
                        <span className="badge bg-primary me-1">
                          概念: {r.concept_name || "N/A"}
                        </span>
                        <span className="badge bg-secondary me-1">
                          類型: {r.concept_type || "N/A"}
                        </span>
                      </>
                    ) : retrievalMethod === "adaptive" ? (
                      <>
                        <span className="badge bg-warning me-1">
                          融合分數: {r.fused_score?.toFixed(3) || "N/A"}
                        </span>
                        <span className="badge bg-info me-1">
                          策略: {r.contributing_strategies?.join(",") || "N/A"}
                        </span>
                        <span className="badge bg-success me-1">
                          數量: {r.strategy_count || 0}
                        </span>
                      </>
                    ) : retrievalMethod === "hybrid" ? (
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
                    ) : (
                      `分數: ${r.score?.toFixed(3) || "N/A"}`
                    )}
                    <span className="ms-2">
                      doc={r.doc_id || "N/A"} idx={r.chunk_index || "N/A"}
                    </span>
                    {r.metadata && (
                      <span className="ms-2">
                        {r.metadata.category &&
                          `category=${r.metadata.category}`}
                        {r.metadata.article_label &&
                          ` article=${r.metadata.article_label}`}
                      </span>
                    )}
                  </div>
                  <p className="mb-0">{r.content}</p>
                </li>
              ))}
            </ol>
          </div>
        )}

        {/* 生成答案 */}
        {answer && (
          <div className="mt-3">
            <h3 className="h6">生成答案</h3>
            <p>{answer}</p>
            {legalReferences && legalReferences.length > 0 && (
              <div className="mt-2">
                <h6 className="mb-1">法律參考</h6>
                <ul>
                  {legalReferences.map((ref, index) => (
                    <li key={index}>{ref}</li>
                  ))}
                </ul>
              </div>
            )}
            {steps && steps.length > 0 && (
              <div className="mt-2">
                <h6 className="mb-1">生成步驟</h6>
                <ol>
                  {steps.map((step, index) => (
                    <li key={index}>{step}</li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
