import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useRag } from "../lib/ragStore";
import { api } from "../lib/api";

export function RetrievePage() {
  const nav = useNavigate();
  const location = useLocation();
  const {
    canRetrieve,
    retrieve,
    hybridRetrieve,
    hierarchicalRetrieve,
    multiLevelRetrieve,
    multiLevelFusionRetrieve,
    retrieval,
    generate,
    answer,
    steps,
    legalReferences,
    embedProvider,
    setEmbedProvider,
  } = useRag();
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [retrievalMethod, setRetrievalMethod] = useState("vector");
  const [selectedDatabase, setSelectedDatabase] = useState<any>(null);
  const [databaseMessage, setDatabaseMessage] = useState<string | null>(null);

  // E/C/U標註相關狀態
  const [annotationMode, setAnnotationMode] = useState(false);
  const [annotations, setAnnotations] = useState<Record<number, string>>({});
  const [annotationStats, setAnnotationStats] = useState<any>(null);

  // E/C/U標註處理函數
  const handleAnnotationToggle = (index: number, label: "E" | "C" | "U") => {
    setAnnotations((prev) => ({
      ...prev,
      [index]: prev[index] === label ? "" : label, // 切換選擇
    }));
  };

  const handleSubmitAnnotations = async () => {
    if (!retrieval || !query) return;

    try {
      // 提交標註
      const result = await api.saveAnnotations({
        query: query,
        results: retrieval,
        annotations: annotations,
      });

      // 計算並顯示統計結果
      const stats = calculateAnnotationStats(annotations, retrieval.length);
      setAnnotationStats(stats);

      console.log("標註已保存:", result);
    } catch (error) {
      console.error("保存標註失敗:", error);
      alert("保存標註失敗，請重試");
    }
  };

  // 統計計算
  const calculateAnnotationStats = (
    annotations: Record<number, string>,
    total: number
  ) => {
    const eCount = Object.values(annotations).filter((a) => a === "E").length;
    const cCount = Object.values(annotations).filter((a) => a === "C").length;
    const uCount = Object.values(annotations).filter((a) => a === "U").length;

    return {
      total: total,
      essential: {
        count: eCount,
        percentage: ((eCount / total) * 100).toFixed(1),
      },
      complementary: {
        count: cCount,
        percentage: ((cCount / total) * 100).toFixed(1),
      },
      unnecessary: {
        count: uCount,
        percentage: ((uCount / total) * 100).toFixed(1),
      },
      relevant: {
        count: eCount + cCount,
        percentage: (((eCount + cCount) / total) * 100).toFixed(1),
      },
    };
  };

  // 處理從embedding資料庫列表跳轉過來的情況
  useEffect(() => {
    if (location.state?.selectedDatabase) {
      const database = location.state.selectedDatabase;
      setSelectedDatabase(database);
      setDatabaseMessage(
        location.state.message || `已選擇embedding資料庫: ${database.name}`
      );

      // 設置embedProvider以啟用檢索功能
      if (database.provider) {
        setEmbedProvider(database.provider);
      }

      // 根據資料庫類型自動選擇合適的檢索方法
      if (database.type === "multi_level") {
        setRetrievalMethod("multi_level_fusion");
      } else if (database.type === "standard") {
        setRetrievalMethod("vector");
      }
    }
  }, [location.state, setEmbedProvider]);

  // 僅保留：標準、HybridRAG、多層次融合檢索

  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-3">Retrieve</h2>

        {/* 顯示選中的embedding資料庫信息 */}
        {selectedDatabase && (
          <div className="alert alert-info mb-3">
            <div className="d-flex justify-content-between align-items-start">
              <div>
                <h6 className="alert-heading mb-2">
                  <i className="bi bi-database me-2"></i>
                  當前使用的Embedding資料庫
                </h6>
                <div className="row text-muted small">
                  <div className="col-sm-6">
                    <strong>名稱:</strong> {selectedDatabase.name}
                  </div>
                  <div className="col-sm-6">
                    <strong>類型:</strong>{" "}
                    {selectedDatabase.type === "multi_level"
                      ? "多層次Embedding"
                      : "標準Embedding"}
                  </div>
                  <div className="col-sm-6">
                    <strong>Embedding:</strong> {selectedDatabase.provider} (
                    {selectedDatabase.model})
                  </div>
                  <div className="col-sm-6">
                    <strong>向量數量:</strong>{" "}
                    {selectedDatabase.num_vectors.toLocaleString()}
                  </div>
                  <div className="col-sm-6">
                    <strong>分塊方式:</strong>{" "}
                    {selectedDatabase.chunking_strategy}
                  </div>
                  <div className="col-sm-6">
                    <strong>文檔:</strong>{" "}
                    {selectedDatabase.documents
                      .map((d: any) => d.filename)
                      .join(", ")}
                  </div>
                </div>
              </div>
              <button
                className="btn btn-outline-secondary btn-sm"
                onClick={() => {
                  setSelectedDatabase(null);
                  setDatabaseMessage(null);
                }}
                title="清除選擇"
              >
                <i className="bi bi-x"></i>
              </button>
            </div>
          </div>
        )}

        {databaseMessage && (
          <div
            className="alert alert-success alert-dismissible fade show mb-3"
            role="alert"
          >
            <i className="bi bi-check-circle me-2"></i>
            {databaseMessage}
            <button
              type="button"
              className="btn-close"
              onClick={() => setDatabaseMessage(null)}
            ></button>
          </div>
        )}

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
          <div className="form-check form-check-inline"></div>
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
          <div className="form-check form-check-inline"></div>
        </div>

        {/* 保留的方法：Standard / HybridRAG / Multi-Layered Fusion */}

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

        {/* 移除 HopRAG 使用指導 */}

        {/* 移除 HopRAG 管理面板 */}

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
              disabled={busy || !canRetrieve}
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
              disabled={!retrieval || retrieval.length === 0 || busy}
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

        {retrieval && (
          <div className="mt-3">
            <h3 className="h6">
              Top results{" "}
              {retrievalMethod === "hybrid" && (
                <span className="badge bg-primary">HybridRAG</span>
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

            {/* 標註控制面板 */}
            {retrieval && (
              <div className="card mb-3 border-primary">
                <div className="card-body">
                  <div className="d-flex justify-content-between align-items-center">
                    <div>
                      <h5 className="mb-1">E/C/U標註模式</h5>
                      <small className="text-muted">
                        E=Essential(必需) | C=Complementary(補充) |
                        U=Unnecessary(不必要)
                      </small>
                    </div>
                    <div>
                      <button
                        className={`btn ${
                          annotationMode ? "btn-primary" : "btn-outline-primary"
                        } me-2`}
                        onClick={() => {
                          setAnnotationMode(!annotationMode);
                          if (!annotationMode) {
                            setAnnotations({});
                            setAnnotationStats(null);
                          }
                        }}
                      >
                        {annotationMode ? "關閉標註模式" : "開啟標註模式"}
                      </button>
                      {annotationMode && (
                        <button
                          className="btn btn-success"
                          disabled={
                            Object.keys(annotations).length !== retrieval.length
                          }
                          onClick={handleSubmitAnnotations}
                        >
                          提交標註 ({Object.keys(annotations).length}/
                          {retrieval.length})
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 標註統計結果 */}
            {annotationStats && (
              <div className="alert alert-info">
                <h6 className="alert-heading">標註統計結果</h6>
                <div className="row">
                  <div className="col-md-3">
                    <strong>Essential:</strong>{" "}
                    {annotationStats.essential.count} (
                    {annotationStats.essential.percentage}%)
                  </div>
                  <div className="col-md-3">
                    <strong>Complementary:</strong>{" "}
                    {annotationStats.complementary.count} (
                    {annotationStats.complementary.percentage}%)
                  </div>
                  <div className="col-md-3">
                    <strong>Unnecessary:</strong>{" "}
                    {annotationStats.unnecessary.count} (
                    {annotationStats.unnecessary.percentage}%)
                  </div>
                  <div className="col-md-3">
                    <strong>Relevant (E+C):</strong>{" "}
                    {annotationStats.relevant.count} (
                    {annotationStats.relevant.percentage}%)
                  </div>
                </div>
              </div>
            )}

            <ol>
              {(() => {
                // 確定要顯示的結果數據
                let resultsToShow = retrieval || null;

                console.log("🎯 結果顯示邏輯:", {
                  retrievalMethod,
                  hasRetrieval: !!retrieval,
                  retrievalLength: retrieval?.length,
                  resultsToShow: resultsToShow?.length,
                });

                return resultsToShow?.map((r: any, index: number) => (
                  <li
                    key={r.node_id || `${r.doc_id}-${r.chunk_index || index}`}
                    className="mb-2"
                  >
                    <div className="d-flex justify-content-between align-items-start">
                      <div className="flex-grow-1">
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
                            {r.hierarchical_description ? (
                              <span className="text-primary">
                                {r.hierarchical_description}
                              </span>
                            ) : (
                              `doc=${r.doc_id} idx=${r.chunk_index}`
                            )}
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
                      </div>

                      {/* 標註按鈕（僅在標註模式下顯示） */}
                      {annotationMode && (
                        <div className="ms-3" style={{ minWidth: "200px" }}>
                          <div className="btn-group-vertical" role="group">
                            <button
                              className={`btn btn-sm ${
                                annotations[index] === "E"
                                  ? "btn-success"
                                  : "btn-outline-success"
                              }`}
                              onClick={() => handleAnnotationToggle(index, "E")}
                            >
                              {annotations[index] === "E" ? "✓ " : ""}Essential
                            </button>
                            <button
                              className={`btn btn-sm ${
                                annotations[index] === "C"
                                  ? "btn-info"
                                  : "btn-outline-info"
                              }`}
                              onClick={() => handleAnnotationToggle(index, "C")}
                            >
                              {annotations[index] === "C" ? "✓ " : ""}
                              Complementary
                            </button>
                            <button
                              className={`btn btn-sm ${
                                annotations[index] === "U"
                                  ? "btn-secondary"
                                  : "btn-outline-secondary"
                              }`}
                              onClick={() => handleAnnotationToggle(index, "U")}
                            >
                              {annotations[index] === "U" ? "✓ " : ""}
                              Unnecessary
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
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
