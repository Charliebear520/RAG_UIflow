import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";

export function RetrievePage() {
  const nav = useNavigate();
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
  } = useRag();
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [retrievalMethod, setRetrievalMethod] = useState("vector");

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
              id="multiLevelRag"
              value="multi_level"
              checked={retrievalMethod === "multi_level"}
              onChange={(e) => setRetrievalMethod(e.target.value)}
            />
            <label className="form-check-label" htmlFor="multiLevelRag">
              多層次檢索 (Multi-Layered Retrieval) 📚
            </label>
          </div>
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
        </div>

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
              <strong>多層次檢索</strong>
              ：基於六個粒度級別（文件、章、節、條、項、款/目），智能選擇最合適的層次進行檢索
            </li>
            <li>
              <strong>多層次融合檢索</strong>
              ：從所有六個粒度級別檢索並融合結果，提供最全面的檢索效果
            </li>
          </ul>
        </div>

        <form
          className="row gy-2 gx-2 align-items-end"
          onSubmit={async (e) => {
            e.preventDefault();
            if (!canRetrieve) return;
            setBusy(true);
            if (retrievalMethod === "hybrid") {
              await hybridRetrieve(query, k);
            } else if (retrievalMethod === "multi_level") {
              await multiLevelRetrieve(query, k);
            } else if (retrievalMethod === "multi_level_fusion") {
              await multiLevelFusionRetrieve(query, k);
            } else {
              await retrieve(query, k);
            }
            setBusy(false);
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
              disabled={!canRetrieve || busy}
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
            <ol>
              {retrieval.map((r: any) => (
                <li key={`${r.doc_id}-${r.chunk_index}`} className="mb-2">
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
                          置信度: {r.metadata?.confidence?.toFixed(2) || "N/A"}
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
                                  `${level}:${score?.toFixed(2)}`
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
              ))}
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
