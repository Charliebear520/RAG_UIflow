import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";

export function RetrievePage() {
  const nav = useNavigate();
  const {
    canRetrieve,
    retrieve,
    hybridRetrieve,
    retrieval,
    generate,
    answer,
    steps,
    legalReferences,
  } = useRag();
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [useHybrid, setUseHybrid] = useState(false);

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
              checked={!useHybrid}
              onChange={() => setUseHybrid(false)}
            />
            <label className="form-check-label" htmlFor="vectorOnly">
              向量檢索 (Vector Search)
            </label>
          </div>
          <div className="form-check form-check-inline">
            <input
              className="form-check-input"
              type="radio"
              name="retrievalMethod"
              id="hybridRag"
              checked={useHybrid}
              onChange={() => setUseHybrid(true)}
            />
            <label className="form-check-label" htmlFor="hybridRag">
              HybridRAG (向量 + 法律規則)
            </label>
          </div>
        </div>

        <form
          className="row gy-2 gx-2 align-items-end"
          onSubmit={async (e) => {
            e.preventDefault();
            if (!canRetrieve) return;
            setBusy(true);
            if (useHybrid) {
              await hybridRetrieve(query, k);
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
              {useHybrid && <span className="badge bg-primary">HybridRAG</span>}
              {retrieval.embedding_provider && (
                <span className="badge bg-secondary ms-2">
                  {retrieval.embedding_provider}
                  {retrieval.embedding_model &&
                    ` (${retrieval.embedding_model})`}
                </span>
              )}
            </h3>

            {/* 顯示檢索指標 */}
            {retrieval.metrics && (
              <div className="alert alert-info mb-3">
                <h6 className="mb-2">檢索指標</h6>
                <div className="row">
                  <div className="col-md-3">
                    <span className="badge bg-success me-2">
                      P@{k}: {retrieval.metrics.p_at_k?.toFixed(3) || "N/A"}
                    </span>
                  </div>
                  <div className="col-md-3">
                    <span className="badge bg-warning me-2">
                      R@{k}: {retrieval.metrics.r_at_k?.toFixed(3) || "N/A"}
                    </span>
                  </div>
                  {retrieval.metrics.relevant_articles && (
                    <div className="col-md-6">
                      <small className="text-muted">
                        相關法條:{" "}
                        {retrieval.metrics.relevant_articles.join(", ")}
                      </small>
                    </div>
                  )}
                </div>
                {retrieval.metrics.matched_qa && (
                  <div className="mt-2">
                    <small className="text-muted">
                      匹配QA: {retrieval.metrics.matched_qa}
                    </small>
                  </div>
                )}
                {retrieval.metrics.note && (
                  <div className="mt-1">
                    <small className="text-warning">
                      {retrieval.metrics.note}
                    </small>
                  </div>
                )}
              </div>
            )}
            <ol>
              {retrieval.map((r: any) => (
                <li key={`${r.doc_id}-${r.chunk_index}`} className="mb-2">
                  <div className="small text-muted">
                    {useHybrid ? (
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
