import React, { useMemo, useState } from "react";
import { useRag } from "../lib/ragStore";

export function EvaluatePage() {
  const { docId, chunkingResults, selectedStrategy, embedProvider, retrieval } =
    useRag();
  const [qaGold, setQaGold] = useState<any[] | null>(null);
  const [evalResults, setEvalResults] = useState<any[] | null>(null);
  const [evalSummary, setEvalSummary] = useState<any | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  const canStart = useMemo(
    () => !!docId && !!chunkingResults && !!embedProvider,
    [docId, chunkingResults, embedProvider]
  );

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h2 className="h5 mb-0">Evaluate (beta)</h2>
              <p className="text-muted mb-0">
                使用您選擇的embedding和檢索策略，上傳 qa_gold 計算真實的 P@K /
                R@K 評測指標。
              </p>
            </div>
            <div className="card-body">
              {!docId && (
                <div className="alert alert-warning" role="alert">
                  請先於 Upload 頁上傳文件，產生 doc_id 後再進行評測。
                </div>
              )}

              {docId && !chunkingResults && (
                <div className="alert alert-warning" role="alert">
                  請先於 Chunk 頁完成分塊操作。
                </div>
              )}

              {docId && chunkingResults && !embedProvider && (
                <div className="alert alert-warning" role="alert">
                  請先於 Embed 頁完成向量化操作。
                </div>
              )}

              {docId && chunkingResults && embedProvider && (
                <div className="alert alert-success" role="alert">
                  ✅ 已準備就緒：分塊策略 "{selectedStrategy}"，向量化提供者 "
                  {embedProvider}"，檢索策略 "{retrieval}
                  "。評測將使用真實的embedding和HybridRAG檢索。
                </div>
              )}

              <div className="mb-4">
                <label className="form-label fw-bold">上傳 qa_gold.json</label>
                <input
                  type="file"
                  accept="application/json,.json"
                  className="form-control"
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    try {
                      const txt = await f.text();
                      const parsed = JSON.parse(txt);
                      if (!Array.isArray(parsed))
                        throw new Error("qa_gold 應為陣列");
                      setQaGold(parsed);
                    } catch (err: any) {
                      alert(`qa_gold 讀取失敗：${err.message || err}`);
                    }
                  }}
                  disabled={!canStart}
                />
                {qaGold && (
                  <small className="text-muted d-block mt-1">
                    已載入 {qaGold.length} 條問題
                  </small>
                )}
              </div>

              {chunkingResults && (
                <div className="mb-4">
                  <h6>當前分塊結果</h6>
                  <div className="table-responsive">
                    <table className="table table-sm">
                      <thead>
                        <tr>
                          <th>strategy</th>
                          <th className="text-end">chunk_size</th>
                          <th className="text-end">overlap_ratio</th>
                          <th className="text-end">chunk_count</th>
                          <th className="text-end">avg_length</th>
                        </tr>
                      </thead>
                      <tbody>
                        {chunkingResults.map((r: any, i: number) => (
                          <tr key={i}>
                            <td>{r.strategy}</td>
                            <td className="text-end">{r.config?.chunk_size}</td>
                            <td className="text-end">
                              {r.config?.overlap_ratio}
                            </td>
                            <td className="text-end">{r.chunk_count}</td>
                            <td className="text-end">
                              {(r.metrics?.avg_length ?? 0).toFixed(1)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {chunkingResults && qaGold && (
                <div className="mb-4">
                  <button
                    className="btn btn-success"
                    disabled={!canStart || isEvaluating}
                    onClick={async () => {
                      if (isEvaluating) return;

                      setIsEvaluating(true);
                      try {
                        const base =
                          (import.meta as any).env.VITE_API_BASE_URL || "/api";
                        const payload = {
                          doc_id: docId,
                          qa_gold: qaGold || [],
                          chunking_results: chunkingResults || [],
                          k_values: [1, 3, 5, 10],
                        };
                        const resp = await fetch(`${base}/evaluate/gold`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify(payload),
                        });
                        if (!resp.ok) throw new Error(await resp.text());
                        const data = await resp.json();
                        setEvalResults(data.results || []);
                        setEvalSummary(data.summary || null);
                      } catch (e: any) {
                        alert(`策略評測失敗：${e.message || e}`);
                      } finally {
                        setIsEvaluating(false);
                      }
                    }}
                  >
                    {isEvaluating ? (
                      <>
                        <span
                          className="spinner-border spinner-border-sm me-2"
                          role="status"
                          aria-hidden="true"
                        ></span>
                        評測進行中...
                      </>
                    ) : (
                      "使用真實embedding和檢索策略計算 P@K / R@K"
                    )}
                  </button>

                  {isEvaluating && (
                    <div className="mt-2">
                      <div
                        className="alert alert-info d-flex align-items-center"
                        role="alert"
                      >
                        <span
                          className="spinner-border spinner-border-sm me-2"
                          role="status"
                          aria-hidden="true"
                        ></span>
                        <div>
                          <strong>評測進行中...</strong>
                          <br />
                          <small>
                            正在使用 {embedProvider} embedding 和 {retrieval}{" "}
                            檢索策略計算評估指標，請稍候...
                          </small>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {evalResults && (
                <div className="mb-2">
                  <h6>策略評測</h6>
                  <div className="table-responsive">
                    <table className="table table-sm">
                      <thead>
                        <tr>
                          <th>strategy</th>
                          <th className="text-end">chunk_size</th>
                          <th className="text-end">P@5</th>
                          <th className="text-end">R@5</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(evalResults || []).map((r: any, i: number) => (
                          <tr key={i}>
                            <td>{r.strategy}</td>
                            <td className="text-end">{r.config?.chunk_size}</td>
                            <td className="text-end">
                              {(r.metrics?.precision_at_k?.[5] ?? 0).toFixed(3)}
                            </td>
                            <td className="text-end">
                              {(r.metrics?.recall_at_k?.[5] ?? 0).toFixed(3)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {evalSummary && (
                    <div className="small text-muted">
                      最佳（P@5）：
                      <code>
                        {evalSummary.best_by_p_at_5?.strategy} / size=
                        {evalSummary.best_by_p_at_5?.config?.chunk_size}
                      </code>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
