import React, { useEffect, useState } from "react";
import { api } from "../lib/api";

type ChapterEntry = {
  chunk_id?: string;
  chapter_title?: string;
  chapter_key?: string;
  summary?: string;
  law_name?: string;
  section_count?: number;
  sections?: Array<{
    chunk_id?: string;
    section_title?: string;
    summary?: string;
  }>;
};

export function ExperimentalGroupEPage() {
  const [catalog, setCatalog] = useState<ChapterEntry[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [docOptions, setDocOptions] = useState<string[]>([]);
  const [documents, setDocuments] = useState<
    Array<{
      chunk_id?: string;
      law_name?: string;
      title?: string;
      summary?: string;
      last_updated?: string;
    }>
  >([]);
  const [catalogStats, setCatalogStats] = useState<any>(null);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [summaryMessage, setSummaryMessage] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [result, setResult] = useState<any | null>(null);
  const [loadingResult, setLoadingResult] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    loadCatalog();
  }, []);

  const loadCatalog = async (overrideDocId?: string) => {
    setCatalogLoading(true);
    setError(null);
    try {
      const effectiveDocId = overrideDocId || selectedDocId || undefined;
      const data = await api.getChapterCatalog({
        doc_id: effectiveDocId,
        max_sections: 5,
      });
      if (!effectiveDocId) {
        setDocOptions(data.stats?.doc_ids || []);
        const firstDocId = data.stats?.doc_ids?.[0];
        if (firstDocId) {
          setCatalogLoading(false);
          setSelectedDocId(firstDocId);
          await loadCatalog(firstDocId);
          return;
        }
      }
      setCatalog(data.catalog || []);
      setDocuments(data.documents || []);
      setCatalogStats(data.stats || null);
    } catch (err: any) {
      setError(err.message || "載入章節摘要失敗");
    } finally {
      setCatalogLoading(false);
    }
  };

  const handleSummarize = async () => {
    if (!selectedDocId) {
      setError("請先選擇目標 doc_id");
      return;
    }
    setSummarizing(true);
    setError(null);
    try {
      const res = await api.summarizeChapters({
        doc_id: selectedDocId,
        include_sections: true,
      });
      setSummaryMessage(
        `更新章節 ${res.chapters_updated || 0} 筆、節 ${
          res.sections_updated || 0
        } 筆`
      );
      await loadCatalog(selectedDocId);
    } catch (err: any) {
      setError(err.message || "章節摘要生成失敗");
    } finally {
      setSummarizing(false);
    }
  };

  const toggleChapter = (chunkId?: string) => {
    if (!chunkId) return;
    setExpanded((prev) => ({
      ...prev,
      [chunkId]: !prev[chunkId],
    }));
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDocId) {
      setError("請先選擇目標 doc_id");
      return;
    }
    if (!query.trim()) {
      setError("請輸入查詢");
      return;
    }
    setLoadingResult(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.experimentalGroupERetrieve({
        query,
        k,
        doc_id: selectedDocId,
      });
      if (!res) {
        setError("未取得實驗組E的檢索結果");
      }
      setResult(res || null);
    } catch (err: any) {
      setError(err.message || "實驗組E檢索失敗");
    } finally {
      setLoadingResult(false);
    }
  };

  const llmStage = result?.llm_stage || {};

  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-3">實驗組E：LLM章節導向 + 細節RAG</h2>
        <p className="text-muted">
          本頁面顯示章、節的永久摘要，並提供一鍵檢索流程：Gemini-2.5 Flash
          Thinking 先鎖定章節，再針對章節內的條/款/目進行embedding檢索。
        </p>

        {error && (
          <div className="alert alert-danger">
            <i className="bi bi-exclamation-triangle me-2"></i>
            {error}
          </div>
        )}

        <div className="mb-3">
          <label className="form-label fw-semibold">選擇目標 doc_id</label>
          {docOptions.length > 0 ? (
            <div className="d-flex gap-2">
              <select
                className="form-select"
                value={selectedDocId}
                onChange={(e) => {
                  const value = e.target.value;
                  setSelectedDocId(value);
                  loadCatalog(value || undefined);
                }}
              >
                <option value="">請選擇 doc_id</option>
                {docOptions.map((docId) => (
                  <option key={docId} value={docId}>
                    {docId}
                  </option>
                ))}
              </select>
              <button
                className="btn btn-outline-secondary"
                type="button"
                onClick={() => loadCatalog(selectedDocId || undefined)}
                disabled={!selectedDocId || catalogLoading}
              >
                重新載入
              </button>
            </div>
          ) : (
            <p className="text-muted small mb-0">
              尚無可用 doc_id，請先生成章節摘要。
            </p>
          )}
        </div>

        <div className="d-flex gap-2 mb-3">
          <button
            className="btn btn-outline-primary"
            onClick={() => loadCatalog(selectedDocId || undefined)}
            disabled={catalogLoading}
          >
            {catalogLoading ? "載入中..." : "重新載入摘要"}
          </button>
          <button
            className="btn btn-primary"
            onClick={handleSummarize}
            disabled={summarizing || !selectedDocId}
          >
            {summarizing ? "LLM 摘要生成中..." : "更新章/節摘要"}
          </button>
          {summaryMessage && (
            <span className="text-success small align-self-center">
              {summaryMessage}
            </span>
          )}
        </div>

        {catalogStats && (
          <div className="alert alert-info">
            <div className="d-flex flex-wrap gap-3">
              <div>
                <strong>法規數：</strong>
                {catalogStats.total_documents || 0}
              </div>
              <div>
                <strong>章節數：</strong>
                {catalogStats.total_chapters || 0}
              </div>
              <div>
                <strong>節數：</strong>
                {catalogStats.total_sections || 0}
              </div>
              <div>
                <strong>覆蓋法規：</strong>
                {(catalogStats.doc_ids || []).join(", ") || "N/A"}
              </div>
              <div>
                <strong>最後更新：</strong>
                {catalogStats.last_updated || "N/A"}
              </div>
            </div>
          </div>
        )}

        {documents.length > 0 && (
          <div className="mb-4">
            <h5>法規摘要（Document 層）</h5>
            <div className="list-group">
              {documents.map((doc) => (
                <div
                  className="list-group-item"
                  key={doc.chunk_id || doc.title}
                >
                  <div className="fw-bold">
                    {doc.title || doc.law_name || "法規"}
                    {doc.chunk_id && (
                      <small className="text-muted ms-2">
                        chunk_id: <code>{doc.chunk_id}</code>
                      </small>
                    )}
                  </div>
                  <div>{doc.summary || "（無摘要）"}</div>
                  <small className="text-muted">
                    {doc.law_name} • 更新時間：{doc.last_updated || "N/A"}
                  </small>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="mb-4">
          <h5>章節摘要</h5>
          {catalogLoading ? (
            <p className="text-muted">載入章節摘要中...</p>
          ) : catalog.length === 0 ? (
            <p className="text-muted">尚未建立任何章節摘要。</p>
          ) : (
            <div className="list-group">
              {catalog.map((chapter) => {
                const chunkId = chapter.chunk_id || chapter.chapter_key || "";
                const isExpanded = !!expanded[chunkId];
                return (
                  <div
                    className="list-group-item flex-column align-items-start"
                    key={chunkId}
                  >
                    <div className="d-flex justify-content-between w-100">
                      <div>
                        <h6 className="mb-1">
                          {chapter.chapter_title || "未命名章節"}
                        </h6>
                        <small className="text-muted">
                          {chapter.law_name} • chunk_id:{" "}
                          <code>{chunkId || "N/A"}</code>
                        </small>
                      </div>
                      <button
                        className="btn btn-sm btn-outline-secondary"
                        onClick={() => toggleChapter(chunkId)}
                        disabled={
                          !chapter.sections || chapter.sections.length === 0
                        }
                      >
                        {isExpanded ? "收合節摘要" : "展開節摘要"}
                      </button>
                    </div>
                    <p className="mb-1">{chapter.summary || "（無摘要）"}</p>
                    {chapter.section_count && (
                      <small className="text-muted">
                        節數：{chapter.section_count}（顯示最多5個）
                      </small>
                    )}
                    {isExpanded && chapter.sections && (
                      <div className="mt-2 ms-3">
                        {chapter.sections.map((section) => (
                          <div
                            key={section.chunk_id || section.section_title}
                            className="mb-2"
                          >
                            <div className="fw-bold">
                              {section.section_title || "節"}
                              {section.chunk_id && (
                                <small className="text-muted ms-2">
                                  chunk_id: <code>{section.chunk_id}</code>
                                </small>
                              )}
                            </div>
                            <div>{section.summary || "（無摘要）"}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <hr />

        <h5 className="mt-4">實驗組E檢索</h5>
        <form className="row g-2 align-items-end mb-3" onSubmit={handleSearch}>
          <div className="col-12 col-md-6">
            <label className="form-label">Query</label>
            <input
              className="form-control"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="輸入查詢..."
            />
          </div>
          <div className="col-auto">
            <label className="form-label">Top-K</label>
            <select
              className="form-select"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value, 10))}
            >
              {[3, 5, 10].map((val) => (
                <option key={val} value={val}>
                  {val}
                </option>
              ))}
            </select>
          </div>
          <div className="col-auto">
            <button
              className="btn btn-primary"
              type="submit"
              disabled={loadingResult || !selectedDocId}
            >
              {loadingResult ? "檢索中..." : "執行實驗組E"}
            </button>
          </div>
        </form>

        {result && (
          <div className="mt-4">
            <h6>LLM 思考與選章</h6>
            {llmStage.fallback_used && (
              <div className="alert alert-warning py-2">
                ⚠️ LLM 選章為 fallback 模式（使用預設章節）。
              </div>
            )}
            {llmStage.thinking && (
              <pre
                className="bg-light p-2 rounded"
                style={{ whiteSpace: "pre-wrap" }}
              >
                {llmStage.thinking}
              </pre>
            )}
            {llmStage.selection_details &&
              llmStage.selection_details.length > 0 && (
                <div className="mt-3">
                  <h6>選定章節</h6>
                  <ol>
                    {llmStage.selection_details.map(
                      (chapter: any, idx: number) => (
                        <li key={idx} className="mb-2">
                          <div className="fw-bold">
                            {chapter.chapter_title || "章節"}{" "}
                            {chapter.chunk_id && (
                              <small className="text-muted">
                                (<code>{chapter.chunk_id}</code>)
                              </small>
                            )}
                          </div>
                          {chapter.reason && (
                            <div className="small text-muted">
                              理由：{chapter.reason}
                            </div>
                          )}
                          {chapter.sections && chapter.sections.length > 0 && (
                            <ul className="mt-2">
                              {chapter.sections.map(
                                (sec: any, sIdx: number) => (
                                  <li key={sIdx}>
                                    {sec.section_title}
                                    {sec.chunk_id && (
                                      <small className="text-muted ms-1">
                                        (<code>{sec.chunk_id}</code>)
                                      </small>
                                    )}
                                    {sec.reason && (
                                      <div className="small text-muted">
                                        理由：{sec.reason}
                                      </div>
                                    )}
                                  </li>
                                )
                              )}
                            </ul>
                          )}
                        </li>
                      )
                    )}
                  </ol>
                </div>
              )}

            <h6 className="mt-4">Top {k} 檢索結果</h6>
            {result.fused_results && result.fused_results.length > 0 ? (
              <ol>
                {result.fused_results.map((item: any, idx: number) => (
                  <li
                    key={`${item.chunk_id || item.doc_id}-${idx}`}
                    className="mb-3"
                  >
                    <div className="small text-muted mb-1">
                      相似度: {(item.similarity || 0).toFixed(3)} • 層次:{" "}
                      {item.level || "N/A"} • chunk_id:{" "}
                      <code>{item.chunk_id || "N/A"}</code>
                    </div>
                    <pre
                      className="bg-light p-2 rounded"
                      style={{ whiteSpace: "pre-wrap" }}
                    >
                      {item.content}
                    </pre>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="text-muted">未取得任何檢索結果。</p>
            )}

            {result.level_contributions && (
              <div className="mt-4">
                <h6>層級貢獻</h6>
                <table className="table table-sm">
                  <thead>
                    <tr>
                      <th>層級</th>
                      <th>候選數</th>
                      <th>入選數</th>
                      <th>權重</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.level_contributions).map(
                      ([level, info]: [string, any]) => (
                        <tr key={level}>
                          <td>{level}</td>
                          <td>{info?.candidates ?? 0}</td>
                          <td>{info?.selected ?? 0}</td>
                          <td>{info?.weight ?? 1}</td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
