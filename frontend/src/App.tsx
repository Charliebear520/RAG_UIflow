import React, { useEffect, useMemo, useState } from "react";
import { api } from "./lib/api";

function Section({
  title,
  children,
  id,
}: {
  title: string;
  children: React.ReactNode;
  id?: string;
}) {
  return (
    <section
      id={id}
      style={{
        border: "1px solid #ddd",
        borderRadius: 8,
        padding: 16,
        marginBottom: 16,
      }}
    >
      <h2 style={{ marginTop: 0 }}>{title}</h2>
      {children}
    </section>
  );
}

export default function App() {
  const [docId, setDocId] = useState<string | null>(null);
  const [chunkMeta, setChunkMeta] = useState<{
    size: number;
    overlap: number;
    count: number;
  } | null>(null);
  const [embedProvider, setEmbedProvider] = useState<string | null>(null);
  const [retrieval, setRetrieval] = useState<any[] | null>(null);
  const [answer, setAnswer] = useState<string | null>(null);
  const [steps, setSteps] = useState<any[] | null>(null);
  const [jsonData, setJsonData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState("");

  // 恢復狀態與自動捲動到 Evaluate(beta)
  useEffect(() => {
    try {
      const storedDocId = localStorage.getItem("docId");
      if (storedDocId) setDocId(storedDocId);
    } catch {}

    if (typeof window !== "undefined" && window.location.hash === "#evaluate") {
      setTimeout(() => {
        const el = document.getElementById("evaluate");
        if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 50);
    }
  }, []);

  // Evaluate (beta)
  const [qaGold, setQaGold] = useState<any[] | null>(null);
  const [chunkTaskId, setChunkTaskId] = useState<string | null>(null);
  const [chunkTaskProgress, setChunkTaskProgress] = useState<number>(0);
  const [chunkingResults, setChunkingResults] = useState<any[] | null>(null);
  const [evalResults, setEvalResults] = useState<any[] | null>(null);
  const [evalSummary, setEvalSummary] = useState<any | null>(null);

  const canEmbed = useMemo(() => !!docId && !!chunkMeta, [docId, chunkMeta]);
  const canRetrieve = useMemo(() => !!embedProvider, [embedProvider]);
  const canGenerate = useMemo(() => !!retrieval, [retrieval]);

  const handleUpload = async (file: File) => {
    setLoading(true);
    try {
      const response = await api.convert(file);
      setJsonData(response);
      setFileName(file.name);
    } catch (error) {
      console.error("Error converting file:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: 960,
        margin: "0 auto",
        padding: 24,
        fontFamily: "Inter, system-ui, Avenir, Arial",
      }}
    >
      <h1>RAG Visualizer</h1>
      <p style={{ color: "#666" }}>
        Upload → Chunk → Embed → Retrieve → Generate
      </p>

      {/* 導航按鈕 */}
      <div style={{ marginBottom: 20 }}>
        <button
          style={{
            marginRight: 10,
            padding: "8px 16px",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
          }}
          onClick={() => window.location.reload()}
        >
          主頁面
        </button>
      </div>

      <Section title="1) Upload">
        <input
          type="file"
          onChange={async (e: React.ChangeEvent<HTMLInputElement>) => {
            const f = e.target.files?.[0];
            if (!f) return;
            const res = await api.uploadJson(f);
            setDocId(res.doc_id || res.id || null);
            try {
              localStorage.setItem("docId", res.doc_id);
            } catch {}
            setChunkMeta(null);
            setEmbedProvider(null);
            setRetrieval(null);
            setAnswer(null);
            setSteps(null);
          }}
        />
        {docId && (
          <p>
            Uploaded doc_id: <code>{docId}</code>
          </p>
        )}
      </Section>

      <Section title="2) Chunking">
        <ChunkForm
          disabled={!docId}
          onSubmit={async (size, overlap) => {
            if (!docId) return;
            const res = await api.chunk({
              doc_id: docId,
              chunk_size: size,
              overlap,
            });
            setChunkMeta({ size, overlap, count: res.num_chunks });
          }}
        />
        {chunkMeta && (
          <p>
            chunk_size={chunkMeta.size}, overlap={chunkMeta.overlap}, chunks=
            {chunkMeta.count}
          </p>
        )}
      </Section>

      <Section title="3) Embedding">
        <button
          disabled={!canEmbed}
          onClick={async () => {
            const res = await api.embed();
            setEmbedProvider(res.provider);
          }}
        >
          Compute Embeddings
        </button>
        {embedProvider && (
          <p>
            Provider: <code>{embedProvider}</code>
          </p>
        )}
      </Section>

      <Section title="4) Retrieval">
        <RetrieveForm
          disabled={!canRetrieve}
          onSubmit={async (query, k) => {
            const res = await api.retrieve({ query, k });
            setRetrieval(res.results);
          }}
        />
        {retrieval && (
          <div>
            <h3>Top Results</h3>
            <ol>
              {retrieval.map((r: any) => (
                <li key={`${r.doc_id}-${r.chunk_index}`}>
                  <div style={{ fontSize: 12, color: "#666" }}>
                    score={r.score.toFixed(3)} doc={r.doc_id} idx=
                    {r.chunk_index}
                  </div>
                  <pre style={{ whiteSpace: "pre-wrap" }}>{r.content}</pre>
                </li>
              ))}
            </ol>
          </div>
        )}
      </Section>

      <Section title="5) Generation">
        <GenerateForm
          disabled={!canGenerate}
          onSubmit={async (query, topK) => {
            const res = await api.generate({ query, top_k: topK });
            setAnswer(res.answer);
            setSteps(res.steps);
          }}
        />
        {answer && (
          <div>
            <h3>Answer</h3>
            <pre style={{ whiteSpace: "pre-wrap" }}>{answer}</pre>
          </div>
        )}
        {steps && (
          <div>
            <h3>Reasoning steps</h3>
            <ol>
              {steps.map((s, i) => (
                <li key={i}>
                  <strong>{s.type}:</strong> {s.text}
                </li>
              ))}
            </ol>
          </div>
        )}
      </Section>

      <Section id="evaluate" title="6) Evaluate (beta)">
        <div style={{ display: "grid", gap: 12 }}>
          <div>
            <label style={{ fontWeight: 600 }}>
              上傳 qa_gold.json（可選，用於之後策略評測）
            </label>
            <input
              type="file"
              accept="application/json,.json"
              onChange={async (e) => {
                const f = e.target.files?.[0];
                if (!f) return;
                try {
                  const txt = await f.text();
                  const parsed = JSON.parse(txt);
                  if (!Array.isArray(parsed))
                    throw new Error("qa_gold 應為陣列");
                  setQaGold(parsed);
                  alert(`已載入 qa_gold：${parsed.length} 條問題`);
                } catch (err: any) {
                  alert(`qa_gold 讀取失敗：${err.message || err}`);
                }
              }}
            />
          </div>

          <div>
            <button
              disabled={!docId}
              onClick={async () => {
                if (!docId) return;
                try {
                  // 以後端預設策略與尺寸啟動批量分塊
                  const base =
                    (import.meta as any).env.VITE_API_BASE_URL || "/api";
                  const body = new FormData();
                  // 使用固定三組大小與三組 overlap 作為示例，可改為 UI 多選
                  const payload = {
                    doc_id: docId,
                    strategies: ["fixed_size"],
                    chunk_sizes: [300, 500, 800],
                    overlap_ratios: [0.0, 0.1, 0.2],
                  };
                  const resp = await fetch(`${base}/chunk/multiple`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                  });
                  if (!resp.ok) throw new Error(await resp.text());
                  const data = await resp.json();
                  setChunkTaskId(data.task_id);
                  setChunkTaskProgress(0);
                  setChunkingResults(null);

                  // 開始輪詢狀態
                  const poll = async () => {
                    if (!data.task_id) return;
                    const s = await fetch(
                      `${base}/chunk/status/${data.task_id}`
                    );
                    const st = await s.json();
                    setChunkTaskProgress(st.progress || 0);
                    if (st.status === "completed") {
                      const r = await fetch(
                        `${base}/chunk/results/${data.task_id}`
                      );
                      if (r.ok) {
                        const jr = await r.json();
                        setChunkingResults(jr.results || []);
                      }
                    } else if (st.status === "failed") {
                      alert(`批量分塊失敗：${st.error || "unknown"}`);
                    } else {
                      setTimeout(poll, 1200);
                    }
                  };
                  setTimeout(poll, 800);
                } catch (e: any) {
                  alert(`啟動批量分塊失敗：${e.message || e}`);
                }
              }}
            >
              啟動批量分塊
            </button>
            {chunkTaskId && (
              <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>
                task_id: <code>{chunkTaskId}</code>，進度：
                {(chunkTaskProgress * 100).toFixed(0)}%
              </div>
            )}
          </div>

          {chunkingResults && (
            <div>
              <h3>分塊結果摘要</h3>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th
                      style={{
                        textAlign: "left",
                        borderBottom: "1px solid #ddd",
                      }}
                    >
                      strategy
                    </th>
                    <th
                      style={{
                        textAlign: "right",
                        borderBottom: "1px solid #ddd",
                      }}
                    >
                      chunk_size
                    </th>
                    <th
                      style={{
                        textAlign: "right",
                        borderBottom: "1px solid #ddd",
                      }}
                    >
                      overlap_ratio
                    </th>
                    <th
                      style={{
                        textAlign: "right",
                        borderBottom: "1px solid #ddd",
                      }}
                    >
                      chunk_count
                    </th>
                    <th
                      style={{
                        textAlign: "right",
                        borderBottom: "1px solid #ddd",
                      }}
                    >
                      avg_length
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {chunkingResults.map((r: any, i: number) => (
                    <tr key={i}>
                      <td
                        style={{ padding: 6, borderBottom: "1px solid #eee" }}
                      >
                        {r.strategy}
                      </td>
                      <td
                        style={{
                          padding: 6,
                          borderBottom: "1px solid #eee",
                          textAlign: "right",
                        }}
                      >
                        {r.config?.chunk_size}
                      </td>
                      <td
                        style={{
                          padding: 6,
                          borderBottom: "1px solid #eee",
                          textAlign: "right",
                        }}
                      >
                        {r.config?.overlap_ratio}
                      </td>
                      <td
                        style={{
                          padding: 6,
                          borderBottom: "1px solid #eee",
                          textAlign: "right",
                        }}
                      >
                        {r.chunk_count}
                      </td>
                      <td
                        style={{
                          padding: 6,
                          borderBottom: "1px solid #eee",
                          textAlign: "right",
                        }}
                      >
                        {(r.metrics?.avg_length ?? 0).toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p style={{ color: "#666", fontSize: 12, marginTop: 8 }}>
                提示：你可優先挑選 avg_length 接近目標長度（例如
                450~700）的組合，再到 Retrieve 設定 Hybrid 規則做檢索對比。
              </p>

              <div style={{ marginTop: 12 }}>
                <button
                  disabled={!qaGold}
                  onClick={async () => {
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
                    }
                  }}
                >
                  使用 qa_gold 計算 P@K / R@K
                </button>
              </div>

              {evalResults && (
                <div style={{ marginTop: 12 }}>
                  <h3>策略評測</h3>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        <th
                          style={{
                            textAlign: "left",
                            borderBottom: "1px solid #ddd",
                          }}
                        >
                          strategy
                        </th>
                        <th
                          style={{
                            textAlign: "right",
                            borderBottom: "1px solid #ddd",
                          }}
                        >
                          chunk_size
                        </th>
                        <th
                          style={{
                            textAlign: "right",
                            borderBottom: "1px solid #ddd",
                          }}
                        >
                          P@5
                        </th>
                        <th
                          style={{
                            textAlign: "right",
                            borderBottom: "1px solid #ddd",
                          }}
                        >
                          R@5
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {evalResults.map((r: any, i: number) => (
                        <tr key={i}>
                          <td
                            style={{
                              padding: 6,
                              borderBottom: "1px solid #eee",
                            }}
                          >
                            {r.strategy}
                          </td>
                          <td
                            style={{
                              padding: 6,
                              borderBottom: "1px solid #eee",
                              textAlign: "right",
                            }}
                          >
                            {r.config?.chunk_size}
                          </td>
                          <td
                            style={{
                              padding: 6,
                              borderBottom: "1px solid #eee",
                              textAlign: "right",
                            }}
                          >
                            {(r.metrics?.precision_at_k?.[5] ?? 0).toFixed(3)}
                          </td>
                          <td
                            style={{
                              padding: 6,
                              borderBottom: "1px solid #eee",
                              textAlign: "right",
                            }}
                          >
                            {(r.metrics?.recall_at_k?.[5] ?? 0).toFixed(3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {evalSummary && (
                    <div style={{ marginTop: 8, fontSize: 12, color: "#444" }}>
                      <div>
                        最佳（P@5）：
                        <code>
                          {evalSummary.best_by_p_at_5?.strategy} / size=
                          {evalSummary.best_by_p_at_5?.config?.chunk_size}
                        </code>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </Section>

      <Section title="Upload and Convert PDF">
        {!jsonData ? (
          <div>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => {
                const files = e.target.files;
                const file = files && files[0];
                if (file) handleUpload(file);
              }}
            />
            {loading && <p>正在將PDF轉換為JSON...</p>}
          </div>
        ) : (
          <div>
            <h2>Converted JSON</h2>
            <pre style={{ whiteSpace: "pre-wrap", wordWrap: "break-word" }}>
              {JSON.stringify(jsonData, null, 2)}
            </pre>
            <a
              href={`data:text/json;charset=utf-8,${encodeURIComponent(
                JSON.stringify(jsonData, null, 2)
              )}`}
              download={`${fileName.replace(/\.pdf$/, "")}.json`}
            >
              Download JSON
            </a>
            <button onClick={() => setJsonData(null)}>
              Upload Another File
            </button>
          </div>
        )}
      </Section>
    </div>
  );
}

function ChunkForm({
  disabled,
  onSubmit,
}: {
  disabled?: boolean;
  onSubmit: (size: number, overlap: number) => void;
}) {
  const [size, setSize] = useState(500);
  const [overlap, setOverlap] = useState(50);
  return (
    <form
      onSubmit={(e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        onSubmit(size, overlap);
      }}
    >
      <label>
        size
        <input
          type="number"
          value={size}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setSize(parseInt(e.target.value || "0") || 0)
          }
          style={{ marginLeft: 8, width: 100 }}
        />
      </label>
      <label style={{ marginLeft: 16 }}>
        overlap
        <input
          type="number"
          value={overlap}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setOverlap(parseInt(e.target.value || "0") || 0)
          }
          style={{ marginLeft: 8, width: 100 }}
        />
      </label>
      <button disabled={disabled} type="submit" style={{ marginLeft: 16 }}>
        Chunk
      </button>
    </form>
  );
}

function RetrieveForm({
  disabled,
  onSubmit,
}: {
  disabled?: boolean;
  onSubmit: (query: string, k: number) => void;
}) {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  return (
    <form
      onSubmit={(e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        onSubmit(query, k);
      }}
    >
      <input
        placeholder="Enter query"
        value={query}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
          setQuery(e.target.value)
        }
        style={{ width: 360 }}
      />
      <label style={{ marginLeft: 16 }}>
        top-k
        <input
          type="number"
          value={k}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setK(parseInt(e.target.value || "0") || 0)
          }
          style={{ marginLeft: 8, width: 80 }}
        />
      </label>
      <button disabled={disabled} type="submit" style={{ marginLeft: 16 }}>
        Search
      </button>
    </form>
  );
}

function GenerateForm({
  disabled,
  onSubmit,
}: {
  disabled?: boolean;
  onSubmit: (query: string, topK: number) => void;
}) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  return (
    <form
      onSubmit={(e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        onSubmit(query, topK);
      }}
    >
      <input
        placeholder="Question to answer"
        value={query}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
          setQuery(e.target.value)
        }
        style={{ width: 360 }}
      />
      <label style={{ marginLeft: 16 }}>
        top-k
        <input
          type="number"
          value={topK}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setTopK(parseInt(e.target.value || "0") || 0)
          }
          style={{ marginLeft: 8, width: 80 }}
        />
      </label>
      <button disabled={disabled} type="submit" style={{ marginLeft: 16 }}>
        Generate
      </button>
    </form>
  );
}
