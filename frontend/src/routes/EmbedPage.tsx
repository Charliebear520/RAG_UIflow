import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";

export function EmbedPage() {
  const nav = useNavigate();
  const { canEmbed, embed, embedProvider, embedModel, embedDimension } =
    useRag();
  const [busy, setBusy] = useState(false);

  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-3">Embed</h2>
        <button
          disabled={!canEmbed || busy}
          className="btn btn-primary"
          onClick={async () => {
            setBusy(true);
            await embed();
            setBusy(false);
          }}
        >
          {busy ? "Embedding…" : "Compute embeddings"}
        </button>
        {embedProvider && (
          <div className="mt-3">
            <div className="alert alert-success">
              <h6 className="mb-2">✅ Embedding 完成</h6>
              <div className="row">
                <div className="col-md-4">
                  <strong>提供者:</strong>{" "}
                  <span className="badge bg-primary">{embedProvider}</span>
                </div>
                <div className="col-md-4">
                  <strong>模型:</strong> <code>{embedModel}</code>
                </div>
                <div className="col-md-4">
                  <strong>維度:</strong>{" "}
                  <span className="badge bg-info">{embedDimension}</span>
                </div>
              </div>
            </div>
            <div>
              <button
                className="btn btn-success btn-sm"
                onClick={() => nav("/retrieve")}
              >
                前往 Retrieve 頁面
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
