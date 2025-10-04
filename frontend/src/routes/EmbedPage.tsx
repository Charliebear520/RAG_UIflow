import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";
import {
  EmbeddingTypeSelector,
  EmbeddingType,
} from "../components/EmbeddingTypeSelector";

export function EmbedPage() {
  const nav = useNavigate();
  const {
    canEmbed,
    embed,
    multiLevelEmbed,
    embedProvider,
    embedModel,
    embedDimension,
    selectedStrategy,
  } = useRag();
  const [busy, setBusy] = useState(false);
  const [embeddingType, setEmbeddingType] = useState<EmbeddingType>("standard");

  const handleEmbed = async () => {
    setBusy(true);
    try {
      if (embeddingType === "multi_level") {
        await multiLevelEmbed();
      } else {
        await embed();
      }
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

        {/* Embedding類型選擇 */}
        <div className="mb-4">
          <EmbeddingTypeSelector
            embeddingType={embeddingType}
            onEmbeddingTypeChange={setEmbeddingType}
            selectedStrategy={selectedStrategy}
          />
        </div>

        {/* 執行Embedding */}
        <div className="mb-4">
          <button
            disabled={!canEmbed || busy}
            className="btn btn-primary"
            onClick={handleEmbed}
          >
            {busy
              ? "Embedding…"
              : `計算 ${
                  embeddingType === "multi_level" ? "多層次" : "標準"
                } Embedding`}
          </button>
        </div>

        {/* 結果顯示 */}
        {embedProvider && (
          <div className="mt-3">
            <div className="alert alert-success">
              <h6 className="mb-2">
                ✅ {embeddingType === "multi_level" ? "多層次" : "標準"}{" "}
                Embedding 完成
              </h6>
              <div className="row">
                <div className="col-md-4">
                  <strong>類型:</strong>{" "}
                  <span className="badge bg-primary">
                    {embeddingType === "multi_level" ? "多層次" : "標準"}
                  </span>
                </div>
                <div className="col-md-4">
                  <strong>提供者:</strong>{" "}
                  <span className="badge bg-secondary">{embedProvider}</span>
                </div>
                <div className="col-md-4">
                  <strong>模型:</strong> <code>{embedModel}</code>
                </div>
              </div>
              {embedDimension && (
                <div className="row mt-2">
                  <div className="col-md-4">
                    <strong>維度:</strong>{" "}
                    <span className="badge bg-info">{embedDimension}</span>
                  </div>
                </div>
              )}
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
