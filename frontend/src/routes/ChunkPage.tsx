import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";

// 分塊策略類型定義
type ChunkStrategy =
  | "fixed_size"
  | "hierarchical"
  | "adaptive"
  | "hybrid"
  | "semantic";

// 策略參數接口
interface ChunkParams {
  fixed_size: {
    chunk_size: number;
    overlap: number;
  };
  hierarchical: {
    max_chunk_size: number;
    min_chunk_size: number;
    overlap: number;
    level_depth: number;
  };
  adaptive: {
    target_size: number;
    tolerance: number;
    overlap: number;
    semantic_threshold: number;
  };
  hybrid: {
    primary_size: number;
    secondary_size: number;
    overlap: number;
    switch_threshold: number;
  };
  semantic: {
    target_size: number;
    similarity_threshold: number;
    overlap: number;
    context_window: number;
  };
}

// 策略描述和評估指標
const strategyInfo = {
  fixed_size: {
    name: "固定大小分割",
    description: "將文檔按照固定的字符數進行分割，適合結構化文檔。",
    metrics: ["分塊數量", "平均長度", "長度變異係數", "重疊率"],
    params: {
      chunk_size: {
        label: "分塊大小",
        min: 100,
        max: 2000,
        default: 500,
        unit: "字符",
      },
      overlap: {
        label: "重疊大小",
        min: 0,
        max: 200,
        default: 50,
        unit: "字符",
      },
    },
  },
  hierarchical: {
    name: "層次分割",
    description: "根據文檔結構（標題、段落等）進行層次化分割，保持語義完整性。",
    metrics: ["分塊數量", "層次深度", "結構保持度", "語義連貫性"],
    params: {
      max_chunk_size: {
        label: "最大分塊大小",
        min: 200,
        max: 3000,
        default: 1000,
        unit: "字符",
      },
      min_chunk_size: {
        label: "最小分塊大小",
        min: 50,
        max: 500,
        default: 200,
        unit: "字符",
      },
      overlap: {
        label: "重疊大小",
        min: 0,
        max: 200,
        default: 50,
        unit: "字符",
      },
      level_depth: {
        label: "層次深度",
        min: 1,
        max: 5,
        default: 3,
        unit: "層",
      },
    },
  },
  adaptive: {
    name: "自適應分割",
    description: "根據內容語義自動調整分塊大小，平衡長度和語義完整性。",
    metrics: ["分塊數量", "語義一致性", "長度分佈", "分割點質量"],
    params: {
      target_size: {
        label: "目標大小",
        min: 300,
        max: 1500,
        default: 600,
        unit: "字符",
      },
      tolerance: {
        label: "容差範圍",
        min: 50,
        max: 300,
        default: 100,
        unit: "字符",
      },
      overlap: {
        label: "重疊大小",
        min: 0,
        max: 200,
        default: 50,
        unit: "字符",
      },
      semantic_threshold: {
        label: "語義閾值",
        min: 0.1,
        max: 0.9,
        default: 0.7,
        unit: "分數",
      },
    },
  },
  hybrid: {
    name: "混合分割",
    description: "結合多種策略，根據內容特徵動態選擇最適合的分割方法。",
    metrics: ["分塊數量", "策略使用率", "整體效果", "適應性評分"],
    params: {
      primary_size: {
        label: "主要大小",
        min: 400,
        max: 1200,
        default: 600,
        unit: "字符",
      },
      secondary_size: {
        label: "次要大小",
        min: 200,
        max: 800,
        default: 400,
        unit: "字符",
      },
      overlap: {
        label: "重疊大小",
        min: 0,
        max: 200,
        default: 50,
        unit: "字符",
      },
      switch_threshold: {
        label: "切換閾值",
        min: 0.1,
        max: 0.9,
        default: 0.5,
        unit: "分數",
      },
    },
  },
  semantic: {
    name: "語義分割",
    description: "基於語義相似性進行智能分割，確保每個分塊在語義上保持連貫性。",
    metrics: ["分塊數量", "語義連貫性", "相似度分佈", "分割點質量"],
    params: {
      target_size: {
        label: "目標大小",
        min: 300,
        max: 1200,
        default: 600,
        unit: "字符",
      },
      similarity_threshold: {
        label: "相似度閾值",
        min: 0.1,
        max: 0.9,
        default: 0.6,
        unit: "分數",
      },
      overlap: {
        label: "重疊大小",
        min: 0,
        max: 200,
        default: 50,
        unit: "字符",
      },
      context_window: {
        label: "上下文窗口",
        min: 50,
        max: 300,
        default: 100,
        unit: "字符",
      },
    },
  },
};

export function ChunkPage() {
  const nav = useNavigate();
  const { canChunk, chunk, docId, chunkMeta } = useRag();
  const [selectedStrategy, setSelectedStrategy] =
    useState<ChunkStrategy>("fixed_size");
  const [params, setParams] = useState<ChunkParams>({
    fixed_size: { chunk_size: 500, overlap: 50 },
    hierarchical: {
      max_chunk_size: 1000,
      min_chunk_size: 200,
      overlap: 50,
      level_depth: 3,
    },
    adaptive: {
      target_size: 600,
      tolerance: 100,
      overlap: 50,
      semantic_threshold: 0.7,
    },
    hybrid: {
      primary_size: 600,
      secondary_size: 400,
      overlap: 50,
      switch_threshold: 0.5,
    },
    semantic: {
      target_size: 600,
      similarity_threshold: 0.6,
      overlap: 50,
      context_window: 100,
    },
  });
  const [busy, setBusy] = useState(false);
  const [chunkResults, setChunkResults] = useState<any>(null);

  const handleParamChange = (
    strategy: ChunkStrategy,
    param: string,
    value: number
  ) => {
    setParams((prev) => ({
      ...prev,
      [strategy]: {
        ...prev[strategy],
        [param]: value,
      },
    }));
  };

  const handleRunChunker = async () => {
    if (!canChunk) return;
    setBusy(true);
    try {
      // 暫時使用現有的chunk API，後續需要擴展支持不同策略
      const currentParams = params[selectedStrategy];

      // 根據策略獲取適當的chunk size
      let chunkSize: number;
      switch (selectedStrategy) {
        case "fixed_size":
          chunkSize = (currentParams as ChunkParams["fixed_size"]).chunk_size;
          break;
        case "hierarchical":
          chunkSize = (currentParams as ChunkParams["hierarchical"])
            .max_chunk_size;
          break;
        case "adaptive":
          chunkSize = (currentParams as ChunkParams["adaptive"]).target_size;
          break;
        case "hybrid":
          chunkSize = (currentParams as ChunkParams["hybrid"]).primary_size;
          break;
        case "semantic":
          chunkSize = (currentParams as ChunkParams["semantic"]).target_size;
          break;
        default:
          chunkSize = 500;
      }

      await chunk(chunkSize, currentParams.overlap);

      // 模擬結果數據（後續需要從後端獲取）
      setChunkResults({
        strategy: selectedStrategy,
        metrics: {
          chunk_count: chunkMeta?.count || 0,
          avg_length: 450,
          length_variance: 0.15,
          overlap_rate: currentParams.overlap / chunkSize,
        },
        chunks: ["示例分塊內容1...", "示例分塊內容2...", "示例分塊內容3..."],
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="row g-3">
      {/* 左側：策略選擇和參數配置 */}
      <div className="col-12 col-md-6">
        <div className="card h-100">
          <div className="card-body">
            <h2 className="h5 mb-3">分塊策略配置</h2>

            {!canChunk && (
              <div className="alert alert-warning" role="alert">
                請先上傳文檔後再進行分塊操作。
              </div>
            )}

            {/* 策略選擇 */}
            <div className="mb-4">
              <label className="form-label fw-bold">選擇分塊策略</label>
              <div className="row g-2">
                {Object.entries(strategyInfo).map(([key, info]) => (
                  <div key={key} className="col-6">
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="radio"
                        name="strategy"
                        id={`strategy-${key}`}
                        checked={selectedStrategy === key}
                        onChange={() =>
                          setSelectedStrategy(key as ChunkStrategy)
                        }
                      />
                      <label
                        className="form-check-label"
                        htmlFor={`strategy-${key}`}
                      >
                        {info.name}
                      </label>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 策略描述 */}
            <div className="mb-4">
              <h6 className="text-primary">
                {strategyInfo[selectedStrategy].name}
              </h6>
              <p className="text-muted small mb-2">
                {strategyInfo[selectedStrategy].description}
              </p>
              <div className="small">
                <strong>評估指標：</strong>
                <span className="text-muted">
                  {strategyInfo[selectedStrategy].metrics.join("、")}
                </span>
              </div>
            </div>

            {/* 參數配置 */}
            <div className="mb-4">
              <h6>參數設置</h6>
              <div className="row g-3">
                {Object.entries(strategyInfo[selectedStrategy].params).map(
                  ([paramKey, paramInfo]) => (
                    <div key={paramKey} className="col-6">
                      <label className="form-label small">
                        {paramInfo.label}
                        <span className="text-muted">({paramInfo.unit})</span>
                      </label>
                      <input
                        className="form-control form-control-sm"
                        type="number"
                        min={paramInfo.min}
                        max={paramInfo.max}
                        value={
                          params[selectedStrategy][
                            paramKey as keyof ChunkParams[ChunkStrategy]
                          ]
                        }
                        onChange={(e) =>
                          handleParamChange(
                            selectedStrategy,
                            paramKey,
                            parseFloat(e.target.value) || 0
                          )
                        }
                      />
                      <div className="form-text small">
                        範圍: {paramInfo.min} - {paramInfo.max}
                      </div>
                    </div>
                  )
                )}
              </div>
            </div>

            {/* 執行按鈕 */}
            <div className="d-grid">
              <button
                className="btn btn-primary"
                disabled={!canChunk || busy}
                onClick={handleRunChunker}
              >
                {busy ? (
                  <>
                    <span
                      className="spinner-border spinner-border-sm me-2"
                      role="status"
                    />
                    執行分塊中...
                  </>
                ) : (
                  "Run Chunker"
                )}
              </button>
            </div>

            {docId && (
              <p className="text-muted small mt-3 mb-0">
                doc_id: <code>{docId}</code>
              </p>
            )}
          </div>
        </div>
      </div>

      {/* 右側：結果顯示 */}
      <div className="col-12 col-md-6">
        <div className="card h-100">
          <div className="card-body">
            <h2 className="h5 mb-3">分塊結果</h2>

            {!chunkResults && !chunkMeta && (
              <p className="text-muted mb-0">
                選擇策略並點擊「Run Chunker」後，分塊結果將顯示在這裡。
              </p>
            )}

            {chunkMeta && (
              <div className="mb-4">
                <div className="alert alert-success" role="alert">
                  <h6 className="alert-heading">分塊完成</h6>
                  <p className="mb-2">
                    文檔已成功分割為 {chunkMeta.count} 個分塊
                  </p>
                  <hr />
                  <div className="row g-2 small">
                    <div className="col-6">
                      <strong>分塊大小:</strong> {chunkMeta.size} 字符
                    </div>
                    <div className="col-6">
                      <strong>重疊大小:</strong> {chunkMeta.overlap} 字符
                    </div>
                  </div>
                </div>

                <div className="d-grid">
                  <button
                    className="btn btn-success"
                    onClick={() => nav("/embed")}
                  >
                    繼續到 Embed
                  </button>
                </div>
              </div>
            )}

            {chunkResults && (
              <div>
                {/* 評估指標 */}
                <div className="mb-4">
                  <h6>評估指標</h6>
                  <div className="row g-2">
                    <div className="col-6">
                      <div className="card bg-light">
                        <div className="card-body p-2">
                          <div className="small text-muted">分塊數量</div>
                          <div className="fw-bold">
                            {chunkResults.metrics.chunk_count}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="col-6">
                      <div className="card bg-light">
                        <div className="card-body p-2">
                          <div className="small text-muted">平均長度</div>
                          <div className="fw-bold">
                            {chunkResults.metrics.avg_length}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="col-6">
                      <div className="card bg-light">
                        <div className="card-body p-2">
                          <div className="small text-muted">長度變異</div>
                          <div className="fw-bold">
                            {chunkResults.metrics.length_variance.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="col-6">
                      <div className="card bg-light">
                        <div className="card-body p-2">
                          <div className="small text-muted">重疊率</div>
                          <div className="fw-bold">
                            {(chunkResults.metrics.overlap_rate * 100).toFixed(
                              1
                            )}
                            %
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 分塊內容預覽 */}
                <div>
                  <h6>分塊內容預覽</h6>
                  <div
                    className="border rounded p-3"
                    style={{ maxHeight: "300px", overflowY: "auto" }}
                  >
                    {chunkResults.chunks.map((chunk: string, index: number) => (
                      <div key={index} className="mb-3">
                        <div className="d-flex justify-content-between align-items-center mb-1">
                          <small className="text-muted">
                            分塊 #{index + 1}
                          </small>
                          <small className="text-muted">
                            {chunk.length} 字符
                          </small>
                        </div>
                        <div className="bg-light p-2 rounded small">
                          {chunk}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
