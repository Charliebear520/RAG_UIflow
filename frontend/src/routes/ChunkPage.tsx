import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";
import { api } from "../lib/api";
// 無映射模式：移除與 QA 映射相關的元件

// 擴展Window接口以包含Bootstrap
declare global {
  interface Window {
    bootstrap: any;
  }
}

// 分塊策略類型定義
type ChunkStrategy = "fixed_size" | "structured_hierarchical";

// 策略參數接口
interface ChunkParams {
  fixed_size: {
    chunk_size: number;
    overlap: number;
  };
  structured_hierarchical: {
    // 多層級結構化分割不需要額外參數，按照法律結構自然分割
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
  structured_hierarchical: {
    name: "多層級結構化分割",
    description:
      "基於六個粒度級別（文件-章-節-條-項-款/目）進行智能分割，一次性生成所有層次的分塊，確保法律概念完整性和上下文一致性。",
    metrics: ["多層級覆蓋", "結構準確性", "上下文一致性", "語義完整性"],
    params: {
      preserve_structure: {
        label: "強制結構邊界分割",
        type: "boolean",
        default: true,
        description: "在結構邊界強制分割，確保法律邏輯完整性",
      },
      min_chunk_size: {
        label: "最小分塊大小",
        min: 50,
        max: 500,
        default: 100,
        unit: "字符",
        description: "避免產生過短的分塊",
      },
      max_chunk_size: {
        label: "最大分塊大小",
        min: 500,
        max: 5000,
        default: 2000,
        unit: "字符",
        description: "避免產生過長的分塊",
      },
      enable_overlap: {
        label: "啟用重疊",
        type: "boolean",
        default: false,
        description: "是否在結構邊界之間添加重疊內容",
      },
      overlap_ratio: {
        label: "重疊比例",
        min: 0.05,
        max: 0.3,
        default: 0.1,
        unit: "比例",
        description: "僅在啟用重疊時使用",
      },
    },
  },
};

// 評測相關接口
interface EvaluationResult {
  config: {
    chunk_size: number;
    overlap: number;
    overlap_ratio: number;
    strategy?: string;
    window_size?: number;
    step_size?: number;
    boundary_aware?: boolean;
    preserve_sentences?: boolean;
    min_chunk_size_sw?: number;
    max_chunk_size_sw?: number;
  };
  metrics: {
    precision_omega: number;
    precision_at_k: Record<number, number>;
    recall_at_k: Record<number, number>;
    chunk_count: number;
    avg_chunk_length: number;
    length_variance: number;
  };
  test_queries: string[];
  retrieval_results: Record<string, any[]>;
  timestamp: string;
}

interface EvaluationTask {
  task_id: string;
  status: string;
  created_at: string;
  completed_at?: string;
  error_message?: string;
  total_configs: number;
  completed_configs: number;
  progress: number;
}

export function ChunkPage() {
  const nav = useNavigate();
  const { canChunk, chunk, docId, chunkMeta, setChunkingResultsAndStrategy } =
    useRag();

  // 無映射模式：不需要 QA Set 上傳狀態（移除）

  // 步驟2: 分塊策略配置狀態
  const [selectedStrategy, setSelectedStrategy] = useState<ChunkStrategy>(
    "structured_hierarchical"
  );
  const [isChunking, setIsChunking] = useState(false);
  const [chunkingError, setChunkingError] = useState<string | null>(null);
  const [chunkingProgress, setChunkingProgress] = useState(0);
  const [chunkingTaskId, setChunkingTaskId] = useState<string | null>(null);

  // 分塊參數狀態
  const [chunkParams, setChunkParams] = useState<ChunkParams>({
    fixed_size: {
      chunk_size: 500,
      overlap: 50,
    },
    structured_hierarchical: {
      // 多層級結構化分割不需要額外參數
    },
  });

  // 無映射模式：不需要 QA 映射狀態（移除）

  // 步驟4: 評測狀態
  const [evaluationConfig, setEvaluationConfig] = useState({
    chunk_sizes: [300, 500, 800],
    overlap_ratios: [0.0, 0.1, 0.2],
    strategy: "fixed_size",
    test_queries: [
      "著作權的定義是什麼？",
      "什麼情況下可以合理使用他人作品？",
      "侵犯著作權的法律後果是什麼？",
      "著作權的保護期限是多久？",
      "如何申請著作權登記？",
    ],
    k_values: [1, 3, 5, 10],
  });
  const [currentTask, setCurrentTask] = useState<EvaluationTask | null>(null);
  const [evaluationResults, setEvaluationResults] = useState<
    EvaluationResult[]
  >([]);
  const [evaluationComparison, setEvaluationComparison] = useState<any>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [showAllResults, setShowAllResults] = useState(false);
  const [progressInterval, setProgressInterval] =
    useState<NodeJS.Timeout | null>(null);

  // 輪詢進度更新的函數
  const pollProgress = async (taskId: string) => {
    try {
      const response = await fetch(`/api/evaluate/status/${taskId}`);
      if (response.ok) {
        const taskData = await response.json();
        setCurrentTask(taskData);

        // 如果任務完成或失敗，停止輪詢
        if (taskData.status === "completed" || taskData.status === "failed") {
          if (progressInterval) {
            clearInterval(progressInterval);
            setProgressInterval(null);
          }

          // 如果完成，獲取結果
          if (taskData.status === "completed") {
            const resultsResponse = await fetch(
              `/api/evaluate/results/${taskId}`
            );
            if (resultsResponse.ok) {
              const resultsData = await resultsResponse.json();
              setEvaluationResults(resultsData.results || []);
              setEvaluationComparison(resultsData.comparison || null);
            }
          }
        }
      }
    } catch (error) {
      console.error("輪詢進度時出錯:", error);
    }
  };

  // 清理輪詢的函數
  const clearProgressPolling = () => {
    if (progressInterval) {
      clearInterval(progressInterval);
      setProgressInterval(null);
    }
  };

  // 組件卸載時清理輪詢
  useEffect(() => {
    return () => {
      clearProgressPolling();
    };
  }, []);

  // 計算預計剩餘時間
  const calculateEstimatedTime = (task: EvaluationTask) => {
    if (task.progress === 0 || task.progress === 1) {
      return task.progress === 1 ? "已完成" : "計算中...";
    }

    const elapsedTime = Date.now() - new Date(task.created_at).getTime();
    const avgTimePerProgress = elapsedTime / task.progress;
    const remainingProgress = 1 - task.progress;
    const estimatedRemainingMs = remainingProgress * avgTimePerProgress;
    const estimatedMinutes = Math.ceil(estimatedRemainingMs / (1000 * 60));

    return estimatedMinutes > 0 ? estimatedMinutes.toString() : "< 1";
  };

  // 步驟1: 處理QA Set文件上傳
  // 無映射模式：不需要 QA 上傳處理（移除）

  // 步驟2: 處理分塊策略選擇
  const handleStrategyChange = (strategy: ChunkStrategy) => {
    setSelectedStrategy(strategy);
  };

  // 處理分塊參數變更
  const handleParamsChange = (params: ChunkParams) => {
    setChunkParams(params);
  };

  // 分塊結果狀態
  const [chunkingResults, setChunkingResults] = useState<any[]>([]);

  // 分塊顯示狀態
  const [showAllChunks, setShowAllChunks] = useState(false);
  const [chunkSearchTerm, setChunkSearchTerm] = useState("");
  const [chunkFilterLength, setChunkFilterLength] = useState<{
    min: number;
    max: number;
  }>({ min: 0, max: 10000 });

  // 確保Bootstrap accordion正確初始化
  useEffect(() => {
    if (chunkingResults.length > 0 && typeof window !== "undefined") {
      // 等待DOM更新後重新初始化Bootstrap組件
      const timer = setTimeout(() => {
        // 檢查Bootstrap是否可用
        if (window.bootstrap) {
          // 重新初始化所有accordion
          const accordions = document.querySelectorAll(".accordion");
          accordions.forEach((accordion) => {
            const bsAccordion = new window.bootstrap.Collapse(accordion, {
              toggle: false,
            });
          });
        }
      }, 100);

      return () => clearTimeout(timer);
    }
  }, [chunkingResults, showAllChunks]);

  // 步驟2: 執行分塊操作
  const handleRunChunking = async () => {
    if (!canChunk) return;

    setIsChunking(true);
    setChunkingError(null);
    setChunkingResults([]);
    setChunkingProgress(0);

    try {
      // 根據策略準備參數
      let apiParams: any = {
        doc_id: docId!,
        strategies: [selectedStrategy],
      };

      if (selectedStrategy === "fixed_size") {
        apiParams.chunk_sizes = [chunkParams.fixed_size.chunk_size];
        apiParams.overlap_ratios = [
          chunkParams.fixed_size.overlap / chunkParams.fixed_size.chunk_size,
        ];
      } else if (selectedStrategy === "structured_hierarchical") {
        // 多層級結構化分割不需要額外參數，按照法律結構自然分割
        apiParams.chunk_sizes = [1000]; // 默認值，實際不會使用
        apiParams.overlap_ratios = [0.1]; // 默認值，實際不會使用
      }

      // 調用分塊API
      const response = await api.startMultipleChunking(apiParams);

      setChunkingTaskId(response.task_id);

      // 開始輪詢進度
      const pollProgress = async () => {
        try {
          const statusResponse = await api.getChunkingStatus(response.task_id);
          setChunkingProgress(statusResponse.progress * 100);

          if (statusResponse.status === "completed") {
            const resultsResponse = await api.getChunkingResults(
              response.task_id
            );
            setChunkingResults(resultsResponse.results);

            // 將分塊結果存儲到 RAG store 中
            if (resultsResponse.results && resultsResponse.results.length > 0) {
              setChunkingResultsAndStrategy(
                resultsResponse.results,
                selectedStrategy
              );
            }

            setIsChunking(false);
          } else if (statusResponse.status === "failed") {
            setChunkingError(statusResponse.error_message || "分塊操作失敗");
            setIsChunking(false);
          } else {
            // 繼續輪詢
            setTimeout(pollProgress, 1000);
          }
        } catch (error) {
          console.error("輪詢分塊進度失敗:", error);
          setChunkingError("獲取分塊進度失敗");
          setIsChunking(false);
        }
      };

      // 開始輪詢
      setTimeout(pollProgress, 1000);
    } catch (error) {
      console.error("分塊操作失敗:", error);
      setChunkingError(error instanceof Error ? error.message : "分塊操作失敗");
      setIsChunking(false);
    }
  };

  // 重新進行分塊
  const handleRetryChunking = () => {
    setChunkingResults([]);
    setEvaluationResults([]);
    setCurrentTask(null);
    setShowAllChunks(false);
    setChunkSearchTerm("");
    setChunkFilterLength({ min: 0, max: 10000 });
    setChunkingProgress(0);
    setChunkingTaskId(null);
  };

  // 過濾和搜索分塊
  const getFilteredChunks = () => {
    if (!chunkingResults.length) return [];

    const allChunks = chunkingResults.flatMap((result, resultIndex) => {
      // 優先使用chunks_with_span，如果沒有則回退到chunks
      const chunksData =
        result.chunks_with_span || result.all_chunks || result.chunks || [];

      return chunksData.map((chunkData: any, chunkIndex: number) => {
        // 如果chunkData是帶span信息的對象
        if (typeof chunkData === "object" && chunkData.content) {
          return {
            chunk: chunkData.content,
            chunkId: chunkData.chunk_id,
            span: chunkData.span,
            metadata: chunkData.metadata,
            resultIndex,
            chunkIndex,
            strategy: result.strategy,
            config: result.config,
          };
        } else {
          // 如果chunkData是字符串（舊格式）
          return {
            chunk: chunkData,
            chunkId: `${result.strategy}_chunk_${chunkIndex}`,
            span: null,
            metadata: null,
            resultIndex,
            chunkIndex,
            strategy: result.strategy,
            config: result.config,
          };
        }
      });
    });

    return allChunks.filter(({ chunk }) => {
      // 長度過濾
      const chunkLength = chunk.length;
      if (
        chunkLength < chunkFilterLength.min ||
        chunkLength > chunkFilterLength.max
      ) {
        return false;
      }

      // 搜索過濾
      if (chunkSearchTerm) {
        return chunk.toLowerCase().includes(chunkSearchTerm.toLowerCase());
      }

      return true;
    });
  };

  // 步驟3: 處理QA映射完成的回調函數
  // 無映射模式：不需要 QA 映射完成回調（移除）

  // 步驟3: 開始QA映射
  // 無映射模式：不需要 QA 映射啟動（移除）

  // 評測相關函數
  // 無映射模式：此頁不再啟動評測，改由首頁 Evaluate(beta) 執行
  const startEvaluation = async () => {
    setEvaluationError("請前往首頁 Evaluate(beta) 進行評測");
  };

  const loadEvaluationResults = async (taskId: string) => {
    try {
      const [resultsResponse, comparisonResponse] = await Promise.all([
        api.getEvaluationResults(taskId),
        api.getEvaluationComparison(taskId),
      ]);

      setEvaluationResults(resultsResponse.results);
      setEvaluationComparison(comparisonResponse);
    } catch (err) {
      setEvaluationError(`載入結果失敗: ${err}`);
    }
  };

  // 計算最佳配置
  const getBestConfig = () => {
    if (!evaluationResults.length) return null;

    return evaluationResults.reduce((best, current) => {
      const currentScore =
        (current.metrics.precision_at_k[3] || 0) * 0.3 +
        (current.metrics.recall_at_k[3] || 0) * 0.3 +
        (current.metrics.precision_omega || 0) * 0.4;
      const bestScore =
        (best.metrics.precision_at_k[3] || 0) * 0.3 +
        (best.metrics.recall_at_k[3] || 0) * 0.3 +
        (best.metrics.precision_omega || 0) * 0.4;
      return currentScore > bestScore ? current : best;
    }, evaluationResults[0]);
  };

  const exportEvaluationReport = () => {
    if (!evaluationResults.length || !evaluationComparison) return;

    const report = {
      evaluation_info: {
        task_id: currentTask?.task_id,
        doc_id: docId,
        created_at: currentTask?.created_at,
        completed_at: currentTask?.completed_at,
        total_configs: evaluationResults.length,
      },
      configuration: evaluationConfig,
      results: evaluationResults,
      comparison: evaluationComparison,
      summary: {
        best_config: evaluationResults.reduce((best, current) => {
          const currentScore =
            (current.metrics.precision_at_k[3] || 0) * 0.3 +
            (current.metrics.recall_at_k[3] || 0) * 0.3 +
            (current.metrics.precision_omega || 0) * 0.4;
          const bestScore =
            (best.metrics.precision_at_k[3] || 0) * 0.3 +
            (best.metrics.recall_at_k[3] || 0) * 0.3 +
            (best.metrics.precision_omega || 0) * 0.4;
          return currentScore > bestScore ? current : best;
        }, evaluationResults[0]),
        recommendations: evaluationComparison.recommendations,
      },
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chunk-evaluation-report-${
      new Date().toISOString().split("T")[0]
    }.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // 輪詢任務狀態
  useEffect(() => {
    if (
      !currentTask ||
      currentTask.status === "completed" ||
      currentTask.status === "failed"
    ) {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const status = await api.getEvaluationStatus(currentTask.task_id);
        setCurrentTask(status);

        if (status.status === "completed") {
          await loadEvaluationResults(status.task_id);
        }
      } catch (err) {
        console.error("獲取任務狀態失敗:", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentTask]);

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h2 className="h4 mb-0">分塊策略評測</h2>
              <p className="text-muted mb-0">
                無映射模式：Upload → Chunk → Embed（可選）→ Evaluate(beta)
              </p>
            </div>
            <div className="card-body">
              {!canChunk && (
                <div className="alert alert-warning" role="alert">
                  請先上傳文檔後再進行分塊評測操作。
                </div>
              )}

              {canChunk && (
                <div className="row g-4">
                  {/* 步驟 1: 說明與引導（無映射模式） */}
                  <div className="col-12">
                    <div className="alert alert-info" role="alert">
                      <strong>提示：</strong>{" "}
                      本頁專注於「批量分塊」。評測請到首頁 Evaluate(beta)
                      區塊上傳 <code>qa_gold.json</code>，啟動分塊後一鍵計算 P@K
                      / R@K。
                    </div>
                  </div>

                  {/* 步驟2: 多種分塊組合處理 */}
                  <div className="col-12">
                    <div
                      className={`card ${
                        chunkingResults.length > 0
                          ? "border-success"
                          : "border-secondary"
                      }`}
                    >
                      <div
                        className={`card-header ${
                          chunkingResults.length > 0
                            ? "bg-success text-white"
                            : "bg-secondary text-white"
                        }`}
                      >
                        <h5 className="mb-0">步驟 2: 多種分塊組合處理</h5>
                      </div>
                      <div className="card-body">
                        {chunkingResults.length > 0 ? (
                          <div>
                            <div className="alert alert-success mb-4">
                              <h6>✅ 多種分塊組合處理完成</h6>
                              <p className="mb-0">
                                已成功完成 {chunkingResults.length}{" "}
                                種分塊組合的處理
                              </p>
                            </div>

                            {/* 分塊結果統計 */}
                            <div className="row g-3 mb-4">
                              <div className="col-md-3">
                                <div className="card bg-light">
                                  <div className="card-body text-center">
                                    <h5 className="card-title text-primary">
                                      {chunkingResults.length}
                                    </h5>
                                    <p className="card-text small">
                                      分塊組合數
                                    </p>
                                  </div>
                                </div>
                              </div>
                              <div className="col-md-3">
                                <div className="card bg-light">
                                  <div className="card-body text-center">
                                    <h5 className="card-title text-success">
                                      {Math.round(
                                        chunkingResults.reduce(
                                          (sum, result) =>
                                            sum + result.chunk_count,
                                          0
                                        ) / chunkingResults.length
                                      )}
                                    </h5>
                                    <p className="card-text small">
                                      平均分塊數
                                    </p>
                                  </div>
                                </div>
                              </div>
                              <div className="col-md-3">
                                <div className="card bg-light">
                                  <div className="card-body text-center">
                                    <h5 className="card-title text-warning">
                                      {Math.round(
                                        chunkingResults.reduce(
                                          (sum, result) =>
                                            sum +
                                            (result.metrics?.avg_length || 0),
                                          0
                                        ) / chunkingResults.length
                                      )}
                                    </h5>
                                    <p className="card-text small">平均長度</p>
                                  </div>
                                </div>
                              </div>
                              <div className="col-md-3">
                                <div className="card bg-light">
                                  <div className="card-body text-center">
                                    <h5 className="card-title text-info">1</h5>
                                    <p className="card-text small">
                                      使用策略數
                                    </p>
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* 分塊配置信息 */}
                            <div className="mb-4">
                              <h6>分塊配置組合</h6>
                              <div className="table-responsive">
                                <table className="table table-sm table-striped">
                                  <thead>
                                    <tr>
                                      <th>策略</th>
                                      <th>分塊大小</th>
                                      <th>重疊比例</th>
                                      <th>分塊數量</th>
                                      <th>平均長度</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {chunkingResults.map((result, index) => (
                                      <tr key={index}>
                                        <td>
                                          {strategyInfo[
                                            result.strategy as ChunkStrategy
                                          ]?.name || result.strategy}
                                        </td>
                                        <td>{result.config.chunk_size} 字符</td>
                                        <td>
                                          {(
                                            result.config.overlap_ratio * 100
                                          ).toFixed(1)}
                                          %
                                        </td>
                                        <td>{result.chunk_count}</td>
                                        <td>
                                          {result.metrics?.avg_length?.toFixed(
                                            0
                                          ) || "N/A"}
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </div>

                            {/* 分塊內容查看 */}
                            <div className="mb-4">
                              <div className="d-flex justify-content-between align-items-center mb-3">
                                <h6 className="mb-0">分塊內容查看</h6>
                                <button
                                  className="btn btn-outline-primary btn-sm"
                                  onClick={() =>
                                    setShowAllChunks(!showAllChunks)
                                  }
                                >
                                  {showAllChunks ? (
                                    <>
                                      <i className="bi bi-eye-slash me-1"></i>
                                      隱藏所有分塊
                                    </>
                                  ) : (
                                    <>
                                      <i className="bi bi-eye me-1"></i>
                                      查看所有分塊
                                    </>
                                  )}
                                </button>
                              </div>

                              {showAllChunks && (
                                <div className="mb-3">
                                  {/* 搜索和過濾控制 */}
                                  <div className="row g-2 mb-3">
                                    <div className="col-md-6">
                                      <input
                                        type="text"
                                        className="form-control form-control-sm"
                                        placeholder="搜索分塊內容..."
                                        value={chunkSearchTerm}
                                        onChange={(e) =>
                                          setChunkSearchTerm(e.target.value)
                                        }
                                      />
                                    </div>
                                    <div className="col-md-3">
                                      <input
                                        type="number"
                                        className="form-control form-control-sm"
                                        placeholder="最小長度"
                                        value={chunkFilterLength.min || ""}
                                        onChange={(e) =>
                                          setChunkFilterLength((prev) => ({
                                            ...prev,
                                            min: parseInt(e.target.value) || 0,
                                          }))
                                        }
                                      />
                                    </div>
                                    <div className="col-md-3">
                                      <input
                                        type="number"
                                        className="form-control form-control-sm"
                                        placeholder="最大長度"
                                        value={chunkFilterLength.max || ""}
                                        onChange={(e) =>
                                          setChunkFilterLength((prev) => ({
                                            ...prev,
                                            max:
                                              parseInt(e.target.value) || 10000,
                                          }))
                                        }
                                      />
                                    </div>
                                  </div>

                                  {/* 過濾結果統計 */}
                                  {(() => {
                                    const filteredChunks = getFilteredChunks();
                                    const totalChunks = chunkingResults.reduce(
                                      (sum, result) =>
                                        sum +
                                        (
                                          result.all_chunks ||
                                          result.chunks ||
                                          []
                                        ).length,
                                      0
                                    );
                                    return (
                                      <div className="alert alert-info py-2">
                                        <small>
                                          顯示 {filteredChunks.length} /{" "}
                                          {totalChunks} 個分塊
                                          {chunkSearchTerm &&
                                            ` (搜索: "${chunkSearchTerm}")`}
                                          {(chunkFilterLength.min > 0 ||
                                            chunkFilterLength.max < 10000) &&
                                            ` (長度: ${chunkFilterLength.min}-${chunkFilterLength.max} 字符)`}
                                        </small>
                                      </div>
                                    );
                                  })()}
                                </div>
                              )}

                              {/* 分塊列表 */}
                              <div
                                className="accordion"
                                id={`chunkPreview-${Date.now()}`}
                              >
                                {(showAllChunks ? getFilteredChunks() : []).map(
                                  (chunkInfo, index) => {
                                    const {
                                      chunk,
                                      chunkId,
                                      span,
                                      metadata,
                                      resultIndex,
                                      chunkIndex,
                                      strategy,
                                      config,
                                    } = chunkInfo;
                                    const uniqueId = `chunk-${resultIndex}-${chunkIndex}-${Date.now()}`;
                                    const accordionId = `chunkPreview-${Date.now()}`;
                                    return (
                                      <div
                                        key={uniqueId}
                                        className="accordion-item"
                                      >
                                        <h2 className="accordion-header">
                                          <button
                                            className="accordion-button collapsed"
                                            type="button"
                                            data-bs-toggle="collapse"
                                            data-bs-target={`#${uniqueId}`}
                                            aria-expanded="false"
                                            aria-controls={uniqueId}
                                          >
                                            <div className="d-flex justify-content-between w-100 me-3">
                                              <div className="d-flex flex-column">
                                                <span>
                                                  {strategyInfo[
                                                    strategy as ChunkStrategy
                                                  ]?.name || strategy}{" "}
                                                  - 分塊 {chunkIndex + 1}
                                                </span>
                                                {chunkId && (
                                                  <small className="text-primary fw-bold">
                                                    ID: {chunkId}
                                                  </small>
                                                )}
                                                {span && (
                                                  <small className="text-info">
                                                    Span: [{span.start}-
                                                    {span.end}]
                                                  </small>
                                                )}
                                              </div>
                                              <span className="badge bg-secondary">
                                                {chunk.length} 字符
                                              </span>
                                            </div>
                                          </button>
                                        </h2>
                                        <div
                                          id={uniqueId}
                                          className="accordion-collapse collapse"
                                          data-bs-parent={`#${accordionId}`}
                                          aria-labelledby={`heading-${uniqueId}`}
                                        >
                                          <div className="accordion-body">
                                            <div className="d-flex justify-content-between align-items-center mb-2">
                                              <div className="d-flex flex-column">
                                                <small className="text-muted">
                                                  {strategyInfo[
                                                    strategy as ChunkStrategy
                                                  ]?.name || strategy}{" "}
                                                  | 分塊 {chunkIndex + 1} |
                                                  大小: {config.chunk_size} |
                                                  重疊:{" "}
                                                  {(
                                                    config.overlap_ratio * 100
                                                  ).toFixed(1)}
                                                  %
                                                </small>
                                                {chunkId && (
                                                  <small className="text-primary fw-bold">
                                                    Chunk ID: {chunkId}
                                                  </small>
                                                )}
                                                {span && (
                                                  <small className="text-info">
                                                    原文位置: [{span.start}-
                                                    {span.end}] 字符
                                                  </small>
                                                )}
                                              </div>
                                              <button
                                                className="btn btn-outline-secondary btn-sm"
                                                onClick={() => {
                                                  navigator.clipboard.writeText(
                                                    chunk
                                                  );
                                                }}
                                                title="複製分塊內容"
                                              >
                                                <i className="bi bi-clipboard"></i>
                                              </button>
                                            </div>

                                            {/* 顯示法條JSON spans信息 */}
                                            {metadata?.overlapping_law_spans
                                              ?.length > 0 && (
                                              <div className="mb-3">
                                                <h6 className="text-success mb-2">
                                                  <i className="bi bi-file-text me-1"></i>
                                                  對應法條JSON spans:
                                                </h6>
                                                <div className="row">
                                                  {metadata.overlapping_law_spans
                                                    .slice(0, 3)
                                                    .map(
                                                      (
                                                        lawSpan: any,
                                                        lawIndex: number
                                                      ) => (
                                                        <div
                                                          key={lawIndex}
                                                          className="col-md-4 mb-2"
                                                        >
                                                          <div className="card bg-success bg-opacity-10 border-success">
                                                            <div className="card-body p-2">
                                                              <h6 className="card-title text-success mb-1 small">
                                                                {
                                                                  lawSpan.article_name
                                                                }
                                                              </h6>
                                                              <p className="card-text small mb-1">
                                                                <strong>
                                                                  ID:
                                                                </strong>{" "}
                                                                {
                                                                  lawSpan.article_id
                                                                }
                                                              </p>
                                                              <p className="card-text small mb-1">
                                                                <strong>
                                                                  位置:
                                                                </strong>{" "}
                                                                [
                                                                {
                                                                  lawSpan.start_char
                                                                }
                                                                -
                                                                {
                                                                  lawSpan.end_char
                                                                }
                                                                ]
                                                              </p>
                                                              <p className="card-text small mb-1">
                                                                <strong>
                                                                  重疊:
                                                                </strong>{" "}
                                                                {(
                                                                  lawSpan.overlap_ratio *
                                                                  100
                                                                ).toFixed(1)}
                                                                %
                                                              </p>
                                                              <p className="card-text small mb-0">
                                                                <strong>
                                                                  章節:
                                                                </strong>{" "}
                                                                {
                                                                  lawSpan.chapter_name
                                                                }{" "}
                                                                &gt;{" "}
                                                                {
                                                                  lawSpan.section_name
                                                                }
                                                              </p>
                                                            </div>
                                                          </div>
                                                        </div>
                                                      )
                                                    )}
                                                  {metadata
                                                    .overlapping_law_spans
                                                    .length > 3 && (
                                                    <div className="col-12">
                                                      <small className="text-muted">
                                                        還有{" "}
                                                        {metadata
                                                          .overlapping_law_spans
                                                          .length - 3}{" "}
                                                        個法條spans...
                                                      </small>
                                                    </div>
                                                  )}
                                                </div>
                                              </div>
                                            )}

                                            <pre
                                              className="small text-muted mb-0"
                                              style={{
                                                whiteSpace: "pre-wrap",
                                                maxHeight: "300px",
                                                overflow: "auto",
                                                backgroundColor: "#f8f9fa",
                                                padding: "10px",
                                                borderRadius: "4px",
                                                border: "1px solid #dee2e6",
                                              }}
                                            >
                                              {chunk}
                                            </pre>
                                          </div>
                                        </div>
                                      </div>
                                    );
                                  }
                                )}
                              </div>

                              {showAllChunks &&
                                getFilteredChunks().length === 0 && (
                                  <div className="alert alert-warning">
                                    <i className="bi bi-search me-2"></i>
                                    沒有找到符合條件的分塊
                                  </div>
                                )}
                            </div>

                            {/* 操作按鈕 */}
                            <div className="d-flex gap-2">
                              <button
                                className="btn btn-outline-primary"
                                onClick={handleRetryChunking}
                              >
                                <i className="bi bi-arrow-clockwise me-1"></i>
                                重新分塊
                              </button>
                              <button
                                className="btn btn-success"
                                onClick={() => nav("/embed")}
                              >
                                前往 Embedding
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div>
                            {/* 分塊策略選擇 */}
                            <div className="mb-4">
                              <label className="form-label fw-bold">
                                選擇分塊策略
                              </label>
                              <div className="row g-2">
                                {Object.entries(strategyInfo).map(
                                  ([key, info]) => (
                                    <div key={key} className="col-12">
                                      <div className="form-check">
                                        <input
                                          className="form-check-input"
                                          type="radio"
                                          name="strategy"
                                          id={`strategy-${key}`}
                                          checked={selectedStrategy === key}
                                          onChange={() =>
                                            handleStrategyChange(
                                              key as ChunkStrategy
                                            )
                                          }
                                        />
                                        <label
                                          className="form-check-label"
                                          htmlFor={`strategy-${key}`}
                                        >
                                          <strong>{info.name}</strong>
                                          <br />
                                          <small className="text-muted">
                                            {info.description}
                                          </small>
                                        </label>
                                      </div>
                                    </div>
                                  )
                                )}
                              </div>
                            </div>

                            {/* 策略描述 */}
                            <div className="mb-4">
                              <h6 className="text-primary">已選擇的策略</h6>
                              <div className="alert alert-info">
                                <strong>
                                  {strategyInfo[selectedStrategy].name}
                                </strong>
                                <p className="mb-1">
                                  {strategyInfo[selectedStrategy].description}
                                </p>
                                <div className="small text-muted">
                                  評估指標：
                                  {strategyInfo[selectedStrategy].metrics.join(
                                    "、"
                                  )}
                                </div>
                              </div>
                            </div>

                            {/* 策略參數配置 */}
                            <div className="mb-4">
                              <h6 className="text-primary">策略參數配置</h6>
                              <div className="card">
                                <div className="card-body">
                                  {selectedStrategy === "fixed_size" && (
                                    <div className="row">
                                      <div className="col-md-6">
                                        <label className="form-label">
                                          分塊大小
                                        </label>
                                        <input
                                          type="number"
                                          className="form-control"
                                          value={
                                            chunkParams.fixed_size.chunk_size
                                          }
                                          onChange={(e) =>
                                            setChunkParams({
                                              ...chunkParams,
                                              fixed_size: {
                                                ...chunkParams.fixed_size,
                                                chunk_size:
                                                  parseInt(e.target.value) ||
                                                  500,
                                              },
                                            })
                                          }
                                          min="100"
                                          max="2000"
                                        />
                                      </div>
                                      <div className="col-md-6">
                                        <label className="form-label">
                                          重疊大小
                                        </label>
                                        <input
                                          type="number"
                                          className="form-control"
                                          value={chunkParams.fixed_size.overlap}
                                          onChange={(e) =>
                                            setChunkParams({
                                              ...chunkParams,
                                              fixed_size: {
                                                ...chunkParams.fixed_size,
                                                overlap:
                                                  parseInt(e.target.value) ||
                                                  50,
                                              },
                                            })
                                          }
                                          min="0"
                                          max="200"
                                        />
                                      </div>
                                    </div>
                                  )}

                                  {selectedStrategy ===
                                    "structured_hierarchical" && (
                                    <div>
                                      {/* 多層級分割說明 */}
                                      <div className="alert alert-info mb-3">
                                        <h6 className="alert-heading">
                                          🔍 多層級結構化分割
                                        </h6>
                                        <p className="mb-2">
                                          此策略將
                                          <strong>
                                            一次性生成所有六個粒度級別
                                          </strong>
                                          的分塊：
                                        </p>
                                        <ul className="mb-2 small">
                                          <li>
                                            <strong>文件層級</strong>：整個法規
                                          </li>
                                          <li>
                                            <strong>文件組成部分層級</strong>
                                            ：章級別
                                          </li>
                                          <li>
                                            <strong>
                                              基本單位層次結構層級
                                            </strong>
                                            ：節級別
                                          </li>
                                          <li>
                                            <strong>基本單位層級</strong>
                                            ：條文級別
                                          </li>
                                          <li>
                                            <strong>
                                              基本單位組成部分層級
                                            </strong>
                                            ：項級別
                                          </li>
                                          <li>
                                            <strong>列舉層級</strong>：款/目級別
                                          </li>
                                        </ul>
                                        <p className="mb-0 small text-muted">
                                          <strong>上下文一致性</strong>
                                          ：低層次列舉元素（款/目）會自動包含上級元素的上下文，確保語義一致性。
                                        </p>
                                      </div>

                                      <div className="alert alert-success">
                                        <h6 className="alert-heading">
                                          ✅ 自動化分割
                                        </h6>
                                        <p className="mb-0">
                                          多層級結構化分割會根據法律文檔的天然結構自動進行分割，無需額外參數配置。
                                          系統會智能識別法律文檔的層次結構，並為每個層次生成相應的分塊。
                                        </p>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>

                            {/* 開始分塊按鈕 */}
                            <div className="d-grid">
                              <button
                                className="btn btn-warning btn-lg"
                                onClick={handleRunChunking}
                                disabled={isChunking}
                              >
                                {isChunking ? (
                                  <>
                                    <span
                                      className="spinner-border spinner-border-sm me-2"
                                      role="status"
                                      aria-hidden="true"
                                    ></span>
                                    分塊中... ({chunkingProgress.toFixed(1)}%)
                                  </>
                                ) : (
                                  `開始分塊 (${strategyInfo[selectedStrategy].name})`
                                )}
                              </button>
                            </div>

                            {/* 分塊錯誤提示 */}
                            {chunkingError && (
                              <div
                                className="alert alert-danger mt-3"
                                role="alert"
                              >
                                {chunkingError}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 下一步：策略評測（導向 Evaluate(beta)） */}
                  <div className="col-12">
                    <div
                      className={`card ${
                        chunkingResults.length > 0
                          ? "border-success"
                          : "border-secondary"
                      }`}
                    >
                      <div
                        className={`card-header ${
                          chunkingResults.length > 0
                            ? "bg-success text-white"
                            : "bg-secondary text-white"
                        }`}
                      >
                        <h5 className="mb-0">下一步：策略評測</h5>
                      </div>
                      <div className="card-body">
                        {chunkingResults.length > 0 ? (
                          <div className="d-flex align-items-center justify-content-between">
                            <div className="text-muted small">
                              已完成分塊，前往 Embedding 頁面進行向量化，然後到
                              Retrieve 頁面測試檢索，最後到 Evaluate
                              頁面進行評測。
                            </div>
                            <button
                              className="btn btn-success"
                              onClick={() => nav("/embed")}
                            >
                              前往 Embedding
                            </button>
                          </div>
                        ) : (
                          <div className="text-center text-muted py-3">
                            <i className="bi bi-hourglass-split fs-1 d-block mb-2"></i>
                            <p>請先完成分塊</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
