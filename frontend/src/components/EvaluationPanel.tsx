import React, { useState, useEffect } from "react";
import { api } from "../lib/api";

interface EvaluationPanelProps {
  docId: string;
  onEvaluationComplete: (results: any) => void;
}

interface EvaluationResults {
  results: Array<{
    config: {
      chunk_size: number;
      overlap_ratio: number;
    };
    metrics: {
      precision_omega: number;
      precision_at_k: Record<number, number>;
      recall_at_k: Record<number, number>;
      chunk_count: number;
      avg_chunk_length: number;
      length_variance: number;
    };
  }>;
  summary: {
    best_config: any;
    total_configs: number;
    best_precision_omega: number;
    best_precision_at_5: number;
    best_recall_at_5: number;
  };
}

export const EvaluationPanel: React.FC<EvaluationPanelProps> = ({
  docId,
  onEvaluationComplete,
}) => {
  const [isEvaluationMode, setIsEvaluationMode] = useState(false);
  const [chunkSizes, setChunkSizes] = useState<number[]>([300, 500, 800]);
  const [overlapRatios, setOverlapRatios] = useState<number[]>([0.0, 0.1, 0.2]);
  const [questionTypes, setQuestionTypes] = useState<string[]>([
    "案例應用",
    "情境分析",
    "實務處理",
    "法律後果",
    "合規判斷",
  ]);
  const [numQuestions, setNumQuestions] = useState<number>(10);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationTaskId, setEvaluationTaskId] = useState<string | null>(null);
  const [evaluationProgress, setEvaluationProgress] = useState(0);
  const [evaluationResults, setEvaluationResults] =
    useState<EvaluationResults | null>(null);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);

  const startEvaluation = async () => {
    if (!docId) return;

    setIsEvaluating(true);
    setEvaluationError(null);
    setEvaluationResults(null);

    try {
      // 開始評估
      const response = await api.evaluateChunking({
        doc_id: docId,
        chunk_sizes: chunkSizes,
        overlap_ratios: overlapRatios,
        question_types: questionTypes,
        num_questions: numQuestions,
      });

      setEvaluationTaskId(response.task_id);

      // 開始執行評估
      await api.runEvaluation(response.task_id);

      // 輪詢任務狀態
      const pollTaskStatus = async () => {
        try {
          const statusResponse = await api.getEvaluationTask(response.task_id);

          if (statusResponse.status === "completed") {
            // 獲取評估結果
            const resultsResponse = await api.getEvaluationResults(
              response.task_id
            );
            setEvaluationResults(resultsResponse);
            onEvaluationComplete(resultsResponse);
            setIsEvaluating(false);
            setEvaluationProgress(100);
          } else if (statusResponse.status === "failed") {
            setEvaluationError(statusResponse.error_message || "評估失敗");
            setIsEvaluating(false);
          } else if (statusResponse.status === "running") {
            setEvaluationProgress(statusResponse.progress * 100);
            setTimeout(pollTaskStatus, 2000); // 2秒後再次檢查
          }
        } catch (error) {
          setEvaluationError("獲取評估狀態失敗");
          setIsEvaluating(false);
        }
      };

      // 開始輪詢
      setTimeout(pollTaskStatus, 1000);
    } catch (error) {
      setEvaluationError("啟動評估失敗");
      setIsEvaluating(false);
    }
  };

  const getPerformanceColor = (value: number, metric: string) => {
    if (metric === "precision") {
      return value > 0.4
        ? "text-green-600"
        : value > 0.2
        ? "text-yellow-600"
        : "text-red-600";
    }
    return value > 0.8
      ? "text-green-600"
      : value > 0.6
      ? "text-yellow-600"
      : "text-red-600";
  };

  const getPerformanceBadge = (value: number, metric: string) => {
    if (metric === "precision") {
      return value > 0.4
        ? "bg-success"
        : value > 0.2
        ? "bg-warning"
        : "bg-danger";
    }
    return value > 0.8
      ? "bg-success"
      : value > 0.6
      ? "bg-warning"
      : "bg-danger";
  };

  return (
    <div className="space-y-6">
      {/* 評估模式開關 */}
      <div className="flex items-center space-x-4">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={isEvaluationMode}
            onChange={(e) => setIsEvaluationMode(e.target.checked)}
            className="mr-2"
          />
          <span className="font-medium">啟用評測模式</span>
        </label>
      </div>

      {isEvaluationMode && (
        <>
          {/* 評估配置 */}
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">評測配置</h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  分塊大小
                </label>
                <div className="flex flex-wrap gap-2">
                  {[200, 300, 400, 500, 600, 800, 1000].map((size) => (
                    <label key={size} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={chunkSizes.includes(size)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setChunkSizes([...chunkSizes, size]);
                          } else {
                            setChunkSizes(chunkSizes.filter((s) => s !== size));
                          }
                        }}
                        className="mr-1"
                      />
                      <span className="text-sm">{size}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  重疊比例
                </label>
                <div className="flex flex-wrap gap-2">
                  {[0.0, 0.1, 0.2, 0.3].map((ratio) => (
                    <label key={ratio} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={overlapRatios.includes(ratio)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setOverlapRatios([...overlapRatios, ratio]);
                          } else {
                            setOverlapRatios(
                              overlapRatios.filter((r) => r !== ratio)
                            );
                          }
                        }}
                        className="mr-1"
                      />
                      <span className="text-sm">
                        {(ratio * 100).toFixed(0)}%
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium mb-2">問題類型</label>
              <div className="flex flex-wrap gap-2">
                {[
                  "案例應用",
                  "情境分析",
                  "實務處理",
                  "法律後果",
                  "合規判斷",
                ].map((type) => (
                  <label key={type} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={questionTypes.includes(type)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setQuestionTypes([...questionTypes, type]);
                        } else {
                          setQuestionTypes(
                            questionTypes.filter((t) => t !== type)
                          );
                        }
                      }}
                      className="mr-1"
                    />
                    <span className="text-sm">{type}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium mb-2">問題數量</label>
              <input
                type="number"
                value={numQuestions}
                onChange={(e) => setNumQuestions(parseInt(e.target.value))}
                className="w-32 px-3 py-2 border rounded-md"
                min="5"
                max="20"
              />
            </div>

            <div className="mt-4">
              <button
                onClick={startEvaluation}
                disabled={isEvaluating || !docId}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
              >
                {isEvaluating ? "評測中..." : "開始評測"}
              </button>
            </div>
          </div>

          {/* 評估進度 */}
          {isEvaluating && (
            <div className="bg-yellow-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-2">評測進度</h3>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${evaluationProgress}%` }}
                ></div>
              </div>
              <p className="text-sm text-gray-600 mt-2">
                已完成 {evaluationProgress.toFixed(1)}%
              </p>
            </div>
          )}

          {/* 評估錯誤 */}
          {evaluationError && (
            <div className="bg-red-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-red-800 mb-2">
                評測錯誤
              </h3>
              <p className="text-red-600">{evaluationError}</p>
            </div>
          )}

          {/* 評估結果 */}
          {evaluationResults && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">評測結果</h3>

              {/* 最佳配置推薦 */}
              {evaluationResults.summary.best_config && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h4 className="text-lg font-semibold text-green-800 mb-2">
                    🎯 推薦配置
                  </h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="font-medium">Chunk Size:</span>{" "}
                      {evaluationResults.summary.best_config.config.chunk_size}
                    </div>
                    <div>
                      <span className="font-medium">Overlap Ratio:</span>{" "}
                      {
                        evaluationResults.summary.best_config.config
                          .overlap_ratio
                      }
                    </div>
                  </div>
                </div>
              )}

              {/* 結果表格 */}
              <div className="overflow-x-auto">
                <table className="min-w-full bg-white border border-gray-200 rounded-lg">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        配置
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Precision Omega
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Precision@5
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Recall@5
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        分塊數量
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        平均長度
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {evaluationResults.results.map((result, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-4 py-3 text-sm">
                          {result.config.chunk_size}/
                          {result.config.overlap_ratio}
                        </td>
                        <td className="px-4 py-3 text-sm">
                          <span
                            className={`badge ${getPerformanceBadge(
                              result.metrics.precision_omega,
                              "precision"
                            )}`}
                          >
                            {result.metrics.precision_omega.toFixed(3)}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm">
                          <span
                            className={`badge ${getPerformanceBadge(
                              result.metrics.precision_at_k[5],
                              "precision"
                            )}`}
                          >
                            {result.metrics.precision_at_k[5]?.toFixed(3) ||
                              "0.000"}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm">
                          <span
                            className={`badge ${getPerformanceBadge(
                              result.metrics.recall_at_k[5],
                              "recall"
                            )}`}
                          >
                            {result.metrics.recall_at_k[5]?.toFixed(3) ||
                              "0.000"}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm">
                          {result.metrics.chunk_count}
                        </td>
                        <td className="px-4 py-3 text-sm">
                          {result.metrics.avg_chunk_length.toFixed(1)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
