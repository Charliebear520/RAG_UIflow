import React, { useState, useEffect } from "react";
import { api } from "../lib/api";
import { QASetUploader } from "./QASetUploader";

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
  // Default evaluation mode ON and tuned defaults
  const [isEvaluationMode, setIsEvaluationMode] = useState(true);
  const [evaluationMode, setEvaluationMode] = useState<"qa_set" | "generated">(
    "qa_set"
  ); // 新增：評估模式選擇
  const [qaMappingResult, setQAMappingResult] = useState<any>(null); // 新增：QA映射結果
  const [chunkSizes, setChunkSizes] = useState<number[]>([300, 600, 900]);
  const [overlapRatios, setOverlapRatios] = useState<number[]>([0.0, 0.1]);
  const [questionTypes, setQuestionTypes] = useState<string[]>([
    "案例應用",
    "情境分析",
    "實務處理",
    "法律後果",
    "合規判斷",
  ]);
  const [numQuestions, setNumQuestions] = useState<number>(20);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationTaskId, setEvaluationTaskId] = useState<string | null>(null);
  const [evaluationProgress, setEvaluationProgress] = useState(0);
  const [evaluationResults, setEvaluationResults] =
    useState<EvaluationResults | null>(null);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);

  const totalCombos = chunkSizes.length * overlapRatios.length;
  const tooManyCombos = totalCombos > 8; // guardrail

  const handleQAMappingComplete = (result: any) => {
    setQAMappingResult(result);
    // 可以自動開始評估，或者讓用戶手動開始
  };

  const startEvaluation = async () => {
    if (!docId) return;
    if (tooManyCombos) {
      setEvaluationError("參數組合過多，請減少選項（上限 8 組）");
      return;
    }

    setIsEvaluating(true);
    setEvaluationError(null);
    setEvaluationResults(null);

    try {
      let response;

      if (evaluationMode === "qa_set" && qaMappingResult) {
        // 使用QA set映射結果進行評估
        response = await api.startFixedSizeEvaluation({
          doc_id: docId,
          chunk_sizes: chunkSizes,
          overlap_ratios: overlapRatios,
          test_queries: qaMappingResult.original_qa_set.map(
            (item: any) => item.query
          ),
          k_values: [1, 3, 5, 10],
        });
      } else {
        // 使用生成的問題進行評估
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
            const statusResponse = await api.getEvaluationTask(
              response.task_id
            );

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
        return;
      }

      // 對於QA set模式，使用不同的API流程
      if (evaluationMode === "qa_set") {
        setEvaluationTaskId(response.task_id);

        // 輪詢任務狀態
        const pollTaskStatus = async () => {
          try {
            const statusResponse = await api.getEvaluationStatus(
              response.task_id
            );

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
      }
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
        <span className="text-sm text-gray-600">
          組合數：{totalCombos}（上限 8）
        </span>
      </div>

      {isEvaluationMode && (
        <>
          {/* 評估模式選擇 */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">評估模式</h3>
            <div className="flex space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="qa_set"
                  checked={evaluationMode === "qa_set"}
                  onChange={(e) =>
                    setEvaluationMode(e.target.value as "qa_set" | "generated")
                  }
                  className="mr-2"
                />
                <span>QA Set上傳模式</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="generated"
                  checked={evaluationMode === "generated"}
                  onChange={(e) =>
                    setEvaluationMode(e.target.value as "qa_set" | "generated")
                  }
                  className="mr-2"
                />
                <span>自動生成問題模式</span>
              </label>
            </div>
          </div>

          {/* QA Set上傳模式 */}
          {evaluationMode === "qa_set" && (
            <QASetUploader
              docId={docId}
              onMappingComplete={handleQAMappingComplete}
            />
          )}

          {/* 評估配置 */}
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">
              {evaluationMode === "qa_set"
                ? "評測配置（基於QA Set）"
                : "評測配置（自動生成問題）"}
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  分塊大小
                </label>
                <div className="flex flex-wrap gap-2">
                  {[200, 300, 400, 500, 600, 800, 900, 1000].map((size) => (
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

            {evaluationMode === "generated" && (
              <>
                <div className="mt-4">
                  <label className="block text-sm font-medium mb-2">
                    問題類型
                  </label>
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
                  <label className="block text-sm font-medium mb-2">
                    問題數量
                  </label>
                  <input
                    type="number"
                    value={numQuestions}
                    onChange={(e) => setNumQuestions(parseInt(e.target.value))}
                    className="w-32 px-3 py-2 border rounded-md"
                    min="5"
                    max="50"
                  />
                </div>
              </>
            )}

            <div className="mt-4">
              <button
                onClick={startEvaluation}
                disabled={
                  isEvaluating ||
                  !docId ||
                  (evaluationMode === "qa_set" && !qaMappingResult)
                }
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
              >
                {isEvaluating
                  ? "評測中..."
                  : evaluationMode === "qa_set" && !qaMappingResult
                  ? "請先完成QA Set映射"
                  : "開始評測"}
              </button>
              {evaluationMode === "qa_set" && !qaMappingResult && (
                <p className="text-sm text-gray-600 mt-2">
                  請先上傳QA set並完成chunk映射後再開始評測
                </p>
              )}
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
                      <span className="font-medium">策略:</span>{" "}
                      {evaluationResults.summary.best_config.config.strategy ||
                        "固定大小分割"}
                    </div>
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
                    {evaluationResults.summary.best_config.config.chunk_by && (
                      <div>
                        <span className="font-medium">分割單位:</span>{" "}
                        {evaluationResults.summary.best_config.config.chunk_by}
                      </div>
                    )}
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
                          <div className="font-medium">
                            {(result.config as any).strategy || "固定大小分割"}
                          </div>
                          <div className="text-xs text-gray-500">
                            Size: {result.config.chunk_size} | Overlap:{" "}
                            {(result.config.overlap_ratio * 100).toFixed(0)}%
                            {(result.config as any).chunk_by && (
                              <div className="mt-1">
                                分割單位:{" "}
                                {(result.config as any).chunk_by === "article"
                                  ? "按條文分割"
                                  : (result.config as any).chunk_by === "item"
                                  ? "按項分割"
                                  : (result.config as any).chunk_by ===
                                    "section"
                                  ? "按節分割"
                                  : (result.config as any).chunk_by ===
                                    "chapter"
                                  ? "按章分割"
                                  : (result.config as any).chunk_by}
                              </div>
                            )}
                            {(result.config as any).preserve_structure !==
                              undefined && (
                              <div className="mt-1">
                                保持結構:{" "}
                                {(result.config as any).preserve_structure
                                  ? "是"
                                  : "否"}
                              </div>
                            )}
                            {(result.config as any).level_depth !==
                              undefined && (
                              <div className="mt-1">
                                層次深度: {(result.config as any).level_depth}
                              </div>
                            )}
                            {(result.config as any).similarity_threshold !==
                              undefined && (
                              <div className="mt-1">
                                相似度閾值:{" "}
                                {(result.config as any).similarity_threshold}
                              </div>
                            )}
                            {(result.config as any).semantic_threshold !==
                              undefined && (
                              <div className="mt-1">
                                語義閾值:{" "}
                                {(result.config as any).semantic_threshold}
                              </div>
                            )}
                            {(result.config as any).step_size !== undefined && (
                              <div className="mt-1">
                                步長: {(result.config as any).step_size}
                              </div>
                            )}
                            {(result.config as any).switch_threshold !==
                              undefined && (
                              <div className="mt-1">
                                切換閾值:{" "}
                                {(result.config as any).switch_threshold}
                              </div>
                            )}
                            {(result.config as any).secondary_size !==
                              undefined && (
                              <div className="mt-1">
                                次要大小:{" "}
                                {(result.config as any).secondary_size}
                              </div>
                            )}
                          </div>
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
