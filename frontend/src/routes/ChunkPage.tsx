import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";
import { api } from "../lib/api";
import { QASetUploader } from "../components/QASetUploader";
import { ChunkQASetUploader } from "../components/ChunkQASetUploader";
import { QAMappingDetails } from "../components/QAMappingDetails";

// 擴展Window接口以包含Bootstrap
declare global {
  interface Window {
    bootstrap: any;
  }
}

// 分塊策略類型定義
type ChunkStrategy =
  | "fixed_size"
  | "rcts_hierarchical"
  | "structured_hierarchical"
  | "hybrid"
  | "semantic";

// 策略參數接口
interface ChunkParams {
  fixed_size: {
    chunk_size: number;
    overlap: number;
  };
  rcts_hierarchical: {
    max_chunk_size: number;
    overlap_ratio: number;
    preserve_structure: boolean;
  };
  structured_hierarchical: {
    max_chunk_size: number;
    overlap_ratio: number;
    chunk_by: "chapter" | "section" | "article" | "item";
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
    name: "Fixed-Size",
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
  rcts_hierarchical: {
    name: "RCTS Hierarchical",
    description:
      "結合RecursiveCharacterTextSplitter和層次結構識別，智能分割法律文檔。",
    metrics: ["分隔符準確性", "結構保持度", "長文本處理", "語義完整性"],
    params: {
      max_chunk_size: {
        label: "最大分塊大小",
        min: 200,
        max: 2000,
        default: 1000,
        unit: "字符",
      },
      overlap_ratio: {
        label: "重疊比例",
        min: 0.05,
        max: 0.3,
        default: 0.1,
        unit: "比例",
      },
      preserve_structure: {
        label: "保持層次結構",
        type: "boolean",
        default: true,
        description: "在條文邊界強制分割，確保法律邏輯完整性",
      },
    },
  },
  structured_hierarchical: {
    name: "Structured Hierarchical (pending pilot)",
    description:
      "基於JSON結構數據，按照法律文檔的章-節-條-項結構進行智能分割。",
    metrics: ["結構準確性", "條文完整性", "引用關係保持", "分割粒度"],
    params: {
      max_chunk_size: {
        label: "最大分塊大小",
        min: 200,
        max: 2000,
        default: 1000,
        unit: "字符",
      },
      overlap_ratio: {
        label: "重疊比例",
        min: 0.05,
        max: 0.3,
        default: 0.1,
        unit: "比例",
      },
      chunk_by: {
        label: "分割單位",
        type: "select",
        options: [
          { value: "article", label: "按條文分割" },
          { value: "item", label: "按項分割" },
          { value: "section", label: "按節分割" },
          { value: "chapter", label: "按章分割" },
        ],
        default: "article",
        description: "選擇分割的粒度級別",
      },
    },
  },
  hybrid: {
    name: "Hybrid (Fixed + Semantic)",
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
      overlap_ratio: {
        label: "重疊比例",
        min: 0.05,
        max: 0.3,
        default: 0.1,
        unit: "比例",
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
    name: "Semantic",
    description: "基於語義相似度進行分割，保持語義連貫性。",
    metrics: ["語義連貫性", "相似度", "分塊質量"],
    params: {
      target_size: {
        label: "目標大小",
        min: 200,
        max: 1500,
        default: 500,
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
        min: 100,
        max: 1000,
        default: 200,
        unit: "字符",
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
  const { canChunk, chunk, docId, chunkMeta } = useRag();

  // 步驟1: QA Set上傳狀態
  const [uploadedQASetFile, setUploadedQASetFile] = useState<File | null>(null);

  // 步驟2: 多種分塊組合配置狀態
  const [selectedStrategies, setSelectedStrategies] = useState<ChunkStrategy[]>(
    ["fixed_size"]
  );
  const [chunkSizes, setChunkSizes] = useState<number[]>([300, 500, 800]);
  const [overlapRatios, setOverlapRatios] = useState<number[]>([0.0, 0.1, 0.2]);
  const [isChunking, setIsChunking] = useState(false);
  const [chunkingError, setChunkingError] = useState<string | null>(null);
  const [chunkingProgress, setChunkingProgress] = useState(0);
  const [chunkingTaskId, setChunkingTaskId] = useState<string | null>(null);

  // 步驟3: QA映射狀態
  const [qaMappingResult, setQAMappingResult] = useState<any>(null);
  const [chunkingCompleted, setChunkingCompleted] = useState(false); // 新增：分塊處理完成狀態
  const [qaMappingTaskId, setQAMappingTaskId] = useState<string | null>(null);
  const [qaMappingProgress, setQAMappingProgress] = useState(0);
  const [qaMappingError, setQAMappingError] = useState<string | null>(null);

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
  const handleQASetFileUploaded = (file: File) => {
    setUploadedQASetFile(file);
  };

  // 步驟2: 處理多種分塊組合配置
  const handleStrategyToggle = (strategy: ChunkStrategy) => {
    setSelectedStrategies((prev) =>
      prev.includes(strategy)
        ? prev.filter((s) => s !== strategy)
        : [...prev, strategy]
    );
  };

  const handleChunkSizeToggle = (size: number) => {
    setChunkSizes((prev) =>
      prev.includes(size) ? prev.filter((s) => s !== size) : [...prev, size]
    );
  };

  const handleOverlapRatioToggle = (ratio: number) => {
    setOverlapRatios((prev) =>
      prev.includes(ratio) ? prev.filter((r) => r !== ratio) : [...prev, ratio]
    );
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

  // 步驟2: 執行多種分塊組合操作
  const handleRunMultipleChunking = async () => {
    if (!canChunk) return;

    setIsChunking(true);
    setChunkingError(null);
    setChunkingResults([]);
    setChunkingProgress(0);

    try {
      // 調用新的批量分塊API
      const response = await api.startMultipleChunking({
        doc_id: docId!,
        strategies: selectedStrategies,
        chunk_sizes: chunkSizes,
        overlap_ratios: overlapRatios,
      });

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
            // 注意：不自動設置chunkingCompleted，需要用戶手動點擊「繼續到QA映射」
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
    setChunkingCompleted(false); // 重置分塊完成狀態
    setQAMappingResult(null);
    setQAMappingTaskId(null); // 重置QA映射任務ID
    setQAMappingProgress(0); // 重置QA映射進度
    setQAMappingError(null); // 重置QA映射錯誤
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
  const handleQAMappingComplete = (result: any) => {
    setQAMappingResult(result);
  };

  // 步驟3: 開始QA映射
  const handleStartQAMapping = async () => {
    if (!uploadedQASetFile || !chunkingResults.length) {
      setQAMappingError("請先完成QA set上傳和分塊處理");
      return;
    }

    try {
      setQAMappingError(null);
      setQAMappingProgress(0);

      // 讀取QA set文件內容
      const qaSetContent = await uploadedQASetFile.text();
      const qaSet = JSON.parse(qaSetContent);

      // 驗證QA set格式
      if (!Array.isArray(qaSet)) {
        setQAMappingError("QA set文件格式錯誤：應該是一個包含問題答案對的數組");
        return;
      }

      // 檢查是否有基本的QA結構
      if (qaSet.length === 0) {
        setQAMappingError("QA set文件為空");
        return;
      }

      // 檢查第一個項目是否有必要的字段
      const firstItem = qaSet[0];
      if (!firstItem.query || !firstItem.label) {
        setQAMappingError("QA set文件格式錯誤：缺少必要的字段（query, label）");
        return;
      }

      // 添加調試信息
      console.log("QA映射請求數據:");
      console.log("- doc_id:", docId);
      console.log("- qa_set長度:", qaSet.length);
      console.log("- qa_set結構:", {
        isArray: Array.isArray(qaSet),
        firstItemKeys: Object.keys(firstItem),
        sampleItem: {
          query: firstItem.query?.substring(0, 50) + "...",
          label: firstItem.label,
          hasSpans: !!firstItem.spans,
        },
      });
      console.log("- chunking_results長度:", chunkingResults.length);
      console.log(
        "- chunking_results結構:",
        chunkingResults.map((r) => ({
          strategy: r.strategy,
          has_chunks_with_span: !!r.chunks_with_span,
          chunks_with_span_length: r.chunks_with_span?.length || 0,
        }))
      );

      // 啟動QA映射任務
      const response = await api.startQAMapping({
        doc_id: docId!,
        qa_set: qaSet,
        chunking_results: chunkingResults,
        iou_threshold: 0.5,
      });

      setQAMappingTaskId(response.task_id);

      // 開始輪詢進度
      const pollProgress = async () => {
        try {
          const statusResponse = await api.getQAMappingStatus(response.task_id);
          setQAMappingProgress(statusResponse.progress * 100);

          if (statusResponse.status === "completed") {
            const resultResponse = await api.getQAMappingResult(
              response.task_id
            );
            setQAMappingResult(resultResponse);
            setQAMappingTaskId(null);
          } else if (statusResponse.status === "failed") {
            setQAMappingError(statusResponse.error || "QA映射失敗");
            setQAMappingTaskId(null);
          } else {
            // 繼續輪詢
            setTimeout(pollProgress, 1000);
          }
        } catch (error) {
          console.error("輪詢QA映射進度失敗:", error);
          setQAMappingError("獲取QA映射進度失敗");
          setQAMappingTaskId(null);
        }
      };

      // 開始輪詢
      setTimeout(pollProgress, 1000);
    } catch (error) {
      console.error("QA映射失敗:", error);
      setQAMappingError(error instanceof Error ? error.message : "QA映射失敗");
    }
  };

  // 評測相關函數
  const startEvaluation = async () => {
    if (!docId) {
      setEvaluationError("請先上傳文檔");
      return;
    }

    if (!qaMappingResult) {
      setEvaluationError("請先完成QA Set映射");
      return;
    }

    if (chunkingResults.length === 0) {
      setEvaluationError("請先完成多種分塊組合處理");
      return;
    }

    setEvaluationLoading(true);
    setEvaluationError(null);
    setEvaluationResults([]);
    setEvaluationComparison(null);

    try {
      // 使用QA set中的問題進行評測
      const testQueries = qaMappingResult.original_qa_set.map(
        (item: any) => item.query
      );

      // 使用新的策略評估API，基於已完成的分塊結果
      const response = await api.startStrategyEvaluation({
        doc_id: docId,
        chunking_results: chunkingResults,
        qa_mapping_result: qaMappingResult,
        test_queries: testQueries,
        k_values: [1, 3, 5, 10],
      });

      setCurrentTask({
        task_id: response.task_id,
        status: "running",
        created_at: new Date().toISOString(),
        total_configs: response.total_configs,
        completed_configs: 0,
        progress: 0,
      });

      // 開始輪詢進度更新
      clearProgressPolling(); // 清理之前的輪詢
      const interval = setInterval(() => {
        pollProgress(response.task_id);
      }, 1000); // 每秒輪詢一次
      setProgressInterval(interval);
    } catch (err) {
      setEvaluationError(`啟動評測失敗: ${err}`);
    } finally {
      setEvaluationLoading(false);
    }
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
                通過四個步驟完成分塊策略的評測：上傳QA set → 進行分塊 →
                分塊後映射 → 策略評估
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
                  {/* 步驟1: 上傳QA Set */}
                  <div className="col-12">
                    <div
                      className={`card ${
                        uploadedQASetFile ? "border-success" : "border-primary"
                      }`}
                    >
                      <div
                        className={`card-header ${
                          uploadedQASetFile
                            ? "bg-success text-white"
                            : "bg-primary text-white"
                        }`}
                      >
                        <h5 className="mb-0">步驟 1: 上傳QA Set</h5>
                      </div>
                      <div className="card-body">
                        <ChunkQASetUploader
                          onFileUploaded={handleQASetFileUploaded}
                        />
                      </div>
                    </div>
                  </div>

                  {/* 步驟2: 多種分塊組合處理 */}
                  <div className="col-12">
                    <div
                      className={`card ${
                        chunkingResults.length > 0
                          ? "border-success"
                          : uploadedQASetFile
                          ? "border-warning"
                          : "border-secondary"
                      }`}
                    >
                      <div
                        className={`card-header ${
                          chunkingResults.length > 0
                            ? "bg-success text-white"
                            : uploadedQASetFile
                            ? "bg-warning text-dark"
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
                                    <h5 className="card-title text-info">
                                      {selectedStrategies.length}
                                    </h5>
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
                                className="btn btn-primary"
                                onClick={() => {
                                  // 設置可以進行QA映射的狀態
                                  setChunkingCompleted(true);
                                  console.log("繼續到QA映射步驟");
                                }}
                                disabled={!uploadedQASetFile}
                              >
                                <i className="bi bi-arrow-right me-1"></i>
                                繼續到QA映射
                              </button>
                            </div>
                            {!uploadedQASetFile && (
                              <div className="mt-2">
                                <small className="text-muted">
                                  <i className="bi bi-info-circle me-1"></i>
                                  如需進行QA映射，請先上傳QA set文件
                                </small>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div>
                            {/* 分塊策略選擇 */}
                            <div className="mb-4">
                              <label className="form-label fw-bold">
                                選擇分塊策略（可多選）
                              </label>
                              <div className="row g-2">
                                {Object.entries(strategyInfo).map(
                                  ([key, info]) => (
                                    <div key={key} className="col-6">
                                      <div className="form-check">
                                        <input
                                          className="form-check-input"
                                          type="checkbox"
                                          id={`strategy-${key}`}
                                          checked={selectedStrategies.includes(
                                            key as ChunkStrategy
                                          )}
                                          onChange={() =>
                                            handleStrategyToggle(
                                              key as ChunkStrategy
                                            )
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
                                  )
                                )}
                              </div>
                            </div>

                            {/* 策略描述 */}
                            {selectedStrategies.length > 0 && (
                              <div className="mb-4">
                                <h6 className="text-primary">已選擇的策略</h6>
                                {selectedStrategies.map((strategy) => (
                                  <div key={strategy} className="mb-2">
                                    <strong>
                                      {strategyInfo[strategy].name}
                                    </strong>
                                    <p className="text-muted small mb-1">
                                      {strategyInfo[strategy].description}
                                    </p>
                                    <div className="small text-muted">
                                      評估指標：
                                      {strategyInfo[strategy].metrics.join(
                                        "、"
                                      )}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}

                            {/* 參數配置 */}
                            <div className="mb-4">
                              <h6>分塊參數組合</h6>
                              <div className="row g-3">
                                <div className="col-md-6">
                                  <label className="form-label">
                                    分塊大小（字符）
                                  </label>
                                  <div className="d-flex flex-wrap gap-2">
                                    {[
                                      200, 300, 400, 500, 600, 800, 1000, 1200,
                                    ].map((size) => (
                                      <div key={size} className="form-check">
                                        <input
                                          className="form-check-input"
                                          type="checkbox"
                                          id={`size-${size}`}
                                          checked={chunkSizes.includes(size)}
                                          onChange={() =>
                                            handleChunkSizeToggle(size)
                                          }
                                        />
                                        <label
                                          className="form-check-label small"
                                          htmlFor={`size-${size}`}
                                        >
                                          {size}
                                        </label>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                                <div className="col-md-6">
                                  <label className="form-label">重疊比例</label>
                                  <div className="d-flex flex-wrap gap-2">
                                    {[0.0, 0.1, 0.2, 0.3].map((ratio) => (
                                      <div key={ratio} className="form-check">
                                        <input
                                          className="form-check-input"
                                          type="checkbox"
                                          id={`ratio-${ratio}`}
                                          checked={overlapRatios.includes(
                                            ratio
                                          )}
                                          onChange={() =>
                                            handleOverlapRatioToggle(ratio)
                                          }
                                        />
                                        <label
                                          className="form-check-label small"
                                          htmlFor={`ratio-${ratio}`}
                                        >
                                          {(ratio * 100).toFixed(0)}%
                                        </label>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              </div>
                              <div className="mt-3">
                                <div className="alert alert-info">
                                  <small>
                                    <strong>組合數量：</strong>
                                    {selectedStrategies.length *
                                      chunkSizes.length *
                                      overlapRatios.length}{" "}
                                    種組合
                                    <br />
                                    <strong>選中的策略：</strong>
                                    {selectedStrategies
                                      .map((s) => strategyInfo[s].name)
                                      .join("、")}
                                    <br />
                                    <strong>分塊大小：</strong>
                                    {chunkSizes.join("、")} 字符
                                    <br />
                                    <strong>重疊比例：</strong>
                                    {overlapRatios
                                      .map((r) => `${(r * 100).toFixed(0)}%`)
                                      .join("、")}
                                  </small>
                                </div>
                              </div>
                            </div>

                            {/* 開始分塊按鈕 */}
                            <div className="d-grid">
                              <button
                                className="btn btn-warning btn-lg"
                                onClick={handleRunMultipleChunking}
                                disabled={
                                  isChunking ||
                                  selectedStrategies.length === 0 ||
                                  chunkSizes.length === 0 ||
                                  overlapRatios.length === 0
                                }
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
                                  `開始批量分塊 (${
                                    selectedStrategies.length *
                                    chunkSizes.length *
                                    overlapRatios.length
                                  } 種組合)`
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

                  {/* 步驟3: QA Set映射 (可選) */}
                  <div className="col-12">
                    <div
                      className={`card ${
                        uploadedQASetFile && qaMappingResult
                          ? "border-success"
                          : chunkingResults.length > 0 && uploadedQASetFile
                          ? "border-warning"
                          : "border-secondary"
                      }`}
                    >
                      <div
                        className={`card-header ${
                          uploadedQASetFile && qaMappingResult
                            ? "bg-success text-white"
                            : chunkingResults.length > 0 && uploadedQASetFile
                            ? "bg-warning text-dark"
                            : "bg-secondary text-white"
                        }`}
                      >
                        <h5 className="mb-0">步驟 3: QA Set映射 (可選)</h5>
                      </div>
                      <div className="card-body">
                        {!uploadedQASetFile ? (
                          <div className="text-center text-muted py-3">
                            <i className="bi bi-info-circle fs-1 d-block mb-2"></i>
                            <p>此步驟為可選，用於將QA set與分塊結果進行映射</p>
                            <p className="small">
                              如需進行評測，建議上傳QA set文件
                            </p>
                          </div>
                        ) : chunkingResults.length === 0 ? (
                          <div className="text-center text-muted py-3">
                            <i className="bi bi-hourglass-split fs-1 d-block mb-2"></i>
                            <p>請先完成步驟2：多種分塊組合處理</p>
                          </div>
                        ) : !chunkingCompleted ? (
                          <div className="text-center text-muted py-3">
                            <i className="bi bi-exclamation-triangle fs-1 d-block mb-2"></i>
                            <p>請先點擊「繼續到QA映射」按鈕</p>
                          </div>
                        ) : qaMappingResult ? (
                          <div>
                            <div className="alert alert-success mb-4">
                              <h6>✅ QA Set映射完成</h6>
                              <p className="mb-0">
                                已成功完成QA set與分塊的映射，共處理{" "}
                                {qaMappingResult.original_qa_set?.length || 0}{" "}
                                個問題
                              </p>
                            </div>

                            {/* 映射摘要 */}
                            {qaMappingResult.mapping_summary && (
                              <div className="row g-3 mb-4">
                                <div className="col-md-3">
                                  <div className="card bg-light">
                                    <div className="card-body text-center">
                                      <h5 className="card-title text-primary">
                                        {
                                          qaMappingResult.mapping_summary
                                            .total_configs
                                        }
                                      </h5>
                                      <p className="card-text small">
                                        分塊配置數
                                      </p>
                                    </div>
                                  </div>
                                </div>
                                <div className="col-md-3">
                                  <div className="card bg-light">
                                    <div className="card-body text-center">
                                      <h5 className="card-title text-info">
                                        {qaMappingResult.original_qa_set
                                          ?.length || 0}
                                      </h5>
                                      <p className="card-text small">
                                        QA問題數
                                      </p>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* 映射結果詳情 */}
                            {qaMappingResult.mapping_results && (
                              <div className="mb-4">
                                <h6>映射結果詳情</h6>
                                <div className="table-responsive">
                                  <table className="table table-sm table-striped">
                                    <thead>
                                      <tr>
                                        <th>配置</th>
                                        <th>策略</th>
                                        <th>分塊數量</th>
                                        <th>正例問題</th>
                                        <th>負例問題</th>
                                        <th>映射成功率</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {Object.entries(
                                        qaMappingResult.mapping_results
                                      ).map(
                                        ([configId, result]: [string, any]) => (
                                          <tr key={configId}>
                                            <td>
                                              <small>
                                                {result.strategy} |{" "}
                                                {result.config.chunk_size} |{" "}
                                                {(
                                                  result.config.overlap_ratio *
                                                  100
                                                ).toFixed(0)}
                                                %
                                              </small>
                                            </td>
                                            <td>{result.strategy}</td>
                                            <td>{result.chunk_count}</td>
                                            <td>
                                              {
                                                result.mapping_stats
                                                  .positive_questions
                                              }
                                            </td>
                                            <td>
                                              {
                                                result.mapping_stats
                                                  .negative_questions
                                              }
                                            </td>
                                            <td>
                                              {result.mapping_stats.mapping_success_rate?.toFixed(
                                                1
                                              ) || "0.0"}
                                              %
                                            </td>
                                          </tr>
                                        )
                                      )}
                                    </tbody>
                                  </table>
                                </div>

                                {/* 映射後的QA set詳細內容 */}
                                <QAMappingDetails
                                  mappingResults={
                                    qaMappingResult.mapping_results
                                  }
                                />
                              </div>
                            )}
                          </div>
                        ) : (
                          <div>
                            <div className="alert alert-info">
                              <h6>📋 準備進行QA映射</h6>
                              <p className="mb-0">
                                已上傳QA set文件，可以開始進行QA
                                set與分塊結果的映射
                              </p>
                            </div>

                            <div className="d-grid">
                              <button
                                className="btn btn-primary btn-lg"
                                onClick={handleStartQAMapping}
                                disabled={
                                  !chunkingCompleted || !!qaMappingTaskId
                                }
                              >
                                {qaMappingTaskId ? (
                                  <>
                                    <span
                                      className="spinner-border spinner-border-sm me-2"
                                      role="status"
                                      aria-hidden="true"
                                    ></span>
                                    QA映射中... ({qaMappingProgress.toFixed(1)}
                                    %)
                                  </>
                                ) : (
                                  <>
                                    <i className="bi bi-link-45deg me-2"></i>
                                    開始QA Set映射
                                  </>
                                )}
                              </button>
                            </div>

                            {/* 進度條 */}
                            {qaMappingTaskId && (
                              <div className="mt-3">
                                <div className="progress">
                                  <div
                                    className="progress-bar progress-bar-striped progress-bar-animated"
                                    role="progressbar"
                                    style={{ width: `${qaMappingProgress}%` }}
                                    aria-valuenow={qaMappingProgress}
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                  >
                                    {qaMappingProgress.toFixed(1)}%
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* 錯誤提示 */}
                            {qaMappingError && (
                              <div className="mt-3">
                                <div
                                  className="alert alert-danger"
                                  role="alert"
                                >
                                  <i className="bi bi-exclamation-triangle me-1"></i>
                                  {qaMappingError}
                                </div>
                              </div>
                            )}

                            {!chunkingCompleted && (
                              <div className="mt-2">
                                <div
                                  className="alert alert-warning"
                                  role="alert"
                                >
                                  <i className="bi bi-exclamation-triangle me-1"></i>
                                  請先完成分塊組合處理
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 步驟4: 策略評估 */}
                  <div className="col-12">
                    <div
                      className={`card ${
                        evaluationResults.length > 0
                          ? "border-success"
                          : "border-warning"
                      }`}
                    >
                      <div
                        className={`card-header ${
                          evaluationResults.length > 0
                            ? "bg-success text-white"
                            : "bg-warning text-dark"
                        }`}
                      >
                        <h5 className="mb-0">步驟 4: 策略評估</h5>
                      </div>
                      <div className="card-body">
                        {chunkingResults.length > 0 ? (
                          <div>
                            {/* 評測配置信息 */}
                            <div className="mb-4">
                              <h6>策略評估配置</h6>
                              <div className="alert alert-info">
                                <h6>📊 評估說明</h6>
                                <p className="mb-2">
                                  將使用步驟3中映射的QA Set對步驟2中生成的{" "}
                                  {chunkingResults.length}{" "}
                                  種分塊組合進行檢索測試，
                                  計算P@K和R@K指標來評估各分割策略的表現。
                                </p>
                                <div className="row">
                                  <div className="col-md-6">
                                    <strong>分塊組合數：</strong>{" "}
                                    {chunkingResults.length}
                                  </div>
                                  <div className="col-md-6">
                                    <strong>QA問題數：</strong>{" "}
                                    {qaMappingResult?.original_qa_set?.length ||
                                      0}
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* 開始評測按鈕 */}
                            <div className="d-grid mb-4">
                              <button
                                className="btn btn-primary btn-lg"
                                onClick={startEvaluation}
                                disabled={
                                  evaluationLoading ||
                                  !qaMappingResult ||
                                  chunkingResults.length === 0
                                }
                              >
                                {evaluationLoading ? (
                                  <>
                                    <span
                                      className="spinner-border spinner-border-sm me-2"
                                      role="status"
                                      aria-hidden="true"
                                    ></span>
                                    評測中...
                                  </>
                                ) : !qaMappingResult ? (
                                  "請先完成QA Set映射"
                                ) : chunkingResults.length === 0 ? (
                                  "請先完成多種分塊組合處理"
                                ) : (
                                  "開始策略評估"
                                )}
                              </button>
                            </div>

                            {/* 評測錯誤提示 */}
                            {evaluationError && (
                              <div className="alert alert-danger" role="alert">
                                {evaluationError}
                              </div>
                            )}

                            {/* 評測進度 */}
                            {currentTask && (
                              <div className="mb-4">
                                <h6>評測進度</h6>
                                <div className="progress mb-2">
                                  <div
                                    className="progress-bar"
                                    role="progressbar"
                                    style={{
                                      width: `${currentTask.progress * 100}%`,
                                    }}
                                    aria-valuenow={currentTask.progress * 100}
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                  >
                                    {(currentTask.progress * 100).toFixed(1)}%
                                  </div>
                                </div>
                                <div className="d-flex justify-content-between small text-muted">
                                  <span>
                                    已完成 {currentTask.completed_configs} /{" "}
                                    {currentTask.total_configs} 個配置
                                  </span>
                                  <span>
                                    預計剩餘時間:{" "}
                                    {calculateEstimatedTime(currentTask)} 分鐘
                                  </span>
                                </div>
                              </div>
                            )}

                            {/* 評測結果 */}
                            {evaluationResults.length > 0 && (
                              <div>
                                <h6>評測結果</h6>

                                {/* 最佳配置 */}
                                {getBestConfig() && (
                                  <div className="alert alert-success mb-4">
                                    <h6 className="alert-heading">
                                      🎯 最佳配置 (綜合評分最高)
                                    </h6>
                                    <div className="row">
                                      <div className="col-md-6">
                                        <h6>配置參數</h6>
                                        <div className="table-responsive">
                                          <table className="table table-sm">
                                            <tbody>
                                              <tr>
                                                <td>
                                                  <strong>分塊大小:</strong>
                                                </td>
                                                <td>
                                                  {
                                                    getBestConfig()?.config
                                                      .chunk_size
                                                  }
                                                </td>
                                              </tr>
                                              <tr>
                                                <td>
                                                  <strong>重疊比例:</strong>
                                                </td>
                                                <td>
                                                  {(
                                                    (getBestConfig()?.config
                                                      .overlap_ratio || 0) * 100
                                                  ).toFixed(0)}
                                                  %
                                                </td>
                                              </tr>
                                            </tbody>
                                          </table>
                                        </div>
                                      </div>
                                      <div className="col-md-6">
                                        <h6>性能指標</h6>
                                        <div className="table-responsive">
                                          <table className="table table-sm">
                                            <tbody>
                                              <tr>
                                                <td>
                                                  <strong>Precision@3:</strong>
                                                </td>
                                                <td>
                                                  <span className="badge bg-success">
                                                    {getBestConfig()?.metrics.precision_at_k[3]?.toFixed(
                                                      3
                                                    ) || "0.000"}
                                                  </span>
                                                </td>
                                              </tr>
                                              <tr>
                                                <td>
                                                  <strong>Recall@3:</strong>
                                                </td>
                                                <td>
                                                  <span className="badge bg-info">
                                                    {getBestConfig()?.metrics.recall_at_k[3]?.toFixed(
                                                      3
                                                    ) || "0.000"}
                                                  </span>
                                                </td>
                                              </tr>
                                              <tr>
                                                <td>
                                                  <strong>
                                                    Precision Omega:
                                                  </strong>
                                                </td>
                                                <td>
                                                  <span className="badge bg-warning">
                                                    {getBestConfig()?.metrics.precision_omega?.toFixed(
                                                      3
                                                    ) || "0.000"}
                                                  </span>
                                                </td>
                                              </tr>
                                            </tbody>
                                          </table>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                )}

                                {/* 詳細結果表格 */}
                                <div className="table-responsive mb-4">
                                  <table className="table table-striped">
                                    <thead>
                                      <tr>
                                        <th>配置</th>
                                        <th>Precision@3</th>
                                        <th>Recall@3</th>
                                        <th>Precision Omega</th>
                                        <th>分塊數量</th>
                                        <th>平均長度</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {evaluationResults.map(
                                        (result, index) => (
                                          <tr key={index}>
                                            <td>
                                              <div className="small">
                                                <strong>固定大小分割</strong>
                                                <br />
                                                Size: {result.config.chunk_size}
                                                <br />
                                                Overlap:{" "}
                                                {(
                                                  result.config.overlap_ratio *
                                                  100
                                                ).toFixed(0)}
                                                %
                                              </div>
                                            </td>
                                            <td>
                                              <span
                                                className={`badge ${
                                                  result.metrics
                                                    .precision_at_k[3] > 0.4
                                                    ? "bg-success"
                                                    : result.metrics
                                                        .precision_at_k[3] > 0.2
                                                    ? "bg-warning"
                                                    : "bg-danger"
                                                }`}
                                              >
                                                {result.metrics.precision_at_k[3]?.toFixed(
                                                  3
                                                ) || "0.000"}
                                              </span>
                                            </td>
                                            <td>
                                              <span
                                                className={`badge ${
                                                  result.metrics
                                                    .recall_at_k[3] > 0.8
                                                    ? "bg-success"
                                                    : result.metrics
                                                        .recall_at_k[3] > 0.6
                                                    ? "bg-warning"
                                                    : "bg-danger"
                                                }`}
                                              >
                                                {result.metrics.recall_at_k[3]?.toFixed(
                                                  3
                                                ) || "0.000"}
                                              </span>
                                            </td>
                                            <td>
                                              <span
                                                className={`badge ${
                                                  result.metrics
                                                    .precision_omega > 0.4
                                                    ? "bg-success"
                                                    : result.metrics
                                                        .precision_omega > 0.2
                                                    ? "bg-warning"
                                                    : "bg-danger"
                                                }`}
                                              >
                                                {result.metrics.precision_omega?.toFixed(
                                                  3
                                                ) || "0.000"}
                                              </span>
                                            </td>
                                            <td>
                                              {result.metrics.chunk_count}
                                            </td>
                                            <td>
                                              {result.metrics.avg_chunk_length.toFixed(
                                                1
                                              )}{" "}
                                              字符
                                            </td>
                                          </tr>
                                        )
                                      )}
                                    </tbody>
                                  </table>
                                </div>

                                {/* 導出報告按鈕 */}
                                <div className="d-grid">
                                  <button
                                    className="btn btn-outline-primary"
                                    onClick={exportEvaluationReport}
                                  >
                                    導出評測報告
                                  </button>
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="text-center text-muted py-3">
                            <i className="bi bi-hourglass-split fs-1 d-block mb-2"></i>
                            <p>請先完成前面的步驟</p>
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
