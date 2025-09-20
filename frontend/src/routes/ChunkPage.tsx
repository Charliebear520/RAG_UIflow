import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";
import { api } from "../lib/api";

// 分塊策略類型定義
type ChunkStrategy =
  | "fixed_size"
  | "hierarchical"
  | "rcts_hierarchical"
  | "structured_hierarchical"
  | "semantic"
  | "sliding_window"
  | "llm_semantic"
  | "hybrid";

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
  semantic: {
    max_chunk_size: number;
    similarity_threshold: number;
    overlap: number;
    context_window: number;
  };
  sliding_window: {
    window_size: number;
    step_size: number;
    overlap_ratio: number;
    boundary_aware: boolean;
    min_chunk_size: number;
    max_chunk_size: number;
    preserve_sentences: boolean;
  };
  llm_semantic: {
    max_chunk_size: number;
    semantic_threshold: number;
    overlap: number;
    context_window: number;
  };
  hybrid: {
    primary_size: number;
    secondary_size: number;
    overlap_ratio: number;
    switch_threshold: number;
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
  rcts_hierarchical: {
    name: "RCTS層次分割",
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
    name: "結構化層次分割",
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
  sliding_window: {
    name: "滑動視窗分割",
    description:
      "使用固定大小的滑動視窗進行分割，支援邊界感知和重疊控制，確保內容的連續性。",
    metrics: ["分塊數量", "視窗覆蓋率", "重疊率", "內容連續性", "邊界保持度"],
    params: {
      window_size: {
        label: "視窗大小",
        min: 200,
        max: 1500,
        default: 500,
        unit: "字符",
      },
      step_size: {
        label: "步長",
        min: 50,
        max: 500,
        default: 250,
        unit: "字符",
      },
      overlap_ratio: {
        label: "重疊比例",
        min: 0,
        max: 0.5,
        default: 0.1,
        unit: "比例",
        step: 0.05,
      },
      boundary_aware: {
        label: "邊界感知",
        type: "boolean",
        default: true,
      },
      min_chunk_size: {
        label: "最小分塊大小",
        min: 50,
        max: 500,
        default: 100,
        unit: "字符",
      },
      max_chunk_size: {
        label: "最大分塊大小",
        min: 500,
        max: 2000,
        default: 1000,
        unit: "字符",
      },
      preserve_sentences: {
        label: "保持句子完整性",
        type: "boolean",
        default: true,
      },
    },
  },
  llm_semantic: {
    name: "LLM輔助語義分割",
    description: "結合LLM的語義理解能力進行智能分割，確保語義連貫性。",
    metrics: ["分塊數量", "語義連貫性", "LLM準確性", "分割點質量"],
    params: {
      max_chunk_size: {
        label: "最大分塊大小",
        min: 300,
        max: 1200,
        default: 500,
        unit: "字符",
      },
      semantic_threshold: {
        label: "語義閾值",
        min: 0.1,
        max: 0.9,
        default: 0.7,
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
    name: "語義分割",
    description: "基於語義相似性進行智能分割，確保每個分塊在語義上保持連貫性。",
    metrics: ["分塊數量", "語義連貫性", "相似度分佈", "分割點質量"],
    params: {
      max_chunk_size: {
        label: "最大分塊大小",
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
  const [selectedStrategy, setSelectedStrategy] =
    useState<ChunkStrategy>("fixed_size");

  // 當策略變化時，更新評測配置
  useEffect(() => {
    setEvaluationConfig((prev) => ({
      ...prev,
      strategy: selectedStrategy,
    }));
  }, [selectedStrategy]);
  const [params, setParams] = useState<ChunkParams>({
    fixed_size: { chunk_size: 500, overlap: 50 },
    hierarchical: {
      max_chunk_size: 1000,
      min_chunk_size: 200,
      overlap: 50,
      level_depth: 3,
    },
    rcts_hierarchical: {
      max_chunk_size: 1000,
      overlap_ratio: 0.1,
      preserve_structure: true,
    },
    structured_hierarchical: {
      max_chunk_size: 1000,
      overlap_ratio: 0.1,
      chunk_by: "article",
    },
    semantic: {
      max_chunk_size: 600,
      similarity_threshold: 0.6,
      overlap: 50,
      context_window: 100,
    },
    sliding_window: {
      window_size: 500,
      step_size: 250,
      overlap_ratio: 0.1,
      boundary_aware: true,
      min_chunk_size: 100,
      max_chunk_size: 1000,
      preserve_sentences: true,
    },
    llm_semantic: {
      max_chunk_size: 500,
      semantic_threshold: 0.7,
      overlap: 50,
      context_window: 100,
    },
    hybrid: {
      primary_size: 600,
      secondary_size: 400,
      overlap_ratio: 0.1,
      switch_threshold: 0.5,
    },
  });
  const [busy, setBusy] = useState(false);
  const [chunkResults, setChunkResults] = useState<any>(null);

  // 評測相關狀態
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [evaluationConfig, setEvaluationConfig] = useState({
    chunk_sizes: [300, 500, 800],
    overlap_ratios: [0.0, 0.1, 0.2],
    strategy: "fixed_size", // 新增：分割策略
    test_queries: [
      "著作權的定義是什麼？",
      "什麼情況下可以合理使用他人作品？",
      "侵犯著作權的法律後果是什麼？",
      "著作權的保護期限是多久？",
      "如何申請著作權登記？",
    ],
    k_values: [1, 3, 5, 10],
    // 策略特定參數選項 - 預設包含所有排列組合
    chunk_by_options: ["article", "item", "section", "chapter"], // 結構化層次分割
    preserve_structure_options: [true, false], // RCTS層次分割
    level_depth_options: [2, 3, 4], // 層次分割
    similarity_threshold_options: [0.5, 0.6, 0.7], // 語義分割
    semantic_threshold_options: [0.6, 0.7, 0.8], // LLM語義分割
    switch_threshold_options: [0.3, 0.5, 0.7], // 混合分割
    min_chunk_size_options: [100, 200, 300], // 層次分割
    context_window_options: [50, 100, 150], // 語義分割
    step_size_options: [200, 250, 300], // 滑動視窗
    window_size_options: [400, 500, 600, 800], // 滑動視窗
    boundary_aware_options: [true, false], // 滑動視窗
    preserve_sentences_options: [true, false], // 滑動視窗
    min_chunk_size_options_sw: [50, 100, 150], // 滑動視窗專用
    max_chunk_size_options_sw: [800, 1000, 1200], // 滑動視窗專用
    secondary_size_options: [300, 400, 500], // 混合分割
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

  // 問題生成相關狀態
  const [generatedQuestions, setGeneratedQuestions] = useState<any[]>([]);
  const [questionLoading, setQuestionLoading] = useState(false);
  const [questionError, setQuestionError] = useState<string | null>(null);
  const [numQuestions, setNumQuestions] = useState(10);

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
  const [questionTypes, setQuestionTypes] = useState<string[]>([
    "案例應用",
    "情境分析",
    "實務處理",
    "法律後果",
    "合規判斷",
  ]);
  const [difficultyLevels, setDifficultyLevels] = useState<string[]>([
    "基礎",
    "進階",
    "應用",
  ]);

  const handleParamChange = (
    strategy: ChunkStrategy,
    param: string,
    value: number | boolean | string | string[]
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
        case "rcts_hierarchical":
          chunkSize = (currentParams as ChunkParams["rcts_hierarchical"])
            .max_chunk_size;
          break;
        case "structured_hierarchical":
          chunkSize = (currentParams as ChunkParams["structured_hierarchical"])
            .max_chunk_size;
          break;
        case "semantic":
          chunkSize = (currentParams as ChunkParams["semantic"]).max_chunk_size;
          break;
        case "sliding_window":
          chunkSize = (currentParams as ChunkParams["sliding_window"])
            .window_size;
          break;
        case "llm_semantic":
          chunkSize = (currentParams as ChunkParams["llm_semantic"])
            .max_chunk_size;
          break;
        case "hybrid":
          chunkSize = (currentParams as ChunkParams["hybrid"]).primary_size;
          break;
        default:
          chunkSize = 500;
      }

      // 構建策略特定的參數
      let strategyParams: any = { strategy: selectedStrategy };

      switch (selectedStrategy) {
        case "hierarchical":
          strategyParams.hierarchical_params = {
            min_chunk_size: (currentParams as ChunkParams["hierarchical"])
              .min_chunk_size,
            level_depth: (currentParams as ChunkParams["hierarchical"])
              .level_depth,
          };
          break;
        case "rcts_hierarchical":
          strategyParams.rcts_hierarchical_params = {
            overlap_ratio: (currentParams as ChunkParams["rcts_hierarchical"])
              .overlap_ratio,
            preserve_structure: (
              currentParams as ChunkParams["rcts_hierarchical"]
            ).preserve_structure,
          };
          break;
        case "structured_hierarchical":
          strategyParams.structured_hierarchical_params = {
            overlap_ratio: (
              currentParams as ChunkParams["structured_hierarchical"]
            ).overlap_ratio,
            chunk_by: (currentParams as ChunkParams["structured_hierarchical"])
              .chunk_by,
          };
          break;
        case "semantic":
          strategyParams.semantic_params = {
            similarity_threshold: (currentParams as ChunkParams["semantic"])
              .similarity_threshold,
            context_window: (currentParams as ChunkParams["semantic"])
              .context_window,
          };
          break;
        case "sliding_window":
          strategyParams.sliding_window_params = {
            step_size: (currentParams as ChunkParams["sliding_window"])
              .step_size,
            overlap_ratio: (currentParams as ChunkParams["sliding_window"])
              .overlap_ratio,
            boundary_aware: (currentParams as ChunkParams["sliding_window"])
              .boundary_aware,
            min_chunk_size: (currentParams as ChunkParams["sliding_window"])
              .min_chunk_size,
            max_chunk_size: (currentParams as ChunkParams["sliding_window"])
              .max_chunk_size,
            preserve_sentences: (currentParams as ChunkParams["sliding_window"])
              .preserve_sentences,
          };
          break;
        case "llm_semantic":
          strategyParams.llm_semantic_params = {
            semantic_threshold: (currentParams as ChunkParams["llm_semantic"])
              .semantic_threshold,
            context_window: (currentParams as ChunkParams["llm_semantic"])
              .context_window,
          };
          break;
        case "hybrid":
          strategyParams.hybrid_params = {
            secondary_size: (currentParams as ChunkParams["hybrid"])
              .secondary_size,
            overlap_ratio: (currentParams as ChunkParams["hybrid"])
              .overlap_ratio,
            switch_threshold: (currentParams as ChunkParams["hybrid"])
              .switch_threshold,
          };
          break;
      }

      // 獲取overlap值，根據策略類型處理
      let overlapValue: number;
      if ("overlap" in currentParams) {
        overlapValue = currentParams.overlap;
      } else if ("overlap_ratio" in currentParams) {
        // 對於使用overlap_ratio的策略，計算實際overlap值
        overlapValue = Math.round(chunkSize * currentParams.overlap_ratio);
      } else {
        // 默認值
        overlapValue = 50;
      }

      const result = await chunk(
        chunkSize,
        overlapValue,
        selectedStrategy,
        strategyParams
      );

      // 使用真實的chunking結果
      setChunkResults({
        strategy: selectedStrategy,
        metrics: {
          chunk_count: result.num_chunks,
          avg_length: result.metrics.avg_length,
          length_variance: result.metrics.length_variance,
          overlap_rate: result.metrics.overlap_rate,
          min_length: result.metrics.min_length,
          max_length: result.metrics.max_length,
        },
        chunks: result.sample || [],
        all_chunks: result.all_chunks || [],
        full_chunks: result.num_chunks,
        chunk_size: result.chunk_size,
        overlap: result.overlap,
      });
    } finally {
      setBusy(false);
    }
  };

  // 評測相關函數
  const startEvaluation = async () => {
    if (!docId) {
      setEvaluationError("請先上傳文檔");
      return;
    }

    setEvaluationLoading(true);
    setEvaluationError(null);
    setEvaluationResults([]);
    setEvaluationComparison(null);

    try {
      const response = await api.startFixedSizeEvaluation({
        doc_id: docId,
        ...evaluationConfig,
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

  const generateQuestions = async () => {
    if (!docId) {
      setQuestionError("請先上傳文檔");
      return;
    }

    setQuestionLoading(true);
    setQuestionError(null);
    setGeneratedQuestions([]);

    try {
      const response = await api.generateQuestions({
        doc_id: docId,
        num_questions: numQuestions,
        question_types: questionTypes,
        difficulty_levels: difficultyLevels,
      });

      console.log("問題生成響應:", response); // 調試用

      if (response.success) {
        console.log("成功生成問題:", response.result.questions.length, "個"); // 調試用
        setGeneratedQuestions(response.result.questions);
        // 將生成的問題添加到測試查詢中
        const newQueries = response.result.questions.map(
          (q: any) => q.question
        );
        setEvaluationConfig((prev) => ({
          ...prev,
          test_queries: [...prev.test_queries, ...newQueries],
        }));
      } else {
        console.log("響應顯示失敗:", response); // 調試用
        setQuestionError("問題生成失敗");
      }
    } catch (err) {
      console.error("問題生成異常:", err); // 調試用
      setQuestionError(`問題生成失敗: ${err}`);
    } finally {
      setQuestionLoading(false);
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

  const exportQuestions = () => {
    if (!generatedQuestions.length) return;

    const report = {
      generation_info: {
        doc_id: docId,
        num_questions: generatedQuestions.length,
        question_types: questionTypes,
        difficulty_levels: difficultyLevels,
        generated_at: new Date().toISOString(),
      },
      questions: generatedQuestions,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `legal-questions-${
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

            {/* 評測功能切換 */}
            <div className="mb-4">
              <div className="d-flex justify-content-between align-items-center">
                <h6 className="mb-0">分割策略評測</h6>
                <div className="form-check form-switch">
                  <input
                    className="form-check-input"
                    type="checkbox"
                    id="evaluationSwitch"
                    checked={showEvaluation}
                    onChange={(e) => setShowEvaluation(e.target.checked)}
                  />
                  <label
                    className="form-check-label"
                    htmlFor="evaluationSwitch"
                  >
                    啟用評測模式
                  </label>
                </div>
              </div>
              <small className="text-muted">
                啟用後可以測試不同chunk size和overlap比例的組合效果
              </small>
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
                        <span className="text-muted">
                          {paramInfo.type === "boolean"
                            ? ""
                            : `(${paramInfo.unit})`}
                        </span>
                      </label>
                      {paramInfo.type === "boolean" ? (
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="checkbox"
                            checked={
                              params[selectedStrategy][
                                paramKey as keyof ChunkParams[ChunkStrategy]
                              ] as boolean
                            }
                            onChange={(e) =>
                              handleParamChange(
                                selectedStrategy,
                                paramKey,
                                e.target.checked
                              )
                            }
                          />
                          <label className="form-check-label small">
                            {paramInfo.description || "啟用此選項"}
                          </label>
                        </div>
                      ) : paramInfo.type === "select" ? (
                        <select
                          className="form-select form-select-sm"
                          value={
                            params[selectedStrategy][
                              paramKey as keyof ChunkParams[ChunkStrategy]
                            ] as string
                          }
                          onChange={(e) =>
                            handleParamChange(
                              selectedStrategy,
                              paramKey,
                              e.target.value
                            )
                          }
                        >
                          {paramInfo.options?.map((option: any) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      ) : paramInfo.type === "text" ? (
                        <input
                          className="form-control form-control-sm"
                          type="text"
                          value={
                            Array.isArray(
                              params[selectedStrategy][
                                paramKey as keyof ChunkParams[ChunkStrategy]
                              ]
                            )
                              ? (
                                  params[selectedStrategy][
                                    paramKey as keyof ChunkParams[ChunkStrategy]
                                  ] as string[]
                                ).join(",")
                              : (params[selectedStrategy][
                                  paramKey as keyof ChunkParams[ChunkStrategy]
                                ] as string)
                          }
                          onChange={(e) =>
                            handleParamChange(
                              selectedStrategy,
                              paramKey,
                              e.target.value.split(",").map((s) => s.trim())
                            )
                          }
                        />
                      ) : (
                        <>
                          <input
                            className="form-control form-control-sm"
                            type="number"
                            min={paramInfo.min}
                            max={paramInfo.max}
                            value={
                              params[selectedStrategy][
                                paramKey as keyof ChunkParams[ChunkStrategy]
                              ] as number
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
                        </>
                      )}
                    </div>
                  )
                )}
              </div>
            </div>

            {/* 評測配置 */}
            {showEvaluation && (
              <div className="mb-4">
                <h6 className="text-success">評測配置</h6>

                {/* 問題生成 */}
                <div className="mb-3">
                  <label className="form-label fw-bold">生成繁體中文問題</label>
                  <div className="row g-2 mb-2">
                    <div className="col-6">
                      <input
                        type="number"
                        className="form-control form-control-sm"
                        placeholder="問題數量"
                        value={numQuestions}
                        onChange={(e) =>
                          setNumQuestions(parseInt(e.target.value) || 10)
                        }
                        min="1"
                        max="50"
                      />
                    </div>
                    <div className="col-6">
                      <button
                        className="btn btn-success btn-sm w-100"
                        onClick={generateQuestions}
                        disabled={!docId || questionLoading}
                      >
                        {questionLoading ? (
                          <span className="spinner-border spinner-border-sm me-1" />
                        ) : (
                          <i className="bi bi-question-circle me-1"></i>
                        )}
                        生成問題
                      </button>
                    </div>
                  </div>
                </div>

                {/* 問題生成錯誤顯示 */}
                {questionError && (
                  <div className="alert alert-danger" role="alert">
                    {questionError}
                  </div>
                )}

                {/* 評測參數 */}
                <div className="mb-3">
                  <label className="form-label fw-bold">評測參數</label>
                  <div className="row g-2">
                    <div className="col-6">
                      <input
                        type="text"
                        className="form-control form-control-sm"
                        placeholder="Chunk Sizes (逗號分隔)"
                        value={evaluationConfig.chunk_sizes.join(",")}
                        onChange={(e) => {
                          const sizes = e.target.value
                            .split(",")
                            .map((s) => parseInt(s.trim()))
                            .filter((n) => !isNaN(n));
                          setEvaluationConfig((prev) => ({
                            ...prev,
                            chunk_sizes: sizes,
                          }));
                        }}
                      />
                    </div>
                    <div className="col-6">
                      <input
                        type="text"
                        className="form-control form-control-sm"
                        placeholder="重疊比例 (逗號分隔: 0.0,0.1,0.2)"
                        value={evaluationConfig.overlap_ratios.join(",")}
                        onChange={(e) => {
                          const ratios = e.target.value
                            .split(",")
                            .map((s) => parseFloat(s.trim()))
                            .filter((n) => !isNaN(n));
                          setEvaluationConfig((prev) => ({
                            ...prev,
                            overlap_ratios: ratios,
                          }));
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* 策略特定參數 */}
                {selectedStrategy === "structured_hierarchical" && (
                  <div className="mb-3">
                    <label className="form-label fw-bold">分割單位選項</label>
                    <input
                      type="text"
                      className="form-control form-control-sm"
                      placeholder="分割單位選項 (逗號分隔: article,item,section,chapter)"
                      value={evaluationConfig.chunk_by_options.join(",")}
                      onChange={(e) => {
                        const options = e.target.value
                          .split(",")
                          .map((s) => s.trim())
                          .filter((s) => s);
                        setEvaluationConfig((prev) => ({
                          ...prev,
                          chunk_by_options: options,
                        }));
                      }}
                    />
                    <div className="form-text small">
                      可選值: article(按條文分割), item(按項分割),
                      section(按節分割), chapter(按章分割)
                    </div>
                  </div>
                )}

                {selectedStrategy === "rcts_hierarchical" && (
                  <div className="mb-3">
                    <label className="form-label fw-bold">
                      保持層次結構選項
                    </label>
                    <input
                      type="text"
                      className="form-control form-control-sm"
                      placeholder="保持層次結構選項 (逗號分隔: true,false)"
                      value={evaluationConfig.preserve_structure_options.join(
                        ","
                      )}
                      onChange={(e) => {
                        const options = e.target.value
                          .split(",")
                          .map((s) => s.trim().toLowerCase() === "true")
                          .filter((s) => typeof s === "boolean");
                        setEvaluationConfig((prev) => ({
                          ...prev,
                          preserve_structure_options: options,
                        }));
                      }}
                    />
                    <div className="form-text small">
                      可選值: true(啟用), false(停用)
                    </div>
                  </div>
                )}

                {selectedStrategy === "hierarchical" && (
                  <div className="mb-3">
                    <div className="row g-2">
                      <div className="col-6">
                        <label className="form-label small">層次深度選項</label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="層次深度 (逗號分隔: 2,3,4)"
                          value={evaluationConfig.level_depth_options.join(",")}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseInt(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              level_depth_options: options,
                            }));
                          }}
                        />
                      </div>
                      <div className="col-6">
                        <label className="form-label small">
                          最小分塊大小選項
                        </label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="最小分塊大小 (逗號分隔: 100,200,300)"
                          value={evaluationConfig.min_chunk_size_options.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseInt(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              min_chunk_size_options: options,
                            }));
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {selectedStrategy === "semantic" && (
                  <div className="mb-3">
                    <div className="row g-2">
                      <div className="col-6">
                        <label className="form-label small">
                          相似度閾值選項
                        </label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="相似度閾值 (逗號分隔: 0.5,0.6,0.7)"
                          value={evaluationConfig.similarity_threshold_options.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseFloat(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              similarity_threshold_options: options,
                            }));
                          }}
                        />
                      </div>
                      <div className="col-6">
                        <label className="form-label small">
                          上下文窗口選項
                        </label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="上下文窗口 (逗號分隔: 50,100,150)"
                          value={evaluationConfig.context_window_options.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseInt(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              context_window_options: options,
                            }));
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {selectedStrategy === "llm_semantic" && (
                  <div className="mb-3">
                    <div className="row g-2">
                      <div className="col-6">
                        <label className="form-label small">語義閾值選項</label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="語義閾值 (逗號分隔: 0.6,0.7,0.8)"
                          value={evaluationConfig.semantic_threshold_options.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseFloat(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              semantic_threshold_options: options,
                            }));
                          }}
                        />
                      </div>
                      <div className="col-6">
                        <label className="form-label small">
                          上下文窗口選項
                        </label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="上下文窗口 (逗號分隔: 50,100,150)"
                          value={evaluationConfig.context_window_options.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseInt(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              context_window_options: options,
                            }));
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {selectedStrategy === "sliding_window" && (
                  <div className="mb-3">
                    <div className="row g-2">
                      <div className="col-6">
                        <label className="form-label small">視窗大小選項</label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="視窗大小 (逗號分隔: 400,500,600,800)"
                          value={evaluationConfig.window_size_options.join(",")}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseInt(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              window_size_options: options,
                            }));
                          }}
                        />
                      </div>
                      <div className="col-6">
                        <label className="form-label small">步長選項</label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="步長 (逗號分隔: 200,250,300)"
                          value={evaluationConfig.step_size_options.join(",")}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseInt(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              step_size_options: options,
                            }));
                          }}
                        />
                      </div>
                    </div>
                    <div className="row g-2 mt-2">
                      <div className="col-6">
                        <label className="form-label small">
                          最小分塊大小選項
                        </label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="最小分塊大小 (逗號分隔: 50,100,150)"
                          value={evaluationConfig.min_chunk_size_options_sw.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseInt(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              min_chunk_size_options_sw: options,
                            }));
                          }}
                        />
                      </div>
                      <div className="col-6">
                        <label className="form-label small">
                          最大分塊大小選項
                        </label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="最大分塊大小 (逗號分隔: 800,1000,1200)"
                          value={evaluationConfig.max_chunk_size_options_sw.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseInt(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              max_chunk_size_options_sw: options,
                            }));
                          }}
                        />
                      </div>
                    </div>
                    <div className="row g-2 mt-2">
                      <div className="col-6">
                        <label className="form-label small">邊界感知選項</label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="邊界感知 (逗號分隔: true,false)"
                          value={evaluationConfig.boundary_aware_options.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => s.trim().toLowerCase() === "true")
                              .filter((s) => typeof s === "boolean");
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              boundary_aware_options: options,
                            }));
                          }}
                        />
                      </div>
                      <div className="col-6">
                        <label className="form-label small">
                          保持句子完整性選項
                        </label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="保持句子完整性 (逗號分隔: true,false)"
                          value={evaluationConfig.preserve_sentences_options.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => s.trim().toLowerCase() === "true")
                              .filter((s) => typeof s === "boolean");
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              preserve_sentences_options: options,
                            }));
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {selectedStrategy === "hybrid" && (
                  <div className="mb-3">
                    <div className="row g-2">
                      <div className="col-6">
                        <label className="form-label small">切換閾值選項</label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="切換閾值 (逗號分隔: 0.3,0.5,0.7)"
                          value={evaluationConfig.switch_threshold_options.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseFloat(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              switch_threshold_options: options,
                            }));
                          }}
                        />
                      </div>
                      <div className="col-6">
                        <label className="form-label small">次要大小選項</label>
                        <input
                          type="text"
                          className="form-control form-control-sm"
                          placeholder="次要大小 (逗號分隔: 300,400,500)"
                          value={evaluationConfig.secondary_size_options.join(
                            ","
                          )}
                          onChange={(e) => {
                            const options = e.target.value
                              .split(",")
                              .map((s) => parseInt(s.trim()))
                              .filter((n) => !isNaN(n));
                            setEvaluationConfig((prev) => ({
                              ...prev,
                              secondary_size_options: options,
                            }));
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 統一執行按鈕 */}
            <div className="d-grid">
              <button
                className={`btn ${
                  showEvaluation ? "btn-warning" : "btn-primary"
                }`}
                disabled={
                  !canChunk ||
                  busy ||
                  evaluationLoading ||
                  currentTask?.status === "running"
                }
                onClick={showEvaluation ? startEvaluation : handleRunChunker}
              >
                {busy || evaluationLoading ? (
                  <>
                    <span
                      className="spinner-border spinner-border-sm me-2"
                      role="status"
                    />
                    {showEvaluation ? "啟動評測中..." : "執行分塊中..."}
                  </>
                ) : (
                  <>
                    {showEvaluation ? (
                      <>
                        <i className="bi bi-graph-up me-1"></i>
                        開始評測
                      </>
                    ) : (
                      "Run Chunker"
                    )}
                  </>
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

            {chunkResults &&
              chunkResults.chunks &&
              chunkResults.chunks.length > 0 && (
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
                              {chunkResults.metrics.avg_length} 字符
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
                              {(
                                chunkResults.metrics.overlap_rate * 100
                              ).toFixed(1)}
                              %
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="col-6">
                        <div className="card bg-light">
                          <div className="card-body p-2">
                            <div className="small text-muted">最小長度</div>
                            <div className="fw-bold">
                              {chunkResults.metrics.min_length} 字符
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="col-6">
                        <div className="card bg-light">
                          <div className="card-body p-2">
                            <div className="small text-muted">最大長度</div>
                            <div className="fw-bold">
                              {chunkResults.metrics.max_length} 字符
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* 分塊內容預覽 */}
                  <div>
                    <div className="d-flex justify-content-between align-items-center mb-2">
                      <h6>分塊內容預覽</h6>
                      <div>
                        <small className="text-muted me-2">
                          顯示前 {chunkResults.chunks.length} 個分塊，共{" "}
                          {chunkResults.full_chunks} 個
                        </small>
                        {chunkResults.all_chunks &&
                          chunkResults.all_chunks.length > 3 && (
                            <button
                              className="btn btn-outline-primary btn-sm"
                              onClick={() => {
                                // 切換顯示所有分塊或只顯示前3個
                                if (chunkResults.chunks.length === 3) {
                                  setChunkResults((prev: any) => ({
                                    ...prev,
                                    chunks: prev.all_chunks || [],
                                  }));
                                } else {
                                  setChunkResults((prev: any) => ({
                                    ...prev,
                                    chunks: prev.all_chunks?.slice(0, 3) || [],
                                  }));
                                }
                              }}
                            >
                              {chunkResults.chunks.length === 3
                                ? "顯示全部"
                                : "顯示前3個"}
                            </button>
                          )}
                      </div>
                    </div>
                    <div
                      className="border rounded p-3"
                      style={{ maxHeight: "400px", overflowY: "auto" }}
                    >
                      {chunkResults.chunks.map(
                        (chunk: string, index: number) => (
                          <div key={index} className="mb-3">
                            <div className="d-flex justify-content-between align-items-center mb-1">
                              <small className="text-muted">
                                分塊 #{index + 1}
                              </small>
                              <small className="text-muted">
                                {chunk.length} 字符
                              </small>
                            </div>
                            <div
                              className="bg-light p-2 rounded small"
                              style={{ whiteSpace: "pre-wrap" }}
                            >
                              {chunk}
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                </div>
              )}

            {/* 評測結果顯示 */}
            {showEvaluation && (
              <>
                {/* 錯誤顯示 */}
                {evaluationError && (
                  <div className="alert alert-danger" role="alert">
                    {evaluationError}
                  </div>
                )}

                {/* 任務狀態 */}
                {currentTask && (
                  <div className="card mb-3">
                    <div className="card-header">
                      <h6 className="mb-0">評測任務狀態</h6>
                    </div>
                    <div className="card-body">
                      <div className="row g-3">
                        <div className="col-md-3">
                          <div className="text-center">
                            <div
                              className={`badge bg-${
                                currentTask.status === "completed"
                                  ? "success"
                                  : currentTask.status === "running"
                                  ? "primary"
                                  : currentTask.status === "failed"
                                  ? "danger"
                                  : "secondary"
                              } fs-6`}
                            >
                              {currentTask.status === "completed"
                                ? "已完成"
                                : currentTask.status === "running"
                                ? "運行中"
                                : currentTask.status === "failed"
                                ? "失敗"
                                : currentTask.status}
                            </div>
                          </div>
                        </div>
                        <div className="col-md-3">
                          <div className="text-center">
                            <div className="small text-muted">進度</div>
                            <div className="fw-bold">
                              {(currentTask.progress * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>
                        <div className="col-md-3">
                          <div className="text-center">
                            <div className="small text-muted">已完成配置</div>
                            <div className="fw-bold">
                              {currentTask.completed_configs}/
                              {currentTask.total_configs}
                            </div>
                          </div>
                        </div>
                        <div className="col-md-3">
                          <div className="text-center">
                            <div className="small text-muted">任務ID</div>
                            <div className="fw-bold small">
                              {currentTask.task_id.slice(0, 8)}...
                            </div>
                          </div>
                        </div>
                      </div>

                      {currentTask.status === "running" && (
                        <div className="mt-3">
                          {/* 評測狀態指示器 */}
                          <div
                            className="alert alert-info d-flex align-items-center mb-3"
                            role="alert"
                          >
                            <div
                              className="spinner-border spinner-border-sm me-2"
                              role="status"
                            >
                              <span className="visually-hidden">
                                Loading...
                              </span>
                            </div>
                            <div>
                              <strong>評測進行中</strong>
                              <div className="small">
                                正在評估 {selectedStrategy} 分割策略的{" "}
                                {currentTask.total_configs} 個配置組合
                              </div>
                            </div>
                          </div>

                          <div className="progress" style={{ height: "25px" }}>
                            <div
                              className="progress-bar progress-bar-striped progress-bar-animated bg-primary"
                              role="progressbar"
                              style={{
                                width: `${currentTask.progress * 100}%`,
                              }}
                              aria-valuenow={currentTask.progress * 100}
                              aria-valuemin={0}
                              aria-valuemax={100}
                            >
                              <span className="fw-bold text-white">
                                {(currentTask.progress * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                          <div className="mt-2 d-flex justify-content-between align-items-center">
                            <small className="text-muted">
                              正在處理配置 {currentTask.completed_configs} /{" "}
                              {currentTask.total_configs}
                            </small>
                            <div className="d-flex align-items-center gap-2">
                              <small className="text-muted">
                                預計剩餘時間:{" "}
                                {calculateEstimatedTime(currentTask)} 分鐘
                              </small>
                              <button
                                className="btn btn-sm btn-outline-danger"
                                onClick={() => {
                                  clearProgressPolling();
                                  setCurrentTask(null);
                                  setEvaluationError("評測已取消");
                                }}
                                title="取消評測"
                              >
                                <i className="bi bi-x-circle"></i> 取消
                              </button>
                            </div>
                          </div>
                        </div>
                      )}

                      {currentTask.status === "completed" && (
                        <div className="mt-3">
                          <div
                            className="alert alert-success d-flex align-items-center"
                            role="alert"
                          >
                            <i className="bi bi-check-circle-fill me-2"></i>
                            <div>
                              <strong>評測完成！</strong>
                              <div className="small">
                                成功評估了 {currentTask.total_configs}{" "}
                                個配置組合
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {currentTask.error_message && (
                        <div className="mt-3">
                          <div className="alert alert-danger" role="alert">
                            <strong>錯誤:</strong> {currentTask.error_message}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* 最佳配置 */}
                {evaluationResults.length > 0 && getBestConfig() && (
                  <div className="card mb-3">
                    <div className="card-header bg-success text-white">
                      <h6 className="mb-0">
                        <i className="bi bi-trophy me-2"></i>
                        最佳配置 (綜合評比)
                      </h6>
                    </div>
                    <div className="card-body">
                      {(() => {
                        const bestConfig = getBestConfig();
                        if (!bestConfig) return null;

                        const score =
                          (bestConfig.metrics.precision_at_k[3] || 0) * 0.3 +
                          (bestConfig.metrics.recall_at_k[3] || 0) * 0.3 +
                          (bestConfig.metrics.precision_omega || 0) * 0.4;

                        return (
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
                                      <td>{bestConfig.config.chunk_size}</td>
                                    </tr>
                                    <tr>
                                      <td>
                                        <strong>重疊比例:</strong>
                                      </td>
                                      <td>
                                        {(
                                          bestConfig.config.overlap_ratio * 100
                                        ).toFixed(0)}
                                        %
                                      </td>
                                    </tr>
                                    {bestConfig.config.strategy ===
                                      "sliding_window" && (
                                      <>
                                        <tr>
                                          <td>
                                            <strong>視窗大小:</strong>
                                          </td>
                                          <td>
                                            {bestConfig.config.window_size ||
                                              bestConfig.config.chunk_size}
                                          </td>
                                        </tr>
                                        <tr>
                                          <td>
                                            <strong>步長:</strong>
                                          </td>
                                          <td>
                                            {bestConfig.config.step_size ||
                                              "N/A"}
                                          </td>
                                        </tr>
                                        <tr>
                                          <td>
                                            <strong>邊界感知:</strong>
                                          </td>
                                          <td>
                                            {bestConfig.config.boundary_aware
                                              ? "是"
                                              : "否"}
                                          </td>
                                        </tr>
                                        <tr>
                                          <td>
                                            <strong>保持句子完整:</strong>
                                          </td>
                                          <td>
                                            {bestConfig.config
                                              .preserve_sentences
                                              ? "是"
                                              : "否"}
                                          </td>
                                        </tr>
                                        {bestConfig.config
                                          .min_chunk_size_sw && (
                                          <tr>
                                            <td>
                                              <strong>最小分塊大小:</strong>
                                            </td>
                                            <td>
                                              {
                                                bestConfig.config
                                                  .min_chunk_size_sw
                                              }
                                            </td>
                                          </tr>
                                        )}
                                        {bestConfig.config
                                          .max_chunk_size_sw && (
                                          <tr>
                                            <td>
                                              <strong>最大分塊大小:</strong>
                                            </td>
                                            <td>
                                              {
                                                bestConfig.config
                                                  .max_chunk_size_sw
                                              }
                                            </td>
                                          </tr>
                                        )}
                                      </>
                                    )}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                            <div className="col-md-6">
                              <h6>評估指標</h6>
                              <div className="table-responsive">
                                <table className="table table-sm">
                                  <tbody>
                                    <tr>
                                      <td>
                                        <strong>Precision@3:</strong>
                                      </td>
                                      <td>
                                        <span className="badge bg-success">
                                          {(
                                            bestConfig.metrics
                                              .precision_at_k[3] || 0
                                          ).toFixed(3)}
                                        </span>
                                      </td>
                                    </tr>
                                    <tr>
                                      <td>
                                        <strong>Recall@3:</strong>
                                      </td>
                                      <td>
                                        <span className="badge bg-success">
                                          {(
                                            bestConfig.metrics.recall_at_k[3] ||
                                            0
                                          ).toFixed(3)}
                                        </span>
                                      </td>
                                    </tr>
                                    <tr>
                                      <td>
                                        <strong>PrecisionΩ:</strong>
                                      </td>
                                      <td>
                                        <span className="badge bg-success">
                                          {(
                                            bestConfig.metrics
                                              .precision_omega || 0
                                          ).toFixed(3)}
                                        </span>
                                      </td>
                                    </tr>
                                    <tr>
                                      <td>
                                        <strong>綜合評分:</strong>
                                      </td>
                                      <td>
                                        <span className="badge bg-primary fs-6">
                                          {score.toFixed(3)}
                                        </span>
                                      </td>
                                    </tr>
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                )}

                {/* 評測結果 */}
                {evaluationResults.length > 0 && (
                  <div className="card mb-3">
                    <div className="card-header d-flex justify-content-between align-items-center">
                      <h6 className="mb-0">
                        評測結果
                        <span className="badge bg-secondary ms-2">
                          {evaluationResults.length} 個配置
                        </span>
                      </h6>
                      <div className="d-flex gap-2">
                        <button
                          className="btn btn-outline-secondary btn-sm"
                          onClick={() => setShowAllResults(!showAllResults)}
                        >
                          <i
                            className={`bi bi-chevron-${
                              showAllResults ? "up" : "down"
                            } me-1`}
                          ></i>
                          {showAllResults ? "收起" : "展開全部"}
                        </button>
                        <button
                          className="btn btn-outline-primary btn-sm"
                          onClick={exportEvaluationReport}
                        >
                          <i className="bi bi-download me-1"></i>
                          導出報告
                        </button>
                      </div>
                    </div>
                    <div className="card-body">
                      <div className="table-responsive">
                        <table className="table table-sm table-hover">
                          <thead>
                            <tr>
                              <th>配置</th>
                              <th>Precision@1</th>
                              <th>Precision@3</th>
                              <th>Precision@5</th>
                              <th>Recall@1</th>
                              <th>Recall@3</th>
                              <th>Recall@5</th>
                              <th>PrecisionΩ</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(showAllResults
                              ? evaluationResults
                              : evaluationResults.slice(0, 10)
                            ).map((result, index) => (
                              <tr key={index}>
                                <td>
                                  <div className="small">
                                    <div>Size: {result.config.chunk_size}</div>
                                    <div>
                                      Overlap:{" "}
                                      {(
                                        result.config.overlap_ratio * 100
                                      ).toFixed(0)}
                                      %
                                    </div>
                                  </div>
                                </td>
                                <td>
                                  <span
                                    className={`badge ${
                                      result.metrics.precision_at_k[1] > 0.7
                                        ? "bg-success"
                                        : result.metrics.precision_at_k[1] > 0.5
                                        ? "bg-warning"
                                        : "bg-danger"
                                    }`}
                                  >
                                    {result.metrics.precision_at_k[1]?.toFixed(
                                      3
                                    ) || "0.000"}
                                  </span>
                                </td>
                                <td>
                                  <span
                                    className={`badge ${
                                      result.metrics.precision_at_k[3] > 0.7
                                        ? "bg-success"
                                        : result.metrics.precision_at_k[3] > 0.5
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
                                      result.metrics.precision_at_k[5] > 0.7
                                        ? "bg-success"
                                        : result.metrics.precision_at_k[5] > 0.5
                                        ? "bg-warning"
                                        : "bg-danger"
                                    }`}
                                  >
                                    {result.metrics.precision_at_k[5]?.toFixed(
                                      3
                                    ) || "0.000"}
                                  </span>
                                </td>
                                <td>
                                  <span
                                    className={`badge ${
                                      result.metrics.recall_at_k[1] > 0.8
                                        ? "bg-success"
                                        : result.metrics.recall_at_k[1] > 0.6
                                        ? "bg-warning"
                                        : "bg-danger"
                                    }`}
                                  >
                                    {result.metrics.recall_at_k[1]?.toFixed(
                                      3
                                    ) || "0.000"}
                                  </span>
                                </td>
                                <td>
                                  <span
                                    className={`badge ${
                                      result.metrics.recall_at_k[3] > 0.8
                                        ? "bg-success"
                                        : result.metrics.recall_at_k[3] > 0.6
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
                                      result.metrics.recall_at_k[5] > 0.8
                                        ? "bg-success"
                                        : result.metrics.recall_at_k[5] > 0.6
                                        ? "bg-warning"
                                        : "bg-danger"
                                    }`}
                                  >
                                    {result.metrics.recall_at_k[5]?.toFixed(
                                      3
                                    ) || "0.000"}
                                  </span>
                                </td>
                                <td>
                                  <span
                                    className={`badge ${
                                      result.metrics.precision_omega > 0.7
                                        ? "bg-success"
                                        : result.metrics.precision_omega > 0.5
                                        ? "bg-warning"
                                        : "bg-danger"
                                    }`}
                                  >
                                    {result.metrics.precision_omega?.toFixed(
                                      3
                                    ) || "0.000"}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {!showAllResults && evaluationResults.length > 10 && (
                        <div className="mt-3 text-center">
                          <div className="alert alert-info mb-0">
                            <i className="bi bi-info-circle me-2"></i>
                            顯示前 10 筆結果，共 {evaluationResults.length}{" "}
                            筆配置
                            <br />
                            <small>點擊「展開全部」按鈕查看所有結果</small>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* 對比分析 */}
                {evaluationComparison && (
                  <div className="card mb-3">
                    <div className="card-header">
                      <h6 className="mb-0">對比分析與推薦</h6>
                    </div>
                    <div className="card-body">
                      <div className="row g-4">
                        {/* Chunk Size 分析 */}
                        <div className="col-md-6">
                          <h6>分塊大小分析</h6>
                          <div className="table-responsive">
                            <table className="table table-sm">
                              <thead>
                                <tr>
                                  <th>Size</th>
                                  <th>Precision@3</th>
                                  <th>Recall@3</th>
                                  <th>PrecisionΩ</th>
                                </tr>
                              </thead>
                              <tbody>
                                {Object.entries(
                                  evaluationComparison.chunk_size_analysis
                                ).map(([size, metrics]: [string, any]) => (
                                  <tr key={size}>
                                    <td>{size}</td>
                                    <td>
                                      {metrics.precision_at_k?.[3]?.toFixed(
                                        3
                                      ) || "0.000"}
                                    </td>
                                    <td>
                                      {metrics.recall_at_k?.[3]?.toFixed(3) ||
                                        "0.000"}
                                    </td>
                                    <td>
                                      {metrics.precision_omega?.toFixed(3) ||
                                        "0.000"}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>

                        {/* Overlap 分析 */}
                        <div className="col-md-6">
                          <h6>重疊比例分析</h6>
                          <div className="table-responsive">
                            <table className="table table-sm">
                              <thead>
                                <tr>
                                  <th>Ratio</th>
                                  <th>Precision@3</th>
                                  <th>Recall@3</th>
                                  <th>PrecisionΩ</th>
                                </tr>
                              </thead>
                              <tbody>
                                {Object.entries(
                                  evaluationComparison.overlap_analysis
                                ).map(([ratio, metrics]: [string, any]) => (
                                  <tr key={ratio}>
                                    <td>
                                      {(parseFloat(ratio) * 100).toFixed(0)}%
                                    </td>
                                    <td>
                                      {metrics.precision_at_k?.[3]?.toFixed(
                                        3
                                      ) || "0.000"}
                                    </td>
                                    <td>
                                      {metrics.recall_at_k?.[3]?.toFixed(3) ||
                                        "0.000"}
                                    </td>
                                    <td>
                                      {metrics.precision_omega?.toFixed(3) ||
                                        "0.000"}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>

                      {/* 策略特定參數分析 */}
                      {evaluationComparison.strategy_specific_analysis &&
                        Object.keys(
                          evaluationComparison.strategy_specific_analysis
                        ).length > 0 && (
                          <div className="row g-4 mt-3">
                            <div className="col-12">
                              <h6>策略特定參數分析</h6>
                              <div className="table-responsive">
                                <table className="table table-sm">
                                  <thead>
                                    <tr>
                                      <th>參數類型</th>
                                      <th>參數值</th>
                                      <th>Precision@3</th>
                                      <th>Recall@3</th>
                                      <th>PrecisionΩ</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {Object.entries(
                                      evaluationComparison.strategy_specific_analysis
                                    ).map(
                                      ([paramKey, metrics]: [string, any]) => {
                                        // 解析參數鍵和值
                                        let paramType = "";
                                        let paramValue = "";

                                        if (paramKey.startsWith("chunk_by_")) {
                                          paramType = "分割單位";
                                          const chunkBy = paramKey.replace(
                                            "chunk_by_",
                                            ""
                                          );
                                          paramValue =
                                            chunkBy === "article"
                                              ? "按條文分割"
                                              : chunkBy === "item"
                                              ? "按項分割"
                                              : chunkBy === "section"
                                              ? "按節分割"
                                              : chunkBy === "chapter"
                                              ? "按章分割"
                                              : chunkBy;
                                        } else if (
                                          paramKey === "preserve_structure"
                                        ) {
                                          paramType = "保持結構";
                                          paramValue = "是";
                                        } else if (
                                          paramKey === "no_preserve_structure"
                                        ) {
                                          paramType = "保持結構";
                                          paramValue = "否";
                                        } else if (
                                          paramKey.startsWith("level_depth_")
                                        ) {
                                          paramType = "層次深度";
                                          paramValue = paramKey.replace(
                                            "level_depth_",
                                            ""
                                          );
                                        } else if (
                                          paramKey.startsWith(
                                            "similarity_threshold_"
                                          )
                                        ) {
                                          paramType = "相似度閾值";
                                          paramValue = paramKey.replace(
                                            "similarity_threshold_",
                                            ""
                                          );
                                        } else if (
                                          paramKey.startsWith(
                                            "semantic_threshold_"
                                          )
                                        ) {
                                          paramType = "語義閾值";
                                          paramValue = paramKey.replace(
                                            "semantic_threshold_",
                                            ""
                                          );
                                        } else if (
                                          paramKey.startsWith("step_size_")
                                        ) {
                                          paramType = "步長";
                                          paramValue = paramKey.replace(
                                            "step_size_",
                                            ""
                                          );
                                        } else if (
                                          paramKey.startsWith(
                                            "switch_threshold_"
                                          )
                                        ) {
                                          paramType = "切換閾值";
                                          paramValue = paramKey.replace(
                                            "switch_threshold_",
                                            ""
                                          );
                                        } else if (
                                          paramKey.startsWith("secondary_size_")
                                        ) {
                                          paramType = "次要大小";
                                          paramValue = paramKey.replace(
                                            "secondary_size_",
                                            ""
                                          );
                                        } else {
                                          paramType = paramKey;
                                          paramValue = "";
                                        }

                                        return (
                                          <tr key={paramKey}>
                                            <td>{paramType}</td>
                                            <td>{paramValue}</td>
                                            <td>
                                              {metrics.precision_at_k?.[3]?.toFixed(
                                                3
                                              ) || "0.000"}
                                            </td>
                                            <td>
                                              {metrics.recall_at_k?.[3]?.toFixed(
                                                3
                                              ) || "0.000"}
                                            </td>
                                            <td>
                                              {metrics.precision_omega?.toFixed(
                                                3
                                              ) || "0.000"}
                                            </td>
                                          </tr>
                                        );
                                      }
                                    )}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          </div>
                        )}

                      {/* 推薦 */}
                      <div className="mt-4">
                        <h6>推薦配置</h6>
                        <div className="alert alert-info">
                          {evaluationComparison.recommendations.map(
                            (rec: string, index: number) => (
                              <div key={index}>{rec}</div>
                            )
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 生成問題結果 */}
                {generatedQuestions.length > 0 && (
                  <div className="card">
                    <div className="card-header d-flex justify-content-between align-items-center">
                      <h6 className="mb-0">生成的繁體中文問題</h6>
                      <button
                        className="btn btn-outline-success btn-sm"
                        onClick={exportQuestions}
                      >
                        <i className="bi bi-download me-1"></i>
                        導出問題
                      </button>
                    </div>
                    <div className="card-body">
                      <div className="row g-3">
                        {generatedQuestions.map((question, index) => (
                          <div key={index} className="col-12">
                            <div className="card border-primary">
                              <div className="card-body">
                                <div className="d-flex justify-content-between align-items-start mb-2">
                                  <h6 className="card-title mb-0">
                                    問題 {index + 1}
                                  </h6>
                                  <div>
                                    <span
                                      className={`badge bg-${
                                        question.difficulty === "基礎"
                                          ? "success"
                                          : question.difficulty === "進階"
                                          ? "warning"
                                          : "danger"
                                      } me-1`}
                                    >
                                      {question.difficulty}
                                    </span>
                                    <span className="badge bg-info">
                                      {question.question_type}
                                    </span>
                                  </div>
                                </div>
                                <p className="card-text">{question.question}</p>

                                <div className="mt-3">
                                  <h6 className="small text-muted mb-2">
                                    相關法規：
                                  </h6>
                                  <div className="d-flex flex-wrap gap-1">
                                    {question.references.map(
                                      (ref: string, refIndex: number) => (
                                        <span
                                          key={refIndex}
                                          className="badge bg-secondary"
                                        >
                                          {ref}
                                        </span>
                                      )
                                    )}
                                  </div>
                                </div>

                                {question.keywords.length > 0 && (
                                  <div className="mt-2">
                                    <h6 className="small text-muted mb-2">
                                      關鍵詞：
                                    </h6>
                                    <div className="d-flex flex-wrap gap-1">
                                      {question.keywords.map(
                                        (keyword: string, kwIndex: number) => (
                                          <span
                                            key={kwIndex}
                                            className="badge bg-light text-dark"
                                          >
                                            {keyword}
                                          </span>
                                        )
                                      )}
                                    </div>
                                  </div>
                                )}

                                <div className="mt-2">
                                  <small className="text-muted">
                                    估算Token數: {question.estimated_tokens}
                                  </small>
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
