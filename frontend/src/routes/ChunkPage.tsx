import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";
import { api } from "../lib/api";
import MetadataViewer from "../components/MetadataViewer";
// 無映射模式：移除與 QA 映射相關的元件

// 擴展Window接口以包含Bootstrap
declare global {
  interface Window {
    bootstrap: any;
  }
}

// 法律層級統計顯示組件
const HierarchyStatsDisplay: React.FC = () => {
  const [hierarchyStats, setHierarchyStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHierarchyStats = async () => {
      try {
        setLoading(true);
        const response = await api.get("/chunking-hierarchy-stats");
        setHierarchyStats(response);
        setError(null);
      } catch (err) {
        console.error("獲取層級統計失敗:", err);
        setError("獲取層級統計失敗");
      } finally {
        setLoading(false);
      }
    };

    fetchHierarchyStats();
  }, []);

  if (loading) {
    return (
      <div className="row g-3 mb-4">
        <div className="col-12">
          <div className="text-center">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">載入中...</span>
            </div>
            <p className="mt-2 text-muted">載入分塊層級統計中...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !hierarchyStats) {
    return (
      <div className="row g-3 mb-4">
        <div className="col-12">
          <div className="alert alert-warning">
            <i className="bi bi-exclamation-triangle me-2"></i>
            無法載入分塊層級統計
          </div>
        </div>
      </div>
    );
  }

  const { hierarchy_stats, level_names, total_chunks } = hierarchyStats;

  // 層級順序：章、節、條、項、款、目
  const levelOrder = [
    "document", // 章級
    "document_component", // 節級
    "basic_unit_hierarchy", // 條級
    "basic_unit", // 項級
    "basic_unit_component", // 款級
    "enumeration", // 目級
  ];

  return (
    <div className="mb-4">
      <h6 className="mb-3">
        <i className="bi bi-diagram-3 me-2"></i>
        分塊結果統計 - 按法律層級分類
      </h6>

      {/* 總計統計 */}
      <div className="row g-3 mb-3">
        <div className="col-12">
          <div className="card bg-primary text-white">
            <div className="card-body text-center">
              <h5 className="card-title mb-1">
                <i className="bi bi-collection me-2"></i>
                {total_chunks}
              </h5>
              <p className="card-text small mb-0">總分塊數量</p>
            </div>
          </div>
        </div>
      </div>

      {/* 各層級統計 */}
      <div className="row g-3">
        {levelOrder.map((levelKey, index) => {
          const count = hierarchy_stats[levelKey] || 0;
          const levelName = level_names[levelKey] || levelKey;

          // 根據層級設置不同的顏色
          const colorClass =
            [
              "bg-danger", // 章級 - 紅色
              "bg-warning", // 節級 - 橙色
              "bg-success", // 條級 - 綠色
              "bg-info", // 項級 - 藍色
              "bg-primary", // 款級 - 紫色
              "bg-secondary", // 目級 - 灰色
            ][index] || "bg-light";

          return (
            <div key={levelKey} className="col-md-4 col-lg-2">
              <div className={`card ${colorClass} text-white`}>
                <div className="card-body text-center p-3">
                  <h6 className="card-title mb-1">{count}</h6>
                  <p className="card-text small mb-0">{levelName}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* 層級說明 */}
      <div className="mt-3">
        <small className="text-muted">
          <i className="bi bi-info-circle me-1"></i>
          法律文檔按照「章、節、條、項、款、目」六個層級進行分塊，每個層級的分塊數量如上所示
        </small>
      </div>
    </div>
  );
};

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

interface ChunkHierarchyChunk {
  chunk_id: string;
  level: string;
  content: string;
  content_length: number;
  metadata: Record<string, any>;
  strategy?: string;
  path_label: string;
}

interface ChunkHierarchyNode {
  level: string;
  key: string;
  label: string;
  path_label: string;
  chunk?: ChunkHierarchyChunk | null;
  children: ChunkHierarchyNode[];
  path?: Record<string, any>;
  doc_id?: string;
  doc_name?: string;
}

interface ChunkHierarchyDocument {
  doc_id: string;
  doc_name: string;
  law_nodes: ChunkHierarchyNode[];
}

interface ChunkHierarchyResponse {
  documents: ChunkHierarchyDocument[];
  timestamp: string;
}

interface ChunkSearchResult {
  doc_id: string;
  doc_name: string;
  level: string;
  chunk_id: string;
  chunk_index: number;
  path: Record<string, any>;
  path_label: string;
  content: string;
  content_length: number;
  metadata: Record<string, any>;
}

interface ChunkSearchResponse {
  query: string;
  normalized_filters: Record<string, any>;
  result_count: number;
  results: ChunkSearchResult[];
}

export function ChunkPage() {
  const nav = useNavigate();
  const {
    canChunk,
    chunk,
    docId,
    chunkMeta,
    setChunkingResultsAndStrategy,
    jsonData,
  } = useRag();

  // 無映射模式：不需要 QA Set 上傳狀態（移除）

  // 步驟2: 分塊策略配置狀態
  const [selectedStrategy, setSelectedStrategy] = useState<ChunkStrategy>(
    "structured_hierarchical"
  );
  const [isChunking, setIsChunking] = useState(false);
  const [chunkingError, setChunkingError] = useState<string | null>(null);
  const [chunkingProgress, setChunkingProgress] = useState(0);
  const [chunkingTaskId, setChunkingTaskId] = useState<string | null>(null);
  const [showMetadataViewer, setShowMetadataViewer] = useState(false);

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

  // 分塊查看相關狀態
  const [selectedHierarchyLevel, setSelectedHierarchyLevel] =
    useState<string>("all");
  const [hierarchyChunks, setHierarchyChunks] = useState<any[]>([]);
  const [loadingHierarchyChunks, setLoadingHierarchyChunks] = useState(false);

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

  // 層級視圖與法條搜尋狀態
  const [hierarchyData, setHierarchyData] = useState<ChunkHierarchyDocument[]>([]);
  const [hierarchyLoading, setHierarchyLoading] = useState(false);
  const [hierarchyError, setHierarchyError] = useState<string | null>(null);
  const [hierarchyTimestamp, setHierarchyTimestamp] = useState<string | null>(null);
  const [hierarchyView, setHierarchyView] = useState<"tree" | "list">("tree");
  const [citationQuery, setCitationQuery] = useState("");
  const [citationLoading, setCitationLoading] = useState(false);
  const [citationError, setCitationError] = useState<string | null>(null);
  const [citationResults, setCitationResults] = useState<ChunkSearchResult[]>([]);
  const [searchFilters, setSearchFilters] = useState<Record<string, any> | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [highlightedChunkId, setHighlightedChunkId] = useState<string | null>(null);

  // 取得層級中文標籤
  const getLevelLabel = (md: any) => {
    const level = md?.level_en || md?.level || "";
    const map: Record<string, string> = {
      Chapter: "章",
      Section: "節",
      Article: "條",
      Paragraph: "項",
      Subparagraph: "款",
      Item: "目",
      Law: "法規",
    };
    return map[level] || level || "";
  };

  // 組合層級與所屬法規/章節資訊
  const getLevelPathText = (md: any) => {
    if (!md) return "";
    const levelZh = getLevelLabel(md);
    const parts: string[] = [];
    if (levelZh) parts.push(`層級: ${levelZh}`);
    if (md.law_name) parts.push(`法規: ${md.law_name}`);
    if (md.chapter) parts.push(`章: ${md.chapter}`);
    if (md.section) parts.push(`節: ${md.section}`);
    if (md.article) parts.push(`條: ${md.article}`);
    if (md.paragraph) parts.push(`項: ${md.paragraph}`);
    if (md.subparagraph) parts.push(`款: ${md.subparagraph}`);
    if (md.item) parts.push(`目: ${md.item}`);
    return parts.join(" | ");
  };

  const makeNodeKey = React.useCallback((node: ChunkHierarchyNode) => {
    const docPart = node.doc_id ? `${node.doc_id}-` : "";
    const baseKey = `${docPart}${node.level}-${node.key}`;
    if (node.chunk?.chunk_id) {
      return `${baseKey}-${node.chunk.chunk_id}`;
    }
    return baseKey;
  }, []);

  const findNodePathByChunkId = React.useCallback(
    (
      nodes: ChunkHierarchyNode[],
      targetChunkId: string,
      currentPath: string[] = []
    ): string[] | null => {
      for (const node of nodes) {
        const nodeKey = makeNodeKey(node);
        const nextPath = [...currentPath, nodeKey];
        if (node.chunk?.chunk_id === targetChunkId) {
          return nextPath;
        }
        if (node.children && node.children.length > 0) {
          const childPath = findNodePathByChunkId(
            node.children,
            targetChunkId,
            nextPath
          );
          if (childPath) {
            return childPath;
          }
        }
      }
      return null;
    },
    [makeNodeKey]
  );

  const toggleNodeExpansion = React.useCallback(
    (nodeKey: string, forceExpand?: boolean) => {
      setExpandedNodes((prev) => {
        const next = new Set(prev);
        const shouldExpand =
          forceExpand !== undefined ? forceExpand : !next.has(nodeKey);
        if (shouldExpand) {
          next.add(nodeKey);
        } else {
          next.delete(nodeKey);
        }
        return next;
      });
    },
    []
  );

  const expandPathForChunk = React.useCallback(
    (chunkId: string) => {
      const paths = hierarchyData
        .map((doc) => findNodePathByChunkId(doc.law_nodes, chunkId))
        .filter((path): path is string[] => Array.isArray(path) && path.length);

      if (paths.length > 0) {
        setExpandedNodes((prev) => {
          const next = new Set(prev);
          paths[0].forEach((key) => next.add(key));
          return next;
        });
      }
    },
    [findNodePathByChunkId, hierarchyData]
  );

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

  // 獲取指定層級的chunks
  const fetchHierarchyChunks = async (levelName: string) => {
    if (levelName === "all") {
      setHierarchyChunks([]);
      return;
    }

    try {
      setLoadingHierarchyChunks(true);
      const response = await api.get(`/api/chunks-by-hierarchy/${levelName}`);
      setHierarchyChunks(response.data.chunks || []);
    } catch (err) {
      console.error("獲取層級chunks失敗:", err);
      setHierarchyChunks([]);
    } finally {
      setLoadingHierarchyChunks(false);
    }
  };

  // 當選擇的層級改變時，獲取對應的chunks
  useEffect(() => {
    fetchHierarchyChunks(selectedHierarchyLevel);
  }, [selectedHierarchyLevel]);

  const loadHierarchyData = React.useCallback(
    async (targetDocId?: string) => {
      try {
        setHierarchyLoading(true);
        setHierarchyError(null);
        const response: ChunkHierarchyResponse = await api.getChunkHierarchyTree(
          targetDocId
        );
        const documents = response?.documents || [];
        setHierarchyData(documents);
        setHierarchyTimestamp(response?.timestamp || null);

        const defaultExpanded = new Set<string>();
        documents.forEach((doc) => {
          doc.law_nodes.forEach((lawNode) => {
            const lawKey = makeNodeKey(lawNode);
            defaultExpanded.add(lawKey);
            lawNode.children.forEach((child) => {
              if (["chapter", "section"].includes(child.level)) {
                defaultExpanded.add(makeNodeKey(child));
              }
            });
          });
        });
        setExpandedNodes(defaultExpanded);
      } catch (error) {
        console.error("載入層級樹失敗:", error);
        setHierarchyError(
          error instanceof Error ? error.message : "載入層級樹失敗"
        );
        setHierarchyData([]);
      } finally {
        setHierarchyLoading(false);
      }
    },
    [makeNodeKey]
  );

  useEffect(() => {
    if (!docId || chunkingResults.length === 0) {
      setHierarchyData([]);
      setHierarchyTimestamp(null);
      setCitationResults([]);
      setSearchFilters(null);
      setHighlightedChunkId(null);
      return;
    }
    loadHierarchyData(docId);
  }, [docId, chunkingResults, loadHierarchyData]);

  const handleRefreshHierarchy = React.useCallback(() => {
    if (!docId) return;
    loadHierarchyData(docId);
  }, [docId, loadHierarchyData]);

  useEffect(() => {
    if (hierarchyView === "list" && !showAllChunks) {
      setShowAllChunks(true);
    }
  }, [hierarchyView, showAllChunks]);

  // 步驟2: 執行分塊操作
  const handleRunChunking = async () => {
    if (!canChunk) return;

    // 強制要求：必須先於上傳頁保存 JSON（upload-json 或 convert 後的 metadata 已寫入）
    if (!jsonData) {
      setChunkingError("請先於上傳頁保存法條 JSON，然後再執行分塊。");
      return;
    }

    setIsChunking(true);
    setChunkingError(null);
    setChunkingResults([]);
    setChunkingProgress(0);

    try {
      // 再次顯式同步一次 JSON 到後端，避免不同步導致仍以舊內容分塊
      if (docId && jsonData) {
        try {
          await api.updateJson(docId, jsonData);
        } catch (e) {
          console.warn("在分塊前同步 JSON 失敗：", e);
        }
      }

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

  const handleHighlightChunk = React.useCallback(
    (chunkId: string) => {
      if (!chunkId) return;
      setHierarchyView("tree");
      expandPathForChunk(chunkId);
      setHighlightedChunkId(chunkId);
    },
    [expandPathForChunk]
  );

  const handleCitationSearch = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();
    if (!citationQuery.trim()) {
      setCitationError("請輸入要搜尋的章、節、條或枚舉項");
      setCitationResults([]);
      return;
    }
    if (!docId) {
      setCitationError("請先完成上傳並取得文檔ID");
      return;
    }
    setCitationLoading(true);
    setCitationError(null);
    try {
      const response: ChunkSearchResponse = await api.searchChunkByCitation({
        query: citationQuery.trim(),
        doc_id: docId,
      });
      setCitationResults(response?.results || []);
      setSearchFilters(response?.normalized_filters || null);
      if (response?.results && response.results.length > 0) {
        handleHighlightChunk(response.results[0].chunk_id);
      } else {
        setHighlightedChunkId(null);
      }
    } catch (error) {
      console.error("法條搜尋失敗:", error);
      setCitationError(error instanceof Error ? error.message : "搜尋失敗");
      setCitationResults([]);
    } finally {
      setCitationLoading(false);
    }
  };

  const handleResetCitationSearch = () => {
    setCitationQuery("");
    setCitationResults([]);
    setCitationError(null);
    setSearchFilters(null);
    setHighlightedChunkId(null);
  };

  const renderHierarchyNode = (
    node: ChunkHierarchyNode,
    depth: number = 0
  ): React.ReactNode => {
    const nodeKey = makeNodeKey(node);
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = expandedNodes.has(nodeKey);
    const isHighlighted = node.chunk?.chunk_id === highlightedChunkId;

    return (
      <div key={nodeKey} className={`mb-2 ${depth > 0 ? "ms-3" : ""}`}>
        <div
          className={`d-flex align-items-center gap-2 ${
            isHighlighted ? "bg-warning bg-opacity-25 rounded p-2" : ""
          }`}
        >
          {hasChildren ? (
            <button
              type="button"
              className="btn btn-sm btn-outline-secondary"
              onClick={() => toggleNodeExpansion(nodeKey)}
              aria-expanded={isExpanded}
            >
              {isExpanded ? (
                <i className="bi bi-dash-lg"></i>
              ) : (
                <i className="bi bi-plus-lg"></i>
              )}
            </button>
          ) : (
            <span
              className="d-inline-flex align-items-center justify-content-center btn btn-sm btn-light disabled"
              style={{ width: "32px" }}
            >
              <i className="bi bi-dot"></i>
            </span>
          )}
          <span className="badge bg-secondary text-uppercase">
            {node.level}
          </span>
          <span className="fw-semibold">{node.label}</span>
          {node.chunk && (
            <div className="ms-auto d-flex align-items-center gap-2">
              <span className="text-muted small">
                {node.chunk.content_length} 字
              </span>
              <button
                type="button"
                className="btn btn-outline-secondary btn-sm"
                onClick={() =>
                  navigator.clipboard.writeText(node.chunk!.chunk_id)
                }
              >
                <i className="bi bi-clipboard"></i> ID
              </button>
            </div>
          )}
        </div>
        {node.chunk && (
          <div className={`mt-2 ${depth > 0 ? "ms-4" : ""}`}>
            <details open={isHighlighted}>
              <summary className="small text-muted">
                Chunk ID: {node.chunk.chunk_id}
              </summary>
              <pre
                className="small mt-2 p-3 bg-light border rounded"
                style={{ whiteSpace: "pre-wrap" }}
              >
                {node.chunk.content}
              </pre>
            </details>
          </div>
        )}
        {hasChildren && isExpanded && (
          <div className="mt-2 ms-4">
            {node.children.map((child) =>
              renderHierarchyNode(child, depth + 1)
            )}
          </div>
        )}
      </div>
    );
  };

  const renderSearchFilterSummary = () => {
    if (!searchFilters) return null;
    const parts: string[] = [];
    if (searchFilters.chapter) {
      parts.push(`第${searchFilters.chapter}章`);
    }
    if (searchFilters.section) {
      parts.push(`第${searchFilters.section}節`);
    }
    if (searchFilters.article_canonical) {
      const articleText = searchFilters.article_suffix
        ? `第${searchFilters.article}條之${searchFilters.article_suffix}`
        : `第${searchFilters.article}條`;
      parts.push(articleText);
    }
    if (searchFilters.paragraph) {
      parts.push(`第${searchFilters.paragraph}項`);
    }
    if (searchFilters.subparagraph) {
      parts.push(`第${searchFilters.subparagraph}款`);
    }
    if (searchFilters.item) {
      parts.push(`第${searchFilters.item}目`);
    }
    if (!parts.length) return null;
    return (
      <div className="alert alert-info small mt-3 mb-0">
        <div className="fw-semibold mb-1">套用條件</div>
        <div>{parts.join(" / ")}</div>
      </div>
    );
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
                      本頁專注於「批量分塊」。本頁的所有分割策略會
                      <strong>統一使用上傳頁面的 JSON 結構</strong>進行分割。
                      評測請到首頁 Evaluate(beta) 區塊上傳{" "}
                      <code>qa_gold.json</code>，啟動分塊後一鍵計算 P@K / R@K。
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
                        <div className="d-flex justify-content-between align-items-center">
                          <h5 className="mb-0">步驟 2: 多種分塊組合處理</h5>
                          {chunkingResults.length > 0 && (
                            <button
                              className="btn btn-sm btn-outline-light"
                              onClick={() => setShowMetadataViewer(true)}
                              title="查看和編輯Enhanced Metadata"
                            >
                              <i className="bi bi-tags"></i> Metadata
                            </button>
                          )}
                        </div>
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

                            {/* 分塊結果統計 - 按法律層級分類 */}
                            <HierarchyStatsDisplay />

                            {/* 分塊配置信息 */}
                            <div className="mb-4">
                              <h6>分塊配置組合</h6>
                              <div className="table-responsive">
                                <table className="table table-sm table-striped">
                                  <thead>
                                    <tr>
                                      <th>策略</th>
                                      <th>分塊大小/層級</th>
                                      <th>重疊比例/層級</th>
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
                                        <td>
                                          {result.strategy ===
                                          "structured_hierarchical"
                                            ? "多層級"
                                            : `${result.config.chunk_size} 字符`}
                                        </td>
                                        <td>
                                          {result.strategy ===
                                          "structured_hierarchical"
                                            ? "—"
                                            : `${(
                                                result.config.overlap_ratio *
                                                100
                                              ).toFixed(1)}%`}
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
                              <div className="d-flex flex-wrap align-items-center justify-content-between gap-2 mb-3">
                                <h6 className="mb-0">分塊內容查看</h6>
                                <div className="btn-group btn-group-sm" role="group">
                                  <button
                                    type="button"
                                    className={`btn ${
                                      hierarchyView === "tree"
                                        ? "btn-primary"
                                        : "btn-outline-primary"
                                    }`}
                                    onClick={() => setHierarchyView("tree")}
                                  >
                                    <i className="bi bi-diagram-3 me-1"></i>
                                    層級樹
                                  </button>
                                  <button
                                    type="button"
                                    className={`btn ${
                                      hierarchyView === "list"
                                        ? "btn-primary"
                                        : "btn-outline-primary"
                                    }`}
                                    onClick={() => setHierarchyView("list")}
                                  >
                                    <i className="bi bi-list-ul me-1"></i>
                                    列表
                                  </button>
                                </div>
                              </div>

                              {hierarchyView === "tree" ? (
                                <>
                                  <div className="border rounded p-3 bg-light">
                                    <form
                                      className="row g-2 align-items-end"
                                      onSubmit={handleCitationSearch}
                                    >
                                      <div className="col-md-6">
                                        <label className="form-label fw-semibold mb-1">
                                          法條搜尋（例：第3條第1項第19款第3目）
                                        </label>
                                        <input
                                          type="text"
                                          className="form-control"
                                          placeholder="輸入章、節、條、項、款、目等組合"
                                          value={citationQuery}
                                          onChange={(e) =>
                                            setCitationQuery(e.target.value)
                                          }
                                        />
                                      </div>
                                      <div className="col-md-3">
                                        <button
                                          type="submit"
                                          className="btn btn-primary w-100"
                                          disabled={citationLoading}
                                        >
                                          {citationLoading ? (
                                            <>
                                              <span className="spinner-border spinner-border-sm me-2"></span>
                                              搜尋中...
                                            </>
                                          ) : (
                                            <>
                                              <i className="bi bi-search me-1"></i>
                                              搜尋法條
                                            </>
                                          )}
                                        </button>
                                      </div>
                                      <div className="col-md-3 text-md-end">
                                        <div className="d-flex gap-2">
                                          <button
                                            type="button"
                                            className="btn btn-outline-secondary w-100"
                                            onClick={handleResetCitationSearch}
                                          >
                                            <i className="bi bi-eraser me-1"></i>
                                            清除
                                          </button>
                                          <button
                                            type="button"
                                            className="btn btn-outline-primary w-100"
                                            onClick={handleRefreshHierarchy}
                                            disabled={hierarchyLoading}
                                          >
                                            <i className="bi bi-arrow-clockwise me-1"></i>
                                            重新整理
                                          </button>
                                        </div>
                                      </div>
                                    </form>
                                    {renderSearchFilterSummary()}
                                  </div>

                                  {citationError && (
                                    <div className="alert alert-danger mt-3">
                                      <i className="bi bi-exclamation-octagon me-2"></i>
                                      {citationError}
                                    </div>
                                  )}

                                  {citationResults.length > 0 && (
                                    <div className="mt-3">
                                      <h6 className="fw-semibold mb-2">
                                        <i className="bi bi-search me-1"></i>
                                        搜尋結果（{citationResults.length}）
                                      </h6>
                                      {citationResults.map((result) => (
                                        <div
                                          key={result.chunk_id}
                                          className={`card mb-2 ${
                                            result.chunk_id === highlightedChunkId
                                              ? "border-primary"
                                              : ""
                                          }`}
                                        >
                                          <div className="card-body">
                                            <div className="d-flex justify-content-between align-items-center">
                                              <div>
                                                <div className="fw-semibold">
                                                  {result.path_label}
                                                </div>
                                                <div className="text-muted small">
                                                  {result.doc_name} · 分塊 {" "}
                                                  {result.chunk_index + 1}
                                                </div>
                                              </div>
                                              <div className="d-flex gap-2">
                                                <button
                                                  type="button"
                                                  className="btn btn-sm btn-outline-primary"
                                                  onClick={() =>
                                                    handleHighlightChunk(
                                                      result.chunk_id
                                                    )
                                                  }
                                                >
                                                  <i className="bi bi-diagram-3 me-1"></i>
                                                  樹狀定位
                                                </button>
                                                <button
                                                  type="button"
                                                  className="btn btn-sm btn-outline-secondary"
                                                  onClick={() =>
                                                    navigator.clipboard.writeText(
                                                      result.chunk_id
                                                    )
                                                  }
                                                >
                                                  <i className="bi bi-clipboard me-1"></i>
                                                  複製ID
                                                </button>
                                              </div>
                                            </div>
                                            <pre
                                              className="small mt-3 p-3 bg-light border rounded"
                                              style={{ whiteSpace: "pre-wrap" }}
                                            >
                                              {result.content}
                                            </pre>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  )}

                                  <div className="mt-4">
                                    {hierarchyLoading ? (
                                      <div className="text-center py-4">
                                        <div
                                          className="spinner-border text-primary"
                                          role="status"
                                        >
                                          <span className="visually-hidden">
                                            載入中...
                                          </span>
                                        </div>
                                        <p className="mt-2 text-muted">
                                          載入層級結構...
                                        </p>
                                      </div>
                                    ) : hierarchyError ? (
                                      <div className="alert alert-danger">
                                        <i className="bi bi-exclamation-triangle me-2"></i>
                                        {hierarchyError}
                                      </div>
                                    ) : hierarchyData.length === 0 ? (
                                      <div className="alert alert-warning">
                                        <i className="bi bi-info-circle me-2"></i>
                                        尚無層級資料，請確認已完成多層級分塊
                                      </div>
                                    ) : (
                                      hierarchyData.map((doc) => (
                                        <div
                                          key={doc.doc_id}
                                          className="mb-4 border rounded p-3"
                                        >
                                          <div className="d-flex justify-content-between align-items-center mb-2">
                                            <h6 className="fw-bold mb-0">
                                              <i className="bi bi-file-earmark-text me-2"></i>
                                              {doc.doc_name}
                                            </h6>
                                            {hierarchyTimestamp && (
                                              <small className="text-muted">
                                                更新時間 {hierarchyTimestamp}
                                              </small>
                                            )}
                                          </div>
                                          {doc.law_nodes.map((node) =>
                                            renderHierarchyNode(node)
                                          )}
                                        </div>
                                      ))
                                    )}
                                  </div>
                                </>
                              ) : (
                                <>
                                  <div className="d-flex justify-content-between align-items-center mb-3">
                                    <p className="text-muted mb-0">
                                      使用列表檢視瀏覽原始分塊結果
                                    </p>
                                    <button
                                      className="btn btn-outline-primary btn-sm"
                                      onClick={() => setShowAllChunks(!showAllChunks)}
                                    >
                                      {showAllChunks ? (
                                        <>
                                          <i className="bi bi-eye-slash me-1"></i>
                                          隱藏列表
                                        </>
                                      ) : (
                                        <>
                                          <i className="bi bi-eye me-1"></i>
                                          查看列表
                                        </>
                                      )}
                                    </button>
                                  </div>

                                  {showAllChunks && (
                                    <div className="mb-3">
                                      <div className="row g-2 mb-3">
                                        <div className="col-md-4">
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
                                        <div className="col-md-2">
                                          <select
                                            className="form-select form-select-sm"
                                            value={selectedHierarchyLevel}
                                            onChange={(e) =>
                                              setSelectedHierarchyLevel(
                                                e.target.value
                                              )
                                            }
                                          >
                                            <option value="all">所有層級</option>
                                            <option value="document">章級</option>
                                            <option value="document_component">
                                              節級
                                            </option>
                                            <option value="basic_unit_hierarchy">
                                              條級
                                            </option>
                                            <option value="basic_unit">項級</option>
                                            <option value="basic_unit_component">
                                              款級
                                            </option>
                                            <option value="enumeration">目級</option>
                                          </select>
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
                                                  parseInt(e.target.value) ||
                                                  10000,
                                              }))
                                            }
                                          />
                                        </div>
                                      </div>

                                      {(() => {
                                        if (selectedHierarchyLevel === "all") {
                                          const filteredChunks =
                                            getFilteredChunks();
                                          const totalChunks =
                                            chunkingResults.reduce(
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
                                                顯示 {filteredChunks.length} / {" "}
                                                {totalChunks} 個分塊
                                                {chunkSearchTerm &&
                                                  ` (搜索: "${chunkSearchTerm}")`}
                                                {(chunkFilterLength.min > 0 ||
                                                  chunkFilterLength.max < 10000) &&
                                                  ` (長度: ${chunkFilterLength.min}-${chunkFilterLength.max} 字符)`}
                                              </small>
                                            </div>
                                          );
                                        } else {
                                          const levelNames: Record<string, string> = {
                                            document: "章級",
                                            document_component: "節級",
                                            basic_unit_hierarchy: "條級",
                                            basic_unit: "項級",
                                            basic_unit_component: "款級",
                                            enumeration: "目級",
                                          };
                                          return (
                                            <div className="alert alert-info py-2">
                                              <small>
                                                {loadingHierarchyChunks ? (
                                                  <>
                                                    <i className="spinner-border spinner-border-sm me-1"></i>
                                                    載入 {" "}
                                                    {levelNames[selectedHierarchyLevel] ||
                                                      selectedHierarchyLevel}{" "}
                                                    分塊中...
                                                  </>
                                                ) : (
                                                  <>
                                                    顯示 {hierarchyChunks.length} 個 {" "}
                                                    {levelNames[selectedHierarchyLevel] ||
                                                      selectedHierarchyLevel}{" "}
                                                    分塊
                                                    {chunkSearchTerm &&
                                                      ` (搜索: "${chunkSearchTerm}")`}
                                                  </>
                                                )}
                                              </small>
                                            </div>
                                          );
                                        }
                                      })()}
                                    </div>
                                  )}

                                  <div
                                    className="accordion"
                                    id={`chunkPreview-${Date.now()}`}
                                  >
                                    {showAllChunks &&
                                      selectedHierarchyLevel === "all" &&
                                      getFilteredChunks().map(
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
                                                          Span: [{span.start}-{span.end}]
                                                        </small>
                                                      )}
                                                    </div>
                                                    <span className="badge bg-secondary">
                                                      {getLevelLabel(metadata) ||
                                                        `${chunk.length} 字符`}
                                                    </span>
                                                  </div>
                                                </button>
                                              </h2>
                                              <div
                                                id={uniqueId}
                                                className="accordion-collapse collapse"
                                                data-bs-parent={`#${accordionId}`}
                                              >
                                                <div className="accordion-body">
                                                  <div className="d-flex justify-content-between align-items-center mb-2">
                                                    <div className="d-flex flex-column">
                                                      <small className="text-muted">
                                                        {strategyInfo[
                                                          strategy as ChunkStrategy
                                                        ]?.name || strategy}{" "}
                                                        | 分塊 {chunkIndex + 1}
                                                        {metadata
                                                          ? ` | ${getLevelPathText(metadata)}`
                                                          : ` | 大小: ${config.chunk_size} | 重疊: ${(
                                                              config.overlap_ratio * 100
                                                            ).toFixed(1)}%`}
                                                      </small>
                                                      {chunkId && (
                                                        <small className="text-primary fw-bold">
                                                          Chunk ID: {chunkId}
                                                        </small>
                                                      )}
                                                      {span && (
                                                        <small className="text-info">
                                                          原文位置: [{span.start}-{span.end}] 字符
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
                                    {showAllChunks &&
                                    selectedHierarchyLevel !== "all" ? (
                                      loadingHierarchyChunks ? (
                                        <div className="text-center py-4">
                                          <div
                                            className="spinner-border text-primary"
                                            role="status"
                                          >
                                            <span className="visually-hidden">
                                              載入中...
                                            </span>
                                          </div>
                                          <p className="mt-2 text-muted">
                                            載入分塊中...
                                          </p>
                                        </div>
                                      ) : hierarchyChunks.length > 0 ? (
                                        hierarchyChunks
                                          .filter((chunk: any) => {
                                            if (chunkSearchTerm) {
                                              return chunk.content
                                                .toLowerCase()
                                                .includes(
                                                  chunkSearchTerm.toLowerCase()
                                                );
                                            }
                                            return true;
                                          })
                                          .map((chunkInfo: any, index: number) => (
                                            <div
                                              key={`hierarchy-${index}`}
                                              className="accordion-item"
                                            >
                                              <h2 className="accordion-header">
                                                <button
                                                  className="accordion-button collapsed"
                                                  type="button"
                                                  data-bs-toggle="collapse"
                                                  data-bs-target={`#hierarchy-chunk-${index}`}
                                                  aria-expanded="false"
                                                  aria-controls={`hierarchy-chunk-${index}`}
                                                >
                                                  <div className="d-flex w-100 justify-content-between align-items-center">
                                                    <span className="fw-bold">
                                                      {chunkInfo.chunk_id}
                                                    </span>
                                                    <span className="badge bg-info me-2">
                                                      {chunkInfo.level}
                                                    </span>
                                                    <small className="text-muted">
                                                      {chunkInfo.content.length} 字符
                                                    </small>
                                                  </div>
                                                </button>
                                              </h2>
                                              <div
                                                id={`hierarchy-chunk-${index}`}
                                                className="accordion-collapse collapse"
                                                data-bs-parent="#chunkPreview"
                                              >
                                                <div className="accordion-body">
                                                  <div className="row">
                                                    <div className="col-md-8">
                                                      <h6>分塊內容：</h6>
                                                      <div
                                                        className="bg-light p-3 rounded"
                                                        style={{
                                                          maxHeight: "300px",
                                                          overflowY: "auto",
                                                        }}
                                                      >
                                                        <pre className="mb-0 small">
                                                          {chunkInfo.content}
                                                        </pre>
                                                      </div>
                                                    </div>
                                                    <div className="col-md-4">
                                                      <h6>元數據：</h6>
                                                      <div className="bg-light p-3 rounded small">
                                                        <p className="mb-1">
                                                          <strong>文檔：</strong> {chunkInfo.doc_name}
                                                        </p>
                                                        <p className="mb-1">
                                                          <strong>層級：</strong> {chunkInfo.level}
                                                        </p>
                                                        {chunkInfo.metadata &&
                                                          Object.keys(
                                                            chunkInfo.metadata
                                                          ).length > 0 && (
                                                            <div>
                                                              <strong>詳細信息：</strong>
                                                              <pre className="small mt-1">
                                                                {JSON.stringify(
                                                                  chunkInfo.metadata,
                                                                  null,
                                                                  2
                                                                )}
                                                              </pre>
                                                            </div>
                                                          )}
                                                      </div>
                                                    </div>
                                                  </div>
                                                </div>
                                              </div>
                                            </div>
                                          ))
                                      ) : (
                                        <div className="alert alert-warning">
                                          <i className="bi bi-search me-2"></i>
                                          該層級沒有找到分塊
                                        </div>
                                      )
                                    ) : null}
                                  </div>

                                  {showAllChunks &&
                                    selectedHierarchyLevel === "all" &&
                                    getFilteredChunks().length === 0 && (
                                      <div className="alert alert-warning">
                                        <i className="bi bi-search me-2"></i>
                                        沒有找到符合條件的分塊
                                      </div>
                                    )}
                                </>
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
                                disabled={isChunking || !jsonData}
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
                              {!jsonData && (
                                <div
                                  className="alert alert-warning mt-2"
                                  role="alert"
                                >
                                  請先於上傳頁完成 PDF 轉換或 JSON
                                  上傳，並確保後端已保存 JSON 後再進行分塊。
                                </div>
                              )}
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

      {/* Metadata Viewer Modal */}
      {showMetadataViewer && (
        <MetadataViewer onClose={() => setShowMetadataViewer(false)} />
      )}
    </div>
  );
}
