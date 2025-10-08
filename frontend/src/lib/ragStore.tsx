import React, { createContext, useContext, useMemo, useState } from "react";
import { api } from "./api";

type ChunkMeta = { size: number; overlap: number; count: number };

type ChunkingResult = {
  strategy: string;
  config: any;
  chunk_count: number;
  metrics: any;
  chunks_with_span?: any[];
  all_chunks?: string[];
  chunks?: string[];
};

type RagContextType = {
  // data
  docId: string | null;
  chunkMeta: ChunkMeta | null;
  chunkingResults: ChunkingResult[] | null;
  selectedStrategy: string | null;
  embedProvider: string | null;
  embedModel: string | null;
  embedDimension: number | null;
  retrieval: any[] | null;
  answer: string | null;
  steps: any[] | null;
  legalReferences: string[] | null;
  jsonData: any | null;
  fileName: string | null;
  loading: boolean;

  // derived
  canChunk: boolean;
  canEmbed: boolean;
  canRetrieve: boolean;
  canGenerate: boolean;

  // actions
  upload: (file: File) => Promise<void>;
  convert: (file: File, metadataOptions?: any) => Promise<void>;
  uploadJson: (file: File) => Promise<void>;
  updateJsonData: (newJsonData: any) => void;
  setDocId: (docId: string | null) => void;
  chunk: (
    size: number,
    overlap: number,
    strategy?: string,
    extraParams?: any
  ) => Promise<any>;
  setChunkingResultsAndStrategy: (
    results: ChunkingResult[],
    strategy: string
  ) => void;
  embed: () => Promise<void>;
  multiLevelEmbed: () => Promise<void>;
  retrieve: (query: string, k: number) => Promise<void>;
  hybridRetrieve: (query: string, k: number) => Promise<void>;
  hierarchicalRetrieve: (query: string, k: number) => Promise<void>;
  multiLevelRetrieve: (query: string, k: number) => Promise<void>;
  multiLevelFusionRetrieve: (
    query: string,
    k: number,
    fusionStrategy?: string
  ) => Promise<void>;
  hopragEnhancedRetrieve: (query: string, k: number) => Promise<void>;
  generate: (query: string, topK: number) => Promise<void>;
  reset: () => void;
};

const RagContext = createContext<RagContextType | undefined>(undefined);

export function RagProvider({ children }: { children: React.ReactNode }) {
  const [docId, setDocId] = useState<string | null>(null);
  const [chunkMeta, setChunkMeta] = useState<ChunkMeta | null>(null);
  const [chunkingResults, setChunkingResults] = useState<
    ChunkingResult[] | null
  >(null);
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [embedProvider, setEmbedProvider] = useState<string | null>(null);
  const [embedModel, setEmbedModel] = useState<string | null>(null);
  const [embedDimension, setEmbedDimension] = useState<number | null>(null);
  const [retrieval, setRetrieval] = useState<any[] | null>(null);
  const [answer, setAnswer] = useState<string | null>(null);
  const [steps, setSteps] = useState<any[] | null>(null);
  const [legalReferences, setLegalReferences] = useState<string[] | null>(null);
  const [jsonData, setJsonData] = useState<any | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canChunk = useMemo(() => !!docId, [docId]);
  const canEmbed = useMemo(
    () => !!docId && !!chunkingResults,
    [docId, chunkingResults]
  );
  const canRetrieve = useMemo(() => !!embedProvider, [embedProvider]);
  const canGenerate = useMemo(
    () => !!retrieval && retrieval.length > 0,
    [retrieval]
  );

  async function upload(file: File) {
    // upload函數現在只負責設置文件名，實際轉換由convert函數處理
    setFileName(file.name);
    // reset downstream state
    setDocId(null);
    setChunkMeta(null);
    setChunkingResults(null);
    setSelectedStrategy(null);
    setEmbedProvider(null);
    setEmbedModel(null);
    setEmbedDimension(null);
    setRetrieval(null);
    setAnswer(null);
    setSteps(null);
    setJsonData(null);
  }

  async function convert(file: File, metadataOptions?: any) {
    setLoading(true);
    try {
      const res = await api.convert(file, metadataOptions);

      // 檢查是否直接返回結果（緩存命中）
      if (res.doc_id) {
        setDocId(res.doc_id);
        setJsonData(res.metadata);
        setFileName(file.name);

        // 重置下游狀態
        setChunkMeta(null);
        setChunkingResults(null);
        setSelectedStrategy(null);
        setEmbedProvider(null);
        setEmbedModel(null);
        setEmbedDimension(null);
        setRetrieval(null);
        setAnswer(null);
        setSteps(null);
        return;
      }

      // 異步任務，需要輪詢狀態
      if (res.task_id) {
        await pollConvertStatus(res.task_id, file.name);
      }
    } catch (error) {
      console.error("Convert error:", error);
      // Reset state on error
      setDocId(null);
      setJsonData(null);
      setFileName(null);
      throw error;
    } finally {
      setLoading(false);
    }
  }

  async function pollConvertStatus(taskId: string, filename: string) {
    const maxAttempts = 120; // 最多等待2分鐘
    let attempts = 0;

    while (attempts < maxAttempts) {
      try {
        const status = await api.getConvertStatus(taskId);

        if (status.status === "completed") {
          // 轉換完成
          setDocId(status.result.doc_id);
          setJsonData(status.result.metadata);
          setFileName(filename);

          // 重置下游狀態
          setChunkMeta(null);
          setChunkingResults(null);
          setSelectedStrategy(null);
          setEmbedProvider(null);
          setEmbedModel(null);
          setEmbedDimension(null);
          setRetrieval(null);
          setAnswer(null);
          setSteps(null);
          break;
        } else if (status.status === "failed") {
          throw new Error(status.error || "PDF轉換失敗");
        }

        // 等待1秒後重試
        await new Promise((resolve) => setTimeout(resolve, 1000));
        attempts++;
      } catch (error) {
        if (attempts >= maxAttempts - 1) {
          throw error;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
        attempts++;
      }
    }

    if (attempts >= maxAttempts) {
      throw new Error("PDF轉換超時");
    }
  }

  function updateJsonData(newJsonData: any) {
    setJsonData(newJsonData);

    // 如果有docId，同步JSON數據到後端
    if (docId) {
      api
        .updateJson(docId, newJsonData)
        .then(() => {
          console.log("JSON data updated in backend");
        })
        .catch((error) => {
          console.warn("Failed to update JSON data in backend:", error);
        });
    }
  }

  async function uploadJson(file: File) {
    setLoading(true);
    try {
      const res = await api.uploadJson(file);

      // 設置文檔信息
      setDocId(res.doc_id);
      setJsonData(res.metadata || null);
      setFileName(file.name);

      // 重置下游狀態
      setChunkMeta(null);
      setChunkingResults(null);
      setSelectedStrategy(null);
      setEmbedProvider(null);
      setEmbedModel(null);
      setEmbedDimension(null);
      setRetrieval(null);
      setAnswer(null);
      setSteps(null);
    } catch (error) {
      console.error("Upload JSON error:", error);
      // Reset state on error
      setDocId(null);
      setJsonData(null);
      setFileName(null);
      throw error;
    } finally {
      setLoading(false);
    }
  }

  async function chunk(
    size: number,
    overlap: number,
    strategy?: string,
    extraParams?: any
  ) {
    if (!docId) return;
    const requestBody = {
      doc_id: docId,
      chunk_size: size,
      overlap,
      strategy,
      use_json_structure: !!jsonData, // 如果有JSON數據，啟用結構化分割
      ...extraParams,
    };
    const res = await api.chunk(requestBody);
    setChunkMeta({ size, overlap, count: res.num_chunks });
    return res;
  }

  function setChunkingResultsAndStrategy(
    results: ChunkingResult[],
    strategy: string
  ) {
    setChunkingResults(results);
    setSelectedStrategy(strategy);
    // 重置下游狀態
    setEmbedProvider(null);
    setEmbedModel(null);
    setEmbedDimension(null);
    setRetrieval(null);
    setAnswer(null);
    setSteps(null);
  }

  async function embed() {
    const res = await api.embed();
    setEmbedProvider(res.provider);
    setEmbedModel(res.model);
    setEmbedDimension(res.dimension || res.num_features || null);
  }

  async function multiLevelEmbed() {
    const res = await api.multiLevelEmbed();
    setEmbedProvider(
      res.levels?.conceptual?.provider ||
        res.levels?.procedural?.provider ||
        res.levels?.normative?.provider ||
        "multi-level"
    );
    setEmbedModel(
      res.levels?.conceptual?.model ||
        res.levels?.procedural?.model ||
        res.levels?.normative?.model ||
        "multi-level"
    );
    setEmbedDimension(
      res.levels?.conceptual?.dimension ||
        res.levels?.procedural?.dimension ||
        res.levels?.normative?.dimension ||
        null
    );
  }

  async function retrieve(query: string, k: number) {
    const res = await api.retrieve({ query, k });
    // 將results數組和額外信息合併保存到retrieval
    const retrievalData = res.results.map((result: any) => ({
      ...result,
      metrics: res.metrics,
      embedding_provider: res.embedding_provider,
      embedding_model: res.embedding_model,
    }));
    setRetrieval(retrievalData);
    // 更新 embedding 信息（如果檢索端點返回了這些信息）
    if (res.embedding_provider) {
      setEmbedProvider(res.embedding_provider);
    }
    if (res.embedding_model) {
      setEmbedModel(res.embedding_model);
    }
  }

  async function hybridRetrieve(query: string, k: number) {
    const res = await api.hybridRetrieve({ query, k });
    // 將results數組和額外信息合併保存到retrieval
    const retrievalData = res.results.map((result: any) => ({
      ...result,
      metrics: res.metrics,
      embedding_provider: res.embedding_provider,
      embedding_model: res.embedding_model,
    }));
    setRetrieval(retrievalData);
    // 更新 embedding 信息（如果檢索端點返回了這些信息）
    if (res.embedding_provider) {
      setEmbedProvider(res.embedding_provider);
    }
    if (res.embedding_model) {
      setEmbedModel(res.embedding_model);
    }
  }

  async function hierarchicalRetrieve(query: string, k: number) {
    const res = await api.hierarchicalRetrieve({ query, k });
    // 將results數組和額外信息合併保存到retrieval
    const retrievalData = res.results.map((result: any) => ({
      ...result,
      metrics: res.metrics,
      embedding_provider: res.embedding_provider,
      embedding_model: res.embedding_model,
    }));
    setRetrieval(retrievalData);
    // 更新 embedding 信息（如果檢索端點返回了這些信息）
    if (res.embedding_provider) {
      setEmbedProvider(res.embedding_provider);
    }
    if (res.embedding_model) {
      setEmbedModel(res.embedding_model);
    }
  }

  async function multiLevelRetrieve(query: string, k: number) {
    const res = await api.multiLevelRetrieve({ query, k });
    const retrievalData = res.results.map((result: any) => ({
      ...result,
      metrics: res.metrics,
      query_analysis: res.query_analysis,
    }));
    setRetrieval(retrievalData);
  }

  async function multiLevelFusionRetrieve(
    query: string,
    k: number,
    fusionStrategy: string = "weighted_sum"
  ) {
    const res = await api.multiLevelFusionRetrieve({
      query,
      k,
      fusion_strategy: fusionStrategy,
    });
    const retrievalData = res.results.map((result: any) => ({
      ...result,
      metrics: res.metrics,
      query_analysis: res.query_analysis,
      level_results: res.level_results,
    }));
    setRetrieval(retrievalData);
  }

  async function hopragEnhancedRetrieve(query: string, k: number) {
    try {
      console.log("🔍 開始HopRAG檢索:", { query, k });
      const res = await api.hopragEnhancedRetrieve({
        query,
        k,
      });
      console.log("✅ HopRAG檢索響應:", res);

      const retrievalData = res.results.map((result: any) => ({
        ...result,
        strategy: res.strategy,
        base_strategy: res.base_strategy,
        hoprag_enabled: res.hoprag_enabled,
        hop_level: result.hop_level || 0,
        hop_source: result.hop_source || "base_retrieval",
      }));

      console.log("📊 處理後的檢索數據:", retrievalData);
      setRetrieval(retrievalData);
      console.log("✅ 檢索數據已設置到狀態");
    } catch (error) {
      console.error("❌ HopRAG檢索失敗:", error);
      throw error;
    }
  }

  async function generate(query: string, topK: number) {
    const res = await api.generate({ query, top_k: topK });
    setAnswer(res.answer);
    setSteps(res.steps);
    setLegalReferences(res.legal_references || []);
  }

  function reset() {
    setDocId(null);
    setChunkMeta(null);
    setChunkingResults(null);
    setSelectedStrategy(null);
    setEmbedProvider(null);
    setEmbedModel(null);
    setEmbedDimension(null);
    setRetrieval(null);
    setAnswer(null);
    setSteps(null);
    setLegalReferences(null);
    setJsonData(null);
    setFileName(null);
  }

  const value: RagContextType = {
    docId,
    chunkMeta,
    chunkingResults,
    selectedStrategy,
    embedProvider,
    embedModel,
    embedDimension,
    retrieval,
    answer,
    steps,
    legalReferences,
    jsonData,
    fileName,
    loading,
    canChunk,
    canEmbed,
    canRetrieve,
    canGenerate,
    upload,
    convert,
    uploadJson,
    updateJsonData,
    setDocId,
    chunk,
    setChunkingResultsAndStrategy,
    embed,
    multiLevelEmbed,
    retrieve,
    hybridRetrieve,
    hierarchicalRetrieve,
    multiLevelRetrieve,
    multiLevelFusionRetrieve,
    hopragEnhancedRetrieve,
    generate,
    reset,
  };

  return <RagContext.Provider value={value}>{children}</RagContext.Provider>;
}

export function useRag() {
  const ctx = useContext(RagContext);
  if (!ctx) throw new Error("useRag must be used within RagProvider");
  return ctx;
}
