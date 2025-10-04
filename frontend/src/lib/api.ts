const base = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    try {
      // 先讀取響應文本
      const errorText = await res.text();
      console.log("Error response status:", res.status);
      console.log("Error response text:", errorText.substring(0, 500)); // 調試用，只顯示前500字符

      // 嘗試解析為JSON
      let errorData;
      try {
        errorData = JSON.parse(errorText);
      } catch (parseError) {
        // 如果不是JSON，直接使用文本
        throw new Error(errorText || `${res.status} ${res.statusText}`);
      }

      // 如果有具體的錯誤信息，直接拋出
      if (errorData.error) {
        throw new Error(errorData.error);
      }
      // 檢查FastAPI的detail字段
      if (errorData.detail) {
        throw new Error(errorData.detail);
      }
      // 如果是字符串類型的detail
      if (typeof errorData === "string") {
        throw new Error(errorData);
      }
      throw new Error(`${res.status} ${res.statusText}`);
    } catch (e) {
      console.log("Error in error handling:", e); // 調試用
      // 如果已經是一個Error對象且有具體信息，直接重新拋出
      if (e instanceof Error) {
        throw e;
      }
      // 如果解析失敗或其他錯誤，拋出通用錯誤
      throw new Error(`${res.status} ${res.statusText}`);
    }
  }

  // 檢查響應是否為空
  const text = await res.text();
  if (!text.trim()) {
    throw new Error("Empty response from server");
  }

  try {
    return JSON.parse(text);
  } catch (e) {
    console.log("Failed to parse JSON response:", text.substring(0, 200)); // 調試用，只顯示前200字符
    throw new Error(
      `Invalid JSON response: ${
        e instanceof Error ? e.message : "Unknown error"
      }`
    );
  }
}

export const api = {
  async chunk(body: {
    doc_id: string;
    chunk_size: number;
    overlap: number;
    strategy?: string;
    [key: string]: any;
  }) {
    const res = await fetch(`${base}/chunk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  async embed(body?: { doc_ids?: string[] }) {
    const res = await fetch(`${base}/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return json<any>(res);
  },
  async multiLevelEmbed(body?: { doc_ids?: string[] }) {
    const res = await fetch(`${base}/multi-level-embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return json<any>(res);
  },
  async retrieve(body: { query: string; k: number }) {
    const res = await fetch(`${base}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  async hybridRetrieve(body: { query: string; k: number }) {
    const res = await fetch(`${base}/hybrid-retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  async hierarchicalRetrieve(body: { query: string; k: number }) {
    const res = await fetch(`${base}/hierarchical-retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  async multiLevelRetrieve(body: { query: string; k: number }) {
    const res = await fetch(`${base}/multi-level-retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  async multiLevelFusionRetrieve(body: {
    query: string;
    k: number;
    fusion_strategy?: string;
  }) {
    const res = await fetch(`${base}/multi-level-fusion-retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  async generate(body: { query: string; top_k: number }) {
    const res = await fetch(`${base}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  async convert(file: File, metadataOptions?: any) {
    const fd = new FormData();
    fd.append("file", file);
    if (metadataOptions) {
      fd.append("metadata_options", JSON.stringify(metadataOptions));
    }
    const res = await fetch(`${base}/convert`, { method: "POST", body: fd });
    return json<any>(res);
  },
  async uploadJson(file: File) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${base}/upload-json`, {
      method: "POST",
      body: fd,
    });
    return json<any>(res);
  },
  async getConvertStatus(taskId: string) {
    const res = await fetch(`${base}/convert/status/${taskId}`);
    return json<any>(res);
  },
  async updateJson(docId: string, jsonData: any) {
    const res = await fetch(`${base}/update-json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: docId, json_data: jsonData }),
    });
    return json<any>(res);
  },
  // 評測相關API
  async startFixedSizeEvaluation(body: {
    doc_id: string;
    chunk_sizes?: number[];
    overlap_ratios?: number[];
    test_queries?: string[];
    k_values?: number[];
  }) {
    const res = await fetch(`${base}/evaluate/fixed-size`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  async getEvaluationStatus(taskId: string) {
    const res = await fetch(`${base}/evaluate/status/${taskId}`);
    return json<any>(res);
  },
  async getEvaluationComparison(taskId: string) {
    const res = await fetch(`${base}/evaluate/comparison/${taskId}`);
    return json<any>(res);
  },
  // 問題生成相關API
  async generateQuestions(body: {
    doc_id: string;
    num_questions?: number;
    question_types?: string[];
    difficulty_levels?: string[];
  }) {
    const res = await fetch(`${base}/generate-questions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  // Evaluation APIs used by EvaluationPanel
  async evaluateChunking(body: {
    doc_id: string;
    chunk_sizes: number[];
    overlap_ratios: number[];
    question_types: string[];
    num_questions: number;
  }) {
    const res = await fetch(`${base}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  async runEvaluation(taskId: string) {
    const res = await fetch(`${base}/evaluate/run/${taskId}`, {
      method: "POST",
    });
    return json<any>(res);
  },
  async getEvaluationTask(taskId: string) {
    const res = await fetch(`${base}/evaluate/task/${taskId}`);
    return json<any>(res);
  },
  async getEvaluationResults(taskId: string) {
    const res = await fetch(`${base}/evaluate/results/${taskId}`);
    return json<any>(res);
  },
  // QA Set上傳和映射相關API
  async uploadQASet(
    file: File,
    docId: string,
    chunkSizes: number[],
    overlapRatios: number[],
    strategy: string = "fixed_size"
  ) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("doc_id", docId);
    formData.append("chunk_sizes", JSON.stringify(chunkSizes));
    formData.append("overlap_ratios", JSON.stringify(overlapRatios));
    formData.append("strategy", strategy);

    const res = await fetch(`${base}/upload-qa-set`, {
      method: "POST",
      body: formData,
    });
    return json<any>(res);
  },
  async getQAMappingStatus(taskId: string) {
    const res = await fetch(`${base}/qa-mapping/status/${taskId}`);
    return json<any>(res);
  },
  async getQAMappingResult(taskId: string) {
    const res = await fetch(`${base}/qa-mapping/result/${taskId}`);
    return json<any>(res);
  },
  // 直接映射QA set（不包含分塊處理）
  async mapQASet(file: File, docId: string) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("doc_id", docId);

    const res = await fetch(`${base}/map-qa-set`, {
      method: "POST",
      body: formData,
    });
    return json<any>(res);
  },
  // 批量分塊相關API
  async startMultipleChunking(body: {
    doc_id: string;
    strategies: string[];
    chunk_sizes: number[];
    overlap_ratios: number[];
  }) {
    const res = await fetch(`${base}/chunk/multiple`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
  async getChunkingStatus(taskId: string) {
    const res = await fetch(`${base}/chunk/status/${taskId}`);
    return json<any>(res);
  },
  async getChunkingResults(taskId: string) {
    const res = await fetch(`${base}/chunk/results/${taskId}`);
    return json<any>(res);
  },
  // 策略評估相關API
  async startStrategyEvaluation(body: {
    doc_id: string;
    chunking_results: any[];
    qa_mapping_result: any;
    test_queries: string[];
    k_values: number[];
  }) {
    const res = await fetch(`${base}/evaluate/strategy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },

  // QA映射相關API
  async startQAMapping(body: {
    doc_id: string;
    qa_set: any[];
    chunking_results: any[];
    iou_threshold?: number;
  }) {
    const res = await fetch(`${base}/qa-mapping/map`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return json<any>(res);
  },
};
