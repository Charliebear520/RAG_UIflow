const base = import.meta.env.VITE_API_BASE_URL || "http://localhost:8001/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    try {
      const errorData = await res.json();
      console.log("Error response data:", errorData); // 調試用
      // 如果有具體的錯誤信息，直接拋出
      if (errorData.error) {
        throw new Error(errorData.error);
      }
      // 檢查FastAPI的detail字段
      if (errorData.detail) {
        throw new Error(errorData.detail);
      }
      throw new Error(`${res.status} ${res.statusText}`);
    } catch (e) {
      console.log("Error parsing response:", e); // 調試用
      // 如果已經是一個Error對象且有具體信息，直接重新拋出
      if (
        e instanceof Error &&
        e.message !== "Unexpected token < in JSON at position 0"
      ) {
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
  async retrieve(body: { query: string; k: number }) {
    const res = await fetch(`${base}/retrieve`, {
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
};
