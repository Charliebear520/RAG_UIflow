const base = import.meta.env.VITE_API_BASE_URL || "/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  async upload(file: File) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${base}/upload`, { method: "POST", body: fd });
    return json<any>(res);
  },
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
  async getEvaluationResults(taskId: string) {
    const res = await fetch(`${base}/evaluate/results/${taskId}`);
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
};
