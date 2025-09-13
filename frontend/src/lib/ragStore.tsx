import React, { createContext, useContext, useMemo, useState } from "react";
import { api } from "./api";

type ChunkMeta = { size: number; overlap: number; count: number };

type RagContextType = {
  // data
  docId: string | null;
  chunkMeta: ChunkMeta | null;
  embedProvider: string | null;
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
  updateJsonData: (newJsonData: any) => void;
  chunk: (
    size: number,
    overlap: number,
    strategy?: string,
    extraParams?: any
  ) => Promise<any>;
  embed: () => Promise<void>;
  retrieve: (query: string, k: number) => Promise<void>;
  generate: (query: string, topK: number) => Promise<void>;
  reset: () => void;
};

const RagContext = createContext<RagContextType | undefined>(undefined);

export function RagProvider({ children }: { children: React.ReactNode }) {
  const [docId, setDocId] = useState<string | null>(null);
  const [chunkMeta, setChunkMeta] = useState<ChunkMeta | null>(null);
  const [embedProvider, setEmbedProvider] = useState<string | null>(null);
  const [retrieval, setRetrieval] = useState<any[] | null>(null);
  const [answer, setAnswer] = useState<string | null>(null);
  const [steps, setSteps] = useState<any[] | null>(null);
  const [legalReferences, setLegalReferences] = useState<string[] | null>(null);
  const [jsonData, setJsonData] = useState<any | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canChunk = useMemo(() => !!docId, [docId]);
  const canEmbed = useMemo(() => !!docId && !!chunkMeta, [docId, chunkMeta]);
  const canRetrieve = useMemo(() => !!embedProvider, [embedProvider]);
  const canGenerate = useMemo(
    () => !!retrieval && retrieval.length > 0,
    [retrieval]
  );

  async function upload(file: File) {
    setLoading(true);
    try {
      const res = await api.upload(file);
      setDocId(res.doc_id);
      // reset downstream state
      setChunkMeta(null);
      setEmbedProvider(null);
      setRetrieval(null);
      setAnswer(null);
      setSteps(null);
      setFileName(file.name);
    } finally {
      setLoading(false);
    }
  }

  async function convert(file: File, metadataOptions?: any) {
    setLoading(true);
    try {
      const res = await api.convert(file, metadataOptions);
      setJsonData(res);
      setFileName(file.name);

      // 如果有docId，同步JSON數據到後端
      if (docId) {
        try {
          await api.updateJson(docId, res);
          console.log("JSON data synchronized to backend");
        } catch (error) {
          console.warn("Failed to sync JSON data to backend:", error);
        }
      }
    } catch (error) {
      console.error("Convert error:", error);
      // Reset state on error
      setJsonData(null);
      setFileName(null);
      // You could also set an error state here if needed
      throw error;
    } finally {
      setLoading(false);
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

  async function embed() {
    const res = await api.embed();
    setEmbedProvider(res.provider);
  }

  async function retrieve(query: string, k: number) {
    const res = await api.retrieve({ query, k });
    setRetrieval(res.results);
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
    setEmbedProvider(null);
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
    embedProvider,
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
    updateJsonData,
    chunk,
    embed,
    retrieve,
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
