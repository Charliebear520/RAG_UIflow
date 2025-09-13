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
  convert: (file: File) => Promise<void>;
  chunk: (size: number, overlap: number) => Promise<void>;
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

  async function convert(file: File) {
    setLoading(true);
    try {
      const res = await api.convert(file);
      setJsonData(res);
      setFileName(file.name);
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

  async function chunk(size: number, overlap: number) {
    if (!docId) return;
    const res = await api.chunk({ doc_id: docId, chunk_size: size, overlap });
    setChunkMeta({ size, overlap, count: res.num_chunks });
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
  }

  function reset() {
    setDocId(null);
    setChunkMeta(null);
    setEmbedProvider(null);
    setRetrieval(null);
    setAnswer(null);
    setSteps(null);
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
    jsonData,
    fileName,
    loading,
    canChunk,
    canEmbed,
    canRetrieve,
    canGenerate,
    upload,
    convert,
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
