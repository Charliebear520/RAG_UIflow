import React, { useMemo, useState } from 'react'
import { api } from './lib/api'

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <h2 style={{ marginTop: 0 }}>{title}</h2>
      {children}
    </section>
  )
}

export default function App() {
  const [docId, setDocId] = useState<string | null>(null)
  const [chunkMeta, setChunkMeta] = useState<{ size: number; overlap: number; count: number } | null>(null)
  const [embedProvider, setEmbedProvider] = useState<string | null>(null)
  const [retrieval, setRetrieval] = useState<any[] | null>(null)
  const [answer, setAnswer] = useState<string | null>(null)
  const [steps, setSteps] = useState<any[] | null>(null)
  const [jsonData, setJsonData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [fileName, setFileName] = useState('')

  const canEmbed = useMemo(() => !!docId && !!chunkMeta, [docId, chunkMeta])
  const canRetrieve = useMemo(() => !!embedProvider, [embedProvider])
  const canGenerate = useMemo(() => !!retrieval, [retrieval])

  const handleUpload = async (file: File) => {
    setLoading(true)
    try {
      const response = await api.convert(file)
      setJsonData(response)
      setFileName(file.name)
    } catch (error) {
      console.error('Error converting file:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: 24, fontFamily: 'Inter, system-ui, Avenir, Arial' }}>
      <h1>RAG Visualizer</h1>
      <p style={{ color: '#666' }}>Upload → Chunk → Embed → Retrieve → Generate</p>

      <Section title="1) Upload">
        <input
          type="file"
          onChange={async (e: React.ChangeEvent<HTMLInputElement>) => {
            const f = e.target.files?.[0]
            if (!f) return
            const res = await api.upload(f)
            setDocId(res.doc_id)
            setChunkMeta(null)
            setEmbedProvider(null)
            setRetrieval(null)
            setAnswer(null)
            setSteps(null)
          }}
        />
        {docId && <p>Uploaded doc_id: <code>{docId}</code></p>}
      </Section>

      <Section title="2) Chunking">
  <ChunkForm
          disabled={!docId}
          onSubmit={async (size, overlap) => {
            if (!docId) return
            const res = await api.chunk({ doc_id: docId, chunk_size: size, overlap })
            setChunkMeta({ size, overlap, count: res.num_chunks })
          }}
        />
        {chunkMeta && (
          <p>
            chunk_size={chunkMeta.size}, overlap={chunkMeta.overlap}, chunks={chunkMeta.count}
          </p>
        )}
      </Section>

      <Section title="3) Embedding">
  <button disabled={!canEmbed} onClick={async () => {
          const res = await api.embed()
          setEmbedProvider(res.provider)
        }}>Compute Embeddings</button>
        {embedProvider && <p>Provider: <code>{embedProvider}</code></p>}
      </Section>

      <Section title="4) Retrieval">
  <RetrieveForm disabled={!canRetrieve} onSubmit={async (query, k) => {
          const res = await api.retrieve({ query, k })
          setRetrieval(res.results)
        }} />
        {retrieval && (
          <div>
            <h3>Top Results</h3>
            <ol>
              {retrieval.map((r: any) => (
                <li key={`${r.doc_id}-${r.chunk_index}`}>
                  <div style={{ fontSize: 12, color: '#666' }}>
                    score={r.score.toFixed(3)} doc={r.doc_id} idx={r.chunk_index}
                  </div>
                  <pre style={{ whiteSpace: 'pre-wrap' }}>{r.content}</pre>
                </li>
              ))}
            </ol>
          </div>
        )}
      </Section>

      <Section title="5) Generation">
  <GenerateForm disabled={!canGenerate} onSubmit={async (query, topK) => {
          const res = await api.generate({ query, top_k: topK })
          setAnswer(res.answer)
          setSteps(res.steps)
        }} />
        {answer && (
          <div>
            <h3>Answer</h3>
            <pre style={{ whiteSpace: 'pre-wrap' }}>{answer}</pre>
          </div>
        )}
        {steps && (
          <div>
            <h3>Reasoning steps</h3>
            <ol>
              {steps.map((s, i) => (
                <li key={i}><strong>{s.type}:</strong> {s.text}</li>
              ))}
            </ol>
          </div>
        )}
      </Section>

      <Section title="Upload and Convert PDF">
        {!jsonData ? (
          <div>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => {
                const file = e.target.files[0]
                if (file) handleUpload(file)
              }}
            />
            {loading && <p>Converting PDF to JSON...</p>}
          </div>
        ) : (
          <div>
            <h2>Converted JSON</h2>
            <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
              {JSON.stringify(jsonData, null, 2)}
            </pre>
            <a
              href={`data:text/json;charset=utf-8,${encodeURIComponent(
                JSON.stringify(jsonData, null, 2)
              )}`}
              download={`${fileName.replace(/\.pdf$/, '')}.json`}
            >
              Download JSON
            </a>
            <button onClick={() => setJsonData(null)}>Upload Another File</button>
          </div>
        )}
      </Section>
    </div>
  )
}

function ChunkForm({ disabled, onSubmit }: { disabled?: boolean; onSubmit: (size: number, overlap: number) => void }) {
  const [size, setSize] = useState(500)
  const [overlap, setOverlap] = useState(50)
  return (
    <form onSubmit={(e: React.FormEvent<HTMLFormElement>) => { e.preventDefault(); onSubmit(size, overlap) }}>
      <label>
        size
        <input type="number" value={size} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSize(parseInt(e.target.value || '0') || 0)} style={{ marginLeft: 8, width: 100 }} />
      </label>
      <label style={{ marginLeft: 16 }}>
        overlap
        <input type="number" value={overlap} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setOverlap(parseInt(e.target.value || '0') || 0)} style={{ marginLeft: 8, width: 100 }} />
      </label>
      <button disabled={disabled} type="submit" style={{ marginLeft: 16 }}>Chunk</button>
    </form>
  )
}

function RetrieveForm({ disabled, onSubmit }: { disabled?: boolean; onSubmit: (query: string, k: number) => void }) {
  const [query, setQuery] = useState('')
  const [k, setK] = useState(5)
  return (
    <form onSubmit={(e: React.FormEvent<HTMLFormElement>) => { e.preventDefault(); onSubmit(query, k) }}>
      <input placeholder="Enter query" value={query} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)} style={{ width: 360 }} />
      <label style={{ marginLeft: 16 }}>
        top-k
        <input type="number" value={k} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setK(parseInt(e.target.value || '0') || 0)} style={{ marginLeft: 8, width: 80 }} />
      </label>
      <button disabled={disabled} type="submit" style={{ marginLeft: 16 }}>Search</button>
    </form>
  )
}

function GenerateForm({ disabled, onSubmit }: { disabled?: boolean; onSubmit: (query: string, topK: number) => void }) {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  return (
    <form onSubmit={(e: React.FormEvent<HTMLFormElement>) => { e.preventDefault(); onSubmit(query, topK) }}>
      <input placeholder="Question to answer" value={query} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)} style={{ width: 360 }} />
      <label style={{ marginLeft: 16 }}>
        top-k
        <input type="number" value={topK} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTopK(parseInt(e.target.value || '0') || 0)} style={{ marginLeft: 8, width: 80 }} />
      </label>
      <button disabled={disabled} type="submit" style={{ marginLeft: 16 }}>Generate</button>
    </form>
  )
}
