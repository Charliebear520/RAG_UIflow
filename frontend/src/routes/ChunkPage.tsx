import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useRag } from '../lib/ragStore'

export function ChunkPage() {
  const nav = useNavigate()
  const { canChunk, chunk, docId, chunkMeta } = useRag()
  const [size, setSize] = useState(500)
  const [overlap, setOverlap] = useState(50)
  const [busy, setBusy] = useState(false)

  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-3">Chunk</h2>
        {!canChunk && <p className="text-muted">Upload a document first.</p>}
        <form
          className="row gy-2 gx-3 align-items-center"
          onSubmit={async (e) => {
            e.preventDefault()
            if (!canChunk) return
            setBusy(true)
            await chunk(size, overlap)
            setBusy(false)
          }}
        >
          <div className="col-auto">
            <label className="form-label">Chunk size</label>
            <input className="form-control" type="number" value={size} onChange={(e) => setSize(parseInt(e.target.value || '0') || 0)} />
          </div>
          <div className="col-auto">
            <label className="form-label">Overlap</label>
            <input className="form-control" type="number" value={overlap} onChange={(e) => setOverlap(parseInt(e.target.value || '0') || 0)} />
          </div>
          <div className="col-auto align-self-end">
            <button disabled={!canChunk || busy} className="btn btn-primary" type="submit">
              {busy ? 'Chunking…' : 'Run chunker'}
            </button>
          </div>
        </form>
        {chunkMeta && (
          <div className="mt-3">
            <div className="text-muted">doc_id: <code>{docId}</code></div>
            <div>Chunks: {chunkMeta.count} (size={chunkMeta.size}, overlap={chunkMeta.overlap})</div>
            <button className="btn btn-success btn-sm mt-2" onClick={() => nav('/embed')}>Continue to Embed</button>
          </div>
        )}
      </div>
    </div>
  )
}
