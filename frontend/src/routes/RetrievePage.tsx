import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useRag } from '../lib/ragStore'

export function RetrievePage() {
  const nav = useNavigate()
  const { canRetrieve, retrieve, retrieval } = useRag()
  const [query, setQuery] = useState('')
  const [k, setK] = useState(5)
  const [busy, setBusy] = useState(false)

  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-3">Retrieve</h2>
        <form
          className="row gy-2 gx-2 align-items-end"
          onSubmit={async (e) => { e.preventDefault(); if (!canRetrieve) return; setBusy(true); await retrieve(query, k); setBusy(false) }}
        >
          <div className="col-12 col-md-6">
            <label className="form-label">Query</label>
            <input className="form-control" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask something…" />
          </div>
          <div className="col-auto">
            <label className="form-label">Top-K</label>
            <input className="form-control" type="number" value={k} onChange={(e) => setK(parseInt(e.target.value || '0') || 0)} />
          </div>
          <div className="col-auto">
            <button disabled={!canRetrieve || busy} className="btn btn-primary" type="submit">{busy ? 'Searching…' : 'Search'}</button>
          </div>
        </form>

        {retrieval && (
          <div className="mt-3">
            <h3 className="h6">Top results</h3>
            <ol>
              {retrieval.map((r: any) => (
                <li key={`${r.doc_id}-${r.chunk_index}`} className="mb-2">
                  <div className="small text-muted">score={r.score.toFixed(3)} doc={r.doc_id} idx={r.chunk_index}</div>
                  <pre className="bg-light p-2 rounded" style={{ whiteSpace: 'pre-wrap' }}>{r.content}</pre>
                </li>
              ))}
            </ol>
            <button className="btn btn-success btn-sm" onClick={() => nav('/generate')}>Continue to Generate</button>
          </div>
        )}
      </div>
    </div>
  )
}
