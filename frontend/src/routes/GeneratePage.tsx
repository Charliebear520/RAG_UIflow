import React, { useState } from 'react'
import { useRag } from '../lib/ragStore'

export function GeneratePage() {
  const { canGenerate, generate, answer, steps } = useRag()
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [busy, setBusy] = useState(false)

  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-3">Generate</h2>
        <form
          className="row gy-2 gx-2 align-items-end"
          onSubmit={async (e) => { e.preventDefault(); if (!canGenerate) return; setBusy(true); await generate(query, topK); setBusy(false) }}
        >
          <div className="col-12 col-md-6">
            <label className="form-label">Question</label>
            <input className="form-control" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="What do you want to know?" />
          </div>
          <div className="col-auto">
            <label className="form-label">Top-K</label>
            <input className="form-control" type="number" value={topK} onChange={(e) => setTopK(parseInt(e.target.value || '0') || 0)} />
          </div>
          <div className="col-auto">
            <button disabled={!canGenerate || busy} className="btn btn-primary" type="submit">{busy ? 'Generating…' : 'Generate'}</button>
          </div>
        </form>

        {answer && (
          <div className="mt-3">
            <h3 className="h6">Answer</h3>
            <pre className="bg-light p-2 rounded" style={{ whiteSpace: 'pre-wrap' }}>{answer}</pre>
          </div>
        )}
        {steps && (
          <div className="mt-2">
            <h3 className="h6">Reasoning steps</h3>
            <ol>
              {steps.map((s: any, i: number) => (
                <li key={i}><strong>{s.type}:</strong> {s.text}</li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </div>
  )
}
