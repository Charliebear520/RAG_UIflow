import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useRag } from '../lib/ragStore'

export function EmbedPage() {
  const nav = useNavigate()
  const { canEmbed, embed, embedProvider } = useRag()
  const [busy, setBusy] = useState(false)

  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-3">Embed</h2>
        <button disabled={!canEmbed || busy} className="btn btn-primary" onClick={async () => { setBusy(true); await embed(); setBusy(false) }}>
          {busy ? 'Embedding…' : 'Compute embeddings'}
        </button>
        {embedProvider && (
          <div className="mt-3">
            Provider: <code>{embedProvider}</code>
            <div>
              <button className="btn btn-success btn-sm mt-2" onClick={() => nav('/retrieve')}>Continue to Retrieve</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
