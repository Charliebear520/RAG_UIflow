const base = import.meta.env.VITE_API_BASE_URL || '/api'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  async upload(file: File) {
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch(`${base}/upload`, { method: 'POST', body: fd })
    return json<any>(res)
  },
  async chunk(body: { doc_id: string; chunk_size: number; overlap: number }) {
    const res = await fetch(`${base}/chunk`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    return json<any>(res)
  },
  async embed(body?: { doc_ids?: string[] }) {
    const res = await fetch(`${base}/embed`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
    return json<any>(res)
  },
  async retrieve(body: { query: string; k: number }) {
    const res = await fetch(`${base}/retrieve`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    return json<any>(res)
  },
  async generate(body: { query: string; top_k: number }) {
    const res = await fetch(`${base}/generate`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    return json<any>(res)
  },
  async convert(file: File) {
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch(`${base}/convert`, { method: 'POST', body: fd })
    return json<any>(res)
  },
}
