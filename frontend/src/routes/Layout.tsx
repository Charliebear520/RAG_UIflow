import React from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'

const stages = [
  { key: 'upload', label: 'Upload', path: '/' },
  { key: 'chunk', label: 'Chunk', path: '/chunk' },
  { key: 'embed', label: 'Embed', path: '/embed' },
  { key: 'retrieve', label: 'Retrieve', path: '/retrieve' },
  { key: 'generate', label: 'Generate', path: '/generate' },
]

export function Layout() {
  const loc = useLocation()

  const activeIndex = stages.findIndex(s => s.path === loc.pathname || (s.path === '/' && loc.pathname === '/'))

  return (
    <div className="container py-3">
      <header className="mb-3">
        <h1 className="h3 mb-3">RAG Visualizer</h1>
        <div className="d-flex gap-2 flex-wrap align-items-center">
          {stages.map((s, i) => (
            <StageBadge key={s.key} label={s.label} active={i === activeIndex} />
          ))}
        </div>
      </header>
      <Outlet />
    </div>
  )
}

function StageBadge({ label, active }: { label: string; active?: boolean }) {
  return (
    <span className={`badge ${active ? 'text-bg-primary' : 'text-bg-secondary'}`}>{label}</span>
  )
}
