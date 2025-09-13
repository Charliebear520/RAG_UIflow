import React from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";

const stages = [
  { key: "upload", label: "Upload", path: "/" },
  { key: "chunk", label: "Chunk", path: "/chunk" },
  { key: "embed", label: "Embed", path: "/embed" },
  { key: "retrieve", label: "Retrieve", path: "/retrieve" },
  { key: "generate", label: "Generate", path: "/generate" },
];

export function Layout() {
  const loc = useLocation();
  const navigate = useNavigate();

  const activeIndex = stages.findIndex(
    (s) => s.path === loc.pathname || (s.path === "/" && loc.pathname === "/")
  );

  return (
    <div className="container py-3">
      <header className="mb-3">
        <h1 className="h3 mb-3">RAG Visualizer</h1>
        <div className="d-flex gap-2 flex-wrap align-items-center">
          {stages.map((s, i) => (
            <StageBadge
              key={s.key}
              label={s.label}
              active={i === activeIndex}
              onClick={() => navigate(s.path)}
            />
          ))}
        </div>
      </header>
      <Outlet />
    </div>
  );
}

function StageBadge({
  label,
  active,
  onClick,
}: {
  label: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <span
      className={`badge ${active ? "text-bg-primary" : "text-bg-secondary"} ${
        !active ? "cursor-pointer" : ""
      }`}
      onClick={!active ? onClick : undefined}
      style={{
        cursor: !active ? "pointer" : "default",
        transition: "all 0.2s ease",
      }}
      onMouseEnter={(e) => {
        if (!active) {
          e.currentTarget.style.transform = "scale(1.05)";
          e.currentTarget.style.opacity = "0.8";
        }
      }}
      onMouseLeave={(e) => {
        if (!active) {
          e.currentTarget.style.transform = "scale(1)";
          e.currentTarget.style.opacity = "1";
        }
      }}
    >
      {label}
    </span>
  );
}
