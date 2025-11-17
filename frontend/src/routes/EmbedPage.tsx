import React, { useState, useEffect, useCallback } from "react";
import { useRag } from "../lib/ragStore";
import { api } from "../lib/api";

// 實驗組定義
const EXPERIMENTAL_GROUPS = {
  group_a: {
    name: "A組：僅條文層 (Baseline)",
    description: "僅使用基本單元層（條文）",
    levels: ["basic_unit"],
    research_purpose: "傳統平面法的表現，作為基線對照組",
  },
  group_b: {
    name: "B組：條文+章節結構",
    description: "基本單元層級層 + 基本單元層（章、節、編 + 條文）",
    levels: ["document_component", "basic_unit_hierarchy", "basic_unit"],
    research_purpose: "評估結構分組的嵌入是否能更好地捕捉廣泛主題",
  },
  group_c: {
    name: "C組：條文+細節層次",
    description: "基本單元層 + 基本單元組成層 + 列舉層（條文 + 項 + 款目）",
    levels: ["basic_unit", "basic_unit_component", "enumeration"],
    research_purpose: "評估細節化層次對列舉式規定的精確度增益",
  },
  group_d: {
    name: "D組：完整多層次ML-RAG",
    description: "章、節、條文、項、款、目層級",
    levels: [
      "document_component",
      "basic_unit_hierarchy",
      "basic_unit",
      "basic_unit_component",
      "enumeration",
    ],
    research_purpose: "作為最佳效能的對比組，評估完整多層次方法的綜合表現",
  },
  group_e: {
    name: "E組：LLM章節導向 + 細節檢索",
    description:
      "先以 LLM 章節路由鎖定章節，再在章節下的條／項／款／目層級執行微觀檢索",
    levels: ["basic_unit", "basic_unit_component", "enumeration"],
    research_purpose:
      "評估 LLM 先鎖章後做細節 embedding 的路由策略是否優於傳統多層檢索",
  },
};

type ExperimentalGroup = "group_a" | "group_b" | "group_c" | "group_d" | "group_e";

type SectionSummaryStatus = {
  chunk_id: string;
  title: string;
  chapter_title?: string;
  has_summary: boolean;
  summary?: string;
  last_updated?: string;
};

type ChapterSummaryStatus = {
  chunk_id: string;
  title: string;
  has_summary: boolean;
  summary?: string;
  last_updated?: string;
  sections: SectionSummaryStatus[];
};

type SummaryStatusResponse = {
  doc_id: string;
  doc_name: string;
  document_chunks: Array<{
    chunk_id: string;
    title: string;
    has_summary: boolean;
    summary?: string;
    last_updated?: string;
  }>;
  chapters: ChapterSummaryStatus[];
  orphan_sections: SectionSummaryStatus[];
  stats: {
    total_chapters: number;
    total_sections: number;
    summarized_chapters: number;
    summarized_sections: number;
  };
};

export function EmbedPage() {
  const {
    canEmbed,
    docId,
    multiLevelEmbed,
    embedProvider,
    embedModel,
    embedDimension,
  } = useRag();
  const [busy, setBusy] = useState(false);
  const [selectedGroup, setSelectedGroup] =
    useState<ExperimentalGroup>("group_a");
  const [summaryStatus, setSummaryStatus] = useState<SummaryStatusResponse | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryActionLoading, setSummaryActionLoading] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [summaryMessage, setSummaryMessage] = useState<string | null>(null);
  const [chunkSummaryLoading, setChunkSummaryLoading] = useState<string | null>(null);
  const [groupEQuery, setGroupEQuery] = useState("");
  const [groupERouteResult, setGroupERouteResult] = useState<any | null>(null);
  const [groupERouting, setGroupERouting] = useState(false);
  const [groupERouteError, setGroupERouteError] = useState<string | null>(null);
  const [groupEChapterSelections, setGroupEChapterSelections] = useState<
    Record<string, boolean>
  >({});
  const [groupEConfirmed, setGroupEConfirmed] = useState(false);

  const fetchSummaryStatus = useCallback(async () => {
    if (!docId) {
      setSummaryStatus(null);
      return;
    }
    setSummaryLoading(true);
    setSummaryError(null);
    try {
      const data = await api.getChapterSummaryStatus({ doc_id: docId });
      setSummaryStatus(data);
    } catch (error) {
      console.error("載入章節摘要狀態失敗:", error);
      setSummaryError(
        error instanceof Error ? error.message : "載入章節摘要狀態失敗"
      );
    } finally {
      setSummaryLoading(false);
    }
  }, [docId]);

  const handleGenerateAllSummaries = useCallback(async () => {
    if (!docId) {
      setSummaryError("請先完成上傳並取得文檔 ID");
      return;
    }
    setSummaryActionLoading(true);
    setSummaryMessage(null);
    setSummaryError(null);
    try {
      const res = await api.summarizeChapters({
        doc_id: docId,
        include_sections: true,
      });
      setSummaryMessage(
        `法規 ${res.documents_updated || 0} / 章 ${res.chapters_updated || 0} / 節 ${res.sections_updated || 0}`
      );
      await fetchSummaryStatus();
    } catch (error) {
      console.error("摘要生成失敗:", error);
      setSummaryError(
        error instanceof Error ? error.message : "摘要生成失敗"
      );
    } finally {
      setSummaryActionLoading(false);
    }
  }, [docId, fetchSummaryStatus]);

  const handleGenerateChunkSummary = useCallback(
    async (chunkId: string) => {
      if (!docId) {
        setSummaryError("請先完成上傳並取得文檔 ID");
        return;
      }
      setChunkSummaryLoading(chunkId);
      setSummaryError(null);
      try {
        await api.summarizeChapters({
          doc_id: docId,
          target_chunk_ids: [chunkId],
          include_sections: true,
        });
        await fetchSummaryStatus();
      } catch (error) {
        console.error("章節摘要生成失敗:", error);
        setSummaryError(
          error instanceof Error ? error.message : "章節摘要生成失敗"
        );
      } finally {
        setChunkSummaryLoading(null);
      }
    },
    [docId, fetchSummaryStatus]
  );

  useEffect(() => {
    if (selectedGroup === "group_e" && docId) {
      fetchSummaryStatus();
    } else if (selectedGroup !== "group_e") {
      setSummaryStatus(null);
      setSummaryMessage(null);
      setSummaryError(null);
    }
    setGroupERouteResult(null);
    setGroupEChapterSelections({});
    setGroupEConfirmed(false);
    setGroupERouteError(null);
    setGroupEQuery("");
  }, [selectedGroup, docId, fetchSummaryStatus]);

  useEffect(() => {
    setChunkSummaryLoading(null);
  }, [docId]);

  const groupESummaryStats = summaryStatus?.stats;
  const groupESummaryMissing =
    !!groupESummaryStats &&
    (groupESummaryStats.total_chapters > groupESummaryStats.summarized_chapters ||
      groupESummaryStats.total_sections > groupESummaryStats.summarized_sections);
  const groupESummaryReady =
    !!groupESummaryStats &&
    groupESummaryStats.total_chapters > 0 &&
    groupESummaryStats.total_chapters === groupESummaryStats.summarized_chapters;

  const handleGroupERoute = useCallback(
    async (event?: React.FormEvent<HTMLFormElement>) => {
      if (event) {
        event.preventDefault();
      }
      if (!docId) {
        setGroupERouteError("請先完成上傳並取得文檔 ID");
        return;
      }
      if (!groupESummaryReady) {
        setGroupERouteError("請先完成章節摘要生成，再執行章節路由");
        return;
      }
      if (!groupEQuery.trim()) {
        setGroupERouteError("請輸入查詢內容");
        return;
      }
      setGroupERouting(true);
      setGroupERouteError(null);
      setGroupERouteResult(null);
      setGroupEChapterSelections({});
      setGroupEConfirmed(false);
      try {
        const res = await api.routeGroupEChapters({
          doc_id: docId,
          query: groupEQuery.trim(),
        });
        const rawDetails = res.llm_stage?.selection_details || [];
        const keyedDetails = rawDetails.map((detail: any, idx: number) => ({
          ...detail,
          selection_key:
            detail.chunk_id ||
            detail.chapter_key ||
            detail.chapter_title ||
            `chapter_${idx}`,
        }));
        if (res.llm_stage) {
          res.llm_stage.selection_details = keyedDetails;
        } else {
          res.llm_stage = { selection_details: keyedDetails };
        }
        const initialSelections: Record<string, boolean> = {};
        keyedDetails.forEach((detail: any) => {
          if (detail.selection_key) {
            initialSelections[detail.selection_key] = true;
          }
        });
        setGroupERouteResult(res);
        setGroupEChapterSelections(initialSelections);
      } catch (error) {
        console.error("章節路由失敗:", error);
        setGroupERouteError(
          error instanceof Error ? error.message : "章節路由失敗"
        );
      } finally {
        setGroupERouting(false);
      }
    },
    [docId, groupEQuery, groupESummaryReady]
  );

  const toggleGroupEChapterSelection = useCallback((key: string) => {
    setGroupEChapterSelections((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
    setGroupEConfirmed(false);
  }, []);

  const handleSelectAllChapters = useCallback((checked: boolean) => {
    setGroupEChapterSelections((prev) => {
      const next: Record<string, boolean> = {};
      Object.keys(prev).forEach((key) => {
        next[key] = checked;
      });
      return next;
    });
    setGroupEConfirmed(false);
  }, []);

  const groupEHasSelection = Object.values(groupEChapterSelections).some(
    (selected) => selected
  );

  const getGroupEFiltersPayload = useCallback(() => {
    if (!docId || !groupERouteResult) {
      return null;
    }
    const details: any[] = groupERouteResult.llm_stage?.selection_details || [];
    const chapterChunkIds: string[] = [];
    const chapterTitles: string[] = [];
    const sectionChunkIds: string[] = [];
    const sectionTitles: string[] = [];
    details.forEach((detail: any, idx: number) => {
      const key =
        detail.selection_key ||
        detail.chunk_id ||
        detail.chapter_key ||
        detail.chapter_title ||
        `chapter_${idx}`;
      if (!groupEChapterSelections[key]) {
        return;
      }
      if (detail.chunk_id) {
        chapterChunkIds.push(detail.chunk_id);
      }
      if (detail.chapter_title) {
        chapterTitles.push(detail.chapter_title);
      }
      (detail.sections || []).forEach((section: any) => {
        if (section.chunk_id) {
          sectionChunkIds.push(section.chunk_id);
        }
        if (section.section_title) {
          sectionTitles.push(section.section_title);
        }
      });
    });
    if (
      chapterChunkIds.length === 0 &&
      chapterTitles.length === 0
    ) {
      return null;
    }
    const unique = (arr: string[]) => Array.from(new Set(arr));
    return {
      doc_id: docId,
      chapter_chunk_ids: unique(chapterChunkIds),
      chapter_titles: unique(chapterTitles),
      section_chunk_ids: unique(sectionChunkIds),
      section_titles: unique(sectionTitles),
    };
  }, [docId, groupERouteResult, groupEChapterSelections]);

  const handleEmbed = useCallback(async () => {
    setBusy(true);
    try {
      let options: { doc_ids?: string[]; group_e_filters?: any[] } | undefined;
      if (selectedGroup === "group_e") {
        if (!docId) {
          setGroupERouteError("請先完成上傳並取得文檔 ID");
          return;
        }
        if (!groupERouteResult) {
          setGroupERouteError("請先執行章節路由並確認章節");
          return;
        }
        const filters = getGroupEFiltersPayload();
        if (!filters) {
          setGroupERouteError("請至少勾選一個章節再執行 Embedding");
          return;
        }
        options = {
          doc_ids: [docId],
          group_e_filters: [filters],
        };
      }
      await multiLevelEmbed([selectedGroup], options);
    } catch (error) {
      console.error("Embedding failed:", error);
      setSummaryError(
        error instanceof Error ? error.message : "Embedding 失敗"
      );
    } finally {
      setBusy(false);
    }
  }, [
    selectedGroup,
    docId,
    groupERouteResult,
    getGroupEFiltersPayload,
    multiLevelEmbed,
  ]);


  return (
    <div className="card">
      <div className="card-body">
        <h2 className="h5 mb-4">Embedding 設置</h2>

        {!canEmbed ? (
          <div className="alert alert-warning">
            <i className="bi bi-exclamation-triangle me-2"></i>
            請先完成文檔上傳和分塊處理
          </div>
        ) : (
          <>
            {/* 實驗組選擇 */}
            <div className="mb-4">
              <h5>選擇實驗組</h5>
              <p className="text-muted small mb-3">
                請選擇要生成的實驗組embedding，用於後續的對比實驗：
              </p>
              <div className="row">
                {Object.entries(EXPERIMENTAL_GROUPS).map(([key, group]) => (
                  <div key={key} className="col-md-6 mb-3">
                    <div
                      className={`card ${
                        selectedGroup === key
                          ? "border-success bg-light"
                          : "border-light"
                      }`}
                    >
                      <div className="card-body">
                        <div className="form-check">
                          <input
                            className="form-check-input"
                            type="radio"
                            name="experimentalGroup"
                            id={key}
                            value={key}
                            checked={selectedGroup === key}
                            onChange={(e) =>
                              setSelectedGroup(
                                e.target.value as ExperimentalGroup
                              )
                            }
                          />
                          <label className="form-check-label" htmlFor={key}>
                            <strong>{group.name}</strong>
                          </label>
                        </div>
                        <p className="card-text small text-muted mt-2">
                          {group.description}
                        </p>
                        <div className="small">
                          <strong>包含層次:</strong> {group.levels.join(", ")}
                        </div>
                        <div className="small text-muted">
                          <em>{group.research_purpose}</em>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* 選中實驗組的詳細信息 */}
              <div className="alert alert-info">
                <h6 className="alert-heading">
                  選中的實驗組：{EXPERIMENTAL_GROUPS[selectedGroup].name}
                </h6>
                <p className="mb-1">
                  <strong>描述：</strong>
                  {EXPERIMENTAL_GROUPS[selectedGroup].description}
                </p>
                <p className="mb-1">
                  <strong>包含層次：</strong>
                  {EXPERIMENTAL_GROUPS[selectedGroup].levels.join(", ")}
                </p>
                <p className="mb-0">
                  <strong>研究目的：</strong>
                  {EXPERIMENTAL_GROUPS[selectedGroup].research_purpose}
                </p>
              </div>
            </div>

            {selectedGroup === "group_e" && (
              <div className="card border-info mb-4">
                <div className="card-header bg-info text-white">
                  <h5 className="mb-0">實驗組E：章節摘要檢查</h5>
                  <small>
                    確認 document / document_component 層已有摘要，再執行細節層 Embedding
                  </small>
                </div>
                <div className="card-body">
                  {!docId ? (
                    <div className="alert alert-warning mb-0">
                      請先於上傳頁取得文檔 ID 並完成多層級分塊。
                    </div>
                  ) : (
                    <>
                      <div className="d-flex flex-wrap gap-2 mb-3">
                        <button
                          type="button"
                          className="btn btn-outline-primary btn-sm"
                          onClick={fetchSummaryStatus}
                          disabled={summaryLoading}
                        >
                          {summaryLoading ? "載入中..." : "重新整理摘要狀態"}
                        </button>
                        <button
                          type="button"
                          className="btn btn-primary btn-sm"
                          onClick={handleGenerateAllSummaries}
                          disabled={summaryActionLoading}
                        >
                          {summaryActionLoading
                            ? "LLM 摘要生成中..."
                            : "一鍵生成章 / 節摘要"}
                        </button>
                      </div>

                      {summaryMessage && (
                        <div className="alert alert-success py-2">{summaryMessage}</div>
                      )}
                      {summaryError && (
                        <div className="alert alert-danger py-2">{summaryError}</div>
                      )}

                      {summaryStatus ? (
                        <>
                          <div className="alert alert-info small">
                            <div>
                              <strong>章：</strong>
                              {summaryStatus.stats.total_chapters}（已完成{" "}
                              {summaryStatus.stats.summarized_chapters}）
                            </div>
                            <div>
                              <strong>節：</strong>
                              {summaryStatus.stats.total_sections}（已完成{" "}
                              {summaryStatus.stats.summarized_sections}）
                            </div>
                            {summaryStatus.stats.total_chapters === 0 && (
                              <div className="text-warning mt-2">
                                尚未偵測到章節分塊，請確認已完成多層級分塊。
                              </div>
                            )}
                            {groupESummaryMissing && summaryStatus.stats.total_chapters > 0 && (
                              <div className="text-danger mt-2">
                                尚有章或節未生成摘要，建議先補齊再執行實驗組E的 Embedding。
                              </div>
                            )}
                          </div>

                          {summaryStatus.chapters.length > 0 ? (
                            summaryStatus.chapters.map((chapter) => (
                              <div key={chapter.chunk_id} className="card mb-3">
                                <div className="card-header d-flex flex-wrap justify-content-between align-items-center gap-2">
                                  <div>
                                    <strong>{chapter.title || "未命名章節"}</strong>
                                    <div className="small text-muted">
                                      chunk_id: <code>{chapter.chunk_id}</code>
                                    </div>
                                  </div>
                                  <div className="d-flex align-items-center gap-2">
                                    <span
                                      className={`badge ${
                                        chapter.has_summary
                                          ? "bg-success"
                                          : "bg-warning text-dark"
                                      }`}
                                    >
                                      {chapter.has_summary ? "已生成" : "未生成"}
                                    </span>
                                    {!chapter.has_summary && (
                                      <button
                                        type="button"
                                        className="btn btn-outline-primary btn-sm"
                                        onClick={() => handleGenerateChunkSummary(chapter.chunk_id)}
                                        disabled={chunkSummaryLoading === chapter.chunk_id}
                                      >
                                        {chunkSummaryLoading === chapter.chunk_id
                                          ? "生成中..."
                                          : "生成章摘要"}
                                      </button>
                                    )}
                                  </div>
                                </div>
                                <div className="card-body">
                                  {chapter.summary ? (
                                    <p className="mb-0">{chapter.summary}</p>
                                  ) : (
                                    <p className="text-muted mb-0">尚未生成章節摘要。</p>
                                  )}
                                </div>
                                {chapter.sections && chapter.sections.length > 0 && (
                                  <div className="list-group list-group-flush">
                                    {chapter.sections.map((section) => (
                                      <div
                                        key={section.chunk_id}
                                        className="list-group-item d-flex flex-wrap justify-content-between align-items-center gap-2"
                                      >
                                        <div>
                                          <div className="fw-semibold">
                                            {section.title || "節"}
                                          </div>
                                          <small className="text-muted d-block">
                                            chunk_id: <code>{section.chunk_id}</code>
                                          </small>
                                          {section.summary ? (
                                            <p className="mb-0 small text-muted">
                                              {section.summary}
                                            </p>
                                          ) : (
                                            <p className="mb-0 small text-muted">
                                              尚未生成摘要。
                                            </p>
                                          )}
                                        </div>
                                        <div className="d-flex align-items-center gap-2">
                                          <span
                                            className={`badge ${
                                              section.has_summary
                                                ? "bg-success"
                                                : "bg-warning text-dark"
                                            }`}
                                          >
                                            {section.has_summary ? "已生成" : "未生成"}
                                          </span>
                                          {!section.has_summary && (
                                            <button
                                              type="button"
                                              className="btn btn-outline-secondary btn-sm"
                                              onClick={() =>
                                                handleGenerateChunkSummary(section.chunk_id)
                                              }
                                              disabled={chunkSummaryLoading === section.chunk_id}
                                            >
                                              {chunkSummaryLoading === section.chunk_id
                                                ? "生成中..."
                                                : "生成節摘要"}
                                            </button>
                                          )}
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            ))
                          ) : (
                            <p className="text-muted mb-0">
                              尚未偵測到章節分塊，請確認已完成多層級分塊。
                            </p>
                          )}

                          {summaryStatus.orphan_sections &&
                            summaryStatus.orphan_sections.length > 0 && (
                              <details className="mt-3">
                                <summary className="fw-semibold">
                                  未配對章節的節（{summaryStatus.orphan_sections.length}）
                                </summary>
                                <div className="list-group mt-2">
                                  {summaryStatus.orphan_sections.map((section) => (
                                    <div key={section.chunk_id} className="list-group-item">
                                      <div className="d-flex justify-content-between align-items-center">
                                        <div>
                                          <div className="fw-semibold">
                                            {section.title || "節"}
                                          </div>
                                          <small className="text-muted">
                                            chunk_id: <code>{section.chunk_id}</code>
                                          </small>
                                        </div>
                                        {!section.has_summary && (
                                          <button
                                            type="button"
                                            className="btn btn-outline-secondary btn-sm"
                                            onClick={() =>
                                              handleGenerateChunkSummary(section.chunk_id)
                                            }
                                            disabled={chunkSummaryLoading === section.chunk_id}
                                          >
                                            {chunkSummaryLoading === section.chunk_id
                                              ? "生成中..."
                                              : "生成節摘要"}
                                          </button>
                                        )}
                                      </div>
                                      {section.summary ? (
                                        <p className="mb-0 small text-muted">{section.summary}</p>
                                      ) : (
                                        <p className="mb-0 small text-muted">
                                          尚未生成摘要。
                                        </p>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </details>
                            )}
                          <hr className="my-4" />
                          <div>
                            <h6 className="fw-semibold">步驟 2：輸入查詢，請 LLM 鎖定章節</h6>
                            {groupERouteError && (
                              <div className="alert alert-danger py-2">
                                {groupERouteError}
                              </div>
                            )}
                            <form
                              className="row g-2 align-items-end mb-3"
                              onSubmit={handleGroupERoute}
                            >
                              <div className="col-12 col-md-8">
                                <label className="form-label fw-semibold">
                                  查詢問題
                                </label>
                                <input
                                  type="text"
                                  className="form-control"
                                  placeholder="例如：網拍平台若收到侵權通知該如何處理？"
                                  value={groupEQuery}
                                  onChange={(e) => setGroupEQuery(e.target.value)}
                                  disabled={groupERouting}
                                />
                              </div>
                              <div className="col-12 col-md-4">
                                <button
                                  type="submit"
                                  className="btn btn-outline-info w-100 mt-4 mt-md-0"
                                  disabled={!groupESummaryReady || groupERouting}
                                >
                                  {groupERouting ? (
                                    <>
                                      <span className="spinner-border spinner-border-sm me-2"></span>
                                      LLM 配對章節中...
                                    </>
                                  ) : (
                                    "產生章節建議"
                                  )}
                                </button>
                              </div>
                            </form>

                            {groupERouteResult && (
                              <>
                                {groupERouteResult.llm_stage?.thinking && (
                                  <div className="alert alert-secondary small">
                                    <strong>LLM 思考：</strong>
                                    <pre className="mb-0 bg-light p-2 rounded" style={{ whiteSpace: "pre-wrap" }}>
                                      {groupERouteResult.llm_stage.thinking}
                                    </pre>
                                  </div>
                                )}
                                {groupERouteResult.llm_stage?.selection_details?.length ? (
                                  <>
                                    <div className="d-flex flex-wrap gap-2 mb-2">
                                      <button
                                        type="button"
                                        className="btn btn-sm btn-outline-secondary"
                                        onClick={() => handleSelectAllChapters(true)}
                                      >
                                        全選
                                      </button>
                                      <button
                                        type="button"
                                        className="btn btn-sm btn-outline-secondary"
                                        onClick={() => handleSelectAllChapters(false)}
                                      >
                                        全部取消
                                      </button>
                                    </div>
                                    {groupERouteResult.llm_stage.selection_details.map(
                                      (chapter: any, idx: number) => {
                                        const selectionKey =
                                          chapter.selection_key ||
                                          chapter.chunk_id ||
                                          chapter.chapter_key ||
                                          chapter.chapter_title ||
                                          `chapter_${idx}`;
                                        const checkboxId = `group-e-chapter-${selectionKey}`;
                                        const isSelected =
                                          groupEChapterSelections[selectionKey] ?? false;
                                        return (
                                          <div key={selectionKey} className="card mb-2">
                                            <div className="card-header d-flex flex-wrap justify-content-between align-items-center gap-2">
                                              <div className="form-check">
                                                <input
                                                  className="form-check-input"
                                                  type="checkbox"
                                                  id={checkboxId}
                                                  checked={isSelected}
                                                  onChange={() =>
                                                    toggleGroupEChapterSelection(selectionKey)
                                                  }
                                                />
                                                <label className="form-check-label" htmlFor={checkboxId}>
                                                  <strong>{chapter.chapter_title || "章節"}</strong>{" "}
                                                  {chapter.chunk_id && (
                                                    <small className="text-muted">
                                                      (<code>{chapter.chunk_id}</code>)
                                                    </small>
                                                  )}
                                                </label>
                                              </div>
                                              {chapter.reason && (
                                                <span className="text-muted small">
                                                  理由：{chapter.reason}
                                                </span>
                                              )}
                                            </div>
                                            {chapter.sections && chapter.sections.length > 0 && (
                                              <div className="card-body">
                                                <div className="small text-muted mb-2">
                                                  相關節：
                                                </div>
                                                <ul className="mb-0 small">
                                                  {chapter.sections.map((section: any, sIdx: number) => (
                                                    <li key={`${selectionKey}-sec-${sIdx}`}>
                                                      {section.section_title || "節"}
                                                      {section.chunk_id && (
                                                        <small className="text-muted ms-1">
                                                          (<code>{section.chunk_id}</code>)
                                                        </small>
                                                      )}
                                                      {section.reason && (
                                                        <div className="text-muted">
                                                          理由：{section.reason}
                                                        </div>
                                                      )}
                                                    </li>
                                                  ))}
                                                </ul>
                                              </div>
                                            )}
                                          </div>
                                        );
                                      }
                                    )}
                                  </>
                                ) : (
                                  <div className="alert alert-warning">
                                    LLM 未提供章節建議，請重新輸入問題或檢查摘要是否完整。
                                  </div>
                                )}
                                <div className="form-check mt-3">
                                  <input
                                    className="form-check-input"
                                    type="checkbox"
                                    id="group-e-confirm"
                                    checked={groupEConfirmed && groupEHasSelection}
                                    disabled={!groupEHasSelection}
                                    onChange={(e) => setGroupEConfirmed(e.target.checked)}
                                  />
                                  <label className="form-check-label" htmlFor="group-e-confirm">
                                    我確認以上勾選的章節作為本次 Embedding 範圍
                                  </label>
                                </div>
                              </>
                            )}
                          </div>
                        </>
                      ) : (
                        <p className="text-muted mb-0">
                          {summaryLoading
                            ? "載入章節摘要狀態..."
                            : "尚未載入章節摘要狀態。"}
                        </p>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}

            {/* 執行Embedding */}
            <div className="mb-4">
              <button
                className="btn btn-primary"
                onClick={handleEmbed}
                disabled={
                  busy ||
                  (selectedGroup === "group_e" &&
                    (!groupESummaryReady ||
                      !groupERouteResult ||
                      !groupEHasSelection ||
                      !groupEConfirmed ||
                      groupERouting))
                }
              >
                {busy ? (
                  <>
                    <span
                      className="spinner-border spinner-border-sm me-2"
                      role="status"
                      aria-hidden="true"
                    ></span>
                    生成 Embedding 中...
                  </>
                ) : (
                  <>
                    <i className="bi bi-play-circle me-2"></i>
                    生成 {EXPERIMENTAL_GROUPS[selectedGroup].name} Embedding
                  </>
                )}
              </button>
            </div>

            {/* 當前狀態顯示 */}
            {embedProvider && (
              <div className="alert alert-success">
                <h6 className="alert-heading">Embedding 狀態</h6>
                <div className="row">
                  <div className="col-md-4">
                    <strong>Provider:</strong> {embedProvider}
                  </div>
                  <div className="col-md-4">
                    <strong>Model:</strong> {embedModel}
                  </div>
                  <div className="col-md-4">
                    <strong>Dimension:</strong> {embedDimension}
                  </div>
                </div>
              </div>
            )}

            {/* 測試按鈕 */}
            <div className="mt-3">
              <button
                className="btn btn-sm btn-outline-info me-2"
                onClick={async () => {
                  try {
                    const response = await fetch(
                      "/api/test-experimental-groups",
                      {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          experimental_groups: [selectedGroup],
                        }),
                      }
                    );
                    const result = await response.json();
                    console.log("🧪 實驗組測試結果:", result);
                    alert(
                      `實驗組測試完成！\n選中層次: ${result.selected_levels.join(
                        ", "
                      )}\n跳過層次: ${result.skipped_levels.join(
                        ", "
                      )}\n\n詳細結果請查看控制台`
                    );
                  } catch (error) {
                    console.error("測試失敗:", error);
                    alert("測試失敗，請檢查控制台");
                  }
                }}
              >
                🧪 測試實驗組層次選擇
              </button>

              <button
                className="btn btn-sm btn-outline-warning"
                onClick={async () => {
                  try {
                    const response = await fetch("/api/debug-store");
                    const result = await response.json();
                    console.log("🔍 Store狀態:", result);
                    alert(
                      `Store狀態檢查完成！\n多層次embedding: ${
                        result.has_multi_level_embeddings ? "是" : "否"
                      }\n可用層次: ${result.available_levels.join(
                        ", "
                      )}\n\n詳細結果請查看控制台`
                    );
                  } catch (error) {
                    console.error("檢查失敗:", error);
                    alert("檢查失敗，請檢查控制台");
                  }
                }}
              >
                🔍 檢查Embedding狀態
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
