import React, { useState } from "react";

interface QAMappingDetailsProps {
  mappingResults: Record<string, any>;
}

export const QAMappingDetails: React.FC<QAMappingDetailsProps> = ({
  mappingResults,
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set());
  const [showJsonViewer, setShowJsonViewer] = useState(false);
  const [selectedConfigForJson, setSelectedConfigForJson] = useState<
    string | null
  >(null);
  const [editableJson, setEditableJson] = useState<string>("");
  const [isEditingJson, setIsEditingJson] = useState(false);

  const toggleChunkExpansion = (qaIndex: number, configIndex: number) => {
    const key = `${configIndex}-${qaIndex}`;
    const newExpanded = new Set(expandedChunks);
    if (newExpanded.has(key)) {
      newExpanded.delete(key);
    } else {
      newExpanded.add(key);
    }
    setExpandedChunks(newExpanded);
  };

  // JSON處理函數
  const generateMappedQASetJson = (configId: string) => {
    const result = mappingResults[configId];
    if (!result?.mapped_qa_set) return [];

    return result.mapped_qa_set.map((qaItem: any) => ({
      query: qaItem.query,
      snippets: qaItem.snippets || [],
      label: qaItem.label,
      answer: qaItem.answer || "",
      relevant_chunks: qaItem.relevant_chunks || [],
    }));
  };

  const handleViewJson = (configId: string) => {
    const jsonData = generateMappedQASetJson(configId);
    setEditableJson(JSON.stringify(jsonData, null, 2));
    setSelectedConfigForJson(configId);
    setShowJsonViewer(true);
    setIsEditingJson(false);
  };

  const handleEditJson = () => {
    setIsEditingJson(true);
  };

  const handleSaveJson = () => {
    try {
      const parsedJson = JSON.parse(editableJson);
      // 這裡可以添加保存邏輯，比如更新mappingResults
      console.log("保存的JSON數據:", parsedJson);
      setIsEditingJson(false);
      alert("JSON已保存！");
    } catch (error) {
      alert("JSON格式錯誤，請檢查語法！");
    }
  };

  const handleDownloadJson = () => {
    if (!selectedConfigForJson) return;

    const jsonData = generateMappedQASetJson(selectedConfigForJson);
    const jsonString = JSON.stringify(jsonData, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mapped_qa_set_${selectedConfigForJson}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mt-4">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h6 className="mb-0">映射後的QA Set詳細內容</h6>
        <button
          className="btn btn-outline-primary btn-sm"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? (
            <>
              <i className="bi bi-eye-slash me-1"></i>
              隱藏映射詳情
            </>
          ) : (
            <>
              <i className="bi bi-eye me-1"></i>
              查看映射詳情
            </>
          )}
        </button>
      </div>

      {showDetails && (
        <div className="accordion" id="mappedQAAccordion">
          {Object.entries(mappingResults).map(
            ([configId, result]: [string, any], configIndex) => (
              <div key={configId} className="accordion-item">
                <h2 className="accordion-header">
                  <div
                    className="accordion-button collapsed d-flex justify-content-between align-items-center"
                    data-bs-toggle="collapse"
                    data-bs-target={`#mappedQA-${configIndex}`}
                    aria-expanded="false"
                    aria-controls={`mappedQA-${configIndex}`}
                  >
                    <span>
                      {result.strategy} - 配置 {configIndex + 1}(
                      {result.config.chunk_size} 字符,{" "}
                      {(result.config.overlap_ratio * 100).toFixed(0)}% 重疊)
                    </span>
                    <div
                      className="d-flex gap-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        className="btn btn-outline-primary btn-sm"
                        onClick={() => handleViewJson(configId)}
                        title="查看JSON"
                      >
                        <i className="bi bi-file-text"></i>
                      </button>
                      <button
                        className="btn btn-outline-success btn-sm"
                        onClick={() => {
                          setSelectedConfigForJson(configId);
                          handleDownloadJson();
                        }}
                        title="下載JSON"
                      >
                        <i className="bi bi-download"></i>
                      </button>
                    </div>
                  </div>
                </h2>
                <div
                  id={`mappedQA-${configIndex}`}
                  className="accordion-collapse collapse"
                  data-bs-parent="#mappedQAAccordion"
                >
                  <div className="accordion-body">
                    <div className="mb-3">
                      <h6 className="text-primary">配置統計</h6>
                      <div className="row g-2">
                        <div className="col-md-3">
                          <div className="card bg-light">
                            <div className="card-body text-center p-2">
                              <h6 className="card-title text-primary mb-1">
                                {result.chunk_count}
                              </h6>
                              <small className="card-text">分塊數量</small>
                            </div>
                          </div>
                        </div>
                        <div className="col-md-3">
                          <div className="card bg-light">
                            <div className="card-body text-center p-2">
                              <h6 className="card-title text-success mb-1">
                                {result.mapping_stats.questions_with_chunks}
                              </h6>
                              <small className="card-text">有映射的問題</small>
                            </div>
                          </div>
                        </div>
                        <div className="col-md-3">
                          <div className="card bg-success bg-opacity-10">
                            <div className="card-body text-center p-2">
                              <h6 className="card-title text-success mb-1">
                                {result.mapping_stats.mapping_success_rate?.toFixed(
                                  1
                                ) || "0.0"}
                                %
                              </h6>
                              <small className="card-text">映射成功率</small>
                            </div>
                          </div>
                        </div>
                        <div className="col-md-3">
                          <div className="card bg-light">
                            <div className="card-body text-center p-2">
                              <h6 className="card-title text-info mb-1">
                                {result.mapping_stats.positive_questions}
                              </h6>
                              <small className="card-text">正例問題</small>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <h6 className="text-primary">映射後的QA項目</h6>
                    <div className="table-responsive">
                      <table className="table table-sm table-bordered">
                        <thead className="table-light">
                          <tr>
                            <th style={{ width: "5%" }}>#</th>
                            <th style={{ width: "35%" }}>問題</th>
                            <th style={{ width: "10%" }}>標籤</th>
                            <th style={{ width: "25%" }}>答案</th>
                            <th style={{ width: "10%" }}>相關chunks</th>
                            <th style={{ width: "15%" }}>原始spans</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.mapped_qa_set?.map(
                            (qaItem: any, qaIndex: number) => (
                              <tr key={qaIndex}>
                                <td className="text-center">
                                  <span className="badge bg-secondary">
                                    {qaIndex + 1}
                                  </span>
                                </td>
                                <td>
                                  <div className="small">
                                    {qaItem.query?.substring(0, 100)}
                                    {qaItem.query?.length > 100 && "..."}
                                  </div>
                                </td>
                                <td className="text-center">
                                  <span
                                    className={`badge ${
                                      qaItem.label?.toLowerCase() === "yes"
                                        ? "bg-success"
                                        : qaItem.label?.toLowerCase() === "no"
                                        ? "bg-danger"
                                        : "bg-secondary"
                                    }`}
                                  >
                                    {qaItem.label}
                                  </span>
                                </td>
                                <td>
                                  <div className="small">
                                    {qaItem.answer?.substring(0, 80)}
                                    {qaItem.answer?.length > 80 && "..."}
                                  </div>
                                </td>
                                <td className="text-center">
                                  {qaItem.relevant_chunks?.length > 0 ? (
                                    <div>
                                      <button
                                        className="btn btn-sm btn-outline-primary"
                                        onClick={(e) => {
                                          e.preventDefault();
                                          e.stopPropagation();
                                          toggleChunkExpansion(
                                            qaIndex,
                                            configIndex
                                          );
                                        }}
                                      >
                                        <span className="badge bg-primary">
                                          {qaItem.relevant_chunks.length}
                                        </span>
                                        <i
                                          className={`bi ms-1 ${
                                            expandedChunks.has(
                                              `${configIndex}-${qaIndex}`
                                            )
                                              ? "bi-chevron-up"
                                              : "bi-chevron-down"
                                          }`}
                                        ></i>
                                      </button>
                                      <div className="mt-1">
                                        <small className="text-muted">
                                          {qaItem.relevant_chunks.join(", ")}
                                        </small>
                                      </div>
                                      {expandedChunks.has(
                                        `${configIndex}-${qaIndex}`
                                      ) && (
                                        <div className="mt-2">
                                          {qaItem.relevant_chunks.map(
                                            (
                                              chunkId: string,
                                              chunkIdx: number
                                            ) => {
                                              const chunkInfo =
                                                result.chunks_with_span?.find(
                                                  (c: any) =>
                                                    c.chunk_id === chunkId
                                                );
                                              if (!chunkInfo) return null;
                                              return (
                                                <div
                                                  key={chunkIdx}
                                                  className="card mt-1"
                                                >
                                                  <div className="card-header p-1">
                                                    <small className="fw-bold">
                                                      {chunkId} [
                                                      {chunkInfo.span.start}-
                                                      {chunkInfo.span.end}]
                                                    </small>
                                                  </div>
                                                  <div className="card-body p-1">
                                                    <pre
                                                      className="small mb-0"
                                                      style={{
                                                        whiteSpace: "pre-wrap",
                                                        maxHeight: "100px",
                                                        overflow: "auto",
                                                        backgroundColor:
                                                          "#f8f9fa",
                                                        padding: "4px",
                                                        borderRadius: "2px",
                                                      }}
                                                    >
                                                      {chunkInfo.content}
                                                    </pre>

                                                    {/* 顯示對應的法條JSON spans */}
                                                    {chunkInfo.metadata
                                                      ?.overlapping_law_spans
                                                      ?.length > 0 && (
                                                      <div className="mt-2">
                                                        <small className="text-success fw-bold">
                                                          法條JSON spans:
                                                        </small>
                                                        <div className="mt-1">
                                                          {chunkInfo.metadata.overlapping_law_spans
                                                            .slice(0, 2)
                                                            .map(
                                                              (
                                                                lawSpan: any,
                                                                lawIndex: number
                                                              ) => (
                                                                <div
                                                                  key={lawIndex}
                                                                  className="small bg-success bg-opacity-10 p-1 rounded mb-1"
                                                                >
                                                                  <div className="text-success fw-bold">
                                                                    {
                                                                      lawSpan.article_name
                                                                    }
                                                                  </div>
                                                                  <div className="text-muted">
                                                                    [
                                                                    {
                                                                      lawSpan.start_char
                                                                    }
                                                                    -
                                                                    {
                                                                      lawSpan.end_char
                                                                    }
                                                                    ] (重疊:{" "}
                                                                    {(
                                                                      lawSpan.overlap_ratio *
                                                                      100
                                                                    ).toFixed(
                                                                      1
                                                                    )}
                                                                    %)
                                                                  </div>
                                                                </div>
                                                              )
                                                            )}
                                                          {chunkInfo.metadata
                                                            .overlapping_law_spans
                                                            .length > 2 && (
                                                            <small className="text-muted">
                                                              +
                                                              {chunkInfo
                                                                .metadata
                                                                .overlapping_law_spans
                                                                .length -
                                                                2}{" "}
                                                              個...
                                                            </small>
                                                          )}
                                                        </div>
                                                      </div>
                                                    )}
                                                  </div>
                                                </div>
                                              );
                                            }
                                          )}
                                        </div>
                                      )}
                                    </div>
                                  ) : (
                                    <span className="badge bg-secondary">
                                      0
                                    </span>
                                  )}
                                </td>
                                <td>
                                  {qaItem.spans?.length > 0 ? (
                                    <div className="small">
                                      {qaItem.spans.map(
                                        (span: any, spanIndex: number) => (
                                          <div
                                            key={spanIndex}
                                            className="text-muted"
                                          >
                                            [{span.start_char}-{span.end_char}]
                                          </div>
                                        )
                                      )}
                                    </div>
                                  ) : (
                                    <span className="text-muted small">無</span>
                                  )}
                                </td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>

                    {/* 相關chunks詳細信息 */}
                    <div className="mt-3">
                      <h6 className="text-primary">相關chunks詳細信息</h6>
                      <div className="row">
                        {result.chunks_with_span?.map(
                          (chunkInfo: any, chunkIndex: number) => (
                            <div key={chunkIndex} className="col-md-6 mb-3">
                              <div className="card">
                                <div className="card-header p-2">
                                  <h6 className="mb-0">
                                    Chunk {chunkInfo.chunk_id}
                                    <span className="badge bg-info ms-2">
                                      [{chunkInfo.span.start}-
                                      {chunkInfo.span.end}]
                                    </span>
                                  </h6>
                                </div>
                                <div className="card-body p-2">
                                  <pre
                                    className="small mb-0"
                                    style={{
                                      whiteSpace: "pre-wrap",
                                      maxHeight: "150px",
                                      overflow: "auto",
                                      backgroundColor: "#f8f9fa",
                                      padding: "8px",
                                      borderRadius: "4px",
                                    }}
                                  >
                                    {chunkInfo.content}
                                  </pre>
                                </div>
                              </div>
                            </div>
                          )
                        )}
                      </div>
                      {result.chunks_with_span?.length > 0 && (
                        <div className="text-center">
                          <small className="text-muted">
                            總共 {result.chunks_with_span.length} 個chunks
                          </small>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )
          )}
        </div>
      )}

      {/* JSON查看器Modal */}
      {showJsonViewer && (
        <div
          className="modal show d-block"
          style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
        >
          <div className="modal-dialog modal-xl">
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title">
                  QA映射結果JSON - {selectedConfigForJson}
                </h5>
                <button
                  type="button"
                  className="btn-close"
                  onClick={() => setShowJsonViewer(false)}
                ></button>
              </div>
              <div className="modal-body">
                <div className="mb-3">
                  <div className="d-flex gap-2">
                    {!isEditingJson && (
                      <button
                        className="btn btn-outline-primary btn-sm"
                        onClick={handleEditJson}
                      >
                        <i className="bi bi-pencil me-1"></i>
                        編輯
                      </button>
                    )}
                    {isEditingJson && (
                      <button
                        className="btn btn-success btn-sm"
                        onClick={handleSaveJson}
                      >
                        <i className="bi bi-check me-1"></i>
                        保存
                      </button>
                    )}
                    {isEditingJson && (
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => setIsEditingJson(false)}
                      >
                        <i className="bi bi-x me-1"></i>
                        取消
                      </button>
                    )}
                    <button
                      className="btn btn-outline-success btn-sm"
                      onClick={handleDownloadJson}
                    >
                      <i className="bi bi-download me-1"></i>
                      下載
                    </button>
                  </div>
                </div>

                <div
                  className="border rounded p-3"
                  style={{ backgroundColor: "#f8f9fa" }}
                >
                  {isEditingJson ? (
                    <textarea
                      className="form-control"
                      value={editableJson}
                      onChange={(e) => setEditableJson(e.target.value)}
                      rows={20}
                      style={{
                        fontFamily: "monospace",
                        fontSize: "14px",
                        border: "none",
                        backgroundColor: "transparent",
                        resize: "vertical",
                      }}
                    />
                  ) : (
                    <pre
                      style={{
                        margin: 0,
                        fontFamily: "monospace",
                        fontSize: "14px",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        maxHeight: "500px",
                        overflow: "auto",
                      }}
                    >
                      {editableJson}
                    </pre>
                  )}
                </div>

                <div className="mt-3">
                  <small className="text-muted">
                    <strong>格式說明：</strong>
                    <ul className="mb-0 mt-1">
                      <li>
                        <code>query</code>: 問題內容
                      </li>
                      <li>
                        <code>snippets</code>: 原始span信息（如果有的話）
                      </li>
                      <li>
                        <code>label</code>: 標籤（Yes/No）
                      </li>
                      <li>
                        <code>answer</code>: 答案內容
                      </li>
                      <li>
                        <code>relevant_chunks</code>: 映射到的chunk ID列表
                      </li>
                    </ul>
                  </small>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
