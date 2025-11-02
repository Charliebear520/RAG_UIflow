import React, { useState, useEffect } from "react";
import { api } from "../lib/api";

interface EnhancedMetadata {
  legal_concepts?: Array<{
    concept_name: string;
    concept_type: string;
    legal_domain: string;
    importance_score: number;
    synonyms: string[];
    confidence: number;
  }>;
  semantic_keywords?: {
    primary_keywords: string[];
    keyword_weights: Record<string, number>;
  };
  article_type?: {
    article_type: string;
    confidence: number;
  };
  legal_domain?: {
    legal_domain: string;
    confidence: number;
  };
  chapter_section_type?: {
    chapter_section_type: string;
    type_description: string;
    confidence: number;
  };
  legal_relations?: Array<{
    relation_type: string;
    subject: string;
    object: string;
    relation: string;
    confidence: number;
  }>;
  query_intent_tags?: string[];
  semantic_similarity?: {
    core_concept_similarity: Record<string, number>;
    confidence: number;
  };
  enhancement_level?: string;
  is_article_level?: boolean;
  is_chapter_section_level?: boolean;
  inherited_from?: string;
  inheritance_type?: string;
}

interface MetadataViewerProps {
  onClose: () => void;
}

const MetadataViewer: React.FC<MetadataViewerProps> = ({ onClose }) => {
  const [enhancedMetadata, setEnhancedMetadata] = useState<
    Record<string, EnhancedMetadata>
  >({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null);
  const [editingMetadata, setEditingMetadata] =
    useState<EnhancedMetadata | null>(null);
  const [isEditing, setIsEditing] = useState(false);

  // 載入enhanced metadata
  const loadEnhancedMetadata = async () => {
    setLoading(true);
    setError(null);
    try {
      // 先檢查是否有enhanced metadata
      const response = await api.get("/enhanced-metadata-stats");
      if (response.enhanced_metadata_count === 0) {
        setError("尚未生成enhanced metadata，請先執行metadata生成");
        setLoading(false);
        return;
      }

      // 獲取enhanced metadata列表
      const metadataResponse = await api.get("/enhanced-metadata-list");
      setEnhancedMetadata(metadataResponse.enhanced_metadata);
    } catch (err) {
      console.error("載入enhanced metadata失敗:", err);
      setError(err instanceof Error ? err.message : "載入失敗");
    } finally {
      setLoading(false);
    }
  };

  // 生成enhanced metadata
  const generateEnhancedMetadata = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.post("/generate-enhanced-metadata", {});
      if (response.success) {
        await loadEnhancedMetadata(); // 重新載入
        alert(
          `Enhanced metadata生成完成！\n統計信息：\n- 總chunks: ${response.stats.total_chunks}\n- 條層級chunks: ${response.stats.article_level_chunks}\n- 章節層級chunks: ${response.stats.chapter_section_chunks}\n- 繼承chunks: ${response.stats.inherited_chunks}`
        );
      } else {
        setError(response.error || "生成失敗");
      }
    } catch (err) {
      console.error("生成enhanced metadata失敗:", err);
      setError(err instanceof Error ? err.message : "生成失敗");
    } finally {
      setLoading(false);
    }
  };

  // 保存編輯的metadata
  const saveEditedMetadata = async () => {
    if (!selectedChunkId || !editingMetadata) return;

    setLoading(true);
    try {
      await api.post("/update-enhanced-metadata", {
        chunk_id: selectedChunkId,
        enhanced_metadata: editingMetadata,
      });

      // 更新本地狀態
      setEnhancedMetadata((prev) => ({
        ...prev,
        [selectedChunkId]: editingMetadata,
      }));

      setIsEditing(false);
      setEditingMetadata(null);
      alert("Metadata更新成功！");
    } catch (err) {
      console.error("更新metadata失敗:", err);
      setError(err instanceof Error ? err.message : "更新失敗");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEnhancedMetadata();
  }, []);

  const renderMetadataField = (key: string, value: any, level: number = 0) => {
    const indent = "  ".repeat(level);

    if (value === null || value === undefined) {
      return null;
    }

    if (typeof value === "object" && !Array.isArray(value)) {
      return (
        <div key={key} style={{ marginLeft: `${level * 20}px` }}>
          <strong>
            {indent}
            {key}:
          </strong>
          {Object.entries(value).map(([subKey, subValue]) =>
            renderMetadataField(subKey, subValue, level + 1)
          )}
        </div>
      );
    }

    if (Array.isArray(value)) {
      return (
        <div key={key} style={{ marginLeft: `${level * 20}px` }}>
          <strong>
            {indent}
            {key}:
          </strong>
          {value.map((item, index) => (
            <div key={index} style={{ marginLeft: `${(level + 1) * 20}px` }}>
              {typeof item === "object" ? (
                Object.entries(item).map(([itemKey, itemValue]) =>
                  renderMetadataField(itemKey, itemValue, level + 2)
                )
              ) : (
                <span>{JSON.stringify(item)}</span>
              )}
            </div>
          ))}
        </div>
      );
    }

    return (
      <div key={key} style={{ marginLeft: `${level * 20}px` }}>
        <strong>
          {indent}
          {key}:
        </strong>{" "}
        {JSON.stringify(value)}
      </div>
    );
  };

  const renderEditableField = (
    key: string,
    value: any,
    path: string[] = []
  ) => {
    const fullPath = [...path, key];
    const pathStr = fullPath.join(".");

    if (typeof value === "string") {
      return (
        <div key={key} className="mb-2">
          <label className="form-label">{key}:</label>
          <input
            type="text"
            className="form-control"
            value={value}
            onChange={(e) => {
              const newValue = { ...editingMetadata };
              let current = newValue;
              for (let i = 0; i < fullPath.length - 1; i++) {
                if (!current[fullPath[i]]) current[fullPath[i]] = {};
                current = current[fullPath[i]];
              }
              current[fullPath[fullPath.length - 1]] = e.target.value;
              setEditingMetadata(newValue);
            }}
          />
        </div>
      );
    }

    if (typeof value === "number") {
      return (
        <div key={key} className="mb-2">
          <label className="form-label">{key}:</label>
          <input
            type="number"
            className="form-control"
            value={value}
            onChange={(e) => {
              const newValue = { ...editingMetadata };
              let current = newValue;
              for (let i = 0; i < fullPath.length - 1; i++) {
                if (!current[fullPath[i]]) current[fullPath[i]] = {};
                current = current[fullPath[i]];
              }
              current[fullPath[fullPath.length - 1]] = parseFloat(
                e.target.value
              );
              setEditingMetadata(newValue);
            }}
          />
        </div>
      );
    }

    if (Array.isArray(value)) {
      return (
        <div key={key} className="mb-3">
          <label className="form-label">{key}:</label>
          <textarea
            className="form-control"
            rows={3}
            value={JSON.stringify(value, null, 2)}
            onChange={(e) => {
              try {
                const parsed = JSON.parse(e.target.value);
                const newValue = { ...editingMetadata };
                let current = newValue;
                for (let i = 0; i < fullPath.length - 1; i++) {
                  if (!current[fullPath[i]]) current[fullPath[i]] = {};
                  current = current[fullPath[i]];
                }
                current[fullPath[fullPath.length - 1]] = parsed;
                setEditingMetadata(newValue);
              } catch (err) {
                // 忽略JSON解析錯誤
              }
            }}
          />
        </div>
      );
    }

    if (typeof value === "object") {
      return (
        <div key={key} className="mb-3">
          <h6>{key}:</h6>
          {Object.entries(value).map(([subKey, subValue]) =>
            renderEditableField(subKey, subValue, fullPath)
          )}
        </div>
      );
    }

    return null;
  };

  return (
    <div
      className="modal show d-block"
      style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
    >
      <div className="modal-dialog modal-xl">
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">Enhanced Metadata 查看與編輯</h5>
            <button
              type="button"
              className="btn-close"
              onClick={onClose}
            ></button>
          </div>

          <div className="modal-body">
            {error && <div className="alert alert-danger">{error}</div>}

            {loading && (
              <div className="text-center">
                <div className="spinner-border" role="status">
                  <span className="visually-hidden">載入中...</span>
                </div>
              </div>
            )}

            {!loading && !error && (
              <>
                <div className="d-flex justify-content-between align-items-center mb-3">
                  <div>
                    <button
                      className="btn btn-primary"
                      onClick={generateEnhancedMetadata}
                      disabled={loading}
                    >
                      生成 Enhanced Metadata
                    </button>
                  </div>
                  <div>
                    <span className="badge bg-info">
                      總計 {Object.keys(enhancedMetadata).length} 個chunks
                    </span>
                  </div>
                </div>

                <div className="row">
                  <div className="col-md-4">
                    <div className="list-group">
                      <div className="list-group-item list-group-item-action active">
                        <strong>Chunk 列表</strong>
                      </div>
                      {Object.entries(enhancedMetadata).map(
                        ([chunkId, metadata]) => (
                          <button
                            key={chunkId}
                            className={`list-group-item list-group-item-action ${
                              selectedChunkId === chunkId ? "active" : ""
                            }`}
                            onClick={() => {
                              setSelectedChunkId(chunkId);
                              setEditingMetadata(null);
                              setIsEditing(false);
                            }}
                          >
                            <div className="d-flex w-100 justify-content-between">
                              <small>{chunkId}</small>
                              <div>
                                {metadata.is_article_level && (
                                  <span className="badge bg-success me-1">
                                    條層級
                                  </span>
                                )}
                                {metadata.is_chapter_section_level && (
                                  <span className="badge bg-warning me-1">
                                    章節層級
                                  </span>
                                )}
                                {metadata.inherited_from && (
                                  <span className="badge bg-info">繼承</span>
                                )}
                              </div>
                            </div>
                          </button>
                        )
                      )}
                    </div>
                  </div>

                  <div className="col-md-8">
                    {selectedChunkId && enhancedMetadata[selectedChunkId] && (
                      <div>
                        <div className="d-flex justify-content-between align-items-center mb-3">
                          <h6>Chunk ID: {selectedChunkId}</h6>
                          <div>
                            {!isEditing ? (
                              <button
                                className="btn btn-sm btn-outline-primary"
                                onClick={() => {
                                  setEditingMetadata({
                                    ...enhancedMetadata[selectedChunkId],
                                  });
                                  setIsEditing(true);
                                }}
                              >
                                編輯
                              </button>
                            ) : (
                              <>
                                <button
                                  className="btn btn-sm btn-success me-2"
                                  onClick={saveEditedMetadata}
                                  disabled={loading}
                                >
                                  保存
                                </button>
                                <button
                                  className="btn btn-sm btn-secondary"
                                  onClick={() => {
                                    setIsEditing(false);
                                    setEditingMetadata(null);
                                  }}
                                >
                                  取消
                                </button>
                              </>
                            )}
                          </div>
                        </div>

                        <div
                          className="border rounded p-3"
                          style={{ maxHeight: "500px", overflowY: "auto" }}
                        >
                          {isEditing ? (
                            <form>
                              {Object.entries(editingMetadata || {}).map(
                                ([key, value]) =>
                                  renderEditableField(key, value)
                              )}
                            </form>
                          ) : (
                            <div className="metadata-display">
                              {Object.entries(
                                enhancedMetadata[selectedChunkId]
                              ).map(([key, value]) =>
                                renderMetadataField(key, value)
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>

          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
            >
              關閉
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetadataViewer;
