import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";

interface EmbeddingDatabase {
  id: string;
  type: "standard" | "multi_level";
  name: string;
  provider: string;
  model: string;
  num_vectors: number;
  dimension: number;
  chunking_strategy: string;
  documents: Array<{
    filename: string;
    json_data: boolean;
  }>;
  created_at: string;
  level?: string;
  levels?: Array<{
    level: string;
    description: string;
    num_vectors: number;
  }>;
}

export function EmbeddingDatabaseList() {
  const navigate = useNavigate();
  const [databases, setDatabases] = useState<EmbeddingDatabase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<{
    show: boolean;
    database: EmbeddingDatabase | null;
  }>({
    show: false,
    database: null,
  });

  useEffect(() => {
    fetchDatabases();
  }, []);

  const fetchDatabases = async () => {
    try {
      setLoading(true);
      const data = await api.getEmbeddingDatabases();
      setDatabases(data);
      setError(null);
    } catch (err) {
      console.error("獲取embedding資料庫失敗:", err);
      setError(err instanceof Error ? err.message : "獲取資料失敗");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteClick = (database: EmbeddingDatabase) => {
    setDeleteConfirm({ show: true, database });
  };

  const handleDeleteConfirm = async () => {
    if (!deleteConfirm.database) return;

    const database = deleteConfirm.database;
    try {
      setDeletingId(database.id);
      console.log("開始刪除embedding資料庫:", database.id);

      const result = await api.deleteEmbeddingDatabase(database.id);
      console.log("刪除結果:", result);

      // 刪除成功，重新載入列表
      console.log("重新載入embedding資料庫列表...");
      await fetchDatabases();

      // 關閉確認對話框
      setDeleteConfirm({ show: false, database: null });
      console.log("刪除完成，對話框已關閉");
    } catch (err) {
      console.error("刪除embedding資料庫失敗:", err);
      alert(err instanceof Error ? err.message : "刪除失敗");
    } finally {
      setDeletingId(null);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteConfirm({ show: false, database: null });
  };

  const getChunkingStrategyDisplay = (strategy: string) => {
    const strategyMap: { [key: string]: string } = {
      basic: "基礎分塊",
      hierarchical: "多層次分塊",
      sliding_window: "滑動視窗分塊",
      legal_semantic: "法律語義分塊",
    };
    return strategyMap[strategy] || strategy;
  };

  const getEmbeddingProviderDisplay = (provider: string) => {
    const providerMap: { [key: string]: string } = {
      gemini: "Gemini Embedding",
      "bge-m3": "BGE-M3",
      openai: "OpenAI Embedding",
    };
    return providerMap[provider] || provider;
  };

  const getDocumentDisplay = (documents: EmbeddingDatabase["documents"]) => {
    if (documents.length === 0) return "無文檔";
    if (documents.length === 1) {
      const doc = documents[0];
      return `${doc.filename}${doc.json_data ? " (JSON)" : ""}`;
    }
    return `${documents.length} 個文檔`;
  };

  const handleUseDatabase = (database: EmbeddingDatabase) => {
    // 跳轉到Retrieve頁面，並攜帶資料庫信息
    navigate("/retrieve", {
      state: {
        selectedDatabase: database,
        message: `已選擇embedding資料庫: ${database.name}`,
      },
    });
  };

  if (loading) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="d-flex justify-content-center">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">載入中...</span>
            </div>
          </div>
          <p className="text-center mt-2">正在載入embedding資料庫...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="alert alert-danger" role="alert">
            <h6 className="alert-heading">載入失敗</h6>
            <p className="mb-0">{error}</p>
            <button
              className="btn btn-outline-danger btn-sm mt-2"
              onClick={fetchDatabases}
            >
              重新載入
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (databases.length === 0) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="text-center py-4">
            <i
              className="bi bi-database-x text-muted"
              style={{ fontSize: "3rem" }}
            ></i>
            <h5 className="mt-3 text-muted">尚無embedding資料庫</h5>
            <p className="text-muted">請先上傳文檔並進行分塊和embedding處理</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-body">
        <div className="d-flex justify-content-between align-items-center mb-3">
          <h5 className="card-title mb-0">
            <i className="bi bi-database me-2"></i>
            Embedding 資料庫
          </h5>
          <button
            className="btn btn-outline-primary btn-sm"
            onClick={fetchDatabases}
            title="重新載入"
          >
            <i className="bi bi-arrow-clockwise"></i>
          </button>
        </div>

        <div className="row g-3">
          {databases.map((database) => (
            <div key={database.id} className="col-12">
              <div className="card border">
                <div className="card-body">
                  <div className="row align-items-center">
                    <div className="col-md-8">
                      <h6 className="card-title mb-2">
                        {database.name}
                        {database.type === "multi_level" && (
                          <span className="badge bg-info ms-2">多層次</span>
                        )}
                      </h6>

                      <div className="row text-muted small mb-2">
                        <div className="col-sm-6">
                          <i className="bi bi-file-text me-1"></i>
                          <strong>文檔:</strong>{" "}
                          {getDocumentDisplay(database.documents)}
                        </div>
                        <div className="col-sm-6">
                          <i className="bi bi-scissors me-1"></i>
                          <strong>分塊方式:</strong>{" "}
                          {getChunkingStrategyDisplay(
                            database.chunking_strategy
                          )}
                        </div>
                        <div className="col-sm-6">
                          <i className="bi bi-cpu me-1"></i>
                          <strong>Embedding:</strong>{" "}
                          {getEmbeddingProviderDisplay(database.provider)}
                        </div>
                        <div className="col-sm-6">
                          <i className="bi bi-hash me-1"></i>
                          <strong>總向量數量:</strong>{" "}
                          {database.num_vectors.toLocaleString()}
                        </div>
                        {database.dimension > 0 && (
                          <div className="col-sm-6">
                            <i className="bi bi-vector-pen me-1"></i>
                            <strong>維度:</strong> {database.dimension}
                          </div>
                        )}
                      </div>

                      {/* 多層次embedding的詳細層次信息 */}
                      {database.levels && database.levels.length > 0 && (
                        <div className="mt-2 mb-2">
                          <div className="alert alert-light py-2 mb-0">
                            <div className="row text-muted small">
                              <div className="col-12 mb-1">
                                <i className="bi bi-layers me-1"></i>
                                <strong>包含層次:</strong>
                              </div>
                              {database.levels.map((levelInfo, index) => (
                                <div key={index} className="col-md-6 col-lg-4">
                                  <span className="badge bg-secondary me-1">
                                    {levelInfo.num_vectors}
                                  </span>
                                  <small>{levelInfo.description}</small>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}

                      <div className="text-muted small">
                        <i className="bi bi-clock me-1"></i>
                        創建時間:{" "}
                        {new Date(database.created_at).toLocaleString()}
                      </div>
                    </div>

                    <div className="col-md-4 text-end">
                      <div className="d-flex gap-2 justify-content-end">
                        <button
                          className="btn btn-success"
                          onClick={() => handleUseDatabase(database)}
                        >
                          <i className="bi bi-play-circle me-1"></i>
                          使用此資料庫
                        </button>
                        <button
                          className="btn btn-outline-danger"
                          onClick={() => handleDeleteClick(database)}
                          disabled={deletingId === database.id}
                          title="刪除此embedding資料庫"
                        >
                          {deletingId === database.id ? (
                            <i className="bi bi-hourglass-split"></i>
                          ) : (
                            <i className="bi bi-trash"></i>
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {databases.length > 0 && (
          <div className="mt-3 text-center">
            <small className="text-muted">
              點擊「使用此資料庫」將跳轉到檢索頁面進行不同策略的檢索與生成
            </small>
          </div>
        )}
      </div>

      {/* 刪除確認對話框 */}
      {deleteConfirm.show && deleteConfirm.database && (
        <div
          className="modal show d-block"
          style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
        >
          <div className="modal-dialog modal-dialog-centered">
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title">
                  <i className="bi bi-exclamation-triangle text-warning me-2"></i>
                  確認刪除
                </h5>
                <button
                  type="button"
                  className="btn-close"
                  onClick={handleDeleteCancel}
                  disabled={deletingId !== null}
                ></button>
              </div>
              <div className="modal-body">
                <p>您確定要刪除此embedding資料庫嗎？</p>
                <div className="alert alert-info">
                  <h6 className="alert-heading">資料庫信息：</h6>
                  <ul className="mb-0">
                    <li>
                      <strong>名稱:</strong> {deleteConfirm.database.name}
                    </li>
                    <li>
                      <strong>類型:</strong>{" "}
                      {deleteConfirm.database.type === "multi_level"
                        ? "多層次Embedding"
                        : "標準Embedding"}
                    </li>
                    <li>
                      <strong>總向量數量:</strong>{" "}
                      {deleteConfirm.database.num_vectors.toLocaleString()}
                    </li>
                    <li>
                      <strong>文檔:</strong>{" "}
                      {deleteConfirm.database.documents
                        .map((d) => d.filename)
                        .join(", ")}
                    </li>
                    {deleteConfirm.database.levels &&
                      deleteConfirm.database.levels.length > 0 && (
                        <li>
                          <strong>包含層次:</strong>{" "}
                          {deleteConfirm.database.levels.length} 個層次
                          <ul className="mt-1 mb-0">
                            {deleteConfirm.database.levels.map(
                              (levelInfo, index) => (
                                <li key={index} className="small">
                                  {levelInfo.description}:{" "}
                                  {levelInfo.num_vectors} 個向量
                                </li>
                              )
                            )}
                          </ul>
                        </li>
                      )}
                  </ul>
                </div>
                <p className="text-danger">
                  <i className="bi bi-exclamation-circle me-1"></i>
                  此操作無法復原，刪除後將無法進行檢索操作。
                </p>
              </div>
              <div className="modal-footer">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleDeleteCancel}
                  disabled={deletingId !== null}
                >
                  取消
                </button>
                <button
                  type="button"
                  className="btn btn-danger"
                  onClick={handleDeleteConfirm}
                  disabled={deletingId !== null}
                >
                  {deletingId ? (
                    <>
                      <i className="bi bi-hourglass-split me-1"></i>
                      刪除中...
                    </>
                  ) : (
                    <>
                      <i className="bi bi-trash me-1"></i>
                      確認刪除
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
