import React, { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";
import Editor from "@monaco-editor/react";

interface MetadataOptions {
  include_id: boolean;
  include_page_range: boolean;
  include_spans: boolean;
}

export function UploadPage() {
  const nav = useNavigate();
  const {
    upload,
    convert,
    uploadJson,
    updateJsonData,
    jsonData,
    fileName,
    docId,
    reset,
    setDocId,
  } = useRag();
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [converting, setConverting] = useState(false);
  const [convertError, setConvertError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedJson, setEditedJson] = useState<string>("");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const [metadataOptions, setMetadataOptions] = useState<MetadataOptions>({
    include_id: true,
    include_page_range: true,
    include_spans: true,
  });

  const isPDF = (f: File) =>
    f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf");

  const isJSON = (f: File) =>
    f.type === "application/json" || f.name.toLowerCase().endsWith(".json");

  const convertMultiple = async (
    files: File[],
    metadataOptions: MetadataOptions
  ) => {
    // 只處理PDF文件的多文件轉換
    const formData = new FormData();
    files.forEach((file, index) => {
      formData.append(`files`, file);
    });
    formData.append("metadata_options", JSON.stringify(metadataOptions));

    const base = import.meta.env.VITE_API_BASE_URL || "/api";
    const response = await fetch(`${base}/convert-multiple`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = "多文件轉換失敗";
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorData.detail || errorMessage;
      } catch (e) {
        // 如果響應不是JSON格式，使用狀態文本
        errorMessage = `${response.status} ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }

    const result = await response.json();

    if (result.doc_id) {
      // 直接返回結果
      setDocId(result.doc_id);
      updateJsonData(result.metadata);
      return;
    }

    if (result.task_id) {
      // 異步任務，需要輪詢狀態
      await pollConvertStatus(result.task_id);
    }
  };

  const pollConvertStatus = async (taskId: string) => {
    const maxAttempts = 120; // 最多等待2分鐘
    let attempts = 0;

    while (attempts < maxAttempts) {
      try {
        const base = import.meta.env.VITE_API_BASE_URL || "/api";
        const response = await fetch(
          `${base}/convert-multiple/status/${taskId}`
        );

        if (!response.ok) {
          let errorMessage = "獲取轉換狀態失敗";
          try {
            const errorData = await response.json();
            errorMessage = errorData.error || errorData.detail || errorMessage;
          } catch (e) {
            errorMessage = `${response.status} ${response.statusText}`;
          }
          throw new Error(errorMessage);
        }

        const status = await response.json();

        if (status.status === "completed") {
          setDocId(status.result.doc_id);
          updateJsonData(status.result.metadata);
          break;
        } else if (status.status === "failed") {
          throw new Error(status.error || "多文件轉換失敗");
        }

        // 等待1秒後重試
        await new Promise((resolve) => setTimeout(resolve, 1000));
        attempts++;
      } catch (error) {
        if (attempts >= maxAttempts - 1) {
          throw error;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
        attempts++;
      }
    }

    if (attempts >= maxAttempts) {
      throw new Error("多文件轉換超時");
    }
  };

  const handleEditJson = () => {
    if (jsonData) {
      setEditedJson(JSON.stringify(jsonData, null, 2));
      setIsEditing(true);
      setJsonError(null);
    }
  };

  const handleSaveJson = () => {
    try {
      const parsed = JSON.parse(editedJson);
      // 更新ragStore中的jsonData
      updateJsonData(parsed);
      setIsEditing(false);
      setJsonError(null);
      alert("JSON已保存！");
    } catch (error) {
      setJsonError("JSON格式錯誤: " + (error as Error).message);
    }
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditedJson("");
    setJsonError(null);
  };

  const handleDownloadJson = () => {
    if (jsonData) {
      const dataStr = JSON.stringify(jsonData, null, 2);
      const dataUri =
        "data:application/json;charset=utf-8," + encodeURIComponent(dataStr);

      const exportFileDefaultName = fileName
        ? fileName.replace(".pdf", ".json")
        : "document.json";

      const linkElement = document.createElement("a");
      linkElement.setAttribute("href", dataUri);
      linkElement.setAttribute("download", exportFileDefaultName);
      linkElement.click();
    }
  };

  return (
    <div className="row g-3">
      {/* Left: Upload Area */}
      <div className="col-12 col-md-6">
        <div className="card h-100">
          <div className="card-body">
            <h2 className="h5 mb-3">Upload</h2>
            <input
              ref={inputRef}
              className="form-control"
              type="file"
              accept="application/pdf,.pdf,application/json,.json"
              multiple
              onChange={async (e) => {
                const files = Array.from(e.target.files || []);
                const pdfFiles = files.filter(isPDF);
                const jsonFiles = files.filter(isJSON);

                if (jsonFiles.length > 0) {
                  // JSON文件：只設置選中文件，不自動上傳
                  setSelectedFiles(jsonFiles);
                } else if (pdfFiles.length > 0) {
                  // PDF文件：設置選中文件（恢復原本邏輯）
                  setSelectedFiles(pdfFiles);
                } else {
                  setSelectedFiles([]);
                }
              }}
            />

            {selectedFiles.length > 0 && (
              <div className="mt-3 d-flex align-items-center justify-content-between">
                <div className="text-truncate" style={{ maxWidth: "60%" }}>
                  <span className="badge text-bg-light me-2">Selected</span>
                  <span title={selectedFiles.map((f) => f.name).join(", ")}>
                    {selectedFiles.length === 1
                      ? selectedFiles[0].name
                      : `${selectedFiles.length} 個文件`}
                  </span>
                </div>
                <div className="d-flex gap-2">
                  {selectedFiles.length > 0 && (
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={async () => {
                        if (selectedFiles.length === 0) return;
                        setConverting(true);
                        setConvertError(null);
                        try {
                          if (selectedFiles.length === 1) {
                            const file = selectedFiles[0];
                            if (isJSON(file)) {
                              await uploadJson(file);
                            } else {
                              await convert(file, metadataOptions);
                            }
                          } else {
                            // 多文件處理
                            const allPDF = selectedFiles.every(isPDF);
                            const allJSON = selectedFiles.every(isJSON);

                            if (allJSON) {
                              throw new Error(
                                "多個JSON文件上傳，請一次只上傳一個JSON文件"
                              );
                            } else if (allPDF) {
                              await convertMultiple(
                                selectedFiles,
                                metadataOptions
                              );
                            } else {
                              throw new Error(
                                "不能混合上傳不同類型的文件，請分別上傳PDF或JSON文件"
                              );
                            }
                          }
                        } catch (error) {
                          console.error("Convert failed:", error);
                          setConvertError(
                            error instanceof Error
                              ? error.message
                              : "轉換失敗，請重試"
                          );
                        } finally {
                          setConverting(false);
                        }
                      }}
                      disabled={converting}
                    >
                      {converting
                        ? "處理中..."
                        : selectedFiles.length === 1
                        ? selectedFiles[0] && isJSON(selectedFiles[0])
                          ? "上傳JSON"
                          : "Confirm Convert"
                        : `轉換 ${selectedFiles.length} 個PDF`}
                    </button>
                  )}
                  {(jsonData || selectedFiles.length > 0) && (
                    <button
                      className="btn btn-outline-secondary btn-sm"
                      onClick={() => {
                        reset();
                        setSelectedFiles([]);
                        setConvertError(null);
                        if (inputRef.current) inputRef.current.value = "";
                      }}
                    >
                      Re-upload
                    </button>
                  )}
                </div>
              </div>
            )}

            {docId && (
              <p className="text-muted small mt-2 mb-0">
                doc_id: <code>{docId}</code>
              </p>
            )}

            {/* Metadata Options - 只對PDF文件顯示 */}
            {selectedFiles.length > 0 && selectedFiles.every(isPDF) && (
              <div className="mt-3">
                <h6 className="mb-2">Metadata 選項</h6>
                <div className="row g-2">
                  <div className="col-6">
                    <div className="form-check form-check-sm">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="include_id"
                        checked={metadataOptions.include_id}
                        onChange={(e) =>
                          setMetadataOptions((prev) => ({
                            ...prev,
                            include_id: e.target.checked,
                          }))
                        }
                      />
                      <label className="form-check-label" htmlFor="include_id">
                        ID (唯一識別碼)
                      </label>
                    </div>
                  </div>
                  <div className="col-6">
                    <div className="form-check form-check-sm">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="include_page_range"
                        checked={metadataOptions.include_page_range}
                        onChange={(e) =>
                          setMetadataOptions((prev) => ({
                            ...prev,
                            include_page_range: e.target.checked,
                          }))
                        }
                      />
                      <label
                        className="form-check-label"
                        htmlFor="include_page_range"
                      >
                        頁碼範圍
                      </label>
                    </div>
                  </div>
                  <div className="col-6">
                    <div className="form-check form-check-sm">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="include_spans"
                        checked={metadataOptions.include_spans}
                        onChange={(e) =>
                          setMetadataOptions((prev) => ({
                            ...prev,
                            include_spans: e.target.checked,
                          }))
                        }
                      />
                      <label
                        className="form-check-label"
                        htmlFor="include_spans"
                      >
                        文字片段範圍
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right: JSON Preview */}
      <div className="col-12 col-md-6">
        <div className="card h-100">
          <div className="card-body">
            <h2 className="h5 mb-3">JSON Preview</h2>
            {!jsonData && !converting && !convertError && (
              <p className="text-muted mb-0">
                Select a PDF on the left and click "Confirm Convert" to preview
                structured JSON here.
              </p>
            )}
            {converting && (
              <div className="d-flex align-items-center gap-2 text-secondary">
                <div
                  className="spinner-border spinner-border-sm"
                  role="status"
                />
                <span>Converting PDF to JSON…</span>
              </div>
            )}
            {convertError && (
              <div className="alert alert-danger" role="alert">
                <strong>轉換失敗:</strong> {convertError}
              </div>
            )}
            {jsonError && (
              <div className="alert alert-danger" role="alert">
                <strong>JSON錯誤:</strong> {jsonError}
              </div>
            )}
            {!converting && jsonData && !isEditing && (
              <div>
                <pre
                  className="bg-light p-2 rounded"
                  style={{
                    maxHeight: 360,
                    overflow: "auto",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {JSON.stringify(jsonData, null, 2)}
                </pre>
                <div className="d-flex gap-2">
                  <button
                    className="btn btn-outline-primary btn-sm"
                    onClick={handleEditJson}
                  >
                    <i className="bi bi-pencil"></i> Edit JSON
                  </button>
                  <a
                    className="btn btn-outline-secondary btn-sm"
                    href={`data:text/json;charset=utf-8,${encodeURIComponent(
                      JSON.stringify(jsonData, null, 2)
                    )}`}
                    download={`${(fileName || "document").replace(
                      /\.pdf$/i,
                      ""
                    )}.json`}
                  >
                    <i className="bi bi-download"></i> Download JSON
                  </a>
                  <button
                    className="btn btn-success btn-sm"
                    onClick={() => nav("/chunk")}
                  >
                    Continue to Chunk
                  </button>
                </div>
              </div>
            )}
            {isEditing && (
              <div>
                <div
                  style={{
                    height: "400px",
                    border: "1px solid #dee2e6",
                    borderRadius: "0.375rem",
                  }}
                >
                  <Editor
                    height="100%"
                    defaultLanguage="json"
                    value={editedJson}
                    onChange={(value) => setEditedJson(value || "")}
                    options={{
                      minimap: { enabled: false },
                      scrollBeyondLastLine: false,
                      fontSize: 14,
                      lineNumbers: "on",
                      roundedSelection: false,
                      scrollbar: {
                        vertical: "auto",
                        horizontal: "auto",
                      },
                      automaticLayout: true,
                      formatOnPaste: true,
                      formatOnType: true,
                    }}
                    theme="vs-light"
                  />
                </div>
                <div className="d-flex gap-2 mt-2">
                  <button
                    className="btn btn-success btn-sm"
                    onClick={handleSaveJson}
                  >
                    <i className="bi bi-check"></i> Save Changes
                  </button>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={handleCancelEdit}
                  >
                    <i className="bi bi-x"></i> Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
