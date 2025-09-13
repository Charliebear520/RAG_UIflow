import React, { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";
import Editor from "@monaco-editor/react";

interface MetadataOptions {
  include_id: boolean;
  include_page_range: boolean;
  include_keywords: boolean;
  include_cross_references: boolean;
  include_importance: boolean;
  include_length: boolean;
  include_extracted_entities: boolean;
  include_spans: boolean;
}

export function UploadPage() {
  const nav = useNavigate();
  const { upload, convert, updateJsonData, jsonData, fileName, docId, reset } =
    useRag();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [converting, setConverting] = useState(false);
  const [convertError, setConvertError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedJson, setEditedJson] = useState<string>("");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const [metadataOptions, setMetadataOptions] = useState<MetadataOptions>({
    include_id: true,
    include_page_range: true,
    include_keywords: true,
    include_cross_references: true,
    include_importance: true,
    include_length: true,
    include_extracted_entities: false,
    include_spans: true,
  });

  const isPDF = (f: File | null) =>
    !!f &&
    (f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));

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
              accept="application/pdf,.pdf"
              onChange={async (e) => {
                const f = e.target.files?.[0] || null;
                setSelectedFile(f);
                if (f) {
                  // Create backend doc for downstream steps
                  await upload(f);
                }
              }}
            />

            {selectedFile && (
              <div className="mt-3 d-flex align-items-center justify-content-between">
                <div className="text-truncate" style={{ maxWidth: "60%" }}>
                  <span className="badge text-bg-light me-2">Selected</span>
                  <span title={selectedFile.name}>{selectedFile.name}</span>
                </div>
                <div className="d-flex gap-2">
                  {isPDF(selectedFile) && (
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={async () => {
                        if (!selectedFile) return;
                        setConverting(true);
                        setConvertError(null);
                        try {
                          await convert(selectedFile, metadataOptions);
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
                    >
                      Confirm Convert
                    </button>
                  )}
                  {(jsonData || selectedFile) && (
                    <button
                      className="btn btn-outline-secondary btn-sm"
                      onClick={() => {
                        reset();
                        setSelectedFile(null);
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

            {/* Metadata Options */}
            {isPDF(selectedFile) && (
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
                        id="include_keywords"
                        checked={metadataOptions.include_keywords}
                        onChange={(e) =>
                          setMetadataOptions((prev) => ({
                            ...prev,
                            include_keywords: e.target.checked,
                          }))
                        }
                      />
                      <label
                        className="form-check-label"
                        htmlFor="include_keywords"
                      >
                        關鍵詞
                      </label>
                    </div>
                  </div>
                  <div className="col-6">
                    <div className="form-check form-check-sm">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="include_cross_references"
                        checked={metadataOptions.include_cross_references}
                        onChange={(e) =>
                          setMetadataOptions((prev) => ({
                            ...prev,
                            include_cross_references: e.target.checked,
                          }))
                        }
                      />
                      <label
                        className="form-check-label"
                        htmlFor="include_cross_references"
                      >
                        交叉引用
                      </label>
                    </div>
                  </div>
                  <div className="col-6">
                    <div className="form-check form-check-sm">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="include_importance"
                        checked={metadataOptions.include_importance}
                        onChange={(e) =>
                          setMetadataOptions((prev) => ({
                            ...prev,
                            include_importance: e.target.checked,
                          }))
                        }
                      />
                      <label
                        className="form-check-label"
                        htmlFor="include_importance"
                      >
                        重要性權重
                      </label>
                    </div>
                  </div>
                  <div className="col-6">
                    <div className="form-check form-check-sm">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        id="include_length"
                        checked={metadataOptions.include_length}
                        onChange={(e) =>
                          setMetadataOptions((prev) => ({
                            ...prev,
                            include_length: e.target.checked,
                          }))
                        }
                      />
                      <label
                        className="form-check-label"
                        htmlFor="include_length"
                      >
                        內容長度
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
