import React, { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useRag } from "../lib/ragStore";

export function UploadPage() {
  const nav = useNavigate();
  const { upload, convert, jsonData, fileName, docId, reset } = useRag();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [converting, setConverting] = useState(false);
  const [convertError, setConvertError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const isPDF = (f: File | null) =>
    !!f &&
    (f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));

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
                          await convert(selectedFile);
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
            {!converting && jsonData && (
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
                    Download JSON
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
          </div>
        </div>
      </div>
    </div>
  );
}
