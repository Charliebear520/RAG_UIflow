import React, { useState, useRef } from "react";
import { api } from "../lib/api";

interface SimpleQASetUploaderProps {
  docId: string;
  onUploadComplete: (result: any) => void;
}

interface QASetItem {
  query: string;
  label: string;
  answer?: string;
  snippets?: Array<{
    file_path: string;
    span: [number, number];
  }>;
  spans?: Array<{
    start_char: number;
    end_char: number;
    text: string;
    page?: number;
    confidence?: number;
    found?: boolean;
  }>;
  relevant_chunks?: string[];
}

export const SimpleQASetUploader: React.FC<SimpleQASetUploaderProps> = ({
  docId,
  onUploadComplete,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith(".json")) {
        alert("請選擇JSON格式的QA set文件");
        return;
      }
      setSelectedFile(file);
      setUploadError(null);
      setUploadResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !docId) return;

    setIsUploading(true);
    setUploadError(null);

    try {
      // 直接映射模式
      const response = await api.mapQASet(selectedFile, docId);
      setUploadResult(response);
      onUploadComplete(response);
    } catch (error) {
      console.error("QA set upload error:", error);
      setUploadError(error instanceof Error ? error.message : "上傳失敗");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div>
      {/* QA Set Upload Section */}
      <div className="mb-4">
        <h6 className="text-info mb-2">
          <i className="bi bi-question-circle me-2"></i>
          QA數據集上傳
        </h6>
        <p className="text-muted small mb-2">
          上傳QA數據集，系統會自動將答案對應到法條JSON中的具體位置，並生成與chunking後格式一致的映射關係
        </p>

        <input
          ref={fileInputRef}
          className="form-control"
          type="file"
          accept="application/json,.json"
          onChange={handleFileSelect}
        />
        <div className="mt-2">
          <small className="text-muted">
            <i className="bi bi-info-circle me-1"></i>
            QA Set文件應包含 <code>query</code>、<code>label</code>、
            <code>answer</code> 字段，支持新的 <code>snippets</code> 格式：
            <br />
            <code>
              {
                '[{ "query": "...", "label": "Yes/No", "answer": "...", "snippets": [{"file_path": "copyright&tradmark.json", "span": [start, end]}] }]'
              }
            </code>
            <br />
            系統會自動將答案對應到法條JSON中的具體位置
          </small>
        </div>
      </div>

      {selectedFile && (
        <div className="mt-3 d-flex align-items-center justify-content-between">
          <div className="text-truncate" style={{ maxWidth: "60%" }}>
            <span className="badge text-bg-light me-2">Selected</span>
            <span title={selectedFile.name}>{selectedFile.name}</span>
          </div>
          <div className="d-flex gap-2">
            <button
              className="btn btn-info btn-sm"
              onClick={handleUpload}
              disabled={isUploading}
            >
              {isUploading ? (
                <>
                  <span
                    className="spinner-border spinner-border-sm me-2"
                    role="status"
                  ></span>
                  處理中...
                </>
              ) : (
                <>
                  <i className="bi bi-diagram-3 me-2"></i>
                  直接映射
                </>
              )}
            </button>
            <button
              className="btn btn-outline-secondary btn-sm"
              onClick={() => {
                setSelectedFile(null);
                setUploadError(null);
                setUploadResult(null);
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
            >
              Re-upload
            </button>
          </div>
        </div>
      )}

      {/* 上傳錯誤 */}
      {uploadError && (
        <div className="alert alert-danger mt-3" role="alert">
          <strong>上傳失敗:</strong> {uploadError}
        </div>
      )}

      {/* 上傳成功統計 */}
      {uploadResult && (
        <div className="mt-3">
          <div className="alert alert-success" role="alert">
            <h6 className="alert-heading">
              <i className="bi bi-check-circle me-2"></i>
              QA Set上傳成功！
            </h6>
            <div className="row">
              <div className="col-6">
                <small className="text-muted">
                  <strong>總配置數:</strong> {uploadResult.total_configs || 0}
                </small>
              </div>
              <div className="col-6">
                <small className="text-muted">
                  <strong>原始QA問題數:</strong>{" "}
                  {uploadResult.original_qa_set?.length || 0}
                </small>
              </div>
            </div>

            {/* 直接映射結果 */}
            {uploadResult.mapped_qa_set && (
              <div className="mt-2 pt-2 border-top">
                <h6 className="mb-2">
                  <i className="bi bi-check-circle me-2"></i>
                  映射結果
                </h6>
                <small className="text-muted">
                  <strong>映射後QA項目數:</strong>{" "}
                  {uploadResult.mapped_qa_set.length}
                </small>
              </div>
            )}

            {/* 映射統計信息 */}
            {uploadResult.conversion_stats && (
              <div className="mt-2 pt-2 border-top">
                <h6 className="mb-2">
                  <i className="bi bi-diagram-3 me-2"></i>
                  映射統計
                </h6>
                <div className="row">
                  <div className="col-4">
                    <small className="text-muted">
                      <strong>總項目數:</strong>{" "}
                      {uploadResult.conversion_stats.total_items || 0}
                    </small>
                  </div>
                  <div className="col-4">
                    <small className="text-muted">
                      <strong>有Snippets:</strong>{" "}
                      {uploadResult.conversion_stats.items_with_snippets || 0}
                    </small>
                  </div>
                  <div className="col-4">
                    <small className="text-muted">
                      <strong>有效Span:</strong>{" "}
                      {uploadResult.conversion_stats.items_with_valid_spans ||
                        0}
                    </small>
                  </div>
                </div>
                <div className="row mt-1">
                  <div className="col-4">
                    <small className="text-muted">
                      <strong>Snippet覆蓋率:</strong>{" "}
                      {uploadResult.conversion_stats.snippet_coverage
                        ? (
                            uploadResult.conversion_stats.snippet_coverage * 100
                          ).toFixed(1) + "%"
                        : "0%"}
                    </small>
                  </div>
                  <div className="col-4">
                    <small className="text-muted">
                      <strong>Span覆蓋率:</strong>{" "}
                      {uploadResult.conversion_stats.span_coverage
                        ? (
                            uploadResult.conversion_stats.span_coverage * 100
                          ).toFixed(1) + "%"
                        : "0%"}
                    </small>
                  </div>
                  <div className="col-4">
                    <small className="text-muted">
                      <strong>文件路徑覆蓋率:</strong>{" "}
                      {uploadResult.conversion_stats.file_path_coverage
                        ? (
                            uploadResult.conversion_stats.file_path_coverage *
                            100
                          ).toFixed(1) + "%"
                        : "0%"}
                    </small>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
