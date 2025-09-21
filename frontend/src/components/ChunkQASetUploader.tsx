import React, { useState, useRef } from "react";

interface ChunkQASetUploaderProps {
  onFileUploaded: (file: File) => void;
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

export const ChunkQASetUploader: React.FC<ChunkQASetUploaderProps> = ({
  onFileUploaded,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith(".json")) {
        setUploadError("請選擇JSON格式的QA set文件");
        return;
      }

      setIsValidating(true);
      setUploadError(null);

      try {
        // 驗證文件格式
        const text = await file.text();
        const parsedData = JSON.parse(text);

        let qaSet;

        // 檢查是否為映射結果格式（包含original_qa_set）
        if (
          parsedData &&
          typeof parsedData === "object" &&
          parsedData.original_qa_set
        ) {
          qaSet = parsedData.original_qa_set;
        }
        // 檢查是否為純粹的QA Set數組
        else if (Array.isArray(parsedData)) {
          qaSet = parsedData;
        }
        // 檢查是否為法條JSON格式
        else if (
          parsedData &&
          typeof parsedData === "object" &&
          parsedData.laws
        ) {
          setUploadError(
            "錯誤：您上傳的是法條JSON文件，請上傳QA Set文件。QA Set文件應該是一個包含問題答案對的數組格式。"
          );
          return;
        }
        // 其他格式錯誤
        else {
          setUploadError(
            `QA set文件格式錯誤：應該是QA Set數組或包含original_qa_set的映射結果，但得到的是 ${typeof parsedData}。請確認上傳的是正確的QA Set JSON文件。`
          );
          return;
        }

        // 確保提取的數據是數組
        if (!Array.isArray(qaSet)) {
          setUploadError("QA set數據格式錯誤：提取的數據不是數組格式");
          return;
        }

        if (qaSet.length === 0) {
          setUploadError("QA set文件格式錯誤：數組不能為空");
          return;
        }

        // 檢查第一個項目的必要字段
        const firstItem = qaSet[0];
        if (!firstItem.query || !firstItem.label) {
          setUploadError("QA set文件格式錯誤：缺少必要的字段（query, label）");
          return;
        }

        setSelectedFile(file);
        onFileUploaded(file);
      } catch (error) {
        setUploadError(
          "文件解析失敗：" +
            (error instanceof Error ? error.message : "未知錯誤")
        );
      } finally {
        setIsValidating(false);
      }
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
          上傳QA數據集文件，用於後續與分塊結果進行映射
        </p>
        <div className="alert alert-info small mb-3">
          <i className="bi bi-info-circle me-1"></i>
          <strong>支持格式：</strong>
          <br />• QA Set數組格式：
          <code>[{`{"query": "...", "label": "...", "answer": "..."}`}]</code>
          <br />• Upload頁面映射結果：
          <code>{`{"original_qa_set": [...]}`}</code>
          <br />• 請勿上傳法條JSON文件
        </div>

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
              className="btn btn-outline-secondary btn-sm"
              onClick={() => {
                setSelectedFile(null);
                setUploadError(null);
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
            >
              Re-upload
            </button>
          </div>
        </div>
      )}

      {/* 驗證中 */}
      {isValidating && (
        <div className="mt-3 d-flex align-items-center gap-2 text-secondary">
          <div className="spinner-border spinner-border-sm" role="status" />
          <span>正在驗證文件格式...</span>
        </div>
      )}

      {/* 上傳錯誤 */}
      {uploadError && (
        <div className="alert alert-danger mt-3" role="alert">
          <strong>上傳失敗:</strong> {uploadError}
        </div>
      )}
    </div>
  );
};
