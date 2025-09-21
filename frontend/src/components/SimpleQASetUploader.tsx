import React, { useState, useRef } from "react";

interface SimpleQASetUploaderProps {
  onFileUploaded: (file: File) => void;
}

export const SimpleQASetUploader: React.FC<SimpleQASetUploaderProps> = ({
  onFileUploaded,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith(".json")) {
        alert("請選擇JSON格式的QA set文件");
        return;
      }
      setSelectedFile(file);
      onFileUploaded(file);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">
          選擇QA Set文件 (JSON格式)
        </label>

        {/* 文件格式說明 */}
        <div className="alert alert-info mb-3">
          <h6>📋 QA Set文件格式說明</h6>
          <p className="mb-2">
            <strong>正確格式：</strong>包含問題答案對的JSON數組
          </p>
          <pre className="small bg-light p-2 rounded">
            {`[
  {
    "query": "問題內容",
    "label": "Yes/No",
    "answer": "答案內容",
    "spans": [{"start_char": 100, "end_char": 200}]
  }
]`}
          </pre>
          <p className="mb-0 small text-muted">
            <strong>注意：</strong>
            請使用QA目錄下的copyright.json等文件，不要使用corpus目錄下的法律文檔文件
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleFileSelect}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
          >
            選擇文件
          </button>
          {selectedFile && (
            <span className="text-sm text-gray-600">
              {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
            </span>
          )}
        </div>
      </div>

      {selectedFile && (
        <div className="alert alert-success">
          <h6>✅ QA Set文件已選擇</h6>
          <p className="mb-0">
            文件: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)}{" "}
            KB)
          </p>
        </div>
      )}
    </div>
  );
};
