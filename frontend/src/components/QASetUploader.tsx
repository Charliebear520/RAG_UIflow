import React, { useState, useRef } from "react";
import { api } from "../lib/api";
import { QAChunkMappingViewer } from "./QAChunkMappingViewer";

interface QASetUploaderProps {
  docId: string;
  onMappingComplete: (result: any) => void;
}

interface QASetItem {
  query: string;
  label: string;
  answer: string;
  snippets?: Array<{
    file_path: string;
    span: [number, number];
  }>;
  spans?: Array<{
    start_char: number;
    end_char: number;
    text: string;
  }>;
  relevant_chunks?: string[];
}

interface MappingConfig {
  chunk_size: number;
  overlap_ratio: number;
  overlap: number;
  strategy: string;
}

interface MappingResult {
  config: MappingConfig;
  chunks: string[];
  mapped_qa_set: QASetItem[];
  chunk_count: number;
}

export const QASetUploader: React.FC<QASetUploaderProps> = ({
  docId,
  onMappingComplete,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [chunkSizes, setChunkSizes] = useState<number[]>([300, 500, 800]);
  const [overlapRatios, setOverlapRatios] = useState<number[]>([0.0, 0.1, 0.2]);
  const [strategy, setStrategy] = useState<string>("fixed_size");
  const [isUploading, setIsUploading] = useState(false);
  const [mappingTaskId, setMappingTaskId] = useState<string | null>(null);
  const [mappingProgress, setMappingProgress] = useState(0);
  const [mappingError, setMappingError] = useState<string | null>(null);
  const [mappingResult, setMappingResult] = useState<any>(null);
  const [expandedItems, setExpandedItems] = useState<Set<number>>(new Set());
  const [showDetailedView, setShowDetailedView] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith(".json")) {
        alert("請選擇JSON格式的QA set文件");
        return;
      }
      setSelectedFile(file);
      setMappingError(null);
      setMappingResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !docId) return;

    setIsUploading(true);
    setMappingError(null);

    try {
      const response = await api.uploadQASet(
        selectedFile,
        docId,
        chunkSizes,
        overlapRatios,
        strategy
      );

      setMappingTaskId(response.task_id);

      // 開始輪詢映射狀態
      const pollMappingStatus = async () => {
        try {
          const statusResponse = await api.getQAMappingStatus(response.task_id);

          if (statusResponse.status === "completed") {
            const resultResponse = await api.getQAMappingResult(
              response.task_id
            );
            setMappingResult(resultResponse);
            onMappingComplete(resultResponse);
            setIsUploading(false);
            setMappingProgress(100);
          } else if (statusResponse.status === "failed") {
            setMappingError(statusResponse.error || "映射失敗");
            setIsUploading(false);
          } else if (statusResponse.status === "processing") {
            setMappingProgress(statusResponse.progress * 100);
            setTimeout(pollMappingStatus, 2000);
          }
        } catch (error) {
          setMappingError("獲取映射狀態失敗");
          setIsUploading(false);
        }
      };

      setTimeout(pollMappingStatus, 1000);
    } catch (error) {
      setMappingError("上傳失敗");
      setIsUploading(false);
    }
  };

  const toggleExpanded = (index: number) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedItems(newExpanded);
  };

  const renderQASetPreview = (
    qaSet: QASetItem[],
    title: string,
    chunksWithSpan?: any[]
  ) => {
    const displayCount = 3;
    const shouldShowMore = qaSet.length > displayCount;

    return (
      <div className="bg-white border rounded-lg p-4 mb-4">
        <h4 className="text-lg font-semibold mb-3">{title}</h4>
        <div className="space-y-3">
          {qaSet.slice(0, displayCount).map((item, index) => (
            <div key={index} className="border-l-4 border-blue-500 pl-4 py-2">
              <div className="flex items-center justify-between">
                <span className="font-medium text-sm">
                  {item.label === "Yes" ? "✅" : "❌"}{" "}
                  {item.query.substring(0, 80)}...
                </span>
                <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                  {item.relevant_chunks?.length || 0} chunks
                </span>
              </div>
              {item.relevant_chunks && item.relevant_chunks.length > 0 && (
                <div className="mt-2">
                  <div className="text-xs text-gray-600 mb-1">
                    相關chunks: {item.relevant_chunks.join(", ")}
                  </div>
                  {/* 顯示對應的chunk內容 */}
                  {chunksWithSpan && (
                    <div className="mt-2 space-y-2">
                      {item.relevant_chunks.map((chunkId, chunkIndex) => {
                        const chunkInfo = chunksWithSpan.find(
                          (c) => c.chunk_id === chunkId
                        );
                        if (!chunkInfo) return null;
                        return (
                          <div
                            key={chunkIndex}
                            className="bg-gray-50 p-2 rounded text-xs"
                          >
                            <div className="font-medium text-blue-600 mb-1">
                              {chunkId} [{chunkInfo.span.start}-
                              {chunkInfo.span.end}]
                            </div>
                            <div className="text-gray-700 max-h-20 overflow-y-auto">
                              {chunkInfo.content.substring(0, 200)}
                              {chunkInfo.content.length > 200 && "..."}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {shouldShowMore && (
            <div className="border-t pt-3">
              <button
                onClick={() => toggleExpanded(qaSet.length)}
                className="text-blue-600 hover:text-blue-800 text-sm font-medium"
              >
                {expandedItems.has(qaSet.length)
                  ? "收起"
                  : `顯示更多 (${qaSet.length - displayCount} 個問題)`}
              </button>

              {expandedItems.has(qaSet.length) && (
                <div className="mt-3 space-y-3">
                  {qaSet.slice(displayCount).map((item, index) => (
                    <div
                      key={displayCount + index}
                      className="border-l-4 border-blue-500 pl-4 py-2"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-sm">
                          {item.label === "Yes" ? "✅" : "❌"}{" "}
                          {item.query.substring(0, 80)}...
                        </span>
                        <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                          {item.relevant_chunks?.length || 0} chunks
                        </span>
                      </div>
                      {item.relevant_chunks &&
                        item.relevant_chunks.length > 0 && (
                          <div className="mt-1">
                            <span className="text-xs text-gray-600">
                              相關chunks: {item.relevant_chunks.join(", ")}
                            </span>
                          </div>
                        )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">QA Set上傳與Chunk映射</h3>

        {/* 文件上傳 */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            選擇QA Set文件 (JSON格式)
          </label>
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

        {/* 分塊配置 */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-2">分塊大小</label>
            <div className="flex flex-wrap gap-2">
              {[200, 300, 400, 500, 600, 800, 900, 1000].map((size) => (
                <label key={size} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={chunkSizes.includes(size)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setChunkSizes([...chunkSizes, size]);
                      } else {
                        setChunkSizes(chunkSizes.filter((s) => s !== size));
                      }
                    }}
                    className="mr-1"
                  />
                  <span className="text-sm">{size}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">重疊比例</label>
            <div className="flex flex-wrap gap-2">
              {[0.0, 0.1, 0.2, 0.3].map((ratio) => (
                <label key={ratio} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={overlapRatios.includes(ratio)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setOverlapRatios([...overlapRatios, ratio]);
                      } else {
                        setOverlapRatios(
                          overlapRatios.filter((r) => r !== ratio)
                        );
                      }
                    }}
                    className="mr-1"
                  />
                  <span className="text-sm">{(ratio * 100).toFixed(0)}%</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* 分塊策略 */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">分塊策略</label>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            className="w-48 px-3 py-2 border rounded-md"
          >
            <option value="fixed_size">固定大小分割</option>
            <option value="structured_hierarchical">結構化層次分割</option>
            <option value="hierarchical">層次分割</option>
            <option value="semantic">語義分割</option>
            <option value="sliding_window">滑動視窗分割</option>
          </select>
        </div>

        {/* 上傳按鈕 */}
        <button
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          className="bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700 disabled:bg-gray-400"
        >
          {isUploading ? "處理中..." : "開始映射"}
        </button>
      </div>

      {/* 映射進度 */}
      {isUploading && mappingTaskId && (
        <div className="bg-yellow-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">映射進度</h3>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
              style={{ width: `${mappingProgress}%` }}
            ></div>
          </div>
          <p className="text-sm text-gray-600 mt-2">
            已完成 {mappingProgress.toFixed(1)}%
          </p>
        </div>
      )}

      {/* 映射錯誤 */}
      {mappingError && (
        <div className="bg-red-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-red-800 mb-2">映射錯誤</h3>
          <p className="text-red-600">{mappingError}</p>
        </div>
      )}

      {/* 映射結果 */}
      {mappingResult && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">映射結果</h3>
            <button
              onClick={() => setShowDetailedView(!showDetailedView)}
              className="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700 text-sm"
            >
              {showDetailedView ? "簡化視圖" : "詳細視圖"}
            </button>
          </div>

          {showDetailedView ? (
            <QAChunkMappingViewer
              mappingResults={mappingResult.mapping_results || {}}
            />
          ) : (
            <>
              {/* 配置統計 */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">配置統計</h4>
                <p className="text-sm text-gray-600">
                  總配置數: {mappingResult.total_configs} | 原始QA問題數:{" "}
                  {mappingResult.original_qa_set?.length || 0}
                </p>
              </div>

              {/* 顯示前幾個配置的映射結果 */}
              {Object.entries(mappingResult.mapping_results || {})
                .slice(0, 3)
                .map(([configId, result]: [string, any]) => (
                  <div key={configId} className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">
                      配置: {result.config.chunk_size} 字符,{" "}
                      {(result.config.overlap_ratio * 100).toFixed(0)}% 重疊
                      <span className="ml-2 text-sm text-gray-600">
                        ({result.chunk_count} chunks)
                      </span>
                    </h4>
                    {renderQASetPreview(
                      result.mapped_qa_set,
                      "映射後的QA Set",
                      result.chunks_with_span
                    )}
                  </div>
                ))}

              {/* 顯示更多配置的按鈕 */}
              {Object.keys(mappingResult.mapping_results || {}).length > 3 && (
                <div className="text-center">
                  <button
                    onClick={() => toggleExpanded(-1)}
                    className="text-blue-600 hover:text-blue-800 font-medium"
                  >
                    {expandedItems.has(-1)
                      ? "收起其他配置"
                      : `顯示其他 ${
                          Object.keys(mappingResult.mapping_results || {})
                            .length - 3
                        } 個配置`}
                  </button>

                  {expandedItems.has(-1) && (
                    <div className="mt-4 space-y-4">
                      {Object.entries(mappingResult.mapping_results || {})
                        .slice(3)
                        .map(([configId, result]: [string, any]) => (
                          <div key={configId} className="border rounded-lg p-4">
                            <h4 className="font-semibold mb-3">
                              配置: {result.config.chunk_size} 字符,{" "}
                              {(result.config.overlap_ratio * 100).toFixed(0)}%
                              重疊
                              <span className="ml-2 text-sm text-gray-600">
                                ({result.chunk_count} chunks)
                              </span>
                            </h4>
                            {renderQASetPreview(
                              result.mapped_qa_set,
                              "映射後的QA Set",
                              result.chunks_with_span
                            )}
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              )}

              {/* 確認按鈕 */}
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">映射完成</h4>
                <p className="text-green-600 text-sm mb-3">
                  請檢查映射結果是否正確。如果無誤，可以繼續進行P@K/R@K評測。
                </p>
                <button
                  onClick={() => onMappingComplete(mappingResult)}
                  className="bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700"
                >
                  確認並開始評測
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};
