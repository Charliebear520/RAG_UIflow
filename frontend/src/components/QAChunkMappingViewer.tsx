import React, { useState } from "react";

interface QAChunkMappingViewerProps {
  mappingResults: Record<string, any>;
}

interface QAMappingItem {
  query: string;
  label: string;
  answer: string;
  relevant_chunks?: string[];
  spans?: Array<{
    start_char: number;
    end_char: number;
    text: string;
  }>;
}

export const QAChunkMappingViewer: React.FC<QAChunkMappingViewerProps> = ({
  mappingResults,
}) => {
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);
  const [expandedQuestions, setExpandedQuestions] = useState<Set<number>>(
    new Set()
  );

  const toggleQuestionExpansion = (questionIndex: number) => {
    const newExpanded = new Set(expandedQuestions);
    if (newExpanded.has(questionIndex)) {
      newExpanded.delete(questionIndex);
    } else {
      newExpanded.add(questionIndex);
    }
    setExpandedQuestions(newExpanded);
  };

  const renderChunkContent = (chunkId: string, chunksWithSpan: any[]) => {
    const chunkInfo = chunksWithSpan.find((c: any) => c.chunk_id === chunkId);
    if (!chunkInfo) return null;

    const lawSpans = chunkInfo.metadata?.overlapping_law_spans || [];

    return (
      <div className="bg-gray-50 border rounded-lg p-3 mb-2">
        <div className="flex items-center justify-between mb-2">
          <h6 className="font-semibold text-blue-600 mb-0">{chunkId}</h6>
          <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
            [{chunkInfo.span.start}-{chunkInfo.span.end}]
          </span>
        </div>
        <div className="text-sm text-gray-700 bg-white p-2 rounded border max-h-32 overflow-y-auto">
          {chunkInfo.content}
        </div>
        
        {/* 顯示對應的法條JSON spans */}
        {lawSpans.length > 0 && (
          <div className="mt-2">
            <h6 className="font-medium text-green-600 mb-1">對應法條JSON spans:</h6>
            <div className="space-y-1">
              {lawSpans.slice(0, 3).map((lawSpan: any, index: number) => (
                <div key={index} className="bg-green-50 p-2 rounded text-xs">
                  <div className="font-medium text-green-800">
                    {lawSpan.article_name} ({lawSpan.article_id})
                  </div>
                  <div className="text-green-700">
                    [{lawSpan.start_char}-{lawSpan.end_char}] 
                    (重疊: {(lawSpan.overlap_ratio * 100).toFixed(1)}%)
                  </div>
                  <div className="text-green-600 mt-1">
                    {lawSpan.chapter_name} > {lawSpan.section_name}
                  </div>
                </div>
              ))}
              {lawSpans.length > 3 && (
                <div className="text-xs text-gray-500">
                  還有 {lawSpans.length - 3} 個法條spans...
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderQuestionItem = (
    qaItem: QAMappingItem,
    questionIndex: number,
    chunksWithSpan: any[]
  ) => {
    const isExpanded = expandedQuestions.has(questionIndex);

    return (
      <div key={questionIndex} className="border rounded-lg p-4 mb-3">
        <div className="flex items-start justify-between mb-2">
          <div className="flex-1">
            <div className="flex items-center mb-2">
              <span
                className={`inline-block w-6 h-6 rounded-full text-center text-sm font-bold mr-2 ${
                  qaItem.label === "Yes"
                    ? "bg-green-100 text-green-800"
                    : "bg-red-100 text-red-800"
                }`}
              >
                {qaItem.label === "Yes" ? "✓" : "✗"}
              </span>
              <h5 className="font-medium text-gray-900 mb-0">
                問題 {questionIndex + 1}
              </h5>
            </div>
            <p className="text-gray-700 mb-2">{qaItem.query}</p>
            {qaItem.answer && (
              <div className="bg-blue-50 p-2 rounded mb-2">
                <p className="text-sm text-blue-800">
                  <strong>答案：</strong>
                  {qaItem.answer}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* 相關chunks信息 */}
        <div className="border-t pt-3">
          <div className="flex items-center justify-between mb-2">
            <h6 className="font-medium text-gray-900 mb-0">
              相關chunks ({qaItem.relevant_chunks?.length || 0})
            </h6>
            {qaItem.relevant_chunks && qaItem.relevant_chunks.length > 0 && (
              <button
                onClick={() => toggleQuestionExpansion(questionIndex)}
                className="text-blue-600 hover:text-blue-800 text-sm font-medium"
              >
                {isExpanded ? "隱藏chunk內容" : "查看chunk內容"}
              </button>
            )}
          </div>

          {qaItem.relevant_chunks && qaItem.relevant_chunks.length > 0 ? (
            <div>
              <div className="flex flex-wrap gap-1 mb-2">
                {qaItem.relevant_chunks.map((chunkId, index) => (
                  <span
                    key={index}
                    className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded"
                  >
                    {chunkId}
                  </span>
                ))}
              </div>

              {isExpanded && (
                <div className="mt-3">
                  <h6 className="font-medium text-gray-900 mb-2">
                    Chunk內容詳情：
                  </h6>
                  {qaItem.relevant_chunks.map((chunkId, index) => (
                    <div key={index}>
                      {renderChunkContent(chunkId, chunksWithSpan)}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="text-gray-500 text-sm">
              {qaItem.label === "No"
                ? "負例問題，無相關chunks"
                : "未找到相關chunks"}
            </div>
          )}
        </div>

        {/* 原始spans信息 */}
        {qaItem.spans && qaItem.spans.length > 0 && (
          <div className="border-t pt-3 mt-3">
            <h6 className="font-medium text-gray-900 mb-2">原始spans：</h6>
            <div className="space-y-1">
              {qaItem.spans.map((span, index) => (
                <div
                  key={index}
                  className="text-sm text-gray-600 bg-gray-50 p-2 rounded"
                >
                  <span className="font-medium">
                    [{span.start_char}-{span.end_char}]
                  </span>
                  <div className="mt-1 text-xs text-gray-500">
                    {span.text.substring(0, 100)}
                    {span.text.length > 100 && "..."}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const configEntries = Object.entries(mappingResults);
  const selectedResult = selectedConfig ? mappingResults[selectedConfig] : null;

  return (
    <div className="space-y-6">
      <div className="bg-white border rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4">QA Set映射結果查看器</h3>

        {/* 配置選擇器 */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            選擇分塊配置：
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {configEntries.map(([configId, result]) => (
              <button
                key={configId}
                onClick={() => setSelectedConfig(configId)}
                className={`p-3 text-left border rounded-lg transition-colors ${
                  selectedConfig === configId
                    ? "border-blue-500 bg-blue-50"
                    : "border-gray-200 hover:border-gray-300"
                }`}
              >
                <div className="font-medium text-sm">
                  {result.strategy} - {result.config.chunk_size}字符
                </div>
                <div className="text-xs text-gray-600">
                  {(result.config.overlap_ratio * 100).toFixed(0)}% 重疊
                </div>
                <div className="text-xs text-gray-500">
                  {result.chunk_count} chunks
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 選中配置的詳細信息 */}
      {selectedResult && (
        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold">
              配置詳情：{selectedResult.strategy} -{" "}
              {selectedResult.config.chunk_size}字符
            </h4>
            <div className="text-sm text-gray-600">
              共 {selectedResult.mapped_qa_set?.length || 0} 個問題
            </div>
          </div>

          {/* 配置統計 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-blue-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">
                {selectedResult.chunk_count}
              </div>
              <div className="text-sm text-blue-800">總chunks</div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-green-600">
                {selectedResult.mapping_stats?.questions_with_chunks || 0}
              </div>
              <div className="text-sm text-green-800">有映射的問題</div>
            </div>
            <div className="bg-yellow-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-green-600">
                {(selectedResult.mapping_stats?.mapping_success_rate || 0).toFixed(1)}%
              </div>
              <div className="text-sm text-green-800">映射成功率</div>
            </div>
          </div>

          {/* 問題列表 */}
          <div className="space-y-4">
            <h5 className="font-semibold">問題映射詳情：</h5>
            {selectedResult.mapped_qa_set?.map(
              (qaItem: QAMappingItem, index: number) =>
                renderQuestionItem(
                  qaItem,
                  index,
                  selectedResult.chunks_with_span || []
                )
            )}
          </div>
        </div>
      )}
    </div>
  );
};
