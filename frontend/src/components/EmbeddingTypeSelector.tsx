import React from "react";

// Embedding類型定義
export type EmbeddingType = "standard" | "multi_level";

// Embedding類型信息
export const embeddingTypeInfo = {
  standard: {
    name: "標準 Embedding",
    description: "基於分塊結果創建單一層次的embedding向量，適合傳統檢索需求。",
    pros: ["簡單快速", "資源消耗低", "適合大部分場景"],
    cons: ["缺乏語義層次", "檢索精度有限"],
    use_cases: ["一般性查詢", "快速檢索", "資源受限環境"],
    api_endpoint: "/api/embed",
  },
  multi_level: {
    name: "多層次 Embedding",
    description:
      "為不同語義層次創建獨立的embedding向量，支持智能查詢分類和層次感知檢索。",
    pros: ["語義層次化", "智能查詢分類", "檢索精度高", "支持多種融合策略"],
    cons: ["資源消耗較高", "計算複雜度大"],
    use_cases: ["複雜法律查詢", "精確檢索", "多層次分析"],
    api_endpoint: "/api/multi-level-embed",
  },
};

interface EmbeddingTypeSelectorProps {
  embeddingType: EmbeddingType;
  onEmbeddingTypeChange: (type: EmbeddingType) => void;
  selectedStrategy: string | null;
}

export const EmbeddingTypeSelector: React.FC<EmbeddingTypeSelectorProps> = ({
  embeddingType,
  onEmbeddingTypeChange,
  selectedStrategy,
}) => {
  // 檢查是否支持多層次embedding
  const supportsMultiLevel = selectedStrategy === "structured_hierarchical";

  return (
    <div className="space-y-4">
      {/* Embedding類型選擇 */}
      <div>
        <label className="block text-sm font-medium mb-2">Embedding 類型</label>
        <div className="grid grid-cols-1 gap-3">
          {Object.entries(embeddingTypeInfo).map(([key, info]) => {
            const isDisabled = key === "multi_level" && !supportsMultiLevel;
            const isSelected = embeddingType === key;

            return (
              <button
                key={key}
                onClick={() =>
                  !isDisabled && onEmbeddingTypeChange(key as EmbeddingType)
                }
                disabled={isDisabled}
                className={`p-4 text-left border rounded-lg transition-colors ${
                  isSelected
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : isDisabled
                    ? "border-gray-200 bg-gray-50 text-gray-400 cursor-not-allowed"
                    : "border-gray-200 hover:border-gray-300"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="font-medium text-lg">{info.name}</div>
                  {key === "multi_level" && !supportsMultiLevel && (
                    <span className="text-xs bg-gray-200 text-gray-600 px-2 py-1 rounded">
                      需要層次分割
                    </span>
                  )}
                </div>
                <div className="text-sm text-gray-600 mt-2">
                  {info.description}
                </div>
                {isDisabled && (
                  <div className="text-xs text-gray-500 mt-2">
                    ⚠️ 多層次 Embedding 需要先使用「層次分割」策略進行分塊
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* 類型詳細信息 */}
      {embeddingTypeInfo[embeddingType] && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="font-medium mb-3">
            {embeddingTypeInfo[embeddingType].name}
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-green-700 mb-2">優點:</h5>
              <ul className="text-xs text-gray-600 space-y-1">
                {embeddingTypeInfo[embeddingType].pros.map((pro, index) => (
                  <li key={index}>• {pro}</li>
                ))}
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-red-700 mb-2">缺點:</h5>
              <ul className="text-xs text-gray-600 space-y-1">
                {embeddingTypeInfo[embeddingType].cons.map((con, index) => (
                  <li key={index}>• {con}</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="mt-4">
            <h5 className="text-sm font-medium text-blue-700 mb-2">
              適用場景:
            </h5>
            <div className="flex flex-wrap gap-2">
              {embeddingTypeInfo[embeddingType].use_cases.map(
                (useCase, index) => (
                  <span
                    key={index}
                    className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded"
                  >
                    {useCase}
                  </span>
                )
              )}
            </div>
          </div>

          {/* 多層次embedding的特殊說明 */}
          {embeddingType === "multi_level" && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
              <h6 className="text-sm font-medium text-blue-800 mb-2">
                🔍 多層次 Embedding 特性
              </h6>
              <ul className="text-xs text-blue-700 space-y-1">
                <li>
                  • <strong>文件層級</strong>：整個法規，適合宏觀查詢
                </li>
                <li>
                  • <strong>文件組成部分層級</strong>：章級別，適合主題性查詢
                </li>
                <li>
                  • <strong>基本單位層次結構層級</strong>
                  ：節級別，適合結構性查詢
                </li>
                <li>
                  • <strong>基本單位層級</strong>：條文級別，適合具體法條查詢
                </li>
                <li>
                  • <strong>基本單位組成部分層級</strong>
                  ：項級別，適合詳細規定查詢
                </li>
                <li>
                  • <strong>列舉層級</strong>：款/目級別，適合具體細節查詢
                </li>
                <li>
                  • <strong>智能分類</strong>：自動識別查詢類型並選擇合適層次
                </li>
                <li>
                  • <strong>結果融合</strong>：支持多種融合策略提升檢索效果
                </li>
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
