import React from "react";

// 分塊策略類型定義
export type ChunkStrategy = "fixed_size" | "structured_hierarchical";

// 策略參數接口
export interface ChunkParams {
  fixed_size: {
    chunk_size: number;
    overlap: number;
  };
  structured_hierarchical: {
    max_chunk_size: number;
    overlap_ratio: number;
    preserve_structure: boolean;
  };
}

// 策略描述和評估指標
export const strategyInfo = {
  fixed_size: {
    name: "固定大小分割",
    description: "將文檔按照固定的字符數進行分割，適合結構化文檔。",
    pros: ["簡單易用", "可控性強", "適合大部分場景"],
    cons: ["可能切斷語義", "不考慮內容結構"],
    metrics: ["平均長度", "長度變異", "分塊數量"],
  },
  structured_hierarchical: {
    name: "結構化層次分割",
    description:
      "基於法律文檔的天然結構（章-節-條-項-款-目）進行智能分割，保持法律概念完整性。",
    pros: ["結構完整性", "法律邏輯保持", "語義連貫性", "專業性強"],
    cons: ["需要結構化數據", "依賴文檔格式"],
    metrics: ["結構準確性", "條文完整性", "引用關係保持"],
  },
};

interface ChunkStrategySelectorProps {
  strategy: ChunkStrategy;
  onStrategyChange: (strategy: ChunkStrategy) => void;
  params: ChunkParams;
  onParamsChange: (params: ChunkParams) => void;
}

export const ChunkStrategySelector: React.FC<ChunkStrategySelectorProps> = ({
  strategy,
  onStrategyChange,
  params,
  onParamsChange,
}) => {
  const handleParamChange = (
    strategyType: ChunkStrategy,
    key: string,
    value: number | boolean
  ) => {
    const newParams = { ...params };
    (newParams[strategyType] as any)[key] = value;
    onParamsChange(newParams);
  };

  const renderStrategyParams = () => {
    switch (strategy) {
      case "fixed_size":
        return (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">分塊大小</label>
              <input
                type="number"
                value={params.fixed_size.chunk_size}
                onChange={(e) =>
                  handleParamChange(
                    "fixed_size",
                    "chunk_size",
                    parseInt(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="100"
                max="2000"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">重疊大小</label>
              <input
                type="number"
                value={params.fixed_size.overlap}
                onChange={(e) =>
                  handleParamChange(
                    "fixed_size",
                    "overlap",
                    parseInt(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="0"
                max="200"
              />
            </div>
          </div>
        );

      case "structured_hierarchical":
        return (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                最大分塊大小
              </label>
              <input
                type="number"
                value={params.structured_hierarchical.max_chunk_size}
                onChange={(e) =>
                  handleParamChange(
                    "structured_hierarchical",
                    "max_chunk_size",
                    parseInt(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="200"
                max="2000"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">重疊比例</label>
              <input
                type="number"
                step="0.1"
                value={params.structured_hierarchical.overlap_ratio}
                onChange={(e) =>
                  handleParamChange(
                    "structured_hierarchical",
                    "overlap_ratio",
                    parseFloat(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="0"
                max="0.5"
              />
            </div>
            <div className="col-span-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={params.structured_hierarchical.preserve_structure}
                  onChange={(e) =>
                    handleParamChange(
                      "structured_hierarchical",
                      "preserve_structure",
                      e.target.checked
                    )
                  }
                  className="mr-2"
                />
                保持層次結構
              </label>
              <div className="text-xs text-gray-500 mt-1">
                RCTS層次分割：結合智能分隔符識別和層次結構，適合處理複雜法律文檔
                <br />
                保持結構：在條文邊界強制分割，確保法律邏輯完整性
              </div>
            </div>
          </div>
        );

      default:
        return (
          <div className="text-gray-500 text-sm">
            該策略的參數配置將在後續版本中實現
          </div>
        );
    }
  };

  return (
    <div className="space-y-4">
      {/* 策略選擇 */}
      <div>
        <label className="block text-sm font-medium mb-2">分割策略</label>
        <div className="grid grid-cols-1 gap-3">
          {Object.entries(strategyInfo).map(([key, info]) => (
            <button
              key={key}
              onClick={() => onStrategyChange(key as ChunkStrategy)}
              className={`p-4 text-left border rounded-lg transition-colors ${
                strategy === key
                  ? "border-blue-500 bg-blue-50 text-blue-700"
                  : "border-gray-200 hover:border-gray-300"
              }`}
            >
              <div className="font-medium text-lg">{info.name}</div>
              <div className="text-sm text-gray-600 mt-2">
                {info.description}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 策略描述 */}
      {strategyInfo[strategy] && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="font-medium mb-2">{strategyInfo[strategy].name}</h4>
          <p className="text-sm text-gray-700 mb-3">
            {strategyInfo[strategy].description}
          </p>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-green-700 mb-1">優點:</h5>
              <ul className="text-xs text-gray-600 space-y-1">
                {strategyInfo[strategy].pros.map((pro, index) => (
                  <li key={index}>• {pro}</li>
                ))}
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-red-700 mb-1">缺點:</h5>
              <ul className="text-xs text-gray-600 space-y-1">
                {strategyInfo[strategy].cons.map((con, index) => (
                  <li key={index}>• {con}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* 參數配置 */}
      <div>
        <label className="block text-sm font-medium mb-2">策略參數</label>
        {renderStrategyParams()}
      </div>
    </div>
  );
};
