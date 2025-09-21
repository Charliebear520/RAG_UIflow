import React from "react";

// 分塊策略類型定義
export type ChunkStrategy =
  | "fixed_size"
  | "rcts_hierarchical"
  | "structured_hierarchical"
  | "hybrid"
  | "semantic";

// 策略參數接口
export interface ChunkParams {
  fixed_size: {
    chunk_size: number;
    overlap: number;
  };
  rcts_hierarchical: {
    max_chunk_size: number;
    overlap_ratio: number;
    preserve_structure: boolean;
  };
  structured_hierarchical: {
    max_chunk_size: number;
    overlap_ratio: number;
    chunk_by: "chapter" | "section" | "article" | "item";
  };
  hybrid: {
    primary_size: number;
    secondary_size: number;
    overlap: number;
    switch_threshold: number;
  };
  semantic: {
    target_size: number;
    similarity_threshold: number;
    overlap: number;
    context_window: number;
  };
}

// 策略描述和評估指標
export const strategyInfo = {
  fixed_size: {
    name: "Fixed-Size",
    description: "將文檔按照固定的字符數進行分割，適合結構化文檔。",
    pros: ["簡單易用", "可控性強", "適合大部分場景"],
    cons: ["可能切斷語義", "不考慮內容結構"],
    metrics: ["平均長度", "長度變異", "分塊數量"],
  },
  rcts_hierarchical: {
    name: "RCTS Hierarchical",
    description:
      "結合RecursiveCharacterTextSplitter和層次結構識別，智能分割法律文檔。",
    pros: ["智能分隔符識別", "保持結構完整性", "處理長文本", "中文優化"],
    cons: ["計算複雜度較高", "需要調優參數"],
    metrics: ["分隔符準確性", "結構保持度", "長文本處理"],
  },
  structured_hierarchical: {
    name: "Structured Hierarchical (pending pilot)",
    description:
      "基於JSON結構數據，按照法律文檔的章-節-條-項結構進行智能分割。",
    pros: ["精確結構識別", "保持法律邏輯", "支持條文引用", "適合法律文檔"],
    cons: ["需要結構化數據", "依賴PDF解析質量"],
    metrics: ["結構準確性", "條文完整性", "引用關係保持"],
  },
  hybrid: {
    name: "Hybrid (Fixed + Semantic)",
    description: "結合固定大小和語義分割策略，根據內容特徵選擇最適合的方法。",
    pros: ["綜合優勢", "適應性強", "效果最佳"],
    cons: ["實現複雜", "參數調優難"],
    metrics: ["綜合評分", "策略切換", "效果提升"],
  },
  semantic: {
    name: "Semantic",
    description: "基於語義相似度進行分割，保持語義連貫性。",
    pros: ["語義完整", "連貫性好", "適合問答"],
    cons: ["計算複雜", "需要預訓練模型"],
    metrics: ["語義連貫性", "相似度", "分塊質量"],
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

      case "rcts_hierarchical":
        return (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                最大分塊大小
              </label>
              <input
                type="number"
                value={params.rcts_hierarchical.max_chunk_size}
                onChange={(e) =>
                  handleParamChange(
                    "rcts_hierarchical",
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
                value={params.rcts_hierarchical.overlap_ratio}
                onChange={(e) =>
                  handleParamChange(
                    "rcts_hierarchical",
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
                  checked={params.rcts_hierarchical.preserve_structure}
                  onChange={(e) =>
                    handleParamChange(
                      "rcts_hierarchical",
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
              <label className="block text-sm font-medium mb-1">分割單位</label>
              <select
                value={params.structured_hierarchical.chunk_by}
                onChange={(e) => {
                  const newParams = { ...params };
                  newParams.structured_hierarchical.chunk_by = e.target
                    .value as "chapter" | "section" | "article" | "item";
                  onParamsChange(newParams);
                }}
                className="w-full px-3 py-2 border rounded-md"
              >
                <option value="article">按條文分割</option>
                <option value="item">按項分割</option>
                <option value="section">按節分割</option>
                <option value="chapter">按章分割</option>
              </select>
              <div className="text-xs text-gray-500 mt-1">
                按條文分割：每條法律條文獨立成塊（推薦）
                <br />
                按項分割：每個條文項目獨立成塊（精細化）
                <br />
                按節分割：每個章節獨立成塊
                <br />
                按章分割：每個章獨立成塊
              </div>
            </div>
          </div>
        );

      case "hybrid":
        return (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                主要分塊大小
              </label>
              <input
                type="number"
                value={params.hybrid.primary_size}
                onChange={(e) =>
                  handleParamChange(
                    "hybrid",
                    "primary_size",
                    parseInt(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="200"
                max="1500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                次要分塊大小
              </label>
              <input
                type="number"
                value={params.hybrid.secondary_size}
                onChange={(e) =>
                  handleParamChange(
                    "hybrid",
                    "secondary_size",
                    parseInt(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="100"
                max="800"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">重疊大小</label>
              <input
                type="number"
                value={params.hybrid.overlap}
                onChange={(e) =>
                  handleParamChange(
                    "hybrid",
                    "overlap",
                    parseInt(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="0"
                max="200"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">切換閾值</label>
              <input
                type="number"
                step="0.1"
                value={params.hybrid.switch_threshold}
                onChange={(e) =>
                  handleParamChange(
                    "hybrid",
                    "switch_threshold",
                    parseFloat(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="0"
                max="1"
              />
            </div>
          </div>
        );

      case "semantic":
        return (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">目標大小</label>
              <input
                type="number"
                value={params.semantic.target_size}
                onChange={(e) =>
                  handleParamChange(
                    "semantic",
                    "target_size",
                    parseInt(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="200"
                max="1500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                相似度閾值
              </label>
              <input
                type="number"
                step="0.1"
                value={params.semantic.similarity_threshold}
                onChange={(e) =>
                  handleParamChange(
                    "semantic",
                    "similarity_threshold",
                    parseFloat(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="0"
                max="1"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">重疊大小</label>
              <input
                type="number"
                value={params.semantic.overlap}
                onChange={(e) =>
                  handleParamChange(
                    "semantic",
                    "overlap",
                    parseInt(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="0"
                max="200"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                上下文窗口
              </label>
              <input
                type="number"
                value={params.semantic.context_window}
                onChange={(e) =>
                  handleParamChange(
                    "semantic",
                    "context_window",
                    parseInt(e.target.value)
                  )
                }
                className="w-full px-3 py-2 border rounded-md"
                min="100"
                max="1000"
              />
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
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(strategyInfo).map(([key, info]) => (
            <button
              key={key}
              onClick={() => onStrategyChange(key as ChunkStrategy)}
              className={`p-3 text-left border rounded-lg transition-colors ${
                strategy === key
                  ? "border-blue-500 bg-blue-50 text-blue-700"
                  : "border-gray-200 hover:border-gray-300"
              }`}
            >
              <div className="font-medium">{info.name}</div>
              <div className="text-xs text-gray-600 mt-1">
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
