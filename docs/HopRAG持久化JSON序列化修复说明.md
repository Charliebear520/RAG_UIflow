# HopRAG 持久化 JSON 序列化修复说明

## 问题描述

在保存 HopRAG 图谱时遇到 JSON 序列化错误：

```
❌ 保存HopRAG图谱失败: keys must be str, int, float, bool or None, not EdgeType
TypeError: keys must be str, int, float, bool or None, not EdgeType
```

错误发生在`hoprag_persistence.py`第 64 行，尝试将 metadata 保存为 JSON 时失败。

## 根本原因

JSON 只支持基本类型（str, int, float, bool, None）作为字典键，但代码中有两处将`EdgeType`枚举对象用作字典键或值：

### 1. 边类型存储问题

**文件**: `backend/app/hoprag_graph_builder.py`  
**位置**: 第 777 行

```python
# 问题代码
'edge_type': self._determine_edge_type(node_a, node_b)
```

`_determine_edge_type`方法返回`EdgeType`枚举对象，直接存储在边属性字典中。

### 2. 统计信息聚合问题

**文件**: `backend/app/hoprag_system_modular.py`  
**位置**: 第 219-226 行

```python
# 问题代码
edge_types = {}
for edge_list in self.edges.values():
    for edge_data in edge_list:
        edge_type = edge_data.get('edge_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
```

当`edge_type`是`EdgeType`枚举时，会被用作字典键，导致 JSON 序列化失败。

## 解决方案

### 修复 1: 存储边类型时转换为字符串

**文件**: `backend/app/hoprag_graph_builder.py`  
**行号**: 777

```python
# 修复后
'edge_type': self._determine_edge_type(node_a, node_b).value  # 轉換為字符串
```

通过添加`.value`，将`EdgeType`枚举转换为其字符串值（如`"basic_unit_to_component"`）。

### 修复 2: 统计时检测并转换枚举

**文件**: `backend/app/hoprag_system_modular.py`  
**行号**: 222-226

```python
# 修复后
edge_type = edge_data.get('edge_type', 'unknown')
# 如果是EdgeType枚舉，轉換為字符串
if isinstance(edge_type, EdgeType):
    edge_type = edge_type.value
edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
```

添加类型检查，确保枚举对象在使用前转换为字符串。

## 已验证的正确处理

### export_graph_data 方法

**文件**: `backend/app/hoprag_system_modular.py`  
**行号**: 296

```python
# 正确处理
"node_type": node.node_type.value,  # 已正确转换
```

节点类型在导出时已经正确转换为字符串值。

## EdgeType 枚举定义

**文件**: `backend/app/hoprag_config.py`  
**行号**: 24-29

```python
class EdgeType(Enum):
    """邊類型枚舉 - 對應新的層級命名"""
    BASIC_UNIT_TO_BASIC_UNIT = "basic_unit_to_basic_unit"
    BASIC_UNIT_TO_COMPONENT = "basic_unit_to_component"
    COMPONENT_TO_BASIC_UNIT = "component_to_basic_unit"
    COMPONENT_TO_COMPONENT = "component_to_component"
```

## 测试建议

修复后应测试以下场景：

1. **构建图谱**: 确保边类型正确存储为字符串

   ```python
   # 构建图谱后检查边属性
   for edge_list in edges.values():
       for edge in edge_list:
           assert isinstance(edge['edge_type'], str)
   ```

2. **保存图谱**: 确保 JSON 序列化成功

   ```python
   persistence = HopRAGPersistence()
   success = persistence.save_graph(hoprag_system)
   assert success == True
   ```

3. **加载图谱**: 确保图谱可以正确加载

   ```python
   success = persistence.load_graph(hoprag_system)
   assert success == True
   assert hoprag_system.is_graph_built
   ```

4. **统计信息**: 确保统计信息可以正确生成和序列化
   ```python
   stats = hoprag_system.get_graph_statistics()
   json.dumps(stats)  # 不应抛出异常
   ```

## 影响范围

- ✅ HopRAG 图谱持久化功能恢复正常
- ✅ 统计信息可以正确生成和序列化
- ✅ 边类型分布统计正确显示
- ✅ 向后兼容：旧代码使用字符串的地方不受影响

## 相关文件

- `backend/app/hoprag_persistence.py` - 持久化管理器
- `backend/app/hoprag_graph_builder.py` - 图构建器（边类型存储）
- `backend/app/hoprag_system_modular.py` - HopRAG 系统（统计信息生成）
- `backend/app/hoprag_config.py` - 配置和枚举定义

## 注意事项

1. **枚举使用原则**: 在内部逻辑中可以使用枚举，但在序列化前必须转换为基本类型
2. **JSON 兼容性**: 所有需要 JSON 序列化的数据结构都应只包含 JSON 支持的类型
3. **类型安全**: 使用`isinstance()`检查确保类型转换的安全性

## 修复时间

2025-10-07
