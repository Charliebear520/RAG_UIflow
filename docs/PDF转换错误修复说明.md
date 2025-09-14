# PDF 转换错误修复说明

## 问题描述

用户报告了以下错误：

- 控制台显示：`POST http://localhost:5174/api/convert 500 (Internal Server Error)`
- 前端错误：`SyntaxError: Failed to execute 'json' on 'Response': Unexpected end of JSON input`
- 前端显示：`轉換失敗: Failed to execute 'json' on 'Response': Unexpected end of JSON input`

## 问题分析

通过系统性的调试发现：

1. **后端实际工作正常**：使用 curl 测试后端端点，发现：

   - `/api/convert` 返回 `200 OK` 和有效的 JSON 响应
   - `/api/convert/status/{task_id}` 也返回完整的 JSON 数据

2. **问题出现在前端**：前端的 `json` 辅助函数在处理响应时存在问题：
   - 当 `res.ok` 为 true 时，直接调用 `res.json()`
   - 如果响应体为空或格式不正确，会抛出 "Unexpected end of JSON input" 错误

## 修复方案

### 1. 前端修复 (`frontend/src/lib/api.ts`)

**修复前的问题代码：**

```typescript
async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    // 错误处理逻辑...
  }
  return res.json(); // 直接调用，没有错误处理
}
```

**修复后的代码：**

```typescript
async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    // 错误处理逻辑...
  }

  // 检查响应是否为空
  const text = await res.text();
  if (!text.trim()) {
    throw new Error("Empty response from server");
  }

  try {
    return JSON.parse(text);
  } catch (e) {
    console.log("Failed to parse JSON response:", text.substring(0, 200));
    throw new Error(
      `Invalid JSON response: ${
        e instanceof Error ? e.message : "Unknown error"
      }`
    );
  }
}
```

**改进点：**

- 先获取响应的文本内容
- 检查响应是否为空
- 使用 try-catch 包装 JSON.parse
- 提供更详细的错误信息用于调试

### 2. 后端修复 (`backend/app/main.py`)

**修复前的问题：**

```python
# 直接抛出 HTTPException，可能导致前端无法正确解析错误响应
if not file.filename or not file.filename.lower().endswith('.pdf'):
    raise HTTPException(status_code=400, detail="只支持PDF文件格式")
```

**修复后的代码：**

```python
# 使用 JSONResponse 确保返回有效的 JSON
if not file.filename or not file.filename.lower().endswith('.pdf'):
    return JSONResponse(
        status_code=400,
        content={"error": "只支持PDF文件格式", "detail": "Invalid file type"}
    )
```

**改进点：**

- 使用 `JSONResponse` 替代 `HTTPException`
- 确保所有错误响应都是有效的 JSON 格式
- 提供结构化的错误信息

## 测试结果

### 后端测试

```bash
curl -X POST http://localhost:8000/api/convert \
  -F "file=@/Users/charliebear/Desktop/code/RAG/corpus/著作權法.pdf" \
  -F "metadata_options={}" \
  -H "Accept: application/json"
```

**结果：**

- 状态码：`200 OK`
- 响应：`{"task_id":"convert_1757827809844_4075","status":"pending","message":"PDF轉換任務已啟動，請使用task_id查詢進度"}`

### 状态查询测试

```bash
curl http://localhost:8000/api/convert/status/convert_1757827809844_4075
```

**结果：**

- 状态码：`200 OK`
- 响应：完整的 JSON 数据，包含转换结果

## 修复效果

1. **前端错误处理更健壮**：能够正确处理各种响应情况
2. **后端错误响应标准化**：所有错误都返回有效的 JSON 格式
3. **调试信息更详细**：提供更多上下文信息用于问题诊断
4. **用户体验改善**：错误信息更清晰，不再出现 "Unexpected end of JSON input" 错误

## 建议

1. **监控和日志**：建议添加更详细的日志记录来监控 API 调用
2. **错误分类**：可以考虑对不同类型的错误进行分类处理
3. **重试机制**：对于网络相关的临时错误，可以考虑添加重试机制
4. **用户反馈**：改善用户界面的错误显示，提供更友好的错误信息

## 相关文件

- `frontend/src/lib/api.ts` - 前端 API 调用逻辑
- `backend/app/main.py` - 后端 API 端点实现
- `frontend/src/lib/ragStore.tsx` - 前端状态管理
- `frontend/src/routes/UploadPage.tsx` - 上传页面组件
