#!/bin/bash

echo "=== Gemini API Key 設置工具 ==="
echo ""
echo "請按照以下步驟設置您的Google API Key："
echo ""
echo "1. 前往 https://makersuite.google.com/app/apikey"
echo "2. 登入您的Google帳戶"
echo "3. 點擊 'Create API Key'"
echo "4. 複製生成的API Key"
echo ""
echo "請輸入您的API Key："
read -p "API Key: " api_key

if [ -z "$api_key" ]; then
    echo "錯誤：API Key不能為空"
    exit 1
fi

# 更新.env文件
if [ -f ".env" ]; then
    # 移除舊的API Key設置
    sed -i.bak '/^GOOGLE_API_KEY=/d' .env
    sed -i.bak '/^GEMINI_API_KEY=/d' .env
fi

# 添加新的API Key
echo "GOOGLE_API_KEY=$api_key" >> .env
echo "USE_GEMINI_COMPLETION=true" >> .env

echo ""
echo "✅ API Key已成功設置！"
echo ""
echo "現在您可以重啟後端服務："
echo "python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "重啟後，您應該看到："
echo "GOOGLE_API_KEY: 已設置"
echo "USE_GEMINI_COMPLETION: True"
