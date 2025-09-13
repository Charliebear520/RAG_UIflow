"""
RAG系統主應用程序
模組化重構版本
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .routes import router

# 加載環境變量
load_dotenv()

# 創建FastAPI應用
app = FastAPI(
    title="RAG系統API",
    description="基於FastAPI的RAG系統，支持文檔分割、問題生成和評估",
    version="2.0.0"
)

# 添加CORS中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊路由
app.include_router(router, prefix="/api", tags=["RAG API"])

# 根路由
@app.get("/")
async def root():
    """根路由"""
    return {
        "message": "RAG系統API",
        "version": "2.0.0",
        "status": "運行中",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
