from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import os
import sys
from datetime import datetime
import json
import asyncio

# 导入你的FactChecker类
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fact_checker import FactChecker

# 创建FastAPI应用
app = FastAPI(
    title="AI虚假新闻检测API",
    description="提供虚假新闻检测服务的REST API",
    version="2.0.0",
)

# 配置CORS，允许来自Chrome扩展的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为你的扩展ID
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class FactCheckRequest(BaseModel):
    text: str
    api_base: Optional[str] = "http://localhost:11434/v1"
    model: Optional[str] = "qwen2.5:3b"
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 1000
    embedding_base_url: Optional[str] = None
    embedding_model: Optional[str] = "bge-m3:latest"
    search_engine: Optional[str] = "searxng"
    searxng_url: Optional[str] = "http://localhost:8090"


# 响应模型
class FactCheckResponse(BaseModel):
    claim: str
    verdict: str
    reasoning: str
    evidence: List[Dict[str, Any]]
    pipeline: Optional[Dict[str, Any]] = None
    timestamp: str


# 活跃任务缓存
active_tasks = {}


@app.post("/check", response_model=FactCheckResponse)
async def check_fact(request: FactCheckRequest, background_tasks: BackgroundTasks):
    """
    检查新闻文本的真实性
    """
    try:
        # 初始化FactChecker
        fact_checker = FactChecker(
            request.api_base,
            request.model,
            request.temperature,
            request.max_tokens,
            embedding_base_url=request.embedding_base_url or request.api_base,
            embedding_model=request.embedding_model,
            search_engine=request.search_engine,
            searxng_url=request.searxng_url,
        )

        # Search providers and model clients are synchronous. Run them outside
        # the ASGI event loop (also lets the optional MCP client own its loop).
        result = await run_in_threadpool(fact_checker.check_news, request.text)

        # 构建响应
        response = {
            "claim": result.claim,
            "verdict": result.verdict,
            "reasoning": result.reasoning,
            "evidence": result.evidence_chunks,
            "pipeline": result.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"事实检查过程中出错: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


# 启动服务器的入口点（用于开发）
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
