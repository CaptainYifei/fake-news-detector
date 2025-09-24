from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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
    api_base: Optional[str] = "http://localhost:8000/v1"
    model: Optional[str] = "jan-v1-4b"
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 1000


# 响应模型
class FactCheckResponse(BaseModel):
    claim: str
    verdict: str
    reasoning: str
    evidence: List[Dict[str, Any]]
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
            request.api_base, request.model, request.temperature, request.max_tokens
        )

        # 提取声明
        claim = fact_checker.extract_claim(request.text)
        # 处理claim字符串，提取"claim:"后面的内容
        if "claim:" in claim.lower():
            claim = claim.split("claim:")[-1].strip()

        # 搜索证据
        evidence_docs = fact_checker.search_evidence(claim)

        # 获取相关证据块
        evidence_chunks = fact_checker.get_evidence_chunks(evidence_docs, claim)

        # 评估声明
        evaluation = fact_checker.evaluate_claim(claim, evidence_chunks)

        # 构建响应
        response = {
            "claim": claim,
            "verdict": evaluation["verdict"],
            "reasoning": evaluation["reasoning"],
            "evidence": evidence_chunks,
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
