#!/bin/bash

# Docker启动脚本 - AI虚假新闻检测器

echo "🐳 启动AI虚假新闻检测器Docker环境"
echo "=================================="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 创建必要的目录
echo "📁 创建数据目录..."
mkdir -p data
mkdir -p searxng

# 构建并启动服务
echo "🔨 构建Docker镜像..."
docker-compose build

echo "🚀 启动服务..."
docker-compose up -d

# 检查服务状态
echo ""
echo "⏳ 等待服务启动..."
sleep 10

echo "🔍 检查服务状态..."
docker-compose ps

echo ""
echo "✅ 服务启动完成！"
echo ""
echo "🌐 访问地址："
echo "   • Streamlit Web界面: http://localhost:8501"
echo "   • FastAPI接口文档: http://localhost:5000/docs"
echo ""
echo "📋 管理命令："
echo "   • 查看日志: docker-compose logs -f"
echo "   • 停止服务: docker-compose down"
echo "   • 重启服务: docker-compose restart"
echo ""
echo "⚠️  注意："
echo "   • 需要宿主机运行Ollama (11434) 或 LM Studio (11435)"