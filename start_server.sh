#!/bin/bash

# 安装必要的依赖
# pip install fastapi uvicorn gunicorn

# 启动服务器
echo "正在启动虚假新闻检测API服务器..."
gunicorn -c gunicorn.conf.py api:app