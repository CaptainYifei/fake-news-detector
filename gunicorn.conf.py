#!/bin/bash
# gunicorn配置文件

# 基本配置
bind = "0.0.0.0:8080"
workers = 4  # 通常为CPU核心数的2-4倍
worker_class = "uvicorn.workers.UvicornWorker"
keepalive = 120
timeout = 120

# 日志配置
accesslog = "access.log"
errorlog = "error.log"
loglevel = "info"

# 重启配置
max_requests = 1000
max_requests_jitter = 50

# 进程名称
proc_name = "fact_checker_api"

# 预加载应用，减少每个worker的启动时间
preload_app = True