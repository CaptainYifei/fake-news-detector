# 使用Alpine Linux基础镜像，体积更小且网络更稳定
FROM python:3.12-alpine

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 安装系统依赖
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p /app/data

# 暴露端口
EXPOSE 8501

# 默认启动命令 - Streamlit应用
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]