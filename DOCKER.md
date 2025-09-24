# 🐳 Docker 部署

## 快速开始

```bash
./start-docker.sh
```

或：

```bash
docker-compose up -d
```

访问: http://localhost:8501

## 构建问题

如遇到构建失败，重试：
```bash
docker-compose build --no-cache
```

## 前提条件

- Docker
- 宿主机运行Ollama (端口11434) 或 LM Studio (端口11435)

## 服务地址说明

项目已配置Docker兼容的服务地址：
- **本地运行**: 自动使用`localhost`地址
- **Docker运行**: 自动使用`host.docker.internal`访问宿主机服务
- **自定义**: 可通过环境变量覆盖默认地址

## 管理

```bash
# 停止
docker-compose down

# 查看日志
docker-compose logs -f

# 重新构建
docker-compose build
```

## 注意

如需要更好的搜索体验，可单独运行SearXNG:
```bash
docker run -d -p 8090:8080 searxng/searxng
```
