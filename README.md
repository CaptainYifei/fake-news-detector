# 🔍 AI虚假新闻检测器 (AI Fake News Detector)

一个基于事实核查的智能新闻验证系统，支持多语言、多模型提供商，使用先进的语义嵌入技术和大型语言模型进行准确的事实核查。

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.43+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

![应用截图](docs/images/screenshot.png)

## ✨ 核心特性

### 🌍 多语言支持
- **智能语言检测**：自动识别中文、英文、日文、韩文输入
- **多语言输出**：支持用户自定义输出语言或自动检测
- **本地化界面**：完整的中英文界面支持

### 🤖 多模型提供商支持
- **Ollama**：本地部署模型（默认：GPT-OSS 120B Cloud + Nomic Embed）
- **LM Studio**：本地模型服务
- **OpenAI**：官方 GPT 系列模型
- **自定义API**：兼容 OpenAI 格式的任意模型服务

### 🔍 高精度事实核查
- **声明提取**：智能提取新闻核心可验证声明
- **多源搜索**：支持 SearXNG、DuckDuckGo 等搜索引擎
- **语义匹配**：使用先进嵌入模型计算证据相关性
- **推理透明**：提供详细的推理过程和证据来源

### 📊 完整的数据管理
- **历史记录**：保存和查看所有事实核查历史
- **PDF导出**：生成专业的核查报告
- **用户系统**：支持多用户独立使用

## 🚀 快速开始

### 前提条件

- **Python 3.12+**
- **Ollama** (推荐) 或其他兼容 OpenAI API 的模型服务
- **SearXNG** (可选，用于搜索功能)

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/CaptainYifei/fake-news-detector.git
cd fake-news-detector
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置模型服务** (推荐使用 Ollama)
```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取推荐模型
ollama pull gpt-oss:120b-cloud
ollama pull nomic-embed-text:latest
```

4. **配置搜索服务** (可选)
```bash
# 使用 Docker 启动 SearXNG
docker run -d -p 8090:8080 searxng/searxng
```

### 启动应用

```bash
streamlit run app.py
```

应用将在 http://localhost:8501 启动

## 📋 项目结构

```
fake-news-detector/
├── app.py                 # Streamlit 主应用
├── fact_checker.py        # 事实核查核心逻辑
├── model_manager.py       # 模型管理和配置
├── model_config.json      # 模型和服务配置文件
├── auth.py                # 用户认证系统
├── db_utils.py            # 数据库操作
├── pdf_export.py          # PDF 报告生成
├── requirements.txt       # 项目依赖
├── api.py                 # RESTful API 接口
├── docs/                  # 文档和使用说明
└── test/                  # 测试文件
```

## ⚙️ 配置说明

### 模型配置 (`model_config.json`)

系统通过 `model_config.json` 进行统一配置，支持：

```json
{
  "providers": {
    "ollama": {
      "name": "Ollama",
      "type": "openai_compatible",
      "base_url": "http://localhost:11434/v1",
      "models": {
        "gpt-oss:120b-cloud": {
          "name": "GPT-OSS 120B Cloud",
          "type": "chat",
          "max_tokens": 8192
        },
        "nomic-embed-text:latest": {
          "name": "Nomic Embed Text",
          "type": "embedding",
          "dimensions": 768
        }
      }
    }
  },
  "defaults": {
    "llm_provider": "ollama",
    "llm_model": "gpt-oss:120b-cloud",
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text:latest",
    "output_language": "zh"
  }
}
```

### 搜索引擎配置

支持多种搜索引擎，可在配置文件中设置：
- **SearXNG**: 本地部署的隐私搜索引擎
- **DuckDuckGo**: 在线搜索（支持代理配置）

## 🔄 工作流程

1. **声明提取** - 使用 LLM 从输入文本提取核心声明
2. **证据搜索** - 通过搜索引擎获取相关网络证据
3. **语义排序** - 使用嵌入模型计算证据相关性
4. **事实判断** - 基于证据进行 TRUE/FALSE/PARTIALLY TRUE 判断
5. **结果呈现** - 提供详细推理过程和证据来源

## 🌐 多语言支持

- **自动检测**: 根据输入文本自动选择合适的语言模板
- **手动选择**: 用户可指定输出语言（中/英/日/韩）
- **智能切换**: 基于 Unicode 字符模式的语言识别

## 📖 使用说明

### Web 界面使用

1. 选择模型提供商和具体模型
2. 配置搜索引擎和输出语言
3. 输入需要核查的新闻内容
4. 查看实时处理进度和最终结果
5. 导出 PDF 报告或查看历史记录

### API 接口使用

```bash
# 启动 API 服务
python api.py

# 发送核查请求
curl -X POST http://localhost:5000/fact-check \
  -H "Content-Type: application/json" \
  -d '{"text": "需要核查的新闻内容"}'
```

详细 API 文档请参见 `docs/api_doc.html`

## 🛠️ 开发指南

### 环境设置

```bash
# 开发环境安装
pip install -r requirements.txt

# 运行测试
python -m pytest test/

# 启动开发服务器
streamlit run app.py --server.runOnSave true
```

### 贡献代码

1. Fork 此仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📝 更新日志

### v2.0.0 (最新版本)
- ✨ 新增多语言支持 (中/英/日/韩)
- 🔧 统一模型管理系统
- 🌐 支持多搜索引擎 (SearXNG/DuckDuckGo)
- 📱 改进用户界面和交互体验
- 🛡️ 增强错误处理和配置管理
- 📄 完善 PDF 导出功能

### v1.0.0
- 🎉 初始版本发布
- ✅ 基础事实核查功能
- 👤 用户认证系统
- 💾 数据持久化存储

## 🐛 故障排除

### 常见问题

**Q: 模型无响应或返回空结果**
A: 检查模型服务是否正常运行，确认 API 地址和端口配置正确

**Q: 搜索功能无法使用**
A: 检查网络连接，确认搜索引擎服务状态，必要时配置代理

**Q: 多语言输出异常**
A: 确认使用的模型支持目标语言，尝试切换到更强的模型

更多问题请查看 [Issues](https://github.com/CaptainYifei/fake-news-detector/issues) 或提交新的问题报告。

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件

## 🔗 相关链接

- **GitHub**: [https://github.com/CaptainYifei/fake-news-detector](https://github.com/CaptainYifei/fake-news-detector)
- **Gitee**: [https://gitee.com/love2eat/fake-news-detector](https://gitee.com/love2eat/fake-news-detector)
- **文档**: [docs/usage.md](docs/usage.md)
- **API文档**: [docs/api_doc.html](docs/api_doc.html)

---

⭐ 如果这个项目对你有帮助，请给我们一个 Star！