# 🔍 AI虚假新闻检测器 (AI Fake News Detector)

一个基于事实核查的智能新闻验证系统，支持多语言、多模型提供商，使用先进的语义嵌入技术和大型语言模型进行准确的事实核查。

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.43+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

![AI 虚假新闻检测器应用总览](docs/images/app-overview.png)

## 🎬 项目预览

系统依次完成新闻中心声明抽取、查询改写与证据检索、多步分解声明验证：

![三节点事实核查流程演示](docs/images/fact-check-workflow.gif)

### 🔬 方法流程

下图以“西班牙 1:0 阿根廷夺冠”为简单案例，展示三个节点的输入、处理方法和中间产物：

![AI 虚假新闻三节点核查手绘科研流程图](docs/images/research-workflow-handdrawn.png)

## ✨ 核心特性

### 🌍 多语言支持
- **智能语言检测**：自动识别中文、英文、日文、韩文输入
- **多语言输出**：支持用户自定义输出语言或自动检测
- **本地化界面**：完整的中英文界面支持

### 🤖 多模型提供商支持
- **Ollama**：本地部署模型（默认：Qwen2.5 3B + BGE-M3）
- **LM Studio**：本地模型服务
- **OpenAI**：官方 GPT 系列模型
- **自定义API**：兼容 OpenAI 格式的任意模型服务

### 🔍 三节点事实核查
- **中心声明抽取**：用 LLM 从新闻中抽取核心可验证声明
- **查询改写检索**：用 LLM 改写多角度检索查询，默认通过 DuckDuckGo 获取网络证据
- **多步分解验证**：用 LLM 将复杂声明拆成原子声明，结合证据逐步判断
- **语义证据门控**：使用 BGE 系列嵌入模型计算相关性，按阈值和实体命中过滤后取 Top K

### 📊 完整的数据管理
- **历史记录**：保存和查看所有事实核查历史
- **PDF导出**：生成专业的核查报告
- **用户系统**：支持多用户独立使用

## 🚀 快速开始

### 前提条件

- **Python 3.12+**
- **Ollama** (推荐) 或其他兼容 OpenAI API 的模型服务
- **DuckDuckGo** (默认搜索引擎)
- **SearXNG** (可选，用于本地搜索后端)

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
ollama pull qwen2.5:3b
ollama pull bge-m3:latest
```

4. **配置搜索服务** (默认使用 DuckDuckGo)

系统默认通过 `ddgs` 调用 DuckDuckGo，无需额外安装或部署搜索服务。如需使用本地 SearXNG，可在 `model_config.json` 中将 `search_provider` 切换为 `searxng`。

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
├── fact_checking/         # 三节点事实核查流水线
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
        "qwen2.5:3b": {
          "name": "Qwen2.5 3B",
          "type": "chat",
          "max_tokens": 32768
        },
        "bge-m3:latest": {
          "name": "BGE-M3",
          "type": "embedding",
          "dimensions": 1024
        }
      }
    }
  },
  "defaults": {
    "llm_provider": "ollama",
    "llm_model": "qwen2.5:3b",
    "embedding_provider": "ollama",
    "embedding_model": "bge-m3:latest",
    "search_provider": "duckduckgo",
    "output_language": "zh",
    "evidence_top_k": 8,
    "evidence_min_similarity": 0.45,
    "evidence_entity_gate": true,
    "evidence_ranker_multiplier": 3
  }
}
```

### 搜索引擎配置

支持多种搜索引擎，可在配置文件中设置：
- **DuckDuckGo**: 默认搜索引擎，使用 `ddgs` 并带多后端 fallback。默认尝试 `duckduckgo/wt-wt`、`duckduckgo/cn-zh`、`auto/wt-wt`、`brave/wt-wt`
- **SearXNG**: 可选本地搜索服务

### 证据门控配置

节点 2 会先扩大候选证据池，再执行过滤，最后只把高质量证据交给节点 3：

- `evidence_top_k`: 最终进入验证节点的证据数量，默认 `8`
- `evidence_min_similarity`: BGE-M3 相关度阈值，默认 `0.45`
- `evidence_entity_gate`: 是否要求证据命中中心声明实体，默认 `true`
- `evidence_ranker_multiplier`: 排序候选扩展倍数，默认 `3`

如果过滤后没有证据，系统会返回 `UNVERIFIABLE`，避免 LLM 在无关网页上强行生成结论。

## 🔄 工作流程

1. **节点 1：新闻中心声明抽取** - 对应论文第三章，使用 LLM 生成中心思想、核心可验证声明和关键实体
2. **节点 2：查询改写与层次化证据检索** - 对应论文第四章，使用 LLM 生成多角度查询，通过 DuckDuckGo 检索，并用 BGE 系列嵌入进行语义排序、相关度阈值过滤、实体命中门控和 Top K 选择
3. **节点 3：多步分解声明验证** - 对应论文第五章，使用 LLM 拆解原子声明、抽取 5W 要素、结合证据问答并汇总结论

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
curl -X POST http://localhost:8080/check \
  -H "Content-Type: application/json" \
  -d '{
    "text": "需要核查的新闻内容",
    "api_base": "http://localhost:11434/v1",
    "model": "qwen2.5:3b",
    "embedding_model": "bge-m3:latest",
    "search_engine": "duckduckgo"
  }'
```

详细 API 文档请参见 `docs/api_doc.html`

## 🛠️ 开发指南

### 环境设置

```bash
# 开发环境安装
pip install -r requirements.txt

# 配置文件校验
python -m json.tool model_config.json >/dev/null

# Python 语法检查
python -m py_compile fact_checker.py app.py api.py fact_checking/*.py fact_checking/nodes/*.py

# 启动开发服务器
streamlit run app.py --server.runOnSave true
```

### 本地验证清单

完成配置后，建议确认以下服务可用：

```bash
# Ollama 模型
ollama list

# DuckDuckGo/ddgs 检索
python - <<'PY'
from ddgs import DDGS
with DDGS(timeout=30) as ddgs:
    results = ddgs.text("2026年世界杯决赛结果", backend="auto", region="wt-wt", max_results=3)
print([item["title"] for item in results])
PY

# Streamlit 服务
curl -I http://127.0.0.1:8501/
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
- 🧩 重构为声明抽取、查询改写检索、多步分解验证三节点流水线
- 🔎 DuckDuckGo 默认检索改用 `ddgs`，并增加多后端 fallback
- 🧪 新增证据 Top K、相关度阈值和实体命中门控
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

**Q: DuckDuckGo 返回 No results found**
A: 系统会自动尝试 fallback 后端。如果全部失败，可临时切换到 SearXNG，或调整 `model_config.json` 中 `duckduckgo.fallback_strategies`

**Q: 检索到了网页但结论是无法验证**
A: 这通常表示网页没有通过相关度阈值或实体命中门控。可以检查页面中显示的候选数、保留数、相关度阈值和每条证据分数

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
