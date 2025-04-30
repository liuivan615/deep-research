# deep-research
**Deep Research**

A modular, open-source research automation toolkit for legal and interdisciplinary studies, integrating LLM-driven query generation, multi-source web search, context extraction, and narrative report generation.

---

## 📖 English Version

### 1. Project Overview
Deep Research is an open-source Python framework designed to streamline and automate in-depth research tasks. By combining Large Language Models (LLMs) with customizable search engines and multi-round retrieval strategies, the toolkit supports rapid collection, filtering, and narrative synthesis of information across legal, academic, and technical domains.

### 2. Key Features
- **LLM Integration** (`init_llm_client`): Connect to remote or local LLMs (OpenAI, Ollama, DeepSeek, etc.) with a unified client interface.
- **Smart Query Generation** (`generate_query`, `get_new_search_queries`): Use LLMs to generate and refine search terms over multiple iterations for comprehensive coverage.
- **Multi-Source Web Search** (`web_search`): Query SearXNG (local or fallback public instances) with retry logic, returning top-k links or images.
- **Context Extraction** (`process_link`, `extract_relevant_context`): Scrape page content, clean text, and extract relevant information snippets via LLM evaluation.
- **Parallel Filtering**: Offload large text filtering to lightweight models (e.g., Qwen3:0.6b) in separate threads for efficiency.
- **Narrative Assembly** (`generate_narrative`): Automatically compose structured reports, highlighting "Problem – Solution – Risk Mitigation" sections.
- **Web UI Template** (`index.html`): A ChatGPT-style interface for interactive research sessions, supporting model selection and history.
- **Prompt Templates** (`prompts.py`): System and task-specific prompts for clarification, initial analysis, iterative search, and final report generation.

### 3. Installation
```bash
# Clone repository
git clone https://github.com/your-org/deep-research.git
cd deep-research

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Optional: install sentence-transformers
pip install sentence-transformers
```

### 4. Configuration
1. **LLM Models**: Edit `deepresearch.py` > `MODELS` to configure API keys and endpoints.
2. **Search Engine**: Modify `search_mcp.py` > `SEARXNG_URL` and `SEARXNG_FALLBACKS` for SearXNG instances.
3. **History Storage**: `research_history.json` tracks past sessions.

### 5. Usage
```bash
# Launch research server
python deepresearch.py

# Open the UI
o pen index.html in your browser

# In Python code
from search_mcp import init_llm_client, generate_query, web_search, process_link
from deepresearch import run_research

client = LLMClient('gpt-4o')
init_llm_client(client)
research_id = run_research('Cross-border e-commerce arbitration analysis', 'gpt-4o')
```

### 6. Project Structure
```
├── deepresearch.py       # Core server, threading, research workflow
├── search_mcp.py         # Search, query generation, page processing
├── prompts.py            # LLM prompt templates
├── index.html            # Front-end UI template
├── requirements.txt      # Python dependencies
└── research_data/        # Saved reports & logs
```

### 7. License
This project is licensed under the MIT License.

---

## 📖 中文版

### 1. 项目概述
Deep Research 是一个开源 Python 框架，旨在简化和自动化深入研究流程。该工具包将大型语言模型（LLM）与可定制的检索引擎、多轮检索策略相结合，支持跨法律、学术及技术领域的快速信息收集、过滤与报告生成。

### 2. 核心功能
- **LLM 客户端** (`init_llm_client`): 统一接口对接远程或本地 LLM（OpenAI、Ollama、DeepSeek 等）。
- **智能检索词生成** (`generate_query`, `get_new_search_queries`): 利用 LLM 多轮生成与优化搜索关键词，确保覆盖广泛。  
- **多源网页搜索** (`web_search`): 调用本地或公共 SearXNG 实例，带重试逻辑，返回前 k 条链接或图片。  
- **上下文抽取** (`process_link`, `extract_relevant_context`): 页面抓取、文本清理，并通过 LLM 提取相关信息片段。  
- **并行文本过滤**: 利用轻量级模型（如 Qwen3:0.6b）在独立线程中处理大文本，提高效率。  
- **报告拼接** (`generate_narrative`): 自动生成“问题—方案—风险防控”三段式结构化报告。  
- **Web 界面模板** (`index.html`): 类 ChatGPT 风格的交互式研究前端，支持模型切换与历史记录。  
- **提示模板** (`prompts.py`): 澄清、初步分析、多轮检索及报告生成的系统与任务级提示。

### 3. 安装步骤
```bash
# 克隆仓库
git clone https://github.com/your-org/deep-research.git
cd deep-research

# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
# 可选：安装句子嵌入库
pip install sentence-transformers
```

### 4. 配置指南
1. **LLM 模型**: 在 `deepresearch.py` 的 `MODELS` 中配置 API Key 与模型端点。  
2. **检索引擎**: 修改 `search_mcp.py` 中的 `SEARXNG_URL` 与 `SEARXNG_FALLBACKS`。  
3. **历史记录**: `research_history.json` 会保存历史会话信息。

### 5. 使用示例
```bash
# 启动研究服务
python deepresearch.py

# 打开前端
o pen index.html

# 在 Python 代码中调用
from search_mcp import init_llm_client, generate_query, web_search, process_link
from deepresearch import run_research

client = LLMClient('gpt-4o')
init_llm_client(client)
research_id = run_research('跨境电子商务仲裁分析', 'gpt-4o')
```

### 6. 项目结构
```
├── deepresearch.py       # 核心流程与多线程实现
├── search_mcp.py         # 检索、查询生成、页面处理模块
├── prompts.py            # LLM 提示模板
├── index.html            # 前端界面模板
├── requirements.txt      # 依赖列表
└── research_data/        # 保存的研究报告与日志
```

### 7. 开源协议
本项目遵循 MIT 协议，欢迎自由使用与贡献。

