# Super-Memory-for-openwebui
作者：南风的AI进化日记。知乎，b站，抖音同号

超级记忆助手，专为中文用户优化。现在它已经被**完全重构为 Model Context Protocol (MCP) 服务器**，遵循 OpenAI 在 basic-memory 示例中采用的模块化设计理念，可直接对接 Open WebUI MCP 运行时或其他兼容主机。

## 主要特性

1. 自动分析并且存储记忆，分为 fact 模式（直接提取需要记忆的事实写入记忆）和 summary 模式（以 AI 自己的视角总结记忆写入）。
2. 为每一条新存入的记忆，都打上一个精确到分钟的时间戳，让 AI 知道每段记忆记住的时间。
3. 可通过 `summarise_clusters` 工具定期生成记忆摘要，它像一个 **“深思熟虑的图书管理员”** 主动巡视整个记忆库，将关于同一主题的零散记忆（比如十几条关于咖啡的记录）找出来，用 LLM 提炼成一条高质量的精华摘要。
4. 拥有查重功能，不会重复记忆已有的事实，并可选启用语义相似度判定（依赖 sentence-transformers）。
5. 智能事实整合自动更新记忆：当您说出与过去记忆矛盾的新话时，它会立即调整，保证记忆库的**实时准确性**。
6. 全面的性能数据：处理流程会返回首字时间、总耗时、生成速度 (TPS) 和输出的 Token 数。

## 与 basic-memory 的关系

* basic-memory MCP 展示了「轻量工具 + 清晰数据模型」的最佳实践。本项目在此基础上进一步拆分为 `config`、`backend`、`llm`、`orchestrator` 等模块，便于二次开发。
* 所有提示词、语义去重与聚类逻辑都重新实现为可复用的服务，方便将来替换 LLM 或持久化后端。

## 目录结构

```
src/super_memory_mcp/
├── __init__.py
├── __main__.py          # python -m super_memory_mcp 入口
├── backend.py           # OpenWebUI / 本地 JSON 存储后端
├── config.py            # 环境变量加载与默认值
├── embeddings.py        # 语义向量工具（可选）
├── llm.py               # OpenAI 兼容 LLM 封装
├── models.py            # 数据模型
├── orchestrator.py      # 核心业务逻辑
├── prompts.py           # 提示词模板
└── server.py            # MCP 服务器注册入口
```

## 快速开始

1. **安装依赖**（可选，视语义搜索需求而定）：

   ```bash
   pip install sentence-transformers numpy aiohttp pytz
   ```

2. **配置环境变量**（示例）：

   ```bash
   export SUPER_MEMORY_API_KEY="sk-..."
   export SUPER_MEMORY_API_URL="https://api.openai.com/v1/chat/completions"
   export SUPER_MEMORY_MODEL="gpt-4o-mini"
   export SUPER_MEMORY_STORAGE_PATH="~/openwebui/memories.json"
   ```

   更多变量可参考 `src/super_memory_mcp/config.py`。

3. **启动 MCP 服务器**：

   ```bash
   python -m super_memory_mcp
   ```

   运行后即可通过 Open WebUI 的 MCP 客户端或其他兼容工具注册该服务。

4. **在主机中调用**：

   * `process_conversation` 工具：输入用户 ID 与消息列表，返回新增记忆与性能数据。
   * `summarise_clusters` 工具：对历史记忆执行聚类并生成摘要。
   * `status` 工具：查看当前配置开关。

## 开发提示

* 若要与 Open WebUI 的记忆 API 打通，可设置 `SUPER_MEMORY_OPENWEBUI_BASE_URL` 与 `SUPER_MEMORY_OPENWEBUI_TOKEN`，此时会自动切换到 REST 后端。
* 默认情况下使用本地 JSON 文件存储，便于开发调试。
* MCP 工程结构遵循 basic-memory 的分层设计，所有模块均提供类型注解，方便自定义或扩展。


