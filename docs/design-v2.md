# 项目 2 设计方案 v2：MeetMind（会议语音转纪要 AI Agent）

> 按浩民要求：不做单片机相关。定位为课程与现有简历之外的个人作品，主攻大厂 AI / FDE 岗位；DigitASR 完成后由 @codex 实现。
> 依据：2026 年 FDE/Agent 实习招聘要求（安永 FDE、阿里 Agent 开发实习「有 ASR/语音经验优先」、小红书 Agent 平台等）：RAG、Tool Calling、MCP、评测指标、FastAPI+Docker 部署、开源作品。

## 1. 为什么选它
- 覆盖 FDE 核心能力：语音识别接入 → 大模型结构化抽取 → Agent 工具调用 → RAG 历史会议问答 → MCP → 评测 → 部署
- 与 DigitASR（语音信号处理）形成「从信号到 AI 应用」的连续叙事，面试好讲
- 真实 ToB 场景（会议纪要、待办分派），贴近飞书/钉钉类 FDE 的客户需求

## 2. 一句话
上传会议录音（或实时录音），自动转写并区分说话人，生成结构化纪要（议题 / 结论 / 待办 / 负责人 / 截止时间），可调用工具把待办写入任务表或生成日程，并支持对历史会议提问（"上周关于预算谁负责？"）且附原文时间戳引用。

## 3. 功能（MVP）
1. **语音转写**：faster-whisper（本地）或云 ASR，输出带时间戳分段；可选 pyannote 说话人分离
2. **结构化纪要**：大模型 + JSON Schema 结构化输出，失败重试与字段校验
3. **Agent 工具调用**：`create_todo`、`create_calendar_event`（生成 .ics）、`export_markdown`/`export_docx`、`search_meetings`
4. **RAG 历史会议问答**：转写分块 + 向量（Chroma）+ BM25 混合检索，回答带时间戳引用
5. **MCP Server**：把检索与工具暴露为 MCP，可接入 Cursor / Claude Desktop 等
6. **评测**：自建 20–30 段公开/自录会议样本，统计转写字错率 CER、待办抽取 F1、检索召回率@k、工具调用成功率，附 bad case 分析
7. **服务与界面**：FastAPI 后端 + Web 界面（Streamlit 起步，可升 React），流式输出，Docker 一键启动
8. **模型可切换**：OpenAI 兼容接口，默认 GPT（OpenAI 接口），可配置切换其他兼容模型

## 4. 非目标（首版不做）
多租户权限、真实接入飞书/钉钉开放平台（留接口与 Mock）、移动端。

## 5. 技术栈
Python 3.10+、FastAPI、faster-whisper、（可选）pyannote.audio、Chroma、rank_bm25、MCP Python SDK、Streamlit、pytest、Docker、GitHub Actions、MIT。

## 6. 仓库结构
```text
MeetMind/
├── README.md / LICENSE / pyproject.toml / Dockerfile / docker-compose.yml
├── .github/workflows/ci.yml
├── src/meetmind/
│   ├── asr/         # 转写、说话人分离
│   ├── summarize/   # Prompt 模板、结构化输出与校验
│   ├── agent/       # 工具路由、多轮上下文
│   ├── tools/       # todo / calendar / export / search
│   ├── retrieval/   # 混合检索
│   ├── mcp_server.py
│   └── api.py
├── web/app.py
├── eval/            # 评测数据说明、脚本、报告
├── samples/         # 可公开的示例音频
└── tests/
```

## 7. 里程碑
| 阶段 | 内容 |
|---|---|
| M1 | 音频转写 + 结构化纪要（命令行跑通） |
| M2 | Agent 工具调用 + 历史会议 RAG 问答 + FastAPI + Web 界面 |
| M3 | 评测报告、MCP Server、Docker、CI、README（架构图、Demo GIF、评测表、bad case） |

## 8. 简历要点草稿（数字以实测为准）
**MeetMind｜会议语音转纪要 AI Agent｜独立开发｜开源地址**
- 搭建「语音转写→说话人分离→大模型结构化抽取」链路，自动生成议题/结论/待办/负责人，JSON Schema 校验与失败重试保证输出稳定
- 设计 Function Calling 工具（待办创建、日程生成、纪要导出、历史检索），并封装为 MCP Server 接入主流 AI 客户端
- 构建向量+BM25 混合检索的历史会议问答，回答附时间戳引用；自建评测集，待办抽取 F1 达 XX%，召回率@5 达 XX%
- FastAPI + Docker 部署、GitHub Actions CI，支持 DeepSeek/千问/本地模型切换

## 9. 待确认
1. 大模型已定：默认 GPT（OpenAI 兼容接口），无需再确认；密钥只放本地环境变量
2. 仓库名默认 `MeetMind`，届时在 GitHub 新建空仓（纯英文名）
