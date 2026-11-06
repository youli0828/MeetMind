# MeetMind 路线图

设计范围以 [design-v2.md](design-v2.md) 为准；原始设计保留不改。

| 阶段 | 状态 | 已完成 / 下一步 |
|---|---|---|
| M1 | 完成 | CLI 音频/转写输入、结构化纪要、来源校验和有限重试、JSON/Markdown、离线样例；原有 37 项测试保留 |
| M2 | 完成 | 有界多轮工具调用、持久化待办、ICS/Markdown/DOCX、转写分块、Chroma + BM25/RRF、原文时间戳引用问答、FastAPI、Web 交互与 SSE 工具进度、离线演示及新增测试 |
| M3 | 已交付，小样本评测 | MCP stdio/SSE、Docker/Compose、CI 镜像构建、真实 CER 对比和 bad case；[业务评测](e2e/business-eval/README.md)已测离线 Recall@1/@2/@3=1.000（2 块/8 问）、已保存真实纪要 token 集合 F1=0.6437（单场/3 项）。线上 embedding、大样本和工具成功率仍 TODO |

M2 不包含多租户、外部任务平台写入、系统日历同步、实时音频或说话人分离。
Web 使用 FastAPI 直接服务的 HTML/JavaScript，避免额外的 Streamlit 进程；符合 M2 的 Web 界面范围。
SSE 是工具进度与最终消息事件流，尚无逐 token 输出。
检索及长会议摘要已分块；日期范围过滤及检索质量拒答阈值待后续验证。

离线演示只使用哈希向量、原文摘录与固定 Agent 搜索流程；测试中的模型响应为 mock。
真实 Chroma 本地存储通过离线测试；历史真实 ASR、聊天和 embeddings 端到端结果见 [e2e](e2e/README.md)。
不得把测试通过数解释为模型质量，也不得填写未经实测的评测指标。

当前本地验收与唯一测试总数见 [README](../README.md#开发与验证)。
历史 CLI 离线 demo 与 wheel 资源检查通过。Edge 离线浏览器流程覆盖导入、引用、待办、DOCX/ICS 下载、
Agent SSE；桌面及移动端布局检查通过。GitHub Actions 已运行 Python 3.10/3.12/3.14 检查及 Docker 构建。
## 12.1 可选说话人分离

已实现独立可 mock 的 pyannote 后端及 ASR 后按时间重叠归属，默认关闭。
需安装 `.[diarization]` 并配置 HF 令牌及模型访问权限；未满足条件明确失败。
本机缺少令牌及 pyannote，尚无真实分离质量指标，见 [验证记录](e2e/optional-diarization.md)。

## 12.2 长会议摘要

超过阈值时按时间顺序分块，默认重叠两段，逐块结构化摘要后合并去重、保留来源引用。
各块采用原校验重试策略；合并失败保留逐块纪要。CLI/API 共用摘要器，短会议路径保持不变。
配置、文件位置、极长会议上下文限制见 [README](../README.md#可选说话人分离与长会议摘要)。
新增验证使用伪造长转写和 mock 模型，不代表真实长会议摘要质量。
