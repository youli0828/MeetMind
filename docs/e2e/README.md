# 真实端到端评测

本目录只保存复现脚本和文本结果。原有 `evaluate.py` 和
`ami-es2002a-base/metrics.json` 是此前遗留的 AMI 尝试；后者记录下载阶段
`ReadTimeout`，不是成功的端到端结果。本轮优先中文，跳过 AMI，不报告英语指标。

## 本次实测结果

结果位于 [alimeeting-20261105-base-rerun](alimeeting-20261105-base-rerun/metrics.json)：
300 秒音频，转写 19.847557 秒，RTF 0.0661585，107 个分段；纪要 35.734023 秒，
Schema/来源校验 1 次通过，结构重试 0、传输重试 0；索引和问答共 4.837143 秒。
实际发起 2 次 Chat Completions、2 次 embeddings 请求。

问：“这段会议讨论了哪些主要问题？请依据原文回答。”
答：“会议讨论了手机设计、目标人群、产品功能、市场定位、年龄适应性以及不同年龄段用户的需求差异等问题。”
引用为 00:06.860–04:50.300（分段 0–106），原文和时间核对通过。
该次转写只形成 1 个检索块，未评估多会议排序或召回率；新增小型离线对照见[业务评测](business-eval/README.md)。

[局部参考评分](alimeeting-20261105-base-rerun/cer_subset.json)为
**CER 58.43%（201/344）**，覆盖 7 段、66.53 秒。
这是独立裁切解码的子集 CER，完整连续 300 秒 CER/WER 没有可用的完整对齐参考。
繁简、数字写法和口头语差异也计入当前归一化方案的误差。

坏例：开场参考讨论“公司新出的一款手机产品”，ASR 却转写成“新规管理的主席”，
纪要随后生成同名议题。该问题说明转写错误会传递到纪要，Schema 合法不代表语义正确。
该次未人工修正模型输出，未计算纪要事实准确率；保存的纪要已纳入后续[业务评测](business-eval/README.md)，
报告行动项 token/set F1 和离线检索召回率，并说明单场真实会议及哈希 embedding 的限制。

历史检查输出见 [verification.json](alimeeting-20261105-base-rerun/verification.json)；
当前测试总数统一见[主 README](../../README.md#开发与验证)。
`docs/e2e/` 只保存文本与脚本，不提交音频、权重或密钥。

## 数据与许可

- [AliMeeting / OpenSLR 119](https://www.openslr.org/119/)，Alibaba Group，
  Yu et al., ICASSP 2022，许可为 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。
- 使用官方页面链接的[国内阿里云 Eval 下载源](https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz)。
  仅取压缩包前 64 MiB，提取
  `Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav` 的前 300 秒，
  取通道索引 0，保存为 PCM16、16 kHz、单声道。音频 SHA-256：
  `123c057c80145117cf5efa3d2c985447fdbd48504e33f5ad6bc773e26844e8ff`。
- 局部参考来自公开的
  [playwithmino/alimeeting-eval-8k](https://huggingface.co/datasets/playwithmino/alimeeting-eval-8k)
  元数据，前 300 秒内的 7 个单说话人片段合计 66.53 秒。
  归一化后的参考文本均与
  [Icefall 发布的 AliMeeting Eval 日志](https://huggingface.co/desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7/tree/main/log/greedy_search)
  中的 `ref` 条目完全一致。评分不使用该日志的 `hyp`。
  这是第三方提供的局部参考子集，完整 300 秒的对齐参考尚未取得。
- 与该语料相关的转写、参考摘录及衍生文本结果沿用 CC BY-SA 4.0；项目代码仍按 MIT 许可。

音频、下载分块、参考缓存和本地权重只保存在已 gitignore 的 `data/external/`，
不上传音频，不在 `docs/e2e/` 放音频。ASR 在本机 CPU 执行；转写文本发送到
`.env` 配置的智谱 Chat Completions，转写分块及问题发送到该服务的 embeddings 接口。
SQLite 临时检索数据在被忽略的 `outputs/`，不使用离线假响应。

## 复现（PowerShell）

使用已有 `.env` 配置 `glm-4-flash`、`json_object`、`embedding-3`。
密钥只在进程内读取并用于请求认证；脚本不打印、复制或保存密钥。
复现输出目录必须尚不存在，可自行换用新目录名。

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[dev,asr]"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
& .\.venv\Scripts\python.exe docs/e2e/download_alimeeting.py
& .\.venv\Scripts\python.exe docs/e2e/download_base.py
& .\.venv\Scripts\python.exe docs/e2e/evaluate_alimeeting.py `
  --model-path data/external/faster-whisper-base `
  --output-dir docs/e2e/alimeeting-20261105-base-rerun
& .\.venv\Scripts\python.exe docs/e2e/evaluate_cer_subset.py `
  --run-dir docs/e2e/alimeeting-20261105-base-rerun
& .\.venv\Scripts\ruff.exe check .
& .\.venv\Scripts\ruff.exe format --check .
& .\.venv\Scripts\python.exe -m pytest -q
```

base 权重固定到 Systran/faster-whisper-base 的
`ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`，`model.bin` 的 SHA-256 为
`d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9`。
镜像连接可能超时；下载脚本会保留已完成分块，重新运行可续传，最终核对完整哈希。
本次 Hugging Face 单连接和部分分段请求发生超时，最后一块由
[ModelScope 国内镜像](https://modelscope.cn/models/pengzhendong/faster-whisper-base)
补齐，拼接后通过上述完整哈希校验。复现脚本也包含该镜像作为下载失败后的备用源。

首次运行因 `faster-whisper 1.2.1` 与 PyAV 19.0.2 不兼容，在音频解码处失败：
`av.open()` 不再接受 `metadata_errors`。参见
[上游问题 #1589](https://github.com/SYSTRAN/faster-whisper/issues/1589)。
失败证据保存在 `alimeeting-20261105-base/`，未改写成成功。
已在 ASR 可选依赖中限制 `av>=11,<19`，本轮重跑使用 PyAV 18.1.0。

转写耗时包含模型初始化和实际解码，不含预先下载权重的时间；
RTF = 转写墙钟秒数 / 300。纪要的结构校验尝试、结构重试和 HTTP 传输重试分别记录。
问答使用真实 embedding、Chroma + BM25 混合检索和真实模型回答，
逐条验证引用的分段 ID、起止秒数及原文。
结构和引用合法不代表事实全部正确，也不代表检索召回率或纪要质量已被量化评估。

CER 对 7 个参考片段分别裁切原始 16 kHz 通道 0 音频，使用相同 base / CPU / int8 / zh
配置独立识别。归一化为 NFKC、小写、仅保留 Unicode 字母和数字，不做繁简转换。
按片段求 Levenshtein 字符距离后累加，除以参考字符总数。
该值不等于连续 300 秒转写的 CER；未计算依赖中文分词方案的 WER。
