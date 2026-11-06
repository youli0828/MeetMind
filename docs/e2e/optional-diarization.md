# 可选说话人分离验证（2026-11-07）

实现覆盖 CLI `run`、API `/audio` 的 ASR 后分离，默认 `DIARIZATION=0`。
采用 [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
及 pyannote.audio 4 的 `token=` / `output.speaker_diarization` 接口。

本机检查只输出配置是否存在，不输出令牌或 `.env` 内容：

- `HUGGINGFACE_TOKEN` / `HF_TOKEN` 在进程环境和本机配置文件中均未配置。
- 当前虚拟环境未安装 pyannote。
- 本机存在既有 AliMeeting 近场音频：
  `data/external/alimeeting/R8001_M8004.headset-mean.first300s.wav`。

因此本次未下载或运行分离模型，未做真实分离前后对照，**没有可报告的 after speaker
fill-rate 或 DER**。不得将默认空标签、mock 标签或历史 CER 当作分离质量结果。
已交付可选功能与离线 mock 测试，包括重叠归属、默认关闭、启用路径、缺少令牌/包、
pyannote API 映射及敏感错误屏蔽；单元测试通过数不代表分离准确率。

补测方式：安装 `.[asr,diarization]`，接受模型访问条件，配置令牌后，对上述同一音频
使用相同 ASR 配置分别设置 `DIARIZATION=0` 与 `1`，输出到两个新的 `outputs/` 目录。
以各自 `transcript.json` 中非空 `speaker` 的段数除以总段数计算 fill-rate，
同时记录音频标识、配置及段数；fill-rate 仅衡量标签覆盖，不是说话人正确率。
DER 需要另行准备时间对齐的真实说话人参考和明确评分协议，本次不报告。
音频、模型权重和令牌均不纳入 Git。
