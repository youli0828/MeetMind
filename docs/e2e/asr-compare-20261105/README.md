# AliMeeting ASR 模型比较（2026-11-05，北京时间）

使用历史端到端评测的同一段 `R8001_M8004_MS801` 远场通道 0、前 300 秒。
远场文件 SHA-256 与历史记录一致；匹配的 4 路头戴麦克风已从
[官方国内阿里云 Eval 源](https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz)
下载，分别为 `N_SPK8013` 至 `N_SPK8016`，同取原始时间轴 0–300 秒。
近场条件为这 4 路的算术平均、四舍五入为 PCM16，未做峰值归一化、时间平移或按参考选择说话人。
每位说话人的近场与远场 TextGrid 全部时间边界、文本一致，原 7 段参考也全部与官方标注匹配。
来源、处理方式、哈希及核对记录见 [data_provenance.json](data_provenance.json)。

音频、完整压缩包、下载分块、原始标注和模型权重均位于被 gitignore 的 `data/external/`。
本轮只调用本地 ASR，不读取 `.env`，不调用聊天或 embedding API。
[AliMeeting / OpenSLR 119](https://www.openslr.org/119/) 的音频、参考及衍生文本沿用
CC BY-SA 4.0，署名 Alibaba Group / Yu et al., ICASSP 2022；项目代码仍为 MIT。

## 实测结果

比较表见 [comparison.md](comparison.md)，机器可读汇总见 [summary.json](summary.json)。
每组的 `metrics.json` 和 `transcript.json` 为原始实测依据。
本次续跑保留已有的 base 近、远场及 small 远场结果，没有重新运行这些项目。
large-v3-turbo 首次因下载清单遗漏 `preprocessor_config.json` 而在解码开始时报特征维度错误；
补齐同一固定 revision 的 128 维配置并验证 Git blob 哈希后，以新进程重试。
失败没有产生性能指标，也不计入有效测量；原因见
[失败记录](large-v3-turbo-initial-failure.json)。

八组均完成，没有跳过模型。旧子集上，medium 的远场 39.53%、近场 28.20% 最低；
turbo 为 41.28% / 29.65%，但完整转写用时 99.98 / 112.82 秒，低于 medium 的
134.62 / 159.16 秒。base 最快，近、远场 CER 也最高。
扩展无重叠子集仍是 medium 最低；含重叠发言的长前缀则是 turbo 最低
（远场 44.55%、近场 26.42%），不能用单一小范围的排序替代所有场景。
同模型近场 CER 均低于远场，但近场是多路耳麦混合、远场是阵列单通道，
差异同时包含录音方式和通道合并的影响。

最终核验：八组音频、权重和参考哈希、逐段编辑距离、CER 汇总、RTF、耳麦混合及
近远场标注对齐均通过；README 表格与自动生成表格完全一致。
`ruff check .`、`ruff format --check .`、`git diff --check` 均通过，`pytest -q` 为
**86 passed**（1 条既有 Chroma / Python 3.14 弃用警告）。音频和权重已忽略，未执行 commit/push。
实际检查命令和输出见 [verification.json](verification.json)。

## 测量口径

- Windows、Intel Core Ultra 7 255HX、20 个逻辑处理器、约 32 GB 内存。
  faster-whisper 1.2.1、CTranslate2 4.8.5、PyAV 18.1.0；完整版本见各组 `metrics.json`。
- 四个模型均为 CPU / int8 / zh、4 个推理线程、1 个 worker、随机种子 0，
  `vad_filter=True`，其余沿用 faster-whisper 默认值（包括 beam size 5 和 temperature fallback）。
  模型仓库与 faster-whisper 内置名称映射一致，revision 和权重 SHA-256 逐组记录。
- `HF_ENDPOINT=https://hf-mirror.com`。镜像发生 TLS/超时错误后使用 ModelScope 国内备用源；
  权重 SHA-256 与固定 HF revision 的 LFS 哈希一致，小文件核对 Git blob SHA-1。
  音频下载也发生过分块超时，保留成功分块续传后完整取得，未因此跳过任何条件。
- 每个模型/条件使用独立新进程，串行单次测量，计时前下载已全部结束。
  墙钟时间包含 ASR 导入、模型初始化、完整 300 秒解码，不含下载、CER 片段解码和文件写入。
  RTF = 墙钟秒数 / 300。未进行多次重复、热缓存控制或温度控制，细小耗时差异不宜过度解读。
- 峰值内存来自 Windows `GetProcessMemoryInfo.PeakWorkingSetSize`，包含解释器和依赖。
  每组记录完整转写结束时及所有评分完成后的进程峰值；表格采用完整转写结束时的值。

## CER 的三个范围

统一使用 NFKC、小写、仅保留 Unicode 字母与数字；不做繁简转换、数字改写或口头语清理。
各片段的 Levenshtein 字符编辑距离累加后除以参考字符总数，不取逐段 CER 的算术平均。

1. **旧子集（主比较）**：与历史 `cer_subset.json` 完全相同的 7 段、66.53 秒、344 字，
   逐段独立解码。base 远场复现 201/344；完整转写也与历史文本一致。
2. **扩展无重叠子集**：[expanded_references.json](expanded_references.json)，
   官方标注中完整落在 0–300 秒且与其他发言不重叠的全部 9 段，75.45 秒、393 字，逐段独立解码。
   会排除连短促应答也重叠的整句话，因此覆盖范围仍较小。
3. **连续长前缀（辅助指标）**：[prefix_reference.json](prefix_reference.json)，
   独立解码 0–295.17 秒，74 条官方标注、1605 字。最后一句 295.31–303.36 秒跨过截断点，
   无法确定 300 秒前应保留哪些字，故选择之前无句子跨越的完整边界。
   参考按 `(起点, 终点, 说话人)` 拼接，包含重叠发言；该顺序会影响 CER，
   **不是官方重叠感知评分，也不是整段 300 秒 CER**。

所有条件采用相同参考和边界。各组 `metrics.json` 保存原始参考、原始假设、归一化文本、
逐段编辑距离及评分耗时；`transcript.json` 保存实际完整 300 秒转写，未人工修正。
三种 CER 都不能外推为整个 AliMeeting 数据集的指标，也不等于纪要质量。

## 复现

从仓库根目录执行，先安装 `.[dev,asr]`。无需 API key。
最后两步使用一个新的输出目录，以保留已有实测结果：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
& .\.venv\Scripts\python.exe docs/e2e/download_alimeeting.py
& .\.venv\Scripts\python.exe docs/e2e/download_base.py
& .\.venv\Scripts\python.exe docs/e2e/download_asr_comparison.py audio
foreach ($modelName in @('small', 'medium', 'large-v3-turbo')) {
    & .\.venv\Scripts\python.exe docs/e2e/download_asr_comparison.py $modelName
}
& .\.venv\Scripts\python.exe docs/e2e/prepare_asr_comparison.py `
    --output-dir docs/e2e/asr-compare-local
foreach ($modelName in @('base', 'small', 'medium', 'large-v3-turbo')) {
    foreach ($audioCondition in @('far', 'near')) {
        & .\.venv\Scripts\python.exe docs/e2e/compare_asr.py `
            --model $modelName --condition $audioCondition `
            --output-dir docs/e2e/asr-compare-local
        if ($LASTEXITCODE -ne 0) { throw "ASR comparison failed" }
    }
}
& .\.venv\Scripts\python.exe docs/e2e/summarize_asr_comparison.py `
    --output-dir docs/e2e/asr-compare-local
```

`compare_asr.py` 会跳过 `status=completed` 的已有结果；空目录可直接续跑。
完整转写、旧子集、扩展子集分别完成后立即保存阶段指标，后续运行采用临时文件加原子替换；
所有评分完成才标记 `completed`。如目录已有未完成的 `metrics.json`，脚本会停止以保护记录，
应保留该目录并改用新目录重跑该组，再明确记录采用哪次测量。
汇总脚本可随时重新执行，只读取实测文件；未完成项目明确标记，不估算缺失指标。
