"""Build comparison tables from saved measurements, including partial runs."""

import argparse
import json
from pathlib import Path

MODELS = ["base", "small", "medium", "large-v3-turbo"]
CONDITIONS = ["far", "near"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            path = args.output_dir / f"{model}-{condition}" / "metrics.json"
            if not path.exists():
                rows.append({"model": model, "condition": condition, "status": "pending"})
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "model": model,
                "condition": condition,
                "status": metrics["status"],
                "metrics_file": path.relative_to(args.output_dir).as_posix(),
                "wall_seconds": metrics["transcription_wall_seconds"],
                "rtf": metrics["real_time_factor"],
                "continuous_peak_working_set_bytes": metrics["continuous_peak_working_set_bytes"],
            }
            for scope in ["cer_subset", "cer_expanded", "cer_long_prefix"]:
                if scope in metrics:
                    row[scope] = {
                        key: metrics[scope][key]
                        for key in ["cer", "edit_distance", "reference_characters"]
                    }
            rows.append(row)
    summary = {
        "status": "completed" if all(r["status"] == "completed" for r in rows) else "partial",
        "audio_seconds": 300,
        "device": "cpu",
        "compute_type": "int8",
        "timing_scope": "ASR import + initialization + continuous 300s decode; "
        "excludes downloads and CER scoring",
        "rows": rows,
    }
    lines = [
        "| 模型 | 远场 CER | 远场耗时(s) | 远场 RTF | 近场 CER | 近场耗时(s) | 近场 RTF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        cells = [model]
        for condition in CONDITIONS:
            row = next(r for r in rows if r["model"] == model and r["condition"] == condition)
            if row["status"] != "completed":
                cells.extend([row["status"], "—", "—"])
            else:
                cells.extend(
                    [
                        f"{row['cer_subset']['cer']:.2%}",
                        f"{row['wall_seconds']:.2f}",
                        f"{row['rtf']:.3f}",
                    ]
                )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "主表 CER：原 7 段、66.53 秒、344 字子集，逐段独立解码。耗时为完整 300 秒转写，",
            "含导入及初始化，不含下载和 CER 评分；RTF = 耗时 / 300。近场为 4 路耳麦等权混合。",
            "",
            "| 模型 | 条件 | 扩展无重叠 CER (393字) | 长前缀 CER (1605字) | 转写峰值内存(MiB) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["status"] != "completed":
            continue
        memory = row["continuous_peak_working_set_bytes"]
        memory_text = f"{memory / 1024**2:.1f}" if memory is not None else "未测量"
        lines.append(
            f"| {row['model']} | {'远场' if row['condition'] == 'far' else '近场'} | "
            f"{row['cer_expanded']['cer']:.2%} | {row['cer_long_prefix']['cer']:.2%} | "
            f"{memory_text} |"
        )
    lines.extend(
        [
            "",
            "扩展无重叠子集为 9 段、75.45 秒。长前缀为 0–295.17 秒，包含重叠发言，",
            "按发言起点拼接参考，顺序会影响 CER；不是官方重叠感知评分，也不是整段 300 秒 CER。",
            "峰值内存包含 Python 与依赖；每组使用独立进程，单次测量。",
        ]
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary["status"], sum(r["status"] == "completed" for r in rows), "/", len(rows))


if __name__ == "__main__":
    main()
