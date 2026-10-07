#!/usr/bin/env python3
"""生成视频工作流四策略 A/B 评估 JSON 与 Markdown 报告。

示例::

    python server/scripts/video_ab_evaluate.py \
        --input server/tests/fixtures/video_ab_evaluation/fixture.json \
        --output-dir output/video-ab-evaluation

本命令是离线报告工具：只读输入、只写输出目录，不调用视频 Provider，也不修改
默认 Provider、质量阈值或其它线上生成策略。
"""

from __future__ import annotations

import argparse
import sys

from services.video_ab_evaluation import (  # noqa: E402
    VideoABEvaluationError,
    build_evaluation_report,
    load_evaluation_payload,
    write_evaluation_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成视频工作流 A/B 评估报告（离线、只读、无策略自动修改）")
    parser.add_argument("--input", required=True, help="A/B 评估输入 JSON")
    parser.add_argument("--output-dir", required=True, help="JSON 与 Markdown 报告输出目录")
    parser.add_argument("--json-name", default="video_ab_evaluation.json", help="JSON 报告文件名")
    parser.add_argument("--markdown-name", default="video_ab_evaluation.md", help="Markdown 报告文件名")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = load_evaluation_payload(args.input)
        report = build_evaluation_report(payload)
        json_path, markdown_path = write_evaluation_reports(
            report,
            args.output_dir,
            json_name=args.json_name,
            markdown_name=args.markdown_name,
        )
    except VideoABEvaluationError as exc:
        print(f"VIDEO_AB_EVALUATION_FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"VIDEO_AB_EVALUATION_OK\nJSON: {json_path}\nMARKDOWN: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
