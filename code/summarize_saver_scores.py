#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from saver_agent.metrics import summarize_saver_metrics
from saver_agent.offline_scoring import load_rollout_records
from saver_agent.offline_scoring import ReferenceDataProvider
from saver_agent.runtime import distributed_runtime_from_env, runtime_log
from saver_agent.score_summary import summarize_scored_rollouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize scored SAVER rollout files into verifier-status distributions and mean rewards."
    )
    parser.add_argument("--input", required=True, help="Input scored rollout path (.json, .jsonl, or directory).")
    parser.add_argument("--output", default="", help="Optional summary JSON output path.")
    parser.add_argument("--data", default="", help="Optional raw saver_agent/oracle JSONL used to compute full SAVER metrics.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths for metric computation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = distributed_runtime_from_env()
    if runtime.is_distributed and not runtime.is_main_process:
        runtime_log("summarize runs on the main process only; skipping duplicate worker.", runtime=runtime)
        return

    runtime_log(f"loading scored rollouts from {args.input}", runtime=runtime)
    try:
        records, _ = load_rollout_records(args.input)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    summary = summarize_scored_rollouts(records)
    if args.data:
        reference_data = ReferenceDataProvider(data_path=args.data, data_root=args.data_root)
        summary = {
            **summary,
            "saver_metrics": summarize_saver_metrics(
                records,
                reference_data=reference_data,
            ),
        }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        runtime_log(f"summarized {len(records)} records to {output_path}", runtime=runtime)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
