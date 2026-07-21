import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pf_ablation import read_ablation_jsonl, summarize_ablation_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize per-feature PF accuracy, stability, and runtime records.",
    )
    parser.add_argument("recording", type=Path, help="pf_ablation_*.jsonl recording")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Discard this many initial frames independently after every PF reset/revision.",
    )
    parser.add_argument("--output", type=Path, help="Optional machine-readable JSON summary.")
    return parser.parse_args()


def compact_changes(changes):
    if not changes:
        return "configured"
    return ", ".join(f"{key}={'on' if value else 'off'}" for key, value in changes.items())


def print_summary(summary):
    print(
        f"Ablation summary | warm-up={summary['warmup_frames_per_revision']} frames/revision"
    )
    for revision in summary["revisions"]:
        print(
            f"\nrevision {revision['revision']} | retained "
            f"{revision['retained_frame_count']}/{revision['frame_count']} | "
            f"{compact_changes(revision['changes_from_configured'])}"
        )
        for name, values in revision["metrics"].items():
            comparison = revision.get("comparison_to_baseline", {}).get(name, {})
            delta = comparison.get("median_delta")
            delta_text = "" if delta is None else f" delta={delta:+.6g}"
            print(
                f"  {name:34s} n={values['count']:4d} "
                f"median={values['median']:.6g} mean={values['mean']:.6g} "
                f"std={values['std']:.6g} p95={values['p95']:.6g}{delta_text}"
            )


def main():
    args = parse_args()
    metadata, frames = read_ablation_jsonl(args.recording)
    summary = summarize_ablation_records(
        metadata,
        frames,
        warmup_frames=max(0, int(args.warmup_frames)),
    )
    print_summary(summary)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
