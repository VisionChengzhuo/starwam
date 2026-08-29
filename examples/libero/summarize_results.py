"""Aggregate LIBERO rollout ``results.json`` files across suites.

Each ``rollout.py`` run writes one ``results.json`` per suite under
``<output_dir>/libero_rollout/<checkpoint_name>/<suite_name>/results.json``.
This script scans those per-suite files, prints a per-suite table plus the
micro-average success rate across all suites, and (optionally) writes a
combined ``summary.json``.

Example:

    python examples/libero/summarize_results.py \
      --rollout-dir /path/to/output/<recipe>/libero_rollout/<checkpoint-name>

    # Or point directly at specific results.json files:
    python examples/libero/summarize_results.py \
      --results a/results.json b/results.json --output summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Conventional LIBERO suite ordering for stable, readable output.
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _find_results_files(rollout_dir: Path) -> list[Path]:
    """Return all ``results.json`` files directly under each suite subdir."""
    files = sorted(rollout_dir.glob("*/results.json"))
    if not files:
        # Fall back to a recursive search in case of a deeper layout.
        files = sorted(rollout_dir.rglob("results.json"))
    return files


def _suite_sort_key(name: str) -> tuple[int, str]:
    return (SUITE_ORDER.index(name) if name in SUITE_ORDER else len(SUITE_ORDER), name)


def load_suite_results(paths: list[Path]) -> list[dict[str, Any]]:
    suites: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("task_suite_name") or path.parent.name
        successes = int(data.get("total_successes", 0))
        trials = int(data.get("total_trials", 0))
        rate = data.get("success_rate")
        if rate is None:
            rate = successes / trials if trials else 0.0
        suites.append(
            {
                "task_suite_name": name,
                "total_successes": successes,
                "total_trials": trials,
                "success_rate": float(rate),
                "checkpoint": data.get("checkpoint"),
                "num_trials": data.get("num_trials"),
                "results_path": str(path),
            }
        )
    suites.sort(key=lambda s: _suite_sort_key(s["task_suite_name"]))
    return suites


def build_summary(suites: list[dict[str, Any]]) -> dict[str, Any]:
    total_successes = sum(s["total_successes"] for s in suites)
    total_trials = sum(s["total_trials"] for s in suites)
    micro = total_successes / total_trials if total_trials else 0.0
    checkpoints = {s["checkpoint"] for s in suites if s.get("checkpoint")}
    return {
        "checkpoint": next(iter(checkpoints)) if len(checkpoints) == 1 else sorted(checkpoints),
        "num_suites": len(suites),
        "per_suite": suites,
        "total_successes": total_successes,
        "total_trials": total_trials,
        "success_rate": micro,
    }


def format_table(summary: dict[str, Any]) -> str:
    rows = [("Suite", "Success", "Trials", "Success rate")]
    for s in summary["per_suite"]:
        rows.append(
            (
                s["task_suite_name"],
                str(s["total_successes"]),
                str(s["total_trials"]),
                f"{s['success_rate'] * 100:.1f}%",
            )
        )
    rows.append(
        (
            "Overall (micro)",
            str(summary["total_successes"]),
            str(summary["total_trials"]),
            f"{summary['success_rate'] * 100:.1f}%",
        )
    )
    widths = [max(len(r[c]) for r in rows) for c in range(4)]
    lines = []
    for i, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[c]) for c, cell in enumerate(row))
        lines.append(line)
        if i == 0 or i == len(rows) - 2:
            lines.append("  ".join("-" * widths[c] for c in range(4)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--rollout-dir",
        type=str,
        help="Directory containing per-suite subdirs, each with a results.json "
        "(e.g. <output_dir>/libero_rollout/<checkpoint-name>).",
    )
    group.add_argument(
        "--results",
        type=str,
        nargs="+",
        help="Explicit list of results.json file paths to aggregate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the combined summary.json. "
        "Defaults to <rollout-dir>/summary.json when --rollout-dir is used.",
    )
    args = parser.parse_args()

    if args.rollout_dir:
        rollout_dir = Path(args.rollout_dir)
        paths = _find_results_files(rollout_dir)
        if not paths:
            raise SystemExit(f"No results.json found under {rollout_dir}")
        default_output = rollout_dir / "summary.json"
    else:
        paths = [Path(p) for p in args.results]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"results.json not found: {', '.join(missing)}")
        default_output = None

    suites = load_suite_results(paths)
    summary = build_summary(suites)

    print(format_table(summary))

    output_path = Path(args.output) if args.output else default_output
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
