#!/usr/bin/env python3
"""scripts/collect_results.py — Collect ablation metrics → Markdown table.

Reads the ``results.json`` written by ``train.py`` from each experiment
subdirectory under ``OUTPUT_ROOT`` and prints (or writes) a Markdown table
suitable for copy-pasting into a paper or README.

Usage::

    # Print to stdout
    python scripts/collect_results.py --output_root ./output/ablation

    # Save to file
    python scripts/collect_results.py --output_root ./output/ablation \\
        --save ablation_table.md

    # Include only a subset of metrics columns
    python scripts/collect_results.py --output_root ./output/ablation \\
        --metrics dice iou sens

    # Show all per-fold numbers in addition to mean±std
    python scripts/collect_results.py --output_root ./output/ablation \\
        --show_folds

Output format::

    | # | Experiment               | Dice ↑        | IoU ↑         | Sens ↑        | Spec ↑        | Prec ↑        |
    |---|--------------------------|---------------|---------------|---------------|---------------|---------------|
    | 1 | NBI-only baseline        | 0.7234±0.0312 | 0.6018±0.0291 | 0.7891±0.0403 | 0.9512±0.0067 | 0.7634±0.0287 |
    ...
    | **9** | **+ WLI aux loss**   | **0.8921±0.0124** | ...                                                          |
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Experiment registry — defines canonical order, display name, and key params
# ---------------------------------------------------------------------------

# Each entry: (exp_number, dir_glob_prefix, display_name, ablation_delta)
EXPERIMENTS: List[Tuple[int, str, str, str]] = [
    (1, "exp1_",  "NBI-only baseline",         "single ResNet34, NBI close-up"),
    (2, "exp2_",  "WLI-only baseline",          "single PVTv2-B2, WLI close-up"),
    (3, "exp3_",  "Early fusion (6-ch)",         "concat WLI+NBI → PVTv2-B2"),
    (4, "exp4_",  "Dual-stream + concat",        "two encoders, simple concat"),
    (5, "exp5_",  "+ CMFIM",                     "+ cross-modal attention"),
    (6, "exp6_",  "+ SAM",                       "+ Spatial Alignment Module"),
    (7, "exp7_",  "+ Boundary Head",             "full Phase 1 DSCMFNet"),
    (8, "exp8_",  "Cross-scale (Phase 2)",        "WLI distant + NBI close-up"),
    (9, "exp9_",  "+ WLI aux loss",              "Phase 2 + WLI mask supervision"),
]

# Metric keys in results.json["summary"]  →  (column header, ↑/↓)
METRIC_OPTIONS: Dict[str, Tuple[str, str]] = {
    "dice": ("Dice",  "↑"),
    "iou":  ("IoU",   "↑"),
    "sens": ("Sens",  "↑"),
    "spec": ("Spec",  "↑"),
    "prec": ("Prec",  "↑"),
}

DEFAULT_METRICS = ["dice", "iou", "sens", "spec", "prec"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_exp_dir(output_root: Path, prefix: str) -> Optional[Path]:
    """Return the first directory under *output_root* whose name starts with *prefix*."""
    matches = sorted(output_root.glob(f"{prefix}*"))
    for m in matches:
        if m.is_dir():
            return m
    return None


def _load_results(exp_dir: Path) -> Optional[Dict]:
    """Load and return ``results.json`` from *exp_dir*, or ``None`` if missing."""
    p = exp_dir / "results.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt_mean_std(mean: float, std: float) -> str:
    """Format ``mean±std`` with 4 decimal places."""
    return f"{mean:.4f}±{std:.4f}"


def _bold(text: str) -> str:
    return f"**{text}**"


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_table(
    output_root: Path,
    metrics: List[str],
    show_folds: bool = False,
    bold_best: bool = True,
) -> Tuple[str, List[Dict]]:
    """Build the Markdown ablation table.

    Args:
        output_root: Directory containing ``exp{N}_*`` subdirectories.
        metrics:     Ordered list of metric short-names to include as columns.
        show_folds:  If True, append a per-fold breakdown section after the table.
        bold_best:   If True, bold the best value in each metric column.

    Returns:
        (markdown_string, rows) where rows is the raw data for further processing.
    """
    # ── Collect data ─────────────────────────────────────────────────────────
    rows: List[Dict] = []
    for exp_num, prefix, display_name, delta in EXPERIMENTS:
        exp_dir = _find_exp_dir(output_root, prefix)
        if exp_dir is None:
            rows.append({"num": exp_num, "name": display_name, "delta": delta,
                         "data": None, "exp_dir": None})
            continue
        data = _load_results(exp_dir)
        rows.append({"num": exp_num, "name": display_name, "delta": delta,
                     "data": data, "exp_dir": exp_dir})

    # ── Determine best value per metric (for bolding) ────────────────────────
    best_val: Dict[str, float] = {}
    if bold_best:
        for m in metrics:
            vals = [
                r["data"]["summary"][f"{m}_mean"]
                for r in rows
                if r["data"] is not None
            ]
            if vals:
                best_val[m] = max(vals)   # all our metrics are ↑

    # ── Column headers ────────────────────────────────────────────────────────
    metric_headers = [
        f"{METRIC_OPTIONS[m][0]} {METRIC_OPTIONS[m][1]}"
        for m in metrics
    ]
    headers = ["#", "Experiment"] + metric_headers
    sep_widths = [3, 26] + [15] * len(metrics)
    sep_row = ["-" * w for w in sep_widths]

    def _pad(text: str, width: int, align: str = "<") -> str:
        # Strip markdown bold for width calculation
        plain = re.sub(r"\*\*", "", text)
        padding = max(0, width - len(plain))
        if align == "<":
            return text + " " * padding
        return " " * padding + text

    def _row_str(cells: List[str]) -> str:
        padded = [_pad(c, sep_widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    lines: List[str] = []
    lines.append(_row_str(headers))
    lines.append(_row_str(sep_row))

    # ── Data rows ─────────────────────────────────────────────────────────────
    for row in rows:
        num_str  = str(row["num"])
        name_str = row["name"]

        if row["data"] is None:
            metric_cells = ["—"] * len(metrics)
            # check if dir exists but no results yet
            if row["exp_dir"] is not None:
                metric_cells = ["*(running)*"] * len(metrics)
        else:
            summary = row["data"]["summary"]
            metric_cells: List[str] = []
            for m in metrics:
                mean = summary.get(f"{m}_mean", float("nan"))
                std  = summary.get(f"{m}_std",  float("nan"))
                cell = _fmt_mean_std(mean, std)
                if bold_best and m in best_val and abs(mean - best_val[m]) < 1e-9:
                    cell = _bold(cell)
                    num_str  = _bold(str(row["num"]))
                    name_str = _bold(row["name"])
                metric_cells.append(cell)

        lines.append(_row_str([num_str, name_str] + metric_cells))

    table = "\n".join(lines)

    # ── Optional per-fold breakdown ───────────────────────────────────────────
    if show_folds:
        fold_lines: List[str] = [
            "",
            "## Per-fold breakdown",
            "",
        ]
        for row in rows:
            if row["data"] is None:
                continue
            fold_results = row["data"].get("fold_results", [])
            if not fold_results:
                continue
            fold_lines.append(f"### Exp {row['num']}: {row['name']}")
            fold_lines.append("")
            fold_headers = ["Fold", "Best epoch"] + metric_headers
            fold_sep     = ["-" * 6, "-" * 11] + ["-" * 15] * len(metrics)
            fold_lines.append("| " + " | ".join(fold_headers) + " |")
            fold_lines.append("| " + " | ".join(fold_sep)    + " |")
            for fr in fold_results:
                fold_num = fr.get("fold", "—")
                best_ep  = fr.get("best_epoch", "—")
                metric_map = {
                    "dice": fr.get("best_val_dice", float("nan")),
                    "iou":  fr.get("val_iou",        float("nan")),
                    "sens": fr.get("val_sensitivity", float("nan")),
                    "spec": fr.get("val_specificity", float("nan")),
                    "prec": fr.get("val_precision",   float("nan")),
                }
                cells = [str(fold_num), str(best_ep)] + [
                    f"{metric_map[m]:.4f}" for m in metrics
                ]
                fold_lines.append("| " + " | ".join(cells) + " |")
            fold_lines.append("")
        table += "\n" + "\n".join(fold_lines)

    return table, rows


# ---------------------------------------------------------------------------
# Summary stats block
# ---------------------------------------------------------------------------

def build_summary_block(rows: List[Dict], metrics: List[str]) -> str:
    """Return a short text block summarising missing / completed experiments."""
    completed = [r for r in rows if r["data"] is not None]
    missing   = [r for r in rows if r["data"] is None and r["exp_dir"] is None]
    pending   = [r for r in rows if r["data"] is None and r["exp_dir"] is not None]

    lines = [
        f"Experiments completed : {len(completed)} / {len(rows)}",
    ]
    if pending:
        lines.append("Still running         : " + ", ".join(
            f"#{r['num']}" for r in pending))
    if missing:
        lines.append("Not started           : " + ", ".join(
            f"#{r['num']}" for r in missing))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect DSCMFNet ablation results into a Markdown table.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--output_root",
        default="./output/ablation",
        help="Root directory containing exp{N}_* subdirectories\n"
             "(default: ./output/ablation)",
    )
    p.add_argument(
        "--save",
        default=None,
        metavar="FILE",
        help="Write the Markdown table to FILE instead of stdout.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=list(METRIC_OPTIONS.keys()),
        default=DEFAULT_METRICS,
        metavar="METRIC",
        help="Metrics to include as columns (default: dice iou sens spec prec).\n"
             f"Choices: {list(METRIC_OPTIONS.keys())}",
    )
    p.add_argument(
        "--show_folds",
        action="store_true",
        help="Append a per-fold breakdown section after the summary table.",
    )
    p.add_argument(
        "--no_bold",
        action="store_true",
        help="Disable bolding of the best value in each column.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    if not output_root.exists():
        print(
            f"ERROR: --output_root '{output_root}' does not exist.\n"
            "Run scripts/run_ablation.sh first.",
            file=sys.stderr,
        )
        sys.exit(1)

    table, rows = build_table(
        output_root=output_root,
        metrics=args.metrics,
        show_folds=args.show_folds,
        bold_best=not args.no_bold,
    )

    summary = build_summary_block(rows, args.metrics)

    # ── Header block ─────────────────────────────────────────────────────────
    header = (
        "## DSCMFNet Ablation Study\n\n"
        "> Metrics are mean ± std over 5-fold cross-validation on 47 EGC cases.\n"
        "> Best value per column is **bolded**.\n"
        "> ↑ = higher is better.\n"
    )

    output = f"{header}\n{table}\n\n<!-- {summary} -->\n"

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"Ablation table written → {args.save}")
        # Also print the status summary to terminal
        print(summary)
    else:
        print(output)
        print(summary)


if __name__ == "__main__":
    main()
