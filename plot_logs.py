#!/usr/bin/env python3
"""Parse SLTrust experiment .log files and generate paper-ready figures.

This script is intentionally independent of the JSON result files. It parses lines like

    Round  20/100  acc=90.57%  loss=0.3524  D=0.4075

from a downloaded log tree whose top-level directories are ``main_table``,
``main_table_cifar10``, ``ablation``, ``tau``, and ``tau_decay`` variants.

Examples
--------
python plot_logs.py "C:/Users/I604741/Downloads/result (1)"
python plot_logs.py results --output figures_from_logs --formats png pdf
python plot_logs.py results --include-alie --dpi 300

Important limitation: the current logs print accuracy, loss, and consensus diameter D,
but do not print tau_byz. Consequently, tau folders produce accuracy/loss/D figures;
tau trajectories cannot be reconstructed from these text logs.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HEADER_RE = re.compile(
    r"^\[(?P<body>.+?)\]\s+Starting\.\.\.\s*$", re.IGNORECASE
)
NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf))"
ROUND_RE = re.compile(
    rf"^\s*Round\s+(?P<round>\d+)\s*/\s*(?P<total>\d+)"
    rf"\s+acc=(?P<acc>{NUMBER})%"
    rf"\s+loss=(?P<loss>{NUMBER})"
    rf"\s+D=(?P<diameter>{NUMBER})\s*$",
    re.IGNORECASE,
)
SEED_RE = re.compile(r"^seed\s*=\s*(\d+)$", re.IGNORECASE)

DISPLAY_METHOD = {
    "fedavg": "FedAvg (baseline)",
    "sama": "SAMA",
    "balance": "BALANCE",
    "krum": "Krum",
    "trimmed_mean": "Trimmed Mean",
    "contra": "CONTRA (soft EMA)",
    "contra_hard": "CONTRA-Hard (hard cutoff)",
    "sltrust": "SLTrust (ours)",
    "sltrust_m": "SLTrust-M (median, ours)",
    "sltrust_no_ratio": "SLTrust-noRatio (EMA only)",
    "sltrust_short_only": "SLTrust-shortOnly (no long memory)",
    "sltrust_long_only": "SLTrust-longOnly (no short window)",
}
DISPLAY_ATTACK = {
    "none": "No Attack",
    "sign_flip": "Sign Flip",
    "gaussian": "Gaussian Noise",
    "alie": "ALIE",
    "burn_in": "Burn-in / Reputation Farming",
    "oscillating": "Oscillating",
    "slow_poison": "Slow Poison",
    "gradual_flip": "Gradual Flip",
    "burn_in_slow": "BurnIn+SlowPoison",
    "backdoor": "Backdoor",
}
METHOD_ORDER = [
    "fedavg", "sama", "balance", "krum", "trimmed_mean", "contra",
    "contra_hard", "sltrust", "sltrust_m", "sltrust_no_ratio",
    "sltrust_short_only", "sltrust_long_only",
]
ATTACK_ORDER = [
    "none", "sign_flip", "gaussian", "alie", "burn_in", "oscillating",
    "slow_poison", "gradual_flip", "burn_in_slow", "backdoor",
]
# Keep the palette and emphasis rules of the original experiment scripts.
METHOD_COLORS = {
    "fedavg": "#9E9E9E", "sama": "#2196F3", "balance": "#4CAF50",
    "krum": "#9C27B0", "trimmed_mean": "#00BCD4", "contra": "#FF9800",
    "contra_hard": "#FF5722", "sltrust": "#E53935", "sltrust_m": "#B71C1C",
    "sltrust_no_ratio": "#FF9800", "sltrust_short_only": "#2196F3",
    "sltrust_long_only": "#4CAF50",
}
ATTACK_COLORS = {
    "none": "#9E9E9E", "sign_flip": "#E53935", "gaussian": "#FF9800",
    "alie": "#F06292", "burn_in": "#2196F3", "oscillating": "#4CAF50",
    "slow_poison": "#00BCD4", "gradual_flip": "#9C27B0",
    "burn_in_slow": "#795548", "backdoor": "#FF5722",
}


@dataclass(frozen=True)
class Point:
    round: int
    total_rounds: int
    accuracy: float
    loss: float
    diameter: float


@dataclass
class Run:
    path: Path
    group: str
    dataset: str
    method: str
    attack: str
    seed: int
    completed: bool
    points: list[Point]

    @property
    def final(self) -> Point:
        return self.points[-1]


def natural_key(value: str) -> tuple:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", value))


def ordered(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    values = set(values)
    rank = {value: index for index, value in enumerate(preferred)}
    return sorted(values, key=lambda x: (rank.get(x, len(rank)), natural_key(x)))


def dataset_and_group(path: Path, root: Path) -> tuple[str, str]:
    top = path.relative_to(root).parts[0].lower()
    suffix = "_cifar10"
    if top.endswith(suffix):
        return "cifar10", top[:-len(suffix)]
    return "mnist", top


def parse_header(line: str) -> tuple[str, str, int] | None:
    match = HEADER_RE.match(line.strip())
    if not match:
        return None
    fields = [part.strip().lower() for part in match.group("body").split("|")]
    seed_index = next((i for i, field in enumerate(fields) if SEED_RE.match(field)), None)
    if seed_index is None or seed_index < 2:
        return None
    seed = int(SEED_RE.match(fields[seed_index]).group(1))
    # Some headers are [group | method | attack | seed=...], others omit group.
    method, attack = fields[seed_index - 2], fields[seed_index - 1]
    return method, attack, seed


def parse_log(path: Path, root: Path) -> tuple[Run | None, str | None]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return None, f"read error: {exc}"
    header = next((parse_header(line) for line in lines if parse_header(line)), None)
    if header is None:
        return None, "no training header"
    points = []
    for line in lines:
        match = ROUND_RE.match(line)
        if match:
            points.append(Point(
                round=int(match.group("round")),
                total_rounds=int(match.group("total")),
                accuracy=float(match.group("acc")),
                loss=float(match.group("loss")),
                diameter=float(match.group("diameter")),
            ))
    if not points:
        return None, "no Round records"
    method, attack, seed = header
    dataset, group = dataset_and_group(path, root)
    return Run(
        path=path, group=group, dataset=dataset, method=method, attack=attack,
        seed=seed, completed=any("Done ->" in line for line in lines), points=points,
    ), None


def load_runs(root: Path, include_alie: bool) -> tuple[list[Run], list[tuple[Path, str]]]:
    parsed, skipped = [], []
    for path in sorted(root.rglob("*.log")):
        run, reason = parse_log(path, root)
        if run is None:
            skipped.append((path, reason or "unknown"))
        elif run.attack == "alie" and not include_alie:
            skipped.append((path, "ALIE excluded (use --include-alie)"))
        else:
            parsed.append(run)

    # Resolve accidental reruns/checkpoints for the same experimental key. A completed
    # run wins, then the run with more logged rounds, then the newer file.
    best: dict[tuple[str, str, str, str, int], Run] = {}
    for run in parsed:
        key = (run.dataset, run.group, run.method, run.attack, run.seed)
        score = (run.completed, len(run.points), run.final.round, run.path.stat().st_mtime)
        old = best.get(key)
        if old is None:
            best[key] = run
        else:
            old_score = (old.completed, len(old.points), old.final.round, old.path.stat().st_mtime)
            if score > old_score:
                skipped.append((old.path, f"superseded by {run.path.name}"))
                best[key] = run
            else:
                skipped.append((run.path, f"superseded by {old.path.name}"))
    return list(best.values()), skipped


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def method_style(method: str) -> dict:
    return {
        "color": METHOD_COLORS.get(method, "#607D8B"),
        "linestyle": "-" if method in {"sltrust", "sltrust_m"} else "--",
        "linewidth": 2.5 if method in {"sltrust", "sltrust_m"} else 1.8,
    }


def display_method(value: str) -> str:
    return DISPLAY_METHOD.get(value, value.replace("_", " ").title())


def display_attack(value: str) -> str:
    return DISPLAY_ATTACK.get(value, value.replace("_", " ").title())


def aggregate_curve(runs: Sequence[Run], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rounds = np.array(sorted({point.round for run in runs for point in run.points}), dtype=int)
    matrix = np.full((len(runs), len(rounds)), np.nan)
    index = {round_: i for i, round_ in enumerate(rounds)}
    for row, run in enumerate(runs):
        for point in run.points:
            matrix[row, index[point.round]] = getattr(point, metric)
    count = np.sum(~np.isnan(matrix), axis=0)
    sums = np.nansum(matrix, axis=0)
    mean = np.divide(sums, count, out=np.full(len(rounds), np.nan), where=count > 0)
    std = np.zeros(len(rounds))
    for column in range(len(rounds)):
        finite = matrix[:, column][np.isfinite(matrix[:, column])]
        if len(finite) > 1:
            std[column] = finite.std(ddof=1)
    return rounds, mean, std, count


def save_figure(fig: plt.Figure, base: Path, formats: Sequence[str], dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(base.with_suffix(f".{extension}"), dpi=dpi)
    plt.close(fig)


def plot_curves(runs: Sequence[Run], metric: str, title: str, output: Path,
                formats: Sequence[str], dpi: int, attack: str | None = None,
                group: str | None = None) -> None:
    groups: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        groups[run.method].append(run)
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in ordered(groups, METHOD_ORDER):
        subset = groups[method]
        x, mean, std, count = aggregate_curve(subset, metric)
        if not np.any(np.isfinite(mean)):
            continue
        style = method_style(method)
        ax.plot(x, mean, label=display_method(method),
                color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], alpha=0.9)
    if attack == "burn_in":
        ax.axvline(x=40, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
        ymin, ymax = ax.get_ylim()
        text_y = ymin + (ymax - ymin) * (0.92 if group in {"tau", "tau_decay"} else 0.05)
        ax.text(40.5, text_y, "attack start", fontsize=9, alpha=0.8)
    elif attack == "oscillating" and group == "tau":
        last_round = max(run.final.total_rounds for run in runs)
        first = True
        for start in range(0, last_round + 1, 25):
            ax.axvspan(start, min(start + 10, last_round), color="red", alpha=0.08,
                       label="attack phase" if first else "_nolegend_")
            first = False
    labels = {
        "accuracy": "Test Accuracy (%)",
        "loss": "Test Loss",
        "diameter": "Consensus Diameter D",
    }
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel(labels[metric], fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        location = "lower right" if metric == "accuracy" else "best"
        ax.legend(fontsize=9, loc=location)
    fig.tight_layout()
    save_figure(fig, output, formats, dpi)


def plot_attack_curves(runs: Sequence[Run], metric: str, title: str, output: Path,
                       formats: Sequence[str], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for attack in ordered((run.attack for run in runs), ATTACK_ORDER):
        selected = [run for run in runs if run.attack == attack]
        x, values, _, _ = aggregate_curve(selected, metric)
        if not np.any(np.isfinite(values)):
            continue
        ax.plot(x, values, label=display_attack(attack),
                color=ATTACK_COLORS.get(attack, "#607D8B"),
                linewidth=1.8, alpha=0.9)
    labels = {
        "accuracy": "Test Accuracy (%)",
        "loss": "Test Loss",
        "diameter": "Consensus Diameter D",
    }
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel(labels[metric], fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend(fontsize=9, loc="lower right" if metric == "accuracy" else "best")
    fig.tight_layout()
    save_figure(fig, output, formats, dpi)


def plot_heatmap(runs: Sequence[Run], title: str, output: Path,
                 formats: Sequence[str], dpi: int) -> None:
    methods = ordered((run.method for run in runs), METHOD_ORDER)
    attacks = ordered((run.attack for run in runs), ATTACK_ORDER)
    values = np.full((len(methods), len(attacks)), np.nan)
    counts = np.zeros_like(values, dtype=int)
    for i, method in enumerate(methods):
        for j, attack in enumerate(attacks):
            selected = [run.final.accuracy for run in runs if run.method == method and run.attack == attack]
            if selected:
                values[i, j], counts[i, j] = np.mean(selected), len(selected)
    valid = values[np.isfinite(values)]
    if not valid.size:
        return
    lower = max(10.0, math.floor(float(valid.min()) / 10.0) * 10.0)
    fig, ax = plt.subplots(figsize=(16, 7))
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=lower, vmax=100)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Test Accuracy (%)")
    colorbar.ax.tick_params(labelsize=10)
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([display_attack(x) for x in attacks], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([display_method(x) for x in methods], fontsize=10)
    for i in range(len(methods)):
        for j in range(len(attacks)):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            threshold_low = lower + (100 - lower) * 0.25
            threshold_high = lower + (100 - lower) * 0.85
            color = "black" if threshold_low < value < threshold_high else "white"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    save_figure(fig, output, formats, dpi)


def plot_summary_bar(runs: Sequence[Run], title: str, output: Path,
                     formats: Sequence[str], dpi: int) -> None:
    methods = ordered((run.method for run in runs), METHOD_ORDER)
    attacks = ordered((run.attack for run in runs), ATTACK_ORDER)
    x = np.arange(len(attacks))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(20, 7))
    for index, method in enumerate(methods):
        values = []
        for attack in attacks:
            match = next((run for run in runs if run.method == method and run.attack == attack), None)
            values.append(match.final.accuracy if match else np.nan)
        offset = (index - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=display_method(method),
               color=METHOD_COLORS.get(method, "#607D8B"), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([display_attack(attack) for attack in attacks],
                       rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_figure(fig, output, formats, dpi)


def plot_ablation_bar(runs: Sequence[Run], title: str, output: Path,
                      formats: Sequence[str], dpi: int) -> None:
    methods = ordered((run.method for run in runs), METHOD_ORDER)
    attacks = ordered((run.attack for run in runs), ATTACK_ORDER)
    x = np.arange(len(attacks))
    width = 0.7 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, method in enumerate(methods):
        values = []
        for attack in attacks:
            match = next((run for run in runs if run.method == method and run.attack == attack), None)
            values.append(match.final.accuracy if match else np.nan)
        offset = (index - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=display_method(method),
                      color=METHOD_COLORS.get(method, "#607D8B"), alpha=0.85)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.1f}%",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([display_attack(attack) for attack in attacks], fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_figure(fig, output, formats, dpi)


def write_csvs(runs: Sequence[Run], skipped: Sequence[tuple[Path, str]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "parsed_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "group", "method", "attack", "seed", "completed",
                         "logged_points", "final_round", "total_rounds", "final_accuracy_pct",
                         "final_loss", "final_diameter", "source_log"])
        for run in sorted(runs, key=lambda r: (r.dataset, r.group, r.method, r.attack, r.seed)):
            writer.writerow([run.dataset, run.group, run.method, run.attack, run.seed,
                             int(run.completed), len(run.points), run.final.round,
                             run.final.total_rounds, run.final.accuracy, run.final.loss,
                             run.final.diameter, str(run.path)])
    buckets: dict[tuple[str, str, str, str], list[Run]] = defaultdict(list)
    for run in runs:
        buckets[(run.dataset, run.group, run.method, run.attack)].append(run)
    with (output / "final_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "group", "method", "attack", "n_seeds", "seeds",
                         "accuracy_mean_pct", "accuracy_std_pct", "loss_mean", "loss_std",
                         "diameter_mean", "diameter_std"])
        for key, subset in sorted(buckets.items()):
            row = list(key) + [len(subset), ";".join(str(r.seed) for r in sorted(subset, key=lambda r: r.seed))]
            for attribute in ("accuracy", "loss", "diameter"):
                vals = np.array([getattr(r.final, attribute) for r in subset])
                row += [float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0]
            writer.writerow(row)
    with (output / "skipped_logs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_log", "reason"])
        writer.writerows((str(path), reason) for path, reason in skipped)


def write_tau_notice(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    text = (
        "The downloaded text logs contain test accuracy, test loss, and consensus diameter D,\n"
        "but do not contain tau_byz. The figures in this directory therefore show only the\n"
        "available metrics and are not tau-evolution or tau-decay curves. True tau figures\n"
        "must be generated from JSON or mechanism traces that persist tau_byz.\n"
    )
    (directory / "README.txt").write_text(text, encoding="utf-8")


def generate_figures(runs: Sequence[Run], output: Path, formats: Sequence[str], dpi: int) -> int:
    by_seed: dict[tuple[str, int], list[Run]] = defaultdict(list)
    for run in runs:
        by_seed[(run.dataset, run.seed)].append(run)
    figure_count = 0
    for (dataset, seed), seed_runs in sorted(by_seed.items()):
        seed_root = output / dataset / f"seed_{seed}"
        main_runs = [run for run in seed_runs if run.group == "main_table"]
        if main_runs:
            main_dir = seed_root / "main_table"
            plot_heatmap(main_runs, f"Final Test Accuracy (%)  |  {dataset.upper()}  seed={seed}",
                         main_dir / "heatmap", formats, dpi)
            plot_summary_bar(main_runs, f"Final Accuracy Summary  |  {dataset.upper()}  seed={seed}",
                             main_dir / "summary_bar", formats, dpi)
            figure_count += 2
            for attack in ordered((run.attack for run in main_runs), ATTACK_ORDER):
                selected = [run for run in main_runs if run.attack == attack]
                title = f"{display_attack(attack)}  |  {dataset.upper()}  seed={seed}"
                plot_curves(selected, "accuracy", title,
                            seed_root / "by_attack" / attack / "all_methods",
                            formats, dpi, attack=attack, group="main_table")
                figure_count += 1
            for method in ordered((run.method for run in main_runs), METHOD_ORDER):
                selected = [run for run in main_runs if run.method == method]
                title = f"{display_method(method)}  |  {dataset.upper()}  seed={seed}"
                plot_attack_curves(selected, "accuracy", title,
                                   seed_root / "by_method" / method / "all_attacks",
                                   formats, dpi)
                figure_count += 1

        ablation_runs = [run for run in seed_runs if run.group == "ablation"]
        if ablation_runs:
            ablation_dir = seed_root / "ablation"
            for attack in ordered((run.attack for run in ablation_runs), ATTACK_ORDER):
                selected = [run for run in ablation_runs if run.attack == attack]
                title = f"Ablation — {display_attack(attack)}  |  {dataset.upper()}  seed={seed}"
                plot_curves(selected, "accuracy", title, ablation_dir / attack,
                            formats, dpi, attack=attack, group="ablation")
                figure_count += 1
            plot_ablation_bar(ablation_runs,
                              f"Ablation Final Accuracy  |  {dataset.upper()}  seed={seed}",
                              ablation_dir / "ablation_bar", formats, dpi)
            figure_count += 1

        for group, directory_name, heading in (
            ("tau", "tau", "Tau Experiment"),
            ("tau_decay", "tau_decay", "Tau-Decay Experiment"),
        ):
            available = [run for run in seed_runs if run.group == group]
            if not available:
                continue
            metric_root = seed_root / "available_metrics" / directory_name
            write_tau_notice(metric_root)
            for attack in ordered((run.attack for run in available), ATTACK_ORDER):
                selected = [run for run in available if run.attack == attack]
                for metric in ("accuracy", "loss", "diameter"):
                    title = f"{heading} — {display_attack(attack)}  |  {dataset.upper()}  seed={seed}"
                    plot_curves(selected, metric, title, metric_root / f"{attack}_{metric}",
                                formats, dpi, attack=attack, group=group)
                    figure_count += 1
    return figure_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SLTrust experiments directly from .log files.")
    parser.add_argument("log_root", type=Path, help="Root directory containing experiment log folders")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: <log_root>/figures_by_seed)")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--include-alie", action="store_true",
                        help="Include legacy ALIE runs (excluded by default)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.log_root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: log root does not exist or is not a directory: {root}", file=sys.stderr)
        return 2
    output = (args.output or (root / "figures_by_seed")).expanduser().resolve()
    configure_style()
    runs, skipped = load_runs(root, args.include_alie)
    if not runs:
        print("error: no parseable training logs found", file=sys.stderr)
        return 1
    write_csvs(runs, skipped, output)
    figure_count = generate_figures(runs, output, args.formats, args.dpi)
    incomplete = sum(not run.completed for run in runs)
    print(f"Parsed {len(runs)} unique runs; skipped {len(skipped)} logs; incomplete selected runs: {incomplete}.")
    print(f"Generated {figure_count} figure groups in {len(args.formats)} format(s): {output}")
    print("Note: tau_byz is absent from the text logs, so true tau-evolution figures were not generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
