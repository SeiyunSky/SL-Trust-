"""Generate formal figures from verified schema-v3 result JSON without Torch.

Only ``schema_version=3``, ``candidate-model-state-v3`` records produced with the
current SLTrust window (S=10, L=30) are eligible.  The script never reads text
logs and writes a manifest describing every rejected or missing comparison cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCHEMA_VERSION = 3
PROTOCOL = "candidate-model-state-v3"
EXPERIMENT_CONTRACT = "sltrust-dfl-2026-10-current-attacks-sl10-l30"
METHODS = ["fedavg", "krum", "trimmed_mean", "balance", "sama", "contra", "contra_hard", "sltrust"]
ATTACKS = [
    "none", "gaussian", "label_flipping", "alie", "broken_node", "burn_in",
    "oscillating", "gradual_gaussian", "direction_perturbation",
]
METHOD_LABELS = {
    "fedavg": "FedAvg", "krum": "Krum", "trimmed_mean": "Trimmed Mean",
    "balance": "BALANCE", "sama": "SAMA", "contra": "CONTRA",
    "contra_hard": "CONTRA-Hard", "sltrust": "SLTrust",
}
ATTACK_LABELS = {
    "none": "No attack", "gaussian": "Gaussian", "label_flipping": "Label flipping",
    "alie": "ALIE", "broken_node": "Broken node", "burn_in": "Burn-in Gaussian",
    "oscillating": "Oscillating Gaussian", "gradual_gaussian": "Gradual Gaussian",
    "direction_perturbation": "Direction perturbation",
}
COLORS = {
    "fedavg": "#6B6B6B", "krum": "#8C564B", "trimmed_mean": "#9467BD",
    "balance": "#4C78A8", "sama": "#2A9D8F", "contra": "#F2A541",
    "contra_hard": "#E76F51", "sltrust": "#C62828",
}

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def _seed(meta: dict) -> int | None:
    try:
        return int(meta["config"]["experiment"]["seed"])
    except (KeyError, TypeError, ValueError):
        return None


def _current_window(meta: dict) -> bool:
    """Reject stale SLTrust runs rather than silently treating them as current."""
    if meta.get("method") not in {"sltrust", "sltrust_median", "sltrust_m"}:
        return True
    config = meta.get("config", {}).get("method_config", {})
    return config.get("S") == 10 and config.get("L") == 30


def _read_record(path: Path, dataset: str) -> tuple[dict | None, str | None]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"unreadable: {exc}"
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("protocol") != PROTOCOL:
        return None, "not schema-v3 candidate-model-state-v3"
    if meta.get("experiment_contract") != EXPERIMENT_CONTRACT:
        return None, "not produced under current experiment contract"
    if meta.get("config", {}).get("data", {}).get("dataset", "").lower() != dataset:
        return None, "different dataset"
    if not _current_window(meta):
        return None, "stale SLTrust window (requires S=10,L=30)"
    acc = history.get("test_acc")
    if not isinstance(acc, list) or not acc or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in acc):
        return None, "invalid test_acc history"
    if not (method := meta.get("method")) or not (attack := meta.get("attack")) or _seed(meta) is None:
        return None, "missing method, attack, or seed metadata"
    return {"meta": meta, "history": history, "path": path, "method": method,
            "attack": attack, "seed": _seed(meta)}, None


def collect(root: Path, dataset: str, methods: list[str], attacks: list[str]):
    """Return valid records keyed by seed/method/attack plus a rejection manifest."""
    records, manifest = {}, []
    for path in root.rglob("*.json"):
        if path.name.endswith(".meta.json"):
            continue
        record, reason = _read_record(path, dataset)
        if reason:
            manifest.append((str(path), reason))
            continue
        key = (record["seed"], record["method"], record["attack"])
        if record["method"] not in methods or record["attack"] not in attacks:
            manifest.append((str(path), "outside requested formal comparison"))
        elif key in records:
            manifest.append((str(path), f"duplicate semantic result for {key}; existing={records[key]['path']}"))
        else:
            records[key] = record
    return records, manifest


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _round_axis(record: dict) -> np.ndarray:
    interval = record["meta"]["config"].get("logging", {}).get("log_interval", 1)
    return np.arange(1, len(record["history"]["test_acc"]) + 1) * int(interval)


def plot_seed(seed: int, records: dict, methods: list[str], attacks: list[str], out: Path, manifest: list):
    target = out / f"seed_{seed}"
    for attack in attacks:
        row = [records.get((seed, method, attack)) for method in methods]
        if not all(row):
            manifest.append((f"seed={seed},attack={attack}", "incomplete method comparison; figure omitted"))
            continue
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        for method, record in zip(methods, row):
            ax.plot(_round_axis(record), record["history"]["test_acc"], label=METHOD_LABELS.get(method, method),
                    color=COLORS.get(method), linewidth=2.2 if method == "sltrust" else 1.35)
        ax.set(title=f"{ATTACK_LABELS.get(attack, attack)} (seed {seed})", xlabel="Communication round", ylabel="Test accuracy (%)", ylim=(0, 100))
        ax.grid(axis="y", color="#D9D9D9", linewidth=.55)
        ax.legend(ncol=2, fontsize=7.5)
        _save(fig, target / f"accuracy_{attack}")


def plot_complete_heatmap(seeds: list[int], records: dict, methods: list[str], attacks: list[str], out: Path, manifest: list):
    complete = [attack for attack in attacks if all((seed, method, attack) in records for seed in seeds for method in methods)]
    for attack in attacks:
        if attack not in complete:
            manifest.append((f"cross-seed,attack={attack}", "incomplete seed/method comparison; excluded from heatmap"))
    if not complete:
        return
    data = np.array([[np.mean([records[(seed, method, attack)]["history"]["test_acc"][-1] for seed in seeds])
                      for attack in complete] for method in methods])
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(complete)), 4.8))
    image = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    fig.colorbar(image, ax=ax, label="Final test accuracy (%)")
    ax.set(title=f"Final accuracy, complete cases (n={len(seeds)} seeds)",
           xticks=np.arange(len(complete)), xticklabels=[ATTACK_LABELS[a] for a in complete],
           yticks=np.arange(len(methods)), yticklabels=[METHOD_LABELS.get(m, m) for m in methods])
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right")
    for i in range(len(methods)):
        for j in range(len(complete)):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if data[i, j] < 35 else "black")
    _save(fig, out / "cross_seed" / "final_accuracy_heatmap_complete_cases")


def _mechanism_series(record: dict) -> dict[str, dict[str, list[float]]] | None:
    """Validate and summarize dedicated edge traces; never infer them from logs."""
    trace = record['history'].get('mechanism_trace')
    expected = int(record['meta']['config']['federated']['num_rounds'])
    rounds = trace.get('rounds') if isinstance(trace, dict) and trace.get('version') == 1 else None
    if not isinstance(rounds, list) or len(rounds) != expected:
        return None
    result = {role: {metric: [] for metric in ('phi', 'ratio', 'tau', 'gamma')}
              for role in ('byzantine', 'honest')}
    for index, entry in enumerate(rounds, 1):
        if not isinstance(entry, dict) or entry.get('round') != index or not isinstance(entry.get('edges'), list):
            return None
        edges = entry['edges']
        # A reputation method must have observer-edge values on every round.
        if not edges:
            return None
        for role, is_byzantine in (('byzantine', True), ('honest', False)):
            role_edges = [edge for edge in edges if edge.get('is_byzantine') is is_byzantine]
            if not role_edges:
                return None
            for metric in result[role]:
                values = [edge.get(metric) for edge in role_edges]
                if not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in values):
                    return None
                result[role][metric].append(float(np.mean(values)))
    return result


def plot_mechanism(records: dict, out: Path, manifest: list) -> None:
    """Plot phi/ratio/tau/gamma only where full dedicated traces exist."""
    for (seed, method, attack), record in records.items():
        if method not in {'sltrust', 'contra', 'contra_hard'}:
            continue
        series = _mechanism_series(record)
        item = f'seed={seed},method={method},attack={attack}'
        if series is None:
            manifest.append((item, 'mechanism trace incomplete; mechanism figure omitted'))
            continue
        rounds = np.arange(1, len(series['byzantine']['phi']) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4), sharex=True)
        for axis, metric in zip(axes.ravel(), ('phi', 'ratio', 'tau', 'gamma')):
            axis.plot(rounds, series['honest'][metric], color='#4C78A8', label='honest-edge mean')
            axis.plot(rounds, series['byzantine'][metric], color='#C62828', label='Byzantine-edge mean')
            axis.set(title=metric, ylim=(0, 1), ylabel='value')
            axis.grid(axis='y', color='#D9D9D9', linewidth=.5)
        axes[1, 0].set_xlabel('Communication round'); axes[1, 1].set_xlabel('Communication round')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'{METHOD_LABELS.get(method, method)} — {ATTACK_LABELS.get(attack, attack)} (seed {seed})')
        _save(fig, out / f'seed_{seed}' / 'mechanism' / f'{method}_{attack}')


def write_manifest(path: Path, manifest: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "reason"])
        writer.writerows(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Root containing result JSON and .meta.json files")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--attacks", default=",".join(ATTACKS))
    parser.add_argument("--mechanism", action="store_true", help="Generate phi/ratio/tau/gamma figures only from complete edge traces")
    args = parser.parse_args()
    methods, attacks = args.methods.split(","), args.attacks.split(",")
    records, manifest = collect(args.results, args.dataset, methods, attacks)
    seeds = sorted({key[0] for key in records})
    for seed in seeds:
        plot_seed(seed, records, methods, attacks, args.out, manifest)
    if seeds:
        plot_complete_heatmap(seeds, records, methods, attacks, args.out, manifest)
    if args.mechanism:
        plot_mechanism(records, args.out, manifest)
    if not records:
        manifest.append((str(args.results), "no eligible formal schema-v3 records found; no figures generated"))
    write_manifest(args.out / "manifest.csv", manifest)
    print(f"Eligible records: {len(records)}; seeds: {seeds}; manifest: {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
