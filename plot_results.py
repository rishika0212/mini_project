"""
plot_results.py
===============
Loads evaluation CSVs produced by evaluate.py and generates comparison graphs.

Usage:
    # 1. Run evaluate.py for each policy first:
    #    python evaluate.py --policy dqn    --episodes 30
    #    python evaluate.py --policy fixed  --episodes 30
    #    python evaluate.py --policy random --episodes 30

    # 2. Then plot:
    python plot_results.py

Outputs (saved to data/plots/):
    avg_wait_comparison.png   — DQN vs Fixed vs Random: average waiting time per episode
    queue_comparison.png      — queue length over time
    throughput_comparison.png — vehicles per minute over time
    reward_curve.png          — DQN training reward curve (from dqn_weights_int1.json)
"""

import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = "data/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

POLICIES       = ['dqn', 'fixed', 'random']
POLICY_LABELS  = {'dqn': 'DQN (ours)', 'fixed': 'Fixed-Time', 'random': 'Random'}
POLICY_COLORS  = {'dqn': '#2ecc71',    'fixed': '#e74c3c',    'random': '#95a5a6'}
POLICY_STYLES  = {'dqn': '-',          'fixed': '--',         'random': ':'}

# ── Load CSV helpers ────────────────────────────────────────────────────────

def load_eval_csv(policy):
    path = f"data/eval_{policy}.csv"
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def episode_means(rows, metric):
    """Return list of per-episode mean values for the given metric."""
    if rows is None:
        return []
    by_ep = {}
    for r in rows:
        ep = int(r['episode'])
        by_ep.setdefault(ep, []).append(r[metric])
    return [np.mean(v) for v in sorted(by_ep.values())]

def rolling_mean(series, window=5):
    if len(series) < window:
        return series
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode='valid')

# ── Plot 1: Average waiting time per episode ────────────────────────────────

def plot_avg_wait():
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for pol in POLICIES:
        rows  = load_eval_csv(pol)
        means = episode_means(rows, 'avg_waiting_time')
        if not means:
            continue
        x = np.arange(1, len(means) + 1)
        ax.plot(x, means,
                label=POLICY_LABELS[pol],
                color=POLICY_COLORS[pol],
                linestyle=POLICY_STYLES[pol],
                linewidth=2, alpha=0.4)
        smooth = rolling_mean(means, window=5)
        xs     = np.arange(len(means) - len(smooth) + 1,
                           len(means) + 1)
        ax.plot(xs, smooth,
                color=POLICY_COLORS[pol],
                linestyle=POLICY_STYLES[pol],
                linewidth=2.5)
        plotted = True

    if not plotted:
        print("  No eval CSV files found — run evaluate.py first.")
        plt.close()
        return

    ax.set_title("Average Vehicle Waiting Time per Episode\n"
                 "(lower is better)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Waiting Time (s)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/avg_wait_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# ── Plot 2: Queue length over time ──────────────────────────────────────────

def plot_queue():
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for pol in POLICIES:
        rows  = load_eval_csv(pol)
        means = episode_means(rows, 'queue_length')
        if not means:
            continue
        x      = np.arange(1, len(means) + 1)
        smooth = rolling_mean(means, window=5)
        xs     = np.arange(len(means) - len(smooth) + 1, len(means) + 1)
        ax.plot(xs, smooth,
                label=POLICY_LABELS[pol],
                color=POLICY_COLORS[pol],
                linestyle=POLICY_STYLES[pol],
                linewidth=2.5)
        plotted = True

    if not plotted:
        plt.close(); return

    ax.set_title("Average Queue Length per Episode\n"
                 "(lower is better)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Queue Length (vehicles)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/queue_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# ── Plot 3: Throughput over time ────────────────────────────────────────────

def plot_throughput():
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for pol in POLICIES:
        rows  = load_eval_csv(pol)
        means = episode_means(rows, 'throughput_vpm')
        if not means:
            continue
        smooth = rolling_mean(means, window=5)
        xs     = np.arange(len(means) - len(smooth) + 1, len(means) + 1)
        ax.plot(xs, smooth,
                label=POLICY_LABELS[pol],
                color=POLICY_COLORS[pol],
                linestyle=POLICY_STYLES[pol],
                linewidth=2.5)
        plotted = True

    if not plotted:
        plt.close(); return

    ax.set_title("Vehicles Cleared per Minute (Throughput) per Episode\n"
                 "(higher is better)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Throughput (vehicles/min)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/throughput_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# ── Plot 4: DQN training reward curve ───────────────────────────────────────

def plot_reward_curve():
    rewards_all = []
    for i in range(1, 4):
        path = f"data/dqn_weights_int{i}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        ep_r = data.get('episode_rewards', [])
        if ep_r:
            rewards_all.append((i, ep_r))

    if not rewards_all:
        print("  No episode_rewards in weight files — train first.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ['#2ecc71', '#3498db', '#9b59b6']

    for (int_id, rewards), color in zip(rewards_all, colors):
        x      = np.arange(1, len(rewards) + 1)
        ax.plot(x, rewards, color=color, alpha=0.25, linewidth=1)
        smooth = rolling_mean(rewards, window=10)
        xs     = np.arange(len(rewards) - len(smooth) + 1, len(rewards) + 1)
        ax.plot(xs, smooth, color=color, linewidth=2.5,
                label=f"Intersection {int_id}")

    ax.set_title("DQN Training Reward Curve\n"
                 "(upward trend = agent improving)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Episode Reward", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/reward_curve.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# ── Summary bar chart ────────────────────────────────────────────────────────

def plot_summary_bar():
    summary = {}
    for pol in POLICIES:
        rows  = load_eval_csv(pol)
        means = episode_means(rows, 'avg_waiting_time')
        if means:
            summary[pol] = np.mean(means)

    if not summary:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    labels  = [POLICY_LABELS[p] for p in summary]
    values  = list(summary.values())
    colors  = [POLICY_COLORS[p] for p in summary]
    bars    = ax.bar(labels, values, color=colors, width=0.5, edgecolor='white')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.2f}s", ha='center', va='bottom', fontsize=11)

    ax.set_title("Mean Waiting Time Across All Episodes\n"
                 "(lower is better)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean Avg Waiting Time (s)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/summary_bar.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nGenerating plots → {OUTPUT_DIR}/\n")
    plot_avg_wait()
    plot_queue()
    plot_throughput()
    plot_reward_curve()
    plot_summary_bar()
    print("\nDone.")
