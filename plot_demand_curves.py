# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Lund University
# Part of: Station-Level Charging Demand Profiles for Heavy-Duty Electric Trucks
# Repository: https://doi.org/10.5281/zenodo.XXXXXXX

"""
plot_demand_curves.py
=====================
Reads the pre-computed probabilistic demand-curve CSVs and creates a
publication-quality figure.

No external dependencies beyond pandas, numpy and matplotlib.

Usage
-----
    python plot_demand_curves.py

The CSVs must be in the same directory (or adjust DATA_DIR below).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HOURS = np.arange(24)


def load_curve(csv_name):
    path = os.path.join(DATA_DIR, csv_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    return pd.read_csv(path)


def plot(pub, dep, save_path=None):
    """Two-panel figure with percentile bands."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    hour_labels = [f"{h:02d}:00" for h in HOURS]

    panels = [
        (axes[0], pub, "Public (Fast) Charging — Skåne Region", "#d62728"),
        (axes[1], dep, "Depot (Optimised) Charging — Skåne Region", "#1f77b4"),
    ]

    for ax, df, title, color in panels:
        h = df["hour"]

        ax.fill_between(h, df["p5_kW"]  / 1e3, df["p95_kW"] / 1e3,
                        alpha=0.15, color=color, label="P5 – P95")
        ax.fill_between(h, df["p25_kW"] / 1e3, df["p75_kW"] / 1e3,
                        alpha=0.30, color=color, label="P25 – P75")
        ax.plot(h, df["p50_kW"] / 1e3, "--", color=color, lw=1.5, label="Median")
        ax.plot(h, df["mean_kW"] / 1e3, "-",  color=color, lw=2.5, label="Mean")

        total_mwh = df["mean_kW"].sum() / 1e3
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Power demand (MW)")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 23)
        ax.annotate(
            f"Mean daily energy: {total_mwh:,.1f} MWh",
            xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    axes[1].set_xlabel("Hour of day")
    axes[1].set_xticks(HOURS)
    axes[1].set_xticklabels(hour_labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def main():
    pub = load_curve("public_demand_curve.csv")
    dep = load_curve("depot_demand_curve.csv")

    fig_path = os.path.join(DATA_DIR, "probabilistic_demand_curves.png")
    plot(pub, dep, save_path=fig_path)


if __name__ == "__main__":
    main()
