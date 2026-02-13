# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Lund University
# Part of: Station-Level Charging Demand Profiles for Heavy-Duty Electric Trucks
# Repository: https://doi.org/10.5281/zenodo.XXXXXXX

"""
plot_normalized_profiles.py
===========================
Self-contained plotting script for the normalised electric-truck
charging demand profiles distributed in this open-access package.

Generates four figures from the bundled CSV files:
  1. Normalised profiles (fraction of daily energy) — public & depot
  2. Scaled MW demand curves — public & depot
  3. Normalised profiles by battery capacity bin — public
  4. Scaled MW demand by battery capacity bin — public

No external dependencies beyond pandas, numpy and matplotlib.

Usage
-----
    pip install pandas numpy matplotlib
    python plot_normalized_profiles.py

Scaling
-------
By default the script uses the simulation mean daily energy.
To scale the normalised profiles to a different scenario, set
DAILY_ENERGY_PUBLIC_MWh / DAILY_ENERGY_DEPOT_MWh below, or
set N_TRUCKS_PUBLIC / N_TRUCKS_DEPOT.

Author: Alice / Mattias — Lund University, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ═══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION — Adjust these to scale for your own scenario
# ═══════════════════════════════════════════════════════════════════════════════

# Total daily fast-charging energy (MWh/day).  None → use simulation mean.
DAILY_ENERGY_PUBLIC_MWh = None    # e.g. 500.0

# Total daily depot-charging energy (MWh/day).  None → use simulation mean.
DAILY_ENERGY_DEPOT_MWh  = None    # e.g. 140.0

# Alternatively specify fleet size; energy = N × mean_energy_per_truck.
N_TRUCKS_PUBLIC = None    # e.g. 2000
N_TRUCKS_DEPOT  = None    # e.g. 1500

# Simulation reference values (from 78 MATSim iterations):
SIM_MEAN_ENERGY_PUBLIC_MWh = 1269.0     # mean daily fast-charging energy
SIM_MEAN_ENERGY_DEPOT_MWh  =  141.6     # mean daily depot-charging energy
SIM_MEAN_TRUCKS_PUBLIC     = 1894       # mean unique trucks / day (public)
SIM_MEAN_TRUCKS_DEPOT      =  756       # mean unique trucks / day (depot)
SIM_ENERGY_PER_TRUCK_PUBLIC_kWh = 664.6  # mean kWh / truck / day (public)
SIM_ENERGY_PER_TRUCK_DEPOT_kWh  = 187.2  # mean kWh / truck / day (depot)

# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HOURS = np.arange(24)

BATTERY_BINS = ["0-200", "200-300", "300-400", "400-500", "500-600", "600-1200"]

# Energy share per battery bin (from simulation):
BIN_ENERGY_SHARE_PUBLIC = {
    "0-200":    0.001,
    "200-300":  0.019,
    "300-400":  0.206,
    "400-500":  0.487,
    "500-600":  0.288,
    "600-1200": 0.000,
}
BIN_ENERGY_SHARE_DEPOT = {
    "0-200":    0.002,
    "200-300":  0.072,
    "300-400":  0.144,
    "400-500":  0.392,
    "500-600":  0.391,
    "600-1200": 0.000,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load(csv_name):
    path = os.path.join(DATA_DIR, csv_name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _find_bin_files(prefix):
    """Find all battery-bin CSVs for a given prefix like 'public_normalized'."""
    found = {}
    for bl in BATTERY_BINS:
        df = _load(f"{prefix}_bat{bl}.csv")
        if df is not None:
            found[bl] = df
    return found


def _hour_labels():
    return [f"{h:02d}:00" for h in HOURS]


def _resolve_energy(user_mwh, user_n, sim_mwh, sim_epk, label):
    """Determine daily energy (MWh) for scaling."""
    if user_mwh is not None:
        return user_mwh
    if user_n is not None:
        return user_n * sim_epk / 1000
    return sim_mwh


def _dress_ax(ax, ylabel, title):
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(HOURS[::2])
    ax.set_xticklabels(
        [_hour_labels()[i] for i in range(0, 24, 2)],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_xlabel("Hour of day")


def _band(ax, h, mean, p5, p25, p50, p75, p95, color):
    ax.fill_between(h, p5,  p95, alpha=0.15, color=color, label="P5-P95")
    ax.fill_between(h, p25, p75, alpha=0.30, color=color, label="P25-P75")
    ax.plot(h, p50, "--", color=color, lw=1.5, label="Median")
    ax.plot(h, mean, "-", color=color, lw=2.5, label="Mean")


# ─── Figures ─────────────────────────────────────────────────────────────────

def plot_normalized_overall(pub, dep, save_dir):
    """Figure 1: Normalised profiles (fraction of daily energy)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, df, title, color, mwh in [
        (axes[0], pub, "Public (Fast) Charging", "#d62728",
         SIM_MEAN_ENERGY_PUBLIC_MWh),
        (axes[1], dep, "Depot (Optimised) Charging", "#1f77b4",
         SIM_MEAN_ENERGY_DEPOT_MWh),
    ]:
        if df is None:
            continue
        _band(ax, df["hour"], df["mean"], df["p5"], df["p25"],
              df["p50"], df["p75"], df["p95"], color)
        _dress_ax(ax, "Fraction of daily energy", title)
        ax.annotate(
            f"Sim. mean daily energy: {mwh:,.1f} MWh",
            xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    fig.suptitle("Normalised Charging Profiles  (sum = 1.0 per day)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, "normalized_profiles.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def plot_scaled_overall(pub, dep, pub_mwh, dep_mwh, save_dir):
    """Figure 2: Scaled MW demand curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, title, color, mwh in [
        (axes[0], pub, f"Public Charging  ({pub_mwh:.1f} MWh/day)",
         "#d62728", pub_mwh),
        (axes[1], dep, f"Depot Charging  ({dep_mwh:.1f} MWh/day)",
         "#1f77b4", dep_mwh),
    ]:
        if df is None:
            continue
        mw = {c: df[c] * mwh for c in ("mean", "p5", "p25", "p50", "p75", "p95")}
        _band(ax, df["hour"], mw["mean"], mw["p5"], mw["p25"],
              mw["p50"], mw["p75"], mw["p95"], color)
        _dress_ax(ax, "Power demand (MW)", title)

    fig.suptitle("Scaled Power Demand Curves", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, "scaled_demand_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def plot_battery_bins_normalized(prefix, title_prefix, color_map, save_dir):
    """Figure 3: Normalised profiles overlaid per battery-size bin."""
    bins = _find_bin_files(f"{prefix}_normalized")
    if not bins:
        return
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(bins)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for (bl, df), col in zip(bins.items(), colors):
        ax.fill_between(df["hour"], df["p25"], df["p75"], alpha=0.12, color=col)
        ax.plot(df["hour"], df["mean"], "-", color=col, lw=2, label=f"{bl} kWh")

    _dress_ax(ax, "Fraction of daily energy",
              f"{title_prefix} — Normalised Profiles by Battery Size")
    ax.legend(loc="upper right", fontsize=9, title="Battery capacity")
    fig.tight_layout()
    path = os.path.join(save_dir, "profiles_by_battery_size.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def plot_battery_bins_scaled(prefix, title_prefix, total_mwh,
                             bin_shares, save_dir):
    """Figure 4: Scaled MW demand per battery-size bin (stacked perspective)."""
    bins = _find_bin_files(f"{prefix}_normalized")
    if not bins:
        return
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(bins)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for (bl, df), col in zip(bins.items(), colors):
        share = bin_shares.get(bl, 0)
        if share < 0.001:
            continue
        mw_mean = df["mean"] * total_mwh * share
        mw_p25  = df["p25"]  * total_mwh * share
        mw_p75  = df["p75"]  * total_mwh * share
        ax.fill_between(df["hour"], mw_p25, mw_p75, alpha=0.12, color=col)
        ax.plot(df["hour"], mw_mean, "-", color=col, lw=2, label=f"{bl} kWh")

    _dress_ax(ax, "Power demand (MW)",
              f"{title_prefix} — Demand by Battery Size  "
              f"({total_mwh:.1f} MWh/day total)")
    ax.legend(loc="upper right", fontsize=9, title="Battery capacity")
    fig.tight_layout()
    path = os.path.join(save_dir, "demand_by_battery_size.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


# ─── Also regenerate the original absolute demand-curve figure ───────────────

def plot_absolute_demand_curves(save_dir):
    """Legacy figure: absolute kW demand curves (from *_demand_curve.csv)."""
    pub = _load("public_demand_curve.csv")
    dep = _load("depot_demand_curve.csv")
    if pub is None or dep is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    hour_labels = _hour_labels()

    for ax, df, title, color in [
        (axes[0], pub, "Public (Fast) Charging — Skåne Region", "#d62728"),
        (axes[1], dep, "Depot (Optimised) Charging — Skåne Region", "#1f77b4"),
    ]:
        h = df["hour"]
        ax.fill_between(h, df["p5_kW"] / 1e3, df["p95_kW"] / 1e3,
                        alpha=0.15, color=color, label="P5-P95")
        ax.fill_between(h, df["p25_kW"] / 1e3, df["p75_kW"] / 1e3,
                        alpha=0.30, color=color, label="P25-P75")
        ax.plot(h, df["p50_kW"] / 1e3, "--", color=color, lw=1.5, label="Median")
        ax.plot(h, df["mean_kW"] / 1e3, "-", color=color, lw=2.5, label="Mean")

        total = df["mean_kW"].sum() / 1e3
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Power demand (MW)")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 23)
        ax.annotate(f"Mean daily energy: {total:,.1f} MWh",
                    xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    axes[1].set_xlabel("Hour of day")
    axes[1].set_xticks(HOURS)
    axes[1].set_xticklabels(hour_labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(save_dir, "probabilistic_demand_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Electric Truck Charging — Open-Access Demand Profiles")
    print("=" * 60)

    pub_norm = _load("public_normalized_overall.csv")
    dep_norm = _load("depot_normalized_overall.csv")

    pub_mwh = _resolve_energy(
        DAILY_ENERGY_PUBLIC_MWh, N_TRUCKS_PUBLIC,
        SIM_MEAN_ENERGY_PUBLIC_MWh, SIM_ENERGY_PER_TRUCK_PUBLIC_kWh, "Public")
    dep_mwh = _resolve_energy(
        DAILY_ENERGY_DEPOT_MWh, N_TRUCKS_DEPOT,
        SIM_MEAN_ENERGY_DEPOT_MWh, SIM_ENERGY_PER_TRUCK_DEPOT_kWh, "Depot")

    print(f"\nScaling:")
    print(f"  Public: {pub_mwh:,.1f} MWh/day  →  peak ≈ "
          f"{pub_norm['mean'].max() * pub_mwh:.1f} MW" if pub_norm is not None else "")
    print(f"  Depot:  {dep_mwh:,.1f} MWh/day  →  peak ≈ "
          f"{dep_norm['mean'].max() * dep_mwh:.1f} MW" if dep_norm is not None else "")

    # Plot all figures
    plot_absolute_demand_curves(DATA_DIR)
    plot_normalized_overall(pub_norm, dep_norm, DATA_DIR)
    plot_scaled_overall(pub_norm, dep_norm, pub_mwh, dep_mwh, DATA_DIR)
    plot_battery_bins_normalized("public", "Public Fast Charging",
                                plt.cm.viridis, DATA_DIR)
    plot_battery_bins_scaled("public", "Public Fast Charging",
                            pub_mwh, BIN_ENERGY_SHARE_PUBLIC, DATA_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
