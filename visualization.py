"""Plotting helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_convergence(conv: pd.DataFrame, out_file: str, title: str = "Convergence") -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(conv["iteration"], conv["current_cost"], label="Current", alpha=0.5)
    plt.plot(conv["iteration"], conv["best_cost"], label="Best", linewidth=2)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_cost_breakdown(metrics: Dict, out_file: str, title: str = "Cost Breakdown") -> None:
    items = ["start_cost", "energy_cost", "time_window_penalty", "carbon_cost"]
    vals = [metrics[k] for k in items]
    plt.figure(figsize=(6, 4))
    plt.bar(["Start", "Energy", "TW/Policy", "Carbon"], vals)
    plt.title(title)
    plt.ylabel("Cost (CNY)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_routes(solution: Dict, coords: pd.DataFrame, out_file: str, title: str = "Routes") -> None:
    cdf = coords.set_index("客户编号")
    depot_x, depot_y = (0.0, 0.0)
    if 0 in cdf.index:
        depot_x = float(cdf.loc[0, "x"])
        depot_y = float(cdf.loc[0, "y"])

    plt.figure(figsize=(7, 6))
    for cid, r in cdf.iterrows():
        if int(cid) == 0:
            continue
        plt.scatter(r["x"], r["y"], c="gray", s=18)
    plt.scatter(depot_x, depot_y, c="red", s=80, marker="*", label="Depot")

    for route in solution.get("routes", []):
        xs, ys = [depot_x], [depot_y]
        for s in route["stops"]:
            cid = int(s["customer_id"])
            if cid in cdf.index:
                xs.append(float(cdf.loc[cid, "x"]))
                ys.append(float(cdf.loc[cid, "y"]))
        xs.append(depot_x)
        ys.append(depot_y)
        plt.plot(xs, ys, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_compare_bar(df: pd.DataFrame, x: str, ys: list[str], out_file: str, title: str = "Compare") -> None:
    ax = df.set_index(x)[ys].plot(kind="bar", figsize=(8, 4))
    ax.set_title(title)
    ax.set_ylabel("Value")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_line(df: pd.DataFrame, x: str, y: str, out_file: str, title: str = "Sensitivity") -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
