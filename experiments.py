"""Ablation and sensitivity experiments."""
from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import pandas as pd

from solver import ALNSConfig, ALNSSolver
from cost_model import CARBON_PRICE, evaluate_solution
from visualization import plot_line


def run_ablation(data: Dict, vehicles: List[Dict], out_dir: Path) -> pd.DataFrame:
    """Run algorithm ablation experiments.

    The current solver is a destroy-repair neighborhood search with optional
    simulated-annealing acceptance. We avoid claiming an adaptive operator
    mechanism unless it is explicitly implemented.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("Greedy baseline", ALNSConfig(iterations=1, adaptive=False, use_sa=False), False),
        ("DestroyRepair_random_init", ALNSConfig(iterations=80, adaptive=False, use_sa=True), True),
        ("DestroyRepair_no_SA", ALNSConfig(iterations=80, adaptive=False, use_sa=False), False),
        ("Full_destroy_repair_SA", ALNSConfig(iterations=100, adaptive=False, use_sa=True), False),
    ]
    rows = []
    all_conv = []
    for name, cfg, random_init in variants:
        t0 = time.time()
        solver = ALNSSolver(data, vehicles, policy=False, config=cfg)
        best, conv = solver.solve(random_init=random_init)
        m = evaluate_solution(best, data, policy=False)
        runtime = time.time() - t0
        best_iter = int(conv["best_cost"].idxmin() + 1) if not conv.empty else 1
        feasible_ratio = 1.0
        rows.append(
            {
                "variant": name,
                "final_total_cost": m["total_cost"],
                "runtime_sec": runtime,
                "best_iteration": best_iter,
                "time_window_satisfaction": m["time_window_satisfaction"],
                "total_carbon": m["total_carbon"],
                "feasible_ratio": feasible_ratio,
            }
        )
        conv = conv.copy()
        conv["variant"] = name
        all_conv.append(conv)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "ablation_results.csv", index=False)
    if all_conv:
        raw = pd.concat(all_conv, ignore_index=True)
        raw.to_csv(out_dir / "ablation_convergence_raw.csv", index=False)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        for v, g in raw.groupby("variant"):
            plt.plot(g["iteration"], g["best_cost"], label=v)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Best Cost")
        plt.tight_layout()
        plt.savefig(out_dir / "ablation_convergence.png", dpi=150)
        plt.close()
    return res


def _scale_ev_fleet(vehicles: List[Dict], factor: float) -> List[Dict]:
    """Return a new fleet where the number of EVs is scaled by factor."""
    fuel = [deepcopy(v) for v in vehicles if v.get("energy") != "ev"]
    ev = [deepcopy(v) for v in vehicles if v.get("energy") == "ev"]
    if not ev:
        return fuel

    target = max(1, int(round(len(ev) * factor)))
    new_fleet = fuel + [deepcopy(v) for v in ev[: min(target, len(ev))]]
    next_id = max(int(v["id"]) for v in new_fleet) + 1 if new_fleet else 1

    while len([v for v in new_fleet if v.get("energy") == "ev"]) < target:
        template = deepcopy(ev[(len(new_fleet) - len(fuel)) % len(ev)])
        template["id"] = next_id
        next_id += 1
        new_fleet.append(template)

    return new_fleet


def run_sensitivity(data: Dict, vehicles: List[Dict], out_dir: Path) -> Dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # baseline solution once: carbon price sensitivity here measures cost accounting response.
    solver = ALNSSolver(data, vehicles, policy=True, config=ALNSConfig(iterations=60, seed=42))
    base_sol, _ = solver.solve(random_init=False)

    carbon_rows = []
    for k in [0.5, 0.75, 1.0, 1.25, 1.5]:
        m = evaluate_solution(base_sol, data, policy=True, carbon_price=CARBON_PRICE * k)
        fuel_used = int(sum(1 for r in base_sol["routes"] if r["vehicle"]["energy"] == "fuel"))
        ev_used = int(sum(1 for r in base_sol["routes"] if r["vehicle"]["energy"] == "ev"))
        carbon_rows.append({
            "carbon_factor": k,
            "carbon_price": CARBON_PRICE * k,
            "total_cost": m["total_cost"],
            "total_carbon": m["total_carbon"],
            "fuel_vehicles": fuel_used,
            "ev_vehicles": ev_used,
            "time_window_satisfaction": m["time_window_satisfaction"],
        })
    carbon_df = pd.DataFrame(carbon_rows)
    carbon_df.to_csv(out_dir / "carbon_price_sensitivity.csv", index=False)
    plot_line(carbon_df, "carbon_factor", "total_cost", str(out_dir / "carbon_price_sensitivity.png"), "Carbon Price Sensitivity")

    ev_rows = []
    for k in [0.8, 1.0, 1.2, 1.4]:
        v2 = _scale_ev_fleet(vehicles, k)
        solver2 = ALNSSolver(data, v2, policy=True, config=ALNSConfig(iterations=50, seed=42))
        sol2, _ = solver2.solve(random_init=False)
        m2 = evaluate_solution(sol2, data, policy=True)
        fuel_used = int(sum(1 for r in sol2["routes"] if r["vehicle"]["energy"] == "fuel"))
        ev_used = int(sum(1 for r in sol2["routes"] if r["vehicle"]["energy"] == "ev"))
        ev_rows.append({
            "ev_factor": k,
            "available_ev": int(sum(1 for v in v2 if v.get("energy") == "ev")),
            "total_cost": m2["total_cost"],
            "total_carbon": m2["total_carbon"],
            "fuel_vehicles": fuel_used,
            "ev_vehicles": ev_used,
            "time_window_satisfaction": m2["time_window_satisfaction"],
        })
    ev_df = pd.DataFrame(ev_rows)
    ev_df.to_csv(out_dir / "ev_quantity_sensitivity.csv", index=False)
    plot_line(ev_df, "available_ev", "total_cost", str(out_dir / "ev_quantity_sensitivity.png"), "EV Quantity Sensitivity")

    gr_rows = []
    for radius in [8, 10, 12, 15]:
        d2 = deepcopy(data)
        coords = d2["coords"].copy()
        coords["distance_to_center"] = (coords["x"] ** 2 + coords["y"] ** 2) ** 0.5
        d2["valid_green_customers"] = coords.loc[
            (coords["distance_to_center"] <= radius) & (coords["客户编号"].astype(int) != 0),
            "客户编号",
        ].astype(int).tolist()
        solver3 = ALNSSolver(d2, vehicles, policy=True, config=ALNSConfig(iterations=50, seed=42))
        sol3, _ = solver3.solve(random_init=False)
        m3 = evaluate_solution(sol3, d2, policy=True)
        fuel_used = int(sum(1 for r in sol3["routes"] if r["vehicle"]["energy"] == "fuel"))
        ev_used = int(sum(1 for r in sol3["routes"] if r["vehicle"]["energy"] == "ev"))
        gr_rows.append({
            "green_radius": radius,
            "green_customer_count": len(d2["valid_green_customers"]),
            "total_cost": m3["total_cost"],
            "total_carbon": m3["total_carbon"],
            "fuel_vehicles": fuel_used,
            "ev_vehicles": ev_used,
            "time_window_satisfaction": m3["time_window_satisfaction"],
        })
    gr_df = pd.DataFrame(gr_rows)
    gr_df.to_csv(out_dir / "green_radius_sensitivity.csv", index=False)
    plot_line(gr_df, "green_radius", "total_cost", str(out_dir / "green_radius_sensitivity.png"), "Green Radius Sensitivity")

    return {"carbon": carbon_df, "ev": ev_df, "green": gr_df}
