"""Main entrypoint for Huazhong Cup A reproducible experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from solver import ALNSConfig, ALNSSolver, solve_problem2_policy, dynamic_reschedule
from cost_model import evaluate_solution
from data_loader import preprocess_data
from experiments import run_ablation, run_sensitivity
from visualization import ensure_dir, plot_compare_bar, plot_convergence, plot_cost_breakdown, plot_routes


def build_vehicles() -> List[Dict]:
    specs = [
        # type, energy, capacity_weight, capacity_volume, available_count, start_cost
        ("fuel_1", "fuel", 3000, 13.5, 60, 400.0),
        ("fuel_2", "fuel", 1500, 10.8, 50, 350.0),
        ("fuel_3", "fuel", 1250, 6.5, 50, 300.0),
        ("ev_1", "ev", 3000, 15.0, 10, 500.0),
        ("ev_2", "ev", 1250, 8.5, 15, 400.0),
    ]
    vehicles = []
    idx = 1
    for t, energy, cw, cv, n, start_cost in specs:
        for _ in range(n):
            vehicles.append(
                {
                    "id": idx,
                    "type": t,
                    "energy": energy,
                    "capacity_weight": float(cw),
                    "capacity_volume": float(cv),
                    "start_cost": float(start_cost),
                }
            )
            idx += 1
    return vehicles


def to_data_dict(pre) -> Dict:
    return {
        "distance": pre.distance_matrix,
        "demand": pre.customer_demand,
        "time_windows": pre.time_windows,
        "coords": pre.customers,
        "valid_green_customers": pre.valid_green_customers,
    }


def save_problem_outputs(name: str, out_dir: Path, solution: Dict, metrics: Dict, conv: pd.DataFrame, coords: pd.DataFrame) -> None:
    ensure_dir(out_dir)
    prefix = name.lower()

    metrics["route_details"].to_csv(out_dir / f"{prefix}_vehicle_routes.csv", index=False)
    metrics["arrival_details"].to_csv(out_dir / f"{prefix}_customer_arrival_times.csv", index=False)
    if "policy_violation_check" in metrics and not metrics["policy_violation_check"].empty:
        metrics["policy_violation_check"].to_csv(out_dir / f"{prefix}_policy_violations.csv", index=False)

    cost_df = pd.DataFrame(
        [
            {
                "total_cost": metrics["total_cost"],
                "start_cost": metrics["start_cost"],
                "energy_cost": metrics["energy_cost"],
                "time_window_penalty": metrics["time_window_penalty"],
                "carbon_cost": metrics["carbon_cost"],
                "total_distance": metrics["total_distance"],
                "total_carbon": metrics["total_carbon"],
                "time_window_satisfaction": metrics["time_window_satisfaction"],
            }
        ]
    )
    cost_df.to_csv(out_dir / f"{prefix}_cost_breakdown.csv", index=False)

    usage = metrics["route_details"].copy()
    if not usage.empty:
        usage = usage.groupby("vehicle_type", as_index=False).agg(used_vehicles=("vehicle_id", "nunique"), total_stops=("num_stops", "sum"))
    usage.to_csv(out_dir / f"{prefix}_vehicle_usage.csv", index=False)

    conv.to_csv(out_dir / f"{prefix}_convergence.csv", index=False)
    plot_convergence(conv, str(out_dir / f"{prefix}_convergence.png"), f"{name} Convergence")
    plot_cost_breakdown(metrics, str(out_dir / f"{prefix}_cost_breakdown.png"), f"{name} Cost Breakdown")
    plot_routes(solution, coords, str(out_dir / f"{prefix}_routes.png"), f"{name} Routes")

    summary = {
        "total_cost": metrics["total_cost"],
        "start_cost": metrics["start_cost"],
        "energy_cost": metrics["energy_cost"],
        "time_window_penalty": metrics["time_window_penalty"],
        "carbon_cost": metrics["carbon_cost"],
        "total_distance": metrics["total_distance"],
        "total_carbon": metrics["total_carbon"],
        "time_window_satisfaction": metrics["time_window_satisfaction"],
        "used_vehicles": int(metrics["route_details"]["vehicle_id"].nunique()) if not metrics["route_details"].empty else 0,
        "policy_violations": int(len(metrics.get("policy_violation_check", []))),
    }
    with open(out_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Huazhong Cup A pipeline")
    parser.add_argument(
        "--only",
        choices=["all", "problem1", "problem2", "problem3", "ablation", "sensitivity"],
        default="all",
        help="Run full pipeline or one module only.",
    )
    args = parser.parse_args()

    pre = preprocess_data(".", green_radius_km=10.0)
    vehicles = build_vehicles()
    data = to_data_dict(pre)

    out_root = ensure_dir("output")
    prep_dir = ensure_dir(out_root / "preprocess")
    pre.customer_demand.to_csv(prep_dir / "customer_demand_summary.csv", index=False)
    pre.missing_report.to_csv(prep_dir / "missing_value_report.csv", index=False)
    with open(prep_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "valid_customer_count": len(pre.valid_customers),
                "no_order_customers": pre.no_order_customers,
                "green_customers": pre.green_customers,
                "valid_green_customers": pre.valid_green_customers,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    p1_sol = p1_metrics = p2_sol = p2_metrics = None

    if args.only in ["all", "problem1", "problem2", "problem3"]:
        p1_solver = ALNSSolver(data, vehicles, policy=False, config=ALNSConfig(iterations=70, seed=42))
        p1_sol, p1_conv = p1_solver.solve(random_init=False)
        p1_metrics = evaluate_solution(p1_sol, data, policy=False)
        p1_dir = ensure_dir(out_root / "problem1")
        save_problem_outputs("problem1", p1_dir, p1_sol, p1_metrics, p1_conv, pre.customers)

    if args.only in ["all", "problem2", "problem3"]:
        p2_sol, p2_conv, p2_metrics = solve_problem2_policy(data, vehicles, iterations=80)
        p2_dir = ensure_dir(out_root / "problem2")
        save_problem_outputs("problem2", p2_dir, p2_sol, p2_metrics, p2_conv, pre.customers)

        policy_check = p2_metrics["arrival_details"].copy()
        policy_check = policy_check[
            policy_check["vehicle_type"].str.contains("fuel")
            & policy_check["customer_id"].isin(pre.valid_green_customers)
        ]
        if not policy_check.empty:
            policy_check = policy_check[["vehicle_id", "vehicle_type", "customer_id", "arrival_time", "policy_feasible"]]
            policy_check["is_green_customer"] = True
            policy_check["violation"] = policy_check["arrival_time"].between(8, 16, inclusive="left")
        policy_check.to_csv(p2_dir / "problem2_policy_violation_check.csv", index=False)

        if p1_metrics is not None:
            compare_df = pd.DataFrame(
                [
                    {
                        "scenario": "before_policy",
                        "total_cost": p1_metrics["total_cost"],
                        "start_cost": p1_metrics["start_cost"],
                        "energy_cost": p1_metrics["energy_cost"],
                        "time_window_penalty": p1_metrics["time_window_penalty"],
                        "carbon_cost": p1_metrics["carbon_cost"],
                        "total_distance": p1_metrics["total_distance"],
                        "total_carbon": p1_metrics["total_carbon"],
                        "fuel_used": int(sum(1 for r in p1_sol["routes"] if r["vehicle"]["energy"] == "fuel")),
                        "ev_used": int(sum(1 for r in p1_sol["routes"] if r["vehicle"]["energy"] == "ev")),
                        "tw_satisfaction": p1_metrics["time_window_satisfaction"],
                    },
                    {
                        "scenario": "after_policy",
                        "total_cost": p2_metrics["total_cost"],
                        "start_cost": p2_metrics["start_cost"],
                        "energy_cost": p2_metrics["energy_cost"],
                        "time_window_penalty": p2_metrics["time_window_penalty"],
                        "carbon_cost": p2_metrics["carbon_cost"],
                        "total_distance": p2_metrics["total_distance"],
                        "total_carbon": p2_metrics["total_carbon"],
                        "fuel_used": int(sum(1 for r in p2_sol["routes"] if r["vehicle"]["energy"] == "fuel")),
                        "ev_used": int(sum(1 for r in p2_sol["routes"] if r["vehicle"]["energy"] == "ev")),
                        "tw_satisfaction": p2_metrics["time_window_satisfaction"],
                    },
                ]
            )
            compare_df.to_csv(p2_dir / "problem2_emission_summary.csv", index=False)
            plot_compare_bar(compare_df, "scenario", ["total_cost", "energy_cost", "carbon_cost"], str(p2_dir / "problem2_cost_compare.png"), "Policy Cost Compare")
            plot_compare_bar(compare_df, "scenario", ["fuel_used", "ev_used"], str(p2_dir / "problem2_vehicle_structure_compare.png"), "Vehicle Structure Compare")
            plot_compare_bar(compare_df, "scenario", ["total_carbon"], str(p2_dir / "problem2_emission_compare.png"), "Emission Compare")

    if args.only in ["all", "problem3"]:
        if p2_sol is None:
            p2_sol, _, p2_metrics = solve_problem2_policy(data, vehicles, iterations=80)
        p3_dir = ensure_dir(out_root / "problem3")
        scA = dynamic_reschedule(data, vehicles, p2_sol, scenario="A", event_time=9.5)
        scB = dynamic_reschedule(data, vehicles, p2_sol, scenario="B", event_time=10.0)

        for sc in [scA, scB]:
            name = sc["scenario"]
            before = sc["before_metrics"]["arrival_details"]
            after = sc["after_metrics"]["arrival_details"]
            before.to_csv(p3_dir / f"scenario_{name}_before.csv", index=False)
            after.to_csv(p3_dir / f"scenario_{name}_after.csv", index=False)
            cmp = pd.DataFrame([
                {
                    "before_total_cost": sc["before_metrics"]["total_cost"],
                    "after_total_cost": sc["after_metrics"]["total_cost"],
                    "delayed_customers_before": int((sc["before_metrics"]["arrival_details"]["late_hours"] > 0).sum()),
                    "delayed_customers_after": int((sc["after_metrics"]["arrival_details"]["late_hours"] > 0).sum()),
                    "tw_satisfaction_before": sc["before_metrics"]["time_window_satisfaction"],
                    "tw_satisfaction_after": sc["after_metrics"]["time_window_satisfaction"],
                    "reassigned_customers": sc["reassigned_customers"],
                    "route_change_rate": sc["disruption"],
                    "new_vehicles": sc["new_vehicles"],
                    "dynamic_objective": sc["dynamic_objective"],
                }
            ])
            cmp.to_csv(p3_dir / f"scenario_{name}_compare.csv", index=False)
            plot_routes(sc["before_solution"], pre.customers, str(p3_dir / f"scenario_{name}_routes_before.png"), f"Scenario {name} Routes Before")
            plot_routes(sc["after_solution"], pre.customers, str(p3_dir / f"scenario_{name}_routes_after.png"), f"Scenario {name} Routes After")

        with open(p3_dir / "problem3_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "method_note": "滚动重优化：冻结已完成客户，对剩余客户集合重新优化，并加入方案扰动惩罚。",
                    "scenario_A": {"dynamic_objective": scA["dynamic_objective"], "disruption": scA["disruption"]},
                    "scenario_B": {"dynamic_objective": scB["dynamic_objective"], "disruption": scB["disruption"]},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    if args.only in ["all", "ablation"]:
        run_ablation(data, vehicles, Path(out_root) / "ablation")

    if args.only in ["all", "sensitivity"]:
        run_sensitivity(data, vehicles, Path(out_root) / "sensitivity")

    summary = {
        "preprocess_ok": True,
        "valid_customer_count": len(pre.valid_customers),
        "green_customer_count": len(pre.green_customers),
    }
    if p1_metrics is not None:
        summary["problem1_total_cost"] = p1_metrics["total_cost"]
    if p2_metrics is not None:
        summary["problem2_total_cost"] = p2_metrics["total_cost"]

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== 总报告 summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
