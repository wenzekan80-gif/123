"""Unified solver module: ALNS, policy solve, and dynamic rescheduling."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from cost_model import evaluate_solution


def check_capacity(load_w: float, load_v: float, vehicle: Dict) -> bool:
    """Capacity feasibility check."""
    return load_w <= vehicle["capacity_weight"] + 1e-9 and load_v <= vehicle["capacity_volume"] + 1e-9


@dataclass
class ALNSConfig:
    iterations: int = 100
    remove_ratio: float = 0.15
    init_temp: float = 500.0
    cooling: float = 0.99
    seed: int = 42
    adaptive: bool = True
    use_sa: bool = True


class ALNSSolver:
    """Fast ALNS-like heuristic solver."""

    def __init__(self, data: Dict, vehicles: List[Dict], policy: bool = False, config: ALNSConfig | None = None):
        self.data = data
        self.vehicles = deepcopy(vehicles)
        self.policy = policy
        self.cfg = config or ALNSConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def solve(self, random_init: bool = False) -> Tuple[Dict, pd.DataFrame]:
        current = self._build_initial_solution(random_init=random_init)
        eval_cur = evaluate_solution(current, self.data, policy=self.policy)
        best = deepcopy(current)
        eval_best = deepcopy(eval_cur)
        hist = []
        temp = self.cfg.init_temp

        for it in range(1, self.cfg.iterations + 1):
            cand = deepcopy(current)
            removed = self._random_remove(cand)
            self._fast_repair(cand, removed)
            eval_cand = evaluate_solution(cand, self.data, policy=self.policy)
            delta = eval_cand["total_cost"] - eval_cur["total_cost"]
            accept = delta < 0 or (self.cfg.use_sa and self.rng.random() < np.exp(-max(0, delta) / max(temp, 1e-9)))
            if accept:
                current, eval_cur = cand, eval_cand
            if eval_cur["total_cost"] < eval_best["total_cost"]:
                best, eval_best = deepcopy(current), deepcopy(eval_cur)
            temp *= self.cfg.cooling
            hist.append({"iteration": it, "current_cost": eval_cur["total_cost"], "best_cost": eval_best["total_cost"]})

        return best, pd.DataFrame(hist)

    def _customer_list(self) -> List[Dict]:
        dm = self.data["demand"].sort_values("客户编号")
        tw = self.data["time_windows"].set_index("客户编号")
        out = []
        for _, r in dm.iterrows():
            cid = int(r["客户编号"])
            out.append(
                {
                    "customer_id": cid,
                    "weight": float(r["重量"]),
                    "volume": float(r["体积"]),
                    "tw_early": float(tw.loc[cid, "最早"]) if cid in tw.index else 8.0,
                }
            )
        return out

    def _build_initial_solution(self, random_init: bool = False) -> Dict:
        customers = self._customer_list()
        coords = self.data["coords"].set_index("客户编号")
        green_set = set(self.data.get("valid_green_customers", []))

        if customers:
            X = np.array(
                [
                    [
                        float(coords.loc[c["customer_id"], "x"]) if c["customer_id"] in coords.index else 0.0,
                        float(coords.loc[c["customer_id"], "y"]) if c["customer_id"] in coords.index else 0.0,
                        c["tw_early"],
                        c["weight"],
                        c["volume"],
                    ]
                    for c in customers
                ]
            )
            k = min(max(2, len(customers) // 8), 8)
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X) if len(customers) >= k and k >= 2 else np.zeros(len(customers), dtype=int)
            for i, c in enumerate(customers):
                c["cluster"] = int(labels[i])

        if random_init:
            self.rng.shuffle(customers)
        else:
            customers = sorted(customers, key=lambda x: (x.get("cluster", 0), x["tw_early"]))

        routes: List[Dict] = []
        v_idx = 0
        for c in customers:
            rem_w, rem_v = c["weight"], c["volume"]
            while rem_w > 1e-9 and rem_v > 1e-9:
                placed = False
                for r in routes:
                    veh = r["vehicle"]
                    if self.policy and c["customer_id"] in green_set and veh["energy"] == "fuel":
                        continue
                    lw = sum(s["weight"] for s in r["stops"])
                    lv = sum(s["volume"] for s in r["stops"])
                    fw, fv = veh["capacity_weight"] - lw, veh["capacity_volume"] - lv
                    if fw <= 1e-9 or fv <= 1e-9:
                        continue
                    pw, pv = min(rem_w, fw), min(rem_v, fv)
                    if pw <= 1e-9 or pv <= 1e-9:
                        continue
                    r["stops"].append({"customer_id": c["customer_id"], "weight": pw, "volume": pv})
                    rem_w -= pw
                    rem_v -= pv
                    placed = True
                    break

                if placed:
                    continue

                if v_idx >= len(self.vehicles):
                    break

                if self.policy and c["customer_id"] in green_set:
                    found = next((j for j in range(v_idx, len(self.vehicles)) if self.vehicles[j]["energy"] == "ev"), None)
                    if found is not None:
                        self.vehicles[v_idx], self.vehicles[found] = self.vehicles[found], self.vehicles[v_idx]

                veh = deepcopy(self.vehicles[v_idx])
                v_idx += 1
                if self.policy and c["customer_id"] in green_set and veh["energy"] == "fuel":
                    continue

                pw, pv = min(rem_w, veh["capacity_weight"]), min(rem_v, veh["capacity_volume"])
                routes.append({"vehicle": veh, "stops": [{"customer_id": c["customer_id"], "weight": pw, "volume": pv}]})
                rem_w -= pw
                rem_v -= pv

        return {"routes": routes}

    def _random_remove(self, sol: Dict) -> List[Dict]:
        unique = list(set(int(s["customer_id"]) for r in sol["routes"] for s in r["stops"]))
        if not unique:
            return []
        k = max(1, int(len(unique) * self.cfg.remove_ratio))
        pick = set(self.rng.choice(unique, size=min(k, len(unique)), replace=False).tolist())
        removed = []
        for r in sol["routes"]:
            keep = []
            for s in r["stops"]:
                if int(s["customer_id"]) in pick:
                    removed.append(deepcopy(s))
                else:
                    keep.append(s)
            r["stops"] = keep
        sol["routes"] = [r for r in sol["routes"] if r["stops"]]
        return removed

    def _fast_repair(self, sol: Dict, removed: List[Dict]) -> None:
        dist = self.data["distance"]
        green_set = set(self.data.get("valid_green_customers", []))
        used_ids = {r["vehicle"]["id"] for r in sol["routes"]}
        free_vehicles = [deepcopy(v) for v in self.vehicles if v["id"] not in used_ids][:20]

        for c in removed:
            best = None
            for ridx, r in enumerate(sol["routes"][:30]):
                veh = r["vehicle"]
                if self.policy and c["customer_id"] in green_set and veh["energy"] == "fuel":
                    continue
                lw = sum(x["weight"] for x in r["stops"])
                lv = sum(x["volume"] for x in r["stops"])
                if not check_capacity(lw + c["weight"], lv + c["volume"], veh):
                    continue
                stops = r["stops"]
                for pos in [0, len(stops)]:
                    prev = 0 if pos == 0 else int(stops[pos - 1]["customer_id"])
                    nxt = 0 if pos >= len(stops) else int(stops[pos]["customer_id"])
                    delta = float(dist.loc[prev, c["customer_id"]]) + float(dist.loc[c["customer_id"], nxt]) - float(dist.loc[prev, nxt])
                    if best is None or delta < best[0]:
                        best = (delta, ridx, pos)

            if best is not None:
                _, ridx, pos = best
                sol["routes"][ridx]["stops"].insert(pos, deepcopy(c))
            elif free_vehicles:
                if self.policy and c["customer_id"] in green_set:
                    free_vehicles.sort(key=lambda x: 0 if x["energy"] == "ev" else 1)
                v = free_vehicles.pop(0)
                if self.policy and c["customer_id"] in green_set and v["energy"] == "fuel":
                    continue
                sol["routes"].append({"vehicle": v, "stops": [deepcopy(c)]})


def solve_problem2_policy(data: Dict, vehicles: List[Dict], iterations: int = 320) -> Tuple[Dict, pd.DataFrame, Dict]:
    """Policy-aware solve wrapper for Problem 2."""
    cfg = ALNSConfig(iterations=iterations, seed=42)
    solver = ALNSSolver(data=data, vehicles=vehicles, policy=True, config=cfg)
    best, conv = solver.solve(random_init=False)
    metrics = evaluate_solution(best, data, policy=True)
    return best, conv, metrics


def classify_event(event: Dict) -> str:
    """Classify dynamic event disturbance level."""
    changed = int(event.get("num_changes", 1))
    if changed <= 1:
        return "light"
    if changed <= 5:
        return "medium"
    return "heavy"


def _arrival_map(metrics: Dict) -> Dict[int, float]:
    arr = metrics["arrival_details"]
    if arr.empty:
        return {}
    return dict(zip(arr["customer_id"].astype(int), arr["arrival_time"].astype(float)))


def _scenario_from_base(data: Dict, base_solution: Dict, event_time: float, scenario: str) -> Dict:
    base_metrics = evaluate_solution(base_solution, data, policy=True)
    arr = base_metrics["arrival_details"]
    done_customers = set(arr.loc[arr["arrival_time"] <= event_time, "customer_id"].astype(int).tolist())

    demand = data["demand"].copy()
    tw = data["time_windows"].copy()
    coords = data["coords"].copy()

    if scenario == "A" and len(demand) >= 3:
        cancel_id = int(demand.iloc[0]["客户编号"])
        demand = demand[demand["客户编号"] != cancel_id]
        candidate_ids = [int(x) for x in data["distance"].index if int(x) != 0]
        new_id = candidate_ids[-1]
        demand = demand[demand["客户编号"] != new_id]
        demand = pd.concat([demand, pd.DataFrame([{"客户编号": new_id, "重量": 120.0, "体积": 1.1}])], ignore_index=True)
        if (tw["客户编号"] == new_id).any():
            tw.loc[tw["客户编号"] == new_id, ["最早", "最晚"]] = [10.5, 16.0]
        else:
            tw = pd.concat([tw, pd.DataFrame([{"客户编号": new_id, "最早": 10.5, "最晚": 16.0}])], ignore_index=True)

    elif scenario == "B" and len(tw) >= 2:
        cid = int(tw.iloc[1]["客户编号"])
        tw.loc[tw["客户编号"] == cid, "最早"] = tw.loc[tw["客户编号"] == cid, "最早"] - 1.0
        tw.loc[tw["客户编号"] == cid, "最晚"] = tw.loc[tw["客户编号"] == cid, "最晚"] - 0.5
        coords.loc[coords["客户编号"] == cid, "x"] += 1.5
        coords.loc[coords["客户编号"] == cid, "y"] -= 1.0

    remaining = demand[~demand["客户编号"].astype(int).isin(done_customers)].reset_index(drop=True)
    return {
        "distance": data["distance"],
        "demand": remaining,
        "time_windows": tw,
        "coords": coords,
        "valid_green_customers": data["valid_green_customers"],
    }


def dynamic_reschedule(data: Dict, vehicles: List[Dict], base_solution: Dict, scenario: str, event_time: float, lamb: float = 50.0) -> Dict:
    """Run event-driven rescheduling for scenario A/B."""
    before_metrics = evaluate_solution(base_solution, data, policy=True)
    before_map = _arrival_map(before_metrics)
    changed_data = _scenario_from_base(data, base_solution, event_time, scenario)

    evt = {"num_changes": 2 if scenario == "A" else 1}
    level = classify_event(evt)
    iters = 40 if level == "light" else (60 if level == "medium" else 90)

    solver = ALNSSolver(changed_data, vehicles, policy=True, config=ALNSConfig(iterations=iters, seed=42))
    after_solution, _ = solver.solve(random_init=False)
    after_metrics = evaluate_solution(after_solution, changed_data, policy=True)
    after_map = _arrival_map(after_metrics)

    common = set(before_map) & set(after_map)
    reassigned = len([c for c in common if abs(before_map[c] - after_map[c]) > 1e-6])
    delta_arrival = sum(abs(before_map[c] - after_map[c]) for c in common)
    new_vehicles = max(0, len(after_solution["routes"]) - len(base_solution["routes"]))
    disruption = reassigned + 0.05 * delta_arrival + 2.0 * new_vehicles

    final_obj = after_metrics["total_cost"] + lamb * disruption
    return {
        "scenario": scenario,
        "event_level": level,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "before_solution": base_solution,
        "after_solution": after_solution,
        "changed_data": changed_data,
        "disruption": disruption,
        "dynamic_objective": final_obj,
        "reassigned_customers": reassigned,
        "new_vehicles": new_vehicles,
    }
