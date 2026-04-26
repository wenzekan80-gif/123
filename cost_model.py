"""Cost and route evaluation model."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_START_COST = 400.0
POLICY_VIOLATION_PENALTY = 1_000_000.0
WAIT_COST_PER_H = 20.0
LATE_COST_PER_H = 50.0
SERVICE_HOURS = 20.0 / 60.0
FUEL_PRICE = 7.61
ELEC_PRICE = 1.64
FUEL_CF = 2.547
ELEC_CF = 0.501
CARBON_PRICE = 0.65


def is_policy_feasible(vehicle: Dict, customer_id: int, arrival_time: float, green_customer_set: set[int]) -> bool:
    """Green-zone policy feasibility check."""
    if vehicle.get("energy") == "fuel" and customer_id in green_customer_set and 8.0 <= arrival_time < 16.0:
        return False
    return True


def speed_at_time(t: float) -> float:
    """Deterministic speed profile according to requested time periods."""
    hour = t % 24
    if 8 <= hour < 9 or 11.5 <= hour < 13:
        return 9.8
    if 10 <= hour < 11.5 or 15 <= hour < 17 or 17 <= hour < 19:
        return 35.4
    if 9 <= hour < 10 or 13 <= hour < 15 or hour >= 19:
        return 55.3
    return 35.4


def energy_per_100km(vehicle: Dict, speed: float, load_ratio: float) -> float:
    if vehicle["energy"] == "fuel":
        base = 0.0025 * speed**2 - 0.2554 * speed + 31.75
        return base * (1 + 0.4 * load_ratio)
    base = 0.0014 * speed**2 - 0.12 * speed + 36.19
    return base * (1 + 0.35 * load_ratio)


def evaluate_solution(solution: Dict, data: Dict, policy: bool = False, carbon_price: float = CARBON_PRICE) -> Dict:
    """Evaluate solution and return detailed metrics.

    solution['routes']: list of {'vehicle':dict,'stops':[{'customer_id':int,'weight':float,'volume':float}]}
    """
    dist = data["distance"]
    tw = data["time_windows"].set_index("客户编号")
    green = set(data["valid_green_customers"])

    total_start = total_energy_cost = total_tw_cost = total_carbon_cost = 0.0
    total_dist = total_carbon = 0.0
    arrival_records: List[Dict] = []
    route_details: List[Dict] = []
    tw_ok = 0
    tw_total = 0
    policy_violations: List[Dict] = []

    for r_idx, route in enumerate(solution.get("routes", [])):
        vehicle = route["vehicle"]
        stops = route.get("stops", [])
        if not stops:
            continue
        route_start_cost = float(vehicle.get("start_cost", DEFAULT_START_COST))
        total_start += route_start_cost

        current = 0
        t = 8.0
        remain_w = float(sum(s["weight"] for s in stops))
        for s in stops:
            cid = int(s["customer_id"])
            d = float(dist.loc[current, cid])
            v = speed_at_time(t)
            tt = d / max(v, 1e-6)
            arrive = t + tt
            load_ratio = min(1.0, max(0.0, remain_w / vehicle["capacity_weight"]))
            e100 = energy_per_100km(vehicle, v, load_ratio)
            amount = e100 * d / 100.0

            if vehicle["energy"] == "fuel":
                eco = amount * FUEL_PRICE
                carbon = amount * FUEL_CF
            else:
                eco = amount * ELEC_PRICE
                carbon = amount * ELEC_CF

            total_energy_cost += eco
            total_carbon += carbon
            total_carbon_cost += carbon * carbon_price
            total_dist += d

            earliest = float(tw.loc[cid, "最早"]) if cid in tw.index else 8.0
            latest = float(tw.loc[cid, "最晚"]) if cid in tw.index else 20.0
            wait = max(0.0, earliest - arrive)
            late = max(0.0, arrive - latest)
            total_tw_cost += wait * WAIT_COST_PER_H + late * LATE_COST_PER_H

            service_begin = arrive + wait
            depart = service_begin + SERVICE_HOURS

            feasible = True
            if policy:
                feasible = is_policy_feasible(vehicle, cid, arrive, green)
                if not feasible:
                    total_tw_cost += POLICY_VIOLATION_PENALTY
                    policy_violations.append(
                        {
                            "vehicle_id": vehicle["id"],
                            "vehicle_type": vehicle["type"],
                            "customer_id": cid,
                            "arrival_time": round(arrive, 4),
                            "is_green_customer": cid in green,
                            "violation": True,
                            "penalty": POLICY_VIOLATION_PENALTY,
                        }
                    )

            tw_total += 1
            tw_ok += int(late <= 1e-9 and feasible)

            arrival_records.append(
                {
                    "vehicle_id": vehicle["id"],
                    "vehicle_type": vehicle["type"],
                    "customer_id": cid,
                    "arrival_time": arrive,
                    "wait_hours": wait,
                    "late_hours": late,
                    "service_start": service_begin,
                    "depart_time": depart,
                    "policy_feasible": feasible,
                }
            )

            remain_w = max(0.0, remain_w - s["weight"])
            t = depart
            current = cid

        d_back = float(dist.loc[current, 0])
        v_back = speed_at_time(t)
        total_dist += d_back
        e100b = energy_per_100km(vehicle, v_back, 0.0)
        amt_back = e100b * d_back / 100.0
        if vehicle["energy"] == "fuel":
            total_energy_cost += amt_back * FUEL_PRICE
            carb = amt_back * FUEL_CF
        else:
            total_energy_cost += amt_back * ELEC_PRICE
            carb = amt_back * ELEC_CF
        total_carbon += carb
        total_carbon_cost += carb * carbon_price

        route_details.append(
            {
                "vehicle_id": vehicle["id"],
                "vehicle_type": vehicle["type"],
                "num_stops": len(stops),
                "load_weight": float(sum(s["weight"] for s in stops)),
                "load_volume": float(sum(s["volume"] for s in stops)),
                "capacity_weight": float(vehicle["capacity_weight"]),
                "capacity_volume": float(vehicle["capacity_volume"]),
                "start_cost": route_start_cost,
                "route": "0-" + "-".join(str(int(s["customer_id"])) for s in stops) + "-0",
            }
        )

    tw_rate = float(tw_ok / tw_total) if tw_total else 1.0
    total = total_start + total_energy_cost + total_tw_cost + total_carbon_cost
    return {
        "total_cost": total,
        "start_cost": total_start,
        "energy_cost": total_energy_cost,
        "time_window_penalty": total_tw_cost,
        "carbon_cost": total_carbon_cost,
        "total_distance": total_dist,
        "total_carbon": total_carbon,
        "time_window_satisfaction": tw_rate,
        "route_details": pd.DataFrame(route_details),
        "arrival_details": pd.DataFrame(arrival_records),
        "policy_violation_check": pd.DataFrame(policy_violations),
    }
