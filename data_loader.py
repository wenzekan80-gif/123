"""Data loading and preprocessing for Huazhong Cup A logistics problem."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreprocessResult:
    orders_raw: pd.DataFrame
    customers: pd.DataFrame
    distance_matrix: pd.DataFrame
    time_windows: pd.DataFrame
    customer_demand: pd.DataFrame
    valid_customers: List[int]
    no_order_customers: List[int]
    green_customers: List[int]
    valid_green_customers: List[int]
    missing_report: pd.DataFrame


def _read_first_existing(paths: List[Path], **kwargs) -> pd.DataFrame:
    for p in paths:
        if p.exists():
            return pd.read_excel(p, **kwargs)
    raise FileNotFoundError(f"No file found among: {paths}")


def load_raw_data(base_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw excel files from the contest attachments.

    The competition version deliberately fails fast if any required file is
    missing. This avoids silently falling back to synthetic data and producing
    invalid contest results.
    """
    base = Path(base_dir)
    data_dir = base / "data"

    try:
        orders = _read_first_existing([
            base / "订单信息.xlsx",
            base / "订单信息(1).xlsx",
            data_dir / "订单信息.xlsx",
            data_dir / "订单信息(1).xlsx",
        ])
        dist = _read_first_existing([base / "距离矩阵.xlsx", data_dir / "距离矩阵.xlsx"], index_col=0)
        coords = _read_first_existing([base / "客户坐标信息.xlsx", data_dir / "客户坐标信息.xlsx"])
        tw = _read_first_existing([base / "时间窗.xlsx", data_dir / "时间窗.xlsx"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "未找到真实附件数据，请检查当前目录或 data/ 目录中是否包含："
            "订单信息.xlsx/订单信息(1).xlsx、距离矩阵.xlsx、客户坐标信息.xlsx、时间窗.xlsx。"
        ) from exc

    return orders, dist, coords, tw


def _norm_cols(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """Normalize common Chinese/English attachment column names.

    Matching is case-insensitive and substring-based, so columns such as
    "目标客户编号", "ID", "X (km)", "开始时间" can be mapped safely.
    """
    renamed = {}
    used_targets = set()
    for target, aliases in mapping.items():
        aliases_norm = [str(a).lower() for a in aliases]
        for c in df.columns:
            if c in renamed:
                continue
            c_norm = str(c).lower().strip()
            if c == target or c_norm == target.lower() or any(a in c_norm for a in aliases_norm):
                if target not in used_targets:
                    renamed[c] = target
                    used_targets.add(target)
                    break
    return df.rename(columns=renamed)


def _to_hour(value) -> float:
    """Convert Excel/string time values to decimal hours."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v * 24.0 if 0 <= v < 1 else v
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return float(value.hour) + float(value.minute) / 60.0 + float(getattr(value, "second", 0)) / 3600.0
    text = str(value).strip()
    if ":" in text:
        parts = text.split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        sec = int(float(parts[2])) if len(parts) > 2 else 0
        return h + m / 60.0 + sec / 3600.0
    return float(text)


def preprocess_data(base_dir: str = ".", green_radius_km: float = 10.0) -> PreprocessResult:
    """Run all preprocessing tasks and return structured result."""
    orders, dist, coords, tw = load_raw_data(base_dir)

    orders = _norm_cols(orders, {
        "订单编号": ["订单", "order"],
        "客户编号": ["目标客户", "客户", "customer"],
        "重量": ["重量", "kg", "weight"],
        "体积": ["体积", "m3", "立方", "volume"],
    })
    coords = _norm_cols(coords, {
        "客户编号": ["客户", "编号", "id"],
        "x": ["x", "横", "经度"],
        "y": ["y", "纵", "纬度"],
    })
    tw = _norm_cols(tw, {
        "客户编号": ["客户", "编号", "id"],
        "最早": ["最早", "开始", "start"],
        "最晚": ["最晚", "结束", "end"],
    })

    if "客户编号" not in orders:
        raise ValueError("订单信息.xlsx 未识别到客户编号列")
    for col in ["重量", "体积"]:
        if col not in orders:
            raise ValueError(f"订单信息.xlsx 未识别到{col}列")
    if "客户编号" not in coords:
        raise ValueError("客户坐标信息.xlsx 未识别到客户编号列")
    for col in ["x", "y"]:
        if col not in coords:
            raise ValueError(f"客户坐标信息.xlsx 未识别到{col}坐标列")
    if "客户编号" not in tw:
        raise ValueError("时间窗.xlsx 未识别到客户编号列")
    for col in ["最早", "最晚"]:
        if col not in tw:
            raise ValueError(f"时间窗.xlsx 未识别到{col}时间列")

    orders["客户编号"] = orders["客户编号"].astype(int)
    coords["客户编号"] = coords["客户编号"].astype(int)

    missing_entries = []
    for col in ["重量", "体积"]:
        global_mean = orders[col].mean(skipna=True)
        customer_mean = orders.groupby("客户编号")[col].transform("mean")
        missing_mask = orders[col].isna()
        fill_vals = customer_mean.where(customer_mean.notna(), global_mean)
        orders.loc[missing_mask, col] = fill_vals[missing_mask]
        for idx in orders.index[missing_mask]:
            missing_entries.append(
                {
                    "row": int(idx),
                    "customer_id": int(orders.at[idx, "客户编号"]),
                    "field": col,
                    "method": "customer_mean" if pd.notna(customer_mean.iloc[idx]) else "global_mean",
                    "filled_value": float(orders.at[idx, col]),
                }
            )

    demand = orders.groupby("客户编号", as_index=False)[["重量", "体积"]].sum()
    valid_customers = sorted(demand["客户编号"].astype(int).unique().tolist())

    customers_all = sorted(coords["客户编号"].astype(int).unique().tolist()) if "客户编号" in coords else valid_customers
    customer_nodes = [c for c in customers_all if c != 0]
    no_order_customers = sorted(list(set(customer_nodes) - set(valid_customers)))

    coords = coords.copy()
    coords["x"] = coords["x"].astype(float)
    coords["y"] = coords["y"].astype(float)
    coords["distance_to_center"] = np.sqrt(coords["x"] ** 2 + coords["y"] ** 2)
    coords["is_green"] = (coords["distance_to_center"] <= green_radius_km) & (coords["客户编号"] != 0)
    green_customers = sorted(coords.loc[coords["is_green"], "客户编号"].astype(int).tolist())
    valid_green_customers = sorted(list(set(green_customers) & set(valid_customers)))

    tw = tw[["客户编号", "最早", "最晚"]].copy()
    tw["客户编号"] = tw["客户编号"].astype(int)
    tw["最早"] = tw["最早"].apply(_to_hour)
    tw["最晚"] = tw["最晚"].apply(_to_hour)

    try:
        dist.index = dist.index.astype(int)
        dist.columns = dist.columns.astype(int)
    except Exception:
        pass

    return PreprocessResult(
        orders_raw=orders,
        customers=coords,
        distance_matrix=dist,
        time_windows=tw,
        customer_demand=demand,
        valid_customers=valid_customers,
        no_order_customers=no_order_customers,
        green_customers=green_customers,
        valid_green_customers=valid_green_customers,
        missing_report=pd.DataFrame(missing_entries),
    )
