from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np


ALLOWED_POOL_TYPES = {"input", "doc", "pom", "mb", "enz", "nec", "maom", "inert"}
CO2_NAME = "CO2"


@dataclass
class Pool:
    name: str
    type: str
    C0: float


@dataclass
class DriverParams:
    T: Dict[str, float] = field(default_factory=dict)
    M: Dict[str, float] = field(default_factory=dict)
    Y_M: Dict[str, float] = field(default_factory=dict)
    P: Dict[str, float] = field(default_factory=dict)
    Mc: Dict[str, float] = field(default_factory=dict)
    G: Dict[str, float] = field(default_factory=dict)


@dataclass
class Edge:
    from_pool: str
    to_pool: str
    k0: float
    y0: float
    drivers: DriverParams = field(default_factory=DriverParams)
    auto_generated: bool = False


@dataclass
class Settings:
    auto_edges: bool = True
    include_optional_edges: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ThetaConfig:
    constant: Dict[str, float] = field(default_factory=dict)
    time_varying: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SoilCUEConfig:
    pools: List[Pool]
    edges: List[Edge]
    settings: Settings
    theta: ThetaConfig


# (from_type, to_type): (k0, y0)
DEFAULT_EDGE_SEEDS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("input", "doc"): (0.25, 0.80),
    ("doc", "mb"): (0.08, 0.55),
    ("doc", CO2_NAME): (0.05, 0.0),
    ("mb", "nec"): (0.06, 0.45),
    ("mb", CO2_NAME): (0.07, 0.0),
    ("pom", "doc"): (0.03, 0.65),
    ("pom", CO2_NAME): (0.01, 0.0),
    ("nec", "maom"): (0.04, 0.60),
    ("nec", "doc"): (0.02, 0.45),
    ("nec", CO2_NAME): (0.01, 0.0),
    ("maom", "doc"): (0.01, 0.35),
    ("maom", CO2_NAME): (0.002, 0.0),
    ("maom", "inert"): (0.002, 0.90),
    ("inert", "doc"): (0.0005, 0.10),
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _ramp_value(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    frac = (t - t0) / (t1 - t0)
    return v0 + frac * (v1 - v0)


def evaluate_theta(t: float, theta_cfg: ThetaConfig) -> Dict[str, float]:
    theta = {
        "T": float(theta_cfg.constant.get("T", 20.0)),
        "Moisture": float(theta_cfg.constant.get("Moisture", 0.6)),
        "Plant": float(theta_cfg.constant.get("Plant", 0.0)),
        "Community": float(theta_cfg.constant.get("Community", 0.0)),
        "Geology": float(theta_cfg.constant.get("Geology", 0.0)),
    }
    tv = theta_cfg.time_varying

    if "T_ramp" in tv:
        r = tv["T_ramp"]
        theta["T"] = _ramp_value(t, r["t0"], r["t1"], r["v0"], r["v1"])
    if "M_ramp" in tv:
        r = tv["M_ramp"]
        theta["Moisture"] = _ramp_value(t, r["t0"], r["t1"], r["v0"], r["v1"])
    return theta


# Driver templates

def fT_q10(T: float, q10: float, tref: float) -> float:
    return q10 ** ((T - tref) / 10.0)


def fM_bell(M: float, Mopt: float, width: float, floor: float) -> float:
    width = max(width, 1e-9)
    return floor + (1.0 - floor) * np.exp(-((M - Mopt) / width) ** 2)


def delta_y_moisture(M: float, Mopt: float, width: float, alpha: float) -> float:
    width = max(width, 1e-9)
    return -alpha * abs((M - Mopt) / width)


def k_multiplier(edge: Edge, theta: Dict[str, float]) -> float:
    mult = 1.0

    tcfg = edge.drivers.T
    q10 = float(tcfg.get("q10", 2.0))
    tref = float(tcfg.get("tref", 20.0))
    mult *= fT_q10(theta["T"], q10, tref)

    mcfg = edge.drivers.M
    Mopt = float(mcfg.get("Mopt", 0.6))
    width = float(mcfg.get("width", 0.25))
    floor = float(mcfg.get("floor", 0.1))
    mult *= fM_bell(theta["Moisture"], Mopt, width, floor)

    pcfg = edge.drivers.P
    mult *= (1.0 + float(pcfg.get("coef", 0.0)) * theta["Plant"])

    ccfg = edge.drivers.Mc
    mult *= (1.0 + float(ccfg.get("coef", 0.0)) * theta["Community"])

    gcfg = edge.drivers.G
    beta = float(gcfg.get("beta", 0.0))
    gamma = float(gcfg.get("gamma", 0.0))
    geology = theta["Geology"]

    # Requested special geology effects for specific edge classes.
    if edge.from_pool and edge.to_pool and edge.to_pool != CO2_NAME:
        # Actual matching happens by names in matrix builder using pool type map,
        # so here we only apply optional edge-level generic factors.
        mult *= (1.0 + beta * geology)
        mult *= np.exp(-gamma * geology)

    return max(mult, 0.0)


def y_effective(edge: Edge, theta: Dict[str, float]) -> float:
    if edge.to_pool == CO2_NAME:
        return 0.0

    y = edge.y0
    mcfg = edge.drivers.M
    ymcfg = edge.drivers.Y_M
    Mopt = float(mcfg.get("Mopt", 0.6))
    width = float(mcfg.get("width", 0.25))
    alpha = float(ymcfg.get("alpha", 0.03))
    y += delta_y_moisture(theta["Moisture"], Mopt, width, alpha)

    ccfg = edge.drivers.Mc
    y += float(ccfg.get("y_shift", 0.0)) * theta["Community"]

    return _clip01(y)


def validate_config(cfg: SoilCUEConfig) -> None:
    names = [p.name for p in cfg.pools]
    if len(names) != len(set(names)):
        raise ValueError("Pool names must be unique.")

    for p in cfg.pools:
        if p.type not in ALLOWED_POOL_TYPES:
            raise ValueError(f"Unsupported pool type '{p.type}' for pool '{p.name}'.")

    pool_set = set(names)
    for e in cfg.edges:
        if e.k0 < 0:
            raise ValueError(f"Edge {e.from_pool}->{e.to_pool} has negative k0.")
        if not (0.0 <= e.y0 <= 1.0):
            raise ValueError(f"Edge {e.from_pool}->{e.to_pool} has y0 outside [0,1].")
        if e.from_pool not in pool_set:
            raise ValueError(f"Edge source '{e.from_pool}' not in pools.")
        if e.to_pool != CO2_NAME and e.to_pool not in pool_set:
            raise ValueError(f"Edge target '{e.to_pool}' not in pools and not CO2.")

    m = cfg.theta.constant.get("Moisture", 0.6)
    if not (0.0 <= m <= 1.0):
        warnings.warn("Constant Moisture is outside [0,1].", RuntimeWarning)


def build_matrices(
    cfg: SoilCUEConfig,
    t: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Returns K, Y, A, l, pool_index_map.
    - K and Y are (n,n) for internal edges only.
    - l is respiration vector (n,), includes internal inefficiency + edges to CO2.
    """
    theta = evaluate_theta(t, cfg.theta)
    if not (0.0 <= theta["Moisture"] <= 1.0):
        warnings.warn("Time-varying Moisture is outside [0,1].", RuntimeWarning)

    pool_idx = {p.name: i for i, p in enumerate(cfg.pools)}
    type_map = {p.name: p.type for p in cfg.pools}
    n = len(cfg.pools)
    K = np.zeros((n, n), dtype=float)
    Y = np.zeros((n, n), dtype=float)
    l = np.zeros(n, dtype=float)

    for e in cfg.edges:
        i = pool_idx[e.from_pool]

        mult = k_multiplier(e, theta)

        # Hard-coded geology effects required by edge class.
        g = theta["Geology"]
        if e.to_pool != CO2_NAME:
            from_t = type_map[e.from_pool]
            to_t = type_map[e.to_pool]
            if from_t == "nec" and to_t == "maom":
                betaG = float(e.drivers.G.get("beta", 0.0))
                mult *= (1.0 + betaG * g)
            if from_t == "maom" and to_t == "doc":
                gammaG = float(e.drivers.G.get("gamma", 0.0))
                mult *= np.exp(-gammaG * g)

        k_eff = e.k0 * mult

        if e.to_pool == CO2_NAME:
            l[i] += k_eff
            continue

        j = pool_idx[e.to_pool]
        y_eff = y_effective(e, theta)
        K[i, j] += k_eff
        Y[i, j] = y_eff

    A = Y * K

    # l_u = sum_v (1-y_uv)k_uv + sum_{u->CO2} k
    l += np.sum((1.0 - Y) * K, axis=1)

    return K, Y, A, l, pool_idx


def build_M_from_AK(A: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    D_uu = sum_v k_{u->v}
    M = A^T - D
    """
    D = np.diag(np.sum(K, axis=1))
    return A.T - D
