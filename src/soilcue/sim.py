from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.integrate import solve_ivp

from .model import SoilCUEConfig, build_M_from_AK, build_matrices


@dataclass
class SimulationResult:
    t: np.ndarray
    C: np.ndarray
    F_CO2: np.ndarray
    F_prod: np.ndarray
    F_total: np.ndarray
    CUE_eff: np.ndarray


def _input_vector(t: float, cfg: SoilCUEConfig, n: int) -> np.ndarray:
    # Optional user extension point: currently default zero, runtime-safe.
    return np.zeros(n, dtype=float)


def run_simulation(cfg: SoilCUEConfig, t_end: float, dt_out: float) -> SimulationResult:
    n = len(cfg.pools)
    C0 = np.array([p.C0 for p in cfg.pools], dtype=float)
    t_eval = np.arange(0.0, t_end + 1e-12, dt_out)

    def rhs(t: float, C: np.ndarray) -> np.ndarray:
        K, Y, A, _, _ = build_matrices(cfg, t)
        M = build_M_from_AK(A, K)
        return M @ C + _input_vector(t, cfg, n)

    sol = solve_ivp(rhs, (0.0, t_end), C0, t_eval=t_eval, method="RK45", vectorized=False)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    C = sol.y.T
    F_CO2 = np.zeros_like(sol.t)
    F_prod = np.zeros_like(sol.t)
    F_total = np.zeros_like(sol.t)

    for idx, t in enumerate(sol.t):
        K, _, A, l, _ = build_matrices(cfg, t)
        Ct = C[idx]
        F_CO2[idx] = float(l @ Ct)
        F_prod[idx] = float(np.sum(A.T @ Ct))
        F_total[idx] = float(np.sum(K.T @ Ct))

    with np.errstate(divide="ignore", invalid="ignore"):
        CUE_eff = np.where(F_total > 0.0, F_prod / F_total, np.nan)

    return SimulationResult(t=sol.t, C=C, F_CO2=F_CO2, F_prod=F_prod, F_total=F_total, CUE_eff=CUE_eff)


def save_outputs(cfg: SoilCUEConfig, result: SimulationResult, outdir: str | Path) -> Dict[str, Path]:
    import csv
    import matplotlib.pyplot as plt

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    pools_csv = out / "pools_timeseries.csv"
    with open(pools_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", *[p.name for p in cfg.pools]])
        for i, t in enumerate(result.t):
            w.writerow([float(t), *map(float, result.C[i])])

    flux_csv = out / "fluxes.csv"
    with open(flux_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "F_CO2", "F_prod", "F_total", "CUE_eff"])
        for i, t in enumerate(result.t):
            w.writerow([float(t), float(result.F_CO2[i]), float(result.F_prod[i]), float(result.F_total[i]), float(result.CUE_eff[i])])

    fig1 = out / "pools_trajectories.png"
    plt.figure(figsize=(10, 6))
    for i, p in enumerate(cfg.pools):
        plt.plot(result.t, result.C[:, i], label=p.name)
    plt.xlabel("Time")
    plt.ylabel("Carbon stock C")
    plt.title("Pool trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=150)
    plt.close()

    fig2 = out / "co2_flux.png"
    plt.figure(figsize=(8, 4))
    plt.plot(result.t, result.F_CO2)
    plt.xlabel("Time")
    plt.ylabel("F_CO2")
    plt.title("Respiration flux")
    plt.tight_layout()
    plt.savefig(fig2, dpi=150)
    plt.close()

    fig3 = out / "cue_eff.png"
    plt.figure(figsize=(8, 4))
    plt.plot(result.t, result.CUE_eff)
    plt.xlabel("Time")
    plt.ylabel("CUE_eff")
    plt.title("System CUE")
    plt.tight_layout()
    plt.savefig(fig3, dpi=150)
    plt.close()

    cum_co2 = float(np.trapz(result.F_CO2, result.t))
    mean_cue = float(np.nanmean(result.CUE_eff))
    final = result.C[-1]
    print("\nSimulation summary")
    print("------------------")
    for p, v in zip(cfg.pools, final):
        print(f"Final C[{p.name}] = {v:.6g}")
    print(f"Cumulative CO2 = {cum_co2:.6g}")
    print(f"Mean CUE_eff = {mean_cue:.6g}")

    return {
        "pools_csv": pools_csv,
        "flux_csv": flux_csv,
        "pools_png": fig1,
        "co2_png": fig2,
        "cue_png": fig3,
    }
