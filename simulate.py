#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import numpy as np

from src.soilcue.io import dump_effective_edges_yaml, load_config
from src.soilcue.sim import run_simulation, save_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic multi-pool soil CUE network simulator")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--t_end", type=float, default=200.0)
    p.add_argument("--dt_out", type=float, default=1.0)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    result = run_simulation(cfg, t_end=args.t_end, dt_out=args.dt_out)
    paths = save_outputs(cfg, result, outdir=args.outdir)
    dump_effective_edges_yaml(cfg, f"{args.outdir}/effective_edges.yaml")

    print("\nSaved files:")
    for k, v in paths.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
