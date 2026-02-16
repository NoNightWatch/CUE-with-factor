# Dynamic multi-pool soil CUE network simulator

Runtime-defined soil carbon pool network with edge-level turnover (`K`) and CUE (`Y`) controls.

## Features

- Dynamic pools from YAML (no code edits needed when adding/removing pools).
- Optional default edge generation by pool **type** rules.
- Supports explicit edge overrides/partials merged onto generated defaults.
- ODE system:
  - `A = Y ∘ K`
  - `D_uu = Σ_v k_{u→v}`
  - `M = A^T - D`
  - `dC/dt = M C + u(t,θ)` with `u=0` by default
- CO2 is handled as a special sink edge target, not a state pool.
- Outputs CSV + PNG plots + summary statistics.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python simulate.py --config example_config.yaml --t_end 200 --dt_out 1 --outdir outputs --seed 0
```

## Config notes

- If `settings.auto_edges: true`, generated default edges are created from pool type rules.
- Optional defaults are controlled in `settings.include_optional_edges`.
- Explicit `edges:` entries override defaults by exact `(from, to)` pair.
- Driver parameters can be set per-edge; omitted parameters use built-in defaults.

## Output files

- `pools_timeseries.csv`: pools over time
- `fluxes.csv`: `F_CO2`, `F_prod`, `F_total`, `CUE_eff`
- `pools_trajectories.png`, `co2_flux.png`, `cue_eff.png`
- `effective_edges.yaml`: final merged edge set used for simulation
