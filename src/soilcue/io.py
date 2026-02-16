from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from .model import (
    CO2_NAME,
    DEFAULT_EDGE_SEEDS,
    DriverParams,
    Edge,
    Pool,
    Settings,
    SoilCUEConfig,
    ThetaConfig,
    validate_config,
)


def _parse_pool(d: Dict) -> Pool:
    return Pool(name=str(d["name"]), type=str(d["type"]), C0=float(d.get("C0", 0.0)))


def _parse_drivers(d: Dict | None) -> DriverParams:
    d = d or {}
    return DriverParams(
        T=dict(d.get("T", {})),
        M=dict(d.get("M", {})),
        Y_M=dict(d.get("Y_M", {})),
        P=dict(d.get("P", {})),
        Mc=dict(d.get("Mc", {})),
        G=dict(d.get("G", {})),
    )


def _parse_edge(d: Dict, auto_generated: bool = False) -> Edge:
    to_pool = str(d["to"])
    y0_default = 0.0 if to_pool == CO2_NAME else 0.5
    return Edge(
        from_pool=str(d["from"]),
        to_pool=to_pool,
        k0=float(d.get("k0", 0.0)),
        y0=float(d.get("y0", y0_default)),
        drivers=_parse_drivers(d.get("drivers")),
        auto_generated=auto_generated,
    )


def _default_edge_specs_for_types(include_optional: Dict[str, bool]) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = [
        ("input", "doc"),
        ("doc", "mb"),
        ("doc", CO2_NAME),
        ("mb", "nec"),
        ("mb", CO2_NAME),
        ("pom", "doc"),
        ("pom", CO2_NAME),
        ("nec", "maom"),
        ("nec", "doc"),
        ("maom", "doc"),
    ]

    if include_optional.get("nec_to_co2", False):
        specs.append(("nec", CO2_NAME))
    if include_optional.get("maom_to_co2", False):
        specs.append(("maom", CO2_NAME))
    if include_optional.get("inert_pool", False):
        specs.extend([("maom", "inert"), ("inert", "doc")])

    return specs


def _generate_default_edges(pools: List[Pool], include_optional: Dict[str, bool]) -> List[Edge]:
    by_type: Dict[str, List[str]] = {}
    for p in pools:
        by_type.setdefault(p.type, []).append(p.name)

    generated: List[Edge] = []
    for from_type, to_type in _default_edge_specs_for_types(include_optional):
        if from_type not in by_type:
            continue

        dest_names = [CO2_NAME] if to_type == CO2_NAME else by_type.get(to_type, [])
        if not dest_names:
            continue

        k0, y0 = DEFAULT_EDGE_SEEDS[(from_type, to_type)]
        for from_name in by_type[from_type]:
            for to_name in dest_names:
                generated.append(
                    Edge(
                        from_pool=from_name,
                        to_pool=to_name,
                        k0=k0,
                        y0=0.0 if to_name == CO2_NAME else y0,
                        drivers=DriverParams(),
                        auto_generated=True,
                    )
                )
    return generated


def _merge_edges(default_edges: List[Edge], explicit_edges: List[Edge]) -> List[Edge]:
    merged: Dict[Tuple[str, str], Edge] = {(e.from_pool, e.to_pool): e for e in default_edges}

    for e in explicit_edges:
        key = (e.from_pool, e.to_pool)
        if key not in merged:
            merged[key] = e
            continue

        base = merged[key]
        # explicit edge overrides defaults, but keeps absent driver groups from base
        merged[key] = Edge(
            from_pool=e.from_pool,
            to_pool=e.to_pool,
            k0=e.k0,
            y0=e.y0,
            drivers=DriverParams(
                T={**base.drivers.T, **e.drivers.T},
                M={**base.drivers.M, **e.drivers.M},
                Y_M={**base.drivers.Y_M, **e.drivers.Y_M},
                P={**base.drivers.P, **e.drivers.P},
                Mc={**base.drivers.Mc, **e.drivers.Mc},
                G={**base.drivers.G, **e.drivers.G},
            ),
            auto_generated=False,
        )

    return list(merged.values())


def load_config(path: str | Path) -> SoilCUEConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    pools = [_parse_pool(p) for p in raw.get("pools", [])]
    settings_raw = raw.get("settings", {})
    settings = Settings(
        auto_edges=bool(settings_raw.get("auto_edges", True)),
        include_optional_edges=dict(settings_raw.get("include_optional_edges", {})),
    )

    theta_raw = raw.get("theta", {})
    theta = ThetaConfig(
        constant=dict(theta_raw.get("constant", {})),
        time_varying=dict(theta_raw.get("time_varying", {})),
    )

    explicit_edges = [_parse_edge(e, auto_generated=False) for e in raw.get("edges", [])]

    if settings.auto_edges:
        default_edges = _generate_default_edges(pools, settings.include_optional_edges)
        edges = _merge_edges(default_edges, explicit_edges)
    else:
        edges = explicit_edges

    cfg = SoilCUEConfig(pools=pools, edges=edges, settings=settings, theta=theta)
    validate_config(cfg)
    return cfg


def dump_effective_edges_yaml(cfg: SoilCUEConfig, path: str | Path) -> None:
    payload = {
        "edges": [
            {
                "from": e.from_pool,
                "to": e.to_pool,
                "k0": e.k0,
                "y0": e.y0,
                "drivers": asdict(e.drivers),
                "auto_generated": e.auto_generated,
            }
            for e in cfg.edges
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
