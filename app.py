#!/usr/bin/env python3
from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

from src.soilcue.io import load_config
from src.soilcue.model import CO2_NAME, Edge, Pool, SoilCUEConfig, validate_config
from src.soilcue.sim import SimulationResult, run_simulation


st.set_page_config(page_title="Soil CUE 动态控制台", layout="wide")
st.title("Soil CUE 动态网络可视化控制台")
st.caption("池、路径和环境参数调整后会自动重算，无需先点击输出。")


def _cfg_to_frames(cfg: SoilCUEConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    pools_df = pd.DataFrame(
        [{"name": p.name, "type": p.type, "C0": float(p.C0)} for p in cfg.pools]
    )
    edges_df = pd.DataFrame(
        [
            {
                "from": e.from_pool,
                "to": e.to_pool,
                "k0": float(e.k0),
                "y0": float(e.y0),
            }
            for e in cfg.edges
        ]
    )
    return pools_df, edges_df


def _frames_to_cfg(base_cfg: SoilCUEConfig, pools_df: pd.DataFrame, edges_df: pd.DataFrame) -> SoilCUEConfig:
    pools = [
        Pool(name=str(row["name"]).strip(), type=str(row["type"]).strip(), C0=float(row["C0"]))
        for _, row in pools_df.iterrows()
        if str(row["name"]).strip()
    ]

    edges = []
    for _, row in edges_df.iterrows():
        from_pool = str(row["from"]).strip()
        to_pool = str(row["to"]).strip()
        if not from_pool or not to_pool:
            continue
        y0 = 0.0 if to_pool == CO2_NAME else float(row["y0"])
        edges.append(
            Edge(
                from_pool=from_pool,
                to_pool=to_pool,
                k0=max(0.0, float(row["k0"])),
                y0=min(1.0, max(0.0, y0)),
            )
        )

    cfg = replace(base_cfg, pools=pools, edges=edges)
    validate_config(cfg)
    return cfg


def _summary(result: SimulationResult, cfg: SoilCUEConfig) -> dict:
    final = result.C[-1]
    return {
        "final_pools": {p.name: float(v) for p, v in zip(cfg.pools, final)},
        "cumulative_co2": float((result.F_CO2[:-1] * (result.t[1:] - result.t[:-1])).sum()),
        "mean_cue_eff": float(pd.Series(result.CUE_eff).mean(skipna=True)),
    }


def _build_snapshot(result: SimulationResult, cfg: SoilCUEConfig, summary: dict) -> bytes:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for i, p in enumerate(cfg.pools):
        ax.plot(result.t, result.C[:, i], label=p.name)
    ax.set_title("Pool trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel("C")
    ax.legend(fontsize=8)

    axes[0, 1].plot(result.t, result.F_CO2)
    axes[0, 1].set_title("CO2 Flux")

    axes[1, 0].plot(result.t, result.CUE_eff)
    axes[1, 0].set_title("System CUE")

    axes[1, 1].axis("off")
    text_lines = [
        "Summary",
        f"Cumulative CO2: {summary['cumulative_co2']:.4f}",
        f"Mean CUE_eff: {summary['mean_cue_eff']:.4f}",
        "Final pools:",
    ]
    text_lines.extend([f"- {k}: {v:.4f}" for k, v in summary["final_pools"].items()])
    axes[1, 1].text(0.0, 1.0, "\n".join(text_lines), va="top", fontsize=10)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


with st.sidebar:
    st.header("配置")
    cfg_file = st.file_uploader("上传 YAML 配置", type=["yaml", "yml"])
    t_end = st.number_input("t_end", min_value=1.0, value=200.0, step=10.0)
    dt_out = st.number_input("dt_out", min_value=0.1, value=1.0, step=0.1)

if cfg_file:
    raw_cfg = yaml.safe_load(cfg_file.getvalue())
    tmp_cfg_path = Path(".streamlit_tmp_config.yaml")
    tmp_cfg_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")
    base_cfg = load_config(tmp_cfg_path)
else:
    base_cfg = load_config("example_config.yaml")

pools_df, edges_df = _cfg_to_frames(base_cfg)

st.subheader("池设置（可实时编辑）")
edited_pools = st.data_editor(
    pools_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={"C0": st.column_config.NumberColumn(min_value=0.0)},
    key="pools_editor",
)

st.subheader("路径设置（可实时编辑）")
edited_edges = st.data_editor(
    edges_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "k0": st.column_config.NumberColumn(min_value=0.0),
        "y0": st.column_config.NumberColumn(min_value=0.0, max_value=1.0),
    },
    key="edges_editor",
)

try:
    cfg = _frames_to_cfg(base_cfg, edited_pools, edited_edges)
except Exception as exc:
    st.error(f"配置无效：{exc}")
    st.stop()

result = run_simulation(cfg, t_end=float(t_end), dt_out=float(dt_out))
summary = _summary(result, cfg)

st.success("已按当前 UI 自动更新结果。")

m1, m2 = st.columns(2)
m1.metric("累计 CO2", f"{summary['cumulative_co2']:.4f}")
m2.metric("平均 CUE_eff", f"{summary['mean_cue_eff']:.4f}")

st.subheader("最终各池结果")
st.dataframe(
    pd.DataFrame(
        [{"pool": k, "final_C": v} for k, v in summary["final_pools"].items()]
    ),
    use_container_width=True,
)

plot_tab1, plot_tab2, plot_tab3 = st.tabs(["池轨迹", "CO2", "CUE_eff"])

with plot_tab1:
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, p in enumerate(cfg.pools):
        ax.plot(result.t, result.C[:, i], label=p.name)
    ax.set_xlabel("Time")
    ax.set_ylabel("C")
    ax.legend(ncol=3, fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

with plot_tab2:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.t, result.F_CO2)
    ax.set_xlabel("Time")
    ax.set_ylabel("F_CO2")
    st.pyplot(fig)
    plt.close(fig)

with plot_tab3:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.t, result.CUE_eff)
    ax.set_xlabel("Time")
    ax.set_ylabel("CUE_eff")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("输出")
output_options = st.multiselect(
    "点击输出时选择内容",
    ["当前UI页面图片", "汇总统计信息"],
    default=["当前UI页面图片", "汇总统计信息"],
)

if st.button("输出"):
    st.info("已按所选项生成输出文件。")
    if "当前UI页面图片" in output_options:
        png_data = _build_snapshot(result, cfg, summary)
        st.download_button(
            "下载UI快照(PNG)",
            data=png_data,
            file_name="ui_snapshot.png",
            mime="image/png",
        )
    if "汇总统计信息" in output_options:
        st.download_button(
            "下载汇总(JSON)",
            data=json.dumps(summary, ensure_ascii=False, indent=2),
            file_name="summary_stats.json",
            mime="application/json",
        )
