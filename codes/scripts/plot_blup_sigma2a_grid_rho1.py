"""
plot_blup_sigma2a_grid_rho1.py — Boxplots of BLUP sigma2_a estimates across
methods, combined over all rho values, with the stabilizing model fixed at
rho=1.

Usage:
    python scripts/plot_blup_sigma2a_grid_rho1.py --output plots/blup_sigma2a_estimates_rho1_fixed.pdf
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Nimbus Sans"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot BLUP sigma2_a estimates across methods and parameter settings with fixed rho=1."
    )
    parser.add_argument("--results-base", default="results/blup_rho1_fixed")
    parser.add_argument("--output", default="plots/blup_sigma2a_estimates_rho1_fixed.pdf")
    return parser.parse_args()


def load_config():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    sim = cfg["simulation_parameters"]
    vs_scales = [float(v) for v in sim["VS_SCALE"]]
    sigma2_as = [float(s) for s in sim["sigma2_a"]]
    alpha_grid = [float(a) for a in sim["ALPHA_GRID"]]
    rhos = sorted(float(r) for r in sim["RHO"])
    return vs_scales, sigma2_as, alpha_grid, rhos


def alpha_tag(alpha_model: float) -> str:
    return f"{alpha_model:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def alpha_colors(n_alpha: int) -> list[str]:
    palette = ["#f3a14b", "#5ca55c", "#d95f5f", "#8c6bb1", "#6baed6", "#e377c2"]
    if n_alpha <= len(palette):
        return palette[:n_alpha]
    cmap = plt.get_cmap("tab10")
    return [matplotlib.colors.to_hex(cmap(i % 10)) for i in range(n_alpha)]


def display_rho_sq(rho_true: float) -> str:
    return f"{round(rho_true ** 2, 1):.1f}"


def load_results_frame(results_base: str, rho_true: float, vs_scale: float, sigma2_a: float) -> pd.DataFrame:
    dir_path = f"{results_base}/rho{rho_true}/vs{vs_scale}_sa{sigma2_a}"
    csvs = sorted(glob.glob(f"{dir_path}/rep_*.csv"))
    if not csvs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)


def detect_method_specs(sample_df: pd.DataFrame, alpha_grid: list[float]):
    method_specs = [("sigma2_a_est", "Evolutionary", "#4c9be8")]
    alpha_palette = alpha_colors(len(alpha_grid))

    for alpha_model, color in zip(alpha_grid, alpha_palette):
        alpha_col = f"sigma2_a_alpha_{alpha_tag(alpha_model)}_est"
        if alpha_col in sample_df.columns:
            method_specs.append((alpha_col, fr"$\alpha={alpha_model:g}$", color))
            continue

        if np.isclose(alpha_model, 1.0) and "sigma2_a_vanilla_est" in sample_df.columns:
            method_specs.append(("sigma2_a_vanilla_est", r"$\alpha=1$", color))

    return method_specs


def plot_method_groups(ax, series_by_method, base_positions, xticklabels, colors):
    n_methods = len(series_by_method)
    offsets = np.linspace(-0.36, 0.36, n_methods)
    width = min(0.72 / max(n_methods, 1), 0.22)
    bp_common = dict(
        patch_artist=True,
        widths=width,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=2.0),
    )

    for idx, data in enumerate(series_by_method):
        positions = [x + offsets[idx] for x in base_positions]
        ax.boxplot(
            data,
            positions=positions,
            boxprops=dict(facecolor=colors[idx], edgecolor=colors[idx], linewidth=1.2),
            whiskerprops=dict(color=colors[idx], linewidth=1.2),
            capprops=dict(color=colors[idx], linewidth=1.2),
            **bp_common,
        )

    ax.set_xticks(base_positions)
    ax.set_xticklabels(xticklabels, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)


def pathlike_parent(path_str: str) -> str:
    parent = os.path.dirname(path_str)
    return parent if parent else "."


def main() -> int:
    args = parse_args()
    os.makedirs(pathlike_parent(args.output), exist_ok=True)

    vs_scales, sigma2_as, alpha_grid, rhos = load_config()
    ws_trues = [v / 2.0 for v in vs_scales]

    sample_df = pd.DataFrame()
    for rho_true in rhos:
        for sa_true in sigma2_as:
            for vs_scale in vs_scales:
                sample_df = load_results_frame(args.results_base, rho_true, vs_scale, sa_true)
                if not sample_df.empty:
                    break
            if not sample_df.empty:
                break
        if not sample_df.empty:
            break

    if sample_df.empty:
        print("No BLUP fixed-rho1 result CSVs found.")
        return 0

    method_specs = detect_method_specs(sample_df, alpha_grid)
    if not method_specs:
        print("No sigma2_a estimate columns found in BLUP fixed-rho1 results.")
        return 0

    fig, axes = plt.subplots(
        len(rhos),
        len(sigma2_as),
        figsize=(2.88 * len(sigma2_as), 2.592 * len(rhos) + 0.72),
        sharey=False,
        squeeze=False,
    )
    fig.suptitle(r"Estimates of $\sigma_b^2$ with fixed $\rho_{ab}^2=1.0$", fontsize=16, y=0.99)
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=8, label=label)
        for _, label, color in method_specs
    ]
    handles.append(
        plt.Line2D([0], [0], color="#444444", linestyle=":", linewidth=2.0, label="True value")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(handles),
        frameon=False,
        fontsize=10,
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.supxlabel(r"True $W_s$", fontsize=12, y=0.01)

    for row, rho_true in enumerate(rhos):
        for col, sa_true in enumerate(sigma2_as):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(fr"True $\sigma_b^2 = {sa_true}$", fontsize=13)

            series_by_method = [[] for _ in method_specs]
            xticklabels = []
            base_positions = []

            for i, vs_scale in enumerate(vs_scales):
                df = load_results_frame(args.results_base, rho_true, vs_scale, sa_true)
                for method_idx, (column_name, _, _) in enumerate(method_specs):
                    if df.empty or column_name not in df.columns:
                        series_by_method[method_idx].append(pd.Series(dtype=float))
                    else:
                        series_by_method[method_idx].append(df[column_name].dropna())
                base_positions.append(i + 1)
                xticklabels.append(f"{ws_trues[i]:.3g}")

            plot_method_groups(
                ax,
                series_by_method=series_by_method,
                base_positions=base_positions,
                xticklabels=xticklabels,
                colors=[spec[2] for spec in method_specs],
            )

            ax.axhline(sa_true, color="#444444", linestyle=":", linewidth=2.0, zorder=1)
            if col == 0:
                ax.set_ylabel(
                    fr"$\rho_{{ab}}^2={display_rho_sq(rho_true)}$"
                    "\n"
                    r"Estimated $\sigma_b^2$",
                    fontsize=12,
                )

    plt.tight_layout(rect=[0.02, 0.0, 1.0, 0.975])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved fixed-rho1 sigma2_a estimate plot to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
