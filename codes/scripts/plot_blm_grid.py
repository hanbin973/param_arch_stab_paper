"""
plot_blm_grid.py — Generate combined BLM summary plots across all rho values.

The sigma2_a / W_S panel includes both uncorrected and corrected reference
lines.

Usage:
    python scripts/plot_blm_grid.py <out_sigma2_a_over_ws> <out_sigma2_a>
"""

import glob
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Nimbus Sans"]

UNCORRECTED_COLOR = "#7a7a7a"
CORRECTED_COLOR = "#e8462a"


def load_config():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sim = cfg["simulation_parameters"]
    l_int = int(float(sim["L"]))
    rec_rate = float(sim["REC_RATE"])
    return {
        "N": float(sim["N"]),
        "L": float(sim["L"]),
        "L_int": l_int,
        "mut_rate": float(sim["MUT_RATE"]),
        "rec_rate": rec_rate,
        "recomb_hmean": compute_recomb_hmean(l_int, rec_rate),
        "vs_scales": [float(v) for v in sim["VS_SCALE"]],
        "sigma2_as": [float(s) for s in sim["sigma2_a"]],
        "rhos": sorted(float(r) for r in sim["RHO"]),
    }


def compute_theta(n: float, mut_rate: float) -> float:
    return 4.0 * n * mut_rate


def compute_recomb_hmean(length: int, rec_rate: float, chunk_size: int = 1_000_000) -> float:
    if length < 2:
        raise ValueError(f"L must be at least 2 to compute the recombination harmonic mean; got {length}")
    if rec_rate <= 0.0:
        raise ValueError(f"REC_RATE must be positive to compute the recombination harmonic mean; got {rec_rate}")

    base = 1.0 - 2.0 * rec_rate
    total = 0.0
    tiny = np.finfo(np.float64).tiny

    for start in range(1, length, chunk_size):
        stop = min(length, start + chunk_size)
        d = np.arange(start, stop, dtype=np.float64)
        c_d = 0.5 * (1.0 - np.power(base, d))
        c_d = np.maximum(c_d, tiny)
        total += float(np.sum(2.0 * (length - d) / c_d))

    return (length * (length - 1.0)) / total


def compute_z(sigma2_a: float, vs_scale: float, n: float, l: float,
              theta: float, recomb_hmean: float) -> float:
    w_s = vs_scale
    v_s = 2.0 * n * vs_scale
    v_g = 2.0 * l * theta * sigma2_a / (2.0 + sigma2_a / w_s)
    return v_g / (2.0 * recomb_hmean * v_s)


def display_rho_sq(rho_true: float) -> str:
    return f"{round(rho_true ** 2, 1):.1f}"


def load_results_frame(results_root: str, rho_true: float, vs_scale: float, sigma2_a: float) -> pd.DataFrame:
    dir_path = f"{results_root}/rho{rho_true}/vs{vs_scale}_sa{sigma2_a}"
    csvs = sorted(glob.glob(f"{dir_path}/rep_*.csv"))
    if not csvs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)


def load_ratio_estimates(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if "sigma2_a_over_Ws_est" in df.columns:
        return df["sigma2_a_over_Ws_est"].dropna()
    if {"sigma2_a_est", "Ws_est"}.issubset(df.columns):
        return (df["sigma2_a_est"] / df["Ws_est"]).dropna()
    return pd.Series(dtype=float)


def boxplot_style():
    return dict(
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        boxprops=dict(facecolor="#4c9be8", color="#1a3a5c", linewidth=1.5),
        medianprops=dict(color=CORRECTED_COLOR, linewidth=2.5),
        whiskerprops=dict(color="#1a3a5c", linewidth=1.5),
        capprops=dict(color="#1a3a5c", linewidth=1.5),
        flierprops=dict(
            marker="o",
            markerfacecolor="#4c9be8",
            markeredgecolor="#1a3a5c",
            markersize=4,
            alpha=0.5,
        ),
    )


def reference_legend_handles():
    return [
        plt.Line2D(
            [0], [0],
            color=UNCORRECTED_COLOR,
            linestyle=":",
            linewidth=2,
            label=r"Uncorrected: $\sigma_a^2 / W_S$",
        ),
        plt.Line2D(
            [0], [0],
            color=CORRECTED_COLOR,
            linestyle=":",
            linewidth=2,
            label="Bulmer correction",
        ),
    ]


def plot_sigma2a_over_ws(results_root: str, cfg: dict, output: str) -> None:
    n = cfg["N"]
    l = cfg["L"]
    vs_scales = sorted(cfg["vs_scales"])
    sigma2_as = cfg["sigma2_as"]
    rhos = cfg["rhos"]
    theta = compute_theta(n, cfg["mut_rate"])
    recomb_hmean = cfg["recomb_hmean"]
    bp_kw = boxplot_style()

    fig, axes = plt.subplots(
        len(rhos),
        len(sigma2_as),
        figsize=(3.2 * len(sigma2_as), 2.88 * len(rhos) + 0.96),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    fig.suptitle(
        r"Estimates of $\sigma_a^2 / W_S$ with uncorrected and corrected reference lines",
        fontsize=15,
        y=0.99,
    )
    fig.legend(
        handles=reference_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=2.8,
        columnspacing=2.0,
    )
    fig.supxlabel(r"True $W_S$", fontsize=14, y=0.006)

    for row, rho_true in enumerate(rhos):
        row_unused = np.isclose(rho_true, 0.0)
        for col, sa_true in enumerate(sigma2_as):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(fr"True $\sigma_a^2 = {sa_true}$", fontsize=14)

            data_to_plot = []
            positions = []
            labels = []

            for i, vs_scale in enumerate(vs_scales):
                base_ref = sa_true / vs_scale
                pos = i + 1
                positions.append(pos)
                labels.append(f"{vs_scale:g}")

                if row_unused:
                    continue

                z = compute_z(sa_true, vs_scale, n=n, l=l, theta=theta, recomb_hmean=recomb_hmean)
                corrected_ref = base_ref * ((1.0 - z) ** 2)
                df = load_results_frame(results_root, rho_true, vs_scale, sa_true)
                data_to_plot.append(load_ratio_estimates(df))
                ax.hlines(
                    base_ref,
                    xmin=pos - 0.4,
                    xmax=pos + 0.4,
                    color=UNCORRECTED_COLOR,
                    linestyle=":",
                    linewidth=2,
                    zorder=2,
                )
                ax.hlines(
                    corrected_ref,
                    xmin=pos - 0.4,
                    xmax=pos + 0.4,
                    color=CORRECTED_COLOR,
                    linestyle=":",
                    linewidth=2,
                    zorder=3,
                )

            if row_unused:
                ax.text(
                    0.5,
                    0.5,
                    "Unused",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=22,
                    fontweight="bold",
                    color="#7a7a7a",
                    alpha=0.75,
                )
            elif any(len(d) > 0 for d in data_to_plot):
                ax.boxplot(data_to_plot, positions=positions, **bp_kw)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.spines[["top", "right"]].set_visible(False)

            if col == 0:
                ax.set_ylabel(
                    fr"$\rho_{{ab}}^2={display_rho_sq(rho_true)}$"
                    "\n"
                    r"Estimated $\sigma_a^2 / W_S$",
                    fontsize=12,
                )

    plt.figure(fig.number)
    plt.tight_layout(rect=[0.02, 0.0, 1.0, 0.98125])
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved sigma2_a / W_s grid to {output}", flush=True)


def plot_sigma2a(results_root: str, rhos: list[float], vs_scales: list[float],
                 sigma2_as: list[float], output: str) -> None:
    fig, axes = plt.subplots(
        len(rhos),
        len(vs_scales),
        figsize=(3.2 * len(vs_scales), 2.88 * len(rhos) + 0.72),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    bp_kw = boxplot_style()

    fig.suptitle(r"Estimates of $\sigma_b^2$", fontsize=16, y=0.99)
    fig.supxlabel(r"True $\sigma_b^2$", fontsize=14, y=0.0125)

    for row, rho_true in enumerate(rhos):
        for col, vs_scale in enumerate(vs_scales):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(fr"True $W_S = {vs_scale}$", fontsize=14)

            data_to_plot = []
            positions = []
            labels = []

            for j, sa_true in enumerate(sigma2_as):
                pos = j + 1
                positions.append(pos)
                labels.append(f"{sa_true:g}")

                df = load_results_frame(results_root, rho_true, vs_scale, sa_true)
                data_to_plot.append(
                    df["sigma2_a_est"].dropna() if "sigma2_a_est" in df.columns else pd.Series(dtype=float)
                )
                ax.hlines(
                    sa_true,
                    xmin=pos - 0.4,
                    xmax=pos + 0.4,
                    color="#e8462a",
                    linestyle="--",
                    linewidth=2,
                    zorder=3,
                )

            if any(len(d) > 0 for d in data_to_plot):
                ax.boxplot(data_to_plot, positions=positions, **bp_kw)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.spines[["top", "right"]].set_visible(False)

            if col == 0:
                ax.set_ylabel(
                    fr"$\rho_{{ab}}^2={display_rho_sq(rho_true)}$"
                    "\n"
                    r"Estimated $\sigma_b^2$",
                    fontsize=12,
                )

    plt.figure(fig.number)
    plt.tight_layout(rect=[0.02, 0.0, 1.0, 0.9875])
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved sigma2_a grid to {output}", flush=True)


def main(out_s2a_over_ws: str, out_s2a: str):
    results_root = "results/blm"
    out_dir = os.path.dirname(out_s2a_over_ws) or "."
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config()
    vs_scales = cfg["vs_scales"]
    sigma2_as = cfg["sigma2_as"]
    rhos = cfg["rhos"]

    plot_sigma2a_over_ws(results_root, cfg, out_s2a_over_ws)
    plot_sigma2a(results_root, rhos, vs_scales, sigma2_as, out_s2a)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/plot_blm_grid.py <out_sigma2_a_over_ws> <out_sigma2_a>")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2])
