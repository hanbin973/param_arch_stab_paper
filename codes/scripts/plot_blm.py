"""
plot_blm.py — Boxplot of BLM parameter estimates across all replicates.

Usage:
    python plot_blm.py <blm_results_dir> <output_png> <vs_scale> <sigma2_a_true> <rho_true>

Arguments:
    sigma2_a_true : Prior variance (= wildcard value, e.g. 0.5)
"""

import sys
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(results_dir: str, output_png: str, vs_scale: float,
         sigma2_a_true: float, rho_true: float) -> None:
    csvs = sorted(glob.glob(f"{results_dir}/rep_*.csv"))
    if not csvs:
        print(f"No CSV files found in {results_dir}", flush=True)
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    print(f"Loaded {len(df)} replicates", flush=True)


    # True parameter values
    Ws_true = vs_scale
    sigma2_a_over_Ws_true = sigma2_a_true / Ws_true
    # sigma2_a_true is already the true wildcard value
    if "sigma2_a_over_Ws_est" not in df.columns:
        if {"sigma2_a_est", "Ws_est"}.issubset(df.columns):
            df["sigma2_a_over_Ws_est"] = df["sigma2_a_est"] / df["Ws_est"]
        else:
            raise ValueError(
                "Input CSVs must contain 'sigma2_a_over_Ws_est' "
                "or both 'sigma2_a_est' and 'Ws_est'."
            )
    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle(
        fr"BLM Estimates  ($\rho_{{ab}}={rho_true}$, VS_SCALE={vs_scale}, σ²_a={sigma2_a_true}, "
        fr"true ref $=\sigma_a^2/W_s$, n={len(df)} reps)",
        fontsize=12, fontweight="bold",
    )

    bp_kw = dict(
        patch_artist=True,
        widths=0.4,
        boxprops=dict(facecolor="#4c9be8", color="#1a3a5c", linewidth=1.5),
        medianprops=dict(color="#e8462a", linewidth=2.5),
        whiskerprops=dict(color="#1a3a5c", linewidth=1.5),
        capprops=dict(color="#1a3a5c", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="#4c9be8",
                        markeredgecolor="#1a3a5c", markersize=5, alpha=0.7),
    )

    for ax, col, true_val, label in [
        (axes[0], "sigma2_a_over_Ws_est", sigma2_a_over_Ws_true, r"$\widehat{\sigma_a^2/W_s}$" "\n" r"(true ref: $\sigma_a^2/W_s$)"),
        (axes[1], "sigma2_a_est", sigma2_a_true, r"$\hat{\sigma}_a^2$"),
    ]:
        ax.boxplot(df[col].dropna(), **bp_kw)
        ax.axhline(true_val, color="#e8462a", linestyle="--", linewidth=1.8,
                   label=f"True = {true_val}", zorder=3)

        ax.set_xticks([])
        ax.set_ylabel(label, fontsize=13)
        ax.set_title(label, fontsize=13)
        ax.legend(fontsize=10, framealpha=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate median
        med = df[col].median()
        ax.text(1, med, f"  median = {med:.3g}",
                va="center", ha="left", fontsize=9, color="#333333")

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_png}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python plot_blm.py <blm_results_dir> <output_png> <vs_scale> <sigma2_a_true> <rho_true>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
