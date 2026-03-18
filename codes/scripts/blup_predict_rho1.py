"""
blup_predict_rho1.py — Train/validate BLUP prediction on one replicate with
the stabilizing model fixed at rho=1 for robustness checks.

Usage:
    python blup_predict_rho1.py <vcf_file> <pheno_file> <rep_id> <sigma2_a_true> <vs_scale> <rho_true> <output_csv>
"""

import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blup_predict import (
    SIGMA2_E_INIT,
    SIGMA2_E_TRUE,
    alpha_tag,
    blup_predict,
    load_alpha_grid,
    load_phenotype_raw,
    optimize_alpha_params,
    optimize_stabilizing_params,
    prediction_metrics,
    split_train_test,
)
from fit_blm import (
    compute_allele_freqs,
    extract_genotype_matrix,
    snp_variances,
)

FIT_RHO = 1.0


def run(vcf_file: str, pheno_file: str, rep_id: int,
        sigma2_a_true: float, vs_scale: float, rho_true: float,
        output_csv: str) -> None:
    print(f"[rep {rep_id}] Loading genotype/phenotype for BLUP with fixed rho=1", flush=True)
    G = extract_genotype_matrix(vcf_file)
    y = load_phenotype_raw(pheno_file, sigma2_e=SIGMA2_E_TRUE, seed=rep_id)

    if y.shape[0] != G.shape[0]:
        raise ValueError(
            f"Phenotype length {y.shape[0]} does not match number of individuals {G.shape[0]}"
        )

    train_idx, test_idx = split_train_test(G.shape[0], seed=rep_id)
    G_train = G[train_idx, :]
    G_test = G[test_idx, :]
    y_train = y[train_idx]
    y_test = y[test_idx]

    mu_train = float(np.mean(y_train))
    y_train_c = y_train - mu_train
    y_test_c = y_test - mu_train

    p_train = compute_allele_freqs(G_train)
    het_safe = np.maximum(p_train * (1.0 - p_train), 1e-6)
    alpha_models = load_alpha_grid()
    Ws_true = vs_scale / 2.0
    ratio_true = sigma2_a_true / Ws_true

    print(
        f"[rep {rep_id}] Fitting stabilizing-prior parameters on n_train={len(train_idx)} with fixed rho={FIT_RHO:g}",
        flush=True,
    )
    ratio_est, s2a_est, sigma2e_stab, conv_stab, nll_stab = optimize_stabilizing_params(
        y=y_train_c,
        G=G_train,
        p_hat=p_train,
        sigma2_a_true=sigma2_a_true,
        vs_scale=vs_scale,
        rho_true=FIT_RHO,
    )
    Ws_est = s2a_est / ratio_est
    sv_stab, _, _ = snp_variances(p_train, Ws_est, s2a_est, rho=FIT_RHO)

    pred_stab = blup_predict(G_train, y_train_c, G_test, sv_stab, sigma2e_stab)
    corr_stab, r2_stab, mse_stab = prediction_metrics(y_test_c, pred_stab)

    alpha_results = []
    for alpha_model in alpha_models:
        print(
            f"[rep {rep_id}] Fitting alpha={alpha_model:g} BLUP sigma2_a,sigma2_e on n_train={len(train_idx)}",
            flush=True,
        )
        s2a_alpha_est, sigma2e_alpha, conv_alpha, nll_alpha = optimize_alpha_params(
            y=y_train_c,
            G=G_train,
            p_hat=p_train,
            sigma2_a_init=sigma2_a_true,
            alpha_model=alpha_model,
        )
        sv_alpha = s2a_alpha_est / (het_safe ** alpha_model)
        pred_alpha = blup_predict(G_train, y_train_c, G_test, sv_alpha, sigma2e_alpha)
        corr_alpha, r2_alpha, mse_alpha = prediction_metrics(y_test_c, pred_alpha)
        alpha_results.append({
            "alpha": alpha_model,
            "tag": alpha_tag(alpha_model),
            "sigma2_a_est": s2a_alpha_est,
            "sigma2_e_est": sigma2e_alpha,
            "converged": conv_alpha,
            "neg_loglik": nll_alpha,
            "corr": corr_alpha,
            "r2": r2_alpha,
            "mse": mse_alpha,
        })

    print(
        f"[rep {rep_id}] fixed-rho1 stabilized(s2e={sigma2e_stab:.4f}, r={corr_stab:.4f}, r2={r2_stab:.4f})  "
        + "  ".join(
            f"alpha={result['alpha']:g}(s2e={result['sigma2_e_est']:.4f}, r={result['corr']:.4f}, r2={result['r2']:.4f})"
            for result in alpha_results
        ),
        flush=True,
    )

    fieldnames = [
        "rep_id", "n_train", "n_test", "sigma2_e", "sigma2_e_true", "sigma2_e_init",
        "rho_true", "rho_fit",
        "sigma2_a_true", "Ws_true", "sigma2_a_over_Ws_true",
        "sigma2_a_over_Ws_est", "sigma2_a_est", "Ws_implied",
        "sigma2_e_stabilized_est", "stabilized_converged", "stabilized_neg_loglik",
        "corr_stabilized", "r2_stabilized", "mse_stabilized",
    ]
    row = {
        "rep_id": rep_id,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "sigma2_e": sigma2e_stab,
        "sigma2_e_true": SIGMA2_E_TRUE,
        "sigma2_e_init": SIGMA2_E_INIT,
        "rho_true": rho_true,
        "rho_fit": FIT_RHO,
        "sigma2_a_true": sigma2_a_true,
        "Ws_true": Ws_true,
        "sigma2_a_over_Ws_true": ratio_true,
        "sigma2_a_over_Ws_est": ratio_est,
        "sigma2_a_est": s2a_est,
        "Ws_implied": Ws_est,
        "sigma2_e_stabilized_est": sigma2e_stab,
        "stabilized_converged": int(conv_stab),
        "stabilized_neg_loglik": nll_stab,
        "corr_stabilized": corr_stab,
        "r2_stabilized": r2_stab,
        "mse_stabilized": mse_stab,
    }

    for result in alpha_results:
        tag = result["tag"]
        fieldnames.extend([
            f"sigma2_a_alpha_{tag}_est",
            f"sigma2_e_alpha_{tag}_est",
            f"alpha_{tag}_converged",
            f"alpha_{tag}_neg_loglik",
            f"corr_alpha_{tag}",
            f"r2_alpha_{tag}",
            f"mse_alpha_{tag}",
        ])
        row.update({
            f"sigma2_a_alpha_{tag}_est": result["sigma2_a_est"],
            f"sigma2_e_alpha_{tag}_est": result["sigma2_e_est"],
            f"alpha_{tag}_converged": int(result["converged"]),
            f"alpha_{tag}_neg_loglik": result["neg_loglik"],
            f"corr_alpha_{tag}": result["corr"],
            f"r2_alpha_{tag}": result["r2"],
            f"mse_alpha_{tag}": result["mse"],
        })

    for result in alpha_results:
        if np.isclose(result["alpha"], 1.0):
            fieldnames.extend([
                "sigma2_e_vanilla_est",
                "sigma2_a_vanilla_est", "vanilla_converged", "vanilla_neg_loglik",
                "corr_vanilla", "r2_vanilla", "mse_vanilla",
            ])
            row.update({
                "sigma2_e_vanilla_est": result["sigma2_e_est"],
                "sigma2_a_vanilla_est": result["sigma2_a_est"],
                "vanilla_converged": int(result["converged"]),
                "vanilla_neg_loglik": result["neg_loglik"],
                "corr_vanilla": result["corr"],
                "r2_vanilla": result["r2"],
                "mse_vanilla": result["mse"],
            })
            break

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"[rep {rep_id}] BLUP fixed-rho1 comparison written to {output_csv}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python blup_predict_rho1.py <vcf_file> <pheno_file> <rep_id> <sigma2_a_true> <vs_scale> <rho_true> <output_csv>"
        )
        sys.exit(1)

    run(
        vcf_file=sys.argv[1],
        pheno_file=sys.argv[2],
        rep_id=int(sys.argv[3]),
        sigma2_a_true=float(sys.argv[4]),
        vs_scale=float(sys.argv[5]),
        rho_true=float(sys.argv[6]),
        output_csv=sys.argv[7],
    )
