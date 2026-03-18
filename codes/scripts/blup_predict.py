"""
blup_predict.py — Train/validate BLUP prediction on one replicate.

Workflow
--------
1. Load genotype dosages (N x M) and phenotype vector (N).
2. Split individuals into train/validation halves (50/50) using rep_id as seed.
3. Fit prior models on the train set:
   a) Stabilizing prior (fit_blm logic): estimate sigma2_a, sigma2_a / W_s, and sigma2_e.
   b) Alpha-model baselines from config.yaml with sv_j = sigma2_a / het_j**alpha,
      jointly estimating sigma2_a and sigma2_e.
4. Build BLUP predictor from train set and evaluate prediction on validation set.

Usage:
    python blup_predict.py <vcf_file> <pheno_file> <rep_id> <sigma2_a_true> <vs_scale> <rho_true> <output_csv>
"""

import csv
import sys
import numpy as np
import yaml
from scipy.optimize import minimize, minimize_scalar

# Ensure sibling imports work whether launched as script or module.
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fit_blm import (
    extract_genotype_matrix,
    compute_allele_freqs,
    snp_variances,
    snp_variance_derivatives,
    factor_v,
    solve_v,
    projected_grad_maxnorm,
    solve_reduced_ai_step,
    raw_step_metrics,
    LOG_PARAM_MIN,
    LOG_PARAM_MAX,
)

SIGMA2_E_TRUE = 100.0
SIGMA2_E_INIT = 100.0
SIGMA2_E_MIN = 1e-4
SIGMA2_E_MAX = 1e5
LOG_SIGMA2_E_MIN = float(np.log(SIGMA2_E_MIN))
LOG_SIGMA2_E_MAX = float(np.log(SIGMA2_E_MAX))
LBFGSB_MAXITER = 200
LBFGSB_MAXLS = 50
LBFGSB_GTOL = 1e-5
LBFGSB_FTOL = 1e-9


def load_alpha_grid() -> list[float]:
    """Read the alpha-model grid from config.yaml."""
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return [float(a) for a in cfg["simulation_parameters"]["ALPHA_GRID"]]


def alpha_tag(alpha_model: float) -> str:
    """Return a filesystem/CSV-safe identifier for alpha."""
    return f"{alpha_model:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def load_phenotype_raw(pheno_file: str, sigma2_e: float, seed: int | None = None) -> np.ndarray:
    """Load phenotype values and add Gaussian noise with variance sigma2_e."""
    y = np.loadtxt(pheno_file, dtype=np.float64)
    if y.ndim != 1:
        y = np.ravel(y)
    if sigma2_e > 0.0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, np.sqrt(sigma2_e), size=y.shape[0])
    return y


def split_train_test(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test index arrays with an exact 50/50 split when n is even."""
    if n < 2:
        raise ValueError(f"Need at least 2 individuals; got n={n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = n // 2
    return perm[:n_train], perm[n_train:]


def reml_stats_from_sv(y: np.ndarray, G: np.ndarray, sv: np.ndarray,
                       sigma2_e: float, need_trace: bool = False):
    """REML NLL with helper matrices used for gradient/AI updates."""
    n = y.shape[0]
    ones = np.ones(n, dtype=np.float64)

    backend, factor_data, log_det = factor_v(G, sv, sigma2_e)
    if backend is None:
        return np.inf, None

    alpha = solve_v(backend, factor_data, y)
    B = solve_v(backend, factor_data, G)
    v1 = solve_v(backend, factor_data, ones)

    c_r_safe = max(float(ones @ v1), 1e-300)
    scale = 1.0 / c_r_safe

    Py = alpha - v1 * ((ones @ alpha) * scale)
    PG = B - np.outer(v1, (ones @ B) * scale)
    trace_p = None
    if need_trace:
        if backend == "dense":
            trace_inv_v = float(np.trace(solve_v(backend, factor_data, np.eye(n))))
        else:
            _, sfac, slow, sigma2_e_used = factor_data
            s_inv = solve_v("dense", (sfac, slow), np.eye(G.shape[1]))
            trace_inv_v = (n - G.shape[1]) / sigma2_e_used + float(np.trace(s_inv)) / sigma2_e_used
        trace_p = trace_inv_v - float(v1 @ v1) * scale

    quad_reml = y @ Py
    nreml = 0.5 * (log_det + np.log(c_r_safe) + quad_reml + (n - 1) * np.log(2.0 * np.pi))

    return nreml, (backend, factor_data, v1, scale, Py, PG, trace_p)


def optimize_sigma2e_given_sv(y: np.ndarray, G: np.ndarray, sv: np.ndarray,
                              sigma2_e_init: float) -> tuple[float, bool, float]:
    """Optimize sigma2_e with other prior-variance terms held fixed."""
    def objective(log_sigma2_e: float) -> float:
        sigma2_e = float(np.exp(np.clip(log_sigma2_e, -25.0, 25.0)))
        nreml, _ = reml_stats_from_sv(y, G, sv, sigma2_e)
        return nreml

    init = float(np.clip(sigma2_e_init, SIGMA2_E_MIN, SIGMA2_E_MAX))
    bounds = (np.log(SIGMA2_E_MIN), np.log(SIGMA2_E_MAX))
    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-3})
    sigma2_e_est = float(np.exp(result.x)) if result.success else init
    nreml = float(result.fun) if result.success else float(objective(np.log(init)))
    return sigma2_e_est, bool(result.success), nreml


def optimize_stabilizing_given_sigma2e(y: np.ndarray, G: np.ndarray, p_hat: np.ndarray,
                                       sigma2_a_true: float, vs_scale: float,
                                       sigma2_e: float, rho_true: float,
                                       ratio_init: float | None = None,
                                       sigma2_a_init: float | None = None):
    """Compatibility helper: AI-REML fit at fixed sigma2_e for benchmarking/comparison."""
    Ws_true = vs_scale / 2.0
    s2a_over_Ws_true = sigma2_a_true / Ws_true
    ratio_seed = float(ratio_init) if ratio_init is not None else s2a_over_Ws_true
    sigma2_a_seed = float(sigma2_a_init) if sigma2_a_init is not None else sigma2_a_true
    x_curr = np.array([np.log(ratio_seed), np.log(sigma2_a_seed)], dtype=np.float64)

    max_iter = 50
    ftol = 1e-8
    gtol = 1e-5
    xtol_abs = 1e-3
    xtol_rel = 1e-3
    converged = False

    def nreml_grad_ai(params):
        log_ratio, log_s2a = params
        log_ratio = np.clip(log_ratio, -25.0, 25.0)
        log_s2a = np.clip(log_s2a, -25.0, 25.0)

        s2a_over_Ws = np.exp(log_ratio)
        sigma2_a = np.exp(log_s2a)
        Ws = sigma2_a / s2a_over_Ws
        sv, d_logWs_ind, d_logs2a_ind = snp_variance_derivatives(
            p_hat, Ws, sigma2_a, rho=rho_true
        )

        nreml, aux = reml_stats_from_sv(y, G, sv, sigma2_e)
        if aux is None:
            return np.inf, np.zeros(2), np.eye(2)
        backend, factor_data, v1, scale, Py, PG, _ = aux

        d_ratio = -d_logWs_ind
        d_s2a = d_logWs_ind + d_logs2a_ind

        Gt_Py = G.T @ Py
        diag_GPG = (G * PG).sum(axis=0)
        residual = diag_GPG - Gt_Py ** 2

        grad = 0.5 * np.array([
            np.dot(d_ratio, residual),
            np.dot(d_s2a, residual),
        ])

        w_ratio = d_ratio * Gt_Py
        w_s2a = d_s2a * Gt_Py

        v_ratio = G @ w_ratio
        v_s2a = G @ w_s2a

        v1_ratio = solve_v(backend, factor_data, v_ratio)
        v1_s2a = solve_v(backend, factor_data, v_s2a)

        P_v_ratio = v1_ratio - v1 * ((np.ones(y.shape[0]) @ v1_ratio) * scale)
        P_v_s2a = v1_s2a - v1 * ((np.ones(y.shape[0]) @ v1_s2a) * scale)

        ai_11 = 0.5 * np.dot(v_ratio, P_v_ratio)
        ai_12 = 0.5 * np.dot(v_ratio, P_v_s2a)
        ai_22 = 0.5 * np.dot(v_s2a, P_v_s2a)
        AI = np.array([[ai_11, ai_12], [ai_12, ai_22]])

        if not np.isfinite(nreml) or not np.all(np.isfinite(grad)) or not np.all(np.isfinite(AI)):
            return np.inf, np.zeros(2), np.eye(2)
        return nreml, grad, AI

    def nreml_only(params):
        log_ratio, log_s2a = np.clip(params, -25.0, 25.0)
        s2a_over_Ws = np.exp(log_ratio)
        sigma2_a = np.exp(log_s2a)
        Ws = sigma2_a / s2a_over_Ws
        sv, _, _ = snp_variances(p_hat, Ws, sigma2_a, rho=rho_true)
        nreml, _ = reml_stats_from_sv(y, G, sv, sigma2_e)
        return nreml

    nreml, grad, AI = nreml_grad_ai(x_curr)

    for _ in range(max_iter):
        if projected_grad_maxnorm(x_curr, grad, LOG_PARAM_MIN, LOG_PARAM_MAX) < gtol:
            converged = True
            break

        step, free = solve_reduced_ai_step(
            x_curr, grad, AI, LOG_PARAM_MIN, LOG_PARAM_MAX
        )
        if not np.any(free):
            converged = True
            break

        alpha = 1.0
        moved = False
        for _ in range(10):
            proposal = x_curr - alpha * step
            x_next = np.clip(proposal, LOG_PARAM_MIN, LOG_PARAM_MAX)
            nreml_next = nreml_only(x_next)
            if np.isfinite(nreml_next) and (nreml_next < nreml or abs(nreml - nreml_next) < ftol):
                max_param_change, max_rel_param_change = raw_step_metrics(x_curr, x_next)
                if abs(nreml - nreml_next) < ftol:
                    converged = True
                elif max_param_change < xtol_abs or max_rel_param_change < xtol_rel:
                    converged = True
                x_curr = x_next
                moved = True
                break
            alpha *= 0.5

        if not moved:
            if projected_grad_maxnorm(x_curr, grad, LOG_PARAM_MIN, LOG_PARAM_MAX) < gtol:
                converged = True
            break
        if converged:
            break
        nreml, grad, AI = nreml_grad_ai(x_curr)

    ratio_est = np.exp(x_curr[0])
    s2a_est = np.exp(x_curr[1])
    return ratio_est, s2a_est, converged, nreml


def optimize_stabilizing_params(y: np.ndarray, G: np.ndarray, p_hat: np.ndarray,
                                sigma2_a_true: float, vs_scale: float, rho_true: float):
    """Joint L-BFGS-B REML fit for stabilizing-prior parameters and sigma2_e."""
    Ws_true = vs_scale / 2.0
    ratio_init = float(sigma2_a_true / Ws_true)
    sigma2_e_init = max(float(np.var(y)) * 0.5, SIGMA2_E_INIT)
    sigma2_e_init = float(np.clip(sigma2_e_init, SIGMA2_E_MIN, SIGMA2_E_MAX))
    lower_bounds = np.array([LOG_PARAM_MIN, LOG_PARAM_MIN, LOG_SIGMA2_E_MIN], dtype=np.float64)
    upper_bounds = np.array([LOG_PARAM_MAX, LOG_PARAM_MAX, LOG_SIGMA2_E_MAX], dtype=np.float64)
    x_curr = np.clip(np.array(
        [np.log(ratio_init), np.log(sigma2_a_true), np.log(sigma2_e_init)],
        dtype=np.float64,
    ), lower_bounds, upper_bounds)

    bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))

    def nreml_and_grad(params):
        log_ratio = float(np.clip(params[0], LOG_PARAM_MIN, LOG_PARAM_MAX))
        log_s2a = float(np.clip(params[1], LOG_PARAM_MIN, LOG_PARAM_MAX))
        log_sigma2_e = float(np.clip(params[2], LOG_SIGMA2_E_MIN, LOG_SIGMA2_E_MAX))
        s2a_over_Ws = np.exp(log_ratio)
        sigma2_a = np.exp(log_s2a)
        sigma2_e = np.exp(log_sigma2_e)
        Ws = sigma2_a / s2a_over_Ws

        sv, d_logWs_ind, d_logs2a_ind = snp_variance_derivatives(
            p_hat, Ws, sigma2_a, rho=rho_true
        )
        nreml, aux = reml_stats_from_sv(y, G, sv, sigma2_e, need_trace=True)
        if aux is None:
            return np.inf, np.zeros(3), np.eye(3)
        backend, factor_data, v1, scale, Py, PG, trace_p = aux

        d_ratio = -d_logWs_ind
        d_s2a = d_logWs_ind + d_logs2a_ind

        Gt_Py = G.T @ Py
        diag_GPG = (G * PG).sum(axis=0)
        residual = diag_GPG - Gt_Py ** 2

        grad = np.empty(3, dtype=np.float64)
        grad[0] = 0.5 * float(np.dot(d_ratio, residual))
        grad[1] = 0.5 * float(np.dot(d_s2a, residual))
        grad[2] = 0.5 * sigma2_e * (trace_p - float(Py @ Py))

        if not np.isfinite(nreml) or not np.all(np.isfinite(grad)):
            return np.inf, np.zeros(3)
        return float(nreml), grad

    result = minimize(
        nreml_and_grad,
        x_curr,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": LBFGSB_MAXITER,
            "maxls": LBFGSB_MAXLS,
            "gtol": LBFGSB_GTOL,
            "ftol": LBFGSB_FTOL,
        },
    )
    x_final = np.clip(result.x, lower_bounds, upper_bounds)
    nreml, _ = nreml_and_grad(x_final)
    ratio_est = float(np.exp(x_final[0]))
    s2a_est = float(np.exp(x_final[1]))
    sigma2_e_est = float(np.exp(x_final[2]))
    converged = bool(result.success)
    return ratio_est, s2a_est, sigma2_e_est, converged, nreml


def optimize_alpha_given_sigma2e(y: np.ndarray, G: np.ndarray, p_hat: np.ndarray,
                                 sigma2_a_init: float, alpha_model: float,
                                 sigma2_e: float):
    """AI-REML optimizer for alpha-BLUP with sv_j = sigma2_a / het_j**alpha at fixed sigma2_e."""
    het = p_hat * (1.0 - p_hat)
    het_safe = np.maximum(het, 1e-6)
    alpha_weight = het_safe ** alpha_model
    x_curr = np.array([np.log(sigma2_a_init)], dtype=np.float64)

    max_iter = 50
    ftol = 1e-8
    gtol = 1e-5
    xtol_abs = 1e-3
    xtol_rel = 1e-3
    converged = False
    boundary_stall_count = 0
    boundary_stall_limit = 2

    def nreml_grad_ai_1d(x):
        log_s2a = float(np.clip(x[0], -25.0, 25.0))
        sigma2_a = np.exp(log_s2a)
        sv = sigma2_a / alpha_weight

        nreml, aux = reml_stats_from_sv(y, G, sv, sigma2_e)
        if aux is None:
            return np.inf, np.zeros(1), np.array([[1.0]])
        backend, factor_data, v1, scale, Py, PG, _ = aux

        d_s2a = sv  # d sv / d log(sigma2_a)

        Gt_Py = G.T @ Py
        diag_GPG = (G * PG).sum(axis=0)
        residual = diag_GPG - Gt_Py ** 2
        grad = 0.5 * np.array([np.dot(d_s2a, residual)])

        w = d_s2a * Gt_Py
        v = G @ w
        v1_v = solve_v(backend, factor_data, v)
        P_v = v1_v - v1 * ((np.ones(y.shape[0]) @ v1_v) * scale)
        ai = 0.5 * np.dot(v, P_v)
        AI = np.array([[ai]])

        if not np.isfinite(nreml) or not np.all(np.isfinite(grad)) or not np.all(np.isfinite(AI)):
            return np.inf, np.zeros(1), np.array([[1.0]])
        return nreml, grad, AI

    def nreml_only_1d(x):
        log_s2a = float(np.clip(x[0], -25.0, 25.0))
        sigma2_a = np.exp(log_s2a)
        sv = sigma2_a / alpha_weight
        nreml, _ = reml_stats_from_sv(y, G, sv, sigma2_e)
        return nreml

    nreml, grad, AI = nreml_grad_ai_1d(x_curr)

    for _ in range(max_iter):
        if projected_grad_maxnorm(x_curr, grad, LOG_PARAM_MIN, LOG_PARAM_MAX) < gtol:
            converged = True
            break

        denom = AI[0, 0]
        if not np.isfinite(denom) or abs(denom) < 1e-10:
            denom = 1e-10
        step = grad[0] / denom

        alpha = 1.0
        moved = False
        attempted_boundary_stall = False
        for _ in range(10):
            proposal = x_curr[0] - alpha * step
            x_next = np.array([np.clip(proposal, LOG_PARAM_MIN, LOG_PARAM_MAX)])
            was_clipped = not np.isclose(x_next[0], proposal)
            nreml_next = nreml_only_1d(x_next)
            if np.isfinite(nreml_next) and (nreml_next < nreml or abs(nreml - nreml_next) < ftol):
                max_param_change, max_rel_param_change = raw_step_metrics(x_curr, x_next)
                if abs(nreml - nreml_next) < ftol:
                    converged = True
                elif max_param_change < xtol_abs or max_rel_param_change < xtol_rel:
                    converged = True
                else:
                    boundary_stall_count = 0
                x_curr = x_next
                moved = True
                break
            if was_clipped:
                max_param_change, max_rel_param_change = raw_step_metrics(x_curr, x_next)
                if max_param_change < xtol_abs or max_rel_param_change < xtol_rel:
                    attempted_boundary_stall = True
            alpha *= 0.5

        if not moved:
            if attempted_boundary_stall:
                boundary_stall_count += 1
                if boundary_stall_count >= boundary_stall_limit:
                    converged = True
            break
        if converged:
            break
        nreml, grad, AI = nreml_grad_ai_1d(x_curr)

    s2a_est = np.exp(x_curr[0])
    return s2a_est, converged, nreml


def alpha_reml_rotated_stats(y_rot: np.ndarray, x_rot: np.ndarray, kernel_evals: np.ndarray,
                             sigma2_a: float, sigma2_e: float,
                             need_trace: bool = False):
    """REML quantities for V = sigma2_a K + sigma2_e I after diagonalizing K."""
    lam = sigma2_a * kernel_evals + sigma2_e
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        return np.inf, None

    inv_lam = 1.0 / lam
    alpha_rot = y_rot * inv_lam
    v1_rot = x_rot * inv_lam
    c_r_safe = max(float(np.dot(x_rot, v1_rot)), 1e-300)
    scale = 1.0 / c_r_safe

    proj_y = float(np.dot(x_rot, alpha_rot))
    Py_rot = alpha_rot - v1_rot * (proj_y * scale)

    quad_reml = float(np.dot(y_rot, Py_rot))
    nreml = 0.5 * (
        float(np.sum(np.log(lam))) + np.log(c_r_safe) + quad_reml
        + (y_rot.shape[0] - 1) * np.log(2.0 * np.pi)
    )

    trace_p = None
    if need_trace:
        trace_inv_v = float(np.sum(inv_lam))
        trace_p = trace_inv_v - float(np.dot(v1_rot, v1_rot)) * scale

    return nreml, (lam, inv_lam, v1_rot, scale, Py_rot, trace_p)


def rotated_reml_trace_term(x_rot: np.ndarray, inv_lam: np.ndarray,
                            scale: float, d_diag: np.ndarray) -> float:
    """Return tr(P dV) when V and dV are diagonal in the rotated basis."""
    return float(
        np.sum(d_diag * inv_lam)
        - np.sum((x_rot ** 2) * d_diag * (inv_lam ** 2)) * scale
    )


def optimize_alpha_params(y: np.ndarray, G: np.ndarray, p_hat: np.ndarray,
                          sigma2_a_init: float, alpha_model: float):
    """Joint L-BFGS-B REML fit for alpha-model sigma2_a and sigma2_e."""
    sigma2_e_init = max(float(np.var(y)) * 0.5, SIGMA2_E_INIT)
    sigma2_e_init = float(np.clip(sigma2_e_init, SIGMA2_E_MIN, SIGMA2_E_MAX))
    lower_bounds = np.array([LOG_PARAM_MIN, LOG_SIGMA2_E_MIN], dtype=np.float64)
    upper_bounds = np.array([LOG_PARAM_MAX, LOG_SIGMA2_E_MAX], dtype=np.float64)
    x_curr = np.clip(
        np.array([np.log(sigma2_a_init), np.log(sigma2_e_init)], dtype=np.float64),
        lower_bounds,
        upper_bounds,
    )
    het_safe = np.maximum(p_hat * (1.0 - p_hat), 1e-6)
    alpha_weight = het_safe ** alpha_model
    kernel = (G / alpha_weight[np.newaxis, :]) @ G.T
    kernel = 0.5 * (kernel + kernel.T)
    kernel_evals, kernel_evecs = np.linalg.eigh(kernel)
    y_rot = kernel_evecs.T @ y
    x_rot = kernel_evecs.T @ np.ones(y.shape[0], dtype=np.float64)
    bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))

    def nreml_and_grad(params):
        log_s2a = float(np.clip(params[0], LOG_PARAM_MIN, LOG_PARAM_MAX))
        log_sigma2_e = float(np.clip(params[1], LOG_SIGMA2_E_MIN, LOG_SIGMA2_E_MAX))
        sigma2_a = np.exp(log_s2a)
        sigma2_e = np.exp(log_sigma2_e)

        nreml, aux = alpha_reml_rotated_stats(
            y_rot, x_rot, kernel_evals, sigma2_a, sigma2_e, need_trace=True
        )
        if aux is None:
            return np.inf, np.zeros(2)
        lam, inv_lam, _, scale, Py_rot, trace_p = aux

        d_sigma2_a = sigma2_a * kernel_evals
        d_sigma2_e = np.full_like(lam, sigma2_e)

        grad = np.empty(2, dtype=np.float64)
        grad[0] = 0.5 * (
            rotated_reml_trace_term(x_rot, inv_lam, scale, d_sigma2_a)
            - float(np.dot(d_sigma2_a, Py_rot ** 2))
        )
        grad[1] = 0.5 * (
            trace_p * sigma2_e
            - float(np.dot(d_sigma2_e, Py_rot ** 2))
        )
        if not np.isfinite(nreml) or not np.all(np.isfinite(grad)):
            return np.inf, np.zeros(2)
        return float(nreml), grad

    result = minimize(
        nreml_and_grad,
        x_curr,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": LBFGSB_MAXITER,
            "maxls": LBFGSB_MAXLS,
            "gtol": LBFGSB_GTOL,
            "ftol": LBFGSB_FTOL,
        },
    )
    x_final = np.clip(result.x, lower_bounds, upper_bounds)
    nreml, _ = nreml_and_grad(x_final)
    s2a_est = float(np.exp(x_final[0]))
    sigma2_e_est = float(np.exp(x_final[1]))
    converged = bool(result.success)
    return s2a_est, sigma2_e_est, converged, nreml


def blup_predict(G_train: np.ndarray, y_train: np.ndarray, G_test: np.ndarray,
                 sv: np.ndarray, sigma2_e: float) -> np.ndarray:
    """Posterior mean BLUP predictions for test individuals."""
    backend, factor_data, _ = factor_v(G_train, sv, sigma2_e)
    if backend is None:
        raise np.linalg.LinAlgError("Failed to factor BLUP covariance matrix.")
    alpha = solve_v(backend, factor_data, y_train)
    b_hat = sv * (G_train.T @ alpha)
    return G_test @ b_hat


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Return (corr, r2, mse)."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    y_std = float(np.std(y_true))
    p_std = float(np.std(y_pred))
    if y_std == 0.0 or p_std == 0.0:
        return np.nan, np.nan, mse
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return corr, corr * corr, mse


def run(vcf_file: str, pheno_file: str, rep_id: int,
        sigma2_a_true: float, vs_scale: float, rho_true: float,
        output_csv: str) -> None:
    print(f"[rep {rep_id}] Loading genotype/phenotype for BLUP", flush=True)
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

    # Intercept from train set only.
    mu_train = float(np.mean(y_train))
    y_train_c = y_train - mu_train
    y_test_c = y_test - mu_train

    p_train = compute_allele_freqs(G_train)
    het_safe = np.maximum(p_train * (1.0 - p_train), 1e-6)
    alpha_models = load_alpha_grid()
    Ws_true = vs_scale / 2.0
    ratio_true = sigma2_a_true / Ws_true

    print(f"[rep {rep_id}] Fitting stabilizing-prior parameters on n_train={len(train_idx)}", flush=True)
    ratio_est, s2a_est, sigma2e_stab, conv_stab, nll_stab = optimize_stabilizing_params(
        y=y_train_c, G=G_train, p_hat=p_train, sigma2_a_true=sigma2_a_true,
        vs_scale=vs_scale, rho_true=rho_true
    )
    Ws_est = s2a_est / ratio_est
    sv_stab, _, _ = snp_variances(p_train, Ws_est, s2a_est, rho=rho_true)

    pred_stab = blup_predict(G_train, y_train_c, G_test, sv_stab, sigma2e_stab)
    corr_stab, r2_stab, mse_stab = prediction_metrics(y_test_c, pred_stab)

    alpha_results = []
    for alpha_model in alpha_models:
        print(
            f"[rep {rep_id}] Fitting alpha={alpha_model:g} BLUP sigma2_a,sigma2_e on n_train={len(train_idx)}",
            flush=True,
        )
        s2a_alpha_est, sigma2e_alpha, conv_alpha, nll_alpha = optimize_alpha_params(
            y=y_train_c, G=G_train, p_hat=p_train,
            sigma2_a_init=sigma2_a_true, alpha_model=alpha_model,
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
        f"[rep {rep_id}] stabilized(s2e={sigma2e_stab:.4f}, r={corr_stab:.4f}, r2={r2_stab:.4f})  "
        + "  ".join(
            f"alpha={result['alpha']:g}(s2e={result['sigma2_e_est']:.4f}, r={result['corr']:.4f}, r2={result['r2']:.4f})"
            for result in alpha_results
        ),
        flush=True,
    )

    fieldnames = [
        "rep_id", "n_train", "n_test", "sigma2_e", "sigma2_e_true", "sigma2_e_init",
        "rho_true",
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

    # Backward-compatible aliases for the historical alpha=1 baseline.
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

    print(f"[rep {rep_id}] BLUP comparison written to {output_csv}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python blup_predict.py <vcf_file> <pheno_file> <rep_id> <sigma2_a_true> <vs_scale> <rho_true> <output_csv>"
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
