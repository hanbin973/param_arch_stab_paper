"""
fit_blm.py — Fit a Bayesian Linear Model on genotype data from a SLiM tree sequence.

Model:
    y = G b + e
    b_j ~ N(0, sigma_j^2)
    e   ~ N(0, sigma_e^2 * I)       [sigma_e^2 is fixed; see SIGMA2_E below]

Per-SNP prior variance (with rho fixed from input):
    sigma_j^2 = sigma2_a * (1 - rho^2 * x_j / (1 + x_j))
    x_j       = 2 * sigma2_a * p_hat_j*(1-p_hat_j) / Ws
    where p_hat_j = mean(G[:,j]) / 2

Marginal likelihood (b integrated out):
    y | Ws, sigma2_a, rho ~ N(0, V)
    V = G diag(sigma^2) G^T + sigma_e^2 * I

We estimate sigma2_a and (sigma2_a / Ws) at fixed rho by maximising
log p(y | Ws, sigma2_a, rho), with Ws recovered as:
    Ws = sigma2_a / (sigma2_a / Ws)

Usage:
    python fit_blm.py <vcf_file> <pheno_file> <rep_id> <sigma2_a_true> <vs_scale> <rho_true> <output_csv>

Arguments:
    sigma2_a_true : the prior variance (from the Snakefile wildcard)
    vs_scale      : true ratio of V_S to 2*N (from the Snakefile)

Output CSV columns:
    rep_id, sigma2_e, sigma2_a_true, rho_true, Ws_true, sigma2_a_over_Ws_true,
    sigma2_a_over_Ws_est, sigma2_a_est, neg_loglik, converged
"""

import sys
import csv
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

# ── Fixed hyperparameters ────────────────────────────────────────────────────
# Small fixed noise variance — keeps the N×N covariance matrix well-conditioned
# and makes (Ws, sigma2_a) identifiable.
SIGMA2_E = 10
# ─────────────────────────────────────────────────────────────────────────────
LOG_PARAM_MIN = -15.0
LOG_PARAM_MAX = 5.0


def extract_genotype_matrix(vcf_file: str) -> np.ndarray:
    """Return diploid dosage matrix G of shape (n_individuals, n_variants).

    Reads the VCF file output by SLiM. Each row is a variant, each column
    after 9 is a diploid genotype like '0|0', '1|0', '1|1'.
    """
    dosages = []
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            # Sum the '1' characters to compute diploid dosage (0, 1, or 2)
            row = [g.count('1') for g in parts[9:]]
            dosages.append(row)

    if not dosages:
        print("  Genotype matrix is empty (no variants).", flush=True)
        return np.zeros((0, 0))

    # G shape becomes (N, M)
    G = np.array(dosages, dtype=np.float64).T

    # Drop monomorphic sites (fixed or absent across all individuals)
    n_ind = G.shape[0]
    allele_counts = G.sum(axis=0)
    max_count = 2 * n_ind
    keep = (allele_counts > 0) & (allele_counts < max_count)
    G = G[:, keep]

    print(f"  Genotype matrix: {G.shape[0]} individuals × {G.shape[1]} polymorphic SNPs",
          flush=True)
    return G


def compute_allele_freqs(G: np.ndarray) -> np.ndarray:
    """p_hat_j = mean(G[:,j]) / 2  (estimated from dosage column)."""
    return G.mean(axis=0) / 2.0


def snp_variances(p_hat: np.ndarray, Ws: float,
                  sigma2_a: float, rho: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-SNP prior variance sigma_j^2 = E[beta_j^2 | p_hat_j].

    sigma_j^2 = sigma2_a * (1 - rho^2 * x / (1 + x))
    x         = 2 * sigma2_a * p_hat*(1-p_hat) / Ws

    Returns
    -------
    sv    : (M,) per-SNP prior variances
    a     : (M,) legacy stabilising-selection term  het / Ws
    b     : float legacy prior term  0.5 / sigma2_a

    Numerical notes
    ---------------
    * `a` and `b` are kept for backward compatibility with helper scripts that
      still inspect the historical rho=1 components.
    * The main optimizers use `snp_variance_derivatives(...)` for derivatives.
    """
    het = p_hat * (1.0 - p_hat)
    rho_sq = float(rho) ** 2
    x = (2.0 * sigma2_a / Ws) * het
    frac = np.divide(x, 1.0 + x, out=np.ones_like(x), where=np.isfinite(x))
    sv = sigma2_a * (1.0 - rho_sq * frac)
    sv = np.where(np.isfinite(sv) & (sv >= 0.0), sv, 0.0)

    a = het / Ws
    b = 0.5 / sigma2_a
    return sv, a, b


def snp_variance_derivatives(p_hat: np.ndarray, Ws: float, sigma2_a: float,
                             rho: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sv plus independent derivatives wrt log(Ws) and log(sigma2_a)."""
    sv, _, _ = snp_variances(p_hat, Ws, sigma2_a, rho=rho)
    het = p_hat * (1.0 - p_hat)
    rho_sq = float(rho) ** 2
    x = (2.0 * sigma2_a / Ws) * het
    frac = np.divide(x, 1.0 + x, out=np.ones_like(x), where=np.isfinite(x))
    inv_one_plus_x = np.divide(1.0, 1.0 + x, out=np.zeros_like(x), where=np.isfinite(x))
    d_logWs_ind = sigma2_a * rho_sq * frac * inv_one_plus_x
    d_logs2a_ind = sv - d_logWs_ind
    d_logWs_ind = np.where(np.isfinite(d_logWs_ind) & (d_logWs_ind >= 0.0), d_logWs_ind, 0.0)
    d_logs2a_ind = np.where(
        np.isfinite(d_logs2a_ind) & (d_logs2a_ind >= 0.0), d_logs2a_ind, 0.0
    )
    return sv, d_logWs_ind, d_logs2a_ind


def factor_v(G: np.ndarray, sv: np.ndarray,
             sigma2_e: float) -> tuple[str, tuple, float] | tuple[None, None, None]:
    """Factor V = G diag(sv) G^T + sigma2_e I using an adaptive backend.

    When M < N, Woodbury works with an M×M system instead of forming the full
    N×N covariance. Otherwise, fall back to the dense N×N Cholesky path.
    """
    N, M = G.shape

    if M < N:
        U = G * np.sqrt(sv)[np.newaxis, :]
        S = np.eye(M) + (U.T @ U) / sigma2_e
        try:
            sfac, slow = cho_factor(S, lower=True)
        except np.linalg.LinAlgError:
            return None, None, None
        log_det = N * np.log(sigma2_e) + 2.0 * np.sum(
            np.log(np.maximum(np.diag(sfac), 1e-300))
        )
        return "woodbury", (U, sfac, slow, sigma2_e), log_det

    K = (G * sv[np.newaxis, :]) @ G.T
    V = K + sigma2_e * np.eye(N)
    try:
        cfac, low = cho_factor(V, lower=True)
    except np.linalg.LinAlgError:
        return None, None, None
    log_det = 2.0 * np.sum(np.log(np.maximum(np.diag(cfac), 1e-300)))
    return "dense", (cfac, low), log_det


def solve_v(backend: str, factor_data: tuple, rhs: np.ndarray) -> np.ndarray:
    """Solve V x = rhs for either the dense or Woodbury factorization."""
    if backend == "dense":
        return cho_solve(factor_data, rhs)

    U, sfac, slow, sigma2_e = factor_data
    rhs_2d = rhs if rhs.ndim == 2 else rhs[:, None]
    Ut_rhs = U.T @ rhs_2d
    solved = cho_solve((sfac, slow), Ut_rhs)
    out = rhs_2d / sigma2_e - (U @ solved) / (sigma2_e ** 2)
    return out if rhs.ndim == 2 else out[:, 0]


def projected_grad_maxnorm(x: np.ndarray, grad: np.ndarray,
                           lower: float, upper: float,
                           tol: float = 1e-10) -> float:
    """Infinity norm of the gradient projected onto box constraints."""
    pg = grad.copy()
    at_lower = x <= lower + tol
    at_upper = x >= upper - tol

    # For minimization with lower/upper bounds:
    # * at lower bound, outward gradient points negative
    # * at upper bound, outward gradient points positive
    pg[at_lower & (grad >= 0.0)] = 0.0
    pg[at_upper & (grad <= 0.0)] = 0.0
    return float(np.max(np.abs(pg)))


def active_box_mask(x: np.ndarray, grad: np.ndarray,
                    lower: float, upper: float,
                    tol: float = 1e-10) -> np.ndarray:
    """Return coordinates that should stay fixed at the current box boundary."""
    at_lower = x <= lower + tol
    at_upper = x >= upper - tol
    return (at_lower & (grad >= 0.0)) | (at_upper & (grad <= 0.0))


def solve_reduced_ai_step(x: np.ndarray, grad: np.ndarray, ai: np.ndarray,
                          lower: float, upper: float,
                          ridge: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Solve the AI system only on coordinates not active at a box boundary."""
    active = active_box_mask(x, grad, lower, upper)
    free = ~active
    step = np.zeros_like(grad)
    if not np.any(free):
        return step, free

    ai_free = 0.5 * (ai[np.ix_(free, free)] + ai[np.ix_(free, free)].T)
    grad_free = grad[free]
    try:
        step[free] = np.linalg.solve(ai_free, grad_free)
    except np.linalg.LinAlgError:
        step[free] = np.linalg.solve(ai_free + np.eye(np.sum(free)) * ridge, grad_free)
    return step, free


def raw_step_metrics(x_prev: np.ndarray, x_next: np.ndarray) -> tuple[float, float]:
    """Absolute and relative parameter change on the raw reported scale."""
    raw_prev = np.exp(x_prev)
    raw_next = np.exp(x_next)
    d_raw = raw_next - raw_prev
    raw_scale = np.maximum(1.0, np.abs(raw_prev))
    max_param_change = float(np.max(np.abs(d_raw)))
    max_rel_param_change = float(np.max(np.abs(d_raw) / raw_scale))
    return max_param_change, max_rel_param_change


def reml_objective_grad_ai(params: np.ndarray, y: np.ndarray, G: np.ndarray,
                           p_hat: np.ndarray, sigma2_e: float, rho: float):
    """REML NLL, its analytic gradient, and the Average Information (AI) matrix."""
    log_s2a_over_Ws, log_s2a = params
    # ── Site C: clip before exp to prevent float64 underflow/overflow ────────
    _LOG_SAFE = 25.0
    sigma2_a_over_Ws = np.exp(np.clip(log_s2a_over_Ws, -_LOG_SAFE, _LOG_SAFE))
    sigma2_a = np.exp(np.clip(log_s2a, -_LOG_SAFE, _LOG_SAFE))
    Ws = sigma2_a / sigma2_a_over_Ws

    N    = y.shape[0]
    ones = np.ones(N)
    sv, d_logWs_ind, d_logs2a_ind = snp_variance_derivatives(
        p_hat, Ws, sigma2_a, rho=rho
    )

    # ── Build and factor V ───────────────────────────────────────────────────
    backend, factor_data, log_det = factor_v(G, sv, sigma2_e)
    if backend is None:
        return np.inf, np.zeros(2), np.eye(2)

    # ── ML quantities ────────────────────────────────────────────────────────
    alpha   = solve_v(backend, factor_data, y)         # V^{-1} y   (N,)
    B       = solve_v(backend, factor_data, G)         # V^{-1} G   (N, M)

    # ── REML projection: compute P = V^{-1} - V^{-1} 1 (1^T V^{-1} 1)^-1 1^T V^{-1}
    v1    = solve_v(backend, factor_data, ones)        # V^{-1} 1   (N,)
    c_r   = ones @ v1                                  # 1^T V^{-1} 1  (scalar)
    # ── Site D: guard c_r before division and log ─────────────────────────────
    c_r_safe = max(float(c_r), 1e-300)
    scale    = 1.0 / c_r_safe

    Py    = alpha - v1 * ((ones @ alpha) * scale)      # P y        (N,)
    PG    = B     - np.outer(v1, (ones @ B) * scale)   # P G        (N, M)

    # ── REML log-likelihood ──────────────────────────────────────────────────
    quad_reml = y @ Py                                 # y^T P y    scalar
    nreml = 0.5 * (log_det + np.log(c_r_safe) + quad_reml
                   + (N - 1) * np.log(2.0 * np.pi))

    # ── Per-SNP sensitivities ∂sv_j / ∂theta_k ──────────────────────────────
    # Reparameterize to theta = [log(s2a/Ws), log(s2a)]:
    # log(Ws) = log(s2a) - log(s2a/Ws)
    d_s2a_over_Ws = -d_logWs_ind
    d_s2a = d_logWs_ind + d_logs2a_ind

    # ── REML gradient ────────────────────────────────────────────────────────
    Gt_Py    = G.T @ Py                               # G^T P y    (M,)
    diag_GPG = (G * PG).sum(axis=0)                   # diag(G^T P G) (M,)
    residual = diag_GPG - Gt_Py ** 2                  # (M,)
    grad = 0.5 * np.array([
        np.dot(d_s2a_over_Ws, residual),
        np.dot(d_s2a, residual),
    ])

    # ── Average Information (AI) Matrix ──────────────────────────────────────
    w_s2a_over_Ws = d_s2a_over_Ws * Gt_Py             # (M,)
    w_s2a = d_s2a * Gt_Py                             # (M,)

    v_s2a_over_Ws = G @ w_s2a_over_Ws                 # (N,)
    v_s2a = G @ w_s2a                                 # (N,)

    v1_s2a_over_Ws = solve_v(backend, factor_data, v_s2a_over_Ws)  # V^{-1} v_s2a_over_Ws
    v1_s2a = solve_v(backend, factor_data, v_s2a)     # V^{-1} v_s2a

    P_v_s2a_over_Ws = v1_s2a_over_Ws - v1 * ((ones @ v1_s2a_over_Ws) * scale)
    P_v_s2a = v1_s2a - v1 * ((ones @ v1_s2a) * scale)

    ai_11 = 0.5 * np.dot(v_s2a_over_Ws, P_v_s2a_over_Ws)
    ai_12 = 0.5 * np.dot(v_s2a_over_Ws, P_v_s2a)
    ai_22 = 0.5 * np.dot(v_s2a, P_v_s2a)

    AI = np.array([
        [ai_11, ai_12],
        [ai_12, ai_22]
    ])

    if not np.isfinite(nreml) or not np.all(np.isfinite(grad)) or not np.all(np.isfinite(AI)):
        return np.inf, np.zeros(2), np.eye(2)

    return nreml, grad, AI


def reml_objective_only(params: np.ndarray, y: np.ndarray, G: np.ndarray,
                        p_hat: np.ndarray, sigma2_e: float, rho: float) -> float:
    """Compute only the REML NLL for step-halving line search."""
    log_s2a_over_Ws, log_s2a = params
    _LOG_SAFE = 25.0
    sigma2_a_over_Ws = np.exp(np.clip(log_s2a_over_Ws, -_LOG_SAFE, _LOG_SAFE))
    sigma2_a = np.exp(np.clip(log_s2a, -_LOG_SAFE, _LOG_SAFE))
    Ws = sigma2_a / sigma2_a_over_Ws

    N    = y.shape[0]
    ones = np.ones(N)
    sv, _, _ = snp_variances(p_hat, Ws, sigma2_a, rho=rho)

    backend, factor_data, log_det = factor_v(G, sv, sigma2_e)
    if backend is None:
        return np.inf

    alpha   = solve_v(backend, factor_data, y)
    v1      = solve_v(backend, factor_data, ones)

    c_r_safe = max(float(ones @ v1), 1e-300)
    scale    = 1.0 / c_r_safe

    Py        = alpha - v1 * ((ones @ alpha) * scale)
    quad_reml = y @ Py

    nreml = 0.5 * (log_det + np.log(c_r_safe) + quad_reml
                   + (N - 1) * np.log(2.0 * np.pi))
    return nreml


def load_phenotype(pheno_file: str) -> np.ndarray:
    """Load per-individual phenotype vector written by SLiM.

    The .pheno file is a plain text file with one floating-point value per line,
    written by main.slim at the terminal tick.
    """
    y = np.loadtxt(pheno_file, dtype=np.float64)
    y += np.random.normal(0, np.sqrt(SIGMA2_E), y.shape[0])
    print(f"  Phenotype loaded: n={y.shape[0]}  mean={y.mean():.4f}  var={y.var():.4f}",
          flush=True)
    return y


def fit_blm(vcf_file: str, pheno_file: str, rep_id: int,
            sigma2_a_true: float, vs_scale: float, rho_true: float,
            output_csv: str) -> None:
    """Main fitting routine for one replicate.

    Parameters
    ----------
    sigma2_a_true : float
        Prior variance (= wildcard value, e.g. 0.5).
    vs_scale : float
        True ratio of V_S to 2*N (= wildcards.vs_scale).
    """
    print(f"[rep {rep_id}] Loading {vcf_file}", flush=True)
    G = extract_genotype_matrix(vcf_file)
    p_hat = compute_allele_freqs(G)

    y = load_phenotype(pheno_file)

    if y.shape[0] != G.shape[0]:
        raise ValueError(
            f"Phenotype length {y.shape[0]} does not match "
            f"number of individuals {G.shape[0]}"
        )

    # ── Optimise (AI-REML) ───────────────────────────────────────────────────
    Ws_eff = vs_scale / 2.0
    sigma2_a_over_Ws_true = sigma2_a_true / Ws_eff
    x_curr = np.array([np.log(sigma2_a_over_Ws_true), np.log(sigma2_a_true)])
    
    print(f"[rep {rep_id}] Optimising marginal likelihood (AI-REML) "
          f"(rho={rho_true}, vs_scale={vs_scale}, sigma2_a_true={sigma2_a_true}, "
          f"Ws_eff={Ws_eff:.6g}, "
          f"s2a_over_Ws_true={sigma2_a_over_Ws_true:.6g}) ...", flush=True)

    max_iter = 50
    ftol = 1e-8
    gtol = 1e-5
    xtol_abs = 1e-3
    xtol_rel = 1e-3
    converged = False
    
    # Compute initial state
    nreml, grad, AI = reml_objective_grad_ai(x_curr, y, G, p_hat, SIGMA2_E, rho_true)
    
    for it in range(max_iter):
        g_norm = projected_grad_maxnorm(x_curr, grad, LOG_PARAM_MIN, LOG_PARAM_MAX)
        if g_norm < gtol:
            converged = True
            break
            
        step, free = solve_reduced_ai_step(
            x_curr, grad, AI, LOG_PARAM_MIN, LOG_PARAM_MAX
        )
        if not np.any(free):
            converged = True
            break
            
        # Step-halving line search
        alpha = 1.0
        success_step = False
        
        for _ in range(10):  # max 10 halving steps
            x_prev = x_curr
            x_next = x_curr - alpha * step
            
            # Bound parameters
            x_next = np.clip(x_next, LOG_PARAM_MIN, LOG_PARAM_MAX)
            was_clipped = not np.allclose(x_next, x_curr - alpha * step)
            
            nreml_next = reml_objective_only(x_next, y, G, p_hat, SIGMA2_E, rho_true)
            
            if not np.isfinite(nreml_next):
                alpha *= 0.5
                continue
                
            # Strict decrease (or extremely small change)
            if nreml_next < nreml or np.abs(nreml - nreml_next) < ftol:
                max_param_change, max_rel_param_change = raw_step_metrics(x_prev, x_next)

                if np.abs(nreml - nreml_next) < ftol:
                    converged = True
                elif max_param_change < xtol_abs or max_rel_param_change < xtol_rel:
                    converged = True
                    
                x_curr = x_next
                success_step = True
                break
                
            alpha *= 0.5
            
        if not success_step:
            if projected_grad_maxnorm(x_curr, grad, LOG_PARAM_MIN, LOG_PARAM_MAX) < gtol:
                converged = True
            break
            
        if converged:
            break
            
        # Compute new gradient and AI for next iteration
        nreml, grad, AI = reml_objective_grad_ai(x_curr, y, G, p_hat, SIGMA2_E, rho_true)

    sigma2_a_over_Ws_est = np.exp(x_curr[0])
    sigma2_a_est = np.exp(x_curr[1])
    Ws_est = sigma2_a_est / sigma2_a_over_Ws_est
    neg_ll = nreml

    print(f"[rep {rep_id}] "
          f"s2a_over_Ws_est={sigma2_a_over_Ws_est:.6f}  "
          f"s2a_est={sigma2_a_est:.6f}  Ws_implied={Ws_est:.2f}  "
          f"neg_loglik={neg_ll:.4f}  success={converged}",
          flush=True)

    # ── Write output ─────────────────────────────────────────────────────────
    # sigma2_a_true stored as the *scaled* (simulation) value so that
    # plot_blm.py can back-scale both columns uniformly by dividing by scale.
    fieldnames = [
        "rep_id", "sigma2_e", "sigma2_a_true", "rho_true", "Ws_true",
        "sigma2_a_over_Ws_true", "sigma2_a_over_Ws_est",
        "sigma2_a_est", "neg_loglik", "converged",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "rep_id":         rep_id,
            "sigma2_e":       SIGMA2_E,
            "sigma2_a_true":  sigma2_a_true,
            "rho_true":       rho_true,
            "Ws_true":        Ws_eff,
            "sigma2_a_over_Ws_true": sigma2_a_over_Ws_true,
            "sigma2_a_over_Ws_est":  sigma2_a_over_Ws_est,
            "sigma2_a_est":   sigma2_a_est,
            "neg_loglik":     neg_ll,
            "converged":      int(converged),
        })
    print(f"[rep {rep_id}] Results written to {output_csv}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python fit_blm.py <vcf_file> <pheno_file> <rep_id> <sigma2_a_true> <vs_scale> <rho_true> <output_csv>")
        sys.exit(1)

    vcf_file       = sys.argv[1]
    pheno_file     = sys.argv[2]
    rep_id         = int(sys.argv[3])
    sigma2_a_true  = float(sys.argv[4])
    vs_scale       = float(sys.argv[5])
    rho_true       = float(sys.argv[6])
    output_csv     = sys.argv[7]

    fit_blm(vcf_file, pheno_file, rep_id, sigma2_a_true, vs_scale, rho_true, output_csv)
