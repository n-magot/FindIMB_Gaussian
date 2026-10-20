"""
============================================================================
        FindIMB Experimental Step with the BGe Score (continuous outcome)
============================================================================

Given the observational Markov boundary MB of a continuous outcome Y (from
FindIMB_BGeScore_ContinuousOutcome.py), searches over subsets Z of
MB \\ {X} with a pruned forward search and scores, for each visited Z, the
two FindIMB hypotheses, with parent set P = {X} u Z:

  H_Z^c    : P(Y | X, Z) is the same in D_o and D_e (D_o can be pooled)
             log P(D_e | D_o, H_Z^c)  = s(Y | P ; D_o u D_e) - s(Y | P ; D_o)
  H_Z^cbar : D_e has its own P(Y | X, Z)
             log P(D_e | H_Z^cbar)    = s(Y | P ; D_e)

where s(Y | P ; D) = log p(d^{P u {Y}}) - log p(d^{P}) is the BGe local
score (Kuipers, Moffa & Heckerman 2014, Eq. 2), together with the
observational score log P(D_o | H_Z^o) = s(Y | P ; D_o).

FORWARD SEARCH (as in the discrete FindIMB code):
  The search starts from Z = {} (parent set {X}) and proceeds layer by layer;
  each kept node is expanded by adding one more covariate group. Every
  (parent, child) edge is recorded in a lattice, including edges into
  children that were already scored. After each layer the scores of all
  nodes visited so far are normalized: H^c and H^cbar jointly, H^o
  separately. A new node is expanded further only if at least one of its
  three normalized scores exceeds THRESHOLD. With THRESHOLD < 0 every subset
  is visited and the result equals exhaustive enumeration.

Posterior over the 2 x #visited hypotheses, with prior P(H_Z^c) = P_HC / #visited,
P(H_Z^cbar) = (1 - P_HC) / #visited:

  P(H_Z^c | D)    proportional to  P(D_e | D_o, H_Z^c) P(H_Z^c)
  P(H_Z^cbar | D) proportional to  P(D_e | H_Z^cbar)   P(H_Z^cbar)

With USE_OBS_TERM = True the observational term of Eq. 3 is included:
P(D_o | H_Z^o) multiplies H_Z^c, and P(D_o | H_Z^obar) multiplies H_Z^cbar,
where P(D_o | H_Z^obar) is the weighted average of P(D_o | H_U^o) over the
nodes U reachable from Z in the lattice (supersets of Z), with weights
1 / [C(m-|Z|, |U|-|Z|) (m-|Z|+1)] renormalized over the reachable nodes
(m = number of covariate groups in MB \\ {X}). Without pruning the weights
already sum to 1.

PREDICTION (Bayesian model averaging over the visited Z and both hypotheses):
  p(y | x, MB, D) = sum_Z P(H_Z^c | D) p(y | x, Z, D_o u D_e)
                          + P(H_Z^cbar | D) p(y | x, Z, D_e),
each term a Student-t posterior predictive; the point prediction is the
same average of the predictive means.
--------------------
BGe assumes all variables (outcome and covariates) are jointly Gaussian.

THE PRIOR (weakly informative Normal-Wishart)
---------------------------------------------
All columns are standardized with the OBSERVATIONAL (D_o) mean and sd, and
the SAME transformation is applied to D_e. A common affine transformation
preserves P(Y | X, Z), which H_Z^c assumes equal across D_o and D_e; a
separate transformation of D_e would not, because the intervention changes
the distribution of X. With n = total number of columns (Y, X and every MB
covariate column, each dummy counted once):

    mu | W ~ N(nu, (a_mu W)^{-1}),   W ~ Wishart(a_w, T)
    nu = 0,  a_mu = 1,  a_w = n + 2,  T = t I,  t = 1/2.

This is the standard weak BGe prior (Kuipers, Moffa & Heckerman 2014,
supplement Eq. B.6). It is centred on independence and on the scale of the
standardized data:
t = a_mu (a_w - n - 1) / (a_mu + 1) = 1/2, the smallest integer prior sample
sizes (a_mu = 1, a_w = n + 2) for which this calibration holds. It plays the
role of the weak Dirichlet prior a_jk = 1 of Triantafillou et al. (2021),
though it is derived from one joint prior rather than set per conditional.
The alternative a_w = n + 1, t = 2.

nu = 0 is the D_o mean for EVERY dataset (not re-centred on the data being
scored): re-centring would use D_e's own mean under H_Z^cbar and erase the
mean shift the H_Z^c test is meant to detect.
"""

import itertools
import numpy as np
import pandas as pd
from scipy.special import comb, gammaln, logsumexp, multigammaln


# ===========================================================================
# 0. USER CONFIG -- edit this block for your dataset
# ===========================================================================

OBS_PATH = "path/to/observational.csv"     # <-- EDIT: observational data D_o
EXP_PATH = "path/to/experimental.csv"      # <-- EDIT: experimental data D_e
TEST_PATH = None                           # optional: new cases to predict

OUTCOME_COL = "Y"                           # <-- EDIT: the continuous outcome
TREATMENT_COL = "X"                         # <-- EDIT: the treatment
TREATMENT_TYPE = "binary"                   # "binary", "nominal" or "continuous"

# Markov boundary covariates (without X), e.g. from the MB search
MB_NOMINAL_COLS = [
    # "your_nominal_col_1",
]
MB_BINARY_COLS = [
    # "your_binary_col_1",
]
MB_CONTINUOUS_COLS = [
    # "your_continuous_col_1",
]

# Optional: deliberate reference level per nominal column (drops that level
# in dummy coding). If unset, the alphabetically-first level is used.
NOMINAL_REF = {}

# Optional: deliberate "positive" value per binary column (maps to 1). If
# unset, the second value (sorted) is used.
BINARY_POSITIVE = {}

# --- BGe hyperparameters (Normal-Wishart) ---
A_MU = 1.0            # alpha_mu > 0
A_W  = None           # alpha_w > n-1; None -> n + 2 (standard weak BGe, KMH 2014)
T_SCALE = None        # None -> t = 1/2 (KMH supp Eq. B.6)

# --- hypotheses ---
P_HC = 0.5            # prior mass on H^c (spread uniformly over Z); H^cbar gets 1 - P_HC
USE_OBS_TERM = False  # True: include P(D_o | H_Z^o), P(D_o | H_Z^obar) (Eq. 3)

# --- forward search ---
THRESHOLD = 0.1       # expand a node if any normalized score exceeds this; < 0 -> exhaustive


# ===========================================================================
# 1. Load data
# ===========================================================================

def load_data(path):
    df = pd.read_csv(path)
    needed = ([OUTCOME_COL, TREATMENT_COL] + MB_NOMINAL_COLS + MB_BINARY_COLS
              + MB_CONTINUOUS_COLS)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: configured columns not in data: {missing}\n"
                         f"Available: {list(df.columns)}")
    return df


# ===========================================================================
# 2. Encoding + standardization (one scale for D_o, D_e and the test set)
# ===========================================================================

def resolve_nominal_ref(pooled, col):
    if col in NOMINAL_REF:
        return NOMINAL_REF[col]
    default = sorted(pooled[col].unique())[0]
    print(f"NOTE: no reference level set for nominal '{col}' -> using '{default}'.")
    return default


def resolve_binary_positive(pooled, col):
    if col in BINARY_POSITIVE:
        return BINARY_POSITIVE[col]
    vals = sorted(pooled[col].unique())
    default = vals[1] if len(vals) > 1 else vals[0]
    print(f"NOTE: no positive value set for binary '{col}' -> using '{default}'.")
    return default


def encode_group(pooled, col, kind):
    """Numeric (N, k) block for one variable (a nominal variable is one group
    of L-1 dummy columns, added to / removed from parent sets as a unit)."""
    if kind == "nominal":
        ref = resolve_nominal_ref(pooled, col)
        levels = [l for l in sorted(pooled[col].unique()) if l != ref]
        return np.column_stack([(pooled[col] == l).astype(float).values for l in levels])
    if kind == "binary":
        pos = resolve_binary_positive(pooled, col)
        return (pooled[col] == pos).astype(float).values.reshape(-1, 1)
    if kind == "continuous":
        return pooled[col].astype(float).values.reshape(-1, 1)
    raise ValueError(f"Unknown variable type: {kind}")


def build_data(df_o, df_e, df_test=None):
    """
    Returns standardized matrices D_o, D_e (and D_test), the column layout
    `cols` (group name -> column indices; "__Y__" and "__X__" for the outcome
    and treatment) and the outcome back-transform.

    Standardization constants (mean, sd) and the log1p decision use D_o ONLY,
    and the same transformation is applied to D_e and the test set. H_Z^c
    assumes P(Y | X, Z) is the same in D_o and D_e, which a common affine
    transformation preserves; estimating the transformation on D_e too, or
    pooling, would partly absorb a genuine shift and weaken the invariance
    test. Encoding conventions (reference levels, the "positive" binary value)
    are read from the pooled frame -- they are label choices, not leakage.
    """
    N_o = len(df_o)
    n_train = len(df_o) + len(df_e)
    parts = [df_o, df_e] + ([df_test] if df_test is not None else [])
    pooled = pd.concat(parts, ignore_index=True)

    # outcome: skew check on D_o only -> log1p if strongly right-skewed
    y_raw = pooled[OUTCOME_COL].astype(float)
    y_obs = y_raw.iloc[:N_o]
    log_y = bool(y_obs.skew() > 1.0 and (y_raw >= 0).all())
    if log_y:
        print(f"NOTE: D_o outcome skew = {y_obs.skew():.2f} > 1 -> applying log1p.")
    y = np.log1p(y_raw.values) if log_y else y_raw.values

    blocks = [("__Y__", y.reshape(-1, 1)),
              ("__X__", encode_group(pooled, TREATMENT_COL, TREATMENT_TYPE))]
    for col in MB_NOMINAL_COLS:
        blocks.append((col, encode_group(pooled, col, "nominal")))
    for col in MB_BINARY_COLS:
        blocks.append((col, encode_group(pooled, col, "binary")))
    for col in MB_CONTINUOUS_COLS:
        blocks.append((col, encode_group(pooled, col, "continuous")))

    raw = np.hstack([b for _, b in blocks])
    mean = raw[:N_o].mean(axis=0)          # D_o-only standardization
    sd = raw[:N_o].std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Z = (raw - mean) / sd

    cols, c = {}, 0
    for name, b in blocks:
        cols[name] = list(range(c, c + b.shape[1]))
        c += b.shape[1]

    y_transform = {"mean": mean[0], "sd": sd[0], "log1p": log_y}
    D_test = Z[n_train:] if df_test is not None else None
    return Z[:N_o], Z[N_o:n_train], cols, y_transform, D_test


# ===========================================================================
# 3. BGe score core  (Kuipers-Moffa-Heckerman 2014, formula 2)
# ===========================================================================

def default_hyperparams(n, a_mu=A_MU, a_w=A_W, t_scale=T_SCALE):
    """Standard weak BGe prior on n variables: a_w = n + 2, T = (1/2) I
    (Kuipers, Moffa & Heckerman 2014, supplement Eq. B.6)."""
    if a_w is None:
        a_w = n + 2                       # standard weak BGe choice
    if a_w <= n - 1:
        raise ValueError(f"a_w must exceed n-1 = {n-1}; got {a_w}")
    nu = np.zeros(n)
    if t_scale is None:
        # t = a_mu (a_w - n - 1) / (a_mu + 1); with a_mu=1, a_w=n+2 -> t = 1/2
        t_scale = a_mu * (a_w - n - 1) / (a_mu + 1)
    if t_scale <= 0:
        raise ValueError(
            f"t_scale must be > 0; the calibration t = a_mu(a_w-n-1)/(a_mu+1) "
            f"gives t <= 0 for a_w = {a_w}, a_mu = {a_mu}. Need a_w > n + 1.")
    T = t_scale * np.eye(n)
    return T, nu, a_mu, a_w


class BGe:
    """One Normal-Wishart prior on all n columns; BGe scores and posterior
    predictives on any dataset with that column layout."""

    def __init__(self, n, a_mu=A_MU, a_w=A_W, t_scale=T_SCALE):
        self.n = n
        self.T, self.nu, self.a_mu, self.a_w = default_hyperparams(
            n, a_mu=a_mu, a_w=a_w, t_scale=t_scale)

    def posterior(self, data):
        """N, mu_N and R = T + S_N + (N a_mu/(N+a_mu)) (nu - xbar)(nu - xbar)^T."""
        N = data.shape[0]
        xbar = data.mean(axis=0)
        centered = data - xbar
        diff = (self.nu - xbar).reshape(-1, 1)
        R = self.T + centered.T @ centered + (N * self.a_mu / (N + self.a_mu)) * (diff @ diff.T)
        mu_N = (self.a_mu * self.nu + N * xbar) / (self.a_mu + N)
        return {"N": N, "R": R, "mu_N": mu_N}

    def log_marginal(self, post, idx):
        """log p(d^S) for the column subset S = idx, per formula (2)."""
        l = len(idx)
        if l == 0:
            return 0.0
        N, n, a_mu, a_w = post["N"], self.n, self.a_mu, self.a_w
        ix = np.ix_(idx, idx)
        _, logdet_T = np.linalg.slogdet(self.T[ix])
        _, logdet_R = np.linalg.slogdet(post["R"][ix])
        return float(
            (l / 2.0) * (np.log(a_mu) - np.log(N + a_mu))
            + multigammaln((N + a_w - n + l) / 2.0, l)
            - (l * N / 2.0) * np.log(np.pi)
            - multigammaln((a_w - n + l) / 2.0, l)
            + ((a_w - n + l) / 2.0) * logdet_T
            - ((N + a_w - n + l) / 2.0) * logdet_R
        )

    def score(self, post, y_col, par_cols):
        """BGe local score  s(Y | P ; D) = log p(d^{P u Y}) - log p(d^P)."""
        return self.log_marginal(post, par_cols + [y_col]) - self.log_marginal(post, par_cols)

    def predictive_mean(self, post, y_col, par_cols, x_par):
        """E[Y | x_P, D] = mu_N[Y] + R[Y,P] R[P,P]^{-1} (x_P - mu_N[P])."""
        mu, R = post["mu_N"], post["R"]
        if not par_cols:
            return np.full(x_par.shape[0], mu[y_col])
        beta = np.linalg.solve(R[np.ix_(par_cols, par_cols)], R[par_cols, y_col])
        return mu[y_col] + (x_par - mu[par_cols]) @ beta

    def predictive_logpdf(self, post, y_col, par_cols, x_par, y):
        """
        log p(y | x_P, D): Student-t posterior predictive of Y given P.
        Predictive of x_F, F = P u {Y}: multivariate t with df = a_w + N - n + 1,
        location mu_N[F], scale c R[F, F], c = (a_mu + N + 1)/((a_mu + N) df).
        Conditioning on x_P (p = |P|): univariate t with df + p degrees of
        freedom, location as in predictive_mean, and
        scale^2 = c R_{Y.P} (df + d2) / (df + p),
        d2 = (x_P - mu_N[P])' (c R[P,P])^{-1} (x_P - mu_N[P]).
        """
        N, mu, R = post["N"], post["mu_N"], post["R"]
        df = self.a_w + N - self.n + 1
        c = (self.a_mu + N + 1) / ((self.a_mu + N) * df)
        y = np.asarray(y, dtype=float)
        if par_cols:
            R_PP = R[np.ix_(par_cols, par_cols)]
            R_PY = R[par_cols, y_col]
            beta = np.linalg.solve(R_PP, R_PY)
            dx = x_par - mu[par_cols]
            m = mu[y_col] + dx @ beta
            r_cond = R[y_col, y_col] - R_PY @ beta
            d2 = np.einsum("ij,ij->i", dx, np.linalg.solve(R_PP, dx.T).T) / c
            p = len(par_cols)
        else:
            m = np.full(y.shape, mu[y_col])
            r_cond = R[y_col, y_col]
            d2 = np.zeros(y.shape)
            p = 0
        df_c = df + p
        s2 = c * r_cond * (df + d2) / df_c
        z2 = (y - m) ** 2 / (df_c * s2)
        return (gammaln((df_c + 1) / 2.0) - gammaln(df_c / 2.0)
                - 0.5 * np.log(df_c * np.pi * s2)
                - (df_c + 1) / 2.0 * np.log1p(z2))


# ===========================================================================
# 4. The two hypotheses, scored over a pruned forward search of Z
# ===========================================================================

def forward_search_hypotheses(bge, D_o, D_e, cols, threshold=THRESHOLD,
                              p_hc=P_HC, use_obs_term=USE_OBS_TERM):
    """
    Pruned forward search over subsets Z of MB \\ {X} (as in the discrete
    FindIMB code), then the posterior over (Z, H_Z^c / H_Z^cbar).

    Each visited node Z gets three BGe log-likelihoods:
        log P(D_e | D_o, H_Z^c),  log P(D_e | H_Z^cbar),  log P(D_o | H_Z^o).
    After each layer, all visited nodes are normalized (H^c and H^cbar
    jointly, with the prior P_HC; H^o separately), and a new node is expanded
    only if any of its three normalized scores exceeds `threshold`.

    Returns (res, posts): one row per visited Z with
      log_P_De_given_Do_HZc  = s(Y|P; D_o u D_e) - s(Y|P; D_o)
      log_P_De_given_HZc_bar = s(Y|P; D_e)
      [log_P_Do_given_HZo, log_P_Do_given_HZo_bar if use_obs_term]
      P_HZc, P_HZc_bar       posterior over the 2 x #visited hypotheses
      P_c_given_Z            P(H_Z^c | D) / (P(H_Z^c | D) + P(H_Z^cbar | D))
    and the posteriors of D_o, D_e and D_o u D_e.
    """
    posts = {"o": bge.posterior(D_o), "e": bge.posterior(D_e),
             "oe": bge.posterior(np.vstack([D_o, D_e]))}
    y_col, x_cols = cols["__Y__"][0], cols["__X__"]
    covs = [g for g in cols if g not in ("__Y__", "__X__")]
    m = len(covs)

    def canon(groups):
        """Canonical node label: the groups of Z in the order of `covs`."""
        return tuple(g for g in covs if g in groups)

    lp_c, lp_cb, lp_o = {}, {}, {}     # log P(De|Do,HZc), log P(De|HZcbar), log P(Do|HZo)
    lattice = {}                        # parent -> children

    def score_node(Z):
        par = x_cols + [c for g in Z for c in cols[g]]
        s_o = bge.score(posts["o"], y_col, par)
        lp_c[Z] = bge.score(posts["oe"], y_col, par) - s_o
        lp_cb[Z] = bge.score(posts["e"], y_col, par)
        lp_o[Z] = s_o
        lattice[Z] = []

    # --- initial node: treatment only ---
    root = ()
    score_node(root)
    frontier = [root]

    # --- forward layered search ---
    for _ in range(m):
        next_layer = set()
        for Z in frontier:
            for g in covs:
                if g not in Z:
                    next_layer.add((canon(set(Z) | {g}), Z))

        frontier = []
        # sorted() makes the search order, and so the results, reproducible
        for Z, parent in sorted(next_layer):
            # every edge is recorded, also into already-scored children,
            # because P(D_o | H_Z^obar) averages over all nodes reachable from Z
            lattice[parent].append(Z)
            if Z in lp_c:
                continue
            score_node(Z)
            frontier.append(Z)

        # normalization is global, over every node visited so far
        nodes = list(lp_c)
        a = np.array([lp_c[z] for z in nodes]) + np.log(p_hc)
        b = np.array([lp_cb[z] for z in nodes]) + np.log(1 - p_hc)
        o = np.array([lp_o[z] for z in nodes])
        lz, lzo = logsumexp(np.concatenate([a, b])), logsumexp(o)
        keep = {z for z, ai, bi, oi in zip(nodes, a, b, o)
                if np.exp(ai - lz) > threshold
                or np.exp(bi - lz) > threshold
                or np.exp(oi - lzo) > threshold}

        # expand only subsets whose score passes the threshold
        frontier = [z for z in frontier if z in keep]
        if not frontier:
            break

    # --- final phase: posterior over (Z, hypothesis) ---
    nodes = list(lp_c)
    res = pd.DataFrame({
        "Z": nodes,
        "log_P_De_given_Do_HZc": [lp_c[z] for z in nodes],
        "log_P_De_given_HZc_bar": [lp_cb[z] for z in nodes],
    })

    # the uniform prior 1 / #visited is the same for every node and cancels
    a = res["log_P_De_given_Do_HZc"].values + np.log(p_hc)
    b = res["log_P_De_given_HZc_bar"].values + np.log(1 - p_hc)

    if use_obs_term:
        def reachable(Z):
            """All nodes reachable from Z in the lattice, including Z."""
            seen, stack = set(), [Z]
            while stack:
                n = stack.pop()
                if n not in seen:
                    seen.add(n)
                    stack.extend(lattice[n])
            return seen

        def log_P_Do_HZo_bar(Z):
            """Weighted average of P(D_o | H_U^o) over U reachable from Z."""
            log_terms, log_w = [], []
            for U in reachable(Z):
                lw = (-np.log(comb(m - len(Z), len(U) - len(Z), exact=True))
                      - np.log(m - len(Z) + 1))
                log_w.append(lw)
                log_terms.append(lp_o[U] + lw)
            # renormalize over the reachable nodes (= 0 when nothing was pruned)
            return logsumexp(log_terms) - logsumexp(log_w)

        lo = np.array([lp_o[z] for z in nodes])
        lo_bar = np.array([log_P_Do_HZo_bar(z) for z in nodes])
        res["log_P_Do_given_HZo"] = lo
        res["log_P_Do_given_HZo_bar"] = lo_bar
        a, b = a + lo, b + lo_bar

    lz = logsumexp(np.concatenate([a, b]))
    res["P_HZc"] = np.exp(a - lz)
    res["P_HZc_bar"] = np.exp(b - lz)
    res["P_c_given_Z"] = res["P_HZc"] / (res["P_HZc"] + res["P_HZc_bar"])
    return res, posts


# ===========================================================================
# 5. Prediction (Bayesian model averaging over Z and both hypotheses)
# ===========================================================================

def bma_predict(bge, res, posts, cols, D_new):
    """
    For the rows of D_new (standardized, same layout as D_o / D_e) returns
      mean   : sum_Z P(H_Z^c|D) E[Y | x, Z, D_o u D_e] + P(H_Z^cbar|D) E[Y | x, Z, D_e]
      logpdf : log of the same mixture of predictive densities at y
               (only meaningful when D_new contains the outcome)
    both on the standardized outcome scale.
    """
    y_col, x_cols = cols["__Y__"][0], cols["__X__"]
    y = D_new[:, y_col]
    mean = np.zeros(D_new.shape[0])
    terms = []
    for _, r in res.iterrows():
        par = x_cols + [c for g in r["Z"] for c in cols[g]]
        x_par = D_new[:, par]
        for w, post in [(r["P_HZc"], posts["oe"]), (r["P_HZc_bar"], posts["e"])]:
            mean += w * bge.predictive_mean(post, y_col, par, x_par)
            terms.append(np.log(w + 1e-300) + bge.predictive_logpdf(post, y_col, par, x_par, y))
    return mean, logsumexp(np.vstack(terms), axis=0)


def to_outcome_scale(pred_z, y_transform):
    """Undo the standardization (and log1p, if applied) of the outcome."""
    y = pred_z * y_transform["sd"] + y_transform["mean"]
    return np.expm1(y) if y_transform["log1p"] else y


def to_outcome_logpdf(logpdf_z, y_orig, y_transform):
    """Log density on the original outcome scale (Jacobian of the transforms)."""
    out = logpdf_z - np.log(y_transform["sd"])
    if y_transform["log1p"]:
        out = out - np.log1p(np.asarray(y_orig, dtype=float))
    return out


# ===========================================================================
# 6. Main
# ===========================================================================

def main():
    df_o, df_e = load_data(OBS_PATH), load_data(EXP_PATH)
    df_t = load_data(TEST_PATH) if TEST_PATH else None
    D_o, D_e, cols, y_transform, D_t = build_data(df_o, df_e, df_t)
    bge = BGe(D_o.shape[1])

    covs = [g for g in cols if g not in ("__Y__", "__X__")]
    print(f"Score      : BGe local scores  [Kuipers-Moffa-Heckerman 2014]")
    print(f"Hypotheses : H_Z^c, H_Z^cbar for subsets Z of MB \\ {{X}}, forward search "
          f"(threshold {THRESHOLD}; observational term: {USE_OBS_TERM})")
    print(f"MB covariate groups ({len(covs)}): {covs}")
    print(f"N_o = {D_o.shape[0]}, N_e = {D_e.shape[0]}, n = {bge.n} columns")
    print(f"Prior: Normal-Wishart  a_mu={bge.a_mu}, a_w={bge.a_w}, T={bge.T[0, 0]:.4f}*I, "
          f"P(H^c) = {P_HC}")

    res, posts = forward_search_hypotheses(bge, D_o, D_e, cols)
    print(f"Visited {len(res)} of {2 ** len(covs)} subsets.")
    out = res.copy()
    out["Z"] = out["Z"].apply(lambda z: ", ".join(z) if z else "(empty)")
    out = out.sort_values("P_HZc", ascending=False)
    pd.set_option("display.width", 140)
    print("\n=== Hypotheses (parent set = {X} u Z) ===")
    print(out.round(4).to_string(index=False))
    print(f"\nTotal P(H^c) = {res['P_HZc'].sum():.3f}   "
          f"Total P(H^cbar) = {res['P_HZc_bar'].sum():.3f}")
    top = out.iloc[0]
    print(f"Most probable H^c set: Z = {top['Z']}  (P = {top['P_HZc']:.3f})")

    if D_t is not None:
        mean_z, lp_z = bma_predict(bge, res, posts, cols, D_t)
        pred = to_outcome_scale(mean_z, y_transform)
        print(f"\nPredicted {len(pred)} test cases (BMA mean, original scale).")
        if OUTCOME_COL in df_t and df_t[OUTCOME_COL].notna().all():
            y = df_t[OUTCOME_COL].values.astype(float)
            lp = to_outcome_logpdf(lp_z, y, y_transform)
            print(f"Test RMSE = {np.sqrt(np.mean((y - pred) ** 2)):.4f}   "
                  f"mean log predictive density = {lp.mean():.4f}")
    return res


if __name__ == "__main__":
    main()
