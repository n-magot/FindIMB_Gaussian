"""
============================================================================
        BGe Score for Continuous-Outcome Variable Selection
============================================================================

Scores candidate covariate sets Z for a continuous outcome Y with the
Bayesian Gaussian equivalent (BGe) score, and searches over Z (greedy or
exhaustive). Covariates may be continuous, binary or nominal.

SCOPE
-----
This file computes the observational term P(D_o | Z) of FindIMB, i.e. the
observational Markov boundary step:
    score(Y | Z) = log p(d^{Z u {Y}}) - log p(d^{Z}),

where each term is the closed-form Normal-Wishart marginal likelihood of
Kuipers, Moffa & Heckerman (2014, Eq. 2), evaluated on the sub-blocks of T
and R for the columns in that set. All candidate sets are scored against
ONE joint prior on (Y, all covariates), so the scores for Z and Z u {Z0}
are marginals of the same model and are directly comparable.
--------------------
BGe assumes all variables (outcome and covariates) are jointly Gaussian.

THE PRIOR (weakly informative Normal-Wishart)
---------------------------------------------
All columns are standardized (mean 0, sd 1). With n = total number of
columns (Y plus every covariate column, each dummy counted once):

    mu | W ~ N(nu, (a_mu W)^{-1}),   W ~ Wishart(a_w, T)
    nu = 0,  a_mu = 1,  a_w = n + 2,  T = t I,  t = 1/2.

This is the standard weak BGe prior (Kuipers, Moffa & Heckerman 2014,
supplement Eq. B.6; the bnlearn defaults iss.mu = 1, iss.w = ncol + 2).
It is centred on independence and on the scale of the standardized data:
t = a_mu (a_w - n - 1) / (a_mu + 1) = 1/2, the smallest integer prior sample
sizes (a_mu = 1, a_w = n + 2) for which this calibration holds. It plays the
role of the weak Dirichlet prior a_jk = 1 of Triantafillou et al. (2021),
though it is derived from one joint prior rather than set per conditional.
The alternative a_w = n + 1, t = 2 is reported in the sensitivity analysis.
"""

import itertools
import numpy as np
import pandas as pd
from scipy.special import multigammaln

# OLS p-values need a t CDF; scipy is already required, but keep the
# dependency-free fallback for portability.
try:
    from scipy import stats as _scipy_stats
    def _two_sided_p_value(t_stat, dof):
        return float(2 * _scipy_stats.t.sf(np.abs(t_stat), dof))
except ImportError:
    import math
    def _two_sided_p_value(t_stat, dof):
        z = abs(float(t_stat))
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


# ===========================================================================
# 0. USER CONFIG -- edit this block for your dataset
# ===========================================================================

DATA_PATH = "path/to/your/data.csv"    # <-- EDIT: your file path

OUTCOME_COL = "Y"                       # <-- EDIT: the continuous outcome

NOMINAL_COLS = [
    # "your_nominal_col_1",
]
BINARY_COLS = [
    # "your_binary_col_1",
]
CONTINUOUS_COLS = [
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
T_SCALE = None        # None -> t = 1/2 (KMH supp Eq. B.6; bnlearn default)

# --- search ---
SEARCH_MODE = "greedy"          # "exhaustive" or "greedy"
GREEDY_THRESHOLD = 0.01
CATEGORICAL_ENCODING = "dummy"  # "dummy" (recommended) or "integer"

RUN_OLS_DIAGNOSTIC = True


# ===========================================================================
# 1. Load data
# ===========================================================================

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    missing = [c for c in [OUTCOME_COL] + NOMINAL_COLS + BINARY_COLS + CONTINUOUS_COLS
               if c not in df.columns]
    if missing:
        raise ValueError(f"Configured columns not in data: {missing}\n"
                         f"Available: {list(df.columns)}")
    return df


# ===========================================================================
# 2. Encoding + standardization  (all columns end up numeric & standardized)
# ===========================================================================
# Under the full-Gaussian BGe assumption every column is treated as Gaussian,
# so the whole design (outcome + covariates) is standardized to mean 0, sd 1.
# This is the natural scale for the Normal-Wishart prior with nu = 0 and T = t I.

def zscore(x):
    x = np.asarray(x, dtype=float)
    s = x.std(ddof=0)
    return (x - x.mean()) / (s if s > 0 else 1.0)


def dummy_encode(series, ref_level):
    """L-1 indicator columns, reference level dropped."""
    levels = [l for l in sorted(series.unique()) if l != ref_level]
    return pd.DataFrame(
        {f"{series.name}_{lvl}": (series == lvl).astype(float) for lvl in levels},
        index=series.index,
    )


def integer_encode(series, level_order):
    """Single 0,1,2,... column -- imposes a false ordering on a nominal
    variable; provided only for comparison, not recommended."""
    mapping = {level: i for i, level in enumerate(level_order)}
    return pd.DataFrame({f"{series.name}_int": series.map(mapping).astype(float)},
                        index=series.index)


def resolve_nominal_ref(df, col):
    if col in NOMINAL_REF:
        return NOMINAL_REF[col]
    default = sorted(df[col].unique())[0]
    print(f"NOTE: no reference level set for nominal '{col}' -> using '{default}'.")
    return default


def resolve_binary_positive(df, col):
    if col in BINARY_POSITIVE:
        return BINARY_POSITIVE[col]
    vals = sorted(df[col].unique())
    default = vals[1] if len(vals) > 1 else vals[0]
    print(f"NOTE: no positive value set for binary '{col}' -> using '{default}'.")
    return default


def build_design(df, encoding=CATEGORICAL_ENCODING):
    """
    Returns:
      col_matrix : dict {var_group_name: (N, k) standardized ndarray}
                   (a nominal dummy-block is one group with k = L-1 columns,
                   so it is added to / removed from parent sets as a unit)
      y_z        : (N,) standardized outcome
    Every numeric column is z-scored, including binary/dummy columns, since
    the Gaussian prior treats them all on a common scale.
    """
    groups = {}

    for col in NOMINAL_COLS:
        ref = resolve_nominal_ref(df, col)
        if encoding == "dummy":
            block = dummy_encode(df[col], ref_level=ref)
        elif encoding == "integer":
            order = [ref] + [l for l in sorted(df[col].unique()) if l != ref]
            block = integer_encode(df[col], level_order=order)
        else:
            raise ValueError(f"Unknown CATEGORICAL_ENCODING: {encoding}")
        groups[col] = np.column_stack([zscore(block[c].values) for c in block.columns])

    for col in BINARY_COLS:
        pos = resolve_binary_positive(df, col)
        b = (df[col] == pos).astype(float).values
        groups[col] = zscore(b).reshape(-1, 1)

    for col in CONTINUOUS_COLS:
        groups[col] = zscore(df[col].values).reshape(-1, 1)

    if not groups:
        raise ValueError("No covariates configured -- set NOMINAL/BINARY/CONTINUOUS_COLS.")

    # outcome: skew check -> log1p if strongly right-skewed, then z-score
    y_raw = df[OUTCOME_COL].astype(float)
    if y_raw.skew() > 1.0 and (y_raw >= 0).all():
        print(f"NOTE: outcome skew = {y_raw.skew():.2f} > 1 -> applying log1p.")
        y_model = np.log1p(y_raw.values)
    else:
        y_model = y_raw.values
    y_z = zscore(y_model)

    return groups, y_z


# ===========================================================================
# 3. BGe score core  (Kuipers-Moffa-Heckerman 2014, formula 2)
# ===========================================================================

def default_hyperparams(n, a_mu=A_MU, a_w=A_W, t_scale=T_SCALE):
    """Standard weak BGe prior on n variables: a_w = n + 2, T = (1/2) I
    (Kuipers, Moffa & Heckerman 2014, supplement Eq. B.6; bnlearn defaults)."""
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


def _bge_log_marginal(YY_idx, data, precomp, T, nu, a_mu, a_w, N, n):
    """
    log p(d^Y | m) for columns YY_idx, per formula (2). Submatrices of T and
    R are selected AFTER R is formed (the paper's correction). `precomp`
    caches R (it does not depend on the subset).
    """
    l = len(YY_idx)
    if l == 0:
        return 0.0
    R = precomp["R"]
    idx = np.asarray(YY_idx, dtype=int)
    T_YY = T[np.ix_(idx, idx)]
    R_YY = R[np.ix_(idx, idx)]
    _, logdet_T = np.linalg.slogdet(T_YY)
    _, logdet_R = np.linalg.slogdet(R_YY)
    e_T = (a_w - n + l) / 2.0
    e_R = (N + a_w - n + l) / 2.0
    return float(
        (l / 2.0) * (np.log(a_mu) - np.log(N + a_mu))
        + multigammaln((N + a_w - n + l) / 2.0, l)
        - (l * N / 2.0) * np.log(np.pi)
        - multigammaln((a_w - n + l) / 2.0, l)
        + e_T * logdet_T
        - e_R * logdet_R
    )


def precompute_R(data, T, nu, a_mu, N):
    """R = T + S_N + (N a_mu/(N+a_mu)) (nu - xbar)(nu - xbar)^T."""
    xbar = data.mean(axis=0)
    centered = data - xbar
    S_N = centered.T @ centered
    diff = (nu - xbar).reshape(-1, 1)
    R = T + S_N + (N * a_mu / (N + a_mu)) * (diff @ diff.T)
    return {"R": R}


class BGeScorer:
    """
    Holds the full standardized data matrix (outcome + all covariate columns)
    and scores node-conditional likelihoods p(Y | Z) via the BGe ratio.

    Column layout: the outcome Y is column 0; each covariate group occupies a
    contiguous block of columns (nominal dummy-blocks span k = L-1 columns).
    A "parent set" is a set of GROUP names; scoring expands each group to its
    columns.
    """
    def __init__(self, groups, y_z, a_mu=A_MU, a_w=A_W, t_scale=T_SCALE):
        self.group_names = list(groups.keys())
        # build the joint data matrix: column 0 = Y, then each group's columns
        cols = [y_z.reshape(-1, 1)]
        self.group_cols = {}       # group -> list of column indices in `data`
        c = 1
        for g in self.group_names:
            block = groups[g]
            k = block.shape[1]
            self.group_cols[g] = list(range(c, c + k))
            cols.append(block)
            c += k
        self.data = np.hstack(cols)             # (N, n_total)
        self.N, self.n = self.data.shape
        self.Y_COL = 0

        self.T, self.nu, self.a_mu, self.a_w = default_hyperparams(
            self.n, a_mu=a_mu, a_w=a_w, t_scale=t_scale)
        self.precomp = precompute_R(self.data, self.T, self.nu, self.a_mu, self.N)

    def _cols_of(self, groups):
        idx = []
        for g in groups:
            idx.extend(self.group_cols[g])
        return idx

    def log_marginal(self, col_idx):
        return _bge_log_marginal(col_idx, self.data, self.precomp,
                                 self.T, self.nu, self.a_mu, self.a_w,
                                 self.N, self.n)

    def score(self, parent_groups):
        """BGe local score  log p(Y | parents) = log p(parents u Y) - log p(parents)."""
        par_cols = self._cols_of(parent_groups)
        fam_cols = par_cols + [self.Y_COL]
        return self.log_marginal(fam_cols) - self.log_marginal(par_cols)


# ===========================================================================
# 4. Subset search over covariate GROUPS
# ===========================================================================

def all_subsets(names):
    out = []
    for r in range(len(names) + 1):
        out.extend(itertools.combinations(names, r))
    return out


def exhaustive_search(scorer, verbose=True):
    names = scorer.group_names
    rows = []
    for Z in all_subsets(names):
        rows.append({"Z": ", ".join(Z) if Z else "(empty)",
                     "score_Y_given_Z": scorer.score(list(Z))})
    dfres = pd.DataFrame(rows).sort_values("score_Y_given_Z", ascending=False)
    # normalized weights over all scored subsets (softmax of local scores)
    s = dfres["score_Y_given_Z"].values
    w = np.exp(s - s.max()); w /= w.sum()
    dfres["weight"] = w
    if verbose:
        print(f"\nScored {len(rows)} subsets (exhaustive).")
    return dfres.reset_index(drop=True)


def greedy_forward_search(scorer, threshold=GREEDY_THRESHOLD, verbose=True):
    names = list(scorer.group_names)
    scores = {}
    base = tuple()
    scores[base] = scorer.score(list(base))
    frontier = [base]
    n_eval = 1

    for layer in range(len(names)):
        candidates = set()
        for subset in frontier:
            used = set(subset)
            for v in names:
                if v not in used:
                    candidates.add(tuple(sorted(used | {v})))
        new_frontier = []
        for Z in candidates:
            if Z in scores:
                continue
            scores[Z] = scorer.score(list(Z))
            n_eval += 1
            new_frontier.append(Z)
        if not new_frontier:
            break
        # normalized weights over all scored so far -> prune
        s = np.array(list(scores.values()))
        w = np.exp(s - s.max()); w /= w.sum()
        wmap = {z: wi for z, wi in zip(scores.keys(), w)}
        frontier = [z for z in new_frontier if wmap[z] > threshold]
        if verbose:
            print(f"Layer {layer+1}: +{len(candidates)} candidates "
                  f"({n_eval} total), {len(frontier)} pass threshold.")
        if not frontier:
            break

    s = np.array(list(scores.values()))
    w = np.exp(s - s.max()); w /= w.sum()
    dfres = pd.DataFrame({
        "Z": [", ".join(z) if z else "(empty)" for z in scores],
        "score_Y_given_Z": list(scores.values()),
        "weight": w,
    }).sort_values("score_Y_given_Z", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"\nEvaluated {n_eval} subsets (vs {2**len(names)} exhaustive).")
    return dfres


# ===========================================================================
# 5. OLS diagnostic (independent frequentist cross-check)
# ===========================================================================

def run_ols_diagnostic(scorer, df):
    """
    Plain OLS of the standardized outcome on the FULL standardized design
    (all covariate columns). Independent of the BGe prior -- a cross-check:
    a variable significant here but absent from the top BGe subset is worth
    a second look.
    """
    names = scorer.group_names
    X_slopes = np.hstack([scorer.data[:, scorer.group_cols[g]] for g in names])
    N = X_slopes.shape[0]
    X = np.hstack([np.ones((N, 1)), X_slopes])
    # column labels expanded per dummy
    col_names = ["const"]
    for g in names:
        k = len(scorer.group_cols[g])
        col_names += ([g] if k == 1 else [f"{g}[{j}]" for j in range(k)])
    y = scorer.data[:, scorer.Y_COL]

    p = X.shape[1]
    dof = N - p
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    rss = float(resid @ resid)
    sigma2 = rss / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stats = beta / se
    pvals = np.array([_two_sided_p_value(t, dof) for t in t_stats])
    r2 = 1.0 - rss / float(((y - y.mean()) ** 2).sum())

    print("\n" + "=" * 70)
    print("OLS DIAGNOSTIC (standardized outcome ~ full standardized design)")
    print("=" * 70)
    out = pd.DataFrame({"variable": col_names, "coef": beta, "std_err": se,
                        "t_stat": t_stats, "p_value": pvals})
    out["sig(p<0.05)"] = out["p_value"] < 0.05
    pd.set_option("display.width", 100)
    print(out.to_string(index=False))
    print(f"\nR-squared: {r2:.4f}   (n={N}, dof={dof})")
    return out


# ===========================================================================
# 6. Main
# ===========================================================================

def main():
    df = load_data()
    groups, y_z = build_design(df, encoding=CATEGORICAL_ENCODING)
    scorer = BGeScorer(groups, y_z, a_mu=A_MU, a_w=A_W, t_scale=T_SCALE)

    print(f"Score      : BGe node score  p(Y | Z)  [Kuipers-Moffa-Heckerman 2014]")
    print(f"Encoding   : {CATEGORICAL_ENCODING}")
    print(f"Search     : {SEARCH_MODE}")
    print(f"Covariate groups ({len(scorer.group_names)}): {scorer.group_names}")
    print(f"Joint Gaussian dimension n = {scorer.n} (Y + {scorer.n-1} covariate cols), N = {scorer.N}")
    print(f"Prior: Normal-Wishart  a_mu={scorer.a_mu}, a_w={scorer.a_w}, "
          f"T={scorer.T[0,0]:.4f}*I")

    if RUN_OLS_DIAGNOSTIC:
        run_ols_diagnostic(scorer, df)

    if SEARCH_MODE == "exhaustive":
        res = exhaustive_search(scorer)
    elif SEARCH_MODE == "greedy":
        res = greedy_forward_search(scorer, threshold=GREEDY_THRESHOLD)
    else:
        raise ValueError(f"Unknown SEARCH_MODE: {SEARCH_MODE}")

    pd.set_option("display.width", 100)
    print("\n=== BGe subset ranking (higher score_Y_given_Z is better) ===")
    print(res.to_string(index=False))
    print("\nBest subset:", res.iloc[0].to_dict())
    return res


if __name__ == "__main__":
    main()
