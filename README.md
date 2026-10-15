# FindIMB with the BGe score for a continuous outcome

Python code for FindIMB (Triantafillou, Jabbari & Cooper, 2021) with a continuous outcome. The Dirichlet (BDe) scores of the discrete version are replaced by closed-form Normal–Wishart (BGe) scores (Geiger & Heckerman, 1994; Kuipers, Moffa & Heckerman, 2014).

## Files

| File | What it does |
|---|---|
| `FindIMB_BGeScore_ContinuousOutcome.py` | Scores candidate covariate sets `Z` with the BGe local score `log p(Y \| Z)` on the observational data `D_o`, and searches over `Z` (greedy or exhaustive). Optional OLS diagnostic. |
| `FindIMB_BGe_ExperimentalStep.py` | Always conditions on the treatment `X`, plus a subset `Z` of the other Markov boundary covariates. For every such `Z`, it scores the hypotheses `H_Z^c` (`D_o` and the experimental data `D_e` share `P(Y \| X, Z)`) and `H_Z^cbar` (they do not), computes their posterior probabilities, and predicts `Y` for new cases by Bayesian model averaging. |

## Method

**Prediction.** Posterior predictive densities are Student-t. The prediction averages over all subsets `Z` and both hypotheses:

```
p(y | x, MB, D) = sum_Z  P(H_Z^c | D) p(y | x, Z, D_o u D_e)  +  P(H_Z^cbar | D) p(y | x, Z, D_e)
```

**Assumption.** BGe assumes all variables (outcome and covariates) are jointly Gaussian.

## Prior

All columns are standardized to mean 0 and variance 1. In the experimental step the standardization uses the pooled `D_o u D_e` mean and sd. With `n` the total number of columns (each dummy counted once):

```
mu | W ~ N(nu, (a_mu W)^{-1}),   W ~ Wishart(a_w, T)
nu = 0,   a_mu = 1,   a_w = n + 1,   T = t I,   t = 2
```

- `a_mu = 1`: one imaginary prior case for the mean.
- `a_w = n + 1`: every marginal correlation is uniform on (-1, 1) a priori (Barnard, McCulloch & Meng, 2000).
- `t = 2`: each variable's precision has prior mean `2/t = 1`, the precision of standardized data.
- Since `a_w - n` and `t` do not depend on `n`, the score of a candidate set does not depend on the number of variables in the domain.

All of these can be changed with `A_MU`, `A_W` and `T_SCALE` in the config block of each file.

## Requirements

Python 3 with `numpy`, `pandas` and `scipy`:

```
pip install numpy pandas scipy
```

## Usage

**Step 1: Markov boundary.** In the config block of `FindIMB_BGeScore_ContinuousOutcome.py`, set:
- `DATA_PATH`: the observational data;
- `OUTCOME_COL`;
- the covariates by type (`NOMINAL_COLS`, `BINARY_COLS`, `CONTINUOUS_COLS`), including the treatment.

Then run:

```
python FindIMB_BGeScore_ContinuousOutcome.py
```

It prints the subsets ranked by `log p(Y | Z)`, with normalized weights. The top subset is the estimated Markov boundary.

**Step 2: Experimental step.** In the config block of `FindIMB_BGe_ExperimentalStep.py`, set:
- `OBS_PATH` and `EXP_PATH`: the two datasets, with the same columns;
- optionally `TEST_PATH`: new cases to predict;
- `OUTCOME_COL`, `TREATMENT_COL` and `TREATMENT_TYPE`;
- the Markov boundary covariates other than `X` (`MB_NOMINAL_COLS`, `MB_BINARY_COLS`, `MB_CONTINUOUS_COLS`). `X` is set by `TREATMENT_COL` and is always included.

Then run:

```
python FindIMB_BGe_ExperimentalStep.py
```

It prints, for every subset `Z`:
- the log scores of both hypotheses;
- the posterior probabilities `P(H_Z^c | D)` and `P(H_Z^cbar | D)`;
- `P(H_Z^c | D, Z)`.

It also prints the total posterior mass on pooling. If a test file is given, it reports the test RMSE and the mean log predictive density of the model-averaged prediction.

**Encoding.** Nominal variables are dummy-coded, and each dummy block is added to or removed from a set as one unit. Reference levels (`NOMINAL_REF`) and positive values of binary variables (`BINARY_POSITIVE`) can be set in the config block. An outcome with skew > 1 is log1p-transformed, and predictions are returned on the original scale.

## References

- Barnard, J., McCulloch, R. and Meng, X.-L. (2000). Modeling covariance matrices in terms of standard deviations and correlations, with application to shrinkage. *Statistica Sinica* 10, 1281–1311.
- Geiger, D. and Heckerman, D. (1994). Learning Gaussian networks. In *Proceedings of UAI*.
- Kuipers, J., Moffa, G. and Heckerman, D. (2014). Addendum on the scoring of Gaussian directed acyclic graphical models. *The Annals of Statistics* 42, 1689–1691.
- Triantafillou, S., Jabbari, F. and Cooper, G. F. (2021). Causal Markov boundaries. In *Proceedings of UAI*, PMLR 161.
