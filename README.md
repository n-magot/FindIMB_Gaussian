# FindIMB with the BGe score for a continuous outcome

Python code for FindIMB (Triantafillou, Jabbari & Cooper, 2021) with a continuous outcome. The Dirichlet (BDe) scores of the discrete version are replaced by closed-form Normal–Wishart (BGe) scores (Geiger & Heckerman, 1994; Kuipers, Moffa & Heckerman, 2014).

## Assumptions

The BGe score assumes that all variables (outcome, treatment and covariates) are jointly Gaussian, and that the rows are i.i.d. draws from this distribution. Binary and categorical variables satisfy this only approximately. What the scores use is that `Y`, given the treatment and the covariate set, follows a linear model with Gaussian noise of constant variance. The treatment and covariates enter only as regressors.

## Files

- **`Find_MarkovBoundary.py`**: finds the Markov boundary of the outcome `Y` in the observational data.
- **`FindIMB_Gaussian.py`**: for each conditioning set `{X} u Z`, where `X` is the treatment and `Z` a subset of the other Markov boundary covariates, scores two hypotheses:
  - `H^c`: the observational and experimental data share `P(Y | X, Z)`, so they can be pooled;
  - `H^cbar`: they do not.

  It then predicts `Y` by averaging over all sets and both hypotheses.

## Prior

The variables are standardized, and the prior is the weakly informative Normal–Wishart

```
nu = 0,  a_mu = 1,  a_w = n + 2,  T = (1/2) I
```

This is the standard weak BGe prior (Kuipers, Moffa & Heckerman, 2014, supplement Eq. B.6). It is centred on independence and on the scale of the standardized data, with `t = a_mu (a_w − n − 1) / (a_mu + 1) = 1/2` and the smallest integer prior sample sizes for which this calibration holds. It plays the role of the weak Dirichlet prior `a_jk = 1` of the discrete version, though it is derived from one joint prior rather than set per conditional. The values can be changed in the config block of each file.

## Output

- **Step 1** prints:
  - every scored covariate set ranked by `log p(Y | Z)`, with its normalized weight (the top set is the Markov boundary);

- **Step 2** prints, for every set `Z`:
  - the log scores of `H^c` and `H^cbar` and their posterior probabilities;
  - the total posterior probability of pooling;
  - the most probable set to pool on.

  If a test file is given, it also predicts `Y` for the new cases and reports the RMSE and the log predictive density.

Both scripts return the results table as a pandas DataFrame.
## References

- Geiger, D. and Heckerman, D. (1994). Learning Gaussian networks. In *Proceedings of UAI*.
- Kuipers, J., Moffa, G. and Heckerman, D. (2014). Addendum on the scoring of Gaussian directed acyclic graphical models. *The Annals of Statistics* 42, 1689–1691.
- Triantafillou, S., Jabbari, F. and Cooper, G. F. (2021). Causal Markov boundaries. In *Proceedings of UAI*, PMLR 161.
