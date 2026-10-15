# FindIMB for continuous outcomes (BGe score)

FindIMB (Triantafillou, Jabbari & Cooper, 2021) combines observational and experimental data to predict an outcome under intervention. It pools the observational data only where they agree with the experiment.

This repository implements FindIMB for a **continuous outcome**. The Dirichlet scores of the discrete version are replaced by closed-form Normal–Wishart (BGe) scores (Kuipers, Moffa & Heckerman, 2014).

## Files

- **`Find_MarkovBoundary.py`**: finds the Markov boundary of the outcome `Y` in the observational data.
- **`FindIMB_Gaussian.py`**: for each conditioning set `{X} u Z`, where `X` is the treatment and `Z` a subset of the other Markov boundary covariates, scores two hypotheses:
  - `H^c`: the observational and experimental data share `P(Y | X, Z)`, so they can be pooled;
  - `H^cbar`: they do not.

  It then predicts `Y` by averaging over all sets and both hypotheses.

## Prior

The variables are standardized, and the prior is the weakly informative Normal–Wishart

```
nu = 0,  a_mu = 1,  a_w = n + 1,  T = 2 I
```

Under this prior every correlation is uniform a priori and each variable's precision has prior mean 1.


## Output

- **Step 1** prints:
  - every scored covariate set ranked by `log p(Y | Z)`, with its normalized weight (the top set is the Markov boundary);

- **Step 2** prints, for every set `Z`:
  - the log scores of `H^c` and `H^cbar`;
  - their posterior probabilities;
  - the total posterior probability of pooling;
  - the most probable set to pool on.

  If a test file is given, it also predicts `Y` for the new cases and reports the RMSE and the log predictive density.

Both scripts return the results table as a pandas DataFrame.
