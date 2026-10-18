# Recursive forest profiles

Deterministic numerical checks and figure generation for fixed-generation
profiles in recursive forests with decaying immigration. The two parent-choice
rules favor newer vertices (`N`) or older vertices (`O`).

## Run

Tested with Python 3.12.14. Create a virtual environment and install the pinned
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_and_figures.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.
The program uses no network access after installation. It runs the checks,
generates four figures, and writes `verification_results.json`. A failed check
raises an assertion and terminates the run with a nonzero exit status.

## What is checked

| Calculation | Independent comparison |
| --- | --- |
| Finite-size means and covariances | Exhaustive enumeration of 82,768 weighted parent-choice configurations, 14 parameter settings and 84 size/parameter comparisons through size 7 |
| Scalar covariance systems | A separately expressed matrix recurrence |
| Critical newer-preference constants | Closed formulas and an 8-by-8 triangular recurrence |
| Subcritical covariance constants | 96 comparisons between integral representations and triangular systems |
| Root deletion | Exact rational polynomial identities and finite-size generating functions |
| General scale kernels | Discrete descendant path sums for three additional densities |
| Slow critical convergence | A harmonic-number identity and exact mean ratios at size 2,000,000 |

The computations are deterministic. Numerical checks support the algebraic
audit; they do not replace proofs of asymptotic statements. The general-kernel
checks concern descendant response, not a general profile central limit theorem.

## Generated files

The program creates these PDF vector figures and corresponding PNG previews in
`figures/`:

- `forest_growth_schematic`: roots, generations, immigration and attachment;
- `attachment_response`: limiting parent-choice densities and direct-descendant responses;
- `exact_mean_phases`: exact recurrence means in three immigration regimes;
- `limiting_correlations`: analytic correlation matrices.

The first figure is a schematic. The quantitative figures use exact deterministic
recurrences or analytic formulas, without Monte Carlo sampling. Generated files
are excluded from version control and can be recreated by the command above.

`verification_results.json` records the numerical residuals, finite-size mean
ratios and selected values used in the figures. Small floating-point differences
between platforms are possible. PDF metadata can differ between runs.

## Main functions

- `birth_matrix`, `moment_step`, `scalar_cov_step`: finite-size moment updates;
- `exact_means`: deterministic profile means;
- `exhaustive_checks`, `analytic_checks`, `appendix_generating_functions`:
  independent checks;
- `general_response_checks`, `critical_finite_size_checks`: response and
  finite-size convergence checks;
- `figures`: all four illustrations.

The parameters for each calculation are explicit in the script. NumPy, SciPy and
Matplotlib are the only external Python dependencies.

## Authors and license

Marek Gałązka and Hanna Wdowicka. Released under the MIT License; see `LICENSE`.
