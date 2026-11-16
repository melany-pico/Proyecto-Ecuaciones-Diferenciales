# Research protocol roadmap

**Research question:** When can an ODE vector field learned from sparse/noisy trajectories be distilled into compact governing equations that generalize to unseen initial conditions?

**Hypotheses:** H1: neural-to-symbolic outperforms direct finite-difference SINDy in selected noisy regimes; H2: increased trajectory coverage improves structural recovery; H3: low symbolic complexity can be achieved without unacceptable out-of-distribution degradation.

**Benchmarks:** Lorenz-63, Duffing (autonomous unforced), Van der Pol, Lotka–Volterra. Later: forced Duffing, partially observed dynamics, additional systems, weather with exogenous drivers.

**Baselines to implement:** direct SINDy, smoothed derivative SINDy, Neural ODE without distillation, optionally weak-form SINDy, PySR, and a known physics model.

**Primary metrics:** symbolic support F1, coefficient error (aligned libraries), vector-field RMSE, short/long horizon trajectory error, number of nonzero terms, numerical stability, runtime. Report seeds and confidence intervals.

**Design:** trajectory-level holdout, multiple initial conditions, noise and irregular sampling sweeps, ablations (network capacity, distillation sampling, sparsity threshold, library order), extrapolation tests.

**Weather caveat:** ERA5/CAMS are gridded observational/reanalysis products, not known closed-form autonomous ODE ground truth. Treat as empirical predictive/interpretability case studies; use forcing and nonstationarity checks.

**Publication:** preregister comparisons, freeze test sets, log versions and seeds, document failure cases, and compare against current peer-reviewed literature before claiming novelty.
