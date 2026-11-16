# Neural ODE → Symbolic Equations | Research scaffold v0.1

## Setup (Windows Anaconda Prompt)
```bat
conda env create -f environment.yml
conda activate dynadistill
python -m pytest tests -q
python scripts/run_all.py
```
Run from the extracted **project root**. CPU is the default; no GPU is required. Initial training is a **smoke test**, not a validated scientific result.

## Run scripts individually (in this order)
```bat
python scripts/01_generate_data.py
python scripts/02_train_neural_ode.py
python scripts/03_distill_equations.py
python scripts/04_evaluate.py
python scripts/05_plot_results.py
```
Optional (requires Internet):
```bat
python scripts/06_download_openml.py
python scripts/07_download_era5.py
python scripts/08_download_cams.py
```
ERA5 needs CDS API access and accepted license terms. CAMS currently prints setup guidance. UCR datasets and meteorological NetCDF files are **not yet connected to training**. Do not treat OpenML Iris as a dynamical benchmark.

## Changing systems
Edit `configs/default.yml`: `system` = `lorenz`, `duffing`, `vanderpol`, or `lotka_volterra`, then rerun scripts 01–05. Separate outputs are named by system. Adjust training epochs, polynomial degree and threshold after the smoke test.

## Methodological caveats
- Baseline Neural ODE uses autonomous fully observed state and short-segment trajectory loss.
- The sparse stage uses polynomial STLSQ fitted to Neural ODE derivatives, not measured finite differences.
- Train/validation/test are split by **whole trajectories**. Validation exists but early stopping is not implemented yet.
- Test metrics use clean simulated ground truth; noisy observations are inputs to training only.
- Coefficient recovery, support precision/recall, noise sweeps, multiple seeds, SINDy direct baseline, uncertainty intervals and ERA5/CAMS integration are future work.
- Chaotic Lorenz long-horizon MSE can be large even for useful models; use short-horizon and invariant-based evaluations in the full study.
- Recovered coefficients are in original state units, but optimization can be sensitive to state scale.

## Data folders
`data/raw/` original raw data; `data/processed/` simulated NPZ splits; `data/external/openml/`, `era5/`, `cams/` downloaded or manually supplied datasets.

## Reproducibility
Record package versions (`conda env export > results/environment_resolved.yml`), seeds, platform and hardware for each experiment. Version 0.1 is an executable research foundation, not a Q1-ready implementation.
