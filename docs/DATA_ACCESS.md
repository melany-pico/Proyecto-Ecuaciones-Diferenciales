# Dataset acquisition notes

## Synthetic systems
No web required. `01_generate_data.py` creates ground-truth trajectories and noisy observations.

## ERA5
1. Register at https://cds.climate.copernicus.eu/ and accept terms for `reanalysis-era5-single-levels`.
2. Install/configure the CDS API credentials according to https://cds.climate.copernicus.eu/how-to-api (typically `%USERPROFILE%\.cdsapirc` on Windows).
3. Run `python scripts/07_download_era5.py`.
4. If API access fails, use the CDS dataset download interface with the same region/date/variables and put `.nc` files in `data/external/era5/`.
5. Relative humidity can be derived from temperature/dew point with an explicitly documented formula; do not label dew point as humidity.

## CAMS
Browse https://ads.atmosphere.copernicus.eu/ ; choose a specific product, approve license and obtain API access. Save NetCDF/GRIB files to `data/external/cams/`. The placeholder script does not download data automatically yet.

## UCR
Visit https://www.timeseriesclassification.com/ and put downloaded archives/TS files under `data/external/ucr/`. Classification samples generally are not interchangeable with physical ODE trajectories. A dedicated adapter is required.

## OpenML
`06_download_openml.py` tests access using Iris (not used in dynamical discovery). Select an appropriate time-series dataset and implement a timestamp-aware adapter before using it in experiments.
