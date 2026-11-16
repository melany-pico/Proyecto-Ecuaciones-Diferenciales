from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'data/external/cams';out.mkdir(parents=True,exist_ok=True)
print('CAMS requires selecting a specific atmospheric product, variables, region and API credentials.')
print('Catalog: https://ads.atmosphere.copernicus.eu/ ; place downloaded NetCDF/GRIB in:',out)
print('No automatic request is made until product-specific parameters are validated.')
