import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dynadistill.utils.common import ROOT
out=ROOT/'data/external/era5';out.mkdir(parents=True,exist_ok=True)
try:
 import cdsapi
 target=out/'era5_sample_2023_01.nc'
 client=cdsapi.Client()
 client.retrieve('reanalysis-era5-single-levels',{'product_type':['reanalysis'],'variable':['2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind','surface_pressure','2m_dewpoint_temperature'],'year':['2023'],'month':['01'],'day':['01','02'],'time':['00:00','06:00','12:00','18:00'],'area':[1.0,-79.0,0.0,-77.0],'data_format':'netcdf','download_format':'unarchived'},str(target))
 print('Downloaded:',target)
except Exception as e:
 print('ERA5 request could not finish:',e)
 print('Configure your CDS personal access token and accept the dataset terms at https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels')
 print('Manual NetCDF location: data/external/era5/')
