import pandas as pd
import xarray as xr
import numpy as np

from utils.netcdf import load_copernicus_ammonia
from utils.plot import plot_feature
from utils.regressors import time_to_int, classify_surface, add_seasons, get_value_at_location_and_time
from utils.regressors import add_time_variant_regressor
from utils.regressors import add_time_invariant_regressor
from utils.regressors import add_registry_information
from utils.regressors import add_one_hot_encoded_regressors
from utils.regressors import add_dummy_regressor

# configuration
start, end = '2016-01-01', '2020-12-31'
time_slice = slice(start, end)

lat_min, lat_max = 44.75, 46.65
lat_slice = slice(lat_min, lat_max)

lon_min, lon_max = 8.5, 11.25
lon_slice = slice(lon_min, lon_max)

bounds = [[lat_min, lat_max], [lon_min, lon_max]]

units = {
    'Wind_u': 'm/s',
    'Altitude': 'm',
    'Wind_v': 'm/s',
    'Wind Speed': 'm/s',
    'Dewpoint': '°C',
    'Temperature': '°C',
    'TP': 'm',
    'NH3': 'Tg/yr',
    'BLH': 'm'
}

# response variable
pollutant = pd.read_csv('./data/arpa_pollutants/NO2.csv', index_col=0)

# pollutant stations registry
df, registry = add_registry_information(pollutant, './data/arpa_registry/registry.csv', 'NO2', start, end)

# time
df['Time'] = df['Date'].apply(time_to_int)

# ammonia
nh3 = load_copernicus_ammonia(['ags', 'agl'], time_slice, lat_slice, lon_slice)
df = add_time_variant_regressor(df, nh3, registry, 'NH3', start, end)
plot_feature(nh3, df, feature='NH3', loc=(9.5, 45.5), label='NH₃', save=True, unit=units['NH3'])

# altitude
altitude = xr.load_dataset('./data/DEM_Lombardia_100mx100m.nc').altitude
df = add_time_invariant_regressor(df, altitude, registry, 'Altitude', )
plot_feature(altitude, df, feature='Altitude', loc=(9.5, 45.5), label='Altitude', save=True, unit=units['Altitude'],
             time_variant=False, with_boundary=False)

# temperature
temp = xr.open_dataset('./data/copernicus/temperature_2m/ERA5-LAND_0.1x0.1_temperature_daily.nc').t2m
df = add_time_variant_regressor(df, temp, registry, 'Temperature', start, end)
plot_feature(temp, df, feature='Temperature', loc=(9.5, 45.5), label='Temperature', save=True, unit=units['Temperature'])

# boundary layer height
blh = xr.open_dataset('./data/copernicus/boundary_layer_height/ERA5-0.25x0.25_boundary_layer_heigth_daily.nc').blh
df = add_time_variant_regressor(df, blh, registry, 'BLH', start, end)
plot_feature(blh, df, feature='BLH', loc=(9.5, 45.5), label='Boundary Layer Height', save=True, unit=units['BLH'])

# total precipitation
tp = xr.open_dataset('./data/copernicus/total_precipitation/ERA5-LAND_0.1x0.1_total_precipitation_daily.nc').tp
df = add_time_variant_regressor(df, tp, registry, 'TP', start, end)
plot_feature(tp, df, feature='TP', loc=(9.5, 45.5), label='Total Precipitation', save=True, unit=units['TP'])

quantile_25 = df['TP'].quantile(.25)
df['Binary TP'] = df['TP'].apply(lambda val: 1 if val > quantile_25 else 0)

# 2 meter dewpoint
d2m = xr.open_dataset('./data/copernicus/dewpoint/ERA5-LAND_0.1x0.1_2m_dewpoint_daily.nc').d2m
d2m = d2m - 273.15
df = add_time_variant_regressor(df, d2m, registry, 'Dewpoint', start, end)
plot_feature(d2m, df, feature='Dewpoint', loc=(9.5, 45.5), label='Dew Point', save=True, unit=units['Dewpoint'])

# wind 10m u-component
wind_u = xr.open_dataset('./data/copernicus/wind_10m_u/ERA5-LAND_0.1x0.1_wind_10m_u_daily.nc').u10
df = add_time_variant_regressor(df, wind_u, registry, 'Wind_u', start, end)
plot_feature(wind_u, df, feature='Wind_u', loc=(9.5, 45.5), label='Wind u-component', save=True, unit=units['Wind_u'])

# wind 10m v-component
wind_v = xr.open_dataset('./data/copernicus/wind_10m_v/ERA5-LAND_0.1x0.1_wind_10m_v_daily.nc').v10
df = add_time_variant_regressor(df, wind_v, registry, 'Wind_v', start, end)
plot_feature(wind_v, df, feature='Wind_v', loc=(9.5, 45.5), label='Wind v-component', save=True, unit=units['Wind_v'])

# wind velocity
wind_speed = np.sqrt(np.power(wind_u, 2) + np.power(wind_v, 2))
df['Wind Speed'] = np.sqrt(np.power(df['Wind_u'], 2) + np.power(df['Wind_v'], 2))
plot_feature(wind_speed, df, feature='Wind Speed', loc=(9.5, 45.5), label='Wind Speed', save=True, unit=units['Wind Speed'])

# surface class
df, classes = classify_surface(df, encoded=True)

# stations tipology
df, encoded_regressors = add_one_hot_encoded_regressors(df, 'Typology', prefix='Type')

# dummy is_weekend
# df = add_dummy_regressor(df, 'Weekend', lambda row: int(pd.to_datetime(row['Date']).dayofweek > 4))
lockdown_start = pd.to_datetime('9 March 2020')
lockdown_end = pd.to_datetime('18 May 2020')
df = add_dummy_regressor(df, 'Lockdown', lambda row: int(lockdown_start < pd.to_datetime(row['Date']) < lockdown_end))

# seasons
df, seasons = add_seasons(df)

# data cleaning
df = df.dropna()
df = df[df['NO2'] > 0]

# save
df.to_csv('./data/learning_dataset.csv')
