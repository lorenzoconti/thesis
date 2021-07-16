import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np

from scipy import interpolate

from utils.netcdf import load_copernicus_ammonia
from utils.regressors import time_to_int, classify_surface, add_dummy_regressor, \
    add_seasons, get_value_at_location, get_value_at_time

# configuration
verbose = True

start, end = '2016-01-01', '2020-12-31'
time_slice = slice(start, end)

lat_min, lat_max = 45, 46
lat_slice = slice(lat_min, lat_max)

lon_min, lon_max = 9.05, 10.45
lon_slice = slice(lon_min, lon_max)

kriging_resolution = 250

# kriging_dates
# Winter: 15 January 2020
# Spring: 15 April   2020
# Summer: 15 July    2020
# Fall  : 15 October 2020
kriging_date_string = '15 January 2020'
kriging_date = pd.to_datetime(kriging_date_string)

lats = np.linspace(lat_min, lat_max, kriging_resolution)
lons = np.linspace(lon_min, lon_max, kriging_resolution)
lats_mesh, lons_mesh = np.meshgrid(lats, lons)
lats_mesh, lons_mesh = lats_mesh.reshape(-1), lons_mesh.reshape(-1)

df = pd.DataFrame({'Latitude': lats_mesh, 'Longitude': lons_mesh})

df['Date'] = kriging_date
df['Time'] = time_to_int(kriging_date)

# ammonia
nh3 = load_copernicus_ammonia(['ags', 'agl'], time_slice, lat_slice, lon_slice)

# altitude
altitude = xr.load_dataset('./data/DEM_Lombardia_100mx100m.nc').altitude

# temperature
temp = xr.open_dataset('./data/copernicus/temperature_2m/ERA5-LAND_0.1x0.1_temperature_daily.nc').t2m

# boundary layer height
blh = xr.open_dataset('./data/copernicus/boundary_layer_height/ERA5-0.25x0.25_boundary_layer_heigth_daily.nc').blh

# total precipitation
tp = xr.open_dataset('./data/copernicus/total_precipitation/ERA5-LAND_0.1x0.1_total_precipitation_daily.nc').tp

# 2 meter dewpoint
d2m = xr.open_dataset('./data/copernicus/dewpoint/ERA5-LAND_0.1x0.1_2m_dewpoint_daily.nc').d2m

# wind 10m u-component
wind_u = xr.open_dataset('./data/copernicus/wind_10m_u/ERA5-LAND_0.1x0.1_wind_10m_u_daily.nc').u10

# wind 10m v-component
wind_v = xr.open_dataset('./data/copernicus/wind_10m_v/ERA5-LAND_0.1x0.1_wind_10m_v_daily.nc').v10

feature_names = ['NH3', 'Temperature', 'BLH', 'TP', 'Dewpoint', 'Wind_u', 'Wind_v']
feature_values = [nh3, temp, blh, tp, d2m, wind_u, wind_v]

assert len(feature_values) == len(feature_names)

for name, value in zip(feature_names, feature_values):
    if verbose:
        print(f'Adding {name}')

    netcdf_lons = np.array(value.lon.values)
    netcdf_lats = np.array(value.lat.values)
    netcdf_data = np.array(get_value_at_time(value, kriging_date))

    f = interpolate.interp2d(netcdf_lons, netcdf_lats, netcdf_data, kind='linear')
    upsampled_data = f(lons, lats)

    ds = xr.DataArray(data=upsampled_data,
                      dims=["lat", "lon"],
                      coords=dict(lon=lons, lat=lats))

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    value.sel(time=kriging_date, method='nearest').plot(ax=axs[0], add_colorbar=False)
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    axs[0].set_title('')

    ds.plot(ax=axs[1], add_colorbar=False)
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')

    plt.show()

    df[name] = df[['Latitude', 'Longitude']].apply(lambda row:
                                                   get_value_at_location(
                                                               ds,
                                                               lat=row['Latitude'],
                                                               lon=row['Longitude']),
                                                   axis=1)

netcdf_lons = np.array(altitude.lon.values)
netcdf_lats = np.array(altitude.lat.values)
netcdf_data = np.array(altitude.values)

f = interpolate.interp2d(netcdf_lons, netcdf_lats, netcdf_data, kind='linear')
upsampled_data = f(lons, lats)
df['Altitude'] = df[['Latitude', 'Longitude', 'Date']].apply(lambda row:
                                                             get_value_at_location(
                                                                 altitude,
                                                                 lat=row['Latitude'],
                                                                 lon=row['Longitude']),
                                                             axis=1)

# wind speed
df['Wind Speed'] = np.sqrt(np.power(df['Wind_u'], 2) + np.power(df['Wind_v'], 2))

temp = df.copy()

# surface class
df, classes = classify_surface(df, encoded=True, how='left')

# dummy is_weekend
df = add_dummy_regressor(df, 'Weekend', lambda row: int(pd.to_datetime(row['Date']).dayofweek > 4))

# seasons
df, seasons = add_seasons(df)

for s in seasons:
    if s not in df.columns:
        df[s] = 0


# modify total precipitations to categorical
quantile_25 = df['TP'].quantile(.25)
df['Binary TP'] = df['TP'].apply(lambda val: 1 if val > quantile_25 else 0)

# adjust dewpoing measure unit
df['Dewpoint'] = df['Dewpoint'] - 273.15

# data cleaning
df.to_csv(f'data/kriging_dataset_{kriging_date_string.replace(" ", "_")}_250x250_interpolated.csv')
