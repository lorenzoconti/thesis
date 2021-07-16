import xarray as xr

# settings
start = '2016-01-01'
end = '2020-12-31'
time_slice = slice(start, end)

lat_min, lat_max = 44.75, 46.65
lon_min, lon_max = 8.5, 11.25

lat_slice = slice(lat_min, lat_max)
lon_slice = slice(lon_min, lon_max)

# ERA5-Land Temperature: hourly data

temp = xr.open_dataset('./data/copernicus/ERA5-LAND_0.1x0.1_temperature_hourly.nc').t2m
temp = temp.reindex(latitude=list(reversed(temp.latitude)))
temp = temp.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
temp = temp.rename({'latitude': 'lat', 'longitude': 'lon'})
temp = temp.resample(time='D').mean()
temp = temp - 273.15

blh = xr.open_dataset('./data/copernicus/ERA5-0.1x0.1_boundary_layer_heigth_hourly.nc').blh
blh = blh.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
blh = blh.rename({'latitude': 'lat', 'longitude': 'lon'})
blh = blh.resample(time='D').mean()

tp = xr.open_dataset('./data/copernicus/ERA5-0.1x0.1_boundary_layer_heigth_hourly.nc').tp
tp = tp.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
tp = tp.rename({'latitude': 'lat', 'longitude': 'lon'})
tp = tp.resample(time='D').mean()

wind_u = xr.load_dataset('./data/copernicus/wind_10m_u/ERA5-LAND_0.1x0.1_wind_10m_u_hourly.nc').u10
wind_u = wind_u.reindex(latitude=list(reversed(wind_u.latitude)))
wind_u = wind_u.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
wind_u = wind_u.rename({'latitude': 'lat', 'longitude': 'lon'})
wind_u = wind_u.resample(time='D').mean()

wind_v = xr.load_dataset('./data/copernicus/wind_10m_v/ERA5-LAND_0.1x0.1_wind_10m_v_hourly.nc').v10
wind_v = wind_v.reindex(latitude=list(reversed(wind_v.latitude)))
wind_v = wind_v.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
wind_v = wind_v.rename({'latitude': 'lat', 'longitude': 'lon'})
wind_v = wind_v.resample(time='D').mean()

d2m = xr.load_dataset('./data/copernicus/dewpoint/ERA5-LAND_0.1x0.1_2m_dewpoint_hourly.nc').d2m
d2m = d2m.reindex(latitude=list(reversed(d2m.latitude)))
d2m = d2m.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
d2m = d2m.rename({'latitude': 'lat', 'longitude': 'lon'})
d2m = d2m.resample(time='D').mean()

tp = xr.load_dataset('./data/copernicus/total_precipitation/ERA5-LAND_0.1x0.1_total_precipitation_hourly.nc').tp
tp = tp.reindex(latitude=list(reversed(tp.latitude)))
tp = tp.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
tp = tp.rename({'latitude': 'lat', 'longitude': 'lon'})
tp = tp.resample(time='D').mean()