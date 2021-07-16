import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal, ogr, osr
import numpy as np


def load_copernicus_ammonia(layers, time_slice, lat_slice, lon_slice, verbose=False):
    """ Load the ammonia netcdf dataset as a sum of different desired layers, one for each pollutant source.

    If [verbose] plots each dataset and the resulting sum on several heatmaps.

    [layers] is a list containing the pollutant's sources of interest.
    [time_slice, lat_slice, lon_slice] are used to filter the dataset in a specific area and in a specific time interval.
    """
    xr_layers = []

    if 'agl' in layers:
        xr_layers.append(xr.load_dataset(
            './data/copernicus/ammonia/CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_nh3_v4.2_monthly_agl.nc').agl.sel(
            time=time_slice, lat=lat_slice, lon=lon_slice))

    if 'ags' in layers:
        xr_layers.append(xr.load_dataset(
            './data/copernicus/ammonia/CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_nh3_v4.2_monthly_ags.nc').ags.sel(
            time=time_slice, lat=lat_slice, lon=lon_slice))

    nh3 = sum(xr_layers)
    nh3.name = 'nh3'

    if verbose:

        shape = gpd.read_file('./shp/lombardia/lombardia.shp').to_crs(epsg=4326)

        ncols = len(xr_layers) + 1
        fig, axs = plt.subplots(ncols=ncols, figsize=(8 * ncols, 5))

        for i in range(len(xr_layers)):
            shape.plot(ax=axs[i], color='black', alpha=0.5)
            xr_layers[i].mean(dim='time').plot(ax=axs[i], alpha=0.5)

        shape.plot(ax=axs[len(xr_layers)], color='black', alpha=0.5)
        nh3.mean(dim='time').plot(ax=axs[len(xr_layers)], alpha=0.5)

        plt.show()

    return nh3


def find_pixel(data, lat, lon, bounds, resolution=0.1):
    '''Find the grid area within a specific point is located and return the values corresponding to the grid area corners.'''

    lat_min, lat_max = bounds[0][0], bounds[0][1]
    lon_min, lon_max = bounds[1][0], bounds[1][1]

    assert (lat > lat_min and lat < lat_max and lon > lon_min and lon < lon_max)

    nearest = data.sel(lat=lat, lon=lon, method='nearest')

    if lat in data.lat.values and lon in data.lon.values:
        return [data.sel(lat=lat, lon=lon)]

    nearest_lat = round(float(nearest.lat.values), 2)
    nearest_lon = round(float(nearest.lon.values), 2)

    lat_min = nearest_lat - int(nearest_lat > lat) * resolution
    lat_max = nearest_lat + int(nearest_lat <= lat) * resolution
    lon_min = nearest_lon - int(nearest_lon > lon) * resolution
    lon_max = nearest_lon + int(nearest_lon <= lon) * resolution

    return [data.sel(lat=lat, lon=lon, method='nearest') for lat in [lat_min, lat_max] for lon in [lon_min, lon_max]]


def get_time_series_at_location(data, lat, lon, feature):
    """Return the closest time series to a specific point."""

    ts = data.sel(lat=lat, lon=lon, method='nearest', drop=True).to_series()
    index = ts.index.get_level_values('time')
    values = ts.values

    return pd.DataFrame({'Date': index.values, feature: values})


def geotiff_to_netcdf(path, epsg, download=False, output_path='', name='', non_negative=True, savefig=False,
                      fig_path=''):
    """Convert a raster geotiff image to a netcdf dataset."""

    ds = gdal.Open(path)

    # upper left, resolution and skew of x and y
    ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()

    # lower right x and y
    lrx = ulx + (ds.RasterXSize * xres)
    lry = uly + (ds.RasterYSize * yres)

    # setup the source projection - you can also import from epsg, proj4...
    source = osr.SpatialReference()
    source.ImportFromWkt(ds.GetProjection())

    # the target projection
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)

    # create the transform - this can be used repeatedly
    transform = osr.CoordinateTransformation(source, target)

    # transform the point. You can also create an ogr geometry and use the more generic `point.Transform()`
    uly, ulx, _ = transform.TransformPoint(ulx, uly)
    lry, lrx, _ = transform.TransformPoint(lrx, lry)

    lons = np.linspace(ulx, lrx, ds.RasterXSize)
    lats = np.linspace(lry, uly, ds.RasterYSize)

    data = ds.ReadAsArray()

    if non_negative: data[data < 0] = np.nan

    ds = xr.DataArray(np.flipud(data), coords={'lat': lats, 'lon': lons, }, dims=['lat', 'lon'])

    ds.name = name

    if download and len(output_path) > 0:
        ds.to_netcdf(output_path)

    if savefig and len(fig_path) > 0:
        fig, ax = plt.subplots(figsize=(15, 10))
        ds.plot(ax=ax)
        plt.savefig(fig_path)
        plt.close()

    return ds
