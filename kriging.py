import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pickle
import xarray as xr
import geopandas as gpd

from model.gp import gp_create, gp_predict
from model.nn import nn_create, nn_predict
from utils.netcdf import load_copernicus_ammonia
from utils.plot import plot_grid

kriging_months = ['January', 'April', 'July', 'October']

# features
classes = ['Class_Agg_Urbano', 'Class_Rurale', 'Class_Urbano', 'Class_Montagna']
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
other_categoricals = ['Binary TP', 'Weekend']
to_be_scaled = ['NH3', 'Altitude', 'Temperature', 'Dewpoint', 'BLH', 'Wind_u', 'Wind_v', 'Wind Speed']

nn_features = to_be_scaled + classes + seasons + other_categoricals
gp_features = ['Longitude', 'Latitude', 'Time']
target = ['NO2']

scaler = pickle.load(open('output/scaler/scaler.save', 'rb'))

# nn model
nn_model = nn_create(NUM_FEATURES=len(nn_features), NUM_LAYERS=2, NUM_NEURONS=64)
nn_model.load_state_dict(torch.load('output/models_state_dict/nn_model.pth'))


def df_to_tensor(df, features):
    return torch.from_numpy(np.array(df[features].to_numpy())).float()


kriging_images = []
kriging_variances = []

for kmonth in kriging_months:
    # NO2 value placeholder
    kriging_date = f'15 {kmonth} 2020'
    # full_df = pd.read_csv(f'data/kriging_dataset_{kriging_date.replace(" ", "_")}_250x250_interpolated.csv',
    # index_col=0)
    full_df = pd.read_csv(f'data/kriging_dataset_{kriging_date.replace(" ", "_")}_250x250.csv',
                          index_col=0)
    full_df['NO2'] = -1

    df = full_df.reset_index().copy().dropna()

    # scale dataset
    df[to_be_scaled + target] = scaler.transform(df[to_be_scaled + target])

    # create dataset
    x_nn_t = df_to_tensor(df, nn_features)
    y_nn_t = df_to_tensor(df, target)
    x_gp_t = df_to_tensor(df, gp_features)

    nn_preds = nn_predict(x_nn_t, y_nn_t, nn_model, 'cuda:0', BATCH_SIZE=1024)
    nn_preds = np.array(nn_preds)
    y_nn = y_nn_t.numpy().reshape(-1)

    nn_errors = y_nn - nn_preds
    nn_errors_t = torch.from_numpy(nn_errors)

    gp_model, likelihood = gp_create(x_gp_t, torch.from_numpy(nn_preds), None, NU=1 / 2, INDUCING_POINTS=500)
    gp_model.load_state_dict(torch.load('output/models_state_dict/gp_model.pth'))

    gp_preds, gp_vars = gp_predict(x_gp_t, nn_errors_t, gp_model, 'cuda:0', BATCH_SIZE=512)

    gp_preds = np.array(gp_preds)
    gp_vars = np.array(gp_vars)

    df['NN'] = list(nn_preds)
    df['GP'] = list(gp_preds)
    df['NN+GP'] = list(nn_preds + gp_preds)
    df['NO2'] = list(nn_preds + gp_preds)
    df['Variance'] = list(gp_vars)

    df[to_be_scaled + target] = scaler.inverse_transform(df[to_be_scaled + target])

    j_df = pd.merge(full_df.reset_index(), df, on='index', how='outer')

    data = np.array(j_df['NO2_y'].values).reshape(250, -1)
    var = np.array(j_df['Variance'].values).reshape(250, -1)

    kriging_images.append(data)
    kriging_variances.append(var)

# clipping data
min_val = -10
max_val = 120
kriging_images = [np.clip(data, min_val, max_val) for data in kriging_images]

# plot distributions
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

for i, ax in enumerate(axs.flatten()):
    ax.set_title(kriging_months[i])
    sns.histplot(kriging_images[i].reshape(-1), ax=ax)

plt.show()

# set ticks and labels
lats = sorted(list(set(df['Latitude'].values)))
lons = sorted(list(set(df['Longitude'].values)))
min_lat, max_lat = round(np.min(lats), 2), round(np.max(lats), 2)
min_lon, max_lon = round(np.min(lons), 2), round(np.max(lons), 2)

num_ticks = 6
lat_ticks = [round(lat, 2) for lat in np.linspace(min_lat, max_lat, num_ticks)]
lon_ticks = [round(lon, 2) for lon in np.linspace(min_lon, max_lon, num_ticks)]

# plot seasons kriging
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
fig.subplots_adjust(wspace=0.1, hspace=0.2)

for i, ax in enumerate(axs.flatten()):

    ds = xr.DataArray(data=np.rot90(np.fliplr(kriging_images[i]), 1),
                      dims=["lat", "lon"],
                      coords=dict(lon=lons, lat=lats))

    img = ds.plot.pcolormesh(vmin=min_val, vmax=max_val, ax=ax, add_colorbar=False)
    ax.set_title(f'15 {kriging_months[i]} 2020')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

fig.colorbar(img, ax=axs, location='bottom', shrink=0.9, pad=0.05, label='NO₂ (µg/m³)')

plt.show()
# plt.savefig('output/kriging/seasons.png', bbox_inches='tight', pad_inches=0)


# plot seasons kriging variances
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
fig.subplots_adjust(wspace=0.1, hspace=0.2)

for i, ax in enumerate(axs.flatten()):

    ds = xr.DataArray(data=np.rot90(np.fliplr(kriging_variances[i]), 1),
                      dims=["lat", "lon"],
                      coords=dict(lon=lons, lat=lats))

    img = ds.plot.pcolormesh(ax=ax, add_colorbar=False)
    ax.set_title(f'15 {kriging_months[i]} 2020')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

fig.colorbar(img, ax=axs, location='bottom', shrink=0.9, pad=0.05, label='NO₂ (µg/m³)')
plt.show()

# January kriging analysis
ds = xr.DataArray(data=np.rot90(np.fliplr(kriging_images[0]), 1),
                  dims=["lat", "lon"],
                  coords=dict(lon=lons, lat=lats))

# load temp and blh datasets
temp = xr.load_dataset('data/copernicus/temperature_2m/ERA5-LAND_0.1x0.1_temperature_daily.nc').t2m
nh3 = load_copernicus_ammonia(['agl', 'ags'], time_slice=slice('2016-01-01', '2020-12-31'),
                              lat_slice=slice(44.75, 46.65), lon_slice=slice(8.5, 11.25))
blh = xr.load_dataset('data/copernicus/boundary_layer_height/ERA5-0.25x0.25_boundary_layer_heigth_daily.nc').blh

# plot grids over the kriging image
fig, ax = plt.subplots(ncols=1, figsize=(8, 8))

img = ds.plot.pcolormesh(vmin=min_val, vmax=max_val, ax=ax, add_colorbar=False)

plot_grid(nh3, ax, color='C0', label='NH₃ grid')
plot_grid(temp, ax, color='white', label='Other features grid')
plot_grid(blh, ax, color='red', label='BLH grid')
ax.legend(facecolor='lightgrey')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# plt.savefig(f'output/images/kriging/kriging_over_features_resolution.png', bbox_inches='tight', pad_inches=0)
plt.show()

# plot surface classification over kriging image
rename_dict = {
    'Agg_BG': 'Agg_Urbano',
    'Agg_BS': 'Agg_Urbano',
    'Agg_MI': 'Agg_Urbano',
    'A': 'Urbano',
    'B': 'Rurale',
    'C': 'Montagna',
    'D': 'Montagna'
}

zoning_shp = gpd.read_file('./shp/zoning/zone.shp').to_crs(epsg=4326)
zoning_shp.rename(columns={'COD_ZONA': 'Class'}, inplace=True)
zoning_shp.drop(['COD_ZONA2C'], axis=1, inplace=True)
zoning_shp = zoning_shp[~zoning_shp['Class'].isin(['D'])]

# plot
fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
img = ds.plot.pcolormesh(vmin=min_val, vmax=max_val, ax=ax, add_colorbar=False)
zoning_shp.boundary.plot(ax=ax, color='white')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# plt.savefig(f'output/kriging/kriging_over_surface_classification.png', bbox_inches='tight', pad_inches=0)
plt.show()

# plot bad predictions
corrected_image = kriging_images[0]
corrected_image[corrected_image < 0] = np.nan

ds = xr.DataArray(data=np.rot90(np.fliplr(corrected_image), 1),
                  dims=["lat", "lon"],
                  coords=dict(lon=lons, lat=lats))

fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
img = ds.plot.pcolormesh(vmin=0, vmax=max_val, ax=ax, add_colorbar=False)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# plt.savefig(f'output/kriging/bad_predictions.png', bbox_inches='tight', pad_inches=0)
plt.show()
