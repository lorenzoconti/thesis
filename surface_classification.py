import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.regressors import add_one_hot_encoded_regressors

plt.style.use('ggplot')

# reclassification
rename_dict = {
    'Agg_BG': 'Agg_Urbano',
    'Agg_BS': 'Agg_Urbano',
    'Agg_MI': 'Agg_Urbano',
    'A': 'Urbano',
    'B': 'Rurale',
    'C': 'Montagna',
    'D': 'Montagna'
}
# Lombardia
basemap_shp = gpd.read_file('./shp/lombardia/lombardia.shp').to_crs(epsg=4326)

# ARPA's zoning classification
zoning_shp = gpd.read_file('./shp/zoning/zone.shp').to_crs(epsg=4326)
zoning_shp.rename(columns={'COD_ZONA': 'Class'}, inplace=True)
zoning_shp['Class'].replace(rename_dict, inplace=True)
zoning_shp.drop(['COD_ZONA2C'], axis=1, inplace=True)
zoning_shp['Class'].replace(rename_dict, inplace=True)
zoning_shp, classes = add_one_hot_encoded_regressors(zoning_shp, 'Class', 'Class')

classes = [str.split(c, '_', 1)[1] for c in classes]

reclass_dict = {c: i+1 for i, c in enumerate(classes)}
zoning_shp['Level'] = zoning_shp['Class'].replace(reclass_dict)

# ARPA's NO2 data
df = pd.read_csv('data/learning_dataset.csv', index_col=0)
df = df[['NO2', 'IDStation', 'Latitude', 'Longitude', 'Typology', 'Altitude']]

mean_df = df.groupby(['IDStation', 'Typology', 'Latitude', 'Longitude']).mean('NO2').reset_index()

gdf = gpd.GeoDataFrame(mean_df, geometry=gpd.points_from_xy(mean_df['Longitude'], mean_df['Latitude'])).set_crs(4326)
full_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'])).set_crs(4326)

# spatial join
j_gdf = gpd.sjoin(gdf, zoning_shp, how='inner')
j_gdf.drop('index_right', axis=1, inplace=True)

full_j_gdf = gpd.sjoin(full_gdf, zoning_shp, how='inner')
full_j_gdf.drop('index_right', axis=1, inplace=True)
full_j_gdf.sort_values('Class', inplace=True)

# plot
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

cmap = plt.cm.get_cmap('cividis', 4)
# ax 0
basemap_shp.plot(ax=axs[0], alpha=0.2, color='grey')
j_gdf.plot('Class', ax=axs[0], cmap=cmap)

axs[0].set_ylabel('Latitude')
axs[0].set_xlabel('Longitude')

# ax 1
zoning_shp.plot('Class', ax=axs[1],  alpha=0.7, cmap=cmap, legend=True)
axs[1].set_ylabel('Latitude')
axs[1].set_xlabel('Longitude')

plt.show()

# plot
_, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 24))

typos = list(set(df['Typology']))
for i, ax in enumerate(axs.flatten()):
    coords = j_gdf[j_gdf['Typology'] == typos[i]][['Longitude', 'Latitude']].to_numpy()
    ax.plot(coords[:, 0], coords[:, 1], 'x', label=typos[i], color='black', ms=7, mew=3)
    lgnd_stations = ax.legend(loc='center', bbox_to_anchor=[0.85, 0.7])

    zoning_shp.plot('Class', ax=ax, alpha=0.7, cmap=cmap, legend=True)
    ax.add_artist(lgnd_stations)
    ax.set_facecolor('white')

plt.show()

# plot
fig, ax = plt.subplots(figsize=(10, 10))
colors = ['yellow', 'blue', 'red', 'green', 'white', 'black', 'pink', 'orange']

typos = list(set(df['Typology']))
typos.sort()
counts = list(mean_df['Typology'].value_counts().reset_index().sort_values('index')['Typology'].values)

for i, t in enumerate(typos):
    coords = j_gdf[j_gdf['Typology'] == typos[i]][['Longitude', 'Latitude']].to_numpy()
    ax.plot(coords[:, 0], coords[:, 1], 'x', label=f'{typos[i]} ({counts[i]})', color=colors[i], ms=8, mew=3)

lgnd_stations = ax.legend(loc='center', bbox_to_anchor=[0.9, 0.7])
zoning_shp.plot('Class', ax=ax, alpha=0.7, cmap=cmap, legend=True)
ax.add_artist(lgnd_stations)
plt.show()

#
classes_mean = []
classes_median = []
means = j_gdf[['NO2', 'Class']].groupby('Class').mean().reset_index()
medians = j_gdf[['NO2', 'Class']].groupby('Class').median().reset_index()

for c in classes:
    classes_mean.append(means[means['Class'] == c]['NO2'].values[0])
    classes_median.append(medians[medians['Class'] == c]['NO2'].values[0])

for c, m in zip(classes, classes_mean):
    print(f'{c}: {m}')

# plot
_, ax = plt.subplots(figsize=(8, 8))

ax.plot(full_j_gdf['Class'], full_j_gdf['NO2'], 'x', color='grey')

min_x = [value-0.1 for value in range(len(classes))]
max_x = [value+0.1 for value in range(len(classes))]

ax.hlines(classes_mean, min_x, max_x, label='Mean NO₂ value', colors=['red'])
ax.hlines(classes_median, min_x, max_x, label='Median NO₂ value', colors=['black'])

ax.set_ylabel('NO₂ (µg/m³)')
ax.set_xlabel('Surface Class')
plt.legend(facecolor='white', loc='upper center')

plt.show()

# fitlered dataframe: removing altitude outliers (over 600m)
f_df = j_gdf[j_gdf['Altitude'] < 600]

# plot
_, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

for i, ax in enumerate(axs.flatten()):

    ax.plot(f_df['Altitude'], f_df['NO2'], 'x', color='grey', ms=10, label='Stations mean value')

    c_df = f_df[f_df['Class'] == classes[i]]
    m, q = np.polyfit(c_df['Altitude'], c_df['NO2'], 1)
    x = np.linspace(f_df['Altitude'].min(), f_df['Altitude'].max(), 2)

    ax.plot(c_df['Altitude'], c_df['NO2'], 'o', color='black', ms=12, label=classes[i], markerfacecolor='None')
    ax.plot(x, m*x + q)
    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel(f'NO₂ (µg/m³)')
    ax.legend(facecolor='white', framealpha=1)

plt.show()
