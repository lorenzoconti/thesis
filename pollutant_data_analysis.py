import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from matplotlib.dates import DateFormatter

from utils.processing import clean_registry
from utils.processing import clean_pollutant
from utils.plot import plot_stations_location
from utils.plot import plot_bubbles
from utils.plot import plot_distributions

warnings.filterwarnings("ignore")
path = 'D:/thesis/fonts/cmunss.ttf'
plt.style.use('ggplot')

# bounding box
start, end = '2016-01-01', '2020-12-31'

# ARPA data
NO2 = pd.read_csv('data/arpa_pollutants/NO2.csv', index_col=0)
registry = pd.read_csv('data/arpa_registry/registry.csv')

pollutants = [NO2]
pollutant_names = ['NO2']

# stations of interest
stations = list(set(registry['IDStation'].to_list()))

# clean registry
registry = clean_registry(registry, pollutant_names, start, end)

# clean pollutants
for pollutant, name in zip(pollutants, pollutant_names):
    clean_pollutant(pollutant, stations, name)

del pollutant

# plot stations location of each pollutant of interest
# plot_stations_location(registry, pollutant_names, nrows=1, ncols=1, save=True, filename='no2_stations.png')
plot_stations_location(registry, pollutant_names, nrows=1, ncols=1)

# plot NO2 bubble map
plot_bubbles(registry, pollutants, pollutant_names, nrows=1, ncols=1, coefficients=[6])

# plot NO2 distribution
print(NO2['NO2'].min(), NO2['NO2'].max())
plot_distributions(pollutants, pollutant_names, xlabels=['NO₂ (µg/m³)'])

NO2['NO2'][NO2['NO2'] < 0] = np.nan

means = NO2.groupby(pd.to_datetime(NO2['Date']))['NO2'].mean()
mins = NO2.groupby(pd.to_datetime(NO2['Date']))['NO2'].min()
maxs = NO2.groupby(pd.to_datetime(NO2['Date']))['NO2'].max()

# plot NO2 time series
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlabel('time')
ax.set_ylabel('NO₂ (µg/m³)')
ax.plot(means, label='Mean NO₂ (µg/m³) over ARPA Stations')
ax.fill_between(means.index, mins, maxs, alpha=0.5, label='NO₂ (µg/m³) Excursion')

date_form = DateFormatter("%b-%Y")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))

ax.legend(facecolor='white', framealpha=1)
plt.show()
