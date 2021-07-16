import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns
import geopandas as gpd
import pandas as pd
import numpy as np
import datetime

from itertools import product


__output_path = './output/images/'

def _save_plot(save, filename):
    '''
    Saves the plot to the default output folder with a given filename.

    Parameters:
        save     (boolean) : If True saves the image
        filename (string)  : The name of the output file

    Returns:
        None
    '''
    if save:
        if len(filename) > 0:
            plt.savefig(__output_path + filename + '.png', bbox_inches='tight', pad_inches=0)
        else:
            raise Warning('Cannot save the image without a properly defined image name')


def _plot_markers(lons, lats, shp_path, ax, epsg=4326):
    """
    Plots the position of the stations.

    Parameters:
        lons       (float) : The point longitudes
        lats       (float) : The point latitudes
        shp_path  (string) : The path of the shapefile to be plotted below
        ax          (Axes) : The axe on which the points must be plotted
        epsg         (int) : The EPSG code identifying the coordinates reference system

    Returns:
        None
    """
    lombardia = gpd.read_file(shp_path)
    lombardia.to_crs(epsg=epsg, inplace=True)
    lombardia.plot(ax=ax, color='#BCCBD6')

    prov = gpd.read_file('shp/province/province.shp')
    prov.to_crs(epsg=epsg, inplace=True)
    prov['coords'] = prov['geometry'].apply(lambda x: x.centroid.coords[:])
    prov['coords'] = [coords[0] for coords in prov['coords']]
    prov.boundary.plot(ax=ax, color='gray', linewidth=0.5)
    for idx, row in prov.iterrows():
        ax.annotate(s=row['SIGLA'], xy=row['coords'],
                    horizontalalignment='center', color='grey')

    ax.plot(lons, lats, linestyle='none', marker="o", markersize=7, c="#005A8F", markeredgecolor="white",
            markeredgewidth=.5, label='ARPA Stations')


def plot_stations_location(registry, pollutants, nrows=None, ncols=None, show=True, save=False, filename='', grid=None):
    """
    Plots the stations position for several pollutants.

    Shows the position of the stations with at least one sensor measuring a pollutant
    within the list of the pollutants of interest.

    Parameters:
        registry   (DataFrame) : The DataFrame containing the stations information
        pollutants      (list) : The list containing the name of the pollutants of interest
        nrows            (int) : The number of rows in the plot
        ncols            (int) : The number of columns in the plot
        show         (boolean) : If True shows the image
        save         (boolean) : If True saves the image
        filename      (string) : The name of the output file
        grid       (DataArray) : The DataArray containing latitudes and longitudes information useful for displaying a grid

    Returns:
        None
    """
    assert((nrows is not None and ncols is not None) or (ncols is None and ncols is None))

    if ncols is None and nrows is None: 
        ncols = len(pollutants)
        nrows = 1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*6))

    iterator = enumerate(axs.flatten()) if hasattr(axs, '__iter__') else [(0, axs)]

    for i, ax in iterator:

        plt.sca(ax)

        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude', labelpad=10)        

        data = registry[(registry['Pollutant'] == pollutants[i])]

        _plot_markers(data['Longitude'], data['Latitude'], ax=ax, shp_path='shp/lombardia/lombardia.shp')

        if grid is not None: plot_grid(grid, ax)

        ax.legend(facecolor='white', framealpha=1)

    _save_plot(save, filename)

    if show: plt.show()    
    else: plt.close()


def _subplot_bubbles(registry, data, pollutant, on, ax, shp_path, epsg=4326, scale=1, cmap='Reds', unit='µg/m³'):
    """
    Plots qualitatively the mean concentration of a specific pollutant over the whole time interval.

    Parameters:
        registry  (DataFrame) : is the dataset containing the station informations
        data      (DataFrame) : is the dataset containing the pollutant concentration values
        pollutant    (string) : is the pollutant's name
        shp_path     (string) : is the path of the shapefile to be plotted in the background
        scale           (int) : is a coefficient to change the circles dimension
        cmap         (string) : is the heatmap's color map

    Returns:
        None
    """
    lombardia = gpd.read_file(shp_path)
    lombardia.to_crs(epsg=epsg, inplace=True)
    lombardia.plot(ax=ax, color='#BCCBD6')

    prov = gpd.read_file('shp/province/province.shp')
    prov.to_crs(epsg=epsg, inplace=True)
    prov['coords'] = prov['geometry'].apply(lambda x: x.centroid.coords[:])
    prov['coords'] = [coords[0] for coords in prov['coords']]
    prov.boundary.plot(ax=ax, color='gray', linewidth=0.5)
    for idx, row in prov.iterrows():
        ax.annotate(s=row['SIGLA'], xy=row['coords'],
                    horizontalalignment='center', color='grey')

    grouped_data = data[pd.notna(data[pollutant])].groupby(on).mean()
    merge = registry[(registry['Pollutant'] == pollutant) & pd.isna(registry['DateStop'])].join(grouped_data[pollutant],
                                                                                                on=on)

    col_name = pollutant + 't'
    merge[col_name] = list(map(lambda value: value*scale, merge[pollutant]))
    pollutant = col_name

    ms = [float(value) for value in merge[pollutant]]

    min_value = merge[pollutant].min()
    max_value = merge[pollutant].max()

    colors = [value/max_value for value in merge[pollutant]]

    plt.scatter(merge['Longitude'], merge['Latitude'], s=ms, c=colors, alpha=0.8, cmap=cmap)

    ticks = [int(num) for num in np.linspace(min_value, max_value, 8)]

    cbar = plt.colorbar(fraction=0.04, pad=0.1)  
    cbar.ax.set_yticks(ticks)
    cbar.ax.set_yticklabels([str(tick) for tick in ticks])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(unit, fontsize=10, rotation=270,
                       fontdict={'fontsize': 8, 'fontweight': 'medium'})
    # fontdict={'fontsize': 8, 'fontweight': 'medium', 'fontfamily': 'Times New Roman'})


def plot_bubbles(registry, pollutants, names, nrows=None, ncols=None, coefficients=[],
                 cmaps=['Blues', 'Greys', 'Reds', 'Oranges'], show=True, save=False, filename=''):
    """
    Plots qualitatively the mean concentration of a list of pollutants over the whole time interval.

    Paramaters:
        registry  (DataFrame) : The DataFrame containing the stations information
        pollutants     (list) : The list of DataFrames containing the pollutants concentration values
        names          (list) : The pollutants name list
        coefficients   (list) : The list of scale coefficients to change the heatmap's circles dimension
        cmaps          (list) : The heatmap's color map list, one for each pollutant
        show        (boolean) : If True shows the image
        save        (boolean) : If True saves the image
        filename     (string) : The name of the output file

    Returns:
        None

    """
    assert((nrows is not None and ncols is not None) or (ncols is None and ncols is None))

    if ncols is None and nrows is None: 
        ncols = len(names)
        nrows = 1

    if not len(coefficients) > 0 : coefficients = [1]*len(pollutants)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 6*nrows))

    fig.subplots_adjust(wspace=0.4, hspace=0.2)

    iterator = enumerate(axs.flatten()) if hasattr(axs, '__iter__') else [(0, axs)]

    for i, ax in iterator:

        plt.sca(ax)

        # ax.set_title(names[i])
        ax.set_ylabel('Latitude', labelpad=10)
        ax.set_xlabel('Longitude', labelpad=10)  

        data = pollutants[i]
        _subplot_bubbles(registry, pollutants[i], names[i], on='IDStation', ax=ax,
                         shp_path='shp/lombardia/lombardia.shp', scale=coefficients[i], cmap=cmaps[i])

    _save_plot(save, filename)

    if show: plt.show() 
    else: plt.close()


def plot_distributions(pollutants, names, xlabels=None, nrows=None, ncols=None, show=True, save=False, filename=''):
    """
    Plots the distribution of a single pollutant or of a list of pollutants.

    Paramaters:
        pollutants   (list) : The DataFrame containing the pollutants concentration values or the list of pollutants values dataset
        names        (list) : The pollutants of interest's name
        nrows         (int) : The number of rows in the plot
        ncols         (int) : The number of columns in the plot
        show      (boolean) : If True shows the image
        save      (boolean) : If True saves the image
        filename   (string) : The name of the output file

    Returns:
        None
    """
    assert((nrows is not None and ncols is not None) or (ncols is None and ncols is None))

    if ncols is None and nrows is None: 
        ncols = len(names)
        nrows = 1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 6*nrows))

    fig.subplots_adjust(wspace=0.4, hspace=0.2)

    iterator = enumerate(axs.flatten()) if hasattr(axs, '__iter__') else [(0, axs)]

    for i, ax in iterator:

        plt.sca(ax)
        pollutant = names[i]
        data = pollutants[i]

        print(data[pollutant].min(), data[pollutant].max())
        # dist = sns.distplot(data[(pd.notna(data[pollutant]) & (data[pollutant] > 0))][pollutant])
        hist = sns.histplot(data[pollutant], kde=True, stat='probability')
        hist.set(xlabel=xlabels[i])

    _save_plot(save, filename)

    if show: plt.show() 
    else: plt.close()


def correlation_plot(data, pollutants, station='', show=True, save=False, filename=''):

    """
    Plots the correlation between the specified pollutants ad a given station.

    Parameters:
        data        (list) : The list of the pollutants time series
        pollutants  (list) : The list containing the pollutants name
        station   (string) : The station's name at which the pollutants are detected

    Returns:
        None
    """
    plt.style.use('ggplot')

    assert len(data) == len(pollutants)

    combinations = [(i, j) for (i, j) in product(list(range(len(pollutants))), repeat=2)]    

    fig, axs = plt.subplots(nrows=len(data), ncols=len(data)-1, figsize=(20, 20))
    fig.tight_layout()

    title = 'Pollutants correlation - ' + station if len(station) > 0 else 'Pollutants correlation'
    fig.suptitle(title, fontsize=20, fontweight='bold')
    fig.subplots_adjust(top=0.95)

    skip = False
    ni = 1

    for i, j in combinations:

        if i == j: 
            skip = True
            ni = i + 1
        
        else:

            if skip and i == ni:
                skip = False

            plt.sca(axs[i, j-1 if skip else j])

            plt.scatter(data[i][pollutants[i]], data[j][pollutants[j]], color='grey', alpha=0.8)
            plt.xticks([])
            plt.xlabel(pollutants[i])
            plt.ylabel(pollutants[j])
            plt.yticks([])   

            sns.regplot(data[i][pollutants[i]], data[j][pollutants[j]], scatter=False, line_kws={"color": "red"})

    _save_plot(save, filename)

    if show: plt.show() 
    else: plt.close()
    

def plot_grid(data, ax, label='', color='grey'):
    """
    Plots a NetCDF Observation grid on a specific ax.

    Parameters:
        data   (DataArray) : The DataArray containing latitudes and longitudes information useful for displaying a grid
        ax          (Axes) : The axe on which the points must be plotted
        color     (String) : The color of the grid
        label     (String) : The label on the legend

    Returns:
        None
    """
    lats = data.lat.values
    lons = data.lon.values

    ax.hlines(lats, xmin=lons.min(), xmax=lons.max(), colors=color, alpha=0.8, label=label)
    ax.vlines(lons, ymin=lats.min(), ymax=lats.max(), colors=color, alpha=0.8)


def plot_time_series(df, column, show=True, save=False, filename='', title='', ylabel=''):
    """
    Plots the time series.

    Parameters:
        df      (DataArray) : The DataFrame containing the time series
        column       (Axes) : The DataFrame column containing the time series
        show      (boolean) : If True shows the image
        save      (boolean) : If True saves the image
        filename   (string) : The name of the output file

    Returns:
        None
    """
    fig, axs = plt.subplots(figsize=(15, 5))

    dates = list(df['Date'].values)

    formatter = mdates.DateFormatter("%Y-%m")
    locator = mdates.MonthLocator(interval=4)

    axs.xaxis.set_major_formatter(formatter)
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_minor_locator(locator)

    axs.set_title(title)
    fig.subplots_adjust(top=0.95)
    
    axs.set_ylabel(ylabel)    

    axs.plot(dates, df[column].values)

    _save_plot(save, filename)

    if show: plt.show() 
    else: plt.close()


def plot_feature(darray, dframe, feature, label, unit, loc, time_variant=True, with_boundary=True, save=False):

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.2)

    # heatmap

    if time_variant:
        img = darray.mean(dim='time').plot.pcolormesh(ax=axs[0], add_colorbar=False)
        axs[0].set_title(f'Mean {label} over time')
    else:
        img = darray.plot.pcolormesh(ax=axs[0], add_colorbar=False)
        axs[0].set_title(f'{feature}')

    axs[0].xaxis.set_major_locator(plt.MaxNLocator(10))
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')

    if with_boundary:
        shape = gpd.read_file('shp/lombardia/lombardia.shp')
        shape.to_crs(epsg=4326, inplace=True)
        shape.boundary.plot(ax=axs[0], color='#BCCBD6')

    # distribution dframe
    sns.histplot(np.array(dframe[feature].values).reshape(-1), ax=axs[1], color='red', alpha=1,
                 label='Learning Dataset')
    # dsitribution darray
    sns.histplot(np.array(darray.values).reshape(-1), ax=axs[1], color='C0', alpha=0.6, label='Surface')

    axs[1].set_title('Surface and learning dataset distributions')
    axs[1].set_xlabel(f'{label} ({unit})')
    axs[1].legend()

    fig.colorbar(img, ax=axs, shrink=0.95, pad=0.08, label=feature, location='left')

    if save:
        plt.savefig(f'output/images/features/{feature}_distribution.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    # time series at location
    darray.sel(lon=loc[0], lat=loc[1], method='nearest').plot(ax=ax)
    ax.set_title(f'Longitude: {loc[0]}, Latitude: {loc[1]}')
    ax.set_xlabel('Time')
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    ax.set_ylabel(f'{label} ({unit})')

    if save:
        plt.savefig(f'output/images/features/{feature}_ts.png', bbox_inches='tight', pad_inches=0)
    plt.show()
