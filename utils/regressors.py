import pandas as pd
import numpy as np
import geopandas as gpd

from utils.netcdf import get_time_series_at_location

from utils.processing import upsample
from utils.processing import clean_registry


def time_to_int(date, start='2016/01/01'):
    """
    Converts a date into an integer by counting the number of days between that date and the stat date.

    Parameters:
        date  (string)  : The date to be converted
        start (string)  : The starting date

    Returns:
        diff    (int)   : The number of days between start and date
    """
    diff = pd.to_datetime(date) - pd.to_datetime(start)
    diff = np.abs(diff.days)
    return diff


def add_registry_information(df, reg_path, pollutant, start, end):
    """
    Returns a decorated DataFrame with registry informations.

    Loads the registry file and cleans it by deleting inactive stations or stations with an unspecified typology.
    Appends to every row of the input dataset information about Latitude, Longitude and Typology of the station
    related to that row.

    Parameters:
        df      (DataFrame)     : The DataFrame to be augmented
        reg_path   (string)     : The registry file's path
        pollutant  (string)     : The pollutant of interest's name
        start      (string)     : The start date of the regressor's time interval
        end        (string)     : The end date of the regressor's time interval

    Returns:
        df       (DataFrame)    : The augmented DataFrame
        registry (DataFrame)    : The cleaned registry DataFrame with only relevant information (ID, coordinates and the
                                  typology)
    """
    registry = pd.read_csv(reg_path)

    registry = clean_registry(registry, [pollutant], start, end)

    registry = registry[registry['Pollutant'] == pollutant]
    registry = registry[['IDStation', 'Latitude', 'Longitude', 'Typology', 'Altitude']]

    df = pd.merge(df, registry, how='inner')

    return df, registry


def add_time_variant_regressor(df, X, registry, feature, start, end):
    """
    Returns a DataFrame with a new column containing the time variant regressor value.

    For every station, searches the closest time series in the DataArray object and updates the entire station's time series
    with the time series found.

    Parameters:
        df   (xarray.DataArray)  : The DataArray containing the regressor's values
        X           (DataFrame)  : The DataFrame to be augmented
        registry    (DataFrame)  : The DataFrame containing the registry information
        feature        (string)  : The name of the regressor column to be added
        start          (string)  : The start date of the regressor's time interval
        end            (string)  : The end date of the regressor's time interval

    Returns:
        df      (DataFrame)  : The augmented DataFrame
    """
    unique_stations = list(set(registry['IDStation'].values))

    for station in unique_stations:

        lat, lon = registry[registry['IDStation'] == station]['Latitude'].values[0], registry[registry['IDStation'] == station]['Longitude'].values[0]

        idxs = list(df[df['IDStation'] == station]['Time'].values)

        ts = upsample(get_time_series_at_location(X, lat, lon, feature), feature, start, end, fillna=True)
        ts = list(ts[feature].values)
        ts = [ts[idx] for idx in idxs]

        df.loc[df['IDStation'] == station, [feature]] = ts  

    return df


def add_time_invariant_regressor(df, X, registry, feature):
    """
    Returns a DataFrame with a new column containing the time invariant regressor value.

    Parameters:
        df   (xarray.DataArray)  : The DataArray containing the regressor's values
        X           (DataFrame)  : The DataFrame to be augmented
        registry    (DataFrame)  : The DataFrame containing the registry information
        feature        (string)  : The name of the regressor column to be added

    Returns:
        df      (DataFrame)  : The augmented DataFrame
    """
    unique_stations = list(set(registry['IDStation'].values))

    for station in unique_stations:

        lat = registry[registry['IDStation'] == station]['Latitude'].values[0]
        lon = registry[registry['IDStation'] == station]['Longitude'].values[0]

        altitude_value = X.sel(lat=lat, lon=lon, method='nearest', drop=True).values
        df.loc[df['IDStation'] == station, [feature]] = altitude_value

    return df


def add_one_hot_encoded_regressors(df, feature, prefix):
    """
    Returns a DataFrame with a set of new columns based on a one-hot encoding on a given feature.

    Parameters:
        df   (DataFrame)  : The DataFrame to be augmented
        feature (string)  : The name of the column to be encoded
        prefix  (string)  : The prefix for every encoded column

    Returns:
        df   (DataFrame)  : The augmented DataFrame
        columns   (list)  : The encoded columns name
    """
    dummies = pd.get_dummies(df[feature], prefix=prefix)
    columns = list(dummies.columns)
    X = pd.concat([df, dummies], axis=1)

    return X, columns


def add_dummy_regressor(df, feature, condition):
    """
    Returns a DataFrame with a feature column containing boolean values in an integer format.

    The value is 0 (False) or 1 (True) if the value meets a specified condition.

    Parameters:
        df      (DataFrame) : The DataFrame to be augmented
        feature    (string) : The name of the column to be inserted
        condition  (lambda) : The condition that every row must satisfy

    Returns:
        df      (DataFrame) : The augmented DataFrame
    """
    df[feature] = df.apply(condition, axis=1)

    return df


def classify_surface(df, encoded=False, how='inner'):
    """

    """
    rename_dict = {
        'Agg_BG': 'Agg_Urbano',
        'Agg_BS': 'Agg_Urbano',
        'Agg_MI': 'Agg_Urbano',
        'A': 'Urbano',
        'B': 'Rurale',
        'C': 'Montagna',
        'D': 'Montagna'
    }

    # ARPA's zoning classification
    zoning_shp = gpd.read_file('./shp/zoning/zone.shp').to_crs(epsg=4326)
    zoning_shp.rename(columns={'COD_ZONA': 'Class'}, inplace=True)
    zoning_shp.drop(['COD_ZONA2C'], axis=1, inplace=True)
    zoning_shp['Class'].replace(rename_dict, inplace=True)

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'])).set_crs(4326)
    j_gdf = gpd.sjoin(gdf, zoning_shp, how=how)
    j_gdf.drop('index_right', axis=1, inplace=True)

    df = pd.DataFrame(j_gdf).drop('geometry', axis=1)

    if encoded:
        df, classes = add_one_hot_encoded_regressors(df, feature='Class', prefix='Class')
    else:
        classes = None

    return df, classes


def add_seasons(df):

    seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
    month_to_season = dict(zip(range(1, 13), seasons))
    df['Season'] = df['Date'].apply(lambda date: month_to_season[pd.to_datetime(date).month])
    df, seasons = add_one_hot_encoded_regressors(df, 'Season', 'Season')
    df.drop('Season', axis=1, inplace=True)
    df.rename(columns={'Season_1': 'Winter', 'Season_2': 'Spring', 'Season_3': 'Summer', 'Season_4': 'Fall'},
              inplace=True)
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    return df, seasons


def normalize_column(df):
    return (df - df.mean())/df.std()


def get_value_at_location_and_time(data, lat, lon, time):
    return data.sel(lat=lat, lon=lon, time=time, method='nearest').values.item()


def get_value_at_location(data, lat, lon):
    return data.sel(lat=lat, lon=lon, method='nearest').values.item()


def get_value_at_time(data, time):
    return data.sel(time=time, method='nearest').values
