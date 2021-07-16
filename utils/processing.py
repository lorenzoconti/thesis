import pandas as pd
import numpy as np
import scipy.stats as stats

from utils.export import correlations_to_latex

def clean_registry(registry, pollutants, start, end):
    '''
    Cleans the registry dataset.

    Removes the stations outside the time interval of interest with a DateStart after
    the start value, and with a DateStop different then NaN and before the end value.
    Removes sensors that measure unanalyzed pollutants.

    Parameters:
        registry   (DataFrame) : The DataFrame to be cleaned
        pollutants      (list) : The names of the pollutants of interest
        start         (string) : The start date of the time interval of interest
        end           (string) : The end date of the time interval of interest  

    Returns:
        registry    (DatFrame) : The cleaned DataFrame
    '''
    assert(isinstance(registry, pd.DataFrame))
    assert(isinstance(pollutants, list))

    cleaning_filter = lambda row: (
        (row['DateStop'].isna()) | (pd.to_datetime(row['DateStop']) > pd.to_datetime(end))) & \
        (pd.to_datetime(row['DateStart']) < pd.to_datetime(start)) & \
        (row['Pollutant'].isin(pollutants))
    
    registry = registry[cleaning_filter]

    if 'Tipology' in list(registry.columns):
        registry = registry[~registry['Tipology'].isna()]

    return registry


def clean_pollutant(df, stations, pollutant):
    """
    Cleans the pollutant dataset.

    Removes data from uninteresting stations and replaces the negative values with np.nan.

    Parameters:
        df        (DataFrame)  : The pollutant dataset to be cleaned
        stations       (list)  : The stations of interest
        pollutant    (string)  : The pollutant name

    Returns:
        None, modifies the DataFrame inplace

    """
    df.drop(df[~(df['IDStation'].isin(stations))].index, inplace=True)

    df[pollutant].where(df[pollutant] > 0, np.nan, inplace=True)


def get_complete_stations(registry, pollutants):
    '''
    Gets the list of stations capable of detecting the concentration of all pollutants.

    Parameters:
        registry    (DataFrame) : The dataset containing the stations information
        pollutants       (list) : The list of all the pollutant names the stations must detect
    
    Returns:
        complete_stations   (DataFrame) :   The DataFrame containing the stations that detect all the specified pollutants.
    '''
    detects_all_pollutants = lambda group: set(pollutants) <= set(group['Pollutant'])
    filter_pollutants_of_interest = lambda row: row['Pollutant'].isin(pollutants) 

    filter_active_stations = lambda row: pd.isna(row['DateStop'])

    complete_stations = registry[filter_active_stations].groupby('IDStation') \
                            .filter(detects_all_pollutants)[filter_pollutants_of_interest] \
                            .sort_values(by='IDStation')

    return complete_stations


def fill_missing_rows(df, column, fillwith):
    '''
    Fills the rows of a specific column with a provided list of values.

    Parameters:
        df       (DataFrame)  : The DataFrame to be filled
        column      (string)  : The target column
        fillwith      (list)  : The list of values to fill the column with
    
    Returns:
        df       (DataFrame)  : The filled DataFrame
    '''
    df = df.set_index(column).reindex(pd.Index(fillwith))

    df.reset_index(inplace=True)

    df = df.rename(columns={'index': column})

    return df


def fill_missing_dates(df, start, end, freq='d', column='Date'):    
    '''
    Fills the rows of a specific column with the dates between start and end and with a given frequency.

    Parameters:
        df      (DataFrame) : The DataFrame to be filled
        column     (string) : The target column
        start      (string) : The time interval start date
        end        (string) : The time interval end date
        freq       (string) : The frequency of the date ranging
    
    Returns:
        df      (DataFrame) : The filled DataFrame 
    '''
    index = pd.Index(pd.date_range(start, end, freq=freq))

    if len(df) < len(index):

        df[column] = pd.to_datetime(df[column])

        df = fill_missing_rows(df, column, index)

        assert(len(df) == len(index))

    return df


def remove_nan(dfs, columns):
    '''
    Given multiple DataFrames, on every DataFrame removes the rows where
    at least one DataFrame has a missing value on a specific target column. 

    Parameters:
        dfs     (list) : The list of DataFrame objects.
        columns (list) : The list of the target columns.
    
    Returns:
        dfs     (tuple) : A tuple containing DataFrames cleaned by NaNs    
    '''
    idxs = []

    for i, df in enumerate(dfs):

        df.index = [idx for idx in range(len(df))]
        idxs.extend(list(df[df[columns[i]].isna()].index.values))
    
    idxs = list(set(idxs))
    
    for i, df in enumerate(dfs):

        if df[columns[i]].isna().sum() > 0:
             df.drop([df.index[i] for i in idxs], inplace=True)

    return tuple(dfs)


def get_time_series_by_station(pollutants, id_station):
    '''
    Returns a list containing the time series of the pollutants detected by a specific station.

    Parameters:   
        pollutants   (list) : The list of pollutants DataFrames
        id_station (string) : The id of the station of interest
    
    Returns:
        pollutants_at_station (list) : The list of DataFrames, one for each pollutant detected by the given station

    '''
    pollutants_at_station = []
        
    for pollutant in pollutants:
        
        pollutant_at_station = pollutant[pollutant['IDStation'] == id_station]
        pollutants_at_station.append(pollutant_at_station)

    return pollutants_at_station
    

def mean_correlation(pollutants, stations, names, start, end, fillna=True, diag=False, export=False, output_path=''):
    
    '''
    Computes the mean correlation between the pollutants across all the stations that detect all the pollutants.

    If diag is True computes the correlation between the same pollutant across the stations.

    Paramaters:
        pollutants  (list) : The DataFrames containing the pollutants time series
        stations    (list) : The stations identifiers
        names       (list) : The pollutants names

    Returns:  
        df  (DataFrame) : A table with the correlations results
    '''
    correlations = {}

    default_correlation = { 'correlations' : [], 'p-values' : [] }

    # compute the correlations between the pollutants of interests for every station of interest
    for station in stations:

        pollutants_at_station = []

        # fill the missing dates to have equally long time series
        # missing values imputation with pad method (forward and backword)
        for pollutant in pollutants:
            p = pollutant[pollutant['IDStation'] == station]
            p =  fill_missing_dates(p, start, end)

            if fillna:  p = p.fillna(method='ffill').fillna(method='bfill')       

            pollutants_at_station.append(p)

        skip = 0 if diag else 1
        
        for i in range(len(pollutants_at_station)):

            for j in range(skip, len(pollutants_at_station)):

                couple = '{}-{}'.format(names[i], names[j])

                if not fillna: 

                    p_i = fill_missing_dates(pollutants_at_station[i], start, end)
                    p_j = fill_missing_dates(pollutants_at_station[j], start, end)

                    p_i, p_j = remove_nan([p_i, p_j], [names[i], names[j]])
                    p_i = p_i[names[i]]
                    p_j = p_j[names[j]]
                else:
                    p_i, p_j = pollutants_at_station[i][names[i]], pollutants_at_station[j][names[j]]

                r, p = stats.pearsonr(p_i, p_j)

                couple_corr = correlations.get(couple, default_correlation)

                correlations.update({couple: {
                    'correlations' : couple_corr['correlations'] + [r],
                    'p-values': couple_corr['p-values'] + [p],
                }})

            skip += 1
    
    df = _correlations_to_dataframe(correlations, names)

    if export: correlations_to_latex(df, output_path)

    return df


def _correlations_to_dataframe(correlations, columns):
    '''
    Transfoms the correlation dictionary to a dataframe.

    Parameters:
        correlations (dict) : The dictionary containing the correlations results
        columns      (list) : The returned DataFrame column names
    
    Returns:
        df  (DataFrame) : The summary of the correlation results in a table format
    '''
    dim = len(columns)
    df = pd.DataFrame(data=np.ones((dim, dim)), columns=columns, index=columns)

    str_and_round = lambda value: str(round(value, 2))

    for k, v in correlations.items():
        
        p1, p2 = k.split('-')
        value = '{} ± {}'.format(str_and_round(np.array(v['correlations']).mean()), str_and_round(np.array(v['correlations']).std()))
        df[p1][p2] = value
        df[p2][p1] = value

    return df


def upsample(df, column, start, end, fillna=False):
    '''
    Upsamples the time series by augmenting the DataFrame with the missing dates and interpolating them.

    Parameters:
        df    (DataFrame) : The DataFrame containing the time series to be upsampled
        column   (string) : The column name of the DataFrame where the time series is located
        start    (string) : The time interval start date
        end      (string) : The time interval end date
        fillna  (boolean) : If True fills the remaining missing values

    Returns:
        filled_df (DataFrame) : The DataFrame with the upsampled time series at the specified column
    '''
    filled_df = fill_missing_dates(df, start, end, freq='d', column='Date')
    filled_df[column] = filled_df[column].interpolate()

    if fillna: filled_df[column] = filled_df[column].fillna(method='ffill').fillna(method='bfill')

    return filled_df




        






            

