import xarray as xr
import pandas as pd
from netCDF4 import Dataset, num2date
import numpy as np
import numpy.ma as ma
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib import patches, colors
from datetime import datetime, timedelta
from collections import defaultdict
from collections import Counter
import scipy.optimize as opt
import plotly.express as px
import scipy.stats as stats
from scipy.stats import linregress
import re
import pickle 
import time
import warnings
import seaborn as sn
import cmasher as cmr
import mpl_axes_aligner

from functions_diff_warming import *

pickle_path = '/fs/yedoma/home/vpo001/VikScriptsTests/Python_Pickles/'


def assign_value_reanalysis_stat_generic(var_name, df, forcings,
                                         temp_air, SW_flux, SW_direct_flux, SW_diffuse_flux,
                                         time_bkg_air, time_trans_air, time_pre_trans_air,
                                         extension=''):
    """ Creates a dictionary of mean quntities over the background and transient periods
    
    Parameters
    ----------
    var_name : str
        Should ONLY take var_name = reanalysis_stats
    df : pandas.core.frame.DataFrame
        Panda DataFrame df, should at least include the following columns: 'altitude', 'aspect', 'slope'
    forcings : list of str
        List of forcings provided, with a number of entries between 1 and 3 in 'era5', 'merra2', and 'jra55'. E.g. ['era5', 'merra2']
    temp_air : list of netCDF4._netCDF4.Variable
        list of netCDF air temperature time series for each forcing, e.g. ['temp_air_era5', 'temp_air_merra2']
    SW_flux : list of netCDF4._netCDF4.Variable
        list of netCDF SW flux time series for each forcing
    SW_direct_flux : list of netCDF4._netCDF4.Variable
        list of netCDF SW direct flux time series for each forcing
    SW_diffuse_flux : list of netCDF4._netCDF4.Variable
        list of netCDF SW diffuse flux time series for each forcing
    time_bkg_air : list of numpy.ma.core.MaskedArray
        list of masks to select only background period, for each forcing
    time_trans_air : list of numpy.ma.core.MaskedArray
        list of masks to select only transient period, for each forcing
    time_pre_trans_air : list of numpy.ma.core.MaskedArray
        list of masks to select only timestamps before the end of the transient period, for each forcing
    extension : str, optional
        Location of the event, e.g. 'Aksaut_Ridge' (not a list, 1 single location)    

    Returns
    -------
        dict

    """

    file_name = f"reanalysis_stats{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"reanalysis_stats{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    # if the variable has no assigned value yet, we need to assign it
    if var_name is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode 
            with open(my_path, 'rb') as file: 
                # Call load method to deserialize 
                var_name = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
        except (OSError, IOError) as e:
            # difference (trend) between trans and bkg periods for the air temperature:
            # amount of warming, at different altitudes

            # list of sorted altitudes at the site studied
            list_alt = sorted(list(set(df.altitude)))

            var_name = {forcings[i]: {list_alt[alt]:
                                    {'temp_bkg': temp_air[i][time_bkg_air[i], alt].mean(),
                                     # background air temperature at different elevations
                                     'temp_trans': temp_air[i][time_trans_air[i], alt].mean(),
                                     # transient air temperature at different elevations
                                     'air_warming': temp_air[i][time_trans_air[i], alt].mean()
                                     - temp_air[i][time_bkg_air[i], alt].mean(),
                                     # air warming at different elevations
                                     'SW': SW_flux[i][time_pre_trans_air[i], alt].mean(),
                                     # total flux mean over the whole study, at different elevations
                                     'SW_direct': SW_direct_flux[i][time_pre_trans_air[i], alt].mean(),
                                     # total direct flux mean over the whole study, at different elevations
                                     'SW_diffuse': SW_diffuse_flux[i][time_pre_trans_air[i], alt].mean()
                                     # total diffuse flux mean over the whole study, at different elevations
                                    }
                                    for alt in range(temp_air[i].shape[1])}
                        for i in range(len(forcings))}
            
            # Open a file and use dump() 
            with open(my_path, 'wb') as file: 
                # A new file will be created 
                pickle.dump(var_name, file)
            print('Created a new pickle:', file_name)

            # useless line just to use the variable 'e' so that I don't get an error
            if e == 0:
                pass

    else:
        print('The variable already existed:', var_name_full)

    return var_name

def glacier_filter_generic(var_name, snow_height, extension='', glacier=False, min_glacier_depth=100, max_glacier_depth=20000):
    """ Function returns a list of valid simulations regarding the glacier criteria
    
    Parameters
    ----------
    var_name : str
        will be named 'list_valid_sim' and is the name of the input and output list
    snow_height : netCDF4._netCDF4.Variable
        netCDF file with the height of snow as a functionof time for each simulation
    extension : str, optional
        Location of the event, e.g. 'Aksaut_Ridge'
    glacier : bool, optional
        By default only keeps non-glacier simulations but can be changed to True to select only glaciated simulations
    min_glacier_depth : float, optional
        Selects simulation with minimum snow height higher than this threshold (in mm)
    max_glacier_depth : float, optional
        Selects simulation with minimum snow height lower than this threshold (in mm)
    

    Returns
    -------
    var_name : list
        List of valid simulations.
    """
    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"list_valid_sim{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"list_valid_sim{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # if the variable has no assigned value yet, we need to assign it
    if var_name is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode 
            with open(my_path, 'rb') as file: 
                # Call load method to deserialize 
                var_name = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
        except (OSError, IOError) as e:
            # we create a dictionary of all valid simulations
            var_name = []
            for sim_index in range(snow_height.shape[0]):
                min_snow_height = np.min(snow_height[sim_index,:])
                if glacier:
                    if (min_snow_height >= min_glacier_depth) & (min_snow_height < max_glacier_depth):
                        var_name.append(sim_index)
                    else:
                        pass
                else:
                    if min_snow_height < min_glacier_depth:
                        var_name.append(sim_index)
                    else:
                        pass
            
            # Open a file and use dump() 
            with open(my_path, 'wb') as file:
                # A new file will be created 
                pickle.dump(var_name, file)
            print('Created a new pickle:', file_name)

            # useless line just to use the variable 'e' so that I don't get an error
            if e == 0:
                pass

    else:
        print('The variable already existed:', var_name_full)

    return var_name

def assign_weight_sim_Aksaut(df_stats):
    """ Function returns a statistical weight for each simulation according to the importance in rockfall starting zone 
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda DataFrame df_stats, should at least include the following columns: 'altitude', 'aspect', 'slope'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'

    Returns
    -------
    pd_weight : pandas.core.frame.DataFrame
        Panda DataFrame assigning a statistical weight to each simulation for each of 'altitude', 'aspect', 'slope'
        and an overall weight.
    """

    dict_weight = {}
    dict_weight = {i: [1,1,1]
                       for i in list(df_stats.index.values)}
    pd_weight = pd.DataFrame.from_dict(dict_weight, orient='index',
                                       columns=['altitude', 'aspect', 'slope'])
    pd_weight['weight'] = pd_weight['altitude']*pd_weight['aspect']*pd_weight['slope']
    
    return pd_weight

def assign_tot_water_prod_generic(swe_mean, mean_prec, time_ground, time_pre_trans_ground, time_air_merra2):
    """ Function returns the total water production at daily intervals in [mm s-1] 
    
    Parameters
    ----------
    swe_mean : list
        Mean swe over all 3 reanalyses
    mean_prec : list
        Mean precipitation over all simulations (weighted average)
    time_ground : netCDF4._netCDF4.Variable
        Time series for the ground timestamps
    time_pre_trans_ground : numpy.ma.core.MaskedArray
        Selects only the timestamps before the cutoff of the end of the transient era
    time_air_merra2 : netCDF4._netCDF4.Variable
        Time series for the air timestamps

    Returns
    -------
    list of daily total water production
    """

    # convert mean precipitation into a panda dataframe to facilitate grouping by 24.
    # This way the data is reported daily (as it is for the swe) rather than hourly
    pd_prec_day = ((pd.DataFrame(mean_prec).rolling(24).mean()).iloc[23::24, :]).reset_index(drop=True)

    # convert swe into panda df to make the diff() operation easier. Swe is a total and
    # we get the instantaneous flux by taking the difference between consecutive datapoints.
    # The total precipitation between 2 datapoints is over 1 day = 864000 hence we divide by 86400.
    # Finally, swe is in [kg m-2] which is equivalent to [mm] so we're good.

    # Number of days between the start of the ground and air timeseries
    num_day = int((num2date(time_ground[0], time_ground.units) - num2date(time_air_merra2[0], time_air_merra2.units)).total_seconds()/86400)
    # we add num_days zeros at the beginning to make sure the dfs have the same dimension and span Jan 1, 1980 to Dec 31, 2019
    # We make sure to multiply by (-1) and to divide by 86400
    pd_diff_swe = pd.concat([pd.DataFrame([0]*num_day), (pd.DataFrame(swe_mean[:len(time_ground[time_pre_trans_ground][:])]).diff()).fillna(0)/86400*(-1)], ignore_index=True)

    if len(pd_prec_day)!=len(pd_diff_swe):
        raise ValueError('VictorCustomError: The lengths of the precipitation and swe time series are different!')

    # The total water production is returned in the list type as the sum of precipitation and swe
    tot_water_prod = list(pd_prec_day.add(pd_diff_swe.reset_index(drop=True), fill_value=0)[0])

    return tot_water_prod

def mean_all_reanalyses_generic(time_files, files_to_smooth, year_bkg_end=2010, year_trans_end=2023):
    """ Function returns the mean time series over a number of reanalysis 
    
    Parameters
    ----------
    time_file : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored
    file_to_smooth : list of list
        List ime series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that it needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts [temp_air_era5[:,0]] but not [temp_air_era5]
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period

    Returns
    -------
    mean : dict
        average time series over reanalyses (3 or less)
    """

    list_years = [list_tokens_year(time_files[i], year_bkg_end, year_trans_end)[0] for i in range(len(time_files))]
    len_years = {y: [len(list_years[i][y]) for i in range(len(time_files))] for y in list(list_years[0].keys()) if y<year_trans_end}

    mean = []

    # presumably this means that y goes from 1980 to 2019
    for y in list(len_years.keys()):
        # check if all lists have the same amount of indices in a given year
        order = sorted(len_years[y])
        min_len = order[0]
        max_len = order[-1]
        if min_len == max_len:
            # if True, then we can proceed to computing the mean
            mean += list(np.mean([files_to_smooth[i][list_years[i][y]] for i in range(len(time_files))], axis=0))
        else:
            for j in range(len(order)):
                mean += list(np.mean([files_to_smooth[i][list_years[i][y]][(order[j-1] if j>0 else 0):order[j]]
                                        for i in range(len(time_files)) if len(files_to_smooth[i][list_years[i][y]]) > (order[j-1] if j>0 else 0)], axis=0))

    return mean

def aggregating_distance_temp_generic(time_file, file_to_smooth, window, year=0, year_bkg_end=2010, year_trans_end=2023, fill_before=False):
    """ Function returns the distance to the mean in units of standard deviation for a given date yy-mm-dd
        relative to the sensemble of same date for previous years: {year-mm-dd for year<=yy}
        Should only be applied to a previously 'smoothed' function through a running average
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored        
    file_to_smooth : list
        Time series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that it needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts temp_air_era5[:,0] but not temp_air_era5
    window : str
        Time window of datapoints to average over.
        Can take the following arguments: 'day', 'week', 'year', 'climate'
    year : int, optional
        One can choose to display a specific year or the whole set by choosing an integer not in the study sample, e.g. 0.
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    fill_before : bool, optional
        Option to fill the first empty values of the smoothed data with the first running average value if True

    Returns
    -------
    distance : dict
        Distance from mean in units of standard deviation

    """

    # here temp_smoothed is the smoothed temperature data series in the form of a dictionary 
    # keys are timestamps (end of window) and values are running average of temperatures over window
    temp_smoothed = running_mean_std(time_file, file_to_smooth, window, fill_before)[0]

    list_dates = list_tokens_year(time_file, year_bkg_end=2010, year_trans_end=2023)[0]

    num_min = np.min([len(list_dates[year]) for year in list_dates.keys() if year < year_trans_end])
    min_temp_index = list(temp_smoothed.keys())[0]

    # range of the timestamp whether the user chose a given year of the whole study period
    # index_range = list(range(len(file_to_smooth)))

    aggregate_mean = {}
    aggreagte_std = {}
    distance = {}

    start_year = num2date(time_file[0], time_file.units).year

    for year in [year_bkg_end]:
        aggregate_mean[year] = {day: np.mean([temp_smoothed[list_dates[y][day]] for y in range(start_year,year+1) if list_dates[y][day]>=min_temp_index]) for day in range(num_min)}
        aggreagte_std[year] = {day: np.std([temp_smoothed[list_dates[y][day]] for y in range(start_year,year+1) if list_dates[y][day]>=min_temp_index]) for day in range(num_min)}
        distance[year] = {day: (temp_smoothed[list_dates[year][day]]-aggregate_mean[year][day])/aggreagte_std[year][day] for day in range(num_min)}

    for year in range(year_bkg_end+1, year_trans_end):
        aggregate_mean[year] = {day: aggregate_mean[year-1][day]
                                     + (temp_smoothed[list_dates[year][day]] - aggregate_mean[year-1][day]) / (year-start_year+1)
                                     for day in range(num_min)}
        aggreagte_std[year] = {day: np.sqrt( ( (year-start_year) * aggreagte_std[year-1][day]**2
                                              + (temp_smoothed[list_dates[year][day]] - aggregate_mean[year][day])
                                              *(temp_smoothed[list_dates[year][day]] - aggregate_mean[year-1][day]) )
                                              / (year-start_year+1) )
                                    for day in range(num_min)}
        distance[year] = {day: (temp_smoothed[list_dates[year][day]]-aggregate_mean[year][day])/aggreagte_std[year][day] for day in range(num_min)}


    return distance

def plot_aggregating_distance_temp_generic(name_series, time_file, file_to_smooth, window, site, year=0, year_bkg_end=2010, year_trans_end=2023, fill_before=False):
    """ Plots the distance to the mean in units of standard deviation for a specific year or for the whole length
    
    Parameters
    ----------
    name_series : str
        Name of the quantity to plot, has to be 'air temperature', precipitation', 'snow depth'
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored        
    file_to_smooth : list
        Time series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that it needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts temp_air_era5[:,0] but not temp_air_era5
    window : str
        Time window of datapoints to average over.
        Can take the following arguments: 'day', 'week', 'year', 'climate'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    year : int, optional
        One can choose to display a specific year or the whole set by choosing an integer not in the study sample, e.g. 0.
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    fill_before : bool, optional
        Option to fill the first empty values of the smoothed data with the first running average value if True

    Returns
    -------
        plot

    """

    distance = aggregating_distance_temp_generic(time_file, file_to_smooth, window, year, year_bkg_end, year_trans_end, fill_before)
    list_dates = list_tokens_year(time_file, year_bkg_end=2010, year_trans_end=2023)[0]

    if year in list(list_dates.keys()):
        index_range = list_dates[year]
    else:
        index_range = list(range(len(file_to_smooth)))[list_dates[np.min(list(distance.keys()))][0]:]

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']

    plt.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
    if year in list(list_dates.keys()):
        plt.plot(time_file[list_dates[year][:][:len(distance[year].values())]], distance[year].values(), label='Distance')
    else:
        for y in list(distance.keys()):
            plt.plot(time_file[list_dates[y][:][:len(distance[y].values())]], distance[y].values(), label=('Deviation' if y==year_bkg_end else ''), color= colorcycle[0])

    for indx in rockfall_values(site)['time_index']:
        if indx in index_range:
            if int((num2date(time_file[indx], time_file.units)-rockfall_values(site)['datetime']).total_seconds()) == 0:
                plt.axvline(x = time_file[indx], color = 'r', linestyle='--', label = 'Landslide')

    for i in range(-3,4):
        plt.axhline(y = i, color = 'black', linestyle='--', linewidth=(1 if i ==0 else 0.5))
    # Because the data is stored in seconds since 0001-01-01 00:00:00 (ugh!!) 
    # we have to do some formatting of the x bar
    locs, labels = plt.xticks()

    if year in list(list_dates.keys()):
        locs = np.linspace(time_file[index_range[0]], time_file[index_range[-1]], num=12, endpoint=False)
        labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    else:
        locs = np.linspace(time_file[index_range[0]], time_file[list_dates[year_trans_end][0]], num=11, endpoint=True)
        labels = list(range(year_bkg_end,year_trans_end))
    plt.xticks(locs, labels)

    plt.title(f"Deviation from {name_series} data at %s%s" % (f"{site}", f" in {year}" if year in list(list_dates.keys()) else ""))
    plt.xlabel('Date')
    plt.ylabel((f'{name_series} deviation [$\sigma$]').capitalize())
    plt.legend()
    plt.show()

    # Closing figure
    plt.close()
    plt.clf()

def plot_aggregating_distance_temp_mean_reanalyses_generic(name_series, time_files, files_to_smooth, window, year=0, year_bkg_end=2010, year_trans_end=2023, fill_before=False):

    mean_plot = mean_all_reanalyses_generic(time_files, files_to_smooth, year_bkg_end, year_trans_end)

    plot_aggregating_distance_temp_generic(name_series, time_files[0], mean_plot, window, year, year_bkg_end, year_trans_end, fill_before=False)

def plot_aggregating_distance_temp_all_generic(yaxes, xdata, ydata, window, site, year, year_bkg_end=2010, year_trans_end=2023, fill_before=False):
    """ Plots the distance to the mean in units of standard deviation for a specific year or for the whole length
        Vertical subplots for different variables
    
    Parameters
    ----------
    yaxes : list of str
        List of names of quantities to plot, e.g. ['Air temperature', 'Water production', 'Ground temperature', 'Depth of thaw']
        Will be used for labels
    xdata : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored, e.g. [time_air_era5, time_air_era5, time_ground, time_ground]
        len(xdata) should be equal to len(yaxes) 
    ydata : list of list
        List of time series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that each individual time series needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts temp_air_era5[:,0] but not temp_air_era5
        example: [mean_air_temp, mean_air_temp, temp_ground_mean, temp_ground_mean]
        len(ydata) should be equal to len(yaxes)
    window : list of str
        Time window of datapoints to average over.
        Can take the following arguments: 'day', 'week', 'year', 'climate'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    year : int, optional
        One can choose to display a specific year or the whole set by choosing an integer not in the study sample, e.g. 0.
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    fill_before : bool, optional
        Option to fill the first empty values of the smoothed data with the first running average value if True

    Returns
    -------
        plot containing len(yaxes) subplots sharing the x axis

    """
    num_rows = len(xdata)
    num_cols = len(window)

    f, a = plt.subplots(num_rows, num_cols, figsize=(8, 2*num_rows), sharey='row')
    for idx,ax in enumerate(a):
        distance = [aggregating_distance_temp_generic(xdata[idx], ydata[idx], i, year, year_bkg_end, year_trans_end, fill_before) for i in window]
        list_dates = list_tokens_year(xdata[idx], year_bkg_end, year_trans_end)[0]

        if year in list(list_dates.keys()):
            index_range = list_dates[year]
        else:
            index_range = list(range(len(ydata[idx])))[list_dates[np.min(list(distance[0].keys()))][0]:]

        colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']

        if year in list(list_dates.keys()):
            if num_cols == 1:
                ax.plot(xdata[idx][list_dates[year][:][:len(distance[0][year].values())]], distance[0][year].values(), label='Deviation')
                ax.fill_between(xdata[idx][list_dates[year][:][:len(distance[0][year].values())]], -2, 2, alpha = 0.2, color = 'blue')
            else:
                for i in range(num_cols):
                    ax[i].plot(xdata[idx][list_dates[year][:][:len(distance[i][year].values())]], distance[i][year].values(), label='Deviation')
                    ax[i].fill_between(xdata[idx][list_dates[year][:][:len(distance[i][year].values())]], -2, 2, alpha = 0.2, color = 'blue')
        else:
            if num_cols == 1:
                for y in list(distance[0].keys()):
                    ax.plot(xdata[idx][list_dates[y][:][:len(distance[0][y].values())]], distance[0][y].values(), label=('Deviation' if y==year_bkg_end else ''), color= colorcycle[0])
                    ax.fill_between(xdata[idx][list_dates[y][:][:len(distance[0][y].values())]], -2, 2, alpha = 0.2, color = 'blue')
            else:
                for i in range(num_cols):
                    for y in list(distance[i].keys()):
                        ax[i].plot(xdata[idx][list_dates[y][:][:len(distance[i][y].values())]], distance[i][y].values(), label=('Deviation' if y==year_bkg_end else ''), color= colorcycle[0])
                        ax[i].fill_between(xdata[idx][list_dates[y][:][:len(distance[i][y].values())]], -2, 2, alpha = 0.2, color = 'blue')

        for indx in rockfall_values(site)['time_index']:
            if indx in index_range:
                if int((num2date(xdata[idx][indx], xdata[idx].units)-rockfall_values(site)['datetime']).total_seconds()) == 0:
                    if num_cols == 1:
                        ax.axvline(x = xdata[idx][indx], color = 'r', linestyle='--', label = 'Landslide')
                    else:
                        for i in range(num_cols):
                            ax[i].axvline(x = xdata[idx][indx], color = 'r', linestyle='--', label = 'Landslide')
        if num_cols == 1:
            ax.axhline(y = 0, color = 'black', linestyle='--', linewidth=1)
        else:
            for i in range(num_cols):
                ax[i].axhline(y = 0, color = 'black', linestyle='--', linewidth=1)

        if year in list(list_dates.keys()):
            locs = np.linspace(xdata[idx][index_range[0]], xdata[idx][index_range[-1]], num=12, endpoint=False)
            if num_cols == 1:
                labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            else:
                labels = ['Jan','','Mar','','May','','Jul','','Sep','','Nov','']
        else:
            # locs = np.linspace(xdata[idx][index_range[0]], xdata[idx][index_range[-1]], num=14, endpoint=True)
            loc_init = xdata[idx][list_dates[year_bkg_end][0]]
            len_year = xdata[idx][list_dates[year_bkg_end+1][0]] - loc_init
            num_year = year_trans_end - year_bkg_end
            locs = np.linspace(loc_init, loc_init + num_year * len_year, num=num_year+1, endpoint=True)
            labels = list(range(year_bkg_end,year_trans_end+1,1))

        if idx < num_rows-1:
            labels_end = ['']*len(labels)
        else:
            labels_end = labels
        if num_cols == 1:
            ax.set_xticks(locs, labels)
            ax.set_ylabel(yaxes[idx])
            ax.grid(axis = 'y')
        else: 
            for i in range(num_cols):
                ax[i].set_xticks(locs, labels_end)
                ax[i].grid(axis = 'y')
                if idx == 0:
                    ax[i].set_title(window[i].capitalize())
            ax[0].set_ylabel(yaxes[idx])
            ax[1].yaxis.set_tick_params(labelleft=False)

    f.supylabel(r'Normalized deviation $d_{norm}$ [$\sigma$]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.clf()

def plot_table_mean_GST_aspect_slope_generic(df_stats, site, altitude, background=True, box=True):
    """ Function returns a plot of the table of either mean background GST (ground-surface temperature)
        or its evolution vetween the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    altitude : int
        Altitude at which we want the table
    background : bool, optional 
        If True, plots the mean background value, else, plots the evolution of the mean
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box


    Returns
    -------
    Table
    """
    
    rf_values = rockfall_values(site)
    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}
    
    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if altitude == j:
            alt_index = i

    list_mean = (table_background_evolution_mean_GST_aspect_slope(df_stats)[1][alt_index] if background else table_background_evolution_mean_GST_aspect_slope(df_stats)[3][alt_index])
    df_temp = pd.DataFrame(list_mean, index=list(dic_var['slope']), columns=list(dic_var['aspect']))

    vals = np.around(df_temp.values, 2)
    vals = vals[~np.isnan(vals)]
    dilute = 1
    min_vals = np.min(vals)
    max_vals = np.max(vals)
    range_vals = max_vals- min_vals
    normal = plt.Normalize((np.max(max_vals - 2*dilute*range_vals,0) if np.min(vals)>0 else -dilute*np.max(np.abs(vals))), dilute*np.max(np.abs(vals)))
    colours = plt.cm.seismic(normal(df_temp))

    for i in range(len(list_mean)):
        for j in range(len(list_mean[0])):
            if np.isnan(list_mean[i][j]):
                colours[i][j] = list(colors.to_rgba('silver'))

    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    the_table=plt.table(cellText=list_mean, rowLabels=df_temp.index, colLabels=df_temp.columns, 
                        loc='center', cellLoc='center',
                        cellColours=colours)
    
    if box:
        if altitude == rockfall_values(site)['altitude']:
            ax.add_patch(Rectangle(((rf_values['aspect'])/45*1/8, (70-rf_values['slope'])/10*1/6), 1/8, 1/6,
                        edgecolor = 'black', transform=ax.transAxes,
                        fill=False,
                        lw=4))

    the_table.scale(1, 3.7)
    the_table.set_fontsize(16)
    ax.axis('off')

    plt.text(-0.05, 5/12,'Slope [°]', fontsize= 16, rotation=90, horizontalalignment='right', verticalalignment='center')
    plt.text(0.5, 1,'Aspect [°]', fontsize= 16, rotation=0, horizontalalignment='center', verticalalignment='bottom')

    plt.tight_layout()
    plt.show()
    plt.show()
    plt.close()
    plt.clf()

def plot_table_aspect_slope_all_altitudes_generic(df, df_stats, site, show_glacier=True, box=True):
    """ Function returns 1 plot per altitude of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Panda DataFrame df, should at least include the following columns: 'altitude', 'aspect', 'slope'
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    show_glacier : bool, opional
        Whether or not to plot the glacier fraction
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    (2 or 3)*(# altitudes) tables
    """
    
    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}
    # number of simulations for a given cell with fixed (altitude, slope, aspect) triplet
    sim_per_cell = len(df)/np.prod([len(dic_var[i]) for i in dic_var.keys()])

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if box:
            if rockfall_values(site)['altitude'] == j:
                alt_index = i

    # setting the parameter values 
    annot = True
    center = [0, 0]
    cmap = ['seismic', 'seismic']

    table_all = table_background_evolution_mean_GST_aspect_slope(df_stats)

    list_mean = [# evolution mean GST
                table_all[3],
                # background mean GST
                table_all[1]]
    labels_plot = ['Mean GST evolution [°C]', 'Mean background GST [°C]']
    if show_glacier:
        # glacier fraction
        list_mean.append([[[int((sim_per_cell-k)/sim_per_cell*100) for k in j] for j in i] for i in table_all[4]])
        labels_plot.append('Glacier fraction')
        cmap.append('BrBG')
        center.append(50)
    data = [[pd.DataFrame(i, index=list(dic_var['slope']), columns=list(dic_var['aspect'])) for i in j] for j in list_mean]

    nrows= len(data)
    ncols = len(data[0])

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3*nrows), constrained_layout=True)
    no_nan = [[l for j in i for k in j.values for l in k if not np.isnan(l)] for i in data]
    vmin = [np.min(i) for i in no_nan]
    vmax = [np.max(i) for i in no_nan]
    if len(vmin) == 3:
        vmin[2] = 0
        vmax[2] = 100

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    # plotting the heatmap 
    for j in range(nrows):
        if ncols == 1:
            sn.heatmap(data=data[nrows-1-j][0], annot=annot, center=center[nrows-1-j], cmap=cmap[nrows-1-j], ax=axs[j], vmin=vmin[nrows-1-j], vmax=vmax[nrows-1-j],
                    cbar=True, yticklabels=True, xticklabels=(j==nrows-1), cbar_kws={'label': labels_plot[nrows-1-j]})
            axs[0].figure.axes[-1].yaxis.label.set_size(13)
            if box:
                axs[j].add_patch(Rectangle(((rockfall_values(site)['aspect'])/45*1/8, (70-rockfall_values(site)['slope'])/10*1/5), 1/8, 1/5,
                                        edgecolor = 'black', transform=axs[j].transAxes, fill=False, lw=4))
        if ncols > 1:
            for i in range(ncols):
                sn.heatmap(data=data[nrows-1-j][i], annot=annot, center=center[nrows-1-j], cmap=cmap[nrows-1-j], ax=axs[j,i], vmin=vmin[nrows-1-j], vmax=vmax[nrows-1-j],
                            cbar=(i==ncols-1), yticklabels=(i==0), xticklabels=(j==nrows-1), cbar_kws={'label': labels_plot[nrows-1-j]})
            axs[0,0].figure.axes[-1].yaxis.label.set_size(13)
            if box:
                axs[j,alt_index].add_patch(Rectangle(((rockfall_values(site)['aspect'])/45*1/8, (70-rockfall_values(site)['slope'])/10*1/5), 1/8, 1/5,
                                                    edgecolor = 'black', transform=axs[j,alt_index].transAxes, fill=False, lw=4))

    if ncols == 1:
        axs[0].set_title('%s m' % alt_list[0])
    if ncols > 1:
        for i in range(ncols):
            axs[0,i].set_title('%s m' % alt_list[i])
    fig.supxlabel('Aspect [°]')
    fig.supylabel('Slope [°]')

    # displaying the plotted heatmap 
    plt.show()
    plt.close()
    plt.clf() 
 
def plot_table_aspect_slope_all_altitudes_polar_generic(df_stats, site, box=True):
    """ Function returns 3 polar plots (1 per altitude) of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    2*3 polar plots
    """

    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if box:
            if rockfall_values(site)['altitude'] == j:
                alt_index = i

    list_mean = [table_background_evolution_mean_GST_aspect_slope(df_stats)[1], table_background_evolution_mean_GST_aspect_slope(df_stats)[3]]
    data = [[pd.DataFrame(i, index=list(dic_var['slope']), columns=list(dic_var['aspect'])) for i in j] for j in list_mean]
    no_nan = [[l for j in i for k in j.values for l in k if not np.isnan(l)] for i in data]
    vmin = [np.min(i) for i in no_nan]
    vmax = [np.max(i) for i in no_nan]
    
    nbin_x = len(list_mean[0][0])
    nbin_y = len(list_mean[0][0][0])

    # binning
    rbins = np.linspace(30, 70, nbin_x)
    abins = np.linspace(0, (315/360)*2*np.pi, nbin_y)
    subdivs = 100
    abins2 = np.linspace((0-45/2)/360*2*np.pi, (360-45/2)/360*2*np.pi, subdivs*nbin_y)
    A, R = np.meshgrid(abins2, rbins)

    ncols = len(data[0])
    nrows = len(data)

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 5*nrows), subplot_kw=dict(projection="polar"),
                            gridspec_kw={'hspace': -0.1, 'wspace': 0.3})
    abs_max = [np.max(np.abs([vmin[i], vmax[i]])) for i in range(len(vmin))]

    cmap = [cmr.get_sub_cmap('seismic', 0.5 + 0.5*(vmin[i]/abs_max[i]), 0.5 + 0.5*(vmax[i]/abs_max[i])) for i in range(len(vmin))]

    ticks = [[],[]]
    tick_pos = [[],[]]
    for i in range(len(vmin)):
        for space in [0.05, 0.1, 0.2, 0.5, 1]:
            pre_ticks = np.arange(np.ceil(vmin[i]/space)*space+0, np.min([vmax[i], np.floor(vmax[i]/space+1)*space]), space)
            if (len(pre_ticks)>=5 and len(pre_ticks)<10):
                ticks[i] = [round(j,(2 if space<0.1 else 1)) for j in pre_ticks]
        tick_pos[i] = [(j-vmin[i])/(vmax[i]-vmin[i]) for j in ticks[i] if (j>=vmin[i] and j<vmax[i])]

    for j in range(nrows):
        for i in range(ncols):
            pc = axs[j,i].pcolormesh(A, R, np.repeat(list_mean[j][i], subdivs, axis=1), cmap=cmap[j], vmin=vmin[j], vmax=vmax[j])
            axs[j,i].set_facecolor("silver")
            axs[j,i].set_theta_zero_location('N')
            axs[j,i].set_theta_direction(-1)
            axs[j,i].tick_params(axis='y', labelcolor='black')
            axs[j,i].set_xticks(abins)
            axs[j,i].set_yticks([])
            axs[j,i].xaxis.grid(False)
            axs[j,i].yaxis.grid(False)
            axs[j,i].set_rlim(95,25)
            axs[0,i].set_title('%s m' % alt_list[i])
            axs[j,i].bar(0, 1).remove()
            for k in range(30,80,10):
                axs[j,i].text(np.pi/5, k, ('%s°' % k), horizontalalignment='center', verticalalignment='center')
        if box:
            axs[j,alt_index].add_patch(patches.Rectangle(((rockfall_values(site)['aspect']-45/2)/360*2*np.pi, rockfall_values(site)['slope']-5),
                                                        width=np.pi/4, height=10, edgecolor = 'black', fill=False, lw=2))
        
        
    cbar = [fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap[j]),
                         shrink=.8,
                         ax=axs[j,:].ravel().tolist(), orientation='vertical',
                         label=('Mean background GST [°C]' if j==0 else 'Mean GST evolution [°C]'))
            for j in range(nrows)]
    for idx,i in enumerate(range(-nrows,0)):
        axs[0,0].figure.axes[i].yaxis.set_ticks(tick_pos[idx])
        axs[0,0].figure.axes[i].set_yticklabels(ticks[idx]) 
    
    [axs[0,0].figure.axes[i].yaxis.label.set_size(15) for i in [6,7]]

    plt.show()
    plt.close()
    plt.clf()

def plot_permafrost_all_altitudes_polar_generic(df_stats, site, depth_thaw, box=True):
    """ Function returns 3 polar plots (1 per altitude) of the permafrost and glacier spatial distribution,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    depth_thaw : netCDF4._netCDF4.Variable
        NetCDF variable encoding the thaw depth
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    2*3 polar plots
    """

    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if box:
            if rockfall_values(site)['altitude'] == j:
                alt_index = i

    list_valid_sim = list(df_stats.index.values)
    list_no_perma = []
    for sim in list_valid_sim:
        if np.std(depth_thaw[sim,:]) < 1 and np.max(depth_thaw[sim,:])> 19:
            list_no_perma.append(sim)

    list_perma = list(set(list_valid_sim) - set(list_no_perma))
    check_perma_list = [[[[i in list_perma for i in list(df_stats[(df_stats['altitude']==altitude) & (df_stats['aspect']==aspect) & (df_stats['slope']==slope)].index.values)] for aspect in dic_var['aspect']]for slope in dic_var['slope']]  for altitude in alt_list]

    # 1/6=there is at least one glacier, 0=all simulations are valid and have permafrost, 5/6=all valid simulations but not all permafrost
    list_data = [[[(5/6 if len(k)<3 else (1/6 if all(k) else 1/2)) for k in j] for j in i] for i in check_perma_list]
    data = [pd.DataFrame(i, index=list(dic_var['slope']), columns=list(dic_var['aspect'])) for i in list_data]

    nbin_x = len(list_data[0])
    nbin_y = len(list_data[0][0])

    # binning
    rbins = np.linspace(30, 70, nbin_x)
    abins = np.linspace(0, (315/360)*2*np.pi, nbin_y)
    subdivs = 100
    abins2 = np.linspace((0-45/2)/360*2*np.pi, (360-45/2)/360*2*np.pi, subdivs*nbin_y)
    A, R = np.meshgrid(abins2, rbins)

    ncols = len(data)
    nrows = 1

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 5*nrows), subplot_kw=dict(projection="polar"),
                            gridspec_kw={'hspace': -0.1, 'wspace': 0.3})
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    cmap = matplotlib.colors.ListedColormap([colorcycle[1],colorcycle[2],colorcycle[0]]) 

    for i in range(ncols):
        pc = axs[i].pcolormesh(A, R, np.repeat(list_data[i], subdivs, axis=1), cmap=cmap, vmin=0, vmax=1)
        axs[i].set_facecolor("silver")
        axs[i].set_theta_zero_location('N')
        axs[i].set_theta_direction(-1)
        axs[i].tick_params(axis='y', labelcolor='black')
        axs[i].set_xticks(abins)
        axs[i].set_yticks([])
        axs[i].xaxis.grid(False)
        axs[i].yaxis.grid(False)
        axs[i].set_rlim(95,25)
        axs[i].set_title('%s m' % alt_list[i])
        axs[i].bar(0, 1).remove()
        for k in range(30,80,10):
            axs[i].text(np.pi/5, k, ('%s°' % k), horizontalalignment='center', verticalalignment='center')
    if box:
        axs[alt_index].add_patch(patches.Rectangle(((rockfall_values(site)['aspect']-45/2)/360*2*np.pi, rockfall_values(site)['slope']-5),
                                                        width=np.pi/4, height=10, edgecolor = 'black', fill=False, lw=2))

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap),
                            shrink=.8,
                            ax=axs[:].ravel().tolist(), orientation='vertical')
    axs[0].figure.axes[-1].yaxis.set_ticks([1/6,1/2,5/6])
    axs[0].figure.axes[-1].set_yticklabels(['Permafrost', 'No permafrost, no glaciers', 'Glaciers']) 
    axs[0].figure.axes[-1].yaxis.label.set_size(15)

    plt.show()
    plt.close()
    plt.clf()

def fit_stat_model_grd_temp_Aksaut(df_stats, all=True, diff_forcings=True):
    """ Function returns the value of the statistical model 
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'forcing', 'aspect', 'slope', 'bkg_grd_temp'
    all : bool, optional
        If True, considers all data at once
    diff_forcings : bool, optional
        If True, separates data by 'forcing'

    Returns
    -------
    xdata : list
        List of xdata (actual) grouped by forcing if all=False
    ydata : list
        List of ydata (predicted) grouped by forcing if all=False
    optimizedParameters : list
        List of optimized model parameters grouped by forcing if all=False
    pcov : list
        List of covariances grouped by forcing if all=False
    corr_matrix : list
        List of correlation matrices grouped by forcing if all=False
    R_sq : list
        List of R^2 grouped by forcing if all=False
    parity plot (predicted vs actual)
    """
    
    plt.figure(figsize=(6,6))

    forcings = ['merra2']
    data_set = []
    if all:
        data_set.append(df_stats)
    else: 
        pass
    if diff_forcings:
        for i in forcings:
            data_set.append(df_stats[df_stats['forcing']==i])
    else:
        pass

    input = [np.array([i['aspect'], i['slope'], i['altitude']]) for i in data_set]
    # all the measured differential warmings (from valid simulations) are in xdata
    xdata = [np.array(i['bkg_grd_temp']) for i in data_set]
    
    # The actual curve fitting happens here
    ydata = []
    R_sq = []
    optimizedParameters = []
    pcov = []
    corr_matrix = []
    bounds=((-50, -np.inf, -np.inf, -np.inf, -np.inf), (50, np.inf, np.inf, np.inf, np.inf))
    p0 = (0,0,1000,0,0)
    for i in range(len(input)):
        optimizedParameters.append(opt.curve_fit(stat_model_aspect_slope_alt, input[i], xdata[i], bounds=bounds, p0=p0)[0])
        pcov.append(opt.curve_fit(stat_model_aspect_slope_alt, input[i], xdata[i], bounds=bounds, p0=p0)[0])

        # this represents the fitted values, hence they are the statistically-modelled values 
        # of differential warming: we call them ydata
        ydata.append(stat_model_aspect_slope_alt(input[i], *optimizedParameters[i]))

        # R^2 from numpy package, to check!
        corr_matrix.append(np.corrcoef(xdata[i], ydata[i]))
        corr = corr_matrix[i][0,1]
        R_sq.append(corr**2)

        plt.scatter(xdata[i], ydata[i], marker=("D" if (all and i==0) else "o"),
                    s=20, label=('all data' if (all and i==0) else forcings[len(forcings)-len(data_set)+i]) )
        
    # plot the y=x diagonal
    # start by setting the bounds
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    lim_up = float("{:.2g}".format(max(np.max([np.max(i) for i in xdata]), np.max([np.max(i) for i in ydata]))))
    lim_down = float("{:.2g}".format(min(np.min([np.min(i) for i in xdata]), np.min([np.min(i) for i in ydata]))))
    x = np.arange(lim_down, lim_up, 0.01)
    plt.plot(x, x, color=colorcycle[len(data_set)], linestyle='dashed', label = 'y=x', linewidth=2)
    plt.legend(loc='upper right')

    margin = 0.1
    plt.ylim(ymin= lim_down - margin, ymax= lim_up + margin)
    plt.xlim(xmin= lim_down - margin, xmax= lim_up + margin)

    plt.xlabel(r'Numerically-simulated background GST $\overline{T_{\rm GST}^{\rm bkg}}_{(NS)}$ [°C]')
    plt.ylabel(r'Statistically-modelled background GST $\overline{T_{\rm GST}^{\rm bkg}}_{(SM)}$ [°C]')
    for i in range(len(R_sq)):
        plt.figtext(.7, .3 - i/30, f"$R^2$ = %s" % float("{:.2f}".format(R_sq[i])),
                    c=colorcycle[i])

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()

    return xdata, ydata, optimizedParameters, pcov, corr_matrix, R_sq

def plot_box_yearly_stat(name_series, time_file, file_to_plot, year_bkg_end=2010, year_trans_end=2023):
    """ Plots the distance to the mean in units of standard deviation for a specific year or for the whole length
    
    Parameters
    ----------
    name_series : str
        Name of the quantity to plot, has to be one of 'GST', 'Air temperature', 'Precipitation', 'SWE', 'Water production'
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored        
    file_to_plot : list
        Mean time series (temp_ground_mean, mean_air_temp, mean_prec, swe_mean, tot_water_prod)
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period

    Returns
    -------
        plot

    """

    fig, ax = plt.subplots()

    list_dates = list_tokens_year(time_file, year_bkg_end, year_trans_end)[0]
    overall_mean = np.mean(file_to_plot)
    exponent = int(np.floor(np.log10(np.abs(overall_mean))))
    a = []
    for i in list_dates.keys():
        if i < year_trans_end:
            a = a + [i]*len(list_dates[i])
    x = pd.DataFrame(file_to_plot[:len(a)], columns=[name_series], index=a)
    x['Year'] = a

    mean = [np.mean(x[x['Year']<year_bkg_end][name_series]), np.mean(x[(x['Year']>=year_bkg_end) & (x['Year']<year_trans_end)][name_series])]

    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
    sn.boxplot(x='Year', y=name_series, data=x, showmeans=True, showfliers=False, meanprops=meanpointprops, color='grey', linecolor='black')

    formatted_mean = ["{:.2e}".format(i) for i in mean] if ((exponent < -1) | (exponent>2)) else [float("{:.2f}".format(i)) for i in mean]
    print(formatted_mean)

    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm s-1', 'SWE': 'mm s-1', 'Water production': 'mm s-1'}
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    ax.hlines(mean[0], 0, year_bkg_end - list(list_dates.keys())[0] - 1 + 1/2, linewidth=2, color=colorcycle[0],
              label=f'Background mean: %s%s' % (formatted_mean[0], units[name_series]))
    ax.hlines(mean[1], year_bkg_end - list(list_dates.keys())[0] - 1/2, year_trans_end - list(list_dates.keys())[0] - 1, linewidth=2, color=colorcycle[1],
              label=f'Transient mean: %s%s' % (formatted_mean[1], units[name_series]))

    plt.tight_layout()
    locs, labels = plt.xticks()  # Get the current locations and labels.
    locs = locs[::2]
    labels = labels[::2]
    plt.xticks(locs, labels, rotation=0)
    plt.ylabel(name_series+' ['+units[name_series]+']')
    ax.ticklabel_format(axis='y', style='sci', useMathText=True, scilimits=(-3,3))

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def coordinates_percentile_cdf(data_sorted, proba_bins, percentile):
    """ Function returns coordinates of the point corresponing to the percentile of the 
        Cumulated Distribution Dunction (CDF)
        
    Parameters
    ----------
    data_sorted : numpy.ndarray
        Sorted data
    proba_bins : numpy.ndarray
        list of sorted probability bins, equally spaced points between 0 and 1 with number of bins equals number of data points
    percentile : int
        Percentile of the CDF

    Returns
    -------
    low_point : list
        Coordinates of the point corresponding to the given percentile on the CDF, in the form: [data, threshold]
    """

    threshold = percentile/100
    k=0
    while proba_bins[k] < threshold:
        k += 1
    if proba_bins[k] == threshold:
        low_point = [data_sorted[k], proba_bins[k]]
    low_point = [(threshold - proba_bins[k-1])/(proba_bins[k] - proba_bins[k-1])*(data_sorted[k] - data_sorted[k-1]) + data_sorted[k-1], threshold]

    return low_point

def plot_cdf_GST(df_stats):
    """ Function returns coordinates of the point corresponing to the percentile of the 
        Cumulated Distribution Dunction (CDF)
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'

    Returns
    -------
    2 plots: left panel shows the CDF of background, transient, and evolution of mean GST
             right panel shows the CDF of background, transient, and evolution of mean SO
    """
    
    # sort the data:
    data_bkg_sorted = np.sort(df_stats['bkg_grd_temp'])
    data_trans_sorted = np.sort(df_stats['trans_grd_temp'])
    data_evol_sorted = np.sort(df_stats['evol_grd_temp'])

    data_bkg_SO_sorted = np.sort(df_stats['bkg_SO'])
    data_trans_SO_sorted = np.sort(df_stats['trans_SO'])
    data_diff_warming_sorted = np.sort(df_stats['diff_warming'])


    # calculate the proportional values of samples
    p = np.linspace(0, 1, len(data_bkg_sorted))

    point = [[[] for _ in range(3)] for _ in range(4)]

    list_data = [data_bkg_sorted, data_trans_sorted, data_bkg_SO_sorted, data_trans_SO_sorted]
    list_quant = [10,50,90]

    for indx_i, i in enumerate(list_data):
        for indx_j, j in enumerate(list_quant):
            point[indx_i][indx_j] = coordinates_percentile_cdf(i, p, j)

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    # plot the sorted data:
    fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.plot(data_bkg_sorted, p, label='Background', color=colorcycle[0], linewidth=2)
    ax1.plot(data_trans_sorted, p, label='Transient', color=colorcycle[1], linewidth=2)
    ax1.plot(data_evol_sorted, p, label='Evolution', color=colorcycle[2], linewidth=2)

    for i in range(2):
        for j in range(3):
            ax1.scatter(point[i][j][0], point[i][j][1], color=colorcycle[i])

    ylim_ax1 = ax1.get_ylim()
    xlim_ax1 = ax1.get_xlim()
    ax1.axvline(x=0, color='grey', linestyle="dashed")


    ax1.hlines([i/100 for i in list_quant], xlim_ax1[0],
            [np.max(i) for i in np.array(point).transpose()[0].transpose()[:2].transpose()],
            color='black', linestyle="dashed", linewidth=0.5)
    for i in range(2):
            ax1.vlines([j[0] for j in point[i]], ylim_ax1[0],
                    [j/100 for j in list_quant], color=colorcycle[i], linestyle="dashed", linewidth=0.5)

    ax1.set_xlim(xlim_ax1)
    ax1.set_ylim(ylim_ax1)

    ax1.set_xlabel('Mean GST [°C]')
    ax1.set_ylabel('Probability')
    ax1.legend()

    ax2.plot(data_bkg_SO_sorted, p, label='Background SO', color=colorcycle[0], linewidth=2)
    ax2.plot(data_trans_SO_sorted, p, label='Transient SO', color=colorcycle[1], linewidth=2)
    ax2.plot(data_diff_warming_sorted, p, label='Evolution SO', color=colorcycle[2], linewidth=2)
    ylim_ax2 = ax2.get_ylim()
    xlim_ax2 = ax2.get_xlim()
    ax2.axvline(x=0, color='grey', linestyle="dashed")

    for i in range(2,4):
        for j in range(3):
            ax2.scatter(point[i][j][0], point[i][j][1], color=colorcycle[i-2])

    ax2.hlines([i/100 for i in list_quant], xlim_ax1[0],
            [np.max(i) for i in np.array(point).transpose()[0].transpose()[2:].transpose()],
            color='black', linestyle="dashed", linewidth=0.5)
    for i in range(2,4):
            ax2.vlines([j[0] for j in point[i]], ylim_ax1[0],
                    [j/100 for j in list_quant], color=colorcycle[i-2], linestyle="dashed", linewidth=0.5)

    ax2.set_xlim(xlim_ax2)
    ax2.set_ylim(ylim_ax2)

    ax2.set_xlabel('Mean SO [°C]')

    # Show the graph
    plt.show()
    plt.close()
    plt.clf()

def plot_10_cold_warm(df_stats):
    """ Function returns a plot of mean GST evolution vs background GST, with an emphasis on the 10% colder and warmer simulations
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'

    Returns
    -------
    Plot of mean GST evolution vs background GST, with an emphasis on the 10% colder and warmer simulations
    """

    table_all = table_background_evolution_mean_GST_aspect_slope(df_stats)

    # create some randomly ddistributed data:
    data_bkg = [l for i in table_all[0] for j in i for k in j for l in k]
    data_evol = [l for i in table_all[2] for j in i for k in j for l in k]
    data_trans = [data_bkg[i]+data_evol[i] for i in range(len(data_bkg))]

    # sort the data:
    data_bkg_sorted = np.sort(data_bkg)
    data_trans_sorted = np.sort(data_trans)

    x = [data_bkg_sorted, data_trans_sorted]
    proba_bins = np.linspace(0, 1, len(data_bkg))
    percentiles = [10, 25, 50, 75, 90]
    table_points = [[coordinates_percentile_cdf(data, proba_bins, p)[0] for data in x] for p in percentiles]

    pd_data = pd.DataFrame(table_points, index=percentiles, columns=list(['Background', 'Transient']))
    pd_data['Difference'] = pd_data['Transient'] - pd_data['Background']

    df_stats_bis = pd.DataFrame(data=df_stats, columns=['bkg_grd_temp', 'trans_grd_temp'])
    df_stats_bis['bkg_grd_temp'] = pd.Categorical(df_stats_bis['bkg_grd_temp'], data_bkg_sorted)
    df_stats_bis = df_stats_bis.sort_values('bkg_grd_temp')

    list_x = list(df_stats_bis['bkg_grd_temp'])
    list_y = [df_stats_bis['trans_grd_temp'].iloc[i] - df_stats_bis['bkg_grd_temp'].iloc[i] for i in range(len(df_stats_bis))]

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    pos_10 = int(np.ceil(len(data_bkg)/10))

    plt.scatter(list_x[:pos_10], list_y[:pos_10], c=colorcycle[0])
    plt.scatter(list_x[pos_10:-pos_10], list_y[pos_10:-pos_10], c=colorcycle[1])
    plt.scatter(list_x[-pos_10:], list_y[-pos_10:], c=colorcycle[2])

    mean = [np.mean(list_y[:pos_10]), np.mean(list_y[-pos_10:])]

    plt.axvline((list_x[pos_10-1]+list_x[pos_10])/2, c=colorcycle[0])
    plt.axvline((list_x[-pos_10-1]+list_x[-pos_10])/2, c=colorcycle[2])
    plt.axhline(mean[0], linestyle="dashed",
                label=(r"$\overline{{\rm GST}_{\rm low 10}}$ =" + ('+' if mean[0]>0 else '') + r"%.3f°C" % (mean[0])),
                c=colorcycle[0])
    plt.axhline(mean[1], linestyle="dashed",
                label=(r"$\overline{{\rm GST}_{\rm high 10}}$ =" + ('+' if mean[1]>0 else '') + r"%.3f°C" % (mean[1])),
                c=colorcycle[2])
    
    xlim = plt.gca().get_xlim()
    plt.axvspan(xlim[0], (list_x[pos_10-1]+list_x[pos_10])/2, facecolor=colorcycle[0], alpha=0.3)
    plt.axvspan(xlim[1], (list_x[-pos_10-1]+list_x[-pos_10])/2, facecolor=colorcycle[2], alpha=0.3)
    plt.gca().set_xlim(xlim)

    plt.xlabel('Mean background GST [°C]')
    plt.ylabel('Mean GST evolution [°C]')

    # annotation_string_1 = r"$\overline{{\rm GST}_{\rm low 10}}$ =" + ('+' if mean[0]>0 else '') + r"%.3f°C" % (mean[0])
    # annotation_string_2 = r"$\overline{{\rm GST}_{\rm high 10}}$ =" + ('+' if mean[1]>0 else '') + r"%.3f°C" % (mean[1])
    # plt.annotate(annotation_string_1, (0.5, 0.9), xycoords='axes fraction', va='center', ha='center', color=colorcycle[0])
    # plt.annotate(annotation_string_2, (0.5, 0.85), xycoords='axes fraction', va='center', ha='center', color=colorcycle[2])

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()
    
def heatmap_percentile_GST(df_stats):
    """ Function returns a heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'

    Returns
    -------
    Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    """

    table_all = table_background_evolution_mean_GST_aspect_slope(df_stats)

    # create some randomly ddistributed data:
    data_bkg = [l for i in table_all[0] for j in i for k in j for l in k]
    data_evol = [l for i in table_all[2] for j in i for k in j for l in k]
    data_trans = [data_bkg[i]+data_evol[i] for i in range(len(data_bkg))]

    # sort the data:
    data_bkg_sorted = np.sort(data_bkg)
    data_trans_sorted = np.sort(data_trans)

    x = [data_bkg_sorted, data_trans_sorted]
    proba_bins = np.linspace(0, 1, len(data_bkg))
    percentiles = [10, 25, 50, 75, 90]
    table_points = [[coordinates_percentile_cdf(data, proba_bins, p)[0] for data in x] for p in percentiles]

    percentiles.append('Mean')
    table_points.append([np.mean(data_bkg), np.mean(data_trans)])

    pd_data = pd.DataFrame(table_points, index=percentiles, columns=list(['Background', 'Transient']))
    pd_data['Difference'] = pd_data['Transient'] - pd_data['Background']

    pd_data['Background'] = pd.Categorical(pd_data['Background'], np.sort(pd_data['Background']))
    pd_data = pd_data.sort_values('Background')

    sn.heatmap(data=pd_data, annot=True, fmt=".2f", cmap='seismic',
                cbar=True, yticklabels=True, xticklabels=True, cbar_kws={'label': 'Mean GST [°C]'},
                norm=matplotlib.colors.TwoSlopeNorm(vmin=np.min([np.min(pd_data),-0.05]), vcenter=0, vmax=np.max([np.max(pd_data),0.05])))
    plt.ylabel('Percentile')

    # Show the graph
    plt.show()
    plt.close()
    plt.clf()

def stats_all_years_simulations_to_single_year(time_file, time_series, list_valid_sim, mask_period=None):
    """ Function returns daily mean and several quantiles of a multi-year timeseries over a 1-year period
        e.g. assigns the GST mean over all years and simulations on a given day, e.g. Dec 21st
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
    time_series : netCDF4._netCDF4.Variable
        Time series (could be temperature, precipitation, snow depth, etc.)
    list_valid_sim : list
        List of the indices of all valid simulations
    mask_period : numpy.ma.core.MaskedArray, opt
        Period could be 'background' or 'transient' and is a list of booleans, i.e. a mask.
        Can take values 'time_bkg_ground' or 'time_trans_ground' for instance.
        If None, it considers everything.

    Returns
    -------
    quantiles : pandas.core.frame.DataFrame
        Panda dataframe in the shape (5, 366)
        There is a row per quantile in [0.023, 0.16, 0.5, 0.84, 0.977] and a column per timestamp in the year (daily: 366)
    mean_end : pandas.core.series.Series
        One year time series of the mean

    """

    # create a panda dataframe with month, day, hour for each timestamp
    panda_test = pd.DataFrame(num2date(time_file[:], time_file.units), columns=['date'])
    panda_test['month'] = [i.month for i in panda_test['date']]
    panda_test['day'] = [i.day for i in panda_test['date']]
    panda_test['hour'] = [i.hour for i in panda_test['date']]
    panda_test = panda_test.drop(columns=['date'])

    if mask_period is not None:
        panda_test = panda_test.drop(index=np.arange(0,len(mask_period),1)[[not i for i in mask_period]])
 

    mean_test = []
    # for all simulations
    for sim in list_valid_sim:
        if mask_period is None:
            panda_test['timeseries'] = time_series[sim,:,0] if len(time_series.shape) == 3 else time_series[sim,:]
        else:
            panda_test['timeseries'] = time_series[sim,mask_period[:],0] if len(time_series.shape) == 3 else time_series[sim,mask_period[:]]
        # here we group the values of the multi-year timeseries by month, day, and hour and take the mean
        # this produces a single year average timeseries
        mean_test.append([i for i in panda_test.groupby(['month', 'day', 'hour']).mean().reset_index()['timeseries']])
        panda_test = panda_test.drop(columns=['timeseries'])

    mean_test_df = pd.DataFrame(np.array(mean_test))
    # 1-year timeseries of 5 quantiles corresponding to the median plus minus 1 and 2 sigmas
    quantiles = mean_test_df.quantile([0.023, 0.16, 0.5, 0.84, 0.977])
    # 1-year timeseries of the mean
    mean_end = mean_test_df.mean()

    return quantiles, mean_end

def stats_air_all_years_simulations_to_single_year(time_file, time_series, mask_period=None):
    """ Function returns daily mean and several quantiles of a multi-year timeseries over a 1-year period
        for atmospheric quantities (no simulations). The quantiles are over the years, not over simulations.
        e.g. assigns the GST mean over all years and simulations on a given day, e.g. Dec 21st
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
    time_series : netCDF4._netCDF4.Variable
        Time series (could be temperature, precipitation, snow depth, etc.)
    mask_period : numpy.ma.core.MaskedArray, opt
        Period could be 'background' or 'transient' and is a list of booleans, i.e. a mask.
        Can take values 'time_bkg_ground' or 'time_trans_ground' for instance.
        If None, it considers everything.

    Returns
    -------
    quantiles : pandas.core.frame.DataFrame
        Panda dataframe in the shape (5, 8784)
        There is a row per quantile in [0.023, 0.16, 0.5, 0.84, 0.977] and a column per timestamp in the year (hourly: 8784)
    mean_end : pandas.core.series.Series
        One year time series of the mean

    """

    # create a panda dataframe with month, day, hour for each timestamp
    panda_test = pd.DataFrame(num2date(time_file[:], time_file.units), columns=['date'])
    panda_test['month'] = [i.month for i in panda_test['date']]
    panda_test['day'] = [i.day for i in panda_test['date']]
    panda_test['hour'] = [i.hour for i in panda_test['date']]
    panda_test = panda_test.drop(columns=['date'])

    if mask_period is None:
        panda_test['timeseries'] = time_series[:,2]
    else:
        panda_test = panda_test.drop(index=np.arange(0,len(mask_period),1)[[not i for i in mask_period]])
        panda_test['timeseries'] = time_series[mask_period[:],2]


    mean_end = pd.DataFrame(np.array([i for i in panda_test.groupby(['month', 'day', 'hour']).mean().reset_index()['timeseries']]))

    list_quant = [0.023, 0.16, 0.5, 0.84, 0.977]

    panda_test = panda_test.groupby(['month', 'day', 'hour']).quantile(list_quant)
    panda_test.index.names = ['month', 'day', 'hour', 'quantiles']
    panda_test = panda_test.reset_index().drop(columns=['month', 'day', 'hour'])
    panda_test

    quantiles = []
    for i in range(len(list_quant)):
        quantiles.append(panda_test.loc[i::len(list_quant)]['timeseries'])

    quantiles = pd.DataFrame(np.array(quantiles))

    return quantiles, mean_end

def plot_sanity_one_year_quantiles_two_periods(time_file, time_series_list, list_valid_sim_list, axis_label, list_label, list_mask_period):
    """ Function returns a plot of 2 timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread.
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
    time_series_list : list of netCDF4._netCDF4.Variable
        List of time series (could be temperature, precipitation, snow depth, etc.)
    list_valid_sim_list : list of list
        List of list of the indices of all valid simulations
    axis_label : str
        Label of the y axis, could be 'GST' or 'Snow depth' for instance
    list_label : list
        List of the labels associated to each entry (if both entries are identical, can put a single label)
    list_mask_period : list
        List of the time masks for each entry, could be 'time_bkg_ground' or 'time_trans_ground' for instance.


    Returns
    -------
    Plot of 2 (or more?) timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread.
    Both series have their own y axis if they have different units.

    """
    
    # fig, ax1 = plt.subplots()
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}
    
    delta_hours = int((num2date(time_file[1], time_file.units)-num2date(time_file[0], time_file.units)).total_seconds()/3600)

    if delta_hours == 1:
        quantiles, mean_end = stats_air_all_years_simulations_to_single_year(time_file, time_series_list[0], list_mask_period[0])
    else:
        quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[0], list_valid_sim_list[0], list_mask_period[0])
    xdata = range(len(mean_end))

    indx=0
    plt.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_label[0])
    plt.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
    plt.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)
    plt.ylabel(axis_label+' ['+units[axis_label]+']')

    if delta_hours == 1:
        quantiles, mean_end = stats_air_all_years_simulations_to_single_year(time_file, time_series_list[1], list_mask_period[1])
    else:
        quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[1], list_valid_sim_list[1], list_mask_period[1])
    indx=1

    # ax2 = ax1.twinx()
    plt.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_label[1])
    plt.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
    plt.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)
    # ax2.set_ylabel(axis_label+' ['+units[axis_label]+']')

    # mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.5)

    if axis_label in ['GST', 'Air temperature']:
        plt.axhline(y=0, color='grey', linestyle='dashed')

    locs, labels = plt.xticks()
    locs = np.linspace(0, len(mean_end), num=13, endpoint=True)
    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan']
    plt.xticks(locs, labels)

    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.legend(loc="upper right")
    plt.show()
    plt.close()
    plt.clf()

def plot_sanity_two_variables_one_year_quantiles(time_file, time_series_list, list_valid_sim_list, list_label, list_site=None):
    """ Function returns a plot of 2 timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread.
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
    time_series_list : list of netCDF4._netCDF4.Variable
        List of time series (could be temperature, precipitation, snow depth, etc.)
    list_valid_sim_list : list of list
        List of list of the indices of all valid simulations
    list_label : list
        List of the labels associated to each entry (if both entries are identical, can put a single label)
    list_site : list, optional
        List of labels for the site of each entry
        If both labels are identical (i.e. if only 1 label is provided in list_label) then this means we are plotting 
        the same quantity for different sites, in which case we want to label the curves by their site

    Returns
    -------
    Plot of 2 (or more?) timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread.
    Both series have their own y axis if they have different units.

    """
    
    fig, ax1 = plt.subplots()
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}

    quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[0], list_valid_sim_list[0])
    xdata = range(len(mean_end))

    indx=0
    ax1.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_site[0] if len(list_label) == 1 else list_label[0])
    ax1.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
    ax1.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)
    ax1.set_ylabel(list_label[0]+' ['+units[list_label[0]]+']')

    quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[1], list_valid_sim_list[1])
    indx=1

    if len(list_label) == 1:
        ax1.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_site[1])
        ax1.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
        ax1.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)

    else:
        ax2 = ax1.twinx()
        ax2.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_label[1])
        ax2.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
        ax2.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)
        ax2.set_ylabel(list_label[1]+' ['+units[list_label[1]]+']')

        mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.5)

    for i in list_label:
        if i in ['GST', 'Air temperature']:
            plt.axhline(y=0, color='grey', linestyle='dashed')

    locs, labels = plt.xticks()
    locs = np.linspace(0, len(mean_end), num=13, endpoint=True)
    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan']
    plt.xticks(locs, labels)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()
    plt.close()
    plt.clf()

def plot_sanity_two_variables_one_year_quantiles_side_by_side(time_file, time_series_list, list_valid_sim_list, list_label, list_site):
    """ Function returns 2 plots side by side of 2 timeseries each reduced to a 1-year window with mean and 1 and 2-sigma spread.
        Each plot is a plot of two timeseries of the same variable at two different sites
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
    time_series_list : list of netCDF4._netCDF4.Variable
        List of time series (could be temperature, precipitation, snow depth, etc.). 1 per plot.
    list_valid_sim_list : list of list
        List of list of the indices of all valid simulations
    list_label : list
        List of the labels associated to each plot
    list_site : list
        List of labels for the site of each entry

    Returns
    -------
    Plot of 2 (or more?) timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread.
    Both series have their own y axis if they have different units.

    """
    
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}

    f, a = plt.subplots(1, 2, figsize=(10, 5))
    for idx,ax in enumerate(a):

        quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[idx][0], list_valid_sim_list[0])
        xdata = range(len(mean_end))

        indx=0
        ax.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_site[indx])
        ax.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
        ax.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)
        ax.set_ylabel(list_label[idx]+' ['+units[list_label[idx]]+']')

        quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[idx][1], list_valid_sim_list[1])
        indx=1

        ax.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_site[indx])
        ax.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
        ax.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)

        if list_label[idx] in ['GST', 'Air temperature']:
            ax.axhline(y=0, color='grey', linestyle='dashed')

        locs = np.linspace(0, len(mean_end), num=13, endpoint=True)
        labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan']
        ax.set_xticks(locs)
        ax.set_xticklabels(labels)

        if idx == 1:
            ax.legend(loc="upper right")


    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.show()
    plt.close()
    plt.clf()

def stats_yearly_quantiles_air(time_file, time_series, label_plot, year_trans_end):
    """ Function returns yearly statistics for 'air' timeseries
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (air time)
    time_series : netCDF4._netCDF4.Variable
        List of time series with air time shape (could be air temperature, precipitation, SW, etc.)
    label_plot : str
        label associated to the plot, if 'Precipitation' or 'Water production', rescales data to mm/day
    year_trans_end : int
        Transient period is BEFORE the start of the year corresponding to the variable
    
    Returns
    -------
    panda_test : pandas.core.frame.DataFrame
        Panda dataframe of the timeseries value per day, grouped per year, month, and day
    quantiles : pandas.core.frame.DataFrame
        Panda dataframe of the timeseries [0.023, 0.16, 0.5, 0.84, 0.977] quantiles per year.
        The lenght of the table is #quantiles x #years
    mean_end : pandas.core.series.Series
        Panda dataframe of mean of the timeseries for each year
    dict_indices_quantiles : dict
        Dictionary assigning all the rows of 'quantiles' (dataframe) having information about quantile n to quantile n
    xdata : numpy.ndarray
        List of years
    """

    # create a panda dataframe with month, day, hour for each timestamp
    panda_test = pd.DataFrame(num2date(time_file[:], time_file.units), columns=['date'])
    panda_test['year'] = [i.year for i in panda_test['date']]
    panda_test['month'] = [i.month for i in panda_test['date']]
    panda_test['day'] = [i.day for i in panda_test['date']]
    panda_test['hour'] = [i.hour for i in panda_test['date']]
    panda_test = panda_test.drop(columns=['date', 'hour'])
    # Note that this is selecting the elevation in the 'middle': index 2 in the list [0,1,2,3,4]
    alt_index = int(np.floor((time_series.shape[1]-1)/2))
    panda_test['timeseries'] = time_series[:,alt_index]

    panda_test = panda_test.groupby(['year', 'month', 'day']).mean()

    if label_plot in ['Precipitation', 'Water production']:
        panda_test['timeseries'] = panda_test['timeseries']*86400

    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]
    
    list_drop = [i for i in np.unique(panda_test.index.get_level_values('year')) if i >= year_trans_end]

    quantiles = panda_test.groupby(['year']).quantile(list_quantiles).drop(index=list_drop)
    quantiles.index.names = ['year','quantile']
    quantiles = quantiles.swaplevel()
    dict_indices_quantiles = quantiles.groupby(['quantile']).indices
    mean_end = panda_test.groupby(['year']).mean().drop(index=list_drop)
    xdata = np.array(mean_end.index)

    return panda_test, quantiles, mean_end, dict_indices_quantiles, xdata

def plot_yearly_quantiles_air(time_file, time_series, label_plot, year_bkg_end, year_trans_end):
    """ Function plots yearly statistics for 'air' timeseries
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (air time)
    time_series : netCDF4._netCDF4.Variable
        List of time series with air time shape (could be air temperature, precipitation, SW, etc.)
    label_plot : str
        label associated to the plot, if 'Precipitation' or 'Water production', rescales data to mm/day
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    
    Returns
    -------
    Plot of yearly statistics for 'air' timeseries. Mean and several quantiles for each year.
    """

    panda_test, quantiles, mean_end, dict_indices_quantiles, xdata = stats_yearly_quantiles_air(time_file, time_series, label_plot, year_trans_end)
    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]

    mean_bkg = np.mean(mean_end.loc[xdata[0]:year_bkg_end-1])
    mean_trans = np.mean(mean_end.loc[year_bkg_end:year_trans_end-1])
    formatted_mean = ["{:.2f}".format(i) for i in [mean_bkg, mean_trans]]

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}
    dict_points = {0: {'alpha': 0.2, 'width': 0.5},
                1: {'alpha': 0.4, 'width': 1.0},
                2: {'alpha': 1.0, 'width': 2.0},
                3: {'alpha': 0.4, 'width': 1.0},
                4: {'alpha': 0.2, 'width': 0.5}}

    plt.scatter(xdata, mean_end, color=colorcycle[0], linestyle='None', label='Yearly mean')
    # plt.plot(xdata, mean_end, color=colorcycle[0], label='Mean')
    # plt.plot(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[2]]]['timeseries'], color=colorcycle[0])
    for i in [0,1,3,4]:
        plt.scatter(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[i]]]['timeseries'], color=colorcycle[0], alpha=dict_points[i]['alpha'], linewidth=dict_points[i]['width'])
    plt.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[1]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[3]]]['timeseries'],
                        alpha = 0.4, color=colorcycle[0], linewidth=1,
                        # label='Quantiles 16-84'
                        )
    plt.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[0]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[4]]]['timeseries'],
                        alpha = 0.2, color=colorcycle[0], linewidth=0.5,
                        # label='Quantiles 2.3-97.7'
                        )
    
    plt.hlines(mean_bkg, xdata[0], year_bkg_end, color=colorcycle[1],
               label=f'Background mean: %s%s' % (formatted_mean[0], units[label_plot]))
    plt.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[2],
               label=f'Transient mean: %s%s' % (formatted_mean[1], units[label_plot]))

    ylim = plt.gca().get_ylim()

    plt.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')

    plt.gca().set_ylim(ylim)

    if label_plot in ['GST', 'Air temperature']:
        plt.axhline(y=0, color='grey', linestyle='dashed')
    
    plt.ylabel(label_plot+' ['+units[label_plot]+']')

    # plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
    plt.clf()

def plot_yearly_quantiles_all_sims(time_file, time_series, list_valid_sim, label_plot, year_bkg_end, year_trans_end):
    """ Function plots yearly statistics for 'ground' timeseries over all simulations
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (ground time)
    time_series : netCDF4._netCDF4.Variable
        List of time series with ground time shape (could be ground temperature, SWE, etc.)
    list_valid_sim : list
        List of the indices of all valid simulations
    label_plot : str
        label associated to the plot.
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    
    Returns
    -------
    Plot of yearly statistics for 'ground' timeseries. Mean and several quantiles for each year.
    """

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}
    
    dict_points = {0: {'alpha': 0.2, 'width': 0.5},
                1: {'alpha': 0.4, 'width': 1.0},
                2: {'alpha': 1.0, 'width': 2.0},
                3: {'alpha': 0.4, 'width': 1.0},
                4: {'alpha': 0.2, 'width': 0.5}}

    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]
    
    long_timeseries = []
    for sim in list_valid_sim:
        long_timeseries.append(time_series[sim,:,0] if len(time_series.shape) == 3 else time_series[sim,:])

    long_timeseries = np.array(long_timeseries).flatten()
    long_years = np.array([int(i.year) for i in num2date(time_file[:], time_file.units)]*len(list_valid_sim))

    panda_test = pd.DataFrame(long_years.transpose(), columns=['year'])
    panda_test['timeseries'] = long_timeseries.transpose()

    list_drop = [i for i in np.unique(panda_test['year']) if i >= year_trans_end]

    quantiles = panda_test.groupby(['year']).quantile(list_quantiles).drop(index=list_drop)
    quantiles.index.names = ['year','quantile']
    quantiles = quantiles.swaplevel()
    dict_indices_quantiles = quantiles.groupby(['quantile']).indices
    mean_end = panda_test.groupby(['year']).mean().drop(index=list_drop)
    xdata = np.array(mean_end.index)

    mean_bkg = np.mean(mean_end.loc[xdata[0]:year_bkg_end-1])
    mean_trans = np.mean(mean_end.loc[year_bkg_end:year_trans_end-1])
    formatted_mean = ["{:.2f}".format(i) for i in [mean_bkg, mean_trans]]
    
    plt.scatter(xdata, mean_end, color=colorcycle[0], linestyle='None', label='Yearly mean')
    # plt.plot(xdata, mean_end, color=colorcycle[0], label='Mean')
    # plt.plot(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[2]]]['timeseries'], color=colorcycle[0])
    for i in [0,1,3,4]:
        plt.scatter(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[i]]]['timeseries'], color=colorcycle[0], alpha=dict_points[i]['alpha'], linewidth=dict_points[i]['width'])
    plt.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[1]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[3]]]['timeseries'],
                        alpha = 0.4, color=colorcycle[0], linewidth=1,
                        # label='Quantiles 16-84'
                        )
    plt.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[0]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[4]]]['timeseries'],
                        alpha = 0.2, color=colorcycle[0], linewidth=0.5,
                        # label='Quantiles 2.3-97.7'
                        )
    if label_plot in ['GST', 'Air temperature']:
        plt.axhline(y=0, color='grey', linestyle='dashed')

    plt.hlines(mean_bkg, xdata[0], year_bkg_end, color=colorcycle[1],
               label=f'Background mean: %s%s' % (formatted_mean[0], units[label_plot]))
    plt.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[2],
               label=f'Transient mean: %s%s' % (formatted_mean[1], units[label_plot]))

    ylim = plt.gca().get_ylim()
    plt.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')
    plt.gca().set_ylim(ylim)

    plt.ylabel(label_plot+' ['+units[label_plot]+']')

    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
    plt.clf()

def plot_yearly_quantiles_all_sims_side_by_side(time_file, time_series, list_valid_sim, label_plot, list_site, year_bkg_end, year_trans_end):
    """ Function plots yearly statistics for 'ground' timeseries over all simulations for 1 same metric
        at 2 different sites. 1 plot per site, both plots side by side.
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (ground time)
    time_series : netCDF4._netCDF4.Variable
        List of time series with ground time shape (could be ground temperature, SWE, etc.)
    list_valid_sim : list
        List of the indices of all valid simulations
    label_plot : str
        label associated to the plot.
    list_site : list
        List of labels for the site of each entry
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    
    Returns
    -------
    Plots of yearly statistics for 'ground' timeseries over all simulations for 1 same metric
    at 2 different sites. 1 plot per site, both plots side by side.
    """

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}
    
    dict_points = {0: {'alpha': 0.2, 'width': 0.5},
                1: {'alpha': 0.4, 'width': 1.0},
                2: {'alpha': 1.0, 'width': 2.0},
                3: {'alpha': 0.4, 'width': 1.0},
                4: {'alpha': 0.2, 'width': 0.5}}

    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]

    f, a = plt.subplots(1, 2, figsize=(8, 4), sharey='row')
    for idx,ax in enumerate(a):
        long_timeseries = []
        for sim in list_valid_sim[idx]:
            long_timeseries.append(time_series[idx][sim,:,0] if len(time_series[idx].shape) == 3 else time_series[idx][sim,:])

        long_timeseries = np.array(long_timeseries).flatten()
        long_years = np.array([int(i.year) for i in num2date(time_file[:], time_file.units)]*len(list_valid_sim[idx]))

        panda_test = pd.DataFrame(long_years.transpose(), columns=['year'])
        panda_test['timeseries'] = long_timeseries.transpose()

        list_drop = [i for i in np.unique(panda_test['year']) if i >= year_trans_end]

        quantiles = panda_test.groupby(['year']).quantile(list_quantiles).drop(index=list_drop)
        quantiles.index.names = ['year','quantile']
        quantiles = quantiles.swaplevel()
        dict_indices_quantiles = quantiles.groupby(['quantile']).indices
        mean_end = panda_test.groupby(['year']).mean().drop(index=list_drop)
        xdata = np.array(mean_end.index)

        mean_bkg = np.mean(mean_end.loc[xdata[0]:year_bkg_end-1])
        mean_trans = np.mean(mean_end.loc[year_bkg_end:year_trans_end-1])
        formatted_mean = ["{:.2f}".format(i) for i in [mean_bkg, mean_trans]]
        
        ax.scatter(xdata, mean_end, color=colorcycle[idx], linestyle='None', label='Yearly mean')
        # ax.plot(xdata, mean_end, color=colorcycle[idx], label='Mean')
        # plt.plot(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[2]]]['timeseries'], color=colorcycle[0])
        for i in [0,1,3,4]:
            ax.scatter(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[i]]]['timeseries'], color=colorcycle[idx], alpha=dict_points[i]['alpha'], linewidth=dict_points[i]['width'])
        ax.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[1]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[3]]]['timeseries'],
                            alpha = 0.4, color=colorcycle[idx], linewidth=1,
                            # label='Quantiles 16-84'
                            )
        ax.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[0]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[4]]]['timeseries'],
                            alpha = 0.2, color=colorcycle[idx], linewidth=0.5,
                            # label='Quantiles 2.3-97.7'
                            )

        ax.hlines(mean_bkg, xdata[0], year_bkg_end, color=colorcycle[2],
               label=f'Background mean: %s%s' % (formatted_mean[0], units[label_plot]))
        ax.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[3],
               label=f'Transient mean: %s%s' % (formatted_mean[1], units[label_plot]))
            
        if label_plot in ['GST', 'Air temperature']:
            ax.axhline(y=0, color='grey', linestyle='dashed')

        ax.title.set_text(list_site[idx])
        
        if idx==0:
            ax.set_ylabel(label_plot+' ['+units[label_plot]+']')

        ax.legend(loc='upper right' if idx==0 else 'lower right')

    ylim = plt.gca().get_ylim()

    for ax in a:
        ax.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')

    plt.gca().set_ylim(ylim)

    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.show()
    plt.close()
    plt.clf()

def plot_GST_bkg_vs_evol_quantile_bins_fit_single_site(df_stats):
    """ Function return scatter plot of background GST vs GST evolution for a single site.
    The site is binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced too.
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'bkg_grd_temp' and 'evol_grd_temp'
    
    Returns
    -------
    slope : float
        Linear regression slope
    intercept : float
        Linear regression intercept
    r : float
        Linear regression r-value. Need to square it to get R^2.
    Scatter plot of background GST vs GST evolution for the single site.
    The single site is binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced too.
    """

    df_stats_bis = pd.DataFrame(data=df_stats, columns=['bkg_grd_temp', 'evol_grd_temp'])
    df_stats_bis['bkg_grd_temp'] = pd.Categorical(df_stats_bis['bkg_grd_temp'], np.sort(df_stats['bkg_grd_temp']))
    df_stats_bis = df_stats_bis.sort_values('bkg_grd_temp')

    list_x = list(df_stats_bis['bkg_grd_temp'])
    list_y = list(df_stats_bis['evol_grd_temp'])

    quantiles = np.arange(0, 101, 10)

    cmap = plt.cm.seismic

    list_x_mean = []

    for i in range(len(quantiles)-1):
        low = int(np.ceil(len(df_stats)*quantiles[i]/100))
        up = int(np.ceil(len(df_stats)*quantiles[i+1]/100))
        list_x_mean.append(np.mean(list_x[low:up]))
        # plt.hlines(np.mean(list_y[low:up]),list_x[low],list_x[up-1], color=colorcycle[i], linewidth=2)

    vmax = np.max(np.abs(list_x_mean))

    for i in range(len(quantiles)-1):
        color = cmap((list_x_mean[i] + vmax)/(2*vmax))
        low = int(np.ceil(len(df_stats)*quantiles[i]/100))
        up = int(np.ceil(len(df_stats)*quantiles[i+1]/100))
        plt.scatter(list_x[low:up], list_y[low:up], c=color,s=0.8)
        plt.errorbar(np.mean(list_x[low:up]), np.mean(list_y[low:up]), np.std(list_y[low:up]), np.std(list_x[low:up]), c=color)
        plt.scatter(np.mean(list_x[low:up]), np.mean(list_y[low:up]), c=color, s=50)
        # plt.hlines(np.mean(list_y[low:up]),list_x[low],list_x[up-1], color=colorcycle[i], linewidth=2)

    slope, intercept, r, p, se = linregress(list_x, list_y)
    u = np.arange(np.min(list_x)-0.1, np.max(list_x)+0.1, 0.01)
    print('R-square:', r**2, ', regression slope:', slope , ', regression intercept:', intercept)
    plt.plot(u, slope*u+intercept, c='grey', label=('slope: %s' % (round(slope,3))))

    plt.xlabel('Mean background GST [°C]')
    plt.ylabel('Mean GST evolution [°C]')

    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

    return slope, intercept, r

def plot_GST_bkg_vs_evol_quantile_bins_fit(df_stats, list_site):
    """ Function return scatter plot of background GST vs GST evolution for 2 sites.
    Both sites are binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced for each site.
    
    Parameters
    ----------
    df_stats : list of pandas.core.frame.DataFrame
        List of 2 panda dataframe with at least the following columns: 'bkg_grd_temp' and 'evol_grd_temp'
    list_site : list
        List of labels for the site of each entry
    
    Returns
    -------
    slope : list
        List of linear regression slope (1 for each site)
    intercept : list
        List of linear regression intercept (1 for each site)
    r : list
        List of linear regression r-value (1 for each site). Need to square it to get R^2.
    Scatter plot of background GST vs GST evolution for 2 sites.
    Both sites are binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced for each site.
    """

    num = len(df_stats)

    df_stats_bis = [pd.DataFrame(data=i, columns=['bkg_grd_temp', 'evol_grd_temp']) for i in df_stats]
    for i in range(num):
        df_stats_bis[i]['bkg_grd_temp'] = pd.Categorical(df_stats_bis[i]['bkg_grd_temp'], np.sort(df_stats[i]['bkg_grd_temp']))
        df_stats_bis[i] = df_stats_bis[i].sort_values('bkg_grd_temp')

    list_x = [list(i['bkg_grd_temp']) for i in df_stats_bis]
    list_y = [list(i['evol_grd_temp']) for i in df_stats_bis]

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    quantiles = np.arange(0, 101, 10)

    cmap = plt.cm.seismic
    # colors = cmap(np.linspace(0, 1, len(quantiles)+(1 if len(quantiles)%2 else 0)))

    vmax = [[] for _ in df_stats]
    list_x_mean = [[] for _ in df_stats]

    for j in range(num):

        for i in range(len(quantiles)-1):
            low = int(np.ceil(len(df_stats[j])*quantiles[i]/100))
            up = int(np.ceil(len(df_stats[j])*quantiles[i+1]/100))
            list_x_mean[j].append(np.mean(list_x[j][low:up]))

        vmax[j] = np.max(np.abs(list_x_mean[j]))

    vmax = np.max(vmax)

    points = []
    slope = []
    intercept = []
    r = []

    for j in range(num):

        for i in range(len(quantiles)-1):
            color = cmap((list_x_mean[j][i] + vmax)/(2*vmax))
            low = int(np.ceil(len(df_stats[j])*quantiles[i]/100))
            up = int(np.ceil(len(df_stats[j])*quantiles[i+1]/100))
            plt.scatter(list_x[j][low:up], list_y[j][low:up], c=color,s=0.8)
            plt.errorbar(np.mean(list_x[j][low:up]), np.mean(list_y[j][low:up]), np.std(list_y[j][low:up]), np.std(list_x[j][low:up]), c=color)
            plt.scatter(np.mean(list_x[j][low:up]), np.mean(list_y[j][low:up]), c=color, s=50)
            # plt.hlines(np.mean(list_y[low:up]),list_x[low],list_x[up-1], color=colorcycle[i], linewidth=2)

        slope_i, intercept_i, r_i, p, se = linregress(list_x[j], list_y[j])
        slope.append(slope_i)
        intercept.append(intercept_i)
        r.append(r_i)
        u = np.arange(np.min(list_x[j])-0.05, np.max(list_x[j])+0.05, 0.01)
        print('R-square:', r_i**2, ', regression slope:', slope_i , ', regression intercept:', intercept_i)
        line_j, = plt.plot(u, slope_i*u+intercept_i, c=colorcycle[j], label=list_site[j])
        if j==0:
            first_legend = plt.legend(handles=[line_j], loc='upper left')

        xpts = np.max(list_x[j])+0.05 if j==0 else np.min(list_x[j])-0.05
        ypts = slope_i*xpts+intercept_i

        points.append([xpts, ypts])

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    u = np.arange(xlim[0], xlim[1], 0.01)
    slope_divide = (points[0][1]-points[1][1])/(points[0][0]-points[1][0])
    intercept_divide = points[1][1] - slope_divide*points[1][0]
    line = slope_divide*u+intercept_divide
    plt.plot(u, line, c='grey', linestyle='dashed')

    plt.fill_between(u, ylim[0]*len(u), line, alpha = 0.1, color=colorcycle[0], linewidth=1)
    plt.fill_between(u, line, ylim[1]*len(u), alpha = 0.1, color=colorcycle[1], linewidth=1)

    plt.plot()

    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)

    plt.xlabel('Mean background GST [°C]')
    plt.ylabel('Mean GST evolution [°C]')

    # Show the graph
    # first_legend = plt.legend(handles=[line[0]], loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[line_j], loc='lower right')
    plt.show()
    plt.close()
    plt.clf()

    return slope, intercept, r

def visual_running_stat(year_start):
    """ Function returns a visualization of random time series plotted year after year,
    meaning to represent the way data is sampled for the running deviation calculations.
    
    Parameters
    ----------
    year_start : int
        Year at which we want the plot to start, should coincide with the first year of the background period

    Returns
    -------
    Plot of the running statistics data selection
    """  

    years_a = list(range(year_start, year_start+6))
    years_b = list(range(year_start+10, year_start+13))
    years = years_a + years_b
    a = [0.2*np.random.randn(200) + i for i in years]
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']

    # If we were to simply plot pts, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax1) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    for i in range(len(years_a)):
        ax2.plot(a[i][:121], color=colorcycle[0])
        ax2.plot(range(120,141), a[i][120:141], color=colorcycle[1],
                linewidth=1, markevery=[140])
        ax2.plot(range(140,200), a[i][140:], color=colorcycle[0])

    for i in range(len(years_b)):
        ax1.plot(a[i+len(years_a)][:121], color=colorcycle[0])
        ax1.plot(range(120,141), a[i+len(years_a)][120:141], color=(colorcycle[1] if i<len(years_b)-1 else 'r'),
                linewidth=(1 if i<len(years_b)-1 else 2), markevery=[140])
        ax1.plot(range(140,200), a[i+len(years_a)][140:], color=colorcycle[0])

    ax1.plot(140, year_start+12, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")

    delta=3
    y1 = year_start+9.1
    y2 = year_start+5.9
    for i in range(3):
        ax1.plot(60-delta+delta*i, y1, marker="o", markersize=2, markeredgecolor=colorcycle[0], markerfacecolor=colorcycle[0])
        ax1.plot(130-delta+delta*i, y1, marker="o", markersize=2, markeredgecolor=colorcycle[1], markerfacecolor=colorcycle[1])
        ax1.plot(170-delta+delta*i, y1, marker="o", markersize=2, markeredgecolor=colorcycle[0], markerfacecolor=colorcycle[0])

        ax2.plot(60-delta+delta*i, y2, marker="o", markersize=2, markeredgecolor=colorcycle[0], markerfacecolor=colorcycle[0])
        ax2.plot(130-delta+delta*i, y2, marker="o", markersize=2, markeredgecolor=colorcycle[1], markerfacecolor=colorcycle[1])
        ax2.plot(170-delta+delta*i, y2, marker="o", markersize=2, markeredgecolor=colorcycle[0], markerfacecolor=colorcycle[0])

    # zoom-in / limit the view to different portions of the data
    ax2.set_ylim(year_start-1, year_start+6)  # outliers only
    ax1.set_ylim(year_start+9, year_start+13)  # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax2.set_xticks([0,120,140,200], ['Jan 1', 't-w', 't', 'Dec 31'])
    ax1.set_xticks([], [])
    ax1.set_yticks([year_start+10, year_start+12], ['year-2', 'year'])
    ax2.set_yticks([year_start, year_start+5], [year_start, year_start+5])

    plt.show()

def plot_visible_skymap_from_horizon_file(hor_path):
    """ Function returns a fisheye view of the sky with the visible portion in blue and the blocked one in black.
    
    Parameters
    ----------
    hor_path : str
        Path to the .csv horizon file

    Returns
    -------
    Plot of the sky view from the location
    """  

    hor_file = pd.read_csv(hor_path, usecols=['azi', 'hang'])

    theta_pre = hor_file['azi']
    theta = [i/360*2*np.pi for i in theta_pre]
    r_pre = hor_file['hang']
    r = [90-i for i in r_pre]
    
    # Creating the polar scatter plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    ax.set_ylim(0, 90)
    for i in range(len(r)):
        ax.fill_between( np.linspace(theta[i], (0 if i==len(r)-1 else theta[i+1]), 2), 0, r[i], color='deepskyblue', alpha=0.5)
        ax.fill_between( np.linspace(theta[i], (0 if i==len(r)-1 else theta[i+1]), 2), r[i], 90, color='black', alpha=1)
    ax.scatter(theta, r, c='blue', s=10, cmap='hsv', alpha=0.75)
    # plt.title('Scatter Plot on Polar Axis', fontsize=15)
    plt.show()

def get_all_stats_generic(forcings, path_forcings,
                          path_ground, path_snow, path_repository,
                          year_bkg_end=2010, year_trans_end=2024,
                          consecutive=7, extension='',
                          glacier=False, min_glacier_depth=100, max_glacier_depth=20000):
    """ Creates a number of pickle files (if they don't exist yet)
    
    Parameters
    ----------
    forcings : list of str
        List of forcings provided, with a number of entries between 1 and 3 in 'era5', 'merra2', and 'jra55'. E.g. ['era5', 'merra2']
    path_forcings : list
        list of paths to the location of each forcing data set
    path_snow : str
        path to the snow data
    path_repository : str
        path to the repostiroy .csv
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    consecutive : int, optional
        number of consecutive snow-free days to consider that snow has melted for the season
    extension : str, optional
        Location of the event, e.g. 'Aksaut_Ridge' (not a list, 1 single location)
    glacier : bool, optional
        By default only keeps non-glacier simulations but can be changed to True to select only glaciated simulations
    min_glacier_depth : float, optional
        Selects simulation with minimum snow height higher than this threshold (in mm)
    max_glacier_depth : float, optional
        Selects simulation with minimum snow height lower than this threshold (in mm)

    Returns
    -------
        list_valid_sim : list
            list of simulation number for all valid simulations
        reanalysis_stats : dict
            dictionary of mean quntities over the background and transient periods
        df : pandas.core.frame.DataFrame
            basic information for all simulations
        df_stats : pandas.core.frame.DataFrame
            basic statistics for all simulations
        dict_melt_out : collections.defaultdict
            melt out day for each year and each simulation
        dict_melt_out_consecutive : collections.defaultdict
            melt out day for each year and each simulation, with a minimum number of consecutive snow-free days

    """


    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    # we store the paths to all the files we will need for the analysis
    # starting with the 3 data reanalysis forcings
    path_forcings = {forcings[i]: path_forcings[i] for i in range(len(forcings))}

    if 'era5' in forcings:
        try: ncfile_air_era5.close()
        except: pass
        # Open file for air temperature
        ncfile_air_era5 = Dataset(path_forcings['era5'], mode='r')

    if 'merra2' in forcings:
        try: ncfile_air_merra2.close()
        except: pass
        # Open file for air temperature
        ncfile_air_merra2 = Dataset(path_forcings['merra2'], mode='r')

    if 'jra55' in forcings:
        try: ncfile_air_jra55.close()
        except: pass
        # Open file for air temperature
        ncfile_air_jra55 = Dataset(path_forcings['jra55'], mode='r')

    try: ncfile_ground.close()
    except: pass
    # Open file for ground temperature
    ncfile_ground = Dataset(path_ground, mode='r')
    # Select geotop model data
    f_ground = ncfile_ground.groups['geotop'] 
    try: ncfile_snow.close()
    except: pass

    # Open file for snow temperature
    ncfile_snow = Dataset(path_snow, mode='r')
    # Select geotop model data
    f_snow = ncfile_snow.groups['geotop']

    #####################################################################################
    # CREATE VARIABLES
    #####################################################################################

    #####################################################################################
    # GROUND
    #####################################################################################

    temp_ground = f_ground['Tg']
    time_ground = f_ground['Date']

    #####################################################################################
    # SNOW
    #####################################################################################

    snow_height = f_snow['snow_depth_mm']

    #####################################################################################
    # MATCH SIMULATION INDEXES
    #####################################################################################

    # get the topographic info of each simulation, e.g. altitude, slope, etc.
    df_raw = assign_value_df_raw(path_repository)

    try: df
    except NameError: df = None

    df = assign_value_df(df, df_raw, f_ground, f_snow, extension)

    #####################################################################################
    # AIR
    #####################################################################################

    #####################################################################################
    # PRECIPITATION CONSISTENCY TEST
    #####################################################################################

    # Here we print the sum of the precipitation flux over the whole length of the series
    print('Sum of the precipitation fluxes for the whole series:')

    if 'era5' in forcings:
        precipitation_era5 = ncfile_air_era5['PREC_sur']
        print('era5  :', np.sum(precipitation_era5[:,0]))
    if 'merra2' in forcings:
        precipitation_merra2 = ncfile_air_merra2['PREC_sur']
        print('merra2:', np.sum(precipitation_merra2[:,0]))
    if 'jra55' in forcings:
        precipitation_jra55 = ncfile_air_jra55['PREC_sur']
        print('jra55 :', np.sum(precipitation_jra55[:,0]))


    #####################################################################################
    # ERA5
    #####################################################################################

    if 'era5' in forcings:
        time_air_era5 = ncfile_air_era5['time']
        temp_air_era5 = ncfile_air_era5['AIRT_pl']
        SW_flux_era5 = ncfile_air_era5['SW_sur']
        SW_direct_flux_era5 = ncfile_air_era5['SW_topo_direct']
        SW_diffuse_flux_era5 = ncfile_air_era5['SW_topo_diffuse']

    #####################################################################################
    # MERRA2
    #####################################################################################

    if 'merra2' in forcings:
        time_air_merra2 = ncfile_air_merra2['time']
        temp_air_merra2 = ncfile_air_merra2['AIRT_pl']
        SW_flux_merra2 = ncfile_air_merra2['SW_sur']
        SW_direct_flux_merra2 = ncfile_air_merra2['SW_topo_direct']
        SW_diffuse_flux_merra2 = ncfile_air_merra2['SW_topo_diffuse']

    #####################################################################################
    # JRA55
    #####################################################################################

    if 'jra55' in forcings:
        time_air_jra55 = ncfile_air_jra55['time']
        temp_air_jra55 = ncfile_air_jra55['AIRT_pl']
        SW_flux_jra55 = ncfile_air_jra55['SW_sur']
        SW_direct_flux_jra55 = ncfile_air_jra55['SW_topo_direct']
        SW_diffuse_flux_jra55 = ncfile_air_jra55['SW_topo_diffuse']
        
        
    #####################################################################################
    # SEPARATING INTO BACKGROUND AND TRANSIENT
    #####################################################################################

    # for lack of longer datasets, we will define the background as everything happening before 2000 (here it means from 1980)
    # and we will limit the transient analysis to everything up to 2020-1-1

    [time_bkg_ground, time_trans_ground, time_pre_trans_ground] = list_tokens_year(time_ground, year_bkg_end, year_trans_end)[1:]

    if 'era5' in forcings:
        [time_bkg_air_era5, time_trans_air_era5, time_pre_trans_air_era5] = list_tokens_year(time_air_era5, year_bkg_end, year_trans_end)[1:]
    if 'merra2' in forcings:
        [time_bkg_air_merra2, time_trans_air_merra2, time_pre_trans_air_merra2] = list_tokens_year(time_air_merra2, year_bkg_end, year_trans_end)[1:]
    if 'jra55' in forcings:
        [time_bkg_air_jra55, time_trans_air_jra55, time_pre_trans_air_jra55] = list_tokens_year(time_air_jra55, year_bkg_end, year_trans_end)[1:]

    #####################################################################################
    # ALL TOGETHER
    #####################################################################################

    # initialize list of time series over all forcings
    precipitation_all = [0]*len(forcings)
    time_air_all = [0]*len(forcings)
    temp_air_all = [0]*len(forcings)
    SW_flux_all = [0]*len(forcings)
    SW_direct_flux_all = [0]*len(forcings)
    SW_diffuse_flux_all = [0]*len(forcings)
    time_bkg_air_all = [0]*len(forcings)
    time_trans_air_all = [0]*len(forcings)
    time_pre_trans_air_all = [0]*len(forcings)

    if 'era5' in forcings:
        index = forcings.index('era5')
        precipitation_all[index] = precipitation_era5
        time_air_all[index] = time_air_era5
        temp_air_all[index] = temp_air_era5
        SW_flux_all[index] = SW_flux_era5
        SW_direct_flux_all[index] = SW_direct_flux_era5
        SW_diffuse_flux_all[index] = SW_diffuse_flux_era5
        time_bkg_air_all[index] = time_bkg_air_era5
        time_trans_air_all[index] = time_trans_air_era5
        time_pre_trans_air_all[index] = time_pre_trans_air_era5

    if 'merra2' in forcings:
        index = forcings.index('merra2')
        precipitation_all[index] = precipitation_merra2
        time_air_all[index] = time_air_merra2
        temp_air_all[index] = temp_air_merra2
        SW_flux_all[index] = SW_flux_merra2
        SW_direct_flux_all[index] = SW_direct_flux_merra2
        SW_diffuse_flux_all[index] = SW_diffuse_flux_merra2
        time_bkg_air_all[index] = time_bkg_air_merra2
        time_trans_air_all[index] = time_trans_air_merra2
        time_pre_trans_air_all[index] = time_pre_trans_air_merra2

    if 'jra55' in forcings:
        index = forcings.index('jra55')
        precipitation_all[index] = precipitation_jra55
        time_air_all[index] = time_air_jra55
        temp_air_all[index] = temp_air_jra55
        SW_flux_all[index] = SW_flux_jra55
        SW_direct_flux_all[index] = SW_direct_flux_jra55
        SW_diffuse_flux_all[index] = SW_diffuse_flux_jra55
        time_bkg_air_all[index] = time_bkg_air_jra55
        time_trans_air_all[index] = time_trans_air_jra55
        time_pre_trans_air_all[index] = time_pre_trans_air_jra55

    #####################################################################################
    # REANALYSIS STATS
    #####################################################################################

    try: reanalysis_stats
    except NameError: reanalysis_stats = None

    reanalysis_stats = assign_value_reanalysis_stat_generic(reanalysis_stats, df, forcings, temp_air_all,
                                                            SW_flux_all, SW_direct_flux_all, SW_diffuse_flux_all,
                                                            time_bkg_air_all, time_trans_air_all, time_pre_trans_air_all,
                                                            extension)
    
    #####################################################################################
    # FILTER GLACIERS OUT
    #####################################################################################

    try: list_valid_sim
    except NameError: list_valid_sim = None

    list_valid_sim = glacier_filter_generic(list_valid_sim, snow_height, extension,
                                            glacier, min_glacier_depth, max_glacier_depth)


    #####################################################################################
    # MELT-OUT DATES
    #####################################################################################

    try: dict_melt_out
    except NameError: dict_melt_out = None

    dict_melt_out = melt_out_date(dict_melt_out, snow_height, list_valid_sim,
                                  time_ground, year_bkg_end, year_trans_end, extension)
    stats_melt_out_dic = stats_melt_out(dict_melt_out, year_bkg_end)

    try: dict_melt_out_consecutive
    except NameError: dict_melt_out_consecutive = None

    dict_melt_out_consecutive = melt_out_date_consecutive(consecutive, dict_melt_out_consecutive,
                                                          snow_height, list_valid_sim, time_ground,
                                                          year_bkg_end, year_trans_end, extension)
    stats_melt_out_dic_consecutive = stats_melt_out(dict_melt_out_consecutive, year_bkg_end)

    #####################################################################################
    # CREATE LIST OF STATS
    #####################################################################################

    try: df_stats
    except NameError: df_stats = None

    df_stats = assign_value_df_stat(df_stats, temp_ground, snow_height, reanalysis_stats,
                                    stats_melt_out_dic, stats_melt_out_dic_consecutive, df, list_valid_sim,
                                    time_bkg_ground, time_trans_ground, time_pre_trans_ground,
                                    extension)

                                           
    #####################################################################################
    # RETURN
    #####################################################################################

    return list_valid_sim, reanalysis_stats, df, df_stats, dict_melt_out, dict_melt_out_consecutive

def load_all_pickles_generic(extension=''):
    """ Loads all picklescorresponding to the site name
    
    Parameters
    ----------
    extension : str, optional
        Location of the event, e.g. 'Aksaut_Ridge'

    Returns
    -------
        list_valid_sim : list
            list of simulation number for all valid simulations
        reanalysis_stats : dict
            dictionary of mean quntities over the background and transient periods
        df : pandas.core.frame.DataFrame
            basic information for all simulations
        df_stats : pandas.core.frame.DataFrame
            basic statistics for all simulations
        dict_melt_out : collections.defaultdict
            melt out day for each year and each simulation
        dict_melt_out_consecutive : collections.defaultdict
            melt out day for each year and each simulation, with a minimum number of consecutive snow-free days

    """

    list_file_names = [f"list_valid_sim{('' if extension=='' else '_')}{extension}.pkl",
                       f"reanalysis_stats{('' if extension=='' else '_')}{extension}.pkl",
                       f"df{('' if extension=='' else '_')}{extension}.pkl",
                       f"df_stats{('' if extension=='' else '_')}{extension}.pkl",
                       f"dict_melt_out{('' if extension=='' else '_')}{extension}.pkl",
                       f"dict_melt_out_consecutive{('' if extension=='' else '_')}{extension}.pkl"
                       ]

    output = [0]*len(list_file_names)

    for i, file_name in enumerate(list_file_names):
        my_path = pickle_path + file_name
        # Open the file in binary mode 
        with open(my_path, 'rb') as file: 
            # Call load method to deserialze 
            output[i] = pickle.load(file) 
        print('Succesfully opened the pre-existing pickle:', file_name)

    [list_valid_sim, reanalysis_stats, df, df_stats, dict_melt_out, dict_melt_out_consecutive] = output

    return list_valid_sim, reanalysis_stats, df, df_stats, dict_melt_out, dict_melt_out_consecutive

def plot_all_generic(site, forcings, path_forcings,
                     path_ground, path_snow, path_swe, path_thaw_depth,
                     year_bkg_end=2010, year_trans_end=2023,
                     individual_heatmap=False, polar_plots=False,
                     parity_plot=False):
    """ Function returns a series of summary plots for a given site.
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    forcings : list of str
        List of forcings provided, with a number of entries between 1 and 3 in 'era5', 'merra2', and 'jra55'. E.g. ['era5', 'merra2']
    path_forcings : list of str
        List of string paths to the locations of the different forcings for that particular site (.nc)
    path_ground : str
        String path to the location of the ground output file from GTPEM (.nc)
    path_snow : str
        String path to the location of the snow output file from GTPEM (.nc)
    path_thaw_depth : str
        String path to the location of the thaw depth output file from GTPEM (.nc)
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    individual_heatmap : bool, optional
        Show or not heatmaps for unique altitude
    polar_plots : bool, optional
        Show or not polar plots
    parity_plot : bool, optional
        Show or not parity plot

    Returns
    -------
    Plots
    """  

    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    list_valid_sim, reanalysis_stats, df, df_stats, dict_melt_out, dict_melt_out_consecutive = load_all_pickles_generic(site)

    # we store the paths to all the files we will need for the analysis
    # starting with the 3 data reanalysis forcings
    path_forcings = {forcings[i]: path_forcings[i] for i in range(len(forcings))}

    if 'era5' in forcings:
        try: ncfile_air_era5.close()
        except: pass
        # Open file for air temperature
        ncfile_air_era5 = Dataset(path_forcings['era5'], mode='r')

    if 'merra2' in forcings:
        try: ncfile_air_merra2.close()
        except: pass
        # Open file for air temperature
        ncfile_air_merra2 = Dataset(path_forcings['merra2'], mode='r')

    if 'jra55' in forcings:
        try: ncfile_air_jra55.close()
        except: pass
        # Open file for air temperature
        ncfile_air_jra55 = Dataset(path_forcings['jra55'], mode='r')

    try: ncfile_ground.close()
    except: pass

    # Open file for ground temperature
    ncfile_ground = Dataset(path_ground, mode='r')
    # Select geotop model data
    f_ground = ncfile_ground.groups['geotop'] 

    try: ncfile_snow.close()
    except: pass

    # Open file for snow depth
    ncfile_snow = Dataset(path_snow, mode='r')
    # Select geotop model data
    f_snow = ncfile_snow.groups['geotop']

    try: ncfile_swe.close()
    except: pass

    # Open file for snow depth
    ncfile_swe = Dataset(path_swe, mode='r')
    # Select geotop model data
    f_swe = ncfile_swe.groups['geotop']

    try: ncfile_thaw_depth.close()
    except: pass

    # Open file for snow depth
    ncfile_thaw_depth = Dataset(path_thaw_depth, mode='r')
    # Select geotop model data
    f_thaw_depth = ncfile_thaw_depth.groups['geotop']

    #####################################################################################
    # ASSIGN VARIABLES
    #####################################################################################

    temp_ground = f_ground['Tg']
    time_ground = f_ground['Date']

    snow_height = f_snow['snow_depth_mm']
    swe = f_swe['snow_water_equivalent']
    depth_thaw = f_thaw_depth['AL']

    [time_bkg_ground, time_trans_ground, time_pre_trans_ground] = list_tokens_year(time_ground, year_bkg_end, year_trans_end)[1:]

    # initialize list of time series over all forcings
    precipitation_all = [0]*len(forcings)
    time_air_all = [0]*len(forcings)
    temp_air_all = [0]*len(forcings)

    if 'era5' in forcings:
        precipitation_era5 = ncfile_air_era5['PREC_sur']
        time_air_era5 = ncfile_air_era5['time']
        temp_air_era5 = ncfile_air_era5['AIRT_pl']
        SW_flux_era5 = ncfile_air_era5['SW_sur']
        SW_direct_flux_era5 = ncfile_air_era5['SW_topo_direct']
        SW_diffuse_flux_era5 = ncfile_air_era5['SW_topo_diffuse']
        [time_bkg_air_era5, time_trans_air_era5, time_pre_trans_air_era5] = list_tokens_year(time_air_era5, year_bkg_end, year_trans_end)[1:]
        index = forcings.index('era5')
        precipitation_all[index] = precipitation_era5
        time_air_all[index] = time_air_era5
        temp_air_all[index] = temp_air_era5
    if 'merra2' in forcings:
        precipitation_merra2 = ncfile_air_merra2['PREC_sur']
        time_air_merra2 = ncfile_air_merra2['time']
        temp_air_merra2 = ncfile_air_merra2['AIRT_pl']
        SW_flux_merra2 = ncfile_air_merra2['SW_sur']
        SW_direct_flux_merra2 = ncfile_air_merra2['SW_topo_direct']
        SW_diffuse_flux_merra2 = ncfile_air_merra2['SW_topo_diffuse']
        [time_bkg_air_merra2, time_trans_air_merra2, time_pre_trans_air_merra2] = list_tokens_year(time_air_merra2, year_bkg_end, year_trans_end)[1:]
        index = forcings.index('merra2')
        precipitation_all[index] = precipitation_merra2
        time_air_all[index] = time_air_merra2
        temp_air_all[index] = temp_air_merra2
    if 'jra55' in forcings:
        precipitation_jra55 = ncfile_air_jra55['PREC_sur']
        time_air_jra55 = ncfile_air_jra55['time']
        temp_air_jra55 = ncfile_air_jra55['AIRT_pl']
        SW_flux_jra55 = ncfile_air_jra55['SW_sur']
        SW_direct_flux_jra55 = ncfile_air_jra55['SW_topo_direct']
        SW_diffuse_flux_jra55 = ncfile_air_jra55['SW_topo_diffuse']
        [time_bkg_air_jra55, time_trans_air_jra55, time_pre_trans_air_jra55] = list_tokens_year(time_air_jra55, year_bkg_end, year_trans_end)[1:]
        index = forcings.index('jra55')
        precipitation_all[index] = precipitation_jra55
        time_air_all[index] = time_air_jra55
        temp_air_all[index] = temp_air_jra55


    #####################################################################################
    # PLOTS
    #####################################################################################

    # assign a subjective weight to all simulations
    pd_weight = assign_weight_sim_Aksaut(df_stats)
    # weighted mean GST
    temp_ground_mean = list(np.average([temp_ground[i,:,0] for i in list(pd_weight.index.values)], axis=0, weights=pd_weight['weight']))
    print('The following plot is a histogram of the distribution of the statistical weights of all simulations:')
    plot_hist_stat_weights(pd_weight, df, zero=True)
    print('The following plot is a histogram of the distribution of glacier simulations wrt to altitude, aspect, slope, and forcing:')
    plot_hist_valid_sim_all_variables(df, df_stats, depth_thaw)

    alt_list = sorted(set(df_stats['altitude']))
    alt_index = int(np.floor((len(alt_list)-1)/2))
    alt_index_abs = alt_list[alt_index]
    print('List of altitudes:', alt_list)
    print('Altitude at which we plot the time series:', alt_index_abs)

    # Note that this is selecting the elevation in the 'middle': index 2 in the list [0,1,2,3,4]
    # and it returns the mean air temperature over all reanalyses
    mean_air_temp = mean_all_reanalyses_generic(time_air_all, [i[:,alt_index] for i in temp_air_all])
    # here we get the mean precipitation and then water from snow melting 
    mean_prec = mean_all_reanalyses_generic(time_air_all, [i[:,alt_index] for i in precipitation_all])
    swe_mean = list(np.average([swe[i,:] for i in list(pd_weight.index.values)], axis=0, weights=pd_weight['weight']))
    # finally we get the total water production, averaged over all reanalyses
    tot_water_prod = assign_tot_water_prod_generic(swe_mean, mean_prec, time_ground, time_pre_trans_ground, time_air_merra2)

    year_rockfall = rockfall_values(site)['year']
    print('Plots of the normalized distance of air and ground temperature, water production, and thaw_depth as a function of time')
    print('Granularity: week and month side by side')
    plot_aggregating_distance_temp_all_generic(['Air temperature', 'Water production', 'Ground temperature'],
                                       [time_air_all[0], time_ground, time_ground],
                                       [mean_air_temp, tot_water_prod, temp_ground_mean],
                                       ['week', 'month'], site, year_rockfall, year_bkg_end, year_trans_end, False)
    print('Granularity: year, plotted for all years')
    plot_aggregating_distance_temp_all_generic(['Air temperature', 'Water production', 'Ground temperature'],
                                        [time_air_all[0], time_ground, time_ground],
                                        [mean_air_temp, tot_water_prod, temp_ground_mean],
                                        ['year'], site, 0, year_bkg_end, year_trans_end, False)

    print('Yearly statistics for air and ground surface temperature, and also precipitation and water production')
    plot_box_yearly_stat('Air temperature', time_air_all[0], mean_air_temp, year_bkg_end=2010, year_trans_end=2023)
    plot_box_yearly_stat('GST', time_ground, temp_ground_mean, year_bkg_end=2010, year_trans_end=2023)
    plot_box_yearly_stat('Precipitation', time_ground, mean_prec, year_bkg_end=2010, year_trans_end=2023)
    plot_box_yearly_stat('Water production', time_ground, tot_water_prod, year_bkg_end=2010, year_trans_end=2023)

    if individual_heatmap:
        print('Heatmap of the background mean GST as a function of aspect and slope at %s m:' % alt_index_abs)
        plot_table_mean_GST_aspect_slope_generic(df_stats, site, alt_index_abs, True, False)
        print('Heatmap of the evolution of the mean GST between the background and the transient periods as a function of aspect and slope at %s m:' % alt_index_abs)
        plot_table_mean_GST_aspect_slope_generic(df_stats, site, alt_index_abs, False, False)

    print('Heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitude')
    plot_table_aspect_slope_all_altitudes_generic(df, df_stats, site, show_glacier=False, box=False)


    if polar_plots:
        print('Polar heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitude')
        plot_table_aspect_slope_all_altitudes_polar_generic(df_stats, site, False)

        print('Polar plot of the permafrost and glacier spatial distribution as a function of aspect and slope at all altitude')
        plot_permafrost_all_altitudes_polar_generic(df_stats, site, depth_thaw, False)

    print('CDF of background, transient, and evolution GST:')
    plot_cdf_GST(df_stats)
    print('Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference:')
    heatmap_percentile_GST(df_stats)
    print('Plot of mean GST evolution vs background GST, with an emphasis on the 10% colder and warmer simulations')
    plot_10_cold_warm(df_stats)
    print('Plot of mean GST evolution vs background GST, fit, and binning per 10% quntiles')
    plot_GST_bkg_vs_evol_quantile_bins_fit_single_site(df_stats)

    print('Scatter plot of mean background GST vs evolution of mean GST between the background and transient period')
    plot_mean_bkg_GST_vs_evolution(df_stats)

    if parity_plot:
        print('Parity plot (statistically-modeled vs numerically-simulated) of background mean GST:')
        xdata, ydata, optimizedParameters, pcov, corr_matrix, R_sq = fit_stat_model_grd_temp_Aksaut(df_stats, all=False, diff_forcings=True)
        list_ceof = ['offset', 'c_alt', 'd_alt', 'c_asp', 'c_slope']
        pd_coef = pd.DataFrame(list_ceof, columns=['Coefficient'])
        # previously was columns=['all', 'era5', 'merra2', 'jra55'] when had all 3 forcings
        pd_coef = pd.concat([pd_coef, pd.DataFrame((np.array([list(i) for i in optimizedParameters]).transpose()), columns=forcings)], axis=1)
        print('The coefficients of the statistical model for the mean background GST are given by:')
        print(pd_coef)
