"""This module creates running statistics for timerseries"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pickle
import pandas as pd
from netCDF4 import num2date #pylint: disable=no-name-in-module
import numpy as np
import matplotlib.pyplot as plt

from open import open_air_nc, open_ground_nc, open_swe_nc
from mytime import list_tokens_year
from weights import assign_weight_sim
from constants import save_constants

colorcycle, _ = save_constants()

def mean_all_reanalyses(time_files, files_to_smooth, year_bkg_end, year_trans_end):
    """ Function returns the mean time series over a number of reanalysis (has the length of a timeseries)
    
    Parameters
    ----------
    time_files : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored
    files_to_smooth : list of list
        List of time series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that it needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts [temp_air_era5[:,0]] but not [temp_air_era5]
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
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

def assign_tot_water_prod(path_forcing_list, path_ground, path_swe, path_pickle, year_bkg_end, year_trans_end, site, no_weight=True):
    """ Function returns the total water production at daily intervals in [mm s-1] 
    
    Parameters
    ----------
    path_forcing_list : list of str
        List of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_swe : str
        Path to the .nc file where the aggregated SWE simulations are stored
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    no_weight : bool, optional
        If True, all simulations have the same weight, otherwise the weight is computed as a function of altitude, aspect, and slope

    Returns
    -------
    tot_water_prod : list
        list of daily total water production
    mean_swe : list
        Weighted mean SWE time series over all simulations
    mean_prec : list
        Mean precipitation time series over all reanalyses
    """

    file_name = f"df_stats{('' if site=='' else '_')}{site}.pkl"
    my_path = path_pickle + file_name
    with open(my_path, 'rb') as file: 
        # Call load method to deserialize 
        df_stats = pickle.load(file)

    _, swe = open_swe_nc(path_swe)
    _, time_ground, _ = open_ground_nc(path_ground)
    _, _, _, time_pre_trans_ground = list_tokens_year(time_ground, year_bkg_end, year_trans_end)    

    pd_weight = assign_weight_sim(site, path_pickle, no_weight)

    time_air_all = [open_air_nc(i)[0] for i in path_forcing_list]
    precipitation_all = [open_air_nc(i)[-1] for i in path_forcing_list]

    # Note that this is selecting the elevation in the 'middle': index 2 in the list [0,1,2,3,4]
    # and it returns the mean air temperature over all reanalyses
    alt_list = sorted(set(df_stats['altitude']))
    alt_index = int(np.floor((len(alt_list)-1)/2))
    alt_index_abs = alt_list[alt_index]
    print('List of altitudes:', alt_list)
    print('Altitude at which we plot the time series:', alt_index_abs)

    # here we get the mean precipitation and then water from snow melting 
    mean_swe = list(np.average([swe[i,:] for i in list(pd_weight.index.values)], axis=0, weights=pd_weight['weight']))
    mean_prec = mean_all_reanalyses(time_air_all, [i[:,alt_index] for i in precipitation_all], year_bkg_end, year_trans_end)

    # convert mean precipitation into a panda dataframe to facilitate grouping by 24.
    # This way the data is reported daily (as it is for the swe) rather than hourly
    pd_prec_day = ((pd.DataFrame(mean_prec).rolling(24).mean()).iloc[23::24, :]).reset_index(drop=True)

    # convert swe into panda df to make the diff() operation easier. Swe is a total and
    # we get the instantaneous flux by taking the difference between consecutive datapoints.
    # The total precipitation between 2 datapoints is over 1 day = 864000 hence we divide by 86400.
    # Finally, swe is in [kg m-2] which is equivalent to [mm] so we're good.

    # Number of days between the start of the ground and air timeseries
    num_day = int((num2date(time_ground[0], time_ground.units) - num2date(time_air_all[0][0], time_air_all[0].units)).total_seconds()/86400)
    # we add num_days zeros at the beginning to make sure the dfs have the same dimension and span Jan 1, 1980 to Dec 31, 2019
    # We make sure to multiply by (-1) and to divide by 86400
    pd_diff_swe = pd.concat([pd.DataFrame([0]*num_day), (pd.DataFrame(mean_swe[:len(time_ground[time_pre_trans_ground][:])]).diff()).fillna(0)/86400*(-1)], ignore_index=True)

    if len(pd_prec_day)!=len(pd_diff_swe):
        raise ValueError('VictorCustomError: The lengths of the precipitation and swe time series are different!')

    # The total water production is returned in the list type as the sum of precipitation and swe
    tot_water_prod = list(pd_prec_day.add(pd_diff_swe.reset_index(drop=True), fill_value=0)[0])

    return tot_water_prod, mean_swe, mean_prec

def running_mean_std(time_file, file_to_smooth, window, fill_before=False):
    """ Function returns a running average (and standard deviation) of the time series over a given time window
        adpated from Welford's online algorithm, computing everything in a single-pass 
    
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
    fill_before : bool, optional
        Option to fill the first empty values of the smoothed data with the first running average value if True

    Returns
    -------
    running_mean : dict
        Smoothed time series
    running_std : dict
        Running standard deviation

    """

    # number of hours between consecutive measurements: 1 for air, 24 for ground
    cons_hours = int((num2date(time_file[1], time_file.units) - num2date(time_file[0], time_file.units)).total_seconds()/3600)
    window_size = int({'day': 24, 'week': 24*7, 'month': 24*30, 'year':24*365, 'climate':24*35*20}[window]/cons_hours)

    
    # range of the timestamp whether the user chose a given year of the whole study period
    index_range = list(range(len(file_to_smooth)))

    # initialize the running mean, the first value is at (window_size-1+index_range[0])
    # the initial window extends from index_range[0] to window_size-1+index_range[0] and has (window_size) elements
    init = np.sum(file_to_smooth[index_range[0]:window_size+index_range[0]])/window_size
    running_mean = {}
    running_std = {}

    # fill the first values if fill_before==True
    if fill_before:
        for i in range(index_range[0], window_size-1+index_range[0]):
            running_mean[i] = init
            # running_std[i] = 0
    
    running_mean[window_size-1+index_range[0]] = init
    # running_std[window_size-1+index_range[0]] = np.std(file_to_smooth[index_range[0]:window_size+index_range[0]])

    # the following values are evaluated recursively in a single-pass (more efficient)
    # I have derived the running version of the Welford's online algorithm and tested it.
    for i in range(window_size+index_range[0], index_range[-1]+1):
        running_mean[i] = running_mean[i-1] + (file_to_smooth[i] - file_to_smooth[i-window_size])/window_size
        # running_std[i] = np.sqrt( (window_size * running_std[i-1]**2
        #                            + (file_to_smooth[i] - file_to_smooth[i-window_size])**2/window_size
        #                            + (file_to_smooth[i] - running_mean[i])**2
        #                            - (file_to_smooth[i-window_size] - running_mean[i])**2) / window_size )

    return running_mean, running_std

def aggregating_distance_temp(time_file, file_to_smooth, window, year_bkg_end, year_trans_end, year=0, fill_before=False):
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
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    year : int, optional
        One can choose to display a specific year or the whole set by choosing an integer not in the study sample, e.g. 0.
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

    list_dates = list_tokens_year(time_file, year_bkg_end, year_trans_end)[0]

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

def plot_aggregating_distance_temp_all(yaxes, xdata, ydata, window, site, path_pickle, year_bkg_end, year_trans_end, year, fill_before=False):
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
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    year : int
        One can choose to display a specific year or the whole set by choosing an integer not in the study sample, e.g. 0.
    fill_before : bool, optional
        Option to fill the first empty values of the smoothed data with the first running average value if True

    Returns
    -------
        plot containing len(yaxes) subplots sharing the x axis

    """

    file_name_rockfall_values = f"rockfall_values{('' if site=='' else '_')}{site}.pkl"
    my_path_rockfall_values = path_pickle + file_name_rockfall_values

    with open(my_path_rockfall_values, 'rb') as file: 
        # Call load method to deserialize 
        rockfall_values = pickle.load(file)

    num_rows = len(xdata)
    num_cols = len(window)

    f, a = plt.subplots(num_rows, num_cols, figsize=(8, 2*num_rows), sharey='row')
    for idx,ax in enumerate(a):
        distance = [aggregating_distance_temp(xdata[idx], ydata[idx], i, year_bkg_end, year_trans_end, year, fill_before) for i in window]
        list_dates = list_tokens_year(xdata[idx], year_bkg_end, year_trans_end)[0]

        if year in list(list_dates.keys()):
            index_range = list_dates[year]
        else:
            index_range = list(range(len(ydata[idx])))[list_dates[np.min(list(distance[0].keys()))][0]:]

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

        if rockfall_values['exact_date']:
            for indx in rockfall_values['time_index']:
                if indx in index_range:
                    if int((num2date(xdata[idx][indx], xdata[idx].units)-rockfall_values['datetime']).total_seconds()) == 0:
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
            locs = locs[::int(len(locs)/10)]
            labels = list(range(year_bkg_end,year_trans_end+1,1))
            labels = labels[::int(len(labels)/10)]

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
