"""This module creates seasonal plots with interannual means and variations"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pandas as pd
from netCDF4 import num2date #pylint: disable=no-name-in-module
import numpy as np
import matplotlib.pyplot as plt
import mpl_axes_aligner

pickle_path = '/fs/yedoma/home/vpo001/VikScriptsTests/Python_Pickles/'
colorcycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


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
        #pylint: disable=unsupported-assignment-operation
        panda_test['timeseries'] = time_series[:,2]
    else:
        panda_test = panda_test.drop(index=np.arange(0,len(mask_period),1)[[not i for i in mask_period]])
        panda_test['timeseries'] = time_series[mask_period[:],2]


    mean_end = pd.DataFrame(np.array([i for i in panda_test.groupby(['month', 'day', 'hour']).mean().reset_index()['timeseries']]))

    list_quant = [0.023, 0.16, 0.5, 0.84, 0.977]

    panda_test = panda_test.groupby(['month', 'day', 'hour']).quantile(list_quant)
    panda_test.index.names = ['month', 'day', 'hour', 'quantiles']
    panda_test = panda_test.reset_index().drop(columns=['month', 'day', 'hour'])

    quantiles = []
    for i in range(len(list_quant)):
        quantiles.append(panda_test.loc[i::len(list_quant)]['timeseries'])

    quantiles = pd.DataFrame(np.array(quantiles))

    return quantiles, mean_end

def plot_sanity_one_year_quantiles_two_periods(time_file, time_series_list, list_valid_sim_list, axis_label, list_label, list_mask_period):
    """ Function returns a plot of a single timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread,
    for background and transient piods
    
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
    
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}

    _, a = plt.subplots(1, 2, figsize=(10, 5))
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
