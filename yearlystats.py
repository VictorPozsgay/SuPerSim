"""This module creates statistics in yearly bins for timeseries"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pandas as pd
from netCDF4 import num2date #pylint: disable=no-name-in-module
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from mytime import list_tokens_year
from constants import save_constants

colorcycle, units = save_constants()

def plot_box_yearly_stat(name_series, time_file, file_to_plot, year_bkg_end, year_trans_end):
    """ Plots the distance to the mean in units of standard deviation for a specific year or for the whole length
    
    Parameters
    ----------
    name_series : str
        Name of the quantity to plot, has to be one of 'GST', 'Air temperature', 'Precipitation', 'SWE', 'Water production'
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored        
    file_to_plot : list
        Mean time series (temp_ground_mean, mean_air_temp, mean_prec, swe_mean, tot_water_prod)
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period

    Returns
    -------
        plot

    """

    _, ax = plt.subplots()

    list_dates = list_tokens_year(time_file, year_bkg_end, year_trans_end)[0]
    overall_mean = np.mean(file_to_plot)
    exponent = int(np.floor(np.log10(np.abs(overall_mean))))
    a = []
    for i in list_dates.keys():
        if i < year_trans_end:
            a = a + [i]*len(list_dates[i])
    x = pd.DataFrame(file_to_plot[:len(a)], columns=[name_series], index=a)
    x['Year'] = a

    if name_series in ['Precipitation', 'Water production']:
        x[name_series] = x[name_series]*86400

    mean = [np.mean(x[x['Year']<year_bkg_end][name_series]), np.mean(x[(x['Year']>=year_bkg_end) & (x['Year']<year_trans_end)][name_series])]

    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
    sn.boxplot(x='Year', y=name_series, data=x, showmeans=True, showfliers=False, meanprops=meanpointprops, color='grey', linecolor='black')

    formatted_mean = [f"{i:.2e}" for i in mean] if ((exponent < -1) | (exponent>2)) else [float(f"{i:.2f}") for i in mean]

    ax.hlines(mean[0], 0, year_bkg_end - list(list_dates.keys())[0] - 1 + 1/2, linewidth=2, color=colorcycle[0],
              label=f'Background mean: {formatted_mean[0]}{units[name_series]}')
    ax.hlines(mean[1], year_bkg_end - list(list_dates.keys())[0] - 1/2, year_trans_end - list(list_dates.keys())[0] - 1, linewidth=2, color=colorcycle[1],
              label=f'Transient mean: {formatted_mean[1]}{units[name_series]}')

    plt.tight_layout()
    locs, labels = plt.xticks()  # Get the current locations and labels.
    dt = int(np.floor(len(locs)/10))
    locs = locs[::dt]
    labels = labels[::dt]
    plt.xticks(locs, labels, rotation=0)
    plt.ylabel(name_series+' ['+units[name_series]+']')
    ax.ticklabel_format(axis='y', style='sci', useMathText=True, scilimits=(-3,3))

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def stats_yearly_quantiles_air(list_time_file, list_time_series, label_plot, year_trans_end):
    """ Function returns yearly statistics for 'air' timeseries, averaged over all reanalyses and altitudes
    
    Parameters
    ----------
    list_time_file : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored (air time), one per reanalysis
    list_time_series : list of netCDF4._netCDF4.Variable
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

    panda_list = [[] for _ in list_time_series]

    for i,l in enumerate(list_time_file):
        # create a panda dataframe with month, day, hour for each timestamp
        panda_list[i] = pd.DataFrame(num2date(l[:], l.units), columns=['date'])
        panda_list[i]['year'] = [j.year for j in panda_list[i]['date']]
        panda_list[i]['month'] = [j.month for j in panda_list[i]['date']]
        panda_list[i]['day'] = [j.day for j in panda_list[i]['date']]
        panda_list[i]['hour'] = [j.hour for j in panda_list[i]['date']]
        # Note that this is avergaing the timeseries over all altitudes
        panda_list[i]['timeseries'] = np.mean(list_time_series[i], axis=1)
        panda_list[i] = panda_list[i].drop(columns=['date', 'hour'])
    
    panda_test = pd.concat(panda_list)
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

def plot_yearly_quantiles_air(list_time_file, list_time_series, label_plot, year_bkg_end, year_trans_end):
    """ Function plots yearly statistics for 'air' timeseries, averaged over all reanalyses and altitudes
    
    Parameters
    ----------
    list_time_file : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored (air time), one per reanalysis
    list_time_series : list of netCDF4._netCDF4.Variable
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

    _, quantiles, mean_end, dict_indices_quantiles, xdata = stats_yearly_quantiles_air(list_time_file, list_time_series, label_plot, year_trans_end)
    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]

    mean_bkg = np.mean(mean_end.loc[xdata[0]:year_bkg_end-1])
    mean_trans = np.mean(mean_end.loc[year_bkg_end:year_trans_end-1])
    formatted_mean = [f"{i:.2f}" for i in [mean_bkg, mean_trans]]

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
               label=f'Background mean: {formatted_mean[0]}{units[label_plot]}')
    plt.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[2],
               label=f'Transient mean: {formatted_mean[1]}{units[label_plot]}')

    ylim = plt.gca().get_ylim()

    plt.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')

    plt.gca().set_ylim(ylim)

    if label_plot in ['GST', 'Air temperature']:
        plt.axhline(y=0, color='grey', linestyle='dashed')
    
    plt.ylabel(label_plot+' ['+units[label_plot]+']')

    locs = np.arange(xdata[0], xdata[-1]+1, np.floor((xdata[-1]+1-xdata[0])/8), dtype=int)
    plt.xticks(locs, locs)

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
    formatted_mean = [f"{i:.2f}" for i in [mean_bkg, mean_trans]]
    
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
               label=f'Background mean: {formatted_mean[0]}{units[label_plot]}')
    plt.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[2],
               label=f'Transient mean: {formatted_mean[1]}{units[label_plot]}')

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
    
    dict_points = {0: {'alpha': 0.2, 'width': 0.5},
                1: {'alpha': 0.4, 'width': 1.0},
                2: {'alpha': 1.0, 'width': 2.0},
                3: {'alpha': 0.4, 'width': 1.0},
                4: {'alpha': 0.2, 'width': 0.5}}

    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]

    _, a = plt.subplots(1, 2, figsize=(8, 4), sharey='row')
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
        formatted_mean = [f"{i:.2f}" for i in [mean_bkg, mean_trans]]
        
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
               label=f'Background mean: {formatted_mean[0]}{units[label_plot]}')
        ax.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[3],
               label=f'Transient mean: {formatted_mean[1]}{units[label_plot]}')
            
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