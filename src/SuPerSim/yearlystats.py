"""This module creates statistics in yearly bins for timeseries"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pandas as pd
from netCDF4 import num2date #pylint: disable=no-name-in-module
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from SuPerSim.mytime import list_tokens_year
from SuPerSim.constants import colorcycle, units
from SuPerSim.seasonal import stats_all_years_simulations_to_single_year

def data_box_yearly_stat(name_series, time_file, file_to_plot, year_bkg_end, year_trans_end):
    """ Returns dataframe of daily data
    
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
    df_data : pandas.core.frame.DataFrame
        Panda dataframe with columns ['name_series' (e.g. 'Precipitation), 'Year']
    """

    list_dates = list_tokens_year(time_file, year_bkg_end, year_trans_end)[0]
    
    a = []
    for i in list_dates.keys():
        if i < year_trans_end:
            a = a + [i]*len(list_dates[i])

    if name_series in ['Precipitation', 'Water production']:
        file_to_plot = [i*86400 for i in file_to_plot]

    df_data = pd.DataFrame(file_to_plot[:len(a)], columns=[name_series], index=a)
    df_data['Year'] = a

    return df_data

def plot_box_yearly_stat(df_data, year_bkg_end, year_trans_end):
    """ Box plot of yearly statistics of the timeseries, together with mean for the background and transient periods
    
    Parameters
    ----------
    df_data : pandas.core.frame.DataFrame
        Panda dataframe with columns ['name_series' (e.g. 'Precipitation), 'Year']
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period

    Returns
    -------
    fig : Figure
        Box plot of yearly statistics of the timeseries, together with mean for the background and transient periods
    """

    fig, ax = plt.subplots()

    list_dates = sorted(np.unique(df_data['Year']))
    year_start = list_dates[0]
    name_series = df_data.columns[0]
    mean = [np.mean(df_data[df_data['Year']<year_bkg_end][name_series]), np.mean(df_data[(df_data['Year']>=year_bkg_end) & (df_data['Year']<year_trans_end)][name_series])]
    formatted_mean = [f"{i:.2f}" for i in mean]

    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
    sn.boxplot(x='Year', y=name_series, data=df_data, showmeans=True, showfliers=False, meanprops=meanpointprops, color='grey', linecolor='black')

    ax.hlines(mean[0], 0 - 1/2, year_bkg_end - year_start - 1/2, linewidth=2, color=colorcycle[0],
              label=f'Background mean: {formatted_mean[0]}{units[name_series]}')
    ax.hlines(mean[1], year_bkg_end - year_start - 1/2, year_trans_end - year_start - 1/2, linewidth=2, color=colorcycle[1],
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

    return fig

def plot_box_yearly_stat_from_inputs(name_series, time_file, file_to_plot, year_bkg_end, year_trans_end):
    """ Box plot of yearly statistics of the timeseries, together with mean for the background and transient periods
    
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
    fig : Figure
        Box plot of yearly statistics of the timeseries, together with mean for the background and transient periods
    """

    df_data = data_box_yearly_stat(name_series, time_file, file_to_plot, year_bkg_end, year_trans_end)
    fig = plot_box_yearly_stat(df_data, year_bkg_end, year_trans_end)

    return fig

def atmospheric_data_to_panda(list_time_file, list_time_series, label_plot):
    """ Function returns panda data frame for 'air' timeseries, concatenated all reanalyses
        and averaged over all altitudes
    
    Parameters
    ----------
    list_time_file : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored (air time), one per reanalysis
    list_time_series : list of netCDF4._netCDF4.Variable
        List of time series with air time shape (could be air temperature, precipitation, SW, etc.)
    label_plot : str
        label associated to the plot, if 'Precipitation' or 'Water production', rescales data to mm/day from mm/sec
    
    Returns
    -------
    panda_test : pandas.core.frame.DataFrame
        Panda dataframe of the timeseries value per year
        columns are year, label_plot
    """

    panda_list = [[] for _ in list_time_series]

    for i,l in enumerate(list_time_file):
        # create a panda dataframe with month, day, hour for each timestamp
        panda_list[i] = pd.DataFrame(num2date(l[:], l.units), columns=['date'])
        panda_list[i]['year'] = [j.year for j in panda_list[i]['date']]
        # Note that this is averageing the timeseries over all altitudes
        panda_list[i][label_plot] = np.mean(list_time_series[i], axis=1)
        panda_list[i] = panda_list[i].drop(columns=['date'])

    panda_test = pd.concat(panda_list)

    if label_plot in ['Precipitation', 'Water production']:
        panda_test[label_plot] = panda_test[label_plot]*86400

    return panda_test

def sim_data_to_panda(time_file, time_series, list_valid_sim, label_plot):
    """ Function returns panda data frame for 'air' timeseries, concatenated all reanalyses
        and averaged over all altitudes
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (ground time)
    time_series : netCDF4._netCDF4.Variable
        List of time series with ground time shape (could be ground temperature, SWE, etc.)
    list_valid_sim : list
        List of the indices of all valid simulations
    label_plot : str
        label associated to the plot
    
    Returns
    -------
    panda_test : pandas.core.frame.DataFrame
        Panda dataframe of the timeseries value per year
        columns are year, label_plot
    """

    long_timeseries = []
    for sim in list_valid_sim:
        long_timeseries.append(time_series[sim,:,0] if len(time_series.shape) == 3 else time_series[sim,:])

    long_timeseries = np.array(long_timeseries).flatten()
    long_years = np.array([int(i.year) for i in num2date(time_file[:], time_file.units)]*len(list_valid_sim))

    panda_test = pd.DataFrame(long_years.transpose(), columns=['year'])
    panda_test[label_plot] = long_timeseries.transpose()

    return panda_test

def panda_data_to_yearly_stats(panda_test, year_trans_end):
    """ Function returns panda data frame for 'air' timeseries, concatenated all reanalyses
        and averaged over all altitudes
    
    Parameters
    ----------
    panda_test : pandas.core.frame.DataFrame
        Panda dataframe of the timeseries value per year, month, and day
    year_trans_end : int
        Transient period is BEFORE the start of the year corresponding to the variable
    
    Returns
    -------
    yearly_quantiles : pandas.core.frame.DataFrame
        Panda dataframe of the timeseries [0.023, 0.16, 0.5, 0.84, 0.977] quantiles per year.
        The lenght of the table is #quantiles x #years
        Multi-index : (quantile, year) and column: label_plot from function atmospheric_data_to_panda()
    yearly_mean : pandas.core.series.Series
        Panda dataframe of the mean of the timeseries for each year
        index : year and column: label_plot from function atmospheric_data_to_panda()
    """

    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]

    panda_test = panda_test[panda_test['year'] < year_trans_end]

    yearly_quantiles = panda_test.groupby(['year']).quantile(list_quantiles)
    yearly_quantiles.index.names = ['year','quantile']
    yearly_quantiles = yearly_quantiles.swaplevel()
    yearly_mean = panda_test.groupby(['year']).mean()

    return yearly_quantiles, yearly_mean

def plot_yearly_quantiles(yearly_quantiles, yearly_mean, year_bkg_end, year_trans_end, plot_quantiles=True):
    """ Function plots yearly statistics for 'air' timeseries, averaged over all reanalyses and altitudes
    
    Parameters
    ----------
    yearly_quantiles : pandas.core.frame.DataFrame
        Panda dataframe of the timeseries [0.023, 0.16, 0.5, 0.84, 0.977] quantiles per year.
        The lenght of the table is #quantiles x #years
        Multi-index : (quantile, year) and column: label_plot from function atmospheric_data_to_panda()
    yearly_mean : pandas.core.series.Series
        Panda dataframe of the mean of the timeseries for each year
        index : year and column: label_plot from function atmospheric_data_to_panda()
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    plot_quantiles : bool, optional
        Gives the option to plot the 1-sigma and 2-sigma spread. True by default but the range is largely dominated by the 2-sigma envelope,
        which hinders a good representation of the mean's trend (better if False).
    
    Returns
    -------
    fig : Figure
        Plot of yearly statistics for any timeseries. Mean and several quantiles for each year.
    """

    list_quantiles = sorted(np.unique([q for q,_ in yearly_quantiles.index]))
    list_years = sorted(np.unique([y for _,y in yearly_quantiles.index]))
    
    label_plot = yearly_quantiles.columns[0]

    mean_bkg = np.mean(yearly_mean.loc[list_years[0]:year_bkg_end-1])
    mean_trans = np.mean(yearly_mean.loc[year_bkg_end:year_trans_end-1])
    # mean_list = [mean_bkg, mean_trans]

    # exponent = [int(np.floor(np.log10(np.abs(i)))) for i in mean_list]
    # formatted_mean = [f"{m:.2e}" for i, m in enumerate(mean_list) if ((exponent[i] < -1) | (exponent[i]>2)) else float(f"{m:.2f}")]
    formatted_mean = [f"{i:.2f}" for i in [mean_bkg, mean_trans]]

    dict_points = {0: {'alpha': 0.2, 'width': 0.5},
                1: {'alpha': 0.4, 'width': 1.0},
                2: {'alpha': 1.0, 'width': 2.0},
                3: {'alpha': 0.4, 'width': 1.0},
                4: {'alpha': 0.2, 'width': 0.5}}

    fig = plt.subplots()

    plt.scatter(list_years, yearly_mean, color=colorcycle[0], linestyle='None', label='Yearly mean')
    # plt.plot(list_years, yearly_mean, color=colorcycle[0], label='Mean')
    # plt.plot(list_years, yearly_quantiles.iloc[dict_indices_quantiles[list_quantiles[2]]][label_plot], color=colorcycle[0])
    
    if plot_quantiles:
        for i in [0,1,3,4]:
            plt.scatter(list_years, yearly_quantiles.loc[[list_quantiles[i]]][label_plot], color=colorcycle[0], alpha=dict_points[i]['alpha'], linewidth=dict_points[i]['width'])
        plt.fill_between(list_years, yearly_quantiles.loc[[list_quantiles[1]]][label_plot], yearly_quantiles.loc[[list_quantiles[3]]][label_plot],
                            alpha = 0.4, color=colorcycle[0], linewidth=1,
                            # label='Quantiles 16-84'
                            )
        plt.fill_between(list_years, yearly_quantiles.loc[[list_quantiles[0]]][label_plot], yearly_quantiles.loc[[list_quantiles[4]]][label_plot],
                            alpha = 0.2, color=colorcycle[0], linewidth=0.5,
                            # label='Quantiles 2.3-97.7'
                            )
    
    plt.hlines(mean_bkg, list_years[0], year_bkg_end, color=colorcycle[1],
               label=f'Background mean: {formatted_mean[0]}{units[label_plot]}')
    plt.hlines(mean_trans,  year_bkg_end, list_years[-1], color=colorcycle[2],
               label=f'Transient mean: {formatted_mean[1]}{units[label_plot]}')

    ylim = plt.gca().get_ylim()

    plt.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')

    plt.gca().set_ylim(ylim)

    if label_plot in ['GST', 'Air temperature']:
        plt.axhline(y=0, color='grey', linestyle='dashed')
    
    plt.ylabel(label_plot+' ['+units[label_plot]+']')

    locs = np.arange(list_years[0], list_years[-1]+1, np.floor((list_years[-1]+1-list_years[0])/8), dtype=int)
    plt.xticks(locs, locs)

    # Show the graph
    plt.legend(loc='upper right')
    plt.show()
    plt.close()

    return fig 

def plot_yearly_quantiles_atmospheric_from_inputs(list_time_file, list_time_series, label_plot, year_bkg_end, year_trans_end, plot_quantiles=True):
    """ Function returns panda data frame for atmospheric timeseries, concatenated over all reanalyses
        and averaged over all altitudes
        from intended atmospheric input
    
    Parameters
    ----------
    list_time_file : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored (air time), one per reanalysis
    list_time_series : list of netCDF4._netCDF4.Variable
        List of time series with air time shape (could be air temperature, precipitation, SW, etc.)
    label_plot : str
        label associated to the plot, if 'Precipitation' or 'Water production', rescales data to mm/day from mm/sec
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    plot_quantiles : bool, optional
        Gives the option to plot the 1-sigma and 2-sigma spread. True by default but the range is largely dominated by the 2-sigma envelope,
        which hinders a good representation of the mean's trend (better if False).
    
    Returns
    -------
    fig : Figure
        Plot of yearly statistics for atmospheric timeseries. Mean and several quantiles for each year.
    """

    panda_test = atmospheric_data_to_panda(list_time_file, list_time_series, label_plot)
    yearly_quantiles, yearly_mean = panda_data_to_yearly_stats(panda_test, year_trans_end)
    fig = plot_yearly_quantiles(yearly_quantiles, yearly_mean, year_bkg_end, year_trans_end, plot_quantiles)

    return fig

def plot_yearly_quantiles_sim_from_inputs(time_file, time_series, list_valid_sim, label_plot, year_bkg_end, year_trans_end, plot_quantiles=True):
    """ Function returns panda data frame for atmospheric timeseries, concatenated over all reanalyses
        and averaged over all altitudes
        from intended simulated input
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (ground time)
    time_series : netCDF4._netCDF4.Variable
        List of time series with ground time shape (could be ground temperature, SWE, etc.)
    list_valid_sim : list
        List of the indices of all valid simulations
    label_plot : str
        label associated to the plot
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    plot_quantiles : bool, optional
        Gives the option to plot the 1-sigma and 2-sigma spread. True by default but the range is largely dominated by the 2-sigma envelope,
        which hinders a good representation of the mean's trend (better if False).
    
    Returns
    -------
    fig : Figure
        Plot of yearly statistics for simulated timeseries. Mean and several quantiles for each year.
    """

    panda_test = sim_data_to_panda(time_file, time_series, list_valid_sim, label_plot)
    yearly_quantiles, yearly_mean = panda_data_to_yearly_stats(panda_test, year_trans_end)
    fig = plot_yearly_quantiles(yearly_quantiles, yearly_mean, year_bkg_end, year_trans_end, plot_quantiles)

    return fig

# def plot_yearly_quantiles_all_sims(time_file, time_series, list_valid_sim, label_plot, year_bkg_end, year_trans_end):
#     """ Function plots yearly statistics for 'ground' timeseries over all simulations
    
#     Parameters
#     ----------
#     time_file : netCDF4._netCDF4.Variable
#         File where the time index of each datapoint is stored (ground time)
#     time_series : netCDF4._netCDF4.Variable
#         List of time series with ground time shape (could be ground temperature, SWE, etc.)
#     list_valid_sim : list
#         List of the indices of all valid simulations
#     label_plot : str
#         label associated to the plot.
#     year_bkg_end : int, optional
#         Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
#     year_trans_end : int, optional
#         Same for transient period
    
#     Returns
#     -------
#     Plot of yearly statistics for 'ground' timeseries. Mean and several quantiles for each year.
#     """
    
#     dict_points = {0: {'alpha': 0.2, 'width': 0.5},
#                 1: {'alpha': 0.4, 'width': 1.0},
#                 2: {'alpha': 1.0, 'width': 2.0},
#                 3: {'alpha': 0.4, 'width': 1.0},
#                 4: {'alpha': 0.2, 'width': 0.5}}

#     list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]
    
#     long_timeseries = []
#     for sim in list_valid_sim:
#         long_timeseries.append(time_series[sim,:,0] if len(time_series.shape) == 3 else time_series[sim,:])

#     long_timeseries = np.array(long_timeseries).flatten()
#     long_years = np.array([int(i.year) for i in num2date(time_file[:], time_file.units)]*len(list_valid_sim))

#     panda_test = pd.DataFrame(long_years.transpose(), columns=['year'])
#     panda_test['timeseries'] = long_timeseries.transpose()

#     list_drop = [i for i in np.unique(panda_test['year']) if i >= year_trans_end]

#     quantiles = panda_test.groupby(['year']).quantile(list_quantiles).drop(index=list_drop)
#     quantiles.index.names = ['year','quantile']
#     quantiles = quantiles.swaplevel()
#     dict_indices_quantiles = quantiles.groupby(['quantile']).indices
#     mean_end = panda_test.groupby(['year']).mean().drop(index=list_drop)
#     xdata = np.array(mean_end.index)

#     mean_bkg = np.mean(mean_end.loc[xdata[0]:year_bkg_end-1])
#     mean_trans = np.mean(mean_end.loc[year_bkg_end:year_trans_end-1])
#     formatted_mean = [f"{i:.2f}" for i in [mean_bkg, mean_trans]]
    
#     plt.scatter(xdata, mean_end, color=colorcycle[0], linestyle='None', label='Yearly mean')
#     # plt.plot(xdata, mean_end, color=colorcycle[0], label='Mean')
#     # plt.plot(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[2]]]['timeseries'], color=colorcycle[0])
#     for i in [0,1,3,4]:
#         plt.scatter(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[i]]]['timeseries'], color=colorcycle[0], alpha=dict_points[i]['alpha'], linewidth=dict_points[i]['width'])
#     plt.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[1]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[3]]]['timeseries'],
#                         alpha = 0.4, color=colorcycle[0], linewidth=1,
#                         # label='Quantiles 16-84'
#                         )
#     plt.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[0]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[4]]]['timeseries'],
#                         alpha = 0.2, color=colorcycle[0], linewidth=0.5,
#                         # label='Quantiles 2.3-97.7'
#                         )
#     if label_plot in ['GST', 'Air temperature']:
#         plt.axhline(y=0, color='grey', linestyle='dashed')

#     plt.hlines(mean_bkg, xdata[0], year_bkg_end, color=colorcycle[1],
#                label=f'Background mean: {formatted_mean[0]}{units[label_plot]}')
#     plt.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[2],
#                label=f'Transient mean: {formatted_mean[1]}{units[label_plot]}')

#     ylim = plt.gca().get_ylim()
#     plt.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')
#     plt.gca().set_ylim(ylim)

#     plt.ylabel(label_plot+' ['+units[label_plot]+']')

#     plt.tight_layout()  # otherwise the right y-label is slightly clipped

#     # Show the graph
#     plt.legend(loc='upper right')
#     plt.show()
#     plt.close()

def plot_yearly_quantiles_side_by_side(list_yearly_quantiles, list_yearly_mean, list_site, year_bkg_end, year_trans_end, plot_quantiles=True):
    """ Function plots yearly statistics for 'air' timeseries, averaged over all reanalyses and altitudes
    
    Parameters
    ----------
    list_yearly_quantiles : list of pandas.core.frame.DataFrame
        List of panda dataframes of the timeseries [0.023, 0.16, 0.5, 0.84, 0.977] quantiles per year.
        The lenght of the table is #quantiles x #years
        Multi-index : (quantile, year) and column: label_plot from function atmospheric_data_to_panda()
    list_yearly_mean : list of pandas.core.series.Series
        List of panda dataframes of the mean of the timeseries for each year
        index : year and column: label_plot from function atmospheric_data_to_panda()
    list_site : list
        List of labels for the site of each entry
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    plot_quantiles : bool, optional
        Gives the option to plot the 1-sigma and 2-sigma spread. True by default but the range is largely dominated by the 2-sigma envelope,
        which hinders a good representation of the mean's trend (better if False).
    
    Returns
    -------
    fig : Figure
        Plot of 2 side by side subplots of yearly statistics for the same timeseries at two different sites.
        Mean and several quantiles for each year.
    """

    list_quantiles = sorted(np.unique([q for q,_ in list_yearly_quantiles[0].index]))
    list_years = sorted(np.unique([y for _,y in list_yearly_quantiles[0].index]))
    
    label_plot = [i.columns[0] for i in list_yearly_quantiles]

    mean_bkg = [np.mean(i.loc[list_years[0]:year_bkg_end-1]) for i in list_yearly_mean]
    mean_trans = [np.mean(i.loc[year_bkg_end:year_trans_end-1]) for i in list_yearly_mean]
    # mean_list = [mean_bkg, mean_trans]

    # exponent = [int(np.floor(np.log10(np.abs(i)))) for i in mean_list]
    # formatted_mean = [f"{m:.2e}" for i, m in enumerate(mean_list) if ((exponent[i] < -1) | (exponent[i]>2)) else float(f"{m:.2f}")]
    formatted_mean = [[f"{i:.2f}" for i in [mean_bkg[j], mean_trans[j]]] for j in range(2)]

    dict_points = {0: {'alpha': 0.2, 'width': 0.5},
                1: {'alpha': 0.4, 'width': 1.0},
                2: {'alpha': 1.0, 'width': 2.0},
                3: {'alpha': 0.4, 'width': 1.0},
                4: {'alpha': 0.2, 'width': 0.5}}

    fig, a = plt.subplots(1, 2, figsize=(8, 4), sharey='row')
    for idx,ax in enumerate(a):
        ax.scatter(list_years, list_yearly_mean[idx], color=colorcycle[idx], linestyle='None', label='Yearly mean')
        # ax.plot(xdata, mean_end, color=colorcycle[idx], label='Mean')
        # plt.plot(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[2]]]['timeseries'], color=colorcycle[0])
        if plot_quantiles:
            for i in [0,1,3,4]:
                ax.scatter(list_years, list_yearly_quantiles[idx].loc[[list_quantiles[i]]][label_plot[idx]], color=colorcycle[idx], alpha=dict_points[i]['alpha'], linewidth=dict_points[i]['width'])
            ax.fill_between(list_years, list_yearly_quantiles[idx].loc[[list_quantiles[1]]][label_plot[idx]], list_yearly_quantiles[idx].loc[[list_quantiles[3]]][label_plot[idx]],
                                alpha = 0.4, color=colorcycle[idx], linewidth=1,
                                # label='Quantiles 16-84'
                                )
            ax.fill_between(list_years, list_yearly_quantiles[idx].loc[[list_quantiles[0]]][label_plot[idx]], list_yearly_quantiles[idx].loc[[list_quantiles[4]]][label_plot[idx]],
                                alpha = 0.2, color=colorcycle[idx], linewidth=0.5,
                                # label='Quantiles 2.3-97.7'
                                )

        ax.hlines(mean_bkg[idx], list_years[0], year_bkg_end, color=colorcycle[2],
               label=f'Background mean: {formatted_mean[idx][0]}{units[label_plot[idx]]}')
        ax.hlines(mean_trans[idx],  year_bkg_end, list_years[-1], color=colorcycle[3],
               label=f'Transient mean: {formatted_mean[idx][1]}{units[label_plot[idx]]}')
            
        if label_plot[idx] in ['GST', 'Air temperature']:
            ax.axhline(y=0, color='grey', linestyle='dashed')

        ax.title.set_text(list_site[idx])
        
        if idx==0:
            ax.set_ylabel(label_plot[idx]+' ['+units[label_plot[idx]]+']')

        ax.legend(loc='upper right' if idx==0 else 'lower right')

    ylim = plt.gca().get_ylim()

    for ax in a:
        ax.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')

    plt.gca().set_ylim(ylim)

    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.show()
    plt.close()

    return fig

def plot_yearly_quantiles_side_by_side_atmospheric_from_inputs(list_time_file, list_list_time_series, label_plot, list_site, year_bkg_end, year_trans_end, plot_quantiles=True):
    """ Function returns panda data frame for atmospheric timeseries, concatenated over all reanalyses
        and averaged over all altitudes
        from intended atmospheric input
    
    Parameters
    ----------
    list_time_file : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored (air time), one per reanalysis
    list_list_time_series : list of list of netCDF4._netCDF4.Variable
        List (two sites) of list (n reanalyses) of time series with air time shape (could be air temperature, precipitation, SW, etc.)
    label_plot : str
        label associated to the plot, if 'Precipitation' or 'Water production', rescales data to mm/day from mm/sec
    list_site : list
        List of labels for the site of each entry
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    plot_quantiles : bool, optional
        Gives the option to plot the 1-sigma and 2-sigma spread. True by default but the range is largely dominated by the 2-sigma envelope,
        which hinders a good representation of the mean's trend (better if False).
    
    Returns
    -------
    fig : Figure
        Plot of 2 side by side subplots of yearly statistics for the same atmospheric timeseries at two different sites.
        Mean and several quantiles for each year.
    """

    panda_test = [[] for _ in range(2)]
    yearly_quantiles = [[] for _ in range(2)]
    yearly_mean = [[] for _ in range(2)]

    for i in range(2):
        panda_test[i] = atmospheric_data_to_panda(list_time_file, list_list_time_series[i], label_plot[i])
        yearly_quantiles[i], yearly_mean[i] = panda_data_to_yearly_stats(panda_test[i], year_trans_end)
    
    fig = plot_yearly_quantiles_side_by_side(yearly_quantiles, yearly_mean, list_site, year_bkg_end, year_trans_end, plot_quantiles)

    return fig

def plot_yearly_quantiles_side_by_side_sim_from_inputs(time_file, list_time_series, list_list_valid_sim, label_plot, list_site, year_bkg_end, year_trans_end, plot_quantiles=True):
    """ Function returns panda data frame for simulated timeseries, concatenated over all reanalyses
        and averaged over all altitudes
        from intended simulated input
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (ground time)
    list_time_series : netCDF4._netCDF4.Variable
        List of time series with ground time shape (could be ground temperature, SWE, etc.)
    list_list_valid_sim : list of list
        List (2 sites) of list of indices of all valid simulations
    label_plot : str
        label associated to the plot
    list_site : list
        List of labels for the site of each entry
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    plot_quantiles : bool, optional
        Gives the option to plot the 1-sigma and 2-sigma spread. True by default but the range is largely dominated by the 2-sigma envelope,
        which hinders a good representation of the mean's trend (better if False).
    
    Returns
    -------
    Plot of 2 side by side subplots of yearly statistics for the same simulated timeseries at two different sites.
        Mean and several quantiles for each year.
    """

    panda_test = [[] for _ in range(2)]
    yearly_quantiles = [[] for _ in range(2)]
    yearly_mean = [[] for _ in range(2)]

    for i in range(2):
        panda_test[i] = sim_data_to_panda(time_file, list_time_series[i], list_list_valid_sim[i], label_plot)
        yearly_quantiles[i], yearly_mean[i] = panda_data_to_yearly_stats(panda_test[i], year_trans_end)
    
    fig = plot_yearly_quantiles_side_by_side(yearly_quantiles, yearly_mean, list_site, year_bkg_end, year_trans_end, plot_quantiles)

    return fig

# def plot_yearly_quantiles_all_sims_side_by_side(time_file, time_series, list_valid_sim, label_plot, list_site, year_bkg_end, year_trans_end):
#     """ Function plots yearly statistics for 'ground' timeseries over all simulations for 1 same metric
#         at 2 different sites. 1 plot per site, both plots side by side.
    
#     Parameters
#     ----------
#     time_file : netCDF4._netCDF4.Variable
#         File where the time index of each datapoint is stored (ground time)
#     time_series : netCDF4._netCDF4.Variable
#         List of time series with ground time shape (could be ground temperature, SWE, etc.)
#     list_valid_sim : list
#         List of the indices of all valid simulations
#     label_plot : str
#         label associated to the plot.
#     list_site : list
#         List of labels for the site of each entry
#     year_bkg_end : int, optional
#         Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
#     year_trans_end : int, optional
#         Same for transient period
    
#     Returns
#     -------
#     Plots of yearly statistics for 'ground' timeseries over all simulations for 1 same metric
#     at 2 different sites. 1 plot per site, both plots side by side.
#     """
    
#     dict_points = {0: {'alpha': 0.2, 'width': 0.5},
#                 1: {'alpha': 0.4, 'width': 1.0},
#                 2: {'alpha': 1.0, 'width': 2.0},
#                 3: {'alpha': 0.4, 'width': 1.0},
#                 4: {'alpha': 0.2, 'width': 0.5}}

#     list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]

#     _, a = plt.subplots(1, 2, figsize=(8, 4), sharey='row')
#     for idx,ax in enumerate(a):
#         long_timeseries = []
#         for sim in list_valid_sim[idx]:
#             long_timeseries.append(time_series[idx][sim,:,0] if len(time_series[idx].shape) == 3 else time_series[idx][sim,:])

#         long_timeseries = np.array(long_timeseries).flatten()
#         long_years = np.array([int(i.year) for i in num2date(time_file[:], time_file.units)]*len(list_valid_sim[idx]))

#         panda_test = pd.DataFrame(long_years.transpose(), columns=['year'])
#         panda_test['timeseries'] = long_timeseries.transpose()

#         list_drop = [i for i in np.unique(panda_test['year']) if i >= year_trans_end]

#         quantiles = panda_test.groupby(['year']).quantile(list_quantiles).drop(index=list_drop)
#         quantiles.index.names = ['year','quantile']
#         quantiles = quantiles.swaplevel()
#         dict_indices_quantiles = quantiles.groupby(['quantile']).indices
#         mean_end = panda_test.groupby(['year']).mean().drop(index=list_drop)
#         xdata = np.array(mean_end.index)

#         mean_bkg = np.mean(mean_end.loc[xdata[0]:year_bkg_end-1])
#         mean_trans = np.mean(mean_end.loc[year_bkg_end:year_trans_end-1])
#         formatted_mean = [f"{i:.2f}" for i in [mean_bkg, mean_trans]]
        
#         ax.scatter(xdata, mean_end, color=colorcycle[idx], linestyle='None', label='Yearly mean')
#         # ax.plot(xdata, mean_end, color=colorcycle[idx], label='Mean')
#         # plt.plot(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[2]]]['timeseries'], color=colorcycle[0])
#         for i in [0,1,3,4]:
#             ax.scatter(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[i]]]['timeseries'], color=colorcycle[idx], alpha=dict_points[i]['alpha'], linewidth=dict_points[i]['width'])
#         ax.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[1]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[3]]]['timeseries'],
#                             alpha = 0.4, color=colorcycle[idx], linewidth=1,
#                             # label='Quantiles 16-84'
#                             )
#         ax.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[0]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[4]]]['timeseries'],
#                             alpha = 0.2, color=colorcycle[idx], linewidth=0.5,
#                             # label='Quantiles 2.3-97.7'
#                             )

#         ax.hlines(mean_bkg, xdata[0], year_bkg_end, color=colorcycle[2],
#                label=f'Background mean: {formatted_mean[0]}{units[label_plot]}')
#         ax.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[3],
#                label=f'Transient mean: {formatted_mean[1]}{units[label_plot]}')
            
#         if label_plot in ['GST', 'Air temperature']:
#             ax.axhline(y=0, color='grey', linestyle='dashed')

#         ax.title.set_text(list_site[idx])
        
#         if idx==0:
#             ax.set_ylabel(label_plot+' ['+units[label_plot]+']')

#         ax.legend(loc='upper right' if idx==0 else 'lower right')

#     ylim = plt.gca().get_ylim()

#     for ax in a:
#         ax.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')

#     plt.gca().set_ylim(ylim)

#     plt.tight_layout()  # otherwise the right y-label is slightly clipped

#     # Show the graph
#     plt.show()
#     plt.close()

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
