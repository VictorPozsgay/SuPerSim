"""This module creates statistical analysis based on quantiles"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn

from SuPerSim.pickling import load_all_pickles
from SuPerSim.topoheatmap import table_background_evolution_mean_GST_aspect_slope
from SuPerSim.constants import colorcycle

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

def data_cdf_GST(site, path_pickle):
    """ Function returns data to plot a Cumulated Distribution Dunction (CDF)
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved

    Returns
    -------
    data : dict
        List of sorted background, transient and evolution of GST (gound surface temperature)
        and SO (surface offset)
    """

    pkl = load_all_pickles(site, path_pickle)
    df_stats = pkl['df_stats']

    # diff_warming stands for differential warming and is the tevolution of the surface offset (SO)
    # between the background and transient periods
    data = {'bkg_grd_temp': np.sort(df_stats['bkg_grd_temp']),
            'trans_grd_temp': np.sort(df_stats['trans_grd_temp']),
            'evol_grd_temp': np.sort(df_stats['evol_grd_temp']),
            'bkg_SO': np.sort(df_stats['bkg_SO']),
            'trans_SO': np.sort(df_stats['trans_SO']),
            'diff_warming': np.sort(df_stats['diff_warming'])}
    
    return data

def plot_cdf_GST(data):
    """ Function plots the Cumulated Distribution Dunction (CDF) of background, transient, and evolution of mean GST and SO
    
    Parameters
    ----------
    data : dict
        List of sorted background, transient and evolution of GST (gound surface temperature)
        and SO (surface offset)

    Returns
    -------
    2 plots: left panel shows the CDF of background, transient, and evolution of mean GST
             right panel shows the CDF of background, transient, and evolution of mean SO
    """
    
    # sort the data:
    data_bkg_sorted = data['bkg_grd_temp']
    data_trans_sorted = data['trans_grd_temp']
    data_evol_sorted = data['evol_grd_temp']

    data_bkg_SO_sorted = data['bkg_SO']
    data_trans_SO_sorted = data['trans_SO']
    data_diff_warming_sorted = data['diff_warming']


    # calculate the proportional values of samples
    p = np.linspace(0, 1, len(data_bkg_sorted))

    point = [[[] for _ in range(3)] for _ in range(4)]

    list_data = [data_bkg_sorted, data_trans_sorted, data_bkg_SO_sorted, data_trans_SO_sorted]
    list_quant = [10,50,90]

    for indx_i, i in enumerate(list_data):
        for indx_j, j in enumerate(list_quant):
            point[indx_i][indx_j] = coordinates_percentile_cdf(i, p, j)

    # plot the sorted data:
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

    return fig

def plot_cdf_GST_from_inputs(site, path_pickle):
    """ Function plots the Cumulated Distribution Dunction (CDF) of background, transient, and evolution of mean GST and SO
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved

    Returns
    -------
    2 plots: left panel shows the CDF of background, transient, and evolution of mean GST
             right panel shows the CDF of background, transient, and evolution of mean SO
    """

    data = data_cdf_GST(site, path_pickle)
    fig = plot_cdf_GST(data)

    return fig

def panda_percentile_GST(site, path_pickle):
    """ Function returns a dataframe of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved

    Returns
    -------
    pd_data : pandas.core.frame.DataFrame
        Panda dataframe of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    """

    table_all = table_background_evolution_mean_GST_aspect_slope(site, path_pickle)

    # create some randomly distributed data:
    data_bkg = [k for i in table_all['bkg'].values() for j in i.columns for k in i[j]]
    data_evol = [k for i in table_all['evol'].values() for j in i.columns for k in i[j]]
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

    return pd_data

def plot_heatmap_percentile_GST(pd_data):
    """ Function returns a heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    
    Parameters
    ----------
    pd_data : pandas.core.frame.DataFrame
        Panda dataframe of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference

    Returns
    -------
    fig : Figure
        Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    """

    fig, _ = plt.subplots()

    sn.heatmap(data=pd_data, annot=True, fmt=".2f", cmap='seismic',
                cbar=True, yticklabels=True, xticklabels=True, cbar_kws={'label': 'Mean GST [°C]'},
                norm=matplotlib.colors.TwoSlopeNorm(vmin=np.min([np.min(pd_data),-0.05]), vcenter=0, vmax=np.max([np.max(pd_data),0.05])))
    plt.ylabel('Percentile')

    # Show the graph
    plt.show()
    plt.close()

    return fig

def plot_heatmap_percentile_GST_from_inputs(site, path_pickle):
    """ Function returns a heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    
    Parameters
    ----------
    pd_data : pandas.core.frame.DataFrame
        Panda dataframe of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
        
    Returns
    -------
    fig : Figure
        Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    """

    pd_data = panda_percentile_GST(site, path_pickle)
    fig = plot_heatmap_percentile_GST(pd_data)

    return fig
