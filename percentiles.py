"""This module creates statistical analysis based on quantiles"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn

from pickling import load_all_pickles
from topoheatmap import table_background_evolution_mean_GST_aspect_slope
from constants import save_constants

colorcycle, _ = save_constants()

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

def plot_cdf_GST(site, path_pickle):
    """ Function returns coordinates of the point corresponing to the percentile of the 
        Cumulated Distribution Dunction (CDF)
    
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

    _, _, _, _, _, df_stats = load_all_pickles(site, path_pickle)
    
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

    # plot the sorted data:
    _ = plt.figure()
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

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

def plot_10_cold_warm(site, path_pickle):
    """ Function returns a plot of mean GST evolution vs background GST, with an emphasis on the 10% colder and warmer simulations
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved

    Returns
    -------
    Plot of mean GST evolution vs background GST, with an emphasis on the 10% colder and warmer simulations
    """

    _, _, _, _, _, df_stats = load_all_pickles(site, path_pickle)

    table_all = table_background_evolution_mean_GST_aspect_slope(site, path_pickle)

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

    #pylint: disable=unsubscriptable-object
    list_x = list(df_stats_bis['bkg_grd_temp']) 
    #pylint: disable=unsubscriptable-object
    list_y = [df_stats_bis['trans_grd_temp'].iloc[i] - df_stats_bis['bkg_grd_temp'].iloc[i] for i in range(len(df_stats_bis))]

    pos_10 = int(np.ceil(len(data_bkg)/10))

    plt.scatter(list_x[:pos_10], list_y[:pos_10], c=colorcycle[0])
    plt.scatter(list_x[pos_10:-pos_10], list_y[pos_10:-pos_10], c=colorcycle[1])
    plt.scatter(list_x[-pos_10:], list_y[-pos_10:], c=colorcycle[2])

    mean = [np.mean(list_y[:pos_10]), np.mean(list_y[-pos_10:])]

    plt.axvline((list_x[pos_10-1]+list_x[pos_10])/2, c=colorcycle[0])
    plt.axvline((list_x[-pos_10-1]+list_x[-pos_10])/2, c=colorcycle[2])
    plt.axhline(mean[0], linestyle="dashed",
                label=(r"$\overline{{\rm GST}_{\rm low 10}}$ =" + ('+' if mean[0]>0 else '') + f"{mean[0]:.3f}°C"),
                c=colorcycle[0])
    plt.axhline(mean[1], linestyle="dashed",
                label=(r"$\overline{{\rm GST}_{\rm high 10}}$ =" + ('+' if mean[1]>0 else '') + f"{mean[1]:.3f}°C"),
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
    
def heatmap_percentile_GST(site, path_pickle):
    """ Function returns a heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved

    Returns
    -------
    Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    """

    table_all = table_background_evolution_mean_GST_aspect_slope(site, path_pickle)

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
