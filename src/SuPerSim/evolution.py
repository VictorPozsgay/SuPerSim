"""This module creates plots of the evolution of metrics between background and transient periods"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from SuPerSim.pickling import load_all_pickles
from SuPerSim.topoheatmap import table_background_evolution_mean_GST_aspect_slope
from SuPerSim.constants import colorcycle

def sorted_bkg_GST_data(site, path_pickle):
    """ Function returns background GST and GST evolution for a single site.
    The two lists have the same order and are ordered in background GST increasing order
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    
    Returns
    -------
    list_xy : list
        List of lists of background GST and GST evolution for a single site.
        The nth element of each list corresponds to the same simulation
        The background GST is in increasing order
        in the shape [list_x, list_y]
    """
    
    pkl = load_all_pickles(site, path_pickle)
    df_stats = pkl['df_stats']

    df_stats_bis = pd.DataFrame(data=df_stats, columns=['bkg_grd_temp', 'evol_grd_temp'])
    # df_stats_bis['bkg_grd_temp'] = pd.Categorical(df_stats_bis['bkg_grd_temp'], np.sort(df_stats['bkg_grd_temp']))
    df_stats_bis = df_stats_bis.sort_values('bkg_grd_temp')

    list_xy = [list(df_stats_bis.loc[:, 'bkg_grd_temp']), list(df_stats_bis.loc[:, 'evol_grd_temp'])]

    return list_xy

def plot_GST_bkg_vs_evol_quantile_bins_fit_single_site(list_xy):
    """ Function return scatter plot of background GST vs GST evolution for a single site.
    The site is binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced too.
    
    Parameters
    ----------
    list_xy : list
        List of lists of background GST and GST evolution for a single site.
        The nth element of each list corresponds to the same simulation
        The background GST is in increasing order
        in the shape [list_x, list_y]
    
    Returns
    -------
    fig : figure
        Scatter plot of background GST vs GST evolution for the single site.
        The single site is binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
        A linear regression is produced too.
    """

    quantiles = np.arange(0, 101, 10)

    fig, _ = plt.subplots()

    cmap = plt.cm.seismic #pylint: disable=no-member

    list_x, list_y = list_xy

    list_x_mean = []

    for i in range(len(quantiles)-1):
        low = int(np.ceil(len(list_x)*quantiles[i]/100))
        up = int(np.ceil(len(list_x)*quantiles[i+1]/100))
        list_x_mean.append(np.mean(list_x[low:up]))
        # plt.hlines(np.mean(list_y[low:up]),list_x[low],list_x[up-1], color=colorcycle[i], linewidth=2)

    vmax = np.max(np.abs(list_x_mean))

    for i in range(len(quantiles)-1):
        color = cmap((list_x_mean[i] + vmax)/(2*vmax))
        low = int(np.ceil(len(list_x)*quantiles[i]/100))
        up = int(np.ceil(len(list_x)*quantiles[i+1]/100))
        plt.scatter(list_x[low:up], list_y[low:up], color=color,s=0.8)
        plt.errorbar(np.mean(list_x[low:up]), np.mean(list_y[low:up]), np.std(list_y[low:up]), np.std(list_x[low:up]), color=color)
        plt.scatter(np.mean(list_x[low:up]), np.mean(list_y[low:up]), color=color, s=50)
        # plt.hlines(np.mean(list_y[low:up]),list_x[low],list_x[up-1], color=colorcycle[i], linewidth=2)

    slope, intercept, r, _, _ = linregress(list_x, list_y)
    u = np.arange(np.min(list_x)-0.1, np.max(list_x)+0.1, 0.01)
    print('R-square:', f"{r**2:.4f}", ', regression slope:', f"{slope:.4f}" , ', regression intercept:', f"{intercept:.4f}")
    plt.plot(u, slope*u+intercept, c='grey', label=f'slope: {round(slope,3)}')

    plt.xlabel('Mean background GST [°C]')
    plt.ylabel('Mean GST evolution [°C]')

    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()

    return fig

def plot_GST_bkg_vs_evol_quantile_bins_fit_single_site_from_inputs(site, path_pickle):
    """ Function return scatter plot of background GST vs GST evolution for a single site.
    The site is binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced too.
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    
    Returns
    -------
    fig : figure
        Scatter plot of background GST vs GST evolution for the single site.
        The single site is binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
        A linear regression is produced too.
    """

    list_xy = sorted_bkg_GST_data(site, path_pickle)
    fig =plot_GST_bkg_vs_evol_quantile_bins_fit_single_site(list_xy)

    return fig

def sorted_bkg_GST_data_two_sites(list_site, list_path_pickle):
    """ Function returns background GST and GST evolution for two sites.
        For each site, the two lists have the same order and are ordered in background GST increasing order
    
    Parameters
    ----------
    list_site : list
        List of labels for the site of each entry
    list_path_pickle : list of str
        List of string path to the location of the folder where the pickles are saved
    
    Returns
    -------
    list_xy : list
        List of lists of background GST and GST evolution for a single site.
        The nth element of each list corresponds to the same simulation
        in the shape [[list_x_site0, list_y_site0], [list_x_site1, list_y_site1]]
    """

    list_xy = [[] for _ in list_site]

    num = len(list_site)

    for i in range(num):
        list_xy[i] = sorted_bkg_GST_data(list_site[i], list_path_pickle[i])

    return list_xy

def plot_GST_bkg_vs_evol_quantile_bins_fit_two_sites(list_xy, list_label_site):
    """ Function return scatter plot of background GST vs GST evolution for 2 sites.
        Both sites are binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
        A linear regression is produced for each site.
    
    Parameters
    ----------
    list_xy : list
        List of lists of background GST and GST evolution for a single site.
        The nth element of each list corresponds to the same simulation
        in the shape [[list_x_site0, list_y_site0], [list_x_site1, list_y_site1]]
    list_label_site : list of str
        List of label for each site
    
    Returns
    -------
    fig : figure
        List of linear regression r-value (1 for each site). Need to square it to get R^2.
        Scatter plot of background GST vs GST evolution for 2 sites.
        Both sites are binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
        A linear regression is produced for each site.
    """

    num = len(list_xy)

    list_x = [[] for _ in list_xy]
    list_y = [[] for _ in list_xy]

    for i in range(num):
        list_x[i], list_y[i] = list_xy[i]

    quantiles = np.arange(0, 101, 10)

    fig, _ = plt.subplots()

    cmap = plt.cm.seismic #pylint: disable=no-member
    # colors = cmap(np.linspace(0, 1, len(quantiles)+(1 if len(quantiles)%2 else 0)))

    vmax = [[] for _ in list_xy]
    list_x_mean = [[] for _ in list_xy]

    for j in range(num):

        for i in range(len(quantiles)-1):
            low = int(np.ceil(len(list_x[j])*quantiles[i]/100))
            up = int(np.ceil(len(list_x[j])*quantiles[i+1]/100))
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
            low = int(np.ceil(len(list_x[j])*quantiles[i]/100))
            up = int(np.ceil(len(list_x[j])*quantiles[i+1]/100))
            plt.scatter(list_x[j][low:up], list_y[j][low:up], color=color,s=0.8)
            plt.errorbar(np.mean(list_x[j][low:up]), np.mean(list_y[j][low:up]), np.std(list_y[j][low:up]), np.std(list_x[j][low:up]), color=color)
            plt.scatter(np.mean(list_x[j][low:up]), np.mean(list_y[j][low:up]), color=color, s=50)
            # plt.hlines(np.mean(list_y[low:up]),list_x[low],list_x[up-1], color=colorcycle[i], linewidth=2)

        slope_i, intercept_i, r_i, _, _ = linregress(list_x[j], list_y[j])
        slope.append(slope_i)
        intercept.append(intercept_i)
        r.append(r_i)
        u = np.arange(np.min(list_x[j])-0.05, np.max(list_x[j])+0.05, 0.01)
        print('R-square:', r_i**2, ', regression slope:', slope_i , ', regression intercept:', intercept_i)
        line_j, = plt.plot(u, slope_i*u+intercept_i, c=colorcycle[j], label=list_label_site[j])
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

    return fig

def plot_GST_bkg_vs_evol_quantile_bins_fit_two_sites_from_input(list_site, list_path_pickle, list_label_site):
    """ Function return scatter plot of background GST vs GST evolution for 2 sites.
    Both sites are binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced for each site.
    
    Parameters
    ----------
    list_site : list
        List of labels for the site of each entry
    list_path_pickle : list of str
        List of string path to the location of the folder where the pickles are saved
    list_label_site : list of str
        List of label for each site
    
    Returns
    -------
    fig : figure
        List of linear regression r-value (1 for each site). Need to square it to get R^2.
        Scatter plot of background GST vs GST evolution for 2 sites.
        Both sites are binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
        A linear regression is produced for each site.
    """

    list_xy = sorted_bkg_GST_data_two_sites(list_site, list_path_pickle)
    fig = plot_GST_bkg_vs_evol_quantile_bins_fit_two_sites(list_xy, list_label_site)

    return fig

def data_bkg_GST_evol_altitudes(site, path_pickle):
    """ Function returns background GST and GST evolution for a single site for each altitude
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved


    Returns
    -------
    dic_data : dict
        Dictionary with background GST and GST evolution for each altitude with keys ['bkg', 'evol']
        in the form e.g.
        {'bkg': {2900: [...], 3100: [...], ...}, 'evol': {2900: [...], 3100: [...], ...}}
    """

    table_all = table_background_evolution_mean_GST_aspect_slope(site, path_pickle)
    dic_data = {period: {k: [i for j in v.columns for i in v[j]] for k, v in table_all[period].items()}
                for period in ['bkg', 'evol']}
    
    return dic_data
    
def plot_mean_bkg_GST_vs_evolution(dic_data):
    """ Function returns a scatter plot of mean background GST (ground-surface temperature)
        vs evolution of mean GST between the background and transient period.
        Note that each point is computed from an average over all reanalyses to avoid bias.
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved


    Returns
    -------
    fig : figure
        Scatter plot
    """

    xx = [i for i in dic_data['bkg'].values()]
    yy = [i for i in dic_data['evol'].values()]

    alt_list = list(dic_data['bkg'].keys())

    fig, _ = plt.subplots()

    for i,x in enumerate(xx):
        slope, intercept, r, _, _ = linregress(x,yy[i])
        print('altitude:', alt_list[i], ', R-square:', f"{r**2:.4f}", ', regression slope:', f"{slope:.4f}", ', regression intercept:', f"{intercept:.4f}")
        u = np.arange(np.min(x)-0.1, np.max(x)+0.1, 0.01)
        plt.scatter(x,yy[i], c=colorcycle[i], label=f'{alt_list[i]} m')
        plt.plot(u, slope*u+intercept, c=colorcycle[i], label=f'slope: {round(slope,3)}')

    plt.legend(loc='lower left')
    plt.xlabel('Mean background GST [°C]')
    plt.ylabel('Mean GST evolution [°C]')

    # displaying the scatter plot
    plt.show()
    plt.close()

    return fig

def plot_mean_bkg_GST_vs_evolution_from_inputs(site, path_pickle):
    """ Function returns a scatter plot of mean background GST (ground-surface temperature)
        vs evolution of mean GST between the background and transient period.
        Note that each point is computed from an average over all reanalyses to avoid bias.
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved


    Returns
    -------
    fig : figure
        Scatter plot
    """

    dic_data = data_bkg_GST_evol_altitudes(site, path_pickle)
    fig = plot_mean_bkg_GST_vs_evolution(dic_data)

    return fig

def data_evolution_snow_cover_melt_out(site, path_pickle, variable=None, value=None):
    """ Function returns lists of evolution of snow cover (in days) and melt out date
        between background and transient periods
        For all simulations or for a given subset, for instance the ones with 'slope'=55
        Note that only the simulations with snow are accounted for,
        otherwise, we would get a huge spike at 0 for all simulations that had no snow and kept it this way.

    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    variable : str
        A parameter in df_stats such as slope, aspect, altitude, etc.
    value : float
        A value for the parameter, e.g. 50 for a slope

    Returns
    -------
    evol_melt_out : list
        List of the evolution of melt out date
    evol_snow_cover : list
        List of the evolution of snow cover
    """
    
    pkl = load_all_pickles(site, path_pickle)
    df_stats = pkl['df_stats']

    # creates a subset of df_stats given the value of the variable entered as input. e.g. 'slope'=50
    if variable is None:
        data = df_stats
    else:
        data = df_stats[df_stats[variable]==value]

    # creates a list of the time evolution of both parameters
    # makes sure to only keep the simulations that have shown at least 1 day of snow over the whole study period
    evol_melt_out = [data.iloc[k].melt_out_trans - data.iloc[k].melt_out_bkg for k in range(len(data)) if (data.iloc[k].frac_snow_bkg != 0) | (data.iloc[k].frac_snow_trans != 0)]
    evol_snow_cover = [(data.iloc[k].frac_snow_trans - data.iloc[k].frac_snow_bkg)*365.25 for k in range(len(data)) if (data.iloc[k].frac_snow_bkg != 0) | (data.iloc[k].frac_snow_trans != 0)]

    return evol_melt_out, evol_snow_cover

def plot_evolution_snow_cover_melt_out(evol_melt_out, evol_snow_cover):
    """ Function returns a histogram of the evolution of snow cover (in days) and melt out date
        between background and transient periods
        For all simulations or for a given subset, for instance the ones with 'slope'=55
        Note that only the simulations with snow are accounted for,
        otherwise, we would get a huge spike at 0 for all simulations that had no snow and kept it this way.

    Parameters
    ----------
    evol_melt_out : list
        List of the evolution of melt out date
    evol_snow_cover : list
        List of the evolution of snow cover

    Returns
    -------
    fig : figure
        Histogram
    """
    fig, _ = plt.subplots()

    # plots both histograms
    plt.hist(evol_snow_cover, bins=20, alpha=0.75, weights=np.ones_like(evol_snow_cover) / len(evol_snow_cover), label='Snow cover')
    plt.hist(evol_melt_out, bins=20, alpha=0.75, weights=np.ones_like(evol_melt_out) / len(evol_melt_out), label='Melt out date')

    mean_snow_cov = np.mean(evol_snow_cover)
    mean_melt_out = np.mean(evol_melt_out)

    # adds a vertical line denoting the mean values
    plt.axvline(mean_snow_cov, color=colorcycle[0], linestyle='dashed', linewidth=2)
    plt.axvline(mean_melt_out, color=colorcycle[1], linestyle='dashed', linewidth=2)

    plt.annotate(r"$\overline{\Delta}_{\rm snow\ cover}=$%s%s [days]" % (("+" if mean_snow_cov > 0 else ""), float(f"{mean_snow_cov:.2f}")),
                 (0.12,0.5), xycoords='figure fraction',
                 fontsize=12, horizontalalignment='left', verticalalignment='top', color=colorcycle[0])
    plt.annotate(r"$\overline{\Delta}_{\rm mod}=$%s%s [days]" % (("+" if mean_melt_out > 0 else ""), float(f"{mean_melt_out:.2f}")),
                 (0.12,0.45), xycoords='figure fraction',
                 fontsize=12, horizontalalignment='left', verticalalignment='top', color=colorcycle[1])

    plt.xlabel('Evolution [days]')
    plt.ylabel('Frequency')

    # Show the graph
    plt.legend(loc='upper left')
    plt.show()
    plt.close()

    return fig

def plot_evolution_snow_cover_melt_out_from_inputs(site, path_pickle, variable=None, value=None):
    """ Function returns a histogram of the evolution of snow cover (in days) and melt out date
        between background and transient periods
        For all simulations or for a given subset, for instance the ones with 'slope'=55
        Note that only the simulations with snow are accounted for,
        otherwise, we would get a huge spike at 0 for all simulations that had no snow and kept it this way.

    Parameters
    ----------
    evol_melt_out : list
        List of the evolution of melt out date
    evol_snow_cover : list
        List of the evolution of snow cover

    Returns
    -------
    fig : figure
        Histogram
    """

    evol_melt_out, evol_snow_cover = data_evolution_snow_cover_melt_out(site, path_pickle, variable, value)
    fig = plot_evolution_snow_cover_melt_out(evol_melt_out, evol_snow_cover)

    return fig
