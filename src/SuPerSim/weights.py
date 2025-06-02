"""This module creates weights for averaging timeseries"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SuPerSim.open import open_thaw_depth_nc
from SuPerSim.constants import colorcycle
from SuPerSim.pickling import load_all_pickles

def assign_weight_sim(site, path_pickle, no_weight):
    """ Function returns a statistical weight for each simulation according to the importance in rockfall starting zone 
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    no_weight : bool, optional
        If True, all simulations have the same weight, otherwise the weight is computed as a function of altitude, aspect, and slope

    Returns
    -------
    pd_weight : pandas.core.frame.DataFrame
        Panda DataFrame assigning a statistical weight to each simulation for each of 'altitude', 'aspect', 'slope'
        and an overall weight.
    pd_weight_long : pandas.core.frame.DataFrame
        Panda DataFrame assigning a statistical weight to each simulation for each of 'altitude', 'aspect', 'slope'
        and an overall weight, and has information about the topography of each simulations
    """

    pkl = load_all_pickles(site, path_pickle)
    df_stats = pkl['df_stats']
    rockfall_values = pkl['rockfall_values']

    dict_weight = {}
    if no_weight:
        dict_weight = {i: [df_stats['altitude'].loc[i], 1, df_stats['aspect'].loc[i], 1, df_stats['slope'].loc[i], 1]
                       for i in list(df_stats.index.values)}
    else:
        if rockfall_values['exact_topo']:
            alt_distance = np.max([np.abs(i-rockfall_values['altitude']) for i in np.sort(np.unique(df_stats['altitude']))])
            dict_weight = {i: [df_stats['altitude'].loc[i],
                            1 - np.abs(df_stats.loc[i]['altitude']-rockfall_values['altitude'])/(2*alt_distance),
                            df_stats['aspect'].loc[i],
                            np.cos((np.pi)/180*(df_stats.loc[i]['aspect']-rockfall_values['aspect']))/4+3/4,
                            df_stats['slope'].loc[i],
                            np.cos((np.pi)/30*(df_stats.loc[i]['slope']-rockfall_values['slope']))/4+3/4]
                            for i in list(df_stats.index.values)}
        else:
            dict_weight = {i: [df_stats['altitude'].loc[i], 1, df_stats['aspect'].loc[i], 1, df_stats['slope'].loc[i], 1]
                        for i in list(df_stats.index.values)}

    pd_weight_long = pd.DataFrame.from_dict(dict_weight, orient='index',
                                        columns=['altitude', 'altitude_weight', 'aspect', 'aspect_weight', 'slope', 'slope_weight'])
    
    pd_weight_long['weight'] = pd_weight_long['altitude_weight']*pd_weight_long['aspect_weight']*pd_weight_long['slope_weight']
    pd_weight = pd_weight_long.drop(columns=['altitude', 'aspect', 'slope'])

    return pd_weight, pd_weight_long

def count_stat_weights(site, path_pickle, no_weight): 
    """ Function returns a binned count of the weight distribution over all (valid) simulations 
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    no_weight : bool, optional
        If True, all simulations have the same weight, otherwise the weight is computed as a function of altitude, aspect, and slope

    Returns
    -------
    bins : numpy.ndarray
        bins for histogram within the weight disctibution comprised between 0 and 1
    counts : numpy.ndarray
        number of simulations with a weight within each bin
    number_glaciers : int
        number of simulations yielding a 'glacier'
    """

    pd_weight, _ = assign_weight_sim(site, path_pickle, no_weight)

    pkl = load_all_pickles(site, path_pickle)
    df = pkl['df']

    list_hist = pd_weight.loc[:,'weight']
    counts, bins = np.histogram(list_hist, 10, (0, 1))
    
    number_glaciers = len(df)-len(pd_weight)
    
    return bins, counts, number_glaciers

def plot_hist_stat_weights(bins, counts, number_glaciers, show_glaciers): 
    """ Function returns a histogram of the weight distribution over all (valid) simulations 
        given a binned count of simulation weights and a number of glaciers
    
    Parameters
    ----------
    bins : numpy.ndarray
        bins for histogram within the weight disctibution comprised between 0 and 1
    counts : numpy.ndarray
        number of simulations with a weight within each bin
    number_glaciers : int
        number of simulations yielding a 'glacier'
    show_glaciers : bool, optional
        If True, shows the glacier simulations with a 0 weight, if False, those are ignored.

    Returns
    -------
    fig : figure
        Histogram
    """

    fig, _ = plt.subplots()

    tot_count = np.sum(counts) + (number_glaciers if show_glaciers else 0)
    counts_glaciers = [0 for _ in counts]
    counts_glaciers[0] = number_glaciers
    counts_all = counts + (counts_glaciers if show_glaciers else 0)
    
    plt.hist(bins[:-1], bins, weights=counts/tot_count, label='No glaciers', color=colorcycle[1])
    if show_glaciers:
        plt.hist(bins[:-1], bins, weights=counts_glaciers/tot_count, label='Glaciers', color=colorcycle[0])

    max_count = np.ceil((np.max(counts_all)/tot_count)/0.05+1)*0.05
    
    ticks = list(np.arange(0, max_count, 0.05))
    plt.yticks(ticks, [f"{i:0.2f}" for i in ticks])

    # Show the graph
    if show_glaciers:
        plt.legend(loc='upper right')
    plt.xlabel('Statistical weight')
    plt.ylabel('Frequency')
    plt.show()
    plt.close()

    return fig

def plot_hist_stat_weights_from_input(site, path_pickle, no_weight, show_glaciers): 
    """ Function returns a histogram of the weight distribution over all (valid) simulations 
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    no_weight : bool, optional
        If True, all simulations have the same weight, otherwise the weight is computed as a function of altitude, aspect, and slope
    show_glaciers : bool, optional
        If True, shows the glacier simulations with a 0 weight, if False, those are ignored.

    Returns
    -------
    fig : figure
        Histogram
    """

    bins, counts, number_glaciers = count_stat_weights(site, path_pickle, no_weight)
    fig = plot_hist_stat_weights(bins, counts, number_glaciers, show_glaciers)

    return fig

def count_perma_sim_per_variable(site, path_thaw_depth, path_pickle): 
    """ Function returns a binned count over all simulations
        of the number of valid/glacier simulations for each of the following variable
        ('altitude', 'aspect', 'slope', 'forcing') 
        It also shows the breakdown of valid simulations into permafrost and no-permafrost ones

    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    path_thaw_depth : str
        Path to the .nc file where the aggregated thaw depth simulations are stored
    path_pickle : str
        String path to the location of the folder where the pickles are saved

    Returns
    -------
    dict_bins : dict
        Dictionary giving the total number of simulation per variable
        e.g. dict_bins = {'altitude': [2900, 3100, 3300], 'aspect': [22.5, 45.0, 67.5], 'slope': [55, 60, 65, 70, 75], 'forcing': ['merra2']}
    dict_counts : dict
        Dictionary giving the number of simulation per variable and per parmafrost state
        e.g. dict_counts = {'altitude': {'Permafrost': [14, 37, 45], 'No permafrost, no glaciers': [31, 8, 0], 'Glaciers': [0, 0, 0]}, 'aspect': {}, ...}
    """

    _, thaw_depth = open_thaw_depth_nc(path_thaw_depth)

    pkl = load_all_pickles(site, path_pickle)
    df = pkl['df']
    df_stats = pkl['df_stats']

    variables = ['altitude','aspect','slope','forcing']

    list_valid_sim = list(df_stats.index.values)

    list_no_perma = []
    for sim in list_valid_sim:
        if np.std(thaw_depth[sim,:]) < 1 and np.max(thaw_depth[sim,:])> 19:
            list_no_perma.append(sim)

    list_perma = list(set(list_valid_sim) - set(list_no_perma))

    dict_counts = {var: [] for var in variables}
    dict_bins = {var: [] for var in variables}

    for var in variables:
        # number_no_glaciers: number of valid simulations (without glaciers) per value in list_var_prev
        list_vals = list(np.unique(df_stats.loc[:, var]))
        _, number_no_glaciers = [list_vals, [list(df_stats.loc[:, var]).count(i) for i in list_vals]]
        _, number_perma = [list_vals, [list(df_stats.loc[list_perma, var]).count(i) for i in list_vals]]
        
        # total number of simulations per value per variable
        dict_bins[var], tot = [list(i) for i in np.unique(df.loc[:, var], return_counts=True)]

        if len(number_perma) == 0:
            number_perma = [0 for _ in tot]

        # number_glaciers: number of glaciers per value per variable
        number_glaciers = [tot[i] - number_no_glaciers[i] for i in range(len(tot))]
        # number_no_perma: number of simulation swith no glaciers and no permafrost per value per variable
        number_no_perma = [number_no_glaciers[i] - number_perma[i] for i in range(len(tot))]

        counts = {
            'Permafrost': number_perma,
            'No permafrost, no glaciers': number_no_perma,
            'Glaciers': number_glaciers
        }

        dict_counts[var] = counts

    return dict_bins, dict_counts

def plot_hist_valid_sim_all_variables(dict_bins, dict_counts): 
    """ Function returns a histogram of the number of valid/glacier simulations for each of the following variable
        ('altitude', 'aspect', 'slope', 'forcing') 
        It also shows the breakdown of valid simulations into permafrost and no-permafrost ones

    Parameters
    ----------
    dict_bins : dict
        Dictionary giving the total number of simulation per variable
        e.g. dict_bins = {'altitude': [2900, 3100, 3300], 'aspect': [22.5, 45.0, 67.5], 'slope': [55, 60, 65, 70, 75], 'forcing': ['merra2']}
    dict_counts : dict
        Dictionary giving the number of simulation per variable and per parmafrost state
        e.g. dict_counts = {'altitude': {'Permafrost': [14, 37, 45], 'No permafrost, no glaciers': [31, 8, 0], 'Glaciers': [0, 0, 0]}, 'aspect': {}, ...}

    Returns
    -------
    fig : figure
        Histogram (subplot(2,2))
    """

    variables = ['altitude','aspect','slope','forcing']
    xaxes = ['Altitude [m]','Aspect [°]','Slope [°]','Forcing']
    yaxes = ['Number of simulations','','Number of simulations','']

    tot_per_bin = {var: np.sum([i[0] for i in dict_counts[var].values()]) for var in variables}

    fig, axs = plt.subplots(2,2, figsize=(6,6))
    for idx,ax in enumerate(axs.ravel()):
        bottom = np.zeros(len(dict_bins[variables[idx]]))

        colorbar = {
            'Permafrost': colorcycle[1],
            'No permafrost, no glaciers': colorcycle[2],
            'Glaciers': colorcycle[0]
        }

        for name, data in dict_counts[variables[idx]].items():
            p = ax.bar([str(i) for i in dict_bins[variables[idx]]], data, label=name, bottom=bottom, color=colorbar[name])
            bottom += data
            data_no_zero = [i if i>0 else "" for i in data]
            ax.bar_label(p, labels=data_no_zero, label_type='center')

        ax.set_xlabel(xaxes[idx])
        ax.set_ylim(0, tot_per_bin[variables[idx]])
        ax.set_ylabel(yaxes[idx])

    fig.align_ylabels(axs[:,0])
    plt.legend(loc='lower right', reverse=True)
    plt.tight_layout()
    plt.show()
    plt.close()

    return fig

def plot_hist_valid_sim_all_variables_from_input(site, path_thaw_depth, path_pickle): 
    """ Function returns a binned count over all simulations
        of the number of valid/glacier simulations for each of the following variable
        ('altitude', 'aspect', 'slope', 'forcing') 
        It also shows the breakdown of valid simulations into permafrost and no-permafrost ones

    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    path_thaw_depth : str
        Path to the .nc file where the aggregated thaw depth simulations are stored
    path_pickle : str
        String path to the location of the folder where the pickles are saved 

    Returns
    -------
    fig : figure
        Histogram (subplot(2,2))
    """

    dict_bins, dict_counts = count_perma_sim_per_variable(site, path_thaw_depth, path_pickle)
    fig = plot_hist_valid_sim_all_variables(dict_bins, dict_counts)

    return fig
