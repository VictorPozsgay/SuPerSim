"""This module creates weights for averaging timeseries"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from open import open_thaw_depth_nc
from constants import save_constants

colorcycle, _ = save_constants()

def assign_weight_sim(site, path_pickle, no_weight=True):
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
    """

    file_name_df_stats = f"df_stats{('' if site=='' else '_')}{site}.pkl"
    my_path_df_stats = path_pickle + file_name_df_stats

    with open(my_path_df_stats, 'rb') as file: 
        # Call load method to deserialize 
        df_stats = pickle.load(file)

    file_name_rockfall_values = f"rockfall_values{('' if site=='' else '_')}{site}.pkl"
    my_path_rockfall_values = path_pickle + file_name_rockfall_values

    with open(my_path_rockfall_values, 'rb') as file: 
        # Call load method to deserialize 
        rockfall_values = pickle.load(file)

    dict_weight = {}
    if no_weight:
        dict_weight = {i: [1,1,1] for i in list(df_stats.index.values)}
    else:
        if rockfall_values['exact_topo']:
            alt_distance = np.max([np.abs(i-rockfall_values['altitude']) for i in np.sort(np.unique(df_stats['altitude']))])
            dict_weight = {i: [1 - np.abs(df_stats.loc[i]['altitude']-rockfall_values['altitude'])/(2*alt_distance),
                            np.cos((np.pi)/180*(df_stats.loc[i]['aspect']-rockfall_values['aspect']))/4+3/4,
                            np.cos((np.pi)/30*(df_stats.loc[i]['slope']-rockfall_values['slope']))/4+3/4]
                            for i in list(df_stats.index.values)}
        else:
            dict_weight = {i: [1,1,1] for i in list(df_stats.index.values)}
    
    pd_weight = pd.DataFrame.from_dict(dict_weight, orient='index',
                                       columns=['altitude', 'aspect', 'slope'])
    pd_weight['weight'] = pd_weight['altitude']*pd_weight['aspect']*pd_weight['slope']
    
    return pd_weight

def plot_hist_valid_sim_all_variables(site, path_thaw_depth, path_pickle): 
    """ Function returns a histogram of the number of valid/glacier simulations for each of the following variable
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
    Histogram (subplot(2,2))
    """

    file_name_df = f"df{('' if site=='' else '_')}{site}.pkl"
    file_name_df_stats = f"df_stats{('' if site=='' else '_')}{site}.pkl"

    _, thaw_depth = open_thaw_depth_nc(path_thaw_depth)

    with open(path_pickle + file_name_df, 'rb') as file: 
        # Call load method to deserialize 
        df = pickle.load(file)

    with open(path_pickle + file_name_df_stats, 'rb') as file: 
        # Call load method to deserialize 
        df_stats = pickle.load(file)

    data=np.random.random((4,10))
    variables = ['altitude','aspect','slope','forcing']
    xaxes = ['Altitude [m]','Aspect [°]','Slope [°]','Forcing']
    yaxes = ['Number of simulations','','Number of simulations','']

    list_valid_sim = list(df_stats.index.values)

    list_no_perma = []
    for sim in list_valid_sim:
        if np.std(thaw_depth[sim,:]) < 1 and np.max(thaw_depth[sim,:])> 19:
            list_no_perma.append(sim)

    list_perma = list(set(list_valid_sim) - set(list_no_perma))

    f, a = plt.subplots(2,2, figsize=(6,6))
    for idx,ax in enumerate(a.ravel()):
        # list_var_prev: lists values the variable can take, e.g. [0, 45, 90, 135, etc.] for 'aspect'
        # number_no_glaciers: number of valid simulations (without glaciers) per value in list_var_prev
        list_var_prev, number_no_glaciers = np.unique(df_stats.loc[:, variables[idx]], return_counts=True)
        _, number_perma = np.unique((df_stats.loc[list_perma, :]).loc[:, variables[idx]], return_counts=True)
        print(number_perma)
        
        # translate into strings
        list_var = [str(i) for i in list_var_prev]
        # total number of simulations per value in list_var
        tot = list(np.unique(df.loc[:, variables[idx]], return_counts=True)[1])

        if len(number_perma) == 0:
            number_perma = [0]*len(tot)

        # number_glaciers: number of glaciers per value in list_var
        number_glaciers = [tot[i] - number_no_glaciers[i] for i in range(len(tot))]
        # number_no_perma: number of simulation swith no glaciers and no permafrost per value in list_var
        number_no_perma = [number_no_glaciers[i] - number_perma[i] for i in range(len(tot))]


        bottom = np.zeros(len(tot))
        counts = {
            'Permafrost': number_perma,
            'No permafrost, no glaciers': number_no_perma,
            'Glaciers': number_glaciers
        }

        colorbar = {
            'Permafrost': colorcycle[1],
            'No permafrost, no glaciers': colorcycle[2],
            'Glaciers': colorcycle[0]
        }

        for name, data in counts.items():
            p = ax.bar(list_var, data, label=name, bottom=bottom, color=colorbar[name])
            bottom += data
            data_no_zero = [i if i>0 else "" for i in data]
            ax.bar_label(p, labels=data_no_zero, label_type='center')

        ax.set_xlabel(xaxes[idx])
        ax.set_ylim(0, np.max(tot))
        ax.set_ylabel(yaxes[idx])


    f.align_ylabels(a[:,0])
    plt.legend(loc='lower right', reverse=True)
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.clf()

def plot_hist_stat_weights(pd_weight, df, zero=True): 
    """ Function returns a histogram of the weight distribution over all (valid) simulations 
    
    Parameters
    ----------
    pd_weight : pandas.core.frame.DataFrame
        Panda DataFrame assigning a statistical weight to each simulation for each of 'altitude', 'aspect', 'slope'
        and an overall weight.
    df : pandas.core.frame.DataFrame
        Panda DataFrame df, used to count the total number of simulations pre glacier filter and hence count total number of glaciers
    df_stats : pandas.core.frame.DataFrame
        Panda DataFrame df_stats, should at least include the following columns: 'altitude', 'aspect', 'slope'
    zero : bool, optional
        If True, shows the glacier simulations with a 0 weight, if False, those are ignored.

    Returns
    -------
    Histogram
    """

    list_hist = list(pd_weight['weight'])
    list_hist_b = [0 for _ in range(len(df)-len(pd_weight))]

    counts, bins = np.histogram(list_hist, 10, (0, 1))
    counts_b = 0
    if zero:
        counts_b, bins_b = np.histogram(list_hist_b, 10, (0, 1))
    tot_count = np.sum(counts) + (np.sum(counts_b) if zero else 0)
    
    plt.hist(bins[:-1], bins, weights=counts/tot_count, label='No glaciers', color=colorcycle[1])
    if zero:
        plt.hist(bins_b[:-1], bins_b, weights=counts_b/tot_count, label='Glaciers', color=colorcycle[0])

    max_count = np.ceil((np.max([np.max(counts), np.max(counts_b)])/tot_count)/0.05+1)*0.05
    
    ticks = list(np.arange(0, max_count, 0.05))
    plt.yticks(ticks, [f"{i:0.2f}" for i in ticks])

    # Show the graph
    if zero:
        plt.legend(loc='upper right')
    plt.xlabel('Statistical weight')
    plt.ylabel('Frequency')
    plt.show()
    plt.close()
    plt.clf()
