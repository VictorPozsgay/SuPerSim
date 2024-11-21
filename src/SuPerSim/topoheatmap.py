"""This module creates running statistics for timerseries"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sn
import cmasher as cmr

from SuPerSim.pickling import load_all_pickles
from SuPerSim.constants import colorcycle
from SuPerSim.open import open_thaw_depth_nc

def table_background_evolution_mean_GST_aspect_slope(site, path_pickle, path_thaw_depth=None):
    """ Function returns dataframes of mean background and evolution of GST (ground-surface temperature)
        as a function of slope, aspect, and altitude
        same for number of simulations per cell and permafrost state
        and a dictionary of values of ['aspect', 'slope','altitude', 'forcing']

    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    path_thaw_depth : str
        Path to the .nc file where the aggregated thaw depth simulations are stored
    
    Returns
    -------
    table_all : dict
        Dictionray containing:
            df_bkg : dict of pandas.core.frame.DataFrame
                Dictionary assigning a dataframe of background GST for each aspect and slope
                e.g. df_bkg = {2900: panda with columns=aspect and rows=slope, 3000: ...}
            df_evol : dict of pandas.core.frame.DataFrame
                Average GST evolution between background and transient over all simulations in that cell
            df_num_sim : dict of pandas.core.frame.DataFrame
                Number of simulations in that cell
            df_perma : dict of pandas.core.frame.DataFrame
                Permafrost state in that cell, in ['glacier', 'permafrost', 'no permafrost']
            dic_var : dict
                Dictionary of values of ['aspect', 'slope','altitude', 'forcing'], e.g
                dic_var = {'aspect': array([22.5, 45. , 67.5]), 'slope': array([55, 60, 65, 70, 75]), ...}
    """

    pkl = load_all_pickles(site, path_pickle)
    df_stats = pkl['df_stats']
    list_valid_sim = pkl['list_valid_sim']


    variables = ['aspect', 'slope','altitude', 'forcing']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}

    if path_thaw_depth is not None:
        _, thaw_depth = open_thaw_depth_nc(path_thaw_depth)
        list_valid_sim = list(df_stats.index.values)
        list_no_perma = []
        for sim in list_valid_sim:
            if np.std(thaw_depth[sim,:]) < 1 and np.max(thaw_depth[sim,:])> 19:
                list_no_perma.append(sim)
        list_perma = list(set(list_valid_sim) - set(list_no_perma))

    list_grd_temp = {altitude: {slope: {aspect: [] for aspect in dic_var['aspect']} for slope in dic_var['slope']} for altitude in dic_var['altitude']}
    list_mean_grd_temp = {altitude: {slope: {aspect: [] for aspect in dic_var['aspect']} for slope in dic_var['slope']} for altitude in dic_var['altitude']}
    list_diff_temp = {altitude: {slope: {aspect: [] for aspect in dic_var['aspect']} for slope in dic_var['slope']} for altitude in dic_var['altitude']}
    list_mean_diff_temp = {altitude: {slope: {aspect: [] for aspect in dic_var['aspect']} for slope in dic_var['slope']} for altitude in dic_var['altitude']}
    list_num_sim = {altitude: {slope: {aspect: [] for aspect in dic_var['aspect']} for slope in dic_var['slope']} for altitude in dic_var['altitude']}
    check_perma_list = {altitude: {slope: {aspect: [] for aspect in dic_var['aspect']} for slope in dic_var['slope']} for altitude in dic_var['altitude']}

    for altitude in dic_var['altitude']:
        for slope in dic_var['slope']:
            for aspect in dic_var['aspect']:
                df_this_sim = df_stats[(df_stats['aspect']==aspect) & (df_stats['slope']==slope) & (df_stats['altitude']==altitude)]
                list_grd_temp[altitude][slope][aspect] = list(df_this_sim['bkg_grd_temp'])
                list_sim_per_forcing = [list(df_this_sim['forcing']).count(i) for i in dic_var['forcing']]
                list_num_sim[altitude][slope][aspect] = (np.sum(list_sim_per_forcing) if len(set(list_sim_per_forcing))==1 else np.nan)
                list_mean_grd_temp[altitude][slope][aspect] = round(np.mean((list_grd_temp[altitude][slope][aspect])),3)
                list_diff_temp[altitude][slope][aspect] = list(df_this_sim['trans_grd_temp'] - df_this_sim['bkg_grd_temp'])
                list_mean_diff_temp[altitude][slope][aspect] = round(np.mean((list_diff_temp[altitude][slope][aspect])),3)
                if path_thaw_depth is not None:
                    temp_list = [i in list_perma for i in list(df_stats[(df_stats['altitude']==altitude) & (df_stats['aspect']==aspect) & (df_stats['slope']==slope)].index.values)]
                    check_perma_list[altitude][slope][aspect] = ('no permafrost' if len(temp_list)<3 else ('glacier' if all(temp_list) else 'permafrost'))
                else:
                    check_perma_list[altitude][slope][aspect] = np.nan

    df_bkg = {altitude: pd.DataFrame(list_mean_grd_temp[altitude]).T for altitude in dic_var['altitude']}
    df_evol = {altitude: pd.DataFrame(list_mean_diff_temp[altitude]).T for altitude in dic_var['altitude']}
    df_num_sim = {altitude: pd.DataFrame(list_num_sim[altitude]).T for altitude in dic_var['altitude']}
    df_perma = {altitude: pd.DataFrame(check_perma_list[altitude]).T for altitude in dic_var['altitude']}

    table_all = {'bkg': df_bkg, 'evol': df_evol, 'num_sim': df_num_sim, 'perma': df_perma, 'vars': dic_var}

    return table_all

def prepare_data_table_GST(site, path_pickle, path_thaw_depth=None):
    """ Function returns all data needed to produce topographic heatmaps
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    path_thaw_depth : str
        Path to the .nc file where the aggregated thaw depth simulations are stored


    Returns
    -------
    table_all : dict
        Dictionray containing:
            df_bkg : dict of pandas.core.frame.DataFrame
                Dictionary assigning a dataframe of background GST for each aspect and slope
                e.g. df_bkg = {2900: panda with columns=aspect and rows=slope, 3000: ...}
            df_evol : dict of pandas.core.frame.DataFrame
                Average GST evolution between background and transient over all simulations in that cell
            df_num_sim : dict of pandas.core.frame.DataFrame
                Number of simulations in that cell
            df_perma : dict of pandas.core.frame.DataFrame
                Permafrost state in that cell, in ['glacier', 'permafrost', 'no permafrost']
            dic_var : dict
                Dictionary of values of ['aspect', 'slope','altitude', 'forcing'], e.g
                dic_var = {'aspect': array([22.5, 45. , 67.5]), 'slope': array([55, 60, 65, 70, 75]), ...}
    rockfall_values : dict
        Dictionary with date and topography info for the event
    sim_per_cell : int
        number of simulations for a given cell with fixed (altitude, slope, aspect) triplet
    """

    pkl = load_all_pickles(site, path_pickle)
    df = pkl['df']
    rockfall_values = pkl['rockfall_values']

    table_all = table_background_evolution_mean_GST_aspect_slope(site, path_pickle, path_thaw_depth)
    dic_var = table_all['vars']
    
    variables = ['aspect', 'slope', 'altitude']
    # number of simulations for a given cell with fixed (altitude, slope, aspect) triplet
    sim_per_cell = int(len(df)/np.prod([len(dic_var[i]) for i in variables]))

    return table_all, rockfall_values, sim_per_cell

def prepare_data_table_GST_per_slope(site, path_pickle, path_thaw_depth=None):
    """ Function returns all data needed to produce topographic heatmaps
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    path_thaw_depth : str
        Path to the .nc file where the aggregated thaw depth simulations are stored


    Returns
    -------
    table_all : dict
        Dictionray containing:
            df_bkg : dict of pandas.core.frame.DataFrame
                Dictionary assigning a dataframe of background GST for each aspect and altitude, for a fixed slope
                e.g. df_bkg = {slope_1: panda with columns=aspect and rows=altitude, slope_2: ...}
            df_evol : dict of pandas.core.frame.DataFrame
                Average GST evolution between background and transient over all simulations in that cell
            df_num_sim : dict of pandas.core.frame.DataFrame
                Number of simulations in that cell
            df_perma : dict of pandas.core.frame.DataFrame
                Permafrost state in that cell, in ['glacier', 'permafrost', 'no permafrost']
            dic_var : dict
                Dictionary of values of ['aspect', 'slope', 'altitude', 'forcing'], e.g
                dic_var = {'aspect': array([22.5, 45. , 67.5]), 'slope': array([55, 60, 65, 70, 75]), ...}
    rockfall_values : dict
        Dictionary with date and topography info for the event
    sim_per_cell : int
        number of simulations for a given cell with fixed (altitude, slope, aspect) triplet
    """

    table_all, rockfall_values, sim_per_cell = prepare_data_table_GST(site, path_pickle, path_thaw_depth)

    table_all_new = table_all.copy()
    for k in ['bkg', 'evol', 'num_sim', 'perma']:
        table_all_new[k] = []
        table_all_new[k] = {slo: pd.DataFrame({alt: table_all[k][alt].loc[slo]
                                               for alt in table_all['vars']['altitude']}).T
                            for slo in table_all['vars']['slope']}

    return table_all_new, rockfall_values, sim_per_cell

def plot_table_mean_GST_aspect_slope_single_altitude(table_all, rockfall_values, altitude, background=True, box=True):
    """ Function returns a plot of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope and aspect, for a given altitude and higlights the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    table_all : dict
        Dictionray containing:
            df_bkg : dict of pandas.core.frame.DataFrame
                Dictionary assigning a dataframe of background GST for each aspect and slope
                e.g. df_bkg = {2900: panda with columns=aspect and rows=slope, 3000: ...}
            df_evol : dict of pandas.core.frame.DataFrame
                Average GST evolution between background and transient over all simulations in that cell
            df_num_sim : dict of pandas.core.frame.DataFrame
                Number of simulations in that cell
            df_perma : dict of pandas.core.frame.DataFrame
                Permafrost state in that cell, in ['glacier', 'permafrost', 'no permafrost']
            dic_var : dict
                Dictionary of values of ['aspect', 'slope','altitude', 'forcing'], e.g
                dic_var = {'aspect': array([22.5, 45. , 67.5]), 'slope': array([55, 60, 65, 70, 75]), ...}
    rockfall_values : dict
        Dictionary with date and topography info for the event
    altitude :
        desired altitude for the plot
    background : bool, optional 
        If True, plots the mean background value, else, plots the evolution of the mean
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box


    Returns
    -------
    fig : Figure
        heatmap of background or evolution of GST
    """

    list_mean = (table_all['bkg'][altitude] if background else table_all['evol'][altitude])

    vals = np.around(list_mean.values, 2)
    vals = vals[~np.isnan(vals)]
    dilute = 1
    min_vals = np.min(vals)
    max_vals = np.max(vals)
    range_vals = max_vals- min_vals
    normal = plt.Normalize((np.max(max_vals - 2*dilute*range_vals,0) if np.min(vals)>0 else -dilute*np.max(np.abs(vals))), dilute*np.max(np.abs(vals)))
    colours = plt.cm.seismic(normal(list_mean)) #pylint: disable=no-member

    for aspect in list_mean:
        for slope in list_mean.index:
            if np.isnan(list_mean[aspect].loc[slope]):
                colours[aspect][slope] = list(matplotlib.colors.to_rgba('silver'))

    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    the_table=plt.table(cellText=[list(list_mean.loc[i]) for i in list_mean.index], rowLabels=list_mean.index, colLabels=list_mean.columns, 
                        loc='center', cellLoc='center',
                        cellColours=colours)

    if box and rockfall_values['exact_topo']:
        if altitude == rockfall_values['altitude']:
            ax.add_patch(Rectangle(((rockfall_values['aspect'])/45*1/8, (70-rockfall_values['slope'])/10*1/6), 1/8, 1/6,
                        edgecolor = 'black', transform=ax.transAxes,
                        fill=False,
                        lw=4))

    the_table.scale(1, 3.7)
    the_table.set_fontsize(16)
    ax.axis('off')

    plt.text(-0.08, 5/12,'Slope [°]', fontsize= 16, rotation=90, horizontalalignment='right', verticalalignment='center')
    plt.text(0.5, 1,'Aspect [°]', fontsize= 16, rotation=0, horizontalalignment='center', verticalalignment='bottom')

    plt.tight_layout()
    plt.show()
    plt.close()

    return fig

def plot_table_mean_GST_aspect_slope_single_altitude_from_inputs(site, path_pickle, altitude, background=True, box=True):
    """ Function returns a plot of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope and aspect, for a given altitude and higlights the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    altitude :
        desired altitude for the plot
    background : bool, optional 
        If True, plots the mean background value, else, plots the evolution of the mean
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    fig : Figure
        heatmap of background or evolution of GST
    """

    table_all, rockfall_values, _ = prepare_data_table_GST(site, path_pickle)
    fig = plot_table_mean_GST_aspect_slope_single_altitude(table_all, rockfall_values, altitude, background, box)

    return fig

def plot_table_mean_GST_aspect_slope_all_altitudes(table_all, rockfall_values, sim_per_cell, show_glaciers, box=True):
    """ Function returns a plot of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlights the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    table_all : dict
        Dictionray containing:
            df_bkg : dict of pandas.core.frame.DataFrame
                Dictionary assigning a dataframe of background GST for each aspect and slope
                e.g. df_bkg = {2900: panda with columns=aspect and rows=slope, 3000: ...}
            df_evol : dict of pandas.core.frame.DataFrame
                Average GST evolution between background and transient over all simulations in that cell
            df_num_sim : dict of pandas.core.frame.DataFrame
                Number of simulations in that cell
            df_perma : dict of pandas.core.frame.DataFrame
                Permafrost state in that cell, in ['glacier', 'permafrost', 'no permafrost']
            dic_var : dict
                Dictionary of values of ['aspect', 'slope','altitude', 'forcing'], e.g
                dic_var = {'aspect': array([22.5, 45. , 67.5]), 'slope': array([55, 60, 65, 70, 75]), ...}
    rockfall_values : dict
        Dictionary with date and topography info for the event
    sim_per_cell : int
        number of simulations for a given cell with fixed (altitude, slope, aspect) triplet
    show_glaciers : bool, opional
        Whether or not to plot the glacier fraction
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box


    Returns
    -------
    fig : Figure
        (2 or 3)*(# altitudes) heatmaps of background GST, evolution of GST, and possibly number of glaciers
    """

    dic_var = table_all['vars']

    alt_list = list(dic_var['altitude'])
    for i,j in enumerate(alt_list):
        if box and rockfall_values['exact_topo']:
            if rockfall_values['altitude'] == j:
                alt_index = i

    # setting the parameter values 
    annot = True
    center = [0, 0]
    cmap = ['seismic', 'seismic']

    list_mean = [# evolution mean GST
                table_all['evol'],
                # background mean GST
                table_all['bkg']]

    labels_plot = ['Mean GST evolution [°C]', 'Mean background GST [°C]']
    if show_glaciers:
        # glacier fraction
        list_mean.append({alt: table_all['num_sim'][alt].transform(lambda x: ((sim_per_cell-x)/sim_per_cell*100)).astype(int) for alt in alt_list})
        labels_plot.append('Glacier fraction')
        cmap.append('BrBG')
        center.append(50)

    nrows= len(list_mean)
    ncols = len(list_mean[0])

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3*nrows), constrained_layout=True)
    no_nan = [[i[altitude][aspect][slope]
            for altitude in dic_var['altitude'] for aspect in dic_var['aspect'] for slope in dic_var['slope']
            if not np.isnan(i[altitude][aspect][slope])]
            for i in list_mean]
    vmin = [np.min(i) for i in no_nan]
    vmax = [np.max(i) for i in no_nan]
    if len(vmin) == 3:
        vmin[2] = 0
        vmax[2] = 100

    # plotting the heatmap 
    for j in range(nrows):
        if ncols == 1:
            sn.heatmap(data=list_mean[nrows-1-j][alt_list[0]], annot=annot, center=center[nrows-1-j], cmap=cmap[nrows-1-j], ax=axs[j], vmin=vmin[nrows-1-j], vmax=vmax[nrows-1-j],
                    cbar=True, yticklabels=True, xticklabels=(j==nrows-1), cbar_kws={'label': labels_plot[nrows-1-j]})
            axs[0].figure.axes[-1].yaxis.label.set_size(13)
            if box and rockfall_values['exact_topo']:
                axs[j].add_patch(Rectangle(((rockfall_values['aspect'])/45*1/8, (70-rockfall_values['slope'])/10*1/5), 1/8, 1/5,
                                        edgecolor = 'black', transform=axs[j].transAxes, fill=False, lw=4))
        if ncols > 1:
            for i in range(ncols):
                sn.heatmap(data=list_mean[nrows-1-j][alt_list[i]], annot=annot, center=center[nrows-1-j], cmap=cmap[nrows-1-j], ax=axs[j,i], vmin=vmin[nrows-1-j], vmax=vmax[nrows-1-j],
                            cbar=(i==ncols-1), yticklabels=(i==0), xticklabels=(j==nrows-1), cbar_kws={'label': labels_plot[nrows-1-j]})
            axs[0,0].figure.axes[-1].yaxis.label.set_size(13)
            if box and rockfall_values['exact_topo']:
                axs[j,alt_index].add_patch(Rectangle(((rockfall_values['aspect'])/45*1/8, (70-rockfall_values['slope'])/10*1/5), 1/8, 1/5,
                                                    edgecolor = 'black', transform=axs[j,alt_index].transAxes, fill=False, lw=4))

    if ncols == 1:
        axs[0].set_title(f'{alt_list[0]} m')
    if ncols > 1:
        for i in range(ncols):
            axs[0,i].set_title(f'{alt_list[i]} m')
    fig.supxlabel('Aspect [°]')
    fig.supylabel('Slope [°]')

    # displaying the plotted heatmap 
    plt.show()
    plt.close()

    return fig

def plot_table_mean_GST_aspect_slope_all_altitudes_from_inputs(site, path_pickle, show_glaciers, box=True):
    """ Function returns a plot of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlights the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    altitude :
        desired altitude for the plot
    show_glaciers : bool, opional
        Whether or not to plot the glacier fraction
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    fig : Figure
        (2 or 3)*(# altitudes) heatmaps of background GST, evolution of GST, and possibly number of glaciers
    """

    table_all, rockfall_values, sim_per_cell = prepare_data_table_GST(site, path_pickle)
    fig = plot_table_mean_GST_aspect_slope_all_altitudes(table_all, rockfall_values, sim_per_cell, show_glaciers, box)

    return fig

def plot_table_mean_GST_aspect_slope_all_altitudes_polar(table_all, rockfall_values, box=True):
    """ Function returns a polar plot per altitude of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    table_all : dict
        Dictionray containing:
            df_bkg : dict of pandas.core.frame.DataFrame
                Dictionary assigning a dataframe of background GST for each aspect and slope
                e.g. df_bkg = {2900: panda with columns=aspect and rows=slope, 3000: ...}
            df_evol : dict of pandas.core.frame.DataFrame
                Average GST evolution between background and transient over all simulations in that cell
            df_num_sim : dict of pandas.core.frame.DataFrame
                Number of simulations in that cell
            df_perma : dict of pandas.core.frame.DataFrame
                Permafrost state in that cell, in ['glacier', 'permafrost', 'no permafrost']
            dic_var : dict
                Dictionary of values of ['aspect', 'slope','altitude', 'forcing'], e.g
                dic_var = {'aspect': array([22.5, 45. , 67.5]), 'slope': array([55, 60, 65, 70, 75]), ...}
    rockfall_values : dict
        Dictionary with date and topography info for the event
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    fig : Figure
        2*(# altitudes) polar heatmaps of background GST and evolution of GST
    """

    dic_var = table_all['vars']

    alt_list = dic_var['altitude']
    for i,j in enumerate(alt_list):
        if box and rockfall_values['exact_topo']:
            if rockfall_values['altitude'] == j:
                alt_index = i

    list_mean = [table_all['bkg'], table_all['evol']]
    no_nan = [[i[altitude][aspect][slope]
                for altitude in dic_var['altitude'] for aspect in dic_var['aspect'] for slope in dic_var['slope']
                if not np.isnan(i[altitude][aspect][slope])]
                for i in list_mean]
    vmin = [np.min(i) for i in no_nan]
    vmax = [np.max(i) for i in no_nan]

    list_slope = dic_var['slope']
    list_aspect = dic_var['aspect']

    nbin_slope = len(list_slope)
    delta_slope = list_slope[1] - list_slope[0]
    nbin_aspect = len(list_aspect) # int(36/delta_aspect)
    delta_aspect = list_aspect[1] - list_aspect[0]

    # binning
    rbins = np.linspace(np.min(list_slope), np.max(list_slope), nbin_slope)
    abins = np.linspace(list_aspect[0]/360*2*np.pi, list_aspect[-1]/360*2*np.pi, nbin_aspect)
    subdivs = 100
    abins2 = np.linspace((list_aspect[0]-delta_aspect/2)/360*2*np.pi, (list_aspect[-1]+delta_aspect/2)/360*2*np.pi, subdivs*nbin_aspect)
    A, R = np.meshgrid(abins2, rbins)

    nrows = len(list_mean)
    ncols = len(list_mean[0])

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 5*nrows), subplot_kw=dict(projection="polar"),
                            gridspec_kw={'hspace': -0.1, 'wspace': 0.3})
    abs_max = [np.max(np.abs([vmin[i], vmax[i]])) for i in range(len(vmin))]

    cmap = [cmr.get_sub_cmap('seismic', 0.5 + 0.5*(vmin[i]/abs_max[i]), 0.5 + 0.5*(vmax[i]/abs_max[i])) for i in range(len(vmin))]

    ticks = [[],[]]
    tick_pos = [[],[]]
    for i,v in enumerate(vmin):
        for space in [0.05, 0.1, 0.2, 0.5, 1]:
            pre_ticks = np.arange(np.ceil(v/space)*space+0, np.min([vmax[i], np.floor(vmax[i]/space+1)*space]), space)
            if (len(pre_ticks)>=5 and len(pre_ticks)<10):
                ticks[i] = [round(j,(2 if space<0.1 else 1)) for j in pre_ticks]
        tick_pos[i] = [(j-v)/(vmax[i]-v) for j in ticks[i] if (j>=v and j<vmax[i])]

    for j in range(nrows):
        for i,alt in enumerate(dic_var['altitude']):
            temp_list = [list(list_mean[j][alt].loc[k]) for k in list_mean[j][alt].index]
            axs[j,i].pcolormesh(A, R, np.repeat(temp_list, subdivs, axis=1), cmap=cmap[j], vmin=vmin[j], vmax=vmax[j])
            axs[j,i].set_facecolor("silver")
            axs[j,i].set_theta_zero_location('N')
            axs[j,i].set_theta_direction(-1)
            axs[j,i].tick_params(axis='y', labelcolor='black')
            axs[j,i].set_xticks(abins)
            axs[j,i].set_yticks([])
            axs[j,i].xaxis.grid(False)
            axs[j,i].yaxis.grid(False)
            axs[j,i].set_rlim(int(np.max(list_slope)+delta_slope),int(np.min(list_slope)-delta_slope))
            axs[0,i].set_title(f'{alt_list[i]} m')
            axs[j,i].bar(0, 1).remove()
            for k in range(int(np.min(list_slope)), int(np.max(list_slope))+1,
                        int((np.max(list_slope)-np.min(dic_var['slope']))/(nbin_slope-1))):
                axs[j,i].text(np.pi/5, k, (f'{k}°'), horizontalalignment='center', verticalalignment='center')
        if box and rockfall_values['exact_topo']:
            axs[j,alt_index].add_patch(matplotlib.patches.Rectangle(((rockfall_values['aspect']-45/2)/360*2*np.pi, rockfall_values['slope']-5),
                                                        width=np.pi/4, height=10, edgecolor = 'black', fill=False, lw=2))
        

    for j in range(nrows):
        #pylint: disable=no-member
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap[j]), shrink=.7,
                        ax=axs[j,:].ravel().tolist(), orientation='vertical',
                        label=('Mean background GST [°C]' if j==0 else 'Mean GST evolution [°C]'))

    for idx,i in enumerate(range(-nrows,0)):
        axs[0,0].figure.axes[i].yaxis.set_ticks(tick_pos[idx])
        axs[0,0].figure.axes[i].set_yticklabels(ticks[idx]) 

    for i in [6,7]:
        axs[0,0].figure.axes[i].yaxis.label.set_size(13)

    plt.show()
    plt.close()

    return fig

def plot_table_mean_GST_aspect_altitude_all_slopes_polar(table_all, rockfall_values, box=True):
    """ Function returns a polar plot per slope of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    table_all : dict
        Dictionray containing:
            df_bkg : dict of pandas.core.frame.DataFrame
                Dictionary assigning a dataframe of background GST for each aspect and altitude, for a fixed slope
                e.g. df_bkg = {slope_1: panda with columns=aspect and rows=altitude, slope_2: ...}
            df_evol : dict of pandas.core.frame.DataFrame
                Average GST evolution between background and transient over all simulations in that cell
            df_num_sim : dict of pandas.core.frame.DataFrame
                Number of simulations in that cell
            df_perma : dict of pandas.core.frame.DataFrame
                Permafrost state in that cell, in ['glacier', 'permafrost', 'no permafrost']
            dic_var : dict
                Dictionary of values of ['aspect', 'slope','altitude', 'forcing'], e.g
                dic_var = {'aspect': array([22.5, 45. , 67.5]), 'slope': array([55, 60, 65, 70, 75]), ...}
    rockfall_values : dict
        Dictionary with date and topography info for the event
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    fig : Figure
        2*(# slopes) polar heatmaps of background GST and evolution of GST, for all aspects and altitudes
    """

    dic_var = table_all['vars']

    alt_list = dic_var['altitude']
    for i,j in enumerate(alt_list):
        if box and rockfall_values['exact_topo']:
            if rockfall_values['altitude'] == j:
                alt_index = i

    list_mean = [table_all['bkg'], table_all['evol']]
    no_nan = [[i[slope][aspect][altitude]
                for altitude in dic_var['altitude'] for aspect in dic_var['aspect'] for slope in dic_var['slope']
                if not np.isnan(i[slope][aspect][altitude])]
                for i in list_mean]
    vmin = [np.min(i) for i in no_nan]
    vmax = [np.max(i) for i in no_nan]

    list_slope = dic_var['slope']
    list_aspect = dic_var['aspect']

    nbin_alt = len(alt_list)
    delta_alt = alt_list[1] - alt_list[0]
    nbin_aspect = len(list_aspect) # int(36/delta_aspect)
    delta_aspect = list_aspect[1] - list_aspect[0]

    # binning
    rbins = np.linspace(np.min(alt_list), np.max(alt_list), nbin_alt)
    abins = np.linspace(list_aspect[0]/360*2*np.pi, list_aspect[-1]/360*2*np.pi, nbin_aspect)
    subdivs = 100
    abins2 = np.linspace((list_aspect[0]-delta_aspect/2)/360*2*np.pi, (list_aspect[-1]+delta_aspect/2)/360*2*np.pi, subdivs*nbin_aspect)
    A, R = np.meshgrid(abins2, rbins)

    nrows = len(list_mean)
    ncols = len(list_mean[0])

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 5*nrows), subplot_kw=dict(projection="polar"),
                            gridspec_kw={'hspace': -0.1, 'wspace': 0.3})
    abs_max = [np.max(np.abs([vmin[i], vmax[i]])) for i in range(len(vmin))]

    cmap = [cmr.get_sub_cmap('seismic', 0.5 + 0.5*(vmin[i]/abs_max[i]), 0.5 + 0.5*(vmax[i]/abs_max[i])) for i in range(len(vmin))]

    ticks = [[],[]]
    tick_pos = [[],[]]
    for i,v in enumerate(vmin):
        for space in [0.05, 0.1, 0.2, 0.5, 1]:
            pre_ticks = np.arange(np.ceil(v/space)*space+0, np.min([vmax[i], np.floor(vmax[i]/space+1)*space]), space)
            if (len(pre_ticks)>=5 and len(pre_ticks)<10):
                ticks[i] = [round(j,(2 if space<0.1 else 1)) for j in pre_ticks]
        tick_pos[i] = [(j-v)/(vmax[i]-v) for j in ticks[i] if (j>=v and j<vmax[i])]

    for j in range(nrows):
        for i,slo in enumerate(dic_var['slope']):
            temp_list = [list(list_mean[j][slo].loc[k]) for k in list_mean[j][slo].index]
            axs[j,i].pcolormesh(A, R, np.repeat(temp_list, subdivs, axis=1), cmap=cmap[j], vmin=vmin[j], vmax=vmax[j])
            axs[j,i].set_facecolor("silver")
            axs[j,i].set_theta_zero_location('N')
            axs[j,i].set_theta_direction(-1)
            axs[j,i].tick_params(axis='y', labelcolor='black')
            axs[j,i].set_xticks(abins)
            axs[j,i].set_yticks([])
            axs[j,i].xaxis.grid(False)
            axs[j,i].yaxis.grid(False)
            axs[j,i].set_rlim(int(np.max(alt_list)+delta_alt/2),int(np.min(alt_list)-delta_alt/2))
            axs[0,i].set_title(f'{list_slope[i]}°')
            axs[j,i].bar(0, 1).remove()
            for k in alt_list:
                axs[j,i].text(np.pi/5, k, (f'{int(k)}m'), horizontalalignment='center', verticalalignment='center')
            # for k in range(int(np.min(alt_list)), int(np.max(alt_list))+1,
            #             int((np.max(alt_list)-np.min(dic_var['altitude']))/(nbin_alt-1))):
            #     axs[j,i].text(np.pi/5, k, (f'{k}°'), horizontalalignment='center', verticalalignment='center')
        if box and rockfall_values['exact_topo']:
            axs[j,alt_index].add_patch(matplotlib.patches.Rectangle(((rockfall_values['aspect']-45/2)/360*2*np.pi, rockfall_values['altitude']-5),
                                                        width=np.pi/4, height=10, edgecolor = 'black', fill=False, lw=2))
        

    for j in range(nrows):
        #pylint: disable=no-member
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap[j]), shrink=.7,
                        ax=axs[j,:].ravel().tolist(), orientation='vertical',
                        label=('Mean background GST [°C]' if j==0 else 'Mean GST evolution [°C]'))

    for idx,i in enumerate(range(-nrows,0)):
        axs[0,0].figure.axes[i].yaxis.set_ticks(tick_pos[idx])
        axs[0,0].figure.axes[i].set_yticklabels(ticks[idx]) 

    for i in [6,7]:
        axs[0,0].figure.axes[i].yaxis.label.set_size(13)

    plt.show()
    plt.close()

    return fig

def plot_table_mean_GST_aspect_slope_all_altitudes_polar_from_inputs(site, path_pickle, box=True):
    """ Function returns a polar plot per altitude of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    fig : Figure
        2*(# altitudes) polar heatmaps of background GST and evolution of GST
    """

    table_all, rockfall_values, _ = prepare_data_table_GST(site, path_pickle)
    fig = plot_table_mean_GST_aspect_slope_all_altitudes_polar(table_all, rockfall_values, box)

    return fig

def plot_table_mean_GST_aspect_altitude_all_slopes_polar_from_inputs(site, path_pickle, box=True):
    """ Function returns a polar plot per altitude of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    fig : Figure
        2*(# altitudes) polar heatmaps of background GST and evolution of GST
    """

    table_all, rockfall_values, _ = prepare_data_table_GST_per_slope(site, path_pickle)
    fig = plot_table_mean_GST_aspect_altitude_all_slopes_polar(table_all, rockfall_values, box)

    return fig

def plot_permafrost_all_altitudes_polar(table_all, rockfall_values, box=True):
    """ Function returns a polar plot per altitude of the table of the permafrost state
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    table_all : dict
        Dictionray containing:
            df_bkg : dict of pandas.core.frame.DataFrame
                Dictionary assigning a dataframe of background GST for each aspect and slope
                e.g. df_bkg = {2900: panda with columns=aspect and rows=slope, 3000: ...}
            df_evol : dict of pandas.core.frame.DataFrame
                Average GST evolution between background and transient over all simulations in that cell
            df_num_sim : dict of pandas.core.frame.DataFrame
                Number of simulations in that cell
            df_perma : dict of pandas.core.frame.DataFrame
                Permafrost state in that cell, in ['glacier', 'permafrost', 'no permafrost']
            dic_var : dict
                Dictionary of values of ['aspect', 'slope','altitude', 'forcing'], e.g
                dic_var = {'aspect': array([22.5, 45. , 67.5]), 'slope': array([55, 60, 65, 70, 75]), ...}
    rockfall_values : dict
        Dictionary with date and topography info for the event
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    fig : Figure
        (# altitudes) polar heatmaps of permafrost state
    """

    dic_var = table_all['vars']

    alt_list = dic_var['altitude']
    for i,j in enumerate(alt_list):
        if box and rockfall_values['exact_topo']:
            if rockfall_values['altitude'] == j:
                alt_index = i

    data = table_all['perma']
    data_to_num = {alt: data[alt].replace(['glacier', 'permafrost', 'no permafrost'], [1/6, 1/2, 5/6]) for alt in data.keys()}

    list_slope = dic_var['slope']
    list_aspect = dic_var['aspect']

    nbin_slope = len(list_slope)
    delta_slope = list_slope[1] - list_slope[0]
    nbin_aspect = len(list_aspect) # int(36/delta_aspect)
    delta_aspect = list_aspect[1] - list_aspect[0]

    # binning
    rbins = np.linspace(np.min(list_slope), np.max(list_slope), nbin_slope)
    abins = np.linspace(list_aspect[0]/360*2*np.pi, list_aspect[-1]/360*2*np.pi, nbin_aspect)
    subdivs = 100
    abins2 = np.linspace((list_aspect[0]-delta_aspect/2)/360*2*np.pi, (list_aspect[-1]+delta_aspect/2)/360*2*np.pi, subdivs*nbin_aspect)
    A, R = np.meshgrid(abins2, rbins)

    ncols = len(data)
    nrows = 1

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 5*nrows), subplot_kw=dict(projection="polar"),
                            gridspec_kw={'hspace': -0.1, 'wspace': 0.3})

    cmap = matplotlib.colors.ListedColormap([colorcycle[1],colorcycle[2],colorcycle[0]]) 

    for i,alt in enumerate(dic_var['altitude']):
        axs[i].pcolormesh(A, R, np.repeat(data_to_num[alt], subdivs, axis=1), cmap=cmap, vmin=0, vmax=1)
        axs[i].set_facecolor("silver")
        axs[i].set_theta_zero_location('N')
        axs[i].set_theta_direction(-1)
        axs[i].tick_params(axis='y', labelcolor='black')
        axs[i].set_xticks(abins)
        axs[i].set_yticks([])
        axs[i].xaxis.grid(False)
        axs[i].yaxis.grid(False)
        axs[i].set_rlim(int(np.max(list_slope)+delta_slope),int(np.min(list_slope)-delta_slope))
        axs[i].set_title(f'{alt_list[i]} m')
        axs[i].bar(0, 1).remove()
        for k in range(int(np.min(list_slope)), int(np.max(list_slope))+1,
                        int((np.max(list_slope)-np.min(dic_var['slope']))/(nbin_slope-1))):
            axs[i].text(np.pi/5, k, (f'{k}°'), horizontalalignment='center', verticalalignment='center')
    if box and rockfall_values['exact_topo']:
        axs[alt_index].add_patch(matplotlib.patches.Rectangle(((rockfall_values['aspect']-45/2)/360*2*np.pi, rockfall_values['slope']-5),
                                                        width=np.pi/4, height=10, edgecolor = 'black', fill=False, lw=2))

    #pylint: disable=no-member
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap),
                    shrink=.8, ax=axs[:].ravel().tolist(), orientation='vertical')
    axs[0].figure.axes[-1].yaxis.set_ticks([1/6,1/2,5/6])
    axs[0].figure.axes[-1].set_yticklabels(['Permafrost', 'No permafrost, no glaciers', 'Glaciers']) 
    axs[0].figure.axes[-1].yaxis.label.set_size(15)

    plt.show()
    plt.close()

    return fig

def plot_permafrost_all_altitudes_polar_from_inputs(site, path_pickle, path_thaw_depth, box=True):
    """ Function returns a polar plot per altitude of the table of the permafrost state
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    path_thaw_depth : str
        Path to the .nc file where the aggregated thaw depth simulations are stored
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    fig : Figure
        (# altitudes) polar heatmaps of permafrost state
    """
    table_all, rockfall_values, _ = prepare_data_table_GST(site, path_pickle, path_thaw_depth)
    fig = plot_permafrost_all_altitudes_polar(table_all, rockfall_values, box)

    return fig

def plot_permafrost_all_slopes_polar(table_all, rockfall_values, box=True):
    """ Function returns a polar plot per slope of the table of the permafrost state
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    table_all : dict
        Dictionray containing:
            df_bkg : dict of pandas.core.frame.DataFrame
                Dictionary assigning a dataframe of background GST for each aspect and altitude, for a fixed slope
                e.g. df_bkg = {slope_1: panda with columns=aspect and rows=altitude, slope_2: ...}
            df_evol : dict of pandas.core.frame.DataFrame
                Average GST evolution between background and transient over all simulations in that cell
            df_num_sim : dict of pandas.core.frame.DataFrame
                Number of simulations in that cell
            df_perma : dict of pandas.core.frame.DataFrame
                Permafrost state in that cell, in ['glacier', 'permafrost', 'no permafrost']
            dic_var : dict
                Dictionary of values of ['aspect', 'slope','altitude', 'forcing'], e.g
                dic_var = {'aspect': array([22.5, 45. , 67.5]), 'slope': array([55, 60, 65, 70, 75]), ...}
    rockfall_values : dict
        Dictionary with date and topography info for the event
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    fig : Figure
        (# slopes) polar heatmaps of permafrost state
    """

    dic_var = table_all['vars']

    alt_list = dic_var['altitude']
    for i,j in enumerate(alt_list):
        if box and rockfall_values['exact_topo']:
            if rockfall_values['altitude'] == j:
                alt_index = i

    data = table_all['perma']
    data_to_num = {alt: data[alt].replace(['glacier', 'permafrost', 'no permafrost'], [1/6, 1/2, 5/6]) for alt in data.keys()}

    list_slope = dic_var['slope']
    list_aspect = dic_var['aspect']

    nbin_alt = len(alt_list)
    delta_alt = alt_list[1] - alt_list[0]
    nbin_aspect = len(list_aspect) # int(36/delta_aspect)
    delta_aspect = list_aspect[1] - list_aspect[0]

    # binning
    rbins = np.linspace(np.min(alt_list), np.max(alt_list), nbin_alt)
    abins = np.linspace(list_aspect[0]/360*2*np.pi, list_aspect[-1]/360*2*np.pi, nbin_aspect)
    subdivs = 100
    abins2 = np.linspace((list_aspect[0]-delta_aspect/2)/360*2*np.pi, (list_aspect[-1]+delta_aspect/2)/360*2*np.pi, subdivs*nbin_aspect)
    A, R = np.meshgrid(abins2, rbins)

    ncols = len(data)
    nrows = 1

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 5*nrows), subplot_kw=dict(projection="polar"),
                            gridspec_kw={'hspace': -0.1, 'wspace': 0.3})

    cmap = matplotlib.colors.ListedColormap([colorcycle[1],colorcycle[2],colorcycle[0]]) 

    for i,slo in enumerate(dic_var['slope']):
        axs[i].pcolormesh(A, R, np.repeat(data_to_num[slo], subdivs, axis=1), cmap=cmap, vmin=0, vmax=1)
        axs[i].set_facecolor("silver")
        axs[i].set_theta_zero_location('N')
        axs[i].set_theta_direction(-1)
        axs[i].tick_params(axis='y', labelcolor='black')
        axs[i].set_xticks(abins)
        axs[i].set_yticks([])
        axs[i].xaxis.grid(False)
        axs[i].yaxis.grid(False)
        axs[i].set_rlim(int(np.max(alt_list)+delta_alt/2),int(np.min(alt_list)-delta_alt/2))
        axs[i].set_title(f'{list_slope[i]}°')
        axs[i].bar(0, 1).remove()
        for k in alt_list:
            axs[i].text(np.pi/5, k, (f'{int(k)}m'), horizontalalignment='center', verticalalignment='center')
        # for k in range(int(np.min(list_slope)), int(np.max(list_slope))+1,
        #                 int((np.max(list_slope)-np.min(dic_var['slope']))/(nbin_slope-1))):
        #     axs[i].text(np.pi/5, k, (f'{k}°'), horizontalalignment='center', verticalalignment='center')
    if box and rockfall_values['exact_topo']:
        axs[alt_index].add_patch(matplotlib.patches.Rectangle(((rockfall_values['aspect']-45/2)/360*2*np.pi, rockfall_values['altitude']-5),
                                                        width=np.pi/4, height=10, edgecolor = 'black', fill=False, lw=2))

    #pylint: disable=no-member
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap),
                    shrink=.8, ax=axs[:].ravel().tolist(), orientation='vertical')
    axs[0].figure.axes[-1].yaxis.set_ticks([1/6,1/2,5/6])
    axs[0].figure.axes[-1].set_yticklabels(['Permafrost', 'No permafrost, no glaciers', 'Glaciers']) 
    axs[0].figure.axes[-1].yaxis.label.set_size(15)

    plt.show()
    plt.close()

    return fig

def plot_permafrost_all_slopes_polar_from_inputs(site, path_pickle, path_thaw_depth, box=True):
    """ Function returns a polar plot per altitude of the table of the permafrost state
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    path_thaw_depth : str
        Path to the .nc file where the aggregated thaw depth simulations are stored
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    fig : Figure
        (# altitudes) polar heatmaps of permafrost state
    """
    table_all, rockfall_values, _ = prepare_data_table_GST_per_slope(site, path_pickle, path_thaw_depth)
    fig = plot_permafrost_all_slopes_polar(table_all, rockfall_values, box)

    return fig
