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
from SuPerSim.constants import save_constants

colorcycle, _ = save_constants()

def table_background_evolution_mean_GST_aspect_slope(site, path_pickle):
    """ Function returns a table of mean background and evolution of GST (ground-surface temperature)
        as a function of slope, aspect, and altitude
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    
    Returns
    -------
    list_grd_temp : list
        List of background GST per cell of given altitude, aspect, and slope
    list_mean_grd_temp : list
        Average background GST over all simulations in that cell
    list_diff_temp : list
        List of the evolution of mean GST per cell of given altitude, aspect, and slope
    list_mean_diff_temp : list
        Average the evolution of mean GST over all simulations in that cell
    list_num_sim : list
        Number of valid simulation per cell, returns NaN if different number of simulation per forcing for that cell
    """

    _, _, _, _, _, df_stats, _  = load_all_pickles(site, path_pickle)
    
    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}
    forcings = list(np.unique(df_stats['forcing']))

    list_grd_temp = [[[0 for aspect in dic_var['aspect']] for slope in dic_var['slope']] for altitude in dic_var['altitude']]
    list_mean_grd_temp = [[[0 for aspect in dic_var['aspect']] for slope in dic_var['slope']] for altitude in dic_var['altitude']]
    list_diff_temp = [[[0 for aspect in dic_var['aspect']] for slope in dic_var['slope']] for altitude in dic_var['altitude']]
    list_mean_diff_temp = [[[0 for aspect in dic_var['aspect']] for slope in dic_var['slope']] for altitude in dic_var['altitude']]
    list_num_sim = [[[0 for aspect in dic_var['aspect']] for slope in dic_var['slope']] for altitude in dic_var['altitude']]

    for altitude in range(len(dic_var['altitude'])):
        for slope in range(len(dic_var['slope'])):
            for aspect in range(len(dic_var['aspect'])):
                list_grd_temp[altitude][slope][aspect] = list(df_stats[(df_stats['aspect']==dic_var['aspect'][aspect]) & (df_stats['slope']==dic_var['slope'][slope]) & (df_stats['altitude']==dic_var['altitude'][altitude])]['bkg_grd_temp'])
                list_sim_per_forcing = [list(df_stats[(df_stats['aspect']==dic_var['aspect'][aspect]) & (df_stats['slope']==dic_var['slope'][slope]) & (df_stats['altitude']==dic_var['altitude'][altitude])]['forcing']).count(i) for i in forcings]
                list_num_sim[altitude][slope][aspect] = (np.sum(list_sim_per_forcing) if len(set(list_sim_per_forcing))==1 else np.nan)
                list_mean_grd_temp[altitude][slope][aspect] = round(np.mean((list_grd_temp[altitude][slope][aspect])),3)
                list_diff_temp[altitude][slope][aspect] = list(df_stats[(df_stats['aspect']==dic_var['aspect'][aspect]) & (df_stats['slope']==dic_var['slope'][slope]) & (df_stats['altitude']==dic_var['altitude'][altitude])]['trans_grd_temp'] -
                                                               df_stats[(df_stats['aspect']==dic_var['aspect'][aspect]) & (df_stats['slope']==dic_var['slope'][slope]) & (df_stats['altitude']==dic_var['altitude'][altitude])]['bkg_grd_temp'])
                list_mean_diff_temp[altitude][slope][aspect] = round(np.mean((list_diff_temp[altitude][slope][aspect])),3)

    return list_grd_temp, list_mean_grd_temp, list_diff_temp, list_mean_diff_temp, list_num_sim

def plot_table_mean_GST_aspect_slope(site, path_pickle, altitude, background=True, box=True):
    """ Function returns a plot of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    altitude : int
        Altitude at which we want the table
    background : bool, optional 
        If True, plots the mean background value, else, plots the evolution of the mean
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box


    Returns
    -------
    Table
    """
    #pylint: disable=no-member

    _, _, _, _, _, df_stats, rockfall_values = load_all_pickles(site, path_pickle)
    
    variables = ['aspect', 'slope', 'altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}
    
    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if altitude == j:
            alt_index = i

    list_mean = (table_background_evolution_mean_GST_aspect_slope(site, path_pickle)[1][alt_index] if background else table_background_evolution_mean_GST_aspect_slope(site, path_pickle)[3][alt_index])
    df_temp = pd.DataFrame(list_mean, index=list(dic_var['slope']), columns=list(dic_var['aspect']))

    vals = np.around(df_temp.values, 2)
    vals = vals[~np.isnan(vals)]
    dilute = 1
    min_vals = np.min(vals)
    max_vals = np.max(vals)
    range_vals = max_vals- min_vals
    normal = plt.Normalize((np.max(max_vals - 2*dilute*range_vals,0) if np.min(vals)>0 else -dilute*np.max(np.abs(vals))), dilute*np.max(np.abs(vals)))
    colours = plt.cm.seismic(normal(df_temp))

    for i,l in enumerate(list_mean):
        for j in range(len(list_mean[0])):
            if np.isnan(l[j]):
                colours[i][j] = list(matplotlib.colors.to_rgba('silver'))

    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    the_table=plt.table(cellText=list_mean, rowLabels=df_temp.index, colLabels=df_temp.columns, 
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

    plt.text(-0.05, 5/12,'Slope [°]', fontsize= 16, rotation=90, horizontalalignment='right', verticalalignment='center')
    plt.text(0.5, 1,'Aspect [°]', fontsize= 16, rotation=0, horizontalalignment='center', verticalalignment='bottom')

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_table_aspect_slope_all_altitudes(site, path_pickle, show_glacier=True, box=True):
    """ Function returns 1 plot per altitude of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    show_glacier : bool, opional
        Whether or not to plot the glacier fraction
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    (2 or 3)*(# altitudes) tables
    """

    df, _, _, _, _, df_stats, rockfall_values = load_all_pickles(site, path_pickle)
    
    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}
    # number of simulations for a given cell with fixed (altitude, slope, aspect) triplet
    sim_per_cell = len(df)/np.prod([len(dic_var[i]) for i in dic_var.keys()])

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if box and rockfall_values['exact_topo']:
            if rockfall_values['altitude'] == j:
                alt_index = i

    # setting the parameter values 
    annot = True
    center = [0, 0]
    cmap = ['seismic', 'seismic']

    table_all = table_background_evolution_mean_GST_aspect_slope(site, path_pickle)

    list_mean = [# evolution mean GST
                table_all[3],
                # background mean GST
                table_all[1]]
    labels_plot = ['Mean GST evolution [°C]', 'Mean background GST [°C]']
    if show_glacier:
        # glacier fraction
        list_mean.append([[[int((sim_per_cell-k)/sim_per_cell*100) for k in j] for j in i] for i in table_all[4]])
        labels_plot.append('Glacier fraction')
        cmap.append('BrBG')
        center.append(50)
    data = [[pd.DataFrame(i, index=list(dic_var['slope']), columns=list(dic_var['aspect'])) for i in j] for j in list_mean]

    nrows= len(data)
    ncols = len(data[0])

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3*nrows), constrained_layout=True)
    no_nan = [[l for j in i for k in j.values for l in k if not np.isnan(l)] for i in data]
    vmin = [np.min(i) for i in no_nan]
    vmax = [np.max(i) for i in no_nan]
    if len(vmin) == 3:
        vmin[2] = 0
        vmax[2] = 100

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    # plotting the heatmap 
    for j in range(nrows):
        if ncols == 1:
            sn.heatmap(data=data[nrows-1-j][0], annot=annot, center=center[nrows-1-j], cmap=cmap[nrows-1-j], ax=axs[j], vmin=vmin[nrows-1-j], vmax=vmax[nrows-1-j],
                    cbar=True, yticklabels=True, xticklabels=(j==nrows-1), cbar_kws={'label': labels_plot[nrows-1-j]})
            axs[0].figure.axes[-1].yaxis.label.set_size(13)
            if box and rockfall_values['exact_topo']:
                axs[j].add_patch(Rectangle(((rockfall_values['aspect'])/45*1/8, (70-rockfall_values['slope'])/10*1/5), 1/8, 1/5,
                                        edgecolor = 'black', transform=axs[j].transAxes, fill=False, lw=4))
        if ncols > 1:
            for i in range(ncols):
                sn.heatmap(data=data[nrows-1-j][i], annot=annot, center=center[nrows-1-j], cmap=cmap[nrows-1-j], ax=axs[j,i], vmin=vmin[nrows-1-j], vmax=vmax[nrows-1-j],
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
 
def plot_table_aspect_slope_all_altitudes_polar(site, path_pickle, box=True):
    """ Function returns 3 polar plots (1 per altitude) of the table of either mean background GST (ground-surface temperature)
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
    2*3 polar plots
    """

    _, _, _, _, _, df_stats, rockfall_values = load_all_pickles(site, path_pickle)

    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if box and rockfall_values['exact_topo']:
            if rockfall_values['altitude'] == j:
                alt_index = i

    list_mean = [table_background_evolution_mean_GST_aspect_slope(site, path_pickle)[1], table_background_evolution_mean_GST_aspect_slope(site, path_pickle)[3]]
    data = [[pd.DataFrame(i, index=list(dic_var['slope']), columns=list(dic_var['aspect'])) for i in j] for j in list_mean]
    no_nan = [[l for j in i for k in j.values for l in k if not np.isnan(l)] for i in data]
    vmin = [np.min(i) for i in no_nan]
    vmax = [np.max(i) for i in no_nan]
    
    nbin_x = len(list_mean[0][0])
    nbin_y = len(list_mean[0][0][0])

    # binning
    rbins = np.linspace(30, 70, nbin_x)
    abins = np.linspace(0, (315/360)*2*np.pi, nbin_y)
    subdivs = 100
    abins2 = np.linspace((0-45/2)/360*2*np.pi, (360-45/2)/360*2*np.pi, subdivs*nbin_y)
    A, R = np.meshgrid(abins2, rbins)

    ncols = len(data[0])
    nrows = len(data)

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
        for i in range(ncols):
            axs[j,i].pcolormesh(A, R, np.repeat(list_mean[j][i], subdivs, axis=1), cmap=cmap[j], vmin=vmin[j], vmax=vmax[j])
            axs[j,i].set_facecolor("silver")
            axs[j,i].set_theta_zero_location('N')
            axs[j,i].set_theta_direction(-1)
            axs[j,i].tick_params(axis='y', labelcolor='black')
            axs[j,i].set_xticks(abins)
            axs[j,i].set_yticks([])
            axs[j,i].xaxis.grid(False)
            axs[j,i].yaxis.grid(False)
            axs[j,i].set_rlim(95,25)
            axs[0,i].set_title(f'{alt_list[i]} m')
            axs[j,i].bar(0, 1).remove()
            for k in range(30,80,10):
                axs[j,i].text(np.pi/5, k, (f'{k}°'), horizontalalignment='center', verticalalignment='center')
        if box and rockfall_values['exact_topo']:
            axs[j,alt_index].add_patch(matplotlib.patches.Rectangle(((rockfall_values['aspect']-45/2)/360*2*np.pi, rockfall_values['slope']-5),
                                                        width=np.pi/4, height=10, edgecolor = 'black', fill=False, lw=2))
        
    for j in range(nrows):
        #pylint: disable=no-member
        fig.colorbar(matplotlib.colormaps.ScalarMappable(cmap=cmap[j]), shrink=.8,
                     ax=axs[j,:].ravel().tolist(), orientation='vertical',
                     label=('Mean background GST [°C]' if j==0 else 'Mean GST evolution [°C]'))
    
    for idx,i in enumerate(range(-nrows,0)):
        axs[0,0].figure.axes[i].yaxis.set_ticks(tick_pos[idx])
        axs[0,0].figure.axes[i].set_yticklabels(ticks[idx]) 
    
    for i in [6,7]:
        axs[0,0].figure.axes[i].yaxis.label.set_size(15)

    plt.show()
    plt.close()

def plot_permafrost_all_altitudes_polar(site, path_pickle, depth_thaw, box=True):
    """ Function returns 3 polar plots (1 per altitude) of the permafrost and glacier spatial distribution,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    depth_thaw : netCDF4._netCDF4.Variable
        NetCDF variable encoding the thaw depth
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    2*3 polar plots
    """

    _, _, _, _, _, df_stats, rockfall_values = load_all_pickles(site, path_pickle)

    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if box and rockfall_values['exact_topo']:
            if rockfall_values['altitude'] == j:
                alt_index = i

    list_valid_sim = list(df_stats.index.values)
    list_no_perma = []
    for sim in list_valid_sim:
        if np.std(depth_thaw[sim,:]) < 1 and np.max(depth_thaw[sim,:])> 19:
            list_no_perma.append(sim)

    list_perma = list(set(list_valid_sim) - set(list_no_perma))
    check_perma_list = [[[[i in list_perma for i in list(df_stats[(df_stats['altitude']==altitude) & (df_stats['aspect']==aspect) & (df_stats['slope']==slope)].index.values)] for aspect in dic_var['aspect']]for slope in dic_var['slope']]  for altitude in alt_list]

    # 1/6=there is at least one glacier, 0=all simulations are valid and have permafrost, 5/6=all valid simulations but not all permafrost
    list_data = [[[(5/6 if len(k)<3 else (1/6 if all(k) else 1/2)) for k in j] for j in i] for i in check_perma_list]
    data = [pd.DataFrame(i, index=list(dic_var['slope']), columns=list(dic_var['aspect'])) for i in list_data]

    nbin_x = len(list_data[0])
    nbin_y = len(list_data[0][0])

    # binning
    rbins = np.linspace(30, 70, nbin_x)
    abins = np.linspace(0, (315/360)*2*np.pi, nbin_y)
    subdivs = 100
    abins2 = np.linspace((0-45/2)/360*2*np.pi, (360-45/2)/360*2*np.pi, subdivs*nbin_y)
    A, R = np.meshgrid(abins2, rbins)

    ncols = len(data)
    nrows = 1

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 5*nrows), subplot_kw=dict(projection="polar"),
                            gridspec_kw={'hspace': -0.1, 'wspace': 0.3})

    cmap = matplotlib.colors.ListedColormap([colorcycle[1],colorcycle[2],colorcycle[0]]) 

    for i in range(ncols):
        axs[i].pcolormesh(A, R, np.repeat(list_data[i], subdivs, axis=1), cmap=cmap, vmin=0, vmax=1)
        axs[i].set_facecolor("silver")
        axs[i].set_theta_zero_location('N')
        axs[i].set_theta_direction(-1)
        axs[i].tick_params(axis='y', labelcolor='black')
        axs[i].set_xticks(abins)
        axs[i].set_yticks([])
        axs[i].xaxis.grid(False)
        axs[i].yaxis.grid(False)
        axs[i].set_rlim(95,25)
        axs[i].set_title(f'{alt_list[i]} m')
        axs[i].bar(0, 1).remove()
        for k in range(30,80,10):
            axs[i].text(np.pi/5, k, (f'{k}°'), horizontalalignment='center', verticalalignment='center')
    if box and rockfall_values['exact_topo']:
        axs[alt_index].add_patch(matplotlib.patches.Rectangle(((rockfall_values['aspect']-45/2)/360*2*np.pi, rockfall_values['slope']-5),
                                                        width=np.pi/4, height=10, edgecolor = 'black', fill=False, lw=2))

    #pylint: disable=no-member
    fig.colorbar(matplotlib.colormaps.ScalarMappable(cmap=cmap),
                 shrink=.8, ax=axs[:].ravel().tolist(), orientation='vertical')
    axs[0].figure.axes[-1].yaxis.set_ticks([1/6,1/2,5/6])
    axs[0].figure.axes[-1].set_yticklabels(['Permafrost', 'No permafrost, no glaciers', 'Glaciers']) 
    axs[0].figure.axes[-1].yaxis.label.set_size(15)

    plt.show()
    plt.close()
