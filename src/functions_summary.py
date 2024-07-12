"""This module automates the plotting of summary statistics for timeseries"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from open import open_air_nc, open_ground_nc, open_snow_nc, open_swe_nc, open_thaw_depth_nc, open_SW_direct_nc, open_SW_diffuse_nc, open_SW_up_nc, open_SW_down_nc, open_SW_net_nc, open_LW_net_nc
from mytime import list_tokens_year
from pickling import load_all_pickles
from weights import assign_weight_sim, plot_hist_valid_sim_all_variables, plot_hist_stat_weights
from runningstats import mean_all_altitudes, mean_all_reanalyses, assign_tot_water_prod, plot_aggregating_distance_temp_all
from topoheatmap import plot_table_mean_GST_aspect_slope, plot_table_aspect_slope_all_altitudes, plot_table_aspect_slope_all_altitudes_polar, plot_permafrost_all_altitudes_polar
from model import fit_stat_model_grd_temp
from percentiles import plot_cdf_GST, plot_10_cold_warm, heatmap_percentile_GST
from yearlystats import plot_box_yearly_stat, plot_yearly_quantiles_air, plot_yearly_quantiles_all_sims, plot_yearly_quantiles_all_sims_side_by_side
from seasonal import plot_sanity_one_year_quantiles_two_periods, plot_sanity_two_variables_one_year_quantiles, plot_sanity_two_variables_one_year_quantiles_side_by_side
from evolution import plot_GST_bkg_vs_evol_quantile_bins_fit_single_site, plot_GST_bkg_vs_evol_quantile_bins_fit, plot_mean_bkg_GST_vs_evolution, plot_evolution_snow_cover_melt_out
from constants import save_constants

colorcycle, _ = save_constants()

def plot_visible_skymap_from_horizon_file(hor_path):
    """ Function returns a fisheye view of the sky with the visible portion in blue and the blocked one in black.
    
    Parameters
    ----------
    hor_path : str
        Path to the .csv horizon file

    Returns
    -------
    Plot of the sky view from the location
    """  

    hor_file = pd.read_csv(hor_path, usecols=['azi', 'hang'])

    theta_pre = hor_file['azi']
    theta = [i/360*2*np.pi for i in theta_pre]
    r_pre = hor_file['hang']
    r = [90-i for i in r_pre]
    
    # Creating the polar scatter plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    ax.set_ylim(0, 90)
    for i,j in enumerate(r):
        ax.fill_between( np.linspace(theta[i], (0 if i==len(r)-1 else theta[i+1]), 2), 0, j, color='deepskyblue', alpha=0.5)
        ax.fill_between( np.linspace(theta[i], (0 if i==len(r)-1 else theta[i+1]), 2), j, 90, color='black', alpha=1)
    ax.scatter(theta, r, c='blue', s=10, cmap='hsv', alpha=0.75)
    # plt.title('Scatter Plot on Polar Axis', fontsize=15)
    plt.show()

def plot_all(site, forcing_list,
             path_forcing_list, path_ground, path_snow, path_swe, path_thaw_depth, path_pickle,
             year_bkg_end, year_trans_end, no_weight=True,
             individual_heatmap=False, polar_plots=False, parity_plot=False):
    """ Function returns a series of summary plots for a given site.
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    forcing_list : list of str
        List of forcings provided, with a number of entries between 1 and 3 in 'era5', 'merra2', and 'jra55'. E.g. ['era5', 'merra2']
    path_forcing_list : list of str
        List of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
    path_ground : str
        String path to the location of the ground output file from GTPEM (.nc)
    path_snow : str
        Path to the .nc file where the aggregated snow simulations are stored
    path_swe : str
        Path to the .nc file where the aggregated SWE simulations are stored
    path_thaw_depth : str
        String path to the location of the thaw depth output file from GTPEM (.nc)
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    no_weight : bool, optional
        If True, all simulations have the same weight, otherwise the weight is computed as a function of altitude, aspect, and slope
    individual_heatmap : bool, optional
        Show or not heatmaps for unique altitude
    polar_plots : bool, optional
        Show or not polar plots
    parity_plot : bool, optional
        Show or not parity plot

    Returns
    -------
    Plots
    """  

    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    df, _, list_valid_sim, _, _, df_stats, rockfall_values = load_all_pickles(site, path_pickle)

    #####################################################################################
    # PLOTS
    #####################################################################################

    # assign a subjective weight to all simulations
    pd_weight, _ = assign_weight_sim(site, path_pickle, no_weight)
    _, thaw_depth = open_thaw_depth_nc(path_thaw_depth)

    list_vars = ['time_air', 'temp_air', 'SW_flux', 'SW_direct_flux', 'SW_diffuse_flux', 'precipitation']
    list_series = [open_air_nc(i) for i in path_forcing_list]
    list_series_b = [[list_series[j][i] for j in range(len(list_series))] for i in range(len(list_series[0]))]
    air_all_dict = dict(zip(list_vars, list_series_b))

    time_air_all = air_all_dict['time_air']
    temp_air_all = air_all_dict['temp_air']
    precipitation_all = air_all_dict['precipitation']

    _, time_ground, temp_ground = open_ground_nc(path_ground)
    _, snow_height = open_snow_nc(path_snow)
    _, swe = open_swe_nc(path_swe)

    _, time_bkg_ground, time_trans_ground, _ = list_tokens_year(time_ground, year_bkg_end, year_trans_end)

    # weighted mean GST
    temp_ground_mean = list(np.average([temp_ground[i,:,0] for i in list(pd_weight.index.values)], axis=0, weights=pd_weight.loc[:, 'weight']))
    print('The following plot is a histogram of the distribution of the statistical weights of all simulations:')
    plot_hist_stat_weights(pd_weight, df, zero=True)
    print('The following plot is a histogram of the distribution of glacier simulations wrt to altitude, aspect, slope, and forcing:')
    plot_hist_valid_sim_all_variables(site, path_thaw_depth, path_pickle)

    # Mean air temperature over all reanalyses and altitudes
    mean_air_temp = mean_all_reanalyses(time_air_all,
                                        [mean_all_altitudes(i, site, path_pickle, no_weight=True) for i in temp_air_all],
                                        year_bkg_end, year_trans_end)
    # mean_air_temp = mean_all_reanalyses(time_air_all, [i[:,alt_index] for i in temp_air_all], year_bkg_end, year_trans_end)

    # finally we get the total water production, averaged over all reanalyses
    tot_water_prod, _, mean_prec = assign_tot_water_prod(path_forcing_list, path_ground, path_swe, path_pickle, year_bkg_end, year_trans_end, site, no_weight)

    print('Plots of the normalized distance of air and ground temperature, water production, and thaw_depth as a function of time')
    if 'year' in rockfall_values.keys():
        year_rockfall = rockfall_values['year']
        print('Granularity: week and month side by side')
        plot_aggregating_distance_temp_all(['Air temperature', 'Water production', 'Ground temperature'],
                                        [time_air_all[0], time_ground, time_ground],
                                        [mean_air_temp, tot_water_prod, temp_ground_mean],
                                        ['week', 'month'], site, path_pickle, year_bkg_end, year_trans_end, year_rockfall, False)
    print('Granularity: year, plotted for all years')
    plot_aggregating_distance_temp_all(['Air temperature', 'Water production', 'Ground temperature'],
                                        [time_air_all[0], time_ground, time_ground],
                                        [mean_air_temp, tot_water_prod, temp_ground_mean],
                                        ['year'], site, path_pickle, year_bkg_end, year_trans_end, 0, False)

    print('Yearly statistics for air and ground surface temperature, and also precipitation and water production')
    plot_box_yearly_stat('Air temperature', time_air_all[0], mean_air_temp, year_bkg_end, year_trans_end)
    plot_box_yearly_stat('GST', time_ground, temp_ground_mean, year_bkg_end, year_trans_end)
    plot_box_yearly_stat('Precipitation', time_ground, mean_prec, year_bkg_end, year_trans_end)
    plot_box_yearly_stat('Water production', time_ground, tot_water_prod, year_bkg_end, year_trans_end)

    alt_list = sorted(set(df_stats['altitude']))
    alt_index = int(np.floor((len(alt_list)-1)/2))
    alt_index_abs = alt_list[alt_index]

    if individual_heatmap:
        alt_show = rockfall_values['altitude'] if ((rockfall_values['exact_topo']) and (rockfall_values['altitude'] in alt_list)) else alt_index_abs
        print(f'Heatmap of the background mean GST as a function of aspect and slope at {alt_show} m:')
        plot_table_mean_GST_aspect_slope(site, path_pickle, alt_show, True, False)
        print(f'Heatmap of the evolution of the mean GST between the background and the transient periods as a function of aspect and slope at {alt_index_abs} m:')
        plot_table_mean_GST_aspect_slope(site, path_pickle, alt_show, False, False)

    print('Heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitudes')
    plot_table_aspect_slope_all_altitudes(site, path_pickle, show_glacier=False, box=False)


    if polar_plots:
        print('Polar heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitude')
        plot_table_aspect_slope_all_altitudes_polar(site, path_pickle, box=False)

        print('Polar plot of the permafrost and glacier spatial distribution as a function of aspect and slope at all altitude')
        plot_permafrost_all_altitudes_polar(site, path_pickle, thaw_depth, box=False)

    print('CDF of background, transient, and evolution GST:')
    plot_cdf_GST(site, path_pickle)
    print('Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference:')
    heatmap_percentile_GST(site, path_pickle)
    print('Plot of mean GST evolution vs background GST, with an emphasis on the 10% colder and warmer simulations')
    plot_10_cold_warm(site, path_pickle)
    print('Plot of mean GST evolution vs background GST, fit, and binning per 10% quantiles')
    plot_GST_bkg_vs_evol_quantile_bins_fit_single_site(site, path_pickle)

    print('Scatter plot of mean background GST vs evolution of mean GST between the background and transient period')
    plot_mean_bkg_GST_vs_evolution(site, path_pickle)

    if parity_plot:
        print('Parity plot (statistically-modeled vs numerically-simulated) of background mean GST:')
        _, _, optimizedParameters, _, _, _ = fit_stat_model_grd_temp(site, path_pickle, all_data=False, diff_forcings=True)
        list_ceof = ['offset', 'c_alt', 'd_alt', 'c_asp', 'c_slope']
        pd_coef = pd.DataFrame(list_ceof, columns=['Coefficient'])
        # previously was columns=['all', 'era5', 'merra2', 'jra55'] when had all 3 forcings
        pd_coef = pd.concat([pd_coef, pd.DataFrame((np.array([list(i) for i in optimizedParameters]).transpose()), columns=forcing_list)], axis=1)
        print('The coefficients of the statistical model for the mean background GST are given by:')
        print(pd_coef)

    print('Plot of yearly statistics for atmospheric timeseries. Mean and several quantiles for each year:')
    plot_yearly_quantiles_air(time_air_all, temp_air_all, 'Air temperature', year_bkg_end, year_trans_end)
    plot_yearly_quantiles_air(time_air_all, precipitation_all, 'Precipitation', year_bkg_end, year_trans_end)

    print('Plot of yearly statistics for simulated timeseries. Mean and several quantiles for each year:')
    plot_yearly_quantiles_all_sims(time_ground, temp_ground, list_valid_sim, 'GST', year_bkg_end, year_trans_end)
    plot_yearly_quantiles_all_sims(time_ground, snow_height, list_valid_sim, 'Snow depth', year_bkg_end, year_trans_end)
    plot_yearly_quantiles_all_sims(time_ground, swe, list_valid_sim, 'SWE', year_bkg_end, year_trans_end)

    print('Histogram of the evolution of the snow cover (in days) and melt-out date:')
    plot_evolution_snow_cover_melt_out(site, path_pickle)

    print('Plot of 2 timeseries reduced to a 1-year window with mean and 1- and 2-sigma spread:')
    plot_sanity_two_variables_one_year_quantiles(time_ground, [temp_ground, snow_height], [list_valid_sim, list_valid_sim], ['GST', 'Snow depth'])

    print('Plot of a single timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread, for background and transient periods:')
    plot_sanity_one_year_quantiles_two_periods(time_ground, [temp_ground, temp_ground], [list_valid_sim, list_valid_sim], 'GST', ['Background', 'Transient'], [time_bkg_ground, time_trans_ground])
    plot_sanity_one_year_quantiles_two_periods(time_ground, [snow_height, snow_height], [list_valid_sim, list_valid_sim], 'Snow depth', ['Background', 'Transient'], [time_bkg_ground, time_trans_ground])

    # print('')
    # plot_sanity_one_year_quantiles_two_periods(time_air_merra2, [temp_air_merra2, temp_air_merra2], [None, None], 'Air temperature', ['Background', 'Transient'], [time_bkg_air_merra2, time_trans_air_merra2])
    
    print('All done!')

def plot_camparison_two_sites(list_site, list_label_site,
             list_path_forcing_list, list_path_ground, list_path_snow, list_path_swe,
             list_path_SW_direct, list_path_SW_diffuse, list_path_SW_up,
             list_path_SW_down, list_path_SW_net, list_path_LW_net,
             list_path_pickle, year_bkg_end, year_trans_end):
    """ Function returns a series of comparison of summary plots for two sites.
    
    Parameters
    ----------
    list_site : str
        List of location of the event, e.g. ['Joffre', 'Fingerpost'] or ['North', 'South']
    list_label_site : list of str
        List of label for each site
    list_path_forcing_list : list of str
        List of list of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
        E.g. [['path_reanalysis_1_North', .., 'path_reanalysis_n_North'], ['path_reanalysis_1_South', .., 'path_reanalysis_n_South']]
    list_path_ground : list of str
        List of string path to the location of the ground output file from GTPEM (.nc)
    list_path_snow : str
        List of path to the .nc file where the aggregated snow simulations are stored
    list_path_swe : str
        List of path to the .nc file where the aggregated SWE simulations are stored
    list_path_SW_direct : str
        List of path to the .nc file where the aggregated SW direct simulations are stored
    list_path_SW_diffuse : str
        List of path to the .nc file where the aggregated SW diffuse simulations are stored
    list_path_SW_up : str
        List of path to the .nc file where the aggregated SW up simulations are stored
    list_path_SW_down : str
        List of path to the .nc file where the aggregated SW down simulations are stored
    list_path_SW_net : str
        List of path to the .nc file where the aggregated SW net simulations are stored
    list_path_LW_net : str
        List of path to the .nc file where the aggregated LW net simulations are stored
    list_path_pickle : str
        List of path to the location of the folder where the pickles are saved
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period

    Returns
    -------
    Plots
    """ 

    [df, list_valid_sim, df_stats, rockfall_values,
    time_air_all, temp_air_all, precipitation_all,
    time_ground, temp_ground, snow_height, swe,
    SW_direct, SW_diffuse, SW_up, SW_down, SW_net, LW_net] = [[[] for _ in range(2)] for _ in range(17)]

    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    for i in range(2):
        df[i], _, list_valid_sim[i], _, _, df_stats[i], rockfall_values[i] = load_all_pickles(list_site[i], list_path_pickle[i])

        list_vars = ['time_air', 'temp_air', 'precipitation']
        list_series = [open_air_nc(j) for j in list_path_forcing_list[i]]
        list_series_b = [[list_series[j][k] for j in range(len(list_series))] for k in range(len(list_series[0]))]
        air_all_dict = dict(zip(list_vars, list_series_b))

        time_air_all[i] = air_all_dict['time_air']
        temp_air_all[i] = air_all_dict['temp_air']
        precipitation_all[i] = air_all_dict['precipitation']

        _, time_ground[i], temp_ground[i] = open_ground_nc(list_path_ground[i])
        _, snow_height[i] = open_snow_nc(list_path_snow[i])
        _, swe[i] = open_swe_nc(list_path_swe[i])
        _, SW_direct[i] = open_SW_direct_nc(list_path_SW_direct[i])
        _, SW_diffuse[i] = open_SW_diffuse_nc(list_path_SW_diffuse[i])
        _, SW_up[i] = open_SW_up_nc(list_path_SW_up[i])
        _, SW_down[i] = open_SW_down_nc(list_path_SW_down[i])
        _, SW_net[i] = open_SW_net_nc(list_path_SW_net[i])
        _, LW_net[i] = open_LW_net_nc(list_path_LW_net[i])

    print('Series of plots comparing both sites.')
    print('Plot of a single timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread, for both sites:')
    plot_sanity_two_variables_one_year_quantiles(time_ground[0], [temp_ground[0], temp_ground[1]], [list_valid_sim[0], list_valid_sim[1]], ['GST'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles(time_ground[0], [SW_direct[0], SW_direct[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW direct'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles(time_ground[0], [SW_diffuse[0], SW_diffuse[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW diffuse'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles(time_ground[0], [SW_down[0], SW_down[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW up'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles(time_ground[0], [SW_up[0], SW_up[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW down'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles(time_ground[0], [SW_net[0], SW_net[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW net'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles(time_ground[0], [LW_net[0], LW_net[1]], [list_valid_sim[0], list_valid_sim[1]], ['LW net'], list_label_site)

    print('Plot of seasonal statistics for GST and SW net for both sites side by side:')
    plot_sanity_two_variables_one_year_quantiles_side_by_side(time_ground[0], [[temp_ground[0], temp_ground[1]], [SW_net[0], SW_net[1]]], [list_valid_sim[0], list_valid_sim[1]], ['GST', 'SW net'], list_label_site)
    
    print('Plot of yearly, background, and transient statistics for GST for both sites side by side:')
    plot_yearly_quantiles_all_sims_side_by_side(time_ground[0], [temp_ground[0], temp_ground[1]], [list_valid_sim[0], list_valid_sim[1]], 'GST', list_label_site, year_bkg_end, year_trans_end)
    
    print('Plot of mean GST evolution vs background GST, fit, and binning per 10% quantiles for both sites:')
    plot_GST_bkg_vs_evol_quantile_bins_fit(list_site, list_path_pickle, list_label_site)

    print('All done!')
