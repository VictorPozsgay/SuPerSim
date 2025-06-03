"""This module automates the plotting of summary statistics for timeseries"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import numpy as np

from SuPerSim.open import open_air_nc, open_ground_nc, open_snow_nc, open_swe_nc, open_SW_direct_nc, open_SW_diffuse_nc, open_SW_up_nc, open_SW_down_nc, open_SW_net_nc, open_LW_net_nc
from SuPerSim.mytime import list_tokens_year
from SuPerSim.pickling import load_all_pickles
from SuPerSim.weights import assign_weight_sim, plot_hist_stat_weights_from_input, plot_hist_valid_sim_all_variables_from_input
from SuPerSim.runningstats import mean_all_altitudes, mean_all_reanalyses, assign_tot_water_prod, plot_aggregating_distance_temp_all_from_input
from SuPerSim.topoheatmap import plot_table_mean_GST_aspect_slope_single_altitude_from_inputs, plot_table_mean_GST_aspect_slope_all_altitudes_from_inputs, plot_table_mean_GST_aspect_slope_all_altitudes_polar_from_inputs, plot_permafrost_all_altitudes_polar_from_inputs, plot_table_mean_GST_aspect_altitude_all_slopes_polar_from_inputs, plot_permafrost_all_slopes_polar_from_inputs
from SuPerSim.model import fit_stat_model_GST_from_inputs
from SuPerSim.percentiles import plot_cdf_GST_from_inputs, plot_heatmap_percentile_GST_from_inputs
from SuPerSim.yearlystats import plot_box_yearly_stat_from_inputs, plot_yearly_quantiles_atmospheric_from_inputs, plot_yearly_quantiles_sim_from_inputs, plot_yearly_quantiles_side_by_side_sim_from_inputs
from SuPerSim.seasonal import plot_sanity_one_year_quantiles_two_periods_from_inputs, plot_sanity_two_variables_one_year_quantiles_from_inputs, plot_sanity_two_variables_two_sites_one_year_quantiles_side_by_side_from_inputs
from SuPerSim.evolution import plot_evolution_snow_cover_melt_out_from_inputs, plot_GST_bkg_vs_evol_quantile_bins_fit_single_site_from_inputs, plot_mean_bkg_GST_vs_evolution_from_inputs, plot_GST_bkg_vs_evol_quantile_bins_fit_two_sites_from_input
from SuPerSim.horizon import plot_visible_skymap_from_horizon_file

def plot_all(site,
             path_forcing_list, path_ground, path_snow, path_swe, path_thaw_depth, path_pickle,
             year_bkg_end, year_trans_end, path_horizon=None, no_weight=True, show_glaciers=True,
             individual_heatmap=False, polar_plots=False, parity_plot=False,
             show_landslide_time=True):
    """ Function returns a series of summary plots for a given site.
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
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
    path_horizon : str, optional
        If a path to a .csv horizon file is given, a horizon plot is produced, if None, nothing happens.
    no_weight : bool, optional
        If True, all simulations have the same weight, otherwise the weight is computed as a function of altitude, aspect, and slope
    show_glaciers : bool, optional
        If True, shows the glacier simulations with a 0 weight, if False, those are ignored.
    individual_heatmap : bool, optional
        Show or not heatmaps for unique altitude
    polar_plots : bool, optional
        Show or not polar plots
    parity_plot : bool, optional
        Show or not parity plot
    show_landslide_time : bool
        Choose to show or not the vertical dashed line indicating the time of the landslide. For a slow landslide, choose False.

    Returns
    -------
    dic_figs: dict
        Dictionary {k: v} where the keys 'k' are the figure names and the values 'v' the figures
    """  

    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    pkl = load_all_pickles(site, path_pickle)
    # df = pkl['df']
    list_valid_sim = pkl['list_valid_sim']
    df_stats = pkl['df_stats']
    rockfall_values = pkl['rockfall_values']

    #####################################################################################
    # PLOTS
    #####################################################################################

    # assign a subjective weight to all simulations
    pd_weight, _ = assign_weight_sim(site, path_pickle, no_weight)

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
    
    # Mean air temperature over all reanalyses and altitudes
    mean_air_temp = mean_all_reanalyses(time_air_all,
                                        [mean_all_altitudes(i, site, path_pickle, no_weight) for i in temp_air_all],
                                        year_bkg_end, year_trans_end)
    # mean_air_temp = mean_all_reanalyses(time_air_all, [i[:,alt_index] for i in temp_air_all], year_bkg_end, year_trans_end)

    # finally we get the total water production, averaged over all reanalyses
    tot_water_prod, _, mean_prec = assign_tot_water_prod(path_forcing_list, path_ground, path_swe, path_pickle, year_bkg_end, year_trans_end, site, no_weight)

    alt_list = sorted(set(df_stats['altitude']))
    alt_index = int(np.floor((len(alt_list)-1)/2))
    alt_index_abs = alt_list[alt_index]

    list_figs = []
    list_fig_names = []

    if path_horizon is not None:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Fisheye view of the sky with the visible portion in blue and the blocked one in black:')
        list_fig_names.append('fisheye_horizon')
        list_figs.append(plot_visible_skymap_from_horizon_file(path_horizon))

    print('\n---------------------------------------------------------------------------------------------\n')

    print('The following plot is a histogram of the distribution of the statistical weights of all simulations:')
    list_fig_names.append('hist_stat_weight')
    list_figs.append(plot_hist_stat_weights_from_input(site, path_pickle, no_weight, show_glaciers))

    print('\n---------------------------------------------------------------------------------------------\n')

    print('The following plot is a histogram of the distribution of glacier simulations wrt to altitude, aspect, slope, and forcing:')
    list_fig_names.append('hist_distrib_glaciers_perma')
    list_figs.append(plot_hist_valid_sim_all_variables_from_input(site, path_thaw_depth, path_pickle))



    print('\n---------------------------------------------------------------------------------------------')
    print('------------------------------------- TEMPORAL ANALYSIS -------------------------------------')
    print('---------------------------------------------------------------------------------------------\n')

    print('\n---------------------------------------------------------------------------------------------\n')
    print('Yearly statistics for air and ground surface temperature, and also precipitation and water production')
    list_fig_names.append('AirTemp_yearly_stats_box')
    list_figs.append(plot_box_yearly_stat_from_inputs('Air temperature', time_air_all[0], mean_air_temp, year_bkg_end, year_trans_end))
    list_fig_names.append('GST_yearly_stats_box')
    list_figs.append(plot_box_yearly_stat_from_inputs('GST', time_ground, temp_ground_mean, year_bkg_end, year_trans_end))
    list_fig_names.append('Precip_yearly_stats_box')
    list_figs.append(plot_box_yearly_stat_from_inputs('Precipitation', time_air_all[0], mean_prec, year_bkg_end, year_trans_end))
    list_fig_names.append('WaterProd_yearly_stats_box')
    list_figs.append(plot_box_yearly_stat_from_inputs('Water production', time_ground, tot_water_prod, year_bkg_end, year_trans_end))

    print('\n---------------------------------------------------------------------------------------------\n')
    print('Plot of yearly statistics for atmospheric timeseries. Mean and several quantiles for each year:')
    list_fig_names.append('AirTemp_yearly_quantiles')
    list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, temp_air_all, 'Air temperature', year_bkg_end, year_trans_end))
    list_fig_names.append('AirTemp_yearly_mean')
    list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, temp_air_all, 'Air temperature', year_bkg_end, year_trans_end, False))
    list_fig_names.append('Precip_yearly_quantiles')
    list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, precipitation_all, 'Precipitation', year_bkg_end, year_trans_end))
    list_fig_names.append('Precip_yearly_mean')
    list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, precipitation_all, 'Precipitation', year_bkg_end, year_trans_end, False))

    print('\n---------------------------------------------------------------------------------------------\n')
    print('Plot of yearly statistics for simulated timeseries. Mean and several quantiles for each year:')
    list_fig_names.append('GST_yearly_quantiles')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, temp_ground, list_valid_sim, 'GST', year_bkg_end, year_trans_end))
    list_fig_names.append('GST_yearly_mean')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, temp_ground, list_valid_sim, 'GST', year_bkg_end, year_trans_end, False))
    list_fig_names.append('Snow_yearly_quantiles')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, snow_height, list_valid_sim, 'Snow depth', year_bkg_end, year_trans_end))
    list_fig_names.append('Snow_yearly_mean')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, snow_height, list_valid_sim, 'Snow depth', year_bkg_end, year_trans_end, False))
    list_fig_names.append('SWE_yearly_quantiles')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, swe, list_valid_sim, 'SWE', year_bkg_end, year_trans_end))
    list_fig_names.append('SWE_yearly_mean')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, swe, list_valid_sim, 'SWE', year_bkg_end, year_trans_end, False))

    print('\n---------------------------------------------------------------------------------------------\n')
    print('Plot of 2 timeseries reduced to a 1-year window with mean and 1- and 2-sigma spread:')
    list_fig_names.append('GST_v_snow')
    list_figs.append(plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground, [temp_ground, snow_height], [list_valid_sim, list_valid_sim], ['GST', 'Snow depth']))

    print('\n---------------------------------------------------------------------------------------------\n')
    print('Plot of a single timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread, for background and transient periods:')
    list_fig_names.append('GST_1year_bkg_v_transient')
    list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground, temp_ground], [list_valid_sim, list_valid_sim], 'GST', ['Background', 'Transient'], [time_bkg_ground, time_trans_ground]))
    list_fig_names.append('snow_1year_bkg_v_transient')
    list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [snow_height, snow_height], [list_valid_sim, list_valid_sim], 'Snow depth', ['Background', 'Transient'], [time_bkg_ground, time_trans_ground]))

    # # print('')
    # This works well but it would be better to smooth the data
    # plot_sanity_one_year_quantiles_two_periods_from_inputs(time_air_all[0], [temp_air_all[0], temp_air_all[0]], [None, None], 'Air temperature', ['Background', 'Transient'], [time_bkg_air, time_trans_air])

    print('Plots of the normalized distance of air and ground temperature, water production, and thaw_depth as a function of time')
    if 'year' in rockfall_values.keys():
        year_rockfall = rockfall_values['year']
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Granularity: week and month side by side')
        list_fig_names.append('normdev_week_month')
        list_figs.append(plot_aggregating_distance_temp_all_from_input(['Water production', 'Air temperature', 'Ground temperature'],
                                        [time_ground, time_air_all[0], time_ground],
                                        [tot_water_prod, mean_air_temp, temp_ground_mean],
                                        ['week', 'month'], site, path_pickle, year_bkg_end, year_trans_end, year_rockfall, False,
                                        show_landslide_time))
    
    print('\n---------------------------------------------------------------------------------------------\n')
    print('Granularity: year, plotted for all years')
    list_fig_names.append('normdev')
    list_figs.append(plot_aggregating_distance_temp_all_from_input(['Water production', 'Air temperature', 'Ground temperature'],
                                        [time_ground, time_air_all[0], time_ground],
                                        [tot_water_prod, mean_air_temp, temp_ground_mean],
                                        ['year'], site, path_pickle, year_bkg_end, year_trans_end, 0, False,
                                        show_landslide_time))



    print('\n---------------------------------------------------------------------------------------------')
    print('------------------------------------- SPATIAL ANALYSIS --------------------------------------')
    print('---------------------------------------------------------------------------------------------\n')




    if individual_heatmap:
        alt_show = rockfall_values['altitude'] if ((rockfall_values['exact_topo']) and (rockfall_values['altitude'] in alt_list)) else alt_index_abs
        print('\n---------------------------------------------------------------------------------------------\n')
        print(f'Heatmap of the background mean GST as a function of aspect and slope at {alt_show} m:')
        list_fig_names.append('heatmap_centre_GST_bkg')
        list_figs.append(plot_table_mean_GST_aspect_slope_single_altitude_from_inputs(site, path_pickle, alt_show, background=True, box=True))
        print('\n---------------------------------------------------------------------------------------------\n')
        print(f'Heatmap of the evolution of the mean GST between the background and the transient periods as a function of aspect and slope at {alt_show} m:')
        list_fig_names.append('heatmap_centre_GST_evol')
        list_figs.append(plot_table_mean_GST_aspect_slope_single_altitude_from_inputs(site, path_pickle, alt_show, background=False, box=True))

    print('\n---------------------------------------------------------------------------------------------\n')
    print('Heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitudes')
    list_fig_names.append('heatmap_GST')
    list_figs.append(plot_table_mean_GST_aspect_slope_all_altitudes_from_inputs(site, path_pickle, show_glaciers, box=True))


    if polar_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Polar heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitude')
        list_fig_names.append('heatmap_GST_polar_alts')
        list_figs.append(plot_table_mean_GST_aspect_slope_all_altitudes_polar_from_inputs(site, path_pickle, box=True))
        list_fig_names.append('heatmap_GST_polar')
        list_figs.append(plot_table_mean_GST_aspect_altitude_all_slopes_polar_from_inputs(site, path_pickle, box=True))

        print('\n---------------------------------------------------------------------------------------------\n')
        print('Polar plot of the permafrost and glacier spatial distribution as a function of aspect and slope at all altitude')
        list_fig_names.append('heatmap_perma_polar_alts')
        list_figs.append(plot_permafrost_all_altitudes_polar_from_inputs(site, path_pickle, path_thaw_depth, box=True))
        list_fig_names.append('heatmap_perma_polar')
        list_figs.append(plot_permafrost_all_slopes_polar_from_inputs(site, path_pickle, path_thaw_depth, box=True))



    print('\n---------------------------------------------------------------------------------------------')
    print('-------------------------------------- FURTHER  PLOTS ---------------------------------------')
    print('---------------------------------------------------------------------------------------------\n')



    print('\n---------------------------------------------------------------------------------------------\n')
    print('CDF of background, transient, and evolution GST:')
    list_fig_names.append('CDF_GST_SO')
    list_figs.append(plot_cdf_GST_from_inputs(site, path_pickle))

    print('\n---------------------------------------------------------------------------------------------\n')
    print('Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference:')
    list_fig_names.append('heatmap_percentiles_GST_bkg_trans_evol')
    list_figs.append(plot_heatmap_percentile_GST_from_inputs(site, path_pickle))

    print('\n---------------------------------------------------------------------------------------------\n')
    print('Histogram of the evolution of the snow cover (in days) and melt-out date:')
    list_fig_names.append('hist_snow_cover')
    list_figs.append(plot_evolution_snow_cover_melt_out_from_inputs(site, path_pickle))

    print('\n---------------------------------------------------------------------------------------------\n')
    print('Plot of mean GST evolution vs background GST, fit, and binning per 10% quantiles')
    list_fig_names.append('GST_evol_v_bkg')
    list_figs.append(plot_GST_bkg_vs_evol_quantile_bins_fit_single_site_from_inputs(site, path_pickle))

    print('\n---------------------------------------------------------------------------------------------\n')
    print('Scatter plot of mean background GST vs evolution of mean GST between the background and transient period')
    list_fig_names.append('GST_evol_v_bkg_alts')
    list_figs.append(plot_mean_bkg_GST_vs_evolution_from_inputs(site, path_pickle))

    if parity_plot:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Parity plot (statistically-modeled vs numerically-simulated) of background mean GST:')
        list_fig_names.append('parity_plot_stat_model_bkg_mean_GST')
        list_figs.append(fit_stat_model_GST_from_inputs(site, path_pickle, all_data=False, diff_forcings=True))

    
    print('\n---------------------------------------------------------------------------------------------')
    print('---------------------------------- SUCCESSFULLY COMPLETED -----------------------------------')
    print('---------------------------------------------------------------------------------------------\n')

    dic_figs = dict(zip(list_fig_names, list_figs))
    return dic_figs



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
        pkl = load_all_pickles(list_site[i], list_path_pickle[i])
        df[i] = pkl['df']
        list_valid_sim[i] = pkl['list_valid_sim']
        df_stats[i] = pkl['df_stats']
        rockfall_values[i] = pkl['rockfall_values']

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

    print('\n---------------------------------------------------------------------------------------------\n')

    print('Series of plots comparing both sites.')

    print('\n---------------------------------------------------------------------------------------------\n')

    print('Plot of a single timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread, for both sites:')
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [temp_ground[0], temp_ground[1]], [list_valid_sim[0], list_valid_sim[1]], ['GST'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [SW_direct[0], SW_direct[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW direct'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [SW_diffuse[0], SW_diffuse[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW diffuse'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [SW_down[0], SW_down[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW up'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [SW_up[0], SW_up[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW down'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [SW_net[0], SW_net[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW net'], list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [LW_net[0], LW_net[1]], [list_valid_sim[0], list_valid_sim[1]], ['LW net'], list_label_site)

    print('\n---------------------------------------------------------------------------------------------\n')

    print('Plot of seasonal statistics for GST and SW net for both sites side by side:')
    plot_sanity_two_variables_two_sites_one_year_quantiles_side_by_side_from_inputs(time_ground[0], [[temp_ground[0], temp_ground[1]], [SW_net[0], SW_net[1]]], [list_valid_sim[0], list_valid_sim[1]], ['GST', 'SW net'], list_label_site)
    
    print('\n---------------------------------------------------------------------------------------------\n')

    print('Plot of yearly, background, and transient statistics for GST for both sites side by side:')
    plot_yearly_quantiles_side_by_side_sim_from_inputs(time_ground[0], [temp_ground[0], temp_ground[1]], [list_valid_sim[0], list_valid_sim[1]], 'GST', list_label_site, year_bkg_end, year_trans_end)
    plot_yearly_quantiles_side_by_side_sim_from_inputs(time_ground[0], [temp_ground[0], temp_ground[1]], [list_valid_sim[0], list_valid_sim[1]], 'GST', list_label_site, year_bkg_end, year_trans_end, False)
    
    print('\n---------------------------------------------------------------------------------------------\n')

    print('Plot of mean GST evolution vs background GST, fit, and binning per 10% quantiles for both sites:')
    plot_GST_bkg_vs_evol_quantile_bins_fit_two_sites_from_input(list_site, list_path_pickle, list_label_site)

    
    print('\n---------------------------------------------------------------------------------------------')
    print('---------------------------------- SUCCESSFULLY COMPLETED -----------------------------------')
    print('---------------------------------------------------------------------------------------------\n')
