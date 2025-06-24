"""This module sub-divides the plots into smaller chunks"""

from SuPerSim.horizon import plot_visible_skymap_from_horizon_file
from SuPerSim.weights import plot_hist_stat_weights_from_input, plot_hist_valid_sim_all_variables_from_input
from SuPerSim.yearlystats import plot_box_yearly_stat_from_inputs, plot_yearly_quantiles_atmospheric_from_inputs, plot_yearly_quantiles_sim_from_inputs, plot_yearly_max_thaw_depth_from_inputs, sim_data_to_panda, panda_data_to_yearly_stats
from SuPerSim.seasonal import plot_sanity_two_variables_one_year_quantiles_from_inputs, plot_sanity_one_year_quantiles_two_periods_from_inputs
from SuPerSim.mytime import list_tokens_year
from SuPerSim.runningstats import plot_aggregating_distance_temp_all_from_input
from SuPerSim.topoheatmap import plot_table_mean_GST_aspect_slope_single_altitude_from_inputs, plot_table_mean_GST_aspect_slope_all_altitudes_from_inputs, plot_table_mean_GST_aspect_altitude_all_slopes_polar_from_inputs, plot_permafrost_all_altitudes_polar_from_inputs, plot_permafrost_all_slopes_polar_from_inputs, plot_table_mean_GST_aspect_slope_all_altitudes_polar_from_inputs
from SuPerSim.percentiles import plot_cdf_GST_from_inputs, plot_heatmap_percentile_GST_from_inputs
from SuPerSim.evolution import plot_evolution_snow_cover_melt_out_from_inputs, plot_GST_bkg_vs_evol_quantile_bins_fit_single_site_from_inputs, plot_mean_bkg_GST_vs_evolution_from_inputs
from SuPerSim.model import fit_stat_model_GST_from_inputs

def plot_hor(path_horizon, list_fig_names, list_figs, print_plots):
    if path_horizon is not None:
        if print_plots:
            print('\n---------------------------------------------------------------------------------------------\n')
            print('Fisheye view of the sky with the visible portion in blue and the blocked one in black:')
        list_fig_names.append('fisheye_horizon')
        list_figs.append(plot_visible_skymap_from_horizon_file(path_horizon, print_plots))

def plot_hist(site, path_pickle,path_thaw_depth, no_weight, show_glaciers, list_fig_names, list_figs, print_plots, query, query_suffix):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')

        print('The following plot is a histogram of the distribution of the statistical weights of all simulations:')
        if query is not None:
            print(f'followed by the ensemble subset {query_suffix}:')
    list_fig_names.append('hist_stat_weight')
    list_figs.append(plot_hist_stat_weights_from_input(site, path_pickle, no_weight, show_glaciers, print_plots, query=None))
    if query is not None:
        list_fig_names.append('hist_stat_weight'+query_suffix)
        list_figs.append(plot_hist_stat_weights_from_input(site, path_pickle, no_weight, show_glaciers, print_plots, query=query))

    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')

        print('The following plot is a histogram of the distribution of glacier simulations wrt to altitude, aspect, slope, and forcing:')
        if query is not None:
            print(f'followed by the ensemble subset {query_suffix}:')
    list_fig_names.append('hist_distrib_glaciers_perma')
    list_figs.append(plot_hist_valid_sim_all_variables_from_input(site, path_thaw_depth, path_pickle, print_plots, query=None))
    if query is not None:
        list_fig_names.append('hist_distrib_glaciers_perma'+query_suffix)
        list_figs.append(plot_hist_valid_sim_all_variables_from_input(site, path_thaw_depth, path_pickle, print_plots, query=query))

def plot_yearly_box(query, query_suffix, time_air_all, time_ground, mean_air_temp, temp_ground_mean, mean_prec, tot_water_prod,
                    mean_air_temp_query, temp_ground_mean_query, tot_water_prod_query,
                    year_bkg_end, year_trans_end, print_plots, list_fig_names, list_figs):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Yearly statistics for air and ground surface temperature, and also precipitation and water production')
        if query is not None:
            print(f'followed by the ensemble subset {query_suffix}:')
    list_fig_names.append('AirTemp_yearly_stats_box')
    list_figs.append(plot_box_yearly_stat_from_inputs('Air temperature', time_air_all[0], mean_air_temp, year_bkg_end, year_trans_end, print_plots))
    list_fig_names.append('GST_yearly_stats_box')
    list_figs.append(plot_box_yearly_stat_from_inputs('GST', time_ground, temp_ground_mean, year_bkg_end, year_trans_end, print_plots))
    list_fig_names.append('Precip_yearly_stats_box')
    list_figs.append(plot_box_yearly_stat_from_inputs('Precipitation', time_air_all[0], mean_prec, year_bkg_end, year_trans_end, print_plots))
    list_fig_names.append('WaterProd_yearly_stats_box')
    list_figs.append(plot_box_yearly_stat_from_inputs('Water production', time_ground, tot_water_prod, year_bkg_end, year_trans_end, print_plots))
    if query is not None:
        list_fig_names.append('AirTemp_yearly_stats_box'+query_suffix)
        list_figs.append(plot_box_yearly_stat_from_inputs('Air temperature', time_air_all[0], mean_air_temp_query, year_bkg_end, year_trans_end, print_plots))
        list_fig_names.append('GST_yearly_stats_box'+query_suffix)
        list_figs.append(plot_box_yearly_stat_from_inputs('GST', time_ground, temp_ground_mean_query, year_bkg_end, year_trans_end, print_plots))
        # list_fig_names.append('Precip_yearly_stats_box'+query_suffix)
        # list_figs.append(plot_box_yearly_stat_from_inputs('Precipitation', time_air_all[0], mean_prec_query, year_bkg_end, year_trans_end, print_plots))
        list_fig_names.append('WaterProd_yearly_stats_box'+query_suffix)
        list_figs.append(plot_box_yearly_stat_from_inputs('Water production', time_ground, tot_water_prod_query, year_bkg_end, year_trans_end, print_plots))

def plot_yearly_stats_atmos(query, query_suffix, time_air_all, temp_air_all, precipitation_all, temp_air_all_query, precipitation_all_query,
                            year_bkg_end, year_trans_end, print_plots, list_fig_names, list_figs):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Plot of yearly statistics for atmospheric timeseries. Mean and several quantiles for each year:')
        if query is not None:
            print(f'followed by the ensemble subset {query_suffix}:')
    list_fig_names.append('AirTemp_yearly_quantiles')
    list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, temp_air_all, 'Air temperature', year_bkg_end, year_trans_end, print_plots))
    list_fig_names.append('AirTemp_yearly_mean')
    list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, temp_air_all, 'Air temperature', year_bkg_end, year_trans_end, print_plots, False))
    list_fig_names.append('Precip_yearly_quantiles')
    list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, precipitation_all, 'Precipitation', year_bkg_end, year_trans_end, print_plots))
    list_fig_names.append('Precip_yearly_mean')
    list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, precipitation_all, 'Precipitation', year_bkg_end, year_trans_end, print_plots, False))
    if query is not None:
        list_fig_names.append('AirTemp_yearly_quantiles'+query_suffix)
        list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, temp_air_all_query, 'Air temperature', year_bkg_end, year_trans_end, print_plots))
        list_fig_names.append('AirTemp_yearly_mean'+query_suffix)
        list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, temp_air_all_query, 'Air temperature', year_bkg_end, year_trans_end, print_plots, False))
        list_fig_names.append('Precip_yearly_quantiles'+query_suffix)
        list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, precipitation_all_query, 'Precipitation', year_bkg_end, year_trans_end, print_plots))
        list_fig_names.append('Precip_yearly_mean'+query_suffix)
        list_figs.append(plot_yearly_quantiles_atmospheric_from_inputs(time_air_all, precipitation_all_query, 'Precipitation', year_bkg_end, year_trans_end, print_plots, False))

def plot_yearly_stats_sims(query, query_suffix, time_ground, temp_ground, snow_height, swe,
                           idxs_depths, list_valid_sim, list_valid_sim_query,
                           year_bkg_end, year_trans_end, print_plots, list_fig_names, list_figs):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Plot of yearly statistics for simulated timeseries. Mean and several quantiles for each year:')
        if query is not None:
            print(f'followed by the ensemble subset {query_suffix}:')
    list_fig_names.append('GST_yearly_quantiles')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, temp_ground, list_valid_sim, 'GST', year_bkg_end, year_trans_end, print_plots))
    list_fig_names.append('GST_yearly_mean')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, temp_ground, list_valid_sim, 'GST', year_bkg_end, year_trans_end, print_plots, plot_quantiles=False))
    for k,v in idxs_depths.items():
        list_fig_names.append(f'{k}m_grdtemp_yearly_quantiles')
        list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, temp_ground, list_valid_sim, f'{k}m ground temperature', year_bkg_end, year_trans_end, print_plots, idx_depth=v))
        list_fig_names.append(f'{k}m_grdtemp_yearly_mean')
        list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, temp_ground, list_valid_sim, f'{k}m ground temperature', year_bkg_end, year_trans_end, print_plots, plot_quantiles=False, idx_depth=v))
    list_fig_names.append('Snow_yearly_quantiles')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, snow_height, list_valid_sim, 'Snow depth', year_bkg_end, year_trans_end, print_plots))
    list_fig_names.append('Snow_yearly_mean')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, snow_height, list_valid_sim, 'Snow depth', year_bkg_end, year_trans_end, print_plots, plot_quantiles=False))
    list_fig_names.append('SWE_yearly_quantiles')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, swe, list_valid_sim, 'SWE', year_bkg_end, year_trans_end, print_plots))
    list_fig_names.append('SWE_yearly_mean')
    list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, swe, list_valid_sim, 'SWE', year_bkg_end, year_trans_end, print_plots, plot_quantiles=False))
    if query is not None:
        list_fig_names.append('GST_yearly_quantiles'+query_suffix)
        list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, temp_ground, list_valid_sim_query, 'GST', year_bkg_end, year_trans_end, print_plots))
        list_fig_names.append('GST_yearly_mean'+query_suffix)
        list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, temp_ground, list_valid_sim_query, 'GST', year_bkg_end, year_trans_end, print_plots, plot_quantiles=False))
        for k,v in idxs_depths.items():
            list_fig_names.append(f'{k}m_grdtemp_yearly_quantiles'+query_suffix)
            list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, temp_ground, list_valid_sim_query, f'{k}m ground temperature', year_bkg_end, year_trans_end, print_plots, idx_depth=v))
            list_fig_names.append(f'{k}m_grdtemp_yearly_mean'+query_suffix)
            list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, temp_ground, list_valid_sim_query, f'{k}m ground temperature', year_bkg_end, year_trans_end, print_plots, plot_quantiles=False, idx_depth=v))
        list_fig_names.append('Snow_yearly_quantiles'+query_suffix)
        list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, snow_height, list_valid_sim_query, 'Snow depth', year_bkg_end, year_trans_end, print_plots))
        list_fig_names.append('Snow_yearly_mean'+query_suffix)
        list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, snow_height, list_valid_sim_query, 'Snow depth', year_bkg_end, year_trans_end, print_plots, plot_quantiles=False))
        list_fig_names.append('SWE_yearly_quantiles'+query_suffix)
        list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, swe, list_valid_sim_query, 'SWE', year_bkg_end, year_trans_end, print_plots))
        list_fig_names.append('SWE_yearly_mean'+query_suffix)
        list_figs.append(plot_yearly_quantiles_sim_from_inputs(time_ground, swe, list_valid_sim_query, 'SWE', year_bkg_end, year_trans_end, print_plots, plot_quantiles=False))

def plot_thaw_depth(query, query_suffix, time_ground, thaw_depth, thaw_depth_query,
                    list_depths, df, df_query, print_plots, list_fig_names, list_figs):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Plot of the evolution of maximum annual thaw for groups of simulations binned by (altitude,slope) couples:')
        if query is not None:
            print(f'followed by the ensemble subset {query_suffix}:')
    list_fig_names.append('max_thaw_depth')
    list_figs.append(plot_yearly_max_thaw_depth_from_inputs(time_ground, thaw_depth, list_depths, df, print_plots))
    if query is not None:
        list_fig_names.append('max_thaw_depth'+query_suffix)
        list_figs.append(plot_yearly_max_thaw_depth_from_inputs(time_ground, thaw_depth_query, list_depths, df_query, print_plots))

def plot_2metrics_seasonal(query, query_suffix, time_ground, temp_ground, snow_height,
                           list_valid_sim, list_valid_sim_query, print_plots, list_fig_names, list_figs):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Plot of 2 timeseries reduced to a 1-year window with mean and 1- and 2-sigma spread:')
        if query is not None:
            print(f'followed by the ensemble subset {query_suffix}:')
    list_fig_names.append('GST_v_snow')
    list_figs.append(plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground, [temp_ground, snow_height], [list_valid_sim, list_valid_sim], ['GST', 'Snow depth'], print_plots))
    if query is not None:
        list_fig_names.append('GST_v_snow'+query_suffix)
        list_figs.append(plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground, [temp_ground, snow_height], [list_valid_sim_query, list_valid_sim_query], ['GST', 'Snow depth'], print_plots))

def plot_1metric_seasonal(query, query_suffix, time_ground, time_bkg_ground, time_trans_ground,
                          year_bkg_end, year_trans_end, idxs_depths,
                          temp_ground, snow_height,
                          list_valid_sim, list_valid_sim_query, print_plots, show_decades,
                          show_excep_years, custom_years, list_fig_names, list_figs):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Plot of a single timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread, for background and transient periods:')
        if query is not None:
            print(f'followed by the ensemble subset {query_suffix}:')
    list_fig_names.append('GST_1year_bkg_v_transient')
    list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground, temp_ground], [list_valid_sim, list_valid_sim], 'GST', ['Background', 'Transient'], [time_bkg_ground, time_trans_ground], print_plots))
    if show_excep_years:
        panda_test = sim_data_to_panda(time_ground, temp_ground, list_valid_sim, 'GST')
        _, _, excep_years = panda_data_to_yearly_stats(panda_test, year_trans_end)
        list_fig_names.append('GST_1year_bkg_v_transient_excep_years')
        list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground]*(len(excep_years)+1), [list_valid_sim]*(len(excep_years)+1),
                                                                                'GST', ['Background']+[f'Year {k}' for k in excep_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in excep_years], print_plots))
    if custom_years is not None:
        list_fig_names.append('GST_1year_bkg_v_transient_custom_years')
        list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground]*(len(custom_years)+1), [list_valid_sim]*(len(custom_years)+1),
                                                                                'GST', ['Background']+[f'Year {k}' for k in custom_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in custom_years], print_plots))
    if query is not None:
        list_fig_names.append('GST_1year_bkg_v_transient'+query_suffix)
        list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground, temp_ground], [list_valid_sim_query, list_valid_sim_query], 'GST', ['Background', 'Transient'], [time_bkg_ground, time_trans_ground], print_plots))
        if show_excep_years:
            panda_test = sim_data_to_panda(time_ground, temp_ground, list_valid_sim_query, 'GST')
            _, _, excep_years = panda_data_to_yearly_stats(panda_test, year_trans_end)
            list_fig_names.append('GST_1year_bkg_v_transient_excep_years'+query_suffix)
            list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground]*(len(excep_years)+1), [list_valid_sim_query]*(len(excep_years)+1),
                                                                                    'GST', ['Background']+[f'Year {k}' for k in excep_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in excep_years], print_plots))
        if custom_years is not None:
            list_fig_names.append('GST_1year_bkg_v_transient_custom_years'+query_suffix)
            list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground]*(len(custom_years)+1), [list_valid_sim_query]*(len(custom_years)+1),
                                                                                    'GST', ['Background']+[f'Year {k}' for k in custom_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in custom_years], print_plots))
    if show_decades:
        list_decs = []
        for dec in range(int((year_trans_end-year_bkg_end-1)/10)):
            list_decs.append(year_bkg_end+1+dec*10)
        list_fig_names.append('GST_1year_bkg_v_transient_decades')
        list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground]*(len(list_decs)+2), [list_valid_sim]*(len(list_decs)+2),
                                                                                'GST', ['Background', 'Transient']+[f'Decade {dec}-{dec+9}' for dec in list_decs], [time_bkg_ground, time_trans_ground]+[list_tokens_year(time_ground, dec, dec+10)[2] for dec in list_decs], print_plots))
    
    for k,v in idxs_depths.items():
        list_fig_names.append(f'{k}m_grdtemp_1year_bkg_v_transient')
        list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground, temp_ground], [list_valid_sim, list_valid_sim], f'{k}m ground temperature', ['Background', 'Transient'], [time_bkg_ground, time_trans_ground], print_plots, idx_depth=v))
        if show_excep_years:
            panda_test = sim_data_to_panda(time_ground, temp_ground, list_valid_sim, f'{k}m ground temperature', idx_depth=v)
            _, _, excep_years = panda_data_to_yearly_stats(panda_test, year_trans_end)
            list_fig_names.append(f'{k}m_grdtemp_1year_bkg_v_transient_excep_years')
            list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground]*(len(excep_years)+1), [list_valid_sim]*(len(excep_years)+1),
                                                                                    f'{k}m ground temperature', ['Background']+[f'Year {k}' for k in excep_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in excep_years], print_plots, idx_depth=v))
        if custom_years is not None:
            list_fig_names.append(f'{k}m_grdtemp_1year_bkg_v_transient_custom_years')
            list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground]*(len(custom_years)+1), [list_valid_sim]*(len(custom_years)+1),
                                                                                    f'{k}m ground temperature', ['Background']+[f'Year {k}' for k in custom_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in custom_years], print_plots, idx_depth=v))
        if query is not None:
            list_fig_names.append(f'{k}m_grdtemp_1year_bkg_v_transient'+query_suffix)
            list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground, temp_ground], [list_valid_sim_query, list_valid_sim_query], f'{k}m ground temperature', ['Background', 'Transient'], [time_bkg_ground, time_trans_ground], print_plots, idx_depth=v))
            if show_excep_years:
                panda_test = sim_data_to_panda(time_ground, temp_ground, list_valid_sim_query, f'{k}m ground temperature', idx_depth=v)
                _, _, excep_years = panda_data_to_yearly_stats(panda_test, year_trans_end)
                list_fig_names.append(f'{k}m_grdtemp_1year_bkg_v_transient_excep_years'+query_suffix)
                list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground]*(len(excep_years)+1), [list_valid_sim_query]*(len(excep_years)+1),
                                                                                        f'{k}m ground temperature', ['Background']+[f'Year {k}' for k in excep_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in excep_years], print_plots, idx_depth=v))
            if custom_years is not None:
                list_fig_names.append(f'{k}m_grdtemp_1year_bkg_v_transient_custom_years'+query_suffix)
                list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground]*(len(custom_years)+1), [list_valid_sim_query]*(len(custom_years)+1),
                                                                                        f'{k}m ground temperature', ['Background']+[f'Year {k}' for k in custom_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in custom_years], print_plots, idx_depth=v))
        if show_decades:
            list_decs = []
            for dec in range(int((year_trans_end-year_bkg_end-1)/10)):
                list_decs.append(year_bkg_end+1+dec*10)
            list_fig_names.append(f'{k}m_grdtemp_1year_bkg_v_transient_decades')
            list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [temp_ground]*(len(list_decs)+2), [list_valid_sim]*(len(list_decs)+2),
                                                                                    f'{k}m ground temperature', ['Background', 'Transient']+[f'Decade {dec}-{dec+9}' for dec in list_decs], [time_bkg_ground, time_trans_ground]+[list_tokens_year(time_ground, dec, dec+10)[2] for dec in list_decs], print_plots, idx_depth=v))


    list_fig_names.append('Snow_1year_bkg_v_transient')
    list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [snow_height, snow_height], [list_valid_sim, list_valid_sim], 'Snow depth', ['Background', 'Transient'], [time_bkg_ground, time_trans_ground], print_plots))
    if show_excep_years:
        panda_test = sim_data_to_panda(time_ground, snow_height, list_valid_sim, 'Snow depth')
        _, _, excep_years = panda_data_to_yearly_stats(panda_test, year_trans_end)
        list_fig_names.append('Snow_1year_bkg_v_transient_excep_years')
        list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [snow_height]*(len(excep_years)+1), [list_valid_sim]*(len(excep_years)+1),
                                                                                'Snow depth', ['Background']+[f'Year {k}' for k in excep_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in excep_years], print_plots))
    if custom_years is not None:
        list_fig_names.append('Snow_1year_bkg_v_transient_custom_years')
        list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [snow_height]*(len(custom_years)+1), [list_valid_sim]*(len(custom_years)+1),
                                                                                'Snow depth', ['Background']+[f'Year {k}' for k in custom_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in custom_years], print_plots))
    if show_decades:
        list_decs = []
        for dec in range(int((year_trans_end-year_bkg_end-1)/10)):
            list_decs.append(year_bkg_end+1+dec*10)
        list_fig_names.append('Snow_1year_bkg_v_transient_decades')
        list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [snow_height]*(len(list_decs)+2), [list_valid_sim]*(len(list_decs)+2),
                                                                                'Snow depth', ['Background', 'Transient']+[f'Decade {dec}-{dec+9}' for dec in list_decs], [time_bkg_ground, time_trans_ground]+[list_tokens_year(time_ground, dec, dec+10)[2] for dec in list_decs], print_plots))
    if query is not None:
        list_fig_names.append('Snow_1year_bkg_v_transient'+query_suffix)
        list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [snow_height, snow_height], [list_valid_sim_query, list_valid_sim_query], 'Snow depth', ['Background', 'Transient'], [time_bkg_ground, time_trans_ground], print_plots))
        if show_excep_years:
            panda_test = sim_data_to_panda(time_ground, snow_height, list_valid_sim_query, 'Snow depth')
            _, _, excep_years = panda_data_to_yearly_stats(panda_test, year_trans_end)
            list_fig_names.append('Snow_1year_bkg_v_transient_excep_years'+query_suffix)
            list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [snow_height]*(len(excep_years)+1), [list_valid_sim_query]*(len(excep_years)+1),
                                                                                    'Snow depth', ['Background']+[f'Year {k}' for k in excep_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in excep_years], print_plots))
        if custom_years is not None:
            list_fig_names.append('Snow_1year_bkg_v_transient_custom_years'+query_suffix)
            list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [snow_height]*(len(custom_years)+1), [list_valid_sim_query]*(len(custom_years)+1),
                                                                                    'Snow depth', ['Background']+[f'Year {k}' for k in custom_years], [time_bkg_ground]+[list_tokens_year(time_ground, k, k+1)[2] for k in custom_years], print_plots))
        if show_decades:
            list_decs = []
            for dec in range(int((year_trans_end-year_bkg_end-1)/10)):
                list_decs.append(year_bkg_end+1+dec*10)
            list_fig_names.append('Snow_1year_bkg_v_transient_decades'+query_suffix)
            list_figs.append(plot_sanity_one_year_quantiles_two_periods_from_inputs(time_ground, [snow_height]*(len(list_decs)+2), [list_valid_sim_query]*(len(list_decs)+2),
                                                                                    'Snow depth', ['Background', 'Transient']+[f'Decade {dec}-{dec+9}' for dec in list_decs], [time_bkg_ground, time_trans_ground]+[list_tokens_year(time_ground, dec, dec+10)[2] for dec in list_decs], print_plots))

def plot_normdev(query, query_suffix, time_ground, time_air_all,
                 tot_water_prod, mean_air_temp, temp_ground_mean, temp_ground_mean_deep,
                 tot_water_prod_query, mean_air_temp_query, temp_ground_mean_query, temp_ground_mean_deep_query,
                 site, path_pickle, year_bkg_end, year_trans_end,
                 rockfall_values, print_plots, show_landslide_time, list_fig_names, list_figs):
    if print_plots:
        print('Plots of the normalized distance of air and ground temperature, water production, and thaw_depth as a function of time')
        print('We could also called the normalized deviation: standardized anomaly for instance, given by (x-\\mu)/\\sigma)')
    if 'year' in rockfall_values.keys():
        year_rockfall = rockfall_values['year']
        if print_plots:
            print('\n---------------------------------------------------------------------------------------------\n')
            print('Granularity: week and month side by side')
            if query is not None:
                print(f'followed by the ensemble subset {query_suffix}:')
        list_fig_names.append('normdev_week_month')
        list_figs.append(plot_aggregating_distance_temp_all_from_input(['Water production', 'Air temperature', 'Ground temperature'],
                                        [time_ground, time_air_all[0], time_ground],
                                        [tot_water_prod, mean_air_temp, temp_ground_mean],
                                        ['week', 'month'], site, path_pickle, year_bkg_end, year_trans_end, year_rockfall, print_plots, False,
                                        show_landslide_time))
        if query is not None:
            list_fig_names.append('normdev_week_month'+query_suffix)
            list_figs.append(plot_aggregating_distance_temp_all_from_input(['Water production', 'Air temperature', 'Ground temperature'],
                                            [time_ground, time_air_all[0], time_ground],
                                            [tot_water_prod_query, mean_air_temp_query, temp_ground_mean_query],
                                            ['week', 'month'], site, path_pickle, year_bkg_end, year_trans_end, year_rockfall, print_plots, False,
                                            show_landslide_time))

    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Granularity: year, plotted for all years')
        if query is not None:
            print(f'followed by the ensemble subset {query_suffix}:')
    list_fig_names.append('normdev')
    list_figs.append(plot_aggregating_distance_temp_all_from_input(['Water production', 'Air temperature', 'GST']+[f'{k}m ground temperature' for k in temp_ground_mean_deep.keys()],
                                        [time_ground, time_air_all[0], time_ground]+[time_ground]*len(temp_ground_mean_deep),
                                        [tot_water_prod, mean_air_temp, temp_ground_mean]+list(temp_ground_mean_deep.values()),
                                        ['year'], site, path_pickle, year_bkg_end, year_trans_end, 0, print_plots, False,
                                        show_landslide_time))
    if query is not None:
        list_fig_names.append('normdev'+query_suffix)
        list_figs.append(plot_aggregating_distance_temp_all_from_input(['Water production', 'Air temperature', 'GST']+[f'{k}m ground temperature' for k in temp_ground_mean_deep.keys()],
                                            [time_ground, time_air_all[0], time_ground]+[time_ground]*len(temp_ground_mean_deep),
                                            [tot_water_prod_query, mean_air_temp_query, temp_ground_mean_query]+list(temp_ground_mean_deep_query.values()),
                                            ['year'], site, path_pickle, year_bkg_end, year_trans_end, 0, print_plots, False,
                                            show_landslide_time))

def plot_individual_heatmap(site, path_pickle, rockfall_values, alt_list, alt_index_abs,
                            print_plots, list_fig_names, list_figs):
    alt_show = rockfall_values['altitude'] if ((rockfall_values['exact_topo']) and (rockfall_values['altitude'] in alt_list)) else alt_index_abs
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print(f'Heatmap of the background mean GST as a function of aspect and slope at {alt_show} m:')
    list_fig_names.append('heatmap_centre_GST_bkg')
    list_figs.append(plot_table_mean_GST_aspect_slope_single_altitude_from_inputs(site, path_pickle, alt_show, print_plots, background=True, box=True))
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print(f'Heatmap of the evolution of the mean GST between the background and the transient periods as a function of aspect and slope at {alt_show} m:')
    list_fig_names.append('heatmap_centre_GST_evol')
    list_figs.append(plot_table_mean_GST_aspect_slope_single_altitude_from_inputs(site, path_pickle, alt_show, print_plots, background=False, box=True))

def plot_heatmaps(site, path_pickle, show_glaciers, path_thaw_depth,
                  print_plots, list_fig_names, list_figs):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitudes')
    list_fig_names.append('heatmap_GST')
    list_figs.append(plot_table_mean_GST_aspect_slope_all_altitudes_from_inputs(site, path_pickle, show_glaciers, print_plots, box=True))

    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Polar heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitude')
    list_fig_names.append('heatmap_GST_polar_alts')
    list_figs.append(plot_table_mean_GST_aspect_slope_all_altitudes_polar_from_inputs(site, path_pickle, print_plots, box=True))
    list_fig_names.append('heatmap_GST_polar')
    list_figs.append(plot_table_mean_GST_aspect_altitude_all_slopes_polar_from_inputs(site, path_pickle, print_plots, box=True))

    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Polar plot of the permafrost and glacier spatial distribution as a function of aspect and slope at all altitude')
    list_fig_names.append('heatmap_perma_polar_alts')
    list_figs.append(plot_permafrost_all_altitudes_polar_from_inputs(site, path_pickle, path_thaw_depth, print_plots, box=True))
    list_fig_names.append('heatmap_perma_polar')
    list_figs.append(plot_permafrost_all_slopes_polar_from_inputs(site, path_pickle, path_thaw_depth, print_plots, box=True))

def plot_quantiles(site, path_pickle, print_plots, list_fig_names, list_figs):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('CDF of background, transient, and evolution GST:')
    list_fig_names.append('CDF_GST_SO')
    list_figs.append(plot_cdf_GST_from_inputs(site, path_pickle, print_plots))

    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference:')
    list_fig_names.append('heatmap_percentiles_GST_bkg_trans_evol')
    list_figs.append(plot_heatmap_percentile_GST_from_inputs(site, path_pickle, print_plots))

def plot_meltout(site, path_pickle, print_plots, list_fig_names, list_figs):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Histogram of the evolution of the snow cover (in days) and melt-out date:')
    list_fig_names.append('hist_snow_cover')
    list_figs.append(plot_evolution_snow_cover_melt_out_from_inputs(site, path_pickle, print_plots))

def plot_GST_evol(site, path_pickle, print_plots, list_fig_names, list_figs):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Plot of mean GST evolution vs background GST, fit, and binning per 10% quantiles')
    list_fig_names.append('GST_evol_v_bkg')
    list_figs.append(plot_GST_bkg_vs_evol_quantile_bins_fit_single_site_from_inputs(site, path_pickle, print_plots, query=None))

    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Scatter plot of mean background GST vs evolution of mean GST between the background and transient period')
    list_fig_names.append('GST_evol_v_bkg_alts')
    list_figs.append(plot_mean_bkg_GST_vs_evolution_from_inputs(site, path_pickle, print_plots))

def plot_parity(site, path_pickle, print_plots, list_fig_names, list_figs, all_data=False, diff_forcings=True):
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')
        print('Parity plot (statistically-modeled vs numerically-simulated) of background mean GST:')
    list_fig_names.append('parity_plot_stat_model_bkg_mean_GST')
    fig_model, _, _, _, _, _, _ = fit_stat_model_GST_from_inputs(site, path_pickle, print_plots, all_data, diff_forcings)
    list_figs.append(fig_model)
