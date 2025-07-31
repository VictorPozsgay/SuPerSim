"""This module automates the plotting of summary statistics for timeseries"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import textwrap
import pickle
from ReSubPlot.legends import isolate_legend

from SuPerSim.open import open_air_nc, open_ground_nc, open_snow_nc, open_swe_nc, open_SW_direct_nc, open_SW_diffuse_nc, open_SW_up_nc, open_SW_down_nc, open_SW_net_nc, open_LW_net_nc
from SuPerSim.pickling import load_all_pickles
from SuPerSim.yearlystats import plot_yearly_quantiles_side_by_side_sim_from_inputs
from SuPerSim.seasonal import plot_sanity_two_variables_one_year_quantiles_from_inputs, plot_sanity_two_variables_two_sites_one_year_quantiles_side_by_side_from_inputs
from SuPerSim.evolution import plot_GST_bkg_vs_evol_quantile_bins_fit_two_sites_from_input
from SuPerSim.preparation import prep_data_plot, prep_sim_data_plot, prep_atmos_data_plot
from SuPerSim.captions import captions

from SuPerSim.plotblocks import plot_hor, plot_hist, plot_yearly_box, plot_yearly_stats_atmos, plot_yearly_stats_sims, plot_thaw_depth, plot_2metrics_seasonal, plot_1metric_seasonal, plot_normdev, plot_normdev_bar, plot_individual_heatmap, plot_heatmaps, plot_quantiles, plot_meltout, plot_GST_evol, plot_parity

def plot_all(site,
             path_forcing_list, path_ground, path_snow, path_swe, path_thaw_depth, path_pickle,
             year_bkg_end, year_trans_end, path_horizon=None, no_weight=True,
             print_plots=True, split_legend=True, save_plots_pdf=False, custom_years=None, query=None,
             show_hor=True, show_hist=True, show_glaciers=True, show_yearly_box=True,
             show_yearly_stats_atmos=True, show_yearly_stats_sims=True, show_thaw_depth=True,
             show_2metrics_seasonal=True, show_1metric_seasonal=True,
             show_decades=True, show_excep_years=True, show_normdev=True, show_landslide_time=True,
             show_normdev_bar=True, show_individual_heatmap=False, show_heatmaps=True, 
             show_quantiles=True, show_meltout=True, show_GST_evol=True, show_parity=False):
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
    print_plots : bool, optional
        Whether or not to show plots. Usually True but if one simply wants to get the return dictionary of figures and no plots, choose False.
    split_legend : bool, optional
        Whether or not to strip the legend off of all figures. If True, every figure will be returned in 3 objects: original figure with legend,
        same figure without legend, and legend object
        This uses the function ReSubPlot.legends.isolate_legend
    save_plots_pdf : bool, optional
        If True, figure will be saved to PDF with the name
        f'{site}_{suffix}.pdf' with suffix in
        ['legend_only', 'legend_on', legend_off']
    custom_years : list, optional
        If None, nothing happens, but if a list of integer years is given, the tiemeseries for these years will be printed too for the seasonal plots.
    query : dict, optional
        If query is None then we consider ALL simulations
        However, this is the place where we can select a subset of simulations
        This is done in the following way
        query = {'param_1': 'value_1', ..., 'param_n': 'value_n'}
        and the keys should be taken from a valid column of 'df', 
        e.g. 'altitude', 'slope', 'aspect', 'material', 'maxswe', 'snow'
    show_hor : bool, optional
        Whether or not to produce the horizon plot
    show_hist : bool, optional
        Whether or not to plot the histograms (glacier and permafrost disctibution)
    show_glaciers : bool, optional
        If True, shows the glacier simulations with a 0 weight, if False, those are ignored. t only applies if either 'show_hist' or 'show_heatmaps' is True.
    show_yearly_box : bool, optional
        Whether or not to plot the yearly boxplots
    show_yearly_stats_atmos : bool, optional
        Whether or not to plot the yearly stats for atmospheric data
    show_yearly_stats_sims : bool, optional
        Whether or not to plot the yearly stats for simulated data
    show_thaw_depth : bool, optional
        Whether or not to plot the thaw depth evolution
    show_2metrics_seasonal : bool, optional
        Whether or not to plot a seasonal comparison of 2 metrics 
    show_1metric_seasonal : bool, optional
        Whether or not to plot the seasonal mean and deviation of a single metric at a time
    show_decades : bool, optional   
        Whether or not to plot decadal data, if 'show_1metric_seasonal'=True
    show_excep_years : bool, optional
        Whether or not to plot data for the 3 most exceptional years, if 'show_1metric_seasonal'=True
    show_normdev : bool, optional
        Whether or not to plot the normalized deviation
    show_landslide_time : bool, optional
        Whether or not to show the landslide time on the ormalized deviation plot, if 'show_normdev'=True
    show_normdev_bar : bool, optional
        Whether or not to plot the normalized deviation bar plots (mean annual values)
    show_individual_heatmap : bool, optional
        Whether or not to plot heatmaps for a unique altitude
    show_heatmaps : bool, optional
        Whether or not to plot all heatmaps
    show_quantiles : bool, optional
        Whether or not to plot quntile plots (CDF and evolution of percentiles)
    show_meltout : bool, optional
        Whether or not to plot snow meltout evolution (histogram)
    show_GST_evol : bool, optional
        Whether or not to plot the GST evolution vs background
    show_parity : bool, optional
        Whether or not to plot the parity plot (model results)

    Returns
    -------
    dict_final: dict
        Dictionary {k: v} where the keys 'k' are 'fig_legend_on' (and 'fig_legend_off', 'fig_legend_only' if
        split_legend=True), and 'caption'; and 'v' are the associated objects:
            'fig_legend_on' -> original figure object
            'fig_legend_off' -> same figure, no legend
            'fig_legend_only' -> legend object
            'caption' -> long form standardize caption of the figure
    """  

    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    pkl = load_all_pickles(site, path_pickle)
    df = pkl['df']
    list_valid_sim = pkl['list_valid_sim']
    # df_stats = pkl['df_stats']
    rockfall_values = pkl['rockfall_values']

    #####################################################################################
    # PLOTS
    #####################################################################################

    # Check if the plot dictionary already exists
    file_name = f"dict_final{('' if site=='' else '_')}{site}_split_{split_legend}.pkl"
    my_path = path_pickle + file_name

    # try to open the pickle file, if it exists
    try: 
        # Open the file in binary mode 
        with open(my_path, 'rb') as file: 
            # Call load method to deserialze 
            dict_final = pickle.load(file) 
        print('Succesfully opened the pre-existing pickle:', file_name)

    # if the pickle file does not exist, we have to create it
    except (OSError, IOError) as e:

        list_depths, idxs_depths, alt_list, alt_index_abs, alt_query_idx = prep_data_plot(site, path_ground, path_pickle, query)
        time_ground, time_bkg_ground, time_trans_ground, temp_ground, temp_ground_mean, temp_ground_mean_deep, snow_height, swe, thaw_depth, _ = prep_sim_data_plot(site,
                       path_ground, path_snow, path_swe, path_thaw_depth, path_pickle,
                       year_bkg_end, year_trans_end, no_weight, query=None)
        _, _, _, _, temp_ground_mean_query, temp_ground_mean_deep_query, _, _, thaw_depth_query, list_valid_sim_query = prep_sim_data_plot(site,
                       path_ground, path_snow, path_swe, path_thaw_depth, path_pickle,
                       year_bkg_end, year_trans_end, no_weight, query=query)
        time_air_all, temp_air_all, precipitation_all, mean_air_temp, tot_water_prod, mean_prec = prep_atmos_data_plot(site,
                         path_forcing_list, path_ground, path_swe, path_pickle,
                         year_bkg_end, year_trans_end, no_weight, None, alt_query_idx)
        _, temp_air_all_query, precipitation_all_query, mean_air_temp_query, tot_water_prod_query, _ = prep_atmos_data_plot(site,
                         path_forcing_list, path_ground, path_swe, path_pickle,
                         year_bkg_end, year_trans_end, no_weight, query, alt_query_idx)
        
        df_query = df.loc[list_valid_sim_query].reset_index(drop=True)

        list_figs = []
        list_fig_names = []

        query_suffix = ''
        if query is not None:
            for k,v in query.items():
                query_suffix += f'_{k}{v}'

        if show_hor:
            plot_hor(path_horizon, list_fig_names, list_figs, print_plots)

        if show_hist:
            plot_hist(site, path_pickle, path_thaw_depth, no_weight, show_glaciers, list_fig_names, list_figs, print_plots, query, query_suffix)

        if print_plots:
            print('\n---------------------------------------------------------------------------------------------')
            print('------------------------------------- TEMPORAL ANALYSIS -------------------------------------')
            print('---------------------------------------------------------------------------------------------\n')
        
        if show_yearly_box:
            plot_yearly_box(query, query_suffix, time_air_all, time_ground,
                            mean_air_temp, temp_ground_mean, mean_prec, tot_water_prod,
                            mean_air_temp_query, temp_ground_mean_query, tot_water_prod_query,
                            year_bkg_end, year_trans_end, print_plots, list_fig_names, list_figs)

        if show_yearly_stats_atmos:
            plot_yearly_stats_atmos(query, query_suffix, time_air_all, temp_air_all,
                                    precipitation_all, temp_air_all_query, precipitation_all_query,
                                    year_bkg_end, year_trans_end, print_plots, list_fig_names, list_figs)
        
        if show_yearly_stats_sims:
            plot_yearly_stats_sims(query, query_suffix, time_ground, temp_ground, snow_height, swe,
                                   idxs_depths, list_valid_sim, list_valid_sim_query,
                                   year_bkg_end, year_trans_end, print_plots, list_fig_names, list_figs)

        if show_thaw_depth:
            plot_thaw_depth(query, query_suffix, time_ground, thaw_depth, thaw_depth_query,
                            list_depths, df, df_query, print_plots, list_fig_names, list_figs)

        if show_2metrics_seasonal:
            plot_2metrics_seasonal(query, query_suffix, time_ground, temp_ground, snow_height,
                                   list_valid_sim, list_valid_sim_query, print_plots, list_fig_names, list_figs)
        
        if show_1metric_seasonal:
            plot_1metric_seasonal(query, query_suffix, time_ground, time_bkg_ground, time_trans_ground,
                                  year_bkg_end, year_trans_end, idxs_depths,
                                  temp_ground, snow_height,
                                  list_valid_sim, list_valid_sim_query, print_plots, show_decades,
                                  show_excep_years, custom_years, list_fig_names, list_figs)
        
        # # # # print('')
        # # # This works well but it would be better to smooth the data
        # # # ALSO CAREFUL WHEN USING THIS FUNCTION WITH TIME_AIR, NEED TO CHANGE stats_air_all_years_simulations_to_single_year()
        # # # TO DO THAT, DRAW INSPIRATION FROM stats_all_years_simulations_to_single_year()
        # # # plot_sanity_one_year_quantiles_two_periods_from_inputs(time_air_all[0], [temp_air_all[0], temp_air_all[0]], [None, None], 'Air temperature', ['Background', 'Transient'], [time_bkg_air, time_trans_air], print_plots)

        if show_normdev:
            plot_normdev(query, query_suffix, time_ground, time_air_all,
                         tot_water_prod, mean_air_temp, temp_ground_mean, temp_ground_mean_deep,
                         tot_water_prod_query, mean_air_temp_query, temp_ground_mean_query, temp_ground_mean_deep_query,
                         site, path_pickle, year_bkg_end, year_trans_end,
                         rockfall_values, print_plots, show_landslide_time, list_fig_names, list_figs)


        if show_normdev_bar:
            plot_normdev_bar(query, query_suffix, time_ground, time_air_all,
                             tot_water_prod, mean_air_temp, temp_ground_mean, temp_ground_mean_deep,
                             tot_water_prod_query, mean_air_temp_query, temp_ground_mean_query, temp_ground_mean_deep_query,
                             year_bkg_end, year_trans_end, print_plots, list_fig_names, list_figs)

        if print_plots:
            print('\n---------------------------------------------------------------------------------------------')
            print('------------------------------------- SPATIAL ANALYSIS --------------------------------------')
            print('---------------------------------------------------------------------------------------------\n')


        if show_individual_heatmap:
            plot_individual_heatmap(site, path_pickle, rockfall_values, alt_list, alt_index_abs,
                                    print_plots, list_fig_names, list_figs)

        if show_heatmaps:
            plot_heatmaps(site, path_pickle, show_glaciers, path_thaw_depth,
                          print_plots, list_fig_names, list_figs)

        if print_plots:
            print('\n---------------------------------------------------------------------------------------------')
            print('-------------------------------------- FURTHER  PLOTS ---------------------------------------')
            print('---------------------------------------------------------------------------------------------\n')

        if show_quantiles:
            plot_quantiles(site, path_pickle, print_plots, list_fig_names, list_figs)

        if show_meltout:
            plot_meltout(site, path_pickle, print_plots, list_fig_names, list_figs)
        
        if show_GST_evol:
            plot_GST_evol(site, path_pickle, print_plots, list_fig_names, list_figs)

        if show_parity:
            plot_parity(site, path_pickle, print_plots, list_fig_names, list_figs, all_data=False, diff_forcings=True)
        
        print('\n---------------------------------------------------------------------------------------------')
        print('---------------------------------- SUCCESSFULLY COMPLETED -----------------------------------')
        print('---------------------------------------------------------------------------------------------\n')

        caption_extra_query = ' The ensemble analyzed here is a subset of the full ensemble, corresponding to simulations with'
        units_caption = {'altitude': 'm', 'slope': '°', 'aspect': '°', 'material': '', 'maxswe': 'mm', 'snow': ''}
        if query is not None:
            for k,v in query.items():
                caption_extra_query += f' {k}={v}{units_caption[k]},'
        caption_extra_query = caption_extra_query.rstrip(',')
        caption_extra_query += '.'

        dic_figs = dict(zip(list_fig_names, list_figs))
        if query is not None:
            dic_captions = {k: captions[k.split(query_suffix)[0]]+(caption_extra_query if query_suffix in k else '') for k in dic_figs.keys()}
        else:
            dic_captions = {k: captions[k] for k in dic_figs.keys()}
        # dic_captions = {k:v for (k,v) in captions.items() if k in dic_figs.keys()}

        # Write to file
        with open(path_pickle+site+'_captions_readable.txt', "w", encoding="utf-8") as f:
            for key, value in dic_captions.items():
                f.write(f"{key}:\n")
                wrapped_value = textwrap.fill(value, width=80, initial_indent="    ", subsequent_indent="    ")
                f.write(wrapped_value + "\n\n")

        if not split_legend:
            dict_final = {k: {'fig_legend_on': v, 'caption': dic_captions[k]} for k,v in dic_figs.items()}
            if save_plots_pdf:
                for k,v in dic_figs.items():
                    v.savefig(f'{path_pickle+site}_{k}_legend_on.pdf', bbox_inches='tight')
        else:
            dic_figs_legend_on = {k: [] for k in dic_figs.keys()}
            dic_figs_legend_off = {k: [] for k in dic_figs.keys()}
            dic_figs_legend_only = {k: [] for k in dic_figs.keys()}

            for k,v in dic_figs.items():
                dic_figs_legend_on[k], dic_figs_legend_off[k], dic_figs_legend_only[k] = isolate_legend(v, path_pickle+site+'_'+k if save_plots_pdf else False)
            dict_final = {k: {'fig_legend_on': v, 'fig_legend_off': dic_figs_legend_off[k], 'fig_legend_only': dic_figs_legend_only[k], 'caption': dic_captions[k]} for k,v in dic_figs_legend_on.items()}

        # Open a file and use dump() 
        with open(my_path, 'wb') as file:
            # A new file will be created 
            pickle.dump(dict_final, file)
        print('Created a new pickle:', file_name)

        # useless line just to use the variable 'e' so that I don't get an error
        if e == 0:
            pass

    return dict_final




def plot_camparison_two_sites(list_site, list_label_site,
             list_path_forcing_list, list_path_ground, list_path_snow, list_path_swe,
             list_path_SW_direct, list_path_SW_diffuse, list_path_SW_up,
             list_path_SW_down, list_path_SW_net, list_path_LW_net,
             list_path_pickle, year_bkg_end, year_trans_end, print_plots=True):
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
    print_plots : bool, optional
        Whether or not to show plots. Usually True but if one simply wants to get the return dictionary of figures and no plots, choose False.


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

    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')

        print('Series of plots comparing both sites.')

        print('\n---------------------------------------------------------------------------------------------\n')

        print('Plot of a single timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread, for both sites:')
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [temp_ground[0], temp_ground[1]], [list_valid_sim[0], list_valid_sim[1]], ['GST'], print_plots, list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [SW_direct[0], SW_direct[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW direct'], print_plots, list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [SW_diffuse[0], SW_diffuse[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW diffuse'], print_plots, list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [SW_down[0], SW_down[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW up'], print_plots, list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [SW_up[0], SW_up[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW down'], print_plots, list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [SW_net[0], SW_net[1]], [list_valid_sim[0], list_valid_sim[1]], ['SW net'], print_plots, list_label_site)
    plot_sanity_two_variables_one_year_quantiles_from_inputs(time_ground[0], [LW_net[0], LW_net[1]], [list_valid_sim[0], list_valid_sim[1]], ['LW net'], print_plots, list_label_site)

    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')

        print('Plot of seasonal statistics for GST and SW net for both sites side by side:')
    plot_sanity_two_variables_two_sites_one_year_quantiles_side_by_side_from_inputs(time_ground[0], [[temp_ground[0], temp_ground[1]], [SW_net[0], SW_net[1]]], [list_valid_sim[0], list_valid_sim[1]], ['GST', 'SW net'], list_label_site, print_plots)
    
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')

        print('Plot of yearly, background, and transient statistics for GST for both sites side by side:')
    plot_yearly_quantiles_side_by_side_sim_from_inputs(time_ground[0], [temp_ground[0], temp_ground[1]], [list_valid_sim[0], list_valid_sim[1]], 'GST', list_label_site, year_bkg_end, year_trans_end, print_plots)
    plot_yearly_quantiles_side_by_side_sim_from_inputs(time_ground[0], [temp_ground[0], temp_ground[1]], [list_valid_sim[0], list_valid_sim[1]], 'GST', list_label_site, year_bkg_end, year_trans_end, print_plots, False)
    
    if print_plots:
        print('\n---------------------------------------------------------------------------------------------\n')

        print('Plot of mean GST evolution vs background GST, fit, and binning per 10% quantiles for both sites:')
    plot_GST_bkg_vs_evol_quantile_bins_fit_two_sites_from_input(list_site, list_path_pickle, list_label_site, print_plots, query=None)

    
    print('\n---------------------------------------------------------------------------------------------')
    print('---------------------------------- SUCCESSFULLY COMPLETED -----------------------------------')
    print('---------------------------------------------------------------------------------------------\n')
