"""This module prepares the data for plotting the summary statistics for timeseries"""

import numpy as np

from SuPerSim.open import open_air_nc, open_ground_nc, open_snow_nc, open_swe_nc, open_thaw_depth_nc
from SuPerSim.mytime import list_tokens_year
from SuPerSim.pickling import load_all_pickles
from SuPerSim.weights import assign_weight_sim
from SuPerSim.runningstats import mean_all_altitudes, mean_all_reanalyses, assign_tot_water_prod

def prep_data_plot(site, path_ground, path_pickle, query):
    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    pkl = load_all_pickles(site, path_pickle)
    df_stats = pkl['df_stats']

    f_ground, _, _ = open_ground_nc(path_ground)

    list_depths = list(f_ground['soil_depth'][:])
    idxs_depths = {depth: list_depths.index(depth) if depth in list_depths else -1 for depth in [1,5,10]}
    idxs_depths = {k: v for k,v in idxs_depths.items() if v!=-1}

    alt_list = sorted(set(df_stats['altitude']))
    alt_index = int(np.floor((len(alt_list)-1)/2))
    alt_index_abs = alt_list[alt_index]
    alt_query = alt_index_abs

    if query is not None:
        for k,v in query.items():
            if k=='altitude':
                alt_query=v

    alt_query_idx = alt_index
    for i,xx in enumerate(alt_list):
        if xx==alt_query:
            alt_query_idx = i

    return list_depths, idxs_depths, alt_list, alt_index_abs, alt_query_idx

def prep_sim_data_plot(site,
                       path_ground, path_snow, path_swe, path_thaw_depth, path_pickle,
                       year_bkg_end, year_trans_end, no_weight, query):
    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    # assign a subjective weight to all simulations
    pd_weight, _ = assign_weight_sim(site, path_pickle, no_weight, query)

    f_ground, time_ground, temp_ground = open_ground_nc(path_ground)
    _, snow_height = open_snow_nc(path_snow)
    _, swe = open_swe_nc(path_swe)
    _, thaw_depth = open_thaw_depth_nc(path_thaw_depth)

    list_depths = list(f_ground['soil_depth'][:])
    idxs_depths = {depth: list_depths.index(depth) if depth in list_depths else -1 for depth in [1,5,10]}
    idxs_depths = {k: v for k,v in idxs_depths.items() if v!=-1}

    _, time_bkg_ground, time_trans_ground, _ = list_tokens_year(time_ground, year_bkg_end, year_trans_end)

    list_sims = list(pd_weight.index.values)

    # weighted mean GST
    temp_ground_mean = list(np.average([temp_ground[i,:,0] for i in list_sims], axis=0, weights=pd_weight.loc[:, 'weight']))
    temp_ground_mean_deep = {k: list(np.average([temp_ground[i,:,v] for i in list_sims], axis=0, weights=pd_weight.loc[:, 'weight'])) for k,v in idxs_depths.items()}

    if query is not None:
        temp_ground = temp_ground[list_sims,:,0]
        snow_height = snow_height[list_sims,:]
        swe = swe[list_sims,:]
        thaw_depth = thaw_depth[list_sims,:]

    return time_ground, time_bkg_ground, time_trans_ground, temp_ground, temp_ground_mean, temp_ground_mean_deep, snow_height, swe, thaw_depth, list_sims


def prep_atmos_data_plot(site,
                         path_forcing_list, path_ground, path_swe, path_pickle,
                         year_bkg_end, year_trans_end, no_weight, query, alt_query_idx):
    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    list_vars = ['time_air', 'temp_air', 'SW_flux', 'SW_direct_flux', 'SW_diffuse_flux', 'precipitation']
    list_series = [open_air_nc(i) for i in path_forcing_list]
    list_series_b = [[list_series[j][i] for j in range(len(list_series))] for i in range(len(list_series[0]))]
    air_all_dict = dict(zip(list_vars, list_series_b))

    time_air_all = air_all_dict['time_air']
    temp_air_all = air_all_dict['temp_air']
    precipitation_all = air_all_dict['precipitation']

    # Mean air temperature over all reanalyses and altitudes
    mean_air_temp = mean_all_reanalyses(time_air_all,
                                        [mean_all_altitudes(i, site, path_pickle, no_weight, query, alt_query_idx) for i in temp_air_all],
                                        year_bkg_end, year_trans_end)
    # mean_air_temp = mean_all_reanalyses(time_air_all, [i[:,alt_index] for i in temp_air_all], year_bkg_end, year_trans_end)

    # finally we get the total water production, averaged over all reanalyses
    tot_water_prod, _, mean_prec = assign_tot_water_prod(path_forcing_list, path_ground, path_swe, path_pickle, year_bkg_end, year_trans_end, site, no_weight, query, alt_query_idx)

    if query is not None:
        temp_air_all = [i[:,alt_query_idx].reshape(-1, len(i)).T for i in temp_air_all]
        precipitation_all = [i[:,alt_query_idx].reshape(-1, len(i)).T for i in precipitation_all]

    return time_air_all, temp_air_all, precipitation_all, mean_air_temp, tot_water_prod, mean_prec
