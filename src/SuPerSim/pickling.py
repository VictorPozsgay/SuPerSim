"""This module defines the functions that create statistics for the timseries and pickle them"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd

from SuPerSim.open import open_air_nc, open_ground_nc, open_snow_nc
from SuPerSim.mytime import list_tokens_year, specific_time_to_index

def assign_value_global_dict(path_forcing_list, path_ground, path_snow, path_pickle, year_bkg_end, year_trans_end, site):
    """ Function returns a dictionary containing all the important timeseries and saves it to a pickle
    
    Parameters
    ----------
    path_forcing_list : list of str
        List of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_snow : str
        Path to the .nc file where the aggregated snow simulations are stored
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles   


    Returns
    -------
    global_dict : dict
        A large dictionary listing all timeseries, organized in: air/ground/snow > relevant timeseries
        For the 'air', all timeseries are a list with each entry corresponding to a different reanalysis

    """

    file_name = f"global_dict{('' if site=='' else '_')}{site}.pkl"
    my_path = path_pickle + file_name

    # try to open the pickle file, if it exists
    try:
        # Open the file in binary mode
        with open(my_path, 'rb') as file:
            # Call load method to deserialze
            global_dict = pickle.load(file)
        print('Succesfully opened the pre-existing pickle:', file_name)

    # if the pickle file does not exist, we have to create it
    except (OSError, IOError) as e:
        list_vars = ['time_air', 'temp_air', 'SW_flux', 'SW_direct_flux', 'SW_diffuse_flux', 'precipitation',
                     'time_air_bkg', 'time_air_trans', 'time_air_pre_trans']

        list_series = [open_air_nc(i) + list_tokens_year(open_air_nc(i)[0], year_bkg_end, year_trans_end)[1:] for i in path_forcing_list]
        list_series_b = [[list_series[j][i] for j in range(len(list_series))] for i in range(len(list_series[0]))]

        global_dict = {'air': dict(zip(list_vars, list_series_b)),
                    'ground': dict(zip(['time_ground', 'temp_ground'], open_ground_nc(path_ground)[1:])),
                    'snow': dict(zip(['snow_height'], open_snow_nc(path_snow)[1:]))}
        
        # Open a file and use dump()
        with open(my_path, 'wb') as file:
            # A new file will be created
            pickle.dump(global_dict, file)
        print('Created a new pickle:', file_name)

        # useless line just to use the variable 'e' so that I don't get an error
        if e == 0:
            pass

    return global_dict

# def assign_value_df_raw(path_metadata):
#     """ Function converts the .csv ensemble simulation repository into a panda dataframe with a column for each parameter
    
#     Parameters
#     ----------
#     path_metadata : str
#         Path to the .csv file with all the simulation parameters

#     Returns
#     -------
#     df_raw : pandas.core.frame.DataFrame
#         A panda dataframe version of the .csv file where the simulation paramaters have been unpacked into readable columns

#     """

#     df_raw = pd.read_csv(path_metadata, usecols=['site','directory','parameters'])
#     df_raw['altitude'] = (df_raw['site'].str.split('_').str[1]).apply(pd.to_numeric)
#     df_raw['site_name'] = df_raw['site'].str.split('_').str[0]
#     df_raw['forcing'] = df_raw['directory'].str.split('_').str[3]
#     df_raw['aspect'] = [pd.to_numeric(i.replace('p','.')) for i in (df_raw['parameters'].str.split('aspect_').str[1]).str.split('.inpts').str[0]]
#     df_raw['slope'] = ((df_raw['parameters'].str.split('slope_').str[1]).str.split('.inpts').str[0]).apply(pd.to_numeric)
#     df_raw['snow'] = [(y/100 if y > 10 else y) for y in [pd.to_numeric(i.replace('p','.')) for i in ((df_raw['parameters'].str.split('snow_').str[1]).str.split('.inpts').str[0])]]
#     if len(df_raw['parameters'].str.split('pointmaxswe_')[0]) == 2:
#         df_raw['maxswe'] = (df_raw['parameters'].str.split('pointmaxswe_').str[1]).str.split('.inpts').str[0]
#     else:
#         df_raw['maxswe'] = [np.nan for i in df_raw['altitude']]
#     df_raw['material'] = (df_raw['parameters'].str.split('soil_').str[1]).str.split('.inpts').str[0]
#     df_raw.drop('parameters', axis=1, inplace=True)

#     return df_raw

def assign_value_df_raw(path_metadata):
    """ Function converts the .csv ensemble simulation repository into a panda dataframe with a column for each parameter
    
    Parameters
    ----------
    path_metadata : str
        Path to the .csv file with all the simulation parameters

    Returns
    -------
    df_raw : pandas.core.frame.DataFrame
        A panda dataframe version of the .csv file where the simulation paramaters have been unpacked into readable columns

    """

    df_raw = pd.read_csv(path_metadata, usecols=['directory', 'site_name', 'altitude', 'forcing_name', 'aspect', 'slope', 'scf', 'swe', 'soil'])
    df_raw = df_raw.rename(columns={"forcing_name": "forcing", "scf": "snow", "swe": "maxswe", "soil": "material"})

    return df_raw

def assign_value_df(path_metadata, path_ground, path_pickle, site):
    """ Function returns the panda dataframe with all ensemble simulation parameters and saves it to a pickle
    
    Parameters
    ----------
    path_metadata : str
        Path to the .csv file with all the simulation parameters
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles   


    Returns
    -------
    df : pandas.core.frame.DataFrame
        A panda dataframe version of the .csv file where the simulation paramaters have been unpacked into readable columns,
        saved to a pickle file
        The 'directory' column is ordered by the simulation index provided in 'path_ground'

    """
    # INPUT: Should ONLY take var_name = df
    # OUPUT: value of df whether the variable was already assigned or not.
    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"df{('' if site=='' else '_')}{site}.pkl"
    my_path = path_pickle + file_name

    f_ground, _, _ = open_ground_nc(path_ground)

    # try to open the pickle file, if it exists
    try:
        # Open the file in binary mode
        with open(my_path, 'rb') as file:
            # Call load method to deserialze
            df = pickle.load(file) 
        print('Succesfully opened the pre-existing pickle:', file_name)

    # if the pickle file does not exist, we have to create it
    except (OSError, IOError) as e:
        df_raw = assign_value_df_raw(path_metadata)
            
        # this is a method that re-orders the 'directory' column thanks to the simulation index of f_ground
        df_raw.directory = df_raw.directory.astype("category")
        df_raw.directory = df_raw.directory.cat.set_categories(f_ground['simulation'][:])
        # now the order of the csv file corresponds to the order of the ground
        df = df_raw.sort_values(['directory']).reset_index(drop=True)

        if all(f_ground['simulation'][:,:] == df.iloc[:].directory):
            print('All good, same order for all lists!')
        else:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
        # Open a file and use dump() 
        with open(my_path, 'wb') as file: 
            # A new file will be created 
            pickle.dump(df, file)
        print('Created a new pickle:', file_name)

        # useless line just to use the variable 'e' so that I don't get an error
        if e == 0:
            pass

    return df

def assign_value_reanalysis_stat(forcing_list, path_forcing_list, path_pickle, year_bkg_end, year_trans_end, site):
    """ Creates a dictionary of mean quantities over the background and transient periods
    
    Parameters
    ----------
    forcing_list : list of str
        List of forcings provided, with a number of entries between 1 and 3 in 'era5', 'merra2', and 'jra55'. E.g. ['era5', 'merra2']
    path_forcing_list : list of str
        List of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    site : str
        Location of the event, e.g. 'Aksaut_Ridge' (not a list, 1 single location)    

    Returns
    -------
    reanalysis_stats : dict
        dictionary of mean quntities over the background and transient periods

    """

    file_name = f"reanalysis_stats{('' if site=='' else '_')}{site}.pkl"
    file_name_df = f"df{('' if site=='' else '_')}{site}.pkl"
    my_path = path_pickle + file_name
    my_path_df = path_pickle + file_name_df

    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    # try to open the pickle file, if it exists
    try: 
        # Open the file in binary mode 
        with open(my_path, 'rb') as file: 
            # Call load method to deserialize 
            var_name = pickle.load(file) 
        print('Succesfully opened the pre-existing pickle:', file_name)

    # if the pickle file does not exist, we have to create it
    except (OSError, IOError) as e:
        # difference (trend) between trans and bkg periods for the air temperature:
        # amount of warming, at different altitudes

        temp_air, SW_flux, SW_direct_flux, SW_diffuse_flux, time_bkg_air, time_trans_air, time_pre_trans_air = [[] for _ in range(7)]

        for i in range(len(forcing_list)):
            time_air, var1, var2, var3, var4, var5, var6, var7 = [[] for _ in range(8)]
            time_air, var1, var2, var3, var4, _ = open_air_nc(path_forcing_list[i])
            temp_air.append(var1)
            SW_flux.append(var2)
            SW_direct_flux.append(var3)
            SW_diffuse_flux.append(var4)
            _, var5, var6, var7 = list_tokens_year(time_air, year_bkg_end, year_trans_end)
            time_bkg_air.append(var5)
            time_trans_air.append(var6)
            time_pre_trans_air.append(var7)

        # Open the file in binary mode
        with open(my_path_df, 'rb') as file:
            # Call load method to deserialze
            df = pickle.load(file) 

        # list of sorted altitudes at the site studied
        list_alt = sorted(list(set(df.altitude)))

        var_name = {forcing_list[i]: {list_alt[alt]:
                                {'temp_bkg': temp_air[i][time_bkg_air[i], alt].mean(),
                                    # background air temperature at different elevations
                                    'temp_trans': temp_air[i][time_trans_air[i], alt].mean(),
                                    # transient air temperature at different elevations
                                    'air_warming': temp_air[i][time_trans_air[i], alt].mean()
                                    - temp_air[i][time_bkg_air[i], alt].mean(),
                                    # air warming at different elevations
                                    'SW': SW_flux[i][time_pre_trans_air[i], alt].mean(),
                                    # total flux mean over the whole study, at different elevations
                                    'SW_direct': SW_direct_flux[i][time_pre_trans_air[i], alt].mean(),
                                    # total direct flux mean over the whole study, at different elevations
                                    'SW_diffuse': SW_diffuse_flux[i][time_pre_trans_air[i], alt].mean()
                                    # total diffuse flux mean over the whole study, at different elevations
                                }
                                for alt in range(temp_air[i].shape[1])}
                    for i in range(len(forcing_list))}
        
        # Open a file and use dump() 
        with open(my_path, 'wb') as file: 
            # A new file will be created 
            pickle.dump(var_name, file)
        print('Created a new pickle:', file_name)

        # useless line just to use the variable 'e' so that I don't get an error
        if e == 0:
            pass

    return var_name

def glacier_filter(site, path_snow, path_pickle, glacier=False, min_glacier_depth=100, max_glacier_depth=20000):
    """ Function returns a list of valid simulations regarding the glacier criteria
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Aksaut_Ridge'
    path_snow : str
        Path to the .nc file where the aggregated snow simulations are stored
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    glacier : bool, optional
        By default only keeps non-glacier simulations but can be changed to True to select only glaciated simulations
    min_glacier_depth : float, optional
        Selects simulation with minimum snow height higher than this threshold (in mm)
    max_glacier_depth : float, optional
        Selects simulation with minimum snow height lower than this threshold (in mm)
    

    Returns
    -------
    list_valid_sim : list
        list of simulation number for all valid simulations
    """
    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    _, snow_height = open_snow_nc(path_snow)

    file_name = f"list_valid_sim{('' if site=='' else '_')}{site}.pkl"
    my_path = path_pickle + file_name

    # try to open the pickle file, if it exists
    try: 
        # Open the file in binary mode 
        with open(my_path, 'rb') as file: 
            # Call load method to deserialize 
            list_valid_sim = pickle.load(file) 
        print('Succesfully opened the pre-existing pickle:', file_name)

    # if the pickle file does not exist, we have to create it
    except (OSError, IOError) as e:
        # we create a dictionary of all valid simulations
        list_valid_sim = []
        for sim_index in range(snow_height.shape[0]):
            min_snow_height = np.min(snow_height[sim_index,:])
            if glacier:
                if (min_snow_height >= min_glacier_depth) & (min_snow_height < max_glacier_depth):
                    list_valid_sim.append(sim_index)
                else:
                    pass
            else:
                if min_snow_height < min_glacier_depth:
                    list_valid_sim.append(sim_index)
                else:
                    pass
        
        # Open a file and use dump() 
        with open(my_path, 'wb') as file:
            # A new file will be created 
            pickle.dump(list_valid_sim, file)
        print('Created a new pickle:', file_name)

        # useless line just to use the variable 'e' so that I don't get an error
        if e == 0:
            pass

    return list_valid_sim

def melt_out_date(consecutive, path_ground, path_snow, path_pickle, year_bkg_end, year_trans_end, site):
    """ Function returns a list of melt out dates given the criterion of a number of consecutive snow-free days
    
    Parameters
    ----------
    consecutive : int
        Number of minimum consecutive snow-free days to declare seasonal melt out of the snow
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_snow : str
        Path to the .nc file where the aggregated snow simulations are stored
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    

    Returns
    -------
    dict_melt_out : dict
        Dictionary that assings a melt-out date to each simulation and each year
        Note that if snow does NOT melt at all, we assign the maximal value corresponding to the end of the year 
    stats_melt_out_dic : dict
        Background, transient and full mean of the melt-out date for each simulation
    """

    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle. 

    file_name = f"melt_out{('' if site=='' else '_')}{site}.pkl"
    file_name_list_valid_sim = f"list_valid_sim{('' if site=='' else '_')}{site}.pkl"
    my_path = path_pickle + file_name

    snow_height = open_snow_nc(path_snow)[1]
    time_ground = open_ground_nc(path_ground)[1]
    with open(path_pickle + file_name_list_valid_sim, 'rb') as file: 
        # Call load method to deserialize 
        list_valid_sim = pickle.load(file)

    # try to open the pickle file, if it exists
    try: 
        # Open the file in binary mode 
        with open(my_path, 'rb') as file: 
            # Call load method to deserialze 
            dict_melt_out, stats_melt_out_dic = pickle.load(file) 
        print('Succesfully opened the pre-existing pickle:', file_name)

    # if the pickle file does not exist, we have to create it
    except (OSError, IOError) as e:
        # we start by importing the list of time indexes separated by year
        dictionary = list_tokens_year(time_ground, year_bkg_end, year_trans_end)[0]

        # we create the dictionary of melt-out dates
        dict_melt_out = defaultdict(dict)
        for sim in list_valid_sim:
            for year in list(dictionary.keys()):
                # we extract the time series indices corresponding to the year selected
                time_particular_year = dictionary[year]

                # we get a list of snow free day
                test_list = [i for i, x in enumerate(snow_height[sim, time_particular_year]) if x == 0]
                # this counter goes through test_list and checks that there is a number of consecutive days without snow
                counter = 0
                # this will be the final result and is initialized to 0
                result = 0
                # we start by making sure that the list of snow-free days is at least as long as the number of consecutive
                # days we are after, if not we conclude that there is snow all year round
                if len(test_list) >= counter + consecutive + 1:   
                    if test_list[counter+consecutive] == test_list[counter]+consecutive:
                        result = test_list[counter]
                    # keep going through the loop while we don't have the number of snow-free consecutive days
                    while test_list[counter+consecutive] != test_list[counter]+consecutive:
                        counter += 1
                        result = test_list[counter]
                        if len(test_list) < counter + consecutive + 1:
                            result = len(snow_height[sim, time_particular_year])
                            break
                else:
                    result = len(snow_height[sim, time_particular_year])

                dict_melt_out[sim][year] = result
            
        stats_melt_out_dic = {}
        for sim_index in list(dict_melt_out.keys()):
            stats_melt_out_dic[sim_index] = {'bkg_mean': np.mean([dict_melt_out[sim_index][k] for k in list(dict_melt_out[sim_index].keys()) if k < year_bkg_end]),
                                             'trans_mean': np.mean([dict_melt_out[sim_index][k] for k in list(dict_melt_out[sim_index].keys()) if k >= year_bkg_end]),
                                             'full_mean': np.mean(list(dict_melt_out[sim_index].values())),
                                             'full_std': np.std(list(dict_melt_out[sim_index].values()))}
        
        # Open a file and use dump() 
        with open(my_path, 'wb') as file:
            # A new file will be created 
            pickle.dump([dict_melt_out, stats_melt_out_dic], file)
        print('Created a new pickle:', file_name)

        # useless line just to use the variable 'e' so that I don't get an error
        if e == 0:
            pass

    return dict_melt_out, stats_melt_out_dic

def assign_value_df_stats(path_ground, path_snow, path_pickle, year_bkg_end, year_trans_end, site):
    """ Function returns a large panda dataframe with information about the air, ground, snow, topography, etc. for all simulations
    
    Parameters
    ----------
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_snow : str
        Path to the .nc file where the aggregated snow simulations are stored
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    

    Returns
    -------
    df_stats : pandas.core.frame.DataFrame
        Large panda dataframe with information about the air, ground, snow, topography, etc. for all simulations
    """

    file_name = f"df_stats{('' if site=='' else '_')}{site}.pkl"
    file_name_df = f"df{('' if site=='' else '_')}{site}.pkl"
    file_name_reanalysis_stats = f"reanalysis_stats{('' if site=='' else '_')}{site}.pkl"
    file_name_list_valid_sim = f"list_valid_sim{('' if site=='' else '_')}{site}.pkl"
    file_name_melt_out = f"melt_out{('' if site=='' else '_')}{site}.pkl"
    my_path = path_pickle + file_name

    snow_height = open_snow_nc(path_snow)[1]
    _, time_ground, temp_ground = open_ground_nc(path_ground)
    _, time_bkg_ground, time_trans_ground, time_pre_trans_ground = list_tokens_year(time_ground, year_bkg_end, year_trans_end)

    with open(path_pickle + file_name_df, 'rb') as file: 
        # Call load method to deserialize 
        df = pickle.load(file)

    with open(path_pickle + file_name_reanalysis_stats, 'rb') as file: 
        # Call load method to deserialize 
        reanalysis_stats = pickle.load(file)

    with open(path_pickle + file_name_list_valid_sim, 'rb') as file: 
        # Call load method to deserialize 
        list_valid_sim = pickle.load(file)

    with open(path_pickle + file_name_melt_out, 'rb') as file: 
        # Call load method to deserialize 
        _, stats_melt_out_dic = pickle.load(file)

    # try to open the pickle file, if it exists
    try: 
        # Open the file in binary mode 
        with open(my_path, 'rb') as file: 
            # Call load method to deserialze 
            var_name = pickle.load(file) 
        print('Succesfully opened the pre-existing pickle:', file_name)

    # if the pickle file does not exist, we have to create it
    except (OSError, IOError) as e:
        # this dictionary will tell you pretty much everything you need to know for each simulation
        dict_stats = {}
        len_tot = list(time_pre_trans_ground).count(True)
        len_bkg = list(time_bkg_ground).count(True)
        len_trans = list(time_trans_ground).count(True)

        for sim_index in list_valid_sim:
            dict_stats[sim_index] = [df.iloc[sim_index].directory,
                                    # hash code of the simulation, e.g. 'gt_joffre_2000_merra2_ba887ec'
                                    temp_ground[sim_index, time_bkg_ground,0].mean(),
                                    # background ground temperature for that particular simulation
                                    temp_ground[sim_index, time_trans_ground,0].mean(),
                                    # transient ground temperature for that particular simulation
                                    temp_ground[sim_index, time_trans_ground,0].mean()
                                    - temp_ground[sim_index, time_bkg_ground,0].mean(),
                                    # ground temperature evolution between background and transient for that particular simulation
                                    reanalysis_stats[df.loc[sim_index].forcing][df.loc[sim_index].altitude]['temp_bkg'],
                                    # background air temperature for that particular simulation
                                    reanalysis_stats[df.loc[sim_index].forcing][df.loc[sim_index].altitude]['temp_trans'],
                                    # transient air temperature for that particular simulation
                                    reanalysis_stats[df.loc[sim_index].forcing][df.loc[sim_index].altitude]['air_warming'],
                                    # air temperature evolution between background and transient for that particular simulation
                                    temp_ground[sim_index, time_bkg_ground,0].mean()
                                    - reanalysis_stats[df.iloc[sim_index].forcing][df.iloc[sim_index].altitude]['temp_bkg'],
                                    # background surface offset (SO) for that particular simulation
                                    temp_ground[sim_index, time_trans_ground,0].mean()
                                    - reanalysis_stats[df.iloc[sim_index].forcing][df.iloc[sim_index].altitude]['temp_trans'],
                                    # transient surface offset (SO) for that particular simulation
                                    temp_ground[sim_index, time_trans_ground,0].mean()
                                    - temp_ground[sim_index, time_bkg_ground,0].mean()
                                    - reanalysis_stats[df.iloc[sim_index].forcing][df.iloc[sim_index].altitude]['air_warming'],
                                    # differential warming for that particular simulation
                                    temp_ground[sim_index, time_pre_trans_ground, 0].std(),
                                    # standard deviation of the temperature for that particular simulation
                                    np.min(snow_height[sim_index, time_pre_trans_ground]),
                                    # minimum snow depth
                                    np.max(snow_height[sim_index, time_pre_trans_ground]),
                                    # maximum snow depth 
                                    snow_height[sim_index, time_pre_trans_ground].mean(),
                                    # average snow depth
                                    snow_height[sim_index, time_bkg_ground].mean(),
                                    # average snow depth for the background
                                    snow_height[sim_index, time_trans_ground].mean(),
                                    # average snow depth for the transient
                                    stats_melt_out_dic[sim_index]['full_mean'],
                                    # average snow melt out date
                                    stats_melt_out_dic[sim_index]['full_std'],
                                    # standard deviation of the snow melt out date
                                    stats_melt_out_dic[sim_index]['bkg_mean'],
                                    # average snow melt out date for the background
                                    stats_melt_out_dic[sim_index]['trans_mean'],
                                    # average snow melt out date for the transient
                                    np.count_nonzero(snow_height[sim_index, time_pre_trans_ground])/len_tot,
                                    # fraction of days with snow
                                    np.count_nonzero(snow_height[sim_index, time_bkg_ground])/len_bkg,
                                    # fraction of days with snow for the background
                                    np.count_nonzero(snow_height[sim_index, time_trans_ground])/len_trans,
                                    # fraction of days with snow for the transient
                                    reanalysis_stats[df.loc[sim_index].forcing][df.loc[sim_index].altitude]['SW'],
                                    # mean SW for that particular simulation
                                    reanalysis_stats[df.loc[sim_index].forcing][df.loc[sim_index].altitude]['SW_direct'],
                                    # mean direct SW for that particular simulation
                                    reanalysis_stats[df.loc[sim_index].forcing][df.loc[sim_index].altitude]['SW_diffuse'],
                                    # mean diffuse SW for that particular simulation
                                    df.iloc[sim_index].altitude, # altitude of the simulation
                                    df.iloc[sim_index].aspect, # aspect of the simulation
                                    df.iloc[sim_index].slope, # slope of the simulation
                                    df.iloc[sim_index].snow, # snow correction factor of the simulation
                                    df.iloc[sim_index].maxswe, # maximum snow water equivalent of the simulation
                                    df.iloc[sim_index].material, # ground material of the simulation, e.g. 'rock'
                                    df.iloc[sim_index].forcing, # data reanalysis forcing of the simulation, e.g. 'era5'
                                    df.iloc[sim_index].site_name, # site name of the simulation
                                    ]
        
        var_name = pd.DataFrame.from_dict(dict_stats,
                                            orient='index',
                                            columns=['directory', 'bkg_grd_temp', 'trans_grd_temp', 'evol_grd_temp',
                                                    'bkg_air_temp', 'trans_air_temp', 'evol_air_temp',
                                                    'bkg_SO', 'trans_SO', 'diff_warming', 'std_temp',
                                                    'min_snow', 'max_snow', 'mean_snow', 'mean_snow_bkg', 'mean_snow_trans',
                                                    'melt_out_mean', 'melt_out_std', 'melt_out_bkg', 'melt_out_trans',
                                                    'frac_snow', 'frac_snow_bkg', 'frac_snow_trans',
                                                    'mean_SW','mean_SW_direct','mean_SW_diffuse', 'altitude', 'aspect',
                                                    'slope', 'snow_corr', 'max_swe', 'material', 'forcing', 'site_name'])
        
        # Open a file and use dump() 
        with open(my_path, 'wb') as file:
            # A new file will be created 
            pickle.dump(var_name, file)
        print('Created a new pickle:', file_name)

        # useless line just to use the variable 'e' so that I don't get an error
        if e == 0:
            pass

    return var_name

def assign_rockfall_values(site, path_pickle, path_ground, path_forcing_list, date_event, topo_event):
    """ Function returns a dictionary with date and topography information for the event
    
    Parameters
    ----------
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    path_forcing_list : list of str
        List of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
    date_event : list
        Date of the event in the form [yy, mm, dd]. If only partial information are available, [yy] works too.
        If no information, set date_event = []
    topo_event : list
        Topography of the event in the form [aspect, slope, altitude].
        If no information, set topo_event = []
    site : str
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles   

    Returns
    -------
    rockfall_values : dict
        Dictionary with date and topography info for the event
    """

    file_name = f"rockfall_values{('' if site=='' else '_')}{site}.pkl"
    my_path = path_pickle + file_name

    # try to open the pickle file, if it exists
    try:
        # Open the file in binary mode
        with open(my_path, 'rb') as file:
            # Call load method to deserialze
            rockfall_values = pickle.load(file) 
        print('Succesfully opened the pre-existing pickle:', file_name)

    # if the pickle file does not exist, we have to create it
    except (OSError, IOError) as e:
        _, time_ground, _ = open_ground_nc(path_ground)
        time_air_all = [open_air_nc(i)[0] for i in path_forcing_list]

        rockfall_values = {}
        if [isinstance(i, int) for i in date_event] == [True, True, True]:
            rockfall_values['exact_date'] = True
            rockfall_values['year'] = date_event[0] if isinstance(date_event[0], int) else 0
            rockfall_values['datetime'] = datetime(date_event[0], date_event[1], date_event[2], 0, 0, 0, 0)

            time_index = []
        
            for i in [time_ground, *time_air_all]:
                time_index.append(specific_time_to_index(i, rockfall_values['datetime']))

            time_index = list(np.unique(time_index))

            rockfall_values['time_index'] = time_index

        else:
            rockfall_values['exact_date'] = False
            if len(date_event) > 0:
                if isinstance(date_event[0], int):
                    rockfall_values['year'] = date_event[0]
        if [type(i) for i in topo_event] == [int, int, int]:
            rockfall_values['exact_topo'] = True
            rockfall_values['aspect'], rockfall_values['slope'], rockfall_values['altitude'] = topo_event
        else:
            rockfall_values['exact_topo'] = False
        
        # Open a file and use dump() 
        with open(my_path, 'wb') as file: 
            # A new file will be created 
            pickle.dump(rockfall_values, file)
        print('Created a new pickle:', file_name)

        # useless line just to use the variable 'e' so that I don't get an error
        if e == 0:
            pass

    return rockfall_values

def load_all_pickles(site, path_pickle):
    """ Loads all pickles corresponding to the site name
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Aksaut_Ridge'
    path_pickle : str
        String path to the location of the folder where the pickles are saved

    Returns
    -------
    pkl : dict
        Dictionary containing all the pickles with keys:
        {df, reanalysis_stats, list_valid_sim, dict_melt_out, stats_melt_out_dic, df_stats, rockfall_values}
        Each pickle is defined by
        df : pandas.core.frame.DataFrame
            A panda dataframe version of the .csv file where the simulation paramaters have been unpacked into readable columns,
            saved to a pickle file
            The 'directory' column is ordered by the simulation index provided in 'path_ground'
        reanalysis_stats : dict
            dictionary of mean quntities over the background and transient periods
        list_valid_sim : list
            list of simulation number for all valid simulations
        dict_melt_out : dict
            Dictionary that assings a melt-out date to each simulation and each year
            Note that if snow does NOT melt at all, we assign the maximal value corresponding to the end of the year 
        stats_melt_out_dic : dict
            Background, transient and full mean of the melt-out date for each simulation
        df_stats : pandas.core.frame.DataFrame
            Large panda dataframe with information about the air, ground, snow, topography, etc. for all simulations
        rockfall_values : dict
            Dictionary with date and topography info for the event

    """

    list_file_names = [f"df{('' if site=='' else '_')}{site}.pkl",
                       f"reanalysis_stats{('' if site=='' else '_')}{site}.pkl",
                       f"list_valid_sim{('' if site=='' else '_')}{site}.pkl",
                       f"melt_out{('' if site=='' else '_')}{site}.pkl",
                       f"df_stats{('' if site=='' else '_')}{site}.pkl",
                       f"rockfall_values{('' if site=='' else '_')}{site}.pkl"]

    output = [0 for _ in list_file_names]

    for i, file_name in enumerate(list_file_names):
        my_path = path_pickle + file_name
        # Open the file in binary mode 
        with open(my_path, 'rb') as file: 
            # Call load method to deserialze 
            output[i] = pickle.load(file) 

    [df, reanalysis_stats, list_valid_sim, [dict_melt_out, stats_melt_out_dic], df_stats, rockfall_values] = output

    pkl = {'df': df,
           'reanalysis_stats': reanalysis_stats,
           'list_valid_sim': list_valid_sim,
           'dict_melt_out': dict_melt_out,
           'stats_melt_out_dic': stats_melt_out_dic,
           'df_stats': df_stats,
           'rockfall_values': rockfall_values}

    return pkl

def get_all_stats(forcing_list, path_forcing_list, path_metadata, path_ground, path_snow, path_pickle,
                  year_bkg_end, year_trans_end, consecutive,
                  site, date_event, topo_event,
                  glacier=False, min_glacier_depth=100, max_glacier_depth=20000):
    """ Creates a number of pickle files (if they don't exist yet)
    
    Parameters
    ----------
    forcing_list : list of str
        List of forcings provided, with a number of entries between 1 and 3 in 'era5', 'merra2', and 'jra55'. E.g. ['era5', 'merra2']
    path_forcing_list : list of str
        List of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
    path_metadata : str
        Path to the .csv file with all the simulation parameters
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_snow : str
        path to the snow data
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    consecutive : int
        number of consecutive snow-free days to consider that snow has melted for the season
    site : str
        Location of the event, e.g. 'Aksaut_Ridge' (not a list, 1 single location)
    date_event : list
        Date of the event in the form [yy, mm, dd]. If only partial information are available, [yy] works too.
        If no information, set date_event = []
    topo_event : list
        Topography of the event in the form [aspect, slope, altitude].
        If no information, set topo_event = []
    glacier : bool, optional
        By default only keeps non-glacier simulations but can be changed to True to select only glaciated simulations
    min_glacier_depth : float, optional
        Selects simulation with minimum snow height higher than this threshold (in mm)
    max_glacier_depth : float, optional
        Selects simulation with minimum snow height lower than this threshold (in mm)

    Returns
    -------
    pkl : dict
        Dictionary containing all the pickles with keys:
        {df, reanalysis_stats, list_valid_sim, dict_melt_out, stats_melt_out_dic, df_stats, rockfall_values}
        Each pickle is defined by
        df : pandas.core.frame.DataFrame
            A panda dataframe version of the .csv file where the simulation paramaters have been unpacked into readable columns,
            saved to a pickle file
            The 'directory' column is ordered by the simulation index provided in 'path_ground'
        reanalysis_stats : dict
            dictionary of mean quntities over the background and transient periods
        list_valid_sim : list
            list of simulation number for all valid simulations
        dict_melt_out : dict
            Dictionary that assings a melt-out date to each simulation and each year
            Note that if snow does NOT melt at all, we assign the maximal value corresponding to the end of the year 
        stats_melt_out_dic : dict
            Background, transient and full mean of the melt-out date for each simulation
        df_stats : pandas.core.frame.DataFrame
            Large panda dataframe with information about the air, ground, snow, topography, etc. for all simulations
        rockfall_values : dict
            Dictionary with date and topography info for the event

    """

    df = assign_value_df(path_metadata, path_ground, path_pickle, site)
    reanalysis_stats = assign_value_reanalysis_stat(forcing_list, path_forcing_list, path_pickle, year_bkg_end, year_trans_end, site)
    list_valid_sim = glacier_filter(site, path_snow, path_pickle, glacier, min_glacier_depth, max_glacier_depth)
    dict_melt_out, stats_melt_out_dic = melt_out_date(consecutive, path_ground, path_snow, path_pickle, year_bkg_end, year_trans_end, site)
    df_stats = assign_value_df_stats(path_ground, path_snow, path_pickle, year_bkg_end, year_trans_end, site)
    rockfall_values = assign_rockfall_values(site, path_pickle, path_ground, path_forcing_list, date_event, topo_event)

    pkl = {'df': df,
           'reanalysis_stats': reanalysis_stats,
           'list_valid_sim': list_valid_sim,
           'dict_melt_out': dict_melt_out,
           'stats_melt_out_dic': stats_melt_out_dic,
           'df_stats': df_stats,
           'rockfall_values': rockfall_values}

    return pkl
