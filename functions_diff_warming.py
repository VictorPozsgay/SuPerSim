import xarray as xr
import pandas as pd
from netCDF4 import Dataset, num2date
import numpy as np
import numpy.ma as ma
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib import patches, colors
from datetime import datetime, timedelta
from collections import defaultdict
from collections import Counter
import scipy.optimize as opt
import plotly.express as px
import scipy.stats as stats
from scipy.stats import linregress
import re
import pickle 
import time
import warnings
import seaborn as sn
import cmasher as cmr
import mpl_axes_aligner


pickle_path = '/fs/yedoma/home/vpo001/VikScriptsTests/Python_Pickles/'

def time_unit_stamp(time_series):
    # INPUT: name of a particular time series that has attribute .units
    # OUTPUT: exact date and time of the initial datapoint, converted into a datetime

    # this extracts the unit of the time series, i.e. when it starts,
    # e.g. days since 0001-1-1 0:0:0 or seconds since 1900-01-01
    # these are two formats that exist and we have to accomodate for both
    # we partition the string and only keep what comes after 'since ': the date
    time_start_date = time_series.units.partition(' since ')[2]

    # we identify the different delimiters between year, month, day, hour, etc. 
    # to be: '-', ' ', or ':'
    # we get a list in the form [1, 1, 1, 0, 0, 0] or [1900, 1, 1] respectively
    time_start_date_list = list(map(int, map(float, re.split('-| |:', time_start_date))))

    # now we decide to fill the list with 0s if not long enough
    # i.e. if not provided, we assume the start is at exactly midnight 0:0:0
    time_start_date_fill = time_start_date_list[:6] + [0]*(6 - len(time_start_date_list))

    # here we get the string telling us with what frequency data is sampled, e.g. 'days' or 'seconds'
    frequency = time_series.units.partition(' since ')[0]

    # finally, we convert it to a datetime
    return datetime(*time_start_date_fill), frequency

def list_tokens_year(time_file, year_bkg_end=2000, year_trans_end=2020):
    # INPUT: time_file: this will be the time series that we will partition into years
    # OUTPUT: list_years: a list of indices per calendar year that can be used to study data year per year
    #                     in the form of a dictionary like {1980: [0, 1, 2, ..., 363, 364], 1981: [365, ..., 729], ..., 2019: [14244, ..., 14608]}
    #         time_bkg: selects only background data points
    #         time_trans: selects only transient data points
    #         time_pre_trans: selects only background and transient data points

    # we start by creating a short dictionary that allows us to convert the seconds into the unit of the file
    # e.g. 'days' or 'hours'
    # difference in time stamp between two consecutive measurements (could be 1, 3600, etc.)
    consecutive = ma.getdata(time_file[1]-time_file[0])
    # here we get the unit and we express it in seconds
    seconds_divider = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}[time_unit_stamp(time_file)[1]]
    # we are now able to understand how many seconds there are between 2 consecutive measurements
    secs_between_cons = seconds_divider*consecutive

    # we extract the year and month the data starts at
    start_year = num2date(time_file[0], time_file.units).year
    # start_month = num2date(time_file[0], time_file.units).month
    start_day = num2date(time_file[0], time_file.units).day

    # we extract the exact moment the transient era ends
    # time_end_pre_trans = num2date(time_file[time_pre_trans_file][-1], time_file.units)
    time_end = num2date(time_file[-1], time_file.units)

    # we create a dictionary that associates the correct index position to the start of each new year
    # e.g. if the data is taken daily, all values will be spread by ~365
    list_init = [0 for i in range(year_trans_end-start_year+1)]
    for i in range(year_trans_end-start_year+1):
        # note that we allow an arbitrary 10 days wiggle room just to make sure we capture the end date of the last year 
        # of the transient era since technically this last year doesn't go all the way until the new year

        ##################################################################################
        # IT IS VERY IMPORTANT TO HIGHLIGHT THAT THE JRA55 and MERRA2 SCALED DATA DO NOT #
        # SCALE BETWEEN DEC31 6PM AND MIDNIGHT, HENCE LEAVING A 5H GAP IN THE HOURLY     #
        # DATA EVERY YEAR! THIS IS ACCOUNTED FOR BY SUBTRACTING 5 EVERY YEAR BUT ONE HAS #
        # TO BE VERY CAREFUL IF THIS CHANGES SOMEHOW                                     #
        # A SHORT TEST IS IMPLEMENTED AND SHOULD RETURN AN ERROR MESSAGE IF NOT WORKING  #
        ##################################################################################
        if i==0:
            list_init[i] = 0
        else:
            # time difference between yy-01-01 00:00:00 and exactly a year before
            dt = int((datetime(start_year+i,1,1,0,0,0)
                    - datetime(start_year+i-1,1,(start_day if i==1 else 1),0,0,0)).total_seconds()/secs_between_cons)
            # previous entry in the catalogue, in days elapsed from the start
            prev_year = list_init[i-1]
            # previous year + dt should correspond to Jan 1st at 00:00:00 of the current year
            list_init[i] = prev_year + dt
            # this is where we check that there is no gap in the data by making sure current_year
            # indeed corresponds to Jan 1st at 00:00:00 of the current year
            check = int((num2date(time_file[list_init[i]], time_file.units) - datetime(start_year+i,1,1,0,0,0)).total_seconds())
            # finally, we assign the time in days from the start for this month if the check is correct.
            # however, if the check is not, this means that we are in the situation where there is 5 data
            # point missing between 6pm and midnight on December 31st for either merra2 or jra55
            list_init[i] += (0 if check ==0 else -5)

    last = int((time_end - num2date(time_file[0], time_file.units)).total_seconds()/secs_between_cons)

    # finally, we have a dictionary that associates each year in the pre-transient era to a list of
    # indices in the time series corresponding to that particular year
    list_years = {i+start_year: list(range(list_init[i], (list_init[i+1] if i+1+start_year<=year_trans_end else last+1)))
                for i in range(len(list_init))}

    # this function selects all data before the background cutoff, hence selects the background data
    time_bkg = np.less(time_file[:], time_file[list_years[year_bkg_end][0]])
    # this function selects all data after the background cutoff, but before the transient one, hence selects the transient data
    time_trans = np.logical_and(time_file[:] >= time_file[list_years[year_bkg_end][0]], time_file[:] < time_file[list_years[year_trans_end][0]])
    # this function selects all data before the transient cutoff, hence selects the pre-transient data
    time_pre_trans = np.less(time_file[:], time_file[list_years[year_trans_end][0]])


    # It is here that we make sure that the data that should correspond to Jan 1, 0:00:00 of the last year
    # actually does, otherwise it raises an error and stops the evaluation
    diff = int((num2date(time_file[list_years[year_trans_end][0]], time_file.units)-
                datetime(year_trans_end,1,1,0,0,0)).total_seconds())
    if diff!=0:
        raise ValueError('VictorCustomError: scaled JRA55 and MERRA2 have unidentified gaps in data, check script')

    return list_years, time_bkg, time_trans, time_pre_trans

def list_tokens_year_month(time_file, time_pre_trans_file):
    # INPUT: time_file: this will be the time series that we will partition into years
    #        time_pre_trans_file: furthermore, we only use time before the transient cutoff,
    #                             just in case there would be 'bad' values after that
    # OUTPUT: a list of indices per calendar year that can be used to study data year per year
    #         in the form of a dictionary like {1980: [0, 1, 2, ..., 363, 364], 1981: [365, ..., 729], ..., 2019: [14244, ..., 14608]}

    # we start by creating a short dictionary that allows us to convert the seconds into the unit of the file
    # e.g. 'days' or 'hours'
    # difference in time stamp between two consecutive measurements (could be 1, 3600, etc.)
    consecutive = ma.getdata(time_file[1]-time_file[0])
    # here we get the unit and we express it in seconds
    seconds_divider = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}[time_unit_stamp(time_file)[1]]
    # we are now able to understand how many seconds there are between 2 consecutive measurements
    secs_between_cons = seconds_divider*consecutive

    # we extract the year and month the data starts at
    start_year = num2date(time_file[0], time_file.units).year
    # start_month = num2date(time_file[0], time_file.units).month
    start_day = num2date(time_file[0], time_file.units).day

    # we extract the exact moment the transient era ends
    time_end_pre_trans = num2date(time_file[time_pre_trans_file][-1], time_file.units)
    end_year = time_end_pre_trans.year

    # we create a dictionary that associates the correct index position to the start of each new year
    # e.g. if the data is taken daily, all values will be spread by ~365
    list_init = [[0] * 12 for i in range(end_year-start_year+1)]
    for i in range(end_year-start_year+1):
        # note that we allow an arbitrary 10 days wiggle room just to make sure we capture the end date of the last year 
        # of the transient era since technically this last year doesn't go all the way until the new year

        ##################################################################################
        # IT IS VERY IMPORTANT TO HIGHLIGHT THAT THE JRA55 and MERRA2 SCALED DATA DO NOT #
        # SCALE BETWEEN DEC31 6PM AND MIDNIGHT, HENCE LEAVING A 5H GAP IN THE HOURLY     #
        # DATA EVERY YEAR! THIS IS ACCOUNTED FOR BY SUBTRACTING 5 EVERY YEAR BUT ONE HAS #
        # TO BE VERY CAREFUL IF THIS CHANGES SOMEHOW                                     #
        # A SHORT TEST IS IMPLEMENTED AND SHOULD RETURN AN ERROR MESSAGE IF NOT WORKING  #
        ##################################################################################
        for j in range(12):
            if ((i==0)&(j==0)):
                list_init[i][j] = 0
            else:
                # time difference between two yy-mm-01 00:00:00 and exactly a month before
                dt = int((datetime(start_year+i,j+1,1,0,0,0)
                          - (datetime(start_year+i,j,(start_day if ((i==0)&(j==1)) else 1),0,0,0) if j>0 else datetime(start_year+i-1,12,1,0,0,0))).total_seconds()/secs_between_cons)
                # previous entry in the catalogue, in days elapsed from the start
                prev_month = (list_init[i][j-1] if j>0 else list_init[i-1][11])
                # previous year + dt should correspond to Jan 1st at 00:00:00 of the current year
                list_init[i][j] = prev_month + dt
                # this is where we check that there is no gap in the data by making sure current_month
                # indeed corresponds to the 1st at 00:00:00 of the current month
                check = int((num2date(time_file[list_init[i][j]], time_file.units) - datetime(start_year+i,j+1,1,0,0,0)).total_seconds())
                # finally, we assign the time in days from the start for this month if the check is correct.
                # however, if the check is not, this means that we are in the situation where there is 5 data
                # point missing between 6pm and midnight on December 31st for either merra2 or jra55
                list_init[i][j] += (0 if check ==0 else -5)

    last = int((time_end_pre_trans - num2date(time_file[0], time_file.units)).total_seconds()/secs_between_cons)

    # finally, we have a dictionary that associates each year in the pre-transient era to a list of
    # indices in the time series corresponding to that particular year
    list_years_months = {i+start_year: {j: list(range(list_init[i][j-1], (list_init[i][j] if j<12 else (list_init[i+1][0] if i+1+start_year<=end_year else last+1)), 1))
                                           for j in range(1, 13)}
                                       for i in range(len(list_init))}

    # It is here that we make sure that the data that should correspond to Jan 1, 0:00:00 of the last year
    # actually does, otherwise it raises an error and stops the evaluation
    diff = int((num2date(time_file[time_pre_trans_file][list_years_months[end_year][1][0]], time_file.units)-
                datetime(end_year,1,1,0,0,0)).total_seconds())
    if diff!=0:
        raise ValueError('VictorCustomError: scaled JRA55 and MERRA2 have unidentified gaps in data, check script')

    return list_years_months

def assign_value_df_raw(path_repository):
    # INPUT: path_repository: .csv folder with all the simulations' parameters
    # OUTPUT: df_raw: panda dataframe version of the .csv file where the paramaters have been
    #                 unpacked into readable columns

    x = pd.read_csv(path_repository, usecols=['site','directory','parameters'])
    x['altitude'] = (x['site'].str.split('_').str[1]).apply(pd.to_numeric)
    x['site_name'] = x['site'].str.split('_').str[0]
    x['forcing'] = x['directory'].str.split('_').str[3]
    x['aspect'] = [pd.to_numeric(i.replace('p','.')) for i in (x['parameters'].str.split('aspect_').str[1]).str.split('.inpts').str[0]]
    x['slope'] = ((x['parameters'].str.split('slope_').str[1]).str.split('.inpts').str[0]).apply(pd.to_numeric)
    x['snow'] = [(y/100 if y > 10 else y) for y in [pd.to_numeric(i.replace('p','.')) for i in ((x['parameters'].str.split('snow_').str[1]).str.split('.inpts').str[0])]]
    if len(x['parameters'].str.split('pointmaxswe_')[0]) == 2:
        x['maxswe'] = (x['parameters'].str.split('pointmaxswe_').str[1]).str.split('.inpts').str[0]
    else:
        x['maxswe'] = [np.nan for i in x['altitude']]
    x['material'] = (x['parameters'].str.split('soil_').str[1]).str.split('.inpts').str[0]
    x.drop('parameters', axis=1, inplace=True)

    return x

def assign_value_df(df, df_raw, f_ground, f_snow, extension=''):
    # INPUT: Should ONLY take var_name = df
    # OUPUT: value of df whether the variable was already assigned or not.
    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"df{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"df{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # if the variable has no assigned value yet, we need to assign it
    if df is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode 
            with open(my_path, 'rb') as file: 
                # Call load method to deserialze 
                df = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
        except (OSError, IOError) as e:
            # this is a method that re-orders the 'directory' column thanks to the simulation index of f_ground
            df_raw.directory = df_raw.directory.astype("category")
            df_raw.directory = df_raw.directory.cat.set_categories(f_ground['simulation'][:])
            # now the order of the csv file corresponds to the order of the ground and snow files
            df = df_raw.sort_values(['directory']).reset_index(drop=True)

            if all(f_ground['simulation'][:,:] == f_snow['simulation'][:,:]):
                if all(f_ground['simulation'][:,:] == df.iloc[:].directory):
                    print('All good, same order for all lists!')
                else:
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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

    else:
        print('The variable already existed:', var_name_full)

    return df

def assign_value_reanalysis_stat(var_name, df,
                                 temp_air_era5, temp_air_merra2, temp_air_jra55,
                                 SW_flux_era5, SW_flux_merra2, SW_flux_jra55,
                                 SW_direct_flux_era5, SW_direct_flux_merra2, SW_direct_flux_jra55,
                                 SW_diffuse_flux_era5, SW_diffuse_flux_merra2, SW_diffuse_flux_jra55,
                                 time_bkg_air_era5, time_bkg_air_merra2, time_bkg_air_jra55,
                                 time_trans_air_era5, time_trans_air_merra2, time_trans_air_jra55,
                                 time_pre_trans_air_era5, time_pre_trans_air_merra2, time_pre_trans_air_jra55,
                                 extension=''):
    # INPUT: Should ONLY take var_name = reanalysis_stats
    # OUPUT: value of reanalysis_stats whether the variable was already assigned or not.
    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"reanalysis_stats{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"reanalysis_stats{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # if the variable has no assigned value yet, we need to assign it
    if var_name is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode 
            with open(my_path, 'rb') as file: 
                # Call load method to deserialze 
                var_name = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
        except (OSError, IOError) as e:
            # difference (trend) between trans and bkg periods for the air temperature:
            # amount of warming, at different altitudes

            # list of sorted altitudes at the site studied
            list_alt = sorted(list(set(df.altitude)))

            var_name = {'era5': {list_alt[alt]:
                                  {'temp_bkg': temp_air_era5[time_bkg_air_era5, alt].mean(),
                                   # background air temperature at different elevations
                                   'temp_trans': temp_air_era5[time_trans_air_era5, alt].mean(),
                                   # transient air temperature at different elevations
                                   'air_warming': temp_air_era5[time_trans_air_era5, alt].mean()
                                   - temp_air_era5[time_bkg_air_era5, alt].mean(),
                                   # air warming at different elevations
                                   'SW': SW_flux_era5[time_pre_trans_air_era5, alt].mean(),
                                   # total flux mean over the whole study, at different elevations
                                   'SW_direct': SW_direct_flux_era5[time_pre_trans_air_era5, alt].mean(),
                                   # total direct flux mean over the whole study, at different elevations
                                   'SW_diffuse': SW_diffuse_flux_era5[time_pre_trans_air_era5, alt].mean()
                                   # total diffuse flux mean over the whole study, at different elevations
                                  }
                                  for alt in range(temp_air_era5.shape[1])},
                        'merra2': {list_alt[alt]:
                                    {'temp_bkg': temp_air_merra2[time_bkg_air_merra2, alt].mean(),
                                     # background air temperature at different elevations
                                     'temp_trans': temp_air_merra2[time_trans_air_merra2, alt].mean(),
                                     # transient air temperature at different elevations
                                     'air_warming': temp_air_merra2[time_trans_air_merra2, alt].mean()
                                     - temp_air_merra2[time_bkg_air_merra2, alt].mean(),
                                     # air warming at different elevations
                                     'SW': SW_flux_merra2[time_pre_trans_air_merra2, alt].mean(),
                                     # total flux mean over the whole study, at different elevations
                                     'SW_direct': SW_direct_flux_merra2[time_pre_trans_air_merra2, alt].mean(),
                                     # total direct flux mean over the whole study, at different elevations
                                     'SW_diffuse': SW_diffuse_flux_merra2[time_pre_trans_air_merra2, alt].mean()
                                     # total diffuse flux mean over the whole study, at different elevations
                                    }
                                    for alt in range(temp_air_merra2.shape[1])},
                        'jra55': {list_alt[alt]:
                                    {'temp_bkg': temp_air_jra55[time_bkg_air_jra55, alt].mean(),
                                     # background air temperature at different elevations
                                     'temp_trans': temp_air_jra55[time_trans_air_jra55, alt].mean(),
                                     # transient air temperature at different elevations
                                     'air_warming': temp_air_jra55[time_trans_air_jra55, alt].mean()
                                     - temp_air_jra55[time_bkg_air_jra55, alt].mean(),
                                     # air warming at different elevations
                                     'SW': SW_flux_jra55[time_pre_trans_air_jra55, alt].mean(),
                                     # total flux mean over the whole study, at different elevations
                                     'SW_direct': SW_direct_flux_jra55[time_pre_trans_air_jra55, alt].mean(),
                                     # total direct flux mean over the whole study, at different elevations
                                     'SW_diffuse': SW_diffuse_flux_jra55[time_pre_trans_air_jra55, alt].mean()
                                     # total diffuse flux mean over the whole study, at different elevations
                                    }
                                   for alt in range(temp_air_jra55.shape[1])}
                        }
            
            # Open a file and use dump() 
            with open(my_path, 'wb') as file: 
                # A new file will be created 
                pickle.dump(var_name, file)
            print('Created a new pickle:', file_name)

            # useless line just to use the variable 'e' so that I don't get an error
            if e == 0:
                pass

    else:
        print('The variable already existed:', var_name_full)

    return var_name

def assign_value_df_stat(var_name, temp_ground, snow_height, reanalysis_stats,
                         stats_melt_out_dic, stats_melt_out_dic_consecutive, df, list_valid_sim,
                         time_bkg_ground, time_trans_ground, time_pre_trans_ground,
                         extension=''):
    # INPUT: Should ONLY take var_name = df_stats
    # OUPUT: value of df_stats whether the variable was already assigned or not.
    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"df_stats{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"df_stats{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # if the variable has no assigned value yet, we need to assign it
    if var_name is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode 
            with open(my_path, 'rb') as file: 
                # Call load method to deserialze 
                var_name = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
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
                                        stats_melt_out_dic_consecutive[sim_index]['full_mean'],
                                        # average snow melt out date
                                        stats_melt_out_dic_consecutive[sim_index]['full_std'],
                                        # standard deviation of the snow melt out date
                                        stats_melt_out_dic_consecutive[sim_index]['bkg_mean'],
                                        # average snow melt out date for the background
                                        stats_melt_out_dic_consecutive[sim_index]['trans_mean'],
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
                                                       'melt_out_cons_mean', 'melt_out_cons_std', 'melt_out_cons_bkg', 'melt_out_cons_trans',
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

    else:
        print('The variable already existed:', var_name_full)

    return var_name

def glacier_filter(var_name, temp_ground, snow_height, df, time_pre_trans_ground,
                   extension='', std_temp_glacier=1, excessive_height=1e5):
    # INPUT: var_name: Should ONLY take var_name = list_valid_sim
    #        std_temp_glacier: cutoff to what constitutes a glacier in terms of temperature variation (standard deviation)
    #        excessive_height: we define what we consider is an excessive amount of snow
    # OUTPUT: returns a list of valid simulations, i.e. with no glaciers

    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"list_valid_sim{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"list_valid_sim{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # if the variable has no assigned value yet, we need to assign it
    if var_name is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode 
            with open(my_path, 'rb') as file: 
                # Call load method to deserialze 
                var_name = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
        except (OSError, IOError) as e:
            # we create a dictionary of possible failed simulations
            dict_failed_sim_explicit = {}
            for sim_index in range(temp_ground.shape[0]):
                dict_failed_sim_explicit[sim_index] = [temp_ground[sim_index, time_pre_trans_ground, 0].std() > std_temp_glacier,
                                                       # check if temperature variations are great enough, False = glacier
                                                       np.min(snow_height[sim_index, time_pre_trans_ground]) == 0,
                                                       # checks if the snow depth ever drops to 0 mm, False = glacier
                                                       np.max(snow_height[sim_index, time_pre_trans_ground]) < excessive_height,
                                                       # checks that snow depth doesn't reach ridiculous levels, False = glacier
                                                       (np.max(snow_height[sim_index, time_pre_trans_ground]) == 0) or (df.iloc[sim_index].snow != 0)
                                                       # checks there's no simulated snow when the snow correction factor is set to 0
                                                       ]

            # there are simulations with snow_corr = 1 and never any snow (i.e. max_snow = 0)
            # but they all have slope = 90 so that makes sense

            # list of all valid simulations
            var_name = []
            var_name = [k for k, v in dict_failed_sim_explicit.items() if all(v)]
            
            # Open a file and use dump() 
            with open(my_path, 'wb') as file:
                # A new file will be created 
                pickle.dump(var_name, file)
            print('Created a new pickle:', file_name)

            # useless line just to use the variable 'e' so that I don't get an error
            if e == 0:
                pass

    else:
        print('The variable already existed:', var_name_full)

    return var_name

def warming_vs_any_variable(var_name, df, dict_df, df_stats, list_valid_sim, material_name='rock'):
    # INPUT: var_name: is either of 'altitude', 'aspect', 'slope', and 'snow'
    #        material_name: 'rock' by default, could also be either of '25sand' or '100sand'
    # OUTPUT: list of xvalues (values of var_name), yvalues (diff warming) and error on these.
    #         this is useful for the subsequent function plot_warming_vs_any_variable
    #         that will take care of creating the plot
    
    var_values = dict_df[var_name]

    # this is a list of simulation indexes that correspond to the queried topography
    x = [sorted(list(set(list_valid_sim) &
                     set(list(df.query(f"material == '{material_name}' and {var_name} == {var_values[k]} ").index))))
                for k in range(len(var_values))]

    # for each topographic property we get the mean and standard deviation (over all simulations in x)
    # of the warming difference
    list_stat = []
    # here we convert the nested lists into an array, it's easier to extract data for the plot
    list_stat = np.array([[np.mean(df_stats['diff_warming'][x[k]]), np.std(df_stats['diff_warming'][x[k]])] for k in range(len(var_values))])

    # for each altitude, we plot the mean and standard deviation of the warming from
    # background to transient between the air and the bedrock
    xdata = np.array(var_values)
    ydata = list_stat[:,0]
    ydata_err = list_stat[:,1]
    return [xdata, ydata, ydata_err]

def plot_warming_vs_any_variable(var_name, df, dict_df, df_stats,
                                 list_valid_sim, units_df, material_name='rock'):
    # INPUT: var_name: is either of 'altitude', 'aspect', 'slope', and 'snow'
    #        material_name: 'rock' by default, could also be either of '25sand' or '100sand'
    # OUTPUT: plots the differential warming as a function of var_name in the 
    xdata, ydata, ydata_err = warming_vs_any_variable(var_name, df, dict_df, df_stats,
                                                      list_valid_sim, material_name)

    fig, ax = plt.subplots()
    # for each altitude, we plot the mean and standard deviation of the warming from
    # background to transient between the air and the bedrock
    plt.scatter(xdata, ydata, label='data')
    plt.errorbar(xdata, ydata, yerr=ydata_err, fmt="none", markersize=8, capsize=10)

    plt.title(f"Differential warming at Joffre as a function of {var_name} in {material_name}")
    plt.xlabel(f'{var_name.capitalize()} {units_df[var_name]}')
    plt.ylabel('Temperature [°C]')

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def fit_aspect(df, dict_df, df_stats, list_valid_sim, units_df, material_name):    
    # INPUT: pretty much nothing interesting, no real variable
    # OUTPUT: plot of differential warming vs aspect with a cosine fit

    var_name = 'aspect'
    xdata, ydata, ydata_err = warming_vs_any_variable(var_name, df, dict_df, df_stats,
                                                      list_valid_sim, material_name)

    # This is the function we are trying to fit to the data.
    def func_aspect(x, a, b, c):
        return a + b * np.cos(x * 2 * np.pi / 360 ) + c * np.sin(x * 2 * np.pi / 360) 

    # The actual curve fitting happens here
    optimizedParameters, pcov, _, _, _ = opt.curve_fit(func_aspect, xdata, ydata)
    print('The fitted function is a + b*cos(2pi/360*x) + c*sin(2pi/360*x) with')
    print('[a,b,c] =', optimizedParameters)

    fig, ax = plt.subplots()
    # for each altitude, we plot the mean and standard deviation of the warming from
    # background to transient between the air and the bedrock
    plt.scatter(xdata, ydata, label='data')
    plt.errorbar(xdata, ydata, yerr=ydata_err, fmt="none", markersize=8, capsize=10)

    plt.title(f"Warming difference between air and ground at Joffre as a function of {var_name}")
    plt.xlabel(f'{var_name.capitalize()} {units_df[var_name]}')
    plt.ylabel('Temperature [°C]')

    # Use the optimized parameters to plot the best fit
    x = np.arange(dict_df['aspect'][0],dict_df['aspect'][-1],1)
    plt.plot(x, func_aspect(x, *optimizedParameters), color='red', label="fit")

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def fit_slope(df, dict_df, df_stats, list_valid_sim, units_df, material_name):
    # INPUT: pretty much nothing interesting, no real variable
    # OUTPUT: plot of differential warming vs slope with a cosine fit

    var_name = 'slope'
    xdata, ydata, ydata_err = warming_vs_any_variable(var_name, df, dict_df, df_stats,
                                                      list_valid_sim, material_name)

    # This is the function we are trying to fit to the data.
    def func(x, a, b):
        return a * np.sin(x * 2 * np.pi / 360) + b

    # The actual curve fitting happens here
    optimizedParameters, pcov = opt.curve_fit(func, xdata, ydata)
    print('The fitted function is a*sin(2pi*x/360)+b with')
    print('[a,b] =', optimizedParameters)

    fig, ax = plt.subplots()
    # for each altitude, we plot the mean and standard deviation of the warming from
    # background to transient between the air and the bedrock
    plt.scatter(xdata, ydata, label='data')
    plt.errorbar(xdata, ydata, yerr=ydata_err, fmt="none", markersize=8, capsize=10)

    plt.title(f"Warming difference between air and ground at Joffre as a function of {var_name}")
    plt.xlabel(f'{var_name.capitalize()} {units_df[var_name]}')
    plt.ylabel('Temperature [°C]')

    # Use the optimized parameters to plot the best fit
    x = np.arange(dict_df['slope'][0],dict_df['slope'][-1],(dict_df['slope'][-1]-dict_df['slope'][0])/100)
    plt.plot(x, func(x, *optimizedParameters), color='red', label="fit")

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def plot_warming_vs_snow_depth(df, df_stats, list_valid_sim, material_name='rock'):
    # INPUT: pretty much nothing interesting, no real variable
    # OUTPUT: plot of differential warming vs snow depth

    # this is a list of simulation indexes that correspond to the queried topography
    sim = sorted(list(set(list_valid_sim) &
                      set(list(df.query(f"material == '{material_name}'").index))))
        
    # ensemble of inputs
    xdata = np.array(df_stats['mean_snow'][sim])
    ydata = np.array(df_stats['diff_warming'][sim])

    fig, ax = plt.subplots()
    plt.scatter(xdata, ydata)

    plt.title('Differential warming as a function of average snow depth')
    plt.xlabel('Average snow depth [mm]')
    plt.ylabel('Differential warming [°C]')

    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def dic_monthly_mean(file_to_average, time_file, time_pre_trans_file):
    # INPUT: the timeseries to average monthly and the time file (air or ground, forcing)
    # OUTPUT: dictionary that assigns a mean value for each month
    #         the dictionary takes the form
    #         dic_monthly_mean = {year: {month: mean value}}
    list_years_month = list_tokens_year_month(time_file, time_pre_trans_file)
    dic_monthly_mean = {}
    for year in list(list_years_month.keys()):
        dic_monthly_mean[year] = {month: np.mean(file_to_average[list_years_month[year][month]])
                                      for month in list(list_years_month[year].keys())}
    return dic_monthly_mean

def assign_dic_monthly_air_temp(dic_monthly_air_temp, df,
                                temp_air_era5, time_air_era5, time_pre_trans_air_era5,
                                temp_air_merra2, time_air_merra2, time_pre_trans_air_merra2,
                                temp_air_jra55, time_air_jra55, time_pre_trans_air_jra55,
                                extension=''):
    # INPUT: takes all temp and time series from all 3 forcing
    # OUTPUT: dictionary that assigns a mean temperature for each month
    #         the dictionary takes the form
    #         dic_monthly_air_temp = {forcing: {altitude: {year: {month: mean air T}}}}

    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"dic_monthly_air_temp{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"dic_monthly_air_temp{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # if the variable has no assigned value yet, we need to assign it
    if dic_monthly_air_temp is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode 
            with open(my_path, 'rb') as file: 
                # Call load method to deserialze 
                dic_monthly_air_temp = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
        except (OSError, IOError) as e:
            list_alt = sorted(list(set(df.altitude)))
            dic_monthly_air_temp = {'era5': {list_alt[alt]: dic_monthly_mean(temp_air_era5[:, alt], time_air_era5, time_pre_trans_air_era5) for alt in range(temp_air_era5.shape[1])},
                                    'merra2': {list_alt[alt]: dic_monthly_mean(temp_air_merra2[:, alt], time_air_merra2, time_pre_trans_air_merra2) for alt in range(temp_air_merra2.shape[1])},
                                    'jra55': {list_alt[alt]: dic_monthly_mean(temp_air_jra55[:, alt], time_air_jra55, time_pre_trans_air_jra55) for alt in range(temp_air_jra55.shape[1])}}

            # Open a file and use dump() 
            with open(my_path, 'wb') as file:
                # A new file will be created 
                pickle.dump(dic_monthly_air_temp, file)
            print('Created a new pickle:', file_name)

            # useless line just to use the variable 'e' so that I don't get an error
            if e == 0:
                pass

    else:
        print('The variable already existed:', var_name_full)


    return dic_monthly_air_temp

def assign_dic_monthly_ground_temp(dic_monthly_ground_temp, list_valid_sim,
                                   temp_ground, time_ground, time_pre_trans_ground,
                                   extension=''):
    # INPUT: takes all simulated temp and time series from the ground
    # OUTPUT: dictionary that assigns a mean temperature for each month
    #         the dictionary takes the form
    #         dic_monthly_ground_temp = {simulation_index: {year: {month: mean ground T}}}

    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"dic_monthly_ground_temp{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"dic_monthly_ground_temp{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # if the variable has no assigned value yet, we need to assign it
    if dic_monthly_ground_temp is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode 
            with open(my_path, 'rb') as file: 
                # Call load method to deserialze 
                dic_monthly_ground_temp = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
        except (OSError, IOError) as e:
            dic_monthly_ground_temp = {i: dic_monthly_mean(temp_ground[i,:,0], time_ground,
                                                           time_pre_trans_ground)
                                        for i in list_valid_sim}

            # Open a file and use dump() 
            with open(my_path, 'wb') as file:
                # A new file will be created 
                pickle.dump(dic_monthly_ground_temp, file)
            print('Created a new pickle:', file_name)

            # useless line just to use the variable 'e' so that I don't get an error
            if e == 0:
                pass

    else:
        print('The variable already existed:', var_name_full)

    return dic_monthly_ground_temp

def melt_out_date_consecutive(consecutive, dict_melt_out_consecutive,
                              snow_height, list_valid_sim, time_ground,
                              year_bkg_end=2000, year_trans_end=2020, extension=''):
    # INPUT: takes the list of valid simulations together with the ground time series and the pre transient one
    #        consecutive: number of consecutive of snow-free days to consider that snow has melted for the season
    # OUTPUT: dictionary that assings a melt-out date to each simulation and each year
    #         Note that if snow does NOT melt at all, we assign the maximal value corresponding to the end of the year

    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"dict_melt_out_consecutive{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"dict_melt_out_consecutive{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # if the variable has no assigned value yet, we need to assign it
    if dict_melt_out_consecutive is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode 
            with open(my_path, 'rb') as file: 
                # Call load method to deserialze 
                dict_melt_out_consecutive = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
        except (OSError, IOError) as e:
            # we start by importing the list of time indexes separated by year
            dictionary = list_tokens_year(time_ground, year_bkg_end, year_trans_end)[0]

            # we create the dictionary of melt-out dates
            dict_melt_out_consecutive = defaultdict(dict)
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

                    dict_melt_out_consecutive[sim][year] = result

            
            # Open a file and use dump() 
            with open(my_path, 'wb') as file:
                # A new file will be created 
                pickle.dump(dict_melt_out_consecutive, file)
            print('Created a new pickle:', file_name)

            # useless line just to use the variable 'e' so that I don't get an error
            if e == 0:
                pass

    else:
        print('The variable already existed:', var_name_full)

    return dict_melt_out_consecutive

def melt_out_date(dict_melt_out, snow_height, list_valid_sim,
                  time_ground, year_bkg_end=2000, year_trans_end=2020, extension=''):
    # INPUT: takes the list of valid simulations together with the ground time series and the pre transient one
    # OUTPUT: dictionary that assings a melt-out date to each simulation and each year
    #         Note that if snow does NOT melt at all, we assign the maximal value corresponding to the end of the year

    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"dict_melt_out{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"dict_melt_out{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # if the variable has no assigned value yet, we need to assign it
    if dict_melt_out is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode
            with open(my_path, 'rb') as file: 
                # Call load method to deserialze 
                dict_melt_out = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
        except (OSError, IOError) as e:
            # we start by importing the list of time indexes separated by year
            dictionary = list_tokens_year(time_ground, year_bkg_end, year_trans_end)[0]

            # we create the dictionary of melt-out dates
            dict_melt_out = defaultdict(dict)
            for sim in list_valid_sim:
                for year in list(dictionary.keys()):
                    # we extract the time series indices corresponding to the year selected
                    time_particular_year = dictionary[year]

                    # this finds the first iteration of snow_height = 0 which is taken as the melt-out date
                    # if no such index exists, there is no melt-out and we assign the maximal value (e.g. 365 or 366 if frequency is 'days')
                    dict_melt_out[sim][year] = next((i for i, x in enumerate(snow_height[sim, time_particular_year]) if x == 0), len(time_particular_year))

            # Open a file and use dump() 
            with open(my_path, 'wb') as file:
                # A new file will be created 
                pickle.dump(dict_melt_out, file)
            print('Created a new pickle:', file_name)

            # useless line just to use the variable 'e' so that I don't get an error
            if e == 0:
                pass

    else:
        print('The variable already existed:', var_name_full)

    return dict_melt_out

def stats_melt_out(dict_melt_out, year_bkg_end=2000):
    stats_melt_out_dic = {}
    for sim_index in list(dict_melt_out.keys()):
        stats_melt_out_dic[sim_index] = {'bkg_mean': np.mean([dict_melt_out[sim_index][k] for k in list(dict_melt_out[sim_index].keys()) if k < year_bkg_end]),
                                         'trans_mean': np.mean([dict_melt_out[sim_index][k] for k in list(dict_melt_out[sim_index].keys()) if k >= year_bkg_end]),
                                         'full_mean': np.mean(list(dict_melt_out[sim_index].values())),
                                         'full_std': np.std(list(dict_melt_out[sim_index].values()))}
    return stats_melt_out_dic

def stat_model(X, offset, c_alt, c_asp, d_asp, c_slope, d_slope, c_snow, c_SW, c_LW, c_WSPD):
    # INPUT: X: all the variables: altitude, aspect, slope, mean_snow, mean_SW, mean_LW, mean_WSPD
    #        rest is the coefficients of the model
    # OUTPUT: value of the model

    # This is the statistical model we are trying to fit to the data.
    # unpack the variables in X
    altitude, aspect, slope, mean_snow, mean_SW, mean_LW, mean_WSPD = X
    return (offset + c_alt * altitude + c_asp * np.cos(aspect * 2 * np.pi / 360)
            + d_asp * np.sin(aspect * 2 * np.pi / 360)
            + c_slope * np.cos(slope * 2 * np.pi / 360 + d_slope)
            + c_snow * mean_snow + c_SW * mean_SW + c_LW * mean_LW + c_WSPD * mean_WSPD)

def stat_model_reduced(X, offset, c_alt, c_asp, d_asp, c_slope, c_snow, c_frac, c_SW):
    # However, having only one site, there is no variation on any of mean_SW, mean_LW, mean_WSPD
    # Hence, including them just introduces a huge level of degeneracy in the parameters
    # This is the same as the function stat_model but has less variables
    altitude, aspect, slope, mean_snow, frac_snow, mean_SW = X
    return (offset + c_alt * altitude + c_asp * np.cos(aspect * 2 * np.pi / 360)
            + d_asp * np.sin(aspect * 2 * np.pi / 360)
            + c_slope * np.sin(slope * 2 * np.pi / 360)
            + c_snow * mean_snow + c_frac * frac_snow
            + c_SW * mean_SW)

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # INPUT: variables and axes
    # OUTPUT: scatter plot with x and y histograms

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=25, color='black')

    # now determine nice limits by hand:
    binwidth = 0.05
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, alpha=0.5)
    ax_histy.hist(y, bins=bins, orientation='horizontal', alpha=0.5)

def scatter_hist_y(x, y, ax, ax_histy):
    # INPUT: variables and axes
    # OUTPUT: scatter plot with y histogram only

    # no labels
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=25, color='black')

    # now determine nice limits by hand:
    binwidth = 0.05
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histy.hist(y, bins=bins, orientation='horizontal', alpha=0.5)

def fit_stat_model_reduced(df_stats, material_name='rock'):
    # INPUT: could potentially change the material, but it is set to 'rock' by default
    # OUTPUT: xdata: measured differential warmings (from valid simulations)
    #         ydata: modelled differential warmings (from valid simulations)
    #         optimizedParameters: best fit parameters

    # this is a list of simulation indexes that correspond to the queried topography
    # by defauls we set material to be 'rock'
    sim = sorted(list(df_stats.query(f"material == '{material_name}'").index))
    
    # ensemble of inputs for the reduced case of Joffre
    # takes: aspect, slope, mean_snow, frac_snow
    input_reduced = np.array([df_stats['altitude'][sim], df_stats['aspect'][sim],
                              df_stats['slope'][sim], df_stats['mean_snow'][sim],
                              df_stats['frac_snow'][sim], df_stats['mean_SW'][sim]])
    
    # all the measured differential warmings (from valid simulations) are in xdata
    xdata = np.array(df_stats['diff_warming'][sim])
    
    # The actual curve fitting happens here
    optimizedParameters, pcov = opt.curve_fit(stat_model_reduced, input_reduced, xdata)

    # this represents the fitted values, hence they are the statistically-modelled values 
    # of differential warming: we call them ydata
    ydata = stat_model_reduced(input_reduced, *optimizedParameters)

    # R^2 from numpy package, to check!
    corr_matrix = np.corrcoef(xdata, ydata)
    corr = corr_matrix[0,1]
    R_sq = corr**2

    # for each altitude, we plot the modelled vs simulated differential warming
    # turns out that the (x,y) sets are EXACTLY the same for each altitude
    plt.scatter(xdata, ydata, s=2, label='data')

    # plot the y=x diagonal
    # start by setting the bounds
    lim_up = float("{:.2g}".format(max(np.max(xdata), np.max(ydata))))
    lim_down = float("{:.2g}".format(min(np.min(xdata), np.min(ydata))))
    x = np.arange(lim_down, lim_up, 0.01)
    plt.plot(x, x, color='red', linestyle='dashed', label = 'y=x')
    plt.legend(loc='upper right')

    margin = 0.1
    plt.ylim(ymin= lim_down - margin, ymax= lim_up + margin)
    plt.xlim(xmin= lim_down - margin, xmax= lim_up + margin)

    plt.title('Comparison of statistically-modelled and simulated differential warming')
    plt.xlabel('Simulated differential warming [°C]')
    plt.ylabel('Statistically-modelled differential warming [°C]')
    plt.figtext(.7, .2, f"$R^2$ = %s" % float("{:.2f}".format(R_sq)))

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()

    return xdata, ydata, optimizedParameters

def fit_stat_model_marginal_distrib(rotate, df_stats, material_name='rock'):
    # INPUT: rotate: whether or not we want to transform the data to distribute it along the diagonal y=x
    #                rotate NO if 0, otherwise YES
    #        could potentially change the material, but it is set to 'rock' by default
    #        or the altitude to an other one
    # OUTPUT: xdata: measured differential warmings (from valid simulations)
    #         ydata: modelled differential warmings (from valid simulations)
    #         optimizedParameters: best fit parameters
    #         lin_model_init: linear regression between modelled and simulated data
    #         lin_model: adjusted (or not) linear regression between modelled and simulated data
    
    # this is a list of simulation indexes that correspond to the queried topography
    # by defauls we set material to be 'rock' 
    # and the altitude to 2000m (since there is no altitude dependence in this dataset)
    sim = sorted(list(df_stats.query(f"material == '{material_name}' ").index))
    
    # ensemble of inputs for the reduced case of Joffre
    input_reduced = np.array([df_stats['altitude'][sim], df_stats['aspect'][sim],
                              df_stats['slope'][sim], df_stats['mean_snow'][sim],
                              df_stats['frac_snow'][sim], df_stats['mean_SW'][sim]])
    
    # all the measured differential warmings (from valid simulations) are in xdata
    xdata = np.array(df_stats['diff_warming'][sim])

    # The actual curve fitting happens here
    # Unfortunately, it doesn't produce a best fit for model vs simulated value around y=x
    # but rather around some line with a slope not equal to 1 and a non-zero intercept
    optimizedParameters_init, pcov_init = opt.curve_fit(stat_model_reduced, input_reduced, xdata)
    ydata_init = stat_model_reduced(input_reduced, *optimizedParameters_init)

    # this is a very simple linear regression that looks at modelled vs actual (simulated) data
    # we can get the slope and intercetp: D_mod = a*D_sim + b
    #                                           = offset + (...)
    # and hence we get D_sim = 1/a * [ (offset - b) + (...) ]
    lin_model_init = linregress(xdata, ydata_init)

    # we thus start by correcting the first paramter, the offset, by the intercept
    optpar = optimizedParameters_init
    optpar[0] -= lin_model_init.intercept
    # we then contrain the initial guess and the bounds to give exactly the 'rotated' data
    init_guess = (1/lin_model_init.slope) * np.array(optpar)
    # this is by how much we allow the search to deviate from the initial guess
    # set to 1 if do not want to search, will not rotate the data
    # set to 1e-10 or less if want to rotate the data to y=x
    if rotate == 0:
        plusminus = 1
    else:
        plusminus = 1e-10

    bounds_opt = (init_guess - plusminus, init_guess + plusminus)

    # Second curve fitting with the constrained parameters this time
    optimizedParameters, pcov = opt.curve_fit(stat_model_reduced, input_reduced, xdata,
                                              p0=init_guess, bounds=bounds_opt)
    
    # this represents the fitted values, hence they are the statistically-modelled values 
    # of differential warming: we call them ydata
    ydata = stat_model_reduced(input_reduced, *optimizedParameters)

    lin_model = linregress(xdata, ydata)

    return xdata, ydata, optimizedParameters, lin_model_init, lin_model

def plot_stat_model_marginal_distrib(rotate, df_stats, material_name='rock'):
    # INPUT: rotate: whether or not we want to transform the data to distribute it along the diagonal y=x
    #                rotate NO if 0, otherwise YES
    # OUTPUT: plot stat-modelled vs simulated diff warming with histograms, marginales, etc.


    # we can now compute the linear regression between the xdata and the 'adjusted' ydata
    # xdata is simulated
    xdata = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[0]
    # ydata is modelled
    ydata = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[1]
    lin_model = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[4]

    # R^2 from numpy package, to check!
    corr_matrix = np.corrcoef(xdata, ydata)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    # p-value
    p_val = lin_model.pvalue

    # Start with a square Figure.
    fig = plt.figure(figsize=(8, 8))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    # histx is for simulation
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    # histy is for model
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    scatter_hist(xdata, ydata, ax, ax_histx, ax_histy)

    # histx is for simulation
    ax_histx.set_ylabel('Simulation count')
    # histy is for model
    ax_histy.set_xlabel('Model count')

    # we compute the mean and standard deviation for the simulated differential warming
    D_sim = [np.mean(xdata), np.std(xdata)]
    # and for the statistically-modelled one
    D_mod = [np.mean(ydata), np.std(ydata)]

    # simulation data
    ax_histx.axvline(D_sim[0], linestyle='dashed', linewidth=2)
    ax_histx.axvline(D_sim[0] + D_sim[1], linestyle='dashed', linewidth=1)
    ax_histx.axvline(D_sim[0] - D_sim[1], linestyle='dashed', linewidth=1)
 
    ax.axvline(D_sim[0], linestyle='dashed', linewidth=2)
    ax.axvline(D_sim[0] + D_sim[1], linestyle='dashed', linewidth=1)
    ax.axvline(D_sim[0] - D_sim[1], linestyle='dashed', linewidth=1)
    
    # model data
    ax_histy.axhline(D_mod[0], linestyle='dashed', linewidth=2)
    ax_histy.axhline(D_mod[0] + D_mod[1], linestyle='dashed', linewidth=1)
    ax_histy.axhline(D_mod[0] - D_mod[1], linestyle='dashed', linewidth=1)
 
    ax.axhline(D_mod[0], linestyle='dashed', linewidth=2)
    ax.axhline(D_mod[0] + D_mod[1], linestyle='dashed', linewidth=1)
    ax.axhline(D_mod[0] - D_mod[1], linestyle='dashed', linewidth=1)

    ax.set_xlabel('Simulated differential warming [°C]')
    ax.set_ylabel('Statistically-modelled differential warming [°C]')
    # fig.suptitle('Comparison of statistically-modelled and simulated differential warming')
    annotation = r'$R^2$ = %s' % float("{:.2g}".format(R_sq)) 
    annotation += "\n"
    annotation += r'$p$ = %s' % float("{:.2g}".format(p_val)) 
    ax.annotate(annotation, (0.8, 0.3), xycoords='axes fraction', va='center')

    # simulation data
    annotation_string_x = r"$\overline{\mathcal{D}_{\rm sim}}$ = %.3f" % (D_sim[0]) 
    annotation_string_x += "\n"
    annotation_string_x += r"$\sigma(\mathcal{D}_{\rm sim})$ = %.3f" % (D_sim[1]) 
    ax_histx.annotate(annotation_string_x, (0.1, 0.7), xycoords='axes fraction', va='center')

    # model data
    annotation_string_y = r"$\overline{\mathcal{D}_{\rm mod}}$ = %.3f" % (D_mod[0]) 
    annotation_string_y += "\n"
    annotation_string_y += r"$\sigma(\mathcal{D}_{\rm mod})$ = %.3f" % (D_mod[1]) 
    ax_histy.annotate(annotation_string_y, (0.05, 0.9), xycoords='axes fraction', va='center')

    # plot the y=x diagonal
    # start by setting the bounds
    lim_up = float("{:.2g}".format(max(np.max(xdata), np.max(ydata))))
    lim_down = float("{:.2g}".format(min(np.min(xdata), np.min(ydata))))
    x = np.arange(lim_down, lim_up, 0.01)
    ax.plot(x, x, color='red', linestyle='dashed', label = 'y=x')
    ax.legend(loc='upper right')

    margin = 0.1
    ax.set_ylim(ymin= lim_down - margin, ymax= lim_up + margin)
    ax.set_xlim(xmin= lim_down - margin, xmax= lim_up + margin)

    if rotate == 0:
        ax.plot(x, lin_model.slope * x + lin_model.intercept, color='green',
            label="modelled vs simulated linear fit", linestyle='dashed')
        ax.legend(loc='upper left')

    # Show the graph
    plt.show()
    plt.close()
    plt.clf()

def plot_residuals_vs_fit(rotate, df_stats, material_name='rock'):
    # INPUT: rotate: whether or not we want to transform the data to distribute it along the diagonal y=x
    #                rotate NO if 0, otherwise YES
    # OUTPUT: plot residuals vs fitted values of diff warming (i.e. stat-modelled, in y) with y histograms
    xdata = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[0]
    ydata = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[1]
    lin_model = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[4]

    # we define the resiudals as the calculated value (model, ydata)
    # minus the actual (simulation, xdata) value
    residuals = ydata - xdata
    print('residual sum is %s' % np.sum(residuals))
    print('square value sum is %s' % np.sum(np.square(residuals)))

    residuals_bis = ydata - ( lin_model.slope * xdata + lin_model.intercept )
    print('residual bis sum is %s' % np.sum(residuals_bis))
    print('square value sum is %s' % np.sum(np.square(residuals_bis)))

    # Start with a square Figure.
    fig = plt.figure(figsize=(8, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                        left=0.1, right=0.9, wspace=0.05, hspace=0.0)
    # Create the Axes.
    ax = fig.add_subplot(gs[0, 0])
    # histy is for model
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    # plot the residuals vs the fitted data (model, ydata)
    scatter_hist_y(ydata, residuals, ax, ax_hist)

    # linear regression
    res_lin = linregress(ydata, residuals)

    if np.max([np.abs(res_lin.slope), np.abs(res_lin.intercept)]) > 1e-2: 
        # plot the fitted linear regression
        # start by setting the bounds
        lim_up = float("{:.2g}".format(max(np.max(xdata), np.max(ydata))))
        lim_down = float("{:.2g}".format(min(np.min(xdata), np.min(ydata))))
        x = np.arange(lim_down, lim_up, 0.01)
        ax.plot(x, res_lin.slope * x + res_lin.intercept, color='red', linestyle='dashed', label = 'fit')
        ax.legend(loc='upper right')

        margin = 0.05
        ax.set_ylim(ymin= lim_down - margin, ymax= lim_up + margin)
        ax.set_xlim(xmin= lim_down - margin, xmax= lim_up + margin)

    print(res_lin)
    
    # ax is for residuals vs Fitted diff warming
    ax.set_xlabel('Statistically-modelled differential warming [°C]')
    ax.set_ylabel('Residuals [°C]')
    # hist is for residual count
    ax_hist.set_xlabel('Residual count')

    # we compute the mean and standard deviation for the residuals
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    
    # model data
    ax_hist.axhline(res_mean, linestyle='dashed', linewidth=2)
    ax_hist.axhline(res_mean + res_std, linestyle='dashed', linewidth=1)
    ax_hist.axhline(res_mean - res_std, linestyle='dashed', linewidth=1)
 
    ax.axhline(res_mean, linestyle='dashed', linewidth=2)
    ax.axhline(res_mean + res_std, linestyle='dashed', linewidth=1)
    ax.axhline(res_mean - res_std, linestyle='dashed', linewidth=1)

    # Show the graph#
    plt.show()
    plt.close()
    plt.clf()

def plot_residuals_vs_value(rotate, df_stats, material_name='rock'):
    # INPUT: rotate: whether or not we want to transform the data to distribute it along the diagonal y=x
    #                rotate NO if 0, otherwise YES
    # OUTPUT: plot residuals vs actual values of diff warming (i.e. simulated, in x) with y histograms
    xdata = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[0]
    ydata = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[1]
    lin_model = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[4]

    # we define the resiudals as the calculated value (model, ydata)
    # minus the actual (simulation, xdata) value
    residuals = ydata - xdata
    print('residual sum is %s' % np.sum(residuals))
    print('square value sum is %s' % np.sum(np.square(residuals)))

    residuals_bis = ydata - ( lin_model.slope * xdata + lin_model.intercept )
    print('residual bis sum is %s' % np.sum(residuals_bis))
    print('square value sum is %s' % np.sum(np.square(residuals_bis)))

    # plot the residuals vs the actual data (simulation, xdata)
    plt.scatter(xdata, residuals)
    print(linregress(xdata, residuals))

    # x = np.arange(-0.35, 0.35, 0.01)
    # plt.plot(x, x, color='red', linestyle='dashed', label = 'y=x')

    # Show the graph
    plt.xlabel('Simulated differential warming [°C]')
    plt.ylabel('Residuals')
    plt.show()
    plt.close()
    plt.clf()

def plot_qq(rotate, df_stats, material_name='rock'):
    xdata = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[0]
    ydata = fit_stat_model_marginal_distrib(rotate, df_stats, material_name)[1]

    residuals = ydata - xdata
    std_res = (residuals-np.mean(residuals))/np.std(residuals)
    
    # Start with a square Figure.
    plt.figure(figsize=(8, 6))
    stats.probplot(std_res, dist="norm", plot=plt)
    plt.title("")
    # plt.xlabel('xlabel')
    plt.ylabel('Standardized residuals')
    plt.show()
    plt.close()
    plt.clf()

def plot_temp_var(sim_index, df_stats, diff_fontsize = 12):
    # INPUT: sim_index: index of the simulation
    # OUTPUT: plot of the evolution of air and ground temperature, together with
    #         a visual interpretation of the differential warming

    fig, ax = plt.subplots()
    ax.figure.set_size_inches(4,5)

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    # ground and air temperature for background and transient periods
    grd = np.array([df_stats.loc[sim_index]['bkg_grd_temp'], df_stats.loc[sim_index]['trans_grd_temp']])
    air = np.array([df_stats.loc[sim_index]['bkg_air_temp'], df_stats.loc[sim_index]['trans_air_temp']])
    # difference = ground - air
    dif = [grd[0] - air[0], grd[1] - air[1]]

    # plot all points
    plt.scatter([0, 1], grd, label='Ground', s=100)
    plt.scatter([0, 1], air, label='Air', s=100)
    # arrow between air and ground temperature for both background and transient
    ax.plot([0,0], [air[0], grd[0]], color='red', ls='-', marker='_', label='SO')
    # plt.arrow(0, air[0], 0, dif[0], width = 0.01, head_width = 0.1, color='red', length_includes_head=True)
    plt.plot([1,1], [air[1], grd[1]], color='red', ls='-', marker='_')
    # plt.arrow(1, air[1], 0, dif[1], width = 0.01, head_width = 0.1, color='red', length_includes_head=True)

    # plot dashed line for time evolution of ground temperature
    plt.plot([0,1], grd, color=colorcycle[0], ls='--', marker=None)
    # plt.plot([0,1], grd - grd[0] + air[0], color=colorcycle[0], ls='--', marker=None)
    # same for air
    plt.plot([0,1], air, color=colorcycle[1], ls='--', marker=None)
    plt.plot([0,1], air + grd[0] - air[0], color='orange', ls='--', marker=None)

    # plot difference between differences: differential warming
    plt.arrow(2, grd[1] - (dif[1]-dif[0]), 0, dif[1]-dif[0], width = 0.01, head_width = 0.05, color='green', length_includes_head=True)
    plt.arrow(1, grd[1] - (dif[1]-dif[0]), 0, dif[1]-dif[0], width = 0.01, head_width = 0.05, color='green', length_includes_head=True)
    ddif = dif[1]-dif[0]
    alignment = ['top' if ddif < 0 else 'bottom'][0]
    ax.text(2, grd[1] - (dif[1]-dif[0]) + (dif[1]-dif[0]),
            r"$\mathcal{D}$ = %s%s°C" % (("+" if ddif > 0 else ""), float("{:.3f}".format(ddif))),
            color='green', fontsize=diff_fontsize, ha='right', va=alignment)

    plt.plot([1,2], [grd[1] - (dif[1]-dif[0]), grd[1] - (dif[1]-dif[0])], color='green', ls='--', marker=None)
    plt.plot([1,2], [grd[1], grd[1]], color='green', ls='--', marker=None)

    plt.title('Evolution of ground and air mean temperatures')
    # plt.xlabel('Time')
    plt.ylabel('Temperature [°C]')

    # make sure we only have 2 ticks and not 3, and label them
    xticks = np.arange(0, 2, 1)
    xlabels = ['Background', 'Transient']
    ax.set_xticks(xticks, labels=xlabels)

    # Show the graph
    plt.legend(loc='lower right')
    plt.show()
    plt.close()
    plt.clf()

def plot_asp_slo_alt_radial(sim_index, alt_min, alt_max, df_stats):
    # INPUT: sim_index: index of the simulation
    #        alt_min = minimal altitude considered across ALL sites
    #        alt_max = maximal altitude considered across ALL sites
    # OUTPUT: radial plot of aspect and slope, with altitude indicated as a color

    asp = df_stats.loc[sim_index].aspect
    alt = df_stats.loc[sim_index].altitude
    slo = df_stats.loc[sim_index].slope

    # create altitude bins for legend color
    all_alts = np.arange(alt_min - 250, alt_max + 500 + 250, 500)
    cmap = plt.get_cmap('inferno_r', len(all_alts)-1)

    # create figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    # this reverses the radial axis in order to have 90 degrees in the center
    ax.set_rlim(bottom=90, top=20)
    # this reverses the angle direction which is now clockwise or antidirect
    ax.set_theta_direction(-1)
    # set the North to the top
    ax.set_theta_zero_location("N")
    # plots aspect and slope with altitude as a color gradient
    c = ax.scatter(asp*np.pi/180, slo, c=alt, s=100, cmap=cmap, alpha=1,
                vmin=all_alts[0], vmax=all_alts[-1])
    plt.colorbar(c, ax=ax, pad=0.1)

    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def subsets_df_stats(df_stats, variable=None, value=None):
    # INPUT: one can decide to plot for ALL simulations (default)
    #        OR to choose a sample given by the couple (variable, value)
    #        where variable is 'aspect', 'slope', 'altitude', 'forcing', etc.
    #        and value is a legitimate value that the variable can take, for instance 2000 for altitude
    #        ###Note### that 'aspect' is over a full quadrant and hence covers the value +-45°
    #        e.g. we consider NW/N/NE (315/0/45°) when asked for North-facing (0°) slopes
    # OUTPUT: returns the subset of df_stats corresponding to the pair (variable, value)

    # creates a subset of df_stats given the value of the variable entered as input. e.g. 'slope'=50
    if variable==None:
        data = df_stats
    elif variable=='aspect':
        data = df_stats[df_stats['aspect'].isin({(value-45)%360, value%360, (value+45)%360})]
    else:
        data = df_stats[df_stats[variable]==value]

    return data

def plot_evolution_snow_cover_melt_out(df_stats, variable=None, value=None):
    # INPUT: one can decide to plot for ALL simulations (default)
    #        OR to choose a sample given by the couple (variable, value)
    #        see 'subsets_df_stats' function for more info
    # OUTPUT: plots the time evolution (transient - background) of the snow cover and melt out date
    #         note that only the simulations with snow are accounted for. 
    #         otherwise, we would get a huge spike at 0 for all simulations that had no snow and kept it this way.

    # creates a subset of df_stats given the value of the variable entered as input. e.g. 'slope'=50
    data = subsets_df_stats(df_stats, variable, value)

    # creates a list of the time evolution of both parameters
    # makes sure to only keep the simulations that have shown at least 1 day of snow over the whole study period
    evol_melt_out = [data.iloc[k].melt_out_cons_trans - data.iloc[k].melt_out_cons_bkg for k in range(len(data)) if ((data.iloc[k].frac_snow_bkg != 0) | (data.iloc[k].frac_snow_trans != 0))]
    evol_snow_cover = [(data.iloc[k].frac_snow_trans - data.iloc[k].frac_snow_bkg)*365.25 for k in range(len(data)) if ((data.iloc[k].frac_snow_bkg != 0) | (data.iloc[k].frac_snow_trans != 0))]

    # plots both histograms
    plt.hist(evol_snow_cover, bins=20, alpha=0.75, weights=np.ones_like(evol_snow_cover) / len(evol_snow_cover), label='Snow cover')
    plt.hist(evol_melt_out, bins=20, alpha=0.75, weights=np.ones_like(evol_melt_out) / len(evol_melt_out), label='Melt out date')

    mean_snow_cov = np.mean(evol_snow_cover)
    mean_melt_out = np.mean(evol_melt_out)

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    # adds a vertical line denoting the mean values
    plt.axvline(mean_snow_cov, color=colorcycle[0], linestyle='dashed', linewidth=2)
    plt.axvline(mean_melt_out, color=colorcycle[1], linestyle='dashed', linewidth=2)

    plt.annotate(r"$\overline{\Delta}_{\rm snow\ cover}=$%s%s [days]" % (("+" if mean_snow_cov > 0 else ""), float("{:.2f}".format(mean_snow_cov))),
                 (0.12,0.5), xycoords='figure fraction',
                 fontsize=12, horizontalalignment='left', verticalalignment='top', color=colorcycle[0])
    plt.annotate(r"$\overline{\Delta}_{\rm mod}=$%s%s [days]" % (("+" if mean_melt_out > 0 else ""), float("{:.2f}".format(mean_melt_out))),
                 (0.12,0.45), xycoords='figure fraction',
                 fontsize=12, horizontalalignment='left', verticalalignment='top', color=colorcycle[1])

    plt.xlabel('Evolution [days]')
    plt.ylabel('Frequency')
    units = {'altitude': 'm', 'aspect': '°', 'slope': '°', 'forcing': ''}
    # plt.title(f'Time evolution of the snow mantle{('' if variable==None else f' for {variable} = {value}{units[variable]}')}')

    # Show the graph
    plt.legend(loc='upper left')
    plt.show()
    plt.close()
    plt.clf()

def snow_boxplot_variable(df_stats, variable, plot_type='box'):
    # INPUT: one can decide to plot for ALL simulations (default)
    #        OR to choose a sample given by the couple (variable, value)
    #        see 'subsets_df_stats' function for more info
    # OUTPUT: plots the time evolution (transient - background) of the snow cover and melt out date
    #         note that only the simulations with snow are accounted for. 
    #         otherwise, we would get a huge spike at 0 for all simulations that had no snow and kept it this way.

    # xlist is a list of all values of the variable
    xlist = np.sort(list(set(df_stats.loc[:, variable])))

    # creates a subset of df_stats given the value of the variable entered as input. e.g. 'slope'=50
    data = [df_stats[df_stats[variable]==value] for value in xlist]

    # creates a list of the time evolution of both parameters
    # makes sure to only keep the simulations that have shown at least 1 day of snow over the whole study period
    evol_snow_cover = [[data[i].iloc[k].melt_out_cons_trans - data[i].iloc[k].melt_out_cons_bkg for k in range(len(data[i])) if ((data[i].iloc[k].frac_snow_bkg != 0) | (data[i].iloc[k].frac_snow_trans != 0))] for i in range(len(data))]
    evol_melt_out = [[(data[i].iloc[k].frac_snow_trans - data[i].iloc[k].frac_snow_bkg)*365.25 for k in range(len(data[i])) if ((data[i].iloc[k].frac_snow_bkg != 0) | (data[i].iloc[k].frac_snow_trans != 0))] for i in range(len(data))]

    mean_snow_cover = [np.mean(evol_snow_cover[i]) for i in range(len(xlist))]
    mean_melt_out = [np.mean(evol_melt_out[i]) for i in range(len(xlist))]

    ticks = xlist
    colorcycle = [u'#1f77b4', u'#ff7f0e']

    plt.figure()

    if plot_type == 'box':
        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)
            plt.setp(bp['means'], markerfacecolor=color)
            plt.setp(bp['means'], markeredgecolor=color)

        bpl = plt.boxplot(evol_snow_cover, showmeans=True, positions=np.array(range(len(evol_snow_cover)))*2.0-0.4,
                          sym='', widths=0.6)
        bpr = plt.boxplot(evol_melt_out, showmeans=True, positions=np.array(range(len(evol_melt_out)))*2.0+0.4,
                          sym='', widths=0.6)
        set_box_color(bpl, colorcycle[0])
        set_box_color(bpr, colorcycle[1])

    elif plot_type == 'violin':
        bpl = plt.violinplot(evol_snow_cover, showmedians=True, positions=np.array(range(len(evol_snow_cover)))*2.0-0.4, widths=0.6)
        bpr = plt.violinplot(evol_melt_out, showmedians=True, positions=np.array(range(len(evol_melt_out)))*2.0+0.4, widths=0.6)
        plt.scatter(np.array(range(len(evol_snow_cover)))*2.0-0.4, mean_snow_cover, marker="^", c=colorcycle[0])
        plt.scatter(np.array(range(len(evol_melt_out)))*2.0+0.4, mean_melt_out, marker="^", c=colorcycle[1])

    # draw temporary lines and use them to create a legend
    plt.plot([], c=colorcycle[0], label='Snow cover')
    plt.plot([], c=colorcycle[1], label='Melt-out date')
    plt.legend(loc='lower right')

    units = {'altitude': '[m]', 'aspect': '[°]', 'slope': '[°]'}
    plt.xlabel((f'{variable} %s' % (units[variable] if variable in units.keys() else '')).capitalize())

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylabel('Evolution [days]')
    plt.title('Evolution of the snow mantle as a function of %s' % (variable))
    plt.tight_layout()

    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def plot_annual_snow_cover(sim_index, df_stats):
    # INPUT: sim_index: index of the simulation
    # OUTPUT: chart of the mean annual snow cover with melt out date

    # melt out date (in months) 
    a = 12*np.array([0,df_stats.loc[sim_index].melt_out_cons_trans, df_stats.loc[sim_index].melt_out_cons_bkg])/365.25
    # snow-free time (in months)
    b = 12*np.array([0,1-df_stats.loc[sim_index].frac_snow_trans, 1-df_stats.loc[sim_index].frac_snow_bkg])
    # rest of the year
    c = np.array([0,12-a[1]-b[1], 12-a[2]-b[2]])

    ax = pd.DataFrame({'Snow 1' : a,'No snow' : b, 'Snow 2' : c}).plot.barh(stacked=True, color=['blue', 'grey', 'blue'])
    ax.figure.set_size_inches(10,3)

    ax.set_title("Mean annual snow cover")
    ax.legend(['Snow', 'No snow'], loc='lower right')
    ax.margins(0)

    earliest_mo = np.min([a[1],a[2]])
    latest_mo = np.max([a[1],a[2]])

    snow_days_bkg = df_stats.loc[sim_index].frac_snow_bkg * 365.25
    snow_days_trans = df_stats.loc[sim_index].frac_snow_trans * 365.25

    evol_melt_out = (a[1] - a[2])/12*365.25
    evol_snow_days = snow_days_trans - snow_days_bkg

    if a[1] > 0:
        ax.annotate(datetime.strftime(datetime.strptime('01/01/01','%d/%m/%y')+ timedelta(days=df_stats.loc[sim_index].melt_out_cons_bkg), '%d/%m'), (a[2], 2-0.25),
                    fontsize=12, horizontalalignment=('right' if evol_melt_out>0 else 'left'), verticalalignment='top')
        ax.annotate(datetime.strftime(datetime.strptime('01/01/01','%d/%m/%y')+ timedelta(days=df_stats.loc[sim_index].melt_out_cons_trans), '%d/%m'), (a[1], 1-0.25),
                    fontsize=12, horizontalalignment=('left' if evol_melt_out>0 else 'right'), verticalalignment='top')
        ax.annotate(r"$\Delta_{\rm mod}=$%s%s days" % (("+" if evol_melt_out > 0 else ""), float("{:.2f}".format(evol_melt_out))), (latest_mo + 0.1, 0),
                    fontsize=12, horizontalalignment='left', verticalalignment='center')
        ax.annotate(r"$\Delta_{\rm snow}=$%s%s days" % (("+" if evol_snow_days > 0 else ""), float("{:.2f}".format(evol_snow_days))), (12, 1.5),
                    fontsize=12, horizontalalignment='right', verticalalignment='center')
        ax.axvline(a[1], linestyle='dashed', color='black', linewidth=1)
        ax.axvline(a[1]+b[1], linestyle='dashed', color='black', linewidth=1)
        ax.axvline(a[2], linestyle='dashed', color='black', linewidth=1)
        ax.axvline(a[2]+b[2], linestyle='dashed', color='black', linewidth=1)

    # plt.plot([earliest_mo, latest_mo], [0,0], color='green', ls='-', marker='|', linewidth=1)
    # red rectangle if melt out date advances, otherwise green
    ax.add_patch(Rectangle((earliest_mo, 0-0.25), latest_mo-earliest_mo, 2*0.25,
                           color=('r' if earliest_mo == a[1] else 'g')))

    # red rectangle if first snow date is later, otherwise green
    earliest_snow_in = np.min([a[1]+b[1],a[2]+b[2]])
    latest_snow_in = np.max([a[1]+b[1],a[2]+b[2]])
    ax.add_patch(Rectangle((earliest_snow_in, 0-0.25), latest_snow_in-earliest_snow_in, 2*0.25,
                           color=('r' if earliest_snow_in == a[2]+b[2] else 'g')))
    
    # label the xticks for each month
    xlabels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','']
    ax.set_xticklabels(xlabels)
    plt.locator_params(axis='x', nbins=12)

    # label the yticks for each period
    yticks = np.arange(1, 3, 1)
    ylabels = ['Transient', 'Background']
    ax.set_yticks(yticks, labels=ylabels)

    # Show the graph
    # plt.legend(loc='lower right')
    plt.show()
    plt.close()
    plt.clf()

def assign_dic_SO_monthly(dic_SO_monthly, df_stats, list_valid_sim, dic_monthly_ground_temp, dic_monthly_air_temp, extension='', year_end_bkg=2000):
    # INPUT: all sorts of stats
    # OUTPUT: dictionary that assings the mean SO (ground - air) to each month for the background and transient
    #         the dictionary takes the form
    #         dic_SO_monthly = {period: {month: {sim_index: {year: SO}}}}

    # This saves a lot of time since it is not re-evaluated each time but just the first
    # time and then saved into a pickle.

    file_name = f"dic_SO_monthly{('' if extension=='' else '_')}{extension}.pkl"
    var_name_full = f"dic_SO_monthly{('' if extension=='' else '_')}{extension}"
    my_path = pickle_path + file_name

    # if the variable has no assigned value yet, we need to assign it
    if dic_SO_monthly is None:
        # try to open the pickle file, if it exists
        try: 
            # Open the file in binary mode 
            with open(my_path, 'rb') as file: 
                # Call load method to deserialze 
                dic_SO_monthly = pickle.load(file) 
            print('Succesfully opened the pre-existing pickle:', file_name)

        # finally, if the variable AND the pickle file do not exist, we have to create them both
        except (OSError, IOError) as e:
            dic_SO_monthly = {'background': {month: {sim_index: {year: dic_monthly_ground_temp[sim_index][year][month] -
                                                                       dic_monthly_air_temp[df_stats.loc[sim_index].forcing][df_stats.loc[sim_index].altitude][year][month]
                                                                 for year in np.array(list(dic_monthly_ground_temp[list_valid_sim[0]].keys())) if year<year_end_bkg}
                                                     for sim_index in list_valid_sim}
                                             for month in range(1, 13, 1)},
                              'transient': {month: {sim_index: {year: dic_monthly_ground_temp[sim_index][year][month] -
                                                                      dic_monthly_air_temp[df_stats.loc[sim_index].forcing][df_stats.loc[sim_index].altitude][year][month]
                                                                for year in np.array(list(dic_monthly_ground_temp[list_valid_sim[0]].keys())) if year>=year_end_bkg}
                                                    for sim_index in list_valid_sim}
                                            for month in range(1, 13, 1)}
                              }

            # Open a file and use dump() 
            with open(my_path, 'wb') as file:
                # A new file will be created 
                pickle.dump(dic_SO_monthly, file)
            print('Created a new pickle:', file_name)

            # useless line just to use the variable 'e' so that I don't get an error
            if e == 0:
                pass

    else:
        print('The variable already existed:', var_name_full)

    return dic_SO_monthly

def plot_monthly_SO(sim_index, df_stats, dic_monthly_ground_temp, dic_monthly_air_temp):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # fig = plt.figure()
    years = np.array(list(dic_monthly_ground_temp[sim_index].keys()))
    months = np.array(range(1, 13, 1))/12*2*np.pi
    X, Y = np.meshgrid(months, years)

    ax.set_rlim(bottom=years[0]-10, top=years[-1])
    # this reverses the angle direction which is now clockwise or antidirect
    ax.set_theta_direction(-1)
    # set the North to the top
    ax.set_theta_offset(105/360*2*np.pi)

    forcing = df_stats.loc[sim_index].forcing
    altitude = df_stats.loc[sim_index].altitude

    Zmat = [[dic_monthly_ground_temp[sim_index][y][m] -
            dic_monthly_air_temp[forcing][altitude][y][m] for m in range(1, 13, 1)] for y in years]

    ax.set_xticks(months)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_yticklabels([])
    ax.grid(False)

    pc1 = plt.pcolormesh(X, Y, Zmat, vmin=np.min(Zmat), vmax=np.max(Zmat), shading='auto')
    cbar = fig.colorbar(pc1, ax=ax, pad=0.1)
    cbar.ax.set_ylabel('Surface offset [°C]')

    plt.title('Mean monthly surface offset')
    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def plot_monthly_SO_all(df_stats, dic_monthly_ground_temp, dic_monthly_air_temp, list_valid_sim):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # fig = plt.figure()
    years = np.array(list(dic_monthly_ground_temp[0].keys()))
    months = np.array(range(1, 13, 1))/12*2*np.pi
    X, Y = np.meshgrid(months, years)

    ax.set_rlim(bottom=years[0]-10, top=years[-1])
    # this reverses the angle direction which is now clockwise or antidirect
    ax.set_theta_direction(-1)
    # set the North to the top
    ax.set_theta_offset(105/360*2*np.pi)

    Zmat = [[[dic_monthly_ground_temp[sim_index][y][m] -
              dic_monthly_air_temp[df_stats.loc[sim_index].forcing][df_stats.loc[sim_index].altitude][y][m]
              for sim_index in list_valid_sim]
              for m in range(1, 13, 1)]
              for y in years]
    
    Zmat_mean = np.array(Zmat).mean(axis=2)
    print(Zmat_mean.shape)

    ax.set_xticks(months)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_yticklabels([])
    ax.grid(False)

    pc1 = plt.pcolormesh(X, Y, Zmat_mean, vmin=np.min(Zmat_mean), vmax=np.max(Zmat_mean), shading='auto')
    cbar = fig.colorbar(pc1, ax=ax, pad=0.1)
    cbar.ax.set_ylabel('Surface offset [°C]')

    plt.title('Mean monthly surface offset')
    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def plot_evolution_bkg_trans_S0(df_stats, variable=None, value=None): 
    # INPUT: one can decide to plot for ALL simulations (default)
    #        OR to choose a sample given by the couple (variable, value)
    #        see 'subsets_df_stats' function for more info
    # OUTPUT: plots the time evolution (transient - background) of the Surface Offset

    # creates a subset of df_stats given the value of the variable entered as input. e.g. 'slope'=50
    data = subsets_df_stats(df_stats, variable, value)

    fig, ax = plt.subplots()

    # creates a list of the SO (ground - air temp) in the background and transient eras
    SO_bkg = [data.iloc[k]['bkg_grd_temp'] - data.iloc[k]['bkg_air_temp'] for k in range(len(data))]
    SO_trans = [data.iloc[k]['trans_grd_temp'] - data.iloc[k]['trans_air_temp'] for k in range(len(data))]

    # plots both histograms
    plt.hist(SO_bkg, bins=20, alpha=0.75, weights=np.ones_like(SO_bkg) / len(SO_bkg), label='Background SO')
    plt.hist(SO_trans, bins=20, alpha=0.75, weights=np.ones_like(SO_trans) / len(SO_trans), label='Transient SO')

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    # adds a vertical line denoting the mean values

    mean_SO_bkg = np.mean(SO_bkg)
    mean_SO_trans = np.mean(SO_trans)
    mean_diff_warm = np.mean(data['diff_warming'])

    plt.axvline(mean_SO_bkg, color=colorcycle[0], linestyle='dashed', linewidth=2)
    plt.axvline(mean_SO_trans, color=colorcycle[1], linestyle='dashed', linewidth=2)

    plt.annotate(r"$\overline{\mathcal{SO}}_{bkg}=$%s%s [°C]" % (("+" if mean_SO_bkg > 0 else ""), float("{:.2f}".format(mean_SO_bkg))),
                (0.88,0.88), xycoords='figure fraction',
                fontsize=12, horizontalalignment='right', verticalalignment='top', color=colorcycle[0])
    plt.annotate(r"$\overline{\mathcal{SO}}_{trans}=$%s%s [°C]" % (("+" if mean_SO_trans > 0 else ""), float("{:.2f}".format(mean_SO_trans))),
                (0.88,0.84), xycoords='figure fraction',
                fontsize=12, horizontalalignment='right', verticalalignment='top', color=colorcycle[1])
    plt.annotate(r"$\overline{\mathcal{D}}=$%s%s [°C]" % (("+" if mean_diff_warm > 0 else ""), float("{:.2f}".format(mean_diff_warm))),
                (0.88,0.80), xycoords='figure fraction',
                fontsize=12, horizontalalignment='right', verticalalignment='top')

    tform = blended_transform_factory(ax.transData, ax.transAxes)
    plt.annotate('', xy=(np.max([mean_SO_bkg, mean_SO_trans])+2, 0.86), xycoords=tform, 
                xytext=(np.max([mean_SO_bkg, mean_SO_trans])-0.05, 0.86), textcoords=tform, fontsize='xx-large',
                ha='center', va='center', color='r',
                arrowprops=dict(arrowstyle= '<|-', color='black', lw=2, ls='-'))
    plt.annotate('', xy=(np.min([mean_SO_bkg, mean_SO_trans])-2, 0.86), xycoords=tform, 
                xytext=(np.min([mean_SO_bkg, mean_SO_trans])+0.05, 0.86), textcoords=tform, fontsize='xx-large',
                ha='center', va='center', color='r',
                arrowprops=dict(arrowstyle= '<|-', color='black', lw=2, ls='-'))

    plt.xlabel('Temperature [°C]')
    plt.ylabel('Frequency')
    units = {'altitude': 'm', 'aspect': '°', 'slope': '°', 'forcing': ''}
    # plt.title(f'Background and transient SO{('' if variable==None else f' for {variable} = {value}{units[variable]}')}')

    # Show the graph
    plt.legend(loc='upper left')
    plt.show()
    plt.close()
    plt.clf()

def plot_hist_diff_warming(df_stats, variable=None): 
    # INPUT: df_stats: all stats
    #        variable: None by default but can be chosen to be 'altitude', 'aspect',
    #                  'slope', 'forcing', 'material', 'snow_corr', etc.
    #                  Can be any column of df_stats
    # OUTPUT: plots the histogram of differential warming, possibly for all different
    #         calues of a given variable

    if variable == None:
        xlist=['all simulations']
        x = [df_stats['diff_warming']]
    else:
        xlist = np.sort(list(set(df_stats.loc[:, variable])))
        x = [df_stats[df_stats.loc[:, variable] == i]['diff_warming'] for i in xlist]

    # plots histograms
    for i in range(len(x)):
        plt.hist(x[i], bins=20, alpha=0.5, weights=np.ones_like(x[i]) / len(x[i]), label=xlist[i])

    # color cycle
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    # adds a vertical line denoting the mean values
    mean_diff_warm = np.mean(df_stats['diff_warming'])
    annotation = r"$\overline{\mathcal{D}}=$%s%s [°C]" % (("+" if mean_diff_warm > 0 else ""), float("{:.2f}".format(mean_diff_warm)))
    plt.annotate(annotation,
                    (0.86,0.8), xycoords='figure fraction',
                    fontsize=12, horizontalalignment='right', verticalalignment='top')
    dx = 0.05

    if variable == None:
        plt.axvline(mean_diff_warm, color=colorcycle[0], linestyle='dashed', linewidth=2)
    else:
        plt.axvline(mean_diff_warm, color='black', linestyle='dashed', linewidth=2)
        mean_list = [np.mean(x[i]) for i in range(len(xlist))]
        for i in range(len(xlist)):
            plt.axvline(mean_list[i], color=colorcycle[i], linestyle='dashed', linewidth=2) 
            annotation = r"$\overline{\mathcal{D}}_{%s}=$%s%s [°C]" % (xlist[i], ("+" if mean_list[i] > 0 else ""), float("{:.2f}".format(mean_list[i])))
            plt.annotate(annotation,
                         (0.86,0.8-dx*(i+1)), xycoords='figure fraction',
                         fontsize=12, horizontalalignment='right', verticalalignment='top',
                         color=colorcycle[i]) 


    plt.xlabel('Differential warming [°C]')
    plt.ylabel('Frequency')
    # plt.title('')

    # Show the graph
    plt.legend(loc='upper left')
    plt.show()
    plt.close()
    plt.clf()

def diff_warm_boxplot_variable(df_stats, variable, plot_type='box'):
    # INPUT: plot_type: can either take 'box' or 'violin'
    # OUTPUT: boxplot or violinplot of the monthly differential warming for ALL simulations

    # xlist is a list of all values of the variable
    # x is a list of differential warming sorted by values in xlist
    xlist = np.sort(list(set(df_stats.loc[:, variable])))
    x = [df_stats[df_stats.loc[:, variable] == i]['diff_warming'] for i in xlist]

    mean_diff_warm = [np.mean(x[i]) for i in range(len(xlist))]

    ticks = xlist
    colorcycle = [u'#1f77b4', u'#ff7f0e']

    plt.figure()

    if plot_type == 'box':
        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)
            plt.setp(bp['means'], markerfacecolor=color)
            plt.setp(bp['means'], markeredgecolor=color)

        bp = plt.boxplot(x, showmeans=True, positions=np.array(range(len(x)))*2.0,
                          sym='', widths=0.6)
        set_box_color(bp, colorcycle[0])

    elif plot_type == 'violin':
        plt.violinplot(x, showmedians=True, positions=np.array(range(len(x)))*2.0, widths=0.6)
        plt.scatter(np.array(range(len(x)))*2.0, mean_diff_warm, marker="^", c=colorcycle[0])

    units = {'altitude': '[m]', 'aspect': '[°]', 'slope': '[°]'}
    plt.xlabel((f'{variable} %s' % (units[variable] if variable in units.keys() else '')).capitalize())

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylabel('Differential warming [°C]')
    plt.title('Differential warming as a function of %s' % (variable))
    plt.tight_layout()

    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def plot_hist_valid_sim(df, df_stats, variable): 
    """ Function returns a histogram of the number of valid/glacier simulations as a function of a given variable
        ('altitude', 'aspect', 'slope', 'forcing') 
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Panda DataFrame df, should at least include the following columns: 'altitude', 'aspect', 'slope'
    df_stats : pandas.core.frame.DataFrame
        Panda DataFrame df_stats, should at least include the following columns: 'altitude', 'aspect', 'slope'
    variable : str
        can be chosen to be 'altitude', 'aspect', 'slope', 'forcing', 'material', etc.
        Can be any column of both df and df_stats

    Returns
    -------
    Histogram
    """

    # list_var_prev: lists values the variable can take, e.g. [0, 45, 90, 135, etc.] for 'aspect'
    # number_no_glaciers: number of valid simulations (without glaciers) per value in list_var_prev
    list_var_prev, number_no_glaciers = np.unique(df_stats.loc[:, variable], return_counts=True)
    # translate into strings
    list_var = [str(i) for i in list_var_prev]
    # total number of simulations per value in list_var
    tot = list(np.unique(df.loc[:, variable], return_counts=True)[1])
    # number_glaciers: number of glaciers per value in list_var
    number_glaciers = [tot[i] - number_no_glaciers[i] for i in range(len(tot))]

    bottom = np.zeros(len(tot))
    counts = {
        'No glaciers': number_no_glaciers,
        'Glaciers': number_glaciers}

    colorcycle = [u'#1f77b4', u'#ff7f0e']

    for name, data in counts.items():
        p = plt.bar(list_var, data, label=name, bottom=bottom, color=(colorcycle[0] if name=='Glaciers' else colorcycle[1]))
        bottom += data
        data_no_zero = [i if i>0 else "" for i in data]
        plt.bar_label(p, labels=data_no_zero, label_type='center')

    units = {'altitude': '[m]', 'aspect': '[°]', 'slope': '[°]'}

    plt.xlabel((f'{variable} %s' % (units[variable] if variable in units.keys() else '')).capitalize())
    plt.ylabel('Number of simulations')
    plt.title('Simulations producing glaciers as a function of %s' % (variable))

    # Show the graph
    plt.legend(loc='lower right', reverse=True)
    plt.show()
    plt.close()
    plt.clf()

def plot_hist_valid_sim_all_variables(df, df_stats, depth_thaw): 
    """ Function returns a histogram of the number of valid/glacier simulations for each of the following variable
        ('altitude', 'aspect', 'slope', 'forcing') 
        It also shows the breakdown of valid simulations into permafrost and no-permafrost ones

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Panda DataFrame df, should at least include the following columns: 'altitude', 'aspect', 'slope'
    df_stats : pandas.core.frame.DataFrame
        Panda DataFrame df_stats, should at least include the following columns: 'altitude', 'aspect', 'slope'
    depth_thaw : netCDF4._netCDF4.Variable
        NetCDF variable encoding the thaw depth

    Returns
    -------
    Histogram (subplot(2,2))
    """

    data=np.random.random((4,10))
    variables = ['altitude','aspect','slope','forcing']
    xaxes = ['Altitude [m]','Aspect [°]','Slope [°]','Forcing']
    yaxes = ['Number of simulations','','Number of simulations','']

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    list_valid_sim = list(df_stats.index.values)

    list_no_perma = []
    for sim in list_valid_sim:
        if np.std(depth_thaw[sim,:]) < 1 and np.max(depth_thaw[sim,:])> 19:
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
    list_hist_b = [0] * (len(df)-len(pd_weight))

    counts, bins = np.histogram(list_hist, 10, (0, 1))
    counts_b = 0
    if zero == True:
        counts_b, bins_b = np.histogram(list_hist_b, 10, (0, 1))
    tot_count = np.sum(counts) + (np.sum(counts_b) if zero == True else 0)

    colorcycle = [u'#1f77b4', u'#ff7f0e']
    
    plt.hist(bins[:-1], bins, weights=counts/tot_count, label='No glaciers', color=colorcycle[1])
    if zero == True:
        plt.hist(bins_b[:-1], bins_b, weights=counts_b/tot_count, label='Glaciers', color=colorcycle[0])

    max_count = np.ceil((np.max([np.max(counts), np.max(counts_b)])/tot_count)/0.05+1)*0.05
    
    ticks = list(np.arange(0, max_count, 0.05))
    plt.yticks(ticks, ["{:0.2f}".format(i) for i in ticks])

    # Show the graph
    if zero == True:
        plt.legend(loc='upper right')
    plt.xlabel('Statistical weight')
    plt.ylabel('Frequency')
    plt.show()
    plt.close()
    plt.clf()

def monthly_SO_bkg_trans_boxplot(sim_index, df_stats, dic_monthly_ground_temp, dic_monthly_air_temp, plot_type='box', year_end_bkg=2000, show_monthly_diff_warm=True):
    # INPUT: plot_type: can either take 'box' or 'violin'
    # OUTPUT: boxplot or violinplot of the monthly surface offset (SO) for the background and transient eras
    #         for a SINGLE simulation

    # list of years
    years = np.array(list(dic_monthly_ground_temp[sim_index].keys()))
    # forcing of the given simulation
    forcing = df_stats.loc[sim_index].forcing
    altitude = df_stats.loc[sim_index].altitude
    # SO dictionaries for the background and the transient eras, sorted monthly
    dic_SO_bkg = [[dic_monthly_ground_temp[sim_index][y][m] -
                dic_monthly_air_temp[forcing][altitude][y][m] for y in years if y<year_end_bkg] for m in range(1, 13, 1)]
    dic_SO_trans = [[dic_monthly_ground_temp[sim_index][y][m] -
                dic_monthly_air_temp[forcing][altitude][y][m] for y in years if y>=year_end_bkg] for m in range(1, 13, 1)]
    
    mean_bkg = np.mean(dic_SO_bkg, axis=1)
    mean_trans = np.mean(dic_SO_trans, axis=1)

    ticks = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colorcycle = [u'#1f77b4', u'#ff7f0e']

    if show_monthly_diff_warm:
        plt.figure().set_figwidth(8)
    else:
        plt.figure()

    if plot_type == 'box':
        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)
            plt.setp(bp['means'], markerfacecolor=color)
            plt.setp(bp['means'], markeredgecolor=color)

        bpl = plt.boxplot(dic_SO_bkg, showmeans=True, positions=np.array(range(len(dic_SO_bkg)))*2.0-0.4,
                          sym='', widths=0.6)
        bpr = plt.boxplot(dic_SO_trans, showmeans=True, positions=np.array(range(len(dic_SO_trans)))*2.0+0.4,
                          sym='', widths=0.6)
        set_box_color(bpl, colorcycle[0])
        set_box_color(bpr, colorcycle[1])        

        whiskl = [item.get_ydata()[1] for item in bpl['whiskers']]
        whiskr = [item.get_ydata()[1] for item in bpr['whiskers']]
        whiskmax = [np.max([whiskl[i], whiskr[i]]) for i in range(1,24,2)]

    elif plot_type == 'violin':
        bpl = plt.violinplot(dic_SO_bkg, showmedians=True, positions=np.array(range(len(dic_SO_bkg)))*2.0-0.4, widths=0.6)
        bpr = plt.violinplot(dic_SO_trans, showmedians=True, positions=np.array(range(len(dic_SO_trans)))*2.0+0.4, widths=0.6)
        plt.scatter(np.array(range(len(dic_SO_bkg)))*2.0-0.4, mean_bkg, marker="^", c=colorcycle[0])
        plt.scatter(np.array(range(len(dic_SO_trans)))*2.0+0.4, mean_trans, marker="^", c=colorcycle[1])

        whiskmax = [np.max([dic_SO_bkg[i], dic_SO_trans[i]]) for i in range(12)]

    if show_monthly_diff_warm:
        for i in np.array(range(len(dic_SO_bkg))):
            plt.annotate(r"%s%s" % (("+" if mean_trans[i] - mean_bkg[i] > 0 else ""), float("{:.2f}".format(mean_trans[i] - mean_bkg[i]))),
                            ((np.array(range(len(dic_SO_bkg)))*2.0)[i],whiskmax[i]), xycoords='data',
                            fontsize=12, horizontalalignment='center', verticalalignment='bottom', color=('r' if mean_trans[i] - mean_bkg[i] > 0 else 'g'))

    # draw temporary lines and use them to create a legend
    plt.plot([], c=colorcycle[0], label='Background')
    plt.plot([], c=colorcycle[1], label='Transient')
    plt.legend(loc='lower right')

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylabel('Surface offset [°C]')
    plt.title('Monthly analysis of background and transient SO')
    plt.tight_layout()

    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def monthly_SO_bkg_trans_boxplot_all(dic_SO_monthly, list_valid_sim, plot_type='box', show_monthly_diff_warm=True):
    # INPUT: plot_type: can either take 'box' or 'violin'
    # OUTPUT: boxplot or violinplot of the monthly surface offset (SO) for the background and transient eras
    #         for ALL simulations

    list_monthly_bkg = [[dic_SO_monthly['background'][month][sim][year]
                         for sim in list_valid_sim
                         for year in list(dic_SO_monthly['background'][1][list_valid_sim[0]].keys())]
                        for month in range(1,13,1)]
    list_monthly_trans = [[dic_SO_monthly['transient'][month][sim][year]
                           for sim in list_valid_sim
                           for year in list(dic_SO_monthly['transient'][1][list_valid_sim[0]].keys())]
                          for month in range(1,13,1)]

    mean_bkg = list(np.mean(list_monthly_bkg, axis=1))
    mean_trans = list(np.mean(list_monthly_trans, axis=1))

    ticks = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colorcycle = [u'#1f77b4', u'#ff7f0e']

    if show_monthly_diff_warm:
        plt.figure().set_figwidth(8)
    else:
        plt.figure()

    if plot_type == 'box':
        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)
            plt.setp(bp['means'], markerfacecolor=color)
            plt.setp(bp['means'], markeredgecolor=color)

        bpl = plt.boxplot(list_monthly_bkg, showmeans=True, positions=np.array(range(len(list_monthly_bkg)))*2.0-0.4,
                          sym='', widths=0.6)
        bpr = plt.boxplot(list_monthly_trans, showmeans=True, positions=np.array(range(len(list_monthly_trans)))*2.0+0.4,
                          sym='', widths=0.6)
        set_box_color(bpl, colorcycle[0])
        set_box_color(bpr, colorcycle[1])

        whiskl = [item.get_ydata()[1] for item in bpl['whiskers']]
        whiskr = [item.get_ydata()[1] for item in bpr['whiskers']]
        whiskmax = [np.max([whiskl[i], whiskr[i]]) for i in range(1,24,2)]
        

    elif plot_type == 'violin':
        bpl = plt.violinplot(list_monthly_bkg, showmedians=True, positions=np.array(range(len(list_monthly_bkg)))*2.0-0.4, widths=0.6)
        bpr = plt.violinplot(list_monthly_trans, showmedians=True, positions=np.array(range(len(list_monthly_trans)))*2.0+0.4, widths=0.6)
        plt.scatter(np.array(range(len(list_monthly_bkg)))*2.0-0.4, mean_bkg, marker="^", c=colorcycle[0])
        plt.scatter(np.array(range(len(list_monthly_trans)))*2.0+0.4, mean_trans, marker="^", c=colorcycle[1])

        whiskmax = [np.max([list_monthly_bkg[i], list_monthly_trans[i]]) for i in range(12)]

    if show_monthly_diff_warm:
        for i in np.array(range(len(list_monthly_bkg))):
            plt.annotate(r"%s%s" % (("+" if mean_trans[i] - mean_bkg[i] > 0 else ""), float("{:.2f}".format(mean_trans[i] - mean_bkg[i]))),
                            ((np.array(range(len(list_monthly_bkg)))*2.0)[i],whiskmax[i]), xycoords='data',
                            fontsize=12, horizontalalignment='center', verticalalignment='bottom', color=('r' if mean_trans[i] - mean_bkg[i] > 0 else 'g'))

    # draw temporary lines and use them to create a legend
    plt.plot([], c=colorcycle[0], label='Background')
    plt.plot([], c=colorcycle[1], label='Transient')
    plt.legend(loc='lower right')

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylabel('Surface offset [°C]')
    plt.title('Monthly analysis of background and transient SO')
    plt.tight_layout()

    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def monthly_diff_warming_boxplot_all(dic_SO_monthly, list_valid_sim, plot_type='box'):
    # INPUT: plot_type: can either take 'box' or 'violin'
    # OUTPUT: boxplot or violinplot of the monthly differential warming for ALL simulations

    list_monthly_diff_warm = [[np.mean([dic_SO_monthly['transient'][month][sim][year]
                                        for year in list(dic_SO_monthly['transient'][1][list_valid_sim[0]].keys())]) -
                               np.mean([dic_SO_monthly['background'][month][sim][year]
                                        for year in list(dic_SO_monthly['background'][1][list_valid_sim[0]].keys())])
                               for sim in list_valid_sim]
                              for month in range(1,13,1)]

    mean_monthly_diff_warm = list(np.mean(list_monthly_diff_warm, axis=1))

    ticks = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colorcycle = [u'#1f77b4', u'#ff7f0e']

    plt.figure()

    if plot_type == 'box':
        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)
            plt.setp(bp['means'], markerfacecolor=color)
            plt.setp(bp['means'], markeredgecolor=color)

        bp = plt.boxplot(list_monthly_diff_warm, showmeans=True, positions=np.array(range(len(list_monthly_diff_warm)))*2.0,
                          sym='', widths=0.6)
        set_box_color(bp, colorcycle[0])

    elif plot_type == 'violin':
        plt.violinplot(list_monthly_diff_warm, showmedians=True, positions=np.array(range(len(list_monthly_diff_warm)))*2.0, widths=0.6)
        plt.scatter(np.array(range(len(list_monthly_diff_warm)))*2.0, mean_monthly_diff_warm, marker="^", c=colorcycle[0])

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylabel('Differential warming [°C]')
    plt.title('Monthly analysis of differential warming')
    plt.tight_layout()

    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def plot_mean_annual_snow_depth_variable(df_stats, time_ground, snow_height, all=True, sim_index=None, variable=None, value=None):
    # INPUT: We have the choice to either plot for all simulations or a given one
    #        all: if True -> all simulations, else -> a given one specified by sim_index
    #        sim_index: should be None if all is selected to be True, otherwise, it should be an
    #                   index taken from list_valid_sim
    #        variable: None by default but can be chosen to be 'altitude', 'aspect',
    #                  'slope', 'forcing', 'material', 'snow_corr', etc.
    #                  Can be any column of df_stats
    #        value: legitimate value that the variable can take, for instance 2000 for altitude
    #               ###Note### that 'aspect' is over a full quadrant and hence covers the value +-45°
    #               e.g. we consider NW/N/NE (315/0/45°) when asked for North-facing (0°) slopes
    #        IF all=True: sim_index should be None and the pair (variable, value) should be populated
    #        IF all=False: sim_index should have a value but not the pair (variable, value)
    #        Finally, one can output ALL simulations by simply running:
    #        plot_mean_annual_snow_depth_variable(df_stats, time_ground, snow_height)
    # OUTPUT: Plots all background snow depth time series in the form of a mean surrounded by an envelope
    #         at +-2 standard deviations, hence a 95% confidence interval. Same for the transient period.

    # we start by creating a list of time stamps sorted by years
    list_year = list_tokens_year(time_ground)[0]

    # we start by creating a list of time stamps sorted by years
    list_sim = (list(subsets_df_stats(df_stats, variable, value).index) if all==True else [sim_index])

    # we aggregate all background (and then transient) snow depth data and make sure they all have the same length for simplicity
    # since the ground and snow data start on Jan 2nd 1980, we create a Jan 1st 1980 data and then limit all series to 365 points
    bkg_time_series = [ma.append(snow_height[sim_index, list_year[1980]][0],snow_height[sim_index, list_year[1980]][:364]) if year==1980
                       else snow_height[sim_index, list_year[year][:365]]
                       for year in range(1980, 2000)
                       for sim_index in list_sim]
    trans_time_series = [snow_height[sim_index, list_year[year][:365]] for year in range(2000, 2020) for sim_index in list_sim]

    # here we take the mean and standard deviation of both datasets
    bkg_mean = np.mean(bkg_time_series, axis=0)
    bkg_std = np.std(bkg_time_series, axis=0)
    trans_mean = np.mean(trans_time_series, axis=0)
    trans_std = np.std(trans_time_series, axis=0)

    colorcycle = [u'#1f77b4', u'#ff7f0e']

    # finally, we can plot the mean and the 95% confidence envelope
    plt.plot(range(365), bkg_mean, c=colorcycle[0], label='Background')
    plt.fill_between(range(365), np.max([np.zeros(365), bkg_mean - 2*bkg_std], axis=0), bkg_mean + 2*bkg_std, color=colorcycle[0], alpha=.2)
    plt.plot(range(365), trans_mean, c=colorcycle[1], label='Transient')
    plt.fill_between(range(365), np.max([np.zeros(365), trans_mean - 2*trans_std], axis=0), trans_mean + 2*trans_std, color=colorcycle[1], alpha=.2)

    locs, labels = plt.xticks()
    labels = [date.strftime('%m') for date in num2date(locs, time_ground.units)]
    plt.xticks(locs, labels)
    warnings.filterwarnings("ignore") 

    plt.xlabel('Date')
    plt.ylabel('Snow height [mm]')
    units = {'altitude': 'm', 'aspect': '°', 'slope': '°', 'forcing': ''}
    # plt.title(f'Snow height data at Joffre{('' if variable==None else f' for {variable} = {value}{units[variable]}')}')


    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def mean_all_reanalyses(time_files, files_to_smooth):
    """ Function returns the mean time series over a number of reanalysis 
    
    Parameters
    ----------
    time_file : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored
    file_to_smooth : list of list
        List ime series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that it needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts [temp_air_era5[:,0]] but not [temp_air_era5]

    Returns
    -------
    mean : dict
        average time series over reanalyses (3 or less)
    """

    list_years = [list_tokens_year(time_files[i], year_bkg_end=2000, year_trans_end=2020)[0] for i in range(len(time_files))]
    len_years = {y: [len(list_years[i][y]) for i in range(len(time_files))] for y in list(list_years[0].keys()) if y<2020}

    mean = []

    # presumably this means that y goes from 1980 to 2019
    for y in list(len_years.keys()):
        # check if all lists have the same amount of indices in a given year
        order = sorted(len_years[y])
        min_len = order[0]
        max_len = order[-1]
        if min_len == max_len:
            # if True, then we can proceed to computing the mean
            mean += list(np.mean([files_to_smooth[i][list_years[i][y]] for i in range(len(time_files))], axis=0))
        else:
            for j in range(len(order)):
                mean += list(np.mean([files_to_smooth[i][list_years[i][y]][(order[j-1] if j>0 else 0):order[j]]
                                        for i in range(len(time_files)) if len(files_to_smooth[i][list_years[i][y]]) > (order[j-1] if j>0 else 0)], axis=0))

    return mean

def running_mean_std(time_file, file_to_smooth, window, fill_before=False):
    """ Function returns a running average (and standard deviation) of the time series over a given time window
        adpated from Welford's online algorithm, computing everything in a single-pass 
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored
    file_to_smooth : list
        Time series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that it needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts temp_air_era5[:,0] but not temp_air_era5
    window : str
        Time window of datapoints to average over.
        Can take the following arguments: 'day', 'week', 'year', 'climate'
    fill_before : bool, optional
        Option to fill the first empty values of the smoothed data with the first running average value if True

    Returns
    -------
    running_mean : dict
        Smoothed time series
    running_std : dict
        Running standard deviation

    """

    # number of hours between consecutive measurements: 1 for air, 24 for ground
    cons_hours = int((num2date(time_file[1], time_file.units) - num2date(time_file[0], time_file.units)).total_seconds()/3600)
    window_size = int({'day': 24, 'week': 24*7, 'month': 24*30, 'year':24*365, 'climate':24*35*20}[window]/cons_hours)

    
    # range of the timestamp whether the user chose a given year of the whole study period
    index_range = list(range(len(file_to_smooth)))

    # initialize the running mean, the first value is at (window_size-1+index_range[0])
    # the initial window extends from index_range[0] to window_size-1+index_range[0] and has (window_size) elements
    init = np.sum(file_to_smooth[index_range[0]:window_size+index_range[0]])/window_size
    running_mean = {}
    running_std = {}

    # fill the first values if fill_before==True
    if fill_before:
        for i in range(index_range[0], window_size-1+index_range[0]):
            running_mean[i] = init
            # running_std[i] = 0
    
    running_mean[window_size-1+index_range[0]] = init
    # running_std[window_size-1+index_range[0]] = np.std(file_to_smooth[index_range[0]:window_size+index_range[0]])

    # the following values are evaluated recursively in a single-pass (more efficient)
    # I have derived the running version of the Welford's online algorithm and tested it.
    for i in range(window_size+index_range[0], index_range[-1]+1):
        running_mean[i] = running_mean[i-1] + (file_to_smooth[i] - file_to_smooth[i-window_size])/window_size
        # running_std[i] = np.sqrt( (window_size * running_std[i-1]**2
        #                            + (file_to_smooth[i] - file_to_smooth[i-window_size])**2/window_size
        #                            + (file_to_smooth[i] - running_mean[i])**2
        #                            - (file_to_smooth[i-window_size] - running_mean[i])**2) / window_size )

    return running_mean, running_std

def aggregating_distance_temp(time_file, file_to_smooth, window, year=0, fill_before=False):
    """ Function returns the distance to the mean in units of standard deviation for a given date yy-mm-dd
        relative to the sensemble of same date for previous years: {year-mm-dd for year<=yy}
        Should only be applied to a previously 'smoothed' function through a running average
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored        
    file_to_smooth : list
        Time series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that it needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts temp_air_era5[:,0] but not temp_air_era5
    window : str
        Time window of datapoints to average over.
        Can take the following arguments: 'day', 'week', 'year', 'climate'
    year : int, optional
        One can choose to display a specific year or the whole set by choosing an integer not in the study sample, e.g. 0.
    fill_before : bool, optional
        Option to fill the first empty values of the smoothed data with the first running average value if True

    Returns
    -------
    distance : dict
        Distance from mean in units of standard deviation

    """

    # here temp_smoothed is the smoothed temperature data series in the form of a dictionary 
    # keys are timestamps (end of window) and values are running average of temperatures over window
    temp_smoothed = running_mean_std(time_file, file_to_smooth, window, fill_before)[0]

    list_dates = list_tokens_year(time_file, year_bkg_end=2000, year_trans_end=2020)[0]

    num_min = np.min([len(list_dates[year]) for year in list_dates.keys() if year < 2020])
    min_temp_index = list(temp_smoothed.keys())[0]

    # range of the timestamp whether the user chose a given year of the whole study period
    index_range = list(range(len(file_to_smooth)))

    aggregate_mean = {}
    aggreagte_std = {}
    distance = {}

    for year in [1990]:
        aggregate_mean[year] = {day: np.mean([temp_smoothed[list_dates[y][day]] for y in range(1980,year+1) if list_dates[y][day]>=min_temp_index]) for day in range(num_min)}
        aggreagte_std[year] = {day: np.std([temp_smoothed[list_dates[y][day]] for y in range(1980,year+1) if list_dates[y][day]>=min_temp_index]) for day in range(num_min)}
        distance[year] = {day: (temp_smoothed[list_dates[year][day]]-aggregate_mean[year][day])/aggreagte_std[year][day] for day in range(num_min)}

    for year in range(1991, 2020):
        aggregate_mean[year] = {day: aggregate_mean[year-1][day]
                                     + (temp_smoothed[list_dates[year][day]] - aggregate_mean[year-1][day]) / (year-1980+1)
                                     for day in range(num_min)}
        aggreagte_std[year] = {day: np.sqrt( ( (year-1980) * aggreagte_std[year-1][day]**2
                                              + (temp_smoothed[list_dates[year][day]] - aggregate_mean[year][day])
                                              *(temp_smoothed[list_dates[year][day]] - aggregate_mean[year-1][day]) )
                                              / (year-1980+1) )
                                    for day in range(num_min)}
        distance[year] = {day: (temp_smoothed[list_dates[year][day]]-aggregate_mean[year][day])/aggreagte_std[year][day] for day in range(num_min)}


    return distance

def plot_aggregating_distance_temp(name_series, time_file, file_to_smooth, window, site, year=0, fill_before=False):
    """ Plots the distance to the mean in units of standard deviation for a specific year or for the whole length
    
    Parameters
    ----------
    name_series : str
        Name of the quantity to plot, has to be 'air temperature', precipitation', 'snow depth'
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored        
    file_to_smooth : list
        Time series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that it needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts temp_air_era5[:,0] but not temp_air_era5
    window : str
        Time window of datapoints to average over.
        Can take the following arguments: 'day', 'week', 'year', 'climate'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    year : int, optional
        One can choose to display a specific year or the whole set by choosing an integer not in the study sample, e.g. 0.
    fill_before : bool, optional
        Option to fill the first empty values of the smoothed data with the first running average value if True

    Returns
    -------
        plot

    """

    distance = aggregating_distance_temp(time_file, file_to_smooth, window, year, fill_before)
    list_dates = list_tokens_year(time_file, year_bkg_end=2000, year_trans_end=2020)[0]

    if year in list(list_dates.keys()):
        index_range = list_dates[year]
    else:
        index_range = list(range(len(file_to_smooth)))[list_dates[np.min(list(distance.keys()))][0]:]

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']

    plt.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
    if year in list(list_dates.keys()):
        plt.plot(time_file[list_dates[year][:][:len(distance[year].values())]], distance[year].values(), label='Distance')
    else:
        for y in list(distance.keys()):
            plt.plot(time_file[list_dates[y][:][:len(distance[y].values())]], distance[y].values(), label=('Deviation' if y==1990 else ''), color= colorcycle[0])

    for indx in rockfall_values(site)['time_index']:
        if indx in index_range:
            if int((num2date(time_file[indx], time_file.units)-rockfall_values(site)['datetime']).total_seconds()) == 0:
                plt.axvline(x = time_file[indx], color = 'r', linestyle='--', label = 'Landslide')

    for i in range(-3,4):
        plt.axhline(y = i, color = 'black', linestyle='--', linewidth=(1 if i ==0 else 0.5))
    # Because the data is stored in seconds since 0001-01-01 00:00:00 (ugh!!) 
    # we have to do some formatting of the x bar
    locs, labels = plt.xticks()

    if year in list(list_dates.keys()):
        locs = np.linspace(time_file[index_range[0]], time_file[index_range[-1]], num=12, endpoint=False)
        labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    else:
        locs = np.linspace(time_file[index_range[0]], time_file[list_dates[2020][0]], num=11, endpoint=True)
        labels = list(range(1990,2021,3))
    plt.xticks(locs, labels)

    plt.title(f"Deviation from {name_series} data at %s%s" % (f"{site}", f" in {year}" if year in list(list_dates.keys()) else ""))
    plt.xlabel('Date')
    plt.ylabel((f'{name_series} deviation [$\sigma$]').capitalize())
    plt.legend()
    plt.show()

    # Closing figure
    plt.close()
    plt.clf()

def plot_aggregating_distance_temp_mean_reanalyses(name_series, time_files, files_to_smooth, window, year=0, fill_before=False):

    mean_plot = mean_all_reanalyses(time_files, files_to_smooth)

    plot_aggregating_distance_temp(name_series, time_files[0], mean_plot, window, year, fill_before)

def plot_aggregating_distance_temp_all(yaxes, xdata, ydata, window, site, year, fill_before=False):
    """ Plots the distance to the mean in units of standard deviation for a specific year or for the whole length
        Vertical subplots for different variables
    
    Parameters
    ----------
    yaxes : list of str
        List of names of quantities to plot, e.g. ['Air temperature', 'Water production', 'Ground temperature', 'Depth of thaw']
        Will be used for labels
    xdata : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored, e.g. [time_air_era5, time_air_era5, time_ground, time_ground]
        len(xdata) should be equal to len(yaxes) 
    ydata : list of list
        List of time series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that each individual time series needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts temp_air_era5[:,0] but not temp_air_era5
        example: [mean_air_temp, mean_air_temp, temp_ground_mean, temp_ground_mean]
        len(ydata) should be equal to len(yaxes)
    window : list of str
        Time window of datapoints to average over.
        Can take the following arguments: 'day', 'week', 'year', 'climate'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    year : int, optional
        One can choose to display a specific year or the whole set by choosing an integer not in the study sample, e.g. 0.
    fill_before : bool, optional
        Option to fill the first empty values of the smoothed data with the first running average value if True

    Returns
    -------
        plot containing len(yaxes) subplots sharing the x axis

    """
    num_rows = len(xdata)
    num_cols = len(window)

    f, a = plt.subplots(num_rows, num_cols, figsize=(8, 2*num_rows), sharey='row')
    for idx,ax in enumerate(a):
        distance = [aggregating_distance_temp(xdata[idx], ydata[idx], i, year, fill_before) for i in window]
        list_dates = list_tokens_year(xdata[idx], year_bkg_end=2000, year_trans_end=2020)[0]

        if year in list(list_dates.keys()):
            index_range = list_dates[year]
        else:
            index_range = list(range(len(ydata[idx])))[list_dates[np.min(list(distance[0].keys()))][0]:]

        colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']

        if year in list(list_dates.keys()):
            if num_cols == 1:
                ax.plot(xdata[idx][list_dates[year][:][:len(distance[0][year].values())]], distance[0][year].values(), label='Distance')
            else:
                for i in range(num_cols):
                    ax[i].plot(xdata[idx][list_dates[year][:][:len(distance[i][year].values())]], distance[i][year].values(), label='Distance')
        else:
            if num_cols == 1:
                for y in list(distance[0].keys()):
                    ax.plot(xdata[idx][list_dates[y][:][:len(distance[0][y].values())]], distance[0][y].values(), label=('Deviation' if y==1990 else ''), color= colorcycle[0])
            else:
                for i in range(num_cols):
                    for y in list(distance[i].keys()):
                        ax[i].plot(xdata[idx][list_dates[y][:][:len(distance[i][y].values())]], distance[i][y].values(), label=('Deviation' if y==1990 else ''), color= colorcycle[0])

        for indx in rockfall_values(site)['time_index']:
            if indx in index_range:
                if int((num2date(xdata[idx][indx], xdata[idx].units)-rockfall_values(site)['datetime']).total_seconds()) == 0:
                    if num_cols == 1:
                        ax.axvline(x = xdata[idx][indx], color = 'r', linestyle='--', label = 'Landslide')
                    else:
                        for i in range(num_cols):
                            ax[i].axvline(x = xdata[idx][indx], color = 'r', linestyle='--', label = 'Landslide')
        if num_cols == 1:
            ax.axhline(y = 0, color = 'black', linestyle='--', linewidth=1)
        else:
            for i in range(num_cols):
                ax[i].axhline(y = 0, color = 'black', linestyle='--', linewidth=1)

        if year in list(list_dates.keys()):
            locs = np.linspace(xdata[idx][index_range[0]], xdata[idx][index_range[-1]], num=12, endpoint=False)
            if num_cols == 1:
                labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            else:
                labels = ['Jan','','Mar','','May','','Jul','','Sep','','Nov','']
        else:
            locs = np.linspace(xdata[idx][index_range[0]], xdata[idx][list_dates[2020][0]], num=11, endpoint=True)
            labels = list(range(1990,2021,3))

        if idx < num_rows-1:
            labels_end = ['']*len(labels)
        else:
            labels_end = labels
        if num_cols == 1:
            ax.set_xticks(locs, labels_end)
            ax.set_ylabel(yaxes[idx])
            ax.grid(axis = 'y')
        else: 
            for i in range(num_cols):
                ax[i].set_xticks(locs, labels_end)
                ax[i].grid(axis = 'y')
                if idx == 0:
                    ax[i].set_title(window[i].capitalize())
            ax[0].set_ylabel(yaxes[idx])
            ax[1].yaxis.set_tick_params(labelleft=False)

    f.supylabel(r'Normalized deviation $d_{norm}$ [$\sigma$]')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.clf()

def assign_weight_sim(df_stats, site):
    """ Function returns a statistical weight for each simulation according to the importance in rockfall starting zone 
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda DataFrame df_stats, should at least include the following columns: 'altitude', 'aspect', 'slope'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'

    Returns
    -------
    pd_weight : pandas.core.frame.DataFrame
        Panda DataFrame assigning a statistical weight to each simulation for each of 'altitude', 'aspect', 'slope'
        and an overall weight.
    """

    dict_weight = {}
    alt_distance = np.max([np.abs(i-rockfall_values(site)['altitude']) for i in np.sort(np.unique(df_stats['altitude']))])
    dict_weight = {i: [1 - np.abs(df_stats.loc[i]['altitude']-rockfall_values(site)['altitude'])/(2*alt_distance),
                       np.cos((np.pi)/180*(df_stats.loc[i]['aspect']-rockfall_values(site)['aspect']))/4+3/4,
                       np.cos((np.pi)/30*(df_stats.loc[i]['slope']-rockfall_values(site)['slope']))/4+3/4]
                       for i in list(df_stats.index.values)}
    pd_weight = pd.DataFrame.from_dict(dict_weight, orient='index',
                                       columns=['altitude', 'aspect', 'slope'])
    pd_weight['weight'] = pd_weight['altitude']*pd_weight['aspect']*pd_weight['slope']
    
    return pd_weight

def assign_tot_water_prod(swe_mean, mean_prec, time_ground, time_pre_trans_ground, time_air_era5):
    """ Function returns the total water production at daily intervals in [mm s-1] 
    
    Parameters
    ----------
    swe_mean : list
        Mean swe over all 3 reanalyses
    mean_prec : list
        Mean precipitation over all simulations (weighted average)
    time_ground : netCDF4._netCDF4.Variable
        Time series for the ground timestamps
    time_pre_trans_ground : numpy.ma.core.MaskedArray
        Selects only the timestamps before the cutoff of the end of the transient era
    time_air_era5 : netCDF4._netCDF4.Variable
        Time series for the air timestamps

    Returns
    -------
    list of daily total water production
    """

    # convert mean precipitation into a panda dataframe to facilitate grouping by 24.
    # This way the data is reported daily (as it is for the swe) rather than hourly
    pd_prec_day = ((pd.DataFrame(mean_prec).rolling(24).mean()).iloc[23::24, :]).reset_index(drop=True)

    # convert swe into panda df to make the diff() operation easier. Swe is a total and
    # we get the instantaneous flux by taking the difference between consecutive datapoints.
    # The total precipitation between 2 datapoints is over 1 day = 864000 hence we divide by 86400.
    # Finally, swe is in [kg m-2] which is equivalent to [mm] so we're good.

    # Number of days between the start of the ground and air timeseries
    num_day = int((num2date(time_ground[0], time_ground.units) - num2date(time_air_era5[0], time_air_era5.units)).total_seconds()/86400)
    # we add num_days zeros at the beginning to make sure the dfs have the same dimension and span Jan 1, 1980 to Dec 31, 2019
    # We make sure to multiply by (-1) and to divide by 86400
    pd_diff_swe = pd.concat([pd.DataFrame([0]*num_day), (pd.DataFrame(swe_mean[:len(time_ground[time_pre_trans_ground][:])]).diff()).fillna(0)/86400*(-1)], ignore_index=True)

    if len(pd_prec_day)!=len(pd_diff_swe):
        raise ValueError('VictorCustomError: The lengths of the precipitation and swe time series are different!')

    # The total water production is returned in the list type as the sum of precipitation and swe
    tot_water_prod = list(pd_prec_day.add(pd_diff_swe.reset_index(drop=True), fill_value=0)[0])

    return tot_water_prod

def stat_model_aspect_slope_alt(X, offset, c_alt, d_alt, c_asp, c_slope):
    """ Function returns the value of the statistical model 
    
    Parameters
    ----------
    X : list
        List of aspects and altitudes
    offset, c_asp, d_asp, c_slope, d_slope, c_alt : floats
        coefficients of the model

    Returns
    -------
    Value output of the model given the input
    """

    # This is the statistical model we are trying to fit to the data.
    # unpack the variables in X
    aspect, slope, altitude = X
    return (offset
            + c_alt * altitude
            + c_asp * (altitude - d_alt) * np.cos(aspect * 2 * np.pi / 360) 
            + c_slope * slope)

def fit_stat_model_grd_temp(df_stats, all=True, diff_forcings=True):
    """ Function returns the value of the statistical model 
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'forcing', 'aspect', 'slope', 'bkg_grd_temp'
    all : bool, optional
        If True, considers all data at once
    diff_forcings : bool, optional
        If True, separates data by 'forcing'

    Returns
    -------
    xdata : list
        List of xdata (actual) grouped by forcing if all=False
    ydata : list
        List of ydata (predicted) grouped by forcing if all=False
    optimizedParameters : list
        List of optimized model parameters grouped by forcing if all=False
    pcov : list
        List of covariances grouped by forcing if all=False
    corr_matrix : list
        List of correlation matrices grouped by forcing if all=False
    R_sq : list
        List of R^2 grouped by forcing if all=False
    parity plot (predicted vs actual)
    """
    
    plt.figure(figsize=(6,6))

    forcings = ['era5', 'merra2', 'jra55']
    data_set = []
    if all:
        data_set.append(df_stats)
    else: 
        pass
    if diff_forcings:
        for i in forcings:
            data_set.append(df_stats[df_stats['forcing']==i])
    else:
        pass

    input = [np.array([i['aspect'], i['slope'], i['altitude']]) for i in data_set]
    # all the measured differential warmings (from valid simulations) are in xdata
    xdata = [np.array(i['bkg_grd_temp']) for i in data_set]
    
    # The actual curve fitting happens here
    ydata = []
    R_sq = []
    optimizedParameters = []
    pcov = []
    corr_matrix = []
    bounds=((-50, -np.inf, -np.inf, -np.inf, -np.inf), (50, np.inf, np.inf, np.inf, np.inf))
    p0 = (0,0,1000,0,0)
    for i in range(len(input)):
        optimizedParameters.append(opt.curve_fit(stat_model_aspect_slope_alt, input[i], xdata[i], bounds=bounds, p0=p0)[0])
        pcov.append(opt.curve_fit(stat_model_aspect_slope_alt, input[i], xdata[i], bounds=bounds, p0=p0)[0])

        # this represents the fitted values, hence they are the statistically-modelled values 
        # of differential warming: we call them ydata
        ydata.append(stat_model_aspect_slope_alt(input[i], *optimizedParameters[i]))

        # R^2 from numpy package, to check!
        corr_matrix.append(np.corrcoef(xdata[i], ydata[i]))
        corr = corr_matrix[i][0,1]
        R_sq.append(corr**2)

        plt.scatter(xdata[i], ydata[i], marker=("D" if (all and i==0) else "o"),
                    s=20, label=('all data' if (all and i==0) else forcings[3-len(data_set)+i]) )
        
    # plot the y=x diagonal
    # start by setting the bounds
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    lim_up = float("{:.2g}".format(max(np.max([np.max(i) for i in xdata]), np.max([np.max(i) for i in ydata]))))
    lim_down = float("{:.2g}".format(min(np.min([np.min(i) for i in xdata]), np.min([np.min(i) for i in ydata]))))
    x = np.arange(lim_down, lim_up, 0.01)
    plt.plot(x, x, color=colorcycle[len(data_set)], linestyle='dashed', label = 'y=x', linewidth=2)
    plt.legend(loc='upper right')

    margin = 0.1
    plt.ylim(ymin= lim_down - margin, ymax= lim_up + margin)
    plt.xlim(xmin= lim_down - margin, xmax= lim_up + margin)

    plt.xlabel(r'Numerically-simulated background GST $\overline{T_{\rm GST}^{\rm bkg}}_{(NS)}$ [°C]')
    plt.ylabel(r'Statistically-modelled background GST $\overline{T_{\rm GST}^{\rm bkg}}_{(SM)}$ [°C]')
    for i in range(len(R_sq)):
        plt.figtext(.7, .3 - i/30, f"$R^2$ = %s" % float("{:.2f}".format(R_sq[i])),
                    c=colorcycle[i])

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()

    return xdata, ydata, optimizedParameters, pcov, corr_matrix, R_sq

def plot_stat_model_marginal_distrib_grd_temp(df_stats, all=True):
    """ Function returns the value of the statistical model 
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'forcing', 'aspect', 'slope', 'kg_grd_temp'
    all : bool, optional
        If True, considers all data at once, if False, separates by 'forcing'

    Returns
    -------
    parity plot (predicted vs actual) with histograms and marginals
    """
    # xdata is simulated
    xdata, ydata, optimizedParameters, pcov, corr_matrix, R_sq = fit_stat_model_grd_temp(df_stats, all)

    # Start with a square Figure.
    fig = plt.figure(figsize=(8, 8))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    # histx is for simulation
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    # histy is for model
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    forcings = ['era5', 'merra2', 'jra55']
    for i in range(len(xdata)):
        ax.scatter(xdata[i], ydata[i],
                    s=25, label=('all data' if all else forcings[i]) )

    all_xdata = [item for sub_list in xdata for item in sub_list]
    all_ydata = [item for sub_list in ydata for item in sub_list]

    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = max(np.max(np.abs(all_xdata)), np.max(np.abs(all_ydata)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728']

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(all_xdata, bins=bins, alpha=0.5, color=(colorcycle[0] if all else 'grey'))
    ax_histy.hist(all_ydata, bins=bins, orientation='horizontal', alpha=0.5, color=(colorcycle[0] if all else 'grey'))

    # histx is for simulation
    ax_histx.set_ylabel('Simulation count')
    # histy is for model
    ax_histy.set_xlabel('Model count')

    # we compute the mean and standard deviation for the simulated differential warming
    D_sim = [np.mean(all_xdata), np.std(all_xdata)]
    # and for the statistically-modelled one
    D_mod = [np.mean(all_ydata), np.std(all_ydata)]

    color_lines = (colorcycle[0] if all else 'black')

    # simulation data
    ax_histx.axvline(D_sim[0], linestyle='dashed', linewidth=2, color=color_lines)
    ax_histx.axvline(D_sim[0] + D_sim[1], linestyle='dashed', linewidth=1, color=color_lines)
    ax_histx.axvline(D_sim[0] - D_sim[1], linestyle='dashed', linewidth=1, color=color_lines)
 
    ax.axvline(D_sim[0], linestyle='dashed', linewidth=2, color=color_lines)
    ax.axvline(D_sim[0] + D_sim[1], linestyle='dashed', linewidth=1, color=color_lines)
    ax.axvline(D_sim[0] - D_sim[1], linestyle='dashed', linewidth=1, color=color_lines)
    
    # model data
    ax_histy.axhline(D_mod[0], linestyle='dashed', linewidth=2, color=color_lines)
    ax_histy.axhline(D_mod[0] + D_mod[1], linestyle='dashed', linewidth=1, color=color_lines)
    ax_histy.axhline(D_mod[0] - D_mod[1], linestyle='dashed', linewidth=1, color=color_lines)
 
    ax.axhline(D_mod[0], linestyle='dashed', linewidth=2, color=color_lines)
    ax.axhline(D_mod[0] + D_mod[1], linestyle='dashed', linewidth=1, color=color_lines)
    ax.axhline(D_mod[0] - D_mod[1], linestyle='dashed', linewidth=1, color=color_lines)

    ax.set_xlabel('Simulated background GST [°C]')
    ax.set_ylabel('Statistically-modelled background GST [°C]')

    for i in range(len(R_sq)):
        plt.figtext(.6, .25 - i/40, f"$R^2$ = %s" % float("{:.2f}".format(R_sq[i])),
                    c=('black' if all else colorcycle[i]))

    # simulation data
    annotation_string_x = r"$\overline{T_g^{\rm sim}}$ = %.2f°C" % (D_sim[0]) 
    annotation_string_x += "\n"
    annotation_string_x += r"$\sigma(T_g^{\rm sim})$ = %.2f°C" % (D_sim[1]) 
    ax_histx.annotate(annotation_string_x, (0.05, 0.75), xycoords='axes fraction', va='center')

    # model data
    annotation_string_y = r"$\overline{T_g^{\rm mod}}$ = %.2f°C" % (D_mod[0]) 
    annotation_string_y += "\n"
    annotation_string_y += r"$\sigma(T_g^{\rm mod})$ = %.2f°C" % (D_mod[1]) 
    ax_histy.annotate(annotation_string_y, (0.05, 0.9), xycoords='axes fraction', va='center')

    # plot the y=x diagonal
    # start by setting the bounds
    lim_up = float("{:.2g}".format(max(np.max(all_xdata), np.max(all_ydata))))
    lim_down = float("{:.2g}".format(min(np.min(all_xdata), np.min(all_ydata))))
    x = np.arange(lim_down, lim_up, 0.01)
    ax.plot(x, x, color=colorcycle[3], linestyle='dashed', label = 'y=x')
    ax.legend(loc='upper left')

    margin = 0.1
    ax.set_ylim(ymin= lim_down - margin, ymax= lim_up + margin)
    ax.set_xlim(xmin= lim_down - margin, xmax= lim_up + margin)

    # Show the graph
    plt.show()
    plt.close()
    plt.clf()

def rockfall_values(site):
    """ Function returns a dictionary of the topography and other details of the rockfall event at 'site'
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'

    Returns
    -------
    Dictionary
    """

    rockfall_values = {'Joffre': {'aspect': 45, 'slope': 60, 'altitude': 2500, 'year': 2019,
                                  'datetime': datetime(2019, 5, 13, 0, 0, 0, 0), 'time_index': [14376, 345043, 345048]},
                       'Fingerpost': {'aspect': 270, 'slope': 50, 'altitude': 2600, 'year': 2015,
                                      'datetime': datetime(2015, 12, 16, 0, 0, 0, 0), 'time_index': [13132, 315187, 315192]},
                       'Dawson': {'aspect': 90, 'slope': 40, 'altitude': 500, 'year': 2015,
                                      'datetime': datetime(2015, 12, 16, 0, 0, 0, 0), 'time_index': [13132, 315187, 315192]},
                       'Aksaut_North': {'year': 2021,
                                      'datetime': datetime(2021, 12, 31, 0, 0, 0, 0), 'time_index': [8034, 192840]},
                       'Aksaut_North_test_no_SWEtop': {'year': 2021,
                                      'datetime': datetime(2021, 12, 31, 0, 0, 0, 0), 'time_index': [8034, 192840]},
                       'Aksaut_North_test_no_SnowSMIN': {'year': 2021,
                                      'datetime': datetime(2021, 12, 31, 0, 0, 0, 0), 'time_index': [8034, 192840]},
                       'Aksaut_North_slope_scf': {'year': 2021,
                                      'datetime': datetime(2021, 12, 31, 0, 0, 0, 0), 'time_index': [8034, 192840]},
                       'Aksaut_North_LWin': {'year': 2021,
                                      'datetime': datetime(2021, 12, 31, 0, 0, 0, 0), 'time_index': [8034, 192840]},
                       'Aksaut_North_spin_up': {'year': 2021,
                                      'datetime': datetime(2021, 12, 31, 0, 0, 0, 0), 'time_index': [8034, 192840]},
                       'Aksaut_Ridge': {'year': 2021,
                                      'datetime': datetime(2021, 12, 31, 0, 0, 0, 0), 'time_index': [8034, 192840]},
                       'Aksaut_South': {'year': 2021,
                                      'datetime': datetime(2021, 12, 31, 0, 0, 0, 0), 'time_index': [8034, 192840]},
                       'Aksaut_South_slope_scf': {'year': 2021,
                                      'datetime': datetime(2021, 12, 31, 0, 0, 0, 0), 'time_index': [8034, 192840]}}

    return rockfall_values[site]

def table_background_evolution_mean_GST_aspect_slope(df_stats):
    """ Function returns a table of mean background and evolution of GST (ground-surface temperature)
        as a function of slope, aspect, and altitude
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
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

def table_evolution_mean_GST_aspect_slope(df_stats):
    """ Function returns a table of the evolution of mean GST (ground-surface temperature)
        as a function of slope, aspect, and altitude
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
    Returns
    -------
    list_diff_temp : list
        List of the evolution of mean GST per cell of given altitude, aspect, and slope
    list_mean_diff_temp : list
        Average the evolution of mean GST over all simulations in that cell
    list_num_sim : list
        Number of valid simulation per cell, returns NaN if different number of simulation per forcing for that cell
    """
    
    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}
    forcings = list(np.unique(df_stats['forcing']))

    list_diff_temp = [[[0 for aspect in dic_var['aspect']] for slope in dic_var['slope']] for altitude in dic_var['altitude']]
    list_num_sim = [[[0 for aspect in dic_var['aspect']] for slope in dic_var['slope']] for altitude in dic_var['altitude']]
    list_mean_diff_temp = [[[0 for aspect in dic_var['aspect']] for slope in dic_var['slope']] for altitude in dic_var['altitude']]

    for altitude in range(len(dic_var['altitude'])):
        for slope in range(len(dic_var['slope'])):
            for aspect in range(len(dic_var['aspect'])):
                list_diff_temp[altitude][slope][aspect] = list(df_stats[(df_stats['aspect']==dic_var['aspect'][aspect]) & (df_stats['slope']==dic_var['slope'][slope]) & (df_stats['altitude']==dic_var['altitude'][altitude])]['trans_grd_temp'] -
                                                               df_stats[(df_stats['aspect']==dic_var['aspect'][aspect]) & (df_stats['slope']==dic_var['slope'][slope]) & (df_stats['altitude']==dic_var['altitude'][altitude])]['bkg_grd_temp'])
                list_sim_per_forcing = [list(df_stats[(df_stats['aspect']==dic_var['aspect'][aspect]) & (df_stats['slope']==dic_var['slope'][slope]) & (df_stats['altitude']==dic_var['altitude'][altitude])]['forcing']).count(i) for i in forcings]
                list_num_sim[altitude][slope][aspect] = (np.sum(list_sim_per_forcing) if len(set(list_sim_per_forcing))==1 else np.nan)
                list_mean_diff_temp[altitude][slope][aspect] = round(np.mean((list_diff_temp[altitude][slope][aspect])),3)

    return list_diff_temp, list_mean_diff_temp, list_num_sim

def plot_table_mean_GST_aspect_slope(df_stats, site, altitude, background=True):
    """ Function returns a plot of the table of either mean background GST (ground-surface temperature)
        or its evolution vetween the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    altitude : int
        Altitude at which we want the table
    background : bool, optional 
        If True, plots the mean background value, else, plots the evolution of the mean


    Returns
    -------
    Table
    """
    
    rf_values = rockfall_values(site)
    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}
    
    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
       if altitude == j:
          alt_index = i

    list_mean = (table_background_evolution_mean_GST_aspect_slope(df_stats)[1][alt_index] if background else table_background_evolution_mean_GST_aspect_slope(df_stats)[3][alt_index])
    df_temp = pd.DataFrame(list_mean, index=list(dic_var['slope']), columns=list(dic_var['aspect']))

    vals = np.around(df_temp.values, 2)
    vals = vals[~np.isnan(vals)]
    dilute = 1
    min_vals = np.min(vals)
    max_vals = np.max(vals)
    range_vals = max_vals- min_vals
    normal = plt.Normalize((np.max(max_vals - 2*dilute*range_vals,0) if np.min(vals)>0 else -dilute*np.max(np.abs(vals))), dilute*np.max(np.abs(vals)))
    colours = plt.cm.seismic(normal(df_temp))

    for i in range(len(list_mean)):
        for j in range(len(list_mean[0])):
            if np.isnan(list_mean[i][j]):
                colours[i][j] = list(colors.to_rgba('silver'))

    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    the_table=plt.table(cellText=list_mean, rowLabels=df_temp.index, colLabels=df_temp.columns, 
                        loc='center', cellLoc='center',
                        cellColours=colours)
    
    if altitude == rockfall_values(site)['altitude']:
        ax.add_patch(Rectangle(((rf_values['aspect'])/45*1/8, (70-rf_values['slope'])/10*1/6), 1/8, 1/6,
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
    plt.show()
    plt.close()
    plt.clf()

def plot_table_aspect_slope_all_altitudes(df_stats, site):
    """ Function returns 3 plots (1 per altitude) of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'


    Returns
    -------
    2*3 tables
    """
    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
       if rockfall_values(site)['altitude'] == j:
          alt_index = i

    list_mean = [table_background_evolution_mean_GST_aspect_slope(df_stats)[1], table_background_evolution_mean_GST_aspect_slope(df_stats)[3]]
    data = [[pd.DataFrame(i, index=list(dic_var['slope']), columns=list(dic_var['aspect'])) for i in j] for j in list_mean]

    # setting the parameter values 
    annot = True
    center = 0
    cmap='seismic'

    nrows=2
    ncols = len(data[0])

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3*nrows), constrained_layout=True)
    no_nan = [[l for j in i for k in j.values for l in k if not np.isnan(l)] for i in data]
    vmin = [np.min(i) for i in no_nan]
    vmax = [np.max(i) for i in no_nan]

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    # plotting the heatmap 
    for j in range(nrows):
        for i in range(ncols):
            sn.heatmap(data=data[j][i], annot=annot, center=center, cmap=cmap, ax=axs[j,i], vmin=vmin[j], vmax=vmax[j],
                       cbar=(i==ncols-1), yticklabels=(i==0), xticklabels=(j==nrows-1), cbar_kws={'label': ('Mean background GST [°C]' if j==0 else 'Mean GST evolution [°C]')}) 
        axs[0,0].figure.axes[-1].yaxis.label.set_size(13)
        axs[j,alt_index].add_patch(Rectangle(((rockfall_values(site)['aspect'])/45*1/8, (70-rockfall_values(site)['slope'])/10*1/5), 1/8, 1/5,
                                             edgecolor = 'black', transform=axs[j,alt_index].transAxes, fill=False, lw=4))
    
    for i in range(ncols):
        axs[0,i].set_title('%s m' % alt_list[i])
    fig.supxlabel('Aspect [°]')
    fig.supylabel('Slope [°]')

    # displaying the plotted heatmap 
    plt.show()
    plt.close()
    plt.clf() 

def plot_table_aspect_slope_all_altitudes_polar(df_stats, site):
    """ Function returns 3 polar plots (1 per altitude) of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'


    Returns
    -------
    2*3 polar plots
    """

    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
       if rockfall_values(site)['altitude'] == j:
          alt_index = i

    list_mean = [table_background_evolution_mean_GST_aspect_slope(df_stats)[1], table_background_evolution_mean_GST_aspect_slope(df_stats)[3]]
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
    for i in range(len(vmin)):
        for space in [0.05, 0.1, 0.2, 0.5, 1]:
            pre_ticks = np.arange(np.ceil(vmin[i]/space)*space+0, np.min([vmax[i], np.floor(vmax[i]/space+1)*space]), space)
            if (len(pre_ticks)>=5 and len(pre_ticks)<10):
                ticks[i] = [round(j,(2 if space<0.1 else 1)) for j in pre_ticks]
        tick_pos[i] = [(j-vmin[i])/(vmax[i]-vmin[i]) for j in ticks[i] if (j>=vmin[i] and j<vmax[i])]

    for j in range(nrows):
        for i in range(ncols):
            pc = axs[j,i].pcolormesh(A, R, np.repeat(list_mean[j][i], subdivs, axis=1), cmap=cmap[j], vmin=vmin[j], vmax=vmax[j])
            axs[j,i].set_facecolor("silver")
            axs[j,i].set_theta_zero_location('N')
            axs[j,i].set_theta_direction(-1)
            axs[j,i].tick_params(axis='y', labelcolor='black')
            axs[j,i].set_xticks(abins)
            axs[j,i].set_yticks([])
            axs[j,i].xaxis.grid(False)
            axs[j,i].yaxis.grid(False)
            axs[j,i].set_rlim(95,25)
            axs[0,i].set_title('%s m' % alt_list[i])
            axs[j,i].bar(0, 1).remove()
            for k in range(30,80,10):
                axs[j,i].text(np.pi/5, k, ('%s°' % k), horizontalalignment='center', verticalalignment='center')
        axs[j,alt_index].add_patch(patches.Rectangle(((rockfall_values(site)['aspect']-45/2)/360*2*np.pi, rockfall_values(site)['slope']-5),
                                                     width=np.pi/4, height=10, edgecolor = 'black', fill=False, lw=2))
        
        
    cbar = [fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap[j]),
                         shrink=.8,
                         ax=axs[j,:].ravel().tolist(), orientation='vertical',
                         label=('Mean background GST [°C]' if j==0 else 'Mean GST evolution [°C]'))
            for j in range(nrows)]
    for idx,i in enumerate(range(-nrows,0)):
        axs[0,0].figure.axes[i].yaxis.set_ticks(tick_pos[idx])
        axs[0,0].figure.axes[i].set_yticklabels(ticks[idx]) 
    
    [axs[0,0].figure.axes[i].yaxis.label.set_size(15) for i in [6,7]]

    plt.show()
    plt.close()
    plt.clf()

def plot_permafrost_all_altitudes_polar(df_stats, site, depth_thaw):
    """ Function returns 3 polar plots (1 per altitude) of the permafrost and glacier spatial distribution,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    depth_thaw : netCDF4._netCDF4.Variable
        NetCDF variable encoding the thaw depth


    Returns
    -------
    2*3 polar plots
    """

    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if rockfall_values(site)['altitude'] == j:
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
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    cmap = matplotlib.colors.ListedColormap([colorcycle[1],colorcycle[2],colorcycle[0]]) 

    for i in range(ncols):
        pc = axs[i].pcolormesh(A, R, np.repeat(list_data[i], subdivs, axis=1), cmap=cmap, vmin=0, vmax=1)
        axs[i].set_facecolor("silver")
        axs[i].set_theta_zero_location('N')
        axs[i].set_theta_direction(-1)
        axs[i].tick_params(axis='y', labelcolor='black')
        axs[i].set_xticks(abins)
        axs[i].set_yticks([])
        axs[i].xaxis.grid(False)
        axs[i].yaxis.grid(False)
        axs[i].set_rlim(95,25)
        axs[i].set_title('%s m' % alt_list[i])
        axs[i].bar(0, 1).remove()
        for k in range(30,80,10):
            axs[i].text(np.pi/5, k, ('%s°' % k), horizontalalignment='center', verticalalignment='center')
    axs[alt_index].add_patch(patches.Rectangle(((rockfall_values(site)['aspect']-45/2)/360*2*np.pi, rockfall_values(site)['slope']-5),
                                                    width=np.pi/4, height=10, edgecolor = 'black', fill=False, lw=2))

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap),
                            shrink=.8,
                            ax=axs[:].ravel().tolist(), orientation='vertical')
    axs[0].figure.axes[-1].yaxis.set_ticks([1/6,1/2,5/6])
    axs[0].figure.axes[-1].set_yticklabels(['Permafrost', 'No permafrost, no glaciers', 'Glaciers']) 
    axs[0].figure.axes[-1].yaxis.label.set_size(15)

    plt.show()
    plt.close()
    plt.clf()

def plot_mean_bkg_GST_vs_evolution(df_stats):
    """ Function returns a scatter plot of mean background GST (ground-surface temperature)
        vs evolution of mean GST between the background and transient period.
        Note that each point is computed from an average over the 3 reanalyses to avoid bias.
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'


    Returns
    -------
    Scatter plot
    """

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    xx = [[b for a in i for b in a if not np.isnan(b)] for i in table_background_evolution_mean_GST_aspect_slope(df_stats)[1]]
    yy = [[b for a in i for b in a if not np.isnan(b)] for i in table_background_evolution_mean_GST_aspect_slope(df_stats)[3]]

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))

    for i in range(len(xx)):
        slope, intercept, r, p, se = linregress(xx[i],yy[i])
        print('altitude:', alt_list[i],', R-square:', r**2, ', regression slope:', slope , ', regression intercept:', intercept)
        u = np.arange(np.min(xx[i])-0.1, np.max(xx[i])+0.1, 0.01)
        plt.scatter(xx[i],yy[i], c=colorcycle[i], label=('%s m' % (alt_list[i])))
        plt.plot(u, slope*u+intercept, c=colorcycle[i], label=('slope: %s' % (round(slope,3))))

    plt.legend(loc='lower left')
    plt.xlabel('Mean background GST [°C]')
    plt.ylabel('Mean GST evolution [°C]')

    # displaying the scatter plot 
    plt.show()
    plt.close()
    plt.clf() 

def get_all_stats(path_forcing_era5, path_forcing_merra2, path_forcing_jra55,
                  path_ground, path_snow, path_repository,
                  year_bkg_end=2000, year_trans_end=2020,
                  consecutive=7, extension=''):


    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    # we store the paths to all the files we will need for the analysis
    # starting with the 3 data reanalysis forcings
    path_forcings = {'era5': path_forcing_era5,
                    'merra2': path_forcing_merra2,
                    'jra55': path_forcing_jra55}
    # we also get the time series for the ground and the snow
    path_ground = path_ground
    path_snow = path_snow
    # finally, this is a csv file compiling all the parameters of each simulation
    path_repository = path_repository

    try: ncfile_air_era5.close()
    except: pass
    # Open file for air temperature
    ncfile_air_era5 = Dataset(path_forcings['era5'], mode='r')

    try: ncfile_air_merra2.close()
    except: pass
    # Open file for air temperature
    ncfile_air_merra2 = Dataset(path_forcings['merra2'], mode='r')

    try: ncfile_air_jra55.close()
    except: pass
    # Open file for air temperature
    ncfile_air_jra55 = Dataset(path_forcings['jra55'], mode='r')

    try: ncfile_ground.close()
    except: pass
    # Open file for ground temperature
    ncfile_ground = Dataset(path_ground, mode='r')
    # Select geotop model data
    f_ground = ncfile_ground.groups['geotop'] 
    try: ncfile_snow.close()
    except: pass

    # Open file for snow temperature
    ncfile_snow = Dataset(path_snow, mode='r')
    # Select geotop model data
    f_snow = ncfile_snow.groups['geotop']

    #####################################################################################
    # CREATE VARIABLES
    #####################################################################################

    #####################################################################################
    # GROUND
    #####################################################################################

    # variables_ground = list(f_ground.variables.keys())
    # print(f"The variables in this nc file are: {variables_ground}")

    temp_ground = f_ground['Tg']
    time_ground = f_ground['Date']

    # print('The temperature variable shape: %s, dimensions: %s, and units: %s'
    #     % (temp_ground.shape, temp_ground.dimensions, temp_ground.units))
    # print('The time variable shape: %s and dimensions: %s, and units: %s'
    #     % (time_ground.shape, time_ground.dimensions, time_ground.units))
    # print('Our time data is from %s to %s' % (time_ground[0], time_ground[-1]))
    # print('Our time data is from %s to %s' %
    #     (num2date(time_ground[0], time_ground.units), num2date(time_ground[-1], time_ground.units)))

    #####################################################################################
    # SNOW
    #####################################################################################
    
    # variables_snow = list(f_snow.variables.keys())
    # print(f"The variables in this nc file are: {variables_snow}")

    snow_height = f_snow['snow_depth_mm']
    # print('The snow height variable shape: %s and dimensions: %s, and units: %s' % (snow_height.shape, snow_height.dimensions, snow_height.units))

    #####################################################################################
    # MATCH SIMULATION INDEXES
    #####################################################################################

    # get the topographic info of each simulation, e.g. altitude, slope, etc.
    df_raw = assign_value_df_raw(path_repository)

    try: df
    except NameError: df = None

    df = assign_value_df(df, df_raw, f_ground, f_snow, extension)

    #####################################################################################
    # AIR
    #####################################################################################

    #####################################################################################
    # PRECIPITATION CONSISTENCY TEST
    #####################################################################################

    precipitation_era5 = ncfile_air_era5['PREC_sur']
    precipitation_merra2 = ncfile_air_merra2['PREC_sur']
    precipitation_jra55 = ncfile_air_jra55['PREC_sur']

    # Here we print the sum of the precipitation flux over the whole length of the series
    print('Sum of the precipitation fluxes for the whole series:')
    print('era5  :', np.sum(precipitation_era5[:,0]))
    print('merra2:', np.sum(precipitation_merra2[:,0]))
    print('jra55 :', np.sum(precipitation_jra55[:,0]))

    #####################################################################################
    # ERA5
    #####################################################################################

    # variables_air_era5 = list(ncfile_air_era5.variables.keys())
    # print(f"The variables in this nc file are: {variables_air_era5}")

    time_air_era5 = ncfile_air_era5['time']
    temp_air_era5 = ncfile_air_era5['AIRT_pl']
    SW_flux_era5 = ncfile_air_era5['SW_sur']
    SW_direct_flux_era5 = ncfile_air_era5['SW_topo_direct']
    SW_diffuse_flux_era5 = ncfile_air_era5['SW_topo_diffuse']

    # print('The time variable shape: %s, dimensions: %s, and units: %s' % (time_air_era5.shape, time_air_era5.dimensions, time_air_era5.units))
    # print('The temperature variable shape: %s, dimensions: %s, and units: %s' % (temp_air_era5.shape, temp_air_era5.dimensions, temp_air_era5.units))
    # print('The SW variable shape: %s, dimensions: %s, and units: %s' % (SW_flux_era5.shape, SW_flux_era5.dimensions, SW_flux_era5.units))
    # print('Our time data is from %s to %s' % (time_air_era5[0], time_air_era5[-1]))
    # print('Our time data is from %s to %s' % (num2date(time_air_era5[0], time_air_era5.units), num2date(time_air_era5[-1], time_air_era5.units)))

    # the data is hourly
    # for i in range(2):
    #     print((num2date(time_air_era5[i], time_air_era5.units)))

    #####################################################################################
    # MERRA2
    #####################################################################################

    # variables_air_merra2 = list(ncfile_air_merra2.variables.keys())
    # print(f"The variables in this nc file are: {variables_air_merra2}")

    time_air_merra2 = ncfile_air_merra2['time']
    temp_air_merra2 = ncfile_air_merra2['AIRT_pl']
    SW_flux_merra2 = ncfile_air_merra2['SW_sur']
    SW_direct_flux_merra2 = ncfile_air_merra2['SW_topo_direct']
    SW_diffuse_flux_merra2 = ncfile_air_merra2['SW_topo_diffuse']

    # print('The time variable shape: %s, dimensions: %s, and units: %s' % (time_air_merra2.shape, time_air_merra2.dimensions, time_air_merra2.units))
    # print('The temperature variable shape: %s, dimensions: %s, and units: %s' % (temp_air_merra2.shape, temp_air_merra2.dimensions, temp_air_merra2.units))
    # print('The SW variable shape: %s, dimensions: %s, and units: %s' % (SW_flux_merra2.shape, SW_flux_merra2.dimensions, SW_flux_merra2.units))
    # print('Our time data is from %s to %s' % (time_air_merra2[0], time_air_merra2[-1]))
    # print('Our time data is from %s to %s' % (num2date(time_air_merra2[0], time_air_merra2.units), num2date(time_air_merra2[-1], time_air_merra2.units)))

    # the data is hourly
    # for i in range(2):
    #     print((num2date(time_air_merra2[i], time_air_merra2.units)))

    #####################################################################################
    # JRA55
    #####################################################################################

    # variables_air_jra55 = list(ncfile_air_jra55.variables.keys())
    # print(f"The variables in this nc file are: {variables_air_jra55}")

    time_air_jra55 = ncfile_air_jra55['time']
    temp_air_jra55 = ncfile_air_jra55['AIRT_pl']
    SW_flux_jra55 = ncfile_air_jra55['SW_sur']
    SW_direct_flux_jra55 = ncfile_air_jra55['SW_topo_direct']
    SW_diffuse_flux_jra55 = ncfile_air_jra55['SW_topo_diffuse']

    # print('The time variable shape: %s, dimensions: %s, and units: %s' % (time_air_jra55.shape, time_air_jra55.dimensions, time_air_jra55.units))
    # print('The temperature variable shape: %s, dimensions: %s, and units: %s' % (temp_air_jra55.shape, temp_air_jra55.dimensions, temp_air_jra55.units))
    # print('The SW variable shape: %s, dimensions: %s, and units: %s' % (SW_flux_jra55.shape, SW_flux_jra55.dimensions, SW_flux_jra55.units))
    # print('Our time data is from %s to %s' % (time_air_jra55[0], time_air_jra55[-1]))
    # print('Our time data is from %s to %s' % (num2date(time_air_jra55[0], time_air_jra55.units), num2date(time_air_jra55[-1], time_air_jra55.units)))

    # the data is hourly
    # for i in range(2):
    #     print((num2date(time_air_jra55[i], time_air_jra55.units)))

    #####################################################################################
    # SEPARATING INTO BACKGROUND AND TRANSIENT
    #####################################################################################

    # for lack of longer datasets, we will define the background as everything happening before 2000 (here it means from 1980)
    # and we will limit the transient analysis to everything up to 2020-1-1

    [time_bkg_ground, time_trans_ground, time_pre_trans_ground] = list_tokens_year(time_ground, year_bkg_end, year_trans_end)[1:]
    [time_bkg_air_era5, time_trans_air_era5, time_pre_trans_air_era5] = list_tokens_year(time_air_era5, year_bkg_end, year_trans_end)[1:]
    [time_bkg_air_merra2, time_trans_air_merra2, time_pre_trans_air_merra2] = list_tokens_year(time_air_merra2, year_bkg_end, year_trans_end)[1:]
    [time_bkg_air_jra55, time_trans_air_jra55, time_pre_trans_air_jra55] = list_tokens_year(time_air_jra55, year_bkg_end, year_trans_end)[1:]

    # print('The transient period for the ground goes from %s to %s' % (bkg_end_ground, trans_end_ground))
    # print('The transient period for the air (era5) goes from %s to %s' % (bkg_end_air_era5, trans_end_air_era5))
    # print('The transient period for the air (merra2) goes from %s to %s' % (bkg_end_air_merra2, trans_end_air_merra2))
    # print('The transient period for the air (jra55) goes from %s to %s' % (bkg_end_air_jra55, trans_end_air_jra55))

    #####################################################################################
    # REANALYSIS STATS
    #####################################################################################

    try: reanalysis_stats
    except NameError: reanalysis_stats = None

    reanalysis_stats = assign_value_reanalysis_stat(reanalysis_stats, df,
                                    temp_air_era5, temp_air_merra2, temp_air_jra55,
                                    SW_flux_era5, SW_flux_merra2, SW_flux_jra55,
                                    SW_direct_flux_era5, SW_direct_flux_merra2, SW_direct_flux_jra55,
                                    SW_diffuse_flux_era5, SW_diffuse_flux_merra2, SW_diffuse_flux_jra55,
                                    time_bkg_air_era5, time_bkg_air_merra2, time_bkg_air_jra55,
                                    time_trans_air_era5, time_trans_air_merra2, time_trans_air_jra55,
                                    time_pre_trans_air_era5, time_pre_trans_air_merra2, time_pre_trans_air_jra55,
                                    extension)
    
    #####################################################################################
    # FILTER GLACIERS OUT
    #####################################################################################

    try: list_valid_sim
    except NameError: list_valid_sim = None

    list_valid_sim = glacier_filter(list_valid_sim, temp_ground, snow_height, df, time_pre_trans_ground,
                                    extension, std_temp_glacier=1, excessive_height=1e5)

    #####################################################################################
    # MONTHLY AIR AND GROUND TEMPERATURE
    #####################################################################################

    try: dic_monthly_air_temp
    except NameError: dic_monthly_air_temp = None

    dic_monthly_air_temp = assign_dic_monthly_air_temp(dic_monthly_air_temp, df,
                                                       temp_air_era5, time_air_era5, time_pre_trans_air_era5,
                                                       temp_air_merra2, time_air_merra2, time_pre_trans_air_merra2,
                                                       temp_air_jra55, time_air_jra55, time_pre_trans_air_jra55,
                                                       extension)
                
    try: dic_monthly_ground_temp
    except NameError: dic_monthly_ground_temp = None

    dic_monthly_ground_temp = assign_dic_monthly_ground_temp(dic_monthly_ground_temp, list_valid_sim,
                                                             temp_ground, time_ground, time_pre_trans_ground,
                                                             extension)

    #####################################################################################
    # MELT-OUT DATES
    #####################################################################################

    try: dict_melt_out
    except NameError: dict_melt_out = None

    dict_melt_out = melt_out_date(dict_melt_out, snow_height, list_valid_sim,
                  time_ground, year_bkg_end, year_trans_end, extension)
    stats_melt_out_dic = stats_melt_out(dict_melt_out, year_bkg_end)

    try: dict_melt_out_consecutive
    except NameError: dict_melt_out_consecutive = None

    dict_melt_out_consecutive = melt_out_date_consecutive(consecutive, dict_melt_out_consecutive,
                                                          snow_height, list_valid_sim, time_ground,
                                                          year_bkg_end, year_trans_end, extension)
    stats_melt_out_dic_consecutive = stats_melt_out(dict_melt_out_consecutive, year_bkg_end)

    #####################################################################################
    # CREATE LIST OF STATS
    #####################################################################################

    try: df_stats
    except NameError: df_stats = None

    df_stats = assign_value_df_stat(df_stats, temp_ground, snow_height, reanalysis_stats,
                                    stats_melt_out_dic, stats_melt_out_dic_consecutive, df, list_valid_sim,
                                    time_bkg_ground, time_trans_ground, time_pre_trans_ground,
                                    extension)

    #####################################################################################
    # MONTHLY SO
    #####################################################################################

    try: dic_SO_monthly
    except NameError: dic_SO_monthly = None

    dic_SO_monthly = assign_dic_SO_monthly(dic_SO_monthly, df_stats, list_valid_sim,
                                           dic_monthly_ground_temp, dic_monthly_air_temp, extension, year_bkg_end)
                                           
    #####################################################################################
    # RETURN
    #####################################################################################

    return list_valid_sim, reanalysis_stats, df, df_stats, dict_melt_out, dict_melt_out_consecutive, dic_monthly_ground_temp, dic_monthly_air_temp, dic_SO_monthly

def load_all_pickles(extension=''):

    list_file_names = [f"list_valid_sim{('' if extension=='' else '_')}{extension}.pkl",
                       f"reanalysis_stats{('' if extension=='' else '_')}{extension}.pkl",
                       f"df{('' if extension=='' else '_')}{extension}.pkl",
                       f"df_stats{('' if extension=='' else '_')}{extension}.pkl",
                       f"dict_melt_out{('' if extension=='' else '_')}{extension}.pkl",
                       f"dict_melt_out_consecutive{('' if extension=='' else '_')}{extension}.pkl",
                       f"dic_monthly_ground_temp{('' if extension=='' else '_')}{extension}.pkl",
                       f"dic_monthly_air_temp{('' if extension=='' else '_')}{extension}.pkl",
                       f"dic_SO_monthly{('' if extension=='' else '_')}{extension}.pkl"
                       ]

    output = [0]*len(list_file_names)

    for i, file_name in enumerate(list_file_names):
        my_path = pickle_path + file_name
        # Open the file in binary mode 
        with open(my_path, 'rb') as file: 
            # Call load method to deserialze 
            output[i] = pickle.load(file) 
        print('Succesfully opened the pre-existing pickle:', file_name)

    [list_valid_sim, reanalysis_stats, df, df_stats, dict_melt_out, dict_melt_out_consecutive, dic_monthly_ground_temp, dic_monthly_air_temp, dic_SO_monthly] = output

    return list_valid_sim, reanalysis_stats, df, df_stats, dict_melt_out, dict_melt_out_consecutive, dic_monthly_ground_temp, dic_monthly_air_temp, dic_SO_monthly

def plot_all(site,
             path_forcing_era5, path_forcing_merra2, path_forcing_jra55,
             path_ground, path_snow, path_swe, path_thaw_depth,
             year_bkg_end=2000, year_trans_end=2020):
    """ Function returns a series of summary plots for a given site.
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_forcing_era5 : str
        String path to the location of era5 forcing for that particular site (.nc)
    path_forcing_merra2 : str
        String path to the location of merra2 forcing for that particular site (.nc)
    path_forcing_jra55 : str
        String path to the location of jra55 forcing for that particular site (.nc)
    path_ground : str
        String path to the location of the ground output file from GTPEM (.nc)
    path_snow : str
        String path to the location of the snow output file from GTPEM (.nc)
    path_thaw_depth : str
        String path to the location of the thaw depth output file from GTPEM (.nc)

    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'

    Returns
    -------
    Plots
    """  

    #####################################################################################
    # OPEN THE VARIOUS FILES
    #####################################################################################

    list_valid_sim, reanalysis_stats, df, df_stats, dict_melt_out, dict_melt_out_consecutive, dic_monthly_ground_temp, dic_monthly_air_temp, dic_SO_monthly = load_all_pickles(site)

    # we store the paths to all the files we will need for the analysis
    # starting with the 3 data reanalysis forcings
    path_forcings = {'era5': path_forcing_era5,
                     'merra2': path_forcing_merra2,
                     'jra55': path_forcing_jra55}

    try: ncfile_air_era5.close()
    except: pass

    # Open file for air temperature
    ncfile_air_era5 = Dataset(path_forcings['era5'], mode='r')

    try: ncfile_air_merra2.close()
    except: pass

    # Open file for air temperature
    ncfile_air_merra2 = Dataset(path_forcings['merra2'], mode='r')

    try: ncfile_air_jra55.close()
    except: pass

    # Open file for air temperature
    ncfile_air_jra55 = Dataset(path_forcings['jra55'], mode='r')

    try: ncfile_ground.close()
    except: pass

    # Open file for ground temperature
    ncfile_ground = Dataset(path_ground, mode='r')
    # Select geotop model data
    f_ground = ncfile_ground.groups['geotop'] 

    try: ncfile_snow.close()
    except: pass

    # Open file for snow depth
    ncfile_snow = Dataset(path_snow, mode='r')
    # Select geotop model data
    f_snow = ncfile_snow.groups['geotop']

    try: ncfile_swe.close()
    except: pass

    # Open file for snow depth
    ncfile_swe = Dataset(path_swe, mode='r')
    # Select geotop model data
    f_swe = ncfile_swe.groups['geotop']

    try: ncfile_thaw_depth.close()
    except: pass

    # Open file for snow depth
    ncfile_thaw_depth = Dataset(path_thaw_depth, mode='r')
    # Select geotop model data
    f_thaw_depth = ncfile_thaw_depth.groups['geotop']

    #####################################################################################
    # ASSIGN VARIABLES
    #####################################################################################

    temp_ground = f_ground['Tg']
    time_ground = f_ground['Date']

    snow_height = f_snow['snow_depth_mm']
    swe = f_swe['snow_water_equivalent']
    depth_thaw = f_thaw_depth['AL']

    precipitation_era5 = ncfile_air_era5['PREC_sur']
    precipitation_merra2 = ncfile_air_merra2['PREC_sur']
    precipitation_jra55 = ncfile_air_jra55['PREC_sur']

    time_air_era5 = ncfile_air_era5['time']
    temp_air_era5 = ncfile_air_era5['AIRT_pl']
    SW_flux_era5 = ncfile_air_era5['SW_sur']
    SW_direct_flux_era5 = ncfile_air_era5['SW_topo_direct']
    SW_diffuse_flux_era5 = ncfile_air_era5['SW_topo_diffuse']

    time_air_merra2 = ncfile_air_merra2['time']
    temp_air_merra2 = ncfile_air_merra2['AIRT_pl']
    SW_flux_merra2 = ncfile_air_merra2['SW_sur']
    SW_direct_flux_merra2 = ncfile_air_merra2['SW_topo_direct']
    SW_diffuse_flux_merra2 = ncfile_air_merra2['SW_topo_diffuse']

    time_air_jra55 = ncfile_air_jra55['time']
    temp_air_jra55 = ncfile_air_jra55['AIRT_pl']
    SW_flux_jra55 = ncfile_air_jra55['SW_sur']
    SW_direct_flux_jra55 = ncfile_air_jra55['SW_topo_direct']
    SW_diffuse_flux_jra55 = ncfile_air_jra55['SW_topo_diffuse']

    [time_bkg_ground, time_trans_ground, time_pre_trans_ground] = list_tokens_year(time_ground, year_bkg_end, year_trans_end)[1:]
    [time_bkg_air_era5, time_trans_air_era5, time_pre_trans_air_era5] = list_tokens_year(time_air_era5, year_bkg_end, year_trans_end)[1:]
    [time_bkg_air_merra2, time_trans_air_merra2, time_pre_trans_air_merra2] = list_tokens_year(time_air_merra2, year_bkg_end, year_trans_end)[1:]
    [time_bkg_air_jra55, time_trans_air_jra55, time_pre_trans_air_jra55] = list_tokens_year(time_air_jra55, year_bkg_end, year_trans_end)[1:]

    #####################################################################################
    # 
    #####################################################################################

    # assign a subjective weight to all simulations
    pd_weight = assign_weight_sim(df_stats, site)
    # weighted mean GST
    temp_ground_mean = list(np.average([temp_ground[i,:,0] for i in list(pd_weight.index.values)], axis=0, weights=pd_weight['weight']))
    print('The following plot is a histogram of the distribution of the statistical weights of all simulations:')
    plot_hist_stat_weights(pd_weight, df, zero=True)
    print('The following plot is a histogram of the distribution of glacier simulations wrt to altitude, aspect, slope, and forcing:')
    plot_hist_valid_sim_all_variables(df, df_stats, depth_thaw)

    alt_rockfall = rockfall_values(site)['altitude']

    # sorted list of altitudes
    list_alt = list(np.sort(np.unique(df_stats['altitude'])))
    # altitude index corresponding to the rockfall event
    alt_index = list_alt.index(alt_rockfall)

    # Note that this is selecting the elevation corresponding to the index corresponding to the rockfall, e.g. 1 (->2500 m) for 'Joffre'
    # and it returns the mean air temperature over all reanalyses
    mean_air_temp = mean_all_reanalyses([time_air_era5, time_air_merra2, time_air_jra55],
                                        [temp_air_era5[:,alt_index], temp_air_merra2[:,alt_index], temp_air_jra55[:,alt_index]])
    # here we get the mean precipitation and then water from snow melting 
    mean_prec = mean_all_reanalyses([time_air_era5, time_air_merra2, time_air_jra55],
                                    [precipitation_era5[:,alt_index], precipitation_merra2[:,alt_index], precipitation_jra55[:,alt_index]])
    swe_mean = list(np.average([swe[i,:] for i in list(pd_weight.index.values)], axis=0, weights=pd_weight['weight']))
    # finally we get the total water production, averaged over all reanalyses
    tot_water_prod = assign_tot_water_prod(swe_mean, mean_prec, time_ground, time_pre_trans_ground, time_air_era5)

    year_rockfall = rockfall_values(site)['year']
    print('Plots of the normalized distance of air and ground temperature, water production, and thaw_depth as a function of time')
    print('Granularity: week and month side by side')
    plot_aggregating_distance_temp_all(['Air temperature', 'Water production', 'Ground temperature'],
                                       [time_air_era5, time_ground, time_ground],
                                       [mean_air_temp, tot_water_prod, temp_ground_mean],
                                       ['week', 'month'], site, year_rockfall, False)
    print('Granularity: year, plotted for all years')
    plot_aggregating_distance_temp_all(['Air temperature', 'Water production', 'Ground temperature'],
                                        [time_air_era5, time_ground, time_ground],
                                        [mean_air_temp, tot_water_prod, temp_ground_mean],
                                        ['year'], site, 0, False)

    print('Heatmap of the background mean GST as a function of aspect and slope at %s m:' % alt_rockfall)
    plot_table_mean_GST_aspect_slope(df_stats, site, alt_rockfall, True)
    print('Heatmap of the evolution of the mean GST between the background and the transient periods as a function of aspect and slope at %s m:' % alt_rockfall)
    plot_table_mean_GST_aspect_slope(df_stats, site, alt_rockfall, False)

    print('Heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitude')
    plot_table_aspect_slope_all_altitudes(df_stats, site)

    print('Polar heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitude')
    plot_table_aspect_slope_all_altitudes_polar(df_stats, site)

    print('Polar plot of the permafrost and glacier spatial distribution as a function of aspect and slope at all altitude')
    plot_permafrost_all_altitudes_polar(df_stats, site, depth_thaw)

    print('Scatter plot of mean background GST vs evolution of mean GST between the background and transient period')
    plot_mean_bkg_GST_vs_evolution(df_stats)

    print('Parity plot (statistically-modeled vs numerically-simulated) of background mean GST:')
    xdata, ydata, optimizedParameters, pcov, corr_matrix, R_sq = fit_stat_model_grd_temp(df_stats, all=True, diff_forcings=True)
    list_ceof = ['offset', 'c_alt', 'd_alt', 'c_asp', 'c_slope']
    pd_coef = pd.DataFrame(list_ceof, columns=['Coefficient'])
    pd_coef = pd.concat([pd_coef, pd.DataFrame((np.array([list(i) for i in optimizedParameters]).transpose()), columns=['all', 'era5', 'merra2', 'jra55'])], axis=1)
    print('The coefficients of the statistical model for the mean background GST are given by:')
    print(pd_coef)
    