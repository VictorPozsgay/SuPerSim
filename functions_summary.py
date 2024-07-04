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

def open_air_nc(path_forcing):
    """ Function returns data from the .nc file for a given atmospheric forcing
    
    Parameters
    ----------
    path_forcing : str
        Path to the .nc file where the atmospheric forcing data is stored

    Returns
    -------
    time_air : netCDF4._netCDF4.Variable
        Time file for the atmospheric forcing in the shape (time)
    temp_air : netCDF4._netCDF4.Variable
        Air temperature in the shape (time, station)
    SW_flux : netCDF4._netCDF4.Variable
        Shortwave (SW) flux in the shape (time, station)
    SW_direct_flux : netCDF4._netCDF4.Variable
        Direct shortwave (SW) in the shape (time, station)
    SW_diffuse_flux : netCDF4._netCDF4.Variable
        Diffuse shortwave (SW) in the shape (time, station)
    precipitation : netCDF4._netCDF4.Variable
        Precipitation in the shape (time, station)
    """

    # Open file for air temperature
    ncfile_air = Dataset(path_forcing, mode='r')
    
    time_air = ncfile_air['time']
    temp_air = ncfile_air['AIRT_pl']
    SW_flux = ncfile_air['SW_sur']
    SW_direct_flux = ncfile_air['SW_topo_direct']
    SW_diffuse_flux = ncfile_air['SW_topo_diffuse']
    precipitation = ncfile_air['PREC_sur']

    return time_air, temp_air, SW_flux, SW_direct_flux, SW_diffuse_flux, precipitation

def open_ground_nc(path_ground):
    """ Function returns data from the .nc file for the ground simulations
    
    Parameters
    ----------
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored

    Returns
    -------
    f_ground : netCDF4._netCDF4.Group
        geotop netCDF group with description of dimensions, variables, and data for the ground
    time_ground : netCDF4._netCDF4.Variable
        Time file for the ground simulations in the shape (time)
    temp_ground : netCDF4._netCDF4.Variable
        Ground temperature in the shape (simulation, time, soil_depth)
    """

    # Open file for ground temperature
    ncfile_ground = Dataset(path_ground, mode='r')
    # Select geotop model data
    f_ground = ncfile_ground.groups['geotop']
    
    time_ground = f_ground['Date']
    temp_ground = f_ground['Tg']

    return f_ground, time_ground, temp_ground

def open_snow_nc(path_snow):
    """ Function returns data from the .nc file for the snow simulations
    
    Parameters
    ----------
    path_snow : str
        Path to the .nc file where the aggregated snow simulations are stored

    Returns
    -------
    f_snow : netCDF4._netCDF4.Group
        geotop netCDF group with description of dimensions, variables, and data for the snow
    snow_height : netCDF4._netCDF4.Variable
        Depth of snow in the shape (simulation, time)
    """

    # Open file for snow depth
    ncfile_snow = Dataset(path_snow, mode='r')
    # Select geotop model data
    f_snow = ncfile_snow.groups['geotop']
    
    snow_height = f_snow['snow_depth_mm']

    return f_snow, snow_height

def open_swe_nc(path_swe):
    """ Function returns data from the .nc file for the snow water equivalent (SWE) simulation results
    
    Parameters
    ----------
    path_swe : str
        Path to the .nc file where the aggregated SWE simulations are stored

    Returns
    -------
    f_swe : netCDF4._netCDF4.Group
        geotop netCDF group with description of dimensions, variables, and data for the SWE
    swe : netCDF4._netCDF4.Variable
        SWE in the shape (simulation, time)
    """

    # Open file for snow depth
    ncfile_swe = Dataset(path_swe, mode='r')
    # Select geotop model data
    f_swe = ncfile_swe.groups['geotop']
    
    swe = f_swe['snow_water_equivalent']

    return f_swe, swe

def open_thaw_depth_nc(path_thaw_depth):
    """ Function returns data from the .nc file for the depth of thaw simulation results
    
    Parameters
    ----------
    path_thaw_depth : str
        Path to the .nc file where the aggregated thaw depth simulations are stored

    Returns
    -------
    f_thaw_depth : netCDF4._netCDF4.Group
        geotop netCDF group with description of dimensions, variables, and data for the thaw depth
    thaw_depth : netCDF4._netCDF4.Variable
        Depth of thaw in the shape (simulation, time)
    """

    # Open file for snow depth
    ncfile_thaw_depth = Dataset(path_thaw_depth, mode='r')
    # Select geotop model data
    f_thaw_depth = ncfile_thaw_depth.groups['geotop']
    
    thaw_depth = f_thaw_depth['AL']

    return f_thaw_depth, thaw_depth

def time_unit_stamp(time_file):
    """ Function returns frequency of datapoints and exact date and time of the initial datapoint, converted into a datetime
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
        Needs to have a .units attribute

    Returns
    -------
    frequency : str
        Frequency of the data sampling, e.g. 'seconds'
    start_date : datetime.datetime
        Date and time of the first time stamp

    """

    # this extracts the unit of the time series, i.e. when it starts,
    # e.g. days since 0001-1-1 0:0:0 or seconds since 1900-01-01
    # these are two formats that exist and we have to accomodate for both
    # we partition the string and only keep what comes after 'since ': the date
    time_start_date = time_file.units.partition(' since ')[2]

    # we identify the different delimiters between year, month, day, hour, etc. 
    # to be: '-', ' ', or ':'
    # we get a list in the form [1, 1, 1, 0, 0, 0] or [1900, 1, 1] respectively
    time_start_date_list = list(map(int, map(float, re.split('-| |:', time_start_date))))

    # now we decide to fill the list with 0s if not long enough
    # i.e. if not provided, we assume the start is at exactly midnight 0:0:0
    time_start_date_fill = time_start_date_list[:6] + [0]*(6 - len(time_start_date_list))

    # here we get the string telling us with what frequency data is sampled, e.g. 'days' or 'seconds'
    frequency = time_file.units.partition(' since ')[0]
    # and the date of the first time stamp
    start_date = datetime(*time_start_date_fill)

    # finally, we convert it to a datetime
    return start_date, frequency

def list_tokens_year(time_file, year_bkg_end, year_trans_end):
    """ Function returns a list of time stamps per year, and splits the timestamps into background and transient periods
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground or time_air)
        Needs to have a .units attribute
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period

    Returns
    -------
    list_years : dict
        a list of indices per calendar year that can be used to study data year per year
        in the form of a dictionary like {1980: [0, 1, 2, ..., 363, 364], 1981: [365, ..., 729], ..., 2019: [14244, ..., 14608]}
    time_bkg : numpy.ma.core.MaskedArray
        Mask that selects only the background timestamps
    time_trans : numpy.ma.core.MaskedArray
        Mask that selects only the transient timestamps
    time_pre_trans : numpy.ma.core.MaskedArray
        Mask that selects all points before the end of the transient period (background+transient)

    """

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

def assign_value_global_dict(path_forcing_list, path_ground, path_snow, year_bkg_end, year_trans_end, extension=''):
    """ Function returns a dictionary containing all the important timeseries and saves it to a pickle
    
    Parameters
    ----------
    path_forcing_list : list of str
        List of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_snow : str
        Path to the .nc file where the aggregated snow simulations are stored
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    extension : str, optional
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles   


    Returns
    -------
    global_dict : dict
        A large dictionary listing all timeseries, organized in: air/ground/snow > relevant timeseries
        For the 'air', all timeseries are a list with each entry corresponding to a different reanalysis

    """

    file_name = f"global_dict{('' if extension=='' else '_')}{extension}.pkl"
    my_path = pickle_path + file_name

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

def assign_value_df_raw(path_repository):
    """ Function converts the .csv ensemble simulation repository into a panda dataframe with a column for each parameter
    
    Parameters
    ----------
    path_repository : str
        Path to the .csv file with all the simulation parameters

    Returns
    -------
    df_raw : pandas.core.frame.DataFrame
        A panda dataframe version of the .csv file where the simulation paramaters have been unpacked into readable columns

    """

    df_raw = pd.read_csv(path_repository, usecols=['site','directory','parameters'])
    df_raw['altitude'] = (df_raw['site'].str.split('_').str[1]).apply(pd.to_numeric)
    df_raw['site_name'] = df_raw['site'].str.split('_').str[0]
    df_raw['forcing'] = df_raw['directory'].str.split('_').str[3]
    df_raw['aspect'] = [pd.to_numeric(i.replace('p','.')) for i in (df_raw['parameters'].str.split('aspect_').str[1]).str.split('.inpts').str[0]]
    df_raw['slope'] = ((df_raw['parameters'].str.split('slope_').str[1]).str.split('.inpts').str[0]).apply(pd.to_numeric)
    df_raw['snow'] = [(y/100 if y > 10 else y) for y in [pd.to_numeric(i.replace('p','.')) for i in ((df_raw['parameters'].str.split('snow_').str[1]).str.split('.inpts').str[0])]]
    if len(df_raw['parameters'].str.split('pointmaxswe_')[0]) == 2:
        df_raw['maxswe'] = (df_raw['parameters'].str.split('pointmaxswe_').str[1]).str.split('.inpts').str[0]
    else:
        df_raw['maxswe'] = [np.nan for i in df_raw['altitude']]
    df_raw['material'] = (df_raw['parameters'].str.split('soil_').str[1]).str.split('.inpts').str[0]
    df_raw.drop('parameters', axis=1, inplace=True)

    return df_raw

def assign_value_df(path_repository, path_ground, extension=''):
    """ Function returns the panda dataframe with all ensemble simulation parameters and saves it to a pickle
    
    Parameters
    ----------
    path_repository : str
        Path to the .csv file with all the simulation parameters
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    extension : str, optional
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

    file_name = f"df{('' if extension=='' else '_')}{extension}.pkl"
    my_path = pickle_path + file_name

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
        df_raw = assign_value_df_raw(path_repository)
            
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

def assign_value_reanalysis_stat(forcing_list, path_forcing_list, year_bkg_end, year_trans_end, extension=''):
    """ Creates a dictionary of mean quantities over the background and transient periods
    
    Parameters
    ----------
    forcing_list : list of str
        List of forcings provided, with a number of entries between 1 and 3 in 'era5', 'merra2', and 'jra55'. E.g. ['era5', 'merra2']
    path_forcing_list : list of str
        List of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    extension : str, optional
        Location of the event, e.g. 'Aksaut_Ridge' (not a list, 1 single location)    

    Returns
    -------
    reanalysis_stats : dict
        dictionary of mean quntities over the background and transient periods

    """

    file_name = f"reanalysis_stats{('' if extension=='' else '_')}{extension}.pkl"
    file_name_df = f"df{('' if extension=='' else '_')}{extension}.pkl"
    my_path = pickle_path + file_name
    my_path_df = pickle_path + file_name_df

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

def glacier_filter(path_snow, extension='', glacier=False, min_glacier_depth=100, max_glacier_depth=20000):
    """ Function returns a list of valid simulations regarding the glacier criteria
    
    Parameters
    ----------
    path_snow : str
        Path to the .nc file where the aggregated snow simulations are stored
    extension : str, optional
        Location of the event, e.g. 'Aksaut_Ridge'
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

    file_name = f"list_valid_sim{('' if extension=='' else '_')}{extension}.pkl"
    my_path = pickle_path + file_name

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

def melt_out_date(consecutive, path_ground, path_snow, year_bkg_end, year_trans_end, extension=''):
    """ Function returns a list of melt out dates given the criterion of a number of consecutive snow-free days
    
    Parameters
    ----------
    consecutive : int
        Number of minimum consecutive snow-free days to declare seasonal melt out of the snow
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_snow : str
        Path to the .nc file where the aggregated snow simulations are stored
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    extension : str, optional
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

    file_name = f"melt_out{('' if extension=='' else '_')}{extension}.pkl"
    file_name_list_valid_sim = f"list_valid_sim{('' if extension=='' else '_')}{extension}.pkl"
    my_path = pickle_path + file_name

    snow_height = open_snow_nc(path_snow)[1]
    time_ground = open_ground_nc(path_ground)[1]
    with open(pickle_path + file_name_list_valid_sim, 'rb') as file: 
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

def assign_value_df_stats(path_ground, path_snow, year_bkg_end, year_trans_end, extension=''):
    """ Function returns a large panda dataframe with information about the air, ground, snow, topography, etc. for all simulations
    
    Parameters
    ----------
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_snow : str
        Path to the .nc file where the aggregated snow simulations are stored
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    extension : str, optional
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    

    Returns
    -------
    df_stats : pandas.core.frame.DataFrame
        Large panda dataframe with information about the air, ground, snow, topography, etc. for all simulations
    """

    file_name = f"df_stats{('' if extension=='' else '_')}{extension}.pkl"
    file_name_df = f"df{('' if extension=='' else '_')}{extension}.pkl"
    file_name_reanalysis_stats = f"reanalysis_stats{('' if extension=='' else '_')}{extension}.pkl"
    file_name_list_valid_sim = f"list_valid_sim{('' if extension=='' else '_')}{extension}.pkl"
    file_name_melt_out = f"melt_out{('' if extension=='' else '_')}{extension}.pkl"
    my_path = pickle_path + file_name

    snow_height = open_snow_nc(path_snow)[1]
    _, time_ground, temp_ground = open_ground_nc(path_ground)
    _, time_bkg_ground, time_trans_ground, time_pre_trans_ground = list_tokens_year(time_ground, year_bkg_end, year_trans_end)

    with open(pickle_path + file_name_df, 'rb') as file: 
        # Call load method to deserialize 
        df = pickle.load(file)

    with open(pickle_path + file_name_reanalysis_stats, 'rb') as file: 
        # Call load method to deserialize 
        reanalysis_stats = pickle.load(file)

    with open(pickle_path + file_name_list_valid_sim, 'rb') as file: 
        # Call load method to deserialize 
        list_valid_sim = pickle.load(file)

    with open(pickle_path + file_name_melt_out, 'rb') as file: 
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

def assign_weight_sim(extension='', no_weight=True):
    """ Function returns a statistical weight for each simulation according to the importance in rockfall starting zone 
    
    Parameters
    ----------
    extension : str, optional
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    no_weight : bool, optional
        If True, all simulations have the same weight, otherwise the weight is computed as a function of altitude, aspect, and slope

    Returns
    -------
    pd_weight : pandas.core.frame.DataFrame
        Panda DataFrame assigning a statistical weight to each simulation for each of 'altitude', 'aspect', 'slope'
        and an overall weight.
    """

    file_name = f"df_stats{('' if extension=='' else '_')}{extension}.pkl"
    my_path = pickle_path + file_name
    with open(my_path, 'rb') as file: 
        # Call load method to deserialize 
        df_stats = pickle.load(file)

    dict_weight = {}
    if no_weight:
        dict_weight = {i: [1,1,1] for i in list(df_stats.index.values)}
    else:
        alt_distance = np.max([np.abs(i-rockfall_values(extension)['altitude']) for i in np.sort(np.unique(df_stats['altitude']))])
        dict_weight = {i: [1 - np.abs(df_stats.loc[i]['altitude']-rockfall_values(extension)['altitude'])/(2*alt_distance),
                        np.cos((np.pi)/180*(df_stats.loc[i]['aspect']-rockfall_values(extension)['aspect']))/4+3/4,
                        np.cos((np.pi)/30*(df_stats.loc[i]['slope']-rockfall_values(extension)['slope']))/4+3/4]
                        for i in list(df_stats.index.values)}
    
    pd_weight = pd.DataFrame.from_dict(dict_weight, orient='index',
                                       columns=['altitude', 'aspect', 'slope'])
    pd_weight['weight'] = pd_weight['altitude']*pd_weight['aspect']*pd_weight['slope']
    
    return pd_weight

def plot_hist_valid_sim_all_variables(path_thaw_depth, extension=''): 
    """ Function returns a histogram of the number of valid/glacier simulations for each of the following variable
        ('altitude', 'aspect', 'slope', 'forcing') 
        It also shows the breakdown of valid simulations into permafrost and no-permafrost ones

    Parameters
    ----------
    path_thaw_depth : str
        Path to the .nc file where the aggregated thaw depth simulations are stored
    extension : str, optional
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles

    Returns
    -------
    Histogram (subplot(2,2))
    """

    file_name_df = f"df{('' if extension=='' else '_')}{extension}.pkl"
    file_name_df_stats = f"df_stats{('' if extension=='' else '_')}{extension}.pkl"

    _, thaw_depth = open_thaw_depth_nc(path_thaw_depth)

    with open(pickle_path + file_name_df, 'rb') as file: 
        # Call load method to deserialize 
        df = pickle.load(file)

    with open(pickle_path + file_name_df_stats, 'rb') as file: 
        # Call load method to deserialize 
        df_stats = pickle.load(file)

    data=np.random.random((4,10))
    variables = ['altitude','aspect','slope','forcing']
    xaxes = ['Altitude [m]','Aspect [°]','Slope [°]','Forcing']
    yaxes = ['Number of simulations','','Number of simulations','']

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

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

def mean_all_reanalyses(time_files, files_to_smooth, year_bkg_end, year_trans_end):
    """ Function returns the mean time series over a number of reanalysis (has the length of a timeseries)
    
    Parameters
    ----------
    time_files : list of netCDF4._netCDF4.Variable
        List of files where the time index of each datapoint is stored
    files_to_smooth : list of list
        List of time series (could be temperature, precipitation, snow depth, etc.) that needs to be smoothed
        Note that it needs to be in the shape (n,) and not (n,3) for instance, hence the altitude has
        to be pre-selected. E.g. accepts [temp_air_era5[:,0]] but not [temp_air_era5]
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period

    Returns
    -------
    mean : dict
        average time series over reanalyses (3 or less)
    """

    list_years = [list_tokens_year(time_files[i], year_bkg_end, year_trans_end)[0] for i in range(len(time_files))]
    len_years = {y: [len(list_years[i][y]) for i in range(len(time_files))] for y in list(list_years[0].keys()) if y<year_trans_end}

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

def assign_tot_water_prod_generic(path_forcing_list, path_ground, path_swe, year_bkg_end, year_trans_end, extension='', no_weight=True):
    """ Function returns the total water production at daily intervals in [mm s-1] 
    
    Parameters
    ----------
    path_forcing_list : list of str
        List of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_swe : str
        Path to the .nc file where the aggregated SWE simulations are stored
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    extension : str, optional
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    no_weight : bool, optional
        If True, all simulations have the same weight, otherwise the weight is computed as a function of altitude, aspect, and slope

    Returns
    -------
    tot_water_prod : list
        list of daily total water production
    mean_swe : list
        Weighted mean SWE time series over all simulations
    mean_prec : list
        Mean precipitation time series over all reanalyses
    """

    file_name = f"df_stats{('' if extension=='' else '_')}{extension}.pkl"
    my_path = pickle_path + file_name
    with open(my_path, 'rb') as file: 
        # Call load method to deserialize 
        df_stats = pickle.load(file)

    _, swe = open_swe_nc(path_swe)
    _, time_ground, _ = open_ground_nc(path_ground)
    _, _, _, time_pre_trans_ground = list_tokens_year(time_ground, year_bkg_end, year_trans_end)    

    pd_weight = assign_weight_sim(extension, no_weight)

    time_air_all = [open_air_nc(i)[0] for i in path_forcing_list]
    precipitation_all = [open_air_nc(i)[-1] for i in path_forcing_list]

    # Note that this is selecting the elevation in the 'middle': index 2 in the list [0,1,2,3,4]
    # and it returns the mean air temperature over all reanalyses
    alt_list = sorted(set(df_stats['altitude']))
    alt_index = int(np.floor((len(alt_list)-1)/2))
    alt_index_abs = alt_list[alt_index]
    print('List of altitudes:', alt_list)
    print('Altitude at which we plot the time series:', alt_index_abs)

    # here we get the mean precipitation and then water from snow melting 
    mean_swe = list(np.average([swe[i,:] for i in list(pd_weight.index.values)], axis=0, weights=pd_weight['weight']))
    mean_prec = mean_all_reanalyses(time_air_all, [i[:,alt_index] for i in precipitation_all], year_bkg_end, year_trans_end)

    # convert mean precipitation into a panda dataframe to facilitate grouping by 24.
    # This way the data is reported daily (as it is for the swe) rather than hourly
    pd_prec_day = ((pd.DataFrame(mean_prec).rolling(24).mean()).iloc[23::24, :]).reset_index(drop=True)

    # convert swe into panda df to make the diff() operation easier. Swe is a total and
    # we get the instantaneous flux by taking the difference between consecutive datapoints.
    # The total precipitation between 2 datapoints is over 1 day = 864000 hence we divide by 86400.
    # Finally, swe is in [kg m-2] which is equivalent to [mm] so we're good.

    # Number of days between the start of the ground and air timeseries
    num_day = int((num2date(time_ground[0], time_ground.units) - num2date(time_air_all[0][0], time_air_all[0].units)).total_seconds()/86400)
    # we add num_days zeros at the beginning to make sure the dfs have the same dimension and span Jan 1, 1980 to Dec 31, 2019
    # We make sure to multiply by (-1) and to divide by 86400
    pd_diff_swe = pd.concat([pd.DataFrame([0]*num_day), (pd.DataFrame(mean_swe[:len(time_ground[time_pre_trans_ground][:])]).diff()).fillna(0)/86400*(-1)], ignore_index=True)

    if len(pd_prec_day)!=len(pd_diff_swe):
        raise ValueError('VictorCustomError: The lengths of the precipitation and swe time series are different!')

    # The total water production is returned in the list type as the sum of precipitation and swe
    tot_water_prod = list(pd_prec_day.add(pd_diff_swe.reset_index(drop=True), fill_value=0)[0])

    return tot_water_prod, mean_swe, mean_prec

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

def aggregating_distance_temp_generic(time_file, file_to_smooth, window, year=0, year_bkg_end=2010, year_trans_end=2023, fill_before=False):
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
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
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

    list_dates = list_tokens_year(time_file, year_bkg_end=2010, year_trans_end=2023)[0]

    num_min = np.min([len(list_dates[year]) for year in list_dates.keys() if year < year_trans_end])
    min_temp_index = list(temp_smoothed.keys())[0]

    # range of the timestamp whether the user chose a given year of the whole study period
    # index_range = list(range(len(file_to_smooth)))

    aggregate_mean = {}
    aggreagte_std = {}
    distance = {}

    start_year = num2date(time_file[0], time_file.units).year

    for year in [year_bkg_end]:
        aggregate_mean[year] = {day: np.mean([temp_smoothed[list_dates[y][day]] for y in range(start_year,year+1) if list_dates[y][day]>=min_temp_index]) for day in range(num_min)}
        aggreagte_std[year] = {day: np.std([temp_smoothed[list_dates[y][day]] for y in range(start_year,year+1) if list_dates[y][day]>=min_temp_index]) for day in range(num_min)}
        distance[year] = {day: (temp_smoothed[list_dates[year][day]]-aggregate_mean[year][day])/aggreagte_std[year][day] for day in range(num_min)}

    for year in range(year_bkg_end+1, year_trans_end):
        aggregate_mean[year] = {day: aggregate_mean[year-1][day]
                                     + (temp_smoothed[list_dates[year][day]] - aggregate_mean[year-1][day]) / (year-start_year+1)
                                     for day in range(num_min)}
        aggreagte_std[year] = {day: np.sqrt( ( (year-start_year) * aggreagte_std[year-1][day]**2
                                              + (temp_smoothed[list_dates[year][day]] - aggregate_mean[year][day])
                                              *(temp_smoothed[list_dates[year][day]] - aggregate_mean[year-1][day]) )
                                              / (year-start_year+1) )
                                    for day in range(num_min)}
        distance[year] = {day: (temp_smoothed[list_dates[year][day]]-aggregate_mean[year][day])/aggreagte_std[year][day] for day in range(num_min)}


    return distance

def plot_aggregating_distance_temp_all_generic(yaxes, xdata, ydata, window, site, year, year_bkg_end=2010, year_trans_end=2023, fill_before=False):
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
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
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
        distance = [aggregating_distance_temp_generic(xdata[idx], ydata[idx], i, year, year_bkg_end, year_trans_end, fill_before) for i in window]
        list_dates = list_tokens_year(xdata[idx], year_bkg_end, year_trans_end)[0]

        if year in list(list_dates.keys()):
            index_range = list_dates[year]
        else:
            index_range = list(range(len(ydata[idx])))[list_dates[np.min(list(distance[0].keys()))][0]:]

        colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']

        if year in list(list_dates.keys()):
            if num_cols == 1:
                ax.plot(xdata[idx][list_dates[year][:][:len(distance[0][year].values())]], distance[0][year].values(), label='Deviation')
                ax.fill_between(xdata[idx][list_dates[year][:][:len(distance[0][year].values())]], -2, 2, alpha = 0.2, color = 'blue')
            else:
                for i in range(num_cols):
                    ax[i].plot(xdata[idx][list_dates[year][:][:len(distance[i][year].values())]], distance[i][year].values(), label='Deviation')
                    ax[i].fill_between(xdata[idx][list_dates[year][:][:len(distance[i][year].values())]], -2, 2, alpha = 0.2, color = 'blue')
        else:
            if num_cols == 1:
                for y in list(distance[0].keys()):
                    ax.plot(xdata[idx][list_dates[y][:][:len(distance[0][y].values())]], distance[0][y].values(), label=('Deviation' if y==year_bkg_end else ''), color= colorcycle[0])
                    ax.fill_between(xdata[idx][list_dates[y][:][:len(distance[0][y].values())]], -2, 2, alpha = 0.2, color = 'blue')
            else:
                for i in range(num_cols):
                    for y in list(distance[i].keys()):
                        ax[i].plot(xdata[idx][list_dates[y][:][:len(distance[i][y].values())]], distance[i][y].values(), label=('Deviation' if y==year_bkg_end else ''), color= colorcycle[0])
                        ax[i].fill_between(xdata[idx][list_dates[y][:][:len(distance[i][y].values())]], -2, 2, alpha = 0.2, color = 'blue')

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
            # locs = np.linspace(xdata[idx][index_range[0]], xdata[idx][index_range[-1]], num=14, endpoint=True)
            loc_init = xdata[idx][list_dates[year_bkg_end][0]]
            len_year = xdata[idx][list_dates[year_bkg_end+1][0]] - loc_init
            num_year = year_trans_end - year_bkg_end
            locs = np.linspace(loc_init, loc_init + num_year * len_year, num=num_year+1, endpoint=True)
            labels = list(range(year_bkg_end,year_trans_end+1,1))

        if idx < num_rows-1:
            labels_end = ['']*len(labels)
        else:
            labels_end = labels
        if num_cols == 1:
            ax.set_xticks(locs, labels)
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
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.clf()

def table_background_evolution_mean_GST_aspect_slope(extension=''):
    """ Function returns a table of mean background and evolution of GST (ground-surface temperature)
        as a function of slope, aspect, and altitude
    
    Parameters
    ----------
    extension : str, optional
        Location of the event, e.g. 'Aksaut_North', will be used to label all the pickles
    
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

    file_name = f"df_stats{('' if extension=='' else '_')}{extension}.pkl"
    my_path = pickle_path + file_name
    with open(my_path, 'rb') as file: 
        # Call load method to deserialize 
        df_stats = pickle.load(file)
    
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

def plot_table_mean_GST_aspect_slope_generic(df_stats, site, altitude, background=True, box=True):
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
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box


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
    
    if box:
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

def plot_table_aspect_slope_all_altitudes_generic(df, df_stats, site, show_glacier=True, box=True):
    """ Function returns 1 plot per altitude of the table of either mean background GST (ground-surface temperature)
        or its evolution between the background and the transient periods,
        as a function of slope, aspect, and altitude and higlight the cell corresponding to the 
        rockfall starting zone
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Panda DataFrame df, should at least include the following columns: 'altitude', 'aspect', 'slope'
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    show_glacier : bool, opional
        Whether or not to plot the glacier fraction
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    (2 or 3)*(# altitudes) tables
    """
    
    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}
    # number of simulations for a given cell with fixed (altitude, slope, aspect) triplet
    sim_per_cell = len(df)/np.prod([len(dic_var[i]) for i in dic_var.keys()])

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if box:
            if rockfall_values(site)['altitude'] == j:
                alt_index = i

    # setting the parameter values 
    annot = True
    center = [0, 0]
    cmap = ['seismic', 'seismic']

    table_all = table_background_evolution_mean_GST_aspect_slope(df_stats)

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
            if box:
                axs[j].add_patch(Rectangle(((rockfall_values(site)['aspect'])/45*1/8, (70-rockfall_values(site)['slope'])/10*1/5), 1/8, 1/5,
                                        edgecolor = 'black', transform=axs[j].transAxes, fill=False, lw=4))
        if ncols > 1:
            for i in range(ncols):
                sn.heatmap(data=data[nrows-1-j][i], annot=annot, center=center[nrows-1-j], cmap=cmap[nrows-1-j], ax=axs[j,i], vmin=vmin[nrows-1-j], vmax=vmax[nrows-1-j],
                            cbar=(i==ncols-1), yticklabels=(i==0), xticklabels=(j==nrows-1), cbar_kws={'label': labels_plot[nrows-1-j]})
            axs[0,0].figure.axes[-1].yaxis.label.set_size(13)
            if box:
                axs[j,alt_index].add_patch(Rectangle(((rockfall_values(site)['aspect'])/45*1/8, (70-rockfall_values(site)['slope'])/10*1/5), 1/8, 1/5,
                                                    edgecolor = 'black', transform=axs[j,alt_index].transAxes, fill=False, lw=4))

    if ncols == 1:
        axs[0].set_title('%s m' % alt_list[0])
    if ncols > 1:
        for i in range(ncols):
            axs[0,i].set_title('%s m' % alt_list[i])
    fig.supxlabel('Aspect [°]')
    fig.supylabel('Slope [°]')

    # displaying the plotted heatmap 
    plt.show()
    plt.close()
    plt.clf() 
 
def plot_table_aspect_slope_all_altitudes_polar_generic(df_stats, site, box=True):
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
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    2*3 polar plots
    """

    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if box:
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
        if box:
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

def plot_permafrost_all_altitudes_polar_generic(df_stats, site, depth_thaw, box=True):
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
    box : bool, optional 
        If True, highlights the cell corresponding to the rockfall starting zone with a black box

    Returns
    -------
    2*3 polar plots
    """

    variables = ['aspect', 'slope','altitude']
    dic_var = {}
    dic_var = {i: np.sort(np.unique(df_stats.loc[:, i], return_counts=False)) for i in variables}

    alt_list = list(np.sort(np.unique(df_stats['altitude'])))
    for i,j in enumerate(alt_list):
        if box:
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
    if box:
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

    forcings = ['merra2']
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
                    s=20, label=('all data' if (all and i==0) else forcings[len(forcings)-len(data_set)+i]) )
        
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

def plot_box_yearly_stat(name_series, time_file, file_to_plot, year_bkg_end=2010, year_trans_end=2023):
    """ Plots the distance to the mean in units of standard deviation for a specific year or for the whole length
    
    Parameters
    ----------
    name_series : str
        Name of the quantity to plot, has to be one of 'GST', 'Air temperature', 'Precipitation', 'SWE', 'Water production'
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored        
    file_to_plot : list
        Mean time series (temp_ground_mean, mean_air_temp, mean_prec, swe_mean, tot_water_prod)
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period

    Returns
    -------
        plot

    """

    fig, ax = plt.subplots()

    list_dates = list_tokens_year(time_file, year_bkg_end, year_trans_end)[0]
    overall_mean = np.mean(file_to_plot)
    exponent = int(np.floor(np.log10(np.abs(overall_mean))))
    a = []
    for i in list_dates.keys():
        if i < year_trans_end:
            a = a + [i]*len(list_dates[i])
    x = pd.DataFrame(file_to_plot[:len(a)], columns=[name_series], index=a)
    x['Year'] = a

    mean = [np.mean(x[x['Year']<year_bkg_end][name_series]), np.mean(x[(x['Year']>=year_bkg_end) & (x['Year']<year_trans_end)][name_series])]

    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
    sn.boxplot(x='Year', y=name_series, data=x, showmeans=True, showfliers=False, meanprops=meanpointprops, color='grey', linecolor='black')

    formatted_mean = ["{:.2e}".format(i) for i in mean] if ((exponent < -1) | (exponent>2)) else [float("{:.2f}".format(i)) for i in mean]
    print(formatted_mean)

    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm s-1', 'SWE': 'mm s-1', 'Water production': 'mm s-1'}
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    ax.hlines(mean[0], 0, year_bkg_end - list(list_dates.keys())[0] - 1 + 1/2, linewidth=2, color=colorcycle[0],
              label=f'Background mean: %s%s' % (formatted_mean[0], units[name_series]))
    ax.hlines(mean[1], year_bkg_end - list(list_dates.keys())[0] - 1/2, year_trans_end - list(list_dates.keys())[0] - 1, linewidth=2, color=colorcycle[1],
              label=f'Transient mean: %s%s' % (formatted_mean[1], units[name_series]))

    plt.tight_layout()
    locs, labels = plt.xticks()  # Get the current locations and labels.
    locs = locs[::2]
    labels = labels[::2]
    plt.xticks(locs, labels, rotation=0)
    plt.ylabel(name_series+' ['+units[name_series]+']')
    ax.ticklabel_format(axis='y', style='sci', useMathText=True, scilimits=(-3,3))

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()

def coordinates_percentile_cdf(data_sorted, proba_bins, percentile):
    """ Function returns coordinates of the point corresponing to the percentile of the 
        Cumulated Distribution Dunction (CDF)
        
    Parameters
    ----------
    data_sorted : numpy.ndarray
        Sorted data
    proba_bins : numpy.ndarray
        list of sorted probability bins, equally spaced points between 0 and 1 with number of bins equals number of data points
    percentile : int
        Percentile of the CDF

    Returns
    -------
    low_point : list
        Coordinates of the point corresponding to the given percentile on the CDF, in the form: [data, threshold]
    """

    threshold = percentile/100
    k=0
    while proba_bins[k] < threshold:
        k += 1
    if proba_bins[k] == threshold:
        low_point = [data_sorted[k], proba_bins[k]]
    low_point = [(threshold - proba_bins[k-1])/(proba_bins[k] - proba_bins[k-1])*(data_sorted[k] - data_sorted[k-1]) + data_sorted[k-1], threshold]

    return low_point

def plot_cdf_GST(df_stats):
    """ Function returns coordinates of the point corresponing to the percentile of the 
        Cumulated Distribution Dunction (CDF)
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'

    Returns
    -------
    2 plots: left panel shows the CDF of background, transient, and evolution of mean GST
             right panel shows the CDF of background, transient, and evolution of mean SO
    """
    
    # sort the data:
    data_bkg_sorted = np.sort(df_stats['bkg_grd_temp'])
    data_trans_sorted = np.sort(df_stats['trans_grd_temp'])
    data_evol_sorted = np.sort(df_stats['evol_grd_temp'])

    data_bkg_SO_sorted = np.sort(df_stats['bkg_SO'])
    data_trans_SO_sorted = np.sort(df_stats['trans_SO'])
    data_diff_warming_sorted = np.sort(df_stats['diff_warming'])


    # calculate the proportional values of samples
    p = np.linspace(0, 1, len(data_bkg_sorted))

    point = [[[] for _ in range(3)] for _ in range(4)]

    list_data = [data_bkg_sorted, data_trans_sorted, data_bkg_SO_sorted, data_trans_SO_sorted]
    list_quant = [10,50,90]

    for indx_i, i in enumerate(list_data):
        for indx_j, j in enumerate(list_quant):
            point[indx_i][indx_j] = coordinates_percentile_cdf(i, p, j)

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    # plot the sorted data:
    fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.plot(data_bkg_sorted, p, label='Background', color=colorcycle[0], linewidth=2)
    ax1.plot(data_trans_sorted, p, label='Transient', color=colorcycle[1], linewidth=2)
    ax1.plot(data_evol_sorted, p, label='Evolution', color=colorcycle[2], linewidth=2)

    for i in range(2):
        for j in range(3):
            ax1.scatter(point[i][j][0], point[i][j][1], color=colorcycle[i])

    ylim_ax1 = ax1.get_ylim()
    xlim_ax1 = ax1.get_xlim()
    ax1.axvline(x=0, color='grey', linestyle="dashed")


    ax1.hlines([i/100 for i in list_quant], xlim_ax1[0],
            [np.max(i) for i in np.array(point).transpose()[0].transpose()[:2].transpose()],
            color='black', linestyle="dashed", linewidth=0.5)
    for i in range(2):
            ax1.vlines([j[0] for j in point[i]], ylim_ax1[0],
                    [j/100 for j in list_quant], color=colorcycle[i], linestyle="dashed", linewidth=0.5)

    ax1.set_xlim(xlim_ax1)
    ax1.set_ylim(ylim_ax1)

    ax1.set_xlabel('Mean GST [°C]')
    ax1.set_ylabel('Probability')
    ax1.legend()

    ax2.plot(data_bkg_SO_sorted, p, label='Background SO', color=colorcycle[0], linewidth=2)
    ax2.plot(data_trans_SO_sorted, p, label='Transient SO', color=colorcycle[1], linewidth=2)
    ax2.plot(data_diff_warming_sorted, p, label='Evolution SO', color=colorcycle[2], linewidth=2)
    ylim_ax2 = ax2.get_ylim()
    xlim_ax2 = ax2.get_xlim()
    ax2.axvline(x=0, color='grey', linestyle="dashed")

    for i in range(2,4):
        for j in range(3):
            ax2.scatter(point[i][j][0], point[i][j][1], color=colorcycle[i-2])

    ax2.hlines([i/100 for i in list_quant], xlim_ax1[0],
            [np.max(i) for i in np.array(point).transpose()[0].transpose()[2:].transpose()],
            color='black', linestyle="dashed", linewidth=0.5)
    for i in range(2,4):
            ax2.vlines([j[0] for j in point[i]], ylim_ax1[0],
                    [j/100 for j in list_quant], color=colorcycle[i-2], linestyle="dashed", linewidth=0.5)

    ax2.set_xlim(xlim_ax2)
    ax2.set_ylim(ylim_ax2)

    ax2.set_xlabel('Mean SO [°C]')

    # Show the graph
    plt.show()
    plt.close()
    plt.clf()

def plot_10_cold_warm(df_stats):
    """ Function returns a plot of mean GST evolution vs background GST, with an emphasis on the 10% colder and warmer simulations
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'

    Returns
    -------
    Plot of mean GST evolution vs background GST, with an emphasis on the 10% colder and warmer simulations
    """

    table_all = table_background_evolution_mean_GST_aspect_slope(df_stats)

    # create some randomly ddistributed data:
    data_bkg = [l for i in table_all[0] for j in i for k in j for l in k]
    data_evol = [l for i in table_all[2] for j in i for k in j for l in k]
    data_trans = [data_bkg[i]+data_evol[i] for i in range(len(data_bkg))]

    # sort the data:
    data_bkg_sorted = np.sort(data_bkg)
    data_trans_sorted = np.sort(data_trans)

    x = [data_bkg_sorted, data_trans_sorted]
    proba_bins = np.linspace(0, 1, len(data_bkg))
    percentiles = [10, 25, 50, 75, 90]
    table_points = [[coordinates_percentile_cdf(data, proba_bins, p)[0] for data in x] for p in percentiles]

    pd_data = pd.DataFrame(table_points, index=percentiles, columns=list(['Background', 'Transient']))
    pd_data['Difference'] = pd_data['Transient'] - pd_data['Background']

    df_stats_bis = pd.DataFrame(data=df_stats, columns=['bkg_grd_temp', 'trans_grd_temp'])
    df_stats_bis['bkg_grd_temp'] = pd.Categorical(df_stats_bis['bkg_grd_temp'], data_bkg_sorted)
    df_stats_bis = df_stats_bis.sort_values('bkg_grd_temp')

    list_x = list(df_stats_bis['bkg_grd_temp'])
    list_y = [df_stats_bis['trans_grd_temp'].iloc[i] - df_stats_bis['bkg_grd_temp'].iloc[i] for i in range(len(df_stats_bis))]

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    pos_10 = int(np.ceil(len(data_bkg)/10))

    plt.scatter(list_x[:pos_10], list_y[:pos_10], c=colorcycle[0])
    plt.scatter(list_x[pos_10:-pos_10], list_y[pos_10:-pos_10], c=colorcycle[1])
    plt.scatter(list_x[-pos_10:], list_y[-pos_10:], c=colorcycle[2])

    mean = [np.mean(list_y[:pos_10]), np.mean(list_y[-pos_10:])]

    plt.axvline((list_x[pos_10-1]+list_x[pos_10])/2, c=colorcycle[0])
    plt.axvline((list_x[-pos_10-1]+list_x[-pos_10])/2, c=colorcycle[2])
    plt.axhline(mean[0], linestyle="dashed",
                label=(r"$\overline{{\rm GST}_{\rm low 10}}$ =" + ('+' if mean[0]>0 else '') + r"%.3f°C" % (mean[0])),
                c=colorcycle[0])
    plt.axhline(mean[1], linestyle="dashed",
                label=(r"$\overline{{\rm GST}_{\rm high 10}}$ =" + ('+' if mean[1]>0 else '') + r"%.3f°C" % (mean[1])),
                c=colorcycle[2])
    
    xlim = plt.gca().get_xlim()
    plt.axvspan(xlim[0], (list_x[pos_10-1]+list_x[pos_10])/2, facecolor=colorcycle[0], alpha=0.3)
    plt.axvspan(xlim[1], (list_x[-pos_10-1]+list_x[-pos_10])/2, facecolor=colorcycle[2], alpha=0.3)
    plt.gca().set_xlim(xlim)

    plt.xlabel('Mean background GST [°C]')
    plt.ylabel('Mean GST evolution [°C]')

    # annotation_string_1 = r"$\overline{{\rm GST}_{\rm low 10}}$ =" + ('+' if mean[0]>0 else '') + r"%.3f°C" % (mean[0])
    # annotation_string_2 = r"$\overline{{\rm GST}_{\rm high 10}}$ =" + ('+' if mean[1]>0 else '') + r"%.3f°C" % (mean[1])
    # plt.annotate(annotation_string_1, (0.5, 0.9), xycoords='axes fraction', va='center', ha='center', color=colorcycle[0])
    # plt.annotate(annotation_string_2, (0.5, 0.85), xycoords='axes fraction', va='center', ha='center', color=colorcycle[2])

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()
    
def heatmap_percentile_GST(df_stats):
    """ Function returns a heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'aspect', 'slope', 'bkg_grd_temp'

    Returns
    -------
    Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference
    """

    table_all = table_background_evolution_mean_GST_aspect_slope(df_stats)

    # create some randomly ddistributed data:
    data_bkg = [l for i in table_all[0] for j in i for k in j for l in k]
    data_evol = [l for i in table_all[2] for j in i for k in j for l in k]
    data_trans = [data_bkg[i]+data_evol[i] for i in range(len(data_bkg))]

    # sort the data:
    data_bkg_sorted = np.sort(data_bkg)
    data_trans_sorted = np.sort(data_trans)

    x = [data_bkg_sorted, data_trans_sorted]
    proba_bins = np.linspace(0, 1, len(data_bkg))
    percentiles = [10, 25, 50, 75, 90]
    table_points = [[coordinates_percentile_cdf(data, proba_bins, p)[0] for data in x] for p in percentiles]

    percentiles.append('Mean')
    table_points.append([np.mean(data_bkg), np.mean(data_trans)])

    pd_data = pd.DataFrame(table_points, index=percentiles, columns=list(['Background', 'Transient']))
    pd_data['Difference'] = pd_data['Transient'] - pd_data['Background']

    pd_data['Background'] = pd.Categorical(pd_data['Background'], np.sort(pd_data['Background']))
    pd_data = pd_data.sort_values('Background')

    sn.heatmap(data=pd_data, annot=True, fmt=".2f", cmap='seismic',
                cbar=True, yticklabels=True, xticklabels=True, cbar_kws={'label': 'Mean GST [°C]'},
                norm=matplotlib.colors.TwoSlopeNorm(vmin=np.min([np.min(pd_data),-0.05]), vcenter=0, vmax=np.max([np.max(pd_data),0.05])))
    plt.ylabel('Percentile')

    # Show the graph
    plt.show()
    plt.close()
    plt.clf()

def stats_all_years_simulations_to_single_year(time_file, time_series, list_valid_sim, mask_period=None):
    """ Function returns daily mean and several quantiles of a multi-year timeseries over a 1-year period
        e.g. assigns the GST mean over all years and simulations on a given day, e.g. Dec 21st
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
    time_series : netCDF4._netCDF4.Variable
        Time series (could be temperature, precipitation, snow depth, etc.)
    list_valid_sim : list
        List of the indices of all valid simulations
    mask_period : numpy.ma.core.MaskedArray, opt
        Period could be 'background' or 'transient' and is a list of booleans, i.e. a mask.
        Can take values 'time_bkg_ground' or 'time_trans_ground' for instance.
        If None, it considers everything.

    Returns
    -------
    quantiles : pandas.core.frame.DataFrame
        Panda dataframe in the shape (5, 366)
        There is a row per quantile in [0.023, 0.16, 0.5, 0.84, 0.977] and a column per timestamp in the year (daily: 366)
    mean_end : pandas.core.series.Series
        One year time series of the mean

    """

    # create a panda dataframe with month, day, hour for each timestamp
    panda_test = pd.DataFrame(num2date(time_file[:], time_file.units), columns=['date'])
    panda_test['month'] = [i.month for i in panda_test['date']]
    panda_test['day'] = [i.day for i in panda_test['date']]
    panda_test['hour'] = [i.hour for i in panda_test['date']]
    panda_test = panda_test.drop(columns=['date'])

    if mask_period is not None:
        panda_test = panda_test.drop(index=np.arange(0,len(mask_period),1)[[not i for i in mask_period]])
 

    mean_test = []
    # for all simulations
    for sim in list_valid_sim:
        if mask_period is None:
            panda_test['timeseries'] = time_series[sim,:,0] if len(time_series.shape) == 3 else time_series[sim,:]
        else:
            panda_test['timeseries'] = time_series[sim,mask_period[:],0] if len(time_series.shape) == 3 else time_series[sim,mask_period[:]]
        # here we group the values of the multi-year timeseries by month, day, and hour and take the mean
        # this produces a single year average timeseries
        mean_test.append([i for i in panda_test.groupby(['month', 'day', 'hour']).mean().reset_index()['timeseries']])
        panda_test = panda_test.drop(columns=['timeseries'])

    mean_test_df = pd.DataFrame(np.array(mean_test))
    # 1-year timeseries of 5 quantiles corresponding to the median plus minus 1 and 2 sigmas
    quantiles = mean_test_df.quantile([0.023, 0.16, 0.5, 0.84, 0.977])
    # 1-year timeseries of the mean
    mean_end = mean_test_df.mean()

    return quantiles, mean_end

def stats_air_all_years_simulations_to_single_year(time_file, time_series, mask_period=None):
    """ Function returns daily mean and several quantiles of a multi-year timeseries over a 1-year period
        for atmospheric quantities (no simulations). The quantiles are over the years, not over simulations.
        e.g. assigns the GST mean over all years and simulations on a given day, e.g. Dec 21st
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
    time_series : netCDF4._netCDF4.Variable
        Time series (could be temperature, precipitation, snow depth, etc.)
    mask_period : numpy.ma.core.MaskedArray, opt
        Period could be 'background' or 'transient' and is a list of booleans, i.e. a mask.
        Can take values 'time_bkg_ground' or 'time_trans_ground' for instance.
        If None, it considers everything.

    Returns
    -------
    quantiles : pandas.core.frame.DataFrame
        Panda dataframe in the shape (5, 8784)
        There is a row per quantile in [0.023, 0.16, 0.5, 0.84, 0.977] and a column per timestamp in the year (hourly: 8784)
    mean_end : pandas.core.series.Series
        One year time series of the mean

    """

    # create a panda dataframe with month, day, hour for each timestamp
    panda_test = pd.DataFrame(num2date(time_file[:], time_file.units), columns=['date'])
    panda_test['month'] = [i.month for i in panda_test['date']]
    panda_test['day'] = [i.day for i in panda_test['date']]
    panda_test['hour'] = [i.hour for i in panda_test['date']]
    panda_test = panda_test.drop(columns=['date'])

    if mask_period is None:
        panda_test['timeseries'] = time_series[:,2]
    else:
        panda_test = panda_test.drop(index=np.arange(0,len(mask_period),1)[[not i for i in mask_period]])
        panda_test['timeseries'] = time_series[mask_period[:],2]


    mean_end = pd.DataFrame(np.array([i for i in panda_test.groupby(['month', 'day', 'hour']).mean().reset_index()['timeseries']]))

    list_quant = [0.023, 0.16, 0.5, 0.84, 0.977]

    panda_test = panda_test.groupby(['month', 'day', 'hour']).quantile(list_quant)
    panda_test.index.names = ['month', 'day', 'hour', 'quantiles']
    panda_test = panda_test.reset_index().drop(columns=['month', 'day', 'hour'])
    panda_test

    quantiles = []
    for i in range(len(list_quant)):
        quantiles.append(panda_test.loc[i::len(list_quant)]['timeseries'])

    quantiles = pd.DataFrame(np.array(quantiles))

    return quantiles, mean_end

def plot_sanity_one_year_quantiles_two_periods(time_file, time_series_list, list_valid_sim_list, axis_label, list_label, list_mask_period):
    """ Function returns a plot of 2 timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread.
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
    time_series_list : list of netCDF4._netCDF4.Variable
        List of time series (could be temperature, precipitation, snow depth, etc.)
    list_valid_sim_list : list of list
        List of list of the indices of all valid simulations
    axis_label : str
        Label of the y axis, could be 'GST' or 'Snow depth' for instance
    list_label : list
        List of the labels associated to each entry (if both entries are identical, can put a single label)
    list_mask_period : list
        List of the time masks for each entry, could be 'time_bkg_ground' or 'time_trans_ground' for instance.


    Returns
    -------
    Plot of 2 (or more?) timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread.
    Both series have their own y axis if they have different units.

    """
    
    # fig, ax1 = plt.subplots()
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}
    
    delta_hours = int((num2date(time_file[1], time_file.units)-num2date(time_file[0], time_file.units)).total_seconds()/3600)

    if delta_hours == 1:
        quantiles, mean_end = stats_air_all_years_simulations_to_single_year(time_file, time_series_list[0], list_mask_period[0])
    else:
        quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[0], list_valid_sim_list[0], list_mask_period[0])
    xdata = range(len(mean_end))

    indx=0
    plt.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_label[0])
    plt.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
    plt.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)
    plt.ylabel(axis_label+' ['+units[axis_label]+']')

    if delta_hours == 1:
        quantiles, mean_end = stats_air_all_years_simulations_to_single_year(time_file, time_series_list[1], list_mask_period[1])
    else:
        quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[1], list_valid_sim_list[1], list_mask_period[1])
    indx=1

    # ax2 = ax1.twinx()
    plt.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_label[1])
    plt.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
    plt.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)
    # ax2.set_ylabel(axis_label+' ['+units[axis_label]+']')

    # mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.5)

    if axis_label in ['GST', 'Air temperature']:
        plt.axhline(y=0, color='grey', linestyle='dashed')

    locs, labels = plt.xticks()
    locs = np.linspace(0, len(mean_end), num=13, endpoint=True)
    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan']
    plt.xticks(locs, labels)

    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.legend(loc="upper right")
    plt.show()
    plt.close()
    plt.clf()

def plot_sanity_two_variables_one_year_quantiles(time_file, time_series_list, list_valid_sim_list, list_label, list_site=None):
    """ Function returns a plot of 2 timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread.
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
    time_series_list : list of netCDF4._netCDF4.Variable
        List of time series (could be temperature, precipitation, snow depth, etc.)
    list_valid_sim_list : list of list
        List of list of the indices of all valid simulations
    list_label : list
        List of the labels associated to each entry (if both entries are identical, can put a single label)
    list_site : list, optional
        List of labels for the site of each entry
        If both labels are identical (i.e. if only 1 label is provided in list_label) then this means we are plotting 
        the same quantity for different sites, in which case we want to label the curves by their site

    Returns
    -------
    Plot of 2 (or more?) timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread.
    Both series have their own y axis if they have different units.

    """
    
    fig, ax1 = plt.subplots()
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}

    quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[0], list_valid_sim_list[0])
    xdata = range(len(mean_end))

    indx=0
    ax1.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_site[0] if len(list_label) == 1 else list_label[0])
    ax1.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
    ax1.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)
    ax1.set_ylabel(list_label[0]+' ['+units[list_label[0]]+']')

    quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[1], list_valid_sim_list[1])
    indx=1

    if len(list_label) == 1:
        ax1.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_site[1])
        ax1.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
        ax1.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)

    else:
        ax2 = ax1.twinx()
        ax2.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_label[1])
        ax2.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
        ax2.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)
        ax2.set_ylabel(list_label[1]+' ['+units[list_label[1]]+']')

        mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.5)

    for i in list_label:
        if i in ['GST', 'Air temperature']:
            plt.axhline(y=0, color='grey', linestyle='dashed')

    locs, labels = plt.xticks()
    locs = np.linspace(0, len(mean_end), num=13, endpoint=True)
    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan']
    plt.xticks(locs, labels)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()
    plt.close()
    plt.clf()

def plot_sanity_two_variables_one_year_quantiles_side_by_side(time_file, time_series_list, list_valid_sim_list, list_label, list_site):
    """ Function returns 2 plots side by side of 2 timeseries each reduced to a 1-year window with mean and 1 and 2-sigma spread.
        Each plot is a plot of two timeseries of the same variable at two different sites
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
    time_series_list : list of netCDF4._netCDF4.Variable
        List of time series (could be temperature, precipitation, snow depth, etc.). 1 per plot.
    list_valid_sim_list : list of list
        List of list of the indices of all valid simulations
    list_label : list
        List of the labels associated to each plot
    list_site : list
        List of labels for the site of each entry

    Returns
    -------
    Plot of 2 (or more?) timeseries reduced to a 1-year window with mean and 1 and 2-sigma spread.
    Both series have their own y axis if they have different units.

    """
    
    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}

    f, a = plt.subplots(1, 2, figsize=(10, 5))
    for idx,ax in enumerate(a):

        quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[idx][0], list_valid_sim_list[0])
        xdata = range(len(mean_end))

        indx=0
        ax.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_site[indx])
        ax.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
        ax.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)
        ax.set_ylabel(list_label[idx]+' ['+units[list_label[idx]]+']')

        quantiles, mean_end = stats_all_years_simulations_to_single_year(time_file, time_series_list[idx][1], list_valid_sim_list[1])
        indx=1

        ax.plot(xdata, mean_end, color=colorcycle[indx], linewidth=2, label=list_site[indx])
        ax.fill_between(xdata, quantiles.iloc[1], quantiles.iloc[3], alpha = 0.4, color=colorcycle[indx], linewidth=1)
        ax.fill_between(xdata, quantiles.iloc[0], quantiles.iloc[4], alpha = 0.2, color=colorcycle[indx], linewidth=0.5)

        if list_label[idx] in ['GST', 'Air temperature']:
            ax.axhline(y=0, color='grey', linestyle='dashed')

        locs = np.linspace(0, len(mean_end), num=13, endpoint=True)
        labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan']
        ax.set_xticks(locs)
        ax.set_xticklabels(labels)

        if idx == 1:
            ax.legend(loc="upper right")


    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.show()
    plt.close()
    plt.clf()

def stats_yearly_quantiles_air(time_file, time_series, label_plot, year_trans_end):
    """ Function returns yearly statistics for 'air' timeseries
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (air time)
    time_series : netCDF4._netCDF4.Variable
        List of time series with air time shape (could be air temperature, precipitation, SW, etc.)
    label_plot : str
        label associated to the plot, if 'Precipitation' or 'Water production', rescales data to mm/day
    year_trans_end : int
        Transient period is BEFORE the start of the year corresponding to the variable
    
    Returns
    -------
    panda_test : pandas.core.frame.DataFrame
        Panda dataframe of the timeseries value per day, grouped per year, month, and day
    quantiles : pandas.core.frame.DataFrame
        Panda dataframe of the timeseries [0.023, 0.16, 0.5, 0.84, 0.977] quantiles per year.
        The lenght of the table is #quantiles x #years
    mean_end : pandas.core.series.Series
        Panda dataframe of mean of the timeseries for each year
    dict_indices_quantiles : dict
        Dictionary assigning all the rows of 'quantiles' (dataframe) having information about quantile n to quantile n
    xdata : numpy.ndarray
        List of years
    """

    # create a panda dataframe with month, day, hour for each timestamp
    panda_test = pd.DataFrame(num2date(time_file[:], time_file.units), columns=['date'])
    panda_test['year'] = [i.year for i in panda_test['date']]
    panda_test['month'] = [i.month for i in panda_test['date']]
    panda_test['day'] = [i.day for i in panda_test['date']]
    panda_test['hour'] = [i.hour for i in panda_test['date']]
    panda_test = panda_test.drop(columns=['date', 'hour'])
    # Note that this is selecting the elevation in the 'middle': index 2 in the list [0,1,2,3,4]
    alt_index = int(np.floor((time_series.shape[1]-1)/2))
    panda_test['timeseries'] = time_series[:,alt_index]

    panda_test = panda_test.groupby(['year', 'month', 'day']).mean()

    if label_plot in ['Precipitation', 'Water production']:
        panda_test['timeseries'] = panda_test['timeseries']*86400

    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]
    
    list_drop = [i for i in np.unique(panda_test.index.get_level_values('year')) if i >= year_trans_end]

    quantiles = panda_test.groupby(['year']).quantile(list_quantiles).drop(index=list_drop)
    quantiles.index.names = ['year','quantile']
    quantiles = quantiles.swaplevel()
    dict_indices_quantiles = quantiles.groupby(['quantile']).indices
    mean_end = panda_test.groupby(['year']).mean().drop(index=list_drop)
    xdata = np.array(mean_end.index)

    return panda_test, quantiles, mean_end, dict_indices_quantiles, xdata

def plot_yearly_quantiles_air(time_file, time_series, label_plot, year_bkg_end, year_trans_end):
    """ Function plots yearly statistics for 'air' timeseries
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (air time)
    time_series : netCDF4._netCDF4.Variable
        List of time series with air time shape (could be air temperature, precipitation, SW, etc.)
    label_plot : str
        label associated to the plot, if 'Precipitation' or 'Water production', rescales data to mm/day
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    
    Returns
    -------
    Plot of yearly statistics for 'air' timeseries. Mean and several quantiles for each year.
    """

    panda_test, quantiles, mean_end, dict_indices_quantiles, xdata = stats_yearly_quantiles_air(time_file, time_series, label_plot, year_trans_end)
    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]

    mean_bkg = np.mean(mean_end.loc[xdata[0]:year_bkg_end-1])
    mean_trans = np.mean(mean_end.loc[year_bkg_end:year_trans_end-1])
    formatted_mean = ["{:.2f}".format(i) for i in [mean_bkg, mean_trans]]

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}
    dict_points = {0: {'alpha': 0.2, 'width': 0.5},
                1: {'alpha': 0.4, 'width': 1.0},
                2: {'alpha': 1.0, 'width': 2.0},
                3: {'alpha': 0.4, 'width': 1.0},
                4: {'alpha': 0.2, 'width': 0.5}}

    plt.scatter(xdata, mean_end, color=colorcycle[0], linestyle='None', label='Yearly mean')
    # plt.plot(xdata, mean_end, color=colorcycle[0], label='Mean')
    # plt.plot(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[2]]]['timeseries'], color=colorcycle[0])
    for i in [0,1,3,4]:
        plt.scatter(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[i]]]['timeseries'], color=colorcycle[0], alpha=dict_points[i]['alpha'], linewidth=dict_points[i]['width'])
    plt.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[1]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[3]]]['timeseries'],
                        alpha = 0.4, color=colorcycle[0], linewidth=1,
                        # label='Quantiles 16-84'
                        )
    plt.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[0]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[4]]]['timeseries'],
                        alpha = 0.2, color=colorcycle[0], linewidth=0.5,
                        # label='Quantiles 2.3-97.7'
                        )
    
    plt.hlines(mean_bkg, xdata[0], year_bkg_end, color=colorcycle[1],
               label=f'Background mean: %s%s' % (formatted_mean[0], units[label_plot]))
    plt.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[2],
               label=f'Transient mean: %s%s' % (formatted_mean[1], units[label_plot]))

    ylim = plt.gca().get_ylim()

    plt.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')

    plt.gca().set_ylim(ylim)

    if label_plot in ['GST', 'Air temperature']:
        plt.axhline(y=0, color='grey', linestyle='dashed')
    
    plt.ylabel(label_plot+' ['+units[label_plot]+']')

    # plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
    plt.clf()

def plot_yearly_quantiles_all_sims(time_file, time_series, list_valid_sim, label_plot, year_bkg_end, year_trans_end):
    """ Function plots yearly statistics for 'ground' timeseries over all simulations
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (ground time)
    time_series : netCDF4._netCDF4.Variable
        List of time series with ground time shape (could be ground temperature, SWE, etc.)
    list_valid_sim : list
        List of the indices of all valid simulations
    label_plot : str
        label associated to the plot.
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    
    Returns
    -------
    Plot of yearly statistics for 'ground' timeseries. Mean and several quantiles for each year.
    """

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}
    
    dict_points = {0: {'alpha': 0.2, 'width': 0.5},
                1: {'alpha': 0.4, 'width': 1.0},
                2: {'alpha': 1.0, 'width': 2.0},
                3: {'alpha': 0.4, 'width': 1.0},
                4: {'alpha': 0.2, 'width': 0.5}}

    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]
    
    long_timeseries = []
    for sim in list_valid_sim:
        long_timeseries.append(time_series[sim,:,0] if len(time_series.shape) == 3 else time_series[sim,:])

    long_timeseries = np.array(long_timeseries).flatten()
    long_years = np.array([int(i.year) for i in num2date(time_file[:], time_file.units)]*len(list_valid_sim))

    panda_test = pd.DataFrame(long_years.transpose(), columns=['year'])
    panda_test['timeseries'] = long_timeseries.transpose()

    list_drop = [i for i in np.unique(panda_test['year']) if i >= year_trans_end]

    quantiles = panda_test.groupby(['year']).quantile(list_quantiles).drop(index=list_drop)
    quantiles.index.names = ['year','quantile']
    quantiles = quantiles.swaplevel()
    dict_indices_quantiles = quantiles.groupby(['quantile']).indices
    mean_end = panda_test.groupby(['year']).mean().drop(index=list_drop)
    xdata = np.array(mean_end.index)

    mean_bkg = np.mean(mean_end.loc[xdata[0]:year_bkg_end-1])
    mean_trans = np.mean(mean_end.loc[year_bkg_end:year_trans_end-1])
    formatted_mean = ["{:.2f}".format(i) for i in [mean_bkg, mean_trans]]
    
    plt.scatter(xdata, mean_end, color=colorcycle[0], linestyle='None', label='Yearly mean')
    # plt.plot(xdata, mean_end, color=colorcycle[0], label='Mean')
    # plt.plot(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[2]]]['timeseries'], color=colorcycle[0])
    for i in [0,1,3,4]:
        plt.scatter(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[i]]]['timeseries'], color=colorcycle[0], alpha=dict_points[i]['alpha'], linewidth=dict_points[i]['width'])
    plt.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[1]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[3]]]['timeseries'],
                        alpha = 0.4, color=colorcycle[0], linewidth=1,
                        # label='Quantiles 16-84'
                        )
    plt.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[0]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[4]]]['timeseries'],
                        alpha = 0.2, color=colorcycle[0], linewidth=0.5,
                        # label='Quantiles 2.3-97.7'
                        )
    if label_plot in ['GST', 'Air temperature']:
        plt.axhline(y=0, color='grey', linestyle='dashed')

    plt.hlines(mean_bkg, xdata[0], year_bkg_end, color=colorcycle[1],
               label=f'Background mean: %s%s' % (formatted_mean[0], units[label_plot]))
    plt.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[2],
               label=f'Transient mean: %s%s' % (formatted_mean[1], units[label_plot]))

    ylim = plt.gca().get_ylim()
    plt.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')
    plt.gca().set_ylim(ylim)

    plt.ylabel(label_plot+' ['+units[label_plot]+']')

    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
    plt.clf()

def plot_yearly_quantiles_all_sims_side_by_side(time_file, time_series, list_valid_sim, label_plot, list_site, year_bkg_end, year_trans_end):
    """ Function plots yearly statistics for 'ground' timeseries over all simulations for 1 same metric
        at 2 different sites. 1 plot per site, both plots side by side.
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (ground time)
    time_series : netCDF4._netCDF4.Variable
        List of time series with ground time shape (could be ground temperature, SWE, etc.)
    list_valid_sim : list
        List of the indices of all valid simulations
    label_plot : str
        label associated to the plot.
    list_site : list
        List of labels for the site of each entry
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
    
    Returns
    -------
    Plots of yearly statistics for 'ground' timeseries over all simulations for 1 same metric
    at 2 different sites. 1 plot per site, both plots side by side.
    """

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    units = {'GST': '°C', 'Air temperature': '°C', 'Precipitation': 'mm/day', 'SWE': 'mm', 'Water production': 'mm/day', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}
    
    dict_points = {0: {'alpha': 0.2, 'width': 0.5},
                1: {'alpha': 0.4, 'width': 1.0},
                2: {'alpha': 1.0, 'width': 2.0},
                3: {'alpha': 0.4, 'width': 1.0},
                4: {'alpha': 0.2, 'width': 0.5}}

    list_quantiles = [0.023, 0.16, 0.5, 0.84, 0.977]

    f, a = plt.subplots(1, 2, figsize=(8, 4), sharey='row')
    for idx,ax in enumerate(a):
        long_timeseries = []
        for sim in list_valid_sim[idx]:
            long_timeseries.append(time_series[idx][sim,:,0] if len(time_series[idx].shape) == 3 else time_series[idx][sim,:])

        long_timeseries = np.array(long_timeseries).flatten()
        long_years = np.array([int(i.year) for i in num2date(time_file[:], time_file.units)]*len(list_valid_sim[idx]))

        panda_test = pd.DataFrame(long_years.transpose(), columns=['year'])
        panda_test['timeseries'] = long_timeseries.transpose()

        list_drop = [i for i in np.unique(panda_test['year']) if i >= year_trans_end]

        quantiles = panda_test.groupby(['year']).quantile(list_quantiles).drop(index=list_drop)
        quantiles.index.names = ['year','quantile']
        quantiles = quantiles.swaplevel()
        dict_indices_quantiles = quantiles.groupby(['quantile']).indices
        mean_end = panda_test.groupby(['year']).mean().drop(index=list_drop)
        xdata = np.array(mean_end.index)

        mean_bkg = np.mean(mean_end.loc[xdata[0]:year_bkg_end-1])
        mean_trans = np.mean(mean_end.loc[year_bkg_end:year_trans_end-1])
        formatted_mean = ["{:.2f}".format(i) for i in [mean_bkg, mean_trans]]
        
        ax.scatter(xdata, mean_end, color=colorcycle[idx], linestyle='None', label='Yearly mean')
        # ax.plot(xdata, mean_end, color=colorcycle[idx], label='Mean')
        # plt.plot(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[2]]]['timeseries'], color=colorcycle[0])
        for i in [0,1,3,4]:
            ax.scatter(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[i]]]['timeseries'], color=colorcycle[idx], alpha=dict_points[i]['alpha'], linewidth=dict_points[i]['width'])
        ax.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[1]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[3]]]['timeseries'],
                            alpha = 0.4, color=colorcycle[idx], linewidth=1,
                            # label='Quantiles 16-84'
                            )
        ax.fill_between(xdata, quantiles.iloc[dict_indices_quantiles[list_quantiles[0]]]['timeseries'], quantiles.iloc[dict_indices_quantiles[list_quantiles[4]]]['timeseries'],
                            alpha = 0.2, color=colorcycle[idx], linewidth=0.5,
                            # label='Quantiles 2.3-97.7'
                            )

        ax.hlines(mean_bkg, xdata[0], year_bkg_end, color=colorcycle[2],
               label=f'Background mean: %s%s' % (formatted_mean[0], units[label_plot]))
        ax.hlines(mean_trans,  year_bkg_end, xdata[-1], color=colorcycle[3],
               label=f'Transient mean: %s%s' % (formatted_mean[1], units[label_plot]))
            
        if label_plot in ['GST', 'Air temperature']:
            ax.axhline(y=0, color='grey', linestyle='dashed')

        ax.title.set_text(list_site[idx])
        
        if idx==0:
            ax.set_ylabel(label_plot+' ['+units[label_plot]+']')

        ax.legend(loc='upper right' if idx==0 else 'lower right')

    ylim = plt.gca().get_ylim()

    for ax in a:
        ax.vlines(year_bkg_end, ylim[0], ylim[1], color='grey', linestyle='dashed')

    plt.gca().set_ylim(ylim)

    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    # Show the graph
    plt.show()
    plt.close()
    plt.clf()

def plot_GST_bkg_vs_evol_quantile_bins_fit_single_site(df_stats):
    """ Function return scatter plot of background GST vs GST evolution for a single site.
    The site is binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced too.
    
    Parameters
    ----------
    df_stats : pandas.core.frame.DataFrame
        Panda dataframe with at least the following columns: 'bkg_grd_temp' and 'evol_grd_temp'
    
    Returns
    -------
    slope : float
        Linear regression slope
    intercept : float
        Linear regression intercept
    r : float
        Linear regression r-value. Need to square it to get R^2.
    Scatter plot of background GST vs GST evolution for the single site.
    The single site is binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced too.
    """

    df_stats_bis = pd.DataFrame(data=df_stats, columns=['bkg_grd_temp', 'evol_grd_temp'])
    df_stats_bis['bkg_grd_temp'] = pd.Categorical(df_stats_bis['bkg_grd_temp'], np.sort(df_stats['bkg_grd_temp']))
    df_stats_bis = df_stats_bis.sort_values('bkg_grd_temp')

    list_x = list(df_stats_bis['bkg_grd_temp'])
    list_y = list(df_stats_bis['evol_grd_temp'])

    quantiles = np.arange(0, 101, 10)

    cmap = plt.cm.seismic

    list_x_mean = []

    for i in range(len(quantiles)-1):
        low = int(np.ceil(len(df_stats)*quantiles[i]/100))
        up = int(np.ceil(len(df_stats)*quantiles[i+1]/100))
        list_x_mean.append(np.mean(list_x[low:up]))
        # plt.hlines(np.mean(list_y[low:up]),list_x[low],list_x[up-1], color=colorcycle[i], linewidth=2)

    vmax = np.max(np.abs(list_x_mean))

    for i in range(len(quantiles)-1):
        color = cmap((list_x_mean[i] + vmax)/(2*vmax))
        low = int(np.ceil(len(df_stats)*quantiles[i]/100))
        up = int(np.ceil(len(df_stats)*quantiles[i+1]/100))
        plt.scatter(list_x[low:up], list_y[low:up], c=color,s=0.8)
        plt.errorbar(np.mean(list_x[low:up]), np.mean(list_y[low:up]), np.std(list_y[low:up]), np.std(list_x[low:up]), c=color)
        plt.scatter(np.mean(list_x[low:up]), np.mean(list_y[low:up]), c=color, s=50)
        # plt.hlines(np.mean(list_y[low:up]),list_x[low],list_x[up-1], color=colorcycle[i], linewidth=2)

    slope, intercept, r, p, se = linregress(list_x, list_y)
    u = np.arange(np.min(list_x)-0.1, np.max(list_x)+0.1, 0.01)
    print('R-square:', r**2, ', regression slope:', slope , ', regression intercept:', intercept)
    plt.plot(u, slope*u+intercept, c='grey', label=('slope: %s' % (round(slope,3))))

    plt.xlabel('Mean background GST [°C]')
    plt.ylabel('Mean GST evolution [°C]')

    # Show the graph
    # plt.legend()
    plt.show()
    plt.close()
    plt.clf()

    return slope, intercept, r

def plot_GST_bkg_vs_evol_quantile_bins_fit(df_stats, list_site):
    """ Function return scatter plot of background GST vs GST evolution for 2 sites.
    Both sites are binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced for each site.
    
    Parameters
    ----------
    df_stats : list of pandas.core.frame.DataFrame
        List of 2 panda dataframe with at least the following columns: 'bkg_grd_temp' and 'evol_grd_temp'
    list_site : list
        List of labels for the site of each entry
    
    Returns
    -------
    slope : list
        List of linear regression slope (1 for each site)
    intercept : list
        List of linear regression intercept (1 for each site)
    r : list
        List of linear regression r-value (1 for each site). Need to square it to get R^2.
    Scatter plot of background GST vs GST evolution for 2 sites.
    Both sites are binned in 10 bins of equal sizes and each bin is represented by a dot with x and y error bars.
    A linear regression is produced for each site.
    """

    num = len(df_stats)

    df_stats_bis = [pd.DataFrame(data=i, columns=['bkg_grd_temp', 'evol_grd_temp']) for i in df_stats]
    for i in range(num):
        df_stats_bis[i]['bkg_grd_temp'] = pd.Categorical(df_stats_bis[i]['bkg_grd_temp'], np.sort(df_stats[i]['bkg_grd_temp']))
        df_stats_bis[i] = df_stats_bis[i].sort_values('bkg_grd_temp')

    list_x = [list(i['bkg_grd_temp']) for i in df_stats_bis]
    list_y = [list(i['evol_grd_temp']) for i in df_stats_bis]

    colorcycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    quantiles = np.arange(0, 101, 10)

    cmap = plt.cm.seismic
    # colors = cmap(np.linspace(0, 1, len(quantiles)+(1 if len(quantiles)%2 else 0)))

    vmax = [[] for _ in df_stats]
    list_x_mean = [[] for _ in df_stats]

    for j in range(num):

        for i in range(len(quantiles)-1):
            low = int(np.ceil(len(df_stats[j])*quantiles[i]/100))
            up = int(np.ceil(len(df_stats[j])*quantiles[i+1]/100))
            list_x_mean[j].append(np.mean(list_x[j][low:up]))

        vmax[j] = np.max(np.abs(list_x_mean[j]))

    vmax = np.max(vmax)

    points = []
    slope = []
    intercept = []
    r = []

    for j in range(num):

        for i in range(len(quantiles)-1):
            color = cmap((list_x_mean[j][i] + vmax)/(2*vmax))
            low = int(np.ceil(len(df_stats[j])*quantiles[i]/100))
            up = int(np.ceil(len(df_stats[j])*quantiles[i+1]/100))
            plt.scatter(list_x[j][low:up], list_y[j][low:up], c=color,s=0.8)
            plt.errorbar(np.mean(list_x[j][low:up]), np.mean(list_y[j][low:up]), np.std(list_y[j][low:up]), np.std(list_x[j][low:up]), c=color)
            plt.scatter(np.mean(list_x[j][low:up]), np.mean(list_y[j][low:up]), c=color, s=50)
            # plt.hlines(np.mean(list_y[low:up]),list_x[low],list_x[up-1], color=colorcycle[i], linewidth=2)

        slope_i, intercept_i, r_i, p, se = linregress(list_x[j], list_y[j])
        slope.append(slope_i)
        intercept.append(intercept_i)
        r.append(r_i)
        u = np.arange(np.min(list_x[j])-0.05, np.max(list_x[j])+0.05, 0.01)
        print('R-square:', r_i**2, ', regression slope:', slope_i , ', regression intercept:', intercept_i)
        line_j, = plt.plot(u, slope_i*u+intercept_i, c=colorcycle[j], label=list_site[j])
        if j==0:
            first_legend = plt.legend(handles=[line_j], loc='upper left')

        xpts = np.max(list_x[j])+0.05 if j==0 else np.min(list_x[j])-0.05
        ypts = slope_i*xpts+intercept_i

        points.append([xpts, ypts])

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    u = np.arange(xlim[0], xlim[1], 0.01)
    slope_divide = (points[0][1]-points[1][1])/(points[0][0]-points[1][0])
    intercept_divide = points[1][1] - slope_divide*points[1][0]
    line = slope_divide*u+intercept_divide
    plt.plot(u, line, c='grey', linestyle='dashed')

    plt.fill_between(u, ylim[0]*len(u), line, alpha = 0.1, color=colorcycle[0], linewidth=1)
    plt.fill_between(u, line, ylim[1]*len(u), alpha = 0.1, color=colorcycle[1], linewidth=1)

    plt.plot()

    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)

    plt.xlabel('Mean background GST [°C]')
    plt.ylabel('Mean GST evolution [°C]')

    # Show the graph
    # first_legend = plt.legend(handles=[line[0]], loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[line_j], loc='lower right')
    plt.show()
    plt.close()
    plt.clf()

    return slope, intercept, r

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
    for i in range(len(r)):
        ax.fill_between( np.linspace(theta[i], (0 if i==len(r)-1 else theta[i+1]), 2), 0, r[i], color='deepskyblue', alpha=0.5)
        ax.fill_between( np.linspace(theta[i], (0 if i==len(r)-1 else theta[i+1]), 2), r[i], 90, color='black', alpha=1)
    ax.scatter(theta, r, c='blue', s=10, cmap='hsv', alpha=0.75)
    # plt.title('Scatter Plot on Polar Axis', fontsize=15)
    plt.show()

def get_all_stats(forcing_list, path_forcing_list, path_repository, path_ground, path_snow,
                  year_bkg_end, year_trans_end, consecutive, extension='',
                  glacier=False, min_glacier_depth=100, max_glacier_depth=20000):
    """ Creates a number of pickle files (if they don't exist yet)
    
    Parameters
    ----------
    forcing_list : list of str
        List of forcings provided, with a number of entries between 1 and 3 in 'era5', 'merra2', and 'jra55'. E.g. ['era5', 'merra2']
    path_forcing_list : list of str
        List of paths to the .nc file where the atmospheric forcing data for the given reanalysis is stored
    path_repository : str
        Path to the .csv file with all the simulation parameters
    path_ground : str
        Path to the .nc file where the aggregated ground simulations are stored
    path_snow : str
        path to the snow data
    year_bkg_end : int
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int
        Same for transient period
    consecutive : int
        number of consecutive snow-free days to consider that snow has melted for the season
    extension : str, optional
        Location of the event, e.g. 'Aksaut_Ridge' (not a list, 1 single location)
    glacier : bool, optional
        By default only keeps non-glacier simulations but can be changed to True to select only glaciated simulations
    min_glacier_depth : float, optional
        Selects simulation with minimum snow height higher than this threshold (in mm)
    max_glacier_depth : float, optional
        Selects simulation with minimum snow height lower than this threshold (in mm)

    Returns
    -------
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

    """

    df = assign_value_df(path_repository, path_ground, extension)
    reanalysis_stats = assign_value_reanalysis_stat(forcing_list, path_forcing_list, year_bkg_end, year_trans_end, extension)
    list_valid_sim = glacier_filter(path_snow, extension, glacier, min_glacier_depth, max_glacier_depth)
    dict_melt_out, stats_melt_out_dic = melt_out_date(consecutive, path_ground, path_snow, year_bkg_end, year_trans_end, extension)
    df_stats = assign_value_df_stats(path_ground, path_snow, year_bkg_end, year_trans_end, extension)

    return df, reanalysis_stats, list_valid_sim, dict_melt_out, stats_melt_out_dic, df_stats

def load_all_pickles(extension=''):
    """ Loads all pickles corresponding to the site name
    
    Parameters
    ----------
    extension : str, optional
        Location of the event, e.g. 'Aksaut_Ridge'

    Returns
    -------
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

    """

    list_file_names = [f"df{('' if extension=='' else '_')}{extension}.pkl",
                       f"reanalysis_stats{('' if extension=='' else '_')}{extension}.pkl",
                       f"list_valid_sim{('' if extension=='' else '_')}{extension}.pkl",
                       f"melt_out{('' if extension=='' else '_')}{extension}.pkl",
                       f"df_stats{('' if extension=='' else '_')}{extension}.pkl"]

    output = [0 for _ in list_file_names]

    for i, file_name in enumerate(list_file_names):
        my_path = pickle_path + file_name
        # Open the file in binary mode 
        with open(my_path, 'rb') as file: 
            # Call load method to deserialze 
            output[i] = pickle.load(file) 
        print('Succesfully opened the pre-existing pickle:', file_name)

    [df, reanalysis_stats, list_valid_sim, [dict_melt_out, stats_melt_out_dic], df_stats] = output

    return df, reanalysis_stats, list_valid_sim, dict_melt_out, stats_melt_out_dic, df_stats

def plot_all_generic(site, forcings, path_forcings,
                     path_ground, path_snow, path_swe, path_thaw_depth,
                     year_bkg_end=2010, year_trans_end=2023,
                     individual_heatmap=False, polar_plots=False,
                     parity_plot=False):
    """ Function returns a series of summary plots for a given site.
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    forcings : list of str
        List of forcings provided, with a number of entries between 1 and 3 in 'era5', 'merra2', and 'jra55'. E.g. ['era5', 'merra2']
    path_forcings : list of str
        List of string paths to the locations of the different forcings for that particular site (.nc)
    path_ground : str
        String path to the location of the ground output file from GTPEM (.nc)
    path_snow : str
        String path to the location of the snow output file from GTPEM (.nc)
    path_thaw_depth : str
        String path to the location of the thaw depth output file from GTPEM (.nc)
    year_bkg_end : int, optional
        Background period is BEFORE the start of the year corresponding to the variable, i.e. all time stamps before Jan 1st year_bkg_end
    year_trans_end : int, optional
        Same for transient period
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

    df, reanalysis_stats, list_valid_sim, dict_melt_out, stats_melt_out_dic, df_stats = load_all_pickles(extension)

    #####################################################################################
    # PLOTS
    #####################################################################################

    # assign a subjective weight to all simulations
    pd_weight = assign_weight_sim(extension, no_weight)

    # weighted mean GST
    temp_ground_mean = list(np.average([temp_ground[i,:,0] for i in list(pd_weight.index.values)], axis=0, weights=pd_weight['weight']))
    print('The following plot is a histogram of the distribution of the statistical weights of all simulations:')
    plot_hist_stat_weights(pd_weight, df, zero=True)
    print('The following plot is a histogram of the distribution of glacier simulations wrt to altitude, aspect, slope, and forcing:')
    plot_hist_valid_sim_all_variables(df, df_stats, depth_thaw)

    alt_list = sorted(set(df_stats['altitude']))
    alt_index = int(np.floor((len(alt_list)-1)/2))
    alt_index_abs = alt_list[alt_index]
    print('List of altitudes:', alt_list)
    print('Altitude at which we plot the time series:', alt_index_abs)

    # Note that this is selecting the elevation in the 'middle': index 2 in the list [0,1,2,3,4]
    # and it returns the mean air temperature over all reanalyses
    mean_air_temp = mean_all_reanalyses(time_air_all, [i[:,alt_index] for i in temp_air_all])
    # here we get the mean precipitation and then water from snow melting 
    mean_prec = mean_all_reanalyses(time_air_all, [i[:,alt_index] for i in precipitation_all])
    swe_mean = list(np.average([swe[i,:] for i in list(pd_weight.index.values)], axis=0, weights=pd_weight['weight']))
    # finally we get the total water production, averaged over all reanalyses
    tot_water_prod = assign_tot_water_prod_generic(swe_mean, mean_prec, time_ground, time_pre_trans_ground, time_air_merra2)

    year_rockfall = rockfall_values(site)['year']
    print('Plots of the normalized distance of air and ground temperature, water production, and thaw_depth as a function of time')
    print('Granularity: week and month side by side')
    plot_aggregating_distance_temp_all_generic(['Air temperature', 'Water production', 'Ground temperature'],
                                       [time_air_all[0], time_ground, time_ground],
                                       [mean_air_temp, tot_water_prod, temp_ground_mean],
                                       ['week', 'month'], site, year_rockfall, year_bkg_end, year_trans_end, False)
    print('Granularity: year, plotted for all years')
    plot_aggregating_distance_temp_all_generic(['Air temperature', 'Water production', 'Ground temperature'],
                                        [time_air_all[0], time_ground, time_ground],
                                        [mean_air_temp, tot_water_prod, temp_ground_mean],
                                        ['year'], site, 0, year_bkg_end, year_trans_end, False)

    print('Yearly statistics for air and ground surface temperature, and also precipitation and water production')
    plot_box_yearly_stat('Air temperature', time_air_all[0], mean_air_temp, year_bkg_end=2010, year_trans_end=2023)
    plot_box_yearly_stat('GST', time_ground, temp_ground_mean, year_bkg_end=2010, year_trans_end=2023)
    plot_box_yearly_stat('Precipitation', time_ground, mean_prec, year_bkg_end=2010, year_trans_end=2023)
    plot_box_yearly_stat('Water production', time_ground, tot_water_prod, year_bkg_end=2010, year_trans_end=2023)

    if individual_heatmap:
        print('Heatmap of the background mean GST as a function of aspect and slope at %s m:' % alt_index_abs)
        plot_table_mean_GST_aspect_slope_generic(df_stats, site, alt_index_abs, True, False)
        print('Heatmap of the evolution of the mean GST between the background and the transient periods as a function of aspect and slope at %s m:' % alt_index_abs)
        plot_table_mean_GST_aspect_slope_generic(df_stats, site, alt_index_abs, False, False)

    print('Heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitude')
    plot_table_aspect_slope_all_altitudes_generic(df, df_stats, site, show_glacier=False, box=False)


    if polar_plots:
        print('Polar heatmap of the background mean GST and its evolution as a function of aspect and slope at all altitude')
        plot_table_aspect_slope_all_altitudes_polar_generic(df_stats, site, False)

        print('Polar plot of the permafrost and glacier spatial distribution as a function of aspect and slope at all altitude')
        plot_permafrost_all_altitudes_polar_generic(df_stats, site, depth_thaw, False)

    print('CDF of background, transient, and evolution GST:')
    plot_cdf_GST(df_stats)
    print('Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient GST, and the difference:')
    heatmap_percentile_GST(df_stats)
    print('Plot of mean GST evolution vs background GST, with an emphasis on the 10% colder and warmer simulations')
    plot_10_cold_warm(df_stats)
    print('Plot of mean GST evolution vs background GST, fit, and binning per 10% quntiles')
    plot_GST_bkg_vs_evol_quantile_bins_fit_single_site(df_stats)

    print('Scatter plot of mean background GST vs evolution of mean GST between the background and transient period')
    plot_mean_bkg_GST_vs_evolution(df_stats)

    if parity_plot:
        print('Parity plot (statistically-modeled vs numerically-simulated) of background mean GST:')
        xdata, ydata, optimizedParameters, pcov, corr_matrix, R_sq = fit_stat_model_grd_temp_Aksaut(df_stats, all=False, diff_forcings=True)
        list_ceof = ['offset', 'c_alt', 'd_alt', 'c_asp', 'c_slope']
        pd_coef = pd.DataFrame(list_ceof, columns=['Coefficient'])
        # previously was columns=['all', 'era5', 'merra2', 'jra55'] when had all 3 forcings
        pd_coef = pd.concat([pd_coef, pd.DataFrame((np.array([list(i) for i in optimizedParameters]).transpose()), columns=forcings)], axis=1)
        print('The coefficients of the statistical model for the mean background GST are given by:')
        print(pd_coef)
