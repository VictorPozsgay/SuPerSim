"""This module defines the functions that sort the timestamps in bins (years, months, etc.)"""

#pylint: disable=line-too-long

from datetime import datetime
import re
from numpy import ma
import numpy as np
from netCDF4 import num2date #pylint: disable=no-name-in-module

def time_unit_stamp(time_file):
    """ Function returns frequency of datapoints and exact date and time
    of the initial datapoint, converted into a datetime
    
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

def specific_time_to_index(time_file, date):
    """ Function returns the timeseries time index corresponding to a given date
    
    Parameters
    ----------
    time_file : netCDF4._netCDF4.Variable
        File where the time index of each datapoint is stored (time_ground, not time_air)
        Needs to have a .units attribute
    date : datetime.datetime
        Datetime of the event, e.g. datetime(2019, 5, 13, 0, 0)

    Returns
    -------
    time_stamp : int
        Time stamp of the particular date for the timeseries
    """

    delta_cons_time = int((num2date(time_file[1], time_file.units) - num2date(time_file[0], time_file.units)).total_seconds())
    init_delta = int((date - num2date(time_file[0], time_file.units)).total_seconds())
    time_stamp = int(init_delta/delta_cons_time)
    refined_delta = int((date - num2date(time_file[time_stamp], time_file.units)).total_seconds())
    while refined_delta < 0:
        time_stamp -= 1
        refined_delta = int((date - num2date(time_file[time_stamp], time_file.units)).total_seconds())
    while refined_delta > 0:
        time_stamp += 1
        refined_delta = int((date - num2date(time_file[time_stamp], time_file.units)).total_seconds())

    return time_stamp
