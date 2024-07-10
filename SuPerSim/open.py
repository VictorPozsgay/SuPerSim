"""This module defines the functions that will open the netCDF (.nc) files"""

#pylint: disable=invalid-name

from netCDF4 import Dataset #pylint: disable=no-name-in-module

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
    """ Function returns data from the .nc file
    for the snow water equivalent (SWE) simulation results
    
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
