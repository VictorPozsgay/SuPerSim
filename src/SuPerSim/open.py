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

def open_SW_direct_nc(path_SW_direct):
    """ Function returns data from the .nc file for the SW_direct simulations
    
    Parameters
    ----------
    path_SW_direct : str
        Path to the .nc file where the aggregated SW_direct simulations are stored

    Returns
    -------
    f_SW_direct : netCDF4._netCDF4.Group
        geotop netCDF group with description of dimensions, variables, and data for the SW_direct
    SW_direct_height : netCDF4._netCDF4.Variable
        Depth of SW_direct in the shape (simulation, time)
    """

    # Open file for SW_direct depth
    ncfile_SW_direct = Dataset(path_SW_direct, mode='r')
    # Select geotop model data
    f_SW_direct = ncfile_SW_direct.groups['geotop']

    SW_direct = f_SW_direct['SW_direct']

    return f_SW_direct, SW_direct

def open_SW_diffuse_nc(path_SW_diffuse):
    """ Function returns data from the .nc file for the SW_diffuse simulations
    
    Parameters
    ----------
    path_SW_diffuse : str
        Path to the .nc file where the aggregated SW_diffuse simulations are stored

    Returns
    -------
    f_SW_diffuse : netCDF4._netCDF4.Group
        geotop netCDF group with description of dimensions, variables, and data for the SW_diffuse
    SW_diffuse_height : netCDF4._netCDF4.Variable
        Depth of SW_diffuse in the shape (simulation, time)
    """

    # Open file for SW_diffuse depth
    ncfile_SW_diffuse = Dataset(path_SW_diffuse, mode='r')
    # Select geotop model data
    f_SW_diffuse = ncfile_SW_diffuse.groups['geotop']

    SW_diffuse = f_SW_diffuse['SW_diffuse']

    return f_SW_diffuse, SW_diffuse

def open_SW_up_nc(path_SW_up):
    """ Function returns data from the .nc file for the SW_up simulations
    
    Parameters
    ----------
    path_SW_up : str
        Path to the .nc file where the aggregated SW_up simulations are stored

    Returns
    -------
    f_SW_up : netCDF4._netCDF4.Group
        geotop netCDF group with description of dimensions, variables, and data for the SW_up
    SW_up_height : netCDF4._netCDF4.Variable
        Depth of SW_up in the shape (simulation, time)
    """

    # Open file for SW_up depth
    ncfile_SW_up = Dataset(path_SW_up, mode='r')
    # Select geotop model data
    f_SW_up = ncfile_SW_up.groups['geotop']

    SW_up = f_SW_up['SW_up']

    return f_SW_up, SW_up

def open_SW_down_nc(path_SW_down):
    """ Function returns data from the .nc file for the SW_down simulations
    
    Parameters
    ----------
    path_SW_down : str
        Path to the .nc file where the aggregated SW_down simulations are stored

    Returns
    -------
    f_SW_down : netCDF4._netCDF4.Group
        geotop netCDF group with description of dimensions, variables, and data for the SW_down
    SW_down_height : netCDF4._netCDF4.Variable
        Depth of SW_down in the shape (simulation, time)
    """

    # Open file for SW_down depth
    ncfile_SW_down = Dataset(path_SW_down, mode='r')
    # Select geotop model data
    f_SW_down = ncfile_SW_down.groups['geotop']

    SW_down = f_SW_down['SW_down']

    return f_SW_down, SW_down

def open_SW_net_nc(path_SW_net):
    """ Function returns data from the .nc file for the SW_net simulations
    
    Parameters
    ----------
    path_SW_net : str
        Path to the .nc file where the aggregated SW_net simulations are stored

    Returns
    -------
    f_SW_net : netCDF4._netCDF4.Group
        geotop netCDF group with description of dimensions, variables, and data for the SW_net
    SW_net_height : netCDF4._netCDF4.Variable
        Depth of SW_net in the shape (simulation, time)
    """

    # Open file for SW_net depth
    ncfile_SW_net = Dataset(path_SW_net, mode='r')
    # Select geotop model data
    f_SW_net = ncfile_SW_net.groups['geotop']

    SW_net = f_SW_net['SW_net']

    return f_SW_net, SW_net

def open_LW_net_nc(path_LW_net):
    """ Function returns data from the .nc file for the LW_net simulations
    
    Parameters
    ----------
    path_LW_net : str
        Path to the .nc file where the aggregated LW_net simulations are stored

    Returns
    -------
    f_LW_net : netCDF4._netCDF4.Group
        geotop netCDF group with description of dimensions, variables, and data for the LW_net
    LW_net_height : netCDF4._netCDF4.Variable
        Depth of LW_net in the shape (simulation, time)
    """

    # Open file for LW_net depth
    ncfile_LW_net = Dataset(path_LW_net, mode='r')
    # Select geotop model data
    f_LW_net = ncfile_LW_net.groups['geotop']

    LW_net = f_LW_net['LW_net']

    return f_LW_net, LW_net
