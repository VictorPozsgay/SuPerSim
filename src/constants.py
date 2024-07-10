"""This module defines the constants needed for the package"""

def save_constants():
    """ Function returns the constants needed for the package:
    colorcycle for plots and units for labels
    
    Parameters
    ----------

    Returns
    -------
    colorcycle : list
        Matplotlib default color cycle
    units : dict
        Dictionary with units for each metric

    """
    colorcycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    units = {'GST': '°C', 'Air temperature': '°C',
             'Precipitation': 'mm/day', 'Water production': 'mm/day',
             'SWE': 'mm', 'Snow depth': 'mm',
             'SW': 'W m-2', 'LW': 'W m-2'}
    return colorcycle, units
