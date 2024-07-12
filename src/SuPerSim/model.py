"""This module creates a statistical model for ground temperatures vs topography"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from SuPerSim.pickling import load_all_pickles
from SuPerSim.constants import save_constants

colorcycle, _ = save_constants()

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

def fit_stat_model_grd_temp(site, path_pickle, all_data=True, diff_forcings=True):
    """ Function returns the value of the statistical model 
    
    Parameters
    ----------
    site : str
        Location of the event, e.g. 'Joffre' or 'Fingerpost'
    path_pickle : str
        String path to the location of the folder where the pickles are saved
    all_data : bool, optional
        If True, considers all data at once
    diff_forcings : bool, optional
        If True, separates data by 'forcing'

    Returns
    -------
    xdata : list
        List of xdata (actual) grouped by forcing if all_data=False
    ydata : list
        List of ydata (predicted) grouped by forcing if all_data=False
    optimizedParameters : list
        List of optimized model parameters grouped by forcing if all_data=False
    pcov : list
        List of covariances grouped by forcing if all_data=False
    corr_matrix : list
        List of correlation matrices grouped by forcing if all_data=False
    R_sq : list
        List of R^2 grouped by forcing if all_data=False
    parity plot (predicted vs actual)
    """

    _, _, _, _, _, df_stats, _ = load_all_pickles(site, path_pickle)
    
    plt.figure(figsize=(6,6))

    forcings = np.unique(df_stats['forcing'])
    data_set = []
    if all_data:
        data_set.append(df_stats)
    else: 
        pass
    if diff_forcings:
        for i in forcings:
            data_set.append(df_stats[df_stats['forcing']==i])
    else:
        pass

    input_var = [np.array([i['aspect'], i['slope'], i['altitude']]) for i in data_set]
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
    for i,in_var in enumerate(input_var):
        optimizedParameters.append(opt.curve_fit(stat_model_aspect_slope_alt, in_var, xdata[i], bounds=bounds, p0=p0)[0])
        pcov.append(opt.curve_fit(stat_model_aspect_slope_alt, in_var, xdata[i], bounds=bounds, p0=p0)[0])

        # this represents the fitted values, hence they are the statistically-modelled values 
        # of differential warming: we call them ydata
        ydata.append(stat_model_aspect_slope_alt(in_var, *optimizedParameters[i]))

        # R^2 from numpy package, to check!
        corr_matrix.append(np.corrcoef(xdata[i], ydata[i]))
        corr = corr_matrix[i][0,1]
        R_sq.append(corr**2)

        plt.scatter(xdata[i], ydata[i], marker=("D" if (all_data and i==0) else "o"),
                    s=20, label=('all data' if (all_data and i==0) else forcings[len(forcings)-len(data_set)+i]) )
        
    # plot the y=x diagonal
    # start by setting the bounds

    lim_up = float(f"{max(np.max([np.max(i) for i in xdata]), np.max([np.max(i) for i in ydata])):.2g}")
    lim_down = float(f"{min(np.min([np.min(i) for i in xdata]), np.min([np.min(i) for i in ydata])):.2g}")
    x = np.arange(lim_down, lim_up, 0.01)
    plt.plot(x, x, color=colorcycle[len(data_set)], linestyle='dashed', label = 'y=x', linewidth=2)
    plt.legend(loc='upper right')

    margin = 0.1
    plt.ylim(ymin= lim_down - margin, ymax= lim_up + margin)
    plt.xlim(xmin= lim_down - margin, xmax= lim_up + margin)

    plt.xlabel(r'Numerically-simulated background GST $\overline{T_{\rm GST}^{\rm bkg}}_{(NS)}$ [°C]')
    plt.ylabel(r'Statistically-modelled background GST $\overline{T_{\rm GST}^{\rm bkg}}_{(SM)}$ [°C]')
    for i,r in enumerate(R_sq):
        plt.figtext(.7, .3 - i/30, f"$R^2$ = {r:.2f}",
                    c=colorcycle[i])

    # Show the graph
    plt.legend()
    plt.show()
    plt.close()

    return xdata, ydata, optimizedParameters, pcov, corr_matrix, R_sq
