"""This module creates a horizon plot (fisheye sky view)"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_horizon_to_polar(path_horizon):
    """ Function read a .csv horizon file with columns 'azi' (azimuth) and 'hang' (horizon angle) and formats them.
    
    Parameters
    ----------
    path_horizon : str
        Path to the .csv horizon file

    Returns
    -------
    theta : numpy.ndarray
        azimuth angle (clockwise from North) in degrees increasing from 0 to 2pi
    zenith_angle : numpy.ndarray
        horizon angle in degrees (from the zenith) at corresponding theta (azimuth) direction
        from 0 to 90
        0 corresponds to zenith (vertical wall), 90 corresponds to flat horizon
    """

    hor_file = pd.read_csv(path_horizon, usecols=['azi', 'hang'])
    theta_pre = hor_file['azi']
    theta = np.array([i/360*2*np.pi for i in theta_pre])
    zenith_angle_pre = hor_file['hang']
    zenith_angle = np.array([90-i for i in zenith_angle_pre])

    return theta, zenith_angle

def plot_visible_skymap(theta, zenith_angle):
    """ Function returns a fisheye view of the sky with the visible portion in blue and the blocked one in black.
    
    Parameters
    ----------
    theta : numpy.ndarray
        azimuth angle (clockwise from North) in degrees increasing from 0 to 2pi
    zenith_angle : numpy.ndarray
        horizon angle in degrees (from the zenith) at corresponding theta (azimuth) direction
        from 0 to 90
        0 corresponds to zenith (vertical wall), 90 corresponds to flat horizon

    Returns
    -------
    fig : figure
        Plot of the sky view from the location
    """

    # Creating the polar scatter plot
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    ax.set_ylim(0, 90)
    for i,j in enumerate(zenith_angle):
        ax.fill_between(np.linspace(theta[i], (0 if i==len(zenith_angle)-1 else theta[i+1]), 2), 0, j, color='deepskyblue', alpha=0.5)
        ax.fill_between(np.linspace(theta[i], (0 if i==len(zenith_angle)-1 else theta[i+1]), 2), j, 90, color='black', alpha=1)
    ax.scatter(theta, zenith_angle, color='blue', s=10, alpha=0.75)
    # plt.title('Scatter Plot on Polar Axis', fontsize=15)

    plt.show()
    plt.close()

    return fig

def plot_visible_skymap_from_horizon_file(path_horizon):
    """ Function returns a fisheye view of the sky with the visible portion in blue and the blocked one in black
        from a .csv horizon file
    
    Parameters
    ----------
    path_horizon : str
        Path to the .csv horizon file

    Returns
    -------
    fig : figure
        Plot of the sky view from the location
    """

    theta, zenith_angle = read_horizon_to_polar(path_horizon)
    fig = plot_visible_skymap(theta, zenith_angle)

    return fig
