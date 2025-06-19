"""This module keeps track of the captions for all plots."""

#pylint: disable=line-too-long


######### VARIABLES #########
dict_vars_a = {
    'AirTemp': 'air temperature',
    'GST': 'ground surface temperature (GST)',
    'Precip': 'precipitation',
    'WaterProd': 'total water production (defined to be the combined effect of precipitation and snow melt)',
    'Snow': 'snow depth',
    'SWE': 'snow water equivalent (SWE)'
}

dict_vars_b = {f'{k}m_grdtemp': f'{k} meter ground temperature' for k in [1,5,10]}

# final total variable dictionary
dict_vars = {}
for d in [dict_vars_a, dict_vars_b]:
    dict_vars.update(d)



######### DICTS #########

capt_hor = {'fisheye_horizon': 'Fisheye view of the sky with the visible portion in blue and the blocked one in black. The angular and radial graduations represent the north-based azimuth and the altitude angle from the zenith. For a given azimuth, the horizon would lie at 90 degree from the zenith if the view were unobstructed, and closer to 0 degree if fully obstructed by a vertical wall. Note that an overhang wall could lead to negative altitude angles but we do not allow for this in this representation.'}

capt_hist = {
    'hist_stat_weight' : 'Histogram of the distribution of the statistical weights of all simulations. Simulations yielding glaciers are represented in blue, the others are orange. A weight is assigned to each simulation according to how close its configuration is to the one of the rockfall starting zone.',
    'hist_distrib_glaciers_perma': 'Histogram of the distribution of simulations yielding glaciers, permafrost, or none of the above, with respect to altitude, aspect, slope, and forcing.'
}

# Do not forget to change transient mean into transient trends and address the orange transient mean for all the following plots
capt_yearly_stats_box = {
    f'{k}_yearly_stats_box': f'Boxplot of yearly statistics for {v} for the full simulation period. The data is displayed as minimum, first quartile (Q1), median, third quartile (Q3), and maximum. The mean background value is indicated in blue.'
    for (k,v) in dict_vars.items()
}

# Same here, trends in the transient period (green)!
capt_yearly_quantiles = {
    f'{k}_yearly_quantiles': f'Mean annual {v}, and 1- and 2-sigma envelopes for the full simulation period. The gray horizontal dashed line indicates the separation between the background and transient periods. The mean background value is indicated in orange.'
    for (k,v) in dict_vars.items()
}
# Same here, trends in the transient period (green)!
capt_yearly_mean = {
    f'{k}_yearly_mean': f'Mean annual {v} for the full simulation period. The gray horizontal dashed line indicates the separation between the background and transient periods. The mean background value is indicated in orange.'
    for (k,v) in dict_vars.items()
}

capt_thaw = {'max_thaw_depth': 'Plot of the evolution of maximum annual thaw depth for groups of simulations binned by (altitude,slope) couples for the shwole simulation period. For each group, the solid line represents the mean thaw depth over teh simulation members, and the envelope is the standard deviation. The simulation column extends from the ground surface (horiwontal gray line) down to the maximal plotted depth. Simulations with a maximal thaw depth corresponding to the maximal plotted depth most likely do not have permafrost at all.'}

capt_yearly_comp = {
    'GST_v_snow': 'Plot of the mean daily values of GST (blue) and snow depth (orange) over all simulations and the full simulation period. The 1- and 2-sigma envelopes are also shown.'
}

capt_1year_bkg_v_transient = {
    f'{k}_1year_bkg_v_transient': f'Plot of the mean daily values of {v} over all simulations and the full background (blue) and transient (orange) periods. The 1- and 2-sigma envelopes are also shown.'
    for (k,v) in dict_vars.items()
}

capt_1year_bkg_v_transient_excep_years = {
    f'{k}_1year_bkg_v_transient_excep_years': f'Plot of the mean daily values of {v} over all simulations and the full background period in blue. The 1- and 2-sigma envelopes are also shown. On top of this, the 3 most exceptional years are plotted in order (orange, green, red).'
    for (k,v) in dict_vars.items()
}

capt_1year_bkg_v_transient_decades = {
    f'{k}_1year_bkg_v_transient_decades': f'Plot of the mean daily values of {v} over all simulations and the full background (blue) and transient (orange) periods. The 1- and 2-sigma envelopes are also shown. On top of this, the mean value for every transient decade is plotted.'
    for (k,v) in dict_vars.items()
}

capt_norm_dev = {
    'normdev_week_month': 'Plot of the normalized deviation (or standardized anomaly, defined to be the deviation away from the mean in units of standard deviation) for the total water production, air temperature, and ground temperatures (surface, 1m ,5m, and 10m). The time period corresponds to the year of the landslide and the precise date of the event is indicated by a dashed red vertical line. For each time of the year, the value is compared to all values at the same time of the year for all previous years, and in the week (left) or month (right) leading to it.',
    'normdev': 'Plot of the normalized deviation (or standardized anomaly, defined to be the deviation away from the mean in units of standard deviation) for the total water production, air temperature, and ground temperatures (surface, 1m ,5m, and 10m). The time period corresponds to the transient period and each data point is compared to all previous data points. If applicable, the exact time of the landslide is indicated by a dashed red vertical line.'
}

capt_heatmap = {
    'heatmap_centre_GST_bkg': 'Heatmap of the background mean GST (°C) as a function of aspect and slope at the median simulation height.',
    'heatmap_centre_GST_evol': 'Heatmap of the difference in mean GST (°C) between transient and background periods, as a function of aspect and slope at the median simulation height.',
    'heatmap_GST': 'Heatmap of the background mean GST (°C) as a function of aspect and slope at all altitudes on the first row. The second row is the difference in mean GST (°C) between transient and background periods. Note that if there are 3 rows, the first row is acutally the percentage of simulations for that given altitude, slope, and aspect that yield glaciers.',
    'heatmap_GST_polar_alts': 'Polar heatmap of the background mean GST (top row) and its change between transient and background periods (bottom row) as a function of aspect and slope, for each simulated elevation.',
    'heatmap_GST_polar': 'Polar heatmap of the background mean GST (top row) and its change between transient and background periods (bottom row) as a function of aspect and elevation, for each simulated slope.',
    'heatmap_perma_polar_alts': 'Polar plot of the permafrost and glacier spatial distribution as a function of aspect and slope, for each simulated elevation.',
    'heatmap_perma_polar': 'Polar plot of the permafrost and glacier spatial distribution as a function of aspect and elevation, for each simulated slope.'
}

capt_extra = {
    'CDF_GST_SO': 'Cumulative distribution function of the mean ground surface temperature (GST) (right) and the surface offset (SO) (left) over all simulations for the background (blue) and transient (orange) periods. For each case, the cumulative distribution function of the mean change between the background and transient periods is plotted in green. The median, 10th and 90th percentiles are also visually reported.',
    'heatmap_percentiles_GST_bkg_trans_evol': 'Heatmap of 10th, 25th, 50th, 75th, and 90th percentile in background and transient mean ground surface temperature (GST) over all simulations. The difference between them is reported in the third column and is a good indicator of the potential ground warming.',
    'hist_snow_cover': 'Histogram of the evolution of the snow cover and melt-out date over all simulations between the background and transient periods.',
    'GST_evol_v_bkg': 'Plot of the mean GST change between background and transient period as a function of the mean background GST. This plot gives information on wich of the warmest or coldest original slopes warmed the fastest. On top of this, the simulations are sorted by mean background GST and binned for every 10th quantile, resulting in 10 equal size bins. The large dots represent the average values and deviation for all groups. The linear regression is calculated for all initial dots corresponding to all simulations.',
    'GST_evol_v_bkg_alts': 'Plot of the mean GST change between background and transient period as a function of the mean background GST. This plot gives information on wich of the warmest or coldest original slopes warmed the fastest. The simulations are classified by elevation and a linear regression is calculated for all data points at that specific elevation.',
    'parity_plot_stat_model_bkg_mean_GST': 'Parity plot (statistically-modeled vs numerically-simulated) of background mean GST. The model is given by the function: (offset + c_alt * altitude + c_asp * (altitude - d_alt) * np.cos(aspect * 2 * np.pi / 360) + c_slope * slope). The model can be applied either to the full ensemble simulation, or to all simulations forced with the same reanalysis product.'
}

list_all_capt_dicts = [capt_hor, capt_hist, capt_yearly_stats_box, capt_yearly_quantiles, capt_yearly_mean,
                       capt_thaw, capt_yearly_comp, capt_1year_bkg_v_transient, capt_1year_bkg_v_transient_excep_years,
                       capt_1year_bkg_v_transient_decades, capt_norm_dev, capt_heatmap, capt_extra]


captions = {}
for d in list_all_capt_dicts:
    captions.update(d)
