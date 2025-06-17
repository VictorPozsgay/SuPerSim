"""This module keeps track of the captions for all plots."""

captions = {
    'fisheye_horizon': 'Fisheye view of the sky with the visible portion in blue and the blocked one in black. The angular and radial graduations represent the north-based azimuth and the altitude angle from the zenith. For a given azimuth, the horizon would lie at 90 degree from the zenith if the view were unobstructed, and closer to 0 degree if fully obstructed by a vertical wall. Note that an overhang wall could lead to negative altitude angles but we do not allow for this in this representation.',
    'hist_stat_weight' : 'Histogram of the distribution of the statistical weights of all simulations. Simulations yielding glaciers are represented in blue, the others are orange. A weight is assigned to each simulation according to how close its configuration is to the one of the rockfall starting zone.',
    'hist_distrib_glaciers_perma': 'Histogram of the distribution of simulations yielding glaciers, permafrost, or none of the above, with respect to altitude, aspect, slope, and forcing.',
    # Do not forget to change transient mean into transient trends and address the orange transient mean for all the following plots
    'AirTemp_yearly_stats_box': 'Boxplot of yearly statistics for air temperature for the full simulation period. The data is displayed as minimum, first quartile (Q1), median, third quartile (Q3), and maximum. The mean background value is indicated in blue.',
    'GST_yearly_stats_box': 'Boxplot of yearly statistics for ground surface temperature (GST) for the full simulation period. The data is displayed as minimum, first quartile (Q1), median, third quartile (Q3), and maximum. The mean background value is indicated in blue.',
    'Precip_yearly_stats_box': 'Boxplot of yearly statistics for precipitation for the full simulation period. The data is displayed as minimum, first quartile (Q1), median, third quartile (Q3), and maximum. The mean background value is indicated in blue.',
    'WaterProd_yearly_stats_box': 'Boxplot of yearly statistics for total water production (defined to be the combined effect of precipitation and snow melt) for the full simulation period. The data is displayed as minimum, first quartile (Q1), median, third quartile (Q3), and maximum. The mean background value is indicated in blue.',
    # Restart here: Make sure to understand what the dark and light blue areas correspond to!
    'AirTemp_yearly_quantiles': 'XXXXX'
}