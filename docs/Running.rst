Running
=======

Single site
-----------

For a single site, we recommend the user to start any script with the following list of parameter definitions
(such as can be found in the examples/ folder)::

      # Paths to the data
      path_forcing_ranalysis_1 = '<path>'
      ...
      path_forcing_ranalysis_n = '<path>'
      path_ground = '<path>'
      path_snow = '<path>'
      path_swe = '<path>'
      path_thaw_depth = '<path>'
      path_repository = '<path>'
      path_pickle = '<path>'

      forcing_list = ['reanalysis_1', ..., 'reanalysis_n']
      path_forcing_list = [path_forcing_ranalysis_1, ..., path_forcing_ranalysis_n]

      # Site and event characteristics
      site = '<name_site>'
      year_bkg_end = <year>
      year_trans_end = <year>
      date_event = [<year>, <month>, <day>]
      topo_event = [<altitude>, <aspect>, <slope>]

      # Glaciers
      consecutive = <number>
      glacier = <bool>
      min_glacier_depth = <min>
      max_glacier_depth = <max>

      # Plotting options
      no_weight = <bool>
      individual_heatmap = <bool>
      polar_plots = <bool>
      parity_plot = <bool>


Glaciers
^^^^^^^^

**SuPerSim** has the ability to exclude all simulations that build glaciers. For this, the user should select ::

      glacier = True

The filtering will then get rid of all simulation products where the minimum summer snow depth
is above the threshold given by\ *min_glacier_depth*\. These simulations do not melt out in the summer and are discarded.
However, the user can also decide to **only** keep glaciers by setting ::

      glacier = False

In which case the simulations that are kept are the ones with a minimal snow depth comprised
between\  *min_glacier_depth*\  and\  *max_glacier_depth*\.


Finally, the\  *consecutive*\  parameter is only used to determine the melt out date and snow cover statistics. 
The first full melt out date is then given by the first day of the year with a zero snow depth maintained for 
at least\  *consecutive*\  days.


Running the script
^^^^^^^^^^^^^^^^^^

The first thing to do is to build the statistics for all the metrics by calling the function\  *get_all_stats*\  ::

      df, reanalysis_stats, list_valid_sim, dict_melt_out, stats_melt_out_dic, df_stats, rockfall_values = get_all_stats(
            forcing_list, path_forcing_list, path_repository, path_ground, path_snow, path_pickle,
            year_bkg_end, year_trans_end, consecutive,
            site, date_event, topo_event,
            glacier, min_glacier_depth)

This creates a number of pickles that are saved in the directory given by\  *path_pickle*\.
Once the pickles are created, every time the function is called again,
it will first look for them in the directory and if they exist, it will simply retrieve them.
It will only recompute them if they don't exist. If a mistake was made and the user needs to recompute the variables,
first erase the content of the pickle directory, and then run the function again.
Once the pickles are created, there is an easier way to open them than running the function\  *get_all_stats*\  again,
indeed, we have the function\  *load_all_pickles*\  ::

      df, reanalysis_stats, list_valid_sim, dict_melt_out, stats_melt_out_dic, df_stats, rockfall_values = load_all_pickles(site, path_pickle)

Finally, the plotting function\  *plot_all*\  can be called ::

      plot_all(site, forcing_list, path_forcing_list,
         path_ground, path_snow, path_swe, path_thaw_depth, path_pickle,
         year_bkg_end, year_trans_end, no_weight,
         individual_heatmap, polar_plots, parity_plot)


Comparison
----------

In order to compare two sites, the pickles need to be already computed, and hence the first part of the script needs to have 
been run for both sites. Once all the pickles are saved in their folder, one can compare timeseries.