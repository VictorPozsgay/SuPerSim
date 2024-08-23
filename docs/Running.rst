Running
=======

All reusable functions are found in [\_\_init\_\_.py](Introductions.rst)

Single site
-----------

Setup
^^^^^

For a single site, we recommend the user to start any script with the following list of parameter definitions
(such as can be found in the\  *examples/*\  folder under\  *produce_plots_North.ipynb*\  and\  *produce_plots_South.ipynb*)::

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

Setup
^^^^^

In order to compare two sites, the pickles need to be already computed, and hence the first part of the script needs to have 
been run for both sites. Once all the pickles are saved in their folder, one can compare timeseries. An example can be found in the the\  *examples/*\  folder
under\  *comparison.ipynb*\.

The user should start with a definition of all parameters, for instance ::

      ################################################
      # Here, write the paths to your own data files #
      ################################################

      # this is just introduced for convenience
      path_data = path_parent+'/examples/Aksaut_Caucasus/data/'
      path_data_North = path_data+'North/'
      path_data_South = path_data+'South/'
      path_forcing_merra2 = path_data+'/scaled_merra2_Aksaut.nc'

      # those are the real variables
      list_path_forcing_list = [[path_forcing_merra2], [path_forcing_merra2]]
      list_path_ground = [path_data_North+'result_soil_temperature.nc', path_data_South+'result_soil_temperature.nc']
      list_path_snow = [path_data_North+'result_snow_depth.nc', path_data_South+'result_snow_depth.nc']
      list_path_swe = [path_data_North+'result_swe.nc', path_data_South+'result_swe.nc']
      list_path_SW_direct = [path_data_North+'result_SW_direct.nc', path_data_South+'result_SW_direct.nc']
      list_path_SW_diffuse = [path_data_North+'result_SW_diffuse.nc', path_data_South+'result_SW_diffuse.nc']
      list_path_SW_up = [path_data_North+'result_SW_up.nc', path_data_South+'result_SW_up.nc']
      list_path_SW_down = [path_data_North+'result_SW_down.nc', path_data_South+'result_SW_down.nc']
      list_path_SW_net = [path_data_North+'result_SW_net.nc', path_data_South+'result_SW_net.nc']
      list_path_LW_net = [path_data_North+'result_LW_net.nc', path_data_South+'result_LW_net.nc']
      list_path_pickle = [path_parent+'/examples/Aksaut_Caucasus/python_pickles/', path_parent+'/examples/Aksaut_Caucasus/python_pickles/']

      ###############################################################
      # Enter the parameters of your site and of the rockfall event #
      ###############################################################

      list_site = ['Aksaut_North', 'Aksaut_South']
      list_label_site = ['North', 'South']
      year_bkg_end = 2000
      year_trans_end = 2023

Running the script
^^^^^^^^^^^^^^^^^^

The comparison uses the result of the function\  *get_all_stats*\  applied to both sites. This function saves pickles for both sites.
Let us now use the comparison function\  *plot_camparison_two_sites*\  that retrieves all information about the sites from the pickles
and produces a series of plots comparing timeseries and metrics on each site.
The comparison plotting function\  *plot_camparison_two_sites*\  is called in the following way ::

      plot_camparison_two_sites(list_site, list_label_site,
                  list_path_forcing_list, list_path_ground, list_path_snow, list_path_swe,
                  list_path_SW_direct, list_path_SW_diffuse, list_path_SW_up,
                  list_path_SW_down, list_path_SW_net, list_path_LW_net,
                  list_path_pickle, year_bkg_end, year_trans_end)
