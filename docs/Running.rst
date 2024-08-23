Running
=======

Useful functions
----------------

All reusable functions are found in\  *src/SuPerSim/__init__.py*.

All plotting functions combine a data formatting function and an input-independent plotting function.
For example, the function that plots the horizon line (or visible part of the sky from a point location) can be found under\  *src/SuPerSim/horizon.py*\  and is called\  *plot_visible_skymap_from_horizon_file()*\. It takes \  *path_horizon*\  for argument. However, this path points to a csv file with a specific format and the user might have a different format at hand. Thus, there is a first function\  *read_horizon_to_polar()*\  that formats the csv file into 2 lists of angles (numpy arrays) called\  *theta*\  and\  *zenith_angle*. From there, the plotting function\  *plot_visible_skymap()*\  plots the horizon in polar coordinates from the pre-fromatted angles. Schematically,

*path_horizon* -> *read_horizon_to_polar(path_horizon)* -> *(theta, zenith_angle)* -> *plot_visible_skymap(theta, zenith_angle)* -> *fig*

is equivalent to

*path_horizon* -> *plot_visible_skymap_from_horizon_file(path_horizon)* -> *fig*

but provides a chance for the user to jump in the middle of the process if they do not have the csv file where *path_horizon* points to, but instead already have the lists of angles *(theta, zenith_angle)*.


Single site
-----------

Setup
^^^^^

For a single site, we recommend the user to start any script with the following list of parameter definitions
(such as can be found in the\  *examples/*\  folder under\  *produce_plots_North.ipynb*\  and\  *produce_plots_South.ipynb*)::


      ################################################
      # Here, write the paths to your own data files #
      ################################################

      path_forcing_reanalysis_1 = '<path>' # placeholder path
      ...
      path_forcing_reanalysis_n = '<path>'
      path_ground = '<path>'
      path_snow = '<path>'
      path_swe = '<path>'
      path_thaw_depth = '<path>'
      path_repository = '<path>'
      path_pickle = '<path>'

      path_horizon = path_data+'/horizon.csv' # optional and set to 'None' by default

      forcing_list = ['reanalysis_1', ..., 'reanalysis_n'] # list of forcing names
      path_forcing_list = [path_forcing_reanalysis_1, ..., path_forcing_reanalysis_n]

      ###############################################################
      # Enter the parameters of your site and of the rockfall event #
      ###############################################################

      site = '<name_site>'
      year_bkg_end = <year>
      year_trans_end = <year>
      date_event = [<year>, <month>, <day>]
      topo_event = [<altitude>, <aspect>, <slope>]

      # Glaciers
      consecutive = <number>
      glacier = <bool> # optional and set to 'False' by default
      min_glacier_depth = <min> # optional and set to '100' by default
      max_glacier_depth = <max> # optional and set to '20000' by default

      # Plotting options
      no_weight = <bool> # optional and set to 'True' by default
      individual_heatmap = <bool> # optional and set to 'False' by default
      polar_plots = <bool> # optional and set to 'False' by default
      parity_plot = <bool> # optional and set to 'False' by default


Glaciers
^^^^^^^^

**SuPerSim** has the ability to exclude all simulations that build glaciers. For this, the user should select ::

      glacier = False

The filtering will then get rid of all simulation products where the minimum summer snow depth
is above the threshold given by\ *min_glacier_depth*\. These simulations do not melt out in the summer and are discarded.
However, the user can also decide to **only** keep glaciers by setting ::

      glacier = True

In which case the simulations that are kept are the ones with a minimal snow depth comprised
between\  *min_glacier_depth*\  and\  *max_glacier_depth*\.


Finally, the\  *consecutive*\  parameter is only used to determine the melt out date and snow cover statistics. 
The first full melt out date is then given by the first day of the year with a zero snow depth maintained for 
at least\  *consecutive*\  days.


Running the script
^^^^^^^^^^^^^^^^^^

The first thing to do is to build the statistics for all the metrics by calling the function\  *get_all_stats*\  ::

      pkl = get_all_stats(forcing_list, path_forcing_list, path_repository, path_ground, path_snow, path_pickle,
                              year_bkg_end, year_trans_end, consecutive,
                              site, date_event, topo_event,
                              glacier, min_glacier_depth, max_glacier_depth)

      df = pkl['df']
      reanalysis_stats = pkl['reanalysis_stats']
      list_valid_sim = pkl['list_valid_sim']
      dict_melt_out = pkl['dict_melt_out']
      stats_melt_out_dic = pkl['stats_melt_out_dic']
      df_stats = pkl['df_stats']
      rockfall_values = pkl['rockfall_values']

This creates a number of pickles that are saved in the directory given by\  *path_pickle*\.
Once the pickles are created, every time the function is called again,
it will first look for them in the directory and if they exist, it will simply retrieve them.
It will only recompute them if they don't exist. If a mistake was made and the user needs to recompute the variables,
first erase the content of the pickle directory, and then run the function again.
Once the pickles are created, there is an easier way to open them than running the function\  *get_all_stats*\  again,
indeed, we have the function\  *load_all_pickles*\  ::


      pkl = load_all_pickles(site, path_pickle)

      df = pkl['df']
      reanalysis_stats = pkl['reanalysis_stats']
      list_valid_sim = pkl['list_valid_sim']
      dict_melt_out = pkl['dict_melt_out']
      stats_melt_out_dic = pkl['stats_melt_out_dic']
      df_stats = pkl['df_stats']
      rockfall_values = pkl['rockfall_values']

Finally, the plotting function\  *plot_all*\  can be called ::

      plot_all(site, path_forcing_list, path_ground, path_snow, path_swe, path_thaw_depth, path_pickle,
         year_bkg_end, year_trans_end, path_horizon, no_weight, show_glaciers,
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

      list_path_forcing_list = [['<path_forcing_site_1_reanalysis_1>', ..., '<path_forcing_site_1_reanalysis_n>'], ['<path_forcing_site_2_reanalysis_1>', ..., '<path_forcing_site_2_reanalysis_m>']]
      list_path_ground = ['<path_ground_site_1>', '<path_ground_site_2>']
      list_path_snow = ['<..._site_1>', '<..._site_2>']
      list_path_swe = ['<..._site_1>', '<..._site_2>']
      list_path_SW_direct = ['<..._site_1>', '<..._site_2>']
      list_path_SW_diffuse = ['<..._site_1>', '<..._site_2>']
      list_path_SW_up = ['<..._site_1>', '<..._site_2>']
      list_path_SW_down = ['<..._site_1>', '<..._site_2>']
      list_path_SW_net = ['<..._site_1>', '<..._site_2>']
      list_path_LW_net = ['<..._site_1>', '<..._site_2>']
      list_path_pickle = ['<..._site_1>', '<..._site_2>']

      ###############################################################
      # Enter the parameters of your site and of the rockfall event #
      ###############################################################

      list_site = ['<name_site_1>', '<name_site_2>']
      list_label_site = ['<label_site_1>', '<label_site_2>']
      year_bkg_end = <year>
      year_trans_end = <year>

Running the script
^^^^^^^^^^^^^^^^^^

The comparison uses the result of the function\  *get_all_stats*\  applied to both sites. This function saves pickles for both sites.
Let us now use the comparison function\  *plot_camparison_two_sites*\  that retrieves all information about the sites from the pickles
and produces a series of plots comparing timeseries and metrics on each site.
The comparison plotting function\  *plot_camparison_two_sites*\  is called in the following way ::

      plot_camparison_two_sites(list_site, list_label_site,
<<<<<<< HEAD
             list_path_forcing_list, list_path_ground, list_path_snow, list_path_swe,
             list_path_SW_direct, list_path_SW_diffuse, list_path_SW_up,
             list_path_SW_down, list_path_SW_net, list_path_LW_net,
             list_path_pickle, year_bkg_end, year_trans_end)
=======
                  list_path_forcing_list, list_path_ground, list_path_snow, list_path_swe,
                  list_path_SW_direct, list_path_SW_diffuse, list_path_SW_up,
                  list_path_SW_down, list_path_SW_net, list_path_LW_net,
                  list_path_pickle, year_bkg_end, year_trans_end)
>>>>>>> 694e8a5ffae4cb4b07e7e2563d6faf7964c89f3d
