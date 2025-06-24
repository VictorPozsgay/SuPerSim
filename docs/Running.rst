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
      path_ground               = '<path>'
      path_snow                 = '<path>'
      path_swe                  = '<path>'
      path_thaw_depth           = '<path>'
      path_repository           = '<path>'
      path_pickle               = '<path>'

      path_horizon              = path_data+'/horizon.csv' # optional and set to 'None' by default

      ###############################################################
      # Enter the parameters of your site and of the rockfall event #
      ###############################################################

      site              = '<name_site:str>'

      # CREATE THE PICKLES
      year_bkg_end      = <year:int>
      year_trans_end    = <year:int>
      forcing_list      = ['<reanalysis_1:str>', ..., '<reanalysis_n:str>'] # list of forcing names
      path_forcing_list = ['<path_forcing_reanalysis_1:str>', ..., '<path_forcing_reanalysis_n:str>']
      consecutive       = <number_days:int>
      date_event        = [<year:int>, <month:int>, <day:int>]              # empty list '[]' if none available
      topo_event        = [<altitude:int>, <aspect:int>, <slope:int>]       # empty list '[]' if none available
      glacier           = <bool> # optional and set to 'False' by default
      min_glacier_depth = <min:int> # optional and set to '100' by default
      max_glacier_depth = <max:int> # optional and set to '20000' by default

      # CREATE THE PLOTS
      no_weight               = <bool> # optional

      print_plots             = <bool> # optional
      split_legend            = <bool> # optional
      save_plots_pdf          = <bool> # optional

      custom_years            = [<year_1:int>, ..., <year_n:int>] # optional and set to 'None' by default
      query                   = {'<var_1:str>': <value_1:int>, ..., '<var_n:str>': <value_n:int>} # optional and set to 'None' by default

      show_hor                = <bool> # optional
      show_hist               = <bool> # optional
      show_glaciers           = <bool> # optional
      show_yearly_box         = <bool> # optional
      show_yearly_stats_atmos = <bool> # optional
      show_yearly_stats_sims  = <bool> # optional
      show_thaw_depth         = <bool> # optional
      show_2metrics_seasonal  = <bool> # optional
      show_1metric_seasonal   = <bool> # optional
      show_decades            = <bool> # optional
      show_excep_years        = <bool> # optional
      show_normdev            = <bool> # optional
      show_landslide_time     = <bool> # optional
      show_individual_heatmap = <bool> # optional
      show_heatmaps           = <bool> # optional
      show_quantiles          = <bool> # optionalue
      show_meltout            = <bool> # optional
      show_GST_evol           = <bool> # optional
      show_parity             = <bool> # optional


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

      dict_final = plot_all(site, path_forcing_list, path_ground, path_snow, path_swe, path_thaw_depth, path_pickle,
         year_bkg_end, year_trans_end, path_horizon=path_horizon, no_weight=no_weight,
         print_plots=print_plots, split_legend=split_legend, save_plots_pdf=save_plots_pdf,
         custom_years=custom_years, query=query,
         show_hor=show_hor, show_hist=show_hist, show_glaciers=show_glaciers, show_yearly_box=show_yearly_box,
         show_yearly_stats_atmos=show_yearly_stats_atmos, show_yearly_stats_sims=show_yearly_stats_sims,
         show_thaw_depth=show_thaw_depth, show_2metrics_seasonal=show_2metrics_seasonal,
         show_1metric_seasonal=show_1metric_seasonal, show_decades=show_decades,
         show_excep_years=show_excep_years, show_normdev=show_normdev, show_landslide_time=show_landslide_time,
         show_individual_heatmap=show_individual_heatmap, show_heatmaps=show_heatmaps, show_quantiles=show_quantiles,
         show_meltout=show_meltout, show_GST_evol=show_GST_evol, show_parity=show_parity)

Outputs
^^^^^^^

The script shows all plots on the Python notebook but it also pickles them to ba able to re-use them later.
The script returns the dictionary 'dict_final' with all figure objects, their legends, and captions ::

      dict_final = {
            'name_fig1': {
                  'fig_legend_on': 'fig1_legend_on' # first figure object, with its legend,
                  'fig_legend_off': 'fig1_legend_off' # first figure object, without its legend !!! ONLY if\  *split_legend*\=True !!!,
                  'fig_legend_only': 'fig1_legend_only' # first figure legend object only !!! ONLY if\  *split_legend*\=True !!!,
                  'caption': 'fig1_caption' # generic and standardized string description of the figure
            },
            'name_fig2': {...},
            ...,
            'name_fign': {...},
      }

On top of this, the full list of captions is saved to a human readable text under path_pickle/'{site}_captions_readable.txt'.

Finally, the parameter\  *save_plots_pdf*\ controls whether or not to save all produced plots (3 times as much if\  *split_legend*\=True)
to a PDF format.


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
             list_path_forcing_list, list_path_ground, list_path_snow, list_path_swe,
             list_path_SW_direct, list_path_SW_diffuse, list_path_SW_up,
             list_path_SW_down, list_path_SW_net, list_path_LW_net,
             list_path_pickle, year_bkg_end, year_trans_end)
