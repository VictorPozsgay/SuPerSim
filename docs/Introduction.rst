Introduction
============

Purpose
-------
The\  **SuPerSim** \(which stands for\  **Su**\mmary for\  **Per**\mafrost\  **Sim**\ulations) allows a quick and easy visualization of permafrost metrics and timeseries from ensemble simulations.
The user can extract information from ensemble simulations, create statistics, and plot a variety of spatially and temporally summarized data.

Pre-requisites
--------------

The package needs some specific inputs and those are obtained by using the following packages:

#. **GlobSim** \(https://github.com/geocryology/globsim): downloads and scales reanalysis data to produce meteorological time series at any point location.

#. **GEOtop** \(https://github.com/geotopmodel/geotop): Land surface model of the mass and energy balance of the hydrological cycle which simulates ground thermal properties.

#. **GTPEM** \(https://gitlab.com/permafrostnet/gtpem): Submits ensemble simulations to supercomputer and aggregates the results.

Input data
----------

In order to run **SuPerSim**, the user needs to have the following data files:

#. One or multiple netCDF (.nc) files acting as meteorological forcing data. Those are **GlobSim** products.

#. A .csv file that lists metadata for all simulations, this is a **GTPEM** output (of the\  *build*\  command)

#. A list of ensemble simulation products from **GEOtop**:

   #. A ground file where the soil temperature is stored. The file is obtained by adding the following line to the **GTPEM** .toml configuration file::

         soil_temperature = true

   #. A snow file where the snow depth is stored. The file is obtained by adding the following line to the **GTPEM** .toml configuration file::

         snow_depth = true

   #. A file where the snow water equivalent is stored. The file is obtained by adding the following line to the **GTPEM** .toml configuration file::

         swe = true
   
   #. A file where the depth of thaw is stored. The file is obtained by adding the following line to the **GTPEM** .toml configuration file::

         thaw_depth = true

      .. note::
            
            Additionally, for further analysis, we might also want extra aggregated files:

   #. A file where the SW (shortwave) direct is stored. The file is obtained by adding the following line to the **GTPEM** .toml configuration file::

         SW_direct = true

   #. A file where the SW (shortwave) diffuse is stored. The file is obtained by adding the following line to the **GTPEM** .toml configuration file::

         SW_diffuse = true

   #. A file where the SW (shortwave) up is stored. The file is obtained by adding the following line to the **GTPEM** .toml configuration file::

         SW_up = true
   
   #. A file where the SW (shortwave) down is stored. The file is obtained by adding the following line to the **GTPEM** .toml configuration file::

         SW_down = true

   #. A file where the SW (shortwave) net is stored. The file is obtained by adding the following line to the **GTPEM** .toml configuration file::

         SW_net = true

   #. A file where the LW (longwave) net is stored. The file is obtained by adding the following line to the **GTPEM** .toml configuration file::

         LW_net = true
