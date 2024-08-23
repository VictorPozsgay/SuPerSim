# SuPerSim

**SuPerSim** (**Su**mmary for **Per**mafrost **Sim**ulations) is a package that allows for quick and easy visualization of permafrost metrics and time series from ensemble simulations. The user can extract information from ensemble simulations, create statistics, and plot a variety of spatially and temporally summarized data. 

This tool is particularly useful to perform a spatial and temporal analysis of ground, surface, snow, and air metrics on a slope
underlain by permafrost where a fast or slow landslide happened. It allows the user to plot time series relative to the time of the event 
and to visually inspect their behaviour weeks, months, or years before. **SuPerSim** also creates heatmaps of the ground surface temperature 
with respect to topographic parameters (slope, aspect, and altitude) to understand the relative importance in the rockfall starting zone.

There is a number of plots than can be produced for a single site. However, **SuPerSim** is also able to compare two different sites
(or two different faces of a same mountain for instance, see the example with a comparison of north- and south-facing rock slopes).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install **SuPerSim** (find the PyPi page here: https://pypi.org/project/SuPerSim/).

```bash
pip install SuPerSim
```

Install all the required packages (dependencies) from the *requirements.txt*  file.


```bash
pip install -r requirements.txt
```

Place *requirements.txt* in the directory where you plan to run the command. If the file is in a different directory, specify its path, for example, *path/to/requirements.txt*.

## Usage

The package is better used in a *Python* script and is imported with

```python
import SuPerSim
```

## Examples

The user can find some inspiration on how to use **SuPerSim** by looking at the examples provided.

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
