"""This module takes care of the figure legends"""

def get_all_legends_fig(fig):
    """ Function takes a figure object and extracts all legends
    
    Parameters
    ----------
    fig : Figure

    Returns
    -------
    handles : list
        All legend handles
    labels : list
        All legend labels
    """

    handles = []
    labels = []
    for ax in fig.axes:
        handles += get_all_legends_ax(ax)[0]
        labels += get_all_legends_ax(ax)[1]

    return handles, labels

def get_all_legends_ax(ax):
    """ Function takes an axis object and extracts all legends
    
    Parameters
    ----------
    ax : Axis

    Returns
    -------
    handles : list
        All legend handles
    labels : list
        All legend labels
    """

    handles, labels = ax.get_legend_handles_labels()

    return handles, labels

