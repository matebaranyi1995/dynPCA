import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt

from ipywidgets import IntRangeSlider, Dropdown


#####################################################################################
# the main function to efficiently plot a larger timeseries per dimension

def display_graphs(data, cols=None, sel_range=None):
    """
    Builds a rasterized stacked image of multiple timeseries on each other

    Parameters
    ----------
    data : pd.DataFrame
        dataframe to plot, each columns is considered as a timeseries
    cols : list, optional
        list of colnames to be plotted, by default the first col only
        length of the list is at most 10
    sel_range : tuple, optional
        tuple of indeces from/to the data will be plotted,
        by default it is the whole series

    Returns
    -------
    datashader Image 
        rasterized picture of the selected cols on the chosen range
    """
    if sel_range is None:
        sel_range = data.index[0], data.index[-1]

    if cols is None:
        cols = data.columns.to_list()[:1]

    cvs = ds.Canvas(x_range=sel_range, plot_height=500, plot_width=1000)

    sel_data = data.loc[sel_range[0]:sel_range[1], cols]
    sel_data.reset_index(inplace=True)

    time_ind = sel_data.columns.to_list()[0]
    sel_cols = sel_data.columns.to_list()[1:]

    traces = {}
    for c in sel_cols:
        traces[c] = cvs.line(sel_data, x=time_ind, y=c)

    colors = [
        "red", "black", "green", "purple", "pink",
        "yellow", "brown", "gray", "orange", "blue"
    ]

    imgs = [tf.shade(traces[c], cmap=[colors[color]], alpha=100)
            for color, c in enumerate(sel_cols)]

    for color, c in enumerate(sel_cols):
        print("Color {} represents column {}.".format(colors[color], c))

    stacked = tf.stack(*imgs)

    return stacked  # , imgs, traces


def data_organizer(fpaths=[], dataframes=[], nicknames=None):
    """
    It organizes multiple dataframes with the SAME colnames into one.
    It attempts to concatanate dataframes based on nicknames,
    so multiple dataframes can have common columnnames.
    It is mainly used for the display_graphs() function.

    Parameters
    ----------
    fpaths : list, optional
        list of dataframes based on filepaths in csv format, by default []
    dataframes : list, optional
        list of dataframes, by default []
    nicknames : list, optional
        list of nicknames attached to colnames per dataframe

    Returns
    -------
    dict_of_dfs, 
        dictionary of the imported dataframes, keys are the nicknames
    dict_of_dfs_toplot
        concatanated dataframe used for display_graphs()
    cols 
        common colnames
    x_range
        index range of dict_of_dfs_toplot
    """
    if nicknames is None:
        nicknames = [str(i) for i in range(len(fpaths)+len(dataframes))]

    dict_of_dfs = {}
    for fp, nam in zip(fpaths, nicknames[:len(fpaths)]):
        dict_of_dfs[nam] = pd.read_csv(fp)

    for df, nam in zip(dataframes, nicknames[-len(dataframes):]):
        dict_of_dfs[nam] = pd.DataFrame(df)

    # skaler = StandardScaler()
    # dict_of_dfs["small"] = pd.DataFrame(skaler.fit_transform(dict_of_dfs["small"].values),
    #                                 index=dict_of_dfs["small"].index, columns=dict_of_dfs["small"].columns)

    # dr = pd.date_range(start="2000-01-01", periods=dict_of_dfs[nicknames[0]].shape[0], freq="{}N".format(1/512*1e9))
    dr = range(dict_of_dfs[nicknames[0]].shape[0])
    timeframe = pd.DataFrame(dr, columns=["time"])
    timeframe.index = dr

    cols = dict_of_dfs[nicknames[0]].columns.to_list()

    for k in dict_of_dfs.keys():
        dict_of_dfs[k].fillna(method="bfill", inplace=True)
        dict_of_dfs[k].fillna(method="ffill", inplace=True)
        dict_of_dfs[k].index = dr
        dict_of_dfs[k].columns = ['{}_{}'.format(
            str(c), k) for c in dict_of_dfs[k].columns]

    y_ranges = {}
    for c in cols:
        c_min, c_max = 0, 0
        for k in dict_of_dfs.keys():
            c_min = min(c_min, dict_of_dfs[k]['{}_{}'.format(c, k)].min())
            c_max = max(c_max, dict_of_dfs[k]['{}_{}'.format(c, k)].max())
        y_ranges[c] = (c_min*1.1, c_max*1.1)

    sel_data = timeframe
    for k in dict_of_dfs.keys():
        sel_data = sel_data.join(dict_of_dfs[k])

    dict_of_dfs_toplot = sel_data.drop("time", axis=1)
    x_range = [dict_of_dfs_toplot.index[0], dict_of_dfs_toplot.index[-1]]

    return dict_of_dfs, dict_of_dfs_toplot, cols, x_range


def simple_dual_plot(maindf, otherdf, fromhere = 0, tohere = None, nicknames = ["main", "other"], *args, **kwargs):

    if tohere is None:
        tohere = min(maindf.shape[0], otherdf.shape[0])

    fig, axs = plt.subplots(maindf.shape[1], 1, *args, **kwargs)
    datarange = np.array(range(fromhere, tohere))
    for i, j in enumerate(maindf.columns.tolist()):
        axs[i].plot(datarange, maindf[j].iloc[fromhere:tohere],
                    alpha=.8, linewidth=.7, label='{}_{}'.format(nicknames[0], j))
        axs[i].plot(datarange, otherdf[j].iloc[fromhere:tohere],
                    alpha=.8, linewidth=.7, label='{}_{}'.format(nicknames[1], j))
        axs[i].legend(bbox_to_anchor=(1, 1), loc='upper left')
        # axs[i].hlines(0, 0, len(datarange), linewidth=.4, color="black")

    # plt.suptitle("Original(blue) and restored(orange) TS")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def widg_forplot(dict_of_dfs, cols, x_range):
    # dates = list(pd.date_range(start="2000-01-01", periods=dict_of_dfs_toplot.shape[0], freq="{}N".format(1/512*1e9)))
    # options = [(i.strftime('%H:%M:%S.%f'), i) for i in dates]
    # index = (0, len(dates)-1)
    # myslider = SelectionRangeSlider(
    #     options = options,
    #     index = index,
    #     description = 'Test',
    #     orientation = 'horizontal',
    #     layout={'width': '200px'},
    #     continuous_update=False
    #     )

    optionz = list(zip(cols, [['{}_{}'.format(str(c), k)
                               for k in dict_of_dfs.keys()] for c in cols]))

    mydrop = Dropdown(
        options=optionz,
        value=optionz[0][1],  # red, black, green
        description='Col:',
        disabled=False,
        continuous_update=False
    )

    myslider = IntRangeSlider(
        value=x_range,
        min=x_range[0],
        max=x_range[1],
        step=1,
        continuous_update=False,
        layout={'width': '1000px'}
    )

    return optionz, mydrop, myslider
