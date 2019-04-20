import matplotlib.pylab as plt
import numpy as np


def plot_baseline(baseline, ax):
    # --- plot baseline ---
    if "value" in baseline.keys() and "color" in baseline.keys() and "label" in baseline.keys():
        ax.hlines(baseline["value"], colors=baseline["color"],
                  xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
                  linestyle="dashdot", label=baseline["label"])


def plot_optimization(x_lab, y_lab, y_column, tpe_results, color, filename=None, baseline=None, label=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))

    if label is None:
        label = x_lab
    # --- plot all evaluations ---
    ax.plot(tpe_results[x_lab], tpe_results[y_column], '.-', c=color, label=label)

    # --- plot baseline ---
    if baseline is not None:
        plot_baseline(baseline, ax)

    # --- mark for the maximum ---
    max_ix = tpe_results[y_column].idxmax()
    ax.plot(tpe_results.loc[max_ix, x_lab], tpe_results.loc[max_ix, y_column], "o", c="black")
    ax.text(tpe_results.loc[max_ix, x_lab],
            tpe_results.loc[max_ix, y_column] + 0.5 * (
                        ax.get_ylim()[1] - tpe_results.loc[max_ix, y_column]),
            "max: {:.4f}".format(tpe_results.loc[max_ix, y_column]),
            horizontalalignment="center")
    ax.vlines(tpe_results.loc[max_ix, x_lab],
              colors="black",
              ymin=ax.get_ylim()[0],
              ymax=tpe_results.loc[max_ix, y_column],
              linestyle="dashdot", label="max {}: {:.4f}".format(x_lab, tpe_results.loc[max_ix, x_lab]))

    # --- xy labels and legend ---
    ax.set_title("Optimization for "+x_lab)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.legend()
    if filename is not None:
        plt.savefig(filename+".svg")
        plt.close()


def plot_optimization(x_column, y_column, tpe_results, color, ax, label=None):
    # --- plot all evaluations ---
    x = tpe_results[x_column]
    y = tpe_results[y_column]
    sd = tpe_results[y_column.replace(" ", " std ", 1)]
    ax.plot(x, y, '.-', c="black", label=label)
    ax.fill_between(x, y1=y+sd, y2=y-sd, label="68%", color=color, alpha=0.5)
    ax.fill_between(x, y1=y+2*sd, y2=y-2*sd, label="95%", color=color, alpha=0.25)


def mark_maximum(tpe_results, x_column, y_column, ax):
    # --- mark for the maximum ---
    max_ix = tpe_results[y_column].idxmax()
    ax.plot(tpe_results.loc[max_ix, x_column], tpe_results.loc[max_ix, y_column], "o", c="black")
    ax.text(tpe_results.loc[max_ix, x_column],
            tpe_results.loc[max_ix, y_column] + 0.5 * (
                        ax.get_ylim()[1] - tpe_results.loc[max_ix, y_column]),
            "max: {:.4f}".format(tpe_results.loc[max_ix, y_column]),
            horizontalalignment="center")
    ax.vlines(tpe_results.loc[max_ix, x_column],
              colors="black",
              ymin=ax.get_ylim()[0],
              ymax=tpe_results.loc[max_ix, y_column],
              linestyle="dashdot", label="max {}: {:.4f}".format(x_column, tpe_results.loc[max_ix, x_column]))


# def good_bar_charts(names_toghether, groups_toghether, data_toghether, names_alone, groups_alone, data_alone,
#                     data_toghether_color, data_alone_color, ylab="AUC", tit="Comparing AUC", ylim=(0.7, 1),
#                     figsize=(15, 8)):
#     """
#
#     :param names_toghether: list: name of each element of the together group.
#     :param groups_toghether: list of lists: how many different grous are going to be draw toghether.
#     :param data_toghether: values
#     :param names_alone:
#     :param groups_alone:
#     :param data_alone:
#     :param data_toghether_color:
#     :param data_alone_color:
#     :param ylab:
#     :param tit:
#     :param ylim:
#     :param figsize:
#     :return:
#     """
#     # https://matplotlib.org/examples/api/barchart_demo.html
#     ntog = len(data_toghether[0])
#     n_in_tog = float(len(data_toghether))
#     N = ntog + len(data_alone)
#     ind = np.arange(N)  # the x locations for the groups
#
#     width = 0.8 / n_in_tog  # the width of the bars,0.8 para uqe quede un 0.2 de espacio entre barras
#
#     fig, ax = plt.subplots(figsize=figsize)
#     rects = []
#     for i, (dt, dt_color) in enumerate(zip(data_toghether, data_toghether_color)):
#         rects.append(ax.bar(ind[:ntog] + width * ((i + 1 / 2) - n_in_tog / 2), dt, width, color=dt_color, yerr=0))
#
#     for i, (dt, dt_color) in enumerate(zip(data_alone, data_alone_color)):
#         rects.append(ax.bar(ind[ntog + i], dt, width, color=dt_color, yerr=0))
#
#     # add some text for labels, title and axes ticks
#     ax.set_ylabel(ylab, weight="bold")
#     ax.set_title(tit, weight="bold")
#     ax.set_xticks(ind)
#     ax.set_xticklabels(names_toghether + names_alone, weight="bold")
#     ax.set_ylim(ylim)
#
#     ax.legend(list(map(lambda r: r[0], rects)), groups_toghether + groups_alone, loc=0)
#

def good_bar_charts(df_toghether, df_alone, data_toghether_color, data_alone_color, ax):
    """

    :param names_toghether: list: name of each element of the together group.
    :param groups_toghether: list of lists: how many different grous are going to be draw toghether.
    :param data_toghether: values
    :param names_alone:
    :param groups_alone:
    :param data_alone:
    :param data_toghether_color:
    :param data_alone_color:
    :param ylab:
    :param tit:
    :param ylim:
    :param figsize:
    :return:
    """
    # https://matplotlib.org/examples/api/barchart_demo.html
    ntog = df_toghether.shape[0]
    n_in_tog = df_toghether.shape[1]
    N = ntog + df_alone.shape[0]
    ind = np.arange(N)  # the x locations for the groups

    width = 0.8 / n_in_tog  # the width of the bars,0.8 para uqe quede un 0.2 de espacio entre barras

    rects = []
    for i, ((ix, dt), dt_color) in enumerate(zip(df_toghether.items(), data_toghether_color)):
        rects.append(ax.bar(ind[:ntog] + width * ((i + 1 / 2) - n_in_tog / 2), dt, width, color=dt_color, yerr=0))

    for i, ((ix, dt), dt_color) in enumerate(zip(df_alone.items(), data_alone_color)):
        rects.append(ax.bar(ind[ntog + i], dt, width, color=dt_color, yerr=0))

    # add some text for labels, title and axes ticks
    ax.set_xticks(ind)
    ax.set_xticklabels(df_toghether.columns.tolist() + df_alone.columns.tolist(), weight="bold")

    # ax.legend(list(map(lambda r: r[0], rects)), groups_toghether + groups_alone, loc=0)
