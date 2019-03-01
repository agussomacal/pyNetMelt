import matplotlib.pylab as plt


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
