import matplotlib.pylab as plt


def plot_optimization(x_variable, y_variable, optimizer, tpe_results, color, filename=None, baseline=None, label=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))

    if label is None:
        label = x_variable
    # --- plot all evaluations ---
    ax.plot(tpe_results[x_variable], tpe_results[optimizer.__name__], '.-', c=color, label=label)

    # --- plot baseline ---
    if baseline is not None:
        if "value" in baseline.keys() and "color" in baseline.keys() and "label" in baseline.keys():
            ax.hlines(baseline["value"], colors=baseline["color"],
                      xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
                      linestyle="dashdot", label=baseline["label"])

    # --- mark for the maximum ---
    max_ix = tpe_results[optimizer.__name__].idxmax()
    ax.plot(tpe_results.loc[max_ix, x_variable], tpe_results.loc[max_ix, optimizer.__name__], "o", c="black")
    ax.text(tpe_results.loc[max_ix, x_variable],
            tpe_results.loc[max_ix, optimizer.__name__] + 0.5 * (
                        ax.get_ylim()[1] - tpe_results.loc[max_ix, optimizer.__name__]),
            "max: {:.4f}".format(tpe_results.loc[max_ix, optimizer.__name__]),
            horizontalalignment="center")
    ax.vlines(tpe_results.loc[max_ix, x_variable],
              colors="black",
              ymin=ax.get_ylim()[0],
              ymax=tpe_results.loc[max_ix, optimizer.__name__],
              linestyle="dashdot", label="max {}: {:.4f}".format(x_variable, tpe_results.loc[max_ix, x_variable]))

    # --- xy labels and legend ---
    ax.set_title("Optimization for "+x_variable)
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)
    ax.legend()
    if filename is not None:
        plt.savefig(filename+".svg")
        plt.close()
