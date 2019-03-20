import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_corner(samples, savefn=None):
    columns = list()
    for i in range(1, int(len(samples[0])/4)+1):
        columns.append([r"$x_{}$".format(i), r"$y_{}$".format(i),
                        r"$flux_{}$".format(i), r"$bmaj_{}$".format(i)])
    columns = [item for sublist in columns for item in sublist]
    df = pd.DataFrame(samples, columns=columns)
    q = sns.PairGrid(df)
    q.map_diag(sns.distplot, bins=100)
    q.map_upper(plt.scatter, s=0.5)
    q.map_lower(sns.kdeplot, linewidths=0.25)
    if savefn is not None:
        q.fig.savefig(savefn, dpi=300, bbox_inches="tight")
    return q.fig
