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
    q.map_diag(sns.distplot, bins=50, norm_hist=True)
    q.map_upper(plt.scatter, s=1.0)
    q.map_lower(sns.kdeplot, linewidths=0.5)
    if savefn is not None:
        q.fig.savefig(savefn, dpi=300, bbox_inches="tight")
    return q.fig


def plot_position_posterior(samples, savefn=None, ra_lim=(-10, 10),
                            dec_lim=(-10, 10)):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, 1)
    xs = dict()
    ys = dict()
    fluxes = dict()
    n_comps = int(len(samples[0])/4)

    for i_comp in range(n_comps):
        xs[i_comp] = samples[:, 0+i_comp*4]
        ys[i_comp] = samples[:, 1+i_comp*4]
        fluxes[i_comp] = samples[:, 2+i_comp*4]

    for i_comp, color in zip(range(n_comps), colors):
        axes.scatter(xs[i_comp], ys[i_comp], s=0.5, color=color)

    axes.set_xlim(ra_lim)
    axes.set_ylim(dec_lim)
    axes.set_xlabel("RA, mas")
    axes.set_ylabel("DEC, mas")
    axes.invert_xaxis()

    if savefn is not None:
        fig.savefig(savefn, dpi=300, bbox_inches="tight")
    return fig
