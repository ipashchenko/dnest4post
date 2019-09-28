import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import corner


def _plot_corner(samples, savefn=None):
    columns = list()
    for i in range(1, int(len(samples[0])/4)+1):
        columns.append([r"$x_{}$".format(i), r"$y_{}$".format(i),
                        r"$flux_{}$".format(i), r"$bmaj_{}$".format(i)])
    columns = [item for sublist in columns for item in sublist]
    df = pd.DataFrame(samples, columns=columns)
    q = sns.PairGrid(df)
    q.map_diag(sns.distplot, bins=50, norm_hist=False)
    # q.map_upper(plt.scatter, s=1.0)
    q.map_lower(sns.kdeplot, linewidths=0.5)
    if savefn is not None:
        q.fig.savefig(savefn, dpi=100, bbox_inches="tight")
    return q.fig


def plot_corner(samples, savefn=None, truths=None):
    columns = list()
    for i in range(1, int(len(samples[0])/4)+1):
        columns.append([r"$x_{}$".format(i), r"$y_{}$".format(i),
                        r"$flux_{}$".format(i), r"$bmaj_{}$".format(i)])
    columns = [item for sublist in columns for item in sublist]
    fig = corner.corner(samples, labels=columns, truths=truths,
                        show_titles=True, quantiles=[0.16, 0.5, 0.84],
                        color="gray", truth_color="#1f77b4",
                        plot_contours=True, range=[0.975]*len(columns),
                        plot_datapoints=False, fill_contours=True,
                        levels=(0.393, 0.865, 0.989),
                        hist2d_kwargs={"plot_datapoints": False,
                                       "plot_density": False,
                                       "plot_contours": True,
                                       "no_fill_contours": True},
                        hist_kwargs={'ls': 'solid',
                                     'density': True})
    if savefn is not None:
        fig.savefig(savefn, dpi=100, bbox_inches="tight")
    return fig


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
    axes.set_aspect("equal")

    if savefn is not None:
        fig.savefig(savefn, dpi=300, bbox_inches="tight")
    return fig


def rj_plot_ncomponents_distribution(posterior_file="posterior_sample.txt",
                                     picture_fn=None):
    samples = np.loadtxt(posterior_file)
    values, counts = np.unique(samples[:, 7], return_counts=True)
    plt.vlines(values, 0, counts, color='C0', lw=8)
    plt.ylim(0, max(counts) * 1.06)
    plt.xticks(np.arange(min(values), max(values)+1))
    plt.xlabel("# components")
    plt.ylabel("N")
    if picture_fn is not None:
        plt.savefig(picture_fn, bbox_inches="tight")
    plt.show()