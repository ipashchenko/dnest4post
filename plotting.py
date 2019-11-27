import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from spydiff import import_difmap_model
# import seaborn as sns
import corner
from postprocess_labels import sort_samples_by_r
from postprocess_labels_rj import get_samples_for_each_n


def process_rj_samples(post_file, n_comp, n_max, jitter_first=True, savefn=None,
                         ra_lim=(-10, 10), dec_lim=(-10, 10),
                         difmap_model_fn=None):
    data = np.loadtxt(post_file)
    data = get_samples_for_each_n(data, jitter_first=jitter_first, n_max=n_max)[n_comp]
    data = sort_samples_by_r(data)
    fig = plot_position_posterior(data, savefn, ra_lim, dec_lim, difmap_model_fn)
    return fig


def process_norj_samples(post_file, jitter_first=True, savefn=None,
                         ra_lim=(-10, 10), dec_lim=(-10, 10),
                         difmap_model_fn=None):
    data = np.loadtxt(post_file)
    if jitter_first:
        data = data[:, 1:]
    data = sort_samples_by_r(data)
    fig = plot_position_posterior(data, savefn, ra_lim, dec_lim, difmap_model_fn)
    return fig


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
                            dec_lim=(-10, 10), difmap_model_fn=None):
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
        axes.scatter(xs[i_comp], ys[i_comp], s=3.0, color=color)

    if difmap_model_fn is not None:
        comps = import_difmap_model(difmap_model_fn)
        print("Read {} components from {}".format(len(comps), difmap_model_fn))
        for comp in comps:
            if comp.size == 3:
                axes.scatter(comp.p[2], comp.p[1], s=40, color="black", alpha=0.5, marker="x")
            elif comp.size == 4:
                e = Circle((comp.p[2], comp.p[1]), comp.p[3],
                           edgecolor="black", facecolor="red",
                           alpha=0.125)
                axes.add_patch(e)
            else:
                raise Exception("Not implemented for elliptical component!")

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
                                     picture_fn=None, jitter_first=True):
    samples = np.loadtxt(posterior_file)
    if jitter_first:
        ncomp_index = 7
    else:
        ncomp_index = 6
    values, counts = np.unique(samples[:, ncomp_index], return_counts=True)
    plt.vlines(values, 0, counts, color='C0', lw=8)
    plt.ylim(0, max(counts) * 1.06)
    plt.xticks(np.arange(min(values), max(values)+1))
    plt.xlabel("# components")
    plt.ylabel("N")
    if picture_fn is not None:
        plt.savefig(picture_fn, bbox_inches="tight")
    plt.show()