import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from postprocess_labels_gains import (sort_samples_by_r)
from plotting import plot_position_posterior
from plotting import plot_corner as plot_corner_norj


label_size = 16
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_samples_for_each_n(samples, jitter_first=True, n_max=30):
    j = 0
    if jitter_first:
        j += 1
    # dim, max num components, 4 hyperparameters + num components
    j += 7
    n_components = samples[:, j-1]
    out_samples = dict()
    for n in np.array(np.unique(n_components), dtype=int):
        samples_with_n_components = list()
        for sample in samples[n_components == n]:
            one_post_point = list()
            if jitter_first:
                one_post_point.append(sample[0])
            for k in range(n):
                one_post_point.extend([sample[j+k+i*n_max] for i in range(4)])
            samples_with_n_components.append(one_post_point)
        out_samples.update({n: np.atleast_2d(samples_with_n_components)})
    return out_samples


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


def process_rj_samples(post_file, n_comp, n_max, jitter_first=True, savefn=None,
                       ra_lim=(-10, 10), dec_lim=(-10, 10),
                       difmap_model_fn=None):
    data = np.loadtxt(post_file)
    data = get_samples_for_each_n(data, jitter_first=jitter_first, n_max=n_max)[n_comp]
    data = sort_samples_by_r(data, n_comp=n_comp, jitter_first=jitter_first)
    j = 0
    if jitter_first:
        j += 1
    fig = plot_position_posterior(data[:, j:], savefn, ra_lim, dec_lim, difmap_model_fn)
    return fig


def infer_column_positions(post_file, name):
    fo = open(post_file, "r")
    names = fo.readline()
    names = names.strip("\n# ")
    names = names.split()
    return [i for i, x in enumerate(names) if x == name]


def plot_corner(post_file, n_comp, n_max=30, savefn=None, truths=None, range_frac=1.0,
                jitter_first=False, plot_range=None, skip_first_coordinates=True):
    post_samples = np.loadtxt(post_file, skiprows=1)
    n_samples = get_samples_for_each_n(post_samples, jitter_first, n_max=n_max)
    samples = sort_samples_by_r(n_samples[n_comp], n_comp, comp_length=4,
                                jitter_first=jitter_first)
    return plot_corner_norj(samples, savefn, truths, range_frac,
                            jitter_first, plot_range, skip_first_coordinates)