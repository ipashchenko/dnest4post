import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from postprocess_labels_gains import (sort_samples_by_r, sort_samples_by_F,
                                      sort_samples_by_DEC, sort_samples_by_RA)
from plotting import plot_position_posterior
from plotting import plot_corner_gen as plot_corner_norj


label_size = 16
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_samples_for_each_n(samples, jitter_first=True, n_jitters=1, n_max=30,
                           skip_hyperparameters=False,
                           type="cg"):
    if type == "cg":
        nn = 0
        comp_length = 4
    elif type == "eg":
        nn = 2
        comp_length = 6
    j = 0
    if jitter_first:
        j += n_jitters
    # dim, max num components, 4 hyperparameters + num components
    j += (7 + nn)
    if skip_hyperparameters:
        j -= (4 + nn)
    n_components = samples[:, j-1]
    out_samples = dict()
    for n in np.array(np.unique(n_components), dtype=int):
        samples_with_n_components = list()
        for sample in samples[n_components == n]:
            one_post_point = list()
            if jitter_first:
                one_post_point.extend(sample[:n_jitters])
            for k in range(n):
                one_post_point.extend([sample[j+k+i*n_max] for i in range(comp_length)])
            samples_with_n_components.append(one_post_point)
        out_samples.update({n: np.atleast_2d(samples_with_n_components)})
    return out_samples


def rj_plot_ncomponents_distribution(posterior_file="posterior_sample.txt",
                                     picture_fn=None, jitter_first=True,
                                     n_jitters=1, skip_hyperparameters=False,
                                     type="cg", normed=False, show=True):
    samples = np.atleast_2d(np.loadtxt(posterior_file))
    if type == "cg":
        nn = 0
    elif type == "eg":
        nn = 2
    if jitter_first:
        ncomp_index = 6 + nn + n_jitters
    else:
        ncomp_index = 6 + nn
    if skip_hyperparameters:
        ncomp_index -= (4 + nn)
    values, counts = np.unique(samples[:, ncomp_index], return_counts=True)
    fig, axes = plt.subplots(1, 1)
    if normed:
        counts = counts/sum(counts)
    axes.vlines(values, 0, counts, color='C0', lw=8)
    axes.set_ylim(0, max(counts) * 1.06)
    axes.set_xticks(np.arange(min(values), max(values)+1))
    axes.set_xlabel("# components")
    if normed:
        axes.set_ylabel("P")
    else:
        axes.set_ylabel("N")
    if picture_fn is not None:
        fig.savefig(picture_fn, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def process_rj_samples(post_file, n_comp, n_max, jitter_first=True, n_jitters=1,
                       savefn=None, type="cg",
                       ra_lim=(-10, 10), dec_lim=(-10, 10), figsize=None,
                       difmap_model_fn=None, skip_hyperparameters=False, s=0.6,
                       sort_samples_by=None):
    if sort_samples_by is not None:
        assert sort_samples_by in ("r", "dec", "ra", "F")
    if type == "cg":
        comp_length = 4
    elif type == "eg":
        comp_length = 6
    data = np.atleast_2d(np.loadtxt(post_file))
    data = get_samples_for_each_n(data, jitter_first=jitter_first, n_jitters=n_jitters,
                                  n_max=n_max, type=type,
                                  skip_hyperparameters=skip_hyperparameters)[n_comp]
    if sort_samples_by == "r":
        data = sort_samples_by_r(data, n_comp=n_comp, jitter_first=jitter_first,
                                 n_jitters=n_jitters, comp_length=comp_length)
    elif sort_samples_by == "F":
        data = sort_samples_by_F(data, n_comp=n_comp, jitter_first=jitter_first,
                                 n_jitters=n_jitters, comp_length=comp_length)
    elif sort_samples_by == "ra":
        data = sort_samples_by_RA(data, n_comp=n_comp, jitter_first=jitter_first,
                                  n_jitters=n_jitters, comp_length=comp_length)
    elif sort_samples_by == "dec":
        data = sort_samples_by_DEC(data, n_comp=n_comp, jitter_first=jitter_first,
                                   n_jitters=n_jitters, comp_length=comp_length)
    else:
        pass

    j = 0
    if jitter_first:
        j += n_jitters
    if sort_samples_by is None:
        sorted_components = False
    else:
        sorted_components = True
    fig = plot_position_posterior(data[:, j:], savefn, ra_lim, dec_lim,
                                  difmap_model_fn, type=type, s=s, figsize=figsize,
                                  sorted_componets=sorted_components)
    return fig


def estimated_component_std(post_file, n_comp, n_max, ra_lim, dec_lim, jitter_first=True,
                            n_jitters=1, flux_as_log=True):
    data = np.atleast_2d(np.loadtxt(post_file))
    data = get_samples_for_each_n(data, jitter_first=jitter_first, n_jitters=n_jitters,
                                  n_max=n_max)[n_comp]
    ras = list()
    decs = list()
    fluxs = list()
    j_x = 0
    if jitter_first:
        j_x += n_jitters
    j_y = 1
    if jitter_first:
        j_y += n_jitters
    j_flux = 2
    if jitter_first:
        j_flux += n_jitters
    for sample in data:
        for i in range(n_comp):
            ra = sample[i*4+j_x]
            dec = sample[i*4 + j_y]
            flux = sample[i*4 + j_flux]
            if ra_lim[0] < ra < ra_lim[1] and dec_lim[0] < dec < dec_lim[1]:
                ras.append(ra)
                decs.append(dec)
                if flux_as_log:
                    fluxs.append(np.exp(flux))
                else:
                    fluxs.append(flux)

    print(fluxs)
    return fluxs, np.std(ras), np.std(decs), np.std(fluxs)


def infer_column_positions(post_file, name):
    fo = open(post_file, "r")
    names = fo.readline()
    names = names.strip("\n# ")
    names = names.split()
    return [i for i, x in enumerate(names) if x == name]


def get_values_at_given_positions(post_file, names):
    samples = np.loadtxt(post_file)
    indexes = list()
    for name in names:
        indexes.extend(infer_column_positions(post_file, name))
    return samples[np.ix_(np.arange(len(samples)), indexes)]


def plot_per_antenna_jitters(post_file, n_jitters, antenna_mapping=None):
    """
    :param post_file:
    :param n_jitters:
    :param antenna_mapping: (optional)
        Dictionary with keys - position number of antenna jitter (from 0 to
        number of antennas), values - strings with names of the antennas.
    :return:
    """
    samples = np.loadtxt(post_file)
    data = [samples[:, i] for i in range(n_jitters)]
    if antenna_mapping is None:
        labels = [str(i) for i in range(n_jitters)]
    else:
        labels = [antenna_mapping[i] for i in range(n_jitters)]
    df = pd.DataFrame.from_items(zip(labels, data))
    axes = sns.boxplot(data=df, orient='h')
    axes.set_xlabel(r"$\log{\sigma_{\rm ant}}$")
    axes.set_ylabel("Antenna")
    plt.tight_layout()
    plt.show()
    return axes


def plot_single_gain(post_file, ant, IF, n_max=1000, type="amp", rj_ncomp=None):
    df = pd.read_csv(post_file, header=0, delim_whitespace=True)
    suffix = "_ant{}_scan{}".format(ant, IF)
    positions = list()
    for i in range(n_max):
        res = infer_column_positions(post_file, "{}{}{}".format(type, i, suffix))
        if not res:
            break
        else:
            positions.extend(res)
    if not positions:
        raise Exception("No data on this gains in posterior file!")
    if rj_ncomp is not None:
        n_comp_pos = infer_column_positions(post_file, "num_components")[0]
    gains_dict = {}
    samples = np.loadtxt(post_file, skiprows=1)
    for i, pos in enumerate(sorted(positions)):
        if rj_ncomp is not None:
            toad = list()
            for sample in samples:
                if int(sample[n_comp_pos]) == rj_ncomp:
                    toad.append(sample[pos])
            toad = np.array(toad)
        else:
            toad = samples[:, pos]
        print("Updating {} with {}".format(i, toad))
        gains_dict.update({i: toad})
    df = pd.DataFrame.from_dict(gains_dict)
    import seaborn as sns
    axes = sns.boxplot(data=df, orient='v')
    # axes.axhline(1.0, lw=2, color="r")
    # axes.set_xlabel("Residual D-term estimate")
    # axes.set_ylabel("Antenna")
    # axes.set_xlim([0, 0.025])
    plt.tight_layout()
    plt.show()


def plot_corner(post_file, n_comp, n_max=30, savefn=None, truths=None,
                range_frac=1.0, jitter_first=False, n_jitters=0,
                plot_range=None, comp_type="circ", skip_hyperparameters=False):
    post_samples = np.atleast_2d(np.loadtxt(post_file, skiprows=1))
    n_params = {"pt": 3, "circ": 4, "ell": 6}[comp_type]
    n_samples = get_samples_for_each_n(post_samples, jitter_first, n_jitters,
                                       n_max=n_max, comp_length=n_params,
                                       skip_hyperparameters=skip_hyperparameters)
    samples = sort_samples_by_r(n_samples[n_comp], n_comp, comp_length=n_params,
                                jitter_first=jitter_first, n_jitters=n_jitters)
    return plot_corner_norj(samples[:, n_jitters:], savefn, truths, range_frac, jitter_first=False,
                            n_jitters=0, plot_range=plot_range, comp_coordinates_to_skip=0, comp_type=comp_type)


if __name__ == "__main__":
    # post_file = "/home/ilya/CLionProjects/basc/artificial/posterior_sample_art3cg_RJcirc_jitters_brightest.txt"
    # n_comp = 3
    # n_max = 20
    # save_fn = "/home/ilya/CLionProjects/basc/artificial/corner_RJ_3cg_jitters.png"
    # truths = [np.log(2), np.log(1), -5, 0, np.log(1), np.log(2), -20, -5, np.log(0.5), np.log(4)]
    # jitter_first = True
    # n_jitters = 5
    # plot_corner(post_file, n_comp, n_max, save_fn, truths, 1.0, jitter_first,
    #             n_jitters, None, "circ")

    from difmap_utils import convert_sample_to_difmap_model
    from plotting import plot_size_distance_posterior, plot_tb_distance_posterior

    posterior_file = "/home/ilya/github/bam/posterior_sample.txt"
    save_dir = "/home/ilya/data/bam"
    save_rj_ncomp_distribution_file = os.path.join(save_dir, "ncomponents_distribution.png")
    original_ccfits = "/home/ilya/data/bam/0212+735.u.2019_08_15.icn.fits"
    n_max = 20
    n_max_samples_to_plot = 500
    jitter_first = True
    component_type = "eg"
    pixsize_mas = 0.1
    freq_ghz = 15.4
    posterior_samples = np.loadtxt(posterior_file)
    fig = rj_plot_ncomponents_distribution(posterior_file, picture_fn=save_rj_ncomp_distribution_file,
                                           jitter_first=jitter_first, n_jitters=1, type=component_type,
                                           normed=True, show=False)
    samples_for_each_n = get_samples_for_each_n(posterior_samples, jitter_first,
                                                n_jitters=1, n_max=n_max,
                                                skip_hyperparameters=False,
                                                type=component_type)

    n_components_spread = samples_for_each_n.keys()

    import sys
    sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
    from from_fits import create_clean_image_from_fits_file
    from image import plot as iplot
    from spydiff import find_bbox, find_image_std

    ccimage = create_clean_image_from_fits_file(original_ccfits)
    beam = ccimage.beam
    # Number of pixels in beam
    npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*pixsize_mas**2)

    std = find_image_std(ccimage.image, npixels_beam, min_num_pixels_used_to_estimate_std=100,
                         blc=None, trc=None)
    blc, trc = find_bbox(ccimage.image, level=3*std, min_maxintensity_jyperbeam=10*std,
                         min_area_pix=3*npixels_beam, delta=0)
    fig = iplot(ccimage.image, x=ccimage.x, y=ccimage.y,
                min_abs_level=3*std, beam=beam, show_beam=True, blc=blc, trc=trc,
                components=None, close=False, plot_colorbar=False, show=False,
                contour_linewidth=0.25, contour_color='k')
    fig.savefig(os.path.join(save_dir, "CLEAN_image.png"), dpi=600)
    for n_component in n_components_spread:
        samples_to_plot = samples_for_each_n[n_component][:, 1:]
        n_samples = len(samples_to_plot)
        if n_samples > n_max_samples_to_plot:
            n_samples = n_max_samples_to_plot
        fig_out = plot_position_posterior(samples_to_plot[:n_max_samples_to_plot, :],
                                          savefn=None, ra_lim=None, dec_lim=None,
                                          difmap_model_fn=None, type=component_type, s=0.5, figsize=None,
                                          sorted_componets=False, fig=fig,
                                          inverse_xaxis=False,
                                          alpha_opacity=0.03)
        fig_out.savefig(os.path.join(save_dir, f"CLEAN_image_ncomp_{n_component}.png"), dpi=600)
        plt.close(fig_out)
        f = plot_size_distance_posterior(samples_to_plot[:n_max_samples_to_plot, :],
                                         savefn=os.path.join(save_dir, f"r_R_ncomp_{n_component}.png"),
                                         type="eg")
        f = plot_tb_distance_posterior(samples_to_plot[:n_max_samples_to_plot, :], freq_ghz, type="eg",
                                   savefn=os.path.join(save_dir, f"r_Tb_ncomp_{n_component}.png"))
        plt.close(f)