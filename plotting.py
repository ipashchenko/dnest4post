import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from matplotlib.patches import Ellipse, Circle
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from spydiff import import_difmap_model
# import seaborn as sns
import corner
from postprocess_labels import sort_samples_by_r, sort_samples_by_F
from postprocess_labels_rj import get_samples_for_each_n

mas_to_rad = u.mas.to(u.rad)
# Speed of light [cm / s]
c = const.c.cgs.value
k = const.k_B.cgs.value


def gaussian_circ_ft(flux, dx, dy, bmaj, uv):
    """
    FT of circular gaussian at ``uv`` points.

    :param flux:
        Full flux of gaussian.
    :param dx:
        Distance from phase center [mas].
    :param dy:
        Distance from phase center [mas].
    :param bmaj:
        FWHM of a gaussian [mas].
    :param uv:
        2D numpy array of (u,v)-coordinates (dimensionless).
    :return:
        Tuple of real and imaginary visibilities parts.
    """
    shift = [dx*mas_to_rad, dy*mas_to_rad]
    result = np.exp(-2.0*np.pi*1j*(uv @ shift))
    c = (np.pi*bmaj*mas_to_rad)**2/(4. * np.log(2.))
    b = uv[:, 0]**2 + uv[:, 1]**2
    ft = flux*np.exp(-c*b)
    ft = np.array(ft, dtype=complex)
    result *= ft
    return result.real, result.imag


def process_rj_samples(post_file, n_comp, n_max, jitter_first=True, savefn=None,
                         ra_lim=(-10, 10), dec_lim=(-10, 10),
                         difmap_model_fn=None):
    data = np.loadtxt(post_file)
    data = get_samples_for_each_n(data, jitter_first=jitter_first, n_max=n_max)[n_comp]
    data = sort_samples_by_r(data)
    fig = plot_position_posterior(data, savefn, ra_lim, dec_lim, difmap_model_fn)
    return fig


def process_norj_samples(post_file, jitter_first=True,
                         ra_lim=(-10, 10), dec_lim=(-10, 10), freq_ghz=15.4,
                         z=0.0,
                         difmap_model_fn=None, data_file=None, sort_by_r=True,
                         savefn_position_post=None, savefn_fluxsize_post=None,
                         savefn_rtb_post=None,
                         savefn_radplot_post=None):
    data = np.loadtxt(post_file)
    if jitter_first:
        data = data[:, 1:]
    if sort_by_r:
        data = sort_samples_by_r(data)
    else:
        data = sort_samples_by_F(data)
    fig1 = plot_position_posterior(data, savefn_position_post, ra_lim, dec_lim, difmap_model_fn)
    fig2 = plot_flux_size_posterior(data, freq_ghz, z, savefn=savefn_fluxsize_post)
    fig3 = plot_tb_distance_posterior(data, freq_ghz, z, savefn=savefn_rtb_post)
    if data_file is not None:
        fig4 = plot_radplot(data_file, data, savefn=savefn_radplot_post,
                            jitter_first=jitter_first)
    else:
        fig4 = None
    return fig1, fig2, fig3, fig4


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
        axes.scatter(xs[i_comp], ys[i_comp], s=0.6, color=color)

    if difmap_model_fn is not None:
        comps = import_difmap_model(difmap_model_fn)
        print("Read {} components from {}".format(len(comps), difmap_model_fn))
        for comp in comps:
            if comp.size == 3:
                axes.scatter(-comp.p[1], -comp.p[2], s=80, color="black", alpha=1, marker="x")
            elif comp.size == 4:
                e = Circle((-comp.p[1], -comp.p[2]), comp.p[3],
                           edgecolor="black", facecolor="red",
                           alpha=0.05)
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


# TODO: plot iso-Tb curves and resolution limit curves
def plot_flux_size_posterior(samples, freq_ghz=15.4, z=0, D=1, savefn=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, 1)
    log_fluxes = dict()
    log_sizes = dict()
    n_comps = int(len(samples[0])/4)

    for i_comp in range(n_comps):
        log_fluxes[i_comp] = samples[:, 2+i_comp*4]
        log_sizes[i_comp] = samples[:, 3+i_comp*4]

    # Find range of fluxes to plot iso-Tb lines
    lg_all_fluxes = np.log10(np.e)*np.concatenate(list(log_fluxes.values()))
    x = np.linspace(np.min(lg_all_fluxes), np.max(lg_all_fluxes), 100)
    for lg_tb in (9, 10.5, 12, 13):
        y = lg_size_for_given_flux_and_tb(10**x, lg_tb, freq_ghz, z, D)
        axes.plot(x, y, color='black', linestyle='--', label=r"$10^{%s}$ K" % (str(lg_tb)))

    from labellines import labelLines
    lines = axes.get_lines()
    labelLines(lines, xvals=[x[int(len(x)/10)]]*len(lines))

    for i_comp, color in zip(range(n_comps), colors):
        axes.scatter(np.log10(np.e)*log_fluxes[i_comp], np.log10(np.e)*log_sizes[i_comp], s=0.6, color=color)

    axes.set_xlabel("lg(flux [Jy])")
    axes.set_ylabel("lg(size [mas])")

    if savefn is not None:
        fig.savefig(savefn, dpi=300, bbox_inches="tight")
    return fig


def tb_comp(flux, bmaj, freq, z=0, bmin=None, D=1):
    """
    :param flux:
        Flux in Jy.
    :param freq:
        Frequency in GHz.
    """
    bmaj *= mas_to_rad
    if bmin is None:
        bmin = bmaj
    else:
        bmin *= mas_to_rad
    freq *= 10**9
    flux *= 10**(-23)
    return 2.*np.log(2)*(1.+z)*flux*c**2/(freq**2*np.pi*k*bmaj*bmin*D)


def lg_size_for_given_flux_and_tb(flux_jy, lg_tb, freq_ghz=15.4, z=0.0, D=1.0):
    freq = freq_ghz * 10**9
    flux = flux_jy * 10**(-23)
    lg_bmaj = 0.5*(np.log10(2.*np.log(2)*(1.+z)*c**2/(np.pi*k*freq**2*D))
                   + np.log10(flux) - lg_tb)
    return lg_bmaj - np.log10(mas_to_rad)


def plot_tb_distance_posterior(samples, freq_ghz, z=0.0, savefn=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, 1)
    lg_tbs = dict()
    lg_rs = dict()
    n_comps = int(len(samples[0])/4)

    for i_comp in range(n_comps):
        lg_rs[i_comp] = np.log10(np.hypot(samples[:, 0+i_comp*4], samples[:, 1+i_comp*4]))
        lg_tbs[i_comp] = np.log10(tb_comp(np.exp(samples[:, 2+i_comp*4]),
                                          np.exp(samples[:, 3+i_comp*4]),
                                          freq_ghz, z=z))

    for i_comp, color in zip(range(n_comps), colors):
        axes.scatter(lg_rs[i_comp], lg_tbs[i_comp], s=0.6, color=color)

    axes.set_xlabel("lg(r [mas])")
    axes.set_ylabel("lg(Tb [K])")

    if savefn is not None:
        fig.savefig(savefn, dpi=300, bbox_inches="tight")
    return fig


def plot_radplot(data_file, samples=None, samples_file=None, style="a&p",
                 savefn=None, n_samples=12, comp_length=4, jitter_first=True,
                 sort_by_r=True):

    if samples is None:
        if samples_file is None:
            raise Exception("Need samples or samples_file!")
        samples = np.loadtxt(samples_file)
    if jitter_first:
        samples = samples[:, 1:]
    if sort_by_r:
        samples = sort_samples_by_r(samples, comp_length)
    else:
        samples = sort_samples_by_F(samples, comp_length)

    # Plot data
    u, v, re, im, err = np.loadtxt(data_file, unpack=True)
    uv = np.column_stack((u, v))
    uv_radius = np.hypot(u, v)
    if style == "a&p":
        a1 = np.hypot(re, im)
        a2 = np.arctan2(re, im)
    if style == "re&im":
        a1 = re
        a2 = im

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
    axes[0].plot(uv_radius, a1, '_', color="#1f77b4", alpha=0.75, ms=5, mew=1)
    axes[1].plot(uv_radius, a2, '_', color="#1f77b4", alpha=0.75, ms=5, mew=1)

    length = len(samples[0])
    indx = np.random.randint(0, len(samples), n_samples)
    for i in indx:
        sample = samples[i]

        n_gaussians = int(length / comp_length)
        re_model = np.zeros(len(re))
        im_model = np.zeros(len(im))

        for n in range(n_gaussians):
            x = sample[n * comp_length + 0]
            y = sample[n * comp_length + 1]
            flux = np.exp(sample[n * comp_length + 2])
            size = np.exp(sample[n * comp_length + 3])
            re_model_, im_model_ = gaussian_circ_ft(flux, x, y, size, uv)
            re_model += re_model_
            im_model += im_model_

        if style == "a&p":
            a1 = np.hypot(re_model, im_model)
            a2 = np.arctan2(re_model, im_model)
        if style == "re&im":
            a1 = re_model
            a2 = im_model
        axes[0].plot(uv_radius, a1, '_', color="#ff7f0e", alpha=0.1, ms=5, mew=1)
        axes[1].plot(uv_radius, a2, '_', color="#ff7f0e", alpha=0.1, ms=5, mew=1)

    if style == 'a&p':
        axes[0].set_ylabel('Amplitude, [Jy]')
        axes[1].set_ylabel('Phase, [rad]')
    elif style == 're&im':
        axes[0].set_ylabel('Re, [Jy]')
        axes[1].set_ylabel('Im, [Jy]')
    axes[1].set_xlim(left=0)
    axes[1].set_xlabel('UV-radius, wavelengths')

    if savefn is not None:
        fig.savefig(savefn, dpi=100, bbox_inches="tight")
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