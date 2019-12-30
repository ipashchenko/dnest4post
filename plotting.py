import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from matplotlib.patches import Ellipse, Circle
from sklearn.covariance import MinCovDet
from sklearn.mixture import GaussianMixture as GMM
import datetime
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from spydiff import import_difmap_model
# import seaborn as sns
import corner
from postprocess_labels import (sort_samples_by_r, sort_samples_by_F,
                                find_component_xy_location_covariance,
                                find_ellipse_angle)
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


def plot_core_direction_several_epochs(post_files_dict, jitter_first=True, color=None, fig=None):
    if color is None:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    if fig is None:
        fig, axes = plt.subplots(1, 1)
    else:
        axes = fig.get_axes()[0]
    dates = list()
    angles = list()
    for i, (epoch, post_files) in enumerate(post_files_dict.items()):
        print("Processing epoch {}".format(epoch))
        for post_file in post_files:
            samples = np.loadtxt(post_file)
            if jitter_first:
                samples = samples[:, 1:]

            samples = sort_samples_by_F(samples)
            xy = samples[:, 0:2] - np.median(samples[:, 0:2], axis=0)
            try:
                loc, cov = find_component_xy_location_covariance(xy)
            except ValueError:
                loc, cov = find_component_xy_location_covariance(xy, type="gmm")

            v, w = np.linalg.eigh(cov[:2, :2])
            v = np.sqrt(v)
            print(v[0]/v[1])
            if v[0]/v[1] > 0.5:
                continue

            angle = find_ellipse_angle(cov)
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            angles.append(angle)
            dates.append(datetime.datetime.strptime(epoch, '%Y_%m_%d'))

    angles = np.rad2deg(np.unwrap(np.deg2rad(angles)))
    angles[angles < -60] += 180
    axes.scatter(dates, angles, s=10, color=color)
    # axes.set_ylim([0, 180])
    axes.set_xlabel("Date")
    axes.set_ylabel("Core angle - N-E, deg")
    # axes.set_aspect("equal")

    return fig


def plot_several_epochs(post_files_dict, jitter_first=True, ra_lim=(-10, 10),
                        dec_lim=(-10, 10)):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, 1)
    for i, (epoch, post_file) in enumerate(post_files_dict.items()):
        samples = np.loadtxt(post_file)
        if jitter_first:
            samples = samples[:, 1:]

        samples = sort_samples_by_F(samples)

        # Find position of most bright component and subtract its coordinates
        # from all coordinates
        x_core = np.median(samples[:, 0])
        y_core = np.median(samples[:, 1])

        xs = list()
        ys = list()
        n_comps = int(len(samples[0])/4)

        for i_comp in range(n_comps):
            xs.append(samples[:, 0+i_comp*4])
            ys.append(samples[:, 1+i_comp*4])

        xs = np.hstack(xs)
        ys = np.hstack(ys)

        axes.scatter(xs-x_core, ys-y_core, s=0.6, color=colors[i], alpha=0.5,
                     label=epoch)

    axes.set_xlim(ra_lim)
    axes.set_ylim(dec_lim)
    axes.set_xlabel("RA, mas")
    axes.set_ylabel("DEC, mas")
    axes.invert_xaxis()
    axes.set_aspect("equal")
    plt.legend(markerscale=5)

    return fig


def process_norj_samples(post_file, jitter_first=True,
                         ra_lim=(-10, 10), dec_lim=(-10, 10), freq_ghz=15.4,
                         z=0.0,
                         difmap_model_fn=None, data_file=None, sort_by_r=True,
                         savefn_position_post=None,
                         savefn_fluxsize_isot_post=None,
                         savefn_fluxsize_post=None,
                         savefn_rtb_post=None, savefn_sizer_post=None,
                         savefn_radplot_post=None):
    data = np.loadtxt(post_file)
    if jitter_first:
        data = data[:, 1:]
    if sort_by_r:
        data = sort_samples_by_r(data)
    else:
        data = sort_samples_by_F(data)
    from postprocess_labels import cluster_by_flux_size
    cluster_components = cluster_by_flux_size(data)
    fig1 = plot_position_posterior(data, savefn_position_post, ra_lim, dec_lim, difmap_model_fn)
    fig2 = plot_flux_size_posterior_isoT(data, freq_ghz, z, savefn=savefn_fluxsize_isot_post)
    fig3 = plot_flux_size_posterior(data, savefn=savefn_fluxsize_post)
    fig4 = plot_tb_distance_posterior(data, freq_ghz, z, savefn=savefn_rtb_post)
    fig5 = plot_size_distance_posterior(data, savefn=savefn_sizer_post)
    if data_file is not None:
        fig6 = plot_radplot(data_file, data, savefn=savefn_radplot_post,
                            jitter_first=jitter_first)
    else:
        fig6 = None
    fig7 = plot_flux_size_posterior_clusters(cluster_components)
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7


def plot_corner(samples, savefn=None, truths=None, range_frac=1.0,
                jitter_first=False, plot_range=None):
    columns = list()
    j = 0
    if jitter_first:
        j = 1
    for i in range(1, int(len(samples[0, j:])/4)+1):
        columns.append([r"$x_{}$".format(i), r"$y_{}$".format(i),
                        r"$\log{flux_{%s}}$" % str(i), r"$\log{bmaj_{%s}}$" % str(i)])
    columns = [item for sublist in columns for item in sublist]
    if jitter_first:
        columns.insert(0, r"$\log{\sigma_{\rm jitter}}$")
    if plot_range is None:
        plot_range = [range_frac] * len(columns)
    fig = corner.corner(samples, labels=columns, truths=truths,
                        show_titles=True, quantiles=[0.16, 0.5, 0.84],
                        color="gray", truth_color="#1f77b4",
                        plot_contours=True, range=plot_range,
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


def plot_flux_size_posterior_isoT(samples, freq_ghz=15.4, z=0, D=1, savefn=None):
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
    x = np.linspace(np.min(lg_all_fluxes)-0.1*np.ptp(lg_all_fluxes), np.max(lg_all_fluxes), 100)
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


def plot_flux_size_posterior(samples, savefn=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, 1)
    fluxes = dict()
    sizes = dict()
    n_comps = int(len(samples[0])/4)

    for i_comp in range(n_comps):
        fluxes[i_comp] = np.exp(samples[:, 2+i_comp*4])
        sizes[i_comp] = np.exp(samples[:, 3+i_comp*4])

    for i_comp, color in zip(range(n_comps), colors):
        axes.scatter(fluxes[i_comp], sizes[i_comp], s=0.6, color=color)

    axes.set_xlabel("flux [Jy]")
    axes.set_ylabel("FWHM [mas]")
    axes.set_xscale('log')
    axes.set_yscale('log')

    if savefn is not None:
        fig.savefig(savefn, dpi=300, bbox_inches="tight")
    return fig


def plot_flux_size_posterior_clusters(cluster_components, savefn=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, 1)

    for cluster_id, components in cluster_components.items():
        axes.scatter(np.exp(components[:, 2]), np.exp(components[:, 3]), s=0.6, color=colors[cluster_id])

    axes.set_xlabel("flux [Jy]")
    axes.set_ylabel("FWHM [mas]")
    axes.set_xscale('log')
    axes.set_yscale('log')

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
    tbs = dict()
    rs = dict()
    n_comps = int(len(samples[0])/4)

    for i_comp in range(n_comps):
        rs[i_comp] = np.hypot(samples[:, 0+i_comp*4], samples[:, 1+i_comp*4])
        tbs[i_comp] = tb_comp(np.exp(samples[:, 2+i_comp*4]),
                                          np.exp(samples[:, 3+i_comp*4]),
                                          freq_ghz, z=z)

    for i_comp, color in zip(range(n_comps), colors):
        axes.scatter(rs[i_comp], tbs[i_comp], s=0.6, color=color)

    # Need manually set ylim because of matplotlib bug
    lg_tb_min = np.floor((np.log10(np.min([tbs[i] for i in range(n_comps)]))))
    lg_tb_max = np.ceil(np.log10(np.max([tbs[i] for i in range(n_comps)])))
    axes.set_ylim([10**lg_tb_min, 10**lg_tb_max])
    axes.set_xlabel("r [mas]")
    axes.set_ylabel("Tb [K]")
    axes.set_xscale('log')
    axes.set_yscale('log')

    if savefn is not None:
        fig.savefig(savefn, dpi=300, bbox_inches="tight")
    return fig


def plot_size_distance_posterior(samples, savefn=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, 1)
    sizes = dict()
    rs = dict()
    n_comps = int(len(samples[0])/4)

    for i_comp in range(n_comps):
        rs[i_comp] = np.hypot(samples[:, 0+i_comp*4], samples[:, 1+i_comp*4])
        sizes[i_comp] = np.exp(samples[:, 3+i_comp*4])

    for i_comp, color in zip(range(n_comps), colors):
        axes.scatter(rs[i_comp], sizes[i_comp], s=0.6, color=color)

    axes.set_xlabel("r [mas]")
    axes.set_ylabel("FWHM [mas]")
    axes.set_xscale('log')
    axes.set_yscale('log')

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


if __name__ == "__main__":
    import matplotlib

    label_size = 16
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plot_core_direction_several_epochs({"1997_08_18": ["1997_08_18.u.5comp_norj.txt", "1997_08_18.u.6comp_norj.txt", "1997_08_18.u.7comp_norj.txt", "1997_08_18.u.8comp_norj.txt"],
                                              "2000_01_11": ["2000_01_11.u.6comp_norj.txt", "2000_01_11.u.7comp_norj.txt", "2000_01_11.u.8comp_norj.txt", "2000_01_11.u.9comp_norj.txt"],
                                              "2002_08_12": ["2002_08_12.u.4comp_norj.txt", "2002_08_12.u.5comp_norj.txt", "2002_08_12.u.6comp_norj.txt", "2002_08_12.u.7comp_norj.txt", "2002_08_12.u.8comp_norj.txt"],
                                              "2003_03_29": ["2003_03_29.u.6comp_norj.txt", "2003_03_29.u.7comp_norj.txt", "2003_03_29.u.8comp_norj.txt", "2003_03_29.u.9comp_norj.txt"],
                                              "2004_10_18": ["2004_10_18.u.6comp_norj.txt", "2004_10_18.u.7comp_norj.txt", "2004_10_18.u.8comp_norj.txt"],
                                              "2005_05_13": ["2005_05_13.u.6comp_norj.txt", "2005_05_13.u.7comp_norj.txt", "2005_05_13.u.8comp_norj.txt", "2005_05_13.u.9comp_norj.txt"],
                                              "2005_09_23": ["2005_09_23.u.6comp_norj.txt", "2005_09_23.u.7comp_norj.txt", "2005_09_23.u.8comp_norj.txt" ],
                                              "2005_10_29": ["2005_10_29.u.6comp_norj.txt", "2005_10_29.u.7comp_norj.txt", "2005_10_29.u.8comp_norj.txt"],
                                              "2005_11_17": ["2005_11_17.u.6comp_norj.txt", "2005_11_17.u.7comp_norj.txt", "2005_11_17.u.8comp_norj.txt"],
                                              "2006_07_07": ["2006_07_07.u.6comp_norj.txt", "2006_07_07.u.7comp_norj.txt", "2006_07_07.u.8comp_norj.txt"],
                                              "2007_08_16": ["2007_08_16.u.6comp_norj.txt", "2007_08_16.u.7comp_norj.txt", "2007_08_16.u.8comp_norj.txt", "2007_08_16.u.9comp_norj.txt"],
                                              "2008_06_25": ["2008_06_25.u.6comp_norj.txt", "2008_06_25.u.7comp_norj.txt", "2008_06_25.u.8comp_norj.txt"],
                                              "2008_08_06": ["2008_08_06.u.6comp_norj.txt", "2008_08_06.u.7comp_norj.txt", "2008_08_06.u.8comp_norj.txt"],
                                              "2008_11_19": ["2008_11_19.u.6comp_norj.txt", "2008_11_19.u.7comp_norj.txt", "2008_11_19.u.8comp_norj.txt", "2008_11_19.u.9comp_norj.txt"],
                                              "2009_03_25": ["2009_03_25.u.6comp_norj.txt", "2009_03_25.u.7comp_norj.txt", "2009_03_25.u.8comp_norj.txt", "2009_03_25.u.9comp_norj.txt"],
                                              "2009_12_10": ["2009_12_10.u.6comp_norj.txt", "2009_12_10.u.7comp_norj.txt", "2009_12_10.u.8comp_norj.txt"],
                                              "2010_06_19": ["2010_06_19.u.7comp_norj.txt", "2010_06_19.u.8comp_norj.txt", "2010_06_19.u.9comp_norj.txt"],
                                              "2010_06_27": ["2010_06_27.u.7comp_norj.txt", "2010_06_27.u.8comp_norj.txt"],
                                              "2010_08_27": ["2010_08_27.u.5comp_norj.txt", "2010_08_27.u.6comp_norj.txt", "2010_08_27.u.7comp_norj.txt", "2010_08_27.u.8comp_norj.txt"],
                                              "2010_11_13": ["2010_11_13.u.7comp_norj.txt", "2010_11_13.u.8comp_norj.txt", "2010_11_13.u.9comp_norj.txt"],
                                              "2011_02_27": ["2011_02_27.u.6comp_norj.txt", "2011_02_27.u.7comp_norj.txt"],
                                              "2011_08_15": ["2011_08_15.u.5comp_norj.txt", "2011_08_15.u.6comp_norj.txt", "2011_08_15.u.7comp_norj.txt"],
                                              "2019_08_23": ["2019_08_23.u.5comp_norj.txt", "2019_08_23.u.6comp_norj.txt", "2019_08_23.u.7comp_norj.txt", "2019_08_23.u.8comp_norj.txt"],
                                              "2019_08_27": ["2019_08_27.u.3comp_norj.txt", "2019_08_27.u.4comp_norj.txt", "2019_08_27.u.5comp_norj.txt", "2019_08_27.u.6comp_norj.txt", "2019_08_27.u.7comp_norj.txt", "2019_08_27.u.8comp_norj.txt"],
                                              "2019_10_11": ["2019_10_11.u.4comp_norj.txt", "2019_10_11.u.5comp_norj.txt", "2019_10_11.u.6comp_norj.txt", "2019_10_11.u.7comp_norj.txt"]})

    fig = plot_core_direction_several_epochs({"1997_08_18": ["1997_08_18.u.5comp_norj.txt"],
                                              "2000_01_11": ["2000_01_11.u.6comp_norj.txt"],
                                              "2002_08_12": ["2002_08_12.u.4comp_norj.txt"],
                                              "2003_03_29": ["2003_03_29.u.6comp_norj.txt"],
                                              "2004_10_18": ["2004_10_18.u.6comp_norj.txt"],
                                              "2005_05_13": ["2005_05_13.u.6comp_norj.txt"],
                                              "2005_09_23": ["2005_09_23.u.6comp_norj.txt"],
                                              "2005_10_29": ["2005_10_29.u.6comp_norj.txt"],
                                              "2005_11_17": ["2005_11_17.u.6comp_norj.txt"],
                                              "2006_07_07": ["2006_07_07.u.6comp_norj.txt"],
                                              "2007_08_16": ["2007_08_16.u.6comp_norj.txt"],
                                              "2008_06_25": ["2008_06_25.u.6comp_norj.txt"],
                                              "2008_08_06": ["2008_08_06.u.6comp_norj.txt"],
                                              "2008_11_19": ["2008_11_19.u.6comp_norj.txt"],
                                              "2009_03_25": ["2009_03_25.u.6comp_norj.txt"],
                                              "2009_12_10": ["2009_12_10.u.6comp_norj.txt"],
                                              "2010_06_19": ["2010_06_19.u.7comp_norj.txt"],
                                              "2010_06_27": ["2010_06_27.u.7comp_norj.txt"],
                                              "2010_08_27": ["2010_08_27.u.5comp_norj.txt"],
                                              "2010_11_13": ["2010_11_13.u.7comp_norj.txt"],
                                              "2011_02_27": ["2011_02_27.u.6comp_norj.txt"],
                                              "2011_08_15": ["2011_08_15.u.5comp_norj.txt"],
                                              "2019_08_23": ["2019_08_23.u.5comp_norj.txt"],
                                              "2019_08_27": ["2019_08_27.u.3comp_norj.txt"],
                                              "2019_10_11": ["2019_10_11.u.4comp_norj.txt"]
                                              }, fig=fig, color="green")
    fig = plot_core_direction_several_epochs({"1997_08_18": ["1997_08_18.u.8comp_norj.txt"],
                                              "2000_01_11": ["2000_01_11.u.9comp_norj.txt"],
                                              "2002_08_12": ["2002_08_12.u.8comp_norj.txt"],
                                              "2003_03_29": ["2003_03_29.u.9comp_norj.txt"],
                                              "2004_10_18": ["2004_10_18.u.8comp_norj.txt"],
                                              "2005_05_13": ["2005_05_13.u.9comp_norj.txt"],
                                              "2005_09_23": ["2005_09_23.u.8comp_norj.txt"],
                                              "2005_10_29": ["2005_10_29.u.8comp_norj.txt"],
                                              "2005_11_17": ["2005_11_17.u.8comp_norj.txt"],
                                              "2006_07_07": ["2006_07_07.u.8comp_norj.txt"],
                                              "2007_08_16": ["2007_08_16.u.9comp_norj.txt"],
                                              "2008_06_25": ["2008_06_25.u.8comp_norj.txt"],
                                              "2008_08_06": ["2008_08_06.u.8comp_norj.txt"],
                                              "2008_11_19": ["2008_11_19.u.9comp_norj.txt"],
                                              "2009_03_25": ["2009_03_25.u.9comp_norj.txt"],
                                              "2009_12_10": ["2009_12_10.u.8comp_norj.txt"],
                                              "2010_06_19": ["2010_06_19.u.9comp_norj.txt"],
                                              "2010_06_27": ["2010_06_27.u.8comp_norj.txt"],
                                              "2010_08_27": ["2010_08_27.u.8comp_norj.txt"],
                                              "2010_11_13": ["2010_11_13.u.9comp_norj.txt"],
                                              "2011_02_27": ["2011_02_27.u.7comp_norj.txt"],
                                              "2011_08_15": ["2011_08_15.u.7comp_norj.txt"],
                                              "2019_08_23": ["2019_08_23.u.8comp_norj.txt"],
                                              "2019_08_27": ["2019_08_27.u.8comp_norj.txt"],
                                              "2019_10_11": ["2019_10_11.u.7comp_norj.txt"]
                                              }, fig=fig, color="red")
    fig.savefig("core_direction_full_v1.png", bbox_inches="tight", dpi=300)

    # post_files_dict = {#"2004_10_18": "2004_10_18.u.6comp_norj.txt",
    #                                           "2005_05_13": "2005_05_13.u.7comp_norj.txt",
    #                                           "2005_09_23": "2005_09_23.u.7comp_norj.txt",
    #                                           "2005_10_29": "2005_10_29.u.7comp_norj.txt",
    #                                           "2005_11_17": "2005_11_17.u.6comp_norj.txt",
    #                                           "2006_07_07": "2006_07_07.u.8comp_norj.txt"}
    #
    # post_files_dict = {#"2008_06_25": "2008_06_25.u.7comp_norj.txt",
    #                                           # "2008_08_06": "2008_08_06.u.8comp_norj.txt",
    #                                           "2008_11_19": "2008_11_19.u.6comp_norj.txt",
    #                                           "2009_03_25": "2009_03_25.u.8comp_norj.txt",
    #                                           "2009_12_10": "2009_12_10.u.7comp_norj.txt"}
    #                                           # "2010_06_19": "2010_06_19.u.9comp_norj.txt"}
    #
    # # This shows components going to center (gathering)
    # post_files_dict = {"2019_08_27": "2019_08_27.u.6comp_norj.txt",
    #                    "2019_10_11": "2019_10_11.u.6comp_norj.txt"}
    #
    # post_files_dict = {"2019_08_23": "2019_08_23.u.6comp_norj.txt",
    #                    "2019_10_11": "2019_10_11.u.6comp_norj.txt"}
    #
    # #This shows components going from center
    # post_files_dict = {"2019_08_23": "2019_08_23.u.6comp_norj.txt",
    #                    "2019_08_27": "2019_08_27.u.6comp_norj.txt"}
    #
    # post_files_dict = {"2008_06_25": "2008_06_25.u.7comp_norj.txt",
    #                    "2008_08_06": "2008_08_06.u.8comp_norj.txt",
    #                    "2008_11_19": "2008_11_19.u.7comp_norj.txt"}
    #                    # "2009_03_25": "2009_03_25.u.8comp_norj.txt",
    #                    # "2009_12_10": "2009_12_10.u.7comp_norj.txt",
    #                    # "2010_06_19": "2010_06_19.u.9comp_norj.txt",
    #                    # "2010_06_27": "2010_06_27.u.8comp_norj.txt",
    #                    # "2010_08_27": "2010_08_27.u.7comp_norj.txt",
    #                    # "2010_11_13": "2010_11_13.u.9comp_norj.txt",
    #                    # "2011_02_27": "2011_02_27.u.7comp_norj.txt",
    #                    # "2011_08_15": "2011_08_15.u.7comp_norj.txt"}

    post_files_dict = {"1997_08_18": "1997_08_18.u.6comp_norj.txt",
                       "2000_01_11": "2000_01_11.u.7comp_norj.txt",
                       "2002_08_12": "2002_08_12.u.7comp_norj.txt"}

    # TODO: Waiting for 9 components of both epochs
    post_files_dict = {"2002_08_12": "2002_08_12.u.7comp_norj.txt",
                       "2003_03_29": "2003_03_29.u.8comp_norj.txt"}
    # TODO: Changing jet direction (both core PA and whole jet)
    post_files_dict = {"2003_03_29": "2003_03_29.u.7comp_norj.txt",
                       "2004_10_18": "2004_10_18.u.7comp_norj.txt"}
    post_files_dict = {"2004_10_18": "2004_10_18.u.7comp_norj.txt",
                       "2005_05_13": "2005_05_13.u.8comp_norj.txt"}
    # Both 7 and 8 components look consistent
    post_files_dict = {"2005_05_13": "2005_05_13.u.7comp_norj.txt",
                       "2005_09_23": "2005_09_23.u.7comp_norj.txt"}
    # Both 7 and 8 components look consistent
    post_files_dict = {"2005_09_23": "2005_09_23.u.7comp_norj.txt",
                       "2005_10_29": "2005_10_29.u.7comp_norj.txt"}
    # 7 comps for 11_17 results in component at (1.3, -3) with 10 mJy and size
    # ~ 0.3 mas (modelling diffuse emission)
    post_files_dict = {"2005_10_29": "2005_10_29.u.7comp_norj.txt",
                       "2005_11_17": "2005_11_17.u.6comp_norj.txt"}
    # 7 comp for 07_07 results in component at (-0.5, -0.5) with 5 mJy and size
    # ~ 0.1 mas
    post_files_dict = {"2005_11_17": "2005_11_17.u.6comp_norj.txt",
                       "2006_07_07": "2006_07_07.u.6comp_norj.txt"}
    post_files_dict = {"2006_07_07": "2006_07_07.u.6comp_norj.txt",
                       "2007_08_16": "2007_08_16.u.7comp_norj.txt"}
    # Not so clear, but S-shaped jet is evident
    post_files_dict = {"2007_08_16": "2007_08_16.u.7comp_norj.txt",
                       "2008_06_25": "2008_06_25.u.7comp_norj.txt"}
    # 08_06 8 components coincides with difmap model
    post_files_dict = {"2008_06_25": "2008_06_25.u.7comp_norj.txt",
                       "2008_08_06": "2008_08_06.u.8comp_norj.txt"}
    # FIXME: Unclear how many components to use for 11_19. There's one
    # suspicious at r=(0.75, -0.6) with 30-40 mJy and 1 mas size that falls out
    # of thrend with r. Possibly represent extended structure. And even for
    # 8-component model there's at r=(0.5, -0.3) with 20-30 mJy and 0.01-0.1 mas
    # size. Only 7-component model doesn't have such
    post_files_dict = {"2008_08_06": "2008_08_06.u.7comp_norj.txt",
                       "2008_11_19": "2008_11_19.u.7comp_norj.txt"}
    # TODO: Waiting for 9 component 2009_03_25, however even 8 components
    # results in 6 mJy, 0.1 mas component at r=(0.8, -0.1)
    post_files_dict = {"2008_11_19": "2008_11_19.u.9comp_norj.txt",
                       "2009_03_25": "2009_03_25.u.8comp_norj.txt"}
    # Abrupt change in jet direction on 50 deg
    post_files_dict = {"2009_03_25": "2009_03_25.u.7comp_norj.txt",
                       "2009_12_10": "2009_12_10.u.8comp_norj.txt"}
    # TODO: Waiting 7 component of 06_19, but 8 component model coincides with difmap.
    # TODO: Waiting 9 component of 12_10
    post_files_dict = {"2009_12_10": "2009_12_10.u.8comp_norj.txt",
                       "2010_06_19": "2010_06_19.u.8comp_norj.txt"}
    # TODO: Waiting 7 component of 06_19, may be 7 & 7 coincides perfectly
    # Now only one component of 06_19 has no counterpart in 06_27
    post_files_dict = {"2010_06_19": "2010_06_19.u.8comp_norj.txt",
                       "2010_06_27": "2010_06_27.u.7comp_norj.txt"}
    post_files_dict = {"2010_06_27": "2010_06_27.u.7comp_norj.txt",
                       "2010_08_27": "2010_08_27.u.7comp_norj.txt"}
    # Slight change in inner and whole jet direction
    post_files_dict = {"2010_08_27": "2010_08_27.u.7comp_norj.txt",
                       "2010_11_13": "2010_11_13.u.7comp_norj.txt"}
    # TODO: Waiting 6 component for 11_13
    post_files_dict = {"2010_11_13": "2010_11_13.u.7comp_norj.txt",
                       "2011_02_27": "2011_02_27.u.7comp_norj.txt"}
    # Models with 6 components coincide, but 02_27 has reacher structure
    # FIXME: Bad amp self-cal for 2011-08-15 (SC)
    post_files_dict = {"2011_02_27": "2011_02_27.u.7comp_norj.txt",
                       "2011_08_15": "2011_08_15.u.7comp_norj.txt"}




    fig = plot_several_epochs(post_files_dict, ra_lim=(-3, 9), dec_lim=(-3, 2))

