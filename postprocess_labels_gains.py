import numpy as np
import matplotlib.pyplot as plt
# This postprocess samples of fixed number of components + gains


# This getters are applicable to already de-RJ samples, i.e. those that were
# obtained by rj.get_samples_for_each_n
def get_r(sample, comp_length, n_comp, jitter_first=True, n_jitters=1):
    j = 0
    if jitter_first:
        j += n_jitters
    return [np.hypot(sample[i*comp_length+j], sample[i*comp_length+j+1]) for i in
            range(n_comp)]


def get_x(sample, comp_length, n_comp, jitter_first=True, n_jitters=1):
    j = 0
    if jitter_first:
        j += n_jitters
    return [sample[i*comp_length+j] for i in range(n_comp)]


def get_y(sample, comp_length, n_comp, jitter_first=True, n_jitters=1):
    j = 1
    if jitter_first:
        j += n_jitters
    return [sample[i*comp_length+j] for i in range(n_comp)]


def get_F(sample, comp_length, n_comp, jitter_first=True, n_jitters=1):
    j = 2
    if jitter_first:
        j += n_jitters
    return [(sample[i*comp_length+j]) for i in range(n_comp)]


def sort_sample_by_r(sample, n_comp, comp_length=4, jitter_first=True,
                     n_jitters=1):
    r = get_r(sample, comp_length, n_comp, jitter_first, n_jitters)
    indices = np.argsort(r)
    # Construct re-labelled sample
    j = 0
    if jitter_first:
        j += n_jitters
    else:
        n_jitters = 0
    result = np.hstack([sample[j+i*comp_length: j+(i+1)*comp_length] for i in
                        indices])
    return np.hstack((sample[: n_jitters], result))


def sort_sample_by_F(sample, n_comp, comp_length=4, jitter_first=True,
                     n_jitters=1):
    F = get_F(sample, comp_length, n_comp, jitter_first, n_jitters)
    indices = np.argsort(F)[::-1]
    # Construct re-labelled sample
    j = 0
    if jitter_first:
        j += n_jitters
    return np.hstack([sample[j+i*comp_length: j+(i+1)*comp_length] for i in
                      indices])


def sort_sample_by_DEC(sample, n_comp, comp_length=4, jitter_first=True, n_jitters=1, inverse=False):
    dec = get_x(sample, comp_length, n_comp, jitter_first, n_jitters)
    indices = np.argsort(dec)
    if inverse:
        indices = indices[::-1]
    # Construct re-labelled sample
    j = 0
    if jitter_first:
        j += n_jitters
    return np.hstack([sample[j+i*comp_length: j+(i+1)*comp_length] for i in
                      indices])


def sort_sample_by_RA(sample, n_comp, comp_length=4, jitter_first=True, n_jitters=1, inverse=False):
    dec = get_y(sample, comp_length, n_comp, jitter_first, n_jitters)
    indices = np.argsort(dec)
    if inverse:
        indices = indices[::-1]
    # Construct re-labelled sample
    j = 0
    if jitter_first:
        j += n_jitters
    return np.hstack([sample[j+i*comp_length: j+(i+1)*comp_length] for i in
                      indices])

def sort_samples_by_r(samples, n_comp, comp_length=4, jitter_first=True,
                      n_jitters=1):
    new_samples = list()
    for sample in samples:
        sorted_sample = sort_sample_by_r(sample, n_comp, comp_length, jitter_first,
                                         n_jitters)
        new_samples.append(sorted_sample)
    return np.atleast_2d(new_samples)


def sort_samples_by_F(samples, n_comp, comp_length=4, jitter_first=True, n_jitters=1):
    new_samples = list()
    for sample in samples:
        sorted_sample = sort_sample_by_F(sample, n_comp, comp_length, jitter_first, n_jitters)
        if jitter_first:
            sorted_sample = np.append(sorted_sample[::-1], sample[0])[::-1]
        new_samples.append(sorted_sample)
    return np.atleast_2d(new_samples)



def sort_samples_by_DEC(samples, n_comp, comp_length=4, jitter_first=True, n_jitters=1, inverse=False):
    """
    Sort each sample by DEC.
    """
    new_samples = list()
    for sample in samples:
        sorted_sample = sort_sample_by_DEC(sample, n_comp, comp_length, jitter_first, n_jitters, inverse)
        if jitter_first:
            sorted_sample = np.append(sorted_sample[::-n_jitters], sample[0])[::-1]
        new_samples.append(sorted_sample)
    return np.atleast_2d(new_samples)


def sort_samples_by_RA(samples, n_comp, comp_length=4, jitter_first=True, n_jitters=1, inverse=False):
    """
    Sort each sample by RA.
    """
    new_samples = list()
    for sample in samples:
        sorted_sample = sort_sample_by_RA(sample, n_comp, comp_length, jitter_first, n_jitters, inverse)
        if jitter_first:
            sorted_sample = np.append(sorted_sample[::-n_jitters], sample[0])[::-1]
        new_samples.append(sorted_sample)
    return np.atleast_2d(new_samples)


def scans(times):
    times = np.sort(np.array(times))
    a, b = np.histogram(times[1:] - times[:-1])
    scan_borders = times[(np.where((times[1:] - times[:-1]) > b[1])[0])]
    scans_list = [[times[0], scan_borders[0]]]
    if len(scan_borders) > 1:
        for i in range(len(scan_borders) - 1):
            scans_list.append([float(times[np.where(times == scan_borders[i])[0] + 1]),
                               scan_borders[i + 1]])
    else:
        i = -1
    scans_list.append([float(times[np.where(times == scan_borders[i + 1])[0] + 1]),
                       times[-1]])
    return scans_list


def process_sampled_gains(posterior_sample, df_fitted, jitter_first=True, n_comp=1, plotfn=None,
                          with_mean_phase=True, with_mean_amplitude=False, add_mean_phase=False):
    """
    :param posterior_sample:
        DNest file with posterior.
    :param df_fitted:
        Dataframe fitted (created by ``create_data_file`` and ``inject_gains``).
    :param jitter_first: (optional)
        Boolean. Is jitter samples go first? (default: ``True``)
    :param n_comp: (optional)
        Number of components in model. (default: ``1``)
    :param plotfn: (optional)
        File to save picture. If ``None`` than just return figure.
    :return:
        Dictionary with keys - antenna numbers (as in ``gains_dict``), times, ``amp``, ``phase``
        and values - samples of the posterior distribution for amplitude and phase at given time
        for given antenna.
    """
    samples = np.loadtxt(posterior_sample, skiprows=1)
    first_gain_index = n_comp*4
    if jitter_first:
        first_gain_index += 1
    gains_len = dict()
    gains_post = dict()
    j = first_gain_index
    antennas = set(list(df_fitted.ant1.unique()) + list(df_fitted.ant2.unique()))
    for ant in antennas:
        gains_len[ant] = dict()
        gains_len[ant]["amp"] = len(set(df_fitted.query("ant1 == @ant or ant2 == @ant").times_amp.values))
        gains_len[ant]["phase"] = len(set(df_fitted.query("ant1 == @ant or ant2 == @ant").times_phase.values))
        gains_post[ant] = dict()
        gains_post[ant]["amp"] = dict()
        gains_post[ant]["phase"] = dict()
        h = 0
        k = 0
        if with_mean_phase:
            gains_post[ant]["mean_phase"] = dict()
            gains_post[ant]["mean_phase"] = samples[:, j+gains_len[ant]["amp"]]
            h += 1
        if with_mean_amplitude:
            gains_post[ant]["mean_amp"] = dict()
            gains_post[ant]["mean_amp"] = samples[:, j]
            k += 1
        for i, t in enumerate(df_fitted.query("ant1 == @ant or ant2 == @ant").times_amp.unique()):
            gains_post[ant]["amp"][t] = samples[:, j+i+k]
            # +1 mean skip ``mean_phase``
            if with_mean_phase:
                if add_mean_phase:
                    gains_post[ant]["phase"][t] = samples[:, j+gains_len[ant]["amp"]+i+1+k] + gains_post[ant]["mean_phase"]
                else:
                    gains_post[ant]["phase"][t] = samples[:, j+gains_len[ant]["amp"]+i+1+k]
            else:
                gains_post[ant]["phase"][t] = samples[:, j+gains_len[ant]["amp"]+i+k]


        j += gains_len[ant]["amp"] + gains_len[ant]["phase"] + h + k

    # Find scans
    times = list()
    for ant in antennas:
        times.extend(list(gains_post[ant]['amp'].keys()))
    times = set(times)
    times = list(times)
    scans_list = scans(times)
    n_scans = len(scans_list)

    fig, axes = plt.subplots(len(gains_len), n_scans, sharex=True, sharey=False, figsize=(12, 20),
                             gridspec_kw={'wspace': 0.3, 'hspace': 0.2})
    for i, ant in enumerate(gains_post):
        print("Plotting antenna ", ant)

        for scan_number, scan in enumerate(scans_list):
            t_low, t_up = scan
            t0 = None
            for t in sorted(gains_post[ant]["amp"].keys()):
                if not t_low <= t <= t_up:
                    continue
                if t0 is None:
                    t0 = t
                # Array with posterior for given t
                amp = gains_post[ant]["amp"][t]
                ts = np.array([t-t0]*len(amp))
                alpha = 6e-5*len(amp)
                ts += np.random.normal(loc=0, scale=1.0, size=len(ts))
                axes[i, scan_number].scatter(ts, amp, color="#1f77b4", alpha=alpha, s=2)
                axes[i, scan_number].axhline(1.0, color="C1", lw=0.5)

    fig.text(0.5, 0.04, 'scan time, s', ha='center')
    fig.text(0.04, 0.5, 'Gain amplitude', va='center', rotation='vertical')

    if plotfn:
        fig.savefig(plotfn+"_amplitudes.png", bbox_inches="tight", dpi=100)
    fig.show()


    fig, axes = plt.subplots(len(gains_len), n_scans, sharex=True, sharey=False, figsize=(12, 20),
                             gridspec_kw={'wspace': 0.3, 'hspace': 0.2})
    for i, ant in enumerate(gains_post):
        print("Plotting antenna ", ant)

        for scan_number, scan in enumerate(scans_list):
            t_low, t_up = scan
            t0 = None
            for t in sorted(gains_post[ant]["phase"].keys()):
                if not t_low <= t <= t_up:
                    continue
                if t0 is None:
                    t0 = t
                # Array with posterior for given t
                phase = gains_post[ant]["phase"][t]
                ts = np.array([t-t0]*len(phase))
                alpha = 6e-5*len(phase)
                ts += np.random.normal(loc=0, scale=1.0, size=len(ts))
                axes[i, scan_number].scatter(ts, phase, color="#1f77b4", alpha=alpha, s=2)
                axes[i, scan_number].axhline(0.0, color="C1", lw=0.5)

    fig.text(0.5, 0.04, 'scan time, s', ha='center')
    fig.text(0.04, 0.5, 'Gain phase, rad', va='center', rotation='vertical')

    if plotfn:
        fig.savefig(plotfn+"_phases.png", bbox_inches="tight", dpi=100)
    fig.show()

    return gains_post


def process_sampled_gains_many_IFs(posterior_sample, df_fitted, plot_IF=0, jitter_first=True, n_comp=1, plotfn=None,
                          with_mean_phase=True, add_mean_phase=False, n_IF=4):
    """
    :param posterior_sample:
        DNest file with posterior.
    :param df_fitted:
        Dataframe fitted (created by ``create_data_file`` and ``inject_gains``).
    :param jitter_first: (optional)
        Boolean. Is jitter samples go first? (default: ``True``)
    :param n_comp: (optional)
        Number of components in model. (default: ``1``)
    :param plotfn: (optional)
        File to save picture. If ``None`` than just return figure.
    :return:
        Dictionary with keys - antenna numbers (as in ``gains_dict``), times, ``amp``, ``phase``
        and values - samples of the posterior distribution for amplitude and phase at given time
        for given antenna.
    """
    samples = np.loadtxt(posterior_sample, skiprows=1)
    first_gain_index = n_comp*4
    if jitter_first:
        first_gain_index += 1
    gains_len = dict()
    gains_post = dict()
    j = first_gain_index
    antennas = set(list(df_fitted.ant1.unique()) + list(df_fitted.ant2.unique()))
    # Traverse each antenna
    for ant in antennas:
        print("Antenna ", ant)
        # Traverse IFS for given antenna
        gains_len[ant] = dict()
        gains_post[ant] = dict()
        for IF in range(n_IF):
            print("IF ", IF)
            gains_len[ant][IF] = dict()
            n_amp = len(set(df_fitted.query("(ant1 == @ant or ant2 == @ant) and IF == @IF").times_amp.values))
            n_phase = len(set(df_fitted.query("(ant1 == @ant or ant2 == @ant) and IF == @IF").times_phase.values))
            if n_amp == 0 and n_phase == 0:
                print("No amplitudes or phases - skipping")
                gains_len[ant][IF] = 0
                gains_post[ant][IF] = None
                continue
            gains_len[ant][IF]["amp"] = n_amp
            gains_len[ant][IF]["phase"] = n_phase
            gains_post[ant][IF] = dict()
            gains_post[ant][IF]["amp"] = dict()
            gains_post[ant][IF]["phase"] = dict()
            if with_mean_phase:
                print("Mean phase")
                gains_post[ant][IF]["mean_phase"] = samples[:, j+gains_len[ant][IF]["amp"]]
                h = 1
            else:
                h = 0
            for i, t in enumerate(df_fitted.query("(ant1 == @ant or ant2 == @ant) and IF == @IF").times_amp.unique()):
                print("timestamp ", i, ", time ", t, ", index j = ", j)
                gains_post[ant][IF]["amp"][t] = samples[:, j+i]
                # +1 mean skip ``mean_phase``
                if with_mean_phase:
                    if add_mean_phase:
                        gains_post[ant][IF]["phase"][t] = samples[:, j+gains_len[ant][IF]["amp"]+i+1] + gains_post[ant][IF]["mean_phase"]
                    else:
                        gains_post[ant][IF]["phase"][t] = samples[:, j+gains_len[ant][IF]["amp"]+i+1]
                else:
                    gains_post[ant][IF]["phase"][t] = samples[:, j+gains_len[ant][IF]["amp"]+i]
            print("Increasing j...")
            j += gains_len[ant][IF]["amp"] + gains_len[ant][IF]["phase"] + h

    # Find scans
    times = list()
    for ant in antennas:
        for IF in range(n_IF):
            try:
                times.extend(list(gains_post[ant][IF]['amp'].keys()))
            except TypeError:
                pass
    times = set(times)
    times = list(times)
    scans_list = scans(times)
    n_scans = len(scans_list)

    fig, axes = plt.subplots(len(gains_len), n_scans, sharex=True, sharey=False, figsize=(12, 20),
                             gridspec_kw={'wspace': 0.3, 'hspace': 0.2})
    for i, ant in enumerate(gains_post):
        print("Plotting antenna ", ant)
        # if ant == 6:
        #     break

        for IF in range(n_IF):
            print("Plotting IF ", IF)

            if gains_post[ant][IF] is None:
                print("Skipping ant={}, IF={}", format(str(ant), str(IF)))
                continue

            for scan_number, scan in enumerate(scans_list):
                print("Plotting scan ", scan_number)
                t_low, t_up = scan
                t0 = None
                for t in sorted(gains_post[ant][IF]["amp"].keys()):
                    # print("Plotting timestamp ", t)
                    if not t_low <= t <= t_up:
                        # print("Skipping because it is not inside ", [t_low, t_up])
                        continue
                    if t0 is None:
                        print("Scan beginning ", t)
                        t0 = t
                    print("Plotting timestamp ", t)
                    # Array with posterior for given t
                    amp = gains_post[ant][IF]["amp"][t]
                    print("Mean amplitude = ", np.median(amp))
                    ts = np.array([t-t0]*len(amp))
                    alpha = 6e-5*len(amp)
                    ts += np.random.normal(loc=0, scale=1.0, size=len(ts))
                    axes[i, scan_number].scatter(ts, amp, color="C{}".format(IF), alpha=alpha, s=2)
                    axes[i, scan_number].axhline(1.0, color="black", lw=0.5)

    fig.text(0.5, 0.04, 'scan time, s', ha='center')
    fig.text(0.04, 0.5, 'Gain amplitude', va='center', rotation='vertical')

    if plotfn:
        fig.savefig(plotfn+"_amplitudes.png", bbox_inches="tight", dpi=100)
    fig.show()


    fig, axes = plt.subplots(len(gains_len), n_scans, sharex=True, sharey=False, figsize=(12, 20),
                             gridspec_kw={'wspace': 0.3, 'hspace': 0.2})
    for i, ant in enumerate(gains_post):
        print("Plotting antenna ", ant)

        for scan_number, scan in enumerate(scans_list):
            t_low, t_up = scan
            t0 = None
            for t in sorted(gains_post[ant][plot_IF]["phase"].keys()):
                if not t_low <= t <= t_up:
                    continue
                if t0 is None:
                    t0 = t
                # Array with posterior for given t
                phase = gains_post[ant][plot_IF]["phase"][t]
                ts = np.array([t-t0]*len(phase))
                alpha = 6e-5*len(phase)
                ts += np.random.normal(loc=0, scale=1.0, size=len(ts))
                axes[i, scan_number].scatter(ts, phase, color="#1f77b4", alpha=alpha, s=2)
                axes[i, scan_number].axhline(0.0, color="C1", lw=0.5)

    fig.text(0.5, 0.04, 'scan time, s', ha='center')
    fig.text(0.04, 0.5, 'Gain phase, rad', va='center', rotation='vertical')

    if plotfn:
        fig.savefig(plotfn+"_phases.png", bbox_inches="tight", dpi=100)
    fig.show()

    return gains_post



def position_uncertainty(gains_posterior, df_fitted, lims=(-0.5, 0.5),
                         alpha_0=0.05):
    """
    Infer uncertainty of position from posterior of the gains phases.

    :param gains_posterior:
        Dictionary with keys - antenna numbers (as in ``gains_dict``), times,
        ``amp``, ``phase`` and values - samples of the posterior distribution
        for amplitude and phase at given time for given antenna. This is
        prepaired with bsc/data_utils.py.
    :param df_fitted:
        Dataframe fitted (created by ``create_data_file`` and ``inject_gains``).
    """
    from pycircstat.descriptive import cdiff
    from scipy.stats import scoreatpercentile
    import matplotlib.pyplot as plt
    import astropy.units as u
    rad_to_mas = u.rad.to(u.mas)
    fig, axes = plt.subplots(1, 1)
    dx = np.linspace(-10, 10, 100)
    for row in df_fitted.itertuples():
        print("processing visibility measurement...")
        t = row.times_phase
        ant1 = row.ant1
        ant2 = row.ant2
        u = row.u
        v = row.v
        phi_j_post = gains_posterior[ant1]["phase"][t]
        phi_k_post = gains_posterior[ant2]["phase"][t]
        delta_phi_jk_post = cdiff(phi_j_post, phi_k_post)
        low, med, up = scoreatpercentile(delta_phi_jk_post, [16, 50, 84])
        print(low, med, up)
        a = -u/v
        b_low = low*rad_to_mas/(2*np.pi*v)
        b_med = med*rad_to_mas/(2*np.pi*v)
        b_up = up*rad_to_mas/(2*np.pi*v)
        dy_low = a*dx + b_low
        dy_med = a*dx + b_med
        dy_up = a*dx + b_up
        # axes.plot(dx, dy_med, lw=2, alpha=0.003, color="#1f77b4")
        axes.fill_between(dx, dy_low, dy_up, alpha=alpha_0, color="#ff7f0e")

    axes.set_xlim(lims)
    axes.set_ylim(lims)
    axes.set_xlabel("dx, mas")
    axes.set_ylabel("dy, mas")
    fig.tight_layout()
    fig.show()
    return fig