import numpy as np
# This postprocess samples of fixed number of components + gains


def get_r(sample, comp_length, n_comp, jitter_first=True):
    j = 0
    if jitter_first:
        j += 1
    return [np.hypot(sample[i*comp_length+j], sample[i*comp_length+j+1]) for i in
            range(n_comp)]


def get_F(sample, comp_length, n_comp, jitter_first=True):
    j = 2
    if jitter_first:
        j += 1
    return [(sample[i*comp_length+j]) for i in range(n_comp)]


def sort_sample_by_r(sample, n_comp, comp_length=4, jitter_first=True):
    r = get_r(sample, comp_length, n_comp, jitter_first)
    indices = np.argsort(r)
    # Construct re-labelled sample
    j = 0
    if jitter_first:
        j += 1
    return np.hstack([sample[j+i*comp_length: j+(i+1)*comp_length] for i in
                      indices])


def sort_sample_by_F(sample, n_comp, comp_length=4, jitter_first=True):
    F = get_F(sample, comp_length, n_comp, jitter_first)
    indices = np.argsort(F)[::-1]
    # Construct re-labelled sample
    j = 0
    if jitter_first:
        j += 1
    return np.hstack([sample[j+i*comp_length: j+(i+1)*comp_length] for i in
                      indices])


def sort_samples_by_r(samples, n_comp, comp_length=4, jitter_first=True):
    new_samples = list()
    for sample in samples:
        sorted_sample = sort_sample_by_r(sample, n_comp, comp_length, jitter_first)
        if jitter_first:
            sorted_sample = np.append(sorted_sample[::-1], sample[0])[::-1]
        new_samples.append(sorted_sample)
    return np.atleast_2d(new_samples)


def sort_samples_by_F(samples, n_comp, comp_length=4, jitter_first=True):
    new_samples = list()
    for sample in samples:
        sorted_sample = sort_sample_by_F(sample, n_comp, comp_length, jitter_first)
        if jitter_first:
            sorted_sample = np.append(sorted_sample[::-1], sample[0])[::-1]
        new_samples.append(sorted_sample)
    return np.atleast_2d(new_samples)



def process_sampled_gains(posterior_sample, df_fitted, jitter_first=True, n_comp=1, plotfn=None,
                          with_mean_phase=True, add_mean_phase=False):
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
        if with_mean_phase:
            gains_post[ant]["mean_phase"] = dict()
            gains_post[ant]["mean_phase"] = samples[:, j+gains_len[ant]["amp"]]
            h = 1
        else:
            h = 0
        for i, t in enumerate(df_fitted.query("ant1 == @ant or ant2 == @ant").times_amp.unique()):
            gains_post[ant]["amp"][t] = samples[:, j+i]
            # +1 mean skip ``mean_phase``
            if with_mean_phase:
                if add_mean_phase:
                    gains_post[ant]["phase"][t] = samples[:, j+gains_len[ant]["amp"]+i+1] + gains_post[ant]["mean_phase"]
                else:
                    gains_post[ant]["phase"][t] = samples[:, j+gains_len[ant]["amp"]+i+1]
            else:
                gains_post[ant]["phase"][t] = samples[:, j+gains_len[ant]["amp"]+i]


        j += gains_len[ant]["amp"] + gains_len[ant]["phase"] + h

    fig, axes = plt.subplots(len(gains_len), 2, sharex=True, figsize=(8, 20))
    for i, ant in enumerate(gains_post):
        print("Plotting antenna ", ant)
        for t in gains_post[ant]["amp"].keys():
            # Array with posterior for given t
            amp = gains_post[ant]["amp"][t]
            ts = np.array([t]*len(amp))
            alpha = 1e-5*len(amp)
            ts += np.random.normal(loc=0, scale=0.05, size=len(ts))
            axes[i, 0].scatter(ts, amp, color="#1f77b4", alpha=alpha)
        for t in gains_post[ant]["phase"].keys():
            phase = gains_post[ant]["phase"][t]
            ts = np.array([t]*len(phase))
            alpha = 1e-5*len(phase)
            ts += np.random.normal(loc=0, scale=0.05, size=len(ts))
            axes[i, 1].scatter(ts, phase, color="#1f77b4", alpha=alpha)
        axes[i, 1].yaxis.set_ticks_position("right")
    axes[0, 0].set_title("Amplitudes")
    axes[0, 1].set_title("Phases")
    axes[i, 0].set_xlabel("time, s")
    axes[i, 1].set_xlabel("time, s")
    if plotfn:
        fig.savefig(plotfn, bbox_inches="tight", dpi=100)
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
        delta_phi_jk_post = phi_j_post - phi_k_post
        low, med, up = scoreatpercentile(delta_phi_jk_post, [2.5, 50, 97.5])
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