import numpy as np
# This postprocess samples of RJ


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


if __name__ == "__main__":
    n_true = 6

    # truths = [0, 0, np.log(2), np.log(0.1),
    #           -0.5, 0, np.log(1), np.log(0.2),
    #           -1.5, -1, np.log(0.5), np.log(0.3),
    #           -3.5, -2, np.log(0.25), np.log(0.5),
    #           -5, -5, np.log(0.125), np.log(0.5),
    #           -7.5, -8, np.log(0.075), np.log(0.5),
    #           -8, -9, np.log(0.05), np.log(0.75),
    #           -9, -9.5, np.log(0.035), np.log(0.75)]

    post_fn = "/home/ilya/github/dnest4post/data/posterior_sample.txt"
    from plotting import rj_plot_ncomponents_distribution
    rj_plot_ncomponents_distribution(posterior_file=post_fn)
    samples = np.loadtxt(post_fn)
    ns, counts = np.unique(samples[:, 7], return_counts=True)
    n_pmax = int(ns[np.argmax(counts)])
    # n_pmax = 2
    # assert n_pmax == n_true

    result = get_samples_for_each_n(samples)
    samples_n = result[n_pmax]
    np.savetxt("posterior_nbest_{}_1800_S_long.txt".format(n_pmax), samples_n)

    # from postprocess_labels_gains import sort_samples_by_r
    # samples_n = sort_samples_by_r(samples_n, n_pmax)
    # from plotting import plot_corner
    # fig = plot_corner(samples_n[:, 1:], "corner_{}_1800_S_r_larger_field.png".format(n_pmax))