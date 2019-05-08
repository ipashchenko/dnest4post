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
