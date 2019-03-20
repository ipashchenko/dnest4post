import numpy as np


def get_r(sample, comp_length):
    length = len(sample)
    return [np.hypot(sample[i*comp_length+0], sample[i*comp_length+1]) for i in
            range(int(length/comp_length))]


def get_F(sample, comp_length):
    length = len(sample)
    return [(sample[i*comp_length+2]) for i in range(int(length/comp_length))]


def sort_sample_by_r(sample, comp_length=4):
    r = get_r(sample, comp_length)
    indices = np.argsort(r)
    # Construct re-labelled sample
    return np.hstack([sample[i*comp_length: (i+1)*comp_length] for i in
                      indices])


def sort_sample_by_F(sample, comp_length=4):
    F = get_F(sample, comp_length)
    indices = np.argsort(F)[::-1]
    # Construct re-labelled sample
    return np.hstack([sample[i*comp_length: (i+1)*comp_length] for i in
                      indices])


def sort_samples_by_r(samples, comp_length=4):
    new_samples = list()
    for sample in samples:
        new_samples.append(sort_sample_by_r(sample, comp_length))
    return np.atleast_2d(new_samples)


def sort_samples_by_F(samples, comp_length=4):
    new_samples = list()
    for sample in samples:
        new_samples.append(sort_sample_by_F(sample, comp_length))
    return np.atleast_2d(new_samples)
