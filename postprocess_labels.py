import math
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse
from scipy import signal



def get_r(sample, comp_length):
    """
    Get distances of components from the phase center.

    :param sample:
        Iterable of component properties, i.e. [x1, x2, F1, size1, x2, y2, F2,
        size2, ...].
    :param comp_length:
        Number of component parameters (4 for circular Gaussian).
    :return:
        List of component distances.
    """
    length = len(sample)
    return [np.hypot(sample[i*comp_length+0], sample[i*comp_length+1]) for i in
            range(int(length/comp_length))]


def get_x(sample, comp_length):
    """
    Get x-coordinate of components from the phase center.

    :param sample:
        Iterable of component properties, i.e. [x1, x2, F1, size1, x2, y2, F2,
        size2, ...].
    :param comp_length:
        Number of component parameters (4 for circular Gaussian).
    :return:
        List of component x-coordinates.
    """
    length = len(sample)
    return [sample[i*comp_length+0] for i in range(int(length/comp_length))]


def get_y(sample, comp_length):
    """
    Get y-coordinate of components from the phase center.

    :param sample:
        Iterable of component properties, i.e. [x1, x2, F1, size1, x2, y2, F2,
        size2, ...].
    :param comp_length:
        Number of component parameters (4 for circular Gaussian).
    :return:
        List of component y-coordinates.
    """
    length = len(sample)
    return [sample[i*comp_length+1] for i in range(int(length/comp_length))]


def get_F(sample, comp_length):
    """
    Get fluxes of components.

    :param sample:
        Iterable of component properties, i.e. [x1, x2, F1, size1, x2, y2, F2,
        size2, ...].
    :param comp_length:
        Number of component parameters (4 for circular Gaussian).
    :return:
        List of component fluxes.
    """
    length = len(sample)
    return [(sample[i*comp_length+2]) for i in range(int(length/comp_length))]


def get_size(sample, comp_length):
    """
    Get sizes of components.

    :param sample:
        Iterable of component properties, i.e. [x1, x2, F1, size1, x2, y2, F2,
        size2, ...].
    :param comp_length:
        Number of component parameters (4 for circular Gaussian).
    :return:
        List of component sizes.
    """
    length = len(sample)
    return [(sample[i*comp_length+3]) for i in range(int(length/comp_length))]


def sort_sample_by_r(sample, comp_length=4):
    """
    Sort sample such that component with lowest distance from phase centet goes
    first.


    :param sample:
        Iterable of component properties, i.e. [x1, x2, F1, size1, x2, y2, F2,
        size2, ...].
    :param comp_length:
        Number of component parameters (4 for circular Gaussian).
    :return:
        Array with sorted sample.
    """
    r = get_r(sample, comp_length)
    indices = np.argsort(r)
    # Construct re-labelled sample
    return np.hstack([sample[i*comp_length: (i+1)*comp_length] for i in
                      indices])


def sort_sample_by_DEC(sample, comp_length=4, inverse=False):
    dec = get_x(sample, comp_length)
    indices = np.argsort(dec)
    if inverse:
        indices = indices[::-1]
    # Construct re-labelled sample
    return np.hstack([sample[i*comp_length: (i+1)*comp_length] for i in
                      indices])


def sort_sample_by_RA(sample, comp_length=4):
    ra = get_y(sample, comp_length)
    indices = np.argsort(ra)
    # Construct re-labelled sample
    return np.hstack([sample[i*comp_length: (i+1)*comp_length] for i in
                      indices])


def sort_sample_by_F(sample, comp_length=4):
    """
    Sort sample such that component with highest flux goes first.

    :param sample:
        Iterable of component properties, i.e. [x1, x2, F1, size1, x2, y2, F2,
        size2, ...].
    :param comp_length:
        Number of component parameters (4 for circular Gaussian).
    :return:
        Array with sorted sample.
    """
    F = get_F(sample, comp_length)
    indices = np.argsort(F)[::-1]
    # Construct re-labelled sample
    return np.hstack([sample[i*comp_length: (i+1)*comp_length] for i in
                      indices])


def sort_samples_by_r(samples, comp_length=4):
    """
    Sort each sample by distance from phase center..
    """
    new_samples = list()
    for sample in samples:
        new_samples.append(sort_sample_by_r(sample, comp_length))
    return np.atleast_2d(new_samples)


def sort_samples_by_dec(samples, comp_length=4, inverse=False):
    """
    Sort each sample by DEC.
    """
    new_samples = list()
    for sample in samples:
        new_samples.append(sort_sample_by_DEC(sample, comp_length, inverse=inverse))
    return np.atleast_2d(new_samples)


def sort_samples_by_ra(samples, comp_length=4):
    """
    Sort each sample by RA.
    """
    new_samples = list()
    for sample in samples:
        new_samples.append(sort_sample_by_RA(sample, comp_length))
    return np.atleast_2d(new_samples)


def sort_samples_by_F(samples, comp_length=4):
    """
    Sort each sample by flux.
    """
    new_samples = list()
    for sample in samples:
        new_samples.append(sort_sample_by_F(sample, comp_length))
    return np.atleast_2d(new_samples)


def cluster_by_flux_size(samples, comp_length=4):
    cluster_components = dict()
    n_clusters = int(len(samples[0]) / comp_length)
    xs = list()
    ys = list()
    fluxes = list()
    sizes = list()
    for sample in samples:
        xs.extend(get_x(sample, comp_length))
        ys.extend(get_y(sample, comp_length))
        fluxes.extend(get_F(sample, comp_length))
        sizes.extend(get_size(sample, comp_length))

    components = np.vstack((xs, ys, fluxes, sizes)).T

    X = np.vstack((fluxes, sizes)).T

    from sklearn.cluster import SpectralClustering, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    # clustering = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors",
    #                                 n_neighbors=10,
    #                                 assign_labels="discretize", random_state=0)
    # clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="single")
    clustering = GaussianMixture(n_components=n_clusters, n_init=20, reg_covar=1e-10, tol=1e-5, max_iter=1000)
    y_pred = clustering.fit_predict(X)
    for i in range(n_clusters):
        idx = y_pred == i
        cluster_components[i] = components[idx]

    return cluster_components


def find_component_xy_location_covariance(xy, type="mincovdet"):
    if type == "mincovdet":
        robust_cov = MinCovDet().fit(xy)
        location = robust_cov.location_
        covariance = robust_cov.covariance_
    elif type == "gmm":
        gmm = GMM()
        gmm.fit(xy)
        location = gmm.means_[0]
        covariance = gmm.covariances_[0]
    else:
        raise Exception("type must be mincovdet or gmm!")
    return location, covariance


def find_ellipse_angle(cov):
    """
    :param cov:
        2x2 covariance.
    """
    v, w = np.linalg.eigh(cov[:2, :2])
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    return 180 * angle / np.pi


def make_ellipses(cov, location, ax):
    """
    Add ellipses representing components of Gaussian Mixture Model.

    :param cov:
        2x2 covariance.
    :param location:
        2-element location.
    :param ax:
        Matplotlib axes object.
    """
    v, w = np.linalg.eigh(cov[:2, :2])
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    print("before sqrt - {}".format(v))
    v = np.sqrt(v)
    print("after sqrt - {}".format(v))
    ell = Ellipse(location[:2], v[0], v[1], 180 + angle)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)


def cg_flux_to_amplitude(flux, size):
    return 2*np.sqrt(np.log(2))*flux/(np.sqrt(np.pi)*size)


def cg_image(flux, size_pixels, center_pixel):
    """
    Function for plotting circular Gaussian components.

    :param flux:
        Flux of the component.
    :param size_pixels:
        Size of the component in pixels.
    :param center_pixel:
        Tuple of two coordinates of the center in pixels.

    :return:
        Function that takes two 2D arrays of image pixel coordinates and returns
        image of the Gaussian with the shape as coordinate arrays.

    >>> import matplotlib.pyplot as plt
    >>> xx, yy = np.meshgrid(np.arange(512), np.arange(512))
    >>> gauss = cg_image(10., 10., (100, 100))
    >>> plt.matshow(gauss(xx, yy))
    """
    amplitude = cg_flux_to_amplitude(flux, size_pixels)
    return lambda x, y:  amplitude*np.exp(-4*np.log(2)*np.hypot(x-center_pixel[0], y-center_pixel[1])**2/size_pixels**2)


def plot_posterior_image(imsize, pixsize, samples=None, samples_file=None,
                         size=10, comp_length=4, jitter_first=True,
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

    length = len(samples[0])
    image = np.zeros(imsize)
    xx, yy = np.meshgrid(np.arange(imsize[0]), np.arange(imsize[1]))
    # random ``size`` samples
    indx = np.random.randint(0, len(samples), size)

    for i in indx:
        sample = samples[i]

        gaussians = list()
        n_gaussians = int(length / comp_length)

        for n in range(n_gaussians):
            x = sample[n * comp_length + 0]
            x_pixels = int(x/pixsize) + imsize[0]/2
            y = sample[n * comp_length + 1]
            y_pixels = int(y/pixsize) + imsize[1]/2
            flux = np.exp(sample[n * comp_length + 2])
            size = np.exp(sample[n * comp_length + 3])
            print(x_pixels, y_pixels, flux, size/pixsize)
            gaussians.append(cg_image(flux, size/pixsize, (x_pixels, y_pixels,)))

        for cg in gaussians:
            image += cg(xx, yy)

    return xx, yy, image/size


# FIXME: Seems that size of beam here is not the same as in difmap.
def gaussian_beam(size_x, bmaj, bmin, bpa, size_y=None):
    """
    Generate and return a 2D Gaussian function
    of dimensions (size_x,size_y).

    See Briggs PhD (Appendix B) for details.

    :param size_x:
        Size of first dimension [pixels].
    :param bmaj:
        Beam major axis size [pixels].
    :param bmin:
        Beam minor axis size [pixels].
    :param bpa:
        Beam positional angle [deg].
    :param size_y: (optional)
        Size of second dimension. Default is ``size_x``.
    :return:
        Numpy array of shape (``size_x``, ``size_y``,).
    """
    size_y = size_y or size_x
    x, y = np.mgrid[-size_x: size_x, -size_y: size_y]
    # Constructing parameters of gaussian from ``bmaj``, ``bmin``, ``bpa``.
    a0 = 1. / (0.5 * bmaj) ** 2.
    c0 = 1. / (0.5 * bmin) ** 2.
    # This brings PA to VLBI-convention (- = from North counter-clockwise)
    bpa = -bpa
    theta = math.pi * (bpa + 90.) / 180.
    a = math.log(2) * (a0 * math.cos(theta) ** 2. +
                       c0 * math.sin(theta) ** 2.)
    b = (-(c0 - a0) * math.sin(2. * theta)) * math.log(2.)
    c = math.log(2) * (a0 * math.sin(theta) ** 2. +
                       c0 * math.cos(theta) ** 2.)

    g = np.exp(-a * x ** 2. - b * x * y - c * y ** 2.)
    # FIXME: It is already normalized?
    # return g/g.sum()
    return g


def convolve_image_with_beam(image, bmaj_pixels, bmin_pixels, bpa):
    beam_image = gaussian_beam(image.shape[0], bmaj_pixels, bmin_pixels, bpa)
    return signal.fftconvolve(image, beam_image, mode='same')