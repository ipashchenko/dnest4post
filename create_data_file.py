import numpy as np
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData
from model import Model
from components import CGComponent


# Here we have file read by difmap with ``observe fn, dt, true`` and written out
# as ``wobs fn``
uvfits_fname = "/home/ilya/github/DNest4/code/Examples/UV/J0000-3221_S_2017_01_16_pet_vis.fits"
uvdata = UVData(uvfits_fname)
# Array of (N, stokes) errors
# error = uvdata.errors_from_weights_masked_freq_averaged.data
error = uvdata.error(use_V=False, average_freq=True)
model = Model(stokes="RR")
cg1 = CGComponent(1.0, 0, 0, 0.1)
cg2 = CGComponent(0.75, 2, 2, 0.3)
cg3 = CGComponent(0.5, 5, 4, 0.5)
cg4 = CGComponent(0.25, 7, 7, 0.75)
cg5 = CGComponent(0.125, 10, 11, 0.75)
# cg3 = CGComponent(0.5, 2, 2, 0.25)
# cg4 = CGComponent(0.25, 3.0, 3.0, 0.25)
# cg5 = CGComponent(0.25, 5.0, 5.0, 1.0)
model.add_components(cg1, cg2, cg3, cg4, cg5)
noise = uvdata.noise(average_freq=False, use_V=False)
uvdata.substitute([model])
uvdata.noise_add(noise)


if uvdata._check_stokes_present("I"):
    print("I")
    # Masked array of I complex visibility
    vis = uvdata._choose_uvdata(stokes="I", freq_average=True)
    mask = vis.mask
    vis = vis.compressed()
    # I = 0.5*(RR+LL)
    error = 0.5*np.hypot(error[:, 0], error[:, 1])
    error = error[~mask]
    vis_re = vis.real
    vis_im = vis.imag
    uv = uvdata.uv[~mask]
    u = uv[:, 0]
    v = uv[:, 1]

elif uvdata._check_stokes_present("RR"):
    print("RR")
    # Masked array of RR complex visibility
    vis = uvdata._choose_uvdata(stokes="RR", freq_average=True)
    mask = vis.mask
    vis = vis.compressed()
    error = error[:, 0][~mask]
    vis_re = vis.real
    vis_im = vis.imag
    uv = uvdata.uv[~mask]
    u = uv[:, 0]
    v = uv[:, 1]

elif uvdata._check_stokes_present("LL"):
    print("LL")
    # Masked array of LL complex visibility
    vis = uvdata._choose_uvdata(stokes="LL", freq_average=True)
    mask = vis.mask
    vis = vis.compressed()
    error = error[:, 0][~mask]
    vis_re = vis.real
    vis_im = vis.imag
    uv = uvdata.uv[~mask]
    u = uv[:, 0]
    v = uv[:, 1]

else:
    raise Exception("No I, RR or LL Stokes in data!")


data = np.vstack((u, v, vis_re, vis_im, error)).T
np.savetxt("uv_data_art.txt", data)