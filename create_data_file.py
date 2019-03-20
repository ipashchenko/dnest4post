import numpy as np
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData


# Here we have file read by difmap with ``observe fn, dt, true`` and written out
# as ``wobs fn``
uvfits_fname = None
uvdata = UVData(uvfits_fname)


if uvdata._check_stokes_present("I"):
    # Masked array of I complex visibility
    vis = uvdata._choose_uvdata(stokes="I", freq_average=True)
    mask = vis.mask
    vis = vis.compressed()
    # Array of (N, stokes) errors
    error = uvdata.errors_from_weights_masked_freq_averaged.data
    # I = 0.5*(RR+LL)
    error = 0.5*np.hypot(error[:, 0], error[:, 1])
    error = error[~mask]
    vis_re = vis.real
    vis_im = vis.imag
    uv = uvdata.uv[~mask]
    u = uv[:, 0]
    v = uv[:, 1]

elif uvdata._check_stokes_present("RR"):
    # Masked array of RR complex visibility
    vis = uvdata._choose_uvdata(stokes="RR", freq_average=True)
    mask = vis.mask
    vis = vis.compressed()
    # Array of (N, stokes) errors
    error = uvdata.errors_from_weights_masked_freq_averaged.data
    error = error[~mask]
    vis_re = vis.real
    vis_im = vis.imag
    uv = uvdata.uv[~mask]
    u = uv[:, 0]
    v = uv[:, 1]

elif uvdata._check_stokes_present("LL"):
    # Masked array of RR complex visibility
    vis = uvdata._choose_uvdata(stokes="LL", freq_average=True)
    mask = vis.mask
    vis = vis.compressed()
    # Array of (N, stokes) errors
    error = uvdata.errors_from_weights_masked_freq_averaged.data
    error = error[~mask]
    vis_re = vis.real
    vis_im = vis.imag
    uv = uvdata.uv[~mask]
    u = uv[:, 0]
    v = uv[:, 1]

else:
    raise Exception("No I, RR or LL Stokes in data!")


data = np.vstack((u, v, vis_re, vis_im, error)).T
np.savetxt("uv_data.txt", data)