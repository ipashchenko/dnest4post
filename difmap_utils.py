import numpy as np
from astropy import units as u

degree_to_rad = u.deg.to(u.rad)


def export_difmap_model(comps, out_fname, freq_hz):
    """
    :param comps:
        Iterable of tuples with (flux, x, y, bmaj).
    :param out_fname:
        Path for saving file.
    """
    with open(out_fname, "w") as fo:
        fo.write("! Flux (Jy) Radius (mas)  Theta (deg)  Major (mas)  Axial ratio   Phi (deg) T\n\
! Freq (Hz)     SpecIndex\n")
        for comp in comps:
            if len(comp) == 4:
                # Jy, mas, mas, mas
                flux, x, y, bmaj = comp
                e = "1.00000"
                bpa = "000.000"
                type = "1"
                bmaj = "{:.7f}v".format(bmaj)
            elif len(comp) == 6:
                # Jy, mas, mas, mas, -, deg
                flux, x, y, bmaj, e, bpa = comp
                e = "{}v".format(e)
                bpa = "{}v".format((bpa-np.pi/2)/degree_to_rad)
                bmaj = "{}v".format(bmaj)
                type = "1"
            elif len(comp) == 3:
                flux, x, y = comp
                e = "1.00000"
                bmaj = "0.0000"
                bpa = "000.000"
                type = "0"
            else:
                raise Exception
            # mas
            r = np.hypot(x, y)
            # rad
            theta = np.arctan2(x, y)
            theta /= degree_to_rad
            fo.write("{:>11.7f}v {:>13.7f}v {:>13.5f}v {:>13} {:>13} {:>13} {:>3} {:>12.5e} {:>12d}\n".format(flux, r, theta,
                                                              bmaj, e, bpa, type,
                                                             freq_hz, 0))


def convert_sample_to_difmap_model(sample, out_fname, freq_ghz, type="cg"):
    if type == "cg":
        comp_length = 4
    elif type == "eg":
        comp_length = 6
    n_comps = int(len(sample)/comp_length)
    components = list()
    for i in range(n_comps):
        subsample = sample[comp_length*i:comp_length*i+comp_length]
        print(subsample)
        flux = subsample[2]
        bmaj = np.exp(subsample[3])
        x = subsample[0]
        y = subsample[1]
        if type == "eg":
            e = subsample[4]
            bpa = subsample[5]
            comp = (flux, x, y, bmaj, e, bpa)
        elif type == "cg":
            comp = (flux, x, y, bmaj)
        components.append(comp)
    components = sorted(components, key=lambda comp: comp[0], reverse=True)
    export_difmap_model(components, out_fname, 1e9*freq_ghz)
    return components
