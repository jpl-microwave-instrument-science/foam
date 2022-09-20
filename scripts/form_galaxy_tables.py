import os 
import pickle
import numpy as np 
from numba import njit, prange 
import h5py 
import spiceypy as spice 
import scipy.interpolate as spi 
import foam.dielectric as dielectric
from foam.utils.config import cache_path
from foam.utils.mk import manual_furnish

try: 
    from tqdm.auto import tqdm
    lrange = tqdm
except ModuleNotFoundError: 
    lrange = lambda x: x 

import time

""" This script generates an estimate of galactic reflection for 
    a wind-roughened ocean surface.
    These routines use numba compilation and parallelization
    to increase speed
"""

# Defining some functions first
# Override SPICE functions 


@njit
def latrec(radius, lon, lat): 
    x = radius * np.cos(lon) * np.cos(lat)
    y = radius * np.sin(lon) * np.cos(lat)
    z = radius * np.sin(lat)
    return np.array([x, y, z])


@njit
def reclat(rectan): 
    d__1 = abs(rectan[0])
    d__2 = abs(rectan[1])
    d__1 = max(d__1, d__2) 
    d__2 = abs(rectan[2])
    big = max(d__1, d__2)
    if (big > 0.):
        x = rectan[0] / big
        y = rectan[1] / big
        z__ = rectan[2] / big
        radius = big * np.sqrt(x * x + y * y + z__ * z__)
        lat = np.arctan2(z__, np.sqrt(x * x + y * y))
        x = rectan[0]
        y = rectan[1]
        if (x == 0. and y == 0.):
            lon = 0.
        else:
            lon = np.arctan2(y, x)
    else:
        radius = 0.
        lat = 0.
        lon = 0.
    return np.array([radius, lon, lat])


@njit
def vsep(v1, v2): 
    dmag1 = np.linalg.norm(v1)
    u1 = v1 / dmag1
    if (dmag1 == 0.):
        ret_val = 0.
        return ret_val
    
    dmag2 = np.linalg.norm(v2)
    u2 = v2 / dmag2
    if (dmag2 == 0.):
        ret_val = 0.
        return ret_val

    if (np.dot(u1, u2) > 0.):
        vtemp = u1 - u2
        ret_val = np.arcsin(np.linalg.norm(vtemp) * .5) * 2.
    elif (np.dot(u1, u2) < 0.):
        vtemp = u1 + u2
        ret_val = np.pi - np.arcsin(np.linalg.norm(vtemp) * .5) * 2.
    else:
        ret_val = np.pi / 2.
    return ret_val


@njit
def vrotv(v, axis, theta):
    x = axis / np.linalg.norm(axis)
    p = np.dot(v, x) * x
    v1 = v - p 
    v2 = np.cross(x, v1)
    c = np.cos(theta)
    s = np.sin(theta)
    rplane = c * v1 + s * v2
    r = rplane + p
    return r

# Other functions 


@njit
def local_rotation(lat, lon): 
    # Determines rotation matrix from terrestrial to local reference frame
    # Per documentation in SPICE frames required reading 
    # x is north, y is west, z is up 
    colat = np.pi / 2 - lat
    # m1 = spice.rotate(np.pi, 3)
    m1 = np.array([[-1., 0., 0.], 
                   [0., -1., 0.], 
                   [0., 0., 1.]])
    # m2 = spice.rotate(colat, 2)
    m2 = np.array([[np.cos(colat), 0., -np.sin(colat)], 
                   [0., 1., 0.], 
                   [np.sin(colat), 0., np.cos(colat)]])
    # m3 = spice.rotate(lon, 3)
    m3 = np.array([[np.cos(lon), np.sin(lon), 0.], 
                   [-np.sin(lon), np.cos(lon), 0.],
                   [0., 0., 1.]])
    return m1 @ m2 @ m3


@njit
def tilt_rotation(Sx, Sy):
    # Used to determine local to facet local reference frame transformation
    # See Yueh et al. 1997
    denom = np.sqrt(Sx**2 + Sy**2 + 1)
    if denom < 1: denom = 1
    theta = np.arccos(1 / denom)
    if np.sin(theta) == 0: 
        arg = -Sx / denom / np.cos(theta)
    else: 
        arg = -Sy / denom / np.sin(theta)
        
    if abs(arg) >= 1: arg = 0.999 * np.sign(arg)
    phi = np.arcsin(arg)
    beta = np.arctan(np.tan(theta) * np.cos(phi))
    sineconst = np.sin(theta) * np.sin(phi)
    A = np.zeros((3, 3))
    # Xl 
    A[0, 0] = np.cos(beta) 
    A[0, 1] = 0. 
    A[0, 2] = -np.sin(beta)
    # Yl
    A[1, 0] = -sineconst * np.sin(beta)
    A[1, 1] = np.sqrt(1. - sineconst**2)
    A[1, 2] = -sineconst * np.cos(beta)
    # Zl 
    A[2, 0] = np.sin(theta) * np.cos(phi)
    A[2, 1] = np.sin(theta) * np.sin(phi)
    A[2, 2] = np.cos(theta)

    return A 


@njit 
def cm_variance(w, f): 
    # Adjusted Cox/Munk variance 
    variance = 2.9e-3 * (w + 2.) * np.log10(2. * f / 1e3) 
    return variance


@njit(fastmath=True)
def inner_inner_loop(ra, dec, ks, if2topo, hpols, vpols, variance): 
    # Inner inner loop 
    ret_v = 0
    ret_h = 0
    for rrdex in range(len(ra)): 
        rr = ra[rrdex]
        for dddex in range(len(dec)): 
            dd = dec[dddex]
            # ki is the incident direction vector
            # or the vector in the direction of the origin of 
            # galactic radiation
            ki = latrec(1, rr, dd)
            ki = if2topo @ ki
            ang_sep = vsep(ks, ki)
            if abs(ang_sep) < np.radians(89): 
                # slope = np.tan(ang_sep / 2)
                Rh = (np.cos(ang_sep) - (eps - np.sin(ang_sep)**2)**0.5) / \
                     (np.cos(ang_sep) + (eps - np.sin(ang_sep)**2)**0.5)
                Rv = (eps * np.cos(ang_sep) - (eps - np.sin(ang_sep)**2)**0.5) / \
                     (eps * np.cos(ang_sep) + (eps - np.sin(ang_sep)**2)**0.5)
                # Polarization vectors for the incident radiation
                hpoli = np.cross(ki, np.array([0., 0., 1.]))
                if np.linalg.norm(hpoli) == 0: 
                    hpoli = np.array([0., -1., 0])
                hpoli = hpoli / np.linalg.norm(hpoli)
                vpoli = np.cross(hpoli, ks)
                vpoli = vpoli / np.linalg.norm(vpoli)
                # Getting normal vector of facet and facet slopes 
                norm = vrotv(ks, hpoli, ang_sep / 2)
                norm = norm / np.linalg.norm(norm)
                denom = np.sqrt(norm[0]**2 + norm[1]**2 + 1) 
                sx = -norm[0] * denom
                sy = -norm[1] * denom

                # Polarization mixing isn't used for nadir reflection, this will need to change
                A = tilt_rotation(sx, sy)  # Topocentric to facet local frame 
                        
                # Local polarization vectors 
                hpoli_l = A @ hpoli 
                vpoli_l = A @ vpoli 
                hpols_l = A @ hpols 
                vpols_l = A @ vpols 

                z = np.array([0., 0., 1.])
                kiz = np.dot(z, ki)
                ksz = np.dot(z, ks)
                nz = np.dot(norm, z)
                slope = np.sqrt(sx**2 + sy**2)
                pz = 1 / (np.pi * variance) * np.exp(-slope**2 / variance)
                R2v = abs(Rv)**2 * (np.dot(hpoli_l, vpols_l)**2 + np.dot(vpoli_l, vpols_l)**2)
                R2h = abs(Rh)**2 * (np.dot(hpoli_l, hpols_l)**2 + np.dot(vpoli_l, hpols_l)**2)
                sfv = pz * R2v / (4 * kiz * ksz * nz**4)
                sfh = pz * R2h / (4 * kiz * ksz * nz**4)
                ret_v += sfv * galaxy[rrdex, dddex] * dra * ddec
                ret_h += sfh * galaxy[rrdex, dddex] * dra * ddec

    return ret_v, ret_h


# @njit(fastmath=True)
def inner_loop(ra, dec, variance, shape): 
    out_v = np.zeros(shape)
    out_h = np.zeros(shape)
    for rdex in lrange(range(len(ra))):
        r = ra[rdex] 
        for ddex in lrange(range(len(dec))):
            d = dec[ddex]
            # ks is the scattered direction vector
            # or the vector pointed towards the radiometer
            ks = latrec(1, r, d)
            ks = if2ef @ ks   # Swap to Earth centric
            ks_coord = reclat(ks)
            ef2topo = local_rotation(ks_coord[2], ks_coord[1])
            if2topo = ef2topo @ if2ef
            ks = ef2topo @ ks 

            # Scattered polarization vectors 
            hpols = np.cross(ks, np.array([0., 0., 1.]))
            if np.linalg.norm(hpols) == 0: 
                hpols = np.array([0., -1., 0])
            hpols = hpols / np.linalg.norm(hpols)
            vpols = np.cross(ks, hpols)
            vpols = vpols / np.linalg.norm(vpols)
            ret_v, ret_h = inner_inner_loop(ra, dec, ks, if2topo, hpols, vpols, variance)
            out_v[rdex, ddex] = ret_v * np.cos(d)
            out_h[rdex, ddex] = ret_h * np.cos(d)
    return out_v, out_h 


# @njit(fastmath=True)
def loop(galaxy, w, ra, dec, if2ef, eps, angle): 
    # Table shape: polarization, wind, ra, dec
    table = np.zeros((2, len(w) + 1, *np.shape(galaxy)), dtype=np.float32)

    # Zero wind case, Specular reflection 
    # R = (1 - eps**0.5) / (1 + eps**0.5)  # Nadir reflectivity
    Rh = (np.cos(angle) - (eps - np.sin(angle)**2)**0.5) / \
         (np.cos(angle) + (eps - np.sin(angle)**2)**0.5)
    Rv = (eps * np.cos(angle) - (eps - np.sin(angle)**2)**0.5) / \
         (eps * np.cos(angle) + (eps - np.sin(angle)**2)**0.5)

    table[0, 0, :, :] = abs(Rv)**2 * galaxy
    table[1, 0, :, :] = abs(Rh)**2 * galaxy

    for i in range(len(w)): 
        variance = cm_variance(w[i], f)
        out_v, out_h = inner_loop(ra, dec, variance, np.shape(galaxy))
        table[0, i + 1, :, :] = out_v 
        table[1, i + 1, :, :] = out_h 
    return table


# Un-jit-able stuff
file_path = os.path.join(cache_path, 'galaxy', 'TBSkyLbandAquarius.h5')
galaxy_file = h5py.File(file_path, 'r')
ra = galaxy_file['Right_Ascension'][:]
dec = galaxy_file['Declination'][:]
galaxy = galaxy_file['TB_no_Cas_A'][:]
galaxy = galaxy - 2.73  # Floor subtraction

# Decimate by an arbitrary factor to increase computation speed 
gal_interp = spi.RectBivariateSpline(ra, dec, galaxy)
factor = 4 
ra = ra[::factor]
dec = dec[::factor]
galaxy = gal_interp(ra, dec)

ra = np.radians(ra)
dec = np.radians(dec) 
ddec = np.gradient(dec)[0]
dra = np.gradient(ra)[0]
manual_furnish()
if2ef = spice.pxform('J2000', 'ITRF93', 0)

f = 1.41e3  # MHz
w = np.array([5, 10, 15, 20])  # m/s
t = 293.15  # K, 20 C
s = 35  # psu 
eps = dielectric.h2o_liquid_KleinSwift(f, t, s)
angle = 0.

t1 = time.perf_counter() 
table = loop(galaxy, w, ra, dec, if2ef, eps, angle)
t2 = time.perf_counter()
print('Elapsed time: {}'.format(t2 - t1))

# Stash tables for now 
pickle.dump(table, open('test_galtables.p', 'wb'))

ra = np.degrees(ra)
dec = np.degrees(dec) 

# Interpolate 
w = np.concatenate([np.array([0]), w])
gal_interpolator_v = spi.RegularGridInterpolator((w, ra, dec), table[0], method='linear', bounds_error=False, fill_value=None)
gal_interpolator_h = spi.RegularGridInterpolator((w, ra, dec), table[1], method='linear', bounds_error=False, fill_value=None)
gal_interpolator = [gal_interpolator_v, gal_interpolator_h]
pickle.dump(gal_interpolator, open('test_galaxy_interpolator.p', 'wb'))
