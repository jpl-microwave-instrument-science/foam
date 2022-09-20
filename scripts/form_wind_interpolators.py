import numpy as np 
import pandas as pd 
import scipy.interpolate as spi
import pickle

# Forming frequency and wind-dependent interpolation tables

# 1 - Empirical expressions of Meissner and Wentz 
# These are for reference conditions of Ts = 20C and Theta = 55.2
# Salinity has a relatively small impact on isotropic and anisotropic
# contributions up to about 40 psu

# Meissner and Wentz 2012, 6.8-85.5 GHz
# Isotropic terms 
table = np.array([[ 4.96726E-05, -3.03363E-04,  5.60506E-05, -2.86408E-06,  4.88803E-08],
                  [ 3.85750E-03, -5.10844E-04,  4.89469E-05, -1.50552E-06,  1.20306E-08],
                  [-2.35464E-04, -2.76866E-04,  5.73583E-05, -2.94364E-06,  4.89421E-08],
                  [ 4.17650E-03, -6.20751E-04,  6.82607E-05, -2.47982E-06,  2.80155E-08],
                  [ 3.26502E-05, -3.65935E-04,  6.62807E-05, -3.40705E-06,  5.81231E-08],
                  [ 5.06330E-03, -7.41324E-04,  8.54446E-05, -3.28225E-06,  4.01950E-08],
                  [-7.03594E-04, -2.17673E-04,  4.00659E-05, -1.84769E-06,  2.76830E-08],
                  [ 5.63832E-03, -8.43744E-04,  1.06734E-04, -4.61253E-06,  6.67315E-08],
                  [-3.14175E-03,  4.06967E-04, -3.33273E-05,  1.26520E-06, -1.67503E-08],
                  [ 6.01311E-03, -7.00158E-04,  1.26075E-04, -7.27339E-06,  1.35737E-07]])

iterables = [[6.8, 10.7, 18.7, 37, 85.5], ['v', 'h']]
index = pd.MultiIndex.from_product(iterables)
iso_wind_coeffs = pd.DataFrame(table, index=index, columns=[1, 2, 3, 4, 5])

# Anisotropic terms 
## First order 
table = np.array([[ 4.46633E-07,  3.34314E-07,  3.12587E-06, -1.99336E-07,  3.55175E-09], 
                  [ 2.17314E-05, -1.54052E-06,  7.43743E-07, -3.32899E-08,  3.04367E-10],
                  [ 0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00],
                  [ 0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00],
                  [ 4.96132E-05, -2.90991E-05,  9.05913E-06, -5.73703E-07,  1.10332E-08],
                  [-2.20699E-05,  8.92180E-06,  4.69873E-08, -2.41047E-08,  5.71120E-10],
                  [-8.48737E-05,  5.35295E-05, -1.16605E-05,  6.83923E-07, -1.27622E-08],
                  [ 0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00],
                  [-4.88686E-05, -2.26779E-06,  9.94735E-06, -7.51560E-07,  1.55400E-08],
                  [ 3.95872E-05, -2.88339E-05,  6.61597E-06, -4.08181E-07,  7.87906E-09],
                  [-3.29350E-05,  4.32977E-05, -1.33822E-05,  8.75024E-07, -1.74093E-08],
                  [ 0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00],
                  [-2.41163E-04,  7.66737E-05,  3.65641E-06, -5.59326E-07,  1.35655E-08],
                  [-5.43465E-05,  2.24360E-05,  1.16736E-06, -1.58769E-07,  3.60149E-09],
                  [ 2.55925E-04, -1.02271E-04,  3.06653E-06,  6.84854E-08, -2.83830E-09],
                  [ 0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00]])

iterables = [[6.8, 10.7, 18.7, 37], ['v', 'h', 'S3', 'S4']]
index = pd.MultiIndex.from_product(iterables)
aniso_wind_coeffs_1 = pd.DataFrame(table, index=index, columns=[1, 2, 3, 4, 5])

## Second order 
table = np.array([[ 2.21863E-04, -1.18053E-04,  1.68718E-05, -8.94076E-07,  1.60273E-08],
                  [-3.50262E-06,  1.02052E-05, -5.28636E-06,  3.82864E-07, -7.87283E-09],
                  [ 0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00],
                  [ 0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00,  0.00000E+00],
                  [ 1.48213E-04, -7.15954E-05,  1.01992E-05, -5.41575E-07,  9.71451E-09],
                  [-8.09058E-05,  6.06930E-05, -1.42500E-05,  8.86313E-07, -1.69340E-08],
                  [-1.90531E-04,  1.09714E-04, -1.97712E-05,  1.10888E-06, -1.96980E-08],
                  [-9.49332E-05,  3.91201E-05, -1.64418E-06, -2.12315E-08,  1.47529E-09],
                  [ 1.21860E-04, -6.39714E-05,  9.34100E-06, -5.24394E-07,  9.97506E-09],
                  [ 2.65036E-04, -9.32568E-05,  1.41605E-06,  2.98507E-07, -9.64763E-09],
                  [ 1.66139E-04, -4.39714E-05, -5.42274E-06,  6.82097E-07, -1.69151E-08],
                  [-1.62337E-04,  7.13779E-05, -5.42054E-06,  1.26564E-07, -3.00476E-10],
                  [ 2.35250E-04, -1.24502E-04,  1.48805E-05, -7.07241E-07,  1.18776E-08],
                  [ 7.26916E-04, -2.84727E-04,  2.20935E-05, -5.68143E-07,  3.00983E-09],
                  [ 1.37851E-04, -1.58017E-05, -9.08052E-06,  9.03144E-07, -2.16700E-08],
                  [-1.33456E-04,  7.09317E-05, -8.67173E-06,  3.98910E-07, -6.31997E-09]])  

iterables = [[6.8, 10.7, 18.7, 37], ['v', 'h', 'S3', 'S4']]
index = pd.MultiIndex.from_product(iterables)
aniso_wind_coeffs_2 = pd.DataFrame(table, index=index, columns=[1, 2, 3, 4, 5])

# Meissner et al. 2014, 1.4 GHz, Isotropic and anisotropic terms 
# Third and Fourth Stokes are assumed zero
table = np.array([[ 5.7894060320e-001, -1.0473595790e-001,  9.9200140518e-003, -3.6291757411e-004,  4.6589912401e-006],
                  [ 2.1216534279e-002, -7.1813519762e-003,  9.1738225933e-004, -3.9535822500e-005,  5.6502242949e-007],
                  [ 6.6222103677e-002, -2.8388133825e-002,  3.7501350060e-003, -1.8446699362e-004,  3.1286830745e-006],
                  [ 7.7153019419e-001, -1.2715188465e-001,  1.1329089683e-002, -4.0789149957e-004,  5.2183370672e-006],
                  [ 3.3306056615e-003, -3.7204789320e-003,  6.2494297864e-004, -3.2678064089e-005,  5.3603866560e-007],
                  [-1.2192070798e-002,  8.8557322073e-003, -1.2734161054e-003,  5.1704466145e-005, -5.8171521631e-007],
                  [ 5.0281588143e-001, -8.4035755450e-002,  7.8518455026e-003, -2.8030380618e-004,  3.5109805876e-006],
                  [ 1.5596703582e-003,  2.3582984624e-003, -3.4662668911e-004,  2.6606348124e-005, -6.1842609629e-007],
                  [ 5.1196349082e-002, -1.8202980254e-002,  2.0749477038e-003, -8.6995183107e-005,  1.2436524520e-006],
                  [ 8.4965177001e-001, -1.2443856200e-001,  1.0359930394e-002, -3.5549109274e-004,  4.3636667626e-006],
                  [-1.8863269506e-002,  6.5676785996e-003, -7.3111819961e-004,  3.7255111733e-005, -6.8989289663e-007],
                  [-4.8938315835e-002,  2.6541378947e-002, -3.7836135549e-003,  1.9204055354e-004, -3.2620602022e-006],
                  [ 4.7027203005e-001, -7.6334662983e-002,  6.9857495929e-003, -2.4303426684e-004,  2.9390853226e-006],
                  [ 9.1197181127e-003, -3.0431623312e-003,  5.0839571367e-004, -2.0375986729e-005,  2.4580823525e-007],
                  [ 9.3408423686e-002, -3.3492931571e-002,  3.8025601997e-003, -1.6925890570e-004,  2.6396519557e-006],
                  [ 1.0601673642e+000, -1.4677107298e-001,  1.1480019211e-002, -3.8084012081e-004,  4.5485097434e-006],
                  [ 9.6160121528e-003, -4.3505334225e-003,  6.0718079191e-004, -2.7536464802e-005,  4.0733177632e-007],
                  [-5.1974877527e-003,  1.0855313411e-002, -1.8411735248e-003,  9.5714130699e-005, -1.6059448322e-006]])

iterables = [[29.36, 38.44, 46.29], ['v', 'h'], [0, 1, 2]]
index = pd.MultiIndex.from_product(iterables)
aq_wind_coeffs = pd.DataFrame(table / 290, index=index, columns=[1, 2, 3, 4, 5])

# Isotropic Interpolator parameters 
freqs = np.array([0, 1.4, 6.8, 10.7, 18.7, 37])
angles = np.array([0, 10, 20, 29.36, 38.44, 46.29, 55.2])
winds = np.linspace(0, 20, 100)

# Isotropic interpolator 
# Making the assumption that the change in isotropic emissivity decreases
# linearly with frequency with an intercept of 0, which may not be accurate 
# Wind speeds above 20 m/s are linearly extrapolated 
iso_v = np.zeros((len(freqs), len(angles), len(winds)))
iso_h = np.zeros((len(freqs), len(angles), len(winds)))

for f in range(len(freqs)): 
    for a in range(len(angles)): 
        delta_v = 0
        delta_h = 0
        if freqs[f] == 1.4:
            if angles[a] <= 29.36:  
                for i in range(1, 6): 
                    delta_v = delta_v + aq_wind_coeffs.loc[29.36, 'v', 0][i] * winds**i
                    delta_h = delta_h + aq_wind_coeffs.loc[29.36, 'h', 0][i] * winds**i
                if angles[a] < 29.36: 
                    nad = 0.5 * (delta_v + delta_h)
                    delta_v = nad + (delta_v - nad) * (angles[a] / 29.36)**4
                    delta_h = nad + (delta_h - nad) * (angles[a] / 29.36)**1.5
            elif angles[a] == 38.44:
                for i in range(1, 6): 
                    delta_v = delta_v + aq_wind_coeffs.loc[38.44, 'v', 0][i] * winds**i
                    delta_h = delta_h + aq_wind_coeffs.loc[38.44, 'h', 0][i] * winds**i
            elif angles[a] >= 46.29: 
                for i in range(1, 6): 
                    delta_v = delta_v + aq_wind_coeffs.loc[46.29, 'v', 0][i] * winds**i
                    delta_h = delta_h + aq_wind_coeffs.loc[46.29, 'h', 0][i] * winds**i
                if angles[a] > 46.29: 
                    nad = 0.5 * (delta_v + delta_h)
                    delta_v = nad + (delta_v - nad) * (angles[a] / 46.29)**4
                    delta_h = nad + (delta_h - nad) * (angles[a] / 46.29)**1.5

                    # Linear extrapolation for angles higher than 46.29
                    t_inter = spi.RegularGridInterpolator((angles[:-1], winds), iso_v[f, :-1, :], method='linear', bounds_error=False, fill_value=None)
                    delta_v = t_inter((angles[a], winds))
                    t_inter = spi.RegularGridInterpolator((angles[:-1], winds), iso_h[f, :-1, :], method='linear', bounds_error=False, fill_value=None)
                    delta_h = t_inter((angles[a], winds))

        elif freqs[f] != 0:
            for i in range(1, 6): 
                delta_v = delta_v + iso_wind_coeffs.loc[freqs[f], 'v'][i] * winds**i
                delta_h = delta_h + iso_wind_coeffs.loc[freqs[f], 'h'][i] * winds**i
            if angles[a] < 55.2: 
                nad = 0.5 * (delta_v + delta_h)
                delta_v = nad + (delta_v - nad) * (angles[a] / 55.2)**4
                delta_h = nad + (delta_h - nad) * (angles[a] / 55.2)**1.5

        iso_v[f, a, :] = delta_v
        iso_h[f, a, :] = delta_h

iso_v_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), iso_v, method='linear', bounds_error=False, fill_value=None)
iso_h_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), iso_h, method='linear', bounds_error=False, fill_value=None)

# Anisotropic Interpolator parameters 
freqs = np.array([0, 1.4, 6.8, 10.7, 18.7, 37])
angles = np.array([0, 10, 20, 29.36, 38.44, 46.29, 55.2])
winds = np.linspace(3, 20, 100)
winds = np.concatenate([np.zeros(1), winds])  # Interpolating from 0 to 3 m/s per MW2012

# Anisotropic interpolators
# First coefficient 
# Making the assumption that the change in isotropic emissivity decreases
# linearly with frequency with an intercept of 0, which may not be accurate 
# Wind speeds above 20 m/s are linearly extrapolated, which is not correct
aniso1_v = np.zeros((len(freqs), len(angles), len(winds)))
aniso1_h = np.zeros((len(freqs), len(angles), len(winds)))
aniso1_S3 = np.zeros((len(freqs), len(angles), len(winds)))
aniso1_S4 = np.zeros((len(freqs), len(angles), len(winds)))

for f in range(len(freqs)): 
    for a in range(len(angles)): 
        delta_v = 0
        delta_h = 0
        delta_s3 = 0
        delta_s4 = 0
        if freqs[f] == 1.4:
            if angles[a] <= 29.36:  
                for i in range(1, 6): 
                    delta_v = delta_v + aq_wind_coeffs.loc[29.36, 'v', 1][i] * winds**i
                    delta_h = delta_h + aq_wind_coeffs.loc[29.36, 'h', 1][i] * winds**i
                if angles[a] < 29.36: 
                    aniso1_s1 = (delta_v + delta_h) / 2
                    aniso1_s2 = delta_v - delta_h 
                    aniso1_s1 = aniso1_s1 * (angles[a] / 29.36)**2 
                    aniso1_s2 = aniso1_s2 * (angles[a] / 29.36)**1
                    delta_v = aniso1_s1 + aniso1_s2 / 2
                    delta_h = aniso1_s1 - aniso1_s2 / 2
            elif angles[a] == 38.44:
                for i in range(1, 6): 
                    delta_v = delta_v + aq_wind_coeffs.loc[38.44, 'v', 1][i] * winds**i
                    delta_h = delta_h + aq_wind_coeffs.loc[38.44, 'h', 1][i] * winds**i
            elif angles[a] >= 46.29: 
                for i in range(1, 6): 
                    delta_v = delta_v + aq_wind_coeffs.loc[46.29, 'v', 1][i] * winds**i
                    delta_h = delta_h + aq_wind_coeffs.loc[46.29, 'h', 1][i] * winds**i
                if angles[a] > 46.29: 
                    aniso1_s1 = (delta_v + delta_h) / 2
                    aniso1_s2 = delta_v - delta_h 
                    aniso1_s1 = aniso1_s1 * (angles[a] / 46.29)**2 
                    aniso1_s2 = aniso1_s2 * (angles[a] / 46.29)**1
                    delta_v = aniso1_s1 + aniso1_s2 / 2
                    delta_h = aniso1_s1 - aniso1_s2 / 2

                    # Linear extrapolation for angles higher than 46.29
                    t_inter = spi.RegularGridInterpolator((angles[:-1], winds), aniso1_v[f, :-1, :], method='linear', bounds_error=False, fill_value=None)
                    delta_v = t_inter((angles[a], winds))
                    t_inter = spi.RegularGridInterpolator((angles[:-1], winds), aniso1_h[f, :-1, :], method='linear', bounds_error=False, fill_value=None)
                    delta_h = t_inter((angles[a], winds))

        elif freqs[f] != 0:
            for i in range(1, 6): 
                delta_v = delta_v + aniso_wind_coeffs_1.loc[freqs[f], 'v'][i] * winds**i
                delta_h = delta_h + aniso_wind_coeffs_1.loc[freqs[f], 'h'][i] * winds**i
                delta_s3 = delta_s3 + aniso_wind_coeffs_1.loc[freqs[f], 'S3'][i] * winds**i
                delta_s4 = delta_s4 + aniso_wind_coeffs_1.loc[freqs[f], 'S4'][i] * winds**i
            if angles[a] < 55.2: 
                aniso1_s1 = (delta_v + delta_h) / 2
                aniso1_s2 = delta_v - delta_h
                aniso1_s3 = delta_s3
                aniso1_s4 = delta_s4
                aniso1_s1 = aniso1_s1 * (angles[a] / 55.2)**2. 
                aniso1_s2 = aniso1_s2 * (angles[a] / 55.2)**1.
                aniso1_s3 = aniso1_s3 * (angles[a] / 55.2)**1. 
                aniso1_s4 = aniso1_s4 * (angles[a] / 55.2)**2.
                delta_v = aniso1_s1 + aniso1_s2 / 2
                delta_h = aniso1_s1 - aniso1_s2 / 2
                delta_s3 = aniso1_s3
                delta_s4 = aniso1_s4

        aniso1_v[f, a, :] = delta_v
        aniso1_h[f, a, :] = delta_h
        aniso1_S3[f, a, :] = delta_s3
        aniso1_S4[f, a, :] = delta_s4

aniso1_v_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), aniso1_v, method='linear', bounds_error=False, fill_value=None)
aniso1_h_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), aniso1_h, method='linear', bounds_error=False, fill_value=None)
aniso1_S3_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), aniso1_S3, method='linear', bounds_error=False, fill_value=None)
aniso1_S4_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), aniso1_S4, method='linear', bounds_error=False, fill_value=None)

# Second coefficient 
# Making the assumption that the change in isotropic emissivity decreases
# linearly with frequency with an intercept of 0, which may not be accurate 
# Wind speeds above 20 m/s are linearly extrapolated, which is not correct for 1.4 GHz. Fix for this is TBD
aniso2_v = np.zeros((len(freqs), len(angles), len(winds)))
aniso2_h = np.zeros((len(freqs), len(angles), len(winds)))
aniso2_S3 = np.zeros((len(freqs), len(angles), len(winds)))
aniso2_S4 = np.zeros((len(freqs), len(angles), len(winds)))

for f in range(len(freqs)): 
    for a in range(len(angles)): 
        delta_v = 0
        delta_h = 0
        delta_s3 = 0
        delta_s4 = 0
        if freqs[f] == 1.4:
            if angles[a] <= 29.36:  
                for i in range(1, 6): 
                    delta_v = delta_v + aq_wind_coeffs.loc[29.36, 'v', 2][i] * winds**i
                    delta_h = delta_h + aq_wind_coeffs.loc[29.36, 'h', 2][i] * winds**i
                if angles[a] < 29.36: 
                    aniso2_s1 = (delta_v + delta_h) / 2
                    aniso2_s2 = delta_v - delta_h 
                    aniso2_s1 = aniso2_s1 * (angles[a] / 29.36)**2 
                    uW = (winds**2 - winds**3 / 22.5) / 55.5556
                    sf = 2 / 290 * (1 - np.log10(30 / freqs[f]))
                    nadir = uW * sf 
                    aniso2_s2 = nadir + (aniso2_s2 - nadir) * (angles[a] / 29.36)**4
                    delta_v = aniso2_s1 + aniso2_s2 / 2
                    delta_h = aniso2_s1 - aniso2_s2 / 2

            elif angles[a] == 38.44:
                for i in range(1, 6): 
                    delta_v = delta_v + aq_wind_coeffs.loc[38.44, 'v', 2][i] * winds**i
                    delta_h = delta_h + aq_wind_coeffs.loc[38.44, 'h', 2][i] * winds**i
            elif angles[a] >= 46.29: 
                for i in range(1, 6): 
                    delta_v = delta_v + aq_wind_coeffs.loc[46.29, 'v', 2][i] * winds**i
                    delta_h = delta_h + aq_wind_coeffs.loc[46.29, 'h', 2][i] * winds**i
                if angles[a] > 46.29: 
                    aniso2_s1 = (delta_v + delta_h) / 2
                    aniso2_s2 = delta_v - delta_h 
                    aniso2_s1 = aniso2_s1 * (angles[a] / 46.29)**2 
                    uW = (winds**2 - winds**3 / 22.5) / 55.5556
                    sf = 2 / 290 * (1 - np.log10(30 / (freqs[f] / 1e3)))
                    nadir = uW * sf 
                    aniso2_s2 = nadir + (aniso2_s2 - nadir) * (angles[a] / 46.29)**4
                    delta_v = aniso2_s1 + aniso2_s2 / 2
                    delta_h = aniso2_s1 - aniso2_s2 / 2

                    # Linear extrapolation for angles higher than 46.29
                    t_inter = spi.RegularGridInterpolator((angles[:-1], winds), aniso2_v[f, :-1, :], method='linear', bounds_error=False, fill_value=None)
                    delta_v = t_inter((angles[a], winds))
                    t_inter = spi.RegularGridInterpolator((angles[:-1], winds), aniso2_h[f, :-1, :], method='linear', bounds_error=False, fill_value=None)
                    delta_h = t_inter((angles[a], winds))

        elif freqs[f] != 0:
            for i in range(1, 6): 
                delta_v = delta_v + aniso_wind_coeffs_2.loc[freqs[f], 'v'][i] * winds**i
                delta_h = delta_h + aniso_wind_coeffs_2.loc[freqs[f], 'h'][i] * winds**i
                delta_s3 = delta_s3 + aniso_wind_coeffs_2.loc[freqs[f], 'S3'][i] * winds**i
                delta_s4 = delta_s4 + aniso_wind_coeffs_2.loc[freqs[f], 'S4'][i] * winds**i
            if angles[a] < 55.2: 
                aniso2_s1 = (delta_v + delta_h) / 2
                aniso2_s2 = delta_v - delta_h 
                aniso2_s3 = delta_s3 
                aniso2_s4 = delta_s4 
                aniso2_s1 = aniso2_s1 * (angles[a] / 55.2)**2 
                aniso2_s4 = aniso2_s4 * (angles[a] / 55.2)**2 
                uW = (winds**2 - winds**3 / 22.5) / 55.5556
                sf = 2. / 290. * (1. - np.log10(30. / freqs[f]))
                nadir_s2 = uW * sf
                nadir_s3 = -uW * sf 
                aniso2_s2 = nadir_s2 + (aniso2_s2 - nadir_s2) * (angles[a] / 55.2)**4
                aniso2_s3 = nadir_s3 + (aniso2_s3 - nadir_s3) * (angles[a] / 55.2)**4
                delta_v = aniso2_s1 + aniso2_s2 / 2
                delta_h = aniso2_s1 - aniso2_s2 / 2
                delta_s3 = aniso2_s3
                delta_s4 = aniso2_s4

        aniso2_v[f, a, :] = delta_v
        aniso2_h[f, a, :] = delta_h
        aniso2_S3[f, a, :] = delta_s3
        aniso2_S4[f, a, :] = delta_s4

aniso2_v_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), aniso2_v, method='linear', bounds_error=False, fill_value=None)
aniso2_h_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), aniso2_h, method='linear', bounds_error=False, fill_value=None)
aniso2_S3_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), aniso2_S3, method='linear', bounds_error=False, fill_value=None)
aniso2_S4_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), aniso2_S4, method='linear', bounds_error=False, fill_value=None)

interpack = [iso_v_interpolator, iso_h_interpolator, 
             aniso1_v_interpolator, aniso1_h_interpolator, aniso1_S3_interpolator, aniso1_S4_interpolator, 
             aniso2_v_interpolator, aniso2_h_interpolator, aniso2_S3_interpolator, aniso2_S4_interpolator]
pickle.dump(interpack, open('MW_wind_interpolators.p', 'wb'))


# 2- FASTEM-6 Interpolators 
# From RTTOV source code and Kazumori and English 2015

angles = np.linspace(0, 65, 100)
winds = np.linspace(0, 20, 150)

freqs = np.array([6.925, 10.65, 18.7, 23.8, 36.5, 89])  
coef_mk_azi = np.array([4.401E-02, -1.636E+01,  1.478E+00, -4.800E-02,  3.202E-06, -6.002E-05,  # 06V OK
                        4.379E-02, -1.633E+01,  1.453E+00, -4.176E-02,  5.561E-06, -4.644E-05,  # 10V OK
                        5.009E-02, -1.638E+01,  1.520E+00, -3.994E-02,  1.330E-05,  1.113E-05,  # 19V OK
                        5.165E-02, -1.638E+01,  1.543E+00, -4.066E-02,  1.494E-05,  1.010E-05,  # 23V interpolated
                        5.553E-02, -1.638E+01,  1.602E+00, -4.246E-02,  1.903E-05,  7.524E-06,  # 37V OK
                        9.131E-05,  1.251E+00,  6.769E-01, -2.913E-02,  1.092E+00, -1.806E-04,  # 89V OK revised
                        1.234E-07, -8.179E-03, -1.040E+01,  4.477E-01,  0.000E+00,  3.390E-05,  # 06H OK
                        1.938E-05, -8.007E-03, -1.039E+01,  4.610E-01,  0.000E+00,  4.419E-05,  # 10H OK
                        1.362E-04, -1.013E-03, -9.235E+00,  3.844E-01,  0.000E+00,  2.891E-04,  # 19H OK
                        1.519E-04, -7.865E-04, -9.234E+00,  3.884E-01,  0.000E+00,  6.856E-04,  # 23H Interpolated
                        1.910E-04, -2.224E-04, -9.232E+00,  3.982E-01,  0.000E+00,  1.673E-03,  # 37H OK
                        3.554E-04,  5.226E-04,  9.816E-01, -7.783E-03,  0.000E+00,  2.437E+01]).reshape(2, 6, 6)

coef_mk_azi = np.swapaxes(coef_mk_azi, 0, -1)  # Coef number, freq, pol
mask_wind = winds.copy()[:, np.newaxis]
mask_wind[mask_wind > 18] = 18
A1v = coef_mk_azi[0, ..., 0] * (np.exp(-coef_mk_azi[4, ..., 0] * mask_wind**2) - 1.) \
        * (coef_mk_azi[1, ..., 0] * mask_wind + coef_mk_azi[2, ..., 0] * mask_wind**2 +
            coef_mk_azi[3, ..., 0] * mask_wind**3)
A2v = coef_mk_azi[5, :, 0] * mask_wind
A1h = coef_mk_azi[0, :, 1] * mask_wind
A2h = coef_mk_azi[1, :, 1] * (np.exp(-coef_mk_azi[5, :, 1] * mask_wind**2) - 
      1.) * (coef_mk_azi[2, :, 1] * mask_wind + coef_mk_azi[3, :, 1] * mask_wind**2 +
      coef_mk_azi[4, :, 1] * mask_wind**3)

A1s1 = (A1v + A1h) / 2.
A1s2 = A1v - A1h
A2s1 = (A2v + A2h) / 2.
A2s2 = A2v - A2h

# Nadir second harmonic 
mask_freq = freqs.copy()
mask_wind = winds.copy()[:, np.newaxis]
mask_freq[mask_freq > 37] = 37. 
mask_wind[mask_wind > 15] = 15.
A2s2_theta0 = (mask_wind**2 - mask_wind**3 / 22.5) / 55.5556 * (2. / 290) * (1.0 - np.log10(30.0 / mask_freq))
        
theta_ref = 55.2
xs11 = 2
xs12 = 2
xs21 = 1
xs22 = 4

bc_angles = angles[:, np.newaxis, np.newaxis]

A1s1_theta = A1s1 * ((bc_angles / theta_ref)**xs11)
A2s1_theta = A2s1 * ((bc_angles / theta_ref)**xs12)
A1s2_theta = A1s2 * ((bc_angles / theta_ref)**xs21)
A2s2_theta = A2s2_theta0 + (A2s2 - A2s2_theta0) * ((bc_angles / theta_ref)**xs22)

A1v_theta = 0.5 * (2. * A1s1_theta + A1s2_theta)
A1h_theta = 0.5 * (2. * A1s1_theta - A1s2_theta)
A2v_theta = 0.5 * (2. * A2s1_theta + A2s2_theta)
A2h_theta = 0.5 * (2. * A2s1_theta - A2s2_theta)

# FASTEM-6 assumes L band model function is the same as C Band
freqs = np.insert(freqs, 0, 1.4)
A1v_theta = np.insert(A1v_theta, 0, A1v_theta[..., 0], axis=-1)
A1h_theta = np.insert(A1h_theta, 0, A1h_theta[..., 0], axis=-1)
A2v_theta = np.insert(A2v_theta, 0, A2v_theta[..., 0], axis=-1)
A2h_theta = np.insert(A2h_theta, 0, A2h_theta[..., 0], axis=-1)


A1v_theta = np.moveaxis(A1v_theta, (0, 1, 2), (1, 2, 0))
A1h_theta = np.moveaxis(A1h_theta, (0, 1, 2), (1, 2, 0))
A2v_theta = np.moveaxis(A2v_theta, (0, 1, 2), (1, 2, 0))
A2h_theta = np.moveaxis(A2h_theta, (0, 1, 2), (1, 2, 0))


aniso1_v_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), A1v_theta, method='linear', bounds_error=False, fill_value=None)
aniso1_h_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), A1h_theta, method='linear', bounds_error=False, fill_value=None)
aniso2_v_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), A2v_theta, method='linear', bounds_error=False, fill_value=None)
aniso2_h_interpolator = spi.RegularGridInterpolator((freqs * 1e3, angles, winds), A2h_theta, method='linear', bounds_error=False, fill_value=None)

interpack = [aniso1_v_interpolator, aniso1_h_interpolator, aniso2_v_interpolator, aniso2_h_interpolator]
pickle.dump(interpack, open('FASTEM_wind_interpolators.p', 'wb'))
