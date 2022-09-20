import numpy as np
from numpy import exp, pi
import scipy.constants as spc 


# Liquid water

def h2o_liquid_KleinSwift(frequency, temperature, salinity): 
    """ Model from Klein and Swift 1977 for sea water dielectric constant. 
        Valid for frequencies lower than 50 GHz.        

        :param frequency: Frequency in MHz
        :param temperature: Temperature in Kelvin
        :param salinity: Salinity in psu
        :return: Liquid water dielectric constant

    """ 

    SST = temperature - spc.zero_Celsius
    S = salinity
    f = frequency * 1e6

    e0 = 8.8541878128e-12  # spc.physical_constants['vacuum electric permittivity'][0]
    esw_inf = 4.9
    T = SST

    a = 1.0 + 1.613e-5 * T * S - 3.656e-3 * S + 3.210e-5 * S**2 - 4.232e-7 * S**3
    esw_T0 = 87.134 - 1.949e-1 * T - 1.276e-2 * T**2 + 2.491e-4 * T**3
    esw0 = esw_T0 * a

    tau_T0 = (1.1109e-10 - 3.824e-12 * T + 6.938e-14 * T**2 - 5.096e-16 * T**3) / (2 * pi)
    b = 1.0 + 2.282e-5 * T * S - 7.638e-4 * S - 7.760e-6 * S**2 + 1.105e-8 * S**3
    tau_sw = tau_T0 * b

    sigma_25 = S * (0.18252 - 1.4619e-3 * S + 2.093e-5 * S**2 - 1.282e-7 * S**3)
    delt = 25 - T
    phi_e = delt * (2.033e-2 + 1.266e-4 * delt + 2.464e-6 * delt**2 - S * (1.849e-5 - 2.551e-7 * delt + 2.551e-8 * delt**2))
    sigma = sigma_25 * exp(-phi_e)

    esw_real = esw_inf + (esw0 - esw_inf) / (1 + (2 * pi * f * tau_sw)**2)
    esw_img = (2 * pi * f * tau_sw * (esw0 - esw_inf)) / (1 + (2 * pi * f * tau_sw)**2) + sigma / (2 * pi * e0 * f)

    epsilon = esw_real - 1j * esw_img

    return epsilon


def h2o_liquid_MeissnerWentz(frequency, temperature, salinity): 
    """ Model from Meissner and Wentz 2004 & 2012 for sea water dielectric constant

        :param frequency: Frequency in MHz 
        :param temperature: Temperature in Kelvin 
        :param salinity: Salinity in psu 
        :return: Liquid water dielectric constant
    """

    s_ppth = salinity
    celsius = temperature - spc.zero_Celsius
    ghz = frequency / 1e3

    x = np.array([5.723, 2.2379e-2, -7.1237e-4, 5.0478, -7.0315e-2, 6.0059e-4, 3.6143, 2.8841e-2, 1.3652e-1, 1.4825e-3, 2.4166e-4])

    # Pure water parameters
    es = (3.70886e4 - 8.2168e1 * celsius) / (4.21854e2 + celsius)
    e1 = x[0] + x[1] * celsius + x[2] * celsius**2
    v1 = (45 + celsius) / (x[3] + x[4] * celsius + x[5] * celsius**2)
    einf = x[6] + x[7] * celsius
    v2 = (45 + celsius) / (x[8] + x[9] * celsius + x[10] * celsius**2)

    # Conductivity
    sigma_35 = 2.903602 + 8.607e-2 * celsius + 4.738817e-4 * celsius**2 - 2.991e-6 * celsius**3 + 4.3047e-9 * celsius**4
    r_15 = s_ppth * (37.5109 + 5.45216 * s_ppth + 1.4409e-2 * s_ppth**2) / (1004.75 + 182.283 * s_ppth + s_ppth**2)
    a0 = (6.9431 + 3.2841 * s_ppth - 9.9486e-2 * s_ppth**2) / (84.850 + 69.024 * s_ppth + s_ppth**2)
    a1 = 49.843 - 0.2276 * s_ppth + 0.198e-2 * s_ppth**2
    r_t_r_15 = 1 + (a0 * (celsius - 15)) / (a1 + celsius)
    sigma = sigma_35 * r_15 * r_t_r_15

    # Salt water parameters
    # Parameters from Meissner and Wentz 2012
    b = np.array([-3.33330e-3, 4.74868e-6, 0, 2.39357e-3, -3.13530e-5, 2.52477e-7, -6.28908e-3, 1.76032e-4, -9.22144e-5, -1.99723e-2, 1.81176e-4, -2.04265e-3, 1.57883e-4])
    d = np.array([0.23232e-2, -0.79208e-4, 0.36764e-5, -0.35594e-6, 0.89795e-8])  # Correct d3 coefficient, as there's a typo in Meissner and Wentz 2012
    v1 = v1 * (1 + s_ppth * (d[0] + d[1] * celsius + d[2] * celsius**2 + d[3] * celsius**3 + d[4] * celsius**4))
    # Parameters from Meissner and Wentz 2004
    # b = np.array([-3.56417e-3, 4.74868e-6, 1.15574e-5, 2.39357e-3, -3.13530e-5, 2.52477e-7, -6.28908e-3, 1.76032e-4, -9.22144e-5, -1.99723e-2, 1.81176e-4, -2.04265e-3, 1.57883e-4])
    # v1 = v1 * (1 + s_ppth * (b[3] + b[4] * celsius + b[5] * celsius**2))

    es = es * exp(b[0] * s_ppth + b[1] * s_ppth**2 + b[2] * celsius * s_ppth)   
    e1 = e1 * exp(b[6] * s_ppth + b[7] * s_ppth**2 + b[8] * celsius * s_ppth)
    v2 = v2 * (1 + s_ppth * (b[9] + b[10] * celsius))
    einf = einf * (1 + s_ppth * (b[11] + b[12] * celsius))
    v1_mat = ghz * 1 / v1
    v2_mat = ghz * 1 / v2
    epsilon = (es - e1) / (1 + 1j * v1_mat) + (e1 - einf) / (1 + 1j * v2_mat) + einf - 1j * sigma * 17.97510 / ghz

    return epsilon


def h2o_liquid_Ellison(frequency, temperature, salinity): 
    """ Model from Ellison as discussed by Maetzler 2006 - *Thermal Microwave Radiation: Applications for Remote Sensing*
        and Ulaby and Long 2014 - *Microwave Radar and Radiometric Remote Sensing* books.
        Similar formulation to the Meissner and Wentz model.
        Valid for 0-30 C, salinity less than 40 parts per thousand, frequencies lower than 1 THz.
        
        :param frequency: Frequency in MHz 
        :param temperature: Temperature in Kelvin 
        :param salinity: Salinity in psu 
        :return: Liquid water dielectric constant 
    """

    celsius = temperature - spc.zero_Celsius
    s_ppth = salinity
    ghz = frequency / 1e3 

    # Constants 
    # Spare zero added in the 'a' array so that my indexing matches that in Ulaby and Long
    a = np.array([0, 0.46606917e-2, -0.26087876e-4, -0.63926782e-5, 0.63000075e1, 0.26242021e-2, -0.42984155e-2, 0.34414691e-4, 0.17667420e-3, -0.20491560e-6, 0.58366888e3, 0.12684992e3, 0.69227972e-4, 0.38957681e-6, 0.30742330e3, 0.12634992e3, 0.37245044e1, 0.92609781e-2, -0.26093754e-1])
    ew_0 = 87.85306 * exp(-0.00456992 * celsius - a[1] * s_ppth - a[2] * s_ppth**2 - a[3] * s_ppth * celsius)
    ew_1 = a[4] * exp(-a[5] * celsius - a[6] * s_ppth - a[7] * s_ppth * celsius)
    tw_1 = (a[8] + a[9] * s_ppth) * exp(a[10] / (celsius + a[11]))
    tw_2 = (a[12] + a[13] * s_ppth) * exp(a[14] / (celsius + a[15]))
    ew_inf = a[16] + a[17] * celsius + a[18] * s_ppth

    # Conductivity 
    sigma_35 = 2.903602 + 8.607e-2 * celsius + 4.7338819e-4 * celsius**2 - 2.991e-6 * celsius**3 + 4.3041e-9 * celsius**4 
    ps = s_ppth * (37.5109 + 5.45216 * s_ppth + .014409 * s_ppth**2) / (1004.75 + 182.283 * s_ppth + s_ppth**2)  
    a0 = (6.9431 + 3.2841 * s_ppth - 0.099486 * s_ppth**2) / (84.85 + 69.024 * s_ppth + s_ppth**2)
    a1 = 49.843 - 0.2276 * s_ppth + 0.00198 * s_ppth**2
    qs = 1 + (a0 * (celsius - 15)) / (celsius + a1)
    sigma = sigma_35 * ps * qs

    # Double Debye
    eps_r = ew_inf + (ew_0 - ew_1) / (1 + (2 * pi * ghz * tw_1)**2) + (ew_1 - ew_inf) / (1 + (2 * pi * ghz * tw_2)**2)
    eps_i = 2 * pi * ghz * tw_1 * (ew_0 - ew_1) / (1 + (2 * pi * ghz * tw_1)**2) + 2 * pi * ghz * tw_2 * (ew_1 - ew_inf) / (1 + (2 * pi * ghz * tw_2)**2) + sigma * 17.97510 / ghz

    epsilon = eps_r - 1j * eps_i 

    return epsilon


def h2o_liquid_Boutin(frequency, temperature, salinity): 
    """ Model for seawater dielectric constant from Boutin et al. 2020. 
        
        :param frequency: Frequency in MHz 
        :param temperature: Temperature in Kelvin 
        :param salinity: Salinity in psu 
        :return: Liquid water dielectric constant 
    """

    celsius = temperature - spc.zero_Celsius
    s_ppth = salinity
    ghz = frequency / 1e3 
    f0 = 17.97510

    x = [5.7230e+00, 2.2379e-02, -7.1237e-04, 5.0478e+00, -7.0315e-02,
         6.0059e-04, 3.6143e+00, 2.8841e-02, 1.3652e-01, 1.4825e-03, 2.4166e-04]
    z = [-3.56417e-03, 4.74868e-06, 1.15574e-05, 2.39357e-03, -3.13530e-05, 
         2.52477e-07, -6.28908e-03, 1.76032e-04, -9.22144e-05, -1.99723e-02, 
         1.81176e-04, -2.04265e-03, 1.57883e-04]
    PP = [0.000000001749069, 0.000001088535951, -0.000038972693320, 0.003228077425434]

    if celsius < -30.16: celsius = -30.16  

    # Pure water
    e0 = (3.70886e4 - 8.2168e1 * celsius) / (4.21854e2 + celsius)
    e1 = x[0] + x[1] * celsius + x[2] * celsius**2
    n1 = (45.00 + celsius) / (x[3] + x[4] * celsius + x[5] * celsius**2)

    # saline water
    # conductivity [s/m] taken from stogryn et al.
    sig35 = 2.903602 + 8.60700e-2 * celsius + 4.738817e-4 * celsius**2 - 2.9910e-6 * celsius**3 + 4.3047e-9 * celsius**4
    r15 = s_ppth * (37.5109 + 5.45216 * s_ppth + 1.4409e-2 * s_ppth**2) / (1004.75 + 182.283 * s_ppth + s_ppth**2)
    alpha0 = (6.9431 + 3.2841 * s_ppth - 9.9486e-2 * s_ppth**2) / (84.850 + 69.024 * s_ppth + s_ppth**2)
    alpha1 = 49.843 - 0.2276 * s_ppth + 0.198e-2 * s_ppth**2
    rtr15 = 1.0 + (celsius - 15.0) * alpha0 / (alpha1 + celsius)

    sig = sig35 * r15 * rtr15

    fSST = PP[0] * celsius**3 + PP[1] * celsius**2 + PP[2] * celsius + PP[3]

    a0 = 1 - s_ppth * fSST
    e0s = a0 * e0    
    b1 = 1
    n1s = n1 * b1  
    # a1  = np.exp(z[6] * s + z[7] * s2 + z[8] * s * sst); 
    a1 = 1
    e1s = e1 * a1   
    epsr = (e0s - e1s) / (1.0 + 1j * (ghz / n1s)) + e1s - 1j * sig * f0 / ghz

    epsilon = np.real(epsr) - 1j * abs(np.imag(epsr))

    return epsilon


def h2o_liquid_Zhou(frequency, temperature, salinity): 
    """ Model for seawater dielectric constant from Zhou et al. 2021. 
        
        :param frequency: Frequency in MHz 
        :param temperature: Temperature in Kelvin 
        :param salinity: Salinity in psu 
        :return: Liquid water dielectric constant 
    """ 
    celsius = temperature - spc.zero_Celsius
    s_ppth = salinity
    ghz = frequency / 1e3 

    e_inf = 4.9 
    tau_f = np.array([1.75030e-11, -6.12993e-13, 1.24504e-14, -1.14927e-16])
    tau = tau_f[0] + tau_f[1] * celsius + tau_f[2] * celsius**2 + tau_f[3] * celsius**3
    esdw_f = np.array([8.80516e1, -4.01796e-1, -5.10271e-5, 2.55892e-5])
    esdw = esdw_f[0] + esdw_f[1] * celsius + esdw_f[2] * celsius**2 + esdw_f[3] * celsius**3
    rswdw_f = np.array([3.97185e-3, -2.49205e-5, -4.27558e-5, 3.92825e-7, 4.15350e-7])
    rswdw = 1 - s_ppth * (rswdw_f[0] + rswdw_f[1] * celsius + rswdw_f[2] * s_ppth + rswdw_f[3] * celsius * s_ppth + rswdw_f[4] * s_ppth**2)
    sig_f = np.array([9.50470e-2, -4.30858e-4, 2.16182e-6])
    sig = sig_f[0] * s_ppth + sig_f[1] * s_ppth**2 + sig_f[2] * s_ppth**3
    rsig_f = np.array([3.76017e-2, 6.32830e-5, 4.8342e-7, -3.97484e-4, 6.26522e-6])
    rsig = 1 + celsius * (rsig_f[0] + rsig_f[1] * celsius + rsig_f[2] * celsius**2 + rsig_f[3] * s_ppth + rsig_f[4] * s_ppth**2)
    true_sig = sig * rsig

    e0 = 8.8541878128e-12  # spc.physical_constants['vacuum electric permittivity'][0]
    # epsilon = e_inf + (esdw * rswdw - e_inf) / (1 + 1j * 2 * pi * ghz * tau) - 1j * true_sig / (2 * pi * ghz * e0)
    epsilon = e_inf + (esdw * rswdw - e_inf) / (1 + 1j * 2 * pi * 1e9 * ghz * tau) - 1j * true_sig / (2 * pi * ghz * 1e9 * e0)

    return epsilon


def foam_Stogryn(frequency, angle): 
    """ From the empirical polarized foam *emissivity* (not dielectric constant) model of Stogryn 1972

        :param frequency: Frequency in MHz
        :param angle: Satellite look angle in degrees off nadir

        Anguelova parameterization will be added in the future

    """
    angle[angle > 70] = 70
    ghz = frequency / 1e3
    emis_nadir = (208 + 1.29 * ghz) / 288 
    F_h = 1 - 1.748e-3 * angle - 7.336e-5 * angle**2 + 1.004e-7 * angle**3
    F_v = 1 - 9.946e-4 * angle + 3.218e-5 * angle**2 - 1.187e-6 * angle**3 + 7e-20 * angle**10
    emis_H = emis_nadir * F_h
    emis_V = emis_nadir * F_v

    return emis_H, emis_V


# Ice 

def h2o_ice_Maetzler(frequency, temperature, salinity):
    """ Computes permittivity of impure ice from Maetzler 2006. The model was developed for salinity around 0.013 PSU,
        and extrapolation is based on linear assumption to salinity, so over-extrapolation isn't recommended.

        :param frequency: Frequency in MHz 
        :param temperature: Temperature in Kelvin 
        :param salinity: Salinity in psu 
        :return: Ice dielectric constant 

        Code and documentation modified from the Snow Microwave Radiative Transfer Model (SMRT) 
        https://www.smrt-model.science/
    """

    # Modify imaginary component calculated for pure ice
    ghz = frequency / 1e3
    ereal = 3.1884 + 9.1e-4 * (temperature - spc.zero_Celsius)
    theta = 300.0 / temperature - 1.0
    alpha = (0.00504 + 0.0062 * theta) * exp(-22.1 * theta)
    b1 = 0.0207
    b2 = 1.16e-11
    b = 335.0
    deltabeta = exp(- 9.963 + 0.0372 * (temperature - spc.zero_Celsius))
    betam = (b1 / temperature) * (exp(b / temperature) / ((exp(b / temperature) - 1)**2)) + b2 * ghz**2
    beta = betam + deltabeta
    eimag = alpha / ghz + beta * ghz

    pure_ice_permittivity = ereal - 1j * eimag

    # Equation 5.37 from Maetzler 2006: Thermal Microwave Radiation: Applications for Remote Sensing
    g0 = 1866 * np.exp(-0.317 * ghz)
    g1 = 72.2 + 6.02 * ghz

    # Equation 5.36
    delta_Eimag = 1. / (g0 + g1 * (spc.zero_Celsius - temperature))

    # Equation 5.38
    S0 = 0.013 
    epsilon = pure_ice_permittivity - 1j * delta_Eimag * salinity / S0

    return epsilon 


# Soil

def soil_dobson(frequency, moisture, sand, clay, temperature=296, density=1.7): 
    """ Soil model from Dobson et al. 1985 as presented in Ulaby and Long 2014 

        :param frequency: Frequency in MHz
        :param moisture: Volumetric soil moisture in g/cm3
        :param sand: Sand fraction of soil
        :param clay: Clay fraction of soil
        :param temperature: Soil temperature in K
        :param density: Soil density in g/cm3
        :return: Soil dielectric constant

    """
    ghz = frequency / 1e3
    moist = moisture
    san = sand
    cla = clay

    # Constants 
    alpha = 0.65
    beta_1 = 1.27 - 0.519 * san - 0.152 * cla 
    beta_2 = 2.06 - 0.928 * san - 0.255 * cla 
    sigma = -1.65 + 1.939 * density - 2.256 * san + 1.594 * cla 
    eps_w = h2o_liquid_KleinSwift(frequency, temperature, 0)  # This can be replaced with a different water model from above if necessary.
    eps_r = np.real(eps_w)
    eps_i = -np.imag(eps_w) + np.nan_to_num((2.65 - density) / (2.65 * np.array([moist])), nan=0, posinf=0, neginf=0) * sigma * 17.97510 / ghz
    es_r = (1 + 0.66 * density + moist**beta_1 * eps_r**alpha - moist)**(1 / alpha)
    es_i = moist**beta_2 * eps_i

    epsilon = es_r - 1j * es_i

    return epsilon


def soil_peplinski(frequency, moisture, sand, clay, temperature=296, density=1.7): 
    """ Soil model from Dobson et al. 1985 with Peplinski 1995 conductivity term
        The modified conductivity term is specifically relevant for frequencies between 0.3 - 1.3 GHz 

        :param frequency: Frequency in MHz
        :param moisture: Volumetric soil moisture in g/cm3
        :param sand: Sand fraction of soil
        :param clay: Clay fraction of soil
        :param temperature: Soil temperature in K
        :param density: Soil density in g/cm3
        :return: Soil dielectric constant

    """
    ghz = frequency / 1e3
    moist = moisture
    san = sand
    cla = clay

    # Constants 
    alpha = 0.65
    beta_1 = 1.27 - 0.519 * san - 0.152 * cla 
    beta_2 = 2.06 - 0.928 * san - 0.255 * cla 
    sigma = 0.0467 + 0.22 * density - 0.411 * san + 0.661 * cla 
    eps_w = h2o_liquid_KleinSwift(frequency, temperature, 0) 
    eps_r = np.real(eps_w)
    eps_i = -np.imag(eps_w) + np.nan_to_num((2.65 - density) / (2.65 * np.array([moist])), nan=0, posinf=0, neginf=0) * sigma * 17.97510 / ghz

    es_r = (1 + 0.66 * density + moist**beta_1 * eps_r**alpha - moist)**(1 / alpha)
    es_i = moist**beta_2 * eps_i

    epsilon = es_r - 1j * es_i

    return epsilon


def soil_modified_dobson(frequency, moisture, sand, clay): 
    """ Soil model from Dobson et al. 1985 with modification from the Aquarius Algorithm
        Theoretical Basis Document. Performs better for wet soil 

        :param frequency: Frequency in MHz
        :param moisture: Volumetric soil moisture in g/cm3
        :param sand: Sand fraction of soil
        :param clay: Clay fraction of soil
        :return: Soil dielectric constant
    """
    ghz = frequency / 1e3
    beta = 1.09 - 0.11 * sand + 0.18 * clay
    alpha = 0.65
    r_b = 1.3  # Soil bulk density 
    r_ss = 2.65  # Density of solid soil
    eps_ss = 4.7  # Dielectric constant of dry soil
    fo = 18.64  # Relaxation frequency of GHz of pure water

    eps = 1 + r_b / r_ss * (eps_ss**alpha - 1) + moisture**beta * (4.9 + 74.1 / (1 - 6j * ghz / fo)**alpha) - moisture
    epsilon = np.real(eps**(1 / alpha)) - 1j * np.imag(eps**(1 / alpha))

    return epsilon


def soil_hallikainen(moisture, sand, clay): 
    """ L Band soil model from Hallikainen et al. 1985 as discussed in the Aquarius ATBD. 
        Performs better for dry soil. 

        :param moisture: Volumetric soil moisture in g/cm3
        :param sand: Sand fraction of soil
        :param clay: Clay fraction of soil
        :return: Soil dielectric constant
    """

    eps_r = 2.862 - 0.012 * sand + 0.001 * clay + moisture * (3.803 + 0.462 * sand - 0.341 * clay) + moisture**2 * (119.006 - 0.5 * sand + 0.663 * clay)
    eps_i = 0.356 - 0.003 * sand + 0.008 * clay + moisture * (5.507 + 0.044 * sand - 0.002 * clay) + moisture**2 * (17.753 - 0.313 * sand + 0.206 * clay)

    epsilon = eps_r - 1j * eps_i 

    return epsilon 


def soil_wang(moisture): 
    """ L Band soil model from Le Vine and Kao 1997 and Wang 1980. Performs better for wet soil 
        
        :param moisture: Volumetric soil moisture in g/cm3
        :return: Soil dielectric constant
    """

    eps_r = 3.1 + 17.36 * moisture + 63.12 * moisture**2
    eps_i = 0.031 + 4.65 * moisture + 20.42 * moisture**2

    epsilon = eps_r - 1j * eps_i 

    return epsilon
