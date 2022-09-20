# Ocean Plot Test 

import numpy as np
import matplotlib.pyplot as plt 
import datetime as dt 
from foam.ocean import ocean, fastem, two_scale


def tb_spectrum(): 
    f = np.linspace(1e3, 20e3, 100)
    lat = np.zeros(1)
    lon = np.zeros(1)
    theta = 40 * np.ones(1)
    phi = np.zeros(1)
    uwind = 5 * np.ones(1)
    vwind = 5 * np.ones(1)
    o = ocean(datetime=dt.datetime(2017, 1, 1, 12), online=True, verbose=True)  # Specular case 
    TBV, TBH, emisV, emisH = o.get_ocean_TB(f, lat, lon, uwind, vwind, theta, phi)

    plt.figure()
    plt.plot(f / 1e3, np.real(TBV), label='V Pol.', color='r')
    plt.plot(f / 1e3, np.real(TBH), label='H Pol.', color='b')

    o = ocean(datetime=dt.datetime(2017, 1, 1, 12), online=True, verbose=True, mode='full')  # Specular case 
    TBV, TBH, emisV, emisH = o.get_ocean_TB(f, lat, lon, uwind, vwind, theta, phi)

    plt.plot(f / 1e3, np.real(TBV), label='V Pol. GMF', color='r', linestyle='--')
    plt.plot(f / 1e3, np.real(TBH), label='H Pol. GMF', color='b', linestyle='--')

    # FASTEM doesn't work yet
    o = fastem(datetime=dt.datetime(2017, 1, 1, 12), online=True, verbose=True, mode='simple')  # Specular case
    TBV, TBH, emisV, emisH = o.get_ocean_TB(f, lat, lon, uwind, vwind, theta, phi)
    plt.plot(f / 1e3, np.real(TBV), label='V Pol. FASTEM', color='r', linestyle=':')
    plt.plot(f / 1e3, np.real(TBH), label='H Pol. FASTEM', color='b', linestyle=':')

    # Two scale doesn't work yet
    o = two_scale(datetime=dt.datetime(2017, 1, 1, 12), online=True, verbose=True, mode='simple')  # Specular case
    TBV, TBH, emisV, emisH = o.get_ocean_TB(f, lat, lon, uwind, vwind, theta, phi)
    plt.plot(f / 1e3, np.real(TBV), label='V Pol. Two Scale', color='r', linestyle=':')
    plt.plot(f / 1e3, np.real(TBH), label='H Pol. Two Scale', color='b', linestyle=':')

    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Brightness Temperature')


tb_spectrum() 
raise RuntimeError('Two scale model tests do not work at this time')

# Two scale model tests #

# Isotropic wind effects 


def iso_elevation_dependence():
    # Replicating Figure 2 from Yueh 1997
    # Isotropic wind signature from JPL WINRAD experiments 

    frequencies = np.array([19., 37.]) * 1e3
    incidence_angles = np.array([0., 10., 20., 30., 40., 50., 60., 70.])
    phis = np.arange(0, 360, 30)
    winds = np.array([3, 7, 11, 15])
    sst = 285 * np.ones(len(winds))
    sss = 35 * np.ones(len(winds))
    frequencies = frequencies[:, np.newaxis]
    sst = sst[np.newaxis, :]
    sss = sss[np.newaxis, :]
    winds = winds[np.newaxis, :]
    o = ocean.two_scale_ocean()
    Vpol1 = np.zeros((len(incidence_angles), 1, 4))
    Hpol1 = np.zeros((len(incidence_angles), 1, 4))
    Vpol2 = np.zeros((len(incidence_angles), 1, 4))
    Hpol2 = np.zeros((len(incidence_angles), 1, 4))

    for i in range(0, len(incidence_angles)): 
        emis = 0
        for ph in phis: 
            emis += o.get_ocean_emissivity(frequencies, sst, sss, incidence_angles[i], ph, winds, np.zeros(np.shape(winds))) / len(phis)

        emtemp = emis * 285
        Vpol1[i] = emtemp[0, 0]
        Hpol1[i] = emtemp[1, 0]
        Vpol2[i] = emtemp[0, 1]
        Hpol2[i] = emtemp[1, 1]

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(incidence_angles, Vpol1[:, 0, :])
    axs[1, 0].plot(incidence_angles, Hpol1[:, 0, :])
    axs[0, 1].plot(incidence_angles, Vpol2[:, 0, :])
    axs[1, 1].plot(incidence_angles, Hpol2[:, 0, :])
    axs[0, 0].set_ylabel(r'T$_B^V$')
    axs[1, 0].set_ylabel(r'T$_B^H$')
    axs[1, 0].set_xlabel('Incidence Angle')
    axs[1, 1].set_xlabel('Incidence Angle')
    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    plt.tight_layout()
    plt.savefig('iso_elevation_dependence.pdf', format='pdf', dpi=300, transparent=True)


def iso_frequency_dependence(): 
    # Replicating (or attempting to) Figure 9 from Meissner and Wentz 2012 

    frequencies = np.array([6.8, 10.7, 18.7, 23.8, 37, 85.5]) * 1e3 
    winds = np.array([12.5])
    theta = 47  # 55.2 
    sst = np.array([293.15])
    sss = np.array([0])
    phis = np.arange(0, 360, 30)
    o = ocean.two_scale_ocean()
    o_flat = ocean.ocean(full_ocean_active=False, verbose=False)
    emis_flat = o_flat.get_ocean_emissivity(frequencies, sst, sss, theta, 0)

    emis = 0
    for ph in phis: 
        emis += o.get_ocean_emissivity(frequencies, sst, sss, theta, ph, winds, np.zeros(np.shape(winds))) / len(phis)
    emis = (emis - emis_flat) * 290 

    plt.plot(frequencies / 1e3, emis[0, :], color='blue', label='v-pol')
    plt.plot(frequencies / 1e3, emis[1, :], color='red', label='h-pol')
    plt.xscale('log')
    plt.xlim(5, 100)
    plt.xlabel('Frequency (GHz)')
    plt.ylim(-10, 25)
    plt.ylabel(r'$\Delta E_W$ (* 290K)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('iso_frequency_dependence.pdf', format='pdf', dpi=300, transparent=True)


# Anisotropic wind effects 

def aniso_azimuth_dependence(): 
    frequencies = np.array([19., 37.]) * 1e3
    theta = 55 
    phis = np.arange(0, 360, 10)
    winds = np.array([11])
    sst = np.array([285])
    sss = np.array([0])
    o = ocean.two_scale_ocean()
    emis = np.zeros((4, 2, len(phis)))

    for ph in range(len(phis)): 
        emis[:, :, ph] = o.get_ocean_emissivity(frequencies, sst, sss, theta, phis[ph], winds, np.zeros(np.shape(winds)))
    emis = emis * sst 
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(phis, emis[0, 0, :] - np.mean(emis[0, 0, :]), color='red', marker='o')
    axs[0].plot(phis, emis[0, 1, :] - np.mean(emis[0, 1, :]), color='blue', marker='s')
    axs[1].plot(phis, emis[1, 0, :] - np.mean(emis[1, 0, :]), color='red', marker='o')
    axs[1].plot(phis, emis[1, 1, :] - np.mean(emis[1, 1, :]), color='blue', marker='s')
    axs[2].plot(phis, emis[2, 0, :] - np.mean(emis[2, 0, :]), color='red', marker='o')
    axs[2].plot(phis, emis[2, 1, :] - np.mean(emis[2, 1, :]), color='blue', marker='s')
    axs[3].plot(phis, emis[3, 0, :] - np.mean(emis[3, 0, :]), color='red', marker='o')
    axs[3].plot(phis, emis[3, 1, :] - np.mean(emis[3, 1, :]), color='blue', marker='s')

    axs[0].set_ylabel('$T_B^V$')
    axs[1].set_ylabel('$T_B^H$')
    axs[2].set_ylabel('$U$')
    axs[3].set_ylabel('$V$')
    axs[3].set_xlabel('Azimuth (degrees)')
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    axs[3].grid()
    plt.tight_layout()
    plt.savefig('aniso_azimuth_dependence.pdf', format='pdf', dpi=300, transparent=True)

