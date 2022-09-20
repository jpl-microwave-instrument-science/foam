import numpy as np 
import matplotlib 
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime as dt
from cftime import date2num 

import foam.atmosphere as atmosphere
from foam.utils.reader import NCEPReader, MERRAReader


tests = [1, 2, 3, 4, 5]
tests = [5]

if 1 in tests: 
    print('Running Test 1')
    print('Comparing dry air opacity with Liebe 1985 Fig. 2 at 0% RH')
    print('Comparing wet air opacity with Liebe 1985 Fig. 2 at 100% RH')

    atm = atmosphere.atmosphere(datetime=dt.datetime(2017, 1, 1, 12), mode='simple', online=False)

    F = np.linspace(1e3, 350e3, 500)
    w = 23  # g/m^3
    P = 101325 / 1e2  # mbar
    T = 273 + 25  # K

    absn2 = atm.n2abs(T, P, F)
    abso2 = atm.o2abs(T, P, w, F)
    absh2o = atm.h2oabs(T, P, w, F)

    air = 10 * np.log10(np.exp(1)) * (absn2 + abso2)  # dB/km
    wv = 10 * np.log10(np.exp(1)) * absh2o  # dB/km

    plt.figure()
    plt.semilogy(F / 1e3, air, label='Dry Air')
    plt.semilogy(F / 1e3, wv, label='Water Vapor')
    plt.semilogy(F / 1e3, air + wv, label='Combined')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Opacity (dB/km)')
    plt.legend()


if 2 in tests: 
    print('Running Test 2')
    print('Plotting ad hoc ISA vertical profiles')

    atm = atmosphere.atmosphere(datetime=dt.datetime(2017, 1, 1, 12), mode='simple', online=False)
    lat_flat = np.linspace(89, -89, int(180 / 0.5))
    lon_flat = np.linspace(-179, 179, int(360 / 0.5))
    lon, lat = np.meshgrid(lon_flat, lat_flat)
    current_time = date2num(atm.datetime.to_pydatetime(), atm.time_reference)

    prwtr = atm.prwtr_interp((current_time, lat, lon))
    airtemp = atm.airtemp_interp((current_time, lat, lon))
    airpres = atm.airpres_interp((current_time, lat, lon))
    lwtr = atm.lwtr_interp((current_time, lat, lon))
    T, P, dens, wv, lw, alt = atm.isa_profiles(airtemp, airpres, prwtr, lwtr)

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    figs = [fig0, fig1, fig2, fig3, fig4]
    axs = [ax0, ax1, ax2, ax3, ax4]
    for j in range(np.shape(T)[1]): 
        t = T[:, j, 0]
        p = P[:, j, 0]
        d = dens[:, j, 0]
        w = wv[:, j, 0]
        l = lw[:, j, 0]
        a = alt[:, j, 0]
        axs[0].plot(t, a, color=plt.cm.viridis(abs(lat_flat[j]) / 90))
        axs[1].plot(p, a, color=plt.cm.viridis(abs(lat_flat[j]) / 90))
        axs[2].plot(d, a, color=plt.cm.viridis(abs(lat_flat[j]) / 90))
        axs[3].plot(w, a, color=plt.cm.viridis(abs(lat_flat[j]) / 90))
        axs[4].plot(l, a, color=plt.cm.viridis(abs(lat_flat[j]) / 90))

    for i in range(len(axs)): 
        axs[i].set_ylabel('Altitude (km)')
    ax0.set_xlabel('Temperature (K)')
    ax1.set_xlabel('Pressure (Pa)')
    ax2.set_xlabel('Density (kg/m$^3$)')
    ax3.set_xlabel('Precipitable Water Vapor (kg/kg)')
    ax4.set_xlabel('Precipitable Liquid Water (kg/kg)')

    for i in range(len(axs)): 
        divider = make_axes_locatable(axs[i])
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
        norm = matplotlib.colors.Normalize(vmin=0, vmax=90)
        figs[i].colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), cax=ax_cb, orientation='vertical', label='Latitude')
        figs[i].add_axes(ax_cb)

if 3 in tests: 
    print('Running Test 3')
    print('Downloading and plotting NCEP 3D data')

    atm = atmosphere.atmosphere(datetime=dt.datetime(2017, 1, 1, 12), mode='simple', online=True, 
                                reader=NCEPReader, reader_kwargs={'dimension': '3D'})

    lat_flat = np.linspace(89, -89, int(180 / 0.5))
    lon_flat = np.linspace(-179, 179, int(360 / 0.5))
    lon, lat = np.meshgrid(lon_flat, lat_flat)
    current_time = date2num(atm.datetime.to_pydatetime(), atm.time_reference)
    lon = lon[np.newaxis, :]
    lat = lat[np.newaxis, :]
    profile = atm.profile[:, np.newaxis, np.newaxis]

    wv = atm.prwtr_interp((current_time, profile, lat, lon))
    T = atm.airtemp_interp((current_time, profile, lat, lon))
    P = atm.airpres_interp((current_time, profile, lat, lon))
    wv[wv < 0] = 0
    P[P < 0] = 0
    alt = atm.profile

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    figs = [fig0, fig1, fig2]
    axs = [ax0, ax1, ax2]
    for j in range(np.shape(T)[1]): 
        t = T[:, j, 0]
        p = P[:, j, 0]
        w = wv[:, j, 0]
        axs[0].plot(t, alt, color=plt.cm.viridis(abs(lat_flat[j]) / 90))
        axs[1].plot(p, alt, color=plt.cm.viridis(abs(lat_flat[j]) / 90))
        axs[2].plot(w, alt, color=plt.cm.viridis(abs(lat_flat[j]) / 90))

    for i in range(len(axs)): 
        axs[i].set_ylabel('Altitude (km)')
    ax0.set_xlabel('Temperature (K)')
    ax1.set_xlabel('Pressure (Pa)')
    ax2.set_xlabel('Precipitable Water Vapor (kg/kg)')

    for i in range(len(axs)): 
        divider = make_axes_locatable(axs[i])
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
        norm = matplotlib.colors.Normalize(vmin=0, vmax=90)
        figs[i].colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), cax=ax_cb, orientation='vertical', label='Latitude')
        figs[i].add_axes(ax_cb)

if 4 in tests: 
    print('Running Test 4')
    print('Downloading and plotting MERRA 3D data')

    atm = atmosphere.atmosphere(datetime=dt.datetime(2017, 1, 1, 12), mode='simple', online=True, 
                                reader=MERRAReader, reader_kwargs={'dimension': '3D'})

    lat_flat = np.linspace(89, -89, int(180 / 0.5))
    lon_flat = np.linspace(-179, 179, int(360 / 0.5))
    lon, lat = np.meshgrid(lon_flat, lat_flat)
    current_time = date2num(atm.datetime.to_pydatetime(), atm.time_reference)
    lon = lon[np.newaxis, :]
    lat = lat[np.newaxis, :]
    profile = atm.profile[:, np.newaxis, np.newaxis]

    wv = atm.prwtr_interp((current_time, profile, lat, lon))
    lw = atm.lwtr_interp((current_time, profile, lat, lon))
    T = atm.airtemp_interp((current_time, profile, lat, lon))
    P = atm.airpres_interp((current_time, profile, lat, lon))
    wv[wv < 0] = 0
    lw[lw < 0] = 0
    P[P < 0] = 0
    alt = atm.profile

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    figs = [fig0, fig1, fig2, fig3]
    axs = [ax0, ax1, ax2, ax3]
    for j in range(np.shape(T)[1]): 
        t = T[:, j, 0]
        p = P[:, j, 0]
        w = wv[:, j, 0]
        l = lw[:, j, 0]
        axs[0].plot(t, alt, color=plt.cm.viridis(abs(lat_flat[j]) / 90))
        axs[1].plot(p, alt, color=plt.cm.viridis(abs(lat_flat[j]) / 90))
        axs[2].plot(w, alt, color=plt.cm.viridis(abs(lat_flat[j]) / 90))
        axs[2].plot(w, alt, color=plt.cm.viridis(abs(lat_flat[j]) / 90))

    for i in range(len(axs)): 
        axs[i].set_ylabel('Altitude (km)')
    ax0.set_xlabel('Temperature (K)')
    ax1.set_xlabel('Pressure (Pa)')
    ax2.set_xlabel('Precipitable Water Vapor (kg/kg)')

    for i in range(len(axs)): 
        divider = make_axes_locatable(axs[i])
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
        norm = matplotlib.colors.Normalize(vmin=0, vmax=90)
        figs[i].colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), cax=ax_cb, orientation='vertical', label='Latitude')
        figs[i].add_axes(ax_cb)

if 5 in tests: 
    print('Running Test 5')
    print('Plotting AMSU weighting functions')

    res = 0.2
    f0 = 57.290344
    fp = 57.290344 + 0.3222
    fm = 57.290344 - 0.3222
    f = np.array([23.8, 31.4, 50.3, 52.8, 53.596 - 0.115, 53.596 + 0.115, 
                  54.40, 54.94, 55.50, f0, f0 - 0.217, f0 + 0.217, 
                  fm - 0.048, fm + 0.048,  # fp - 0.048, fp + 0.048, 
                  fm - 0.022, fm + 0.022,  # fp - 0.022, fp + 0.022, 
                  fm - 0.010, fm + 0.010,  # fp - 0.010, fp + 0.010, 
                  fm - 0.0045, fm + 0.0045,  # fp - 0.0045, fp + 0.0045, 
                  89.]) * 1e3
    angle = np.array([40.])

    atm = atmosphere.atmosphere(datetime=dt.datetime(2017, 1, 1, 12), mode='full', online=True, 
                                reader=MERRAReader, reader_kwargs={'dimension': '2D'}, profile_res=res)
    tbup, tbdn, prop_dict = atm.get_atmosphere_tb(f, 0., np.array([0.]), np.array([0.]), angle, in_epoch=False, use_time=False)

    z = prop_dict['vertical_grid_km'].ravel()
    wup = prop_dict['upward_weighting_function']
    cmap = plt.cm.viridis(np.linspace(0, 1, len(f)))
    for i in range(len(f)): 
        plt.plot(wup[i, :], z, color=cmap[i], label='%.2f GHz' % (f[i] / 1e3))
    plt.xlabel('Weighting Function')
    plt.ylabel('Altitude')
    plt.xlim(0, np.max(wup))
    plt.ylim(0, 70)
    # plt.legend()
    plt.tight_layout()
    

raise ValueError

print('Comparing full RT model to empirical model')
# lat_flat = np.linspace(89, -89, int(180 / 0.5))
lat_flat = np.array([0, 10, 20])
lon_flat = np.array([0])
atm = atmosphere.atmosphere(verbose=True, datetime=dt.datetime(2005, 1, 1, 12), mode='full', online=False)
f = np.linspace(1e3, 40e3, 500)
tbup, tbdn, prop_dict = atm.get_atmosphere_tb(f, '2005-01-01 12:00', lat_flat, lon_flat)
atm = atmosphere.atmosphere(verbose=True, datetime=dt.datetime(2005, 1, 1, 12), mode='simple', online=False)
f = np.linspace(1e3, 40e3, 500)
tbup2, tbdn2, prop_dict2 = atm.get_atmosphere_tb(f, '2005-01-01 12:00', lat_flat, lon_flat)

nf_c = plt.figure()
nf_r = plt.figure()
for j in range(np.shape(tbup)[1]): 
    plt.figure(nf_c.number)
    plt.plot(f / 1e3, tbup[:, j], label='Full Model', color=plt.cm.Reds(abs(lat_flat[j]) / 90))
    plt.plot(f / 1e3, tbup2[:, j], label='Empirical Model', color=plt.cm.Blues(abs(lat_flat[j]) / 90))
    plt.figure(nf_r.number)
    plt.plot(f / 1e3, abs(tbup2[:, j] - tbup[:, j]), label='Empirical Model', color=plt.cm.Greens(abs(lat_flat[j]) / 90))

plt.figure(nf_c.number)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Brightness Temperature')
plt.figure(nf_r.number)
plt.xlabel('Frequency (GHz)')
plt.ylabel('$\\Delta$ Brightness Temperature')









