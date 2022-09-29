import numpy as np 
import scipy.constants as spc
from matplotlib import pyplot as plt
from netCDF4 import date2num  
import spiceypy as spice
import cartopy.crs as ccrs

from .. import ocean
from .. import atmosphere
from .. import spacecraft as sc
from .. import dielectric


def plot_raw_inputs():
    o = ocean.ocean(mode='flat')

    lat = np.arange(90, -90, -1)
    lon = np.arange(-180, 180, 1)
    llat, llon = np.meshgrid(lat, lon, indexing='ij')
    current_time = date2num(o.time, o.time_reference)
    sst = o.sst_interp((llat, llon)) - spc.zero_Celsius
    sss = o.sss_interp((llat, llon))
    land = o.landmask_interp((llat, llon))
    uwind = o.uwind_interp((current_time, llat, llon))
    vwind = o.vwind_interp((current_time, llat, llon))
    prwtr = o.prwtr_interp((current_time, llat, llon))
    TEC = o.TEC_interp((llat, llon))
    del o
    plt.figure()
    plt.title('Sea Surface Temperature')
    plt.imshow(sst, cmap='jet')
    plt.colorbar(orientation='horizontal')
    plt.clim(vmin=0, vmax=30)

    plt.figure()
    plt.title('Sea Surface Salinity')
    plt.imshow(sss, cmap='jet')
    plt.colorbar(orientation='horizontal')
    plt.clim(vmin=30, vmax=38)

    plt.figure()
    plt.title('Land mask')
    plt.imshow(land, cmap='jet')

    plt.figure()
    plt.title('Zonal Wind')
    plt.imshow(uwind, cmap='jet')
    plt.colorbar(orientation='horizontal')

    plt.figure()
    plt.title('Meridional Wind')
    plt.imshow(vwind, cmap='jet')
    plt.colorbar(orientation='horizontal')

    plt.figure()
    plt.title('Precipitable Water')
    plt.imshow(prwtr, cmap='jet')
    plt.colorbar(orientation='horizontal')

    plt.figure()
    plt.title('Total Electron Content')
    plt.imshow(TEC, cmap='jet')
    plt.colorbar(orientation='horizontal')

    plt.show()
    plt.close('all')


def plot_TB_map(): 
    mesh_spacing = 1
    lat_flat = np.linspace(89.5, -89.5, int(180 / mesh_spacing))
    lon_flat = np.linspace(-179.5, 179.5, int(360 / mesh_spacing))
    lon, lat = np.meshgrid(lon_flat, lat_flat)
    o = ocean.ocean(mode='flat')
    omap = o.get_ocean_TB(np.array([1e3]), theta=40)
    omap_v = omap[0][0] * (1 - o.landmask_interp((lat, lon)))
    omap_h = omap[1][0] * (1 - o.landmask_interp((lat, lon)))
    omap_v[omap_v < 50] = np.nan
    omap_h[omap_h < 50] = np.nan
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(color='k', draw_labels=True)
    extent = (-179.5, 179.5, -89.5, 89.5)
    mode = np.median(omap_v[omap_v > 20])
    offset = 10
    plot = ax.imshow(omap_v, origin='upper', extent=extent, transform=ccrs.PlateCarree(), cmap='turbo', vmin=mode - offset, vmax=mode + offset)    
    fig.colorbar(plot, shrink=1, pad=0.1, label='Brightness Temperature (K)', orientation='horizontal')
    plt.savefig('Vpol_TB.png', dpi=300, transparent=True)

    fig2, ax2 = plt.subplots()
    ax2 = plt.axes(projection=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.gridlines(color='k', draw_labels=True)
    extent = (-179.5, 179.5, -89.5, 89.5)
    mode = np.median(omap_h[omap_h > 20])
    offset = 10
    plot = ax2.imshow(omap_h, origin='upper', extent=extent, transform=ccrs.PlateCarree(), cmap='turbo', vmin=mode - offset, vmax=mode + offset)
    fig2.colorbar(plot, shrink=1, pad=0.1, label='Brightness Temperature (K)', orientation='horizontal')
    plt.savefig('Hpol_TB.png', dpi=300, transparent=True)


def plot_dielectric(): 
    frequency = np.logspace(3, 6, 1000)
    salinity = np.linspace(30, 40, 10)
    temperature = 293 * np.ones(len(salinity))
    eps = dielectric.h2o_liquid_saline_KleinSwift(frequency[:, np.newaxis], temperature[np.newaxis, :], salinity[np.newaxis, :]) 
    plt.figure()
    plt.loglog(frequency, np.real(eps))
    plt.figure()
    plt.loglog(frequency, -np.imag(eps))


def plot_spectrum(): 

    o = ocean.ocean(mode='flat')
    frequency = np.linspace(0.5e3, 100e3, 1000)
    TBV, TBH = o.get_ocean_TB(frequency=frequency, sst=300, sss=35, uwind=0, vwind=0, gal_max=0, lat=0, angle=40)

    plt.figure()
    plt.title(r'Ocean Emission Spectrum')
    plt.semilogx(frequency / 1e3, TBH, label=r'T$_B^H$')
    plt.semilogx(frequency / 1e3, TBV, label=r'T$_B^V$')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Brightness Temperature (K)')
    plt.grid(b=True)
    plt.text(1.1, 201, r'T: 300K, S: 35 psu, $\theta$: 40$^\circ$')
    plt.legend()
    plt.savefig('freq_spectrum.png', format='png', dpi=300, transparent=True)

    angle = np.linspace(0, 80, 100)
    TBV, TBH = o.get_quick_ocean_TB(frequency=1e3, sst=300, sss=35, uwind=0, vwind=0, gal_max=0, lat=0, angle=angle)

    plt.figure()
    plt.title(r'Ocean Emission with Angle')
    plt.plot(angle, TBH, label=r'T$_B^H$')
    plt.plot(angle, TBV, label=r'T$_B^V$')
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Brightness Temperature (K)')
    plt.grid(b=True)
    plt.text(11, 240, r'T: 300K, S: 35 psu, F: 1 GHz')
    plt.legend()
    plt.savefig('angle_spectrum.png', format='png', dpi=300, transparent=True)

    wind = np.linspace(0, 20, 100)
    TBV, TBH = o.get_quick_ocean_TB(frequency=1e3, sst=300, sss=35, uwind=wind, vwind=0, gal_max=0, lat=0, angle=20)

    plt.figure()
    plt.title(r'Ocean Emission with Wind speed')
    plt.plot(wind, TBH, label=r'T$_B^H$')
    plt.plot(wind, TBV, label=r'T$_B^V$')
    plt.xlabel('Wind speed (m/s)')
    plt.ylabel('Brightness Temperature (K)')
    plt.grid(b=True)
    plt.text(2.5, 90.4, r'T: 300K, S: 35 psu, F: 1 GHz, $\theta$: 20$^\circ$')
    plt.legend()
    plt.savefig('wind_spectrum.png', format='png', dpi=300, transparent=True)

    gal = np.linspace(0, 20, 100)
    TBV, TBH = o.get_quick_ocean_TB(frequency=1e3, sst=300, sss=35, uwind=0, vwind=0, gal_max=gal, lat=0, angle=20)

    plt.figure()
    plt.title(r'Ocean Emission with Wind speed')
    plt.plot(gal, TBH, label=r'T$_B^H$')
    plt.plot(gal, TBV, label=r'T$_B^V$')
    plt.xlabel('Galactic Brightness Temperature (K)')
    plt.ylabel('Brightness Temperature (K)')
    plt.grid(b=True)
    plt.text(2.5, 104, r'T: 300K, S: 35 psu, F: 1 GHz, $\theta$: 20$^\circ$')
    plt.legend()
    plt.savefig('gal_spectrum.png', format='png', dpi=300, transparent=True)

    sst = np.linspace(273, 323, 100)
    TBV, TBH = o.get_quick_ocean_TB(frequency=1e3, sst=sst, sss=35, uwind=0, vwind=0, gal_max=0, lat=0, angle=20)

    plt.figure()
    plt.title(r'Ocean Emission with Temperature')
    plt.plot(sst, TBH, label=r'T$_B^H$')
    plt.plot(sst, TBV, label=r'T$_B^V$')
    plt.xlabel('Sea Surface Temperature (K)')
    plt.ylabel('Brightness Temperature (K)')
    plt.grid(b=True)
    plt.text(290, 96, r'S: 35 psu, F: 1 GHz, $\theta$: 20$^\circ$')
    plt.legend()
    plt.savefig('sst_spectrum.png', format='png', dpi=300, transparent=True)

    sss = np.linspace(30, 40, 100)
    TBV, TBH = o.get_quick_ocean_TB(frequency=1e3, sst=300, sss=sss, uwind=0, vwind=0, gal_max=0, lat=0, angle=20)

    plt.figure()
    plt.title(r'Ocean Emission with Salinity')
    plt.plot(sss, TBH, label=r'T$_B^H$')
    plt.plot(sss, TBV, label=r'T$_B^V$')
    plt.xlabel('Sea Surface Salinity (psu)')
    plt.ylabel('Brightness Temperature (K)')
    plt.grid(b=True)
    plt.text(32, 95, r'T: 300K, F: 1 GHz, $\theta$: 20$^\circ$')
    plt.legend()
    plt.savefig('sss_spectrum.png', format='png', dpi=300, transparent=True)


def plot_weighting_function(): 
    res = 0.2
    ghz = np.array([23.8, 31.4, 50.3, 52.8, 53.596, 54.40, 54.94, 55.50, 57.29, 57.61, 89])
    atm = atmosphere.atmosphere()
    angle = np.radians(40)
    t = np.array([300.])
    p = np.array([1e5])
    prwtr = np.array([0])
    lwtr = np.array([0])
    T, P, dens, wv, lw = atm.isa_profiles(t, p, prwtr, lwtr, res=res)
    tau, tbar = atm.mrtm(ghz, P[::-1], T[::-1], wv[::-1], lw[::-1])
    transup = np.exp(-1 * np.cumsum(tau / np.cos(angle), axis=0))
    wup = -np.gradient(transup, axis=0)
    z = np.arange(0, 100 - res, res)[::-1]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(ghz)))
    for i in range(len(ghz)): 
        plt.plot(wup[:, i], z, color=cmap[i], label='%.2f GHz' % ghz[i])
    plt.legend()
    plt.xlabel('Weighting Function')
    plt.ylabel('Altitude')
    plt.xlim(0, np.max(wup) + 0.2 * np.max(wup))
    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()
    plt.savefig('weight.png', dpi=300, transparent=True)


def plot_spacecraft_tracks(): 
    s = sc.spacecraft()
    epochs = np.arange(0, 5 * 24 * 60 * 60, 2)
    s.write_tle_kernels(inclination=np.radians(98.12), height=680e3, look_angle=40, start_epoch=epochs[0], end_epoch=epochs[-1] + 1) 
    pos, times = spice.spkpos('-999', epochs, 'ITRF93', 'NONE', '399')
    ll_arr = np.array([spice.reclat(x) for x in pos])
    # spoint = np.array([spice.sincpt('ELLIPSOID', 'EARTH', epochs[x], 'ITRF93', 'LT+S', '-999', 'SPACECRAFT', point[x])[0] for x in range(0, len(epochs))])
    splist = []
    for x in range(0, len(epochs)):
        try: 
            spoint = spice.sincpt('ELLIPSOID', 'EARTH', epochs[x], 'ITRF93', 'NONE', '-999', 'RADIOMETER', np.array([1, 0, 0]))
        except spice.stypes.NotFoundError: 
            spoint = [np.array([1, 0, 0]), 0]
        splist.append(spoint[0])
        pos, light = spice.spkpos('-999', epochs[x], 'ITRF93', 'LT+S', '399')
        pp = spice.reclat(pos)
        # print('SPK Lat: %f' % np.degrees(pp[2]))
        # print('SPK Lon: %f' % np.degrees(pp[1]))
        # print(spoint[0])
        # pp = spice.reclat(spoint[0])
        # print('CK Lat: %f' % np.degrees(pp[2]))
        # print('CK Lon: %f' % np.degrees(pp[1]))
    ll_parr = np.array([spice.reclat(x) for x in splist])
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.plot(np.degrees(ll_arr[:, 1]), np.degrees(ll_arr[:, 2]), transform=ccrs.PlateCarree(), linestyle='none', marker='.', color='blue')
    ax.plot(np.degrees(ll_parr[:, 1]), np.degrees(ll_parr[:, 2]), transform=ccrs.PlateCarree(), linestyle='none', marker='.', color='red')
    ax.set_extent([-180, 180, -90, 90])

    # Revists 
    # import cartopy.crs as ccrs 
    # from skimage import filters
    # lonbins = np.arange(-180, 180 + 3, 3)
    # latbins = np.arange(-90, -30 + 1, 1)
    # lats = np.degrees(ll_parr[:, 2])
    # lons = np.degrees(ll_parr[:, 1])
    # res, lonb, latb = np.histogram2d(lons, lats, bins=[lonbins, latbins], density=False)
    # gi = filters.gaussian(res.T, sigma=2)
    # ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    # plt.pcolormesh(lonb, latb, gi / np.max(gi.flatten()), transform=ccrs.PlateCarree())
    # plt.colorbar()
    # ax.coastlines()
    # ax.gridlines(color='k', linestyle=':')
    # ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    return ll_parr


def visual_reference(epoch): 
    s = sc.spacecraft()
    height = 700e3
    s.write_tle_kernels(inclination=np.radians(80), height=height) 

    fig = plt.figure(figsize=(10, 20)) 
    ax = fig.add_subplot(211, projection='3d')

    # Plot Earth 
    u = np.linspace(0, 2 * np.pi, 500)
    v = np.linspace(0, np.pi, 500)
    rad = s.earth_mean_radius / 1e3
    x = rad * np.outer(np.cos(u), np.sin(v))
    y = rad * np.outer(np.sin(u), np.sin(v))
    z = rad * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='black', alpha=0.4)
    pos, times = spice.spkpos('-999', epoch, 'ITRF93', 'NONE', '399')
    pos_norm = pos * (rad / np.linalg.norm(pos))
    ax.scatter(pos[0], pos[1], pos[2], color='blue')
    ax.quiver(0, 0, 0, pos_norm[0], pos_norm[1], pos_norm[2], arrow_length_ratio=0.1, color='blue')

    pos2, times = spice.spkpos('399', epoch, 'SPACECRAFT', 'NONE', '-999')
    mat = spice.pxform('SPACECRAFT', 'ITRF93', epoch)
    pos2 = mat @ pos2
    pos2_norm = pos2 * ((height / 1e3) / np.linalg.norm(pos))
    ax.quiver(pos[0], pos[1], pos[2], pos2_norm[0], pos2_norm[1], pos2_norm[2], color='red')

    try:
        spoint = spice.sincpt('ELLIPSOID', 'EARTH', epoch, 'ITRF93', 'NONE', '-999', 'SPACECRAFT', np.array([1, 0, 0]))[0]
        ax.scatter(spoint[0], spoint[1], spoint[2], color='green')
        ll_arr2 = spice.reclat(spoint)
        print("Lon/Lat SPOINT: %f, %f" % (np.degrees(ll_arr2[1]), np.degrees(ll_arr2[2])))

    except: 
        ll_arr2 = np.array([1, 0, 0])

    ll_arr = spice.reclat(pos)
    ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.plot(np.degrees(ll_arr[1]), np.degrees(ll_arr[2]), transform=ccrs.PlateCarree(), linestyle='none', marker='.', color='blue')
    ax2.plot(np.degrees(ll_arr2[1]), np.degrees(ll_arr2[2]), transform=ccrs.PlateCarree(), linestyle='none', marker='.', color='green')
    ax2.set_extent([-180, 180, -90, 90])
    print("Lon/Lat SUB: %f, %f" % (np.degrees(ll_arr[1]), np.degrees(ll_arr[2])))


def plot_spacecraft_temps():
    o = ocean.ocean()
    s = sc.spacecraft()
    s.write_tle_kernels(inclination=np.radians(80), height=700e3, angle=40) 
    epochs = np.arange(0, 1 * 60 * 60, 1)  # np.arange(0, 1 * 24 * 60 * 60, 1 * 60)
    splist = []
    for x in range(0, len(epochs)):
        try: 
            spoint = spice.sincpt('ELLIPSOID', 'EARTH', epochs[x], 'ITRF93', 'NONE', '-999', 'RADIOMETER', np.array([1, 0, 0]))
        except spice.stypes.NotFoundError: 
            spoint = [np.array([1, 0, 0]), 0]
        splist.append(spoint[0])
        pos, light = spice.spkpos('-999', epochs[x], 'ITRF93', 'LT+S', '399')
        pp = spice.reclat(pos)
        print('SPK Lat: %f' % np.degrees(pp[2]))
        print('SPK Lon: %f' % np.degrees(pp[1]))
        print(spoint[0])
        pp = spice.reclat(spoint[0])
        print('CK Lat: %f' % np.degrees(pp[2]))
        print('CK Lon: %f' % np.degrees(pp[1]))
    ll_parr = np.array([spice.reclat(x) for x in splist])
    llat, llon = np.meshgrid(np.degrees(ll_parr[:, 2]), np.degrees(ll_parr[:, 1]))
    TBV, TBH, U, V = o.get_ocean_TB(1e3, llat, llon, theta=40)  # Placeholder for true angle calc

    plt.figure(1)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(np.degrees(ll_parr[:, 1]), np.degrees(ll_parr[:, 2]), c=np.diag(TBV[0, :, :]), cmap='viridis', transform=ccrs.PlateCarree(), s=1)
    # ax.colorbar()

    ax.set_extent([-180, 180, -90, 90])
    plt.savefig('TBV2.pdf', format='pdf', dpi=300)

    plt.figure(2)
    ax2 = plt.axes(projection=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.scatter(np.degrees(ll_parr[:, 1]), np.degrees(ll_parr[:, 2]), c=np.diag(TBH[0, :, :]), cmap='viridis', transform=ccrs.PlateCarree(), s=1)
    # ax.colorbar()

    ax2.set_extent([-180, 180, -90, 90])
    plt.savefig('TBH2.pdf', format='pdf', dpi=300)


