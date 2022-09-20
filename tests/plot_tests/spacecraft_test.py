import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import spiceypy as spice
import h5py 
import cartopy.crs as ccrs

import foam.spacecraft as sc


def smap_pointing(verbose=False):
    """ Compares SMAP L1B pointing data with a conical scanning spacecraft
        configuration initialized from SMAP orbital elements
    """

    smap_file = 'smap_data/SMAP_L1B_TB_29869_A_20200903T204518_R17000_001.h5'
    smap = h5py.File(smap_file, 'r')
    sc_data = smap['Spacecraft_Data']
    scan_time = np.mean(np.gradient(sc_data['antenna_scan_time'][:]))  # Time per one full scan in seconds
    rpm = 60 / scan_time
    fpscan = np.mean(sc_data['footprints_per_scan'][:])
    fpms = rpm * fpscan / 60 / 1e3
    int_time = 1 / fpms
    nadir_lat = sc_data['sc_nadir_lat'][:]
    nadir_lon = sc_data['sc_nadir_lon'][:]
    tb_data = smap['Brightness_Temperature']
    tb_time = tb_data['tb_time_seconds'][:]
    tb_time = tb_time[tb_time != -9999]
    start_epoch = tb_time[0]
    end_epoch = tb_time[-1]
    tb_lat = tb_data['tb_lat'][:].flatten()
    tb_lat = tb_lat[tb_lat != -9999]
    tb_lon = tb_data['tb_lon'][:].flatten()
    tb_lon = tb_lon[tb_lon != -9999]
    look_angle = tb_data['antenna_look_angle'][:]
    look_angle = np.mean(look_angle[look_angle != -9999])
    smap.close() 

    s = sc.spacecraft()
    s.write_tle_kernels(mode='file', file='smap_data/SMAP.tle', int_time=int_time, rpm=rpm, look_angle=look_angle, start_epoch=start_epoch, end_epoch=end_epoch) 

    increment = 1  # second
    epochs = np.arange(start_epoch, end_epoch, increment)   
    splist = []
    for x in range(0, len(epochs)):
        try: 
            spoint = spice.sincpt('ELLIPSOID', 'EARTH', epochs[x], 'ITRF93', 'NONE', '-999', 'RADIOMETER', np.array([1, 0, 0]))
        except spice.stypes.NotFoundError: 
            spoint = [np.array([1, 0, 0]), 0]
        splist.append(spoint[0])
        pos, light = spice.spkpos('-999', epochs[x], 'ITRF93', 'LT+S', '399')
        if verbose: 
            pp = spice.reclat(pos)
            print('SPK Lat: %f' % np.degrees(pp[2]))
            print('SPK Lon: %f' % np.degrees(pp[1]))
            print(spoint[0])
            pp = spice.reclat(spoint[0])
            print('CK Lat: %f' % np.degrees(pp[2]))
            print('CK Lon: %f' % np.degrees(pp[1]))
    ll_parr = np.array([spice.reclat(x) for x in splist])
    llat, llon = np.meshgrid(np.degrees(ll_parr[:, 2]), np.degrees(ll_parr[:, 1]))

    plt.figure(1)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(np.degrees(ll_parr[:, 1]), np.degrees(ll_parr[:, 2]), transform=ccrs.PlateCarree(), s=1, color='r', label='Simulated')
    ax.scatter(tb_lon[::int(increment * 1e3 / int_time)], tb_lat[::int(increment * 1e3 / int_time)], transform=ccrs.PlateCarree(), s=1, color='b', label='SMAP')
    ax.scatter(nadir_lon, nadir_lat, transform=ccrs.PlateCarree(), s=1, color='k', label='SMAP Nadir')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='k', alpha=0.3, linestyle='--')
    ax.set_extent([-180, 180, -90, 90])
    ax.legend()
    plt.savefig('smap_pointing_comparison.pdf', format='pdf', dpi=300)

    s.close()


def visual_reference(): 
    """ Visually illustrates spacecraft track and pointing as an animation """ 
    smap_file = 'smap_data/SMAP_L1B_TB_29869_A_20200903T204518_R17000_001.h5'
    smap = h5py.File(smap_file, 'r')
    times = smap['Brightness_Temperature']['tb_time_seconds'][:].flatten()
    times = times[times != -9999]
    sc_data = smap['Spacecraft_Data']
    scan_time = np.mean(np.gradient(sc_data['antenna_scan_time'][:]))  # Time per one full scan in seconds
    rpm = 60 / scan_time
    time_reference = 'seconds since 2000-01-01 11:58:55.816'  # Adjusted J2000
    fpscan = np.mean(sc_data['footprints_per_scan'][:])
    fpms = rpm * fpscan / 60 / 1e3
    int_time = 1 / fpms

    tb_data = smap['Brightness_Temperature']
    tb_time = tb_data['tb_time_seconds'][:]
    tb_time = tb_time[tb_time != -9999]
    start_epoch = tb_time[0]
    end_epoch = tb_time[-1]
    tb_lat = tb_data['tb_lat'][:].flatten()
    tb_lat = tb_lat[tb_lat != -9999]
    tb_lon = tb_data['tb_lon'][:].flatten()
    tb_lon = tb_lon[tb_lon != -9999]

    look_angle = tb_data['antenna_look_angle'][:]
    look_angle = np.mean(look_angle[look_angle != -9999])

    s = sc.spacecraft()
    s.write_tle_kernels(mode='file', file='smap_data/SMAP.tle', int_time=int_time, rpm=rpm, look_angle=look_angle, start_epoch=start_epoch, end_epoch=end_epoch) 
    fig = plt.figure(figsize=(20, 10)) 
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
    ax2.coastlines()

    # Plot Earth 
    u = np.linspace(0, 2 * np.pi, 500)
    v = np.linspace(0, np.pi, 500)
    rad = s.earth_mean_radius / 1e3
    x = rad * np.outer(np.cos(u), np.sin(v))
    y = rad * np.outer(np.sin(u), np.sin(v))
    z = rad * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='black', alpha=0.1)

    for t in np.arange(0, len(times), int(60e2 / int_time)):
        live_plotter(rad, tb_time, t, ax, ax2)
        if t == 0:
            plt.show(block=False)
        else:
            plt.draw()
        plt.pause(1e-3)


def live_plotter(rad, tb_time, epoch, ax, ax2): 
    pos, times = spice.spkpos('-999', tb_time[epoch], 'ITRF93', 'NONE', '399')
    pos_norm = pos * (rad / np.linalg.norm(pos))
    ax.scatter(pos[0], pos[1], pos[2], color='blue')
    # ax.quiver(0, 0, 0, pos_norm[0], pos_norm[1], pos_norm[2], arrow_length_ratio=0.1, color='blue')

    pos2, times = spice.spkpos('399', tb_time[epoch], 'SPACECRAFT', 'NONE', '-999')
    mat = spice.pxform('SPACECRAFT', 'ITRF93', tb_time[epoch])
    pos2 = mat @ pos2
    height = 685e3
    pos2_norm = pos2 * ((height / 1e3) / np.linalg.norm(pos))
    ax.quiver(pos[0], pos[1], pos[2], pos2_norm[0], pos2_norm[1], pos2_norm[2], color='red')

    try:
        spoint, trgepc, srfvec = spice.sincpt('ELLIPSOID', 'EARTH', tb_time[epoch], 'ITRF93', 'NONE', '-999', 'RADIOMETER', np.array([1, 0, 0]))
        ax.scatter(spoint[0], spoint[1], spoint[2], color='green')
        ll_arr2 = spice.reclat(spoint)
        print("Lon/Lat SPOINT: %f, %f" % (np.degrees(ll_arr2[1]), np.degrees(ll_arr2[2])))
        ax.quiver(spoint[0], spoint[1], spoint[2], -srfvec[0], -srfvec[1], -srfvec[2], color='purple')

    except: 
        ll_arr2 = np.array([1, 0, 0])
    enctime = spice.sce2c(-999, tb_time[epoch])
    cmat, clkout = spice.ckgp(-999000, enctime, 0, 'ITRF93')
    cpoint = cmat.T @ (np.array([1, 0, 0]))
    cpoint_norm = 2 * cpoint * ((height / 1e3) / np.linalg.norm(cpoint))
    ax.quiver(pos[0], pos[1], pos[2], cpoint_norm[0], cpoint_norm[1], cpoint_norm[2], color='g')
    cnormx = cpoint_norm
    cpoint = cmat.T @ (np.array([0, 1, 0]))
    cpoint_norm = 2 * cpoint * ((height / 1e3) / np.linalg.norm(cpoint))
    ax.quiver(pos[0], pos[1], pos[2], cpoint_norm[0], cpoint_norm[1], cpoint_norm[2], color='m')
    cnormy = cpoint_norm
    cpoint = cmat.T @ (np.array([0, 0, 1]))
    cpoint_norm = 2 * cpoint * ((height / 1e3) / np.linalg.norm(cpoint))
    ax.quiver(pos[0], pos[1], pos[2], cpoint_norm[0], cpoint_norm[1], cpoint_norm[2], color='y')
    cnormz = cpoint_norm

    ll_arr = spice.reclat(pos)
    ax2.plot(np.degrees(ll_arr[1]), np.degrees(ll_arr[2]), transform=ccrs.PlateCarree(), linestyle='none', marker='.', color='blue')
    ax2.plot(np.degrees(ll_arr2[1]), np.degrees(ll_arr2[2]), transform=ccrs.PlateCarree(), linestyle='none', marker='.', color='green')
    ax2.set_extent([-180, 180, -90, 90])
    print("Lon/Lat SUB: %f, %f" % (np.degrees(ll_arr[1]), np.degrees(ll_arr[2])))


def smap_revisit(): 
    smap_file = 'smap_data/SMAP_L1B_TB_29869_A_20200903T204518_R17000_001.h5'
    smap = h5py.File(smap_file, 'r')
    sc_data = smap['Spacecraft_Data']
    scan_time = np.mean(np.gradient(sc_data['antenna_scan_time'][:]))  # Time per one full scan in seconds
    rpm = 60 / scan_time
    fpscan = np.mean(sc_data['footprints_per_scan'][:])
    fpms = rpm * fpscan / 60 / 1e3
    int_time = 1 / fpms
    tb_data = smap['Brightness_Temperature']
    tb_time = tb_data['tb_time_seconds'][:]
    tb_time = tb_time[tb_time != -9999]
    start_epoch = tb_time[0]
    end_epoch = tb_time[-1]
    tb_lat = tb_data['tb_lat'][:].flatten()
    tb_lat = tb_lat[tb_lat != -9999]
    tb_lon = tb_data['tb_lon'][:].flatten()
    tb_lon = tb_lon[tb_lon != -9999]
    look_angle = tb_data['antenna_look_angle'][:]
    look_angle = np.mean(look_angle[look_angle != -9999])
    smap.close() 
    revisit_end = start_epoch + 3 * 24 * 60 * 60  # 3 days 
    print(int_time)
    s = sc.spacecraft()
    s.write_tle_kernels(mode='file', file='smap_data/SMAP.tle', int_time=int_time, rpm=rpm, look_angle=look_angle, start_epoch=start_epoch, end_epoch=revisit_end) 
    epoch_samples, lon_samples, lat_samples, theta_samples, phi_samples = s.make_grid(start_epoch, revisit_end, int_time * 10)
    return epoch_samples, lon_samples, lat_samples


def smap_extended_pointing(verbose=False):
    """ Compares SMAP L1B pointing data with a conical scanning spacecraft
        configuration initialized from SMAP orbital elements
    """

    smap_file = 'smap_data/SMAP_L1B_TB_29869_A_20200903T204518_R17000_001.h5'
    smap = h5py.File(smap_file, 'r')
    sc_data = smap['Spacecraft_Data']
    scan_time = np.mean(np.gradient(sc_data['antenna_scan_time'][:]))  # Time per one full scan in seconds
    rpm = 60 / scan_time
    fpscan = np.mean(sc_data['footprints_per_scan'][:])
    fpms = rpm * fpscan / 60 / 1e3
    int_time = 1 / fpms
    nadir_lat = sc_data['sc_nadir_lat'][:]
    nadir_lon = sc_data['sc_nadir_lon'][:]
    tb_data = smap['Brightness_Temperature']
    tb_time = tb_data['tb_time_seconds'][:]
    tb_time = tb_time[tb_time != -9999]
    start_epoch = tb_time[0]
    end_epoch = tb_time[-1]
    tb_lat = tb_data['tb_lat'][:].flatten()
    tb_lat = tb_lat[tb_lat != -9999]
    tb_lon = tb_data['tb_lon'][:].flatten()
    tb_lon = tb_lon[tb_lon != -9999]
    look_angle = tb_data['antenna_look_angle'][:]
    look_angle = np.mean(look_angle[look_angle != -9999])
    smap.close() 
    start_epoch = start_epoch + 10 * 24 * 60 * 60
    end_epoch = start_epoch + 1 * 24 * 60 * 60
    s = sc.spacecraft()
    s.write_tle_kernels(mode='file', file='smap_data/SMAP.tle', int_time=int_time, rpm=rpm, look_angle=look_angle, start_epoch=start_epoch, end_epoch=end_epoch) 

    increment = 20  # second
    epochs = np.arange(start_epoch, end_epoch, increment)   
    splist = []
    for x in range(0, len(epochs)):
        try:  
            spoint = spice.sincpt('ELLIPSOID', 'EARTH', epochs[x], 'ITRF93', 'NONE', '-999', 'RADIOMETER', np.array([1, 0, 0]))
        except spice.stypes.NotFoundError:
            print('Warning, missed point')
            spoint = [np.array([1, 0, 0]), 0]
        splist.append(spoint[0])
        pos, light = spice.spkpos('-999', epochs[x], 'ITRF93', 'LT+S', '399')
        if verbose: 
            pp = spice.reclat(pos)
            print('SPK Lat: %f' % np.degrees(pp[2]))
            print('SPK Lon: %f' % np.degrees(pp[1]))
            print(spoint[0])
            pp = spice.reclat(spoint[0])
            print('CK Lat: %f' % np.degrees(pp[2]))
            print('CK Lon: %f' % np.degrees(pp[1]))
    ll_parr = np.array([spice.reclat(x) for x in splist])
    llat, llon = np.meshgrid(np.degrees(ll_parr[:, 2]), np.degrees(ll_parr[:, 1]))

    plt.figure(1)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(np.degrees(ll_parr[:, 1]), np.degrees(ll_parr[:, 2]), transform=ccrs.PlateCarree(), s=1, color='r', label='Simulated')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='k', alpha=0.3, linestyle='--')
    ax.set_extent([-180, 180, -90, 90])
    ax.legend()

    s.close()



