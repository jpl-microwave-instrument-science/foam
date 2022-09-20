import time 
import numpy as np 
import spiceypy as spice 
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs

from foam.spacecraft import spacecraft
import foam.utils.sc_utils as scu
from foam.utils.mk import manual_furnish


# In this script, we'll demonstrate the use of FOAMs spacecraft revisit tool by generating 
# weekly revisit maps for SMAP and for an arbitrary satellite constellation 
# This will be done using the parallel gridding routine to increase speed

# Monthly coverage, subdivided
# Starting first with one day
manual_furnish()  # Loading some spice kernels necessary to convert times 
datestrings = ['2020 MAY 20 9:00 UTC', '2020 MAY 27 9:00 UTC']
start_epoch = spice.str2et(datestrings[0])
end_epoch = spice.str2et(datestrings[1])
epoch_res = 100e-3  # Coarser resolution for specifying orbit/attitude info
spice.clpool()  # To prevent redundancy, clear the kernel pool

if __name__ == '__main__':  # Namespace guard multiprocessing code

    # 3 satellite constellation with a scanning horn
    # This constellation demonstrates a moderately increased revisit 
    # over the Gulf Stream, which is an interesting target 
    # for observations of ocean salinity 

    # Let's establish orbital parameters
    heights = [400, 500, 800]  # km 
            
    inclination = np.radians([45, 75, 105])
    tle_epoch = start_epoch

    raan = np.array([0, 8, 16])
    raan = raan / 24 * 2 * np.pi
    nproc = 8  # Change the number of cores depending on your system capabilities

    grid_list = []

    t1 = time.perf_counter()
    for i, ra in enumerate(raan):
        # Compute revisit grids for each spacecraft
        look_angle1 = scu.angle_conversion(heights[i], 40, in_angle_type='incidence')
        craft = spacecraft(sc_number=i)
        elems = craft.get_manual_elems(inclination=inclination[i], raan=ra, 
                                    height=heights[i] * 1e3, tle_epoch=start_epoch)
        craft.write_tle_kernels(elems=elems, tle_epoch=tle_epoch, start_epoch=start_epoch,
                             end_epoch=end_epoch, epoch_res=10)
        craft.write_radiometer_ck(look_angle1, 'Y', 10, 'X') 
        grid, llon, llat, obs_dict = scu.revisit_time(craft, start_epoch, end_epoch, epoch_res, 
                                                      plots=False, parallel=True, nproc=nproc,
                                                      grid_mode='cosine', grid_res=1, grid_stop=2)
        grid_list.append(grid)
        craft.unload()

    t2 = time.perf_counter()
    print('Elapsed with {} cores: {}'.format(nproc, t2 - t1))
    grid = np.sum(grid_list, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.coastlines()
    im = ax.pcolormesh(llon, llat, grid, shading='auto', alpha=0.8, cmap='turbo', transform=ccrs.PlateCarree())
    fig.colorbar(im, fraction=0.03, pad=0.04, label='Revisits')
    plt.savefig('constellation_revisit.png', dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(central_latitude=90))
    ax.coastlines()
    im = ax.pcolormesh(llon, llat, grid, shading='auto', alpha=0.8, cmap='turbo', transform=ccrs.PlateCarree())
    fig.colorbar(im, fraction=0.03, pad=0.04, label='Revisits')
    plt.savefig('constellation_polar_revisit.png', dpi=300)

    # Now repeating the same calculation for SMAP
    t1 = time.perf_counter()
    smap_craft = scu.make_smap(start_epoch, end_epoch, epoch_res=10, sc_number=4)
    grid, llon, llat, obs_dict = scu.revisit_time(smap_craft, start_epoch, end_epoch, epoch_res, 
                                                  plots=False, parallel=True, nproc=nproc,
                                                  grid_mode='cosine', grid_res=1, grid_stop=2)

    t2 = time.perf_counter()
    print('Elapsed with {} cores: {}'.format(nproc, t2 - t1))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.coastlines()
    im = ax.pcolormesh(llon, llat, grid, shading='auto', alpha=0.8, cmap='turbo')
    fig.colorbar(im, fraction=0.03, pad=0.04, label='Revisits')
    plt.savefig('smap_revisit.png', dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(central_latitude=90))
    ax.coastlines()
    im = ax.pcolormesh(llon, llat, grid, shading='auto', alpha=0.8, cmap='turbo', transform=ccrs.PlateCarree())
    fig.colorbar(im, fraction=0.03, pad=0.04, label='Revisits')
    plt.savefig('smap_polar_revisit.png', dpi=300)


