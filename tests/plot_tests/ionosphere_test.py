import os 
import sys 
import numpy as np 
import pandas as pd
import scipy.constants as spc 
import datetime as dt 
from cftime import date2num
import matplotlib
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs 

import foam.ionosphere as iono 
import foam.geomag as geomag 

dir_path = os.path.dirname(sys.modules['foam'].__file__) + '/'

print('Testing geomag')

lat_grid = np.linspace(-90, 90, 180)
lon_grid = np.linspace(-180, 180, 180)
lon, lat = np.meshgrid(lon_grid, lat_grid)
mag_file = dir_path + 'assets/magneticfield/WMM2020.COF'
gm = geomag.GeoMag(mag_file)
mag = gm.GeoMag(lat, lon, h=(675e3 / spc.foot))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
pcm = ax.pcolormesh(lon_grid, lat_grid, mag.bz, shading='auto')
cbar = plt.colorbar(pcm, ax=ax)
ax.set_xlabel('Longitude (deg.)')
ax.set_xlabel('Latitude (deg.)')
ax.set_title('Magnetic Field from WMM')
cbar.set_label('Bz (nT)')

print('Testing faraday rotation')
tec = np.array([1, 10, 100])
color = ['k', 'r', 'b']
lat = np.array([45])
lon = np.array([0])
theta = 40 
phi = 0 
freq = np.logspace(np.log(0.5), np.log(5), 100) * 1e3
date = dt.datetime(2005, 1, 1)


ionosphere = iono.ionosphere()
plt.figure()
for t in range(len(tec)): 
    fa = ionosphere.compute_faraday_angle(tec[t], lat, lon, theta, phi, freq)
    plt.plot(freq / 1e3, np.degrees(fa), color=color[t], label='%i TEC' % tec[t])

plt.xlabel('Frequency (GHz)')
plt.ylabel('Faraday angle (degrees)')
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.1, 1000)
plt.xlim(0.5, 5)
plt.tight_layout()

print('Testing online CDDIS access')
ionosphere = iono.ionosphere(datetime=dt.datetime(2017, 1, 1, 12), online=True)
date_list = pd.date_range(dt.datetime(2017, 1, 1, 12), periods=20).to_pydatetime().tolist()
lat_flat = np.linspace(89, -89, int(180 / 0.5))
lon_flat = np.linspace(-179, 179, int(360 / 0.5))
lon, lat = np.meshgrid(lon_flat, lat_flat)

tec_list = []
for d in date_list: 
    ionosphere = iono.ionosphere(datetime=d, online=True)
    current_time = date2num(ionosphere.datetime, ionosphere.time_reference)
    tec_list.append(ionosphere.TEC_interp((current_time, lat, lon)))

fig, ax = plt.subplots()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(color='k', draw_labels=True)
extent = (-180, 180, -90, 90)
out = [[ax.imshow(tec_list[x], origin='upper', extent=extent, transform=ccrs.PlateCarree())] for x in range(0, len(date_list))]
fig.colorbar(out[0][0], shrink=0.7, pad=0.1)
anim = matplotlib.animation.ArtistAnimation(fig, out, interval=500)
anim.save('tec.gif', writer='imagemagick', fps=3, dpi=300)

print('Testing iri2016 simulation')
ionosphere = iono.ionosphere(IRI=True, verbose=True)
fig, ax = plt.subplots()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(color='k', draw_labels=True)
extent = (-180, 180, -90, 90)
lat = np.linspace(90, -90, 36)
lon = np.linspace(-180, 180, 72)
lon, lat = np.meshgrid(lon, lat)
tec = ionosphere.TEC_interp((lat, lon))
pcm = ax.pcolormesh(lon, lat, tec, shading='auto')
cbar = plt.colorbar(pcm, ax=ax)
ax.set_xlabel('Longitude (deg.)')
ax.set_xlabel('Latitude (deg.)')
ax.set_title('IRI TEC')
cbar.set_label('TEC')