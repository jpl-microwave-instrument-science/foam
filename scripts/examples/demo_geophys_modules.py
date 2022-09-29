import numpy as np 
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs

# This script demonstrates several ways of interacting with various 
# FOAM geophysics modules and how to tie them all together with the solver 

# First, we'll start with the ocean module by loading the 'full' ocean
# model with cached ancillary data 
# By default, the ocean model loads several interpolator objects that 
# can be indexed by latitude and longitude
# Let's use this to generate a latitude/longitude map of 
# calm ocean brightness temperature at 1.4 GHz with a 40 degree incidence angle

from foam import ocean
oc = ocean.ocean(mode='rough')
frequency = np.array([1.4e3])  # MHz
lat_flat = np.linspace(89.5, -89.5, 180)
lon_flat = np.linspace(-179.5, 179.5, 360)
lon, lat = np.meshgrid(lon_flat, lat_flat)
phi = uwind = vwind = np.zeros(np.shape(lon))
theta = 40 * np.ones(np.shape(lon))
oTB, emis_dict = oc.get_ocean_TB(frequency, lat, lon, uwind, vwind, theta, phi)

fig = plt.figure(figsize=(5, 7)) 
oTB = oTB[:, 0, :]  # Extract single frequency 

# V-pol
ax = fig.add_subplot(211, projection=ccrs.PlateCarree())
TBV = oTB[0, :]
gl = ax.gridlines(color='k', linestyle='--', alpha=0.5, draw_labels=True)
gl.top_labels = False 
gl.right_labels = False
ax.coastlines()
im = ax.pcolormesh(lon, lat, TBV, shading='auto', transform=ccrs.PlateCarree(), cmap='jet')
ax.set_title('$\\mathregular{TB_V}$')
fig.colorbar(im, label='Brightness Temperature', ax=ax, fraction=0.025)

# H-pol
ax = fig.add_subplot(212, projection=ccrs.PlateCarree())
TBH = oTB[1, :]
gl = ax.gridlines(color='k', linestyle='--', alpha=0.5, draw_labels=True)
gl.top_labels = False 
gl.right_labels = False
ax.coastlines()
im = ax.pcolormesh(lon, lat, TBH, shading='auto', transform=ccrs.PlateCarree(), cmap='jet')
ax.set_title('$\\mathregular{TB_H}$') 
fig.colorbar(im, label='Brightness Temperature', ax=ax, fraction=0.025)

# We can roughen the ocean surface by adding a constant 10m/s wind speed
# and plotting the difference between the roughened surface and the smooth surface
# using interpolated geophysical model functions. 
# The isotropic emissivity signature amounts to a near scalar offset in brightness temperature
# The user can turn off the model wind roughening by setting mode='simple'

phi = vwind = np.zeros(np.shape(lon))
theta = 40 * np.ones(np.shape(lon))
uwind = 10 * np.ones(np.shape(lon))
frequency = np.array([1.4e3])  # MHz
oTBr, emis_dict = oc.get_ocean_TB(frequency, lat, lon, uwind, vwind, theta, phi)

fig = plt.figure(figsize=(5, 7)) 
oTBr = oTBr[:, 0, :]  # Extract single frequency 

# V-pol
ax = fig.add_subplot(211, projection=ccrs.PlateCarree())
TBVr = oTBr[0, :]
gl = ax.gridlines(color='k', linestyle='--', alpha=0.5, draw_labels=True)
gl.top_labels = False 
gl.right_labels = False
ax.coastlines()
im = ax.pcolormesh(lon, lat, TBVr - TBV, shading='auto', transform=ccrs.PlateCarree(), cmap='jet')
ax.set_title('$\\mathregular{TB_V}$')
fig.colorbar(im, label='Brightness Temperature', ax=ax, fraction=0.025)

# H-pol
ax = fig.add_subplot(212, projection=ccrs.PlateCarree())
TBHr = oTBr[1, :]
gl = ax.gridlines(color='k', linestyle='--', alpha=0.5, draw_labels=True)
gl.top_labels = False 
gl.right_labels = False
ax.coastlines()
im = ax.pcolormesh(lon, lat, TBH - TBHr, shading='auto', transform=ccrs.PlateCarree(), cmap='jet')
ax.set_title('$\\mathregular{TB_H}$') 
fig.colorbar(im, label='Brightness Temperature', ax=ax, fraction=0.025)

# Let's check the dependence of specular ocean emissivity on frequency and emission angle next 
sst = 300. 
sss = 34. 
uwind = vwind = 0. 
theta = 40. 
phi = 0. 
frequency = np.linspace(0.5e3, 2e3, 100)
emis_freq = oc.get_ocean_emissivity(frequency, sst, sss, uwind, vwind, theta, phi)

fig = plt.figure(figsize=(5, 7))
ax = fig.add_subplot(211)
ax.plot(frequency / 1e3, emis_freq[0, :] * sst, color='r', label='V-pol')
ax.plot(frequency / 1e3, emis_freq[1, :] * sst, color='b', label='H-pol')
ax.grid(True, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Emissivity $\\times$ SST')
ax.legend()

theta = np.linspace(0, 89.99, 100) 
frequency = np.array([1.4e3])
emis_angle = oc.get_ocean_emissivity(frequency, sst, sss, uwind, vwind, theta, phi)

ax = fig.add_subplot(212)
ax.plot(theta, emis_angle[0, :] * sst, color='r', label='V-pol')
ax.plot(theta, emis_angle[1, :] * sst, color='b', label='H-pol')
ax.grid(True, color='k', linestyle='--', alpha=0.5)
ax.set_ylabel('Emissivity $\\times$ SST')
ax.set_xlabel('Incidence Angle (degrees)')
ax.legend()

# To summarize, the ocean model can generate maps of 
# ocean surface emissivity and the corresponding brightness temperature 
# from ancillary data. Let's move on to the atmosphere module

# First we'll start with a full atmosphere model and simulate 
# brightness temperatures near the center of the 22 GHz water line
from foam import atmosphere 
atm = atmosphere.atmosphere(mode='full')
frequency = np.array([22e3])
lat_flat = np.linspace(89.5, -89.5, 180)
lon_flat = np.linspace(-179, 179.5, 360)
lon, lat = np.meshgrid(lon_flat, lat_flat)
theta = 40 * np.ones(np.shape(lon))
aTBup, aTBdn, atm_dict = atm.get_atmosphere_tb(frequency, lat, lon, angle=theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
TB = aTBup[0, :]
gl = ax.gridlines(color='k', linestyle='--', alpha=0.5, draw_labels=True)
gl.top_labels = False 
gl.right_labels = False
ax.coastlines()
im = ax.pcolormesh(lon, lat, TB, shading='auto', transform=ccrs.PlateCarree(), cmap='jet')
fig.colorbar(im, label='Brightness Temperature', ax=ax, fraction=0.025)

# We can also demonstrate some information about atmospheric propagation 
# by visualizing atmospheric weighting functions around the water line 
prwtr = 0.5 
lwtr = 5e-3 
airtemp = 290 
srfprs = 1e5

