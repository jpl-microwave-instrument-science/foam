import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt 

try: 
    from astropy.wcs import WCS
except ImportError: 
    raise ImportError('User needs to install astropy to use this test')

import foam.sky as sky 
plt.style.use('Akins')

print('Showing specular galaxy')
o = sky.sky(scattered_galaxy=False)
ra = np.linspace(0, 360, 1441)
dec = np.linspace(-90, 90, 721)
rra, ddec = np.meshgrid(ra, dec)

image = o.galaxy_brightness(rra, ddec, 1420, 0)
wcs_input_dict = {'CTYPE1': 'RA', 'CUNIT1': 'deg', 'CDELT1': 0.25, 'CRPIX1': 0, 
                'CRVAL1': 0, 'NAXIS1': 1441, 'CTYPE2': 'DEC', 'CUNIT2': 'deg', 
                'CDELT2': 0.25, 'CRPIX2': 360, 'CRVAL2': 0, 'NAXIS2': 721} 
wcs_map = WCS(wcs_input_dict)

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(projection=wcs_map)
plt.imshow(image, origin='lower', cmap='viridis')  # , aspect='equal')
ax.tick_params(axis='x', which='major', pad=10)
plt.xlabel(r'Right Ascension')
plt.xlim(0, 23.9 * 60)
plt.ylabel(r'Declination')
plt.clim(0, 10)
plt.grid(b=False)
cbar = plt.colorbar(shrink=0.5, pad=0.05)
cbar.set_label('$T_B$ (K)')
fig.tight_layout(pad=5)
plt.savefig('gal_map.png', dpi=300, transparent=True)

print('Showing scattering galaxy')
o = sky.sky(scattered_galaxy=True)
ra = np.linspace(0, 360, 1441)
dec = np.linspace(-90, 90, 721)
rra, ddec = np.meshgrid(ra, dec)

image = o.galaxy_brightness(rra, ddec, 1420, 5)
wcs_input_dict = {'CTYPE1': 'RA', 'CUNIT1': 'deg', 'CDELT1': 0.25, 'CRPIX1': 0, 
                'CRVAL1': 0, 'NAXIS1': 1441, 'CTYPE2': 'DEC', 'CUNIT2': 'deg', 
                'CDELT2': 0.25, 'CRPIX2': 360, 'CRVAL2': 0, 'NAXIS2': 721} 
wcs_map = WCS(wcs_input_dict)

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(projection=wcs_map)
plt.imshow(image, origin='lower', cmap='viridis', aspect='equal')
plt.xlabel(r'Right Ascension')
plt.xlim(0, 23.9 * 60)
plt.ylabel(r'Declination')
plt.grid(b=False)
cbar = plt.colorbar(shrink=0.5, pad=0.05)
cbar.set_label('$T_B$ (K)')
fig.tight_layout(pad=5)
plt.savefig('gal_map_scat.png', dpi=300, transparent=True)
