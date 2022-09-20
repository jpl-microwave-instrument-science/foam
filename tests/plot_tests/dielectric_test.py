import numpy as np 
import matplotlib.pyplot as plt 

from foam.dielectric import *

# Plots for water dielectric 

temp1 = 275.  # * np.ones(1)
temp2 = 300.  # * np.ones(1)
sal1 = 0.  # * np.zeros(1)
sal2 = 35.  # * np.ones(1)  # 32.4

frequency = np.linspace(1, 100, 1000) * 1e3
fig, ax = plt.subplots(2, 2)

meissner_wentz = h2o_liquid_MeissnerWentz(frequency, temp1, sal1)
ellison = h2o_liquid_Ellison(frequency, temp1, sal1)
klein_swift = h2o_liquid_KleinSwift(frequency, temp1, sal1)

ax[0, 0].plot(frequency / 1e3, np.real(meissner_wentz), color='r', label='Meissner Wentz')
ax[0, 0].plot(frequency / 1e3, np.real(ellison), color='g', label='Ellison')
ax[0, 0].plot(frequency / 1e3, np.real(klein_swift), color='b', label='Klein Swift')
ax[0, 0].plot(frequency / 1e3, -np.imag(meissner_wentz), color='r', alpha=0.5, linestyle='--')
ax[0, 0].plot(frequency / 1e3, -np.imag(ellison), color='g', alpha=0.5, linestyle='--')
ax[0, 0].plot(frequency / 1e3, -np.imag(klein_swift), color='b', alpha=0.5, linestyle='--')
ax[0, 0].set_ylabel('Dielectric Constant')
ax[0, 0].legend()
ax[0, 0].set_title('T: {0:.0f} K, S: {1:.0f} psu'.format(temp1, sal1))

meissner_wentz = h2o_liquid_MeissnerWentz(frequency, temp1, sal2)
ellison = h2o_liquid_Ellison(frequency, temp1, sal2)
klein_swift = h2o_liquid_KleinSwift(frequency, temp1, sal2)

ax[0, 1].plot(frequency / 1e3, np.real(meissner_wentz), color='r', label='Meissner Wentz')
ax[0, 1].plot(frequency / 1e3, np.real(ellison), color='g', label='Ellison')
ax[0, 1].plot(frequency / 1e3, np.real(klein_swift), color='b', label='Klein Swift')
ax[0, 1].plot(frequency / 1e3, -np.imag(meissner_wentz), color='r', alpha=0.5, linestyle='--')
ax[0, 1].plot(frequency / 1e3, -np.imag(ellison), color='g', alpha=0.5, linestyle='--')
ax[0, 1].plot(frequency / 1e3, -np.imag(klein_swift), color='b', alpha=0.5, linestyle='--')
ax[0, 1].set_title('T: {0:.0f} K, S: {1:.0f} psu'.format(temp1, sal2))

meissner_wentz = h2o_liquid_MeissnerWentz(frequency, temp2, sal1)
ellison = h2o_liquid_Ellison(frequency, temp2, sal1)
klein_swift = h2o_liquid_KleinSwift(frequency, temp2, sal1)

ax[1, 0].plot(frequency / 1e3, np.real(meissner_wentz), color='r', label='Meissner Wentz')
ax[1, 0].plot(frequency / 1e3, np.real(ellison), color='g', label='Ellison')
ax[1, 0].plot(frequency / 1e3, np.real(klein_swift), color='b', label='Klein Swift')
ax[1, 0].plot(frequency / 1e3, -np.imag(meissner_wentz), color='r', alpha=0.5, linestyle='--')
ax[1, 0].plot(frequency / 1e3, -np.imag(ellison), color='g', alpha=0.5, linestyle='--')
ax[1, 0].plot(frequency / 1e3, -np.imag(klein_swift), color='b', alpha=0.5, linestyle='--')
ax[1, 0].set_ylabel('Dielectric Constant')
ax[1, 0].set_xlabel('Frequency (GHz)')
ax[1, 0].set_title('T: {0:.0f} K, S: {1:.0f} psu'.format(temp2, sal1))

meissner_wentz = h2o_liquid_MeissnerWentz(frequency, temp2, sal2)
ellison = h2o_liquid_Ellison(frequency, temp2, sal2)
klein_swift = h2o_liquid_KleinSwift(frequency, temp2, sal2)

ax[1, 1].plot(frequency / 1e3, np.real(meissner_wentz), color='r', label='Meissner Wentz')
ax[1, 1].plot(frequency / 1e3, np.real(ellison), color='g', label='Ellison')
ax[1, 1].plot(frequency / 1e3, np.real(klein_swift), color='b', label='Klein Swift')
ax[1, 1].plot(frequency / 1e3, -np.imag(meissner_wentz), color='r', alpha=0.5, linestyle='--')
ax[1, 1].plot(frequency / 1e3, -np.imag(ellison), color='g', alpha=0.5, linestyle='--')
ax[1, 1].plot(frequency / 1e3, -np.imag(klein_swift), color='b', alpha=0.5, linestyle='--')
ax[1, 1].set_xlabel('Frequency (GHz)')
ax[1, 1].set_title('T: {0:.0f} K, S: {1:.0f} psu'.format(temp2, sal2))
plt.tight_layout()

# Plot soil models 
sand = 0.3
clay = 0.2
moisture1 = 0.01
moisture2 = 0.3

fig, ax = plt.subplots(1, 2)

dobs = soil_dobson(frequency, moisture1, sand, clay, temperature=temp2)
peps = soil_peplinski(frequency, moisture1, sand, clay, temperature=temp2)
mod_dobs = soil_modified_dobson(frequency, moisture1, sand, clay)
hallikainen = soil_hallikainen(moisture1, sand, clay)
wang = soil_wang(moisture1)


ax[0].plot(frequency / 1e3, np.real(dobs), color='r', label='Dobson')
ax[0].plot(frequency / 1e3, np.real(peps), color='g', label='Peplinski')
ax[0].plot(frequency / 1e3, np.real(mod_dobs), color='b', label='Modified Dobson')
ax[0].plot(frequency / 1e3, -np.imag(dobs), color='r', alpha=0.5)
ax[0].plot(frequency / 1e3, -np.imag(peps), color='g', alpha=0.5)
ax[0].plot(frequency / 1e3, -np.imag(mod_dobs), color='b', alpha=0.5)

ax[0].set_xlabel('Frequency (GHz)')
ax[0].set_ylabel('Dielectric Constant')
ax[0].set_title('Sand: {0:.1f}, Clay: {1:.1f}, Moisture: {2:.1f}'.format(sand, clay, moisture1))

dobs = soil_dobson(frequency, moisture2, sand, clay, temperature=temp2)
peps = soil_peplinski(frequency, moisture2, sand, clay, temperature=temp2)
mod_dobs = soil_modified_dobson(frequency, moisture2, sand, clay)
hallikainen = soil_hallikainen(moisture2, sand, clay)
wang = soil_wang(moisture1)


ax[1].plot(frequency / 1e3, np.real(dobs), color='r', label='Dobson')
ax[1].plot(frequency / 1e3, np.real(peps), color='g', label='Peplinski')
ax[1].plot(frequency / 1e3, np.real(mod_dobs), color='b', label='Modified Dobson')
ax[1].plot(frequency / 1e3, -np.imag(dobs), color='r', alpha=0.5)
ax[1].plot(frequency / 1e3, -np.imag(peps), color='g', alpha=0.5)
ax[1].plot(frequency / 1e3, -np.imag(mod_dobs), color='b', alpha=0.5)
ax[1].set_xlabel('Frequency (GHz)')
ax[1].set_title('Sand: {0:.1f}, Clay: {1:.1f}, Moisture: {2:.1f}'.format(sand, clay, moisture2))
ax[1].legend()

plt.tight_layout()
