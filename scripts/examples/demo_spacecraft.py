import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import spiceypy as spice 

import foam.spacecraft as sc 
from foam.utils.mk import manual_furnish

# In this script, we will demonstrate how to make a spacecraft object and interact with it 
# We will first make our spacecraft, following the approach in utils.sc_utils.make_aquarius()

# The Aquarius L Band instrument consisted of 3 radiometer horns pointed at a common reflector 
# We can approximate this by setting up a spacecraft object with three separate radiometers 
# The pre-launch look angles for the Aquarius instrument is given below (from an Remote Sensing Systems memo)

# Aquarius horns (pre-launch, 0.5 degrees accurate) were pointed as 
# Horn      Elevation       Azimuth
# Horn 1    25.82           9.84
# Horn 2    33.82           -15.29
# Horn 3    40.37           6.55

# The total angle for each radiometer can then be computed as follows
horn = [1, 2, 3]
nadir_angle = [25.82, 33.82, 40.37]
azimuth_angle = [9.84, -15.29, 6.55]
for i in range(len(horn)):  
    rm = spice.rotate(np.radians(azimuth_angle[i]), 3)
    rm2 = spice.rotate(np.radians(-nadir_angle[i]), 2)
    axis, angle = spice.raxisa(rm @ rm2)
    print('Horn {0}: Axis {1}, Angle {2}'.format(horn[i], axis, np.degrees(angle)))

# But first, we must initialize the spacecraft object 
# We do this using a pre-imported TLE set and convert the TLE strings 
# to SPICE-friendly elements using the spice.getelm function.
manual_furnish()
tle = ['1 37673U 11024A   21237.59766439  .00000119  00000-0  26113-4 0  9996',
       '2 37673  98.0067 245.2609 0001526  59.7629  52.4724 14.73133816548758']
epoch, elements = spice.getelm(2000, 500, tle)

# Let's simulate Aquarius shortly after it launched on June 10th, 2011 with 10 second resolution
dates = ['2010 JUN 10 12:00', '2010 JUN 12 12:00']
start_epoch = spice.str2et(dates[0])
end_epoch = spice.str2et(dates[1])
epoch_res = 10 
aq_craft = sc.spacecraft()
aq_craft.write_tle_kernels(elems=elements, tle_epoch=epoch, start_epoch=start_epoch, 
                           end_epoch=end_epoch, epoch_res=epoch_res)

# And now we can write the three radiometer horns with the pre-computed rotation axes. 
# Not that the Aquarius radiometer platform did not spin, and so the rpm arguments to write_radiometer_ck are zero
axis = np.array([0.08032594, 0.93313411, -0.35044041])
aq_craft.write_radiometer_ck(27.60, axis, 0., 'X')
axis = np.array([-0.12187662, 0.90798238, 0.40089161])
aq_craft.write_radiometer_ck(37.02, axis, 0., 'X')
axis = np.array([0.05645089, 0.98652661, -0.15355307])
aq_craft.write_radiometer_ck(40.88, axis, 0., 'X')

# We should now have a fully functioning spacecraft
# Let's try it out by plotting the sub-spacecraft tracks and radiometer look intercepts
# We're now following the approach of the sc_utils.track_viewer_2D() method 

track_dict = aq_craft.make_track_grid(start_epoch, end_epoch, epoch_res)
lons = track_dict['lon']
lats = track_dict['lat']
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.plot(lons, lats, transform=ccrs.PlateCarree(), linestyle='none', marker='.', color='k', 
        label='Spacecraft tracks', ms=5)
obs_dict = aq_craft.make_obs_grid(start_epoch, end_epoch, epoch_res)
lons = obs_dict['lon']
lats = obs_dict['lat']
shape = np.shape(lons)
for i in range(shape[0]): 
    ax.plot(lons[i, :], lats[i, :], transform=ccrs.PlateCarree(), linestyle='none', marker='.',
            label='Radiometer {}'.format(i), ms=5)
ax.set_extent([-180, 180, -90, 90])

# Finally, its a good idea to call the spacecraft.close() method, which unloads the SPICE kernel pool. 
aq_craft.close()
