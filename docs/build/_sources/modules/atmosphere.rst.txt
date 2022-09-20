Atmosphere Module 
=======================

The Atmosphere module defines the atmosphere object, which loads physical characteristics and defines functions relevant to the 
emission by and propagation of microwave radiation through the atmosphere. The module can be used in either 'simple' or 'full' mode, 
both of which read ancillary data from `MERRA-2 Single Level Diagnostics products <https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary>`_ 

Dev note: For now, the default MERRA set will need to be downloaded from https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXSLV.5.12.4/2005/01/MERRA2_300.tavg1_2d_slv_Nx.20050101.nc4 and manually placed in the assets folder. A cleaner way to do this (e.g. local cache) will be implemented before public release 



.. automodule:: foam.atmosphere
    :members:
    

