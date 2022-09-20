Sky Module 
=======================

The Sky module defines the sky object, which defines models for microwave emission of the sun, moon, cosmic microwave background, and galaxy continuum emission. Static brightness temperatures are assumed for the moon (275 K) and the cosmic microwave background (2.73 K). The sun emission model is from the `IPN Progress report <https://ipnpr.jpl.nasa.gov/progress_report/42-175/175E.pdf>`_ by Ho et al. 2008 with an 11 year period and a 100000 K mean brightness temperature.
The sun and moon are currently treated as reflecting from a specular surface, although this may limit accuracy at higher incidence angles. This behavior may be changed in the future. 

The L Band galactic brightness temperature map of Dinnat et al. is included and can be used to model specular reflection of galactic emission from the ocean surface. The frequency dependence of this emission is :math:`T_B = T_{B,1.4 \textrm{GHz}} \left(\frac{f}{1.4 \textrm{GHz}}\right)^{-2.7}`. An approximation to scattering of galactic emission from a wind-roughened ocean surface is also included. This map was generated using the :code:`form_galaxy_tables` utility using the geometric optics approach described in the Aquarius Algorithm Theoretical Basis Document, and a description of this can be found in the utils documentation. Since galactic map described in the ATBD was generated specifically for Aquarius, a map of scattered emission in the nadir direction at the J2000 epoch was generated as a function of right ascension, declination, and wind speed to maintain generality. This scattered emission map is then treated as the true galaxy for specular ocean reflection.  

API
------------------------------------
.. automodule:: foam.sky
    :members:
    

