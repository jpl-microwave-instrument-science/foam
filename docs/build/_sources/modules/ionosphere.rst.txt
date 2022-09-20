Ionosphere Module 
=======================

The Ionosphere module defines the ionosphere object, which loads physical characteristics and defines functions relevant to the propagation of polarimetric microwave radiation through the ionosphere. Currently, this module is only used to determine the Faraday rotation angle for a given magnetic field, ionospheric TEC, and observing frequency. 

Ionospheric TEC maps can either be loaded from ancillary data such as `CDDIS GNSS TEC products <https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/atmospheric_products.html>`_ or generated using the International Reference Ionosphere (via the iri2016 package). The iri2016 package is currently an optional dependency. This module also computes the magnetic field strength using a Python adaptation of the World Magnetic Model written by Christopher Weiss https://github.com/cmweiss/geomag. Due to its relatively small size, the geomag package has been incorporated into FOAM directly (i.e. no pip install dependency). It has also been modified to support numpy arrays. 


API
------------------------------------
.. automodule:: foam.ionosphere
    :members:
    

