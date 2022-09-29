# FOAM
The Forward Ocean Atmosphere Microwave (FOAM) model is a radiative transfer model and mission simulator framework for microwave remote sensing of the ocean. It was designed in the context of sea surface salinity observations, but can also be used to simulate sea surface temperature and ocean vector wind measurements
Full documentation can be found at https://foam.readthedocs.io. 

## Installation
FOAM can be installed (preferably in its own environment) using 

    pip install foam 

FOAM modules download ancillary data from OpenDAP servers hosted by DAACs and other sources.
To use FOAM, the user must configure .netrc, .urs_cookies, and .dodsrc files in their home directories following the instructions at 

https://disc.gsfc.nasa.gov/data-access

It is important to also authorize access to PODAAC, GESDISC, CDDIS, and NSIDC in the "Authorized Apps" section of your NASA Earthdata account.

Before use, FOAM needs to build a local cache. Make sure you have wget and gzip installed on your machine, and execute the following in a Python terminal 

    import foam.utils.config as config 
    config.setup_cache()
    config.get_ancillary_data()

This will setup the cache (.foam) in your home directory and download some ancillary data required to use FOAM offline. 

## Usage
Some examples of FOAM usage have been included in the package scripts/examples directory as Jupyter notebooks. 
Experiment with these notebooks to get an idea for how to use the software! 


If you notice any issues with FOAM, please raise an issue through GitHub or e-mail Alex Akins at alexander.akins@jpl.nasa.gov

&copy; 2022 Alex Akins, developed at Jet Propulsion Laboratory, California Institute of Technology 
License: Apache 2.0
