# FOAM
The Forward Ocean Atmosphere Microwave (FOAM) model is a radiative transfer model framework for microwave remote sensing of ocean state.
Full documentation can be found in the docs/build directory. 

## Installation and Development notes
To test FOAM, download the package from the repository and build locally using `pip install -e .`

FOAM modules download ancillary data from OpenDAP servers hosted by DAACs and other sources.
To use FOAM, the user must configure .netrc, .urs_cookies, and .dodsrc files in their home directories following the instructions at 
https://disc.gsfc.nasa.gov/data-access

Before use, FOAM needs to build a local cache. Make sure you have wget and gzip installed on your machine, and execute the following in a Python terminal 

    import foam.utils.config as config 
    config.setup_cache()
    config.get_ancillary_data()

This will setup the cache (.foam) in your home directory and download some ancillary data required to use FOAM offline. 

If you notice any issues with FOAM, please raise an issue through GitHub or e-mail Alex Akins at alexander.akins@jpl.nasa.gov

