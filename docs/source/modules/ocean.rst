Ocean Module 
=======================

The :code:`ocean` module defines the ocean object, which loads physical characteristics and defines functions relevant to the 
emission by and reflection of microwave radiation by the ocean surface. The :code:`ocean` module contains several different flavors of specular and wind-roughened ocean surface models. 

The :code:`ocean.ocean` base class returns a specular surface if :code:`mode='flat'` and returns a wind-roughened surface approximation if :code:`mode='rough'`. The wind-roughened model function is an interpolation of geophysical model functions developed by Remote Sensing Systems (Meissner and Wentz 2012, 2014). These model functions were developed at 1.4, 6.8, 10.7, 18.7, and 37 GHz from satellite microwave radiometer measurements, and they are extrapolated to lower frequencies assuming a zero intercept. 

The :code:`ocean.fastem` class implements the FASTEM-5 and FASTEM-6 ocean emissivity models developed by the European Centre for Medium-Range Weather Forecasts (Liu et al 2011, Kazumori and English 2016), which are numerical approximations to a two-scale emissivity model. These model are applicable at frequencies from 1.4 to 410 GHz, and are preferred for high-frequency sensing. 

The :code:`ocean.two_scale` implements a full two-scale model emissivity model originally developed by Simon Yueh at JPL in his 1997 paper. The default ocean surface roughness spectrum is that of Durden and Vesecky, but several parameterizations are included from the work of Paul Hwang at the Naval Research Laboratory. This surface emissivity model is the most computationally expensive. 

    **Development note**: *The current implementation of the two-scale model is relatively computationally inefficient, and its use is not recommended for large simulations. Future versions of the software should implement a revised two-scale model with improved performance* 

Ocean ancillary data sources are discussed in the :code:`utils.reader` module documentation. The ocean object can also be specified using different dielectric constant modules, as discussed in the :code:`dielectric` module. 

Setting Up an Ocean object
------------------------------

:code:`ocean` objects can be initialized using different ancillary data sources and dielectric constant modules, as discussed in the :code:`utils.reader` and :code:`dielectric` module documentation. An example initialization is shown below. 

.. code-block:: python
    :linenos:

    from foam.ocean import ocean
    import foam.dielectric as dielectric 

    # Pull data from online 
    oc = ocean(datetime='2015-01-01', mode='rough', online=True, 
               sst_reader=reader.GHRSSTReader, sst_reader_kwargs={'version': 'MUR'},
               sss_reader=reader.OISSSReader, dielectric=dielectric.h2o_liquid_KleinSwift)

    # Use data from cache 
    # or provide your own file using sst_file and sss_file 
    oc = ocean(datetime='2018-01-01', mode='rough', online=False
               dielectric=dielectric.h2o_liquid_Boutin)

After initializing an ocean object, you can compute ocean surface emissivity :code:`ocean.get_ocean_emissivity` and brightness temperature :code:`ocean.get_ocean_TB`

API
------------------------------------
.. automodule:: foam.ocean
    :members:
    

