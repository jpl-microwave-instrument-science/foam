import os
import sys
import shutil
import tarfile

home = os.environ['HOME']
dir_path = os.path.dirname(sys.modules['foam'].__file__)
asset_path = os.path.join(dir_path, 'assets')
cache_path = os.path.join(home, '.foam')


def setup_cache(): 
    print('Setting up cache in home directory')
    home = os.environ['HOME']
    cache_path = os.path.join(home, '.foam')
    exists = os.path.exists(cache_path)
    if not exists: 
        os.mkdir(cache_path)
        paths = ['atmosphere', 'galaxy', 'ionosphere', 'landmask', 'ocean', 'spice']
        for p in paths: 
            make_path = os.path.join(cache_path, p)
            os.mkdir(make_path)


def get_all_ancillary_data(): 
    print('Downloading ancillary data')
    print('Ensure all Earthdata configuration files are set up appropriately in your home directory')
    print('See https://disc.gsfc.nasa.gov/data-access for details')
    print('Requires wget and gzip to be installed')

    cookie_dir = os.path.join(home, '.urs_cookies')
    wget_args = ' --load-cookies {0} --save-cookies {0} --auth-no-challenge=on --keep-session-cookies --content-disposition '.format(cookie_dir)

    print('Downloading atmosphere data from 1/1/2005')
    urlstring = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXSLV.5.12.4/2005/01/MERRA2_300.tavg1_2d_slv_Nx.20050101.nc4'
    save_path = os.path.join(cache_path, 'atmosphere')
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)

    print('Downloading galactic brightness temperature map')
    urlstring = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/aquarius/L3/mapped/galaxy/2018/TBSkyLbandAquarius.h5'
    save_path = os.path.join(cache_path, 'galaxy')
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)

    print('Downloading ionosphere data from 1/1/2005')
    urlstring = 'https://cddis.nasa.gov/archive/gnss/products/ionex/2005/001/jplg0010.05i.Z'
    save_path = os.path.join(cache_path, 'ionosphere')
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    os.system('gzip -d {}'.format(os.path.join(save_path, 'jplg0010.05i.Z')))

    print('Downloading EASE Grid landmask')
    urlstring = 'ftp://sidads.colorado.edu/pub/DATASETS/nsidc0609_loci_ease2/global/EASE2_M03km.LOCImask_land50_coast0km.11568x4872.bin'
    save_path = os.path.join(cache_path, 'landmask')
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    urlstring = 'ftp://sidads.colorado.edu/pub/tools/easegrid2/gridloc.EASE2_M03km.tgz'
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    quick_path = os.path.join(save_path, 'gridloc.EASE2_M03km.tgz')
    my_tar = tarfile.open(quick_path)
    my_tar.extractall(save_path)
    my_tar.close()
    os.system('rm {}'.format(quick_path))

    print('Downloading ocean temperature and salinity maps from 1/1/2018')
    urlstring = 'https://opendap.jpl.nasa.gov/opendap/SalinityDensity/smap/L3/JPL/V5.0/8day_running/2018/001/SMAP_L3_SSS_20180105_8DAYS_V5.0.nc'
    save_path = os.path.join(cache_path, 'ocean')
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    urlstring = 'https://opendap.jpl.nasa.gov/opendap/hyrax/OceanTemperature/ghrsst/data/GDS2/L4/GLOB/ABOM/GAMSSA/v1.0/2018/001/20180101120000-ABOM-L4_GHRSST-SSTfnd-GAMSSA_28km-GLOB-v02.0-fv01.0.nc'
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)

    get_spice_kernels()

    # World Magnetic Model coefficients are small enough to include in the package
    source = os.path.join(asset_path, 'magneticfield')
    dest = os.path.join(cache_path, 'magneticfield')
    shutil.copytree(source, dest)


def get_spice_kernels(): 

    cookie_dir = os.path.join(home, '.urs_cookies')
    wget_args = ' --load-cookies {0} --save-cookies {0} --auth-no-challenge=on --keep-session-cookies --content-disposition '.format(cookie_dir)

    print('Downloading SPICE kernels') 
    urlstring = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp'
    save_path = os.path.join(cache_path, 'spice')
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    urlstring = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de421_1900-2050.bpc'
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    urlstring = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc'
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    urlstring = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc'
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    urlstring = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/geophysical.ker'
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    urlstring = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/latest_leapseconds.tls'
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    urlstring = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/planets/earth_assoc_itrf93.tf'
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
    urlstring = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_080317.tf'
    os.system('wget -P {0}'.format(save_path) + wget_args + urlstring)
