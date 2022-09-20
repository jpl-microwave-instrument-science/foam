import numpy as np 
import pandas as pd
import matplotlib 
from matplotlib import pyplot as plt 
import datetime as dt
from netCDF4 import date2num 
import cartopy.crs as ccrs

from foam.ocean import ocean
from foam.atmosphere import atmosphere
from foam.ionosphere import ionosphere


def ancillary_time_maps(): 
    """ This function generates antimated maps of ancillary data inputs
        starting on the first day of 2017 at noon. Gifs are saved in the 
        directory of execution
    """

    oc = ocean(datetime=dt.datetime(2017, 1, 1, 12), online=True, verbose=True)
    atm = atmosphere(datetime=dt.datetime(2017, 1, 1, 12), online=True, verbose=True)
    ion = ionosphere(datetime=dt.datetime(2017, 1, 1, 12), online=True, verbose=True)

    # date_list = pd.date_range(dt.datetime(2017, 1, 1, 12), periods=20).to_pydatetime().tolist()
    date_list = pd.date_range(dt.datetime(2017, 1, 1, 12), periods=3).to_pydatetime().tolist()

    lat_flat = np.linspace(89, -89, int(180 / 0.5))
    lon_flat = np.linspace(-179, 179, int(360 / 0.5))
    lon, lat = np.meshgrid(lon_flat, lat_flat)

    sst_list = []
    sss_list = []
    prwtr_list = []
    airtemp_list = []
    srfprs_list = []
    lwtr_list = []
    uwind_list = []
    vwind_list = []
    tec_list = []
    for d in date_list: 
        oc.datetime = d 
        atm.datetime = d
        ion.datetime = d
        current_time = date2num(oc.datetime, oc.time_reference)

        # Ocean
        oc.read_ocean()
        sst = oc.sst_interp((lat, lon))
        sst[sst < 260] = np.nan
        sst_list.append(sst)
        sss = oc.sss_interp((lat, lon))
        sss[sss < 30] = np.nan
        sss_list.append(sss)

        # Atmosphere
        atm.read_atmosphere()
        prwtr_list.append(atm.prwtr_interp((current_time, lat, lon)))
        lwtr_list.append(atm.lwtr_interp((current_time, lat, lon)))
        airtemp_list.append(atm.airtemp_interp((current_time, lat, lon)))
        srfprs_list.append(atm.srfpres_interp((current_time, lat, lon)))
        uwind_list.append(atm.uwind_interp((current_time, lat, lon)))
        vwind_list.append(atm.vwind_interp((current_time, lat, lon)))

        # Ionosphere
        ion.read_ionosphere()
        tec_list.append(ion.TEC_interp((current_time, lat, lon)))

    list_list = [sst_list, sss_list, airtemp_list, srfprs_list, prwtr_list, lwtr_list, uwind_list, vwind_list, tec_list]
    name_list = ['sst', 'sss', 'airtemp', 'srfprs', 'prwtr', 'lwtr', 'uwind', 'vwind', 'tec']

    for i in range(0, len(list_list)): 
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(color='k', draw_labels=True)
        extent = (-180, 180, -90, 90)
        out = [[ax.imshow(list_list[i][x], origin='upper', extent=extent, transform=ccrs.PlateCarree())] for x in range(0, len(date_list))]
        fig.colorbar(out[0][0], shrink=0.7, pad=0.1)
        anim = matplotlib.animation.ArtistAnimation(fig, out, interval=500)
        anim.save('%s.gif' % name_list[i], writer='imagemagick', fps=3, dpi=300)

    return sst_list, sss_list, tec_list, prwtr_list, airtemp_list, srfprs_list, lwtr_list, uwind_list, vwind_list
