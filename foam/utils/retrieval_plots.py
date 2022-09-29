import warnings
import numpy as np 
import scipy.constants as spc 
import scipy.interpolate as spi 
import matplotlib.pyplot as plt 


def plot_snapshot_dsss(out_dict, anc_dict, color='tab:blue', label=None, spline=True): 
    ret_sss = out_dict['sss']
    sst = anc_dict['sst']
    sss = anc_dict['sss']
    windspd = anc_dict['windspd']

    mask = np.isnan(ret_sss) | np.isnan(sst) | np.isnan(sss) | np.isnan(windspd)
    mask = ~mask
    ret_sss = ret_sss[mask]
    sst = sst[mask]
    sss = sss[mask]
    windspd = windspd[mask]

    x = sst - spc.zero_Celsius
    xout = np.arange(0, np.ceil(np.nanmax(x)), 1)
    yout = np.zeros(len(xout))
    syout = np.zeros(len(xout))
    y = ret_sss - sss 

    dex = np.digitize(x, xout, right=True)
    for i in range(len(xout)): 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            yout[i] = np.nanmean(y[dex == i])
            syout[i] = np.nanstd(y[dex == i])
    yout = np.nan_to_num(yout)
    syout = np.nan_to_num(syout)
    spl = spi.UnivariateSpline(xout, syout, k=4)
    plt.figure(1)
    plt.plot(xout, syout, linestyle='none', marker='o', color=color, alpha=0.75, label=label)
    if spline: plt.plot(xout, spl(xout), color=color)
    plt.xlabel('SST (C)')
    plt.ylabel('dSSS (psu)')

    x = sss
    xout = np.arange(30, np.floor(np.nanmax(x)), 1 / 3)
    yout = np.zeros(len(xout))
    syout = np.zeros(len(yout))
    y = ret_sss - sss 
    dex = np.digitize(x, xout, right=True)
    for i in range(len(xout)): 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            yout[i] = np.nanmean(y[dex == i])
            syout[i] = np.nanstd(y[dex == i])
    yout = np.nan_to_num(yout)
    syout = np.nan_to_num(syout)
    spl2 = spi.UnivariateSpline(xout, syout, k=4)
    plt.figure(2)
    plt.plot(xout, syout, linestyle='none', marker='o', color=color, alpha=0.75, label=label)
    if spline: plt.plot(xout, spl2(xout), color=color)
    plt.xlabel('SSS (psu)')
    plt.ylabel('dSSS (psu)')

    x = windspd
    xout = np.arange(np.ceil(np.nanmin(x)), np.ceil(np.nanmax(x)), 1)
    yout = np.zeros(len(xout))
    syout = np.zeros(len(xout))
    y = ret_sss - sss 
    dex = np.digitize(x, xout, right=True)
    for i in range(len(xout)): 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            yout[i] = np.nanmean(y[dex == i])
            syout[i] = np.nanstd(y[dex == i])
    yout = np.nan_to_num(yout)
    syout = np.nan_to_num(syout)
    spl3 = spi.UnivariateSpline(xout, syout, k=4)
    plt.figure(3)
    plt.plot(xout, syout, linestyle='none', marker='o', color=color, alpha=0.75, label=label)
    if spline: plt.plot(xout, spl3(xout), color=color)
    plt.xlabel('Wind speed (m/s)')
    plt.ylabel('dSSS (psu)')


def plot_binned_dsss(out_dict, anc_dict, color='tab:blue', label=None, spline=True): 
    ret_sss = out_dict['sss']
    sst = anc_dict['sst']
    sss = anc_dict['sss']
    windspd = anc_dict['windspd']
    lat = anc_dict['lat']
    lon = anc_dict['lon']

    lat_bins = np.arange(-90, 90, 0.5)
    lon_bins = np.arange(-180, 180, 0.5)

    count_hist, xe_ye = np.histogram2d(x=lat, y=lon, bins=(lat_bins, lon_bins))
    weight_hist, xe_ye = np.histogram2d(x=lat, y=lon, bins=(lat_bins, lon_bins), weights=ret_sss)
    out_dict['sss'] = weight_hist / count_hist
    weight_hist, xe_ye = np.histogram2d(x=lat, y=lon, bins=(lat_bins, lon_bins), weights=sss)
    anc_dict['sss'] = weight_hist / count_hist
    weight_hist, xe_ye = np.histogram2d(x=lat, y=lon, bins=(lat_bins, lon_bins), weights=sst)
    anc_dict['sst'] = weight_hist / count_hist
    weight_hist, xe_ye = np.histogram2d(x=lat, y=lon, bins=(lat_bins, lon_bins), weights=windspd)
    anc_dict['windspd'] = weight_hist / count_hist

    plot_snapshot_dsss(out_dict, anc_dict, color=color, label=label, spline=spline)


