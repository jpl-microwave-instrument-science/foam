import os
import re
import warnings
import numpy as np
import scipy.interpolate as spi 
import scipy.constants as spc
import pandas as pd 
import xarray as xr 
from cftime import num2date, date2num


class Reader(): 
    """ Base class for reading from FOAM ancillary data sources 

        :param datetime: Single or tuple of Python datetime object or time string 
        :param online: Reads from OpenDAP servers if True
        :param file: Reads a single, user-provided file if online is False
    
    """
    time_reference = 'seconds since 2000-01-01 12:00:0.0'  # J2000

    def __init__(self, datetime, online=True, file=None): 
        
        self.datetime = pd.to_datetime(datetime)
        self.online = online 
        self.file = file 

    def expand_date(self, freq):
        # Check if iterator
        try: 
            iter(self.datetime)
        except TypeError: 
            self.datetime = pd.to_datetime([self.datetime, self.datetime + pd.Timedelta(freq)])
        
        # Check if length 1 
        if len(self.datetime) == 1: 
            self.datetime = pd.to_datetime([*self.datetime, *self.datetime + pd.Timedelta(freq)])
        elif len(self.datetime) > 2: 
            raise ValueError('Too many datetimes in input')

        dr = pd.date_range(*self.datetime, freq=freq)

        # Now check if the output dr is 1 
        if len(dr) == 1: 
            dr = pd.to_datetime([*dr, *dr + pd.Timedelta(freq)])

        return dr
        

# Ocean readers 


class GHRSSTReader(Reader): 
    """ Group for High Resolution Sea Surface Temperature (GHRSST)
        L4 SST blended analysis data with 1 day time resolution
        Several flavors are available: 
        MUR: JPL Multiscale Ultrahigh Resolution 
            0.25 deg: MUR
            0.01 deg: MUR_highres
        GAMSSA: ABOM product
            0.25 deg: GAMSSA
        OSTIA: UKMO product 
            0.05 deg: OSTIA
        AVHRR_OI: NCEI product 
            0.25 deg, 1981-2020: NCEI_v2
            0.25, 2016-2022: NCEI_v2.1

        :param version: See above 

    """ 
    def __init__(self, datetime, version='MUR', **kwargs): 
        self.version = version 
        super().__init__(datetime, **kwargs)

    def read(self, return_dataset=False): 
        
        dataset = self.get_dataset()
        lat = dataset['lat'].values
        lon = dataset['lon'].values
        time = dataset['time'].values
        # time_date = num2date(time, dataset['time'].units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        time_date = pd.to_datetime(time).to_pydatetime()
        standard_time = date2num(time_date, self.time_reference)
        data = dataset['analysed_sst'].values

        # Scipy interpolators don't like single dimensions
        if len(standard_time) == 1: 
            standard_time = np.concatenate([standard_time, standard_time + 1])
            data = np.concatenate([data, data])
        
        sst_interp = spi.RegularGridInterpolator((standard_time, lat, lon), data, bounds_error=False, fill_value=0)
        
        if return_dataset:
            return sst_interp, dataset
        else:
            dataset.close()
            return sst_interp

    def get_dataset(self): 

        dr = self.expand_date(freq='1D')
        # Aggregate source
        if self.online:
            if self.version == 'MUR': 
                urlstring = 'https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/MUR25-JPL-L4-GLOB-v04.2.nc'
            elif self.version == 'MUR_highres': 
                urlstring = 'https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/MUR-JPL-L4-GLOB-v4.1.nc'
            elif self.version == 'GAMSSA':
                urlstring = 'https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/GAMSSA_28km-ABOM-L4-GLOB-v01.nc'
            elif self.version == 'OSTIA': 
                urlstring = 'https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/OSTIA-UKMO-L4-GLOB-v2.0.nc'
            elif self.version == 'NCEI_v2': 
                urlstring = 'https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/AVHRR_OI-NCEI-L4-GLOB-v2.0.nc'
            elif self.version == 'NCEI_v2.1': 
                urlstring = 'https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/AVHRR_OI-NCEI-L4-GLOB-v2.1.nc'
            else: 
                warnings.warn('Improper version, defaulting to MUR')
                urlstring = 'https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/MUR25-JPL-L4-GLOB-v04.2.nc'
            dataset = xr.open_dataset(urlstring)
        else: 
            dataset = xr.open_dataset(self.file)
            dr = dataset['time'].values
        
        if (dr < dataset['time'][0].values).any() or (dr > dataset['time'][-1].values).any():
            warnings.warn('A value in date range is outside of the valid range for the GHRSST data product')
        dataset = dataset.sel(time=dr, method='nearest')

        return dataset


class SMAPSalinityReader(Reader): 
    """ SMAP L3 produced by JPL 
        SSS data with 1 day time resolution
        and 0.25 deg. grid resolution
    """

    def read(self):
     
        dataset = self.get_dataset()
        lat = dataset['latitude'].values
        lon = dataset['longitude'].values
        time = dataset['times'].values
        time_date = pd.to_datetime(time).to_pydatetime()
        standard_time = date2num(time_date, self.time_reference)
        data = dataset['smap_sss'].values

        # Scipy interpolators don't like single dimensions
        if len(standard_time) == 1: 
            # Arbitrarily padding time by a week in either direction 
            # Due to coarse time resolution of data
            standard_time = np.concatenate([standard_time - (7 * 86.4e3), standard_time + (7 * 86.4e3)])
            data = np.concatenate([data[np.newaxis], data[np.newaxis]])

        sss_interp = spi.RegularGridInterpolator((standard_time, lat, lon), data, bounds_error=False, fill_value=0)

        dataset.close()
        return sss_interp

    def get_dataset(self): 
        dr = self.expand_date(freq='1D')
        if self.online: 
            urlstring = 'https://thredds.jpl.nasa.gov/thredds/dodsC/ncml_aggregation/SalinityDensity/smap/aggregate__SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5.ncml'
            dataset = xr.open_dataset(urlstring)
            if (dr < dataset['times'][0].values).any() or (dr > dataset['times'][-1].values).any():
                warnings.warn('A value in date range is outside of the valid range for the SMAP salinity data product')
            dataset = dataset.sel(times=dr, method='nearest')
            dataset = dataset.reindex(latitude=dataset.latitude[::-1])

        else: 
            # Single SMAP netcdf format is significantly different
            dataset = xr.open_dataset(self.file)
            dataset = dataset.rename({'time': 'times'})
            dataset = dataset.reindex(latitude=dataset.latitude[::-1])
            
        return dataset


class AquariusSalinityReader(Reader): 
    """ Aquarius L3 produced by JPL 
        SSS data with 1 day time resolution and 1 deg. grid resolution
    """

    def read(self, return_dataset=False): 
   
        dataset = self.get_dataset()
        lat = dataset['lat'].values
        lon = dataset['lon'].values
        time = dataset['time'].values
        time_date = pd.to_datetime(time).to_pydatetime()
        standard_time = date2num(time_date, self.time_reference)
        sss_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['sss_cap'].values, bounds_error=False, fill_value=0)
        
        if return_dataset: 
            return sss_interp, dataset
        else: 
            dataset.close()
            return sss_interp

    def get_dataset(self): 
        dr = self.expand_date(freq='1D')

        if self.online:
            # Call will fail without #fillmismatch tag
            urlstring = 'https://thredds.jpl.nasa.gov/thredds/dodsC/ncml_aggregation/SalinityDensity/aquarius/aggregate__AQUARIUS_L3_SSS_CAP_7DAY_V5.ncml#fillmismatch'
            dataset = xr.open_dataset(urlstring)
        else: 
            dataset = xr.open_dataset(self.file)
            dr = dataset['time'].values
         
        if (dr < dataset['time'][0].values).any() or (dr > dataset['time'][-1].values).any():
            warnings.warn('A value in date range is outside of the valid range for the Aquarius salinity data product')
        dataset = dataset.sel(time=dr, method='nearest')
        dataset['lon'] = (dataset['lon'] + 180) % 360 - 180  # Convert lon to -180 to 180 convention
        dataset = dataset.reindex(lon=np.sort(dataset.lon[::-1]))

        return dataset


class OISSSReader(Reader):
    """ Multi-mission Optimally Interpolated SSS analysis 
    """ 

    def read(self): 
        dataset = self.get_dataset()
        lat = dataset['latitude'].values
        lon = dataset['longitude'].values
        time = dataset['time_agg'].values
        time_date = pd.to_datetime(time).to_pydatetime()
        standard_time = date2num(time_date, self.time_reference)
        sss_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['sss'].values, bounds_error=False, fill_value=0)
        
        dataset.close()
        return sss_interp

    def get_dataset(self): 
        dr = self.expand_date(freq='4D')
        if self.online:
            # Call will fail without #fillmismatch tag
            urlstring = 'https://thredds.jpl.nasa.gov/thredds/dodsC/SalinityDensity/OISSS_L4_multimission_7day_v1.nc#fillmismatch'
            dataset = xr.open_dataset(urlstring)
        else: 
            dataset = xr.open_dataset(self.file)
            dr = dataset['time_agg'].values

        if (dr < dataset['time_agg'][0].values).any() or (dr > dataset['time_agg'][-1].values).any():
            warnings.warn('A value in date range is outside of the valid range for the OISSS salinity data product')
        dataset = dataset.sel(time_agg=dr, method='nearest')

        return dataset


class HYCOMReader(Reader):
    """ HYCOM GOFS SST and SSS analysis 
        with 3 hr and 0.08 deg lon x 0.04 deg lat resolution.  
        
        :param sea_ice: Toggle return of sea ice fraction (default False)
        :param lat_bounds: Tuple of latitude bounds for subsetting 
        :param lon_bounds: Tuple of longitude bounds for subsetting. 
                           Note that HYCOM longitudes are indexed from 0 to 360 

        **Development note**: 
        HYCOM files are large, and queries for a large time range may fail. Preliminary 
        attempts to access smaller chunks with dask have not been successful. In this case, 
        it is recommended that the user make calls to the HYCOMReader.get_dataset method 
        covering several shorter blocks, merging those files on system using xarray, and 
        using that merged file as an input in offline mode. 
        This may be implemented in the future. 
    """  

    def __init__(self, datetime, sea_ice=False, 
                 lat_bounds=(-90, 90), lon_bounds=(0, 360), **kwargs): 
        
        self.sea_ice = sea_ice
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        super().__init__(datetime, **kwargs)

    def read(self):
        dataset = self.get_dataset() 
        # Change lon convention to -180 -> 180
        dataset['lon'] = (dataset['lon'] + 180) % 360 - 180
        dataset = dataset.reindex(lon=np.sort(dataset.lon[::-1]))
        lat = dataset['lat'].values
        lon = dataset['lon'].values
        time = dataset['time'].values
        # Filter for unique times
        time, time_mask = np.unique(time, return_index=True)
        time_date = pd.to_datetime(time).to_pydatetime()
        standard_time = date2num(time_date, self.time_reference)
        water_temp = dataset['water_temp'].values[time_mask, ...] + spc.zero_Celsius
        salinity = dataset['salinity'].values[time_mask, ...]
        sst_interp = spi.RegularGridInterpolator((standard_time, lat, lon), water_temp, bounds_error=False, fill_value=0)
        sss_interp = spi.RegularGridInterpolator((standard_time, lat, lon), salinity, bounds_error=False, fill_value=0)
        
        dataset.close()
        return sst_interp, sss_interp 

    def get_dataset(self): 
        dr = self.expand_date(freq='3H')
        if (dr.year < 1994).any():
            raise ValueError("HYCOM products aren't available before 1994")
        elif (dr < '2018-12-04').any():
            warnings.warn('HYCOM data prior to 2018-12-04 do not contain sea ice fraction')

        if self.online:
            set_list = []
            dv = ['tau', 'water_u_bottom', 'water_v_bottom', 'water_temp_bottom',
                 'salinity_bottom']
            if self.sea_ice: 
                dv_ice = ['tau', 'sst', 'sss', 'ssu', 'ssv', 'sih', 'siu', 
                          'siv', 'surtx', 'surty']  # Everything but sea ice fraction

            # From 1994 to 2015 
            yrs = np.unique(dr.year)
            yrs = yrs[(yrs < 2015)]
            for yr in yrs: 
                urlstring = 'https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/{}'.format(yr)
                dataset = xr.open_dataset(urlstring, decode_times=False, drop_variables=dv)
                new_times = num2date(dataset.time, dataset.time.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                dataset['time'] = pd.DatetimeIndex(new_times)
                dataset = dataset.sel(depth=0., lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
                set_list.append(dataset)
            # and beyond
            if ((dr >= '2014-07-01') & (dr <= '2016-04-30')).any(): 
                urlstring = 'https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_56.3'
                dataset = xr.open_dataset(urlstring, decode_times=False, drop_variables=dv)
                new_times = num2date(dataset.time, dataset.time.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                dataset['time'] = pd.DatetimeIndex(new_times)
                dataset = dataset.sel(depth=0., time=slice("2014-07-01", "2016-04-30"), 
                                      lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
                set_list.append(dataset)
            if ((dr >= '2016-05-01') & (dr <= '2017-01-31')).any(): 
                urlstring = 'https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.2'
                dataset = xr.open_dataset(urlstring, decode_times=False, drop_variables=dv)
                new_times = num2date(dataset.time, dataset.time.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                dataset['time'] = pd.DatetimeIndex(new_times)
                dataset = dataset.sel(depth=0., time=slice("2016-05-01", "2017-01-31"), 
                                      lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
                set_list.append(dataset)
            if ((dr >= '2017-02-01') & (dr <= '2017-05-31')).any(): 
                urlstring = 'https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.8'
                dataset = xr.open_dataset(urlstring, decode_times=False, drop_variables=dv)
                new_times = num2date(dataset.time, dataset.time.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                dataset['time'] = pd.DatetimeIndex(new_times)
                dataset = dataset.sel(depth=0., time=slice("2017-02-01", "2017-05-31"), 
                                      lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
                set_list.append(dataset)
            if ((dr >= '2017-06-01') & (dr <= '2017-09-30')).any(): 
                urlstring = 'https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.7'
                dataset = xr.open_dataset(urlstring, decode_times=False, drop_variables=dv)
                new_times = num2date(dataset.time, dataset.time.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                dataset['time'] = pd.DatetimeIndex(new_times)
                dataset = dataset.sel(depth=0., time=slice("2017-06-01", "2017-09-30"), 
                                      lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
                set_list.append(dataset)
            if ((dr >= '2017-10-01') & (dr <= '2017-12-31')).any(): 
                urlstring = 'https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.9'
                dataset = xr.open_dataset(urlstring, decode_times=False, drop_variables=dv)
                new_times = num2date(dataset.time, dataset.time.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                dataset['time'] = pd.DatetimeIndex(new_times)
                dataset = dataset.sel(depth=0., time=slice("2017-10-01", "2017-12-31"), 
                                      lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
                set_list.append(dataset)
            if ((dr >= '2018-01-01') & (dr <= '2018-12-03')).any(): 
                urlstring = 'https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0'
                dataset = xr.open_dataset(urlstring, decode_times=False, drop_variables=dv)
                new_times = num2date(dataset.time, dataset.time.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                dataset['time'] = pd.DatetimeIndex(new_times)
                dataset = dataset.sel(depth=0., time=slice("2018-01-01", "2018-12-03"), 
                                      lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
                set_list.append(dataset)
            if (dr >= '2018-12-04').any():
                urlstring = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'
                dataset = xr.open_dataset(urlstring, decode_times=False, drop_variables=dv)
                new_times = num2date(dataset.time, dataset.time.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                dataset['time'] = pd.DatetimeIndex(new_times)
                thresh_time = np.datetime64("2018-12-04")
                dataset = dataset.sel(depth=0., lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
                if dr[0] < thresh_time: 
                    dataset = dataset.sel(time=slice("2018-12-04", dr[-1].strftime('%Y-%m-%d %X')))
                else: 
                    dataset = dataset.sel(time=slice(dr[0].strftime('%Y-%m-%d %X'), dr[-1].strftime('%Y-%m-%d %X')))
    
                if self.sea_ice: 
                    dataset_ice = xr.open_dataset(urlstring + '/ice', decode_times=False, drop_variables=dv_ice)
                    new_times = num2date(dataset_ice.time, dataset_ice.time.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                    dataset_ice['time'] = pd.DatetimeIndex(new_times)
                    if dr[0] < thresh_time: 
                        dataset_ice = dataset_ice.sel(time=slice("2018-12-04", dr[-1].strftime('%Y-%m-%d %X')), 
                                                      lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
                    else: 
                        dataset_ice = dataset_ice.sel(time=slice(dr[0].strftime('%Y-%m-%d %X'), dr[-1].strftime('%Y-%m-%d %X')), 
                                                      lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
        
                    dataset = xr.combine_by_coords([dataset, dataset_ice])
                set_list.append(dataset)
            if len(set_list) > 1: 
                dataset = xr.combine_by_coords(set_list)
        else: 
            dataset = xr.open_dataset(self.file)
        
        dataset = dataset.sel(time=dr, method='nearest')

        return dataset


# Atmosphere readers 

class NCEPReader(Reader): 
    """ Reads NCEP atmospheric data in 2D (default) or 3D
        with 6 hour time resolution 

        :param dimension: Either '2D' (default) or '3D'
        :param lat_bounds: Tuple of latitude bounds for subsetting 
        :param lon_bounds: Tuple of longitude bounds for subsetting.
                           Note that HYCOM longitudes are indexed from 0 to 360 
        :param use_dask: Toggles dask-based loading (default False).
                         Initial tests suggests dask is slower for small subsets, 
                         but may be necessary for large ones

        **Development note**: 
        This method accesses NCEP data by chaining together the annual aggregates.
        This seems quicker so far than accessing the multi-year parameter aggregates
    """

    def __init__(self, datetime, dimension='2D', 
                 lat_bounds=(-90, 90), lon_bounds=(0, 360), use_dask=False, **kwargs): 
        
        self.dimension = dimension
        self.use_dask = use_dask
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        super().__init__(datetime, **kwargs)

    def read(self, return_dataset=False): 
        dataset = self.get_dataset()
        dataset = dataset.compute()  # Load into memory if it isn't already
        dataset['lon'] = (dataset['lon'] + 180) % 360 - 180
        dataset = dataset.reindex(lon=np.sort(dataset.lon))
        dataset = dataset.reindex(lat=np.sort(dataset.lat))
        lat = dataset['lat'].values
        lon = dataset['lon'].values
        time = dataset['time'].values
        time_date = pd.to_datetime(time).to_pydatetime()
        standard_time = date2num(time_date, self.time_reference)
        if self.dimension == '2D': 
            uwind_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['uwnd'].values, bounds_error=False, fill_value=None)
            vwind_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['vwnd'].values, bounds_error=False, fill_value=None)
            prwtr_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['pr_wtr'].values, bounds_error=False, fill_value=None)
            airtemp_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['air'].values, bounds_error=False, fill_value=None)
            airpres_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['pres'].values, bounds_error=False, fill_value=None)
        else: 
            dataset = self.make_height_index(dataset)
            level = dataset['level'].values / 1e3  # to km 
            uwind_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['uwnd'].values, bounds_error=False, fill_value=None)
            vwind_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['vwnd'].values, bounds_error=False, fill_value=None)
            prwtr_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['shum'].values, bounds_error=False, fill_value=None)
            airtemp_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['air'].values, bounds_error=False, fill_value=None)
            airpres_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['pres'].values * 100, bounds_error=False, fill_value=None)

        # No column liquid water is provided, so a None must be added to match formatting 
        if return_dataset: 
            return airtemp_interp, airpres_interp, uwind_interp, vwind_interp, prwtr_interp, None, dataset 
        else: 
            dataset.close()
            return airtemp_interp, airpres_interp, uwind_interp, vwind_interp, prwtr_interp, None

    def get_dataset(self): 
        dr = self.expand_date(freq='6H')
        if (dr.year < 1948).any():
            raise ValueError("NCEP/NCAR products aren't available before 1948")

        if self.online: 
            set_list = []
            if self.dimension == '2D': 
                params = ['air.sig995.', 'pr_wtr.eatm.', 'pres.sfc.', 'uwnd.sig995.', 'vwnd.sig995.']
                urlstring = 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/surface/'
                yrs = np.unique(dr.year)
                for y in yrs: 
                    for p in params: 
                        url = urlstring + p + str(y) + '.nc'
                        set_list.append(url)
            elif self.dimension == '3D': 
                params = ['air.', 'shum.', 'uwnd.', 'vwnd.', 'hgt.']
                urlstring = 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/pressure/'
                yrs = np.unique(dr.year)
                for y in yrs: 
                    for p in params: 
                        url = urlstring + p + str(y) + '.nc'
                        set_list.append(url)
            else: 
                raise ValueError("Dimension should either be '2D' or '3D'")

            if self.use_dask:
                dataset = xr.open_mfdataset(set_list, chunks='auto')

                # Sel pattern breaks dask loading for time 
                # Uncomment this once xarray fixes this functionality 
                # dataset = dataset.sel(time=dr, method='nearest')
                time = dataset['time'].values
                time = pd.to_datetime(time).values
                # Even explicit location based indexing fails! 
                # mask = (time > dr[0]) & (time < dr[-1])
                # dataset = dataset.loc[dict(time=time[mask])]
                # The only way I've found to make this work so far is via sliced indexing
                lower = time[time >= dr[0]][0]
                upper = time[time <= dr[-1]][-1]
                dataset = dataset.isel(time=slice(int(np.where(time == lower)[0]), 
                                                  int(np.where(time == upper)[0])))
                dataset = dataset.sel(lat=slice(*self.lat_bounds[::-1]), lon=slice(*self.lon_bounds))

            else: 
                dsl = []
                for sl in set_list: 
                    ds = xr.open_dataset(sl)
                    ds = ds.sel(time=dr, method='nearest')
                    ds = ds.sel(lat=slice(*self.lat_bounds[::-1]), lon=slice(*self.lon_bounds))
                    dsl.append(ds)
                dataset = xr.combine_by_coords(dsl, combine_attrs='override')

        else: 
            dataset = xr.open_dataset(self.file)

        return dataset

    def make_height_index(self, dataset): 
        """ Converts vertical indexing of 3D NCEP files from pressure-based to altitude-based
        """ 

        # Get the nominal height
        hgt = np.sort(np.round(dataset['hgt'].mean(['lat', 'lon', 'time']).values))
        # Makes sure coordinates are output correctly 
        frame = dataset.rename_dims({'lat': 'x', 'lon': 'y', 'time': 'z', 'level': 'a'}).to_dataframe().reset_index(drop=True)
        frame['hgt'] = pd.cut(frame['hgt'], hgt)
        frame['hgt'] = frame['hgt'].apply(lambda x: x.mid).astype(float)
        ordered = frame.groupby(['time', 'hgt', 'lat', 'lon']).mean()

        ds = ordered.to_xarray().interpolate_na(dim='hgt', fill_value='extrapolate')
        ds = ds.rename({'level': 'pres'})
        ds = ds.rename({'hgt': 'level'})

        return ds


class MERRAReader(Reader): 
    """ Reads MERRA-2 atmospheric data in 2D (default) or 3D 
        Instantaneous or time-averaged values can be requested, 
        with time-averaged being the default
        2D products have hourly resolution, and 3D products have 3 hour resolution
        Grid spacing is 0.625 deg. 
        
        :param dimension: Either '2D' (default) or '3D'
        :param instant: Either time-averaged (False, default) or 
                        instantaneous (True) fields
        :param lat_bounds: Tuple of latitude bounds for subsetting 
        :param lon_bounds: Tuple of longitude bounds for subsetting
        :param use_dask: Toggles dask-based loading (default False).
                         Initial tests suggests dask is slower for small subsets, 
                         but may be necessary for large ones

        MERRA product codes: 
        - M2T1NXSLV: Time-averaged single-level diagnostics (2D)
        - M2I1NXASM: Instantaneous single-level diagnostics (2D)
        - M2T3NVASM: Time-averaged assimilated meterological fields (3D)
        - M2I3NVASM: Instantaneous assimilated meterological fields (3D)
    """ 

    def __init__(self, datetime, dimension='2D', instant=False, 
                 lat_bounds=(-90, 90), lon_bounds=(-180, 180), use_dask=False, **kwargs): 

        self.dimension = dimension
        self.instant = instant
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.use_dask = use_dask
        super().__init__(datetime, **kwargs)

    def read(self, return_dataset=False): 
        
        dataset = self.get_dataset()
        dataset = dataset.compute()  # Load into memory if it isn't already 

        lat = dataset['lat'].values
        lon = dataset['lon'].values
        time = dataset['time'].values
        time_date = pd.to_datetime(time).to_pydatetime()
        standard_time = date2num(time_date, self.time_reference)
        if self.dimension == '2D': 
            uwind_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['U10M'].values, bounds_error=False, fill_value=None)
            vwind_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['V10M'].values, bounds_error=False, fill_value=None)
            prwtr_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['TQV'].values, bounds_error=False, fill_value=None)
            lwtr_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['TQL'].values, bounds_error=False, fill_value=None)
            airtemp_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['TS'].values, bounds_error=False, fill_value=None)
            airpres_interp = spi.RegularGridInterpolator((standard_time, lat, lon), dataset['PS'].values, bounds_error=False, fill_value=None)
        else: 
            dataset = self.make_height_index(dataset)
            level = dataset['level'].values / 1e3  # to km 
            uwind_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['U'].values, bounds_error=False, fill_value=None)
            vwind_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['V'].values, bounds_error=False, fill_value=None)
            prwtr_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['QV'].values, bounds_error=False, fill_value=None)
            lwtr_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['QL'].values, bounds_error=False, fill_value=None)
            airtemp_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['T'].values, bounds_error=False, fill_value=None)
            airpres_interp = spi.RegularGridInterpolator((standard_time, level, lat, lon), dataset['PL'].values, bounds_error=False, fill_value=None)

        if return_dataset:
            return airtemp_interp, airpres_interp, uwind_interp, vwind_interp, prwtr_interp, lwtr_interp, dataset
        else:
            dataset.close()
            return airtemp_interp, airpres_interp, uwind_interp, vwind_interp, prwtr_interp, lwtr_interp

    def get_dataset(self): 

        if self.dimension == '2D': 
            dr = self.expand_date(freq='1H')
            dim_mask = ['U10M', 'V10M', 'TQV', 'TQL', 'TS', 'PS']
        elif self.dimension == '3D': 
            dr = self.expand_date(freq='3H')
            dim_mask = ['U', 'V', 'QV', 'QL', 'T', 'PL', 'H']
        else: 
            raise ValueError("Dimension should either be '2D' or '3D'")

        if self.online: 
            set_list = []
            if self.dimension == '2D': 
                if self.instant:
                    string = 'M2I1NXASM.5.12.4'
                else: 
                    string = 'M2T1NXSLV.5.12.4'
                urlstring = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/thredds/dodsC/MERRA2_aggregation/{0}/{0}_Aggregation_'.format(string)
                yrs = dr.year.unique()
                for y in yrs: 
                    url = urlstring + str(y) + '.ncml' 
                    set_list.append(url)

            else:
                if self.instant: 
                    string = 'M2I3NVASM.5.12.4' 
                else: 
                    string = 'M2T3NVASM.5.12.4'  
                urlstring = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/thredds/dodsC/MERRA2_aggregation/{0}/'.format(string)
                substring = '/' + string + '_Aggregation_'

                yrmonths = np.unique(dr.floor('d') + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1))
                for ym in yrmonths: 
                    ym = pd.Timestamp(ym)
                    url = '{}{}{}{}_{:02d}.ncml'.format(urlstring, ym.year, substring, ym.year, ym.month)
                    set_list.append(url)
            
            if self.use_dask: 
                dataset = xr.open_mfdataset(set_list, chunks='auto')[dim_mask]

                # Sel provides a great access point for time, but breaks dask loading
                # Uncomment this once xarray fixes this functionality 
                # dataset = dataset.sel(time=dr, method='nearest')
                time = dataset['time'].values
                time = pd.to_datetime(time).values
                # Even explicit location based indexing fails! 
                # mask = (time > dr[0]) & (time < dr[-1])
                # dataset = dataset.loc[dict(time=time[mask])]
                # The only way I've found to make this work so far is via sliced indexing
                # Looks ugly...
                lower = time[time >= dr[0]][0]
                upper = time[time <= dr[-1]][-1]
                dataset = dataset.isel(time=slice(int(np.where(time == lower)[0]), 
                                                  int(np.where(time == upper)[0])))
                dataset = dataset.sel(lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))

            else: 
                dsl = [] 
                for sl in set_list: 
                    ds = xr.open_dataset(sl)[dim_mask]
                    ds = ds.sel(time=dr, method='nearest')
                    ds = ds.sel(lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
                    dsl.append(ds)
                dataset = xr.combine_by_coords(dsl) 

        else: 
            dataset = xr.open_dataset(self.file)

        return dataset

    def make_height_index(self, dataset): 
        # Get the nominal height
        hgt = np.sort(np.round(dataset['H'].mean(['lat', 'lon', 'time']).values))
        # Makes sure coordinates are output correctly 
        frame = dataset.rename_dims({'lat': 'x', 'lon': 'y', 'time': 'z', 'lev': 'a'}).to_dataframe().reset_index(drop=True)
        frame['H'] = pd.cut(frame['H'], hgt)
        frame['H'] = frame['H'].apply(lambda x: x.mid).astype(float)
        ordered = frame.groupby(['time', 'H', 'lat', 'lon']).mean()
        ds = ordered.to_xarray().interpolate_na(dim='H', fill_value='extrapolate')
        ds = ds.rename({'H': 'level'}).drop_vars('lev')

        return ds


# Ionosphere readers 


class IONEXReader(Reader): 
    """ CDDIS Ionospheric Total Electron Content (TEC) from 
        https://cddis.nasa.gov/archive/gnss/products/ionex
        No OpenDAP or THREDDS interface available, 
        so files are downloaded manually using curl and gzip
    """

    def read(self, from_dataset=False):
        """ Reads IONEX files and returns a scipy.interpolate.RegularGridInterpolator
            for ionospheric TEC

            Inputs: 
            :param from_dataset: If True, module assumes the input file is netcdf-like, else use ionex file
            
        """
        
        dataset = self.get_dataset(from_dataset=from_dataset)
        time_date = pd.to_datetime(dataset['time'].values).to_pydatetime()
        standard_time = date2num(time_date, self.time_reference)
        grid_lat = dataset['lat'].values 
        grid_lon = dataset['lon'].values 
        TEC = dataset['TEC'].values

        TEC_interp = spi.RegularGridInterpolator((standard_time, grid_lat[::-1], grid_lon), TEC[:, ::-1, :], bounds_error=False, fill_value=0)
        return TEC_interp

    def get_dataset(self, from_dataset=False): 
        dr = self.expand_date(freq='2H')

        # Assuming CDDIS ionospheric TEC is listed in two hour intervals over a set longitude and latitude grid. 
        hour_inc = np.arange(0, 26, 2)  
        grid_lat = np.arange(87.5, -90, -2.5)
        grid_lon = np.arange(-180, 185, 5)

        if self.online: 
            days = dr.round('1D').unique()
            TECs = []
            sts = []
            # Read over multiple days 
            for d in days: 
                year = d.year
                j_day = int(d.strftime('%j'))
                filename = 'jplg{:03d}0.{:02d}i.Z'.format(j_day, year % 100)
                urlstring = 'https://cddis.nasa.gov/archive/gnss/products/ionex/{:04d}/{:03d}/{}'.format(year, j_day, filename)
                # os.system('curl -c ~/.cddis_cookies -nOL %s' % urlstring)
                cookie_dir = os.path.join(os.environ['HOME'], '.urs_cookies')
                wget_args = ' --load-cookies {0} --save-cookies {0} --auth-no-challenge=on --keep-session-cookies --content-disposition '.format(cookie_dir)
                os.system('wget' + wget_args + urlstring)

                os.system('gzip -d %s' % filename)
                filename = filename[:-2]

                # Read file using comprehension 
                with open(filename) as f:
                    ionex = f.read()
                    TEC = np.array([self.parse_ionex_map(t) for t in ionex.split('START OF TEC MAP')[1:]])
                TECs.append(TEC[:-1, ...])  # Daily files have an overlap
                os.system('rm %s' % filename)  # Get rid of downloaded file
                
                starttime_reference = d.strftime('hours since %Y-%-m-%-d 00:00:0.0')  # J2000 
                time_date = num2date(hour_inc, starttime_reference)
                standard_time = date2num(time_date, self.time_reference)
                sts.append(standard_time[:-1])

            # Collect files 
            TEC = np.concatenate(TECs)
            standard_time = np.concatenate(sts)

            time = num2date(standard_time, self.time_reference)
            dataset = xr.Dataset(data_vars={'TEC': (['time', 'lat', 'lon'], TEC)}, 
                                 coords={'time': time, 'lat': grid_lat, 'lon': grid_lon})
            dataset['TEC'].attrs = {'units': 'TECu', 'description': 'Ionospheric Total Electron Content TEC'}

        else:  
            if not from_dataset:            
                with open(self.file) as f:
                    ionex = f.read()
                    TEC = np.array([self.parse_ionex_map(t) for t in ionex.split('START OF TEC MAP')[1:]])
                starttime_reference = dr[0].strftime('hours since %Y-%-m-%-d 00:00:0.0')  # J2000 
                time_date = num2date(hour_inc, starttime_reference)
                standard_time = date2num(time_date, self.time_reference)
                time = pd.to_datetime(num2date(standard_time, self.time_reference,
                                               only_use_cftime_datetimes=False, only_use_python_datetimes=True))
                dataset = xr.Dataset(data_vars={'TEC': (['time', 'lat', 'lon'], TEC)}, 
                                     coords={'time': time, 'lat': grid_lat, 'lon': grid_lon})
                dataset['TEC'].attrs = {'units': 'TECu', 'description': 'Ionospheric Total Electron Content TEC'}
            else: 
                dataset = xr.open_dataset(self.file)

        return dataset

    @staticmethod
    def parse_ionex_map(tecmap):
        """ Parsing function for CDDIS TEC maps
            
            Inputs: 
            :param tecmap: File handle for CDDIS ionosphere map 
        """
        tecmap = re.split('.*END OF TEC MAP', tecmap)[0]
        split = [np.fromstring(xi, sep=' ') for xi in re.split('.*LAT/LON1/LON2/DLON/H\\n', tecmap)[1:]]
        return 0.1 * np.stack(split)


# Dev. Note: Several data services are down, but future additions include ECCO, NCEI
