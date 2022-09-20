import os
import pickle
import warnings
from multiprocessing import Pool 
from itertools import product
import numpy as np
from numpy import sin, cos, tan, arccos, arcsin, arctan, arctan2, sqrt, radians, exp, log10, log
import pandas as pd
import scipy.interpolate as spi
import scipy.constants as spc
from cftime import date2num

from . import dielectric
from .utils.config import cache_path, dir_path
from .utils import reader


class ocean(): 
    """ Base class for generating an ocean forward model at a given state (temperature, salinity, surface winds) 
        
        :param datetime: Either a single date or a pair of dates bracketing an interval of time.
                         Several formats are permissible, such as
                         - String or iterable of strings
                         - Python datetime or iterable of datetimes 
                         - Numpy datetime64 or iterable
                         - Pandas timestamp or iterable
        :param mode: 'flat' mode implements a specular ocean surface
                     'rough' mode interpolates the Meissner/Wentz GMFs for wind-roughening
        :param online: Toggles online functionality 
        :param sst_file: Sea surface temperature file location. Default is 'cache', which reads a stock file  
                         from the cache. 
        :param sss_file: Sea surface salinity file location. Default is 'cache', which reads a stock file 
                         from the cache. 
        :param sst_reader: File reader method, default is GHRSSTReader
        :param sss_reader: File reader method, default is OISSSReader
        :param verbose: Toggles verbosity 
        
        Note that only one file per parameter is downloaded in the cache.

        For plotting convenience, arrays with geographic info are defined as they would appear on a map, 
        rows of decreasing latitude (90 -> -90) and columns of increasing longitude (-180 -> 180)

    """ 
    time_reference = 'seconds since 2000-01-01 12:00:0.0'  # J2000

    def __init__(self, datetime='2015-01-01', mode='flat', online=False, 
                 sst_file=None, sst_reader=reader.GHRSSTReader, sst_reader_kwargs=None,
                 sss_file=None, sss_reader=reader.OISSSReader, sss_reader_kwargs=None,
                 dielectric=dielectric.h2o_liquid_KleinSwift, verbose=False, **kwargs):

        self.datetime = pd.to_datetime(datetime)
        self.mode = mode
        if self.mode == 'rough': 
            self.use_wind_interpolators = True
        elif self.mode == 'flat': 
            self.use_wind_interpolators = False
        else:
            raise ValueError('Specify mode as either "flat" or "rough"')

        self.online = online
        
        if sst_file is None and not self.online: 
            self.datetime = pd.to_datetime(['2018-01-01 12:00'])
            self.sst_file = os.path.join(cache_path, 'ocean', 
                                         '20180101120000-ABOM-L4_GHRSST-SSTfnd-GAMSSA_28km-GLOB-v02.0-fv01.0.nc')
            self.sst_reader = reader.GHRSSTReader
            self.sst_reader_kwargs = {'version': 'GAMSSA'}
        else: 
            self.sst_file = sst_file 
            self.sst_reader = sst_reader
            if sst_reader_kwargs is None: 
                self.sst_reader_kwargs = {}
            else:    
                self.sst_reader_kwargs = sst_reader_kwargs
        
        if sss_file is None and not self.online: 
            self.sss_file = os.path.join(cache_path, 'ocean', 'SMAP_L3_SSS_20180105_8DAYS_V5.0.nc')
            self.sss_reader = reader.SMAPSalinityReader
            self.sss_reader_kwargs = {}
        else: 
            self.sss_file = sss_file
            self.sss_reader = sss_reader
            if sss_reader_kwargs is None: 
                self.sss_reader_kwargs = {}
            else:    
                self.sss_reader_kwargs = sss_reader_kwargs

        self.dielectric = dielectric
        self.verbose = verbose 

        # Inclusion of misc. kwargs is largely for development 
        for key, value in kwargs.items(): 
            setattr(self, key, value)

        self.read_ocean()
        self.read_land()

    def read_ocean(self):
        """ Reads ocean state from online or local files. 
            Sets up RegularGridInterpolators which take arguments of lat and lon.
        """
        if self.verbose: print('Setting ocean state as a function of latitude and longitude')
        
        if self.sst_reader is self.sss_reader:
            # HYCOM-like case 
            multi_rdr = self.sst_reader(self.datetime, online=self.online, file=self.sst_file, 
                                        **self.sst_reader_kwargs)
            self.sst_interp, self.sss_interp = multi_rdr.read()
        else: 
            sst_rdr = self.sst_reader(self.datetime, online=self.online, file=self.sst_file, 
                                      **self.sst_reader_kwargs)
            self.sst_interp = sst_rdr.read()
            if type(self.sst_interp) is tuple: 
                warnings.warn('Multiple outputs detected for SST interpolator, defaulting to first')
                self.sst_interp = self.sst_interp[0]
            sss_rdr = self.sss_reader(self.datetime, online=self.online, file=self.sss_file, 
                                      **self.sss_reader_kwargs)
            self.sss_interp = sss_rdr.read() 
            if type(self.sss_interp) is tuple: 
                warnings.warn('Multiple outputs detected for SSS interpolator, defaulting to second')
                self.sss_interp = self.sss_interp[1]
        
        if self.use_wind_interpolators: 
            # Loading Meissner and Wentz polarimetric wind interpolators 
            # See the interpolator maker script in scripts/
            if self.verbose: print('Loading wind emissivity interpolators')
            with open(os.path.join(dir_path, 'assets/ocean/MW_wind_interpolators.p'), 'rb') as file: 
                windpack = pickle.load(file)
            self.iso_v = windpack[0]
            self.iso_h = windpack[1]
            self.aniso1_v = windpack[2]
            self.aniso1_h = windpack[3]
            self.aniso1_s3 = windpack[4]
            self.aniso1_s4 = windpack[5]
            self.aniso2_v = windpack[6]
            self.aniso2_h = windpack[7]
            self.aniso2_s3 = windpack[8]
            self.aniso2_s4 = windpack[9]
            
            # Old wind parameterization 
            # self.wind_coeff_0 = 0.35
            # self.wind_coeff_1 = 0.1

    def read_land(self): 
        """ Reads EASE2 Grid 3 km landmask file. 
            Sets up a RegularGridInterpolator which takes arguments of lat and lon.
        """

        if self.verbose: print('Setting Land Mask') 
        self.landmask_file = os.path.join(cache_path, 'landmask', 'EASE2_M03km.LOCImask_land50_coast0km.11568x4872.bin')
        dotstring = self.landmask_file.split('.')
        dimstring = dotstring[-2].split('x')
        dim = (int(dimstring[1]), int(dimstring[0]))
        landmask = np.fromfile(self.landmask_file, dtype=np.uint8)
        self.landmask = landmask.reshape(dim)
        self.landmask[self.landmask < 255] = 1 
        self.landmask[self.landmask == 255] = 0
        land_lat = np.fromfile(os.path.join(cache_path, 'landmask', 'EASE2_M03km.lats.11568x4872x1.double'), dtype=np.float64)
        lat = land_lat.reshape(dim)[::-1, 0]
        land_lon = np.fromfile(os.path.join(cache_path, 'landmask', 'EASE2_M03km.lons.11568x4872x1.double'), dtype=np.float64)
        lon = land_lon.reshape(dim)[0, :]

        self.landmask_interp = spi.RegularGridInterpolator((lat, lon), self.landmask[::-1, :], 
                                                           bounds_error=False, fill_value=1)

    def get_ocean_TB(self, frequency, time, lat, lon, uwind, vwind, theta, phi, in_epoch=False, use_time=True):
        """ Calculates emission temperature of the ocean from frequency, emission angle,
            and meshgrids of latitude and longitude. Type of all inputs should be
            numpy arrays

            :param frequency: Frequency in MHz (dimension M)
            :param time: String or array of times, converted to interpolator
                         epoch reference if in_epoch=False
            :param lat: Latitude in diegrees 
            :param lon: Longitude in degrees 
            :param uwind: E/W 10-m wind 
            :param vwind: N/S 10-m wind 
            :param theta: Elevation angle of spacecraft with respect to surface normal 
            :param phi: Azimuth angle of spacecraft with respect to surface normal 
            :param in_epoch: See above
            :param use_time: Allows for bypass of time resolution in the case of heterogeneous time sources 

            :return: Full Stokes brightness temperatures and emissivity (dimension 4xMx...)
        """

        if not use_time: 
            time = self.datetime[0]
            in_epoch = False
        if not in_epoch: 
            time = date2num(pd.to_datetime(time).to_pydatetime(), self.time_reference)
        
        sst = self.sst_interp((time, lat, lon))
        sst[sst < 270] = np.nan  # Eliminates fill values
        sss = self.sss_interp((time, lat, lon))
        sss[sss <= 0] = np.nan  # Eliminate fill values
        landmask = self.landmask_interp((lat, lon))

        # Reduce data and make duplicate map 
        full = np.vstack([sst, sss, landmask, uwind, vwind, theta, phi])
        full = np.round(full, decimals=2)  # Arbitrary precision, maybe make this a kwarg
        unique, mapping = np.unique(full, axis=1, return_inverse=True)
        sst_unique, sss_unique, landmask_unique, uwind_unique, vwind_unique, theta_unique, phi_unique = unique

        # Pad axis for frequency multiplication
        sst_unique = sst_unique[..., np.newaxis]
        sss_unique = sss_unique[..., np.newaxis]
        landmask_unique = landmask_unique[..., np.newaxis]
        uwind_unique = uwind_unique[..., np.newaxis]
        vwind_unique = vwind_unique[..., np.newaxis]
        theta_unique = theta_unique[..., np.newaxis]
        phi_unique = phi_unique[..., np.newaxis]
        frequency = np.squeeze(frequency)
        emis = self.get_ocean_emissivity(frequency, sst_unique, sss_unique, uwind_unique, vwind_unique, theta_unique, phi_unique)

        # Conform result to original shape
        emis = emis[:, mapping, :]
        
        # Optional emissivity for land is 0.83, setting to zero for now
        landmask = landmask[..., np.newaxis]
        sst = sst[..., np.newaxis]
        emis = emis * (1 - np.ceil(landmask)) + np.ceil(landmask) * -1
        emis[:2][emis[:2] < 0] = np.nan  # Exclude the 3/4 Stokes
        TB = emis * sst
        TB = np.moveaxis(TB, -1, 1)
        emis = np.moveaxis(emis, -1, 1)
        prop_dict = {'emissivity': emis}

        return TB, prop_dict

    def get_ocean_emissivity(self, frequency, sst, sss, uwind, vwind, theta, phi):
        """ Returns ocean emissivity. This is the specular surface emissivity if ocean.mode is 'simple', 
            and GMF wind-roughened emissivity if ocean.mode is 'full'

            :param frequency: Frequency in MHz 
            :param sst: Sea surface temperature in Kelvin 
            :param sss: Sea surface salinity in psu 
            :param uwind: 10-m eastward wind in m/s 
            :param vwind: 10-m northward wind in m/s 
            :param theta: Elevation angle of spacecraft with respect to surface normal
            :param phi: Azimuth angle of spacecraft with respect to surface normal

            :return: Emissivity vector (eV, eH, U, V) (dimension 4x...)
        """ 

        angle = np.radians(theta)
        eps = self.dielectric(frequency, sst, sss)

        Rh = (cos(angle) - (eps - sin(angle)**2)**0.5) / (cos(angle) + (eps - sin(angle)**2)**0.5)
        Rv = (eps * cos(angle) - (eps - sin(angle)**2)**0.5) / (eps * cos(angle) + (eps - sin(angle)**2)**0.5)
        emisV = (1 - Rv * np.conj(Rv))
        emisH = (1 - Rh * np.conj(Rh)) 
        emisS3 = np.zeros(np.shape(emisV))
        emisS4 = np.zeros(np.shape(emisH))
        if self.use_wind_interpolators: 

            ref_eps = self.dielectric(frequency, 273.15 + 20, 35)
            Rh_ref = (cos(angle) - (ref_eps - sin(angle)**2)**0.5) / (cos(angle) + (ref_eps - sin(angle)**2)**0.5)
            Rv_ref = (ref_eps * cos(angle) - (ref_eps - sin(angle)**2)**0.5) / (ref_eps * cos(angle) + (ref_eps - sin(angle)**2)**0.5)
            emisH_ref = (1 - Rh_ref * np.conj(Rh_ref)) 
            emisV_ref = (1 - Rv_ref * np.conj(Rv_ref))

            wind = np.sqrt(uwind**2 + vwind**2)
            wind_phi = arctan2(vwind, uwind)
            phi = np.radians(phi) - wind_phi
            iso_v = self.iso_v((frequency, theta, wind))
            iso_h = self.iso_h((frequency, theta, wind))
            iso_v = iso_v * emisV / emisV_ref
            iso_h = iso_h * emisH / emisH_ref

            aniso1_v = self.aniso1_v((frequency, theta, wind))
            aniso1_h = self.aniso1_h((frequency, theta, wind))
            aniso1_s3 = self.aniso1_s3((frequency, theta, wind))
            aniso1_s4 = self.aniso1_s4((frequency, theta, wind))
            aniso2_v = self.aniso2_v((frequency, theta, wind))
            aniso2_h = self.aniso2_h((frequency, theta, wind))
            aniso2_s3 = self.aniso2_s3((frequency, theta, wind))
            aniso2_s4 = self.aniso2_s4((frequency, theta, wind))

            aniso1_v = aniso1_v * emisV / emisV_ref
            aniso1_h = aniso1_h * emisH / emisH_ref
            aniso2_v = aniso2_v * emisV / emisV_ref
            aniso2_h = aniso2_h * emisH / emisH_ref

            v_wind = iso_v + aniso1_v * cos(phi) + aniso2_v * cos(2 * phi)
            h_wind = iso_h + aniso1_h * cos(phi) + aniso2_h * cos(2 * phi)
            emisV = emisV + v_wind
            emisH = emisH + h_wind
            emisS3 = emisS3 + aniso1_s3 * sin(phi) + aniso2_s3 * sin(2 * phi)
            emisS4 = emisS4 + aniso1_s4 * sin(phi) + aniso2_s4 * sin(2 * phi)

            # Old wind parameterization
            # self.wind_coeff_0 = 0.1150 + 0.5281 * log10(frequency / 1e3)
            # self.wind_coeff_0[self.wind_coeff_0 < 0] = 0
            # emisV = emisV + self.wind_coeff_1 * wind / sst
            # emisH = emisH + self.wind_coeff_0 * wind / sst

        emis = np.array([emisV, emisH, emisS3, emisS4])
        return np.real(emis)

    def get_roughness_correction(self, frequency, sst, sss, uwind, vwind, theta, phi):
        """ Returns only excess emissivity due to wind roughening

            :param frequency: Frequency in MHz 
            :param sst: Sea surface temperature in Kelvin 
            :param sss: Sea surface salinity in psu 
            :param uwind: 10-m eastward wind in m/s 
            :param vwind: 10-m northward wind in m/s 
            :param theta: Elevation angle of spacecraft with respect to surface normal
            :param phi: Azimuth angle of spacecraft with respect to surface normal

            :return: Excess emissivity vector (eV, eH, U, V) (dimension 4x...)
        """

        ref_eps = self.dielectric(frequency, 273.15 + 20, 35)
        angle = np.radians(theta)
        Rh_ref = (cos(angle) - (ref_eps - sin(angle)**2)**0.5) / (cos(angle) + (ref_eps - sin(angle)**2)**0.5)
        Rv_ref = (ref_eps * cos(angle) - (ref_eps - sin(angle)**2)**0.5) / (ref_eps * cos(angle) + (ref_eps - sin(angle)**2)**0.5)
        emisH_ref = (1 - Rh_ref * np.conj(Rh_ref)) 
        emisV_ref = (1 - Rv_ref * np.conj(Rv_ref))

        eps = self.dielectric(frequency, sst, sss)
        Rh = (cos(angle) - (eps - sin(angle)**2)**0.5) / (cos(angle) + (eps - sin(angle)**2)**0.5)
        Rv = (eps * cos(angle) - (eps - sin(angle)**2)**0.5) / (eps * cos(angle) + (eps - sin(angle)**2)**0.5)
        emisH = (1 - Rh * np.conj(Rh)) 
        emisV = (1 - Rv * np.conj(Rv))

        wind = np.sqrt(uwind**2 + vwind**2)
        wind_phi = arctan2(vwind, uwind)
        phi = np.radians(phi) - wind_phi
        iso_v = self.iso_v((frequency, theta, wind))
        iso_h = self.iso_h((frequency, theta, wind))
        
        # Correct emissivity externally
        iso_v = iso_v * emisV / emisV_ref
        iso_h = iso_h * emisH / emisH_ref

        aniso1_v = self.aniso1_v((frequency, theta, wind))
        aniso1_h = self.aniso1_h((frequency, theta, wind))
        aniso1_s3 = self.aniso1_s3((frequency, theta, wind))
        aniso1_s4 = self.aniso1_s4((frequency, theta, wind))
        aniso2_v = self.aniso2_v((frequency, theta, wind))
        aniso2_h = self.aniso2_h((frequency, theta, wind))
        aniso2_s3 = self.aniso2_s3((frequency, theta, wind))
        aniso2_s4 = self.aniso2_s4((frequency, theta, wind))

        # Correct emissivity externally
        aniso1_v = aniso1_v * emisV / emisV_ref
        aniso1_h = aniso1_h * emisH / emisH_ref
        aniso2_v = aniso2_v * emisV / emisV_ref
        aniso2_h = aniso2_h * emisH / emisH_ref

        v_wind = iso_v + aniso1_v * cos(phi) + aniso2_v * cos(2 * phi)
        h_wind = iso_h + aniso1_h * cos(phi) + aniso2_h * cos(2 * phi)
        emisV = v_wind
        emisH = h_wind
        emisS3 = aniso1_s3 * sin(phi) + aniso2_s3 * sin(2 * phi)
        emisS4 = aniso1_s4 * sin(phi) + aniso2_s4 * sin(2 * phi) 

        emis = np.array([emisV, emisH, emisS3, emisS4])
        return np.real(emis)


class fastem(ocean): 
    """ FASTEM ocean emissivity models from 1.4 - 410 GHz. Developed by ECMWF and translated to Python.

        :param version: Either '5' or '6' (default). See Kazumori and English 2015 for differences
    """

    def __init__(self, version='6', **kwargs): 
        self.version = version
        super().__init__(**kwargs)

    def read_ocean(self): 
        super().read_ocean() 
        if self.version == '6': 
            pickle_path = os.path.join(dir_path, 'assets', 'ocean', 'FASTEM_wind_interpolators.p')
            with open(pickle_path, 'rb') as file: 
                windpack = pickle.load(file)
            
            self.aniso1_v = windpack[0]
            self.aniso1_h = windpack[1]
            self.aniso2_v = windpack[2]
            self.aniso2_h = windpack[3]

    def get_ocean_emissivity(self, frequency, sst, sss, uwind, vwind, theta, phi):
        """
        FASTEM emissivity models
        
        :param frequency: Frequency in MHz (dimension O)
        :param sst: Sea surface temperature in Kelvin (dimension MxN)
        :param sss: Sea surface salinity in psu (dimension MxN)
        :param uwind: 10-m eastward wind in m/s 
        :param vwind: 10-m northward wind in m/s 
        :param theta: Elevation angle of spacecraft with respect to surface normal
        :param phi: Azimuth angle of spacecraft with respect to surface normal

        :return emis: Emissivity vector (eV, eH, U, V) (dimension 4xOxMxN)
            
        
        """  
        freq = frequency / 1e3 
        ts = sst - spc.zero_Celsius
        ss = sss 
        x = sin(radians(theta)) * cos(radians(phi))
        y = sin(radians(theta)) * sin(radians(phi))
        z = cos(radians(theta)) 
        senzen = np.degrees(np.arccos(z / sqrt(x**2 + y**2 + z**2)))
        wind = np.sqrt(uwind**2 + vwind**2)
        wind_phi = arctan2(vwind, uwind)
        rel_az = np.radians(phi) - wind_phi 

        # -------------------------------------------------------------------------
        # Permittivity (JCSDA model)
        # -------------------------------------------------------------------------
        A_COEF = np.array([3.8, 0.0248033, 87.9181727, 
                           -0.4031592248, 0.0009493088010, -0.1930858348E-05, -0.002697, 
                           -7.3E-06, -8.9E-06, 5.723, 0.022379,  
                           -0.00071237, -6.28908E-03, 1.76032E-04, -9.22144E-05, 
                           0.1124465, -0.0039815727, 0.00008113381, -0.00000071824242, 
                           -2.39357E-03, 3.1353E-05, -2.52477E-07, 0.003049979018,  
                           -3.010041629E-05, 0.4811910733E-05, -0.4259775841E-07, 0.149, 
                           -8.8E-04, -1.05E-04, 2.033E-02, 1.266E-04,  
                           2.464E-06, -1.849E-05, 2.551E-07, -2.551E-08,  
                           0.182521, -1.46192E-03, 2.09324E-05, -1.28205E-07])

        ts_sq = ts**2
        ts_cu = ts_sq * ts
        e0 = 0.0088419

        einf = A_COEF[0] + A_COEF[1] * ts
        es = A_COEF[2] + A_COEF[3] * ts + A_COEF[4] * ts_sq + A_COEF[5] * ts_cu
        e1 = A_COEF[9] + A_COEF[10] * ts + A_COEF[11] * ts_sq
        tau1 = A_COEF[15] + A_COEF[16] * ts + A_COEF[17] * ts_sq + A_COEF[18] * ts_cu
        tau2 = A_COEF[22] + A_COEF[23] * ts + A_COEF[24] * ts_sq + A_COEF[25] * ts_cu

        es_k = es
        e1_k = e1
        tau1_k = tau1
        tau2_k = tau2
        perm_imag = 0

        delta = 25.0 - ts
        beta = A_COEF[29] + A_COEF[30] * delta + A_COEF[31] * delta**2 + ss * (A_COEF[32] + A_COEF[33] * delta + A_COEF[34] * delta**2)
        sigma25 = ss * (A_COEF[35] + A_COEF[36] * ss + A_COEF[37] * ss**2 + A_COEF[38] * ss**3)
        sigma = sigma25 * np.exp(- delta * beta)

        ONE = 1

        ces = ONE + ss * (A_COEF[6] + A_COEF[7] * ss + A_COEF[8] * ts)
        ce1 = ONE + ss * (A_COEF[12] + A_COEF[13] * ss + A_COEF[14] * ts)
        ctau1 = ONE + ss * (A_COEF[19] + A_COEF[20] * ts + A_COEF[21] * ts_sq)
        ctau2 = ONE + ss * (A_COEF[26] + A_COEF[27] * ts + A_COEF[28] * ss**2)
        es = es_k * ces
        e1 = e1_k * ce1
        tau1 = tau1_k * ctau1
        tau2 = tau2_k * ctau2
        perm_imag = -sigma / (2 * np.pi * e0 * freq)

        f1 = freq * tau1
        f2 = freq * tau2
        del1 = es - e1
        del2 = e1 - einf
        perm_real = einf + del1 / (ONE + f1**2) + del2 / (ONE + f2**2)
        perm_imag = -perm_imag + del1 * f1 / (ONE + f1**2) + del2 * f2 / (ONE + f2**2)
        permittivity = perm_real + (perm_imag * 1j)

        # -------------------------------------------------------------------------
        # Fresnel reflectance coefficients
        # -------------------------------------------------------------------------
        perm1 = np.sqrt(permittivity - np.sin(np.deg2rad(senzen))**2)
        perm2 = permittivity * np.cos(np.deg2rad(senzen))
        rhth = (np.cos(np.deg2rad(senzen)) - perm1) / (np.cos(np.deg2rad(senzen)) + perm1)
        rvth = (perm2 - perm1) / (perm2 + perm1)
        fresnel_v_Real = np.real(rvth)
        fresnel_v_imag = np.imag(rvth)
        fresnel_v = fresnel_v_Real * fresnel_v_Real + fresnel_v_imag * fresnel_v_imag
        fresnel_h_Real = np.real(rhth)
        fresnel_h_imag = np.imag(rhth)
        fresnel_h = fresnel_h_Real * fresnel_h_Real + fresnel_h_imag * fresnel_h_imag

        # -------------------------------------------------------------------------
        # Small scale correction
        # -------------------------------------------------------------------------
        scoef = np.array([-5.0208480E-06, 2.3297951E-08, 4.6625726E-08, -1.9765665E-09, 
        -7.0469823E-04, 7.5061193E-04, 9.8103876E-04, 1.5489504E-04]) 

        scor = scoef[0] * wind * freq + scoef[1] * wind * freq**2 + scoef[2] * wind**2 * freq + scoef[3] * wind**2 * freq**2 + scoef[4] * wind**2 / freq + scoef[5] * wind**2 / freq**2 + scoef[6] * wind + scoef[7] * wind**2
        cos_z = np.cos(np.deg2rad(senzen))
        small_corr = np.exp(-scor * cos_z * cos_z)

        RvS = fresnel_v * small_corr
        RhS = fresnel_h * small_corr

        # -------------------------------------------------------------------------
        # Large scale correction
        # -------------------------------------------------------------------------
        lcoef5 = np.array([-5.994667E-02, 9.341346E-04, -9.566110E-07, 8.360313E-02, -1.085991E-03,
                           6.735338E-07, -2.617296E-02, 2.864495E-04, -1.429979E-07, -5.265879E-04,
                           6.880275E-05, -2.916657E-07, -1.671574E-05, 1.086405E-06, -3.632227E-09,
                           1.161940E-04, -6.349418E-05, 2.466556E-07, -2.431811E-02, -1.031810E-03, 
                           4.519513E-06, 2.868236E-02, 1.186478E-03, -5.257096E-06, -7.933390E-03,
                           -2.422303E-04, 1.089605E-06, -1.083452E-03, -1.788509E-05, 5.464239E-09,
                           -3.855673E-05, 9.360072E-07, -2.639362E-09, 1.101309E-03, 3.599147E-05,
                           -1.043146E-07])

        lcoef = lcoef5

        seczen = ONE / cos_z

        # Compute fitting coefficients for a given frequency
        
        zc = np.zeros((12, *np.shape(freq)))
        for j in range(1, 13):
            zc[j - 1] = lcoef[j * 3 - 2 - 1] + lcoef[j * 3 - 1 - 1] * freq + lcoef[j * 3 - 1] * freq**2

        RvL = zc[0] + zc[1] * seczen + zc[2] * seczen**2 + zc[3] * wind + zc[4] * wind**2 + zc[5] * wind * seczen
        RhL = zc[6] + zc[7] * seczen + zc[8] * seczen**2 + zc[9] * wind + zc[10] * wind**2 + zc[11] * wind * seczen

        # -------------------------------------------------------------------------
        # Foam 
        # -------------------------------------------------------------------------
        # Monahan et al., 1986 without surface stability term
        foam_cover = 1.95E-05 * wind ** 2.55

        # The foam vertical and horizontal reflectance codes, adopted from Masahiro Kazumori, JMA
        FR_COEFF = np.array([0.07, -1.748e-3, -7.336e-5, 1.044e-7, -0.93])
        Foam_Rv = FR_COEFF[0]
        Fh = ONE + senzen * (FR_COEFF[1] + senzen * (FR_COEFF[2] + senzen * FR_COEFF[3]))
        Foam_Rh = ONE + FR_COEFF[4] * Fh

        # Added frequency dependence derived from Stogryn model
        Foam_ref = 0.4 * np.exp(-0.05 * freq)
        Foam_Rv = Foam_Rv * Foam_ref
        Foam_Rh = Foam_Rh * Foam_ref

        # -------------------------------------------------------------------------
        # Emissivity
        # -------------------------------------------------------------------------    
        emisV = (ONE - foam_cover) * (ONE - RvS + RvL) + foam_cover * (ONE - Foam_Rv)
        emisH = (ONE - foam_cover) * (ONE - RhS + RhL) + foam_cover * (ONE - Foam_Rh)

        emis = np.array([emisV, emisH, np.zeros(np.shape(emisV)), np.zeros(np.shape(emisV))])
        # -------------------------------------------------------------------------
        # Azimuth Dependence 
        # ------------------------------------------------------------------------- 
         
        # FASTEM-5 Azimuth model 
        x = np.array([0.0, 1.4, 6.8, 10.7, 19.35, 37., 89., 150., 200.])
        y = np.array([0.0, 0.1, 0.6, 0.9, 1., 1.0, 0.4, 0.2, 0.0])

        b_coef = np.array([ 3.307255E-04, -2.901276E-06, -1.475497E-04,  1.288152E-06,  1.004010E-04,
                           -2.671158E-07,  4.363154E-06, -9.817795E-09, -4.777876E-05,  3.051852E-08,
                            1.369383E-03, -2.215847E-05, -8.099833E-04,  1.767702E-05, -5.977649E-06,
                           -1.784656E-07, -9.355531E-07,  5.495131E-08, -3.479300E-05, -3.751652E-07,
                            2.673536E-04, -1.378890E-06, -8.660113E-05,  2.871488E-07,  1.361118E-05,
                           -1.622586E-08, -1.232439E-07, -3.067416E-09, -1.835366E-06,  8.098728E-09,
                            1.255415E-04, -5.145201E-07, -8.832514E-06, -5.105879E-09,  2.734041E-05,
                           -3.398604E-07,  3.417435E-06, -7.043251E-09,  1.497222E-05, -6.832110E-09,
                           -2.315959E-03, -1.023585E-06,  5.154471E-05,  9.534546E-06, -6.306568E-05,
                           -4.378498E-07, -2.132017E-06,  1.612415E-08, -1.929693E-06, -6.217311E-09,
                           -1.656672E-04,  6.385099E-07,  2.290074E-06,  1.103787E-07, -5.548757E-06,
                            5.275966E-08, -4.653774E-07,  1.427566E-09, -3.197232E-06, -4.048557E-09,
                           -1.909801E-04, -3.387963E-07,  4.641319E-05,  4.502372E-07, -5.055813E-05,
                            2.104201E-07, -4.121861E-06, -1.633057E-08, -2.469888E-05,  4.492103E-08,
                           -4.582853E-03, -5.373940E-06,  9.713047E-04,  1.783009E-05, -4.539091E-04,
                            7.652954E-07, -6.708905E-06,  2.148401E-08,  8.054350E-05,  3.069258E-07,
                           -6.405746E-05, -9.694284E-08,  1.914498E-05,  1.336975E-07, -4.561696E-06,
                            3.769169E-08, -6.105244E-07,  2.433761E-10, -3.961735E-06,  1.995636E-08,
                            1.350148E-06,  3.678149E-07,  1.261701E-05, -2.011440E-07, -2.361347E-05,
                            2.943147E-08, -1.304551E-07, -1.119368E-09,  8.469458E-06, -2.292171E-09,
                            1.419156E-03, -3.838338E-06,  8.222562E-05, -1.106098E-06, -5.482327E-05,
                            3.083137E-07,  4.418828E-06, -1.302562E-08,  3.768883E-05, -5.012753E-08,
                           -9.396649E-06,  2.764698E-07,  1.745336E-05, -1.427031E-07, -3.879930E-06,
                           -1.117458E-08,  5.688281E-08,  1.513582E-09,  6.778764E-06, -7.691286E-09])

        Azimuth_Emi = np.zeros((4, *np.shape(emisV)))
        freq_c = np.zeros(np.shape(freq))
        for i in range(0, len(x) - 1): 
            f_mask = (freq >= x[i]) & (freq < x[i + 1])
            freq_c[f_mask] = y[i] + (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (freq[f_mask] - x[i])
        for m in range(1, 4): 
            # Some fortran/python index switching
            L = 10 * (m - 1) - 1
            ac = b_coef[L + 1] + b_coef[L + 2] * freq + b_coef[L + 3] * seczen                  \
                 + b_coef[L + 4] * seczen * freq                                                \
                 + b_coef[L + 5] * wind + b_coef[L + 6] * wind * freq + b_coef[L + 7] * wind**2 \
                 + b_coef[L + 8] * freq * wind**2 + b_coef[L + 9] * wind * seczen               \
                 + b_coef[L + 10] * wind * seczen * freq                                     
            Azimuth_Emi[0] = Azimuth_Emi[0] + ac * cos(m * rel_az)

            L = 10 * (m - 1) + 30 - 1
            ac = b_coef[L + 1] + b_coef[L + 2] * freq + b_coef[L + 3] * seczen                  \
                 + b_coef[L + 4] * seczen * freq                                                \
                 + b_coef[L + 5] * wind + b_coef[L + 6] * wind * freq + b_coef[L + 7] * wind**2 \
                 + b_coef[L + 8] * freq * wind**2 + b_coef[L + 9] * wind * seczen               \
                 + b_coef[L + 10] * wind * seczen * freq
            Azimuth_Emi[1] = Azimuth_Emi[1] + ac * cos(m * rel_az)

            L = 10 * (m - 1) + 60 - 1
            sc = b_coef[L + 1] + b_coef[L + 2] * freq + b_coef[L + 3] * seczen                  \
                 + b_coef[L + 4] * seczen * freq                                                \
                 + b_coef[L + 5] * wind + b_coef[L + 6] * wind * freq + b_coef[L + 7] * wind**2 \
                 + b_coef[L + 8] * freq * wind**2 + b_coef[L + 9] * wind * seczen               \
                 + b_coef[L + 10] * wind * seczen * freq
            Azimuth_Emi[2] = Azimuth_Emi[2] + sc * sin(m * rel_az)

            L = 10 * (m - 1) + 90 - 1
            sc = b_coef[L + 1] + b_coef[L + 2] * freq + b_coef[L + 3] * seczen                  \
                 + b_coef[L + 4] * seczen * freq                                                \
                 + b_coef[L + 5] * wind + b_coef[L + 6] * wind * freq + b_coef[L + 7] * wind**2 \
                 + b_coef[L + 8] * freq * wind**2 + b_coef[L + 9] * wind * seczen               \
                 + b_coef[L + 10] * wind * seczen * freq
            Azimuth_Emi[3] = Azimuth_Emi[3] + sc * sin(m * rel_az)

        Azimuth_Emi *= freq_c
        
        if self.version == '6': 
            aniso1_v = self.aniso1_v((frequency, theta, wind))
            aniso1_h = self.aniso1_h((frequency, theta, wind))
            aniso2_v = self.aniso2_v((frequency, theta, wind))
            aniso2_h = self.aniso2_h((frequency, theta, wind))

            Azimuth_Emi[0] = aniso1_v * cos(rel_az) + aniso2_v * cos(2 * rel_az)
            Azimuth_Emi[1] = aniso1_h * cos(rel_az) + aniso2_h * cos(2 * rel_az)

        emis += Azimuth_Emi
        return emis


class two_scale(ocean): 
    """ Implements two-scale ocean surface model and associated reflections

        **Development note**: *The current implementation of the two-scale model is relatively computationally inefficient, 
        and its use is not recommended for large simulations. 
        Future versions of the software should implement a revised two-scale model with improved performance* 

        :param spectrum: Specifies isotropic ocean spectrum model. 
                         Options are: 'Durden-Vesecky', 'Pierson-Moskowitz',
                                      'JONSWAP', 'Donelan', or 'General'

        Several modules are translated from Simon Yueh's original two-scale code in Fortran

    """ 

    # Current coordinate system 
    # Theta is decrease in elevation from nadir/zenith
    # Phi is defined moving counter clockwise where 0deg is looking east and 90 deg is looking north
    # Default ocean spectrum parameters. Override with keyword args

    tau = 7.5e-5  # Ratio of surface tension to density of liquid water 

    def __init__(self, spectrum='Durden-Vesecky', **kwargs):

        self.spectrum = spectrum
        super().__init__(**kwargs)

    def get_ocean_emissivity(self, frequency, sst, sss, uwind, vwind, theta, phi, cutoff_factor=3, 
                             parallel=False, nproc=4, Ns=4): 
        """ Calculates emissivity for a wind-roughened ocean surface using a two-scale model 

            :param frequency: Frequency in MHz (dimension O)
            :param sst: Sea surface temperature in Kelvin (dimension MxN)
            :param sss: Sea surface salinity in psu (dimension MxN)
            :param uwind: 10-m eastward wind in m/s (dimension MxN)
            :param vwind: 10-m northward wind in m/s (dimension MxN)
            :param theta: Topocentric elevation angle in degrees
            :param phi: Topocentric azimuth angle in degrees
            :param cutoff_factor: Changes the scale cutoff wavelength.
                                  Default is 3 (e.g. k0/3 is the cutoff).
                                  See Yueh et al. 1997
            :param parallel: If True, compute in parallel (default False)
            :param nproc: Number of parallel processes (default 4)
            :param Ns: Extent of the integrals (Ns * std. dev. of the distribution)
            
            :return emisV, emisH, U, V: Stokes vector emissivity (dimension (OxMxN))
        """

        # Lowering float resolution increases computation speed
        # Order is [data, freq]
        frequency = frequency[..., np.newaxis].astype(np.float32)
        sst = sst[np.newaxis, ...].astype(np.float32)
        sss = sss[np.newaxis, ...].astype(np.float32)
        theta = np.radians(theta)[np.newaxis, ...].astype(np.float32)
        phi = np.radians(phi)[np.newaxis, ...].astype(np.float32)
        uwind = uwind[np.newaxis, ...].astype(np.float32)
        vwind = vwind[np.newaxis, ...].astype(np.float32)

        wind_phi = arctan2(vwind, uwind)

        # Two scale model may start to fail when winds are lower than 1m/s
        # uwind[abs(uwind) < 1] = 1
        # vwind[abs(vwind) < 1] = 1

        ghz = frequency / 1e3
        wavelength = 0.3 / ghz
        k0 = 2 * np.pi / wavelength
        wind = sqrt(uwind**2 + vwind**2)

        # Convention in Yueh 1997, wind_phi - phi also works
        phi = phi - wind_phi 
        eps = self.dielectric(frequency, sst, sss)
        eps = np.conj(eps)  

        conv_fric = speed_10m_to_friction(wind, method='yueh')
        U12 = friction_to_speed(conv_fric, 12.5)
        # Cutoff wavelength should range from 3-5 wavelengths per Yueh 1997
        # kd gets passed forward
        kd = k0 / cutoff_factor  
        R = (0.003 + 1.92E-3 * U12) / (3.16E-3 * U12) 
        s = 1.5e-4

        k_srs = np.logspace(-5, 5, 10000)
        for i in range(wind.ndim): k_srs = k_srs[np.newaxis, :]

        sks = spectrum_DV(k_srs, wind[..., np.newaxis])  # Assume DV to compute c
        k_srs = k_srs * np.ones(np.shape(sks))
        # Computing c term once and carrying it forward
        Dnum = np.trapz(k_srs**2 * sks * exp(-s * k_srs**2), x=k_srs, axis=-1)
        Ddem = np.trapz(k_srs**2 * sks, x=k_srs, axis=-1) 
        D = Dnum / Ddem
        D = np.where(np.isnan(D), 0, D) 
        c = 2 * (1 - R) / (1 + R) / (1 - D)

        Su2, Sc2 = slope_variance(kd, wind, c, self.spectrum)

        # Per Yueh 1997 Section III,
        # "Integration limits of five times the rms upwind and crosswind slopes"
        # Difference between Ns = 4 and Ns = 5 is on the order of 1e-3
        
        Sxmax = Ns * np.sqrt(Su2)
        Symax = Ns * np.sqrt(Sc2)
        dSx = Sxmax / Ns
        dSy = Symax / Ns
        Sx = np.linspace(-Sxmax, Sxmax, 2 * Ns + 1)
        Sy = np.linspace(-Symax, Symax, 2 * Ns + 1)
        
        emis_out = self.two_scale_core(Sx, Sy, dSx, dSy, Su2, Sc2, theta, phi, 
                                       wind, eps, k0, kd, c, self.spectrum, parallel, nproc)
        return emis_out

    @staticmethod
    def two_scale_core(Sx, Sy, dSx, dSy, Su2, Sc2, theta, phi, 
                       wind, eps, k0, kd, c, model, parallel=False, nproc=4):
        
        if not parallel: 
            emis_out = 0 
            for inc_Sx in Sx:
                for inc_Sy in Sy:  
                    res = _tsc([inc_Sx, inc_Sy], dSx, dSy, Su2, Sc2,
                               theta, phi, wind, eps, k0, kd, c, model)
                    emis_out += res
        else: 
            pack = tuple(product(Sx, Sy))
            args = (dSx, dSy, Su2, Sc2, theta, phi, wind, eps, k0, kd, c, model)
            full_args = tuple((x, *args) for x in pack)
    
            with Pool(nproc) as pool:
                res = pool.starmap(_tsc, full_args)
                pool.close()
                pool.join()
            emis_out = np.sum(np.array(res), axis=0)
        return emis_out


# Ocean-related utility methods 
# Mostly in support of the two-scale class 

def speed_10m_to_friction(wind, method='hwang'): 
    """ Converts 10m-winds to wind friction

        'yueh' method is a polynomial fit to an iterative calculation procedure written
        by Simon Yueh, and the 'hwang' method implements Paul Hwang's drag coefficient
        for 10-m winds 

        :param wind: 10-meter wind speed in m/s (must be an array)
        :param method: Calculation method of 'hwang' or 'yueh'
    """
    if method == 'hwang':
        # Below expression is from Hwang 2012 based on empirical measurements 
        # This is more accurate since friction doesn't monotonically increase with speed, 
        # but leads to mismatches when used with friction_to_speed 
        low_speed = 1e-5 * (-0.160 * wind**2 + 9.67 * np.abs(wind) + 80.58)
        high_speed = 2.23e-3 * (wind / 35)**-1
        friction_coeff = np.zeros(np.shape(wind))

        # friction_coeff[wind <= 35] = low_speed[wind <= 35]
        # friction_coeff[wind > 35] = high_speed[wind > 35]

        # Raveling for numba 
        fs = friction_coeff.ravel()
        ls = low_speed.ravel()
        hs = high_speed.ravel()
        ws = wind.ravel()
        fs[ws <= 35] = ls[ws <= 35]
        fs[ws > 35] = hs[ws > 35]
        
        out = wind * friction_coeff**0.5  
    elif method == 'yueh': 
        # Fit to polynomial from Simon's iterative expression 
        # Linear relationship with friction_to_speed, but poor accuracy. 
        out = 1.032e-3 * wind**2 + 28.87e-3 * np.abs(wind) + 14.76e-3
    else: 
        raise ValueError('Specify valid wind speed to friction conversion (hwang or yueh)')

    return out 


def friction_to_speed(friction, z):
    """ Convert wind friction to velocity using formula from Yueh 1997
        :param friction: Wind friction
        :param z: Height of output wind speed 
    """ 

    Zo = 6.84e-5 / np.abs(friction) + 4.28e-3 * friction**2 - 4.43e-4
    return friction / 0.4 * log(z / Zo)


def spectrum_PM(k, wind): 
    """ Isotropic surface spectrum of Pierson and Moskowitz 1964

        :param k: Ocean wavenumber (rad/m)
        :param wind: 10-m wind speed (m/s)

        :return: Wavenumber spectrum in m^3
    """

    # Per the below expression, the peak wavenumber is located at 
    # kp = np.sqrt(2 / 3 * beta) * kc

    a0 = 8e-3  # An alternative value is 6e-3, see Yueh 1997
    kj = 2
    beta = 0.74
    ustar = speed_10m_to_friction(wind, method='yueh')
    u19 = friction_to_speed(ustar, 19.5)
    kc = spc.g / u19**2
    b0 = exp(beta * (kc / kj)**2) * a0
    pm = b0 * k**-3 * exp(-beta * (kc / k)**2)
    pm = np.where(np.isnan(pm), 0, pm) 
    return pm


def spectrum_DV(k, wind): 
    """ Isotropic surface spectrum of Durden and Vesecky 1985

        :param k: Ocean wavenumber (rad/m)
        :param wind: 10-m wind speed (m/s)

        :return: Wavenumber spectrum in m^3
    """

    gamma = 7.25e-5
    a0 = 8e-3  # An alternative value is 6e-3, see Yueh 1997
    a = 0.225
    b = 1.25 
    kj = 2.
    ustar = speed_10m_to_friction(wind, method='yueh')
    u19 = friction_to_speed(ustar, 19.5)
    kc = spc.g / u19**2

    pm = spectrum_PM(k, wind)
    gs = spc.g + gamma * k**2
    dv = a0 * k**-3 * (b * k * ustar**2 / gs)**(a * log10(k / kj))
    out_sk = pm 
    if np.shape(k) != np.shape(dv): 
        k = k * np.ones(np.shape(dv))
        kc = kc * np.ones(np.shape(dv))
    
    # Numpy indexing
    out_sk[k < (kc * 1e-5)] = 0
    out_sk[k > kj] = dv[k > kj]

    # Ravel assignment for jit
    # outs = out_sk.ravel()
    # dvs = dv.ravel()
    # kcs = kc.ravel()
    # ks = k.ravel()
    # outs[ks < (kcs * 1e-5)] = 0 
    # outs[ks > kj] = dvs[ks > kj]

    return out_sk


def spectrum_jonswap(k, wind): 
    """ Isotropic surface spectrum derived from JONSWAP measurements

        Fetch term removed by setting the center frequency equal to the PM spectrum
        for a fully developed ocean surface
            
        :param k: Ocean wavenumber (rad/m)
        :param wind: 10-m wind speed (m/s)

        :return: Wavenumber spectrum in m^3
    """
    tau = 7.5e-5  # Ratio of surface tension to density of liquid water 
    # Compute spectral peak
    beta = 0.74
    ustar = speed_10m_to_friction(wind, method='yueh')
    u19 = friction_to_speed(ustar, 19.5)
    kc = spc.g / u19**2
    kp = np.sqrt(2 / 3 * beta) * kc

    # Convert to wave frequency
    w = np.sqrt(spc.g * k + tau * k**3)
    wp = np.sqrt(spc.g * kp + tau * kp**3)
    w_pound = wp * wind / spc.g 

    # Model spectrum
    aj = 7.33e-3 * w_pound**0.87
    gj = 2.29 * w_pound**0.32 
    sig_a = 9.85e-2 * w_pound**-0.32
    sig_b = 1.05e-1 * w_pound**-0.16
    first_term = aj * spc.g**2 * w**-5 * np.exp(-5 / 4 * (w / wp)**-4)
    gamj_lf = np.exp(-(w - wp)**2 / (2 * sig_a**2 * wp**2)) 
    gamj_hf = np.exp(-(w - wp)**2 / (2 * sig_b**2 * wp**2)) 
    second_lf = gj**gamj_lf
    second_hf = gj**gamj_hf
    S = first_term * second_lf 
    
    # Numpy indexing
    # S[w > wp] = first_term[w > wp] * second_hf[w > wp]
    # S = S * k**-0.5 * np.sqrt(spc.g)  # convert to m^3 units

    # Ravel for numba, assign and return the reference in proper shape 
    Ss = S.ravel()
    fts = first_term.ravel()
    shf = second_hf.ravel()
    ws = w.ravel()
    wps = wp.ravel()
    Ss[ws > wps] = fts[ws > wps] * shf[ws > wps]
    
    return S 


def spectrum_donelan(k, wind): 
    """ Isotropic surface spectrum of Donelan et al. 1985

        Fetch term removed by setting the center frequency equal to the PM spectrum
        for a fully developed ocean surface
            
        :param k: Ocean wavenumber (rad/m)
        :param wind: 10-m wind speed (m/s)

        :return: Wavenumber spectrum in m^3
    """

    tau = 7.5e-5  # Ratio of surface tension to density of liquid water 
    # Compute spectral peak
    beta = 0.74
    ustar = speed_10m_to_friction(wind, method='yueh')
    u19 = friction_to_speed(ustar, 19.5)
    kc = spc.g / u19**2
    kp = np.sqrt(2 / 3 * beta) * kc

    # Convert to wave frequency
    w = np.sqrt(spc.g * k + tau * k**3)
    wp = np.sqrt(spc.g * kp + tau * kp**3)
    w_pound = wp * wind / spc.g 
    
    ad = 6e-3 * w_pound**0.55
    gd = 1.7 * np.ones(np.shape(w_pound))
    
    # Numpy indexing
    # mask = w_pound > 1
    # gd[mask] = gd[mask] + 6 * np.log(w_pound[mask])

    # Raveling for numba 
    gds = gd.ravel()
    wps = w_pound.ravel()
    mask = wps > 1.
    gds[mask] = gds[mask] + 6 * np.log(wps[mask])
    
    sig_d = 8e-2 * (1 + 4 * w_pound**-3)

    first_term = ad * spc.g**2 * wp**-1 * w**-4 * np.exp(-(w / wp)**-4)
    gamd = np.exp(-(w - wp)**2 / (2 * sig_d**2 * wp**2)) 
    second = gd**gamd
    S = first_term * second
    S = S * k**-0.5 * np.sqrt(spc.g)  # convert to m^3 units
    
    return S 


def spectrum_general(k, wind, s_g=4.5, beta_g=4): 
    """ Isotropic general model surface spectrum from Hwang 2017

        :param k: Ocean wavenumber (rad/m)
        :param wind: 10-m wind speed (m/s)
        :param s_g: Variable 

        :return: Wavenumber spectrum in m^3

    """
    tau = 7.5e-5  # Ratio of surface tension to density of liquid water 
    # Compute spectral peak
    beta = 0.74
    ustar = speed_10m_to_friction(wind, method='yueh')
    u19 = friction_to_speed(ustar, 19.5)
    kc = spc.g / u19**2
    kp = np.sqrt(2 / 3 * beta) * kc

    # Convert to wave frequency
    w = np.sqrt(spc.g * k + tau * k**3)
    wp = np.sqrt(spc.g * kp + tau * kp**3)
    w_pound = wp * wind / spc.g 

    A_a = 1.3e-3 * s_g + 1.64e-3
    a_a = 4.83e-1 * s_g - 1.49
    A_g = 4.42e-1 * s_g + 3.93e-1
    a_g = -3.63 * s_g + 19.74
    A_s = -5.39e-2 * s_g + 3.44e-1 
    a_s = 2.05e-9 * s_g + 5.5e-2 
    a_1 = A_a * w_pound**a_a
    g_1 = A_g + a_g * np.log(w_pound)
    s_1 = A_s + a_s * np.log(w_pound)
    ag = a_1 * (1 - 0.3 * np.tanh(0.1 * w_pound))
    gg = g_1 * (1 - 0.5 * np.tanh(0.1 * w_pound))
    sg = s_1     
    zeta = w / wp
    K = (s_g / beta_g)**(1 / beta_g)
    first_term = ag * spc.g**2 * wp**-5 * zeta**-s_g * np.exp(-(zeta / K)**-beta_g)
    gamg = np.exp(-(1 - zeta)**2 / (2 * sg**2))
    second = gg**gamg
    # k = k * np.ones((*np.shape(k)[:-self.kc.ndim], *np.shape(self.kc)), dtype=np.float32)
    S = first_term * second
    S = S * k**-0.5 * np.sqrt(spc.g)  # convert to m^3 units

    return S  


def get_spectrum(k, phi, wind, c=None, model='Durden-Vesecky'):
    """ Returns direction dependent ocean spectrum
        Choice of isotropic spectrum specified by model 
        See Yueh 1997

        :param k: Ocean wavenumber in 1/m
        :param phi: Look angle azimuth in radians
        :param wind: 10-m wind in m/s 
        :param c: C parameter for directional spectrum
                  costly to compute, so provided as an arg
        :param model: Isotropic spectrum model
                 Options are: 'Durden-Vesecky' (default)
                              'Pierson-Moskowitz'
                              'JONSWAP'
                              'Donelan'
                              'General'

    """

    if model == 'Durden-Vesecky': 
        Sk = spectrum_DV(k, wind)
    elif model == 'Pierson-Moskowitz': 
        Sk = spectrum_PM(k, wind)
    elif model == 'JONSWAP': 
        Sk = spectrum_jonswap(k, wind)
    elif model == 'Donelan': 
        Sk = spectrum_donelan(k, wind)
    elif model == 'General': 
        Sk = spectrum_general(k, wind)

    s = 1.5e-4

    if c is None: 
        # This is expensive, so c should generally be precomputed
        conv_fric = speed_10m_to_friction(wind, method='yueh')
        U12 = friction_to_speed(conv_fric, 12.5)
        R = (0.003 + 1.92E-3 * U12) / (3.16E-3 * U12) 
        # k_srs = np.logspace(-5, 5, 10000, dtype=np.float32)
        k_srs = np.logspace(-5, 5, 10000) * np.ones((*np.shape(wind.T), 1))
        k_srs = k_srs.T
        sks = spectrum_DV(k_srs, wind)  # Assume DV to compute c
        k_srs = k_srs.T
        sks = sks.T 
        Dnum = np.trapz(k_srs**2 * sks * exp(-s * k_srs**2), x=k_srs)
        Ddem = np.trapz(k_srs**2 * sks, x=k_srs) 
        D = Dnum.T / Ddem.T
        D = np.where(np.isnan(D), 0, D) 

        c = 2 * (1 - R) / (1 + R) / (1 - D)

    r = c * (1 - exp(-s * k**2))
    Psi = (1 + r * cos(2 * phi))
    return Sk * Psi / (2 * np.pi * k)


def slope_to_normal(psix, psiy):
    """Used to calculate the direction of the surface 
       normal given the slope angle. 
        
        :param psix: angle of the slope in x for the local plane
        :param psiy: angle of the slope in y for the local plane
            
        :return: thetan - zenith angle of surface normal
                 phin - azimuth angle of surface normal
    """

    nx = -tan(psix)
    ny = -tan(psiy)
    nz = 1
    c = sqrt(nx**2 + ny**2 + nz**2)
    nx = nx / c
    ny = ny / c
    nz = nz / c
    thetan = arccos(nz)
    phin = arctan2(ny, nx)

    # Ravel for numba
    phir = phin.ravel()
    thetar = thetan.ravel()
    phir[thetar < 1e-6] = 0

    return thetan, phin


def pdf_slope(Su2, Sc2, Zx, Zy):
    """ Returns the pdf of the slope of the large scale
        surface which is assumed to be Gaussian.
        This PDF was used by S. Durden for his two scale theory
        for a large scale surface. 

        :param Su2: Slope variance in x (up-wind)
        :param Sc2: Slope variance in y (cross-wind)
        :param Zx: Slope in x (up-wind)
        :param Zy: Slope in y (cross-wind)

        :return: Slope PDF
        
    """
    Su = sqrt(Su2)
    Sc = sqrt(Sc2)
    arg = (Zx**2 / Su2 + Zy**2 / Sc2) * 0.5
    slope = 1 / (2 * np.pi * Su * Sc) * exp(-arg)
    return slope


def cox_munk_variance(wind): 
    """ Slope variance expressions of Cox and Munk 

        :param wind: 10-m wind speed 

        :return: Su2 - Slope variance in up wind direction
                 Sc2 - Slope variance in cross wind direction
    """ 
    us = speed_10m_to_friction(wind, method='yueh')
    U_12_5 = friction_to_speed(us, 12.5)
    Su2 = 3.16e-3 * U_12_5
    Sc2 = 3e-3 + 1.92e-3 * U_12_5
    return Su2, Sc2    


def slope_variance(kd, wind, c, model='Durden-Vesecky'):
    """ Used to numerically calculate the variances of 
        the large scale surface with a given cutoff frequency.

        :param kd: high frequency cutoff for the large scale wave
        :param wind: 10-m wind speed 
            
        :return: Su2 - Slope variance in up wind direction
                 Sc2 - Slope variance in cross wind direction
        
    """ 

    s = 1.5e-4
    Nk = 10000
    dk = kd / Nk
    SumS = 0
    Sumf = 0
        
    krange = np.arange(1, Nk + 1)
    dk = dk.reshape((*np.shape(dk), 1))
    arg_wind = wind.reshape((*np.shape(wind), 1))
    k = (krange - 0.5) * dk

    if model == 'Durden-Vesecky': 
        iso_spec = spectrum_DV(k, arg_wind)
    elif model == 'Pierson-Moskowitz': 
        iso_spec = spectrum_PM(k, arg_wind)
    elif model == 'JONSWAP': 
        iso_spec = spectrum_jonswap(k, arg_wind)
    elif model == 'Donelan': 
        iso_spec = spectrum_donelan(k, arg_wind)
    elif model == 'General': 
        iso_spec = spectrum_general(k, arg_wind)
    else: 
        print('Defaulting to DV spectrum')
        iso_spec = spectrum_DV(k, arg_wind)

    SumS = iso_spec * k**2
    Sumf = iso_spec * k**2 * (1 - exp(-s * k**2))
    SumS = np.sum(SumS * dk, axis=-1)
    Sumf = np.sum(Sumf * dk, axis=-1)
    St = SumS  # Total variance
    Su2 = 0.5 * (St + 0.5 * c * Sumf) 
    Sc2 = 0.5 * (St - 0.5 * c * Sumf)

    return Su2, Sc2


def angle_pol_trans(theta, phi, psix, psiy): 
    """This routine is used to covert the incident direction
        in principal frame to local frame. 
            
        :param theta: zenith angle
        :param phi: polar angle
        :param psix: angle of the slope in x for the local plane
        :param psiy: angle of the slope in y for the local plane
            
        :return: thetal - local zenith angle 

                 phil - local azimuth angle 

                 hh - vector product of h pol. vector and h'' pol. vector 

                 hv - vector product of h pol. vector and v'' pol. vector 

        """
    thetan, phin = slope_to_normal(psix, psiy)
    st = sin(thetan)
    ct = cos(thetan)
    sp = sin(phin)
    cp = cos(phin)
    amp = sqrt(ct**2 + (st * cp)**2)
    cb = ct / amp
    sb = st * cp / amp

    A = np.empty((3, 3, *np.shape(cb)))
    A[0, 0] = cb
    A[0, 1] = np.zeros(np.shape(cb))
    A[0, 2] = -sb
    A[1, 0] = -st * sp * sb
    A[1, 1] = ct * cb + st * cp * sb
    A[1, 2] = -cb * st * sp
    A[2, 0] = st * cp
    A[2, 1] = st * sp
    A[2, 2] = ct

    vk = np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta) * np.ones(np.shape(phi))])
    vkpp = np.einsum('ij...,j...->i...', A, vk)
    ampvkpp = np.linalg.norm(vkpp, axis=0)
    vkpp = vkpp / ampvkpp 

    stl = sqrt(vkpp[0]**2 + vkpp[1]**2)
    thetal = arcsin(stl)
    phil = arctan2(vkpp[1], vkpp[0])

    h = np.array([sin(phi), -cos(phi), np.zeros(np.shape(phi))])
    # v = np.array([-cos(theta) * cos(phi), -cos(theta) * sin(phi), sin(theta)])
    hpp = np.array([sin(phil), -cos(phil), np.zeros(np.shape(phil))])
    vpp = np.array([-cos(thetal) * cos(phil), -cos(thetal) * sin(phil), sin(thetal)])

    hh = np.einsum('i...,ji..., j...->...', h, A, hpp)
    hv = np.einsum('i...,ji..., j...->...', h, A, vpp)

    return thetal, phil, hh, hv


def foam_fraction(wind): 
    """ Foam parameterization from Hwang 2012 

        :param wind: 10-m wind speed (m/s) 

        :return: Foam fraction for that wind speed 

    """ 

    foam_exp1 = 3
    foam_exp2 = 2.5
    wind_friction = speed_10m_to_friction(wind, method='yueh')
    low_fraction = 0.3 * (wind_friction - 0.11)**foam_exp1
    low_fraction[wind_friction < 0.11] = 0
    foam_fraction = 0.07 * (wind_friction)**foam_exp2
    loc = np.where(abs(wind_friction - 0.4) == np.min(abs(wind_friction - 0.4)))
    offset = (low_fraction[loc] / foam_fraction[loc])
    foam_fraction = foam_fraction * offset 
    foam_fraction[wind_friction < 0.4] = low_fraction[wind_friction < 0.4]
    foam_fraction[foam_fraction > 0.5] = 0.5
    return foam_fraction


def _tsc(inc_S, dSx, dSy, Su2, Sc2, theta, phi, 
         wind, eps, k0, kd, c, model):
    """ Core computations for slope roughness 
        The user should avoid calling this function on its own
    """
    inc_Sx, inc_Sy = inc_S
    # Large scale - tilted facets        
    projection = 1 - tan(theta) * (inc_Sx * cos(phi) + inc_Sy * sin(phi))
    pdf = pdf_slope(Su2, Sc2, inc_Sx, inc_Sy)
    const = projection * pdf * dSx * dSy

    # From wind-centric (incidence) to slope-centric (local) coordinates
    psix = arctan(inc_Sx)
    psiy = arctan(inc_Sy)
    thetal, phil, hh, hv = angle_pol_trans(theta, phi, psix, psiy)
    x2 = sin(thetal) * cos(phil)
    y2 = sin(thetal) * sin(phil)
    z2 = cos(thetal) 
    local_angle = np.arccos(z2 / sqrt(x2**2 + y2**2 + z2**2))

    # Hydrodynamic modulation 
    hmod = 0.4
    hw = 1 - hmod * (inc_Sx / np.sqrt(Su2))
    hw[hw < 0.5] = 0.5
    hw[hw > 1.5] = 1.5

    # Evaluate the coherent reflectivity
    Rh0, Rv0, shh, svh, svv, shv = R_coherent(thetal, phil + np.pi, eps, wind, k0, kd, c=c, model=model) 

    shh = shh * hw
    shv = shv * hw
    svh = svh * hw
    svv = svv * hw
    Rcoh = np.empty((4, *np.shape(projection)), dtype=np.complex64)
    carg = Rh0 * np.conj(shh)
    Rcoh[0] = abs(Rh0)**2 + 2 * np.real(carg)
    carg = Rv0 * np.conj(svv)
    Rcoh[1] = abs(Rv0)**2 + 2 * np.real(carg)
    carg = svh * np.conj(Rh0) + Rv0 * np.conj(shv)
    Rcoh[2] = 2 * np.real(carg)
    carg = svh * np.conj(Rh0) + Rv0 * np.conj(shv)
    Rcoh[3] = 2 * np.imag(carg)

    # Evaluate the incoherent reflectivity 
    Rincoh = R_incoherent(thetal, phil, eps, wind, k0, kd, c=c, model=model)
    Rincoh = Rincoh * hw

    Rt = np.real(Rcoh + Rincoh)
    emis = np.empty(np.shape(Rt))
    emis_mid = np.empty(np.shape(Rt))
    emis_mid[0] = 1 - Rt[1]
    emis_mid[1] = 1 - Rt[0]
    emis_mid[2] = -Rt[2]
    emis_mid[3] = -Rt[3]

    us = speed_10m_to_friction(wind, method='yueh')
    foam_frac = foam_fraction(us)
            
    # Stogryn foam 
    # emisH_foam, emisV_foam = dielectric.foam_Stogryn(frequency, np.degrees(local_angle))

    # Anguelova foam 
    # Quadratic mixing rule 
    foam_eps = (foam_frac + (1 - foam_frac) * np.sqrt(eps))**2
    Rh = (cos(local_angle) - (foam_eps - sin(local_angle)**2)**0.5) / (cos(local_angle) + (foam_eps - sin(local_angle)**2)**0.5)
    Rv = (foam_eps * cos(local_angle) - (foam_eps - sin(local_angle)**2)**0.5) / (foam_eps * cos(local_angle) + (foam_eps - sin(local_angle)**2)**0.5)
    emisV_foam = (1 - Rv * np.conj(Rv))
    emisH_foam = (1 - Rh * np.conj(Rh))
    emis_foam = np.array([emisV_foam, emisH_foam, np.zeros(np.shape(emisV_foam)), np.zeros(np.shape(emisH_foam))])
    emis_foam = np.real(emis_foam)
    emis_total = foam_frac * emis_foam + (1 - foam_frac) * emis_mid

    # Converting from polarization vector from local to global
    # hh = vv, -hv = vh 
    c2 = hh**2
    cs = hh * hv
    s2 = hv**2
    emis[0] = emis_total[0] * c2 + emis_total[1] * s2 - emis_total[2] * cs
    emis[1] = emis_total[0] * s2 + emis_total[1] * c2 + emis_total[2] * cs
    emis[2] = (emis_total[0] - emis_total[1]) * 2 * cs + emis_total[2] * (c2 - s2)
    emis[3] = emis_total[3]
    emis_out = emis * const  

    return emis_out 


def R_coherent(thetai, phii, eps, wind, k0, kd, c=None, model='Durden-Vesecky'): 
    """ Computes the coherent reflectivity coefficients for Bragg scattering 
        using the small perturbation method.

        :param thetai: Incident elevation angle 
        :param phii: Incident azimuth angle 
        :param eps: Surface dielectric constant 
        :param k0: Electromagnetic wavenumber 
        :param wind: 10-m wind speed 

        :return: Rh0, Rv0 - Specular reflectivities for H and V pol

                 shh, svh, svv, shv - Second order coherent reflection co-pol and cross-pol terms
    """

    k1 = k0 * sqrt(eps)
    kzi = k0 * cos(thetai)
    krhoi = sqrt(k0**2 - kzi**2)
    kxi = krhoi * cos(phii)
    kyi = krhoi * sin(phii)
    k1zi = sqrt(k1**2 - krhoi**2)
    Rh0 = (kzi - k1zi) / (kzi + k1zi)
    Rv0 = (k1**2 * kzi - k0**2 * k1zi) / (k1**2 * kzi + k0**2 * k1zi)

    Nrho = 5000  # Best accuracy is at 10000 samples
    Rkmax = 10 * k0
    Nphi = 180
    dphi = 2 * np.pi / Nphi
    krho = np.logspace(log10(kd) - 3, log10(k0) + 2, Nrho)
    phi = np.arange(0, Nphi)
    for i in range(kzi.ndim): phi = phi[:, np.newaxis]
    krho = krho[:, np.newaxis]
    dkrho = np.gradient(krho, axis=0) 

    kz = sqrt(k0**2 - krho.astype(np.complex64)**2)
    k1z = sqrt(k1**2 - krho.astype(np.complex64)**2)
    t1 = k0**2 - k1**2
    t2 = kz + k1z
    t3 = krho**2 + kz * k1z
    t4 = t1 / t2 / t3
    krho2 = krho**2

    s11 = k1zi + t4 * kz * k1z
    s12 = t4 * krho2
    s21 = krho * krhoi * k1**2 / t3
    s22 = k1zi * t4 * krho2
    s31 = -t4 * krho2 * krhoi**2 / k0**2 + k1zi
    s32 = k1zi * (-2) * krho * krhoi / t3
    s33 = k1zi**2 / k1**2 * t1 / t2
    s34 = -s33 * krho2 / t3
    phi = phi * dphi + phii 
    phi = phi[np.newaxis, :]
    kx = krho * cos(phi)
    ky = krho * sin(phi)

    kdx = -kx + kxi
    kdy = -ky + kyi
    p = cos(phi - phii)
    q = sin(phi - phii)
    s1 = s11 + s12 * p**2 
    s2 = q * (s21 + s22 * p)
    s3 = s31 + s32 * p + s33 + s34 * p**2
    s4 = s2
    kdr = sqrt(kdx**2 + kdy**2)
    kdph = arctan(kdy / kdx)
    sdg = get_spectrum(kdr, kdph, wind, c=c, model=model)
    sdg[kdr < kd] = 0

    ks = krho * sdg * dkrho
    dshh = np.sum(np.sum(ks * s1, axis=0), axis=0)
    dshv = np.sum(np.sum(ks * s2, axis=0), axis=0)
    dsvv = np.sum(np.sum(ks * s3, axis=0), axis=0)
    dsvh = np.sum(np.sum(ks * s4, axis=0), axis=0)
    cross = 2 * kzi * k0 / (k0**2 * k1zi + k1**2 * kzi) * -t1 / (kzi + k1zi)
    shh = 2 * kzi * -t1 / (kzi + k1zi)**2 * dshh * dphi
    shv = cross * dshv * dphi
    svv = 2 * kzi * t1 / (k1**2 * kzi + k0**2 * k1zi)**2 * k1**2 * k0**2 * dsvv * dphi
    svh = -cross * dsvh * dphi

    return Rh0, Rv0, shh, svh, svv, shv


def R_incoherent(thetal, phil, eps, wind, k0, kd, c=None, model='Durden-Vesecky'):
    """ Computes the incoherent reflectivity from a 
        slightly perturbed Gaussian random rough surface
        with ocean spectrum. 

        :param thetal: local observing polar angle
        :param phil: local observing azimuth angle
        :param eps: permittivity of lower half space medium
        :param k0: free space electromagnetic wavenumber
        :param wind: 10m wind speed 
            
        :return: Rx - Incoherent reflectivity

    """

    Ntheta = 90
    Nphi = 360

    dtheta = np.pi / 2 / Ntheta
    dphi = 2 * np.pi / Nphi

    theta = np.arange(1, Ntheta + 1)
    phi = np.arange(1, Nphi + 1)

    for i in range(phil.ndim): 
        theta = theta[:, np.newaxis]
        phi = phi[:, np.newaxis]

    theta = (theta - 0.5) * dtheta
    phi = (phi - 0.5) * dphi + phil
    theta = theta[:, np.newaxis]
    phi = phi[np.newaxis, :]

    # Bistatic SPM
    k1 = k0 * sqrt(eps)
    kxi = k0 * sin(theta) * cos(phi)
    kyi = k0 * sin(theta) * sin(phi)
    kzi = k0 * cos(theta)   
    k1zi = sqrt(k1**2 - kxi**2 - kyi**2)

    kxs = k0 * sin(thetal) * cos(phil)
    kys = k0 * sin(thetal) * sin(phil)
    kzs = k0 * cos(thetal)   
    k1zs = sqrt(k1**2 - kxs**2 - kys**2)
    shape = np.shape(theta * phi)
    S = np.empty((4, *shape), dtype=np.complex64)

    # These coefficients are the same as the first order coefficients in Yueh 1997 Appendix B
    # except multiplied by a factor of k0
    t1 = k1**2 - k0**2
    t2 = k1**2 * kzi + k0**2 * k1zi
    t3 = k1**2 * kzs + k0**2 * k1zs
    c1 = 2 * kzi * t1
    cphi = cos(phil - phi)
    sphi = sin(phil - phi)

    S[0] = c1 / (kzs + k1zs) / (kzi + k1zi) * cphi
    S[1] = c1 * k0 * k1zi / t2 / (kzs + k1zs) * sphi
    S[2] = c1 * k0 * k1zs / t3 / (kzi + k1zi) * sphi
    S[3] = c1 / t3 / t2 * (k1**2 * k0**2 * sin(thetal) * sin(theta) - k0**2 * k1zs * k1zi * cphi)

    kdx = -kxs + kxi
    kdy = -kys + kyi
    kdr = sqrt(kdx**2 + kdy**2)
    kdph = arctan(kdy / kdx)
    sdg = get_spectrum(kdr, kdph, wind, c=c, model=model)
    sdg[kdr < kd] = 0

    Di = 4 * np.pi * k0**2 * cos(thetal)**2 * sdg / cos(theta)
    C = S[:, np.newaxis, :] * np.conj(S[np.newaxis, :]) * Di
    # End bistatic SPM

    shape = np.shape(theta * phi)
    Cx = np.empty((4, *shape), dtype=np.complex64)
    Cx[0] = C[0, 0] + C[1, 1]
    Cx[1] = C[2, 2] + C[3, 3]
    Cx[2] = C[2, 0]
    Cx[3] = C[3, 1]
    const = 1 / (4 * np.pi) * sin(theta) * dtheta * dphi * cos(theta) / cos(thetal)
    dRx = np.empty((4, *shape), dtype=np.complex64)
    dRx[0] = Cx[0]
    dRx[1] = Cx[1]
    carg = Cx[2] + Cx[3]
    dRx[2] = 2 * np.real(carg)
    dRx[3] = 2 * np.imag(carg)
    Rx = np.sum(np.sum(dRx * const, axis=1), axis=1)
    return Rx 
