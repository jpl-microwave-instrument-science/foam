import os 
import numpy as np
from numpy import sin, cos, radians
import pandas as pd 
import scipy.interpolate as spi 
import scipy.constants as spc 
from cftime import date2num

import foam.geomag as geomag

from .utils.config import cache_path, dir_path
from .utils import reader

try:
    import iri2016
    has_IRI = True
except ImportError: 
    has_IRI = False


class ionosphere(): 
    """ The ionosphere class reads maps of Total Electron Content (TEC) from CDDIS ancillary files
        or simulates these maps using the IRI2016 package
        
        :param datetime: Either a single date or a pair of dates bracketing an interval of time
                         Several formats are permissible, such as
                         - String or iterable of strings
                         - Python datetime or iterable of datetimes 
                         - Numpy datetime64 or iterable
                         - Pandas timestamp or iterable 

        :param online: If true, ionospheric data is downloaded from CDDIS servers based on the datetime parameter.
        :param tec_file: Filename for a CDDIS ionosphere TEC map. If None, the FOAM cache default is used
        :param mag_file: Filename for World Magnetic Model coefficients, If None, the FOAM cache default is used 
        :param tec_reader: Reader object for TEC, default is IONEXReader
        :param IRI: If true, ionospheric TEC is simulated using the IRI2016 package. 
                    This mode is very experimental, and subject to be removed
        :param verbose: If true, prints verbose output

    """

    time_reference = 'seconds since 2000-01-01 12:00:0.0'  # J2000

    def __init__(self, datetime='2015-01-01', online=False, tec_file=None, mag_file=None, 
                 tec_reader=reader.IONEXReader, tec_reader_kwargs=None, IRI=False, verbose=False, **kwargs):

        self.datetime = pd.to_datetime(datetime)
        self.online = online 
        self.tec_file = tec_file
        self.mag_file = mag_file
        self.tec_reader = tec_reader
        if tec_reader_kwargs is None: 
            self.tec_reader_kwargs = {}
        else:    
            self.tec_reader_kwargs = tec_reader_kwargs
        self.IRI = IRI
        self.verbose = verbose

        for key, value in kwargs.items(): 
            setattr(self, key, value)

        if self.tec_file is None and not self.online:
            self.datetime = pd.to_datetime(['2015-01-01 12:00'])
            self.tec_file = os.path.join(cache_path, 'ionosphere', 'jplg0010.05i')
        if self.mag_file is None: 
            self.mag_file = os.path.join(dir_path, 'assets', 'magneticfield', 'WMM2020.COF')

        if self.IRI: 
            if not has_IRI: 
                raise RuntimeError('User needs to install the iri2016 package to use this mode')
            self.make_ionosphere()
        else: 
            self.read_ionosphere()

    def read_ionosphere(self):
        """ Reads ionospheric state 
        """ 
        self.tec_reader = self.tec_reader(self.datetime, self.online, file=self.tec_file)
        self.TEC_interp = self.tec_reader.read(self.tec_reader_kwargs)
        
    def make_ionosphere(self): 
        """ Uses the IRI2016 implementation of the International Reference Ionosphere (http://irimodel.org/)
            to generate maps of ionospheric TEC. Due to the slow speed of IRI runs, this is only done for a single time
        """
        print('Using IRI 2016 to calculate ionosphere TEC on a 5 degree grid')
        print('This can take several minutes')

        # 5 degree grid
        grid_lat = np.linspace(90, -90, 36)
        grid_lon = np.linspace(-180, 180, 72)
        TEC = np.array([iri2016.IRI(self.datetime[0], [0, 800, 800], x, y).TEC.values for x in grid_lat for y in grid_lon])
        TEC = TEC.reshape(36, 72) / 1e16  # Convert to TECu
        self.TEC_interp = spi.RegularGridInterpolator((grid_lat[::-1], grid_lon), TEC[::-1, :], bounds_error=False, fill_value=18)

    @staticmethod
    def faraday_rotation(far_angle, TB): 
        """ Applies faraday rotation to the polarized brightness temperatures
            
            :param far_angle: Angle of rotation in radians
            :param TB: Block matrix of polarimetric brightness temperatures [TBV, TBH, U, V] (Size 4xMx...)

            :return: 
        """

        R = np.array([[cos(far_angle)**2, sin(far_angle)**2, 0.5 * sin(2 * far_angle), np.zeros(np.shape(far_angle))],
                    [sin(far_angle)**2, cos(far_angle)**2, -0.5 * sin(2 * far_angle), np.zeros(np.shape(far_angle))],
                    [-sin(2 * far_angle), sin(2 * far_angle), cos(2 * far_angle), np.zeros(np.shape(far_angle))], 
                    [np.zeros(np.shape(far_angle)), np.zeros(np.shape(far_angle)), np.zeros(np.shape(far_angle)), np.ones(np.shape(far_angle))]])
        
        # Correct nans 
        mask = np.isnan(TB)
        TB[mask] = 0 
        TA = np.einsum('ij...,j...->i...', R, TB)
        TA[mask] = np.nan
        return TA

    def compute_faraday_angle(self, frequency, time, lat, lon, theta, phi, in_epoch=False, use_time=True):
        """ Computes the angle of ionospheric faraday rotation for polarized microwave emission.
            This calculation uses a Python adaptation of the World Magnetic Model written by Christopher Weiss 
            https://github.com/cmweiss/geomag. Due to its relatively small size, the geomag package has been 
            included directly with FOAM without requiring a download. It has also been modified to support numpy arrays. 
            
            :param frequency: Measurement frequency in MHz (shape N)
            :param time: String or array of times, converted to interpolator
                         epoch reference if in_epoch=False
            :param lat: Latitude in degrees 
            :param lon: Longitude in degrees 
            :param theta: Radiometer elevation angle in degrees
            :param phi: Radiometer azimuth angle in degrees 
            :param in_epoch: See above
            :param use_time: Toggles use of time in case of heterogeneous ancillary data

            :return: Faraday rotation angle in radians (shape Nx...)
        """ 

        if not use_time: 
            time = self.datetime[0]
            in_epoch = False
        if not in_epoch: 
            time = date2num(pd.to_datetime(time).to_pydatetime(), self.time_reference)

        if self.IRI: 
            TEC = self.TEC_interp((lat, lon))
        else: 
            TEC = self.TEC_interp((time, lat, lon))
        gm = geomag.GeoMag(self.mag_file)
        mag = gm.GeoMag(lat, lon, h=(675e3 / spc.foot))
        x = sin(radians(theta)) * cos(radians(phi))
        y = sin(radians(theta)) * sin(radians(phi))
        z = cos(radians(theta)) 
        mag_field = -(mag.bx * x + mag.by * y + mag.bz * z)
        frequency = np.squeeze(frequency)
        ghz = frequency / 1e3
        TEC = TEC[..., np.newaxis]
        mag_field = mag_field[..., np.newaxis]
        far_angle = np.radians(1.3549e-5 / ghz**2 * TEC * mag_field)
        far_angle = np.moveaxis(far_angle, -1, 0)
        return far_angle
