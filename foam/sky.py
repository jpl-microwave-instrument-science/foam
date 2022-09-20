import os 
import pickle
import h5py 
import numpy as np
import scipy.interpolate as spi 
import scipy.constants as spc

from foam.utils.config import dir_path, cache_path


class sky(): 
    """ The sky class implements models of all contributions to measured microwave emission above the ionosphere. This includes 
        the sun, moon, continuum galactic emission, and cosmic microwave background. See documentation for a description of these 
        models 

        :param scattered_galaxy: Toggles between specular reflection (False) and ocean surface scattering (True). Default is True 
        :param galaxy_file: Input galaxy map as a function of right ascension and declintion. If None, the FOAM default is used.
                            Currently, only the FOAM default is supported, but this may be changed in the future
        :param verbose: Toggles verbose output

    """ 

    galaxy_file = os.path.join(cache_path, 'galaxy/TBSkyLbandAquarius.h5')
    scattered_galaxy_file = os.path.join(dir_path, 'assets/galaxy/galaxy_interpolator_1deg.p')

    def __init__(self, scattered_galaxy=True, galaxy_file=None, verbose=False): 
        self.scattered_galaxy = scattered_galaxy
        if galaxy_file is not None: 
            self.galaxy_file = galaxy_file 
        elif self.scattered_galaxy: 
            self.galaxy_file = self.scattered_galaxy_file    
        self.verbose = verbose

        self.read_galaxy()

    def read_galaxy(self): 
        """ Reads galaxy map as a function of right acension vs declination and creates an interpolator
            If ocean scattering is specified, the interpolator is loaded from a pre-defined pickle. 

        """

        if self.scattered_galaxy: 
            if self.verbose: print('Using approximate galaxy reflection from wind-roughened ocean')
            self.galaxy_interp = pickle.load(open(self.galaxy_file, 'rb'))[0]
        else: 
            if self.verbose:
                print('Using specular galaxy map')
                print('Galaxy map file: %s' % self.galaxy_file)
            gal_map = h5py.File(self.galaxy_file, 'r')  # Cell size is 9.8572e-6 steradians
            self.galaxy_interp = spi.RegularGridInterpolator((gal_map['Right_Ascension'][:], gal_map['Declination'][:]), 
                                                        gal_map['TB_Cas_A_beam'][:, :] - 2.73, bounds_error=False, fill_value=0)  # Other options here include 'TB_Cas_A_1cell' and 'TB_no_Cas_A'
            gal_map.close()

        self.Tcmb = 2.73

    def galaxy_brightness(self, frequency, ra, dec, wind): 
        """ Retrieves galactic microwave brightness temperature from Dinnat et al. map
            
            :param frequency: Frequency in MHz (Size O)
            :param ra: Right ascension in degrees (Size MxN)
            :param dec: Declination in degrees (Size MxN)
            :param wind: 10-m wind speed in m/s (Size MxN)
            :return: Galactic brightness temperature (Size OxMxN)

        """ 

        frequency = frequency[:, np.newaxis]
        ra = ra[np.newaxis, :]
        dec = dec[np.newaxis, :]
        wind = wind[np.newaxis, :]
        if self.scattered_galaxy: 
            gal_Tb = self.galaxy_interp((wind, ra, dec))  
        else:  
            gal_Tb = self.galaxy_interp((ra, dec))
        gal_Tb = gal_Tb * (frequency / 1420)**-2.7 + self.Tcmb
        return gal_Tb

    @staticmethod
    def sun_brightness(frequency, year=2005): 
        """ Simple solar brightness temperature model using average solar flux
            value and sinusoidal dependence from Ho et al. 2008 JPL report. 

            :param frequency: Frequency in MHz 
            :param year: Observation year """

        wave = spc.c / (frequency * 1e6)
        F = 100 + 50 * np.cos(2 * np.pi / 11 * year)
        F = F * 1e-22  # Average solar flux in W/m2/Hz, 1e-22 is 'solar flux unit'
        ster_sun = 5.981149e-5  # Steradians 

        Tb = wave**2 * F / (2 * spc.k * ster_sun)

        return Tb

    @staticmethod
    def moon_brightness():
        """ Returns 275 K for moon brightness temperature
        """
        return 275  
