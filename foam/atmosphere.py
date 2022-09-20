import os 
import numpy as np
from numpy import exp, cos, pi
import pandas as pd
import scipy.interpolate as spi 
import scipy.integrate as sint 
from cftime import date2num

from .utils.config import cache_path
from .utils import reader as rdr


class atmosphere(): 
    """ The atmosphere class reads atmospheric state variables and implements radiative
        transfer calculations to determine atmospheric brightness temperatures. An empirical model for 
        atmospheric absorption and emission is also included if the user does not require the accuracy
        of line-by-line radiative transfer results. 
        
        Inputs: 
        :param datetime: Either a single date or a pair of dates bracketing an interval of time
                         Several formats are permissible, such as
                            - String or iterable of strings
                            - Python datetime or iterable of datetimes 
                            - Numpy datetime64 or iterable ...
                            - Pandas timestamp or iterable ...
        :param mode: 'simple' mode uses model functions to predict atmospheric emission, while 'full' mode
                     performs the full radiative transfer calculations. There is also a 'simple+tdep' mode
                     which uses a temperature dependence in the model function
        :param online: Toggles online or offline mode
        :param file: Atmosphere file location. Default is None, which reads a stock file from the cache.
        :param reader: Atmosphere reader method, default is MERRAReader
        :param reader_kwargs: Reader keyword arguments
        :param verbose: Toggles code verbosity

    """

    time_reference = 'seconds since 2000-01-01 12:00:0.0'  # J2000

    def __init__(self, datetime='2015-01-01', mode='simple', online=False, 
                 file=None, reader=rdr.MERRAReader, reader_kwargs=None,
                 profile_res=1., verbose=False, **kwargs):
        
        self.datetime = pd.to_datetime(datetime)
        self.mode = mode 
        self.online = online 
        self.verbose = verbose
        self.profile_res = profile_res
        self.profile = np.arange(100, 0, -profile_res)  # Vertical profile in km 

        if file is None and not online: 
            self.datetime = pd.to_datetime(['2005-01-01 12:00'])
            self.file = os.path.join(cache_path, 'atmosphere', 'MERRA2_300.tavg1_2d_slv_Nx.20050101.nc4')
            self.reader = rdr.MERRAReader
            self.reader_kwargs = {}
        else: 
            self.file = file
            self.reader = reader
            if reader_kwargs is None: 
                self.reader_kwargs = {}
            else: 
                self.reader_kwargs = reader_kwargs

        for key, value in kwargs.items(): 
            setattr(self, key, value)
        
        self.read_atmosphere()

    def read_atmosphere(self): 
        """ Reads atmospheric state from online or local files 
            Only accepted format is MERRA-2 Time Averaged Single Level Diagnostics file (e.g. ...tavg1_2d_slv_Nx...)
        """

        if self.mode == 'full': 
            if self.verbose: print('Using full radiative transfer computation to compute atmosphere opacity')
        else: 
            if self.verbose: print('Using single value parameters to compute atmosphere opacity')
            
            # Simpler block estimates for optical depth and effective radiating temperature
            # self.tau_coeff = np.array([0.013664614494878, -1.83402804591467e-5, -1.02330938120674e-5])
            # tau = tau_coeff[0] + tau_coeff[1] * sst + tau_coeff[2] * prwtr
            # self.Teff_coeff = np.array([-7.69054064356577e-10, 4.05537390612154e-7, 2.68693677239213e-6, 
            #                         -0.00770149860447519, -0.0576638279677582, 272.404431351231])

            frequencies = np.array([0.3, 0.5, 0.75, 1, 1.4, 2, 4, 6.8, 8.5, 10.7, 
                                    12.5, 15, 18.7, 20, 22.23, 23.8, 25.7, 30, 
                                    33.5, 37, 38.5, 40]) * 1e3  # MHz

            # Optical depth expression coefficients were determined from 0-70 degrees latitude in the MERRA data   
            # See Brown et al. 2006 for the form of the fit. 

            if self.mode == 'simple':
                
                taus = np.array([[3.08716e-03, 9.61362e-05, -1.83726e-05, 2.20786e-03, -5.32170e-03],   # 300 MHz
                                 [4.64444e-03, 1.48537e-04, -3.20937e-05, 4.53491e-03, -1.07514e-02],   # 500 MHz
                                 [5.70544e-03, 1.82513e-04, -4.22078e-05, 6.62276e-03, -1.55460e-02],   # 750 MHz
                                 [6.25752e-03, 2.00259e-04, -4.74112e-05, 7.93105e-03, -1.84255e-02],   # 1.0 GHz
                                 [6.69655e-03, 2.16103e-04, -5.06998e-05, 9.23504e-03, -2.10914e-02],   # 1.4 GHz
                                 [6.98591e-03, 2.34920e-04, -5.11140e-05, 1.05586e-02, -2.32668e-02],   # 2.0 GHz
                                 [7.36422e-03, 3.28218e-04, -4.06684e-05, 1.49071e-02, -2.75905e-02],   # 4.0 GHz
                                 [7.83083e-03, 5.79937e-04, -1.00309e-05, 2.54726e-02, -3.87006e-02],   # 6.8 GHz
                                 [8.20729e-03, 8.49119e-04,  1.37731e-05, 3.38953e-02, -4.61013e-02],   # 8.5 GHz
                                 [8.80894e-03, 1.39200e-03,  5.09926e-05, 4.79529e-02, -5.98844e-02],   # 10.7 GHz
                                 [9.41826e-03, 2.14349e-03,  8.24646e-05, 6.09606e-02, -7.08333e-02],   # 12.5 GHz
                                 [1.04143e-02, 4.16716e-03,  1.23414e-04, 8.28802e-02, -9.14239e-02],   # 15.0 GHz
                                 [1.24646e-02, 1.51802e-02,  1.62492e-04, 1.16229e-01, -1.10136e-01],   # 18.7 GHz
                                 [1.35865e-02, 2.68049e-02,  2.63497e-04, 1.29702e-01, -1.14958e-01],   # 20.0 GHz
                                 [2.01097e-02, 5.69676e-02,  1.58792e-03, 9.94330e-02,  5.69901e-02],   # 22.2 GHz
                                 [1.78031e-02, 4.82584e-02,  5.44257e-04, 1.41326e-01, -3.08155e-02],   # 23.8 GHz
                                 [1.85903e-02, 2.97715e-02,  2.39187e-04, 1.82374e-01, -1.05924e-01],   # 25.7 GHz
                                 [2.42664e-02, 1.63621e-02,  3.65329e-04, 2.50232e-01, -1.81572e-01],   # 30.0 GHz
                                 [3.25417e-02, 1.53154e-02,  3.99717e-04, 2.52397e-01, -6.38977e-02],   # 33.5 GHz
                                 [4.46121e-02, 1.66229e-02,  3.89242e-04, 2.52419e-01,  7.32301e-02],   # 37.0 GHz
                                 [5.15383e-02, 1.78029e-02,  3.26695e-04, 2.50174e-01,  1.49059e-01],   # 38.5 GHz
                                 [6.02604e-02, 1.90441e-02,  2.88297e-04, 2.50453e-01,  1.97543e-01]])  # 40.0 GHz
                
                # Teff is independent of look angle from 0-40 degrees at low frequencies
                # For higher frequency use, it will be necessary to implement some type of look angle correction
                #  Latitude       0-10         10-20        20-30        30-40        40-50        50-60        60-70        70-80        80-90 
                teff = np.array([[2.59541e+02, 2.58138e+02, 2.55928e+02, 2.52102e+02, 2.51574e+02, 2.49970e+02, 2.49377e+02, 2.36872e+02, 2.29746e+02],  # 300 MHz   
                                [2.64251e+02, 2.62197e+02, 2.59672e+02, 2.56015e+02, 2.56265e+02, 2.55295e+02, 2.55521e+02, 2.40959e+02, 2.32195e+02],   # 500 MHz 
                                [2.66529e+02, 2.64078e+02, 2.61383e+02, 2.57969e+02, 2.58753e+02, 2.58233e+02, 2.59076e+02, 2.43331e+02, 2.33539e+02],   # 750 MHz 
                                [2.67550e+02, 2.64887e+02, 2.62092e+02, 2.58834e+02, 2.59881e+02, 2.59592e+02, 2.60782e+02, 2.44507e+02, 2.34224e+02],   # 1.0 GHz 
                                [2.68372e+02, 2.65541e+02, 2.62634e+02, 2.59502e+02, 2.60728e+02, 2.60614e+02, 2.62062e+02, 2.45420e+02, 2.34782e+02],   # 1.4 GHz 
                                [2.69109e+02, 2.66164e+02, 2.63098e+02, 2.60006e+02, 2.61276e+02, 2.61276e+02, 2.62871e+02, 2.46059e+02, 2.35235e+02],   # 2.0 GHz 
                                [2.71265e+02, 2.68203e+02, 2.64475e+02, 2.61163e+02, 2.62125e+02, 2.62384e+02, 2.63991e+02, 2.47199e+02, 2.36336e+02],   # 4.0 GHz 
                                [2.74376e+02, 2.71651e+02, 2.67045e+02, 2.63125e+02, 2.63229e+02, 2.63737e+02, 2.65166e+02, 2.49096e+02, 2.38635e+02],   # 6.8 GHz 
                                [2.76090e+02, 2.73680e+02, 2.68699e+02, 2.64353e+02, 2.63941e+02, 2.64720e+02, 2.66159e+02, 2.50593e+02, 2.40428e+02],   # 8.5 GHz 
                                [2.77977e+02, 2.76124e+02, 2.70988e+02, 2.66103e+02, 2.64931e+02, 2.65852e+02, 2.67286e+02, 2.52687e+02, 2.43083e+02],   # 10.7 GHz 
                                [2.79423e+02, 2.78019e+02, 2.72948e+02, 2.67644e+02, 2.65810e+02, 2.66790e+02, 2.68272e+02, 2.54516e+02, 2.45406e+02],   # 12.5 GHz 
                                [2.81432e+02, 2.80720e+02, 2.76109e+02, 2.70323e+02, 2.67286e+02, 2.67809e+02, 2.69149e+02, 2.57172e+02, 2.49162e+02],   # 15.0 GHz 
                                [2.84357e+02, 2.84398e+02, 2.81319e+02, 2.75865e+02, 2.71365e+02, 2.70220e+02, 2.70567e+02, 2.62064e+02, 2.57067e+02],   # 18.7 GHz 
                                [2.84467e+02, 2.84330e+02, 2.81974e+02, 2.77584e+02, 2.73625e+02, 2.71566e+02, 2.71129e+02, 2.64113e+02, 2.62043e+02],   # 20.0 GHz 
                                [2.80807e+02, 2.79427e+02, 2.77945e+02, 2.77273e+02, 2.76623e+02, 2.74588e+02, 2.73416e+02, 2.65235e+02, 2.69432e+02],   # 22.2 GHz 
                                [2.83704e+02, 2.83183e+02, 2.81180e+02, 2.77857e+02, 2.75053e+02, 2.73545e+02, 2.73405e+02, 2.66426e+02, 2.66018e+02],   # 23.8 GHz 
                                [2.84577e+02, 2.84773e+02, 2.81909e+02, 2.76405e+02, 2.71861e+02, 2.71181e+02, 2.71879e+02, 2.63807e+02, 2.58553e+02],   # 25.7 GHz 
                                [2.82672e+02, 2.82450e+02, 2.78211e+02, 2.72236e+02, 2.69114e+02, 2.70616e+02, 2.71908e+02, 2.60867e+02, 2.51830e+02],   # 30.0 GHz 
                                [2.81541e+02, 2.80467e+02, 2.75570e+02, 2.70268e+02, 2.69154e+02, 2.73010e+02, 2.74746e+02, 2.59656e+02, 2.47305e+02],   # 33.5 GHz 
                                [2.80606e+02, 2.78859e+02, 2.73648e+02, 2.68957e+02, 2.69156e+02, 2.74307e+02, 2.75887e+02, 2.57653e+02, 2.43249e+02],   # 37.0 GHz 
                                [2.80272e+02, 2.78165e+02, 2.72715e+02, 2.68120e+02, 2.68644e+02, 2.74134e+02, 2.75732e+02, 2.56582e+02, 2.41729e+02],   # 38.5 GHz 
                                [2.79749e+02, 2.77304e+02, 2.71849e+02, 2.67749e+02, 2.68989e+02, 2.74698e+02, 2.75652e+02, 2.55334e+02, 2.39981e+02]])  # 40.0 GHz   

            elif self.mode == 'simple+tdep':
                                 
                taus = np.array([[1.04144e-03, -1.38820e-04, 9.76535e-06, 1.87248e-03, -4.19694e-03,  8.32999e-06],     # 300 MHz
                                [7.73812e-04, -2.96162e-04, 2.11658e-05, 3.90201e-03, -8.62718e-03,  1.57612e-05],     # 500 MHz
                                [3.11471e-04, -4.36474e-04, 3.19005e-05, 5.73298e-03, -1.25565e-02,  2.19623e-05],     # 750 MHz
                                [9.54567e-06, -5.19037e-04, 3.87270e-05, 6.87180e-03, -1.48774e-02,  2.55208e-05],     # 1.0 GHz
                                [3.00685e-04, -5.87374e-04, 4.55175e-05, 8.08594e-03, -1.72382e-02,  2.84918e-05],     # 1.4 GHz
                                [5.10641e-04, -6.25900e-04, 5.19705e-05, 9.31671e-03, -1.90912e-02,  3.05257e-05],     # 2.0 GHz
                                [7.86499e-04, -6.09945e-04, 7.16326e-05, 1.38987e-02, -2.42440e-02,  3.31628e-05],     # 4.0 GHz
                                [1.16797e-03, -4.51739e-04, 1.13404e-04, 2.39184e-02, -3.33583e-02,  3.66409e-05],     # 6.8 GHz
                                [1.45177e-03, -2.63469e-04, 1.47009e-04, 3.25249e-02, -4.14330e-02,  3.93167e-05],     # 8.5 GHz
                                [1.95776e-03,  1.57441e-04, 1.98915e-04, 4.57456e-02, -5.24201e-02,  4.38788e-05],     # 10.70GHz
                                [2.47407e-03,  7.78107e-04, 2.46180e-04, 5.85234e-02, -6.26539e-02,  4.84726e-05],     # 12.50GHz
                                [3.27987e-03,  2.59855e-03, 3.11858e-04, 7.91901e-02, -7.91542e-02,  5.58980e-05],     # 15.00GHz
                                [4.98452e-03,  1.31883e-02, 4.00461e-04, 1.11165e-01, -9.15050e-02,  7.12024e-05],     # 18.70GHz
                                [4.34234e-03,  2.47736e-02, 5.08899e-04, 1.20023e-01, -8.31397e-02,  7.36513e-05],     # 20.00GHz
                                [3.11951e-02,  5.81397e-02, 1.45068e-03, 9.52553e-02,  6.92853e-02, -4.37897e-05],     # 22.23GHz
                                [3.12968e-02,  4.92905e-02, 4.32741e-04, 1.39476e-01, -2.42806e-02, -5.21770e-05],     # 23.80GHz
                                [5.99313e-03,  2.67974e-02, 5.94968e-04, 1.92586e-01, -1.42187e-01,  9.92427e-05],     # 25.70GHz
                                [9.55286e-03,  1.23278e-02, 8.49325e-04, 2.47226e-01, -1.64508e-01,  1.37746e-04],     # 30.00GHz
                                [1.15997e-02,  9.68326e-03, 1.05459e-03, 3.02390e-01, -2.04474e-01,  1.74887e-04],     # 33.50GHz
                                [9.46036e-03,  9.46615e-03, 1.20128e-03, 3.54468e-01, -2.19156e-01,  2.10553e-04],     # 37.00GHz
                                [1.50768e-02,  9.09376e-03, 1.33762e-03, 3.50426e-01, -1.46422e-01,  2.62003e-04],     # 38.50GHz
                                [8.62328e-03,  9.62828e-03, 1.35224e-03, 4.00378e-01, -2.21990e-01,  2.66470e-04]])    # 40.00GHz

                #  Latitude       0-10         10-20        20-30        30-40        40-50        50-60        60-70        70-80        80-90 
                teff = np.array([[2.59392e+02, 2.56527e+02, 2.52972e+02, 2.49924e+02, 2.50217e+02,  2.48361e+02, 2.48358e+02, 2.40346e+02, 2.36816e+02],   # 300 MHz
                                [2.64079e+02, 2.60089e+02, 2.55848e+02, 2.53196e+02, 2.54532e+02,  2.53264e+02, 2.54215e+02, 2.45484e+02, 2.41387e+02],   # 500 MHz
                                [2.66341e+02, 2.61624e+02, 2.56968e+02, 2.54716e+02, 2.56774e+02,  2.55941e+02, 2.57597e+02, 2.48591e+02, 2.44206e+02],   # 750 MHz
                                [2.67348e+02, 2.62259e+02, 2.57384e+02, 2.55370e+02, 2.57794e+02,  2.57190e+02, 2.59220e+02, 2.50127e+02, 2.45629e+02],   # 1.0 GHz
                                [2.68149e+02, 2.62808e+02, 2.57743e+02, 2.55905e+02, 2.58563e+02,  2.58111e+02, 2.60418e+02, 2.51316e+02, 2.46773e+02],   # 1.4 GHz
                                [2.68854e+02, 2.63408e+02, 2.58157e+02, 2.56379e+02, 2.59105e+02,  2.58754e+02, 2.61199e+02, 2.52106e+02, 2.47572e+02],   # 2.0 GHz
                                [2.70853e+02, 2.65746e+02, 2.59989e+02, 2.57908e+02, 2.60171e+02,  2.59921e+02, 2.62230e+02, 2.53236e+02, 2.48926e+02],   # 4.0 GHz
                                [2.73871e+02, 2.69763e+02, 2.63467e+02, 2.60595e+02, 2.61810e+02,  2.61804e+02, 2.63696e+02, 2.54825e+02, 2.51005e+02],   # 6.8 GHz
                                [2.75547e+02, 2.72149e+02, 2.65767e+02, 2.62354e+02, 2.62861e+02,  2.63006e+02, 2.64702e+02, 2.56011e+02, 2.52564e+02],   # 8.5 GHz
                                [2.77495e+02, 2.74941e+02, 2.68686e+02, 2.64618e+02, 2.64259e+02,  2.64633e+02, 2.66181e+02, 2.57733e+02, 2.54744e+02],   # 10.70GHz
                                [2.78994e+02, 2.77089e+02, 2.71131e+02, 2.66553e+02, 2.65409e+02,  2.65829e+02, 2.67301e+02, 2.59209e+02, 2.56614e+02],   # 12.50GHz
                                [2.81132e+02, 2.80094e+02, 2.74864e+02, 2.69679e+02, 2.67230e+02,  2.67312e+02, 2.68565e+02, 2.61279e+02, 2.59263e+02],   # 15.00GHz
                                [2.84344e+02, 2.84249e+02, 2.80867e+02, 2.75693e+02, 2.71466e+02,  2.69985e+02, 2.70268e+02, 2.64782e+02, 2.64135e+02],   # 18.70GHz
                                [2.84585e+02, 2.84305e+02, 2.81733e+02, 2.77582e+02, 2.73950e+02,  2.71864e+02, 2.71418e+02, 2.66161e+02, 2.66924e+02],   # 20.00GHz
                                [2.81330e+02, 2.79829e+02, 2.78265e+02, 2.77567e+02, 2.76935e+02,  2.75021e+02, 2.73885e+02, 2.64604e+02, 2.67706e+02],   # 22.23GHz
                                [2.84094e+02, 2.83577e+02, 2.81614e+02, 2.78171e+02, 2.75160e+02,  2.73585e+02, 2.73224e+02, 2.64525e+02, 2.62446e+02],   # 23.80GHz
                                [2.84663e+02, 2.84816e+02, 2.81797e+02, 2.76578e+02, 2.72183e+02,  2.70730e+02, 2.70754e+02, 2.65504e+02, 2.64154e+02],   # 25.70GHz
                                [2.82614e+02, 2.82286e+02, 2.77632e+02, 2.71923e+02, 2.68897e+02,  2.69733e+02, 2.70915e+02, 2.63948e+02, 2.60466e+02],   # 30.00GHz
                                [2.81353e+02, 2.80611e+02, 2.75381e+02, 2.69969e+02, 2.67926e+02,  2.69222e+02, 2.70258e+02, 2.62736e+02, 2.58535e+02],   # 33.50GHz
                                [2.80491e+02, 2.79392e+02, 2.73870e+02, 2.68707e+02, 2.67220e+02,  2.68658e+02, 2.69486e+02, 2.60986e+02, 2.55601e+02],   # 37.00GHz
                                [2.80129e+02, 2.78549e+02, 2.72741e+02, 2.68039e+02, 2.67359e+02,  2.69434e+02, 2.70240e+02, 2.60856e+02, 2.54974e+02],   # 38.50GHz
                                [2.79742e+02, 2.78266e+02, 2.72525e+02, 2.67608e+02, 2.66502e+02,  2.67844e+02, 2.68471e+02, 2.59297e+02, 2.53289e+02]])   # 40.00GHz

            self.tau_coeff = spi.interp1d(frequencies, taus.T, kind='cubic', bounds_error=False, fill_value='extrapolate')
            lats = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85])
            self.teff_coeff = spi.RegularGridInterpolator((frequencies, lats), teff, bounds_error=False, fill_value=None)
        
        atm_rdr = self.reader(self.datetime, online=self.online, file=self.file, 
                              **self.reader_kwargs)

        airtemp_interp, airpres_interp, uwind_interp, vwind_interp, prwtr_interp, lwtr_interp = atm_rdr.read()

        self.uwind_interp = uwind_interp
        self.vwind_interp = vwind_interp
        self.prwtr_interp = prwtr_interp
        self.lwtr_interp = lwtr_interp
        self.airtemp_interp = airtemp_interp
        self.airpres_interp = airpres_interp

        if len(self.airtemp_interp.grid) == 4: 
            self.interpolator_has_vertical = True
        else: 
            self.interpolator_has_vertical = False

    @staticmethod
    def isa_profiles(T0, P0, pr_wv, pr_lw, res=1):
        """ Generates nominal vertical profiles using the International Standard Atmosphere given
            column integrated abundances. The returned profiles are specified in order from top of
            atmosphere to surface.

            :param T0: Air surface temperature in Kelvin
            :param P0: Air surface pressure in Pascals
            :param pr_wtr: Total column precipitable water vapor in kg/m2 (equal to mm)
            :param pr_lw: Total column precipitable water vapor in kg/m2 (equal to mm)
            :param res: Altitude grid resolution (dz) in km

            :returns:
                - T - Temperature profile in K 
                - P - Pressure profile in pascals
                - dens - Density profile in kg/m3
                - wv - Precipitable water vapor profile in kg/kg 
                - lw - Precipitable liquid water profile in kg/kg
                - z - Altitude grid in km 
        """ 

        inflections = np.array([0, 11, 20, 32, 47, 51, 71, 86, 100])
        lapse_rates = np.array([-6.5, 0, 1, 2.8, 0, -2.8, -2, 0])
        z = np.arange(0, 101, 1)   
        for i in range(T0.ndim): z = z[:, np.newaxis]  
        z = z * np.ones((len(z), *np.shape(T0)))
        T = np.ones(np.shape(z))
        T_base = T0[np.newaxis] * T

        for i in range(0, len(lapse_rates)):
            z_mask = (z >= inflections[i])
            T[z_mask] = T_base[z_mask] + lapse_rates[i] * (z[z_mask] - inflections[i])  
            T_base[inflections[i + 1]:, :] = T[inflections[i + 1], :]
        interp_alt = np.arange(0, 101, 1)
        temp_func = spi.interp1d(interp_alt, T, kind='cubic', axis=0)

        P = P0[np.newaxis] * exp(-z / 7)  # original scale height was 7.7 km, but 7 km agrees better with mean MERRA data
        pres_func = spi.interp1d(interp_alt, P, kind='cubic', axis=0)  # pascal 

        dens = P * 1e-2 / T / 2.87
        dens_func = spi.interp1d(interp_alt, dens, kind='cubic', axis=0)  # kg/m3

        H = 2  # Per Ulaby, water vapor scale height is between 2 and 2.5 km 
        wv_0 = pr_wv / H
        wv = wv_0[np.newaxis] * np.exp(-z / H) / 1e3 * 0.98  # Gravity fudge factor 
        wv_func = spi.interp1d(interp_alt, wv / dens, kind='cubic', axis=0)  # kg/kg

        # lw = pr_lw[np.newaxis] / 10 * np.ones(np.shape(z)) / 1e3
        # lw[z > 10] = 0 
        cloud_alt = np.arange(0, 11, 1)
        # This is an empirical cloud model taken by averaging the cloud liquid water profile across all latitudes and longitudes on Earth during a single day 
        # from a MERRA 2 3D dataset. 
        cloud_lw = np.array([0, 0.30819048, 0.35664123, 0.08617703, 0.05782645, 0.05412928, 0.03992631, 0.03518051, 0.03552852, 0.02023061, 0.00616958])
        for i in range(pr_lw.ndim): cloud_lw = cloud_lw[:, np.newaxis]
        lw = pr_lw[np.newaxis] / 1e3 * cloud_lw
 
        lw_func = spi.interp1d(cloud_alt, lw / dens[0:11, :], kind='cubic', axis=0, bounds_error=False, fill_value=0)  # kg/kg
        z = np.arange(100, 0 - res, -res)
        z[z < 1e-3] = 0

        T = temp_func(z)
        P = pres_func(z)
        dens = dens_func(z)
        wv = wv_func(z)
        lw = lw_func(z)

        for i in range(T0.ndim): z = z[:, np.newaxis]  
        z = z * np.ones((len(z), *np.shape(T0)))

        return T, P, dens, wv, lw, z

    def get_atmosphere_tb(self, frequency, time, lat, lon, angle, in_epoch=False, use_time=True): 
        """ Get brightness temperatures and atmospheric opacity from either the empirical model or the line-by-line model 
            
            Inputs: 
            :param frequency: Frequency in MHz 
            :param time: String or array of times, converted to interpolator
                         epoch reference if in_epoch=False
            :param lat: Latitude in degrees 
            :param lon: Longitude in degrees
            :param angle: Emission angle in degrees 
            :param in_epoch: See above

            :returns:
                - tbup - Upwelling brightness temperature 
                - tbdn - Downwelling brightness temperature 
                - transup - Upwelling optical transmisivity
                - transdn - Downwelling optical transmissivity 
                - wup - Upwelling weighting function 
                - wdn - Downwelling weighting function 
            
        """
        # For PWV and PLW, 1kg/m^2 = 1 mm
        if not use_time: 
            try: 
                time = self.datetime[0]
            except TypeError: 
                time = self.datetime
            in_epoch = False
        if not in_epoch: 
            time = date2num(pd.to_datetime(time).to_pydatetime(), self.time_reference)
        
        if self.interpolator_has_vertical: 
            z = self.profile
            try: 
                for i in range(lat.ndim): z = z[:, np.newaxis] 
                z = z * np.ones((len(z), *np.shape(lat)))
            except AttributeError: 
                # Lat and lon are probably floats, profile doesn't need to be expanded
                pass
            interp_tuple = (time, z, lat, lon)
        else: 
            interp_tuple = (time, lat, lon)

        airtemp = self.airtemp_interp(interp_tuple)  # Kelvin
        airtemp[airtemp < 0] = 0
        airpres = self.airpres_interp(interp_tuple)  # Pascal
        airpres[airpres < 0] = 0
        prwtr = self.prwtr_interp(interp_tuple)  # millimeters
        prwtr[prwtr < 0] = 0

        if self.lwtr_interp is not None: 
            lwtr = self.lwtr_interp(interp_tuple)  # millimeters
            lwtr[lwtr < 0] = 0
        else: 
            lwtr = np.zeros(np.shape(prwtr))

        airtemp = airtemp[..., np.newaxis]
        airpres = airpres[..., np.newaxis]
        prwtr = prwtr[..., np.newaxis]
        lwtr = lwtr[..., np.newaxis]
        lat = lat[..., np.newaxis]
        angle = angle[..., np.newaxis]

        # frequency = frequency[np.newaxis, :]
        tbup, tbdn, prop_dict = self.get_atmosphere_prop(frequency, prwtr, lwtr, airtemp, airpres, lat, angle)

        return tbup, tbdn, prop_dict

    def get_atmosphere_prop(self, frequency, prwtr, lwtr, airtemp, airpres, lat, angle):
        """ Add docstring per above """ 
        
        prop_dict = {}
        if self.mode == 'simple':
            prwtr = prwtr * 0.1  # Convert to cm units
            taus = self.tau_coeff(frequency)
            tau = taus[0, :] + taus[1, :] * prwtr + taus[2, :] * prwtr**2 + taus[3, :] * lwtr + taus[4, :] * lwtr**2
            Teff = self.teff_coeff((frequency, abs(lat)))
            transup = exp(-tau / cos(np.radians(angle)))
            transdn = transup
            tbup = (1 - transup) * Teff
            tbdn = (1 - transdn) * transup * Teff 
        elif self.mode == 'simple+tdep': 
            prwtr = prwtr * 0.1  # Convert to cm units
            taus = self.tau_coeff(frequency)
            tau = taus[0, :] + taus[1, :] * prwtr + taus[2, :] * prwtr**2 + taus[3, :] * lwtr + taus[4, :] * lwtr**2 + taus[5, :] * airtemp
            Teff = self.teff_coeff((frequency, abs(lat)))
            transup = exp(-tau / cos(np.radians(angle)))
            transdn = transup
            tbup = (1 - transup) * Teff
            tbdn = (1 - transdn) * transup * Teff 
        else: 
            if not self.interpolator_has_vertical: 
                T, P, dens, wv, lw, alts = self.isa_profiles(airtemp, airpres, prwtr, lwtr, res=self.profile_res)
            else: 
                z = self.profile
                # Quantities have a vertical direction, so index this out 
                for i in range(airtemp[0].ndim): z = z[:, np.newaxis] 
                z = z * np.ones((len(z), *np.shape(airtemp[0])))
                T, P, wv, lw, alts = airtemp, airpres, prwtr, lwtr, z
            tbup, tbdn, wup, wdn, transup, transdn, tau, Teff = self.rad_trans(frequency, P, T, wv, lw, alts, angle)

            prop_dict['vertical_grid_km'] = (alts[:-1] + alts[1:]) / 2
            prop_dict['upward_weighting_function'] = np.moveaxis(wup, -1, 0)
            prop_dict['downward_weighting_function'] = np.moveaxis(wdn, -1, 0)

        prop_dict['upward_transmissivity'] = np.moveaxis(transup, -1, 0)
        prop_dict['downward_transmissivity'] = np.moveaxis(transdn, -1, 0) 
        prop_dict['optical_depth'] = np.moveaxis(tau, -1, 0) 
        prop_dict['effective_temperature'] = np.moveaxis(Teff, -1, 0)
        tbup = np.moveaxis(tbup, -1, 0)
        tbdn = np.moveaxis(tbdn, -1, 0)

        return tbup, tbdn, prop_dict

    def rad_trans(self, freq, p, t, q, lwc, z, angle):
        """ Microwave radiative transfer calculations for input atmosphere compositions. 
            Input convention is: 0 index - top of atmosphere
            The height index should be the first for all variables 

            :param freq: Frequency in MHz
            :param p: Pressure profile in Pa
            :param t: Temperature profile in K
            :param q: Water vapor profile in kg/kg
            :param lwc: Liquid water content in kg/kg
            :param z: Altitude grid in km 
            :param angle: Look angle in degrees
            
            return: Atmospheric opacity in 1/km

            This function and the others that it calls were originally written by P. Rosenkranz and G. Petty, 
            and converted to Python by T. Islam. 
        """

        # Layer of the atmosphere bounded by two successive levels
        
        p_l = 0.5 * (p[:-1] + p[1:])  # Pa
        t_l = 0.5 * (t[:-1] + t[1:])  # K
        q_l = 0.5 * (q[:-1] + q[1:])  # kg/kg
        lwc_l = 0.5 * (lwc[:-1] + lwc[1:])  # kg/kg
        z_l = 0.5 * (z[:-1] + z[1:])  # km
        
        tv = (1. + 0.61 * q_l) * t_l  # Virtual temperature 
        rhoair = p_l / (tv * 287.06)  # Moist air density, kg/m3
        rhov_l = q_l * rhoair  # Vapor density, kg/m3
        rhol_l = lwc_l * rhoair  # Liquid density, kg/m3

        # And the inverse, density to humidity 
        # e = rhowv * (t_l * 461.5)  # convert vapor density to vapor pressure
        # q = 0.622 * e / p_l  # calculate specific humidity

        pmb = p_l / 100.0  # convert pressure from Pa to Mb
        vapden = rhov_l * 1000.0  # convert vapor density from kg/m**3 to g/m**3

        # Compute absorption in each layer
        absairn2 = self.n2abs(t_l, pmb, freq)
        absairo2 = self.o2abs(t_l, pmb, vapden, freq)
        absair = absairn2 + absairo2
        abswv = self.h2oabs(t_l, pmb, vapden, freq)
        abscloud = self.cloudabs(freq, rhol_l, t_l)
        absair[(p_l < 0.1) | (absair <= 0.0)] = 0.0  # set zero below 0.1 hPa
        abswv[(p_l < 0.1) | (abswv <= 0.0)] = 0.0
        abscloud[(p_l < 0.1) | (abscloud <= 0.0)] = 0.0
        abs_total = absair + abswv + abscloud  # 1/km 

        # Optical depth, transmissivity, weighting functions, and brightness temperature
        angle = np.radians(angle)
        tau_up = -sint.cumtrapz(abs_total / np.cos(angle), z_l, axis=0, initial=0)
        tau_dn = sint.cumtrapz(abs_total[::-1] / np.cos(angle), z_l[::-1], axis=0, initial=0)[::-1]
        tau_int = -sint.trapz(abs_total, z_l, axis=0)  # No angle adjustment for the output
        trans_int = np.exp(-1 * tau_int / np.cos(angle))
        transup = np.exp(-1 * tau_up)
        transdn = np.exp(-1 * tau_dn)

        # Gradients with first order edges 
        # wup = np.gradient(transup, z_l, axis=0)
        # wdn = -np.gradient(transdn, z_l, axis=0)
        wup = np.zeros(np.shape(z * freq))
        wup[1:-1] = (transup[1:] - transup[:-1]) / (z_l[1:] - z_l[:-1])
        wup[0] = wup[1] + (wup[1] - wup[2]) / (z[1] - z[2]) * (z[0] - z[1])
        wup[-1] = wup[-2] + (wup[-2] - wup[-3]) / (z[-2] - z[-3]) * (z[-1] - z[-2])
        # And regrid
        wup = 0.5 * (wup[1:] + wup[:-1])

        wdn = np.zeros(np.shape(z * freq))
        wdn[1:-1] = -(transdn[1:] - transdn[:-1]) / (z_l[1:] - z_l[:-1])
        wdn[0] = wdn[1] + (wdn[1] - wdn[2]) / (z[1] - z[2]) * (z[0] - z[1])
        wdn[-1] = wdn[-2] + (wdn[-2] - wdn[-3]) / (z[-2] - z[-3]) * (z[-1] - z[-2])
        # And regrid
        wdn = 0.5 * (wdn[1:] + wdn[:-1])

        tbup = -sint.trapz(wup * t_l, z_l, axis=0)
        tbdn = -sint.trapz(wdn * t_l, z_l, axis=0)
        Teff = tbup / (1 - trans_int)

        return tbup, tbdn, wup, wdn, transup, transdn, tau_int, Teff

    @staticmethod
    def n2abs(T, P, F):
        """ Computes absorption due to nitrogen in air

            :param T: Temperature in Kelvin
            :param P: Pressure in millibars 
            :param F: Frequency in MHz 

            :return: Nitrogen absorption in 1/km
        """
        F = F / 1e3  # Convert to GHz
        TH = 300. / T
        ABSN2 = 6.4e-14 * P * P * F * F * TH**3.55
        return ABSN2

    @staticmethod
    def o2abs(TEMP, PRES, VAPDEN, FREQ):
        """ Computes absorption due to oxygen in air
            
            :param TEMP: Temperature in Kelvin 
            :param PRES: Pressure in millibars. Valid from 3 to 1000 mbar
            :param VAPDEN: Water vapor density in g/m^3 
            :param FREQ: Frequency in MHz. Valid from 0 to 900 GHz

            :return: Oxygen absorption in 1/km

            Originally written by P. Rosenkranz. His comments are below 

            REFERENCE FOR EQUATIONS AND COEFFICIENTS:
            P.W. ROSENKRANZ, CHAP. 2 AND APPENDIX, IN ATMOSPHERIC REMOTE SENSING
            BY MICROWAVE RADIOMETRY (M.A. JANSSEN, ED. 1993)
            AND H.J. LIEBE ET AL, JQSRT V.48, PP.629-643 (1992)
            (EXCEPT: SUBMILLIMETER LINE INTENSITIES FROM HITRAN92)
            LINES ARE ARRANGED 1-,1+,3-,3+,ETC. IN SPIN-ROTATION SPECTRUM
        """
        FREQ = FREQ / 1e3  # Convert to GHz

        F = np.array([118.7503, 56.2648, 62.4863, 58.4466, 60.3061, 59.5910, 
                     59.1642, 60.4348, 58.3239, 61.1506, 57.6125, 61.8002, 
                     56.9682, 62.4112, 56.3634, 62.9980, 55.7838, 63.5685, 
                     55.2214, 64.1278, 54.6712, 64.6789, 54.1300, 65.2241, 
                     53.5957, 65.7648, 53.0669, 66.3021, 52.5424, 66.8368, 
                     52.0214, 67.3696, 51.5034, 67.9009, 368.4984, 424.7631, 
                     487.2494, 715.3932, 773.8397, 834.1453])
    
        S300 = np.array([.2936E-14, .8079E-15, .2480E-14, .2228E-14,  
                     .3351E-14, .3292E-14, .3721E-14, .3891E-14,  
                     .3640E-14, .4005E-14, .3227E-14, .3715E-14,  
                     .2627E-14, .3156E-14, .1982E-14, .2477E-14,  
                     .1391E-14, .1808E-14, .9124E-15, .1230E-14,  
                     .5603E-15, .7842E-15, .3228E-15, .4689E-15,  
                     .1748E-15, .2632E-15, .8898E-16, .1389E-15,  
                     .4264E-16, .6899E-16, .1924E-16, .3229E-16,  
                     .8191E-17, .1423E-16, .6460E-15, .7047E-14, .3011E-14, 
                     .1826E-14, .1152E-13, .3971E-14])
           
        BE = np.array([.009, .015, .083, .084, .212, .212, .391, .391, .626, .626,
                   .915, .915, 1.260, 1.260, 1.660, 1.665, 2.119, 2.115, 2.624, 2.625, 
                   3.194, 3.194, 3.814, 3.814, 4.484, 4.484, 5.224, 5.224, 6.004, 6.004, 
                   6.844, 6.844, 7.744, 7.744, .048, .044, .049, .145, .141, .145])
        #      WIDTHS IN MHZ/MB
        WB300 = .56

        X = .8

        W300 = np.array([1.63, 1.646, 1.468, 1.449, 1.382, 1.360, 
                    1.319, 1.297, 1.266, 1.248, 1.221, 1.207, 1.181, 1.171, 
                    1.144, 1.139, 1.110, 1.108, 1.079, 1.078, 1.05, 1.05, 
                    1.02, 1.02, 1.00, 1.00, .97, .97, .94, .94, .92, .92, .89, .89, 
                    1.92, 1.92, 1.92, 1.81, 1.81, 1.81])
    
        Y300 = np.array([-0.0233, 0.2408, -0.3486, 0.5227, 
                     -0.5430, 0.5877, -0.3970, 0.3237, -0.1348, 0.0311, 
                     0.0725, -0.1663, 0.2832, -0.3629, 0.3970, -0.4599, 
                     0.4695, -0.5199, 0.5187, -0.5597, 0.5903, -0.6246, 
                     0.6656, -0.6942, 0.7086, -0.7325, 0.7348, -0.7546, 
                     0.7702, -0.7864, 0.8083, -0.8210, 0.8439, -0.8529, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        V = np.array([0.0079, -0.0978, 0.0844, -0.1273, 
                 0.0699, -0.0776, 0.2309, -0.2825, 0.0436, -0.0584, 
                 0.6056, -0.6619, 0.6451, -0.6759, 0.6547, -0.6675, 
                 0.6135, -0.6139, 0.2952, -0.2895, 0.2654, -0.2590, 
                 0.3750, -0.3680, 0.5085, -0.5002, 0.6206, -0.6091, 
                 0.6526, -0.6393, 0.6640, -0.6475, 0.6729, -0.6545, 
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
        TH = 300.0 / TEMP
        TH1 = TH - 1.0
        B = TH**X
        PRESWV = VAPDEN * TEMP / 217.0
        PRESDA = PRES - PRESWV
        DEN = .001 * (PRESDA * B + 1.1 * PRESWV * TH)
        DFNR = WB300 * DEN
        SUM = 1.6e-17 * FREQ * FREQ * DFNR / (TH * (FREQ * FREQ + DFNR * DFNR))

        for K in range(40):
            DF = W300[K] * DEN
            Y = .001 * PRES * B * (Y300[K] + V[K] * TH1)
            STR = S300[K] * np.exp(-BE[K] * TH1)
            SF1 = (DF + (FREQ - F[K]) * Y) / ((FREQ - F[K])**2 + DF * DF)
            SF2 = (DF - (FREQ + F[K]) * Y) / ((FREQ + F[K])**2 + DF * DF)
            SUM = SUM + STR * (SF1 + SF2) * (FREQ / F[K])**2

        O2ABS = .5034E12 * SUM * PRESDA * TH**3 / 3.14159
        return O2ABS

    @staticmethod
    def h2oabs(T, P, RHO, F):
        """ Computes absorption in atmosphere due to water vapor

            :param T: Temperature in Kelvin
            :param P: Pressure in millibar - Valid from 0.1 to 1000 mbar
            :param RHO: Water vapor density in g/m^3
            :param F: Frequency in MHz - Valid from 0 to 800 GHz

            :return: Absorption in 1/km
            
            Originally written by P. Rosenkranz. His comments are below 

            REFERENCES-
            LINE INTENSITIES FROM HITRAN92 (SELECTION THRESHOLD=
            HALF OF CONTINUUM ABSORPTION AT 1000 MB).
            WIDTHS MEASURED AT 22,183,380 GHZ, OTHERS CALCULATED:
            H.J.LIEBE AND T.A.DILLON, J.CHEM.PHYS. V.50, PP.727-732 (1969) &
            H.J.LIEBE ET AL., JQSRT V.9, PP. 31-47 (1969)  (22GHz)
            A.BAUER ET AL., JQSRT V.37, PP.531-539 (1987) & 
            ASA WORKSHOP (SEPT. 1989) (380GHz)
            AND A.BAUER ET AL., JQSRT V.41, PP.49-54 (1989) (OTHER LINES).
            AIR-BROADENED CONTINUUM BASED ON LIEBE & LAYTON, NTIA 
            REPORT 87-224 (1987) SELF-BROADENED CONTINUUM BASED ON 
            LIEBE ET AL, AGARD CONF. PROC. 542 (MAY 1993), 
            BUT READJUSTED FOR LINE SHAPE OF
            CLOUGH et al, ATMOS. RESEARCH V.23, PP.229-241 (1989).

        """
        F = F / 1e3  # Convert to GHz
        NLINES = 15

        #     LINE FREQUENCIES:

        FL = np.array([22.2351, 183.3101, 321.2256, 325.1529, 380.1974, 439.1508,
                  443.0183, 448.0011, 470.8890, 474.6891, 488.4911, 556.9360,
                  620.7008, 752.0332, 916.1712])

        #     LINE INTENSITIES AT 300K:
        S1 = np.array([.1310E-13, .2273E-11, .8036E-13, .2694E-11, .2438E-10, 
                   .2179E-11, .4624E-12, .2562E-10, .8369E-12, .3263E-11, .6659E-12, 
                   .1531E-08, .1707E-10, .1011E-08, .4227E-10])

        #     T COEFF. OF INTENSITIES:
        B2 = np.array([2.144, .668, 6.179, 1.541, 1.048, 3.595, 5.048, 1.405, 
                  3.597, 2.379, 2.852, .159, 2.391, .396, 1.441])

        #     AIR-BROADENED WIDTH PARAMETERS AT 300K:
        W3 = np.array([.00281, .00281, .0023, .00278, .00287, .0021, .00186, 
                   .00263, .00215, .00236, .0026, .00321, .00244, .00306, .00267])

        #     T-EXPONENT OF AIR-BROADENING:
        X = np.array([.69, .64, .67, .68, .54, .63, .60, .66, .66, .65, .69, .69, 
                  .71, .68, .70])

        #     SELF-BROADENED WIDTH PARAMETERS AT 300K:
        WS = np.array([.01349, .01491, .0108, .0135, .01541, .0090, .00788, 
                   .01275, .00983, .01095, .01313, .01320, .01140, .01253, .01275])

        #     T-EXPONENT OF SELF-BROADENING:
        XS = np.array([.61, .85, .54, .74, .89, .52, .50, .67, .65, .64, .72, 
                   1.0, .68, .84, .78])
    
        PVAP = RHO * T / 216.68
        PDA = P - PVAP
        DEN = 3.344e16 * RHO
        TI = 300 / T
        TI2 = TI**2.5
    
        #      CONTINUUM TERMS
        CON = (5.4e-10 * PDA * TI**3 + 1.8e-8 * PVAP * TI**7.5) * PVAP * F * F 

        #      ADD RESONANCES
        SUM = 0.0
        for I in range(NLINES):
            WIDTH = W3[I] * PDA * TI**X[I] + WS[I] * PVAP * TI**XS[I]
            WSQ = WIDTH * WIDTH
            S = S1[I] * TI2 * np.exp(B2[I] * (1. - TI))
            DF = [0., 0.]
            DF[0] = F - FL[I]
            DF[1] = F + FL[I]
        #  USE CLOUGH'S DEFINITION OF LOCAL LINE CONTRIBUTION
            BASE = WIDTH / (562500. + WSQ)
        #  DO FOR POSITIVE AND NEGATIVE RESONANCES
            # RES = 0.0
            RES = np.zeros(np.shape(F))
            for J in range(2):
                mask = abs(DF[J]) < 750.
                add_term = WIDTH / (np.where(mask, DF[J]**2, WIDTH / BASE - WSQ) + WSQ) - BASE
                RES = RES + add_term
            SUM = SUM + S * RES * (F / FL[I])**2

        ABH2O = .3183e-4 * DEN * SUM + CON

        ABH2O = np.array(ABH2O)
        ABH2O[(RHO * F) == 0.0] = 0.0

        return ABH2O

    @staticmethod
    def cloudabs(freq, lwc, tk):
        """ Computes absorption in nepers/km by suspended water droplets from 
            dielectric expressions of Liebe, Hufford and Manabe
            (Int. J. IR & MM Waves v.12(17) July 1991

            :param freq: Frequency in MHz
            :param lwc: Cloud liquid bulk density in kg/m^3
            :param tk: Temperature in K

            :return: Cloud absorption in 1/km

            Originally written by P. Rosenkranz. Use of other water dielectric constants here
            would give a similar answer. 
        """

        freq = freq / 1e3  # Convert to GHz
        # Inputs
        M = lwc * 1000.0  # kg/m^3 to g/m^3
        T = tk - 273.15  # K to C
        S = 0  # fresh water
        f = freq * 1e9  # GHz to Hz

        # Getting dielectric constant of water 
        e0 = 8.854e-12
        esw_inf = 4.9

        a = 1.0 + 1.613e-5 * T * S - 3.656e-3 * S + 3.210e-5 * S**2 - 4.232e-7 * S**3
        esw_T0 = 87.134 - 1.949e-1 * T - 1.276e-2 * T**2 + 2.491e-4 * T**3
        esw0 = esw_T0 * a

        tau_T0 = (1.1109e-10 - 3.824e-12 * T + 6.938e-14 * T**2 - 5.096e-16 * T**3)/(2 * np.pi)
        b = 1.0 + 2.282e-5 * T * S - 7.638e-4 * S - 7.760e-6 * S**2 + 1.105e-8 * S**3
        tau_sw = tau_T0 * b

        sigma_25 = S * (0.18252 - 1.4619e-3 * S + 2.093e-5 * S**2 - 1.282e-7 * S**3)
        delta = 25 - T
        phi_e = delta * (2.033e-2 + 1.266e-4 * delta + 2.464e-6 * delta**2 - S * (1.849e-5 - 2.551e-7 * delta + 2.551e-8 * delta**2))
        sigma = sigma_25 * np.exp(-phi_e)

        esw_real = esw_inf + (esw0 - esw_inf) / (1 + (2 * np.pi * f * tau_sw)**2)
        esw_img = (2 * np.pi * f * tau_sw * (esw0 - esw_inf)) / (1 + (2 * np.pi * f * tau_sw)**2) + sigma / (2 * np.pi * e0 * f)

        # Fresh Water Dielectric
        esw_r = esw_real + 1j * esw_img

        # Using Rayleigh approximation for cloud water absorption
        # Cloud absorption in 1/km
        abscloud = 6 * pi * 10**-2 * freq * M * np.imag(esw_r) / np.abs(esw_r + 2)**2

        return abscloud

