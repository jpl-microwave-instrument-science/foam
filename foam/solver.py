from itertools import repeat
from multiprocessing import Pool
import copy
import numpy as np 
import pandas as pd 
from scipy.linalg import svd
from scipy.optimize import least_squares
import scipy.constants as spc
from cftime import date2num
import foam.geomag as geomag
from tqdm.auto import tqdm, trange


class solver():
    """ The solver object combines individual FOAM modules to form a full forward or inverse model. 
        The default for all inputs is 'None', and inputs that are not included in the solver 
        are not called. Appropriate defaults are used in their place.  
    
        Function versions are included to facilitate ease of use with the spacecraft object

        Module inputs: 
        :param ocean: an ocean object
        :param atmosphere: an atmosphere object 
        :param ionosphere: an ionosphere object 
        :param sky: a sky object
    """

    def __init__(self, ocean=None, atmosphere=None, ionosphere=None, sky=None, verbose=False): 

        self.ocean = ocean 
        self.atmosphere = atmosphere 
        self.ionosphere = ionosphere
        self.sky = sky 
        self.verbose = verbose

    def compute_TB(self, frequency, time, lat, lon, theta, phi, ra=None, dec=None, 
                   sun_flag=None, moon_flag=None, in_epoch=False, use_time=True):
        """ This function computes forward model brightness temperatures for the states of provided solver inputs

            Inputs: 
            :param frequency: Observing frequencies in MHz
            :param time: Time in reference coords
            :param lat: Latitudes in degrees 
            :param lon: Longitudes in degrees 
            :param theta: Surface elevation angle (zero is nadir) in degrees
            :param phi: Surface azimuth angle (zero is in orbit plane) in degrees 
            :param ra: Right ascension of reflected surface vector in degrees 
            :param dec: Declination of reflected surface vector in degrees
            :param sun_flag: Boolean array where sun is in path of reflected surface vector
            :param moon_flag: Boolean array where moon is in path of reflected surface vector

        """
        if not use_time: 
            time = self.ocean.datetime[0]
            in_epoch = False
        if not in_epoch: 
            time = date2num(pd.to_datetime(time).to_pydatetime(), self.ocean.time_reference)

        # Atmosphere 
        if self.atmosphere is not None: 
            if not use_time: 
                time = self.atmosphere.datetime[0]
                in_epoch = False 
            if not in_epoch: 
                time = date2num(pd.to_datetime(time).to_pydatetime(), self.atmosphere.time_reference)
            uwind = self.atmosphere.uwind_interp((time, lat, lon))
            vwind = self.atmosphere.vwind_interp((time, lat, lon))
            wind = np.sqrt(uwind**2 + vwind**2)
            tbup, tbdn, prop_dict = self.atmosphere.get_atmosphere_tb(frequency, time, lat, lon, theta, 
                                                                      in_epoch=in_epoch, use_time=use_time)
            transup = prop_dict['upward_transmissivity']
            year = np.mean(self.atmosphere.datetime.year.values)  # Need to adjust for time accuracy later
        else: 
            uwind = vwind = wind = np.zeros(np.shape(lat))
            tbup = tbdn = 0 
            transup = 1
            year = 2005

        # Ocean 
        if self.ocean is not None: 
            if not use_time: 
                time = self.ocean.datetime[0]
                in_epoch = False 
            if not in_epoch: 
                time = date2num(pd.to_datetime(time).to_pydatetime(), self.ocean.time_reference)
            o_tb, odict = self.ocean.get_ocean_TB(frequency, time, lat, lon, uwind, vwind, theta, phi, 
                                                  in_epoch=in_epoch, use_time=use_time)
            emis = odict['emissivity']
        else: 
            o_tb = np.zeros((4, np.shape(frequency), np.shape(lat)))
            emis = np.ones((4, np.shape(frequency), np.shape(lat)))
            emis[2:, :] = 0 
        
        # Sky 
        if self.sky is not None: 
            gal_tb = self.sky.galaxy_brightness(frequency, ra, dec, wind)
            gal_tb[:, moon_flag] = self.sky.moon_brightness() 
            gal_tb[:, sun_flag] = self.sky.sun_brightness(frequency, year)
        else: 
            gal_tb = 0

        TB = o_tb
        TB[:2] = o_tb[:2] * transup + tbup + (1 - emis[:2]) * (tbdn + gal_tb * transup) * transup
        
        # Ionosphere 
        if self.ionosphere is not None: 
            if not use_time: 
                time = self.ionosphere.datetime[0]
                in_epoch = False 
            if not in_epoch: 
                time = date2num(pd.to_datetime(time).to_pydatetime(), self.ionosphere.time_reference)
            far_angle = self.ionosphere.compute_faraday_angle(frequency, time, lat, lon, theta, phi, 
                                                              in_epoch=in_epoch, use_time=use_time)
            TB = self.ionosphere.faraday_rotation(-far_angle, TB)
    
        return np.real(TB)

    def compute_spacecraft_TB(self, frequency, obs_dict, use_time=True): 
        """ Initializes 'compute_TB' using the output from the spacecraft.make_obs_grid 
            function. 

            Inputs: 
            frequency - Observing frequencies in MHz 
            obs_dict - output of spacecraft.make_obs_grid
        """ 
        epoch = obs_dict['epoch']
        lat = obs_dict['lat']
        lon = obs_dict['lon']
        theta = obs_dict['theta']
        phi = obs_dict['phi']
        ra = obs_dict['ra']
        dec = obs_dict['dec']
        sun_flag = obs_dict['sun_flag']
        moon_flag = obs_dict['moon_flag']
        TB = self.compute_TB(frequency, epoch, lat, lon, theta, phi, ra, dec, sun_flag, moon_flag, in_epoch=True, use_time=use_time)
        anc_pack = self.ancillary_pack(epoch, lat, lon, theta, phi, ra=ra, dec=dec, in_epoch=True, use_time=use_time)
        return TB, anc_pack

    def compute_TB_map(self, frequency, theta=0, phi=0, mesh_spacing=1):
        """ Initializes compute_TB for a full map of the Earth's surface with arbitrary 
            pointing information. Sky brightness assumes nadir reflection, and the sun and moon are 
            not included. The map mesh spacing can be manually defined

            Inputs:
            :param frequency: Frequency in MHz (dimension O)
            :param theta: Elevation angle of spacecraft with respect to surface normal
            :param phi: Azimuth angle of spacecraft with respect to surface normal
            :param mesh_spacing: Spacing of latitude and longitude mesh grids

            :return: Polarimetric brightness temperatures 
                    (dimension OxMxN) (M and N correspond to latitude and longitude)
        """

        lat_flat = np.linspace(89.5, -89.5, int(180 / mesh_spacing))
        lon_flat = np.linspace(-179.5, 179.5, int(360 / mesh_spacing))
        lon, lat = np.meshgrid(lon_flat, lat_flat)
        thetas = theta * np.ones(np.shape(lon))
        phis = phi * np.ones(np.shape(lon))

        return self.compute_TB(frequency, lat, lon, thetas, phis, lon, lat)

    def ancillary_pack(self, time, lat, lon, theta, phi, ra=None, dec=None, use_time=True, in_epoch=False): 
        """ Assembles ancillary data from various objects in a form that can be read by the retrieval method 
        """ 

        anc_dict = {}
        anc_dict['lat'] = lat
        anc_dict['lon'] = lon 
        anc_dict['theta'] = theta
        anc_dict['phi'] = phi 
        if (ra is not None) and (dec is not None): 
            anc_dict['ra'] = ra 
            anc_dict['dec'] = dec

        if self.ocean is not None:
            if not use_time: 
                time = self.ocean.datetime[0]
                in_epoch = False 
            if not in_epoch: 
                time = date2num(pd.to_datetime(time).to_pydatetime(), self.ocean.time_reference)
            sst = self.ocean.sst_interp((time, lat, lon))
            sst[sst < 270] = np.nan  # Eliminates fill values
            anc_dict['sst'] = sst 
            sss = self.ocean.sss_interp((time, lat, lon))
            sss[sss <= 0] = np.nan  # Eliminate fill values
            anc_dict['sss'] = sss 
            landmask = self.ocean.landmask_interp((lat, lon))
            anc_dict['landmask'] = landmask
        else: 
            anc_dict['sst'] = np.zeros(np.shape(lat))
            anc_dict['sss'] = np.zeros(np.shape(lat))
            anc_dict['landmask'] = np.zeros(np.shape(lat))

        if self.atmosphere is not None: 
            if not use_time: 
                time = self.atmosphere.datetime[0]
                in_epoch = False 
            if not in_epoch: 
                time = date2num(pd.to_datetime(time).to_pydatetime(), self.atmosphere.time_reference)
            uwind = self.atmosphere.uwind_interp((time, lat, lon))
            anc_dict['uwind'] = uwind
            vwind = self.atmosphere.vwind_interp((time, lat, lon))
            anc_dict['vwind'] = vwind
            wind = np.sqrt(uwind**2 + vwind**2)
            anc_dict['windspd'] = wind
            winddir = np.arctan2(vwind, uwind)
            anc_dict['winddir'] = winddir
            prwtr = self.atmosphere.prwtr_interp((time, lat, lon))
            anc_dict['prwtr'] = prwtr
            if self.atmosphere.lwtr_interp is not None: 
                lwtr = self.atmosphere.lwtr_interp((time, lat, lon))
            else: 
                lwtr = np.zeros(np.shape(prwtr))
            anc_dict['lwtr'] = lwtr
            airtemp = self.atmosphere.airtemp_interp((time, lat, lon))
            anc_dict['airtemp'] = airtemp
            airpres = self.atmosphere.airpres_interp((time, lat, lon))
            anc_dict['srfpres'] = airpres
        else: 
            anc_dict['uwind'] = np.zeros(np.shape(lat))
            anc_dict['vwind'] = np.zeros(np.shape(lat))
            anc_dict['windspd'] = np.zeros(np.shape(lat))
            anc_dict['winddir'] = np.zeros(np.shape(lat))
            anc_dict['prwtr'] = np.zeros(np.shape(lat))
            anc_dict['lwtr'] = np.zeros(np.shape(lat))
            anc_dict['airtemp'] = np.zeros(np.shape(lat))
            anc_dict['srfpres'] = np.zeros(np.shape(lat))

        if self.ionosphere is not None: 
            if not use_time: 
                time = self.ionosphere.datetime[0]
                in_epoch = False 
            if not in_epoch: 
                time = date2num(pd.to_datetime(time).to_pydatetime(), self.ionosphere.time_reference)
            TEC = self.ionosphere.TEC_interp((time, lat, lon))
            anc_dict['tec'] = TEC
            gm = geomag.GeoMag(self.ionosphere.mag_file)
            mag = gm.GeoMag(lat, lon, h=(675e3 / spc.foot))
            x = np.sin(np.radians(theta)) * np.cos(np.radians(phi))
            y = np.sin(np.radians(theta)) * np.sin(np.radians(phi))
            z = np.cos(np.radians(theta)) 
            mag_field = -(mag.bx * x + mag.by * y + mag.bz * z)
            anc_dict['mag_field'] = mag_field 
        else: 
            anc_dict['tec'] = np.zeros(np.shape(lat))
            anc_dict['mag_field'] = np.zeros(np.shape(lat))

        if self.sky is not None: 
            wind = anc_dict['windspd']
            coldsky = self.sky.galaxy_interp((wind, ra, dec))
            anc_dict['coldsky'] = coldsky
        else: 
            anc_dict['coldsky'] = np.zeros(np.shape(lat))

        return anc_dict

    def compute_manual_TB(self, frequency=1e3, lat=0, theta=0, phi=0, 
                          sst=300, sss=35, uwind=5, vwind=5, prwtr=0, lwtr=0, airtemp=0, srfpres=0, 
                          tec=None, coldsky=0, mag_field=0):
        """ This routine can be used for quick calculations of brightness temperature over the ocean, 
            and is used by the inversion routines to retrieve state variables from measurements

            Inputs: 
            :param frequency: Frequency in MHz
            :param sst: Sea surface temperature in Kelvin
            :param sss: Sea surface salinity in psu
            :param uwind: 10 meter U wind in m/s
            :param vwind: 10 meter V wind in m/s
            :param tau: Atmospheric optical depth (only if retrieved)
            :param tec: Ionospheric TEC (only if retrieved)
            :param coldsky: Galactic emission temperature in K
            :param lat: Latitude in degrees north
            :param theta: Incidence angle in degrees 
            :param phi: Azimuth angle in degrees 
            :param prwtr: Precipitable water vapor in kg/kg (for quick tau calculations)
            :param mag_field: Scalar magnetic field in nT

            """
        # print('Arg Stack')
        # print('frequency:{}'.format(frequency))
        # print('lat: {}'.format(lat))
        # print('theta: {}'.format(theta))
        # print('phi: {}'.format(phi))
        # print('sst: {}'.format(sst))
        # print('sss: {}'.format(sss))
        # print('uwind: {}'.format(uwind))
        # print('vwind: {}'.format(vwind))
        # print('prwtr: {}'.format(prwtr))
        # print('lwtr: {}'.format(lwtr))
        # print('airtemp: {}'.format(airtemp))
        # print('srfpres: {}'.format(srfpres))
        # print('tec: {}'.format(tec))
        # print('coldsky: {}'.format(coldsky))
        # print('mag_field: {}'.format(mag_field))

        ghz = frequency / 1e3

        # Ocean surface
        emis = self.ocean.get_ocean_emissivity(frequency, sst, sss, uwind, vwind, theta, phi)
        
        # Propagation
        # Old propagation parameterization 
        # tau_coeff = np.array([0.013664614494878, -1.83402804591467e-5, -1.02330938120674e-5])
        # tau = tau_coeff[0] + tau_coeff[1] * sst + tau_coeff[2] * prwtr
        # Teff_coeff = np.array([-7.69054064356577e-10, 4.05537390612154e-7, 2.68693677239213e-6, 
        #                     -0.00770149860447519, -0.0576638279677582, 272.404431351231])
        # Teff = np.polyval(Teff_coeff, lat)

        if self.atmosphere is not None: 
            angle = np.radians(theta)
            tbup, tbdn, prop_dict = self.atmosphere.get_atmosphere_prop(frequency, prwtr, lwtr, airtemp, srfpres, lat, angle)
            tau = prop_dict['optical_depth']
            Teff = prop_dict['effective_temperature']
            trans = np.exp(-tau / np.cos(angle))
        else: 
            trans = 1
            Teff = 0 

        if self.sky is not None: 
            Tsky = coldsky * (ghz / 1.42)**-2.7 + self.sky.Tcmb
        else: 
            Tsky = 0

        Tup = (1 - trans) * Teff
        Tdown = Tsky * trans + (1 - trans) * Teff

        TB = emis * sst 
        # Fix proper polarimetric behaviour later 
        TB[:2] = emis[:2] * sst * trans + Tup + (1 - emis[:2]) * Tdown * trans
        if self.ionosphere is not None: 
            far_angle = np.radians(1.3549e-5 / ghz**2 * tec * mag_field)
            TB = self.ionosphere.faraday_rotation(-far_angle, TB)

        return np.real(TB)

    def retrieval(self, TB, anc_dict,  # Positional information 
                  frequency=np.array([1.4e3]), bandwidth=np.array([10]), 
                  noise_figure=1, int_time=1, cal_noise=0,  # Instrument parameters 
                  retrieve=['sss', 'sst', 'windspd', 'prwtr', 'lwtr', 'tec', 'coldsky'],  # Retrieval arguments
                  options=None):  # Subselect TB data range  
        """ Inverse model to determine geophysical parameters from brightness temperature data. 
            The user can customize the retrieval process in several ways, as explained by the parameters 

            :param TB: Brightness temperature array with dimension MxNxO
                       M - Dimension of Stokes vector (TBV, TBH, U, V)
                       N - Number of frequency channels 
                       O - Number of latitude/longitude pairs
            :param bandwidth: Bandwidth in MHz 
            :param noise_figure: Receiver noise figure in dB
            :integration_time: Integration time in seconds 
            :param cal_noise: Optional noise source that can be added to NEDT 
            :param retrieve: A list of strings containing parameters that should be retrieved
                             Options are: 'sst', 'sss', 'windspd', 'tau', 'tec' and 'coldsky'

            Note: An alternate versioning scheme for TB is
            TB: Brightness temperature array with dimension MxNxOxP
                       M - Dimension of Stokes vector (TBV, TBH, U, V)
                       N - Number of frequency channels 
                       O - Number of radiometer horns 
                       P - Number of latitude/longitude pairs
        """ 

        if options is None: 
            options = {} 
        options.setdefault('ancillary_guess', True)

        # Flatten horns
        # anc_dict = {k: v.flatten() for k, v in anc_dict.items()}
        # shape = np.shape(TB)
        # TB = TB.reshape(*shape[:-2], shape[-2] * shape[-1])

        out_dict, unc_dict, anc_dict = self._retrieve(self, TB, anc_dict, frequency, bandwidth, 
                                                      noise_figure, int_time, cal_noise, 
                                                      retrieve, options)
        
        return out_dict, unc_dict, anc_dict

    def parallel_retrieval(self, TB, anc_dict,  # Positional information 
                           frequency=np.array([1.4e3]), bandwidth=np.array([10]), 
                           noise_figure=1, int_time=1, cal_noise=0,  # Instrument parameters 
                           retrieve=['sss', 'sst', 'windspd', 'prwtr', 'lwtr', 'tec', 'coldsky'],  # Retrieval arguments
                           options=None, ndiv=4, pool=None, nproc=4):  # Subselect TB data range  
        """ Inverse model to determine geophysical parameters from brightness temperature data. 
            Uses multiprocessing

            :param TB: Brightness temperature array with dimension MxNxO
                       M - Dimension of Stokes vector (TBV, TBH, U, V)
                       N - Number of frequency channels 
                       O - Number of latitude/longitude pairs
            :param bandwidth: Bandwidth in MHz 
            :param noise_figure: Receiver noise figure in dB
            :integration_time: Integration time in seconds 
            :param cal_noise: Optional noise source that can be added to NEDT 
            :param retrieve: A list of strings containing parameters that should be retrieved
                             Options are: 'sst', 'sss', 'windspd', 'tau', 'tec' and 'coldsky'
        """ 

        if options is None: 
            options = {} 
        options.setdefault('ancillary_guess', True)

        # Flatten horns
        # anc_dict = {k: v.flatten() for k, v in anc_dict.items()}
        # shape = np.shape(TB)
        # TB = TB.reshape(*shape[:-2], shape[-2] * shape[-1])
        
        # Partition inputs
        TB_list = np.array_split(TB, ndiv, axis=-1)
        anc_dict_list = [{} for x in range(ndiv)]
        for k, v in anc_dict.items():
            val_list = np.array_split(v, ndiv, axis=-1)
            for i, ad in enumerate(anc_dict_list): 
                ad[k] = val_list[i]

        if not pool: 
            local_pool = True 
            pool = Pool(processes=nproc)
        else: 
            local_pool = False

        # Build solver copy queue
        solver_list = [] 
        for n in range(nproc): 
            solver_list.append(copy.deepcopy(self))
        solver_list.extend(solver_list * (ndiv - 1))

        zipper = list(zip(solver_list, TB_list, anc_dict_list, 
                          repeat(frequency), repeat(bandwidth), 
                          repeat(noise_figure), repeat(int_time),
                          repeat(cal_noise), repeat(retrieve), repeat(options), range(ndiv)))

        tuple_dicts = pool.starmap(self._retrieve, zipper)

        if local_pool: 
            pool.close()
            pool.join()
        
        tuple_dicts = tuple(zip(*tuple_dicts))  # Reorganize by element
        out_dict_list = []
        for td in tuple_dicts: 
            keys = td[0].keys()
            merged_dict = {k: np.concatenate([d.get(k) for d in td]) for k in keys}
            out_dict_list.append(merged_dict)
        out_dict, unc_dict, anc_dict = out_dict_list

        return out_dict, unc_dict, anc_dict

    @staticmethod
    def _retrieve(solver, TB, anc_dict,  # Positional information 
                  frequency, bandwidth, noise_figure, int_time, cal_noise,  # Instrument parameters 
                  retrieve, options, ith=0): 

        # Generate error estimates 
        Trec = 290 * (10**(noise_figure / 10) - 1) 
        Tsys = TB + Trec
        bandwidth = bandwidth[np.newaxis, :, np.newaxis]
        NEDT = Tsys / np.sqrt(bandwidth * 1e6 * int_time)
        tot_error = np.real(np.sqrt(NEDT**2 + cal_noise**2))
        # print('NEDT: {}'.format(np.nanmean(NEDT).flatten()))

        # Setup ancillary data inputs 
        
        low_bounds = []
        high_bounds = [] 
        guess = {}
        good_guess = {}
        if 'sss' in retrieve: 
            low_bounds.append(0)
            high_bounds.append(50)
            guess['sss'] = 34
            good_guess['sss'] = anc_dict['sss']
        if 'sst' in retrieve: 
            low_bounds.append(260)
            high_bounds.append(320)
            guess['sst'] = 293.15
            good_guess['sst'] = anc_dict['sst']
        if 'windspd' in retrieve: 
            low_bounds.append(0)
            high_bounds.append(20)
            guess['windspd'] = 5
            good_guess['windspd'] = anc_dict['windspd']
        if 'prwtr' in retrieve: 
            low_bounds.append(0)
            high_bounds.append(100)
            guess['prwtr'] = 1e-2
            good_guess['prwtr'] = anc_dict['prwtr']
        if 'lwtr' in retrieve: 
            low_bounds.append(0)
            high_bounds.append(100)
            guess['lwtr'] = 1e-2
            good_guess['lwtr'] = anc_dict['lwtr']
        if 'tec' in retrieve: 
            low_bounds.append(0)
            high_bounds.append(100)
            guess['tec'] = 1
            good_guess['tec'] = anc_dict['tec']
        if 'coldsky' in retrieve: 
            low_bounds.append(0)
            high_bounds.append(20)
            guess['coldsky'] = 2
            good_guess['coldsky'] = anc_dict['coldsky']

        bnds = (np.array(low_bounds), np.array(high_bounds))
        guess_array = np.array(list(guess.values()))
        labels = list(guess.keys())
        lat = anc_dict['lat']
        out = np.zeros((len(guess), len(lat)))
        uncertain = np.zeros((len(guess), len(lat)))
        for i in trange(len(lat), colour='green', desc='Thread {}'.format(ith), 
                        mininterval=0.1, maxinterval=1., miniters=1): 
            if not np.isnan(TB[..., i]).any(): 
                y_err = TB[..., i] + np.random.randn(*np.shape(TB[..., i])) * tot_error[..., i]
                argpack = [frequency, y_err, tot_error[..., i], anc_dict, i]

                if options['ancillary_guess']: 
                    start = np.array([good_guess[x][i] for x in good_guess.keys()])
                else: 
                    start = guess_array
                args = (labels, argpack)
                if solver.verbose: 
                    print('Starting Retrieval  #############')
                    print('Guess: {}'.format(start))

                start_scale = start 
                start_scale[start_scale == 0] = 1

                try:
                    res = least_squares(solver._cf, start, method='trf', x_scale=start_scale, args=args,
                                    bounds=bnds, max_nfev=1e10, ftol=1e-4, xtol=1e-4)
                    out[:, i] = res.x

                    # Compute std. dev. using scipy curve_fit algorithm 
                    _, s, VT = svd(res.jac, full_matrices=False)
                    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
                    s = s[s > threshold]
                    VT = VT[:s.size]
                    pcov = np.dot(VT.T / s**2, VT)
                    uncertain[:, i] = np.sqrt(np.diag(pcov))

                except ValueError: 
                    out[:, i] = np.nan
                    uncertain[:, i] = np.nan

                if solver.verbose: 
                    print('Ending Retrieval #############')
                    print('Result: {}'.format(res.x))

            else: 
                out[:, i] = np.nan
        out_dict = dict(zip(guess.keys(), out))
        unc_dict = dict(zip(guess.keys(), uncertain))
        return out_dict, unc_dict, anc_dict

    def _cf(self, guess, labels, argpack): 

        # Unpack argpack 
        frequency = argpack[0]
        y_err = argpack[1]
        tot_error = argpack[2]
        anc_dict = argpack[3]
        index = argpack[4]

        if 'sss' in labels: 
            sss = guess[labels.index('sss')]
        else: 
            sss = anc_dict['sss'][index] 
        if 'sst' in labels: 
            sst = guess[labels.index('sst')]
        else: 
            sst = anc_dict['sst'][index]
        if 'windspd' in labels: 
            windspd = guess[labels.index('windspd')]
        else: 
            windspd = anc_dict['windspd'][index]
        if 'prwtr' in labels: 
            prwtr = guess[labels.index('prwtr')]
        else: 
            prwtr = anc_dict['prwtr'][index]
        if 'lwtr' in labels: 
            lwtr = guess[labels.index('lwtr')]
        else: 
            lwtr = anc_dict['lwtr'][index]            
        if 'tec' in labels: 
            tec = guess[labels.index('tec')]
        else: 
            tec = anc_dict['tec'][index]
        if 'coldsky' in labels: 
            coldsky = guess[labels.index('coldsky')]
        else: 
            coldsky = anc_dict['coldsky'][index]

        winddir = anc_dict['winddir'][index]
        uwind = windspd * np.cos(winddir)
        vwind = windspd * np.sin(winddir)        
        ret_y = self.compute_manual_TB(frequency=frequency, lat=anc_dict['lat'][index], 
                                       theta=anc_dict['theta'][index], phi=anc_dict['phi'][index], 
                                       sst=sst, sss=sss, uwind=uwind, vwind=vwind, prwtr=prwtr, lwtr=lwtr, 
                                       airtemp=anc_dict['airtemp'][index], srfpres=anc_dict['srfpres'][index], 
                                       tec=tec, coldsky=coldsky, mag_field=anc_dict['mag_field'][index])

        cost = (ret_y.ravel() - y_err.ravel()) / tot_error.ravel()
        return cost


def bin_observations(vals, obs_dict, lat_bins, lon_bins, epoch_bins=None): 
    pd_dict = obs_dict.copy()
    pd_dict['vals'] = vals
    nset = pd.DataFrame.from_dict(pd_dict).dropna()
    if epoch_bins is not None: 
        nset = nset[['lat', 'lon', 'epoch', 'vals']]    
    else: 
        nset = nset[['lat', 'lon', 'vals']]

    # Binning 
    lat_dex = np.digitize(nset['lat'], lat_bins)
    lon_dex = np.digitize(nset['lon'], lon_bins)
    nset['lat'] = lat_bins[lat_dex - 1]
    nset['lon'] = lon_bins[lon_dex - 1]
    if epoch_bins is not None: 
        ep_dex = np.digitize(nset['epoch'], epoch_bins)
        nset['epoch'] = epoch_bins[ep_dex - 1]
        nsgb = nset.groupby(['lat', 'lon', 'epoch'])
    else:    
        nsgb = nset.groupby(['lat', 'lon'])
    
    mean = nsgb.mean()
    std = nsgb.std()

    # Conform bins
    if epoch_bins is not None: 
        rdex = pd.MultiIndex.from_product([lat_bins, lon_bins, epoch_bins], names=['lat', 'lon', 'epoch'])
    else: 
        rdex = pd.MultiIndex.from_product([lat_bins, lon_bins], names=['lat', 'lon'])
    mean = mean.reindex(rdex)
    std = std.reindex(rdex)

    return mean, std


