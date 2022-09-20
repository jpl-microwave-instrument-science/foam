import os
import re
from itertools import repeat
from datetime import datetime
from multiprocessing import Pool
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs 
import spiceypy as spice 
from tqdm import tqdm

from .utils.mk import manual_furnish
from .utils.config import dir_path, cache_path
spice_path = os.path.join(cache_path, 'spice')


class spacecraft(): 
    """ Class defining spacecraft instrument and pointing characteristics for 
        for ocean remote sensing on Earth. Each spacecraft object is numbered 
        as a member of a constellation, and each spacecraft can be initialized 
        with several microwave instruments.  

        :param sc_number: Integer identifier of the spacecraft object within a constellation 

        The NAIF ID of a spacecraft in SPICE is -1500 - sc_number
    """

    # Geophysical constants are defined in the geophys.ker file located in assets/ and discussed in SPICE docs (e.g. evlin)
    # The spkw10 documentation is wrong about the order of the constants, and it should follow the 
    # convention in geophys.ker and below
    geophys = np.array([1.082616e-3, -2.53881e-6, -1.65597e-6, 7.43669161e-2, 120, 78, 6378.135, 1])
    mu = 3.986004418e14
    kernel_epoch_range = [-3155716758.8160305, 1577880069.1839132]
    earth_mean_radius = 6371.0088e3

    def __init__(self, sc_number=0):
        self.sc_number = sc_number  # Spacecraft number, relevant if building a constellation
        self.naif_id = -1500 - sc_number
        self.rad_number = 0  # Running count of spacecraft radiometers
    
    def close(self): 
        self.unload()
        
    def furnish(self): 
        manual_furnish() 
        spice.furnsh(self.spk_path)
        spice.furnsh(self.sclk_path)
        spice.furnsh(self.fk_path)
        spice.furnsh(self.ck_path)
        for rad_path in (self.radiometer_fk_paths + self.radiometer_ck_paths): 
            spice.furnsh(rad_path)

    def unload(self): 
        spice.clpool() 

    def write_tle_kernels(self, file=None, elems=None, tle_epoch=None, start_epoch=0., end_epoch=100., epoch_res=10):
        """ Writing kernels for an orbiting spacecraft. Written kernels include spk, sclk, ck, fk.
            Separate CKs must be written for each radiometer

            :param file: File name of the TLE file used to initialized kernels. Must be specified if mode='file'
            :param elems: List of orbital elements (see spacecraft.get_manual_elems)
            :param tle_epoch: Epoch for elements (see spacecraft.get_manual_elems)
            :param start_epoch: Starting epoch of the spacecraft operation period in seconds past J2000
            :param end_epoch: Ending epoch of the spacecraft operation period in seconds past J2000
            :param epoch_res: Time resolution for kernel calculations in seconds

            Spacecraft CK coordinate system is X -> toward planet, Y -> Orbital velocity vector
        """

        manual_furnish()
        # Start and end epochs are extended to the outermost second  
        start_epoch = np.floor(start_epoch)
        end_epoch = np.ceil(end_epoch)

        self.start_epoch = start_epoch 
        self.end_epoch = end_epoch
        self.epoch_res = epoch_res

        if file is not None: 
            tle_epoch, elems = self.read_tle(file)
        elif (elems is not None) and (tle_epoch is not None): 
            pass 
        else: 
            raise RuntimeError('Either a filename or an elements list must be provided')
        self.orbital_period = 2 * np.pi / (elems[-2] / 60)
        # Writing to SPK
        self.spk_path = os.path.join(spice_path, 'spacecraft_{}.bsp'.format(self.sc_number))
        if os.path.exists(self.spk_path): os.remove(self.spk_path)
        file_spk = spice.spkopn(self.spk_path, 'SPACECRAFT_{}'.format(self.sc_number), 0)
        # Reference frame is important, and docs say it should be J2000 (for mkspk)
        spice.spkw10(file_spk, self.naif_id, 399, 'J2000', 
                     self.kernel_epoch_range[0], self.kernel_epoch_range[1], 
                     'SPACECRAFT_{}'.format(self.sc_number), self.geophys, 1, elems, [tle_epoch])
        spice.spkcls(file_spk)
        spice.furnsh(self.spk_path)

        # Writing the SCLK
        # Note: For generality, the SCLK string format is seconds:microseconds
        # This constrains the integration times to microsecond resolution

        now = datetime.now().strftime('@%Y-%m-%d/%H:%M:%S')  # Unique indicator of SCLK is time of definition
        self.sclk_path = os.path.join(spice_path, 'spacecraft_{}.sclk'.format(self.sc_number))
        if os.path.exists(self.sclk_path): os.remove(self.sclk_path)
        file_sclk = open(self.sclk_path, 'w')
        # SCLK Kernel ID: Unique idenfitier, set as the current date 
        # SCLK Data Type: Always type 1
        # SCLK Time System: 1 is BDT and 2 is TDT (see sclk.req)
        # SCLK N Fields: Number of clock fields (e.g. major and minor ticks would be two fields)
        # SCLK Moduli: Modulus value of the fields, or after which value they restart
        # SCLK Offsets: Offsets of clock on initialization 
        # SCLK Output Delim: 1 is '.', 2 is ':', 3 is '-', 4 is ',', and 5 is ' '
        # SCLK Partition: SCLK tick values at the start and end of a time partition. FOAM spacecrafts
        #                 will only have one partition, so this corresponds to the final tick
        # SCLK Coefficients: Triplets of (SCLK values (ticks), parallel time values, rates (same unit as time values))

        diff = end_epoch - start_epoch
        file_sclk.write("""KPL/SCLK
                    \\begindata
                    SCLK_KERNEL_ID              = {1}
                    SCLK_DATA_TYPE_{0}          = 1
                    SCLK01_TIME_SYSTEM_{0}      = 1
                    SCLK01_N_FIELDS_{0}         = 2
                    SCLK01_MODULI_{0}           = ({2}, {3})
                    SCLK01_OFFSETS_{0}          = (0, 0)
                    SCLK01_OUTPUT_DELIM_{0}     = 2

                    SCLK_PARTITION_START_{0}    = 0
                    SCLK_PARTITION_END_{0}      = {4}
                    SCLK01_COEFFICIENTS_{0}     = (

                    0.0000000000000E+00     {5:.13E}     1.0000000000000E+00                           
                    {4:.13E}     {6:.13E}     1.0000000000000E+00 )                            
                    """.format(abs(self.naif_id),
                               now, 
                               int(diff),  # seconds 
                               int(1e6),  # microseconds
                               int(diff * 1e6),  # Clock ticks
                               int(start_epoch),  # seconds
                               int(end_epoch)))  # seconds
                               
        file_sclk.close()
        spice.furnsh(self.sclk_path)

        # Writing the FK
        self.fk_path = os.path.join(spice_path, 'spacecraft_{}.tf'.format(self.sc_number))
        if os.path.exists(self.fk_path): os.remove(self.fk_path)
        file_fk = open(self.fk_path, 'w')
        # FK Frame ID: Same as the CLASS ID for a CK 
        # FK Name: Name of the frame 
        # FK Class: CK frame is class 3 
        # FK Class ID: Integer used internally by SPICE to address the frame 
        #              Safe range specified by NAIF is 1400000 - 2000000
        # FK Center: Center of the frame 
        # SCLK and SPK are specified via reference to respective naif IDs (which are the same for FOAM)
        file_fk.write("""KPL/FK 
                    \\begindata
                    FRAME_SPACECRAFT_{2}        = {0}
                    FRAME_{0}_NAME              = 'SPACECRAFT_{2}'
                    FRAME_{0}_CLASS             = 3
                    FRAME_{0}_CLASS_ID          = {0}
                    FRAME_{0}_CENTER            = {1}
                    CK_{0}_SCLK                 = {1}
                    CK_{0}_SPK                  = {1}
                    """.format(int(self.naif_id * 1e3), self.naif_id, self.sc_number))    
        file_fk.close()
        spice.furnsh(self.fk_path)

        # Writing the spacecraft CK
        # Coordinate system is X -> toward planet, Y -> Orbital velocity vector
        self.ck_path = os.path.join(spice_path, 'spacecraft_{}.bck'.format(self.sc_number))
        if os.path.exists(self.ck_path): os.remove(self.ck_path)
        enc_start = spice.sce2t(self.naif_id, start_epoch)
        enc_stop = spice.sce2t(self.naif_id, end_epoch)

        # Procedures for Type 3 specification
        sepochs = np.arange(start_epoch, end_epoch, epoch_res) 
        sepochs[-1] = end_epoch
        qts = []
        sncs = []
        for i in range(len(sepochs)): 
            starg, light = spice.spkezr('{}'.format(self.naif_id), sepochs[i], 'J2000', 'NONE', '399')
            npos = -starg[:3]  # X vector
            velvec = starg[3:]  # Y vector
            cmat = spice.twovec(spice.vhat(npos), 1, spice.vhat(velvec), 2)

            # If initializing from scratch this is ok, but not so if using an actual TLE
            # lon, lat, tall = spice.recgeo(pos, 6378.137, 1 / 298.257223563)  # WGS84 ellipsoid
            # rm_lat = spice.rotate(-lat, 2)
            # rm_lon = spice.rotate(np.pi - lon, 3)
            # cmat = rm_lon @ rm_lat 
            # qts.append(spice.m2q(cmat.T.copy()))

            qts.append(spice.m2q(cmat.copy()))
            sncs.append(spice.sce2t(self.naif_id, sepochs[i]))

        qts = np.squeeze(np.array([qts]))
        sncs = np.squeeze(np.array([sncs]))
        avvs = np.zeros((len(sncs), 3))
        file_ck = spice.ckopn(self.ck_path, 'SPACECRAFT_{}'.format(self.sc_number), 0)
        spice.ckw03(file_ck, enc_start, enc_stop, int(self.naif_id * 1e3), 'J2000', True, 'SPACECRAFT_{}'.format(self.sc_number),
                    len(sncs), sncs, qts, avvs, 1, np.array([enc_start]))
        spice.ckcls(file_ck)
        spice.furnsh(self.ck_path)

        # Procedures for Type 2 specification
        # Generally aren't good past a day, but I'm leaving them here for now...
        #
        # inclination = elems[3]
        # raan = elems[4]
        # meanmotion = elems[8] / 60
        # pos, light = spice.spkpos('-999', start_epoch, 'J2000', 'LT+S', '399')
        # lon, lat, tall = spice.recgeo(pos, 6378.137, 1 / 298.257223563)  # WGS84 ellipsoid
        # # radius, lon, lat = spice.reclat(pos)
        # rm_lat = spice.rotate(-lat, 2)
        # rm_lon = spice.rotate(np.pi - lon, 3)
        # cmat = rm_lon @ rm_lat 
        # start_qt = spice.m2q(cmat.T.copy())
        # raan_vec = spice.rotvec(np.array([1, 0, 0]), -raan, 3)
        # vel_vec = spice.vrotv(np.array([0, 0, 1]), raan_vec, inclination)
        # ang_vel_vec = meanmotion * vel_vec  # In base reference frame
        # # Check if the vel vec and cmat vec are normal to each other 
        # # c_vec = cmat @ np.array([-1, 0, 0])
        # # print('Normal check: {0}'.format(np.dot(c_vec, vel_vec)))
        # file_ck = spice.ckopn(dir_path + 'assets/spice/current_spacecraft.bck', 'SPACECRAFT', 0)
        # spice.ckw02(file_ck, enc_start, enc_stop, -999000, 'J2000', 'SPACECRAFT', 1, np.array([enc_start]), np.array([enc_stop]),
        #             start_qt, ang_vel_vec, [int_time * 1e-3])
        # spice.ckcls(file_ck)
        # spice.furnsh(spice_path + 'current_spacecraft.bck')

    def write_radiometer_ck(self, look_angle, look_axis, rpm, scan_axis, antenna_pattern=None):
        """ Writes the camera kernel for a given radiometer. 
            This can be called multiple times with different ck numbers 
            to create several radiometers
            
            :param look_angle: Look angle in degrees 
            :param look_axis: Reference axis for look angle rotation
                              Either 'X', 'Y', 'Z' in spacecraft frame or a vector
                              A negative sign can also be added e.g. -X
            :param rpm: Revolutions per minute of scan platform.
            :param scan_axis: Scan rotation axis
                              Either 'X', 'Y', 'Z' in spacecraft frame or a vector
                              A negative sign can also be added e.g. -X
            :param antenna_pattern: Text file describing radiometer antenna pattern
                                    This is a placeholder for future functionality.

            Spacecraft CK coordinate system is X -> toward planet, Y -> Orbital velocity vector
            and radiometer CK is defined relative to the spacecraft system using the input arguments
        """ 
        start_epoch = self.start_epoch
        end_epoch = self.end_epoch

        # Check list of radiometer_ck_paths
        if hasattr(self, 'radiometer_ck_paths'): 
            last = self.radiometer_ck_paths[-1]
            last = re.split('[_.]', last)
            count = int(last[-2]) 
            count += 1
        else:
            self.radiometer_fk_paths = [] 
            self.radiometer_ck_paths = []
            count = 0

        # Select radiometer number by searching kernel pool 
        # for existing radiometer kernels   
        # count = 0
        # stay_flag = True 
        # while stay_flag: 
        #     string = 'spacecraft_{0}_radiometer_{1}.bck'.format(self.sc_number, count)
        #     found_flag = False
        #     num = spice.ktotal('all')
        #     for i in range(0, num): 
        #         nm, tp, src, hnd = spice.kdata(i, 'all')
        #         if string in nm: 
        #             found_flag = True
        #     if found_flag: 
        #         count += 1 
        #     else: 
        #         stay_flag = False

        # Writing the FK
        fk_path = os.path.join(spice_path, 'spacecraft_{0}_radiometer_{1}.tf'.format(self.sc_number, count))
        if os.path.exists(fk_path): os.remove(fk_path)
        file_fk = open(fk_path, 'w')
        file_fk.write("""KPL/FK 
                    \\begindata
                    FRAME_SPACECRAFT_{3}_RADIOMETER_{1}    = {0}
                    FRAME_{0}_NAME          = 'SPACECRAFT_{3}_RADIOMETER_{1}'
                    FRAME_{0}_CLASS         = 3
                    FRAME_{0}_CLASS_ID      = {0}
                    FRAME_{0}_CENTER        = {2}
                    CK_{0}_SCLK             = {2}
                    CK_{0}_SPK              = {2}
                    """.format(int(self.naif_id * 1e3 - 1 - count), count, self.naif_id, self.sc_number))    
        file_fk.close()
        spice.furnsh(fk_path)
        self.radiometer_fk_paths.append(fk_path)

        ck_path = os.path.join(spice_path, 'spacecraft_{0}_radiometer_{1}.bck'.format(self.sc_number, count))
        if os.path.exists(ck_path): os.remove(ck_path)
        enc_start = spice.sce2t(self.naif_id, start_epoch)
        enc_stop = spice.sce2t(self.naif_id, end_epoch)
        file_ck = spice.ckopn(ck_path, 'SPACECRAFT_{0}_RADIOMETER_{1}'.format(self.sc_number, count), 0)
        look_angle = np.radians(look_angle)
        
        if str(look_axis) == 'X': 
            look_axis = np.array([1, 0, 0])
        elif str(look_axis) == '-X': 
            look_axis = np.array([-1, 0, 0])
        elif str(look_axis) == 'Y': 
            look_axis = np.array([0, 1, 0]) 
        elif str(look_axis) == '-Y': 
            look_axis = np.array([0, -1, 0]) 
        elif str(look_axis) == 'Z': 
            look_axis = np.array([0, 0, 1])
        elif str(look_axis) == '-Z': 
            look_axis = np.array([0, 0, -1])
        rm = spice.axisar(look_axis, look_angle)
        start_qt = spice.m2q(rm)

        if str(scan_axis) == 'X': 
            scan_axis = np.array([1, 0, 0])
        elif str(scan_axis) == '-X': 
            scan_axis = np.array([-1, 0, 0])
        elif str(scan_axis) == 'Y': 
            scan_axis = np.array([0, 1, 0]) 
        elif str(scan_axis) == '-Y': 
            scan_axis = np.array([0, -1, 0]) 
        elif str(scan_axis) == 'Z': 
            scan_axis = np.array([0, 0, 1])
        elif str(scan_axis) == '-Z': 
            scan_axis = np.array([0, 0, -1])
        scan_axis = scan_axis
        ang_vel = -rpm * 2 * np.pi / 60 
        ang_vel_vec = ang_vel * scan_axis
        spice.ckw02(file_ck, enc_start, enc_stop, int(self.naif_id * 1e3 - 1 - count), 
                    'SPACECRAFT_{}'.format(self.sc_number), 
                    'SPACECRAFT_{0}_RADIOMETER_{1}'.format(self.sc_number, count), 1,
                    np.array([enc_start]), np.array([enc_stop]), start_qt, ang_vel_vec, [1e-6])
        spice.ckcls(file_ck)
        spice.furnsh(ck_path)
        self.radiometer_ck_paths.append(ck_path)
        self.rad_number += 1

    def write_stationary_kernels(self, sub_long, sub_lat, height): 
        """ Writes SPICE kernels for a stationary spacecraft with a single pointing

            :param sub_long: Sub-Observer longitude in degrees 
            :param sub_lat: Sub-Observer latitude in degrees
            :param height: Height above sub-observer point in km

            This routine is currently broken, and will raise an error if called.  
        """

        raise RuntimeError('This kernel procedure is out of date')
        sub_lat = np.radians(sub_lat)
        sub_long = np.radians(sub_long)
        os.system('rm ' + dir_path + 'assets/spice/current_spacecraft.bsp')
        file = spice.spkopn(dir_path + 'assets/spice/current_spacecraft.bsp', 'SPACECRAFT', 0)
        rect = spice.latrec(self.earth_mean_radius / 1e3 + height, sub_long, sub_lat)
        spice.spkw09(file, -999, 399, 'ITRF93', 
            self.kernel_epoch_range[0], self.kernel_epoch_range[1],  # 1900 and 2500 epochs
            'Spacecraft Ephemeris', 1, 2,
            [[rect[0], rect[1], rect[2], 0, 0, 0], 
            [rect[0], rect[1], rect[2], 0, 0, 0]], 
            [self.kernel_epoch_range[0], self.kernel_epoch_range[1]])
        spice.spkcls(file)
        os.system('rm ' + dir_path + 'assets/spice/current_spacecraft.tf')
        file = open(dir_path + 'assets/spice/current_spacecraft.tf', 'w')
        # Write fixed offset reference frame
        # X towards planet, Y left, Z up 
        # Check ITRF image in spice docs
        file.write("""KPL/FK 
                    \\begindata
                    FRAME_SPACECRAFT            = -999000
                    FRAME_-999000_NAME          = 'SPACECRAFT'
                    FRAME_-999000_CLASS         = 4
                    FRAME_-999000_CLASS_ID      = -999000
                    FRAME_-999000_CENTER        = -999
                    OBJECT_-999_FRAME           = -999000
                    TKFRAME_-999000_SPEC     = 'ANGLES'
                    TKFRAME_-999000_RELATIVE = 'ITRF93'
                    TKFRAME_-999000_ANGLES   = ( {}, {}, {} )
                    TKFRAME_-999000_AXES     = (       3,        2,   3 )
                    TKFRAME_-999000_UNITS    = 'RADIANS'
                    """.format(np.pi, -sub_lat, -sub_long))
        file.close()
        spice.furnsh(dir_path + 'assets/spice/current_spacecraft.bsp')
        spice.furnsh(dir_path + 'assets/spice/current_spacecraft.tf')

    def make_track_grid(self, start_epoch, end_epoch, resolution):
        """ Generates a sequence of sub-spacecraft latitutdes and longitudes 
            
            :param start_epoch: Observation start epoch
            :param end_epoch: Observation end epoch
            :param resolution: Time resolution of tracks

            :return: A dictionary of longitudes, latitudes, and epochs 
        """
        epochs = np.arange(start_epoch, end_epoch, resolution) 
        lons = np.zeros(len(epochs))
        lats = np.zeros(len(epochs))
        for i in range(len(epochs)):
            try: 
                spoint, trgepc, srfvec = spice.sincpt('ELLIPSOID', 'EARTH', epochs[i], 'ITRF93', 'NONE', '{}'.format(self.naif_id), 
                                                      'SPACECRAFT_{}'.format(self.sc_number), np.array([1, 0, 0]))
            except spice.utils.exceptions.NotFoundError: 
                spoint = np.array([1, 0, 0])
                print('WARNING: No intercept determined for epoch {}, setting intercept to zero'.format(epochs[i]))
            lon, lat, tall = spice.recgeo(spoint, 6378.137, 1 / 298.257223563)  # WGS84 ellipsoid
            lons[i] = np.degrees(lon)
            lats[i] = np.degrees(lat)

        track_dict = {'epoch': epochs, 'lon': lons, 'lat': lats}

        return track_dict       

    def make_obs_grid(self, start_epoch, end_epoch, resolution, bounds=None):
        """ Generates a sequence of latitudes, longitudes and angles of observations between the
            start and end epochs with a given time resolution for each radiometer instrument 
            on the spacecraft 

            :param start_epoch: Observation start epoch
            :param end_epoch: Observation end epoch
            :param resolution: Time resolution of tracks
            :param bounds: Tuple of tuples ((lower_lat, upper_lat), (lower_lon, upper_lon))
                           Default is None

            :return: A dictionary of epochs, longitudes, latitudes, incidence elevations (theta), 
                    incidence azimuths (phi), right ascension and declination in the direction of 
                    specular reflection, and flags for if the sun and moon are in that direction 

                    The shape of the arrays in each dictionary value (except epochs) is NxM where N is the number 
                    of radiometers on the spacecraft, and M is the length of the epochs value

        """ 

        epochs = np.arange(start_epoch, end_epoch, resolution) 
        if bounds is None: 
            obs_dict = self._mog(epochs)
        else: 
            obs_dict = self._mog_bound(epochs, bounds)

        return obs_dict

    def _mog(self, epochs, ith=0): 
        """ Grid routine taking a single argument of epochs
        """

        # Commented blocks are a bit slower, but more legible 
        # lons = np.zeros((self.rad_number, len(epochs)))
        # lats = np.zeros((self.rad_number, len(epochs)))
        # thetas = np.zeros((self.rad_number, len(epochs)))
        # phis = np.zeros((self.rad_number, len(epochs)))
        # ras = np.zeros((self.rad_number, len(epochs)))
        # decs = np.zeros((self.rad_number, len(epochs)))
        # sun_flag = np.zeros((self.rad_number, len(epochs)), dtype=bool)
        # moon_flag = np.zeros((self.rad_number, len(epochs)), dtype=bool)

        splist = list(np.zeros(self.rad_number * len(epochs)))
        srlist = list(np.zeros(self.rad_number * len(epochs)))

        for i, eps in enumerate(tqdm(epochs, colour='blue', desc='Thread {}'.format(ith))):
            for j in range(self.rad_number):
                try: 
                    spoint, trgepc, srfvec = spice.sincpt('ELLIPSOID', 'EARTH', eps, 'ITRF93', 'NONE', '{}'.format(self.naif_id), 
                                                          'SPACECRAFT_{0}_RADIOMETER_{1}'.format(self.sc_number, j), np.array([1, 0, 0]))
                except spice.utils.exceptions.NotFoundError: 
                    spoint = np.array([1, 0, 0])
                    srfvec = np.array([1, 0, 0])
                    print('WARNING: No intercept determined for epoch {}, setting intercept to zero'.format(eps))
                
                # Run in loop
                # lon, lat, theta, phi, ra, dec, found_sun, found_moon = self.get_intercept_params(spoint, srfvec, eps)
                # lons[j, i] = np.degrees(lon)
                # lats[j, i] = np.degrees(lat)
                # thetas[j, i] = np.degrees(theta)
                # phis[j, i] = np.degrees(phi)
                # ras[j, i] = np.degrees(ra)
                # decs[j, i] = np.degrees(dec)
                # sun_flag[j, i] = found_sun 
                # moon_flag[j, i] = found_moon 

                splist[i * self.rad_number + j] = spoint
                srlist[i * self.rad_number + j] = srfvec

        # External list comps, harder to read but a bit faster
        zip_eps = epochs[:, np.newaxis] * np.ones(self.rad_number)  # Pad epochs
        zipper = list(zip(splist, srlist, zip_eps.flatten()))  # Zip them up
        pack = np.array([self.get_intercept_params(x[0], x[1], x[2]) for x in zipper], dtype=object)  # Get param pack
        keys = ['epoch', 'lon', 'lat', 'theta', 'phi', 'ra', 'dec', 'sun_flag', 'moon_flag']
        vals = [np.degrees(pack[:, x].astype(float).reshape(len(epochs), self.rad_number).T) for x in range(6)]  # Make obs dict
        vals2 = [pack[:, x].astype(bool).reshape(len(epochs), self.rad_number).T for x in [6, 7]]
        
        vals = [zip_eps.T] + vals + vals2

        obs_dict = dict(zip(keys, vals))

        # obs_dict = {'epoch': epochs, 'lon': lons, 'lat': lats, 'theta': thetas, 'phi': phis, 
        #             'ra': ras, 'dec': decs, 'sun_flag': sun_flag, 'moon_flag': moon_flag}

        return obs_dict

    def _mog_bound(self, epochs, bounds, ith=0): 
        """ Grid routine taking arguments of epochs and bounds 
            Separate since the act of latitude checking may 
            reduce performance in the bound-free case
        """

        splist = list(np.zeros(self.rad_number * len(epochs)))
        srlist = list(np.zeros(self.rad_number * len(epochs)))
        lat_bounds = bounds[0]
        lon_bounds = bounds[1]
        for i, eps in enumerate(tqdm(epochs, colour='blue', desc='Thread {}'.format(ith))):
            pos, lt = spice.spkpos(str(self.naif_id), eps, 'ITRF93', 'NONE', 'EARTH')
            lon, lat, tall = spice.recgeo(pos, 6378.137, 1 / 298.257223563)  # WGS84 ellipsoid
            lon = np.degrees(lon)
            lat = np.degrees(lat)
            if (lon > lon_bounds[0]) and (lon < lon_bounds[1]) and (lat > lat_bounds[0]) and (lat < lat_bounds[1]): 
                for j in range(self.rad_number):
                    try: 
                        spoint, trgepc, srfvec = spice.sincpt('ELLIPSOID', 'EARTH', eps, 'ITRF93', 'NONE', '{}'.format(self.naif_id), 
                                                              'SPACECRAFT_{0}_RADIOMETER_{1}'.format(self.sc_number, j), np.array([1, 0, 0]))
                    except spice.utils.exceptions.NotFoundError: 
                        spoint = np.array([1, 0, 0])
                        srfvec = np.array([1, 0, 0])
                        print('WARNING: No intercept determined for epoch {}, setting intercept to zero'.format(eps))

                    splist[i * self.rad_number + j] = spoint
                    srlist[i * self.rad_number + j] = srfvec

        # Thin zero valued lists
        rlsp = range(len(splist))
        mask = np.array([True if type(splist[x]) is np.ndarray else False for x in rlsp])
        splist = [splist[x] for x in rlsp if type(splist[x]) is np.ndarray]
        srlist = [srlist[x] for x in rlsp if type(srlist[x]) is np.ndarray]
        # Do all epochs check out? 
        mask = np.bitwise_and.reduce(mask.reshape(len(epochs), self.rad_number), axis=1)
        epochs = epochs[mask]
        if (mask == False).all():
            return {}
        else:  
            # External list comps, harder to read but a bit faster
            zip_eps = epochs[:, np.newaxis] * np.ones(self.rad_number)  # Pad epochs
            zipper = list(zip(splist, srlist, zip_eps.flatten()))  # Zip them up
            pack = np.array([self.get_intercept_params(x[0], x[1], x[2]) for x in zipper], dtype=object)  # Get param pack
            keys = ['epoch', 'lon', 'lat', 'theta', 'phi', 'ra', 'dec', 'sun_flag', 'moon_flag']
            vals = [np.degrees(pack[:, x].astype(float).reshape(len(epochs), self.rad_number).T) for x in range(6)]  # Make obs dict
            vals2 = [pack[:, x].astype(bool).reshape(len(epochs), self.rad_number).T for x in [6, 7]]
        
            vals = [zip_eps.T] + vals + vals2
            obs_dict = dict(zip(keys, vals))

            # obs_dict = {'epoch': epochs, 'lon': lons, 'lat': lats, 'theta': thetas, 'phi': phis, 
            #             'ra': ras, 'dec': decs, 'sun_flag': sun_flag, 'moon_flag': moon_flag}

            return obs_dict

    def make_antenna_grid(self, start_epoch, end_epoch, resolution, power_cutoff=0.5): 
        """ Similar to make_obs_grid, but each discrete observation contains an array of
            lat/lon/ra/dec values corresponding to rays sampling an antenna pattern

            This routine is still under construction, and will raise an error when called 
        """

        raise NotImplementedError('Still under construction')
        epochs = np.arange(start_epoch, end_epoch, resolution) 
        packets = np.zeros((self.rad_number, len(epochs)), dtype=object)

        ant_theta = self.ant_dict['theta']
        ant_phi = self.ant_dict['phi']
        
        for i, eps in enumerate(epochs): 
            for j, rads in range(self.rad_number): 
                lons = np.zeros((len(ant_theta), len(ant_phi)))
                lats = np.zeros((len(ant_theta), len(ant_phi)))
                thetas = np.zeros((len(ant_theta), len(ant_phi)))
                phis = np.zeros((len(ant_theta), len(ant_phi)))
                ras = np.zeros((len(ant_theta), len(ant_phi)))
                decs = np.zeros((len(ant_theta), len(ant_phi)))
                ref_ras = np.zeros((len(ant_theta), len(ant_phi)))
                ref_decs = np.zeros((len(ant_theta), len(ant_phi)))
                sun_flag = np.zeros((len(ant_theta), len(ant_phi)), dtype=bool)
                moon_flag = np.zeros((len(ant_theta), len(ant_phi)), dtype=bool)
                for k, th in enumerate(self.ant_dict['theta']): 
                    for l, phi in enumerate(self.ant_dict['phi']): 
                        if self.antenna_pattern(th, phi) >= power_cutoff: 
                            # Note, I'm not defining an azimuth convention yet, which
                            # will be an issue for non-symmetrical power patterns
                            dvec = spice.rotvec(np.array([1, 0, 0]), np.radians(th), 2)
                            dvec = spice.vhat(spice.rotvec(dvec, np.radians(phi), 1))

                            try: 
                                # Check for surface intercept
                                spoint, trgepc, srfvec = spice.sincpt('ELLIPSOID', 'EARTH', eps, 'ITRF93', 'NONE', 
                                                                      '{}'.format(self.naif_id), 
                                                                      'SPACECRAFT_{0}_RADIOMETER_{1}'.format(self.sc_number, j), 
                                                                      dvec)
                                lon, lat, theta, phi, ra, dec, found_sun, found_moon = self.get_intercept_params(spoint, srfvec, eps)

                                lons[k, l] = np.degrees(lon)
                                lats[k, l] = np.degrees(lat)
                                thetas[k, l] = np.degrees(theta)
                                phis[k, l] = np.degrees(phi)
                                ras[k, l] = np.nan
                                decs[k, l] = np.nan
                                ref_ras[k, l] = np.degrees(ra)
                                ref_decs[k, l] = np.degrees(dec)
                                sun_flag[k, l] = found_sun 
                                moon_flag[k, l] = found_moon 

                            except spice.utils.exceptions.NotFoundError: 
                                # Cant find surface intercept, get CK pointing 
                                rad_id = int(self.naif_id * 1e3 - 1 - j)
                                encep = spice.sce2t(eps)
                                cmat, encep = spice.ckgp(rad_id, encep, 0, 'J2000')
                                inertial_pointing = cmat.T @ dvec
                                ran, ra, dec = spice.recrad(inertial_pointing)
                                found_sun, found_moon = self.check_moon_sun(eps, inertial_pointing)

                                lons[k, l] = np.nan
                                lats[k, l] = np.nan
                                thetas[k, l] = np.nan
                                phis[k, l] = np.nan
                                ras[k, l] = np.degrees(ra)
                                decs[k, l] = np.degrees(dec)
                                ref_ras[k, l] = np.nan
                                ref_decs[k, l] = np.nan
                                sun_flag[k, l] = found_sun 
                                moon_flag[k, l] = found_moon 

    def make_parallel_grid(self, start_epoch, end_epoch, resolution, bounds=None, ndiv=4, pool=None, nproc=4): 
        """ Parallel implementation of make_obs_grid, which computes the grid over subwindows
            using spacecraft copies  

            :param start_epoch: Observation start epoch
            :param end_epoch: Observation end epoch
            :param resolution: Time resolution of tracks
            :param bounds: Tuple of tuples ((lower_lat, upper_lat), (lower_lon, upper_lon))
                           Default is None
            :param ndiv: Number of time blocks to feed to the pool (generally the same as ndiv)
            :param pool: Process pool. If not assigned, this function creates one 
            :param nproc: Number of processes in the pool, overriden if pool is provided

            :return: A dictionary of epochs, longitudes, latitudes, incidence elevations (theta), 
                    incidence azimuths (phi), right ascension and declination in the direction of 
                    specular reflection, and flags for if the sun and moon are in that direction 

                    The shape of the arrays in each dictionary value (except epochs) is NxM where N is the number 
                    of radiometers on the spacecraft, and M is the length of the epochs value

        """

        epochs = np.arange(start_epoch, end_epoch, resolution)
        epoch_list = np.array_split(epochs, ndiv)

        if not pool: 
            local_pool = True 
            pool = Pool(processes=nproc)
        else: 
            local_pool = False

        # Build craft copy queue
        craft_list = [] 
        for n in range(nproc): 
            craft_list.append(copy.deepcopy(self))
        craft_list.extend(craft_list * (ndiv - 1))

        zipper = list(zip(craft_list, epoch_list, repeat(bounds), range(ndiv)))
        dicts = pool.starmap(self._mpg, zipper)

        if local_pool: 
            pool.close()
            pool.join()

        # Now that's a comprehension!
        # obs_dict = {k: np.concatenate([d.get(k) for d in dicts]) for k in {k for d in dicts for k in d}}
        keys = ['epoch', 'lon', 'lat', 'theta', 'phi', 'ra', 'dec', 'sun_flag', 'moon_flag']
        dicts = [d for d in dicts if d]  # Filter out empty dicts
        obs_dict = {k: np.concatenate([d.get(k) for d in dicts], axis=1) for k in keys}
        return obs_dict

    @staticmethod
    def _mpg(craft, eps, bounds, i=0):
        """ Parallel gridder map function 
        """
        craft.furnish()
        if bounds is None: 
            obs_dict = craft._mog(eps, i)
        else: 
            obs_dict = craft._mog_bound(eps, bounds, i)
        craft.unload()
        return obs_dict

    def get_intercept_params(self, spoint, srfvec, epoch): 
        # Get lat and lon 
        lon, lat, tall = spice.recgeo(spoint, 6378.137, 1 / 298.257223563)  # WGS84 ellipsoid

        # Get theta and phi incidence angle

        # Slightly slower 
        # colat = np.pi / 2 - lat 
        # m1 = spice.rotate(np.pi / 2, 3)
        # m2 = spice.rotate(colat, 2)
        # m3 = spice.rotate(lon, 3)
        # ef2topo = m1 @ m2 @ m3 

        # Slightly faster
        sinl = np.sin(lon)
        cosl = np.cos(lon)
        sinp = np.sin(lat)
        cosp = np.cos(lat)
        ef2topo = np.array([[-sinl, cosl, 0],
                            [-cosl * sinp, -sinl * sinp, cosp],
                            [cosl * cosp, sinl * cosp, sinp]])
        
        topovec = ef2topo @ srfvec
        theta = np.pi - np.arccos(topovec[2] / spice.vnorm(topovec))
        phi = np.pi + np.arctan2(topovec[1], topovec[0])

        # Get specular ra and dec 
        ra, dec, point_vec = self.specular_reflection(spoint, srfvec)

        # Get sun and moon flags 
        found_sun, found_moon = self.check_moon_sun(epoch, point_vec)

        return lon, lat, theta, phi, ra, dec, found_sun, found_moon

    def get_manual_elems(self, ndt2=0., ndd6=0., bstar=0., inclination=0., raan=0., eccentricity=0.,
                         argperigee=0., meananomaly=0., height=500000., tle_epoch=0.): 
        """ A helper class that forms an element list from TLE inputs. Use this function also to reference
            the units for each parameter required for successful TLE kernel creation 
 
            :param ndt2: First derivative of mean motion in radians/minute^2
            :param ndd6: Second derivative of mean motion in radians/minute^3
            :param bstar: Radiation pressure coefficient (unitless)
            :param inclination: Inclination in radians 
            :param raan: Right Ascension of Ascending Node in radians 
            :param eccentricity: Eccentricity (unitless)
            :param argperigee: Argument of Perigee in radians 
            :param meananomaly: Mean anomaly in radians 
            :param height: Height of semimajor axis above the surface of Earth in meters
            :param tle_epoch: Epoch specified in the TLE set in seconds past J2000

            :return: A list of elements that can be used as an argument to spacecraft.write_tle_kernels

        """
        semimaj = height + self.earth_mean_radius
        meanmotion = np.sqrt(self.mu / semimaj ** 3)  # in radians/sec
        elems = [ndt2, ndd6, bstar, inclination, raan, eccentricity, argperigee, meananomaly, meanmotion * 60, tle_epoch]
        return elems

    @staticmethod
    def read_tle(filename): 
        """ Reads TLE from file using the SPICE getelm function 

            :param filename: TLE text file

            :returns: 
                - epoch - SPICE formatted epoch
                - elements - SPICE formatted elements
        """
        f = open(filename, 'r')
        g = f.readlines()
        h = [g[x].strip() for x in range(len(g))]
        epoch, elements = spice.getelm(2000, 500, h)
        f.close()
        return epoch, elements 

    def write_tle_file(self, epoch=datetime(2000, 1, 1), inclination=0, raan=0, eccentricity=0, argperigee=0, meananomaly=0, height=0, revnumber=0):
        """ Writes two line element strings from inputs 
            All arguments should be floats except for epoch, which should be a datetime object
            Hardcoding the derivatives of mean motion and the radiation pressure drag to zero

            :param inclination: Inclination in radians 
            :param raan: Right Ascension of Ascending Node in radians 
            :param eccentricity: Eccentricity (unitless)
            :param argperigee: Argument of Perigee in radians 
            :param meananomaly: Mean anomaly in radians 
            :param height: Height of semimajor axis above the surface of Earth in meters
            :param revnumber: Number of revolutions around the Earth at the desired epoch

        """  
        if type(epoch) is not datetime: 
            raise ValueError("epoch is not a datetime object")

        semimaj = height + self.earth_mean_radius
        meanmotion = np.sqrt(self.mu / semimaj ** 3) * 60  # in radians/min

        satname = "FOAMSAT    "  # 11 characters

        # Line 1 
        ref_year = datetime(epoch.year, 1, 1)
        days = (epoch - ref_year).total_seconds() / 84600
        epoch_string = "{0}{1:012.8f}".format(epoch.strftime('%y'), days)
        line1 = "1 00000U 000000 {0} 0.00000000 00000-0 00000-0 0 ".format(epoch_string)
        line1 += self.tle_checksum(line1)

        # Line 2 
        ecc = int("{0:.7f}".format(eccentricity).split('.')[1])
        line2 = "2 00000 {0:07.4f} {1:08.4f} {2} {3:08.4f} {4:08.4f} {5:011.8f} {6:05d}".format(inclination, raan, ecc, argperigee, meananomaly, meanmotion, revnumber)
        line2 += self.tle_checksum(line2)
        # Demo TLE, see space-track.org and wiki for formatting
        self.tle = satname + '\n' + line1 + '\n' + line2

    @staticmethod
    def tle_checksum(string): 
        checksum = 0 
        for i in string:
            if i.isdigit(): 
                checksum += int(i)
            elif i == '-':
                checksum += 1
        return str(checksum % 10)

    @staticmethod
    def specular_reflection(spoint, srfvec):
        """ Calculates the specular reflection vector for a given spacecraft look direction 

            :param spoint: Surface intercept point on target body
            :param srfvec: Vector from observer to intercept point 

            :return: RA, Dec in radians and unit vector in specular reflection direction
        
        """

        # Old, expensive way of doing it
        # angle = np.pi - spice.vsep(spoint, srfvec)
        # normal = spice.vcrss(spoint, srfvec)
        # rotframe = spice.twovec(normal, 1, spoint, 2)
        # invframe = spice.invert(rotframe)
        # flipvec = spice.vrotv(rotframe @ srfvec, np.array([1, 0, 0]), np.array([np.pi]))
        # rotvec = spice.vrotv(flipvec, np.array([1, 0, 0]), 2 * angle)
        # point_vec = invframe @ rotvec

        # Cheaper way 
        n = spice.vhat(spoint)
        d = spice.vhat(srfvec)
        point_vec = d - 2 * np.dot(d, n) * n

        ran, ra, dec = spice.recrad(point_vec)

        return ra, dec, point_vec

    @staticmethod
    def check_moon_sun(epoch, point_vec):
        """ Check if the moon and the sun intersect with a pointing vector. 
            This can be used with the outputs of the specular_reflection function 

        """

        # Spiceypy raises errors if intercepts aren't found, so found flags have to deal with this
        try: 
            spoint, trgepc, srfvec = spice.sincpt('ELLIPSOID', 'SUN', epoch, 'IAU_SUN', 'LT+S', 'EARTH', 'ITRF93', point_vec)
            found_sun = True 
        except spice.utils.exceptions.NotFoundError: 
            found_sun = False 
        try: 
            spoint, trgepc, srfvec = spice.sincpt('ELLIPSOID', 'MOON', epoch, 'MOON_ME', 'LT+S', 'EARTH', 'ITRF93', point_vec)
            found_moon = True
        except spice.utils.exceptions.NotFoundError: 
            found_moon = False 
        return found_sun, found_moon


# Spacecraft-related utility methods 

def make_smap(start_epoch, end_epoch, epoch_res=1, sc_number=0): 
    """ Builds the SMAP spacecraft using the spacecraft module.

        Spin platform rotation rate: 14.6 rpm 

        Antenna look angle (emission angle): 35.5 deg. (40 deg.)

        Orbital altitude: 670 km 

        Integration time: 17.5 ms (resolution provided for level 1 products)
    """ 
    manual_furnish()
    smap_craft = spacecraft(sc_number=sc_number)
    tle = ['1 40376U 15003A   20247.33652497  .00000025  00000-0  13841-4 0  9991',
           '2 40376  98.1240 252.7233 0001762  90.8135 269.3273 14.63327320298611']
    epoch, elements = spice.getelm(2000, 500, tle)
    smap_craft.write_tle_kernels(elems=elements, tle_epoch=epoch, start_epoch=start_epoch, 
                                 end_epoch=end_epoch, epoch_res=epoch_res)
    # look_angle, look_axis, rpm, scan_axis
    smap_craft.write_radiometer_ck(35.5, 'Y', 14.6, 'X')

    return smap_craft


def make_aquarius(start_epoch, end_epoch, epoch_res=1, sc_number=0): 
    """ Builds the Aquarius spacecraft using the spacecraft module.

        Horn angles: 25.8 deg., 33.8 deg., 40.3 deg.

        Orbital altitude: 657 km 

        Integration time: 12s (resolution provided for level 1 products)
    """ 
    
    # From an RSS memo
    # Aquarius horns (pre-launch, 0.5 degrees accurate) were pointed as 
    # Horn      Elevation       Azimuth
    # Horn 1    25.82           9.84
    # Horn 2    33.82           -15.29
    # Horn 3    40.37           6.55
    # The total angle for each radiometer can then be computed as follows
    # horn = [1, 2, 3]
    # nadir_angle = [-25.82, -33.82, -40.37]
    # azimuth_angle = [9.84, -15.29, 6.55]
    # for i in range(len(horn)):  
    #     rm = spice.rotate(np.radians(azimuth_angle[i]), 3)
    #     rm2 = spice.rotate(np.radians(nadir_angle[i]), 2)
    #     axis, angle = spice.raxisa(rm @ rm2)
    #     print('Horn {0}: Axis {1}, Angle {2}'.format(horn[i], axis, np.degrees(angle)))
    manual_furnish()
    aq_craft = spacecraft(sc_number=sc_number)
    tle = ['1 37673U 11024A   21237.59766439  .00000119  00000-0  26113-4 0  9996',
           '2 37673  98.0067 245.2609 0001526  59.7629  52.4724 14.73133816548758']
    epoch, elements = spice.getelm(2000, 500, tle)
    aq_craft.write_tle_kernels(elems=elements, tle_epoch=epoch, start_epoch=start_epoch, 
                               end_epoch=end_epoch, epoch_res=epoch_res)
    axis = np.array([0.08032594, 0.93313411, -0.35044041])
    aq_craft.write_radiometer_ck(27.60, axis, 0., 'X')
    axis = np.array([-0.12187662, 0.90798238, 0.40089161])
    aq_craft.write_radiometer_ck(37.02, axis, 0., 'X')
    axis = np.array([0.05645089, 0.98652661, -0.15355307])
    aq_craft.write_radiometer_ck(40.88, axis, 0., 'X')

    return aq_craft


def track_viewer_2D(craft, start_epoch, end_epoch, resolution): 
    """ Plots spacecraft and radiometer look tracks on a 2D map

        :param craft: A spacecraft obect 
        :param start_epoch: Starting epoch in seconds 
        :param end_epoch: Ending epoch in seconds 
        :param resolution: Time resolution in seconds 
    
    """ 
    track_dict = craft.make_track_grid(start_epoch, end_epoch, resolution)
    lons = track_dict['lon']
    lats = track_dict['lat']
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.plot(lons, lats, transform=ccrs.PlateCarree(), linestyle='none', marker='.', color='k', 
            label='Spacecraft tracks', ms=2)
    obs_dict = craft.make_obs_grid(start_epoch, end_epoch, resolution)
    lons = obs_dict['lon']
    lats = obs_dict['lat']
    shape = np.shape(lons)
    for i in range(shape[0]): 
        ax.plot(lons[i, :], lats[i, :], transform=ccrs.PlateCarree(), linestyle='none', marker='.',
                label='Radiometer {}'.format(i), ms=2)
    ax.set_extent([-180, 180, -90, 90])


def track_viewer_animated(craft, start_epoch, end_epoch, resolution, saveplots=True): 
    """ Animated plots spacecraft and radiometer look tracks on a 2D map 
        overtime with a 3D plot of observing geometry

        :param craft: A spacecraft obect 
        :param start_epoch: Starting epoch in seconds 
        :param end_epoch: Ending epoch in seconds 
        :param resolution: Time resolution in seconds 

    """ 
    
    fig = plt.figure(figsize=(20, 10)) 
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
    ax2.coastlines()

    # Plot Earth 
    u = np.linspace(0, 2 * np.pi, 500)
    v = np.linspace(0, np.pi, 500)
    rad = craft.earth_mean_radius / 1e3
    x = rad * np.outer(np.cos(u), np.sin(v))
    y = rad * np.outer(np.sin(u), np.sin(v))
    z = rad * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='black', alpha=0.1)

    epochs = np.arange(start_epoch, end_epoch, resolution)
    obs_dict = craft.make_obs_grid(start_epoch, end_epoch, resolution)
    epochs = obs_dict['epoch']

    for t in range(len(epochs)):
        pos, times = spice.spkpos('{}'.format(craft.naif_id), epochs[t], 'ITRF93', 'NONE', '399')
        ax.scatter(pos[0], pos[1], pos[2], color='blue')
        pos2, times = spice.spkpos('399', epochs[t], 'SPACECRAFT_{}'.format(craft.sc_number), 'NONE', 
                                   '{}'.format(craft.naif_id))
        mat = spice.pxform('SPACECRAFT_{}'.format(craft.sc_number), 'ITRF93', epochs[t])
        pos2 = mat @ pos2
        lon, lat, tall = spice.recgeo(pos, 6378.137, 1 / 298.257223563)  # WGS84 ellipsoid
        height = tall
        pos2_norm = pos2 * (height / np.linalg.norm(pos))
        ax.quiver(pos[0], pos[1], pos[2], pos2_norm[0], pos2_norm[1], pos2_norm[2], color='red')
        for i in range(craft.rad_number): 
            try:
                spoint, trgepc, srfvec = spice.sincpt('ELLIPSOID', 'EARTH', epochs[t], 'ITRF93', 'NONE', 
                                                      '{}'.format(craft.naif_id), 
                                                      'RADIOMETER_{}'.format(i), np.array([1, 0, 0]))
                ax.scatter(spoint[0], spoint[1], spoint[2], color='green')
                ll_arr2 = spice.reclat(spoint)
                # print("Lon/Lat SPOINT: %f, %f" % (np.degrees(ll_arr2[1]), np.degrees(ll_arr2[2])))
                ax.quiver(spoint[0], spoint[1], spoint[2], -srfvec[0], -srfvec[1], -srfvec[2], color='purple')

            except spice.utils.exceptions.NotFoundError: 
                ll_arr2 = np.array([1, 0, 0])

            enctime = spice.sce2c(craft.naif_id, epochs[t])
            cmat, clkout = spice.ckgp(int(craft.naif_id * 1e3 - 1 - i), enctime, 0, 'ITRF93')
            cpoint = cmat.T @ (np.array([1, 0, 0]))
            cpoint_norm = 2 * cpoint * (height / np.linalg.norm(cpoint))
            ax.quiver(pos[0], pos[1], pos[2], cpoint_norm[0], cpoint_norm[1], cpoint_norm[2], color='g')
            cpoint = cmat.T @ (np.array([0, 1, 0]))
            cpoint_norm = 2 * cpoint * (height / np.linalg.norm(cpoint))
            ax.quiver(pos[0], pos[1], pos[2], cpoint_norm[0], cpoint_norm[1], cpoint_norm[2], color='m')
            cpoint = cmat.T @ (np.array([0, 0, 1]))
            cpoint_norm = 2 * cpoint * (height / np.linalg.norm(cpoint))
            ax.quiver(pos[0], pos[1], pos[2], cpoint_norm[0], cpoint_norm[1], cpoint_norm[2], color='y')

            ll_arr = spice.reclat(pos)
            ax2.plot(np.degrees(ll_arr[1]), np.degrees(ll_arr[2]), transform=ccrs.PlateCarree(), linestyle='none', marker='.', color='blue')
            ax2.plot(np.degrees(ll_arr2[1]), np.degrees(ll_arr2[2]), transform=ccrs.PlateCarree(), linestyle='none', marker='.', color='green')
            ax2.set_extent([-180, 180, -90, 90])
            # print("Lon/Lat SUB: %f, %f" % (np.degrees(ll_arr[1]), np.degrees(ll_arr[2])))

        if t == 0:
            # plt.show(block=False)
            if os.path.exists('animated_tracks'): 
                os.rmdir('animated_tracks')
            os.mkdir('animated_tracks')
        else:
            # plt.draw()
            pass
        plt.tight_layout()
        if saveplots: plt.savefig('animated_tracks/chart_{}'.format(t), dpi=90)
        # plt.pause(0.2)


def revisit_time(craft, start_epoch, end_epoch, epoch_res, 
                 grid_mode='linear', grid_res=0.25, grid_stop=1, 
                 plots=False, parallel=False, **kwargs):
    
    """ Computes revisit counts for a spacecraft over a given time range 

        :param craft: Spacecraft object
        :param start_epoch: Starting epoch in seconds since J2000
        :param end_epoch: Ending epoch in seconds since J2000
        :param epoch_res: Time sample resolution in seconds 
        :param grid_mode: Options for grid mode are: 

                            * 'linear' (default): Uniform lat/lon grid resolution set by grid_res
                            
                            * 'ease': Scaled EASE grid with the finest grid resolution set by grid_res
                            
                            * 'cosine': Uniform longitude spacing (grid_res) and a cosine-graded latitude spacing ranging from grid_res to grid_stop. Not exact! 

        :param grid_res: See above, in degrees
        :param grid_stop: See above, in degrees. Only used for grid_mode='cosine'
        :param plots: If True, plot results as pcolormesh
        :param parallel: If true, uses craft.make_parallel_grid (and associated 
                         kwargs), to generate observations in parallel
    """ 

    if parallel: 
        obs_dict = craft.make_parallel_grid(start_epoch, end_epoch, epoch_res, **kwargs)
    else: 
        obs_dict = craft.make_obs_grid(start_epoch, end_epoch, epoch_res, **kwargs)
    obs_dict = {key: value.ravel() for key, value in obs_dict.items()}
    
    if grid_mode == 'linear':
        lat_bins = np.arange(-90, 90 + grid_res, grid_res)
        lon_bins = np.arange(-180, 180 + grid_res, grid_res)
    elif grid_mode == 'ease': 
        dim = (4872, 11568)
        land_lat = np.fromfile(os.path.join(cache_path, 'landmask', 'EASE2_M03km.lats.11568x4872x1.double'), dtype=np.float64)
        lat_bins = land_lat.reshape(dim)[::-1, 0]
        land_lon = np.fromfile(os.path.join(cache_path, 'landmask', 'EASE2_M03km.lons.11568x4872x1.double'), dtype=np.float64)
        lon_bins = land_lon.reshape(dim)[0]
        for i in range(20):
            lat_bins = np.insert(lat_bins, 0, lat_bins[0] - 0.25548359)
            lat_bins = np.append(lat_bins, lat_bins[-1] + 0.25548359)
        lat_bins[0] = -90
        lat_bins[-1] = 90
        sample = int(np.round(grid_res / (360 / dim[-1])))
        lat_bins = lat_bins[::sample]
        lon_bins = lon_bins[::sample]
        lat_bins[0] = -90
        lat_bins[-1] = 90
        lon_bins[0] = -180
        lon_bins[-1] = 180

    elif grid_mode == 'cosine':
        lat_bins = _cosine_grid(0, 90, grid_res, grid_stop)
        lat_bins = np.concatenate([np.flip(-lat_bins[1:]), lat_bins])
        lon_bins = np.arange(-180, 180 + grid_res, grid_res)
    else:
        raise ValueError("Grid mode must be 'linear', 'ease', or 'cosine'")
    
    llat, llon = np.meshgrid(lat_bins, lon_bins)
    grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
    ep_mod = (obs_dict['epoch'] - obs_dict['epoch'][0]) % craft.orbital_period
    dexs = np.where(np.diff(ep_mod) < 0)  # Find zero points, or new orbits
    dexs = np.concatenate([np.zeros(1, dtype=int), dexs[0], [len(ep_mod)]])
    
    # Old method, clean but double-counts 
    # for i in range(1, len(dexs)): 
    #     lats = obs_dict['lat'][dexs[i - 1]:dexs[i]]
    #     lons = obs_dict['lon'][dexs[i - 1]:dexs[i]]
    #     pack = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins])
    #     cg = pack[0].astype(bool).astype(int)
    #     grid += cg

    # New method, using pandas 
    nset = pd.DataFrame.from_dict(obs_dict).iloc[:, :3]
    nset['epoch'] = nset['epoch'] - nset['epoch'][0]
    # Binning 
    lat_dex = np.digitize(nset['lat'], lat_bins)
    lon_dex = np.digitize(nset['lon'], lon_bins)
    nset['lat'] = lat_bins[lat_dex - 1]
    nset['lon'] = lon_bins[lon_dex - 1]

    # Time filtering critera
    # Removes values in bins within less than half 
    # of an orbital period of each other
    # prepend ensures first value is kept
    def func(x):
        z = x['epoch']
        if len(z) > 1:
            return z[np.diff(z, prepend=-1e10) > (craft.orbital_period / 2)]
        else:
            return z

    nsgb = nset.groupby(['lat', 'lon'])
    nsout = nsgb.apply(func)  # Filters out lat and lon
    nsout = nsout.astype(bool).astype(int)
    nsout = nsout.groupby(level=[0, 1]).sum()

    lat_vals = nsout.index.get_level_values(0) 
    lon_vals = nsout.index.get_level_values(1)
    grid, *_ = np.histogram2d(lat_vals, lon_vals, bins=[lat_bins, lon_bins], weights=nsout.values)

    if plots: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.coastlines()
        im = ax.pcolormesh(lon_bins, lat_bins, grid, shading='auto', alpha=0.8, cmap='turbo')
        fig.colorbar(im, fraction=0.03, pad=0.04, label='Revisits')
    return grid, lon_bins, lat_bins, obs_dict


def angle_conversion(h, in_angle, in_angle_type='incidence', R=6371): 
    """ Utility to convert antenna look angle to Earth surface incidence angle 
        or vice versa. Assumes either angle is < 90 deg

        :param h: Height above surface in km 
        :param in_angle: Input angle in degrees
        :param in_angle_type: Either surface 'incidence' or antenna 'look'
        :param R: Radius of body in km (defaults to Earth, 6371)
    """ 
    in_angle = np.radians(in_angle)
    if in_angle_type == 'incidence': 
        out = np.arcsin(R / (R + h) * np.sin(np.pi - in_angle))
    elif in_angle_type == 'look': 
        out = np.arcsin((R + h) / R * np.sin(in_angle))
    else: 
        raise ValueError('in_angle_type must be either "incidence" or "look"')

    return np.degrees(out)


def _cosine_grid(start, stop, dx_start, dx_stop): 
    """ Ad hoc grid spacing method which, when given a start and stop point, 
        generates a sequence of numbers x with properties diff(x[:2]) == dx_start
        and diff(x[-2:]) == dx_stop with a cosine taper. 
        This is an iterative, inexact method which 
        is useful for generating scaled latitude grids for visualization
    """ 

    if (dx_start > 25) or (dx_stop > 25): 
        raise ValueError('Grid too coarse for this method')
    elif dx_start > dx_stop: 
        raise ValueError('dx_start must be smaller than dx_stop')

    a = dx_stop 
    b = dx_stop - dx_start
    n = 3
    x = np.linspace(0, n, n + 1)
    y = a - b * np.cos(np.pi / 2 * x / (n + 1))
    csy = np.cumsum(y)
    y = start + csy / csy[-1] * (stop - start)
    y = np.insert(y, 0, start)
    dy = np.diff(y)

    start_check = abs(dy[0] - dx_start)
    stop_check = abs(dy[-1] - dx_stop)

    going = False
    while True: 
        n += 1
        if n == 1e4: break
        x = np.linspace(0, n, n + 1)
        y = a - b * np.cos(np.pi / 2 * x / (n + 1))
        csy = np.cumsum(y)
        y = start + csy / csy[-1] * (stop - start)
        y = np.insert(y, 0, start)
        dy = np.diff(y)        
        strc = abs(dy[0] - dx_start)
        stpc = abs(dy[-1] - dx_stop)

        if (strc < start_check) and (stpc < stop_check): 
            start_check = strc 
            stop_check = stpc 
            going = True
        elif going:
            break

    x = np.linspace(0, n, n + 1)
    y = a - b * np.cos(np.pi / 2 * x / (n + 1))
    csy = np.cumsum(y)
    y = start + csy / csy[-1] * (stop - start)
    y = np.insert(y, 0, start)

    return y


def strings_to_epochs(*args): 
    """ Converts strings to epochs 
        useable by spacecraft generator objects.

        Example string format - '2020 MAY 24 9:00 UTC'
    """ 

    spice.furnsh(os.path.join(cache_path, 'spice/latest_leapseconds.tls'))
    output = tuple(spice.str2et(arg) for arg in args)
    spice.unload(os.path.join(cache_path, 'spice/latest_leapseconds.tls'))
    return output


def merge_obs_dict(*args): 
    """ Merge any number of observing dictionaries into one 
    """ 
    obs_list = list(args)
    keys = args[0].keys()
    obs_list = [d for d in obs_list if d]  # Filter out empty dicts
    obs_dict = {k: np.concatenate([d.get(k) for d in obs_list], axis=0) for k in keys}
    return obs_dict










    
