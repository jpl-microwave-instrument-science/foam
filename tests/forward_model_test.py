def smap_surface_TB(mode='direct', level_2='JPL'):
    """ Compares SMAP surface TBs with simulated observations 

        :param mode: Specifies method of comparison
                    -   'direct' performs a direct calculation of surface TB 
                        from ancillary data available in level 2 files  
                    -   'simulate' compares level 1 brightness temperatures 
                        with simulated SMAP-like spacecraft observations
        :param level_2: Selects level 2 file source. Options are 
                        'JPL' or 'RSS' 
    """ 

    freq = np.array([1.413e3])

    if mode == 'direct': 
        ocean = o.ocean(mode='simple', use_wind_interpolators=True, online=False, verbose=True)
        if level_2 == 'JPL': 
            smap_file = 'smap_data/SMAP_L2B_SSS_29869_20200903T204716_R17000_V5.0.h5'
            smap = h5py.File(smap_file, 'r')
            qf = smap['quality_flag'][:]
            mask = qf == 0 
            tb_lat = smap['lat'][mask]
            tb_lon = smap['lon'][mask]
            tb_v = smap['tb_v_fore'][mask]
            tb_h = smap['tb_h_fore'][mask]
            anc_sst = smap['anc_sst'][mask]
            smap_sss = smap['smap_sss'][mask]
            azi_fore = smap['azi_fore'][mask]
            inc_fore = smap['inc_fore'][mask]
            anc_dir = smap['anc_dir'][mask]
            anc_spd = smap['smap_spd'][mask]
            uwind = anc_spd * np.cos(np.radians(anc_dir))
            vwind = anc_spd * np.sin(np.radians(anc_dir))
            smap.close()
        elif level_2 == 'RSS': 
            smap_file = 'smap_data/RSS_SMAP_SSS_L2C_r29869_20200903T204628_2020247_FNL_V04.0.nc'
            smap = ncdf.Dataset(smap_file, 'r')
            qc = smap['iqc_flag'][:, :, 0]  # Only grabbing fore looks
            mask = qc == 0
            tb_lat = smap['cellat'][:, :, 0][mask]
            tb_lon = smap['cellon'][:, :, 0][mask]
            tb_v = smap['tb_sur'][:, :, 0, 0][mask]
            tb_h = smap['tb_sur'][:, :, 0, 1][mask]
            tb_v_spec = smap['tb_sur0'][:, :, 0, 0][mask]
            tb_h_spec = smap['tb_sur0'][:, :, 0, 1][mask]
            anc_sst = smap['surtep'][:][mask]
            smap_sss = smap['sss_smap'][:, :, 0][mask]
            azi_fore = smap['eaa'][:, :, 0][mask]
            inc_fore = smap['eia'][:, :, 0][mask]
            anc_dir = smap['windir'][:][mask]
            anc_spd = smap['winspd'][:][mask]
            uwind = anc_spd * np.cos(np.radians(anc_dir))
            vwind = anc_spd * np.sin(np.radians(anc_dir))
            smap.close()
        else: 
            raise ValueError('Wrong source')

        emis = ocean.get_ocean_emissivity(freq, anc_sst, smap_sss, inc_fore, azi_fore, uwind, vwind)
        sim_tb_v = emis[0] * anc_sst
        sim_tb_h = emis[1] * anc_sst

        # Histogram of SMAP brightness temperatures 
        hbins = np.linspace(65, 85, 100)
        vbins = np.linspace(105, 125, 100)
        fig, axs = plt.subplots(2, 1)
        axs[0].hist(tb_v, bins=vbins, label='SMAP', density=True, color='tab:blue')
        axs[0].hist(sim_tb_v, bins=vbins, label='Simulated', alpha=0.5, density=True, color='tab:red')
        axs[0].set_xlabel('TBV (K)')
        axs[0].set_ylabel('Normalized Counts')
        axs[0].legend()

        axs[1].hist(tb_h, bins=hbins, label='SMAP', density=True, color='tab:blue')
        axs[1].hist(sim_tb_h, bins=hbins, label='Simulated', alpha=0.5, density=True, color='tab:red')
        axs[1].set_xlabel('TBH (K)')
        axs[1].set_ylabel('Normalized Counts')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig('compare_salinity_histogram.pdf', dpi=300, transparent=True)

        # Difference histograms
        fig, axs = plt.subplots(2, 1)
        bins = np.linspace(-5, 5, 100)
        axs[0].hist(sim_tb_v - tb_v, bins=bins, density=True, color='tab:blue')
        axs[0].set_xlabel('Model - SMAP TBV(K)')
        axs[0].set_ylabel('Normalized Counts')

        axs[1].hist(sim_tb_h - tb_h, bins=bins, density=True, color='tab:blue')
        axs[1].set_xlabel('Model - SMAP TBH (K)')
        axs[1].set_ylabel('Normalized Counts')
        plt.tight_layout()
        plt.savefig('compare_salinity_histogram_diff.pdf', dpi=300, transparent=True)

        # Wind dependence 

        wind = np.sqrt(uwind**2 + vwind**2)
        fig, axs = plt.subplots(2, 1)
        axs[0].scatter(wind, tb_v, c=tb_lat, marker='.', cmap='Blues', label='SMAP')
        axs[0].scatter(wind, sim_tb_v, c=tb_lat, marker='.', cmap='Reds', label='Simulated')
        axs[0].set_xlabel('Wind Speed')
        axs[0].set_ylabel('TBV')
        legend_elements = [Line2D([0], [0], marker='.', color='b', label='SMAP', linestyle='none'), Line2D([0], [0], marker='.', color='r', label='Simulated', linestyle='none')]
        axs[0].legend(handles=legend_elements)

        axs[1].scatter(wind, tb_h, c=tb_lat, marker='.', cmap='Blues', label='SMAP')
        axs[1].scatter(wind, sim_tb_h, c=tb_lat, marker='.', cmap='Reds', label='Simulated')
        axs[1].set_xlabel('Wind Speed')
        axs[1].set_ylabel('TBH')
        axs[1].legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig('compare_wind_dependence.pdf', dpi=300, transparent=True)

        # Angle dependence 
        tmask = (anc_sst < (273 + 22)) & (anc_sst > (273 + 18)) & (smap_sss > (34)) & (smap_sss < (356))

        plt.figure()
        plt.scatter(abs(azi_fore[tmask] - anc_dir[tmask]), tb_v[tmask], c=wind[tmask], marker='.', cmap='Blues', label='SMAP')
        plt.scatter(abs(azi_fore[tmask] - anc_dir[tmask]), sim_tb_v[tmask], c=wind[tmask], marker='x', cmap='Reds', label='Simulated')
        plt.xlabel('Phi')
        plt.ylabel('TBV')
        plt.legend()

        plt.figure()
        plt.scatter(abs(azi_fore[tmask] - anc_dir[tmask]), tb_h[tmask], c=wind[tmask], marker='.', cmap='Blues', label='SMAP')
        plt.scatter(abs(azi_fore[tmask] - anc_dir[tmask]), sim_tb_h[tmask], c=wind[tmask], marker='x', cmap='Reds', label='Simulated')
        plt.xlabel('Phi')
        plt.ylabel('TBH')
        plt.legend()

    elif mode == 'simulate': 
        ocean = o.ocean(mode='simple', use_wind_interpolators=True, online=True, time=dt.datetime(2020, 9, 3, 12, 0, 0), verbose=True)
        smap_file = 'smap_data/SMAP_L1B_TB_29869_A_20200903T204518_R17000_001.h5'
        smap = h5py.File(smap_file, 'r')
        sc_data = smap['Spacecraft_Data']
        scan_time = np.mean(np.gradient(sc_data['antenna_scan_time'][:]))  # Time per one full scan in seconds
        rpm = 60 / scan_time
        fpscan = np.mean(sc_data['footprints_per_scan'][:])
        fpms = rpm * fpscan / 60 / 1e3
        int_time = 1 / fpms
        tb_data = smap['Brightness_Temperature']
        qfv = tb_data['tb_qual_flag_v'][:]
        qfh = tb_data['tb_qual_flag_h'][:]
        mask = (qfv == 0) & (qfh == 0)
        tb_time = tb_data['tb_time_seconds'][mask]
        start_epoch = tb_time[0]
        end_epoch = tb_time[-1]
        tb_lat = tb_data['tb_lat'][mask]
        tb_lon = tb_data['tb_lon'][mask]
        look_angle = np.mean(tb_data['antenna_look_angle'][mask])
        tb_v = tb_data['tb_v'][mask]
        tb_h = tb_data['tb_h'][mask]
        smap.close() 

        s = sc.spacecraft()
        s.write_tle_kernels(mode='file', file='smap_data/SMAP.tle', int_time=int_time, rpm=rpm, look_angle=look_angle, start_epoch=start_epoch, end_epoch=end_epoch) 

        increment = 1  # Time resolution in seconds  
        epochs, lons, lats, thetas, phis = s.make_grid(start_epoch, end_epoch, increment)
        TBV, TBH, U, V = ocean.get_ocean_TB(freq, lats, lons, theta=thetas, phi=phis)

        # Plots of brightness temperatures at several samples 
        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 6))
        axs[0].coastlines()
        axs[0].scatter(tb_lon[::int(increment * 1e3 / int_time)], tb_lat[::int(increment * 1e3 / int_time)], transform=ccrs.PlateCarree(), s=10, c=tb_v[::int(increment * 1e3 / int_time)], label='SMAP', marker='o', vmin=105, vmax=125)
        axs[0].scatter(lons, lats, transform=ccrs.PlateCarree(), s=10, c=TBV.flatten(), label='Simulated', marker='d', vmin=105, vmax=125)
        axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='k', alpha=0.3, linestyle='--')
        axs[0].set_extent([-180, 180, -90, 90])
        axs[0].legend()

        axs[1].coastlines()
        axs[1].scatter(tb_lon[::int(increment * 1e3 / int_time)], tb_lat[::int(increment * 1e3 / int_time)], transform=ccrs.PlateCarree(), s=10, c=tb_h[::int(increment * 1e3 / int_time)], label='SMAP', marker='o', vmin=65, vmax=85)
        axs[1].scatter(lons, lats, transform=ccrs.PlateCarree(), s=10, c=TBH.flatten(), label='Simulated', marker='d', vmin=65, vmax=85)
        axs[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='k', alpha=0.3, linestyle='--')
        axs[1].set_extent([-180, 180, -90, 90])
        axs[1].legend()

        plt.tight_layout()
        plt.savefig('smap_tb_pointing.pdf', format='pdf', dpi=300)

        s.close()


def smap_atmosphere(): 

    smap_file = 'smap_data/RSS_SMAP_SSS_L2C_r29869_20200903T204628_2020247_FNL_V04.0.nc'
    smap = ncdf.Dataset(smap_file, 'r')
    qc = smap['iqc_flag'][:, :, 0]  # Only grabbing fore looks
    mask = qc == 0
    cellat = smap['cellat'][:, :, 0][mask]
    cellon = smap['cellon'][:, :, 0][mask]
    tran = smap['tran'][:][mask]
    tbup = smap['tbup'][:][mask]
    tbdw = smap['tbdw'][:][mask]
    tb_v = smap['tb_sur'][:, :, 0, 0][mask]
    tb_h = smap['tb_sur'][:, :, 0, 1][mask]
    tb_v_toa = smap['tb_toa_lc'][:, :, 0, 0][mask]
    tb_h_toa = smap['tb_toa_lc'][:, :, 0, 1][mask]
    tb_v_toi = smap['tb_toi'][:, :, 0, 0][mask]
    tb_h_toi = smap['tb_toi'][:, :, 0, 1][mask]
    pra = smap['pra'][:, :, 0][mask]
    pratot_exp = smap['pratot_exp'][:, :, 0][mask]
    anc_sst = smap['surtep'][:][mask]
    smap_sss = smap['sss_smap'][:, :, 0][mask]
    azi_fore = smap['eaa'][:, :, 0][mask]
    inc_fore = smap['eia'][:, :, 0][mask]
    anc_dir = smap['windir'][:][mask]
    anc_spd = smap['winspd'][:][mask]
    uwind = anc_spd * np.cos(np.radians(anc_dir))
    vwind = anc_spd * np.sin(np.radians(anc_dir))
    smap.close()

    smap_file = 'smap_data/SMAP_L1B_TB_29869_A_20200903T204518_R17000_001.h5'
    smap = h5py.File(smap_file, 'r')
    sc_data = smap['Spacecraft_Data']
    scan_time = np.mean(np.gradient(sc_data['antenna_scan_time'][:]))  # Time per one full scan in seconds
    rpm = 60 / scan_time
    fpscan = np.mean(sc_data['footprints_per_scan'][:])
    fpms = rpm * fpscan / 60 / 1e3
    int_time = 1 / fpms
    tb_data = smap['Brightness_Temperature']
    qfv = tb_data['tb_qual_flag_v'][:]
    qfh = tb_data['tb_qual_flag_h'][:]
    mask = (qfv == 0) & (qfh == 0)
    tb_time = tb_data['tb_time_seconds'][mask]
    start_epoch = tb_time[0]
    end_epoch = tb_time[-1]
    tb_lat = tb_data['tb_lat'][mask]
    tb_lon = tb_data['tb_lon'][mask]
    look_angle = np.mean(tb_data['antenna_look_angle'][mask])
    smap.close() 

    s = sc.spacecraft()
    s.write_tle_kernels(mode='file', file='smap_data/SMAP.tle', int_time=int_time, rpm=rpm, look_angle=look_angle, start_epoch=start_epoch, end_epoch=end_epoch) 

    increment = 1  # Time resolution in seconds  
    epochs, lons, lats, thetas, phis = s.make_grid(start_epoch, end_epoch, increment)
    mask = abs(lats < 80)
    lats = lats[mask]
    lons = lons[mask]
    thetas = thetas[mask]
    phis = phis[mask]
    ocean = o.ocean(mode='full', use_wind_interpolators=True, online=True, time=dt.datetime(2020, 9, 3, 12, 0, 0), verbose=True)
    freq = np.array([1.413e3])
    TBV, TBH, U, V = ocean.get_ocean_TB(freq, lats, lons, theta=thetas, phi=phis)

    # Transmittance 
    sim_tran = ocean.transtotal.flatten()
    fig, axs = plt.subplots(1)
    bins = np.linspace(0.985, 0.995, 100)
    axs.hist(tran, bins=bins, label='SMAP', density=True, color='tab:blue')
    axs.hist(sim_tran, bins=bins, label='Simulated', alpha=0.5, density=True, color='tab:red')
    axs.set_xlabel('Atmospheric Transmittance')
    axs.set_ylabel('Normalized Counts')
    axs.legend()
    plt.tight_layout()
    plt.savefig('compare_transmittance.pdf', dpi=300, transparent=True)

    # Atmospheric TB 
    sim_tbup = ocean.TBup.flatten()
    sim_tbdw = ocean.TBdn.flatten()
    fig, axs = plt.subplots(1, 2)
    bins = 100
    axs[0].hist(tbup, bins=bins, label='SMAP', density=True, color='tab:blue')
    axs[0].hist(sim_tbup, bins=bins, label='Simulated', alpha=0.5, density=True, color='tab:red')
    axs[0].set_xlabel('Upwelling Brightness Temperature')
    axs[0].set_ylabel('Normalized Counts')
    axs[0].legend()

    axs[1].hist(tbdw, bins=bins, label='SMAP', density=True, color='tab:blue')
    axs[1].hist(sim_tbdw, bins=bins, label='Simulated', alpha=0.5, density=True, color='tab:red')
    axs[1].set_xlabel('Downwelling Brightness Temperature')
    axs[1].set_ylabel('Normalized Counts')
    axs[1].legend()

    # Check ionospheric rotation 

    hbins = np.linspace(0, 0.02, 50)
    vbins = np.linspace(-0.02, 0, 50)
    sim_diff_v = TBV - ocean.TBV_toa
    sim_diff_h = TBH - ocean.TBH_toa
    fig, axs = plt.subplots(2, 1)
    axs[0].hist(tb_v_toi - tb_v_toa, bins=vbins, label='SMAP', density=True, color='tab:blue')
    axs[0].hist(sim_diff_v.flatten(), bins=vbins, label='Simulated', alpha=0.5, density=True, color='tab:red')
    axs[0].set_xlabel('TOI - TOA TBV (K)')
    axs[0].set_ylabel('Normalized Counts')
    axs[0].legend()

    axs[1].hist(tb_h_toi - tb_h_toa, bins=hbins, label='SMAP', density=True, color='tab:blue')
    axs[1].hist(sim_diff_h.flatten(), bins=hbins, label='Simulated', alpha=0.5, density=True, color='tab:red')
    axs[1].set_xlabel('TOI - TOA TBH (K)')
    axs[1].set_ylabel('Normalized Counts')
    axs[1].legend()
    plt.tight_layout()
    plt.savefig('compare_ionosphere.pdf', dpi=300, transparent=True)

    sim_far = np.degrees(ocean.far_angle.flatten())
    real_far = pratot_exp.flatten() - pra.flatten()
    fig, axs = plt.subplots(1)
    bins = 100
    axs.hist(real_far, bins=bins, label='SMAP', density=True, color='tab:blue')
    axs.hist(sim_far, bins=bins, label='Simulated', alpha=0.5, density=True, color='tab:red')
    axs.set_xlabel('Faraday Rotation')
    axs.set_ylabel('Normalized Counts')
    axs.legend()
    plt.savefig('compare_faraday_rotation.pdf', dpi=300, transparent=True)

    # Catch all 
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(ocean.prwtr.flatten(), ocean.transtotal.flatten())
    axs[0, 0].set_xlabel('prwtr')
    axs[0, 1].scatter(ocean.lwtr.flatten(), ocean.transtotal.flatten())
    axs[0, 1].set_xlabel('lwtr')
    axs[1, 0].scatter(ocean.srfprs.flatten(), ocean.transtotal.flatten())
    axs[1, 0].set_xlabel('srfprs')
    axs[1, 1].scatter(ocean.airtemp.flatten(), ocean.transtotal.flatten())
    axs[1, 1].set_xlabel('airtemp')

    return ocean