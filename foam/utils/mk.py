import os 
import spiceypy as spice 
from .config import cache_path

spice_path = os.path.join(cache_path, 'spice')


def manual_furnish(): 
    spice.furnsh(os.path.join(spice_path, 'de430.bsp'))
    spice.furnsh(os.path.join(spice_path, 'earth_assoc_itrf93.tf'))
    spice.furnsh(os.path.join(spice_path, 'earth_latest_high_prec.bpc'))
    spice.furnsh(os.path.join(spice_path, 'latest_leapseconds.tls'))
    spice.furnsh(os.path.join(spice_path, 'moon_080317.tf'))
    spice.furnsh(os.path.join(spice_path, 'moon_pa_de421_1900-2050.bpc'))
    spice.furnsh(os.path.join(spice_path, 'pck00010.tpc'))
    spice.furnsh(os.path.join(spice_path, 'geophysical.ker'))


def manual_unload(): 
    spice.unload(spice_path + 'de430.bsp')
    spice.unload(spice_path + 'earth_assoc_itrf93.tf')
    spice.unload(spice_path + 'earth_latest_high_prec.bpc')
    spice.unload(spice_path + 'latest_leapseconds.tls')
    spice.unload(spice_path + 'moon_080317.tf')
    spice.unload(spice_path + 'moon_pa_de421_1900-2050.bpc')
    spice.unload(spice_path + 'pck00010.tpc')
    spice.unload(spice_path + 'geophysical.ker')


def form_foam_mk(): 
    foam_mk = open(spice_path + 'foam.mk', 'w')
    foam_mk.write("""KPL/MK
                    \\begindata
                    KERNELS_TO_LOAD = ( '{0}de430.bsp',
                                        '{0}earth_assoc_itrf93.tf',
                                        '{0}earth_latest_high_prec.bpc',
                                        '{0}latest_leapseconds.tls'
                                        '{0}moon_080317.tf',
                                        '{0}moon_pa_de421_1900-2050.bpc',
                                        '{0}pck00010.tpc'
                                        '{0}geophysical.ker')""".format(spice_path))
    foam_mk.close()
    spice.furnsh(spice_path + 'foam.mk')


def form_predict_mk(): 
    predict_mk = open(spice_path + 'predict.mk', 'w')
    predict_mk.write("""KPL/MK
                    \\begindata
                    KERNELS_TO_LOAD = ( '{0}de430.bsp',
                                        '{0}earth_assoc_itrf93.tf',
                                        '{0}earth_latest_high_prec.bpc',
                                        '{0}latest_leapseconds.tls'
                                        '{0}moon_080317.tf',
                                        '{0}moon_pa_de421_1900-2050.bpc',
                                        '{0}pck00010.tpc'
                                        '{0}geophysical.ker'
                                        '{0}current_spacecraft.bsp',
                                        '{0}current_spacecraft.sclk',
                                        '{0}current_spacecraft.tf')""".format(spice_path))
    predict_mk.close()
    spice.furnsh(spice_path + 'predict.mk')
