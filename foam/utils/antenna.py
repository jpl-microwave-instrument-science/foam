import numpy as np 
import scipy.constants as spc 
import scipy.special as sps 
import scipy.integrate as sint 


def read_antenna_pattern(file): 
    """ Reads a simple antenna or feed pattern (i.e. one pattern, no polarization dependence)
        from a text file and returns in a format for input to a FOAM spacecraft. 

        The file format is a csv style with dimensions of Nx3
        Column 1: Elevation/Theta (slowest varying index)
        Column 2: Azimuth/Phi (fasted varying index)
        Column 3: Normalized linear power pattern (Integral = 1)
    """ 

    ap = np.loadtxt(file, delimiter=',', skiprows=1)
    theta = ap[:, 0]
    phi = ap[:, 1]
    pattern = ap[:, 2]
    theta_grid = np.unique(theta)
    phi_grid = np.unique(phi)
    pattern_grid = pattern.reshape(len(theta_grid), len(phi_grid))

    ap_dict = {'theta': theta_grid, 'phi': phi_grid, 'pattern': pattern_grid}
    return ap_dict


def make_uniform_reflector(diameter, frequency, file):  
    """ Generates an antenna pattern assuming uniform illumination 
        of a circular reflector with an arbitrary diameter 

        :param diameter: Reflector diameter in meters
        :param frequency: Frequency in MHz 
        :param file: Output text file to write pattern to
    """ 

    wave = spc.c / (frequency * 1e6)  # In meters
    k = 2 * np.pi / wave
    theta_grid = np.radians(np.arange(0, 90, 0.2))
    azi_grid = np.radians(np.arange(-180, 180, 0.2))
    pattern_grid = np.zeros((len(theta_grid), len(azi_grid)))
    for i, thet in enumerate(theta_grid): 
        out = sint.quad(lambda r, th: sps.j0(k * r * np.sin(th)) * r, 0, diameter, args=(thet))
        pattern_grid[i, :] = 2 * np.pi * out[0]  # This is the E-field pattern

    dphi = np.mean(np.gradient(azi_grid))
    dth = np.mean(np.gradient(theta_grid))
    theta_pattern = np.sum(pattern_grid**2, axis=1)
    total_power = np.trapz(theta_pattern * np.sin(theta_grid) * dth * dphi)
    # Power pattern is sufficient for phase-incoherent signals 
    power_pattern = pattern_grid**2 / total_power

    am, tm = np.meshgrid(azi_grid, theta_grid)
    save_pattern = np.vstack([np.degrees(tm).flatten(), np.degrees(am).flatten(), power_pattern.flatten()]).T 
    np.savetxt(file, save_pattern, header='theta, phi, linear pattern', delimiter=',')





        
