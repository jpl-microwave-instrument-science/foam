import unittest
import numpy as np
import datetime as dt 
from foam.ocean import ocean, fastem


class TestOcean(unittest.TestCase): 

    def setUp(self): 
        self.simple_ocean = ocean(mode='simple', online=False, verbose=True)
        self.full_ocean = ocean(datetime=dt.datetime(2020, 1, 1, 12, 0, 0), mode='full', online=True, verbose=True)

    def test_emis(self): 
        # Single value input test 
        frequency = np.array([1e3, 10e3, 20e3])
        lat = np.array([0, 10, 20])
        lon = np.array([20, 20, -20])
        uwind = np.array([0, 3, 4])
        vwind = np.array([10, 0, 25])
        theta = np.array([20, 30, 40])
        phi = np.array([30, 60, 90])
        emis = self.simple_ocean.get_ocean_emissivity(frequency, lat, lon, uwind, vwind, theta, phi)
        self.assertEqual(np.shape(emis), (4, 3))
        emis = self.full_ocean.get_ocean_emissivity(frequency, lat, lon, uwind, vwind, theta, phi)
        self.assertEqual(np.shape(emis), (4, 3))

    def test_TB(self): 
        # Single value input test 
        frequency = np.array([1e3, 10e3, 20e3, 30e3])
        lat = np.array([0, 10, 20])
        lon = np.array([20, 20, -20])
        uwind = np.array([0, 3, 4])
        vwind = np.array([10, 0, 25])
        theta = np.array([20, 30, 40])
        phi = np.array([30, 60, 90])
        TB, emis = self.simple_ocean.get_ocean_TB(frequency, lat, lon, uwind, vwind, theta, phi)
        self.assertEqual(np.shape(TB), (4, 4, 3))
        TB, emis = self.full_ocean.get_ocean_TB(frequency, lat, lon, uwind, vwind, theta, phi)
        self.assertEqual(np.shape(TB), (4, 4, 3))


class TestFastem(TestOcean): 

    def setUp(self): 
        self.simple_ocean = fastem(mode='simple', online=False, verbose=True)
        self.full_ocean = fastem(mode='full', online=True, verbose=True)


if __name__ == '__main__':
    unittest.main()
