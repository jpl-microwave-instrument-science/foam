import unittest
import numpy as np 
import datetime as dt
from foam.atmosphere import atmosphere


class TestISA(unittest.TestCase): 

    def setUp(self):
        # Tests initialization of the ocean classes in a few modes 
        self.simple_atmosphere = atmosphere(mode='simple', online=False, verbose=True)
        self.full_atmosphere = atmosphere(datetime=dt.datetime(2020, 1, 1, 12, 0, 0), mode='full', online=True, verbose=True) 

    def test_isa_profiles(self): 
        T0 = np.array([200, 250, 300])
        P0 = np.array([1e5, 0.9e5, 1.1e5])
        pr_wv = np.array([0.2, 0.3, 0.4])
        pr_lw = np.array([1e-3, 2e-3, 3e-3])
        T, P, dens, wv, lw, z = self.simple_atmosphere.isa_profiles(T0, P0, pr_wv, pr_lw, res=1)
        s = len(z)
        self.assertEqual(np.shape(T), (s, 3))
        self.assertEqual(np.shape(P), (s, 3))
        self.assertEqual(np.shape(dens), (s, 3))
        self.assertEqual(np.shape(wv), (s, 3))
        self.assertEqual(np.shape(lw), (s, 3))
        T0 = np.array([200, 250, 300, 200, 250, 300]).reshape(3, 2)
        P0 = np.array([1e5, 0.9e5, 1.1e5, 1e5, 0.9e5, 1.1e5]).reshape(3, 2)
        pr_wv = np.array([0.2, 0.3, 0.4, 0.2, 0.3, 0.4]).reshape(3, 2)
        pr_lw = np.array([1e-3, 2e-3, 3e-3, 1e-3, 2e-3, 3e-3]).reshape(3, 2)
        T, P, dens, wv, lw, z = self.full_atmosphere.isa_profiles(T0, P0, pr_wv, pr_lw, res=1)
        s = len(z)
        self.assertEqual(np.shape(T), (s, 3, 2))
        self.assertEqual(np.shape(P), (s, 3, 2))
        self.assertEqual(np.shape(dens), (s, 3, 2))
        self.assertEqual(np.shape(wv), (s, 3, 2))
        self.assertEqual(np.shape(lw), (s, 3, 2))

    def test_mrtm(self): 
        T0 = np.array([200, 250, 300])
        P0 = np.array([1e5, 0.9e5, 1.1e5])
        pr_wv = np.array([0.2, 0.3, 0.4])
        pr_lw = np.array([1e-3, 2e-3, 3e-3])
        T, P, dens, wv, lw, z = self.simple_atmosphere.isa_profiles(T0, P0, pr_wv, pr_lw, res=1)
        freq = np.array([1e3])
        tbup, tbdn, transup, transdn, wup, wdn, tau = self.simple_atmosphere.mrtm(freq, P, T, wv, lw, 30)
        self.assertEqual(len(tbup), 3)
        self.assertEqual(len(tbdn), 3)
        self.assertEqual(np.shape(transup), (len(z) - 1, 3))
        self.assertEqual(np.shape(transdn), (len(z) - 1, 3))
        self.assertEqual(np.shape(wup), (len(z) - 1, 3))
        self.assertEqual(np.shape(wdn), (len(z) - 1, 3))
        self.assertEqual(np.shape(tau), (len(z) - 1, 3))

        T, P, dens, wv, lw, z = self.full_atmosphere.isa_profiles(T0, P0, pr_wv, pr_lw, res=1)
        freq = np.array([1e3])
        tbup, tbdn, transup, transdn, wup, wdn, tau = self.full_atmosphere.mrtm(freq, P, T, wv, lw, 30)
        self.assertEqual(len(tbup), 3)
        self.assertEqual(len(tbdn), 3)
        self.assertEqual(np.shape(transup), (len(z) - 1, 3))
        self.assertEqual(np.shape(transdn), (len(z) - 1, 3))
        self.assertEqual(np.shape(wup), (len(z) - 1, 3))
        self.assertEqual(np.shape(wdn), (len(z) - 1, 3))
        self.assertEqual(np.shape(tau), (len(z) - 1, 3))

    def test_atmos_tb(self): 
        freq = np.array([1e3, 20e3, 30e3])
        lat = np.array([10, 20])
        lon = np.array([30, 40])

        tbup, tbdn, transup, transdn = self.simple_atmosphere.get_atmosphere_tb(freq, lat, lon, angle=0)
        self.assertEqual(len(tbup), 3)
        self.assertEqual(len(tbdn), 3)

        tbup, tbdn, transup, transdn = self.full_atmosphere.get_atmosphere_tb(freq, lat, lon, angle=0)
        self.assertEqual(len(tbup), 3)
        self.assertEqual(len(tbdn), 3)











