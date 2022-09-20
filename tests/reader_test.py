import unittest
import foam.utils.reader as rdr 


class TestReader(unittest.TestCase): 

    def setUp(self): 
        # Base class 
        self.reader = rdr.Reader(['2015-01-01'])

        # Ocean 
        self.ghrsstreader = rdr.GHRSSTReader(['2015-01-01', '2015-01-05'], online=True)
        self.smapreader = rdr.SMAPSalinityReader(['2018-01-01', '2018-01-05'], online=True)
        self.aquariusreader = rdr.AquariusSalinityReader(['2015-01-01', '2015-01-05'], online=True)
        self.oisssreader = rdr.OISSSReader(['2015-01-01', '2015-01-05'], online=True)
        self.hycomreader = rdr.HYCOMReader(['2015-01-01', '2015-01-05'], online=True)

        # Atmosphere 
        self.ncep2dreader = rdr.NCEPReader(['2015-01-01', '2015-01-03'], online=True, dimension='2D')
        self.ncep3dreader = rdr.NCEPReader(['2015-01-01', '2015-01-03'], online=True, dimension='3D')
        self.merra2dreader = rdr.MERRAReader(['2015-01-01', '2015-01-03'], online=True, dimension='2D')
        self.merra2dinstantreader = rdr.MERRAReader(['2015-01-01', '2015-01-03'], online=True, dimension='2D', instant=True)
        self.merra3dreader = rdr.MERRAReader(['2015-01-01', '2015-01-03'], online=True, dimension='3D')
        
        
        # Ionosphere 
        self.ionexreader = rdr.IONEXReader(['2015-01-01', '2015-01-05'], online=True)

    def test_read_ionosphere(self): 
        print('Reading IONEX file')
        self.ionexreader.read()

    def test_read_ocean(self): 
        print('Reading GHRSST SST')
        self.ghrsstreader.read()
        print('Reading SMAP SSS')
        self.smapreader.read()
        print('Reading Aquarius SSS')
        self.aquariusreader.read()
        print('Reading OISSS SSS')
        self.oisssreader.read()
        print('Reading HYCOM SST and SSS')
        self.hycomreader.read()

    def test_read_atmosphere(self): 
        print('Reading NCEP 2D')
        self.ncep2dreader.read()
        print('Reading NCEP 3D')
        self.ncep3dreader.read() 
        print('Reading MERRA 2D')
        self.merra2dreader.read()
        print('Reading MERRA 2D (instantaneous)')
        self.merra2dinstantreader.read()
        print('Reading MERRA 3D')
        self.merra3dreader.read()

 
if __name__ == '__main__':
    unittest.main()
