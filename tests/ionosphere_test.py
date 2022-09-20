import unittest
from foam.ionosphere import ionosphere 


class TestIonosphere(unittest.TestCase):

    def test_constructor(self): 
        def_ion = ionosphere()
        online_ion = ionosphere(datetime=['2015-01-01'], online=True)
        iri_ion = ionosphere(datetime=['2015-01-01'], IRI=True)


if __name__ == '__main__': 
    unittest.main()
