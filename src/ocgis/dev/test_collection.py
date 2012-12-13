import unittest
import numpy as np
import itertools
from ocgis.dev.collection import *


class TestCollection(unittest.TestCase):

    def test_OcgDimension(self):
        bounds = [None,
                  np.array([[0,100],[100,200]])]
        add_bounds = [True,False]
        value = np.array([50,150])
        uid = np.array([1,2])
        
        for bound,add_bounds in itertools.product(bounds,add_bounds):
            dim = OcgDimension('lid',uid,'level',value,bounds=bound)
            self.assertEqual(dim.headers,{'bnds':{0:'bnds0_level',
                                                  1:'bnds1_level'},
                                          'uid':'lid','value':'level'})
            for row in dim.iter_rows(add_bounds=add_bounds):
                if add_bounds and bound is not None:
                    self.assertTrue('bnds' in row)
                else:
                    self.assertFalse('bnds' in row)
                    
    def test_OcgIdentifier(self):
        oid = OcgIdentifier()
        oid.add(55)
        self.assertEqual(oid,{55:1})
        oid.add(55)
        self.assertEqual(oid,{55:1})
        oid.add(56)
        self.assertEqual(oid,{55:1,56:2})

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()