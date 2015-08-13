from numpy.ma import MaskedArray
import numpy as np

from ocgis.interface.nc.field import NcField
from ocgis.test.base import TestBase


class TestNcField(TestBase):
    def get_field_slice(self, rd, slc):
        field = rd.get().__getitem__(slc)
        self.assertIsInstance(field, NcField)
        return field

    def get_netcdf4_slice(self, rd, slc):
        with self.nc_scope(rd.uri, 'r') as ds:
            var = ds.variables['tas']
            actual = var.__getitem__(slc)
        return actual

    def test_get_value_from_source(self):
        rd = self.test_data.get_rd('cancm4_tas')
        data = rd
        variable_name = 'tas'

        def _run_(slc_field, slc_nc, actual_shape):
            field = self.get_field_slice(rd, slc_field)
            actual = self.get_netcdf4_slice(rd, slc_nc)
            res = field._get_value_from_source_(data, variable_name)
            self.assertIsInstance(res, MaskedArray)
            self.assertEqual(res.shape, actual_shape)
            self.assertNumpyAll(res.data.flatten(), actual.flatten())

        sf = (slice(None), 10, slice(None), 5, 23)
        sn = (10, 5, 23)
        ashp = (1, 1, 1, 1, 1)
        _run_(sf, sn, ashp)

        sf = (slice(None), slice(10, 20), slice(None), slice(5, 9), slice(10, 15))
        sn = (slice(10, 20), slice(5, 9), slice(10, 15))
        ashp = (1, 10, 1, 4, 5)
        _run_(sf, sn, ashp)

        sf = (slice(None),
              np.array([10, 13, 15]),
              slice(None),
              np.array([5, 7, 13, 14]),
              np.array([10, 14, 16, 17]))
        sn = (np.array([10, 13, 15]),
              np.array([5, 7, 13, 14]),
              np.array([10, 14, 16, 17]))
        ashp = (1, 3, 1, 4, 4)
        _run_(sf, sn, ashp)
