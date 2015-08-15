import numpy as np

from ocgis.new_interface.adapter import BoundedVariable, AbstractAdapter
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import get_bounds_from_1d


class TestBoundedVariable(AbstractTestNewInterface):
    def get(self, with_bounds=True):
        value = np.array([4, 5, 6], dtype=float)
        var = Variable('x', value=value)
        if with_bounds:
            value_bounds = get_bounds_from_1d(value)
            bounds = Variable('x_bounds', value=value_bounds)
        else:
            bounds = None
        bv = BoundedVariable(var, bounds=bounds)
        return bv

    def test_bases(self):
        self.assertEqual(BoundedVariable.__bases__, (AbstractAdapter,))

    def test_init(self):
        bv = self.get()
        with self.assertRaises(AttributeError):
            bv.shape

    def test_getitem(self):
        bv = self.get()
        sub = bv[1]
        self.assertNumpyAll(sub.bounds.value, bv.bounds[1, :].value)

    def test_set_extrapolated_bounds(self):
        bv = self.get(with_bounds=False)
        self.assertIsNone(bv.bounds)
        self.assertFalse(bv._has_extrapolated_bounds)
        bv.set_extrapolated_bounds()
        self.assertTrue(bv._has_extrapolated_bounds)
        self.assertEqual(bv.bounds.name, 'x_bounds')
        self.assertEqual(bv.bounds.ndim, 2)

    def test_write_netcdf(self):
        bv = self.get()
        dim_x = Dimension('x', 3)
        bv.variable.dimensions = dim_x
        bv.bounds.dimensions = [dim_x, Dimension('bounds', 2)]
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            bv.write_netcdf(ds)
        with self.nc_scope(path, 'r') as ds:
            var = ds.variables[bv.variable.name]
            self.assertEqual(var.bounds, bv.bounds.name)
