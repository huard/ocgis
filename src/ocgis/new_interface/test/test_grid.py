import numpy as np

from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.grid import GridXY
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable


class TestGridXY(AbstractTestNewInterface):

    def get_iter(self, return_kwargs=False):
        poss = [True, False]
        kwds = dict(with_2d_variables=poss,
                    with_dimensions=poss,
                    with_value=poss,
                    with_value_only=poss)
        for k in self.iter_product_keywords(kwds, as_namedtuple=False):
            ret = self.get_gridxy(**k)
            if return_kwargs:
                ret = (ret, k)
            yield ret

    def test_bases(self):
        self.assertEqual(GridXY.__bases__, (Variable,))

    def test_init(self):
        grid = self.get_gridxy()
        self.assertIsInstance(grid, GridXY)

        x = Variable('x', value=[1])
        with self.assertRaises(ValueError):
            GridXY(x=x)

        grid = self.get_gridxy(with_value=True)
        self.assertIsNotNone(grid._value)

    def test_dimensions(self):
        grid = self.get_gridxy()
        self.assertIsNone(grid.dimensions)

        grid = self.get_gridxy(with_dimensions=True)
        self.assertEqual(len(grid.dimensions), 2)
        self.assertEqual(grid.dimensions[0], Dimension('y', 4))

        grid = self.get_gridxy(with_dimensions=True, with_2d_variables=True)
        self.assertEqual(len(grid.dimensions), 2)
        self.assertEqual(grid.dimensions[0], Dimension('y', 4))

        grid = self.get_gridxy(with_value_only=True)
        self.assertIsNone(grid.x)
        self.assertIsNone(grid.y)
        self.assertIsNone(grid.dimensions)

    def test_getitem(self):
        for with_dimensions in [False, True]:
            grid = self.get_gridxy(with_dimensions=with_dimensions)
            self.assertEqual(grid.ndim, 2)
            sub = grid[2, 1]
            self.assertEqual(sub.x.value, 102.)
            self.assertEqual(sub.y.value, 42.)
            self.assertIsNone(grid._value)

            # Test with two-dimensional x and y values.
            grid = self.get_gridxy(with_2d_variables=True, with_dimensions=with_dimensions)
            sub = grid[1:3, 1:3]
            actual_x = [[102.0, 103.0], [102.0, 103.0]]
            self.assertEqual(sub.x.value.tolist(), actual_x)
            actual_y = [[41.0, 41.0], [42.0, 42.0]]
            self.assertEqual(sub.y.value.tolist(), actual_y)
            self.assertIsNone(grid._value)

        # Test with a value.
        grid = self.get_gridxy(with_value_only=True)
        sub = grid[1, :]
        self.assertEqual(sub.value.tolist(), [[[41.0, 41.0, 41.0]], [[101.0, 102.0, 103.0]]])

    def test_resolution(self):
        for grid in self.get_iter():
            self.assertEqual(grid.resolution, 1.)

    def test_shape(self):
        for grid in self.get_iter():
            self.assertEqual(grid.shape, (4, 3))
            self.assertEqual(grid.ndim, 2)

    def test_value(self):
        for grid, kwds in self.get_iter(return_kwargs=True):
            try:
                self.assertIsNone(grid._value)
            except AssertionError:
                self.assertTrue(kwds['with_value'] or kwds['with_value_only'])
            value = grid.value
            self.assertEqual(value.shape, (2, 4, 3))
            self.assertTrue(np.all(grid.value[0, 1, :] == 41.))
            self.assertTrue(np.all(grid.value[1, :, 1] == 102.))

    def test_write_netcdf(self):
        grid = self.get_gridxy()
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            self.assertNumpyAll(var[:], grid.y.value.data)

        # Test with 2-d x and y arrays.
        grid = self.get_gridxy(with_2d_variables=True, with_dimensions=True)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            self.assertNumpyAll(var[:], grid.y.value.data)

        # Test when the value is loaded.
        grid = self.get_gridxy(with_dimensions=True)
        grid._get_value_()
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path, 'r') as ds:
            self.assertEqual(['y', 'x'], [d for d in ds.variables['y'].dimensions])

        # Test with a value only.
        grid = self.get_gridxy(with_value_only=True)
        self.assertIsNone(grid.y)
        self.assertIsNone(grid.x)
        self.assertIsNone(grid.dimensions)
        dimensions = (Dimension('yy', 4), Dimension('xx', 3))
        grid.dimensions = dimensions
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path, 'r') as ds:
            yc = ds.variables['yc']
            self.assertEqual(['yy', 'xx'], [d for d in yc.dimensions])
            self.assertEqual(yc.axis, 'Y')
