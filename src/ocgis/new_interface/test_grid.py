import numpy as np

from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.grid import Grid
from ocgis.new_interface.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable


class TestGrid(AbstractTestNewInterface):
    def get(self, with_z=True, with_2d_variables=False):
        x = [101, 102, 103]
        y = [40, 41, 42, 43]
        z = [100, 200]

        kwds = {}

        if with_2d_variables:
            mx, my = np.meshgrid(x, y)
            vx = Variable('x', value=mx, dtype=float)
            vy = Variable('y', value=my, dtype=float)
        else:
            vx = Variable('x', value=x, dtype=float)
            vy = Variable('y', value=y, dtype=float)
        kwds.update(dict(x=vx, y=vy))

        if with_z:
            assert not with_2d_variables
            vz = Variable('z', value=z, dtype=float)
            kwds.update(dict(z=vz))

        grid = Grid(**kwds)
        return grid

    def test_bases(self):
        self.assertEqual(Grid.__bases__, (AbstractInterfaceObject,))

    def test_init(self):
        grid = self.get()
        self.assertIsInstance(grid, Grid)

        x = Variable('x', value=[1])
        with self.assertRaises(ValueError):
            Grid(x=x)

    def test_getitem(self):
        # tdk: test with 2/3-d level
        grid = self.get()
        sub = grid[1, 2, 0]
        self.assertEqual(sub.x.value, 102.)
        self.assertEqual(sub.y.value, 42.)
        self.assertEqual(sub.z.value, 100.)

        # Test with two-dimensional x and y values.
        grid = self.get(with_2d_variables=True, with_z=False)
        sub = grid[1:3, 1:]
        actual_x = [[102.0, 103.0], [102.0, 103.0]]
        self.assertEqual(sub.x.value.tolist(), actual_x)
        actual_y = [[41.0, 41.0], [42.0, 42.0]]
        self.assertEqual(sub.y.value.tolist(), actual_y)

    def test_resolution(self):
        grid = self.get()
        self.assertEqual(grid.resolution, 1.)

    def test_shape(self):
        grid = self.get()
        self.assertEqual(grid.shape, (3, 4, 2))
        self.assertEqual(grid.ndim, 3)

        grid = self.get(with_z=False)
        self.assertEqual(grid.ndim, 2)

        # Test with two-dimensional x and y values.
        grid = self.get(with_2d_variables=True, with_z=False)
        self.assertEqual(grid.shape, (4, 3))

    def test_write_netcdf(self):
        grid = self.get(with_z=True)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            self.assertNumpyAll(var[:], grid.y.value.data)
