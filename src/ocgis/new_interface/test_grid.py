from copy import deepcopy

import numpy as np

from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.grid import Grid
from ocgis.new_interface.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable


class TestGrid(AbstractTestNewInterface):
    def get(self, with_z=True, with_2d_variables=False, with_zndim=3, with_dimensions=False):
        assert with_zndim in (1, 3)

        x = [101, 102, 103]
        y = [40, 41, 42, 43]
        z = [100, 200]

        x_dim = Dimension('x', length=len(x))
        y_dim = Dimension('y', length=len(y))
        z_dim = Dimension('z', length=len(z))

        kwds = {}

        if with_2d_variables:
            x_value, y_value = np.meshgrid(x, y)
            x_dims = (y_dim, x_dim)
            y_dims = x_dims
        else:
            x_value, y_value = x, y
            x_dims = (x_dim,)
            y_dims = (y_dim,)

        if not with_dimensions:
            x_dims = None
            y_dims = None

        vx = Variable('x', value=x_value, dtype=float, dimensions=x_dims)
        vy = Variable('y', value=y_value, dtype=float, dimensions=y_dims)
        kwds.update(dict(x=vx, y=vy))

        if with_z:
            if with_2d_variables:
                if with_zndim == 3:
                    z_value = np.zeros((3, 4, 3))
                    z_value[0, ...] = 100.
                    z_value[1, ...] = 200.
                    z_value[2, ...] = 300.
                    z_dims = (Dimension('z', length=3), y_dim, x_dim)
                else:
                    z_value = np.zeros((4, 3))
                    z_value[:] = 150.
                    z_dims = (y_dim, x_dim)
            else:
                z_value = z
                z_dims = (z_dim,)

            if not with_dimensions:
                z_dims = None

            vz = Variable('z', value=z_value, dtype=float, dimensions=z_dims)
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

    def test_expand(self):
        grid = self.get()
        grid_orig = deepcopy(grid)
        self.assertEqual(grid.x.ndim, 1)
        self.assertTrue(grid.is_vectorized)
        grid.expand()
        self.assertEqual(grid.x.ndim, 2)
        self.assertFalse(grid.is_vectorized)
        self.assertEqual(grid.shape, grid_orig.shape)

        # Test dimensions are preserved.
        grid = self.get(with_dimensions=True)
        grid.expand()
        for v in [grid.x, grid.y]:
            self.assertEqual(v.dimensions, (Dimension(name='y', length=4), Dimension(name='x', length=3)))

        # Test with a single row and column.
        x = Variable('x', value=[1], dimensions=Dimension('x', length=1))
        y = Variable('y', value=[2], dimensions=Dimension('y', length=1))
        grid = Grid(x=x, y=y)
        grid.expand()
        self.assertEqual(grid.shape, (1, 1))

    def test_getitem(self):
        for with_dimensions in [False, True]:
            grid = self.get(with_dimensions=with_dimensions)
            self.assertEqual(grid.ndim, 3)
            sub = grid[1, 2, 0]
            self.assertEqual(sub.x.value, 102.)
            self.assertEqual(sub.y.value, 42.)
            self.assertEqual(sub.z.value, 100.)

            # Test with two-dimensional x and y values.
            grid = self.get(with_2d_variables=True, with_z=False, with_dimensions=with_dimensions)
            sub = grid[1:3, 1:]
            actual_x = [[102.0, 103.0], [102.0, 103.0]]
            self.assertEqual(sub.x.value.tolist(), actual_x)
            actual_y = [[41.0, 41.0], [42.0, 42.0]]
            self.assertEqual(sub.y.value.tolist(), actual_y)

            # Test with a z-coordinate.
            grid = self.get(with_2d_variables=True, with_dimensions=with_dimensions)
            sub = grid[:, :, 1]
            self.assertTrue(np.all(sub.z.value == 200.))

            # Test with a z-coordinate having one 2-d level.
            grid = self.get(with_2d_variables=True, with_zndim=1, with_dimensions=with_dimensions)
            self.assertEqual(grid.ndim, 3)
            self.assertEqual(grid.shape, (4, 3, 1))
            sub = grid[1:3, :, :]
            self.assertEqual(sub.shape, (2, 3, 1))

    def test_mask(self):
        grid = self.get()
        print grid.mask
        grid.mask[3] = True
        print grid.mask
        print grid.x.value.mask
        print grid.y.value.mask

    def test_resolution(self):
        grid = self.get()
        self.assertEqual(grid.resolution, 1.)

    def test_shape(self):
        grid = self.get()
        self.assertEqual(grid.shape, (4, 3, 2))
        self.assertEqual(grid.ndim, 3)

        grid = self.get(with_z=False)
        self.assertEqual(grid.ndim, 2)

        # Test with two-dimensional x and y values.
        grid = self.get(with_2d_variables=True, with_z=False)
        self.assertEqual(grid.shape, (4, 3))

        # Test with different 3-d configurations. ######################################################################
        grid = self.get(with_2d_variables=True, with_z=False)
        z = Variable('z', value=[100, 200])
        grid.z = z
        self.assertEqual(grid.shape, (4, 3, 2))

        grid = self.get(with_2d_variables=True, with_z=False)
        value = np.zeros(grid.shape)
        value[:] = 100.
        z = Variable('z', value=value)
        grid.z = z
        self.assertEqual(grid.shape, (4, 3, 1))

        grid = self.get(with_2d_variables=True)
        self.assertEqual(grid.shape, (4, 3, 3))
        ################################################################################################################

    def test_write_netcdf(self):
        grid = self.get(with_z=True)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            self.assertNumpyAll(var[:], grid.y.value.data)

        # Test with 2-d x and y arrays.
        grid = self.get(with_z=True, with_2d_variables=True, with_dimensions=True)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            self.assertNumpyAll(var[:], grid.y.value.data)
            var_z = ds.variables['z']
            self.assertEqual(len(var_z.dimensions), 3)
