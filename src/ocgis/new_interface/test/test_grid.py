import itertools

import numpy as np
from shapely.geometry import Point

from ocgis.exc import EmptySubsetError
from ocgis.interface.base.crs import WGS84, CoordinateReferenceSystem
from ocgis.new_interface.dimension import Dimension, SourcedDimension
from ocgis.new_interface.grid import GridXY
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable, BoundedVariable
from ocgis.util.helpers import make_poly, iter_array


class TestGridXY(AbstractTestNewInterface):
    def assertGridCorners(self, grid):
        """
        :type grid: :class:`ocgis.new_interface.grid.GridXY`
        """

        assert grid.corners is not None

        def _get_is_ascending_(arr):
            """
            Return ``True`` if the array is ascending from index 0 to -1.

            :type arr: :class:`numpy.ndarray`
            :rtype: bool
            """

            assert (arr.ndim == 1)
            if arr[0] < arr[-1]:
                ret = True
            else:
                ret = False

            return ret

        # Assert polygon constructed from grid corners contains the associated centroid value.
        for ii, jj in itertools.product(range(grid.shape[0]), range(grid.shape[1])):
            pt = Point(grid.value.data[1, ii, jj], grid.value.data[0, ii, jj])
            poly_corners = grid.corners.data[:, ii, jj]
            rtup = (poly_corners[0, :].min(), poly_corners[0, :].max())
            ctup = (poly_corners[1, :].min(), poly_corners[1, :].max())
            poly = make_poly(rtup, ctup)
            self.assertTrue(poly.contains(pt))

        # Assert masks are equivalent between value and corners.
        for (ii, jj), m in iter_array(grid.value.mask[0, :, :], return_value=True):
            if m:
                self.assertTrue(grid.corners.mask[:, ii, jj].all())
            else:
                self.assertFalse(grid.corners.mask[:, ii, jj].any())

        grid_y = grid._y
        grid_x = grid._x
        if grid_y is not None or grid_x is not None:
            self.assertEqual(_get_is_ascending_(grid_y.value), _get_is_ascending_(grid.corners.data[0, :, 0][:, 0]))
            self.assertEqual(_get_is_ascending_(grid_x.value), _get_is_ascending_(grid.corners.data[1, 0, :][:, 0]))

    def get_iter_gridxy(self, return_kwargs=False):
        poss = [True, False]
        kwds = dict(with_2d_variables=poss,
                    with_dimensions=poss)
        for k in self.iter_product_keywords(kwds, as_namedtuple=False):
            ret = self.get_gridxy(**k)
            if return_kwargs:
                ret = (ret, k)
            yield ret

    def test(self):
        # tdk: add mechanism for asserting variable dimensions match variable values
        # Test behavior of dimension-attached variables.
        dx = SourcedDimension('dx', 3)
        dy = SourcedDimension('dy', 4)
        x = Variable(value=[1, 2, 3], name='x', dimensions=dx)
        y = Variable(value=[10, 20, 30, 40], name='y', dimensions=dy)
        tas = Variable(value=np.random.rand(4, 3), dimensions=[dy, dx])

        dx.attach_variable(x)
        dy.attach_variable(y)

        self.assertTrue(np.may_share_memory(dx._variable.value, x.value))
        self.assertTrue(np.may_share_memory(dy._variable.value, y.value))

        sub = tas[2:4, 1]

        self.assertNumpyAll(sub.dimensions[0]._variable.value, y[2:4].value)
        self.assertNumpyAll(sub.dimensions[1]._variable.value, x[1].value)

        grid = GridXY(x=x, y=y)
        original_dim_ids = [id(d) for d in grid.dimensions]
        self.assertEqual(id(grid._x.dimensions[0]), original_dim_ids[1])
        grid[:] = 1000
        self.assertEqual(grid._x.value.mean(), 1000)
        self.assertEqual(id(grid._x.dimensions[1]), original_dim_ids[1])
        self.assertEqual(id(grid._x.dimensions[0]), original_dim_ids[0])
        self.assertEqual(map(id, grid.dimensions), original_dim_ids)
        self.assertTrue(np.may_share_memory(grid.dimensions[0]._src_idx, dy._src_idx))

    def test_init(self):
        crs = WGS84()
        grid = self.get_gridxy(crs=crs)
        self.assertIsInstance(grid, GridXY)
        self.assertIn('x', grid._variables)
        self.assertIn('y', grid._variables)
        self.assertEqual(grid.crs, crs)

    # tdk: remove
    # def test_corners(self):
    #     # Test constructing from x/y bounds.
    #     grid = self.get_gridxy()
    #     grid._x = BoundedVariable(value=grid._x.value, name='x')
    #     grid._y = BoundedVariable(value=grid._y.value, name='y')
    #     grid._x.set_extrapolated_bounds()
    #     grid._y.set_extrapolated_bounds()
    #     corners = grid.corners.copy()
    #     value = grid.value.copy()
    #     self.assertIsNotNone(corners)
    #     self.assertEqual(corners.shape, (2, grid._y.shape[0], grid._x.shape[0], 4))
    #     self.assertGridCorners(grid)
    #
    #     # Test initializing corners with a value.
    #     grid = GridXY(value=value, corners=corners)
    #     self.assertIsNotNone(grid._corners)
    #     self.assertNumpyAll(grid.corners, corners)
    #
    #     # Test corners are sliced.
    #     sub = grid[2:4, 1]
    #     self.assertEqual(sub.corners.shape, (2, 2, 1, 4))
    #
    #     # Test constructing with two-dimensional variables.
    #     y_value = [[40.0, 40.0, 40.0], [41.0, 41.0, 41.0], [42.0, 42.0, 42.0], [43.0, 43.0, 43.0]]
    #     y_corners = [[[39.5, 39.5, 40.5, 40.5], [39.5, 39.5, 40.5, 40.5], [39.5, 39.5, 40.5, 40.5]],
    #                  [[40.5, 40.5, 41.5, 41.5], [40.5, 40.5, 41.5, 41.5], [40.5, 40.5, 41.5, 41.5]],
    #                  [[41.5, 41.5, 42.5, 42.5], [41.5, 41.5, 42.5, 42.5], [41.5, 41.5, 42.5, 42.5]],
    #                  [[42.5, 42.5, 43.5, 43.5], [42.5, 42.5, 43.5, 43.5], [42.5, 42.5, 43.5, 43.5]]]
    #     x_value = [[101.0, 102.0, 103.0], [101.0, 102.0, 103.0], [101.0, 102.0, 103.0], [101.0, 102.0, 103.0]]
    #     x_corners = [[[100.5, 101.5, 101.5, 100.5], [101.5, 102.5, 102.5, 101.5], [102.5, 103.5, 103.5, 102.5]],
    #                  [[100.5, 101.5, 101.5, 100.5], [101.5, 102.5, 102.5, 101.5], [102.5, 103.5, 103.5, 102.5]],
    #                  [[100.5, 101.5, 101.5, 100.5], [101.5, 102.5, 102.5, 101.5], [102.5, 103.5, 103.5, 102.5]],
    #                  [[100.5, 101.5, 101.5, 100.5], [101.5, 102.5, 102.5, 101.5], [102.5, 103.5, 103.5, 102.5]]]
    #
    #     y_bounds = Variable(value=y_corners, name='y_corners')
    #     y_bounds.create_dimensions(names=['y', 'x', 'cbnds'])
    #     y = BoundedVariable(value=y_value, bounds=y_bounds, name='y')
    #     y.create_dimensions(names=['y', 'x'])
    #
    #     x_bounds = Variable(value=x_corners, name='x_corners')
    #     x_bounds.create_dimensions(names=['y', 'x', 'cbnds'])
    #     x = BoundedVariable(value=x_value, bounds=x_bounds, name='x')
    #     x.create_dimensions(names=['y', 'x'])
    #
    #     grid = GridXY(x=x, y=y)
    #     np.testing.assert_equal(grid.corners[0], y_corners)
    #     np.testing.assert_equal(grid.corners[1], x_corners)
    #
    #     path = self.get_temporary_file_path('foo.nc')
    #     with self.nc_scope(path, 'w') as ds:
    #         grid.write_netcdf(ds)
    #     # subprocess.check_call(['ncdump', path])
    #
    #     # Test writing with dimension variables.
    #     self.assertIsNotNone(grid.corners)
    #     with self.nc_scope(path, 'w') as ds:
    #         grid.write_netcdf(ds)
    #     # subprocess.check_call(['ncdump', path])
    #
    #     with self.nc_scope(path) as ds:
    #         self.assertIn('y_corners', ds.variables)
    #         self.assertIn('x_corners', ds.variables)

    def test_corners_esmf(self):
        x_bounds = Variable(value=[[-100.5, -99.5], [-99.5, -98.5], [-98.5, -97.5], [-97.5, -96.5]], name='x_bounds')
        x = BoundedVariable(value=[-100., -99., -98., -97.], bounds=x_bounds, name='x')

        y_bounds = Variable(value=[[40.5, 39.5], [39.5, 38.5], [38.5, 37.5]], name='y_bounds')
        y = BoundedVariable(value=[40., 39., 38.], bounds=y_bounds, name='y')

        grid = GridXY(x=x, y=y)

        actual = np.array([[[40.5, 40.5, 40.5, 40.5, 40.5], [39.5, 39.5, 39.5, 39.5, 39.5],
                            [38.5, 38.5, 38.5, 38.5, 38.5], [37.5, 37.5, 37.5, 37.5, 37.5]],
                           [[-100.5, -99.5, -98.5, -97.5, -96.5], [-100.5, -99.5, -98.5, -97.5, -96.5],
                            [-100.5, -99.5, -98.5, -97.5, -96.5], [-100.5, -99.5, -98.5, -97.5, -96.5]]],
                          dtype=grid.value.dtype)
        self.assertNumpyAll(actual, grid.corners_esmf)

    def test_dimensions(self):
        grid = self.get_gridxy()
        grid.create_dimensions()
        self.assertEqual(grid.dimensions, (Dimension(name='y', length=4), Dimension(name='x', length=3)))

        grid = self.get_gridxy(with_dimensions=True)
        self.assertEqual(len(grid.dimensions), 2)
        self.assertEqual(grid.dimensions[0], Dimension('y', 4))

        grid = self.get_gridxy(with_dimensions=True, with_2d_variables=True)
        self.assertEqual(len(grid.dimensions), 2)
        self.assertEqual(grid.dimensions[0], Dimension('y', 4))

        grid = self.get_gridxy()
        self.assertIsNone(grid.dimensions)
        grid = self.get_gridxy(with_dimensions=True)
        self.assertIsNotNone(grid.dimensions)

        grid = self.get_gridxy()
        grid.create_dimensions()
        self.assertEqual(len(grid.dimensions), 2)

    def test_expand(self):
        grid = self.get_gridxy(with_value_mask=True)
        self.assertTrue(grid.is_vectorized)
        grid.expand()
        for target in [grid.x, grid.y]:
            self.assertTrue(target.get_mask().any())
        self.assertFalse(grid.is_vectorized)
        self.assertEqual(grid.ndim, 2)
        self.assertEqual(grid.shape, (4, 3))

    def test_getitem(self):
        for with_dimensions in [False, True]:
            grid = self.get_gridxy(with_dimensions=with_dimensions)
            self.assertEqual(grid.ndim, 2)
            sub = grid[2, 1]
            self.assertNotEqual(grid.shape, sub.shape)
            self.assertEqual(sub.x.value, 102.)
            self.assertEqual(sub.y.value, 42.)

            # Test with two-dimensional x and y values.
            grid = self.get_gridxy(with_2d_variables=True, with_dimensions=with_dimensions)
            sub = grid[1:3, 1:3]
            actual_x = [[102.0, 103.0], [102.0, 103.0]]
            self.assertEqual(sub.x.value.tolist(), actual_x)
            actual_y = [[41.0, 41.0], [42.0, 42.0]]
            self.assertEqual(sub.y.value.tolist(), actual_y)

            # Test with backrefs.


    def test_get_mask(self):
        grid = self.get_gridxy()
        self.assertTrue(grid.is_vectorized)
        mask = grid.get_mask()
        self.assertEqual(mask.ndim, 2)
        self.assertFalse(np.any(mask))
        self.assertFalse(grid.is_vectorized)

    def test_get_subset_bbox(self):
        keywords = dict(bounds=[True, False], closed=[True, False])

        for k in self.iter_product_keywords(keywords):
            y = self.get_variable_y(bounds=k.bounds)
            x = self.get_variable_x(bounds=k.bounds)
            grid = GridXY(x, y)
            bg = grid.get_subset_bbox(-99, 39, -98, 39, closed=False)
            self.assertTrue(bg.is_vectorized)
            with self.assertRaises(EmptySubsetError):
                grid.get_subset_bbox(1000, 1000, 1001, 10001, closed=k.closed)

            bg2 = grid.get_subset_bbox(-99999, 1, 1, 1000, closed=k.closed)
            for target in ['x', 'y']:
                original = getattr(grid, target).value
                sub = getattr(bg2, target).value
                self.assertNumpyAll(original, sub)

        # Test mask is not shared with subsetted grid.
        grid = self.get_gridxy()
        new_mask = grid.get_mask()
        new_mask[:, 1] = True
        grid.set_mask(new_mask)
        args = (101.5, 40.5, 102.5, 42.5)
        sub = grid.get_subset_bbox(*args, use_bounds=False)
        self.assertTrue(np.all(sub.get_mask()[1, 0]))
        self.assertTrue(np.all(grid.get_mask()))
        raise self.ToTest('correct copy is made when subsetting')

    def test_resolution(self):
        for grid in self.get_iter_gridxy():
            self.assertEqual(grid.resolution, 1.)

    def test_set_extrapolated_bounds(self):
        value_grid = [[[40.0, 40.0, 40.0, 40.0], [39.0, 39.0, 39.0, 39.0], [38.0, 38.0, 38.0, 38.0]],
                      [[-100.0, -99.0, -98.0, -97.0], [-100.0, -99.0, -98.0, -97.0], [-100.0, -99.0, -98.0, -97.0]]]
        actual_corners = [
            [[[40.5, 40.5, 39.5, 39.5], [40.5, 40.5, 39.5, 39.5], [40.5, 40.5, 39.5, 39.5], [40.5, 40.5, 39.5, 39.5]],
             [[39.5, 39.5, 38.5, 38.5], [39.5, 39.5, 38.5, 38.5], [39.5, 39.5, 38.5, 38.5], [39.5, 39.5, 38.5, 38.5]],
             [[38.5, 38.5, 37.5, 37.5], [38.5, 38.5, 37.5, 37.5], [38.5, 38.5, 37.5, 37.5], [38.5, 38.5, 37.5, 37.5]]],
            [[[-100.5, -99.5, -99.5, -100.5], [-99.5, -98.5, -98.5, -99.5], [-98.5, -97.5, -97.5, -98.5],
              [-97.5, -96.5, -96.5, -97.5]],
             [[-100.5, -99.5, -99.5, -100.5], [-99.5, -98.5, -98.5, -99.5], [-98.5, -97.5, -97.5, -98.5],
              [-97.5, -96.5, -96.5, -97.5]],
             [[-100.5, -99.5, -99.5, -100.5], [-99.5, -98.5, -98.5, -99.5], [-98.5, -97.5, -97.5, -98.5],
              [-97.5, -96.5, -96.5, -97.5]]]]
        y = BoundedVariable(name='y', value=value_grid[0])
        x = BoundedVariable(name='x', value=value_grid[1])
        grid = GridXY(x, y)
        grid.set_extrapolated_bounds()
        np.testing.assert_equal(grid.y.bounds.value, actual_corners[0])
        np.testing.assert_equal(grid.x.bounds.value, actual_corners[1])

    def test_set_mask(self):
        grid = self.get_gridxy()
        self.assertFalse(np.any(grid.get_mask()))
        mask = np.zeros(grid.shape, dtype=bool)
        mask[1, 1] = True
        grid.set_mask(mask)
        self.assertTrue(np.all(grid.y.get_mask()[1, 1]))
        self.assertTrue(np.all(grid.x.get_mask()[1, 1]))

    def test_shape(self):
        for grid in self.get_iter_gridxy():
            self.assertEqual(grid.shape, (4, 3))
            self.assertEqual(grid.ndim, 2)

    def test_update_crs(self):
        grid = self.get_gridxy(crs=WGS84())
        grid.set_extrapolated_bounds()
        self.assertIsNotNone(grid.y.bounds)
        self.assertIsNotNone(grid.x.bounds)
        to_crs = CoordinateReferenceSystem(epsg=3395)
        grid.update_crs(to_crs)
        self.assertEqual(grid.crs, to_crs)
        for element in [grid.x, grid.y]:
            for target in [element.value, element.bounds.value]:
                self.assertTrue(np.all(target > 10000))

    def test_write_netcdf(self):
        grid = self.get_gridxy()
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            self.assertNumpyAll(var[:], grid._y.value.data)

        # Test with 2-d x and y arrays.
        grid = self.get_gridxy(with_2d_variables=True, with_dimensions=True)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            self.assertNumpyAll(var[:], grid._y.value.data)

        # Test when the value is loaded.
        grid = self.get_gridxy(with_dimensions=True)
        self.assertIsNone(grid._value)
        grid._get_value_()
        self.assertIsNotNone(grid._dimensions)
        self.assertTrue(grid.is_vectorized)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path, 'r') as ds:
            self.assertEqual(['y'], [d for d in ds.variables['y'].dimensions])
            self.assertEqual(['x'], [d for d in ds.variables['x'].dimensions])

        # Test with a value only.
        grid = self.get_gridxy(with_value_only=True)
        dimensions = (Dimension('yy', 4), Dimension('xx', 3))
        grid.dimensions = dimensions
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path, 'r') as ds:
            yc = ds.variables['yc']
            self.assertEqual(['yy', 'xx'], [d for d in yc.dimensions])
            self.assertEqual(yc.axis, 'Y')
