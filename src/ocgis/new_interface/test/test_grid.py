import itertools
from copy import deepcopy
from unittest import SkipTest

import fiona
import numpy as np
from numpy.testing.utils import assert_equal
from shapely import wkt
from shapely.geometry import Point, box, MultiPolygon, shape
from shapely.geometry.base import BaseGeometry

from ocgis.api.request.base import RequestDataset
from ocgis.exc import EmptySubsetError, BoundsAlreadyAvailableError
from ocgis.interface.base.crs import WGS84, CoordinateReferenceSystem
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.grid import GridXY, get_polygon_geometry_array, grid_get_intersects, get_geometry_fill
from ocgis.new_interface.mpi import MPI_RANK, MPI_COMM
from ocgis.new_interface.ocgis_logging import log
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable, VariableCollection, SourcedVariable
from ocgis.test.base import attr
from ocgis.util.helpers import make_poly, iter_array


class Test(AbstractTestNewInterface):
    @attr('mpi', 'mpi-4', 'mpi-12')
    def test_grid_get_intersects(self):
        # Test with an empty subset.
        bounds_sequence = (1000., 1000., 1100., 1100.)

        grid = self.get_gridxy()

        with self.assertRaises(EmptySubsetError):
            grid_get_intersects(grid, bounds_sequence)

        # Test combinations.
        bounds_sequence = (101.5, 40.5, 102.5, 42.)
        bounds_sequence_geometry = box(*bounds_sequence)

        keywords = dict(is_vectorized=[True, False], has_bounds=[False, True], use_bounds=[False, True],
                        keep_touches=[True, False], bounds_sequence=[bounds_sequence, bounds_sequence_geometry])

        for k in self.iter_product_keywords(keywords):
            if MPI_RANK == 0:
                log.debug(k)

            grid = self.get_gridxy()
            if k.has_bounds:
                grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
                self.assertTrue(grid.has_bounds)

            grid_sub, slc = grid_get_intersects(grid, bounds_sequence, keep_touches=k.keep_touches,
                                                use_bounds=k.use_bounds)

            if k.has_bounds:
                self.assertTrue(grid.has_bounds)

            log.info('asserts')
            if MPI_RANK == 0:
                # Test geometries are filled appropriately after allocation.
                for t in grid_sub.abstraction_geometry.value.flat:
                    self.assertIsInstance(t, BaseGeometry)
                self.assertIsInstance(grid_sub, GridXY)
                if k.keep_touches:
                    if k.has_bounds and k.use_bounds:
                        desired = (slice(0, 3, None), slice(0, 3, None))
                    else:
                        desired = (slice(1, 3, None), slice(1, 2, None))
                else:
                    if k.has_bounds and k.use_bounds:
                        desired = (slice(1, 3, None), slice(1, 2, None))
                    else:
                        desired = (slice(1, 2, None), slice(1, 2, None))
                self.assertEqual(grid.has_bounds, k.has_bounds)
                self.assertTrue(grid.is_vectorized)
                self.assertEqual(slc, desired)
            else:
                self.assertIsNone(grid_sub)
                self.assertIsNone(slc)
            log.info('asserts end')

        # Test against a file. #########################################################################################
        bounds_sequence = (101.5, 40.5, 102.5, 42.)

        if MPI_RANK == 0:
            path_grid = self.get_temporary_file_path('grid.nc')
            grid_to_write = self.get_gridxy()
            grid_to_write.write_netcdf(path_grid)
            rd = RequestDataset(uri=path_grid)
        else:
            rd = None

        rd = MPI_COMM.bcast(rd, root=0)
        x = SourcedVariable(name='x', request_dataset=rd)
        self.assertIsNone(x._value)
        y = SourcedVariable(name='y', request_dataset=rd)
        self.assertIsNone(x._value)
        self.assertIsNone(y._value)
        grid = GridXY(x, y)
        self.assertIsNone(grid.x._value)

        self.assertTrue(grid.is_vectorized)
        self.assertIsNone(grid.x._value)
        sub, slc = grid_get_intersects(grid, bounds_sequence, mpi_comm=MPI_COMM)

        if MPI_RANK == 0:
            self.assertEqual(slc, (slice(1, 3, None), slice(1, 2, None)))
            self.assertIsInstance(sub, GridXY)
        else:
            self.assertIsNone(slc)
            self.assertIsNone(sub)

        # The file may be deleted before other ranks open.
        MPI_COMM.Barrier()

    @attr('mpi', 'mpi-4', 'mpi-12')
    def test_grid_get_intersects2(self):
        log.info('hello world')
        subset1 = 'Polygon ((100.79558316115701189 41.18854700413223213, 100.79558316115701189 40.80036157024792942, 102.13212035123964938 40.82493026859503971, 102.27953254132229688 41.47354390495867449, 103.62589721074378701 41.55707747933884377, 102.28444628099171609 41.66517975206611624, 102.06332799586775195 42.18112241735536827, 101.72919369834708903 42.13198502066115481, 101.72919369834708903 41.2671668388429751, 100.79558316115701189 41.18854700413223213))'
        subset1 = wkt.loads(subset1)
        subset2 = 'Polygon ((102.82100076148078927 43.17027003280225017, 103.26183018978443329 43.19503573102155514, 103.47481519447048015 42.70467490627928697, 101.38459026476100178 42.53131501874414511, 102.81604762183692969 42.86317537488285012, 102.82100076148078927 43.17027003280225017))'
        subset2 = wkt.loads(subset2)

        ################################################################################################################

        # Test with a single polygon.
        subset = subset1

        grid = self.get_gridxy()

        res = grid_get_intersects(grid, subset)

        if MPI_RANK == 0:
            grid_sub, slc = res

            mask_grid_sub = grid_sub.get_mask()
            self.assertEqual(grid_sub.shape, (2, 2))
            self.assertTrue(np.any(mask_grid_sub))
            self.assertTrue(mask_grid_sub[1, 0])
            self.assertEqual(mask_grid_sub.sum(), 1)
        else:
            self.assertEqual(res, (None, None))

        ################################################################################################################

        # Test with a multi-polygon.

        subset = MultiPolygon([subset1, subset2])
        grid = self.get_gridxy()

        res = grid_get_intersects(grid, subset)

        if MPI_RANK == 0:
            grid_sub, slc = res
            mask_grid_sub = grid_sub.get_mask()
            desired_mask = np.array([[False, False, True], [True, False, True], [True, True, False]])
            self.assertNumpyAll(mask_grid_sub, desired_mask)
        else:
            self.assertEqual(res, (None, None))

        log.info('success')

    @attr('mpi', 'mpi-4')
    def test_grid_get_intersects3(self):
        if MPI_RANK == 0:
            path_shp = self.path_state_boundaries
            geoms = []
            with fiona.open(path_shp) as source:
                for record in source:
                    geom = shape(record['geometry'])
                    geoms.append(geom)

            gvar = GeometryVariable(value=geoms)
            gvar_sub = gvar.get_unioned()

            subset = gvar_sub.value.flatten()[0]
        else:
            subset = None

        subset = MPI_COMM.bcast(subset, root=0)

        resolution = 1.0

        for with_bounds in [False, True]:
            grid = self.get_gridxy_global(resolution=resolution, with_bounds=with_bounds)

            res = grid_get_intersects(grid, subset)

            if MPI_RANK == 0:
                grid_sub, slc = res

                if with_bounds:
                    self.assertEqual(grid_sub.get_mask().sum(), 4595)
                    self.assertEqual(slc, (slice(108, 161, None), slice(1, 113, None)))
                    self.assertEqual(grid_sub.shape, (53, 112))
                else:
                    self.assertEqual(grid_sub.get_mask().sum(), 3506)
                    self.assertEqual(slc, (slice(115, 161, None), slice(12, 112, None)))
                    self.assertEqual(grid_sub.shape, (46, 100))
            else:
                self.assertEqual(res, (None, None))

        log.info('success')

    def test_get_grid_intersects_small(self):
        """Test with a subset inside of one of the cells."""

        as_bounds_tuple = [False, True]

        subset = 'MULTIPOLYGON (((-71.79019426324761 41.60130736620898, -71.79260526324985 41.64175836624665, -71.7882492632458 41.72160336632101, -71.79783126325472 42.00427436658427, -71.49743026297496 42.00925336658891, -71.37864426286433 42.01371336659307, -71.38240526286783 41.97926336656098, -71.38395326286928 41.88843936647639, -71.33308626282189 41.89603136648346, -71.34249326283066 41.8757833664646, -71.33454226282325 41.85790336644796, -71.34548326283345 41.81316136640628, -71.33979826282815 41.78442536637952, -71.31932826280908 41.77219536636813, -71.26662826276001 41.74974336634722, -71.22897626272494 41.70769436630806, -71.28400126277619 41.67954936628185, -71.36738726285384 41.7413503663394, -71.39358026287823 41.76115536635785, -71.36901226285535 41.70329136630396, -71.41924726290215 41.65221236625639, -71.42731826290965 41.48668936610223, -71.48988826296792 41.39208536601413, -71.72226426318434 41.32726436595375, -71.86667826331885 41.32276936594957, -71.84777226330124 41.32534836595197, -71.83686926329108 41.34196136596745, -71.84599526329959 41.40385436602509, -71.8027432632593 41.41582936603623, -71.79019426324761 41.60130736620898)), ((-71.19880826269684 41.67850036628087, -71.14121226264319 41.65527336625924, -71.11713226262077 41.49306236610817, -71.19993726269789 41.46331836608047, -71.19880826269684 41.67850036628087)), ((-71.26916926276238 41.62126836622758, -71.21944726271606 41.63564236624096, -71.23867326273397 41.47484936609121, -71.28800726277991 41.48361936609938, -71.3495252628372 41.4458573660642, -71.26916926276238 41.62126836622758)))'
        subset = wkt.loads(subset)
        for a in as_bounds_tuple:
            grid = self.get_gridxy_global(resolution=3.0)
            if a:
                the_subset = subset.bounds
            else:
                the_subset = subset
            sub, slc = grid_get_intersects(grid, the_subset)
            self.assertEqual(sub.shape, (1, 1))
            self.assertEqual(sub.get_mask().sum(), 0)


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
        kwds = dict(with_2d_variables=poss)
        for k in self.iter_product_keywords(kwds, as_namedtuple=False):
            ret = self.get_gridxy(**k)
            if return_kwargs:
                ret = (ret, k)
            yield ret

    def test_init(self):
        crs = WGS84()
        grid = self.get_gridxy(crs=crs)
        self.assertIsInstance(grid, GridXY)
        self.assertIn('x', grid.parent)
        self.assertIn('y', grid.parent)
        self.assertEqual(grid.crs, crs)
        self.assertEqual(grid.dimensions, (Dimension(name='ydim', length=4), Dimension(name='xdim', length=3)))

        # Test with different variable names.
        x = Variable(name='col', value=[1], dimensions='col')
        y = Variable(name='row', value=[2], dimensions='row')
        grid = GridXY(x, y)
        assert_equal(grid.x.value, [[1]])
        assert_equal(grid.y.value, [[2]])

        # Test point and polygon representations.
        grid = self.get_gridxy(crs=WGS84())
        grid.set_extrapolated_bounds('x_bounds', 'y_bounds', 'bounds')
        targets = ['point', 'polygon']
        targets = [getattr(grid, t) for t in targets]
        for t in targets:
            self.assertIsInstance(t, GeometryVariable)
        sub = grid[1, 1]
        targets = ['point', 'polygon']
        targets = [getattr(sub, t) for t in targets]
        for t in targets:
            self.assertEqual(t.shape, (1, 1))
            self.assertIsInstance(t, GeometryVariable)

    def test_init_from_file(self):
        """Test loading from file."""

        grid = self.get_gridxy()
        path = self.get_temporary_file_path('foo.nc')
        grid.write_netcdf(path)
        rd = RequestDataset(uri=path)
        x = SourcedVariable(name=grid.x.name, request_dataset=rd)
        y = SourcedVariable(name=grid.y.name, request_dataset=rd)
        self.assertIsNone(x._value)
        self.assertIsNone(y._value)
        fgrid = GridXY(x, y)
        self.assertIsNone(fgrid.x._value)
        self.assertIsNone(fgrid.y._value)
        thh

    def test_corners_esmf(self):
        raise SkipTest('move to test for get_esmf_corners_from_ocgis_corners')
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

    def test_create_dimensions(self):
        grid = self.get_gridxy()
        grid.create_dimensions(names=['yy', 'xx'])
        self.assertEqual(grid.dimensions, (Dimension(name='yy', length=4), Dimension(name='xx', length=3)))

    def test_iter(self):
        grid = self.get_gridxy()
        for ctr, (idx, record) in enumerate(grid.iter()):
            self.assertIn(grid.y.name, record)
            self.assertIn(grid.x.name, record)
        self.assertEqual(ctr + 1, grid.shape[0] * grid.shape[1])

    def test_getitem(self):
        grid = self.get_gridxy()
        self.assertEqual(grid.ndim, 2)
        sub = grid[2, 1]
        self.assertNotIn('point', sub.parent)
        self.assertNotIn('polygon', sub.parent)
        self.assertEqual(sub.x.value, 102.)
        self.assertEqual(sub.y.value, 42.)

        # Test with two-dimensional x and y values.
        grid = self.get_gridxy(with_2d_variables=True)
        sub = grid[1:3, 1:3]
        actual_x = [[102.0, 103.0], [102.0, 103.0]]
        self.assertEqual(sub.x.value.tolist(), actual_x)
        actual_y = [[41.0, 41.0], [42.0, 42.0]]
        self.assertEqual(sub.y.value.tolist(), actual_y)

        # Test with parent.
        grid = self.get_gridxy(with_parent=True)
        self.assertEqual(id(grid.x.parent), id(grid.y.parent))
        orig_tas = grid.parent['tas'].value[slice(None), slice(1, 2), slice(2, 4)]
        orig_rhs = grid.parent['rhs'].value[slice(2, 4), slice(1, 2), slice(None)]
        self.assertEqual(grid.shape, (4, 3))

        sub = grid[2:4, 1]
        self.assertEqual(grid.shape, (4, 3))
        self.assertEqual(sub.parent['tas'].shape, (10, 1, 2))
        self.assertEqual(sub.parent['rhs'].shape, (2, 1, 10))
        self.assertNumpyAll(sub.parent['tas'].value, orig_tas)
        self.assertNumpyAll(sub.parent['rhs'].value, orig_rhs)
        self.assertTrue(np.may_share_memory(sub.parent['tas'].value, grid.parent['tas'].value))

    def test_get_mask(self):
        grid = self.get_gridxy()
        self.assertTrue(grid.is_vectorized)
        mask = grid.get_mask()
        self.assertEqual(mask.ndim, 2)
        self.assertFalse(np.any(mask))
        self.assertTrue(grid.is_vectorized)

    @attr('mpi')
    def test_get_intersects_bounds_sequence(self):
        keywords = dict(bounds=[True, False], use_bounds=[True, False])

        for k in self.iter_product_keywords(keywords):
            y = self.get_variable_y(bounds=k.bounds)
            x = self.get_variable_x(bounds=k.bounds)
            grid = GridXY(x, y)
            bounds_sequence = (-99, 39, -98, 39)
            bg = grid.get_intersects(bounds_sequence)
            if MPI_RANK == 0:
                self.assertNotEqual(grid.shape, bg.shape)
            self.assertTrue(bg.is_vectorized)

            with self.assertRaises(EmptySubsetError):
                bounds_sequence = (1000, 1000, 1001, 10001)
                grid.get_intersects(bounds_sequence, use_bounds=k.use_bounds)

            bounds_sequence = (-99999, 1, 1, 1000)
            bg2 = grid.get_intersects(bounds_sequence, use_bounds=k.use_bounds)

            if MPI_RANK == 0:
                for target in ['x', 'y']:
                    original = getattr(grid, target).value
                    sub = getattr(bg2, target).value
                    self.assertNumpyAll(original, sub)

        # Test mask is not shared with subsetted grid.
        grid = self.get_gridxy()
        new_mask = grid.get_mask()
        new_mask[:, 1] = True
        grid.set_mask(new_mask)
        self.assertFalse(grid.has_bounds)

        bounds_sequence = (101.5, 40.5, 103.5, 42.5)
        sub = grid.get_intersects(bounds_sequence, use_bounds=False)
        if MPI_RANK == 0:
            new_mask = sub.get_mask()
            new_mask.fill(True)
            sub.set_mask(new_mask)
            self.assertEqual(grid.get_mask().sum(), 4)

    def test_get_intersects(self):
        subset = box(100.7, 39.71, 102.30, 42.30)
        desired_manual = [[[40.0, 40.0], [41.0, 41.0], [42.0, 42.0]],
                          [[101.0, 102.0], [101.0, 102.0], [101.0, 102.0]]]
        desired_manual = np.array(desired_manual)

        grid = self.get_gridxy()
        self.write_fiona_htmp(grid, 'grid')
        self.write_fiona_htmp(GeometryVariable(value=subset), 'subset')
        sub, sub_slc = grid.get_intersects(subset, return_slice=True)
        self.write_fiona_htmp(sub, 'sub')
        self.assertEqual(sub_slc, (slice(0, 3, None), slice(0, 2, None)))
        self.assertNumpyAll(sub.value_stacked, desired_manual)

        # Test masks are updated.
        grid = self.get_gridxy(with_xy_bounds=True, with_parent=True)
        for t in ['xbounds', 'ybounds']:
            self.assertIn(t, grid.parent)
        subset = 'Polygon ((100.81193771626298883 42.17577854671281301, 101.13166089965399408 42.21211072664360842, 101.34965397923876651 41.18754325259516236, 103.68944636678200766 41.34013840830451159, 103.63858131487890546 41.22387543252595776, 100.77560553633219342 41.08581314878893664, 100.81193771626298883 42.17577854671281301))'
        subset = wkt.loads(subset)
        sub = grid.get_intersects(subset, cascade=True)
        self.assertTrue(sub.get_mask().any())
        self.assertTrue(sub.abstraction_geometry.get_mask().any())
        mask_slice = {'ydim': slice(1, 2), 'xdim': slice(1, 3)}
        for v in sub.parent.values():
            self.assertTrue(v.get_mask().any())
            self.assertFalse(v.get_mask().all())
            actual = v[mask_slice].get_mask()
            self.assertTrue(np.all(actual))

    def test_get_value_polygons(self):
        """Test ordering of vertices when creating from corners is slightly different."""

        keywords = dict(with_bounds=[False, True])
        for k in self.iter_product_keywords(keywords, as_namedtuple=True):
            grid = self.get_polygon_array_grid(with_bounds=k.with_bounds)
            self.assertTrue(grid.is_vectorized)
            if not k.with_bounds:
                grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
            # tdk: rename this
            actual = self.polygon_value_alternate_ordering
            fill = get_geometry_fill(grid.shape, grid.get_mask())
            poly = GeometryVariable(value=get_polygon_geometry_array(grid, fill))
            self.assertGeometriesAlmostEquals(poly, GeometryVariable(value=actual))
            self.assertTrue(grid.is_vectorized)

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

        for should_extrapolate in [False, True]:
            y = Variable(name='y', value=value_grid[0], dimensions=['ydim', 'xdim'])
            x = Variable(name='x', value=value_grid[1], dimensions=['ydim', 'xdim'])
            self.assertIsNone(y.parent)
            self.assertIsNone(x.parent)
            if should_extrapolate:
                y.set_extrapolated_bounds('ybounds', 'bounds')
                x.set_extrapolated_bounds('xbounds', 'bounds')
            grid = GridXY(x, y)
            try:
                grid.set_extrapolated_bounds('ybounds', 'xbounds', 'bounds')
            except BoundsAlreadyAvailableError:
                self.assertTrue(should_extrapolate)
            else:
                np.testing.assert_equal(grid.y.bounds.value, actual_corners[0])
                np.testing.assert_equal(grid.x.bounds.value, actual_corners[1])

        # Test vectorized.
        y = Variable(name='y', value=[1., 2., 3.], dimensions='yy')
        x = Variable(name='x', value=[10., 20., 30.], dimensions='xx')
        grid = GridXY(x, y)
        grid.set_extrapolated_bounds('ybounds', 'xbounds', 'bounds')
        self.assertTrue(grid.is_vectorized)
        self.assertEqual(grid.x.bounds.ndim, 3)

    def test_setitem(self):
        grid = self.get_gridxy()
        self.assertNotIn('point', grid.parent)
        self.assertFalse(np.any(grid.get_mask()))
        grid2 = deepcopy(grid)
        grid2.x[:] = Variable(value=111, mask=True)
        grid2.y[:] = Variable(value=222, mask=True)
        grid2.point
        self.assertIn('ocgis_point', grid2.parent)
        grid[:, :] = grid2
        self.assertTrue(np.all(grid.get_mask()))
        self.assertIn('ocgis_point', grid.parent)
        self.assertEqual(grid.x.value.mean(), 111)
        self.assertEqual(grid.y.value.mean(), 222)

    def test_set_mask(self):
        grid = self.get_gridxy()
        grid.parent['coordinate_system'] = Variable(name='coordinate_system')
        self.assertIsNotNone(grid.point)
        self.assertFalse(np.any(grid.get_mask()))
        mask = np.zeros(grid.shape, dtype=bool)
        mask[1, 1] = True
        self.assertTrue(grid.is_vectorized)
        grid.set_mask(mask)
        self.assertTrue(grid.is_vectorized)
        self.assertTrue(np.all(grid.y.get_mask()[1, 1]))
        self.assertTrue(np.all(grid.x.get_mask()[1, 1]))
        self.assertTrue(np.all(grid.point.get_mask()[1, 1]))
        self.assertIn('coordinate_system', grid.parent)

        path = self.get_temporary_file_path('foo.nc')
        grid.write_netcdf(path)
        # self.ncdump(path)
        nvc = VariableCollection.read_netcdf(path)
        ngrid = GridXY(nvc['x'], nvc['y'], parent=nvc)
        # Mask is not written to coordinate variables.
        self.assertFalse(ngrid.get_mask()[1, 1])
        self.assertNumpyNotAll(grid.masked_value_stacked.compressed(), ngrid.masked_value_stacked.compressed())

        # Test with a parent.
        grid = self.get_gridxy(with_parent=True)
        for k in ['tas', 'rhs']:
            self.assertFalse(grid.parent[k].get_mask().any())
        new_mask = grid.get_mask()
        self.assertFalse(new_mask.any())
        new_mask[1:3, 1] = True
        grid.set_mask(new_mask, cascade=True)
        for k in ['tas', 'rhs']:
            backref_var = grid.parent[k]
            mask = backref_var.get_mask()
            self.assertTrue(mask.any())
            if k == 'tas':
                self.assertTrue(mask[:, 1, 1:3].all())
            if k == 'rhs':
                self.assertTrue(mask[1:3, 1, :].all())
            self.assertEqual(mask.sum(), 20)

    def test_shape(self):
        for grid in self.get_iter_gridxy():
            self.assertEqual(grid.shape, (4, 3))
            self.assertEqual(grid.ndim, 2)

    def test_update_crs(self):
        grid = self.get_gridxy(crs=WGS84())
        grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
        self.assertIsNotNone(grid.y.bounds)
        self.assertIsNotNone(grid.x.bounds)
        to_crs = CoordinateReferenceSystem(epsg=3395)
        grid.update_crs(to_crs)
        self.assertEqual(grid.crs, to_crs)
        for element in [grid.x, grid.y]:
            for target in [element.value, element.bounds.value]:
                self.assertTrue(np.all(target > 10000))

    def test_write_netcdf(self):
        grid = self.get_gridxy(crs=WGS84())
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        self.assertTrue(grid.is_vectorized)
        with self.nc_scope(path) as ds:
            self.assertNumpyAll(ds.variables[grid.x.name][:], grid.x.value[0, :])
            var = ds.variables[grid.y.name]
            self.assertNumpyAll(var[:], grid.y.value[:, 0])
            self.assertEqual(var.axis, 'Y')
            self.assertIn(grid.crs.name, ds.variables)

        # Test with 2-d x and y arrays.
        grid = self.get_gridxy(with_2d_variables=True)
        path = self.get_temporary_file_path('out.nc')
        grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        # self.ncdump(path)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            self.assertNumpyAll(var[:], grid.y.value)

        grid = self.get_gridxy()
        self.assertIsNotNone(grid.dimensions)
        self.assertTrue(grid.is_vectorized)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path, 'r') as ds:
            self.assertEqual(['ydim'], [d for d in ds.variables['y'].dimensions])
            self.assertEqual(['xdim'], [d for d in ds.variables['x'].dimensions])
