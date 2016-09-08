import itertools
from collections import OrderedDict

import fiona
import numpy as np
from nose.plugins.skip import SkipTest
from numpy.ma import MaskedArray
from shapely import wkt
from shapely.geometry import Point, box, MultiPoint, LineString, MultiPolygon
from shapely.geometry.multilinestring import MultiLineString

from ocgis import env, CoordinateReferenceSystem
from ocgis.exc import EmptySubsetError
from ocgis.interface.base.crs import WGS84
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.grid import GridXY, get_geometry_variable, get_point_geometry_array, get_polygon_geometry_array
from ocgis.new_interface.mpi import OcgMpi, MPI_RANK, variable_scatter, MPI_SIZE
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable, VariableCollection
from ocgis.test.base import attr
from ocgis.util.spatial.index import SpatialIndex


class TestGeometryVariable(AbstractTestNewInterface):
    def get_geometryvariable_with_parent(self):
        vpa = np.array([None, None, None])
        vpa[:] = [Point(1, 2), Point(3, 4), Point(5, 6)]
        value = np.arange(0, 30).reshape(10, 3)
        tas = Variable(name='tas', value=value)
        tas.create_dimensions(['time', 'ngeom'])
        backref = VariableCollection(variables=[tas])
        pa = GeometryVariable(value=vpa, parent=backref, name='point', dimensions='ngeom')
        backref[pa.name] = pa
        return pa

    def test_init(self):
        # Test empty.
        gvar = GeometryVariable()
        self.assertEqual(gvar.dtype, object)

        gvar = self.get_geometryvariable()
        self.assertIsInstance(gvar.masked_value, MaskedArray)
        self.assertEqual(gvar.ndim, 1)

        # Test passing a "crs".
        gvar = self.get_geometryvariable(crs=WGS84(), name='my_geom', dimensions='ngeom')
        self.assertEqual(gvar.crs, WGS84())

        # Test using lines.
        line1 = LineString([(0, 0), (1, 1)])
        line2 = LineString([(1, 1), (2, 2)])
        gvar = GeometryVariable(value=[line1, line2])
        self.assertTrue(gvar.value[1].almost_equals(line2))
        self.assertEqual(gvar.geom_type, line1.geom_type)
        lines = MultiLineString([line1, line2])
        lines2 = [lines, lines]
        for actual in [lines, lines2, lines]:
            gvar2 = GeometryVariable(value=actual)
            self.assertTrue(gvar2.value[0].almost_equals(lines))
            self.assertEqual(gvar2.geom_type, lines.geom_type)
            self.assertTrue(gvar2.shape[0] > 0)
            self.assertFalse(gvar2.get_mask().any())

    def test_area(self):
        gvar = self.get_geometryvariable()
        self.assertTrue(np.all(gvar.area == 0))

    def test_crs(self):
        crs = WGS84()
        gvar = GeometryVariable(crs=crs, name='var')
        self.assertEqual(gvar.crs, crs)
        self.assertIn(crs.name, gvar.parent)
        gvar.crs = None
        self.assertIsNone(gvar.crs)
        self.assertEqual(len(gvar.parent), 1)
        self.assertNotIn(crs.name, gvar.parent)

        # Test coordinate system is maintained.
        gvar = GeometryVariable(crs=crs, name='var')
        vc = VariableCollection(variables=gvar)
        self.assertIn(crs.name, vc)

    def test_getitem(self):
        gridxy = self.get_gridxy()
        pa = get_geometry_variable(get_point_geometry_array, gridxy, name='point_array', dimensions=['x', 'y'])
        self.assertEqual(pa.shape, (4, 3))
        self.assertEqual(pa.ndim, 2)
        self.assertIsNotNone(pa._value)
        sub = pa[2:4, 1]
        self.assertEqual(sub.shape, (2, 1))
        self.assertEqual(sub.value.shape, (2, 1))

        # Test slicing with a parent.
        pa = self.get_geometryvariable_with_parent()
        desired_obj = pa.parent['tas']
        self.assertIsNotNone(pa.parent)
        desired = desired_obj[:, 1].value
        self.assertIsNotNone(pa.parent)
        desired_shapes = OrderedDict([('tas', (10, 3)), ('point', (3,))])
        self.assertEqual(pa.parent.shapes, desired_shapes)

        sub = pa[1]
        backref_tas = sub.parent['tas']
        self.assertNumpyAll(backref_tas.value, desired)
        self.assertEqual(backref_tas.shape, (10, 1))

    def test_geom_type(self):
        gvar = GeometryVariable(value=Point(1, 2))
        self.assertEqual(gvar.geom_type, 'Point')

        # Test with a multi-geometry.
        mp = np.array([None])
        mp[0] = MultiPoint([Point(1, 2), Point(3, 4)])
        pa = self.get_geometryvariable(value=mp)
        self.assertEqual(pa.geom_type, 'MultiPoint')

        # Test overloading.
        pa = self.get_geometryvariable(value=mp, geom_type='overload')
        self.assertEqual(pa.geom_type, 'overload')

    @attr('mpi')
    def test_get_intersects(self):
        # tdk: RESUME: test empty subsets
        dist = OcgMpi()
        dimx = dist.create_dimension('x', 5, dist=False)
        dimy = dist.create_dimension('y', 5, dist=True)
        x = dist.create_variable(name='x', dimensions=[dimx, dimy])
        y = dist.create_variable(name='y', dimensions=[dimx, dimy])
        dist.update_dimension_bounds()

        if MPI_RANK == 0:
            x = Variable(value=[1, 2, 3, 4, 5], name='x', dimensions=['x'])
            y = Variable(value=[10, 20, 30, 40, 50], name='y', dimensions=['y'])

        x, dist = variable_scatter(x, dist)
        y, dist = variable_scatter(y, dist)

        self.assertTrue(y.dimensions[0].dist)

        grid = GridXY(x=x, y=y)
        self.assertTrue(grid.dimensions[0].dist)
        pa = get_geometry_variable(get_point_geometry_array, grid, name='points', dimensions=['y', 'x'])
        polygon = box(2.5, 15, 4.5, 45)
        self.assertTrue(pa.dimensions[0].dist)

        # if MPI_RANK == 0:
        #     self.write_fiona_htmp(GeometryVariable(value=polygon), 'polygon')
        # self.write_fiona_htmp(grid.abstraction_geometry, 'grid-{}'.format(MPI_RANK))

        # Try en empty subset.
        with self.assertRaises(EmptySubsetError):
            pa.get_intersects(Point(-8000, 9000))

        sub, slc = pa.get_intersects(polygon, return_slice=True)

        # self.write_fiona_htmp(sub, 'sub-{}'.format(MPI_RANK))

        if MPI_SIZE == 1:
            self.assertEqual(sub.shape, (3, 2))
        else:
            # This is the non-distributed dimension.
            if MPI_SIZE == 2:
                self.assertEqual(sub.shape[1], 2)
            # This is the distributed dimension.
            if MPI_RANK < 5:
                self.assertNotEqual(sub.shape[0], 3)
            else:
                self.assertTrue(sub.is_empty)

        desired_points_manual = [Point(x, y) for x, y in itertools.product(grid.x.value.flat, grid.y.value.flat)]
        desired_points_manual = [pt for pt in desired_points_manual if pt.intersects(polygon)]
        desired_points_slc = pa.get_distributed_slice(slc).value.flat
        for desired_points in [desired_points_manual, desired_points_slc]:
            for pt in desired_points:
                found = False
                for pt_actual in sub.value.flat:
                    if pt_actual.almost_equals(pt):
                        found = True
                        break
                self.assertTrue(found)

    def test_tdk(self):
        # Test w/out an associated grid.
        pa = self.get_geometryvariable()
        polygon = box(0.5, 1.5, 1.5, 2.5)
        sub = pa.get_intersects(polygon)
        self.assertEqual(sub.shape, (1,))
        self.assertEqual(sub.value[0], Point(1, 2))

        # Test no masked data is returned.
        snames = ['Hawaii', 'Utah', 'France']
        snames = Variable(name='state_names', value=snames, dimensions='ngeom')
        snames.create_dimensions('ngeom')
        backref = VariableCollection(variables=snames)
        pts = [Point(1, 2), Point(3, 4), Point(4, 5)]
        subset = MultiPolygon([Point(1, 2).buffer(0.1), Point(4, 5).buffer(0.1)])
        gvar = GeometryVariable(value=pts, parent=backref, dimensions='ngeom', name='points')
        gvar.create_dimensions('ngeom')
        sub, slc = gvar.get_intersects(subset, return_slice=True)
        self.assertFalse(sub.get_mask().any())

        desired = snames.value[slc]
        actual = sub.parent[snames.name].value
        self.assertNumpyAll(actual, desired)

    def test_get_intersection(self):
        for return_indices in [True, False]:
            pa = self.get_geometryvariable()
            polygon = box(0.9, 1.9, 1.5, 2.5)
            lhs = pa.get_intersection(polygon, return_slice=return_indices)
            if return_indices:
                lhs, slc = lhs
                # self.assertEqual(slc, (slice(0, -1, None),))
            self.assertEqual(lhs.shape, (1,))
            self.assertEqual(lhs.value[0], Point(1, 2))
            if return_indices:
                self.assertEqual(pa.value[slc][0], Point(1, 2))

    @attr('mpi')
    def test_get_intersects_masked(self):
        poly = wkt.loads(
            'POLYGON((-98.26574367088608142 40.19952531645570559,-98.71764240506330168 39.54825949367089066,-99.26257911392406186 39.16281645569620906,-99.43536392405064817 38.64446202531645724,-98.78409810126584034 38.33876582278481493,-98.23916139240508016 37.71408227848101546,-97.77397151898735217 37.67420886075949937,-97.62776898734178133 38.15268987341772799,-98.39865506329114453 38.52484177215190186,-98.23916139240508016 39.33560126582278826,-97.73409810126582897 39.58813291139241386,-97.52143987341773368 40.27927215189873777,-97.52143987341773368 40.27927215189873777,-98.26574367088608142 40.19952531645570559))')
        desired_mask = np.array([[True, True, False, True],
                                 [True, False, True, True],
                                 [True, True, False, True]])

        dist = OcgMpi()
        xdim = dist.create_dimension('x', 4, dist=True)
        ydim = dist.create_dimension('y', 3)
        dist.create_dimension('bounds', 2)
        dist.update_dimension_bounds()

        if MPI_RANK == 0:
            x = self.get_variable_x()
            y = self.get_variable_y()
            grid = GridXY(x=x, y=y)
            pa = get_geometry_variable(get_point_geometry_array, grid, crs=WGS84(), name='points',
                                       dimensions=['y', 'x'])
        else:
            pa = None
        pa, dist = variable_scatter(pa, dist)

        keywords = dict(use_spatial_index=[True, False])
        for k in self.iter_product_keywords(keywords):
            ret = pa.get_intersects_masked(poly, use_spatial_index=k.use_spatial_index)
            desired_mask_local = desired_mask[slice(*ydim.bounds_local), slice(*xdim.bounds_local)]
            if MPI_RANK > 3:
                self.assertTrue(pa.is_empty)
            else:
                self.assertNumpyAll(desired_mask_local, ret.get_mask())
                for element in ret.value.flat:
                    self.assertIsInstance(element, Point)

            # This does not test a parallel operation.
            if MPI_RANK == 0:
                # Test pre-masked values in geometry are okay for intersects operation.
                value = [Point(1, 1), Point(2, 2), Point(3, 3)]
                value = np.ma.array(value, mask=[False, True, False], dtype=object)
                pa2 = GeometryVariable(value=value)
                b = box(0, 0, 5, 5)
                res = pa2.get_intersects_masked(b, use_spatial_index=k.use_spatial_index)
                self.assertNumpyAll(res.get_mask(), value.mask)

    def test_get_intersection_masked(self):
        pa = self.get_geometryvariable()
        polygon = box(0.9, 1.9, 1.5, 2.5)
        lhs = pa.get_intersection_masked(polygon)
        self.assertTrue(lhs.get_mask()[1])

    def test_get_nearest(self):
        target1 = Point(0.5, 0.75)
        target2 = box(0.5, 0.75, 0.55, 0.755)
        pa = self.get_geometryvariable()
        for target in [target1, target2]:
            res, slc = pa.get_nearest(target, return_indices=True)
            self.assertIsInstance(res, GeometryVariable)
            self.assertEqual(res.value[0], Point(1, 2))
            self.assertEqual(slc, (0,))
            self.assertEqual(res.shape, (1,))

    def test_get_spatial_index(self):
        pa = self.get_geometryvariable()
        si = pa.get_spatial_index()
        self.assertIsInstance(si, SpatialIndex)
        self.assertEqual(si._index.bounds, [1.0, 2.0, 3.0, 4.0])

    def test_get_unioned(self):
        pa = self.get_geometryvariable()
        u = pa.get_unioned()
        self.assertEqual(u.shape, (1,))
        actual = MultiPoint([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(u.value[0], actual)

    def test_update_crs(self):
        pa = self.get_geometryvariable(crs=WGS84(), name='g', dimensions='gg')
        to_crs = CoordinateReferenceSystem(epsg=2136)
        pa.update_crs(to_crs)
        self.assertEqual(pa.crs, to_crs)
        v0 = [1629871.494956261, -967769.9070825744]
        v1 = [2358072.3857447207, -239270.87548993886]
        np.testing.assert_almost_equal(pa.value[0], v0)
        np.testing.assert_almost_equal(pa.value[1], v1)

    def test_weights(self):
        value = [Point(2, 3), Point(4, 5), Point(5, 6)]
        mask = [False, True, False]
        value = np.ma.array(value, mask=mask, dtype=object)
        pa = self.get_geometryvariable(value=value)
        self.assertNumpyAll(pa.weights, np.ma.array([1, 1, 1], mask=mask, dtype=env.NP_FLOAT))

    def test_write_fiona(self):
        pa = self.get_geometryvariable(crs=WGS84(), name='g', dimensions='gg')
        path = self.get_temporary_file_path('foo.shp')
        pa.write_fiona(path)
        with fiona.open(path, 'r') as records:
            self.assertEqual(len(list(records)), 2)

        # Test with a multi-geometry.
        pts = np.ma.array([Point(1, 2), MultiPoint([Point(3, 4), Point(5, 6)])], dtype=object, mask=False)
        pa = GeometryVariable(value=pts)
        pa.write_fiona(path)
        with fiona.open(path) as source:
            self.assertEqual(source.schema['geometry'], 'MultiPoint')


class TestGeometryVariablePolygons(AbstractTestNewInterface):
    """Test a geometry variable using polygons."""

    def test_init(self):
        row = Variable(value=[2, 3], name='row', dimensions='y')
        col = Variable(value=[4, 5], name='col', dimensions='x')
        grid = GridXY(col, row)
        self.assertIsNone(grid._archetype.bounds)

        row = Variable(value=[2, 3], name='row', dimensions='y')
        row.set_extrapolated_bounds('row_bounds', 'y')
        col = Variable(value=[4, 5], name='col', dimensions='x')
        col.set_extrapolated_bounds('col_bounds', 'x')
        grid = GridXY(y=row, x=col)
        poly = get_geometry_variable(get_polygon_geometry_array, grid, name='geom', dimensions=['row', 'col'])
        self.assertEqual(poly.name, 'geom')
        self.assertEqual(poly.geom_type, 'Polygon')

    def test_area_and_weights(self):
        poly = self.get_polygonarray()
        bbox = box(-98.1, 38.3, -99.4, 39.9)

        sub = poly.get_intersection(bbox)
        actual_area = [[0.360000000000001, 0.1600000000000017], [0.9000000000000057, 0.4000000000000057],
                       [0.18000000000000368, 0.08000000000000228]]
        np.testing.assert_almost_equal(sub.area, actual_area)
        actual_weights = [[0.3999999999999986, 0.17777777777777853], [1.0, 0.444444444444448],
                          [0.20000000000000284, 0.08888888888889086]]
        np.testing.assert_almost_equal(sub.weights, actual_weights)

    def test_get_unioned(self):
        poly = self.get_polygonarray()
        u = poly.get_unioned()
        for p in poly.value.flat:
            self.assertTrue(p.intersects(u.value[0]))

        raise SkipTest('test backref variables are spatially weighted')
