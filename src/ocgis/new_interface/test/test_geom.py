import itertools
from unittest import SkipTest

import fiona
import numpy as np
from numpy.ma import MaskedArray
from numpy.testing.utils import assert_equal
from shapely import wkt
from shapely.geometry import Point, box, MultiPoint, LineString
from shapely.geometry.multilinestring import MultiLineString

from ocgis import env, CoordinateReferenceSystem
from ocgis.interface.base.crs import WGS84
from ocgis.new_interface.geom import SpatialContainer, GeometryVariable
from ocgis.new_interface.grid import GridXY, get_geometry_variable, get_point_geometry_array, get_polygon_geometry_array
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import BoundedVariable, Variable, VariableCollection
from ocgis.util.spatial.index import SpatialIndex


class TestGeometryVariable(AbstractTestNewInterface):
    def get_geometryvariable_with_backref(self):
        vpa = np.array([None, None, None])
        vpa[:] = [Point(1, 2), Point(3, 4), Point(5, 6)]
        value = np.arange(0, 30).reshape(10, 3)
        tas = Variable(name='tas', value=value)
        tas.create_dimensions(['time', 'ngeom'])
        backref = VariableCollection(variables=[tas])
        pa = GeometryVariable(value=vpa, backref=backref)
        pa.create_dimensions('ngeom')
        return pa

    def test_init(self):
        # Test empty.
        gvar = GeometryVariable()
        self.assertEqual(gvar.dtype, object)

        gvar = self.get_geometryvariable()
        self.assertIsInstance(gvar.masked_value, MaskedArray)
        self.assertEqual(gvar.ndim, 1)

        # Test passing a "crs".
        gvar = self.get_geometryvariable(crs=WGS84())
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

    def test_getitem(self):
        gridxy = self.get_gridxy()
        pa = get_geometry_variable(get_point_geometry_array, gridxy)
        self.assertEqual(pa.shape, (4, 3))
        self.assertEqual(pa.ndim, 2)
        self.assertIsNotNone(pa._value)
        sub = pa[2:4, 1]
        self.assertEqual(sub.shape, (2, 1))
        self.assertEqual(sub.value.shape, (2, 1))

        # Test slicing with a backref.
        pa = self.get_geometryvariable_with_backref()
        desired = pa._backref['tas'][:, 1].value
        sub = pa[1]
        backref_tas = sub._backref['tas']
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

    def test_get_intersects(self):
        x = BoundedVariable(value=[1, 2, 3, 4, 5], name='x')
        y = BoundedVariable(value=[10, 20, 30, 40, 50], name='y')
        grid = GridXY(x=x, y=y)
        pa = get_geometry_variable(get_point_geometry_array, grid)
        polygon = box(2.5, 15, 4.5, 45)
        sub, slc = pa.get_intersects(polygon, return_indices=True)
        self.assertEqual(sub.shape, (3, 2))
        desired_points_manual = [Point(x, y) for x, y in itertools.product(grid.x.value.flat, grid.y.value.flat)]
        desired_points_manual = [pt for pt in desired_points_manual if pt.intersects(polygon)]
        desired_points_slc = pa[slc].value.flat
        for desired_points in [desired_points_manual, desired_points_slc]:
            for pt in desired_points:
                found = False
                for pt_actual in sub.value.flat:
                    if pt_actual.almost_equals(pt):
                        found = True
                        break
                self.assertTrue(found)

        # Test w/out an associated grid.
        pa = self.get_geometryvariable()
        polygon = box(0.5, 1.5, 1.5, 2.5)
        sub = pa.get_intersects(polygon)
        self.assertEqual(sub.shape, (1,))
        self.assertEqual(sub.value[0], Point(1, 2))

    def test_get_intersection(self):
        for return_indices in [True, False]:
            pa = self.get_geometryvariable()
            polygon = box(0.9, 1.9, 1.5, 2.5)
            lhs = pa.get_intersection(polygon, return_indices=return_indices)
            if return_indices:
                lhs, slc = lhs
                # self.assertEqual(slc, (slice(0, -1, None),))
            self.assertEqual(lhs.shape, (1,))
            self.assertEqual(lhs.value[0], Point(1, 2))
            if return_indices:
                self.assertEqual(pa.value[slc][0], Point(1, 2))

    def test_get_intersects_masked(self):
        x = self.get_variable_x()
        y = self.get_variable_y()
        grid = GridXY(x=x, y=y)
        pa = get_geometry_variable(get_point_geometry_array, grid, crs=WGS84())
        poly = wkt.loads(
            'POLYGON((-98.26574367088608142 40.19952531645570559,-98.71764240506330168 39.54825949367089066,-99.26257911392406186 39.16281645569620906,-99.43536392405064817 38.64446202531645724,-98.78409810126584034 38.33876582278481493,-98.23916139240508016 37.71408227848101546,-97.77397151898735217 37.67420886075949937,-97.62776898734178133 38.15268987341772799,-98.39865506329114453 38.52484177215190186,-98.23916139240508016 39.33560126582278826,-97.73409810126582897 39.58813291139241386,-97.52143987341773368 40.27927215189873777,-97.52143987341773368 40.27927215189873777,-98.26574367088608142 40.19952531645570559))')
        actual_mask = np.array([[True, True, False, True], [True, False, True, True], [True, True, False, True]])

        keywords = dict(use_spatial_index=[True, False])

        for k in self.iter_product_keywords(keywords):
            ret = pa.get_intersects_masked(poly, use_spatial_index=k.use_spatial_index)
            self.assertNumpyAll(actual_mask, ret.get_mask())
            for element in ret.value.flat:
                self.assertIsInstance(element, Point)

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
        pa = self.get_geometryvariable(crs=WGS84())
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
        pa = self.get_geometryvariable(crs=WGS84())
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

    def test_write_netcdf(self):
        raise SkipTest


class TestGeometryVariablePolygons(AbstractTestNewInterface):
    """Test a geometry variable using polygons."""

    def test_init(self):
        row = BoundedVariable(value=[2, 3], name='row')
        col = BoundedVariable(value=[4, 5], name='col')
        grid = GridXY(col, row)
        self.assertIsNone(grid._archetype.bounds)

        row = BoundedVariable(value=[2, 3], name='row')
        row.set_extrapolated_bounds()
        col = BoundedVariable(value=[4, 5], name='col')
        col.set_extrapolated_bounds()
        grid = GridXY(y=row, x=col)
        poly = get_geometry_variable(get_polygon_geometry_array, grid, name='geom')
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


class TestSpatialContainer(AbstractTestNewInterface):

    def test_init(self):
        to_crs = CoordinateReferenceSystem(epsg=4326)
        keywords = dict(with_xy_bounds=[False, True])
        for k in self.iter_product_keywords(keywords):
            grid = self.get_gridxy(with_xy_bounds=k.with_xy_bounds)
            grid.crs = CoordinateReferenceSystem(epsg=3395)
            sc = SpatialContainer(grid=grid)
            self.assertIsInstance(sc.point, PointArray)
            self.assertEqual(grid.crs, sc.point.crs)
            polygon = sc.polygon
            optimal_geometry = sc.get_optimal_geometry()
            geom = sc.geom
            sub = sc[0, 1]
            if k.with_xy_bounds:
                self.assertIsInstance(polygon, PolygonArray)
                self.assertIsInstance(optimal_geometry, PolygonArray)
                self.assertIsInstance(geom, PolygonArray)
                self.assertEqual(grid.crs, polygon.crs)
                for target in [sub, sub.grid, sub.polygon, sub.point]:
                    self.assertEqual(target.shape, (1, 1))
            else:
                self.assertIsNone(polygon)
                self.assertIsInstance(optimal_geometry, PointArray)
                self.assertIsInstance(geom, PointArray)
            for target in [sc.point, sc.polygon]:
                try:
                    self.assertIsNone(target._value)
                except AttributeError:
                    self.assertFalse(k.with_xy_bounds)
            sc.update_crs(to_crs)
            if k.with_xy_bounds:
                to_test = [sc.grid, sc.point, sc.polygon]
            else:
                to_test = [sc.grid, sc.point]
                self.assertIsNone(sc.polygon)
            for target in to_test:
                self.assertEqual(to_crs, target.crs)

    def test_get_intersects(self):
        keywords = dict(with_xy_bounds=[False, True], return_indices=[False, True])
        polygon = 'Polygon ((100.82351859861583421 42.13422361591698007, 102.11512759515565563 42.12332396193773576, 102.14782655709338144 40.87531358131486314, 101.85353589965392018 40.88621323529410745, 101.83718641868506438 41.86718209342561892, 101.16140787197224427 41.87263192041523752, 100.99246323529403924 41.41484645328719694, 100.82351859861583421 42.13422361591698007))'
        polygon = wkt.loads(polygon)
        for k in self.iter_product_keywords(keywords):
            grid = self.get_gridxy(with_xy_bounds=k.with_xy_bounds)
            sc = SpatialContainer(grid=grid)
            subset = sc.get_intersects(polygon, return_indices=k.return_indices)
            intersection = sc.get_intersection(polygon)
            nearest = sc.get_nearest(polygon)
            if k.return_indices:
                subset, slc = subset
            self.assertEqual(subset.shape, intersection.shape)
            self.assertEqual(subset.shape, (2, 2))
            self.assertEqual(nearest.shape, (1, 1))
            if k.with_xy_bounds:
                self.assertFalse(subset.point.get_mask().any())
                self.assertFalse(subset.polygon.get_mask().any())
                self.assertFalse(subset.grid.get_mask().any())
            else:
                self.assertIsNone(subset.polygon)
                self.assertTrue(subset.point.get_mask().any())
                self.assertTrue(subset.grid.get_mask().any())
                # subset.grid.set_extrapolated_corners()
                subset.grid.set_extrapolated_bounds()
                # self.assertTrue(subset.grid.corners.mask.any())
                self.assertTrue(subset.grid._archetype.bounds.get_mask().any())
            grid_mask = sc.grid.get_mask().copy()
            grid_mask[:] = True
            self.assertFalse(sc.grid.get_mask().any())
            sc.grid.set_mask(grid_mask)
            self.assertTrue(sc.grid.get_mask().all())
            self.assertFalse(subset.grid.get_mask().all())
            self.assertFalse(subset.point.get_mask().all())

