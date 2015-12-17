import fiona
import numpy as np
from numpy.ma import MaskedArray
from shapely import wkt
from shapely.geometry import Point, box, MultiPoint

from ocgis import env, CoordinateReferenceSystem
from ocgis.interface.base.crs import WGS84
from ocgis.new_interface.geom import PointArray, PolygonArray
from ocgis.new_interface.grid import GridXY
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import BoundedVariable, Variable
from ocgis.test.base import ToTest
from ocgis.util.spatial.index import SpatialIndex


class TestPointArray(AbstractTestNewInterface):
    def test_init(self):
        pa = self.get_pointarray()
        self.assertIsInstance(pa.value, MaskedArray)
        self.assertEqual(pa.ndim, 1)

        # Test initializing with a grid object.
        gridxy = self.get_gridxy()
        # Enforce value as none.
        pa = self.get_pointarray(grid=gridxy, value=None)
        self.assertIsInstance(pa.value, MaskedArray)
        self.assertEqual(pa.ndim, 2)

        # Test passing a "crs".
        pa = self.get_pointarray(grid=self.get_gridxy(), crs=WGS84())
        self.assertEqual(pa.crs, WGS84())

    def test_getitem(self):
        gridxy = self.get_gridxy()
        pa = self.get_pointarray(grid=gridxy, value=None)
        self.assertEqual(pa.shape, (4, 3))
        self.assertEqual(pa.ndim, 2)
        self.assertIsNone(pa._value)
        sub = pa[2:4, 1]
        self.assertEqual(sub.shape, (2, 1))
        self.assertEqual(sub.value.shape, (2, 1))
        self.assertEqual(sub.grid.shape, (2, 1))

    def test_set_value(self):
        value = Point(6, 7)
        with self.assertRaises(ValueError):
            self.get_pointarray(value=value)

    def test_geom_type(self):
        gridxy = self.get_gridxy()
        pa = self.get_pointarray(grid=gridxy)
        self.assertEqual(pa.geom_type, 'Point')

        # Test with a multi-geometry.
        mp = np.array([None])
        mp[0] = MultiPoint([Point(1, 2), Point(3, 4)])
        pa = self.get_pointarray(value=mp)
        self.assertEqual(pa.geom_type, 'MultiPoint')

        # Test overloading.
        pa = self.get_pointarray(value=mp, geom_type='overload')
        self.assertEqual(pa.geom_type, 'overload')

    def test_get_intersects(self):
        x = BoundedVariable(value=[1, 2, 3, 4, 5], name='x')
        y = BoundedVariable(value=[10, 20, 30, 40, 50], name='y')
        grid = GridXY(x=x, y=y)
        pa = PointArray(grid=grid)
        polygon = box(2.5, 15, 4.5, 45)
        sub, slc = pa.get_intersects(polygon, return_indices=True)
        np.testing.assert_equal(sub.grid.x.value, [3, 4])
        np.testing.assert_equal(sub.grid.y.value, [20, 30, 40])
        self.assertNumpyAll(grid[slc].value, sub.grid.value)

        # Test w/out an associated grid.
        pa = self.get_pointarray()
        self.assertIsNone(pa.grid)
        polygon = box(0.5, 1.5, 1.5, 2.5)
        sub = pa.get_intersects(polygon)
        self.assertIsNone(sub.grid)
        self.assertEqual(sub.shape, (1,))
        self.assertEqual(sub.value[0], Point(1, 2))

    def test_get_intersection(self):
        for return_indices in [True, False]:
            pa = self.get_pointarray()
            polygon = box(0.9, 1.9, 1.5, 2.5)
            lhs = pa.get_intersection(polygon, return_indices=return_indices)
            if return_indices:
                lhs, slc = lhs
                self.assertEqual(slc, (slice(0, -1, None),))
            self.assertEqual(lhs.shape, (1,))
            self.assertEqual(lhs.value[0], Point(1, 2))

    def test_get_intersects_masked(self):
        x = self.get_variable_x()
        y = self.get_variable_y()
        grid = GridXY(x=x, y=y)
        pa = PointArray(grid=grid, crs=WGS84())
        self.assertIsNotNone(pa.grid)
        poly = wkt.loads(
            'POLYGON((-98.26574367088608142 40.19952531645570559,-98.71764240506330168 39.54825949367089066,-99.26257911392406186 39.16281645569620906,-99.43536392405064817 38.64446202531645724,-98.78409810126584034 38.33876582278481493,-98.23916139240508016 37.71408227848101546,-97.77397151898735217 37.67420886075949937,-97.62776898734178133 38.15268987341772799,-98.39865506329114453 38.52484177215190186,-98.23916139240508016 39.33560126582278826,-97.73409810126582897 39.58813291139241386,-97.52143987341773368 40.27927215189873777,-97.52143987341773368 40.27927215189873777,-98.26574367088608142 40.19952531645570559))')
        actual_mask = np.array([[True, True, False, True], [True, False, True, True], [True, True, False, True]])

        keywords = dict(use_spatial_index=[True, False])

        for k in self.iter_product_keywords(keywords):
            ret = pa.get_intersects_masked(poly, use_spatial_index=k.use_spatial_index)
            self.assertNumpyAll(actual_mask, ret.value.mask)
            for element in ret.value.data.flat:
                self.assertIsInstance(element, Point)
            self.assertIsNotNone(pa.grid)

            # Test pre-masked values in geometry are okay for intersects operation.
            value = [Point(1, 1), Point(2, 2), Point(3, 3)]
            value = np.ma.array(value, mask=[False, True, False], dtype=object)
            pa2 = PointArray(value=value)
            b = box(0, 0, 5, 5)
            res = pa2.get_intersects_masked(b, use_spatial_index=k.use_spatial_index)
            self.assertNumpyAll(res.value.mask, value.mask)

    def test_get_intersection_masked(self):
        pa = self.get_pointarray()
        polygon = box(0.9, 1.9, 1.5, 2.5)
        lhs = pa.get_intersection_masked(polygon)
        self.assertTrue(lhs.value.mask[1])

    def test_get_nearest(self):
        target1 = Point(0.5, 0.75)
        target2 = box(0.5, 0.75, 0.55, 0.755)
        pa = self.get_pointarray()
        for target in [target1, target2]:
            res = pa.get_nearest(target, return_index=True)
            self.assertEqual(res, (Point(1, 2), 0))

    def test_get_spatial_index(self):
        pa = self.get_pointarray()
        si = pa.get_spatial_index()
        self.assertIsInstance(si, SpatialIndex)
        self.assertEqual(si._index.bounds, [1.0, 2.0, 3.0, 4.0])

    def test_get_unioned(self):
        pa = self.get_pointarray()
        u = pa.get_unioned()
        self.assertEqual(u.shape, (1,))
        actual = MultiPoint([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(u.value[0], actual)

    def test_update_crs(self):
        pa = self.get_pointarray(crs=WGS84())
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
        pa = self.get_pointarray(value=value)
        self.assertNumpyAll(pa.weights, np.ma.array([1, 1, 1], mask=mask, dtype=env.NP_FLOAT))

    def test_write_fiona(self):
        pa = self.get_pointarray(crs=WGS84())
        path = self.get_temporary_file_path('foo.shp')
        pa.write_fiona(path)
        with fiona.open(path, 'r') as records:
            self.assertEqual(len(list(records)), 2)

    def test_write_netcdf(self):
        raise ToTest


class TestPolygonArray(AbstractTestNewInterface):
    @property
    def polygon_value(self):
        polys = [['POLYGON ((-100.5 39.5, -100.5 40.5, -99.5 40.5, -99.5 39.5, -100.5 39.5))',
                  'POLYGON ((-99.5 39.5, -99.5 40.5, -98.5 40.5, -98.5 39.5, -99.5 39.5))',
                  'POLYGON ((-98.5 39.5, -98.5 40.5, -97.5 40.5, -97.5 39.5, -98.5 39.5))',
                  'POLYGON ((-97.5 39.5, -97.5 40.5, -96.5 40.5, -96.5 39.5, -97.5 39.5))'],
                 ['POLYGON ((-100.5 38.5, -100.5 39.5, -99.5 39.5, -99.5 38.5, -100.5 38.5))',
                  'POLYGON ((-99.5 38.5, -99.5 39.5, -98.5 39.5, -98.5 38.5, -99.5 38.5))',
                  'POLYGON ((-98.5 38.5, -98.5 39.5, -97.5 39.5, -97.5 38.5, -98.5 38.5))',
                  'POLYGON ((-97.5 38.5, -97.5 39.5, -96.5 39.5, -96.5 38.5, -97.5 38.5))'],
                 ['POLYGON ((-100.5 37.5, -100.5 38.5, -99.5 38.5, -99.5 37.5, -100.5 37.5))',
                  'POLYGON ((-99.5 37.5, -99.5 38.5, -98.5 38.5, -98.5 37.5, -99.5 37.5))',
                  'POLYGON ((-98.5 37.5, -98.5 38.5, -97.5 38.5, -97.5 37.5, -98.5 37.5))',
                  'POLYGON ((-97.5 37.5, -97.5 38.5, -96.5 38.5, -96.5 37.5, -97.5 37.5))']]
        return self.get_shapely_from_wkt_array(polys)

    @property
    def polygon_value_alternate_ordering(self):
        polys = [['POLYGON ((-100.5 40.5, -99.5 40.5, -99.5 39.5, -100.5 39.5, -100.5 40.5))',
                  'POLYGON ((-99.5 40.5, -98.5 40.5, -98.5 39.5, -99.5 39.5, -99.5 40.5))',
                  'POLYGON ((-98.5 40.5, -97.5 40.5, -97.5 39.5, -98.5 39.5, -98.5 40.5))',
                  'POLYGON ((-97.5 40.5, -96.5 40.5, -96.5 39.5, -97.5 39.5, -97.5 40.5))'],
                 ['POLYGON ((-100.5 39.5, -99.5 39.5, -99.5 38.5, -100.5 38.5, -100.5 39.5))',
                  'POLYGON ((-99.5 39.5, -98.5 39.5, -98.5 38.5, -99.5 38.5, -99.5 39.5))',
                  'POLYGON ((-98.5 39.5, -97.5 39.5, -97.5 38.5, -98.5 38.5, -98.5 39.5))',
                  'POLYGON ((-97.5 39.5, -96.5 39.5, -96.5 38.5, -97.5 38.5, -97.5 39.5))'],
                 ['POLYGON ((-100.5 38.5, -99.5 38.5, -99.5 37.5, -100.5 37.5, -100.5 38.5))',
                  'POLYGON ((-99.5 38.5, -98.5 38.5, -98.5 37.5, -99.5 37.5, -99.5 38.5))',
                  'POLYGON ((-98.5 38.5, -97.5 38.5, -97.5 37.5, -98.5 37.5, -98.5 38.5))',
                  'POLYGON ((-97.5 38.5, -96.5 38.5, -96.5 37.5, -97.5 37.5, -97.5 38.5))']]
        return self.get_shapely_from_wkt_array(polys)

    def get_polygonarray(self):
        yb = Variable(value=[[40.5, 39.5], [39.5, 38.5], [38.5, 37.5]], name='yb')
        y = BoundedVariable(value=[40.0, 39.0, 38.0], name='y', bounds=yb)
        xb = Variable(value=[[-100.5, -99.5], [-99.5, -98.5], [-98.5, -97.5], [-97.5, -96.5]], name='xb')
        x = BoundedVariable(value=[-100.0, -99.0, -98.0, -97.0], bounds=xb, name='x')
        grid = GridXY(x=x, y=y)
        poly = PolygonArray(grid=grid)
        return poly

    def get_shapely_from_wkt_array(self, wkts):
        ret = np.array(wkts)
        vfunc = np.vectorize(wkt.loads, otypes=[object])
        ret = vfunc(ret)
        ret = np.ma.array(ret, mask=False)
        return ret

    def test_init(self):
        with self.assertRaises(ValueError):
            PolygonArray()

        row = BoundedVariable(value=[2, 3], name='row')
        col = BoundedVariable(value=[4, 5], name='col')
        grid = GridXY(x=col, y=row)
        self.assertIsNone(grid.corners)
        # Corners are not available.
        with self.assertRaises(ValueError):
            PolygonArray(grid=grid)

        value = grid.value
        grid = GridXY(value=value)
        # Corners are not available.
        with self.assertRaises(ValueError):
            PolygonArray(grid=grid)

        row = BoundedVariable(value=[2, 3], name='row')
        row.set_extrapolated_bounds()
        col = BoundedVariable(value=[4, 5], name='col')
        col.set_extrapolated_bounds()
        grid = GridXY(y=row, x=col)
        poly = PolygonArray(grid=grid)
        self.assertEqual(poly.name, 'geom')
        self.assertEqual(poly.geom_type, 'Polygon')

        # Test bounds are used for bbox subset.
        self.assertTrue(PolygonArray._use_bounds)

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

    def test_get_value(self):
        """Test ordering of vertices when creating from corners is slightly different."""

        keywords = dict(with_grid_row_col_bounds=[True, False],
                        with_grid_mask=[True, False])
        for k in self.iter_product_keywords(keywords, as_namedtuple=True):
            poly = self.get_polygonarray()
            if k.with_grid_mask:
                poly.grid.value.mask[:, 1, 1] = True
            poly.grid.corners
            if not k.with_grid_row_col_bounds:
                poly.grid.y.bounds = None
                poly.grid.x.bounds = None
                actual = self.polygon_value_alternate_ordering
            else:
                actual = self.polygon_value
            if k.with_grid_mask:
                actual.mask[1, 1] = True
            self.assertIsNone(poly._value)
            value_poly = poly.value
            self.assertGeometriesAlmostEquals(value_poly, actual)
