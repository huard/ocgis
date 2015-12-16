import fiona
import numpy as np
from numpy.ma import MaskedArray
from shapely import wkt
from shapely.geometry import Point, box, MultiPoint

from ocgis import env, CoordinateReferenceSystem
from ocgis.interface.base.crs import WGS84
from ocgis.new_interface.geom import PointArray
from ocgis.new_interface.grid import GridXY
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.test.base import ToTest
from ocgis.util.spatial.index import SpatialIndex


class TestPointArray(AbstractTestNewInterface):
    # tdk: test initializing with a grid
    # tdk: bring in spatialgeometrypointdimension tests

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
