import numpy as np
from numpy.ma import MaskedArray
from shapely.geometry import Point

from ocgis import env
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
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

    def test_getitem(self):
        gridxy = self.get_gridxy()
        pa = self.get_pointarray(grid=gridxy, value=None)
        sub = pa[2:4, 1]
        self.assertEqual(sub.shape, (2, 1))
        self.assertEqual(sub.value.shape, (2, 1))
        self.assertEqual(sub.grid.shape, (2, 1))

    def test_set_value(self):
        value = Point(6, 7)
        with self.assertRaises(ValueError):
            self.get_pointarray(value=value)

    def test_spatial_index(self):
        pa = self.get_pointarray()
        self.assertIsInstance(pa.spatial_index, SpatialIndex)
        self.assertEqual(pa.spatial_index._index.bounds, [1.0, 2.0, 3.0, 4.0])

    def test_get_intersects_masked(self):
        sdim = self.get_sdim(crs=WGS84())
        self.assertIsNotNone(sdim.grid)
        sdim.assert_uniform_mask()
        poly = wkt.loads(
            'POLYGON((-98.26574367088608142 40.19952531645570559,-98.71764240506330168 39.54825949367089066,-99.26257911392406186 39.16281645569620906,-99.43536392405064817 38.64446202531645724,-98.78409810126584034 38.33876582278481493,-98.23916139240508016 37.71408227848101546,-97.77397151898735217 37.67420886075949937,-97.62776898734178133 38.15268987341772799,-98.39865506329114453 38.52484177215190186,-98.23916139240508016 39.33560126582278826,-97.73409810126582897 39.58813291139241386,-97.52143987341773368 40.27927215189873777,-97.52143987341773368 40.27927215189873777,-98.26574367088608142 40.19952531645570559))')
        actual_mask = np.array([[True, True, False, True], [True, False, True, True], [True, True, False, True]])
        actual_uid_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int32)
        for use_spatial_index in [True, False]:
            ret = sdim.geom.point.get_intersects_masked(poly, use_spatial_index=use_spatial_index)
            self.assertNumpyAll(actual_mask, ret.value.mask)
            self.assertNumpyAll(actual_mask, ret.uid.mask)
            self.assertNumpyAll(actual_uid_data, ret.uid.data)
            self.assertFalse(sdim.uid.mask.any())
            for element in ret.value.data.flat:
                self.assertIsInstance(element, Point)
            self.assertIsNotNone(sdim.grid)

        # Test pre-masked values in geometry are okay for intersects operation.
        value = [Point(1, 1), Point(2, 2), Point(3, 3)]
        value = np.ma.array(value, mask=[False, True, False], dtype=object).reshape(-1, 1)
        s = SpatialGeometryPointDimension(value=value)
        b = box(0, 0, 5, 5)
        res = s.get_intersects_masked(b)
        self.assertNumpyAll(res.value.mask, value.mask)

    def test_weights(self):
        value = [Point(2, 3), Point(4, 5), Point(5, 6)]
        mask = [False, True, False]
        value = np.ma.array(value, mask=mask, dtype=object)
        pa = self.get_pointarray(value=value)
        self.assertNumpyAll(pa.weights, np.ma.array([1, 1, 1], mask=mask, dtype=env.NP_FLOAT))
