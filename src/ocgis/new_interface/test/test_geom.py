import numpy as np
from numpy.ma import MaskedArray
from shapely import wkt
from shapely.geometry import Point, box

from ocgis import env
from ocgis.interface.base.crs import WGS84
from ocgis.new_interface.geom import PointArray
from ocgis.new_interface.grid import GridXY
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import BoundedVariable
from ocgis.util.spatial.index import SpatialIndex


class TestPointArray(AbstractTestNewInterface):
    # tdk: test initializing with a grid
    # tdk: bring in spatialgeometrypointdimension tests

    def get_variable_x(self, bounds=True):
        value = [-100., -99., -98., -97.]
        if bounds:
            bounds = [[v - 0.5, v + 0.5] for v in value]
        else:
            bounds = None
        x = BoundedVariable(value=value, bounds=bounds, name='x')
        return x

    def get_variable_y(self, bounds=True):
        value = [40., 39., 38.]
        if bounds:
            bounds = [[v + 0.5, v - 0.5] for v in value]
        else:
            bounds = None
        y = BoundedVariable(value=value, bounds=bounds, name='y')
        return y

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

    def test_get_intersects(self):
        x = self.get_variable_x()
        y = self.get_variable_y()
        grid = GridXY(x=x, y=y)
        pa = PointArray(grid=grid, crs=WGS84())
        self.assertIsNotNone(pa.grid)
        poly = wkt.loads(
            'POLYGON((-98.26574367088608142 40.19952531645570559,-98.71764240506330168 39.54825949367089066,-99.26257911392406186 39.16281645569620906,-99.43536392405064817 38.64446202531645724,-98.78409810126584034 38.33876582278481493,-98.23916139240508016 37.71408227848101546,-97.77397151898735217 37.67420886075949937,-97.62776898734178133 38.15268987341772799,-98.39865506329114453 38.52484177215190186,-98.23916139240508016 39.33560126582278826,-97.73409810126582897 39.58813291139241386,-97.52143987341773368 40.27927215189873777,-97.52143987341773368 40.27927215189873777,-98.26574367088608142 40.19952531645570559))')
        actual_mask = np.array([[True, True, False, True], [True, False, True, True], [True, True, False, True]])
        for use_spatial_index in [True, False]:
            ret = pa.get_intersects(poly, use_spatial_index=use_spatial_index)
            self.assertNumpyAll(actual_mask, ret.value.mask)
            for element in ret.value.data.flat:
                self.assertIsInstance(element, Point)
            self.assertIsNotNone(pa.grid)

            # Test pre-masked values in geometry are okay for intersects operation.
            value = [Point(1, 1), Point(2, 2), Point(3, 3)]
            value = np.ma.array(value, mask=[False, True, False], dtype=object)
            pa2 = PointArray(value=value)
            b = box(0, 0, 5, 5)
            res = pa2.get_intersects(b, use_spatial_index=use_spatial_index)
            self.assertNumpyAll(res.value.mask, value.mask)

    def test_get_spatial_index(self):
        pa = self.get_pointarray()
        si = pa.get_spatial_index()
        self.assertIsInstance(si, SpatialIndex)
        self.assertEqual(si._index.bounds, [1.0, 2.0, 3.0, 4.0])

    def test_weights(self):
        value = [Point(2, 3), Point(4, 5), Point(5, 6)]
        mask = [False, True, False]
        value = np.ma.array(value, mask=mask, dtype=object)
        pa = self.get_pointarray(value=value)
        self.assertNumpyAll(pa.weights, np.ma.array([1, 1, 1], mask=mask, dtype=env.NP_FLOAT))
