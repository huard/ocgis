import numpy as np
from shapely.geometry import Point, MultiPoint

from ocgis.interface.base.crs import WGS84, CoordinateReferenceSystem
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface


class TestSpatialAdapter(AbstractTestNewInterface):
    """Tests against a point array."""

    def test_init(self):
        pa = self.get_pointarray(grid=self.get_gridxy(), crs=WGS84())
        self.assertEqual(pa.crs, WGS84())

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

    def test_update_crs(self):
        pa = self.get_pointarray(crs=WGS84())
        to_crs = CoordinateReferenceSystem(epsg=2136)
        pa.update_crs(to_crs)
        self.assertEqual(pa.crs, to_crs)
        v0 = [1629871.494956261, -967769.9070825744]
        v1 = [2358072.3857447207, -239270.87548993886]
        np.testing.assert_almost_equal(pa.value[0], v0)
        np.testing.assert_almost_equal(pa.value[1], v1)
