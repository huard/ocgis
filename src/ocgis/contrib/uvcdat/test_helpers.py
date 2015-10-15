from shapely.geometry import box, Polygon, MultiPolygon
import numpy as np

from ocgis import GeomCabinet
import ocgis
from ocgis.contrib.uvcdat.helpers import convert_multipart_to_singlepart, get_mesh_array_from_polygons, \
    get_coordinate_array_from_polygon, get_polygon_array_from_mesh_array, iter_polygons_and_indices
from ocgis.test.base import TestBase


class Test(TestBase):
    # tdk: file
    # tdk: vcs: test disjoint elements
    def __init__(self, *args, **kwargs):
        self.square = box(10, 20, 15, 30)
        self.triangle = Polygon([(2.5, 2.5), (7.5, 2.5), (5., 7.5)])
        super(Test, self).__init__(*args, **kwargs)

    def get_shapefile_path_state_boundaries_singlepart(self):
        gm = GeomCabinet()
        path = gm.get_shp_path('state_boundaries')
        path_new = self.get_temporary_file_path('state_boundaries_singlepart.shp')
        convert_multipart_to_singlepart(path, path_new)
        return path_new

    def test(self):
        path = self.get_shapefile_path_state_boundaries_singlepart()
        field = ocgis.RequestDataset(path).get()
        mesh_arr = get_mesh_array_from_polygons(field.spatial.geom.polygon.value)

        # Test each shapefile feature is present.
        self.assertEqual(field.shape[-2] * field.shape[-1], mesh_arr.shape[0])

    def test_get_coordinate_array_from_polygon(self):
        node_count = 8
        res = get_coordinate_array_from_polygon(self.square, node_count)
        self.assertEqual(res.shape[1], node_count)

        # Test we have masked data.
        self.assertTrue(res.mask.any())

        # Test latitudes are in the first row.
        self.assertIn(20., res[0, :])
        self.assertIn(30., res[0, :])

        # Test longitudes are in the second row.
        self.assertIn(10., res[1, :])
        self.assertIn(15., res[1, :])

    def test_get_mesh_array_from_polygons(self):
        polygons = [self.square, self.triangle]
        res = get_mesh_array_from_polygons(polygons)
        self.assertEqual(res.tolist(), [[[20.0, 30.0, 30.0, 20.0, 20.0],
                                         [15.0, 15.0, 10.0, 10.0, 15.0]],
                                        [[2.5, 2.5, 7.5, 2.5, None],
                                         [2.5, 7.5, 5.0, 2.5, None]]])

    def test_get_polygon_array_from_mesh_array(self):
        polygons = [self.square, self.triangle]
        res = get_mesh_array_from_polygons(polygons)
        polygon_arr = get_polygon_array_from_mesh_array(res)
        for idx in range(len(polygons)):
            self.assertTrue(polygons[idx].almost_equals(polygon_arr[0, idx]))

    def test_iter_polygons_and_indices(self):
        arr = np.array([self.square, self.triangle, MultiPolygon([self.square, self.triangle])]).reshape(-1, 1)
        res = list(iter_polygons_and_indices(arr))
        # Test multi-polygon yielded as polygons.
        self.assertEqual(len(res), 4)
        # Test the index is repeated.
        self.assertEqual(res[-2][0], res[-1][0])
