from csv import DictReader
import os

import numpy as np
import fiona
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry.multipoint import MultiPoint

from ocgis import constants, SpatialGeometryDimension, SpatialDimension, Field, Variable
from ocgis import GeomCabinet, RequestDataset, OcgOperations, env
from ocgis.interface.base.dimension.spatial import SpatialGeometryPolygonDimension
from ocgis.test.base import TestBase

"""
These tests written to guide bug fixing or issue development. Theses tests are typically high-level and block-specific
testing occurs in tandem. It is expected that any issues identified by these tests have a corresponding test in the
package hierarchy. Hence, these tests in theory may be removed...
"""


class Test20150119(TestBase):
    def test_shapefile_through_operations_subset(self):
        path = GeomCabinet().get_path('state_boundaries')
        rd = RequestDataset(path)
        field = rd.get()
        self.assertIsNone(field.spatial.properties)
        ops = OcgOperations(dataset=rd, output_format='shp', geom='state_boundaries', select_ugid=[15])
        ret = ops.execute()
        rd2 = RequestDataset(ret)
        field2 = rd2.get()
        self.assertAsSetEqual(field.variables.keys(), field2.variables.keys())
        self.assertEqual(tuple([1] * 5), field2.shape)

    def test_shapefile_through_operations(self):
        path = GeomCabinet().get_path('state_boundaries')
        rd = RequestDataset(path)
        field = rd.get()
        self.assertIsNone(field.spatial.properties)
        ops = OcgOperations(dataset=rd, output_format='shp')
        ret = ops.execute()
        rd2 = RequestDataset(ret)
        field2 = rd2.get()
        self.assertAsSetEqual(field.variables.keys(), field2.variables.keys())
        self.assertEqual(field.shape, field2.shape)


class Test20150224(TestBase):
    def test_subset_with_shapefile_no_ugid(self):
        """Test a subset operation using a shapefile without a UGID attribute."""

        output_format = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_CSV_SHAPEFILE]

        geom = self.get_shapefile_path_with_no_ugid()
        geom_select_uid = [8, 11]
        geom_uid = 'ID'
        rd = self.test_data.get_rd('cancm4_tas')

        for of in output_format:
            ops = OcgOperations(dataset=rd, geom=geom, geom_select_uid=geom_select_uid, geom_uid=geom_uid, snippet=True,
                                output_format=of)
            self.assertEqual(len(ops.geom), 2)
            ret = ops.execute()
            if of == constants.OUTPUT_FORMAT_NUMPY:
                for element in geom_select_uid:
                    self.assertIn(element, ret)
                self.assertEqual(ret.properties[8].dtype.names, ('STATE_FIPS', 'ID', 'STATE_NAME', 'STATE_ABBR'))
            else:
                with open(ret) as f:
                    reader = DictReader(f)
                    row = reader.next()
                    self.assertIn(geom_uid, row.keys())
                    self.assertNotIn(env.DEFAULT_GEOM_UID, row.keys())

                shp_path = os.path.split(ret)[0]
                shp_path = os.path.join(shp_path, 'shp', '{0}_gid.shp'.format(ops.prefix))
                with fiona.open(shp_path) as source:
                    record = source.next()
                    self.assertIn(geom_uid, record['properties'])
                    self.assertNotIn(env.DEFAULT_GEOM_UID, record['properties'])


class Test20150327(TestBase):
    def test_sql_where_through_operations(self):
        """Test using a SQL where statement to select some geometries."""

        states = ("Wisconsin", "Vermont")
        s = 'STATE_NAME in {0}'.format(states)
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, geom_select_sql_where=s, geom='state_boundaries', snippet=True)
        ret = ops.execute()
        self.assertEqual(len(ret), 2)
        self.assertEqual(ret.keys(), [8, 10])
        for v in ret.properties.itervalues():
            self.assertIn(v['STATE_NAME'], states)

        # make sure the sql select has preference over uid
        ops = OcgOperations(dataset=rd, geom_select_sql_where=s, geom='state_boundaries', snippet=True,
                            geom_select_uid=[500, 600, 700])
        ret = ops.execute()
        self.assertEqual(len(ret), 2)
        for v in ret.properties.itervalues():
            self.assertIn(v['STATE_NAME'], states)

        # test possible interaction with geom_uid
        path = self.get_shapefile_path_with_no_ugid()
        ops = OcgOperations(dataset=rd, geom=path, geom_select_sql_where=s)
        ret = ops.execute()
        self.assertEqual(ret.keys(), [1, 2])

        ops = OcgOperations(dataset=rd, geom=path, geom_select_sql_where=s, geom_uid='ID')
        ret = ops.execute()
        self.assertEqual(ret.keys(), [13, 15])


class Test20150608(TestBase):
    def test_multipoint_buffering_and_union(self):
        """Test subset behavior using MultiPoint geometries."""

        pts = [Point(3.8, 28.57), Point(9.37, 33.90), Point(17.04, 27.08)]
        mp = MultiPoint(pts)

        rd = self.test_data.get_rd('cancm4_tas')
        coll = OcgOperations(dataset=rd, output_format='numpy', snippet=True, geom=mp).execute()
        mu1 = coll[1]['tas'].variables['tas'].value.mean()
        nc_path = OcgOperations(dataset=rd, output_format='nc', snippet=True, geom=mp).execute()
        with self.nc_scope(nc_path) as ds:
            var = ds.variables['tas']
            mu2 = var[:].mean()
        self.assertEqual(mu1, mu2)


class Test20150811(TestBase):
    def test_write_field_with_ugrid_to_cf_convention(self):
        # tdk: comment
        # tdk: add appropriate convention attribute

        ugrid_polygons = [
            'POLYGON((-1.5019011406844105 0.18377693282636276,-1.25475285171102646 0.02534854245880869,-1.35614702154626099 -0.28517110266159684,-1.68567807351077303 -0.50697084917617241,-1.99619771863117879 -0.41191381495564006,-2.08491761723700897 -0.24714828897338403,-1.9264892268694549 -0.03802281368821281,-1.88212927756653992 0.13307984790874539,-1.5019011406844105 0.18377693282636276))',
            'POLYGON((-2.25602027883396694 0.63371356147021585,-1.76172370088719887 0.51330798479087481,-1.88212927756653992 0.13307984790874539,-1.9264892268694549 -0.03802281368821281,-2.30671736375158432 0.01901140684410674,-2.51584283903675532 0.27249683143219272,-2.52217997465145771 0.48795944233206612,-2.25602027883396694 0.63371356147021585))',
            'POLYGON((-1.55893536121673004 0.86818757921419554,-1.03929024081115307 0.65906210392902409,-1.07097591888466415 0.46261089987325743,-1.5019011406844105 0.18377693282636276,-1.88212927756653992 0.13307984790874539,-1.76172370088719887 0.51330798479087481,-1.55893536121673004 0.86818757921419554))',
            'POLYGON((-2.13561470215462634 0.87452471482889749,-1.83143219264892276 0.98225602027883419,-1.83143219264892276 0.98225602027883419,-1.55893536121673004 0.86818757921419554,-1.58428390367553851 0.66539923954372648,-1.76172370088719887 0.51330798479087481,-2.12294043092522156 0.44993662864385309,-2.25602027883396694 0.63371356147021585,-2.13561470215462634 0.87452471482889749))'
        ]

        polygons = np.array([wkt.loads(xx) for xx in ugrid_polygons]).reshape(-1, 1)

        poly = SpatialGeometryPolygonDimension(value=polygons)
        geom = SpatialGeometryDimension(polygon=poly)
        sdim = SpatialDimension(geom=geom)
        field = Field(spatial=sdim)
        var = Variable(name='tas', value=np.random.rand(*field.shape))
        field.variables.add_variable(var)
        out_path = self.get_temporary_file_path('out.nc')
        print field.shape
