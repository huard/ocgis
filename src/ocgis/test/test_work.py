from csv import DictReader
import os
import numpy as np

import ESMF
import fiona
from shapely import wkt
from shapely.geometry import Polygon
from shapely.geometry import Point

from shapely.geometry.multipoint import MultiPoint

from ocgis import SpatialGeometryPolygonDimension, Field, SpatialCollection, SpatialGeometryDimension, \
    SpatialDimension
from ocgis import constants
from ocgis import ShpCabinet, RequestDataset, OcgOperations, env
from ocgis.api.request.driver.nc import DriverNetcdfUgrid
from ocgis.exc import OcgWarning
from ocgis.interface.base.crs import Spherical, CFWGS84
from ocgis.test.base import TestBase, attr

"""
These tests written to guide bug fixing or issue development. Theses tests are typically high-level and block-specific
testing occurs in tandem. It is expected that any issues identified by these tests have a corresponding test in the
package hierarchy. Hence, these tests in theory may be removed...
"""


class Test20150119(TestBase):
    def test_shapefile_through_operations_subset(self):
        path = ShpCabinet().get_shp_path('state_boundaries')
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
        path = ShpCabinet().get_shp_path('state_boundaries')
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


class Test20150413(TestBase):

    def test_ugrid_read(self):
        """Test reading data from a UGRID Mesh NetCDF file."""

        driver = DriverNetcdfUgrid.key
        polygons = [wkt.loads(xx) for xx in self.test_data_ugrid_polygons]
        polygons = np.atleast_2d(np.array(polygons))
        spoly = SpatialGeometryPolygonDimension(value=polygons)

        path = os.path.join(self.current_dir_output, 'foo.nc')
        with self.nc_scope(path, 'w') as ds:
            spoly.write_to_netcdf_dataset_ugrid(ds)

        rd_name = 'hello_dude'
        rd = RequestDataset(path, driver=driver, name=rd_name)
        self.assertIsInstance(rd.source_metadata, dict)
        self.assertTrue(len(rd.source_metadata) > 1)
        self.assertEqual(rd.crs, env.DEFAULT_COORDSYS)
        self.assertIsNone(rd._crs)
        self.assertIsInstance(rd.get(), Field)

        coll = OcgOperations(dataset=rd).execute()
        self.assertIsInstance(coll, SpatialCollection)
        self.assertEqual(coll[1].keys(), [rd_name])

        path = OcgOperations(dataset=rd, output_format='shp').execute()
        with fiona.open(path) as source:
            ugid = [r['properties'][constants.OCGIS_UNIQUE_GEOMETRY_IDENTIFIER] for r in source]
        self.assertEqual(ugid, [1, 1, 1, 1])

        for output_format in ['shp', 'csv-shp', 'numpy', 'geojson', 'csv', 'nc-ugrid-2d-flexible-mesh']:
            ret = OcgOperations(dataset=rd, output_format=output_format, prefix=output_format).execute()
            if output_format == 'nc-ugrid-2d-flexible-mesh':
                self.assertNcEqual(rd.uri, ret, ignore_attributes={'global': ['history']})

        # Test warning is raised with UGRID outputs and an output coordinate system.
        def _warning_function_():
            output_format = constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH
            ops = OcgOperations(dataset=rd, output_format=output_format, prefix='warning_test', output_crs=CFWGS84())
            ops.execute()

        self.assertWarns(OcgWarning, _warning_function_, suppress=False)


@attr('esmpy7')
class Test20150709(TestBase):
    @staticmethod
    def mesh_create_5():
        '''
        PRECONDITIONS: None
        POSTCONDITIONS: A 5 element Mesh has been created.
        RETURN VALUES: \n Mesh :: mesh \n

          4.0   31 ------ 32 ------ 33
                |         |  22  /   |
                |    21   |     /    |
                |         |   /  23  |
          2.0   21 ------ 22 ------ 23
                |         |          |
                |    11   |    12    |
                |         |          |
          0.0   11 ------ 12 ------ 13

               0.0       2.0        4.0

              Node Ids at corners
              Element Ids in centers

        Note: This mesh is not parallel, it can only be used in serial
        '''
        # Two parametric dimensions, and two spatial dimensions
        mesh = ESMF.Mesh(parametric_dim=2, spatial_dim=2)

        num_node = 9
        num_elem = 5
        nodeId = np.array([11, 12, 13, 21, 22, 23, 31, 32, 33])
        nodeCoord = np.array([0.0, 0.0,  # node 11
                              2.0, 0.0,  # node 12
                              4.0, 0.0,  # node 13
                              0.0, 2.0,  # node 21
                              2.0, 2.0,  # node 22
                              4.0, 2.0,  # node 23
                              0.0, 4.0,  # node 31
                              2.0, 4.0,  # node 32
                              4.0, 4.0])  # node 33
        # If this example were parallel, this indicates which PET owns node.
        nodeOwner = np.zeros(num_node)

        elemId = np.array([11, 12, 21, 22, 23])
        elemType = np.array([ESMF.MeshElemType.QUAD,
                             ESMF.MeshElemType.QUAD,
                             ESMF.MeshElemType.QUAD,
                             ESMF.MeshElemType.TRI,
                             ESMF.MeshElemType.TRI])
        elemConn = np.array([0, 1, 4, 3,  # element 11
                             1, 2, 5, 4,  # element 12
                             3, 4, 7, 6,  # element 21
                             4, 8, 7,  # element 22
                             4, 5, 8])  # element 23
        elemCoord = np.array([1.0, 1.0,
                              3.0, 1.0,
                              1.0, 3.0,
                              2.5, 3.5,
                              3.5, 2.5])

        mesh.add_nodes(num_node, nodeId, nodeCoord, nodeOwner)

        mesh.add_elements(num_elem, elemId, elemType, elemConn, element_coords=elemCoord)

        return mesh, nodeCoord, nodeOwner, elemType, elemConn, elemCoord, nodeId, elemId

    def test_ocgis_ugrid_write_read_by_esmf(self):
        """
        Test a UGRID file written by OCGIS may be read by ESMF.
        """
        ESMF.Manager(debug=True)

        # Write polygons to UGRID file.
        ccw_wkt = [
            'POLYGON((-0.53064516129032269 0.53817204301075283,0.14301075268817209 -0.73763440860215057,-1.25698924731182804 -0.73010752688172054,-0.53064516129032269 0.53817204301075283))',
            'POLYGON((-0.53064516129032269 0.53817204301075283,0.54253642039542171 0.7357162677766218,1.19552983003815516 -0.34940513354144986,0.14301075268817209 -0.73763440860215057,-0.53064516129032269 0.53817204301075283))']
        polygons = [wkt.loads(cw) for cw in ccw_wkt]
        polygons = np.atleast_2d(np.array(polygons))
        spoly = SpatialGeometryPolygonDimension(value=polygons)
        ugrid_path = os.path.join(self.current_dir_output, 'foo.nc')
        with self.nc_scope(ugrid_path, 'w') as ds:
            spoly.write_to_netcdf_dataset_ugrid(ds)

        # path = '/home/benkoziol/Downloads/NFIE_shapefile_ugrid_regrid/catchment_San_Guad_3reaches/catchment_San_Guad_3reaches.shp'
        # rd = RequestDataset(uri=path)
        # ops = OcgOperations(dataset=rd, output_format=constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH)
        # ugrid_path = ops.execute()

        # Read the UGRID file into an ESMF mesh.
        mesh = ESMF.Mesh(filename=ugrid_path, filetype=ESMF.FileFormat.UGRID, meshname="Mesh2")
        # tdk: only allow field build with meshloc of ELEMENT
        efield = ESMF.Field(mesh, meshloc=ESMF.MeshLoc.ELEMENT)
        efield[:] = 15

        nodeCoord = mesh._connectivity
        elemType = mesh._num_nodes_per_elem
        # elemConn = mesh._element_conn
        parametric_dim = mesh.parametric_dim

        assert parametric_dim == 2

        polygons = np.zeros((elemType.shape[0], 1), dtype=object)
        idx_curr_elemConn = 0
        for idx in range(elemType.shape[0]):
            number_of_nodes_in_element = elemType[idx]
            polygon_coords = np.zeros((number_of_nodes_in_element, parametric_dim))
            step = number_of_nodes_in_element * parametric_dim
            polygon_coords[:, 0] = nodeCoord[idx_curr_elemConn:step:parametric_dim]
            polygon_coords[:, 1] = nodeCoord[idx_curr_elemConn + 1:step:parametric_dim]
            polygon = Polygon(polygon_coords)
            polygons[idx] = polygon
            idx_curr_elemConn += step

        poly = SpatialGeometryPolygonDimension(value=polygons)
        geom = SpatialGeometryDimension(polygon=poly)
        sdim = SpatialDimension(geom=geom, crs=CFWGS84())
        sdim.write_fiona('/tmp/foo.shp')

    def test_ocgis_field_to_esmpy_mesh(self):
        """Test creating an ESMF mesh from an OCGIS field."""

        # tdk: need to handle parametric_dim=2 and spatial_dim=3 - mesh on a spherical surface
        mesh, nodeCoord, nodeOwner, elemType, elemConn, elemCoord, nodeId, elemId = self.mesh_create_5()
        # mesh._write_(self.get_temporary_file_path('vtk'))
        parametric_dim = 2

        polygons = np.zeros((elemType.shape[0], 1), dtype=object)
        idx_curr_elemConn = 0
        for idx in range(elemType.shape[0]):
            number_of_nodes_in_element = elemType[idx]
            polygon_coords = np.zeros((number_of_nodes_in_element, parametric_dim))
            node_coordinate_indices = elemConn[idx_curr_elemConn:idx_curr_elemConn + number_of_nodes_in_element]
            node_coordinate_indices_shift = 2 * node_coordinate_indices
            polygon_coords[:, 0] = nodeCoord[node_coordinate_indices_shift]
            polygon_coords[:, 1] = nodeCoord[node_coordinate_indices_shift + 1]
            polygon = Polygon(polygon_coords)
            polygons[idx] = polygon
            idx_curr_elemConn += number_of_nodes_in_element

        poly = SpatialGeometryPolygonDimension(value=polygons)
        geom = SpatialGeometryDimension(polygon=poly)
        sdim = SpatialDimension(geom=geom)
        # shp_path = self.get_temporary_file_path('polygons.shp')
        # poly.write_fiona(shp_path, None)

        # tdk: if this is on a sphere the spatial_dim needs to be three
        if isinstance(sdim.crs, Spherical):
            # tdk: test
            spatial_dim = 3
        else:
            spatial_dim = 2
        mesh = ESMF.Mesh(parametric_dim=2, spatial_dim=spatial_dim)
        # tdk: continue writing mesh from spatial dimension - need support for parallel?

    def test_mesh(self):
        #tdk: remove this test
        import ESMF

        ESMF.Manager(debug=True)
        path = '/home/benkoziol/Dropbox/Share/RK-Share/FVCOM_grid2d_20130228.nc'
        # path = self.get_netcdf_path_ugrid('flavor.nc')
        #
        # with self.nc_scope(path) as ds:
        #     x = ds.variables['Mesh2_node_x'][:]
        #     y = ds.variables['Mesh2_node_y'][:]
        # import matplotlib.pyplot as plt
        # print x[:]
        # print y[:]
        # plt.plot(x[:], y[:])
        # plt.show()
        #
        # with self.nc_scope(path, 'a') as ds:
        #     var = ds.variables['Mesh2']
        #     for attr in ['face_face_connectivity', 'face_edge_connectivity', 'face_coordinates']:
        #         try:
        #             var.delncattr(attr)
        #         except RuntimeError:
        #             pass

        dstgrid = ESMF.Mesh(filename=path, filetype=ESMF.FileFormat.UGRID, meshname="fvcom_mesh")
        nodes = dstgrid.coords[0]
        coords_x, coords_y = nodes
        # The number of coordinate dimensions (i.e. x, y).
        ndims = len(nodes)
        # This is the number of faces in the UGRID file.
        nfaces = dstgrid.size[1]
        self.assertEqual(nfaces, len(dstgrid.coords[1][1]))
        import ipdb;

        ipdb.set_trace()


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

