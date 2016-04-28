import os

import fiona

from ocgis import RequestDataset, GeomCabinetIterator
from ocgis import constants
from ocgis.api.request.driver.base import AbstractDriver
from ocgis.api.request.driver.vector import DriverVector, get_fiona_crs, get_fiona_schema
from ocgis.interface.base.crs import WGS84, CoordinateReferenceSystem
from ocgis.new_interface.geom import GeometryVariable
from ocgis.test.base import TestBase


class TestDriverVector(TestBase):
    def get_driver(self, **kwargs):
        rd = self.get_request_dataset(**kwargs)
        driver = DriverVector(rd)
        return driver

    def get_request_dataset(self, variable=None):
        uri = os.path.join(self.path_bin, 'shp', 'state_boundaries', 'state_boundaries.shp')
        # uri = GeomCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri=uri, driver='vector', variable=variable)
        return rd

    def test_init(self):
        self.assertIsInstances(self.get_driver(), (DriverVector, AbstractDriver))

        actual = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH,
                  constants.OUTPUT_FORMAT_SHAPEFILE]
        self.assertAsSetEqual(actual, DriverVector.output_formats)

    def test_close(self):
        driver = self.get_driver()
        sci = driver.open()
        driver.close(sci)

    def test_get_crs(self):
        driver = self.get_driver()
        self.assertEqual(WGS84(), driver.get_crs())

    def test_get_dimensioned_variables(self):
        driver = self.get_driver()
        target = driver.get_dimensioned_variables()
        self.assertEqual(target, [u'UGID', u'STATE_FIPS', u'ID', u'STATE_NAME', u'STATE_ABBR'])

    def test_get_dump_report(self):
        driver = self.get_driver()
        lines = driver.get_dump_report()
        self.assertTrue(len(lines) > 5)

    def test_get_field(self):
        # tdk: RESUME: loading fails when slicing before the value is loaded. implemented indexed loading.
        driver = self.get_driver()
        field = driver.get_field()
        self.assertEqual(len(field), 7)
        self.assertIsInstance(field.geom, GeometryVariable)
        self.assertIsInstance(field.crs, CoordinateReferenceSystem)
        self.assertIsNone(field.time)
        for v in field.values():
            self.assertIsNotNone(v.value)

    def test_get_variable_collection(self):
        driver = self.get_driver()
        vc = driver.get_variable_collection()
        self.assertEqual(len(vc), 7)
        for v in vc.values():
            if not isinstance(v, CoordinateReferenceSystem):
                self.assertEqual(v.dimensions[0]._src_idx.shape[0], 51)
                self.assertIsNone(v._value)

    def test_inspect(self):
        driver = self.get_driver()
        with self.print_scope() as ps:
            driver.inspect()
        self.assertTrue(len(ps.storage) >= 1)

    def test_metadata(self):
        driver = self.get_driver()
        m = driver.metadata
        self.assertIsInstance(m, dict)
        self.assertTrue(len(m) > 2)
        self.assertIn('variables', m)

    def test_open(self):
        driver = self.get_driver()
        sci = driver.open()
        self.assertIsInstance(sci, GeomCabinetIterator)
        self.assertFalse(sci.as_spatial_dimension)

    def test_write_variable_collection(self):
        # tdk: RESUME: test writing mix polygons and multi-polygons to file
        # Test writing the field to file.
        field = self.get_driver().get_field()
        path1 = self.get_temporary_file_path('out1.shp')
        path2 = self.get_temporary_file_path('out2.shp')
        fiona_crs = get_fiona_crs(field, None)
        fiona_schema = get_fiona_schema(field, None)
        fobject = fiona.open(path2, mode='w', schema=fiona_schema, crs=fiona_crs, driver='ESRI Shapefile')
        for target in [path1, fobject]:
            field.write(target, driver=DriverVector)

            if isinstance(target, basestring):
                path = path1
            else:
                path = path2
                fobject.close()

            with fiona.open(path) as source:
                self.assertEqual(len(source), 51)
            rd = RequestDataset(uri=path)
            field2 = rd.get()
            for v in field.values():
                if isinstance(v, CoordinateReferenceSystem):
                    self.assertEqual(v, field2.crs)
                else:
                    self.assertNumpyAll(v.value, field2[v.name].value)
