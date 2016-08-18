import os
from unittest import SkipTest

import fiona
import numpy as np
from shapely.geometry import Point

from ocgis import RequestDataset, GeomCabinetIterator
from ocgis import constants
from ocgis.api.request.driver.base import AbstractDriver
from ocgis.api.request.driver.vector import DriverVector, get_fiona_crs, get_fiona_schema
from ocgis.interface.base.crs import WGS84, CoordinateReferenceSystem, Spherical
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.mpi import MPI_RANK
from ocgis.new_interface.temporal import TemporalVariable
from ocgis.new_interface.variable import Variable, VariableCollection
from ocgis.test.base import TestBase, attr


class TestDriverVector(TestBase):
    def assertPrivateValueIsNone(self, field_like):
        for v in field_like.values():
            if not isinstance(v, CoordinateReferenceSystem):
                self.assertIsNone(v._value)

    def get_driver(self, **kwargs):
        rd = self.get_request_dataset(**kwargs)
        driver = DriverVector(rd)
        return driver

    def get_request_dataset(self, variable=None):
        uri = os.path.join(self.path_bin, 'shp', 'state_boundaries', 'state_boundaries.shp')
        rd = RequestDataset(uri=uri, driver='vector', variable=variable)
        return rd

    def test_init(self):
        self.assertIsInstances(self.get_driver(), (DriverVector, AbstractDriver))

        actual = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH,
                  constants.OUTPUT_FORMAT_SHAPEFILE]
        self.assertAsSetEqual(actual, DriverVector.output_formats)

    @attr('data')
    def test_system_cf_data(self):
        rd = self.test_data.get_rd('cancm4_tas')
        path = self.get_temporary_file_path('grid.shp')
        field = rd.get()[{'time': slice(3, 6), 'lat': slice(10, 20), 'lon': slice(21, 27)}]
        variable_names = ['time', 'lat', 'lon', 'tas']
        field.write(path, driver=DriverVector, variable_names=variable_names)
        read = RequestDataset(path).get()
        self.assertEqual(len(read.dimensions.values()[0]), 3 * 10 * 6)
        desired_keys = [u'time', u'lat', u'lon', u'tas', 'ocgis_geom', 'ocgis_coordinate_system']
        self.assertEqual(read.keys(), desired_keys)
        self.assertEqual(rd.get().crs, Spherical())
        self.assertEqual(read.crs, Spherical())

    def test_system_conform_units(self):
        """Test conforming units on data read from shapefile."""

        path = self.get_temporary_file_path('temps.shp')
        gvar = GeometryVariable(value=[Point(1, 2), Point(3, 4)], dimensions='g', name='geom')
        var = Variable(name='temp', value=[10., 20.], dimensions='g')
        vc = VariableCollection(variables=[gvar, var])
        vc.write(path, driver=DriverVector)

        field = RequestDataset(path, units='celsius', variable='temp', conform_units_to='fahrenheit').get()
        self.assertNumpyAllClose(field['temp'].value, np.array([50., 68.]))

    def test_system_with_time_data(self):
        """Test writing data with a time dimension."""

        path = self.get_temporary_file_path('what.shp')
        t = TemporalVariable(value=[1.5, 2.5], name='time', dimensions='time')
        geom = GeometryVariable(value=[Point(1, 2), Point(3, 4)], name='geom', dimensions='time')
        field = OcgField(variables=[t, geom], dimension_map={'time': {'variable': 'time'},
                                                             'geom': {'variable': 'geom'}})
        field.write(path, driver=DriverVector)

        rd = RequestDataset(uri=path)
        field2 = rd.get()
        self.assertEqual(field2['time'].value.tolist(), ['0001-01-02 12:00:00', '0001-01-03 12:00:00'])

    def test_close(self):
        driver = self.get_driver()
        sci = driver.open(rd=driver.rd)
        driver.close(sci)

    def test_get_crs(self):
        driver = self.get_driver()
        self.assertEqual(WGS84(), driver.get_crs())

    def test_get_dimensioned_variables(self):
        driver = self.get_driver()
        target = driver.get_dimensioned_variables()
        self.assertEqual(target, [u'UGID', u'STATE_FIPS', u'ID', u'STATE_NAME', u'STATE_ABBR'])

    def test_get_dimensions(self):
        driver = self.get_driver()
        actual = driver.get_dimensions().dimensions[MPI_RANK]
        desired = {None: {'dimensions': [Dimension(name='ocgis_geom', size=51, size_current=51, src_idx='auto')],
                          'groups': {}}}
        self.assertEqual(actual, desired)

    def test_get_dump_report(self):
        driver = self.get_driver()
        lines = driver.get_dump_report()
        self.assertTrue(len(lines) > 5)

    def test_get_field(self):
        driver = self.get_driver()
        field = driver.get_field()
        self.assertPrivateValueIsNone(field)
        self.assertEqual(len(field), 7)
        self.assertIsInstance(field.geom, GeometryVariable)
        self.assertIsInstance(field.crs, CoordinateReferenceSystem)
        self.assertIsNone(field.time)
        for v in field.values():
            self.assertIsNotNone(v.value)

        # Test slicing does not break loading from file.
        field = driver.get_field()
        self.assertPrivateValueIsNone(field)
        sub = field.geom[10, 15, 25].parent
        self.assertPrivateValueIsNone(sub)
        self.assertEqual(len(sub.dimensions[constants.NAME_GEOMETRY_DIMENSION]), 3)

    def test_get_variable_collection(self):
        driver = self.get_driver()
        vc = driver.get_variable_collection()
        self.assertEqual(len(vc), 7)
        for v in vc.values():
            if not isinstance(v, CoordinateReferenceSystem):
                self.assertEqual(len(v.dimensions[0]), 51)
                self.assertIsNone(v._value)

    def test_inspect(self):
        raise SkipTest('inspect needs to be reorganized')
        driver = self.get_driver()
        with self.print_scope() as ps:
            driver.inspect()
        self.assertTrue(len(ps.storage) >= 1)

    def test_metadata(self):
        driver = self.get_driver()
        m = driver.metadata
        self.assertIsInstance(m, dict)
        self.assertIsInstance(m['groups'], dict)
        self.assertTrue(len(m) > 2)
        self.assertIn('variables', m)

    def test_open(self):
        driver = self.get_driver()
        sci = driver.open(rd=driver.rd)
        self.assertIsInstance(sci, GeomCabinetIterator)
        self.assertFalse(sci.as_spatial_dimension)

    def test_write_variable_collection(self):
        # Test writing the field to file.
        field = self.get_driver().get_field()
        path1 = self.get_temporary_file_path('out1.shp')
        path2 = self.get_temporary_file_path('out2.shp')
        fiona_crs = get_fiona_crs(field, None)
        fiona_schema = get_fiona_schema(field, field.geom, None, None)
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

        # Attempt to write without a geometry variable.
        v = Variable('a', value=[1, 2], dimensions='bb')
        field = OcgField(variables=v)
        path = self.get_temporary_file_path('out.shp')
        with self.assertRaises(ValueError):
            field.write(path, driver=DriverVector)

        # Test writing a field with two-dimensional geometry storage.
        value = [Point(1, 2), Point(3, 4), Point(5, 6), Point(6, 7), Point(8, 9), Point(10, 11)]
        gvar = GeometryVariable(value=value, name='points')
        gvar.reshape(2, 3, dimensions=['lat', 'lon'])
        var1 = Variable(name='dummy', value=[6, 7, 8], dimensions=['a'])
        var2 = Variable(name='some_lats', value=[41, 41], dimensions=['lat'])
        var3 = Variable(name='some_lons', value=[0, 90, 280], dimensions=['lon'])
        var4 = Variable(name='data', value=np.random.rand(4, 3, 2), dimensions=['time', 'lon', 'lat'])
        vc = VariableCollection(variables=[gvar, var1, var2, var3, var4])
        path = self.get_temporary_file_path('2d.shp')
        vc.write(path, driver=DriverVector)
        read = RequestDataset(uri=path).get()
        self.assertTrue(len(read) > 2)
        self.assertEqual(read.keys(), ['some_lats', 'some_lons', 'data', constants.NAME_GEOMETRY_DIMENSION])

        # Test writing a subset of the variables.
        path = self.get_temporary_file_path('limited.shp')
        value = [Point(1, 2), Point(3, 4), Point(5, 6)]
        gvar = GeometryVariable(value=value, name='points', dimensions='points')
        var1 = Variable('keep', value=[1, 2, 3], dimensions='points')
        var2 = Variable('remove', value=[4, 5, 6], dimensions='points')
        vc = VariableCollection(variables=[gvar, var1, var2])
        vc.write(path, variable_names=['keep'], driver=DriverVector)
        read = RequestDataset(uri=path).get()
        self.assertNotIn('remove', read)
