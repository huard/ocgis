import datetime
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ocgis import constants
from ocgis.api.request.base import RequestDataset
from ocgis.interface.base.crs import CoordinateReferenceSystem, WGS84
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.grid import GridXY, expand_grid
from ocgis.new_interface.temporal import TemporalVariable
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import reduce_multiply


class TestOcgField(AbstractTestNewInterface):
    def get_ocgfield(self, *args, **kwargs):
        return OcgField(*args, **kwargs)

    def get_ocgfield_example(self):
        dtime = Dimension(name='time')
        t = TemporalVariable(value=[1, 2, 3, 4], name='time', dimensions=dtime)
        lon = Variable(value=[30., 40., 50., 60.], name='longitude', dimensions='lon')
        lat = Variable(value=[-10., -20., -30., -40., -50.], name='latitude', dimensions='lat')
        tas_shape = [t.shape[0], lat.shape[0], lon.shape[0]]
        tas = Variable(value=np.arange(reduce_multiply(tas_shape)).reshape(*tas_shape),
                       dimensions=('time', 'lat', 'lon'), name='tas')
        time_related = Variable(value=[7, 8, 9, 10], name='time_related', dimensions=dtime)
        garbage1 = Variable(value=[66, 67, 68], dimensions='three', name='garbage1')
        dmap = {'time': {'variable': t.name},
                'x': {'variable': lon.name},
                'y': {'variable': lat.name}}
        field = OcgField(variables=[t, lon, lat, tas, garbage1, time_related], dimension_map=dmap)
        return field

    def test_init(self):
        f = self.get_ocgfield()
        self.assertIsInstance(f, OcgField)

    def test_system_crs_and_grid_abstraction(self):
        f = OcgField(grid_abstraction='point')
        grid = self.get_gridxy(with_xy_bounds=True)
        f.add_variable(grid.x)
        f.add_variable(grid.y)

        crs = CoordinateReferenceSystem(epsg=2136, name='location')
        f.add_variable(crs)
        self.assertIsNone(f.crs)
        f.dimension_map['crs']['variable'] = crs.name
        f.dimension_map['x']['variable'] = grid.x.name
        f.dimension_map['y']['variable'] = grid.y.name
        self.assertEqual(f.grid.crs, crs)
        self.assertEqual(f.grid.point.crs, crs)
        self.assertEqual(f.grid.polygon.crs, crs)
        self.assertEqual(f.grid.abstraction, 'point')
        self.assertEqual(f.grid.abstraction_geometry.geom_type, 'Point')
        self.assertEqual(f.geom.geom_type, 'Point')
        self.assertEqual(len(f), 7)

    def test_system_grid_mapping(self):
        # tdk: RESUME: determine way to identify "dimensionsed" variable. this should probably occur using the request dataset (i.e. required dimensions)
        raise self.ToTest('test grid_mapping_name applied to dimensioned variables')

    def test_system_properties(self):
        """Test field properties."""

        time = TemporalVariable(value=[20, 30, 40], dimensions=['the_time'], dtype=float, name='time')
        time_bounds = TemporalVariable(value=[[15, 25], [25, 35], [35, 45]], dimensions=['times', 'bounds'],
                                       dtype=float, name='time_bounds')
        other = Variable(value=[44, 55, 66], name='other', dimensions=['times_again'])
        x = Variable(value=[1, 2, 3], name='xc', dimensions=['x'])
        y = Variable(value=[10, 20, 30, 40], name='yc', dimensions=['y'])

        crs = CoordinateReferenceSystem(epsg=2136)
        f = self.get_ocgfield(variables=[time, time_bounds, other, x, y])
        f2 = deepcopy(f)

        self.assertIsNone(f.realization)
        self.assertIsNone(f.time)
        f.dimension_map['time']['variable'] = time.name
        self.assertNumpyAll(f.time.value, time.value)
        self.assertEqual(f.time.attrs['axis'], 'T')
        self.assertIsNone(f.time.bounds)
        f.dimension_map['time']['bounds'] = time_bounds.name
        self.assertNumpyAll(f.time.bounds.value, time_bounds.value)
        self.assertIn('other', f.time.parent)

        f.dimension_map['time']['names'] += ['times', 'times_again']
        sub = f.get_field_slice({'time': slice(1, 2)})
        desired = OrderedDict([('time', (1,)), ('time_bounds', (1, 2)), ('other', (1,)), ('xc', (3,)), ('yc', (4,))])
        self.assertEqual(sub.shapes, desired)
        self.assertIsNone(sub.grid)
        sub.dimension_map['x']['variable'] = 'xc'
        sub.dimension_map['y']['variable'] = 'yc'

        # Test writing to netCDF will load attributes.
        path = self.get_temporary_file_path('foo.nc')
        sub.write(path)
        with self.nc_scope(path) as ds:
            self.assertEqual(ds.variables[x.name].axis, 'X')
            self.assertEqual(ds.variables[y.name].axis, 'Y')

        self.assertEqual(sub.x.attrs['axis'], 'X')
        self.assertEqual(sub.y.attrs['axis'], 'Y')
        self.assertIsInstance(sub.grid, GridXY)
        desired = OrderedDict([('time', (1,)), ('time_bounds', (1, 2)), ('other', (1,)), ('xc', (3,)), ('yc', (4,))])
        self.assertEqual(sub.shapes, desired)
        # sub.grid.expand()
        desired = OrderedDict(
            [('time', (1,)), ('time_bounds', (1, 2)), ('other', (1,)), ('xc', (3,)), ('yc', (4,))])
        self.assertEqual(sub.shapes, desired)
        self.assertIsNotNone(sub.grid.point)
        self.assertEqual(sub.shapes[sub.grid.point.name], (4, 3))

        # Test a subset.
        bbox = [1.5, 15, 2.5, 35]
        data = Variable(name='data', value=np.random.rand(3, 4), dimensions=['x', 'y'])
        f2.add_variable(data)
        f2.dimension_map['x']['variable'] = 'xc'
        f2.dimension_map['y']['variable'] = 'yc'
        spatial_sub = f2.grid.get_intersects(bbox).parent
        desired = OrderedDict(
            [('time', (3,)), ('time_bounds', (3, 2)), ('other', (3,)), ('xc', (2, 1)), ('yc', (2, 1)),
             ('data', (1, 2))])
        self.assertEqual(spatial_sub.shapes, desired)

        # path = self.get_temporary_file_path('foo.nc')
        # spatial_sub.write_netcdf(psath)
        # self.ncdump(path)

    def test_system_subsetting(self):
        """Test subsetting operations."""

        field = self.get_ocgfield_example()
        field.grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
        sub = field.time.get_between(datetime.datetime(1, 1, 2, 12, 0),
                                     datetime.datetime(1, 1, 4, 12, 0)).parent
        sub = sub.grid.get_intersects([35, -45, 55, -15]).parent
        self.assertTrue(sub.grid.is_vectorized)

    def test_dimensions(self):
        crs = CoordinateReferenceSystem(epsg=2136)
        field = OcgField(variables=[crs])
        self.assertEqual(len(field.dimensions), 0)

    def test_get_by_tag(self):
        v1 = Variable(name='tas')
        v2 = Variable(name='tasmax')
        v3 = Variable(name='tasmin')
        tags = {'avg': ['tas'], 'other': ['tasmax', 'tasmin']}
        field = OcgField(variables=[v1, v2, v3], tags=tags)
        t = field.get_by_tag('other')
        self.assertAsSetEqual([ii.name for ii in t], tags['other'])

    def test_time(self):
        units = [None, 'days since 2012-1-1']
        calendar = [None, '365_day']
        value = [10, 20]
        bounds = [[5, 15], [15, 25]]
        variable_type = [Variable, TemporalVariable]
        bounds_variable_type = [Variable, TemporalVariable]

        keywords = dict(units=units, calendar=calendar, variable_type=variable_type,
                        bounds_variable_type=bounds_variable_type)

        for k in self.iter_product_keywords(keywords):
            attrs = {'units': k.units, 'calendar': k.calendar}
            dimension_map = {'time': {'variable': 'time', 'bounds': 'time_bnds', 'attrs': attrs}}
            var = k.variable_type(name='time', value=value, attrs=attrs, dimensions=['one'])
            bounds_var = k.bounds_variable_type(name='time_bnds', value=bounds, dimensions=['one', 'two'])
            f = OcgField(variables=[var, bounds_var], dimension_map=dimension_map)
            self.assertTrue(len(f.dimension_map) > 1)
            self.assertTrue(f.time.has_bounds)
            self.assertIsInstance(f.time, TemporalVariable)
            self.assertIsInstance(f.time.bounds, TemporalVariable)
            self.assertEqual(f.time.value_datetime.shape, (2,))
            self.assertEqual(f.time.bounds.value_datetime.shape, (2, 2))
            if k.units is None:
                desired = constants.DEFAULT_TEMPORAL_UNITS
            else:
                desired = k.units
            self.assertEqual(f.time.units, desired)
            self.assertEqual(f.time.bounds.units, desired)
            if k.calendar is None:
                desired = constants.DEFAULT_TEMPORAL_CALENDAR
            else:
                desired = k.calendar
            self.assertEqual(f.time.calendar, desired)
            self.assertEqual(f.time.bounds.calendar, desired)


            # path = self.get_temporary_file_path('foo.nc')
            # f.write_netcdf(path)
            # self.ncdump(path)

    def test_write(self):
        # Test writing a basic grid.
        path = self.get_temporary_file_path('foo.nc')
        x = Variable(name='x', value=[1, 2], dimensions='x')
        y = Variable(name='y', value=[3, 4, 5, 6, 7], dimensions='y')
        dmap = {'x': {'variable': 'x'}, 'y': {'variable': 'y'}}
        field = OcgField(variables=[x, y], dimension_map=dmap)

        self.assertEqual(field.grid.parent['x'].value.shape, (2,))
        expand_grid(field.grid)
        self.assertTrue(field.grid.is_vectorized)

        field.write(path)
        out_field = RequestDataset(path).get()
        self.assertTrue(out_field.grid.is_vectorized)
        self.assertNumpyAll(field.grid.value_stacked, out_field.grid.value_stacked)

        # Test another grid.
        grid = self.get_gridxy(crs=WGS84())
        self.assertTrue(grid.is_vectorized)
        field = OcgField(grid=grid)
        self.assertTrue(field.grid_is_vectorized)
        self.assertTrue(field.grid.is_vectorized)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            field.write(ds)
        self.assertTrue(field.grid.is_vectorized)
        with self.nc_scope(path) as ds:
            self.assertNumpyAll(ds.variables[grid.x.name][:], grid.x.value[0, :])
            var = ds.variables[grid.y.name]
            self.assertNumpyAll(var[:], grid.y.value[:, 0])
            self.assertEqual(var.axis, 'Y')
            self.assertIn(grid.crs.name, ds.variables)

        # Test with 2-d x and y arrays.
        grid = self.get_gridxy(with_2d_variables=True)
        field = OcgField(grid=grid)
        path = self.get_temporary_file_path('out.nc')
        field.grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
        with self.nc_scope(path, 'w') as ds:
            field.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            self.assertNumpyAll(var[:], grid.y.value)

        # Test writing a vectorized grid with corners.
        grid = self.get_gridxy()
        field = OcgField(grid=grid)
        self.assertIsNotNone(field.grid.dimensions)
        self.assertFalse(field.grid.has_bounds)
        field.grid.set_extrapolated_bounds('xbnds', 'ybnds', 'corners')
        self.assertTrue(field.grid.is_vectorized)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            field.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path, 'r') as ds:
            self.assertEqual(['ydim'], [d for d in ds.variables['y'].dimensions])
            self.assertEqual(['xdim'], [d for d in ds.variables['x'].dimensions])
