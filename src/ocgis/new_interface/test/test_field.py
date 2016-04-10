from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ocgis import constants
from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.grid import GridXY
from ocgis.new_interface.temporal import TemporalVariable
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable


class TestOcgField(AbstractTestNewInterface):
    def get_ocgfield(self, *args, **kwargs):
        return OcgField(*args, **kwargs)

    def test_init(self):
        f = self.get_ocgfield()
        self.assertIsInstance(f, OcgField)

    def test_combo_properties(self):
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
        sub = f[{'time': slice(1, 2)}]
        desired = OrderedDict([('time', (1,)), ('time_bounds', (1, 2)), ('other', (1,)), ('xc', (3,)), ('yc', (4,))])
        self.assertEqual(sub.shapes, desired)
        self.assertIsNone(sub.grid)
        sub.dimension_map['x']['variable'] = 'xc'
        sub.dimension_map['y']['variable'] = 'yc'

        # Test writing to netCDF will load attributes.
        path = self.get_temporary_file_path('foo.nc')
        sub.write_netcdf(path)
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
        # spatial_sub.write_netcdf(path)
        # self.ncdump(path)

    def test_combo_grid_mapping(self):
        # tdk: RESUME: determine way to identify "dimensionsed" variable. this should probably occur using the request dataset (i.e. required dimensions)
        raise self.ToTest('test grid_mapping_name applied to dimensioned variables')

    def test_combo_crs_and_grid_abstraction(self):
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


# class TestFieldBundle2(AbstractTestNewInterface):
#
#     def get_fieldbundle(self, **kwargs):
#         variables = []
#         if 'fields' not in kwargs:
#             dims = [Dimension('time'), Dimension('level', length=4)]
#             value = np.arange(0, 3 * 4).reshape(3, 4)
#             variable = BoundedVariable(value=value, name='bvar', dimensions=dims, units='kelvin')
#
#             value2 = np.swapaxes(value * 2, 0, 1)
#             dims.reverse()
#             variable2 = BoundedVariable(value=value2, name='cvar', dimensions=dims)
#
#             kwargs['fields'] = [variable, variable2]
#         if 'level' not in kwargs:
#             level = self.get_levelvariable()
#             variables.append(level)
#         if 'time' not in kwargs:
#             time = self.get_temporalvariable()
#             variables.append(time)
#         kwargs['variables'] = variables
#         return FieldBundle2(**kwargs)
#
#     def get_levelvariable(self):
#         level = BoundedVariable(value=[0, 25, 50, 75], name='level')
#         level.create_dimensions('level')
#         return level
#
#     def get_temporalvariable(self):
#         dim = Dimension('time')
#         time = TemporalVariable(value=[1, 2, 3], name='time', dimensions=dim)
#         return time
#
#     def test_init(self):
#         fb = self.get_fieldbundle()
#         self.assertEqual(fb.fields.keys(), ['bvar', 'cvar'])
#         self.assertAsSetEqual(fb.variables.keys(), ['bvar', 'cvar', 'time', 'level'])
#         self.assertIsInstance(fb.fields['bvar'], Variable)
#         self.assertIn('time', fb.variables)
#         self.assertIsNotNone(fb.time)
#
#     def test_getitem(self):
#         fb = self.get_fieldbundle()
#
#         sub = fb[1, 3]
#         self.assertEqual(sub.fields.keys(), ['bvar', 'cvar'])
#         vars_fb = fb.variables
#         vars_sub = sub.variables
#         for v in vars_sub.values():
#             self.assertTrue(all([ii == 1 for ii in v.shape]))
#         for v in vars_fb.values():
#             self.assertTrue(all([ii != 1 for ii in v.shape]))
#         for f, s in zip(vars_fb.values(), vars_sub.values()):
#             self.assertTrue(np.may_share_memory(f.value, s.value))
#         self.assertNumpyAll(vars_sub['time'].value, vars_fb['time'][1].value)
#         self.assertAsSetEqual(vars_sub.keys(), ['bvar', 'cvar', 'level', 'time'])
#         self.assertEqual(vars_fb.keys(), vars_sub.keys())
#
#         sub2 = fb[:, 3]
#         for field in sub2.fields.values():
#             if field.name == 'bvar':
#                 self.assertEqual(field.shape, (3, 1))
#             elif field.name == 'cvar':
#                 self.assertEqual(field.shape, (1, 3))
#             self.assertIsNone(field._backref)
#         self.assertEqual(sub2.variables['level'].shape, (1,))
#         self.assertEqual(sub2.variables['time'].shape, (3,))
#
#     def test_set_dimension_variable(self):
#         fb = self.get_fieldbundle()
#         fb.crs = WGS84()
#         fb.variables.pop('time')
#         self.assertIsNone(fb.time)
#         time = Variable(value=[4, 5, 6], name='time')
#         time.create_dimensions('time')
#         fb.variables.add_variable(time)
#         fb.set_dimension_variable('time', 'time')
#         self.assertNumpyAll(fb.time.value, time.value)
#         sub = fb[1, :]
#         for field in sub.fields.values():
#             for _, record in field.iter():
#                 self.assertIn(time.name, record)
#         bvar, cvar = sub.fields.values()
#         self.assertEqual(np.mean(cvar.value / 2), np.mean(bvar.value))
#         self.assertEqual(set((cvar.value / 2).flatten().tolist()),
#                          set(bvar.value.flatten().tolist()))
#         self.assertNumpyAll(sub.variables['time'].value, time[1].value)
#         self.assertNumpyAll(sub.time.value, time[1].value)
#         self.assertEqual(time.shape, (3,))
#         self.assertNumpyMayShareMemory(time.value, fb.time.value)
#         path = self.get_temporary_file_path('foo.nc')
#         sub.write_netcdf(path)
#         vc = VariableCollection.read_netcdf(path)
#         self.assertEqual(vc['time'].attrs['axis'], 'T')
#         self.assertEqual(vc['level'].attrs['axis'], 'L')
#         path2 = self.get_temporary_file_path('foo1.nc')
#         vc.write_netcdf(path2)
#         self.assertNcEqual(path, path2)
#
#     def test_spatial(self):
#         x = BoundedVariable(value=[1, 2, 3], dtype=float, name='x')
#         x.create_dimensions('x')
#         y = BoundedVariable(value=[10, 20, 30, 40], dtype=float, name='y')
#         y.create_dimensions('y')
#
#         fb = FieldBundle2(variables=[x, y])
#         fb.spatial.grid.x.value[:] = 50
#         self.assertTrue(np.all(fb.variables['x'].value == 50))
#         spatial = fb.spatial
#         desired_crs = CoordinateReferenceSystem(epsg=2136)
#         spatial.grid.crs = desired_crs
#         fb2 = FieldBundle2(spatial=spatial)
#         self.assertEqual(desired_crs, fb2.crs)
#         self.assertTrue(np.all(fb2.x.value == 50))
#         path = self.get_temporary_file_path('foo.nc')
#         fb2.write_netcdf(path)
#         # self.ncdump(path)
#
#         value = np.array([Point(1, 2), Point(3, 4), Point(5, 6)], dtype=object)
#         point = PointArray(value=value, name='geom', crs=WGS84())
#         point.create_dimensions('ngeom')
#         fb3 = FieldBundle2(variables=[point])
#         self.assertIsInstance(fb3.spatial.point, PointArray)
#         self.assertIsNone(fb3.spatial.grid)
#         sheep = Variable(name='sheep', value=[1, 2, 3])
#         sheep.create_dimensions('ngeom')
#         fb3.variables.add_variable(sheep)
#         sub = fb3[1]
#         self.assertEqual(sub.spatial.shape, (1,))
#         self.assertTrue(sub.spatial.point.value[0].almost_equals(value[1]))
#         self.assertEqual(sub.crs, WGS84())

# class TestFieldBundle(AbstractTestNewInterface):
#     def setUp(self):
#         self.attrs = {'some': 'notes', 'we_are_number': 1}
#         super(TestFieldBundle, self).setUp()
#
#     def get_fieldbundle(self, **kwargs):
#         if 'name' not in kwargs:
#             kwargs['name'] = 'fb1'
#         if 'spatial' not in kwargs:
#             grid = self.get_gridxy(crs=Spherical())
#             kwargs['spatial'] = SpatialContainer(grid=grid)
#         if 'attrs' not in kwargs:
#             kwargs['attrs'] = self.attrs
#         if 'extra' not in kwargs:
#             extra = VariableCollection()
#             extra.add_variable(Variable(name='height', value=2.0))
#             kwargs['extra'] = extra
#         if 'level' not in kwargs:
#             level = BoundedVariable(name='level', value=[100, 200, 300, 400, 500], dtype=np.float32)
#             level.set_extrapolated_bounds(name='level_bounds')
#             kwargs['level'] = level
#         if 'time' not in kwargs:
#             time = TemporalVariable(name='time', value=[850, 900, 950, 1000, 1050, 1100, 1150])
#             time.set_extrapolated_bounds(name='time_bounds')
#             kwargs['temporal'] = time
#         if 'realization' not in kwargs:
#             realization = Variable('realization', value=[1, 2], dtype=np.int32)
#             kwargs['realization'] = realization
#         fb = FieldBundle(**kwargs)
#         if 'fields' not in kwargs:
#             np.random.seed(1)
#             value = np.random.rand(*fb.shape)
#             var = Variable(name='tas', alias='tas', value=value, units='K')
#             schema = {'realization': 0, 'time': 1, 'level': 2, 'y': 3, 'x': 4}
#             fb.create_field(var, schema=schema)
#         return fb
#
#     def test_tdk(self):
#         np.random.seed(1)
#         value = np.random.rand(5, 4, 3)
#         v = Variable(value=value, name='tas')
#         v.create_dimensions(['time', 'y', 'x'])
#         backref = VariableCollection(variables=[v])
#         t = TemporalVariable(value=[1, 2, 3, 4, 5])
#         t.create_dimensions('time')
#         grid = self.get_gridxy()
#         grid.create_dimensions()
#         spatial = SpatialContainer(grid=grid)
#
#         t._backref = backref
#         sub_t = t[2:4]
#         backref = sub_t._backref
#         t._backref = None
#
#         spatial.grid._backref = backref
#         sub_s = spatial[1:4, 1]
#         backref = sub_s.grid._backref
#         sub_s.grid._backref = None
#
#     def test(self):
#         # Test with no instrumented dimensions.
#         np.random.seed(1)
#         fb = FieldBundle(name='m1')
#         var = Variable(name='output', value=[[1, 2, 3], [4, 5, 6]])
#         var.create_dimensions(['one', 'two'])
#         fb.create_field(var)
#         sub = fb[1, 1]
#         np.testing.assert_equal(sub.fields['output'].value, [[5]])
#         self.assertEqual(fb.shape, (2, 3))
#
#         # Test with an instrumented dimension.
#         time = TemporalVariable(value=[1, 2, 3, 4, 5])
#         fb = FieldBundle(name='m2', time=time)
#         self.assertIsNotNone(fb.time)
#         var = Variable(name='output2', value=np.random.rand(50, time.shape[0]))
#         var.create_dimensions(['bs', 'time'])
#         fb.create_field(var)
#         sub = fb[20:40, 1:3]
#         np.testing.assert_equal(sub.time.value, [2, 3])
#         self.assertEqual(fb.time.shape, (5,))
#         self.assertTrue(np.may_share_memory(sub.time.value, fb.time.value))
#
#         # Test with two instrumented dimensions.
#         grid = self.get_gridxy()
#         spatial = SpatialContainer(grid=grid)
#         fb = FieldBundle(name='m3', time=time, spatial=spatial)
#         self.assertIsNone(fb.shape)
#         var = Variable(name='tas', value=np.random.rand(fb.time.shape[0], fb.spatial.shape[0], fb.spatial.shape[1]))
#         var.create_dimensions(['time', 'y', 'x'])
#         fb.create_field(var)
#         sub = fb[0:2, 2:4, 1]
#         self.assertEqual(sub.spatial.grid.x.value, spatial.grid.x.value[1])
#         self.assertNumpyAll(sub.spatial.grid.y.value, spatial.grid.y.value[2:4])
#
#         # Test with a schema.
#         for rename_dimensions in [False, True]:
#             fb = FieldBundle(name='m4', time=time, spatial=spatial)
#             var = Variable(name='tas', value=np.random.rand(fb.time.shape[0],
#                                                             fb.spatial.shape[0],
#                                                             fb.spatial.shape[1],
#                                                             6))
#             var.create_dimensions(['the_time', 'lon', 'lat', 'six'])
#             fb.create_field(var, schema={'time': 'the_time', 'x': 'lon', 'y': 'lat'},
#                             rename_dimensions=rename_dimensions)
#             if not rename_dimensions:
#                 self.assertEqual(fb.spatial.grid.x.dimensions[0].name, 'lon')
#                 self.assertEqual(fb.time.dimensions[0].name, 'the_time')
#             sub = fb[0:2, 1:3, 1, -3:]
#             self.assertNumpyAll(sub.spatial.grid.x.value, spatial.grid.x.value[1:3])
#             self.assertEqual(sub.spatial.grid.y.value, spatial.grid.y.value[1])
#
#         path = self.get_temporary_file_path('foo.nc')
#         sub.write_netcdf(path)
#         self.ncdump(path)
#
#     def test_init(self):
#         fb = self.get_fieldbundle()
#         self.assertEqual(fb.attrs, self.attrs)
#         path = self.get_temporary_file_path('foo.nc')
#         fb.write_netcdf(path)
#         self.ncdump(path, header_only=True)
#         vc = VariableCollection.read_netcdf(path)
#         keys = vc.keys()
#         self.assertIn('height', keys)
#         self.assertIn('level', keys)
#         self.assertIn('realization', keys)
#         self.assertIn('time', keys)
#         self.assertIn(fb.spatial.crs.name, keys)
#         self.assertEqual(fb.shape, (2, 7, 5, 4, 3))
#         self.assertEqual(fb.shape_dict,
#                          OrderedDict([('realization', (2,)), ('time', (7,)), ('level', (5,)), ('spatial', (4, 3))]))
#         sub = fb[1, 1, 1, 1, 1]
#         for target in sub.dimensions:
#             print target.name
#             print target.shape
#         tkk
#         with self.nc_scope(path) as ds:
#             ncvar = ds.variables['tas']
#             self.assertEqual(ncvar.dimensions, ('time', 'x', 'y'))
#         tkk
#         print fb
#         print fb.spatial.point.value

