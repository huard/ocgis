from netCDF4 import OrderedDict

import numpy as np

from ocgis.interface.base.crs import Spherical
from ocgis.new_interface.field import FieldBundle, DSlice, FieldBundle2
from ocgis.new_interface.geom import SpatialContainer
from ocgis.new_interface.temporal import TemporalVariable
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import VariableCollection, Variable, BoundedVariable


class TestDSlice(AbstractTestNewInterface):
    def test(self):
        ds = DSlice(['x', 'time', 'y'])
        slc = (1, 3, 4)
        res = ds.get_reordered(slc, ['time', 'y', 'x'])
        actual = (slice(4, 5, None), slice(1, 2, None), slice(3, 4, None))
        self.assertEqual(res, actual)

        np.random.seed(1)
        variable_value = np.random.rand(5, 3, 4)
        variable_value_actual = variable_value[4, 1, 3]
        self.assertEqual(variable_value_actual, variable_value[actual])


class TestFieldBundle2(AbstractTestNewInterface):

    def get_fieldbundle(self, **kwargs):
        if 'fields' not in kwargs:
            fields = VariableCollection()
            variable = BoundedVariable(value=[10, 20, 30], name='bvar')
            variable.create_dimensions('the_time')
            fields.add_variable(variable)
            kwargs['fields'] = fields
        if 'time' not in kwargs:
            time = self.get_temporalvariable()
            kwargs['time'] = time
        return FieldBundle2(**kwargs)

    def get_temporalvariable(self):
        time = TemporalVariable(value=[1, 2, 3])
        time.create_dimensions('temporal')
        return time

    def test_init(self):
        field = self.get_fieldbundle(fields=None)
        self.assertIsNotNone(field.time)

    def test_set_time_dimension(self):
        time = self.get_temporalvariable()
        fb = self.get_fieldbundle(time=None)
        self.assertIsNone(fb.time)
        fb.set_time_dimension(time, dim_name='the_time')
        self.assertNumpyAll(fb.time.value, time.value)
        self.assertNumpyAll(fb.fields['bvar'].dimensions_dict['the_time']._variable.value, time.value)


class TestFieldBundle(AbstractTestNewInterface):
    def setUp(self):
        self.attrs = {'some': 'notes', 'we_are_number': 1}
        super(TestFieldBundle, self).setUp()

    def get_fieldbundle(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'fb1'
        if 'spatial' not in kwargs:
            grid = self.get_gridxy(crs=Spherical())
            kwargs['spatial'] = SpatialContainer(grid=grid)
        if 'attrs' not in kwargs:
            kwargs['attrs'] = self.attrs
        if 'extra' not in kwargs:
            extra = VariableCollection()
            extra.add_variable(Variable(name='height', value=2.0))
            kwargs['extra'] = extra
        if 'level' not in kwargs:
            level = BoundedVariable(name='level', value=[100, 200, 300, 400, 500], dtype=np.float32)
            level.set_extrapolated_bounds(name='level_bounds')
            kwargs['level'] = level
        if 'time' not in kwargs:
            time = TemporalVariable(name='time', value=[850, 900, 950, 1000, 1050, 1100, 1150])
            time.set_extrapolated_bounds(name='time_bounds')
            kwargs['temporal'] = time
        if 'realization' not in kwargs:
            realization = Variable('realization', value=[1, 2], dtype=np.int32)
            kwargs['realization'] = realization
        fb = FieldBundle(**kwargs)
        if 'fields' not in kwargs:
            np.random.seed(1)
            value = np.random.rand(*fb.shape)
            var = Variable(name='tas', alias='tas', value=value, units='K')
            schema = {'realization': 0, 'time': 1, 'level': 2, 'y': 3, 'x': 4}
            fb.create_field(var, schema=schema)
        return fb

    def test_tdk(self):
        np.random.seed(1)
        value = np.random.rand(5, 4, 3)
        v = Variable(value=value, name='tas')
        v.create_dimensions(['time', 'y', 'x'])
        backref = VariableCollection(variables=[v])
        t = TemporalVariable(value=[1, 2, 3, 4, 5])
        t.create_dimensions('time')
        grid = self.get_gridxy()
        grid.create_dimensions()
        spatial = SpatialContainer(grid=grid)

        t._backref = backref
        sub_t = t[2:4]
        backref = sub_t._backref
        t._backref = None

        spatial.grid._backref = backref
        sub_s = spatial[1:4, 1]
        backref = sub_s.grid._backref
        sub_s.grid._backref = None

        print backref['tas'].shape

    def test(self):
        # Test with no instrumented dimensions.
        np.random.seed(1)
        fb = FieldBundle(name='m1')
        var = Variable(name='output', value=[[1, 2, 3], [4, 5, 6]])
        var.create_dimensions(['one', 'two'])
        fb.create_field(var)
        sub = fb[1, 1]
        np.testing.assert_equal(sub.fields['output'].value, [[5]])
        self.assertEqual(fb.shape, (2, 3))

        # Test with an instrumented dimension.
        time = TemporalVariable(value=[1, 2, 3, 4, 5])
        fb = FieldBundle(name='m2', time=time)
        self.assertIsNotNone(fb.time)
        var = Variable(name='output2', value=np.random.rand(50, time.shape[0]))
        var.create_dimensions(['bs', 'time'])
        fb.create_field(var)
        sub = fb[20:40, 1:3]
        np.testing.assert_equal(sub.time.value, [2, 3])
        self.assertEqual(fb.time.shape, (5,))
        self.assertTrue(np.may_share_memory(sub.time.value, fb.time.value))

        # Test with two instrumented dimensions.
        grid = self.get_gridxy()
        spatial = SpatialContainer(grid=grid)
        fb = FieldBundle(name='m3', time=time, spatial=spatial)
        self.assertIsNone(fb.shape)
        var = Variable(name='tas', value=np.random.rand(fb.time.shape[0], fb.spatial.shape[0], fb.spatial.shape[1]))
        var.create_dimensions(['time', 'y', 'x'])
        fb.create_field(var)
        sub = fb[0:2, 2:4, 1]
        self.assertEqual(sub.spatial.grid.x.value, spatial.grid.x.value[1])
        self.assertNumpyAll(sub.spatial.grid.y.value, spatial.grid.y.value[2:4])

        # Test with a schema.
        for rename_dimensions in [False, True]:
            fb = FieldBundle(name='m4', time=time, spatial=spatial)
            var = Variable(name='tas', value=np.random.rand(fb.time.shape[0],
                                                            fb.spatial.shape[0],
                                                            fb.spatial.shape[1],
                                                            6))
            var.create_dimensions(['the_time', 'lon', 'lat', 'six'])
            fb.create_field(var, schema={'time': 'the_time', 'x': 'lon', 'y': 'lat'},
                            rename_dimensions=rename_dimensions)
            if not rename_dimensions:
                self.assertEqual(fb.spatial.grid.x.dimensions[0].name, 'lon')
                self.assertEqual(fb.time.dimensions[0].name, 'the_time')
            sub = fb[0:2, 1:3, 1, -3:]
            self.assertNumpyAll(sub.spatial.grid.x.value, spatial.grid.x.value[1:3])
            self.assertEqual(sub.spatial.grid.y.value, spatial.grid.y.value[1])

        path = self.get_temporary_file_path('foo.nc')
        sub.write_netcdf(path)
        self.ncdump(path)

    # tdk: test wrong variable shape
    # tdk: test with a mask
    # tdk: test w/out all dimensions on input field
    # tdk: test w/out standard ordering (i.e. x before y)
    # tdk: test with a geometry instead of a grid as the only data
    def test_init(self):
        fb = self.get_fieldbundle()
        self.assertEqual(fb.attrs, self.attrs)
        path = self.get_temporary_file_path('foo.nc')
        fb.write_netcdf(path)
        self.ncdump(path, header_only=True)
        vc = VariableCollection.read_netcdf(path)
        keys = vc.keys()
        self.assertIn('height', keys)
        self.assertIn('level', keys)
        self.assertIn('realization', keys)
        self.assertIn('time', keys)
        self.assertIn(fb.spatial.crs.name, keys)
        self.assertEqual(fb.shape, (2, 7, 5, 4, 3))
        self.assertEqual(fb.shape_dict,
                         OrderedDict([('realization', (2,)), ('time', (7,)), ('level', (5,)), ('spatial', (4, 3))]))
        sub = fb[1, 1, 1, 1, 1]
        for target in sub.dimensions:
            print target.name
            print target.shape
        tkk
        with self.nc_scope(path) as ds:
            ncvar = ds.variables['tas']
            self.assertEqual(ncvar.dimensions, ('time', 'x', 'y'))
        tkk
        print fb
        print fb.spatial.point.value
