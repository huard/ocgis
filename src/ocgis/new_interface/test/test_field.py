from netCDF4 import OrderedDict

import numpy as np

from ocgis.interface.base.crs import Spherical
from ocgis.new_interface.field import FieldBundle
from ocgis.new_interface.geom import SpatialContainer
from ocgis.new_interface.temporal import TemporalVariable
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import VariableCollection, Variable, BoundedVariable


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
