import numpy as np

from ocgis.new_interface.field import FieldBundle
from ocgis.new_interface.geom import SpatialContainer
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
            grid = self.get_gridxy()
            kwargs['spatial'] = SpatialContainer(grid=grid)
        if 'attrs' not in kwargs:
            kwargs['attrs'] = self.attrs
        if 'extra' not in kwargs:
            extra = VariableCollection()
            extra.add_variable(Variable(name='height', value=2.0))
            kwargs['extra'] = extra
        if 'level' not in kwargs:
            level = BoundedVariable(name='level', value=[100, 200, 300], dtype=np.float32)
            level.set_extrapolated_bounds(name='level_bounds')
            kwargs['level'] = level
        if 'realization' not in kwargs:
            realization = Variable('realization', value=[1, 2, 3], dtype=np.int32)
            kwargs['realization'] = realization
        return FieldBundle(**kwargs)

    def test_init(self):
        fb = self.get_fieldbundle()
        self.assertEqual(fb.attrs, self.attrs)
        path = self.get_temporary_file_path('foo.nc')
        fb.write_netcdf(path)
        self.ncdump(path, header_only=False)
        vc = VariableCollection.read_netcdf(path)
        keys = vc.keys()
        self.assertIn('height', keys)
        self.assertIn('level', keys)
        self.assertIn('realization', keys)
        tkk
        print fb
        print fb.spatial.point.value
