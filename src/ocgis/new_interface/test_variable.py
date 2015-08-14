import numpy as np

from ocgis import RequestDataset
from ocgis.interface.base.attributes import Attributes
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable, SourcedVariable


class TestVariable(AbstractTestNewInterface):
    def test_bases(self):
        self.assertEqual(Variable.__bases__, (AbstractInterfaceObject, Attributes))

    def test_init(self):
        value = [2, 3, 4, 5, 6, 7]
        time = Dimension('time', length=len(value))

        var = Variable('time_value', value=value, dimensions=time)
        self.assertEqual(var.dimensions, (time,))
        self.assertEqual(id(time), id(var.dimensions[0]))
        self.assertEqual(var.name, 'time_value')
        self.assertEqual(var.alias, var.name)
        self.assertEqual(var.shape, (len(value),))
        self.assertNumpyAll(var.value, np.ma.array(value))
        sub = var[2:4]
        self.assertIsInstance(sub, Variable)
        self.assertEqual(sub.shape, (2,))

        dtype = np.float32
        fill_value = 33.0
        var = Variable('foo', value=value, dimensions=time, dtype=dtype, fill_value=fill_value)
        self.assertEqual(var.dtype, dtype)
        self.assertEqual(var.value.dtype, dtype)
        self.assertEqual(var.value.fill_value, fill_value)

        var = Variable('foo')
        self.assertEqual(var.shape, tuple())


class TestSourcedVariable(AbstractTestNewInterface):
    def get(self):
        data = self.test_data.get_rd('cancm4_tas')
        sv = SourcedVariable('tas', data=data)
        self.assertIsNone(sv._value)
        self.assertIsNone(sv._dimensions)
        return sv

    def test_bases(self):
        self.assertEqual(SourcedVariable.__bases__, (Variable,))

    def test_init(self):
        sv = self.get()
        self.assertIsInstance(sv._data, RequestDataset)

    def test_getitem(self):
        sv = self.get()
        sub = sv[10:20, 5, 6]
        self.assertEqual(sub.shape, (10, 1, 1))
        self.assertIsNone(sub.dimensions[0].length)
        self.assertEqual(sub.dimensions[0].length_current, 10)

    def test_get_dimensions(self):
        sv = self.get()
        self.assertTrue(len(sv.dimensions), 3)

    def test_get_dimensions_from_source_data(self):
        sv = self.get()
        dims = sv._get_dimensions_from_source_data_()
        self.assertIsNone(dims[0].length)
        self.assertEqual(dims[0].length_current, 3650)
        self.assertEqual(['time', 'lat', 'lon'], [d.name for d in dims])
        for d in dims:
            self.assertIsNone(d.__src_idx__)
