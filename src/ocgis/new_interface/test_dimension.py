import numpy as np

from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.dimension import Dimension, SourcedDimension
from ocgis.new_interface.test_new_interface import AbstractTestNewInterface


class TestDimension(AbstractTestNewInterface):
    def test_bases(self):
        self.assertEqual(Dimension.__bases__, (AbstractInterfaceObject,))

    def test_init(self):
        dim = Dimension('foo')
        self.assertEqual(dim.name, 'foo')
        self.assertIsNone(dim.length)

        dim = Dimension('foo', length=23)
        self.assertEqual(dim.length, 23)

    def test_getitem(self):
        dim = Dimension('foo', length=50)
        sub = dim[30:40]
        self.assertEqual(len(sub), 10)

        dim = Dimension('foo', length=None)
        sub = dim[400:500]
        self.assertEqual(len(sub), 100)

    def test_len(self):
        dim = Dimension('foo')
        self.assertEqual(len(dim), 0)


class TestSourcedDimension(AbstractTestNewInterface):
    def get(self):
        return SourcedDimension('foo', length=10)

    def test_bases(self):
        self.assertEqual(SourcedDimension.__bases__, (Dimension,))

    def test_init(self):
        dim = self.get()
        self.assertNumpyAll(dim._src_idx, np.arange(0, 10, dtype=SourcedDimension._default_dtype))
        self.assertEqual(dim._src_idx.shape[0], 10)

    def test_getitem(self):
        dim = self.get()

        sub = dim[4]
        self.assertEqual(sub.length, 1)

        sub = dim[4:5]
        self.assertEqual(sub.length, 1)

        sub = dim[4:6]
        self.assertEqual(sub.length, 2)

        sub = dim[[4, 5, 6]]
        self.assertEqual(sub.length, 3)

        sub = dim[[2, 4, 6]]
        self.assertEqual(sub.length, 3)
        self.assertNumpyAll(sub._src_idx, dim._src_idx[[2, 4, 6]])
