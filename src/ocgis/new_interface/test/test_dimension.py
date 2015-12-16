import numpy as np

from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.dimension import Dimension, SourcedDimension
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface


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

        # Test with negative indexing.
        dim = Dimension(name='geom', length=2)
        slc = slice(0, -1, None)
        self.assertEqual(dim[slc], Dimension('geom', length=1))

        dim = Dimension(name='geom', length=5)
        slc = slice(1, -2, None)
        a = np.arange(5)
        self.assertEqual(dim[slc], Dimension('geom', length=a[slc].shape[0]))

    def test_len(self):
        dim = Dimension('foo')
        self.assertEqual(len(dim), 0)


class TestSourcedDimension(AbstractTestNewInterface):
    def get(self, **kwargs):
        name = kwargs.pop('name', 'foo')
        kwargs['length'] = kwargs.get('length', 10)
        return SourcedDimension(name, **kwargs)

    def test_bases(self):
        self.assertEqual(SourcedDimension.__bases__, (Dimension,))

    def test_init(self):
        dim = self.get()
        self.assertNumpyAll(dim._src_idx, np.arange(0, 10, dtype=SourcedDimension._default_dtype))
        self.assertEqual(dim._src_idx.shape[0], 10)

    def test_eq(self):
        lhs = self.get()
        rhs = self.get()
        self.assertEqual(lhs, rhs)
        self.assertIsNone(lhs.__src_idx__)
        self.assertIsNone(rhs.__src_idx__)

        # Test when source index is loaded.
        lhs._src_idx
        rhs._src_idx
        self.assertEqual(lhs, rhs)

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

        sub = dim[:]
        self.assertEqual(len(sub), len(dim))

        dim = self.get()
        sub = dim[2:]
        self.assertEqual(len(sub), 8)
