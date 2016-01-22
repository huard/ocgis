import numpy as np
from numpy.testing.utils import assert_array_equal

from ocgis.new_interface.dimension import Dimension, SourcedDimension
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable


class TestDimension(AbstractTestNewInterface):

    def test_init(self):
        dim = Dimension('foo')
        self.assertEqual(dim.name, 'foo')
        self.assertIsNone(dim.length)

        dim = Dimension('foo', length=23)
        self.assertEqual(dim.length, 23)

    def test_attach_variable(self):
        # tdk: test unlimited dimensions
        # tdk: test attaching a variable with the same dimension as the target
        dim = Dimension('foo', 3)
        var = Variable(value=[1, 2, 3], name='foobar')
        dim.attach_variable(var)
        var.dimensions = Dimension('loop', 3)
        self.assertIsNone(dim._variable.dimensions)
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            dim.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            ncvar = ds.variables['foobar']
            self.assertEqual(ncvar.dimensions, ('foo',))

        # Test with an unlimited dimension.
        dim = Dimension('time')
        value = [4, 5, 6, 7, 8]
        var = Variable('time_data', value, dimensions=dim)
        dim.attach_variable(var)
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            dim.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            ncvar = ds.variables['time_data']
            self.assertEqual(ncvar.dimensions, ('time',))
            ncdim_time = ds.dimensions['time']
            self.assertTrue(ncdim_time.isunlimited())
        sub = dim[2:4]
        self.assertIsNotNone(sub._variable)
        self.assertNumpyAll(sub._variable.value, var.value[2:4])

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

    def test_init(self):
        dim = self.get()
        self.assertNumpyAll(dim._src_idx, np.arange(0, 10, dtype=SourcedDimension._default_dtype))
        self.assertEqual(dim._src_idx.shape[0], 10)

    def test_copy(self):
        sd = self.get()
        self.assertIsNotNone(sd._src_idx)
        sd2 = sd.copy()
        self.assertTrue(np.may_share_memory(sd._src_idx, sd2._src_idx))
        sd3 = sd2[2:5]
        self.assertEqual(sd, sd2)
        self.assertNotEqual(sd2, sd3)
        self.assertTrue(np.may_share_memory(sd2._src_idx, sd._src_idx))
        self.assertTrue(np.may_share_memory(sd2._src_idx, sd3._src_idx))
        self.assertTrue(np.may_share_memory(sd3._src_idx, sd._src_idx))

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

        dim = self.get()
        sub = dim[3:-1]
        np.testing.assert_equal(sub._src_idx, [3, 4, 5, 6, 7, 8])

        dim = self.get()
        sub = dim[-3:]
        self.assertEqual(sub._src_idx.shape[0], sub.length)

        dim = self.get()
        sub = dim[-7:-3]
        self.assertEqual(sub._src_idx.shape[0], sub.length)

        dim = self.get()
        sub = dim[:-3]
        self.assertEqual(sub._src_idx.shape[0], sub.length)
