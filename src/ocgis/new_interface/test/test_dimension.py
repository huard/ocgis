import numpy as np

from ocgis.new_interface.dimension import Dimension, SourcedDimension
from ocgis.new_interface.mpi import MPI_SIZE, MPI_RANK
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.test.base import attr


class TestDimension(AbstractTestNewInterface):

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
    def get_sourced_dimension(self, **kwargs):
        name = kwargs.pop('name', 'foo')
        kwargs['length'] = kwargs.get('length', 10)
        return SourcedDimension(name, **kwargs)

    def test_init(self):
        dim = self.get_sourced_dimension()
        self.assertNumpyAll(dim._src_idx, np.arange(0, 10, dtype=SourcedDimension._default_dtype))
        self.assertEqual(dim._src_idx.shape[0], 10)

    @attr('mpi-2', 'mpi-5', 'mpi-8')
    def test_init_mpi(self):
        kwds = {'dist': [True, False], 'src_idx': [None, [10, 20, 30, 40, 50]]}
        for k in self.iter_product_keywords(kwds):
            dim = SourcedDimension('the_d', 5, src_idx=k.src_idx, dist=k.dist)
            self.assertEqual(len(dim), 5)
            self.assertEqual(dim.mpi.bounds_global, (0, 5))
            if MPI_SIZE == 1:
                self.assertEqual(dim.mpi.bounds_local, (0, 5))
            if k.dist:
                if MPI_SIZE == 5:
                    bounds = dim.mpi.bounds_local
                    self.assertEqual(bounds[1] - bounds[0], 1)
                elif MPI_SIZE == 8:
                    if MPI_RANK > 4:
                        self.assertIsNone(dim._src_idx)
                    else:
                        self.assertEqual(len(dim._src_idx), 1)
            else:
                self.assertEqual(dim.mpi.bounds_local, dim.mpi.bounds_global)

        # Test with zero length.
        dim = SourcedDimension('zero', length=0, dist=True)
        self.assertEqual(len(dim), 0)
        self.assertIsNone(dim._src_idx)

        # Test unlimited dimension.
        dim = SourcedDimension('unlimited', length=None, dist=True, src_idx=[3, 4, 5, 6])
        self.assertEqual(len(dim), 4)
        if MPI_SIZE == 2:
            if MPI_RANK == 0:
                self.assertEqual(dim._src_idx.tolist(), [3, 4])
            elif MPI_RANK == 1:
                self.assertEqual(dim._src_idx.tolist(), [5, 6])
            else:
                self.assertIsNone(dim._src_idx)
        elif MPI_SIZE == 1:
            self.assertIsNotNone(dim._src_idx)

        # Test some length designation is needed for an unlimited dimension in the distributed case.
        with self.assertRaises(ValueError):
            SourcedDimension('ua', dist=True)

    def test_copy(self):
        sd = self.get_sourced_dimension()
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
        lhs = self.get_sourced_dimension()
        rhs = self.get_sourced_dimension()
        self.assertEqual(lhs, rhs)
        self.assertIsNone(lhs.__src_idx__)
        self.assertIsNone(rhs.__src_idx__)

        # Test when source index is loaded.
        lhs._src_idx
        rhs._src_idx
        self.assertEqual(lhs, rhs)

    @attr('mpi-2', 'mpi-5', 'mpi-8')
    def test_gather(self):
        s = SourcedDimension('first_dist', length=5, dist=True)
        self.assertIsNone(s.__src_idx__)
        actual = s.gather(root=1)
        if MPI_RANK == 1 or MPI_SIZE == 1:
            self.assertFalse(actual.dist)
            self.assertEqual(actual, s)
            self.assertTrue(np.may_share_memory(actual._src_idx, s._src_idx))
            self.assertEqual(actual._src_idx.sum(), np.arange(5).sum())
        else:
            self.assertIsNone(actual)

    def test_getitem(self):
        dim = self.get_sourced_dimension()

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

        dim = self.get_sourced_dimension()
        sub = dim[2:]
        self.assertEqual(len(sub), 8)

        dim = self.get_sourced_dimension()
        sub = dim[3:-1]
        np.testing.assert_equal(sub._src_idx, [3, 4, 5, 6, 7, 8])

        dim = self.get_sourced_dimension()
        sub = dim[-3:]
        self.assertEqual(sub._src_idx.shape[0], sub.length)

        dim = self.get_sourced_dimension()
        sub = dim[-7:-3]
        self.assertEqual(sub._src_idx.shape[0], sub.length)

        dim = self.get_sourced_dimension()
        sub = dim[:-3]
        self.assertEqual(sub._src_idx.shape[0], sub.length)

    @attr('mpi-2', 'mpi-5', 'mpi-8')
    def test_getitem_mpi(self):
        # tdk: test slicing with an integer
        # tdk: test slicing with bool
        # tdk: test slicing with integer array
        # tdk: test slicing with None
        sd = SourcedDimension('yui', 5, src_idx=[10, 20, 30, 40, 50], dist=True)
        sub = sd[1:4]
        if MPI_SIZE == 2:
            if MPI_RANK == 0:
                self.assertEqual(sub._src_idx.tolist(), [20, 30])
            else:
                self.assertEqual(sub._src_idx.tolist(), [40])
        elif MPI_SIZE > 2:
            raise NotImplementedError

    @attr('mpi-2')
    def test_scatter(self):
        d = SourcedDimension('scatter', 6, dist=False)
        self.assertEqual(len(d._src_idx), 6)
        ds = d.scatter()
        if MPI_SIZE == 2:
            self.log.debug(ds.mpi.bounds_local)
            self.log.debug(ds._src_idx)
            llb, lub = ds.mpi.bounds_local
            self.assertEqual(lub - llb, 3)
            self.assertEqual(len(ds._src_idx), 3)