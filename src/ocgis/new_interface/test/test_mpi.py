import itertools
from unittest import SkipTest

import numpy as np
from mpi4py.MPI import COMM_NULL

from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.mpi import MPI_SIZE, MPI_COMM, create_nd_slices, hgather, \
    get_optimal_splits, get_rank_bounds, OcgMpi, MPI_RANK, get_global_to_local_slice
from ocgis.new_interface.ocgis_logging import log
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.test.base import attr
from ocgis.util.helpers import get_local_to_global_slices


class Test(AbstractTestNewInterface):
    def test_groups(self):
        if MPI_SIZE == 1:
            raise SkipTest('MPI only')
        world_group = MPI_COMM.Get_group()
        sub_group = world_group.Incl([0, 1])
        new_comm = MPI_COMM.Create(sub_group)
        if new_comm != COMM_NULL:
            log.debug(new_comm.Get_size())

        if new_comm != COMM_NULL:
            sub_group.Free()
            new_comm.Free()

    def test_get_optimal_splits(self):
        size = 11
        shape = (4, 3)
        splits = get_optimal_splits(size, shape)
        self.assertEqual(splits, (3, 3))

        size = 2
        shape = (4, 3)
        splits = get_optimal_splits(size, shape)
        self.assertEqual(splits, (2, 1))

    def test_create_nd_slices2(self):

        size = (1, 1)
        shape = (4, 3)
        actual = create_nd_slices(size, shape)
        self.assertEqual(actual, ((slice(0, 4, None), slice(0, 3, None)),))

        size = (2, 1)
        shape = (4, 3)
        actual = create_nd_slices(size, shape)
        self.assertEqual(actual, ((slice(0, 2, None), slice(0, 3, None)), (slice(2, 4, None), slice(0, 3, None))))

        size = (4, 2)
        shape = (4, 3)
        actual = create_nd_slices(size, shape)
        to_test = np.arange(12).reshape(*shape)
        pieces = []
        for a in actual:
            pieces.append(to_test[a].reshape(-1))
        self.assertNumpyAll(hgather(pieces).reshape(*shape), to_test)

    def test_get_rank_bounds(self):

        def _run_(arr, nproc):
            desired = arr.sum()

            actual = 0
            for pet in range(nproc):
                bounds = get_rank_bounds(len(arr), nproc=nproc, pet=pet)
                if bounds is None:
                    self.assertTrue(pet >= (nproc - len(arr)))
                else:
                    actual += arr[bounds[0]:bounds[1]].sum()

            assert np.isclose(actual, desired)

        lengths = [1, 2, 3, 4, 100, 333, 1333, 10001]
        nproc = [1, 2, 3, 4, 1000, 1333]

        for l, n in itertools.product(lengths, nproc):
            if l >= n:
                arr = np.random.rand(l) * 100.0
                _run_(arr, n)

        # Test with Nones.
        res = get_rank_bounds(10)
        self.assertEqual(res, [0, 10])

        # Test outside the number of elements.
        res = get_rank_bounds(4, nproc=1000, pet=900)
        self.assertIsNone(res)

        # Test on the edge.
        ret = get_rank_bounds(5, nproc=8, pet=5)
        self.assertIsNone(ret)

    def test_get_local_to_global_slices(self):
        # tdk: consider removing this function
        slices_global = (slice(2, 4, None), slice(0, 2, None))
        slices_local = (slice(0, 1, None), slice(0, 2, None))

        lm = get_local_to_global_slices(slices_global, slices_local)
        self.assertEqual(lm, (slice(2, 3, None), slice(0, 2, None)))

    def test_get_global_to_local_slice(self):
        start_stop = (1, 4)
        bounds_local = (0, 3)
        desired = (1, 3)
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (1, 4)
        bounds_local = (3, 5)
        desired = (0, 1)
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (1, 4)
        bounds_local = (4, 8)
        desired = None
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (3, 4)
        bounds_local = (3, 4)
        desired = (0, 1)
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (10, 20)
        bounds_local = (8, 10)
        desired = None
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (10, 20)
        bounds_local = (12, 15)
        desired = (0, 3)
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)


class TestOcgMpi(AbstractTestNewInterface):
    @attr('mpi-2', 'mpi-8')
    def test_system_bounds(self):
        """Test global and local bounds passing dimensions to the OCGIS MPI interface."""

        # Test passing dimensions to calculated local bounds.
        d1 = Dimension('d1', size=5, dist=True)
        d2 = Dimension('d2', size=10, dist=False)
        d3 = Dimension('d3', size=3, dist=True)
        dimensions = [d1, d2, d3]
        om = OcgMpi(dimensions=dimensions)

        actual = om.bounds_local
        desired_bounds_global = ((0, 5), (0, 10), (0, 3))
        self.assertEqual(om.bounds_global, desired_bounds_global)
        if MPI_SIZE == 1:
            desired = desired_bounds_global
        elif MPI_SIZE == 2:
            if MPI_RANK == 0:
                desired = ((0, 3), (0, 10), (0, 2))
            else:
                desired = ((3, 5), (0, 10), (2, 3))
        else:
            self.assertEqual(actual[1], (0, 10))
            if MPI_RANK < 3:
                for bl in actual:
                    self.assertIsNotNone(bl)
            else:
                self.assertIsNone(actual[0])
                self.assertIsNone(actual[2])

        if MPI_SIZE <= 2:
            self.assertEqual(actual, desired)
