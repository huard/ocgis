import itertools
from copy import deepcopy
from unittest import SkipTest

import numpy as np
from mpi4py.MPI import COMM_NULL

from ocgis.constants import MPIDistributionMode
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.mpi import MPI_SIZE, MPI_COMM, create_nd_slices, hgather, \
    get_optimal_splits, get_rank_bounds, OcgMpi, get_global_to_local_slice, MPI_RANK, variable_scatter
from ocgis.new_interface.ocgis_logging import log
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable, VariableCollection
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
            length = len(arr)
            for pet in range(nproc + 3):
                bounds = get_rank_bounds(length, size=nproc, rank=pet)
                if bounds is None:
                    try:
                        self.assertTrue(pet >= (nproc - length) or (nproc > length and pet >= length))
                        self.assertTrue(length < nproc or pet >= nproc)
                    except AssertionError:
                        self.log.debug('   args: {}, {}, {}'.format(length, nproc, pet))
                        self.log.debug(' bounds: {}'.format(bounds))
                        raise
                else:
                    actual += arr[bounds[0]:bounds[1]].sum()

            try:
                assert np.isclose(actual, desired)
            except AssertionError:
                self.log.debug('   args: {}, {}, {}'.format(length, nproc, pet))
                self.log.debug(' bounds: {}'.format(bounds))
                self.log.debug(' actual: {}'.format(actual))
                self.log.debug('desired: {}'.format(desired))
                raise

        lengths = [1, 2, 3, 4, 5, 6, 8, 100, 333, 1333, 10001]
        nproc = [1, 2, 3, 4, 5, 6, 8, 1000, 1333]

        for l, n in itertools.product(lengths, nproc):
            arr = np.random.rand(l) * 100.0
            _run_(arr, n)

        # Test with Nones.
        res = get_rank_bounds(10)
        self.assertEqual(res, (0, 10))

        # Test outside the number of elements.
        res = get_rank_bounds(4, size=1000, rank=900)
        self.assertIsNone(res)

        # Test on the edge.
        ret = get_rank_bounds(5, size=8, rank=5)
        self.assertIsNone(ret)

        # Test with more elements than procs.
        _run_(np.arange(6), 5)

        # Test with rank higher than size.
        res = get_rank_bounds(6, size=5, rank=6)
        self.assertIsNone(res)

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

    @attr('mpi')
    def test_variable_scatter(self):
        var_value = np.arange(5, dtype=float) + 50
        var_mask = np.array([True, True, False, True, False])

        if MPI_RANK == 0:
            src_mpi = OcgMpi()

            dim = src_mpi.create_dimension('five', 5, src_idx='auto')
            dim_src_idx = dim._src_idx.copy()
            var = Variable('the_five', value=var_value, mask=var_mask, dimensions=dim)
            var.set_extrapolated_bounds('the_five_bounds', 'bounds')
            var_bounds_value = var.bounds.value

            dest_mpi = deepcopy(src_mpi)
            for drank in range(MPI_SIZE):
                dest_mpi.get_dimension('five', rank=drank).dist = True
            dest_mpi.update_dimension_bounds()
        else:
            var, src_mpi, dest_mpi, var_bounds_value, dim_src_idx = [None] * 5

        svar, dest_mpi = variable_scatter(var, dest_mpi)
        var_bounds_value = MPI_COMM.bcast(var_bounds_value)
        dim_src_idx = MPI_COMM.bcast(dim_src_idx)

        if MPI_RANK > 4:
            self.assertTrue(svar.is_empty)
        else:
            dest_dim = dest_mpi.get_dimension('five')
            self.assertNumpyAll(var_value[slice(*dest_dim.bounds_local)], svar.value)
            self.assertNumpyAll(var_mask[slice(*dest_dim.bounds_local)], svar.get_mask())
            self.assertNumpyAll(var_bounds_value[slice(*dest_dim.bounds_local)], svar.bounds.value)
            self.assertNumpyAll(dim_src_idx[slice(*dest_dim.bounds_local)], svar.dimensions[0]._src_idx)
            self.assertNumpyAll(dim_src_idx[slice(*dest_dim.bounds_local)], svar.bounds.dimensions[0]._src_idx)

    @attr('mpi')
    def test_variable_scatter_ndimensions(self):
        if MPI_RANK == 0:
            r = Dimension('realization', 3)
            t = Dimension('time', 365)
            l = Dimension('level', 10)
            y = Dimension('y', 90, dist=True)
            x = Dimension('x', 360, dist=True)

            var = Variable('tas', dimensions=[r, t, l, y, x])

            dest_mpi = OcgMpi()
            for d in var.dimensions:
                dest_mpi.add_dimension(d)
            dest_mpi.update_dimension_bounds()
        else:
            var, dest_mpi = [None] * 2

        svar, dest_mpi = variable_scatter(var, dest_mpi)

        self.assertIsNotNone(svar)


class TestOcgMpi(AbstractTestNewInterface):
    def get_ocgmpi_01(self):
        s = Dimension('first_dist', size=5, dist=True, src_idx='auto')
        ompi = OcgMpi()
        ompi.add_dimension(s)
        ompi.create_dimension('not_dist', size=8, dist=False)
        ompi.create_dimension('another_dist', size=6, dist=True)
        ompi.create_dimension('another_not_dist', size=100, dist=False)
        ompi.create_dimension('coordinate_reference_system', size=0)
        self.assertIsNotNone(s._src_idx)
        return ompi

    def test_init(self):
        ompi = OcgMpi(size=2)
        self.assertEqual(len(ompi.mapping), 2)

        dim_x = Dimension('x', 5, dist=False)
        dim_y = Dimension('y', 11, dist=True)
        var_tas = Variable('tas', value=np.arange(0, 5 * 11).reshape(5, 11), dimensions=(dim_x, dim_y))
        thing = Variable('thing', value=np.arange(11) * 10, dimensions=(dim_y,))

        vc = VariableCollection(variables=[var_tas, thing])
        child = VariableCollection(name='younger')
        vc.add_child(child)
        childer = VariableCollection(name='youngest')
        child.add_child(childer)
        dim_six = Dimension('six', 6)
        hidden = Variable('hidden', value=[6, 7, 8, 9, 0, 10], dimensions=dim_six)
        childer.add_variable(hidden)

        ompi.add_dimensions([dim_x, dim_y])
        ompi.add_dimension(dim_six, group=hidden.group)
        ompi.add_variables([var_tas, thing])
        ompi.add_variable(hidden, dist=MPIDistributionMode.ISOLATED, ranks=1)

        hidden_actual = ompi.get_variable(hidden, rank=1)
        self.assertEqual(hidden_actual['dist'], MPIDistributionMode.ISOLATED)
        variable_ranks_actual = ompi.get_variable_ranks(hidden)
        self.assertEqual(variable_ranks_actual, (1,))

        variable_ranks_actual = ompi.get_variable_ranks(thing)
        self.assertEqual(variable_ranks_actual, (0, 1))

        # self.pprint(ompi.mapping)
        # import ipdb;ipdb.set_trace()

    @attr('mpi')
    def test(self):
        ompi = OcgMpi()
        src_idx = [2, 3, 4, 5, 6]
        dim = ompi.create_dimension('foo', size=5, group='subroot', dist=True, src_idx=src_idx)
        self.assertEqual(dim, ompi.get_dimension(dim.name, group='subroot'))
        self.assertEqual(dim.bounds_local, (0, len(dim)))
        self.assertFalse(dim.is_empty)
        ompi.update_dimension_bounds()
        with self.assertRaises(ValueError):
            ompi.update_dimension_bounds()

        if ompi.size == 2:
            if MPI_RANK == 0:
                self.assertEqual(dim.bounds_local, (0, 3))
                self.assertEqual(dim._src_idx.tolist(), [2, 3, 4])
            elif MPI_RANK == 1:
                self.assertEqual(dim.bounds_local, (3, 5))
                self.assertEqual(dim._src_idx.tolist(), [5, 6])
        elif ompi.size == 8:
            if MPI_RANK <= 4:
                self.assertEqual(len(dim), 1)
                self.assertEqual(dim._src_idx[0], src_idx[MPI_RANK])
            else:
                self.assertTrue(dim.is_empty)

        # Test with multiple dimensions.
        d1 = Dimension('d1', size=5, dist=True, src_idx='auto')
        d2 = Dimension('d2', size=10, dist=False)
        d3 = Dimension('d3', size=3, dist=True)
        dimensions = [d1, d2, d3]
        ompi = OcgMpi()
        for dim in dimensions:
            ompi.add_dimension(dim)
        ompi.update_dimension_bounds()
        bounds_local = ompi.get_bounds_local()
        if ompi.size <= 2:
            desired = {(1, 0): ((0, 5), (0, 10), (0, 3)),
                       (2, 0): ((0, 3), (0, 10), (0, 3)),
                       (2, 1): ((3, 5), (0, 10), (0, 3))}
            self.assertAsSetEqual(bounds_local, desired[(ompi.size, MPI_RANK)])
        else:
            if MPI_RANK <= 1:
                self.assertTrue(dimensions[0]._src_idx.shape[0] <= 2)
            for dim in dimensions:
                if MPI_RANK > 4:
                    self.assertTrue(dim.is_empty)
                else:
                    self.assertFalse(dim.is_empty)

        # Test adding an existing dimension.
        ompi = OcgMpi()
        ompi.create_dimension('one')
        with self.assertRaises(ValueError):
            ompi.create_dimension('one')

    def test_create_or_get_group(self):
        ocmpi = OcgMpi()
        _ = ocmpi._create_or_get_group_(['moon', 'base'])
        _ = ocmpi._create_or_get_group_(None)
        _ = ocmpi._create_or_get_group_('flower')
        ocmpi.create_dimension('foo', group=['moon', 'base'])
        ocmpi.create_dimension('end_of_days')
        ocmpi.create_dimension('start_of_days')

        actual = ocmpi.get_dimension('foo', ['moon', 'base'])
        desired = Dimension('foo')
        self.assertEqual(actual, desired)

        desired = {0: {None: {'variables': {},
                              'dimensions': {'end_of_days': Dimension(name='end_of_days', size=None, size_current=None),
                                             'start_of_days': Dimension(name='start_of_days', size=None,
                                                                        size_current=None)},
                              'groups': {'flower': {'variables': {}, 'dimensions': {}, 'groups': {}},
                                         'moon': {'variables': {}, 'dimensions': {},
                                                  'groups': {'base': {'variables': {},
                                                                      'dimensions': {
                                                                          'foo': Dimension(
                                                                              name='foo',
                                                                              size=None,
                                                                              size_current=None)},
                                                                      'groups': {}}}}}}}}
        # print(ocmpi.mapping)
        # self.pprint(ocmpi.mapping)
        self.assertDictEqual(ocmpi.mapping, desired)

    @attr('mpi')
    def test_gather_dimensions(self):
        raise SkipTest('Gather dimensions will be implemented differently.')
        ompi = self.get_ocgmpi_01()
        desired = deepcopy(ompi.get_group())

        ompi.update_dimension_bounds()

        if MPI_SIZE == 1:
            root = 0
        else:
            root = 1

        actual = ompi.gather_dimensions(root=root)

        if ompi.rank == root:
            for actual_dim, desired_dim in zip(actual['dimensions'], desired['dimensions']):
                try:
                    self.assertEqual(actual_dim, desired_dim)
                except:
                    self.log.debug(actual_dim.__dict__)
                    self.log.debug(desired_dim.__dict__)
                    raise
                self.assertFalse(actual_dim.is_empty)
                self.assertEqual(actual_dim, ompi.get_dimension(actual_dim.name))
        else:
            self.assertIsNone(actual)

    @attr('mpi')
    def test_update_dimension_bounds(self):
        ompi = OcgMpi()
        dim = ompi.create_dimension('five', 5, dist=True)
        ompi.update_dimension_bounds()
        self.assertEqual(dim.bounds_global, (0, 5))
        if MPI_SIZE > 1:
            self.assertNotEqual(dim.bounds_local, dim.bounds_global)
            if MPI_SIZE == 2:
                if MPI_RANK == 0:
                    self.assertEqual(dim.bounds_local, (0, 3))
                else:
                    self.assertEqual(dim.bounds_local, (3, 5))

        # Test updating on single processor.
        if MPI_SIZE == 1:
            ompi = OcgMpi(size=2)
            dim = ompi.create_dimension('five', 5, dist=True)
            ompi.update_dimension_bounds()
            self.assertEqual(dim.bounds_global, (0, 5))
            for rank in range(2):
                actual = ompi.get_dimension('five', rank=rank)
                self.assertEqual(actual.bounds_global, (0, 5))
                if rank == 0:
                    self.assertEqual(actual.bounds_local, (0, 3))
                else:
                    self.assertEqual(actual.bounds_local, (3, 5))

            # Test two dimensions.
            ompi = OcgMpi(size=2)
            ompi.create_dimension('lat', 64, dist=True)
            ompi.create_dimension('lon', 128, dist=True)
            ompi.update_dimension_bounds()
            for rank in range(2):
                lat = ompi.get_dimension('lat', rank=rank)
                self.assertEqual(lat.bounds_local, (0, 64))
                lon = ompi.get_dimension('lon', rank=rank)
                if rank == 0:
                    self.assertEqual(lon.bounds_local, (0, 64))
                else:
                    self.assertEqual(lon.bounds_local, (64, 128))
