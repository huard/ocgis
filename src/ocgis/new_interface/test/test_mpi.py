import itertools
from copy import deepcopy

import numpy as np
from shapely.geometry import box

from ocgis.exc import EmptySubsetError
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.mpi import MPI_RANK, MPI_SIZE, MPI_COMM
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.util.helpers import get_local_to_global_slices, get_optimal_slice_from_array


def create_slices(size, shape):
    remaining = size
    if size == 1:
        ret = [tuple([slice(0, s) for s in shape])]
    else:
        slices_ii_shape = []
        for idx_shape, ii_shape in enumerate(shape):
            if remaining <= 0:
                app_slices_ii_shape = [slice(0, ii_shape)]
            else:
                if remaining < ii_shape:
                    sections = remaining
                else:
                    sections = ii_shape
                # tdk: optimize: remove np.arange generation
                splits = np.array_split(np.arange(ii_shape), sections)
                slices = [get_optimal_slice_from_array(split) for split in splits]
                app_slices_ii_shape = slices
                remaining -= len(slices)
            slices_ii_shape.append(app_slices_ii_shape)
        ret = [slices for slices in itertools.product(*slices_ii_shape)]
    return tuple(ret)


class Test(AbstractTestNewInterface):
    def test_two_to_one(self):
        shape_global = (4, 3)
        idx_global = np.arange(12)
        print idx_global
        print idx_global.reshape(shape_global)

        actual = np.array([4, 5, 6, 7])
        bounds_rows = np.arange(shape_global[0])
        bounds_rows = np.hstack()

    def test_create_slices(self):
        size = 1
        shape = (4, 3)
        actual = create_slices(size, shape)
        self.assertEqual(actual, ((slice(0, 4, None), slice(0, 3, None)),))

        size = 2
        actual = create_slices(size, shape)
        desired = ((slice(0, 2, None), slice(0, 3)), (slice(2, 4, None), slice(0, 3)))
        self.assertEqual(actual, desired)

        size = 12
        shape = (4, 3)
        actual = create_slices(size, shape)
        desired = ((slice(0, 1, None), slice(0, 1, None)), (slice(0, 1, None), slice(1, 2, None)),
                   (slice(0, 1, None), slice(2, 3, None)), (slice(1, 2, None), slice(0, 1, None)),
                   (slice(1, 2, None), slice(1, 2, None)), (slice(1, 2, None), slice(2, 3, None)),
                   (slice(2, 3, None), slice(0, 1, None)), (slice(2, 3, None), slice(1, 2, None)),
                   (slice(2, 3, None), slice(2, 3, None)), (slice(3, 4, None), slice(0, 1, None)),
                   (slice(3, 4, None), slice(1, 2, None)), (slice(3, 4, None), slice(2, 3, None)))
        self.assertEqual(actual, desired)

        size = 5
        shape = (4, 3)
        actual = create_slices(size, shape)
        for a in actual:
            print a

    def test_get_local_to_global_slices(self):
        slices_global = (slice(2, 4, None), slice(0, 2, None))
        slices_local = (slice(0, 1, None), slice(0, 2, None))

        lm = get_local_to_global_slices(slices_global, slices_local)
        self.assertEqual(lm, (slice(2, 3, None), slice(0, 2, None)))

    def test_grid_get_intersects(self):
        subset = box(100.7, 39.71, 102.30, 42.30)

        if MPI_RANK == 0:
            variable = GeometryVariable(value=subset)
            write_fiona_htmp(variable, 'subset')
            grid = self.get_gridxy(with_dimensions=True)
            write_fiona_htmp(grid, 'grid')
        else:
            grid = None

        res = get_mpi_grid_get_intersects(grid, subset)
        if MPI_RANK == 0:
            grid_sub, slc_ret, = res

        if MPI_RANK == 0:
            write_fiona_htmp(grid_sub, 'grid_sub')

            grid_sub_serial, slc_ret_serial = grid.get_intersects(subset, return_indices=True)
            desired_serial = grid_sub_serial.value_stacked

            desired_manual = [[[40.0, 40.0], [41.0, 41.0], [42.0, 42.0]],
                              [[101.0, 102.0], [101.0, 102.0], [101.0, 102.0]]]
            desired_manual = np.array(desired_manual)
            for desc, desired, desired_slc in zip(['serial', 'mpi'], [desired_serial, desired_manual],
                                                  [slc_ret_serial, slc_ret]):
                self.assertEqual(slc_ret, desired_slc)
                self.assertNumpyAll(grid_sub.value_stacked, desired)
                self.assertFalse(np.any(grid_sub.get_mask()))
        else:
            self.assertIsNone(res)


def get_mpi_grid_get_intersects(grid, subset):
    if MPI_RANK == 0:
        if MPI_SIZE != 1:
            assert MPI_SIZE <= reduce(lambda x, y: x * y, grid.shape)

        slices_global = create_slices(MPI_SIZE, grid.shape)
        g_scatter = [grid[slc] for slc in slices_global]
    else:
        g_scatter = None
    grid_local = MPI_COMM.scatter(g_scatter, root=0)

    write_fiona_htmp(grid_local, 'rank{}'.format(MPI_RANK))

    try:
        sub, slc = grid_local.get_intersects(subset, return_indices=True)
    except EmptySubsetError:
        sub = None
        slc = None

    subs = MPI_COMM.gather(sub, root=0)
    slices_local = MPI_COMM.gather(slc, root=0)

    if MPI_RANK == 0:
        for rank, sub in enumerate(subs):
            if sub is not None:
                write_fiona_htmp(sub, 'not_empty_rank{}'.format(rank))

        as_global = []
        for global_slice, local_slice in zip(slices_global, slices_local):
            if local_slice is not None:
                app = get_local_to_global_slices(global_slice, local_slice)
            else:
                app = None
            as_global.append(app)

        slice_map_template = {'starts': [], 'stops': []}
        slice_map = {}
        keys = ['row', 'col']
        for key in keys:
            slice_map[key] = deepcopy(slice_map_template)
        for idx, sub in enumerate(subs):
            if sub is not None:
                for key, idx_slice in zip(keys, [0, 1]):
                    slice_map[key]['starts'].append(as_global[idx][idx_slice].start)
                    slice_map[key]['stops'].append(as_global[idx][idx_slice].stop)
        row, col = slice_map['row'], slice_map['col']
        start_row, stop_row = min(row['starts']), max(row['stops'])
        start_col, stop_col = min(col['starts']), max(col['stops'])

        slc_ret = (slice(start_row, stop_row), slice(start_col, stop_col))
        fill_grid = grid[slc_ret]
        return fill_grid, slc_ret
    else:
        return None

