import itertools
import os
from copy import deepcopy

import numpy as np
from shapely.geometry import box

from ocgis.exc import EmptySubsetError
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.mpi import MPI_RANK, MPI_SIZE, MPI_COMM
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import get_iter, get_optimal_slice_from_array


def create_slices(lengths, ns):
    ret = []
    for length, n in zip(get_iter(lengths, dtype=int), get_iter(ns, dtype=int)):
        slices = [None] * n
        start = 0
        remaining = length
        nlocal = n
        for ii in range(n):
            step = int(np.ceil(float(remaining) / nlocal))
            stop = start + step
            if stop > length:
                stop = length
            index_element = slice(start, stop)
            slices[ii] = index_element
            remaining -= (stop - start)
            start = stop
            nlocal -= 1
        ret.append(slices)
    ret = [ii for ii in itertools.product(*ret)]
    return tuple(ret)

class Test(AbstractTestNewInterface):
    def test_create_slices(self):
        lengths = [4]
        ns = [2]
        actual = create_slices(lengths, ns)
        self.assertEqual(actual, ((slice(0, 2, None),), (slice(2, 4, None),)))

        lengths = [4, 5]
        ns = [2, 2]
        actual = create_slices(lengths, ns)
        self.assertEqual(actual, ((slice(0, 2, None), slice(0, 3, None)), (slice(0, 2, None), slice(3, 5, None)),
                                  (slice(2, 4, None), slice(0, 3, None)), (slice(2, 4, None), slice(3, 5, None))))

    def test(self):
        # tdk: test no dimension i.e. len(dimension) == 0

        def update_map_slices(map_slices, key, target):
            for idx, slc in enumerate(slices):
                map_slices[idx][key] = target.__getitem__(slc)
            return map_slices

        t = Dimension('t', 35)
        l = Dimension('z', 11)
        y = Dimension('y', 7)
        x = Dimension('x', 5)
        dimensions = [t, l, y, x]
        variable = Variable(name='distribute_me', dimensions=dimensions)
        n_elements = reduce(lambda i, j: i * j, variable.shape)
        value = np.arange(n_elements).reshape(variable.shape)
        variable.value = value
        # print variable.value

        n = 5
        actual = create_slices([len(d) for d in variable.dimensions[2:]], n)

        vscatter = []
        for idx in range(n):
            y_slice = actual[0][idx]
            x_slice = actual[1][idx]
            vscatter.append(variable.__getitem__({'y': y_slice, 'x': x_slice}))

            # path_shp_grid = self.get_temporary_file_path('grid.shp')

    def test_get_local_to_global_slices(self):
        slices_global = (slice(2, 4, None), slice(0, 2, None))
        slices_local = (slice(0, 1, None), slice(0, 2, None))

        lm = get_local_to_global_slices(slices_global, slices_local)
        self.assertEqual(lm, (slice(2, 3, None), slice(0, 2, None)))

    def test_tdk(self):
        self.assertEqual(MPI_SIZE, 4)

        subset = box(100.7, 39.71, 102.30, 42.30)

        if MPI_RANK == 0:
            variable = GeometryVariable(value=subset)
            write_fiona_htmp(variable, 'subset')
            grid = self.get_gridxy(with_dimensions=True)
            write_fiona_htmp(grid, 'grid')
        else:
            grid = None

        grid_sub = get_mpi_grid_get_subset_bbox(grid, subset)

        if MPI_RANK == 0:
            write_fiona_htmp(grid_sub, 'grid_sub')
            desired = [[[40.0, 40.0], [41.0, 41.0], [42.0, 42.0]], [[101.0, 102.0], [101.0, 102.0], [101.0, 102.0]]]
            desired = np.array(desired)
            self.assertNumpyAll(grid_sub.value_stacked, desired)
            self.assertFalse(np.any(grid_sub.get_mask()))
        else:
            self.assertIsNone(grid_sub)

    MPI_COMM.Barrier()


def get_mpi_grid_get_subset_bbox(grid, subset):
    if MPI_RANK == 0:
        slices_global = create_slices([len(d) for d in grid.dimensions], (2, 2))
        g_scatter = [grid[slc] for slc in slices_global]
    else:
        g_scatter = None
    grid_local = MPI_COMM.scatter(g_scatter, root=0)

    write_fiona_htmp(grid_local, 'rank{}'.format(MPI_RANK))

    try:
        sub, slc = grid_local.get_subset_bbox(*subset.bounds, return_indices=True)
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

        fill_grid = grid[start_row: stop_row, start_col:stop_col]
        return fill_grid
    else:
        return None


def get_local_to_global_slices(slices_global, slices_local):
    ga = [np.arange(s.start, s.stop) for s in slices_global]
    lm = [get_optimal_slice_from_array(ga[idx][slices_local[idx]]) for idx in range(len(slices_local))]
    lm = tuple(lm)
    return lm


def write_fiona_htmp(obj, name):
    path = os.path.join('/home/benkoziol/htmp/ocgis', 'out_{}.shp'.format(name))
    obj.write_fiona(path)
