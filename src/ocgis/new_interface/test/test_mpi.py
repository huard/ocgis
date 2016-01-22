import itertools

import numpy as np
from mpi4py import MPI
from shapely.geometry import box

from ocgis.exc import EmptySubsetError
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import get_iter


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

    def test_pickle(self):
        MPI_COMM = MPI.COMM_WORLD
        MPI_SIZE = MPI_COMM.Get_size()
        MPI_RANK = MPI_COMM.Get_rank()

        if MPI_RANK == 0:
            grid = self.get_gridxy()
            to_scatter = [grid, grid]
        else:
            to_scatter = None

        grid_local = MPI_COMM.scatter(to_scatter, root=0)
        print grid_local.x.value
        gathered = MPI_COMM.gather(grid_local, root=0)
        print gathered

    def test_tdk(self):
        MPI_COMM = MPI.COMM_WORLD
        MPI_SIZE = MPI_COMM.Get_size()
        MPI_RANK = MPI_COMM.Get_rank()

        self.assertEqual(MPI_SIZE, 4)

        if MPI_RANK == 0:
            grid = self.get_gridxy(with_dimensions=True)

            slices = create_slices([len(d) for d in grid.dimensions], (2, 2))

            g_scatter = []
            for slc in slices:
                g_scatter.append(grid.__getitem__(slc))
        else:
            g_scatter = None

        grid_local = MPI_COMM.scatter(g_scatter, root=0)

        path_local = '/home/benkoziol/htmp/rank_{}.shp'.format(MPI_RANK)
        grid_local.write_fiona(path_local)

        subset = box(100.7, 39.71, 101.32, 40.29)
        try:
            sub = grid_local.get_intersects(subset)
        except EmptySubsetError:
            sub = None

        subs = MPI_COMM.gather(sub, root=0)

        MPI_COMM.Barrier()
        thh

        # MPI_COMM.Barrier()
        # path = self.get_temporary_file_path('out.shp')
        # serial_sub.write_fiona(path)
        # grid.write_fiona(path_shp_grid)
        # new_slc = get_mapped_slice(new_slc, names_src, names_dst)
        # for v in vscatter:
        #     print v.shape
        # print actual

        thh

        key_dimension = 'dimension'
        map_slices = {idx: {'slice': slc} for idx, slc in enumerate(slices)}
        update_map_slices(map_slices, key_dimension, dimension)
        actual_length = 0
        for idx, sc in enumerate(map_slices.values()):
            actual_length += len(sc[key_dimension])
            self.assertIn('slice', sc)
        self.assertEqual(actual_length, len(dimension))

        var = Variable(value=value)
        update_map_slices(map_slices, 'variable', var)
        actual = []
        for v in map_slices.values():
            actual += v['variable'].value.tolist()
        self.assertEqual(np.mean(actual), value.mean())
        thh
