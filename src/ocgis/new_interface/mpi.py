import itertools

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI_ENABLED = False
else:
    MPI_ENABLED = True


class DummyMPIComm(object):
    def Barrier(self):
        pass

    def bcast(self, *args, **kwargs):
        return args[0]

    def gather(self, *args, **kwargs):
        return [args[0]]

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def scatter(self, *args, **kwargs):
        return args[0][0]


if MPI_ENABLED:
    MPI_COMM = MPI.COMM_WORLD
else:
    MPI_COMM = DummyMPIComm()
MPI_SIZE = MPI_COMM.Get_size()
MPI_RANK = MPI_COMM.Get_rank()


def create_slices(length, size=MPI_SIZE):
    step = int(np.ceil(float(length) / size))
    indexes = [None] * size
    start = 0
    for ii in range(size):
        stop = start + step
        if stop > length:
            stop = length
        index_element = slice(start, stop)
        indexes[ii] = index_element
        start = stop
    return tuple(indexes)


def dgather(elements):
    grow = elements[0]
    for idx in range(1, len(elements)):
        for k, v in elements[idx].iteritems():
            grow[k] = v
    return grow


def ogather(elements):
    ret = np.array(elements, dtype=object)
    return ret


def hgather(elements):
    n = sum([e.shape[0] for e in elements])
    fill = np.zeros(n, dtype=elements[0].dtype)
    start = 0
    for e in elements:
        shape_e = e.shape[0]
        if shape_e == 0:
            continue
        stop = start + shape_e
        fill[start:stop] = e
        start = stop
    return fill


def vgather(elements):
    n = sum([e.shape[0] for e in elements])
    fill = np.zeros((n, elements[0].shape[1]), dtype=elements[0].dtype)
    start = 0
    for e in elements:
        shape_e = e.shape
        if shape_e[0] == 0:
            continue
        stop = start + shape_e[0]
        fill[start:stop, :] = e
        start = stop
    return fill


def create_nd_slices(size, shape):
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
                slices = create_slices(ii_shape, sections)
                app_slices_ii_shape = slices
                remaining -= len(slices)
            slices_ii_shape.append(app_slices_ii_shape)
        ret = [slices for slices in itertools.product(*slices_ii_shape)]
    return tuple(ret)
