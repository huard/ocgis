import itertools

import numpy as np

from ocgis.base import AbstractOcgisObject
from ocgis.util.helpers import get_optimal_slice_from_array

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


class OcgMpi(AbstractOcgisObject):
    def __init__(self, nelements=None):
        self._bounds_global = None
        self._bounds_local = None

        self.nelements = nelements
        self.rank = MPI_COMM.Get_rank()
        self.size = MPI_COMM.Get_size()
        self.comm = MPI_COMM

    @property
    def bounds_global(self):
        if self.nelements is not None:
            if self._bounds_global is None:
                self._bounds_global = [0, self.nelements]
        return self._bounds_global

    @property
    def bounds_local(self):
        if self.nelements is not None:
            if self._bounds_local is None:
                self._bounds_local = self.get_rank_bounds()
        return self._bounds_local

    def get_rank_bounds(self, nelements=None):
        nelements = nelements or self.nelements
        return get_rank_bounds(nelements, nproc=self.size, pet=self.rank)


def create_slices(length, size):
    # tdk: optimize: remove np.arange
    r = np.arange(length)
    sections = np.array_split(r, size)
    sections = [get_optimal_slice_from_array(s, check_diff=False) for s in sections]
    return sections


def dgather(elements):
    grow = elements[0]
    for idx in range(1, len(elements)):
        for k, v in elements[idx].iteritems():
            grow[k] = v
    return grow


def get_rank_bounds(nelements, nproc=None, pet=None):
    nproc = nproc or MPI_SIZE
    pet = pet or MPI_RANK
    esplit = int(np.ceil(float(nelements) / float(nproc)))

    if pet == 0:
        lbound = 0
    else:
        lbound = pet * esplit
    ubound = lbound + esplit

    if ubound >= nelements:
        ubound = nelements

    if lbound >= ubound:
        ret = None
    else:
        ret = [lbound, ubound]

    return ret


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


def create_nd_slices(splits, shape):
    ret = [None] * len(shape)
    for idx, (split, shp) in enumerate(zip(splits, shape)):
        ret[idx] = create_slices(shp, split)
    ret = [slices for slices in itertools.product(*ret)]
    return tuple(ret)


def get_optimal_splits(size, shape):
    n_elements = reduce(lambda x, y: x * y, shape)
    if size >= n_elements:
        splits = shape
    else:
        if size <= shape[0]:
            splits = [1] * len(shape)
            splits[0] = size
        else:
            even_split = int(np.power(size, 1.0 / float(len(shape))))
            splits = [None] * len(shape)
            for idx, shp in enumerate(shape):
                if even_split > shp:
                    fill = shp
                else:
                    fill = even_split
                splits[idx] = fill
    return tuple(splits)
