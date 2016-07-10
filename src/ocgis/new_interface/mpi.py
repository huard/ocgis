import itertools
from collections import OrderedDict

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
    def __init__(self, comm=None):
        comm = comm or MPI_COMM
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.dimensions = OrderedDict()

    def create_dimension(self, *args, **kwargs):
        from dimension import Dimension
        group = kwargs.pop('group', 'root')
        dim = Dimension(*args, **kwargs)
        if group not in self.dimensions:
            self.dimensions[group] = OrderedDict()
        self.dimensions[group][dim.name] = dim
        dim.bounds_global = (0, len(dim))
        return dim

    def update_dimension_bounds(self, group='root'):
        lengths = [len(dim) for dim in self.dimensions[group].values() if dim.dist]
        if min(lengths) < self.size:
            the_size = min(lengths)
        else:
            the_size = self.size
        for dim in self.dimensions[group].values():
            omb = OcgMpiBounds(nelements=len(dim), size=the_size)
            bounds_local = omb.bounds_local
            dim.bounds_local = bounds_local
            if bounds_local is not None and dim._src_idx is not None:
                start, stop = bounds_local
                dim._src_idx = dim._src_idx[start:stop]


class OcgMpiBounds(AbstractOcgisObject):
    def __init__(self, nelements=None, dimensions=None, size=None):
        # Require element count or dimensions.
        assert nelements is not None or dimensions is not None

        # Object works off element counts or dimensions exclusively. Passing both is not an option.
        if dimensions is not None:
            assert nelements is None
            if not isinstance(dimensions, (list, tuple)):
                dimensions = [dimensions]
            if size is None:
                lengths = [None] * len(dimensions)
                for idx, dim in enumerate(dimensions):
                    if dim.dist:
                        gl, gu = dim.mpi.bounds_global
                        lengths[idx] = gu - gl
                min_length = min([l for l in lengths if l is not None])
                if min_length < MPI_SIZE:
                    size = min_length
                    for d in dimensions:
                        if d.dist:
                            d.mpi.size = size
        if nelements is not None:
            assert dimensions is None

        self.dimensions = dimensions
        self.nelements = nelements
        self.rank = MPI_RANK
        self.size = size or MPI_SIZE

    @property
    def bounds_global(self):
        if self.nelements is not None:
            ret = (0, self.nelements)
        else:
            ret = tuple([d.mpi.bounds_global for d in self.dimensions])
        return ret

    @property
    def bounds_local(self):
        if self.size == 1:
            ret = self.bounds_global
        else:
            if self.nelements is not None:
                ret = self.get_rank_bounds()
            else:
                bl = [None] * len(self.dimensions)
                for idx, dim in enumerate(self.dimensions):
                    if dim.dist:
                        fill = dim.mpi.bounds_local
                    else:
                        fill = dim.mpi.bounds_global
                    bl[idx] = fill
                ret = tuple(bl)
        return ret

    @property
    def size_global(self):
        lower, upper = self.bounds_global
        return upper - lower

    @property
    def size_local(self):
        bounds = self.bounds_local
        if bounds is None:
            ret = None
        else:
            ret = bounds[1] - bounds[0]
        return ret

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


def get_global_to_local_slice(start_stop, bounds_local):
    """
    :param start_stop: Two-element, integer sequence for the start and stop global indices.
    :type start_stop: tuple
    :param bounds_local: Two-element, integer sequence describing the local bounds.
    :type bounds_local: tuple
    :return: Two-element integer sequence mapping the global to the local slice. If the local bounds are outside the
     global slice, ``None`` will be returned.
    :rtype: tuple or None
    """
    start, stop = start_stop
    lower, upper = bounds_local

    new_start = start
    if start >= upper:
        new_start = None
    else:
        if new_start < lower:
            new_start = lower

    if stop <= lower:
        new_stop = None
    elif upper < stop:
        new_stop = upper
    else:
        new_stop = stop

    if new_start is None or new_stop is None:
        ret = None
    else:
        ret = (new_start - lower, new_stop - lower)
    return ret


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
        ret = (lbound, ubound)

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
