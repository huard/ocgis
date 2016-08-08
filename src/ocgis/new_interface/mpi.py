import itertools
from copy import deepcopy

import numpy as np

from ocgis.base import AbstractOcgisObject
from ocgis.exc import DimensionNotFound
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
    def __init__(self, size=MPI_SIZE):
        self.size = size
        self.dimensions = {}
        for rank in range(size):
            self.dimensions[rank] = get_template_rank_dict()

        self.has_updated_dimensions = False

    def add_dimension(self, dim, group=None, force=False):
        from dimension import Dimension
        if not isinstance(dim, Dimension):
            raise ValueError('"dim" must be a "Dimension" object.')

        for rank in range(self.size):
            if rank > 0:
                dim = dim.copy()
            the_group = self._create_or_get_group_(group, rank=rank)
            if not force and any([dim.name == d.name for d in the_group['dimensions']]):
                raise ValueError('Dimension with name "{}" already in group "{}".'.format(dim.name, group))
            else:
                the_group['dimensions'].append(dim)

    def create_dimension(self, *args, **kwargs):
        from dimension import Dimension
        group = kwargs.pop('group', None)
        dim = Dimension(*args, **kwargs)
        self.add_dimension(dim, group=group)
        return dim

    def gather_dimensions(self, group=None, root=0):
        target_group = self.get_group(group=group)
        new_dimensions = [None] * len(target_group['dimensions'])
        for idx, dim in enumerate(target_group['dimensions']):
            if dim.dist:
                dim = self._gather_dimension_(dim.name, group=group, root=root)
            else:
                pass
            new_dimensions[idx] = dim

        if self.rank == root:
            target_group['dimensions'] = []
            for dim in new_dimensions:
                self.add_dimension(dim, group=group)
        else:
            pass

        if self.rank == root:
            return self.get_group(group=group)
        else:
            return None

    def get_bounds_local(self, group=None, rank=MPI_RANK):
        the_group = self.get_group(group=group, rank=rank)
        ret = [dim.bounds_local for dim in the_group['dimensions']]
        return tuple(ret)

    def get_dimension(self, name, group=None, rank=MPI_RANK):
        group_data = self.get_group(group=group, rank=rank)
        ret = None
        for d in group_data['dimensions']:
            if d.name == name:
                ret = d
                break
        if ret is None:
            raise ValueError('Dimension not found: {}'.format(name))
        return ret

    def get_dimensions(self, names, **kwargs):
        ret = [self.get_dimension(name, **kwargs) for name in names]
        return ret

    def get_group(self, group=None, rank=MPI_RANK):
        return self._create_or_get_group_(group, rank=rank)

    def iter_groups(self, rank=MPI_RANK):
        from ocgis.api.request.driver.base import iter_all_group_keys, get_group
        dimensions = self.dimensions[rank]
        for group_key in iter_all_group_keys(dimensions):
            group_data = get_group(dimensions, group_key)
            yield group_key, group_data

    def update_dimension_bounds(self, rank='all'):
        """

        :param rank: If ``'all'``, update across all ranks. Otherwise, update for the integer rank provided.
        :type rank: str/int
        """
        if self.has_updated_dimensions:
            raise ValueError('Dimensions already updated.')

        if rank == 'all':
            ranks = range(self.size)
        else:
            ranks = [rank]

        for rank in ranks:
            for _, group_data in self.iter_groups(rank=rank):
                dimensions = tuple(group_data['dimensions'])
                lengths = [len(dim) for dim in dimensions if dim.dist]

                # Choose the size of the distribution group. There needs to be at least one element per rank.
                the_size = self.size
                if len(lengths) != 0:
                    if min(lengths) < self.size:
                        the_size = min(lengths)
                    else:
                        pass
                else:
                    pass

                # Update the local bounds for the dimension.
                for dim in dimensions:
                    # For distributed bounds, the local and global bounds will be different.
                    if dim.dist:
                        # Before updating bounds, set the global bounds so they will always be available to associated
                        # variables.
                        dim.bounds_global = (0, len(dim))
                        # Use this to calculate the local bounds for a dimension.
                        omb = MpiBoundsCalculator(nelements=len(dim), size=the_size, rank=rank)
                        bounds_local = omb.bounds_local
                        if bounds_local is not None:
                            start, stop = bounds_local
                            if dim._src_idx is None:
                                src_idx = None
                            else:
                                src_idx = dim._src_idx[start:stop]
                            dim.set_size(stop - start, src_idx=src_idx)
                        else:
                            # If there are no local bounds, the dimension is empty.
                            dim.convert_to_empty()
                            # Remove any source indices on empty dimensions.
                            dim._src_idx = None
                        dim.bounds_local = bounds_local
                    else:
                        # The dimension is not distributed. Local bounds don't need to be set.
                        pass

                # If there are any empty dimensions on the rank, than all dimensions are empty.
                is_empty = [dim.is_empty for dim in dimensions]
                if any(is_empty):
                    for dim in dimensions:
                        dim.convert_to_empty()
                else:
                    pass

        self.has_updated_dimensions = True

    def _create_or_get_group_(self, group, rank=MPI_RANK):
        # The group sequence may be modified.
        group = deepcopy(group)
        # Allow None and single string group selection.
        if group is None or isinstance(group, basestring):
            group = [group]
        # Always start with None, the root group, when searching for data groups.
        if group[0] is not None:
            group.insert(0, None)

        ret = self.dimensions[rank]
        for ctr, g in enumerate(group):
            # No group nesting for the first iteration.
            if ctr > 0:
                ret = ret['groups']
            # This is the default fill for the group.
            if g not in ret:
                ret[g] = {'groups': {}, 'dimensions': []}
            ret = ret[g]
        return ret

    def _gather_dimension_(self, name, group=None, root=0):
        dim = self.get_dimension(name, group=group)
        parts = self.comm.gather(dim, root=root)
        if self.rank == root:
            new_size = 0
            for part in parts:
                if not part.is_empty:
                    new_size += len(part)
                else:
                    pass

            # Only update the size if it is set.
            if dim.size is not None:
                dim._size = new_size
            else:
                pass
            dim._size_current = new_size

            if dim._src_idx is not None:
                new_src_idx = np.zeros(new_size, dtype=dim._src_idx.dtype)
                for part in parts:
                    if not part.is_empty:
                        lower, upper = part.bounds_local
                        new_src_idx[lower:upper] = part._src_idx
                    else:
                        pass
                dim._src_idx = new_src_idx
            else:
                pass

            # Dimension is no longer distributed and should not have local bounds.
            dim._bounds_local = None
            # Dimension is no longer empty as its component parts have been gathered across ranks.
            dim.is_empty = False

            ret = dim
        else:
            ret = None
        return ret


class MpiBoundsCalculator(AbstractOcgisObject):
    def __init__(self, nelements, size=MPI_SIZE, rank=MPI_RANK):
        self.nelements = nelements
        self.rank = rank
        self.size = size

    @property
    def bounds_global(self):
        return 0, self.nelements

    @property
    def bounds_local(self):
        if self.size == 1:
            ret = self.bounds_global
        else:
            ret = self.get_rank_bounds()
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

    def get_rank_bounds(self, length=None, rank=None):
        length = length or self.nelements
        rank = rank or self.rank
        grb = get_rank_bounds(length, size=self.size, rank=rank)
        return grb


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


def dict_get_or_create(ddict, key, default):
    try:
        ret = ddict[key]
    except KeyError:
        ret = ddict[key] = default
    return ret


def get_rank_bounds(length, size=None, rank=None, esplit=None):
    """
    :param int length: Length of the vector containing the bounds.
    :param size: Processor count. If ``None`` use MPI size.
    :param rank: The process's rank. If ``None`` use the MPI rank.
    :param esplit: The split size. If ``None``, compute this internally.
    :return: A tuple of lower and upper bounds using Python slicing rules. Returns ``None`` if no bounds are available
     for the rank. Also returns ``None`` in the case of zero length.
    :rtype: tuple or None

    >>> get_rank_bounds(5, size=4, rank=2, esplit=None)
    (3, 4)
    """

    # This is the edge case for zero-length.
    if length == 0:
        return

    # Set defaults for the rank and size.
    size = size or MPI_SIZE
    rank = rank or MPI_RANK

    # This is the edge case for ranks outside the size. Possible with an overloaded size not related to the MPI
    # environment.
    if rank >= size:
        return

    # Case with more length than size. Do not take this route of a default split is provided.
    if length > size and esplit is None:
        length = int(length)
        size = int(size)
        esplit, remainder = divmod(length, size)

        if remainder > 0:
            # Find the rank bounds with no remainder.
            ret = get_rank_bounds(length - remainder, size=size, rank=rank)
            # Adjust the returned slices accounting for the remainder.
            if rank + 1 <= remainder:
                ret = (ret[0] + rank, ret[1] + rank + 1)
            else:
                ret = (ret[0] + remainder, ret[1] + remainder)
        elif remainder == 0:
            # Provide the default split to compute the bounds and avoid the recursion.
            ret = get_rank_bounds(length, size=size, rank=rank, esplit=esplit)
        else:
            raise NotImplementedError
    # Case with equal length and size or more size than length.
    else:
        if esplit is None:
            if length < size:
                esplit = int(np.ceil(float(length) / float(size)))
            elif length == size:
                esplit = 1
            else:
                raise NotImplementedError
        else:
            esplit = int(esplit)

        if rank == 0:
            lbound = 0
        else:
            lbound = rank * esplit
        ubound = lbound + esplit

        if ubound >= length:
            ubound = length

        if lbound >= ubound:
            # The lower bound is outside the vector length
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


def create_nd_slices(splits, shape):
    ret = [None] * len(shape)
    for idx, (split, shp) in enumerate(zip(splits, shape)):
        ret[idx] = create_slices(shp, split)
    ret = [slices for slices in itertools.product(*ret)]
    return tuple(ret)


def find_dimension_in_sequence(dimension_name, dimensions):
    ret = None
    for dim in dimensions:
        if dimension_name == dim.name:
            ret = dim
            break
    if ret is None:
        raise DimensionNotFound('Dimension not found: {}'.format(dimension_name))
    return ret


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


def get_template_rank_dict():
    return {None: {'dimensions': [], 'groups': {}}}


def variable_scatter(variable, dest_mpi, root=0, comm=None):
    comm = comm or MPI_COMM
    rank = comm.Get_rank()

    # tdk: test a variable with no dimensions
    # Only updated dimensions are allowed for scatters.
    if rank == root:
        assert dest_mpi.has_updated_dimensions

    if rank == root:
        group = variable.group
        dimension_names = [dim.name for dim in variable.dimensions]
    else:
        group = None
        dimension_names = None
    dest_mpi = comm.bcast(dest_mpi, root=root)
    group = comm.bcast(group, root=root)
    dimension_names = comm.bcast(dimension_names, root=root)
    dest_dimensions = dest_mpi.get_dimensions(dimension_names, group=group)

    # Distribute the values and masks if the variable has distributed dimensions.
    if rank == root:
        size = dest_mpi.size
        slices = [None] * size
        for current_rank in range(size):
            current_dimensions = dest_mpi.get_dimensions(dimension_names, group=group, rank=current_rank)
            slices[current_rank] = [slice(d.bounds_local[0], d.bounds_local[1]) for d in current_dimensions]
        variables_to_scatter = [None] * size
        for idx, slc in enumerate(slices):
            variables_to_scatter[idx] = variable[slc]
    else:
        variables_to_scatter = None

    scattered_variable = comm.scatter(variables_to_scatter, root=root)
    try:
        scattered_variable.dimensions = dest_dimensions
    except:
        from ocgis_logging import log
        log.debug(['variable shape', scattered_variable.shape])
        log.debug(['variable value', scattered_variable.value])
        log.debug(['variable mask', scattered_variable.get_mask()])
        log.debug(['dimension shape', [len(d) for d in scattered_variable.dimensions]])
        log.debug(['dimension is_empty', [d.is_empty for d in scattered_variable.dimensions]])
        raise

    return scattered_variable, dest_mpi


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
