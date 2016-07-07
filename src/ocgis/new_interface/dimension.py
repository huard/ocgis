import numpy as np

from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.mpi import OcgMpi, get_global_to_local_slice, MPI_COMM
from ocgis.util.helpers import get_formatted_slice


class Dimension(AbstractInterfaceObject):
    _default_dtype = np.int32

    def __init__(self, name, size=None, size_current=None, src_idx=None, dist=False):
        self._mpi = None
        self._name = name
        self.__src_idx__ = None

        self._size = size
        self._size_current = size_current
        self.dist = dist
        self._src_idx = src_idx

        # If this dimension is distributed, some form of length is required.
        if self.dist:
            if self.size is None and self.size_current is None:
                msg = 'Distributed dimensions required "length" or "length_current".'
                raise ValueError(msg)

        super(Dimension, self).__init__()

    def __eq__(self, other):
        ret = True
        skip = ('__src_idx__', '_mpi')
        for k, v in self.__dict__.items():
            if k in skip:
                continue
            else:
                if v != other.__dict__[k]:
                    ret = False
                    break
        if ret:
            if self._src_idx is None and other._src_idx is not None:
                ret = False
            else:
                if not np.all(self._src_idx == other._src_idx):
                    ret = False
        return ret

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, 1)[0]
        # If this is a distributed dimension, remap the slice accounting for local bounds.
        if self.dist:
            if self.mpi.bounds_local is None:
                slc = None
            else:
                self.log.debug('self.mpi.bounds_local {}'.format(self.mpi.bounds_local))
                remapped = get_global_to_local_slice((slc.start, slc.stop), self.mpi.bounds_local)
                self.log.debug('remapped {}'.format(remapped))
                # If the remapped slice is None, the dimension on this rank is empty.
                if remapped is None:
                    slc = None
                else:
                    slc = slice(*remapped)

        # If there is no local slice (local and global bounds do not overlap), return None for the object.
        ret = self.copy()
        if slc is None:
            ret = None
        else:
            self.__getitem_main__(ret, slc)

        return ret

    def __len__(self):
        if self.size is None:
            ret = self.size_current
        else:
            ret = self.size
        if ret is None:
            ret = 0
        return ret

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        msg = "{0}(name='{1}', size={2}, size_current={3})".format(self.__class__.__name__, self.name, self.size,
                                                                   self.size_current)
        return msg

    @property
    def is_unlimited(self):
        if self.size is None:
            ret = True
        else:
            ret = False
        return ret

    @property
    def mpi(self):
        if self._mpi is None:
            if self.dist:
                size = None
            else:
                size = 1
            self._mpi = OcgMpi(nelements=len(self), size=size)
        return self._mpi

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    @property
    def size_current(self):
        if self._size_current is None:
            if self.size is None:
                if self.__src_idx__ is None:
                    ret = 0
                else:
                    ret = len(self.__src_idx__)
            else:
                ret = self.size
        else:
            ret = self._size_current
        return ret

    @property
    def _src_idx(self):
        if self.__src_idx__ is None and len(self) > 0:
            bounds = self.mpi.bounds_local
            if bounds is not None:
                lower, upper = bounds
                self.__src_idx__ = np.arange(lower, upper, dtype=self._default_dtype)
        return self.__src_idx__

    @_src_idx.setter
    def _src_idx(self, value):
        if value is not None:
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            if self.dist:
                bounds = self.mpi.bounds_local
                if bounds is None:
                    value = None
                else:
                    lower, upper = bounds
                    value = value[lower:upper]
        self.__src_idx__ = value

    def gather(self, root=0, comm=None):
        comm = comm or MPI_COMM
        ret = self
        if self.mpi.size != 1:
            bounds = comm.gather(self.mpi.bounds_local, root=root)
            src_indexes = comm.gather(self._src_idx, root=root)
            if self.mpi.rank == root:
                lower, upper = self.mpi.bounds_global
                src_idx = np.zeros((upper - lower,), dtype=self._default_dtype)
                for idx in range(self.mpi.size):
                    try:
                        lower, upper = bounds[idx]
                    except TypeError:
                        if bounds[idx] is not None:
                            raise
                    else:
                        src_idx[lower:upper] = src_indexes[idx]
                # This dimension is no longer distributed. Setting this to False will allow the local bounds to load
                # correctly.
                ret.dist = False
                ret._src_idx = src_idx
            else:
                ret = None
        else:
            ret.dist = False
        return ret

    def scatter(self):
        self.dist = True
        self._mpi = None
        # Reset the source index forcing a subset by local bounds if applicable for the rank.
        self._src_idx = self._src_idx
        return self

    def __getitem_main__(self, ret, slc):
        # Source index can be None if the dimension has zero length.
        if ret._src_idx is not None:
            ret._src_idx = ret._src_idx.__getitem__(slc)

        try:
            length = len(slc)
        except TypeError:
            # Likely a slice object.
            try:
                length = slc.stop - slc.start
            except TypeError:
                # Likely a NoneType slice.
                if slc.start is None:
                    if slc.stop > 0:
                        length = len(self)
                    elif slc.stop is None:
                        length = len(self)
                    else:
                        length = len(self) + slc.stop
                elif slc.stop is None:
                    if slc.start > 0:
                        length = len(self) - slc.start
                    else:
                        length = abs(slc.start)
                else:
                    raise
        else:
            try:
                # Check for boolean slices.
                if slc.dtype == bool:
                    length = slc.sum()
            except AttributeError:
                # Likely a list/tuple.
                pass
        if self.size is None:
            if length < 0:
                # This is using negative indexing. Subtract from the current length.
                length = length + self.size_current
            ret._size_current = length
        else:
            if length < 0:
                # This is using negative indexing. Subtract from the current length.
                length = length + self.size
            ret._size = length
