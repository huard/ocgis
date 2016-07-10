import numpy as np

from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.mpi import MPI_COMM
from ocgis.util.helpers import get_formatted_slice


class Dimension(AbstractInterfaceObject):
    _default_dtype = np.int32

    def __init__(self, name, size=None, size_current=None, src_idx=None, dist=False):
        super(Dimension, self).__init__()

        self._name = name
        self.__src_idx__ = None
        self.bounds_global = None
        self.bounds_local = None

        self.dist = dist
        self._size = size
        self._size_current = size_current
        self._src_idx = src_idx

        # Adjust object for the distributed case. ----------------------------------------------------------------------

        if dist:
            # A size definition is required.
            if self.size is None and self.size_current is None:
                msg = 'Distributed dimensions require a size definition using "size" or "size_current".'
                raise ValueError(msg)
                # mpi_size = None
                # else:
                #     mpi_size = 1

                # # Always configure the MPI interface.
                # self.mpi = OcgMpi(nelements=len(self), size=mpi_size)
                #
                # if dist:
                #     # Adjust the source index and sizes for the distributed case.
                #     bounds_local = self.mpi.bounds_local
                #     if bounds_local is None:
                #         if self.is_unlimited:
                #             self._size_current = 0
                #         else:
                #             self._size = 0
                #             self._size_current = None
                #         self.__src_idx__ = None
                #     else:
                #         lower, upper = bounds_local
                #         if self.is_unlimited:
                #             self._size_current = upper - lower
                #         else:
                #             self._size = upper - lower
                #             self._size_current = None
                #         if src_idx is not None:
                #             self._src_idx = self._src_idx[lower:upper]

    def __eq__(self, other):
        ret = True
        skip = ('__src_idx__', 'mpi')
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
        # We cannot slice zero length dimensions.
        if len(self) == 0:
            raise IndexError('Zero-length dimensions are not slicable.')

        slc = get_formatted_slice(slc, 1)[0]
        ret = self.copy()
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
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    @property
    def size_current(self):
        if self._size_current is None:
            ret = self.size
        else:
            ret = self._size_current
        return ret

    @property
    def _src_idx(self):
        return self.__src_idx__

    @_src_idx.setter
    def _src_idx(self, value):
        if value is not None:
            if not isinstance(value, np.ndarray):
                value = np.array(value)
        self.__src_idx__ = value

    def gather(self, root=0, comm=None):
        raise NotImplementedError
        comm = comm or MPI_COMM
        ret = self
        if self.mpi.size != 1:
            bounds = comm.gather(self.mpi.bounds_local, root=root)
            src_indexes = comm.gather(self._src_idx, root=root)
            if self.mpi.rank == root:
                lower, upper = self.mpi.bounds_global
                src_idx = np.zeros((upper - lower,), dtype=self._default_dtype)
                for idx in range(self.mpi.size):
                    cbounds = bounds[idx]
                    if cbounds is not None:
                        lower, upper = cbounds
                        src_idx[lower:upper] = src_indexes[idx]
                ret._src_idx = src_idx
            else:
                ret = None
        return ret

    def scatter(self):
        raise NotImplementedError
        return self.__class__(self.name, size=self.size, size_current=self.size_current, src_idx=self._src_idx,
                              dist=True)

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
        if self.is_unlimited:
            if length < 0:
                # This is using negative indexing. Subtract from the current length.
                length = length + self.size_current
            ret._size_current = length
        else:
            if length < 0:
                # This is using negative indexing. Subtract from the current length.
                length = length + self.size
            ret._size = length
