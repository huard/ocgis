import numpy as np

from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.util.helpers import get_formatted_slice


class Dimension(AbstractInterfaceObject):
    # tdk:doc

    def __init__(self, name, length=None, length_current=None):
        # tdk:test
        self.name = name
        self.length = length
        self.length_current = length_current

    def __eq__(self, other):
        if other.__dict__ == self.__dict__:
            ret = True
        else:
            ret = False
        return ret

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, 1)
        try:
            length = len(slc)
        except TypeError:
            # Likely a slice object.
            try:
                length = slc.stop - slc.start
            except TypeError:
                # Likely a NoneType slice.
                if slc.start is None:
                    length = len(self)
                else:
                    raise
        if self.length is None:
            ret = self.__class__(self.name, length_current=length)
        else:
            ret = self.__class__(self.name, length=length)
        return ret

    def __len__(self):
        if self.length is None:
            ret = self.length_current
        else:
            ret = self.length
        if ret is None:
            ret = 0
        return ret

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        msg = '{0}(name={1}, length={2})'.format(self.__class__.__name__, self.name, self.length)
        return msg


class SourcedDimension(Dimension):
    _default_dtype = np.int32

    def __init__(self, *args, **kwargs):
        self.__src_idx__ = None

        self._src_idx = kwargs.pop('src_idx', None)

        super(SourcedDimension, self).__init__(*args, **kwargs)

    def __eq__(self, other):
        try:
            ret = super(SourcedDimension, self).__eq__(other)
        except ValueError:
            # Likely the source index is loaded which requires using a numpy comparison.
            ret = True
            for k, v in self.__dict__.iteritems():
                if k == '__src_idx__':
                    if not np.all(v == other.__dict__[k]):
                        ret = False
                        break
                else:
                    if v != other.__dict__[k]:
                        ret = False
                        break
        return ret

    def __getitem__(self, slc):
        ret = super(SourcedDimension, self).__getitem__(slc)
        ret._src_idx = self._src_idx.__getitem__(slc)
        return ret

    @property
    def _src_idx(self):
        if self.__src_idx__ is None:
            self.__src_idx__ = np.arange(0, len(self), dtype=self._default_dtype)
        return self.__src_idx__

    @_src_idx.setter
    def _src_idx(self, value):
        if value is not None:
            value = np.array(value)
        self.__src_idx__ = value
