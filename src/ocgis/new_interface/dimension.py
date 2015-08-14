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
            length = slc.stop - slc.start
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

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, 1)
        try:
            length = len(slc)
        except TypeError:
            # Likely a slice object.
            length = slc.stop - slc.start
        src_idx = self._src_idx.__getitem__(slc)
        if self.length is None:
            ret = self.__class__(self.name, length_current=length, src_idx=src_idx)
        else:
            ret = self.__class__(self.name, length=length, src_idx=src_idx)
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