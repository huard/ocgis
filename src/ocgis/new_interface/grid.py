from copy import copy

from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import get_formatted_slice


class Grid(Variable):
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            args = (None,)

        self.x = kwargs.pop('x', None)
        self.y = kwargs.pop('y', None)
        self.z = kwargs.pop('z', None)

        super(Grid, self).__init__(*args, **kwargs)

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, self.ndim)
        ret = copy(self)
        if self._value is not None:
            # tdk: test
            raise NotImplementedError
        if self.x is not None:
            ret.x = self.x[slc[1]]
        if self.y is not None:
            ret.y = self.y[slc[0]]
        if self.z is not None:
            ret.z = self.z[slc[2]]
        return ret

    @property
    def shape(self):
        if self.x is not None:
            ret = [len(self.y), len(self.x)]
            if self.z is not None:
                ret.append(len(self.z))
        else:
            # tdk:test
            return self.value.shape[1:]
        ret = tuple(ret)
        return ret
