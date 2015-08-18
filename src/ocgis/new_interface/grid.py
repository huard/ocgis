from copy import copy

import numpy as np

from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.util.helpers import get_formatted_slice


class Grid(AbstractInterfaceObject):
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

        if self.x is None or self.y is None:
            msg = 'At least "x" and "y" are required to make a grid.'
            raise ValueError(msg)

    def __getitem__(self, slc):
        """
        :param slc: The slice sequence with indices corresponding to:

         0 --> x-dimension
         1 --> y-dimension
         2 --> z-dimension (if present)

        :type slc: sequence of slice-compatible arguments
        :returns: Sliced grid components.
        :rtype: :class:`ocgis.new_interface.grid.Grid`
        """

        slc = get_formatted_slice(slc, self.ndim)
        ret = copy(self)
        ret.x = self.x[slc[1]]
        ret.y = self.y[slc[0]]
        if self.z is not None:
            ret.z = self.z[slc[2]]
        return ret

    @property
    def ndim(self):
        if self.z is None:
            ret = 2
        else:
            ret = 3
        return ret

    @property
    def resolution(self):
        ret = np.mean([self.x.resolution, self.y.resolution])
        return ret

    @property
    def shape(self):
        ret = [len(self.x), len(self.y)]
        if self.z is not None:
            ret.append(len(self.z))
        ret = tuple(ret)
        return ret
