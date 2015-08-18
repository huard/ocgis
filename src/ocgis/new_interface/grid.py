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
        if self.x.ndim > 2 or self.y.ndim > 2:
            msg = '"x" and "y" may not have ndim > 2.'
            raise ValueError(msg)
        if self.z is not None and self.z.ndim > 3:
            msg = '"z" may not have ndim > 3.'
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
        if self.is_vectorized:
            ret.x = self.x[slc]
            ret.y = self.y[slc]
        else:
            ret.x = self.x[slc[0]]
            ret.y = self.y[slc[1]]
        if self.z is not None:
            ret.z = self.z[slc[2]]
        return ret

    @property
    def is_vectorized(self):
        if len(self.x.shape) > 1:
            ret = True
        else:
            ret = False
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
        if not self.is_vectorized:
            ret = [len(self.x), len(self.y)]
            if self.z is not None:
                ret.append(len(self.z))
            ret = tuple(ret)
        else:
            ret = self.x.shape
        return ret

    def write_netcdf(self, dataset, **kwargs):
        to_write = [self.x, self.y]
        if self.z is not None:
            to_write.append(self.z)
        for tw in to_write:
            tw.write_netcdf(dataset, **kwargs)
