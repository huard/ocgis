from copy import copy

import numpy as np

from ocgis import constants
from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import get_formatted_slice


class GridXY(Variable):
    def __init__(self, **kwargs):

        self.x = kwargs.pop('x', None)
        self.y = kwargs.pop('y', None)

        super(GridXY, self).__init__(kwargs.pop('name', None), **kwargs)

        if self._value is None:
            if self.x is None or self.y is None:
                msg = 'At least "x" and "y" are required to make a grid.'
                raise ValueError(msg)
            if self.x.ndim > 2 or self.y.ndim > 2:
                msg = '"x" and "y" may not have ndim > 2.'
                raise ValueError(msg)

    def __getitem__(self, slc):
        """
        :param slc: The slice sequence with indices corresponding to:

         0 --> x-dimension
         1 --> y-dimension
         2 --> z-dimension (if present)

        :type slc: sequence of slice-compatible arguments
        :returns: Sliced grid components.
        :rtype: :class:`ocgis.new_interface.grid.GridXY`
        """

        slc = get_formatted_slice(slc, self.ndim)
        ret = copy(self)
        if self._value is None:
            if self.is_vectorized:
                ret.y = self.y[slc[0]]
                ret.x = self.x[slc[1]]
            else:
                ret.y = self.y[:, slc[0], slc[1]]
                ret.x = self.x[:, slc[0], slc[1]]
        else:
            ret._value = self._value[:, slc[0], slc[1]]
        return ret

    @property
    def is_vectorized(self):
        if len(self.x.shape) > 1:
            ret = False
        else:
            ret = True
        return ret

    @property
    def mask(self):
        if self.is_vectorized:
            self.expand()
        return self.x.value.mask

    @mask.setter
    def mask(self, value):
        if self.is_vectorized:
            self.expand()
        self.x.mask = value
        self.y.mask = value

    @property
    def ndim(self):
        return 2

    @property
    def resolution(self):
        if self.is_vectorized:
            to_mean = [self.x.resolution, self.y.resolution]
        else:
            resolution_limit = int(constants.RESOLUTION_LIMIT) / 2
            r_value = self.value[:, 0:resolution_limit, 0:resolution_limit]
            rows = np.mean(np.diff(r_value[0, :, :], axis=0))
            cols = np.mean(np.diff(r_value[1, :, :], axis=1))
            to_mean = [rows, cols]
        ret = np.mean(to_mean)
        return ret

    @property
    def shape(self):
        if self._value is None:
            if self.is_vectorized:
                ret = [len(self.y), len(self.x)]
            else:
                ret = list(self.x.shape)
        else:
            ret = self.value.shape[1:]
        ret = tuple(ret)
        return ret

    def expand(self):
        # tdk: remove
        assert self.x.ndim == 1
        assert self.y.ndim == 1

        new_x, new_y = np.meshgrid(self.x.value, self.y.value)

        if self.x.dimensions is not None:
            new_dims = (self.y.dimensions[0], self.x.dimensions[0])
        else:
            new_dims = None

        self.x = Variable(self.x.name, value=new_x, dimensions=new_dims)
        self.y = Variable(self.y.name, value=new_y, dimensions=new_dims)

        self.x.value.mask = self.y.value.mask

    def write_netcdf(self, dataset, **kwargs):
        if self._value is None:
            to_write = [self.x, self.y]
        else:
            if self.y is not None:
                yname = self.y.name
                yattrs = self.y.attrs
                xname = self.x.name
                xattrs = self.x.attrs
            else:
                yname = 'yc'
                yattrs = {'axis': 'Y'}
                xname = 'xc'
                xattrs = {'axis': 'X'}
            yvar = Variable(yname, value=self.value[0, ...], dimensions=self.dimensions, attrs=yattrs)
            xvar = Variable(xname, value=self.value[1, ...], dimensions=self.dimensions, attrs=xattrs)
            to_write = [yvar, xvar]
        for tw in to_write:
            tw.write_netcdf(dataset, **kwargs)

    def _get_dimensions_(self):
        if self._dimensions is None:
            value = None
            if self.y is not None and self.y.dimensions is not None:
                if self._value is None:
                    if self.is_vectorized:
                        value = (self.y.dimensions[0], self.x.dimensions[0])
                    else:
                        value = self.y.dimensions
            self._dimensions = value
        return self._dimensions

    def _get_value_(self):
        if self._value is None:
            if self.is_vectorized:
                new_x, new_y = np.meshgrid(self.x.value, self.y.value)
                shp = (2, len(self.y), len(self.x))
            else:
                new_x, new_y = self.x.value, self.y.value
                shp = [2] + list(new_x.shape)

            fill = np.zeros(shp)
            fill[0, ...] = new_y
            fill[1, ...] = new_x

            self._set_value_(fill)

        return self._value

    def _validate_value_(self, value):
        if self.dimensions is not None:
            assert value.shape[1:] == self.shape
            assert value.shape[0] == 2
