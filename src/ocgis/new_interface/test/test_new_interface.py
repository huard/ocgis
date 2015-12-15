from abc import ABCMeta
from copy import deepcopy

import numpy as np
from shapely.geometry import Point

from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.geom import PointArray
from ocgis.new_interface.grid import GridXY
from ocgis.new_interface.variable import Variable, BoundedVariable
from ocgis.test.base import TestBase

_VALUE_POINT_ARRAY = np.array([None, None])
_VALUE_POINT_ARRAY[:] = [Point(1, 2), Point(3, 4)]


class AbstractTestNewInterface(TestBase):
    __metaclass__ = ABCMeta

    def get_gridxy(self, with_2d_variables=False, with_dimensions=False, with_value=False,
                   with_value_only=False):

        x = [101, 102, 103]
        y = [40, 41, 42, 43]

        x_dim = Dimension('x', length=len(x))
        y_dim = Dimension('y', length=len(y))

        kwds = {}

        if with_2d_variables:
            x_value, y_value = np.meshgrid(x, y)
            x_dims = (y_dim, x_dim)
            y_dims = x_dims
        else:
            x_value, y_value = x, y
            x_dims = (x_dim,)
            y_dims = (y_dim,)

        if not with_dimensions:
            x_dims = None
            y_dims = None

        if not with_value_only:
            vx = Variable('x', value=x_value, dtype=float, dimensions=x_dims)
            vy = Variable('y', value=y_value, dtype=float, dimensions=y_dims)
            kwds.update(dict(x=vx, y=vy))

        if with_value or with_value_only:
            new_x, new_y = np.meshgrid(x, y)
            fill = np.zeros((2, len(y), len(x)))
            fill[0, ...] = new_y
            fill[1, ...] = new_x
            kwds.update(dict(value=fill))

        grid = GridXY(**kwds)
        return grid

    def get_pointarray(self, **kwargs):
        kwargs['value'] = kwargs.pop('value', deepcopy(_VALUE_POINT_ARRAY))
        pa = PointArray(**kwargs)
        return pa

    def get_variable_x(self, bounds=True):
        value = [-100., -99., -98., -97.]
        if bounds:
            bounds = [[v - 0.5, v + 0.5] for v in value]
        else:
            bounds = None
        x = BoundedVariable(value=value, bounds=bounds, name='x')
        return x

    def get_variable_y(self, bounds=True):
        value = [40., 39., 38.]
        if bounds:
            bounds = [[v + 0.5, v - 0.5] for v in value]
        else:
            bounds = None
        y = BoundedVariable(value=value, bounds=bounds, name='y')
        return y
