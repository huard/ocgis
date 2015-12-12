from abc import ABCMeta

import numpy as np
from shapely.geometry import Point

from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.geom import PointArray
from ocgis.new_interface.grid import GridXY
from ocgis.new_interface.variable import Variable
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
        kwargs['value'] = kwargs.pop('value', _VALUE_POINT_ARRAY)
        pa = PointArray(**kwargs)
        return pa
