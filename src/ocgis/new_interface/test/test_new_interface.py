from abc import ABCMeta
from copy import deepcopy

import numpy as np
from shapely.geometry import Point

from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.geom import PointArray
from ocgis.new_interface.grid import GridXY
from ocgis.new_interface.variable import BoundedVariable, Variable, VariableCollection
from ocgis.test.base import TestBase

_VALUE_POINT_ARRAY = np.array([None, None])
_VALUE_POINT_ARRAY[:] = [Point(1, 2), Point(3, 4)]


class AbstractTestNewInterface(TestBase):
    __metaclass__ = ABCMeta

    def assertGeometriesAlmostEquals(self, a, b):

        def _almost_equals_(a, b):
            return a.almost_equals(b)

        vfunc = np.vectorize(_almost_equals_, otypes=[bool])
        to_test = vfunc(a.data, b.data)
        self.assertTrue(to_test.all())
        self.assertNumpyAll(a.mask, b.mask)

    def get_gridxy(self, with_2d_variables=False, with_dimensions=False, crs=None, with_xy_bounds=False,
                   with_value_mask=False, with_backref=False):

        x = [101, 102, 103]
        y = [40, 41, 42, 43]

        x_dim = Dimension('x', length=len(x))
        y_dim = Dimension('y', length=len(y))

        kwds = {'crs': crs}

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

        if with_value_mask:
            x_value = np.ma.array(x_value, mask=[False, True, False])
            y_value = np.ma.array(y_value, mask=[True, False, True, False])

        vx = BoundedVariable('x', value=x_value, dtype=float, dimensions=x_dims)
        vy = BoundedVariable('y', value=y_value, dtype=float, dimensions=y_dims)
        if with_xy_bounds:
            vx.set_extrapolated_bounds()
            vy.set_extrapolated_bounds()

        if with_backref:
            np.random.seed(1)
            tas = np.random.rand(10, 3, 4)
            tas = Variable(name='tas', value=tas)
            tas.create_dimensions(names=['time', 'x', 'y'])

            rhs = np.random.rand(4, 3, 10) * 100
            rhs = Variable(name='rhs', value=rhs)
            rhs.create_dimensions(names=['y', 'x', 'time'])

            kwds['backref'] = VariableCollection(variables=[tas, rhs])

        grid = GridXY(vx, vy, **kwds)
        return grid

    def get_geometryvariable(self, **kwargs):
        if kwargs.get('value') is None:
            kwargs['value'] = kwargs.pop('value', deepcopy(_VALUE_POINT_ARRAY))
        pa = PointArray(**kwargs)
        return pa

    def get_request_dataset(self):
        data = self.test_data.get_rd('cancm4_tas')
        return data

    def get_variable_x(self, bounds=True):
        value = [-100., -99., -98., -97.]
        if bounds:
            bounds = [[v - 0.5, v + 0.5] for v in value]
            bounds = Variable(value=bounds, name='x_bounds')
        else:
            bounds = None
        x = BoundedVariable(value=value, bounds=bounds, name='x')
        return x

    def get_variable_y(self, bounds=True):
        value = [40., 39., 38.]
        if bounds:
            bounds = [[v + 0.5, v - 0.5] for v in value]
            bounds = Variable(value=bounds, name='y_bounds')
        else:
            bounds = None
        y = BoundedVariable(value=value, bounds=bounds, name='y')
        return y
