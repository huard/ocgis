import os
from abc import ABCMeta
from copy import deepcopy

import numpy as np
from shapely import wkt
from shapely.geometry import Point

from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.grid import GridXY, get_geometry_variable, get_polygon_geometry_array
from ocgis.new_interface.variable import Variable, VariableCollection
from ocgis.test.base import TestBase

_VALUE_POINT_ARRAY = np.array([None, None])
_VALUE_POINT_ARRAY[:] = [Point(1, 2), Point(3, 4)]


class AbstractTestNewInterface(TestBase):
    __metaclass__ = ABCMeta

    def assertGeometriesAlmostEquals(self, a, b):

        def _almost_equals_(a, b):
            return a.almost_equals(b)

        vfunc = np.vectorize(_almost_equals_, otypes=[bool])
        to_test = vfunc(a.value, b.value)
        self.assertTrue(to_test.all())
        self.assertNumpyAll(a.get_mask(), b.get_mask())

    def get_gridxy(self, with_2d_variables=False, crs=None, with_xy_bounds=False, with_value_mask=False,
                   with_parent=False):

        x = [101, 102, 103]
        y = [40, 41, 42, 43]

        x_dim = Dimension('xdim', length=len(x))
        y_dim = Dimension('ydim', length=len(y))

        kwds = {'crs': crs}

        if with_2d_variables:
            x_value, y_value = np.meshgrid(x, y)
            x_dims = (y_dim, x_dim)
            y_dims = x_dims
        else:
            x_value, y_value = x, y
            x_dims = (x_dim,)
            y_dims = (y_dim,)

        if with_value_mask:
            x_value = np.ma.array(x_value, mask=[False, True, False])
            y_value = np.ma.array(y_value, mask=[True, False, True, False])

        vx = Variable('x', value=x_value, dtype=float, dimensions=x_dims)
        vy = Variable('y', value=y_value, dtype=float, dimensions=y_dims)
        if with_xy_bounds:
            vx.set_extrapolated_bounds()
            vy.set_extrapolated_bounds()

        if with_parent:
            np.random.seed(1)
            tas = np.random.rand(10, 3, 4)
            tas = Variable(name='tas', value=tas)
            tas.create_dimensions(names=['time', 'x', 'y'])

            rhs = np.random.rand(4, 3, 10) * 100
            rhs = Variable(name='rhs', value=rhs)
            rhs.create_dimensions(names=['y', 'x', 'time'])

            kwds['parent'] = VariableCollection(variables=[tas, rhs])

        grid = GridXY(vx, vy, **kwds)
        return grid

    def get_geometryvariable(self, **kwargs):
        if kwargs.get('value') is None:
            kwargs['value'] = kwargs.pop('value', deepcopy(_VALUE_POINT_ARRAY))
        pa = GeometryVariable(**kwargs)
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

    @property
    def polygon_value(self):
        polys = [['POLYGON ((-100.5 39.5, -100.5 40.5, -99.5 40.5, -99.5 39.5, -100.5 39.5))',
                  'POLYGON ((-99.5 39.5, -99.5 40.5, -98.5 40.5, -98.5 39.5, -99.5 39.5))',
                  'POLYGON ((-98.5 39.5, -98.5 40.5, -97.5 40.5, -97.5 39.5, -98.5 39.5))',
                  'POLYGON ((-97.5 39.5, -97.5 40.5, -96.5 40.5, -96.5 39.5, -97.5 39.5))'],
                 ['POLYGON ((-100.5 38.5, -100.5 39.5, -99.5 39.5, -99.5 38.5, -100.5 38.5))',
                  'POLYGON ((-99.5 38.5, -99.5 39.5, -98.5 39.5, -98.5 38.5, -99.5 38.5))',
                  'POLYGON ((-98.5 38.5, -98.5 39.5, -97.5 39.5, -97.5 38.5, -98.5 38.5))',
                  'POLYGON ((-97.5 38.5, -97.5 39.5, -96.5 39.5, -96.5 38.5, -97.5 38.5))'],
                 ['POLYGON ((-100.5 37.5, -100.5 38.5, -99.5 38.5, -99.5 37.5, -100.5 37.5))',
                  'POLYGON ((-99.5 37.5, -99.5 38.5, -98.5 38.5, -98.5 37.5, -99.5 37.5))',
                  'POLYGON ((-98.5 37.5, -98.5 38.5, -97.5 38.5, -97.5 37.5, -98.5 37.5))',
                  'POLYGON ((-97.5 37.5, -97.5 38.5, -96.5 38.5, -96.5 37.5, -97.5 37.5))']]
        return self.get_shapely_from_wkt_array(polys)

    @property
    def polygon_value_alternate_ordering(self):
        polys = [['POLYGON ((-100.5 40.5, -99.5 40.5, -99.5 39.5, -100.5 39.5, -100.5 40.5))',
                  'POLYGON ((-99.5 40.5, -98.5 40.5, -98.5 39.5, -99.5 39.5, -99.5 40.5))',
                  'POLYGON ((-98.5 40.5, -97.5 40.5, -97.5 39.5, -98.5 39.5, -98.5 40.5))',
                  'POLYGON ((-97.5 40.5, -96.5 40.5, -96.5 39.5, -97.5 39.5, -97.5 40.5))'],
                 ['POLYGON ((-100.5 39.5, -99.5 39.5, -99.5 38.5, -100.5 38.5, -100.5 39.5))',
                  'POLYGON ((-99.5 39.5, -98.5 39.5, -98.5 38.5, -99.5 38.5, -99.5 39.5))',
                  'POLYGON ((-98.5 39.5, -97.5 39.5, -97.5 38.5, -98.5 38.5, -98.5 39.5))',
                  'POLYGON ((-97.5 39.5, -96.5 39.5, -96.5 38.5, -97.5 38.5, -97.5 39.5))'],
                 ['POLYGON ((-100.5 38.5, -99.5 38.5, -99.5 37.5, -100.5 37.5, -100.5 38.5))',
                  'POLYGON ((-99.5 38.5, -98.5 38.5, -98.5 37.5, -99.5 37.5, -99.5 38.5))',
                  'POLYGON ((-98.5 38.5, -97.5 38.5, -97.5 37.5, -98.5 37.5, -98.5 38.5))',
                  'POLYGON ((-97.5 38.5, -96.5 38.5, -96.5 37.5, -97.5 37.5, -97.5 38.5))']]
        return self.get_shapely_from_wkt_array(polys)

    def get_polygonarray(self):
        grid = self.get_polygon_array_grid()
        poly = get_geometry_variable(get_polygon_geometry_array, grid)
        return poly

    def get_polygon_array_grid(self, with_bounds=True):
        if with_bounds:
            xb = Variable(value=[[-100.5, -99.5], [-99.5, -98.5], [-98.5, -97.5], [-97.5, -96.5]], name='xb')
            yb = Variable(value=[[40.5, 39.5], [39.5, 38.5], [38.5, 37.5]], name='yb')
        else:
            xb, yb = [None, None]
        x = Variable(value=[-100.0, -99.0, -98.0, -97.0], bounds=xb, name='x')
        y = Variable(value=[40.0, 39.0, 38.0], name='y', bounds=yb)
        grid = GridXY(x=x, y=y)
        return grid

    def get_shapely_from_wkt_array(self, wkts):
        ret = np.array(wkts)
        vfunc = np.vectorize(wkt.loads, otypes=[object])
        ret = vfunc(ret)
        ret = np.ma.array(ret, mask=False)
        return ret

    @staticmethod
    def write_fiona_htmp(obj, name):
        path = os.path.join('/home/benkoziol/htmp/ocgis', '{}.shp'.format(name))
        obj.write_fiona(path)
