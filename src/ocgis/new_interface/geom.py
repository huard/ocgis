import inspect
from abc import ABCMeta
from copy import copy

import numpy as np
from numpy.core.multiarray import ndarray
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.prepared import prep

from ocgis import env
from ocgis.exc import EmptySubsetError
from ocgis.new_interface.adapter import SpatialAdapter
from ocgis.new_interface.base import get_keyword_arguments_from_template_keys
from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import iter_array, get_none_or_slice

_KWARGS_SPATIAL_ADAPTER = inspect.getargspec(SpatialAdapter.__init__).args


class AbstractSpatialVariable(Variable, SpatialAdapter):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        kwargs_sa = get_keyword_arguments_from_template_keys(kwargs, _KWARGS_SPATIAL_ADAPTER, pop=True)
        SpatialAdapter.__init__(self, **kwargs_sa)
        Variable.__init__(self, **kwargs)


class PointArray(AbstractSpatialVariable):
    def __init__(self, **kwargs):
        self._grid = kwargs.pop('grid', None)

        super(PointArray, self).__init__(**kwargs)

    def __getitem__(self, slc):
        ret = super(PointArray, self).__getitem__(slc)
        ret._grid = get_none_or_slice(ret._grid, slc)
        return ret

    @property
    def dtype(self):
        # Geometry arrays are always object arrays.
        return object

    @dtype.setter
    def dtype(self, value):
        assert value == object or value is None

    @property
    def grid(self):
        return self._grid

    @property
    def weights(self):
        ret = np.ones(self.value.shape, dtype=env.NP_FLOAT)
        ret = np.ma.array(ret, mask=self.value.mask)
        return ret

    def get_intersects_masked(self, polygon, use_spatial_index=True, keep_touches=False):
        """
        :param polygon: The Shapely geometry to use for subsetting.
        :type polygon: :class:`shapely.geometry.Polygon' or :class:`shapely.geometry.MultiPolygon'
        :param bool use_spatial_index: If ``False``, do not use the :class:`rtree.index.Index` for spatial subsetting.
         If the geometric case is simple, it may marginally improve execution times to turn this off. However, turning
         this off for a complex case will negatively impact (significantly) spatial operation execution times.
        :raises: NotImplementedError, EmptySubsetError
        :rtype: :class:`~ocgis.new_interface.geom.AbstractSpatialVariable`
        :returns: A spatial variable the same geometry type.
        """
        # tdk: doc keep_touches
        # Only polygons are acceptable for subsetting.
        if type(polygon) not in (Polygon, MultiPolygon):
            raise NotImplementedError(type(polygon))

        ret = copy(self)
        # Create the fill array and reference the mask. This is the output geometry value array.
        fill = np.ma.array(ret.value, mask=True)
        ref_fill_mask = fill.mask.reshape(-1)

        if use_spatial_index:
            si = self.get_spatial_index()
            # Return the indices of the geometries intersecting the target geometry, and update the mask accordingly.
            for idx in si.iter_intersects(polygon, self.value.reshape(-1), keep_touches=keep_touches):
                ref_fill_mask[idx] = False
        else:
            # Prepare the polygon for faster spatial operations.
            prepared = prep(polygon)
            # We are not keeping touches at this point. Remember the mask is an inverse.
            for idx, geom in iter_array(self.value.reshape(-1), return_value=True):
                bool_value = False
                if prepared.intersects(geom):
                    if not keep_touches and polygon.touches(geom):
                        bool_value = True
                else:
                    bool_value = True
                ref_fill_mask[idx] = bool_value

        # If everything is masked, this is an empty subset.
        if ref_fill_mask.all():
            raise EmptySubsetError(self.name)

        # Set the returned value to the fill array.
        ret._value = fill

        return ret

    def get_intersection_masked(self, *args, **kwargs):
        # tdk: doc
        ret = self.get_intersects_masked(*args, **kwargs)

        ref_value = ret.value
        for idx, geom in iter_array(ref_value, return_value=True):
            ref_value[idx] = geom.intersection(args[0])

        return ret

    def get_nearest(self, target, return_index=False):
        target = target.centroid
        distances = {}
        for select_nearest_index, geom in iter_array(self.value, return_value=True):
            distances[target.distance(geom)] = select_nearest_index
        select_nearest_index = distances[min(distances.keys())]
        ret = self.value[select_nearest_index]

        if return_index:
            ret = (select_nearest_index[0], ret)

        return ret

    def _get_geometry_fill_(self, shape=None):
        if shape is None:
            shape = (self.grid.shape[0], self.grid.shape[1])
            mask = self.grid.value[0].mask
        else:
            mask = False
        fill = np.ma.array(np.zeros(shape), mask=mask, dtype=object)

        return fill

    def _get_shape_(self):
        ret = super(PointArray, self)._get_shape_()
        # If there is a grid, return this shape. The superclass relies on dimensions and private values.
        if len(ret) == 0 and self.grid is not None:
            ret = self.grid.shape
        return ret

    def _get_value_(self):
        # Create geometries for all the underlying coordinates regardless if the data is masked.
        ref_grid = self.grid.value.data

        fill = self._get_geometry_fill_()
        r_data = fill.data
        for idx_row, idx_col in iter_array(ref_grid[0], use_mask=False):
            y = ref_grid[0, idx_row, idx_col]
            x = ref_grid[1, idx_row, idx_col]
            pt = Point(x, y)
            r_data[idx_row, idx_col] = pt
        return fill

    def _set_value_(self, value):
        if not isinstance(value, ndarray) and value is not None:
            msg = 'Geometry values must be NumPy arrays to avoid automatic shapely transformations.'
            raise ValueError(msg)
        super(PointArray, self)._set_value_(value)
