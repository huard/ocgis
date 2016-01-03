import itertools
from abc import ABCMeta
from collections import deque
from copy import copy

import fiona
import numpy as np
from numpy.core.multiarray import ndarray
from shapely import wkb
from shapely.geometry import Point, Polygon, MultiPolygon, mapping, MultiPoint
from shapely.ops import cascaded_union
from shapely.prepared import prep

from ocgis import constants
from ocgis import env
from ocgis.exc import EmptySubsetError, GridDeficientError
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.grid import GridXY, AbstractSpatialObject
from ocgis.new_interface.variable import Variable
from ocgis.util.environment import ogr
from ocgis.util.helpers import iter_array, get_none_or_slice, get_trimmed_array_by_mask, get_added_slice

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint

GEOM_TYPE_MAPPING = {'Polygon': Polygon, 'Point': Point, 'MultiPoint': MultiPoint, 'MultiPolygon': MultiPolygon}


class SpatialContainer(AbstractInterfaceObject):
    def __init__(self, point=None, polygon=None, grid=None, abstraction=None):
        self._point = None
        self._polygon = None
        self._grid = None

        self.abstraction = abstraction
        self.point = point
        self.poly = polygon
        self.grid = grid

    def __getitem__(self, slc):
        ret = copy(self)
        ret._grid = get_none_or_slice(ret._grid, slc)
        ret._point = get_none_or_slice(ret._point, slc)
        ret._polygon = get_none_or_slice(ret._polygon, slc)
        return ret

    @property
    def crs(self):
        return get_grid_or_geom_attr(self, 'crs')

    @property
    def envelope(self):
        return get_grid_or_geom_attr(self, 'envelope')

    @property
    def extent(self):
        return get_grid_or_geom_attr(self, 'extent')

    @property
    def geom(self):
        if self.abstraction is None:
            ret = self.get_optimal_geometry()
        else:
            ret = getattr(self, self.abstraction)
        assert ret is not None
        return ret

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        if value is not None:
            assert isinstance(value, GridXY)
        self._grid = value

    @property
    def ndim(self):
        return get_grid_or_geom_attr(self, 'ndim')

    @property
    def point(self):
        if self._point is None:
            if self._grid is not None:
                self._point = PointArray(grid=self._grid, crs=self._grid.crs)
        return self._point

    @point.setter
    def point(self, value):
        if value is not None:
            assert isinstance(value, PointArray)
        self._point = value

    @property
    def polygon(self):
        if self._polygon is None:
            if self._grid is not None:
                try:
                    self._polygon = PolygonArray(grid=self._grid, crs=self._grid.crs)
                except GridDeficientError:
                    pass
        return self._polygon

    @polygon.setter
    def polygon(self, value):
        if value is not None:
            assert isinstance(value, PolygonArray)
        self._polygon = value

    @property
    def shape(self):
        return get_grid_or_geom_attr(self, 'shape')

    def copy(self):
        return copy(self)

    def get_intersects(self, *args, **kwargs):
        return get_spatial_operation(self, 'get_intersects', args, kwargs)

    def get_intersection(self, *args, **kwargs):
        return get_spatial_operation(self, 'get_intersection', args, kwargs)

    def get_nearest(self, *args, **kwargs):
        return get_spatial_operation(self, 'get_nearest', args, kwargs)

    def get_optimal_geometry(self):
        return self.polygon or self.point

    def update_crs(self, *args, **kwargs):
        self._apply_(['_grid', '_point', '_polygon'], 'update_crs', args, kwargs=kwargs, inplace=True)

    def write_netcdf(self, *args, **kwargs):
        raise NotImplementedError

    def _apply_(self, targets, func, args, kwargs=None, inplace=False):
        kwargs = kwargs or {}
        for target_name in targets:
            target = self.__dict__[target_name]
            if target is not None and hasattr(target, func):
                ref = getattr(target, func)
                new_value = ref(*args, **kwargs)
                if not inplace:
                    setattr(self, target_name[1:], new_value)


class AbstractSpatialVariable(Variable, AbstractSpatialObject):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        crs = kwargs.pop('crs', None)
        Variable.__init__(self, **kwargs)
        AbstractSpatialObject.__init__(self, crs=crs)


class PointArray(AbstractSpatialVariable):
    # Flag for grid subsetting by bounding box. Bounds should not be used for points.
    _use_bounds = False

    def __init__(self, **kwargs):
        self._grid = kwargs.pop('grid', None)
        self._geom_type = kwargs.pop('geom_type', 'auto')

        kwargs['name'] = kwargs.get('name', 'geom')

        super(PointArray, self).__init__(**kwargs)

        if self._value is None:
            if self._grid is None:
                msg = 'A "value" or "grid" is required.'
                raise ValueError(msg)

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
    def geom_type(self):
        # Geometry objects may change part counts during operations. It is better to scan and update the geometry types
        # to account for these operations.
        if self._geom_type == 'auto':
            self._geom_type = get_geom_type(self.value.data)
        return self._geom_type

    @property
    def grid(self):
        return self._grid

    @property
    def weights(self):
        ret = np.ones(self.value.shape, dtype=env.NP_FLOAT)
        ret = np.ma.array(ret, mask=self.value.mask)
        return ret

    def get_intersects(self, *args, **kwargs):
        return_indices = kwargs.pop('return_indices', False)
        # First, subset the grid by the bounding box.
        if self._grid is not None:
            minx, miny, maxx, maxy = args[0].bounds
            _, slc = self.grid.get_subset_bbox(minx, miny, maxx, maxy, return_indices=True, use_bounds=self._use_bounds)
            ret = self.__getitem__(slc)
        else:
            ret = self
            slc = [slice(None)] * self.ndim
        ret = ret.get_intersects_masked(*args, **kwargs)
        # Barbed and circular geometries may result in rows and or columns being entirely masked. These rows and
        # columns should be trimmed.
        _, adjust = get_trimmed_array_by_mask(ret.get_mask(), return_adjustments=True)
        # Use the adjustments to trim the returned data object.
        ret = ret.__getitem__(adjust)
        # Adjust the returned slices.
        if return_indices:
            ret_slc = tuple([get_added_slice(s, a) for s, a in zip(slc, adjust)])
            ret = (ret, ret_slc)

        return ret

    def get_intersection(self, *args, **kwargs):
        ret = self.get_intersects(*args, **kwargs)
        # If indices are being returned, this will be a tuple.
        if kwargs.get('return_indices'):
            obj = ret[0]
        else:
            obj = ret
        for idx, geom in iter_array(obj.value, return_value=True):
            obj.value[idx] = geom.intersection(args[0])
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
        ret.value = fill

        # Update the grid mask if it is associated with the object.
        if ret._grid is not None:
            ret._grid.set_mask(ret.get_mask())

        return ret

    def get_intersection_masked(self, *args, **kwargs):
        ret = self.get_intersects_masked(*args, **kwargs)

        ref_value = ret.value
        for idx, geom in iter_array(ref_value, return_value=True):
            ref_value[idx] = geom.intersection(args[0])

        return ret

    def get_mask(self):
        if self._value is None:
            ret = self.grid.get_mask()
        else:
            ret = self.value.mask
        return ret

    def set_mask(self, value):
        if self._value is not None:
            self.value.mask = value
        if self._grid is not None:
            self.grid.set_mask(value)

    def get_nearest(self, target, return_indices=False):
        target = target.centroid
        distances = {}
        for select_nearest_index, geom in iter_array(self.value, return_value=True):
            distances[target.distance(geom)] = select_nearest_index
        select_nearest_index = distances[min(distances.keys())]
        ret = self[select_nearest_index]

        if return_indices:
            ret = (ret, select_nearest_index)

        return ret

    def get_spatial_index(self):
        # "rtree" is an optional dependency.
        from ocgis.util.spatial.index import SpatialIndex
        # Fill the spatial index with unmasked values only.
        si = SpatialIndex()
        r_add = si.add
        # Add the geometries to the index.
        for idx, geom in iter_array(self.value.reshape(-1), return_value=True, use_mask=True):
            r_add(idx[0], geom)

        return si

    def get_unioned(self):
        """
        Unions _unmasked_ geometry objects.
        """

        to_union = [geom for geom in self.value.compressed().flat]
        processed_to_union = deque()
        for geom in to_union:
            if isinstance(geom, MultiPolygon) or isinstance(geom, MultiPoint):
                for element in geom:
                    processed_to_union.append(element)
            else:
                processed_to_union.append(geom)
        unioned = cascaded_union(processed_to_union)

        fill = np.ma.array([None], mask=False, dtype=object)
        fill[0] = unioned

        ret = copy(self)
        ret._grid = None
        ret._dimensions = None
        ret._value = fill
        return ret

    def update_crs(self, to_crs):
        # Be sure and project masked geometries to maintain underlying geometries.
        r_value = self.value.data.reshape(-1)
        r_loads = wkb.loads
        r_create = ogr.CreateGeometryFromWkb
        to_sr = to_crs.sr
        from_sr = self.crs.sr
        for idx, geom in enumerate(r_value.flat):
            ogr_geom = r_create(geom.wkb)
            ogr_geom.AssignSpatialReference(from_sr)
            ogr_geom.TransformTo(to_sr)
            r_value[idx] = r_loads(ogr_geom.ExportToWkb())
        self.crs = to_crs
        # The grid is not longer representative of the data.
        self._grid = None

    def iter_records(self, use_mask=True):
        if use_mask:
            to_itr = self.value.compressed()
        else:
            to_itr = self.value.data.flat
        r_geom_class = GEOM_TYPE_MAPPING[self.geom_type]
        name_uid = constants.HEADERS.ID_GEOMETRY.upper()
        for idx, geom in enumerate(to_itr):
            # Convert geometry to a multi-geometry if needed.
            if not isinstance(geom, r_geom_class):
                geom = r_geom_class([geom])
            feature = {'properties': {name_uid: idx}, 'geometry': mapping(geom)}
            yield feature

    def write_fiona(self, path, driver='ESRI Shapefile', use_mask=True):
        if self.crs is None:
            crs = None
        else:
            crs = self.crs.value
        name_uid = constants.HEADERS.ID_GEOMETRY.upper()
        schema = {'geometry': self.geom_type,
                  'properties': {name_uid: 'int'}}

        with fiona.open(path, 'w', driver=driver, crs=crs, schema=schema) as f:
            for record in self.iter_records(use_mask=use_mask):
                f.write(record)
        return path

    def _get_dimensions_(self):
        if self.grid is not None:
            ret = self.grid.dimensions
        else:
            ret = self._dimensions
        return ret

    def _get_extent_(self):
        if self.grid is None:
            raise NotImplementedError
        else:
            return self.grid.extent

    def _get_geometry_fill_(self, shape=None):
        self.grid.expand()
        if shape is None:
            shape = (self.grid.shape[0], self.grid.shape[1])
            mask = self.grid.get_mask()
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
        fill = self._get_geometry_fill_()

        # Create geometries for all the underlying coordinates regardless if the data is masked.
        x_data = self.grid.x.value.data
        y_data = self.grid.y.value.data

        r_data = fill.data
        for idx_row, idx_col in iter_array(y_data, use_mask=False):
            y = y_data[idx_row, idx_col]
            x = x_data[idx_row, idx_col]
            pt = Point(x, y)
            r_data[idx_row, idx_col] = pt
        return fill

    def _set_value_(self, value):
        if not isinstance(value, ndarray) and value is not None:
            msg = 'Geometry values must be NumPy arrays to avoid automatic shapely transformations.'
            raise ValueError(msg)
        super(PointArray, self)._set_value_(value)


class PolygonArray(PointArray):
    _use_bounds = True

    def __init__(self, **kwargs):
        super(PolygonArray, self).__init__(**kwargs)

        if self._value is None:
            if not self.grid.has_bounds:
                msg = 'Grid bounds/corners must be available.'
                raise GridDeficientError(msg)

    @property
    def area(self):
        r_value = self.value
        fill = np.ones(r_value.shape, dtype=env.NP_FLOAT)
        fill = np.ma.array(fill, mask=r_value.mask)
        for (ii, jj), geom in iter_array(r_value, return_value=True):
            fill[ii, jj] = geom.area
        return fill

    @property
    def weights(self):
        return self.area / self.area.max()

    def _get_value_(self):
        fill = self._get_geometry_fill_()
        r_data = fill.data
        grid = self.grid

        if grid.is_vectorized and grid.has_bounds:
            ref_row_bounds = grid.y.bounds.value.data
            ref_col_bounds = grid.x.bounds.value.data
            for idx_row, idx_col in itertools.product(range(ref_row_bounds.shape[0]), range(ref_col_bounds.shape[0])):
                row_min, row_max = ref_row_bounds[idx_row, :].min(), ref_row_bounds[idx_row, :].max()
                col_min, col_max = ref_col_bounds[idx_col, :].min(), ref_col_bounds[idx_col, :].max()
                r_data[idx_row, idx_col] = Polygon(
                    [(col_min, row_min), (col_min, row_max), (col_max, row_max), (col_max, row_min)])
        # The grid dimension may not have row/col or row/col bounds.
        else:
            # We want geometries for everything even if masked.
            x_corners = grid.x.bounds.value.data
            y_corners = grid.y.bounds.value.data
            corners = np.vstack((y_corners, x_corners))
            # tdk: implement mesh value on grid to maintain vectorized grid
            corners = corners.reshape([2] + list(x_corners.shape))
            range_row = range(grid.shape[0])
            range_col = range(grid.shape[1])
            for row, col in itertools.product(range_row, range_col):
                current_corner = corners[:, row, col]
                coords = np.hstack((current_corner[1, :].reshape(-1, 1),
                                    current_corner[0, :].reshape(-1, 1)))
                polygon = Polygon(coords)
                r_data[row, col] = polygon
        return fill


def get_geom_type(data):
    for geom in data.flat:
        geom_type = geom.geom_type
        if geom_type.startswith('Multi'):
            break
    return geom_type


def get_grid_or_geom_attr(sc, attr):
    if sc.grid is None:
        ret = getattr(sc.geom, attr)
    else:
        ret = getattr(sc.grid, attr)
    return ret


def get_spatial_operation(sc, name, args, kwargs):
    """
    :param sc: A spatial containter.
    :type sc: :class:`ocgis.new_interface.geom.SpatialContainer'
    :param str name: Name of the spatial operation.
    :param tuple args: Arguments to the spatial operation.
    :param dict kwargs: Keyword arguments to the spatial operation.
    :returns: Performs the spatial operation on the input spatial container.
    :rtype: :class:`ocgis.new_interface.geom.SpatialContainer'
    """
    ret = copy(sc)

    # Always return indices so the other geometry can be sliced if needed.
    kwargs = kwargs.copy()
    original_return_indices = kwargs.get('return_indices', False)
    kwargs['return_indices'] = True

    # Subset the optimal geometry.
    geom = ret.geom
    operation = getattr(geom, name)
    geom_subset, slc = operation(*args, **kwargs)
    # Synchronize the underlying grid.
    ret.grid = geom_subset.grid
    # Update the other geometry by slicing given the underlying subset. Only slice if it is loaded.
    if isinstance(geom, PolygonArray):
        ret.polygon = geom_subset
        ret.point = get_none_or_slice(ret._point, slc)
    else:
        ret.point = geom_subset
        ret.polygon = get_none_or_slice(ret._polygon, slc)

    if original_return_indices:
        ret = (ret, slc)
    else:
        ret = ret

    return ret
