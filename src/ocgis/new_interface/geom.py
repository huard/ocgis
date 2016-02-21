from abc import ABCMeta, abstractmethod
from collections import deque
from copy import copy

import fiona
import numpy as np
from numpy.core.multiarray import ndarray
from shapely import wkb
from shapely.geometry import Point, Polygon, MultiPolygon, mapping, MultiPoint, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union
from shapely.prepared import prep

from ocgis import constants
from ocgis import env, CoordinateReferenceSystem
from ocgis.exc import EmptySubsetError
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.variable import Variable, VariableCollection, AbstractContainer
from ocgis.util.environment import ogr
from ocgis.util.helpers import iter_array, get_none_or_slice, get_trimmed_array_by_mask, get_added_slice

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint

GEOM_TYPE_MAPPING = {'Polygon': Polygon, 'Point': Point, 'MultiPoint': MultiPoint, 'MultiPolygon': MultiPolygon}


class AbstractSpatialObject(AbstractInterfaceObject):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        self._crs = None
        self.crs = kwargs.pop('crs', None)
        super(AbstractSpatialObject, self).__init__(*args, **kwargs)

    @property
    def envelope(self):
        return box(*self.extent)

    @property
    def extent(self):
        return self._get_extent_()

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        if value is not None:
            assert isinstance(value, CoordinateReferenceSystem)
        self._crs = value

    def write_netcdf(self, dataset, **kwargs):
        if self.crs is not None:
            self.crs.write_to_rootgrp(dataset)
        super(AbstractSpatialObject, self).write_netcdf(dataset, **kwargs)

    @abstractmethod
    def update_crs(self, to_crs):
        """Update coordinate system in-place."""

    @abstractmethod
    def _get_extent_(self):
        """
        :returns: A tuple with order (minx, miny, maxx, maxy).
        :rtype: tuple
        """

    @abstractmethod
    def get_intersects(self, *args, **kwargs):
        """Perform an intersects operations."""

    @abstractmethod
    def get_intersects_masked(self, geometry, use_spatial_index=True, keep_touches=False):
        """Perform an intersects operation and mask non-intersecting elements."""

    @abstractmethod
    def get_nearest(self, target, return_indices=False):
        """Get nearest element to target geometry."""

    @abstractmethod
    def get_spatial_index(self):
        """Get the spatial index."""

    @abstractmethod
    def iter_records(self, use_mask=True):
        """Generate fiona-compatible records."""

    @abstractmethod
    def write_fiona(self, path, driver='ESRI Shapefile', use_mask=True):
        """Write to fiona-compatible drivers."""


class AbstractSpatialContainer(AbstractContainer, AbstractSpatialObject):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        crs = kwargs.pop('crs', None)
        parent = kwargs.pop('parent', None)
        name = kwargs.pop('name', None)
        AbstractContainer.__init__(self, name, parent=parent)
        AbstractSpatialObject.__init__(self, crs=crs)

    def write_netcdf(self, dataset, **kwargs):
        self.parent.write_netcdf(dataset, **kwargs)
        AbstractSpatialObject.write_netcdf(self, dataset)


# class SpatialContainer(AbstractInterfaceObject):
#     def __init__(self, point=None, polygon=None, grid=None, abstraction=None):
#         self._point = None
#         self._polygon = None
#         self._grid = None
#
#         self.abstraction = abstraction
#         self.point = point
#         self.polygon = polygon
#         self.grid = grid
#
#     def __getitem__(self, slc):
#         ret = copy(self)
#         ret._grid = get_none_or_slice(ret._grid, slc)
#         ret._point = get_none_or_slice(ret._point, slc)
#         ret._polygon = get_none_or_slice(ret._polygon, slc)
#         return ret
#
#     @property
#     def crs(self):
#         return get_grid_or_geom_attr(self, 'crs')
#
#     @property
#     def envelope(self):
#         return get_grid_or_geom_attr(self, 'envelope')
#
#     @property
#     def extent(self):
#         return get_grid_or_geom_attr(self, 'extent')
#
#     @property
#     def geom(self):
#         if self.abstraction is None:
#             ret = self.get_optimal_geometry()
#         else:
#             ret = getattr(self, self.abstraction)
#         assert ret is not None
#         return ret
#
#     @property
#     def grid(self):
#         return self._grid
#
#     @grid.setter
#     def grid(self, value):
#         if value is not None:
#             assert isinstance(value, GridXY)
#         self._grid = value
#
#     @property
#     def ndim(self):
#         return get_grid_or_geom_attr(self, 'ndim')
#
#     @property
#     def point(self):
#         if self._point is None:
#             if self._grid is not None:
#                 self._point = PointArray(grid=self._grid, crs=self._grid.crs)
#         return self._point
#
#     @point.setter
#     def point(self, value):
#         if value is not None:
#             assert isinstance(value, PointArray)
#         self._point = value
#
#     @property
#     def polygon(self):
#         if self._polygon is None:
#             if self._grid is not None:
#                 try:
#                     self._polygon = PolygonArray(grid=self._grid, crs=self._grid.crs)
#                 except GridDeficientError:
#                     pass
#         return self._polygon
#
#     @polygon.setter
#     def polygon(self, value):
#         if value is not None:
#             assert isinstance(value, PolygonArray)
#         self._polygon = value
#
#     @property
#     def shape(self):
#         return get_grid_or_geom_attr(self, 'shape')
#
#     def as_variable_collection(self):
#         ret = VariableCollection()
#         if self._grid is not None:
#             ret.add_variable(self._grid.x)
#             ret.add_variable(self._grid.y)
#         for geom in filter(lambda x: x is not None, [self._point, self._polygon]):
#             ret.add_variable(geom)
#         return ret
#
#     def copy(self):
#         return copy(self)
#
#     def get_intersects(self, *args, **kwargs):
#         return get_spatial_operation(self, 'get_intersects', args, kwargs)
#
#     def get_intersection(self, *args, **kwargs):
#         return get_spatial_operation(self, 'get_intersection', args, kwargs)
#
#     def get_nearest(self, *args, **kwargs):
#         return get_spatial_operation(self, 'get_nearest', args, kwargs)
#
#     def get_optimal_geometry(self):
#         return self.polygon or self.point
#
#     def update_crs(self, *args, **kwargs):
#         self._apply_(['_grid', '_point', '_polygon'], 'update_crs', args, kwargs=kwargs, inplace=True)
#
#     def write_netcdf(self, *args, **kwargs):
#         raise NotImplementedError
#
#     def _apply_(self, targets, func, args, kwargs=None, inplace=False):
#         kwargs = kwargs or {}
#         for target_name in targets:
#             target = self.__dict__[target_name]
#             if target is not None and hasattr(target, func):
#                 ref = getattr(target, func)
#                 new_value = ref(*args, **kwargs)
#                 if not inplace:
#                     setattr(self, target_name[1:], new_value)


class AbstractSpatialVariable(Variable, AbstractSpatialObject):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        crs = kwargs.pop('crs', None)
        Variable.__init__(self, **kwargs)
        AbstractSpatialObject.__init__(self, crs=crs)


class GeometryVariable(AbstractSpatialVariable):

    def __init__(self, **kwargs):
        self._name_uid = None

        self._geom_type = kwargs.pop('geom_type', 'auto')
        super(GeometryVariable, self).__init__(**kwargs)

    @property
    def area(self):
        r_value = self.masked_value
        fill = np.ones(r_value.shape, dtype=env.NP_FLOAT)
        fill = np.ma.array(fill, mask=self.get_mask())
        for slc, geom in iter_array(r_value, return_value=True):
            fill[slc] = geom.area
        return fill

    @property
    def dtype(self):
        # Geometry arrays are always object arrays.
        return object

    @dtype.setter
    def dtype(self, value):
        # Geometry data types are always objects. Ignore any passed value.
        pass

    @property
    def geom_type(self):
        # Geometry objects may change part counts during operations. It is better to scan and update the geometry types
        # to account for these operations.
        if self._geom_type == 'auto':
            self._geom_type = get_geom_type(self.value)
        return self._geom_type

    @property
    def weights(self):
        area = self.area
        area.data[area.data == 0] = 1.0
        return area / area.max()

    @property
    def uid(self):
        if self._name_uid is None:
            return None
        else:
            return self.parent[self._name_uid]

    @classmethod
    def read_gis(cls, source, name, name_uid, name_dimension=None):
        if name_dimension is None:
            name_dimension = constants.NAME_GEOMETRY_DIMENSION

        geom_key = constants.DEFAULT_GEOMETRY_KEY
        len_source = len(source)
        for ctr, record in enumerate(source):
            if ctr == 0:
                values = {k: [None] * len_source for k in record['properties'].keys()}
                ret_value = [None] * len_source
                crs = CoordinateReferenceSystem(value=record['meta']['crs'])
            ret_value[ctr] = record[geom_key]
            for k, v in record['properties'].iteritems():
                values[k][ctr] = v
        parent = VariableCollection()
        dimension = Dimension(name_dimension, len_source)
        for k, v in values.iteritems():
            var = Variable(value=v, name=k, dimensions=dimension)
            parent.add_variable(var)
        ret = GeometryVariable(name=name, parent=parent, value=ret_value, crs=crs, dimensions=dimension)
        ret.set_uid(ret.parent[name_uid])
        return ret

    def set_uid(self, variable):
        if self.parent is None:
            self.parent = VariableCollection(variables=[self])
        self.parent.add_variable(variable, force=True)
        self._name_uid = variable.name

    def get_intersects(self, *args, **kwargs):
        return_slice = kwargs.pop('return_slice', False)
        slc = [slice(None)] * self.ndim
        ret = self.get_intersects_masked(*args, **kwargs)

        if self.ndim == 1:
            # For one-dimensional data, assume it is unstructured and compress the returned data.
            adjust = np.where(np.invert(ret.get_mask()))
            ret_slc = adjust
        else:
            # Barbed and circular geometries may result in rows and or columns being entirely masked. These rows and
            # columns should be trimmed.
            _, adjust = get_trimmed_array_by_mask(ret.get_mask(), return_adjustments=True)
            # Adjust the return indices to account for the possible mask trimming.
            ret_slc = tuple([get_added_slice(s, a) for s, a in zip(slc, adjust)])

        # Use the adjustments to trim the returned data object.
        ret = ret.__getitem__(adjust)

        if return_slice:
            ret = (ret, ret_slc)

        return ret

    def get_intersection(self, *args, **kwargs):
        ret = self.get_intersects(*args, **kwargs)
        # If indices are being returned, this will be a tuple.
        if kwargs.get('return_slice'):
            obj = ret[0]
        else:
            obj = ret
        for idx, geom in iter_array(obj.value, return_value=True):
            obj.value[idx] = geom.intersection(args[0])
        return ret

    def get_intersects_masked(self, geometry, use_spatial_index=True, keep_touches=False, cascade=False):
        """
        :param geometry: The Shapely geometry to use for subsetting.
        :type geometry: :class:`shapely.geometry.BaseGeometry'
        :param bool use_spatial_index: If ``False``, do not use the :class:`rtree.index.Index` for spatial subsetting.
         If the geometric case is simple, it may marginally improve execution times to turn this off. However, turning
         this off for a complex case will negatively impact (significantly) spatial operation execution times.
        :raises: NotImplementedError, EmptySubsetError
        :rtype: :class:`~ocgis.new_interface.geom.AbstractSpatialVariable`
        :returns: A spatial variable the same geometry type.
        """
        # tdk: doc keep_touches

        ret = self.copy()
        original_mask = ret.get_mask().copy()
        fill = geometryvariable_get_mask_from_intersects(self, geometry, use_spatial_index=use_spatial_index,
                                                         keep_touches=keep_touches,
                                                         original_mask=original_mask)
        ret.set_mask(np.logical_or(fill, original_mask), cascade=cascade)
        return ret

    def get_mask_from_intersects(self, geometry, use_spatial_index=True, keep_touches=False, original_mask=None):
        return geometryvariable_get_mask_from_intersects(self, geometry,
                                                         use_spatial_index=use_spatial_index,
                                                         keep_touches=keep_touches,
                                                         original_mask=original_mask)

    def get_intersection_masked(self, *args, **kwargs):
        ret = self.get_intersects_masked(*args, **kwargs)

        ref_value = ret.value
        for idx, geom in iter_array(ref_value, return_value=True):
            ref_value[idx] = geom.intersection(args[0])

        return ret

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
        for idx, geom in iter_array(self.masked_value.compressed(), return_value=True):
            r_add(idx[0], geom)

        return si

    def get_unioned(self):
        """
        Unions _unmasked_ geometry objects.
        """

        to_union = [geom for geom in self.masked_value.compressed().flat]
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
        ret._dimensions = None
        ret._value = fill
        return ret

    def update_crs(self, to_crs):
        # Be sure and project masked geometries to maintain underlying geometries.
        r_value = self.value.reshape(-1)
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

    def iter_records(self, use_mask=True):
        if use_mask:
            to_itr = self.masked_value.compressed()
        else:
            to_itr = self.value.flat
        r_geom_class = GEOM_TYPE_MAPPING[self.geom_type]

        for idx, geom in enumerate(to_itr):
            # Convert geometry to a multi-geometry if needed.
            if not isinstance(geom, r_geom_class):
                geom = r_geom_class([geom])
            feature = {'properties': {}, 'geometry': mapping(geom)}
            yield feature

    def write_fiona(self, path, driver='ESRI Shapefile', use_mask=True):
        if self.crs is None:
            crs = None
        else:
            crs = self.crs.value
        schema = {'geometry': self.geom_type,
                  'properties': {}}

        with fiona.open(path, 'w', driver=driver, crs=crs, schema=schema) as f:
            for record in self.iter_records(use_mask=use_mask):
                f.write(record)
        return path

    def write_netcdf(self, *args, **kwargs):
        # tdk: test with a joint netcdf-shapefile output
        pass

    def _get_extent_(self):
        raise NotImplementedError

    def _set_value_(self, value):
        if not isinstance(value, ndarray) and value is not None:
            if isinstance(value, BaseGeometry):
                itr = [value]
                shape = 1
            else:
                itr = value
                shape = len(value)
            value = np.zeros(shape, dtype=self.dtype)
            for idx, element in enumerate(itr):
                value[idx] = element
        super(GeometryVariable, self)._set_value_(value)


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


def geometryvariable_get_mask_from_intersects(gvar, geometry, use_spatial_index=True,
                                              keep_touches=False, original_mask=None):
    # Create the fill array and reference the mask. This is the output geometry value array.
    if original_mask is None:
        original_mask = gvar.get_mask()
    fill = original_mask.copy()
    fill.fill(True)
    ref_fill_mask = fill.reshape(-1)
    # Track global indices because spatial operations only occur on non-masked values.
    global_index = np.arange(reduce(lambda x, y: x * y, original_mask.shape))
    global_index = np.ma.array(global_index, mask=original_mask).compressed()
    if use_spatial_index:
        si = gvar.get_spatial_index()
        # Return the indices of the geometries intersecting the target geometry, and update the mask accordingly.
        for idx in si.iter_intersects(geometry, gvar.masked_value.compressed(), keep_touches=keep_touches):
            ref_fill_mask[global_index[idx]] = False
    else:
        # Prepare the polygon for faster spatial operations.
        prepared = prep(geometry)
        # We are not keeping touches at this point. Remember the mask is an inverse.
        for idx, geom in iter_array(gvar.masked_value.compressed(), return_value=True):
            bool_value = False
            if prepared.intersects(geom):
                if not keep_touches and geometry.touches(geom):
                    bool_value = True
            else:
                bool_value = True
            ref_fill_mask[global_index[idx]] = bool_value

    # If everything is masked, this is an empty subset.
    if ref_fill_mask.all():
        raise EmptySubsetError(gvar.name)

    return fill
