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
from ocgis import env
from ocgis.constants import WrapAction
from ocgis.exc import EmptySubsetError
from ocgis.interface.base.crs import WrappableCoordinateReferenceSystem
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.mpi import variable_gather, MPI_COMM
from ocgis.new_interface.variable import VariableCollection, AbstractContainer, SourcedVariable, Variable
from ocgis.util.environment import ogr
from ocgis.util.helpers import iter_array, get_none_or_slice, get_trimmed_array_by_mask

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint

GEOM_TYPE_MAPPING = {'Polygon': Polygon, 'Point': Point, 'MultiPoint': MultiPoint, 'MultiPolygon': MultiPolygon}


class AbstractSpatialObject(AbstractInterfaceObject):
    def __init__(self, *args, **kwargs):
        self._crs_name = None
        self.crs = kwargs.pop('crs', None)
        super(AbstractInterfaceObject, self).__init__(*args, **kwargs)

    @property
    def crs(self):
        if self.parent is not None and self._crs_name is not None:
            ret = self.parent.get(self._crs_name)
        else:
            ret = None
        return ret

    @crs.setter
    def crs(self, value):
        if value is None:
            if self.crs is not None:
                self.parent.pop(self._crs_name)
                self._crs_name = None
        else:
            if self.parent is None:
                self.initialize_parent()
            if self._crs_name is not None:
                self.parent.pop(self._crs_name)
            self.parent.add_variable(value, force=True)
            self._crs_name = value.name

    @property
    def wrapped_state(self):
        if isinstance(self.crs, WrappableCoordinateReferenceSystem):
            ret = self.crs.get_wrapped_state(self)
        else:
            ret = None
        return ret

    def unwrap(self):
        if not isinstance(self.crs, WrappableCoordinateReferenceSystem):
            raise ValueError("Only spherical coordinate systems may be wrapped/unwrapped.")
        else:
            self.crs.wrap_or_unwrap(WrapAction.UNWRAP, self)

    def wrap(self):
        if not isinstance(self.crs, WrappableCoordinateReferenceSystem):
            raise ValueError("Only spherical coordinate systems may be wrapped/unwrapped.")
        else:
            self.crs.wrap_or_unwrap(WrapAction.WRAP, self)


class AbstractOperationsSpatialObject(AbstractSpatialObject):
    __metaclass__ = ABCMeta

    @property
    def envelope(self):
        return box(*self.extent)

    @property
    def extent(self):
        return self._get_extent_()

    @abstractmethod
    def update_crs(self, to_crs):
        """Update coordinate system in-place."""

        if self.crs is None:
            msg = 'The current CRS is None and cannot be updated.'
            raise ValueError(msg)

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


class AbstractSpatialContainer(AbstractContainer, AbstractOperationsSpatialObject):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        crs = kwargs.pop('crs', None)
        parent = kwargs.pop('parent', None)
        name = kwargs.pop('name', None)
        AbstractContainer.__init__(self, name, parent=parent)
        AbstractOperationsSpatialObject.__init__(self, crs=crs)


class AbstractSpatialVariable(SourcedVariable, AbstractOperationsSpatialObject):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        crs = kwargs.pop('crs', None)
        SourcedVariable.__init__(self, **kwargs)
        AbstractOperationsSpatialObject.__init__(self, crs=crs)

    def deepcopy(self, eager=False):
        ret = super(AbstractSpatialVariable, self).deepcopy(eager)
        ret.crs = ret.crs.deepcopy()
        return ret


class GeometryVariable(AbstractSpatialVariable):
    def __init__(self, *args, **kwargs):
        self._name_uid = None
        self._geom_type = kwargs.pop('geom_type', 'auto')

        if kwargs.get('name') is None:
            kwargs['name'] = 'geom'
        super(GeometryVariable, self).__init__(*args, **kwargs)

    def __add_to_collection_finalize__(self, vc):
        super(GeometryVariable, self).__add_to_collection_finalize__(vc)
        if self.crs is not None:
            vc.add_variable(self.crs, force=True)

    @property
    def area(self):
        r_value = self.masked_value
        fill = np.ones(r_value.shape, dtype=env.NP_FLOAT)
        fill = np.ma.array(fill, mask=self.get_mask().copy())
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

    def set_uid(self, variable):
        if self.parent is None:
            self.parent = VariableCollection(variables=[self])
        self.parent.add_variable(variable, force=True)
        self._name_uid = variable.name
        variable.attrs['ocgis'] = constants.DEFAULT_ATTRIBUTE_VALUE_FOR_GEOMETRY_UNIQUE_IDENTIFIER

    def get_intersects(self, *args, **kwargs):
        """
        :param bool return_slice: (``='False'``) If ``True``, return the _global_ slice that will guarantee no masked
         elements outside the subset geometry.
        :param comm: (``=None``) If ``None``, use the default MPI communicator.
        :return:
        :raises: EmptySubsetError
        """
        return_slice = kwargs.pop('return_slice', False)
        cascade = kwargs.pop('cascade', False)
        comm = kwargs.pop('comm', None) or MPI_COMM
        rank = comm.Get_rank()
        size = comm.Get_size()

        if size > 1 and not self.has_dimensions:
            raise ValueError('Dimensions are required for a distributed intersects operation.')

        ret = self.copy()
        intersects_mask = ret.get_mask_from_intersects(*args, **kwargs)
        intersects_mask = Variable(name='mask_gather', value=intersects_mask, dimensions=ret.dimensions, dtype=bool)
        gathered_mask = variable_gather(intersects_mask, comm=comm)

        adjust = None
        raise_empty_subset = False
        if rank == 0:
            assert gathered_mask.dtype == bool
            if gathered_mask.value.all():
                raise_empty_subset = True
            else:
                _, adjust = get_trimmed_array_by_mask(gathered_mask.value, return_adjustments=True)
        adjust = comm.bcast(adjust)
        raise_empty_subset = comm.bcast(raise_empty_subset)

        if raise_empty_subset:
            raise EmptySubsetError(self.name)

        if size > 1:
            ret = ret.get_distributed_slice(adjust, comm=comm)
            ret_mask = intersects_mask.get_distributed_slice(adjust, comm=comm).value
        else:
            ret = ret.__getitem__(adjust)
            ret_mask = intersects_mask.__getitem__(adjust).value

        ret.set_mask(ret_mask, cascade=cascade)

        # tdk: need to implement fancy index-based slicing for the one-dimensional unstructured case
        # if self.ndim == 1:
        #     # For one-dimensional data, assume it is unstructured and compress the returned data.
        #     adjust = np.where(np.invert(ret.get_mask()))
        #     ret_slc = adjust

        if return_slice:
            ret = (ret, adjust)

        return ret

    def get_intersection(self, *args, **kwargs):
        ret = self.get_intersects(*args, **kwargs)
        # If indices are being returned, this will be a tuple.
        if kwargs.get('return_slice', False):
            obj = ret[0]
        else:
            obj = ret
        for idx, geom in iter_array(obj.value, return_value=True):
            obj.value[idx] = geom.intersection(args[0])
        return ret

    def get_mask_from_intersects(self, geometry_or_bounds, use_spatial_index=True, keep_touches=False,
                                 original_mask=None):
        # Transform bounds sequence to a geometry.
        if not isinstance(geometry_or_bounds, BaseGeometry):
            geometry_or_bounds = box(*geometry_or_bounds)

        # Empty variables return an empty array.
        if self.is_empty:
            ret = np.array([], dtype=bool)
        else:
            ret = geometryvariable_get_mask_from_intersects(self, geometry_or_bounds,
                                                            use_spatial_index=use_spatial_index,
                                                            keep_touches=keep_touches,
                                                            original_mask=original_mask)
        return ret

    def get_intersection_masked(self, *args, **kwargs):
        ret = self.get_intersects_masked(*args, **kwargs)

        if not self.is_empty:
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

    def get_spatial_index(self, target=None):
        """
        Return a spatial index for the geometry variable.
        :param target: If this is a boolean array, use this as the add target. Otherwise, use the compressed masked
         values.
        :return:
        """
        # "rtree" is an optional dependency.
        from ocgis.util.spatial.index import SpatialIndex
        # Fill the spatial index with unmasked values only.
        si = SpatialIndex()
        # Use compressed masked values if target is not available.
        if target is None:
            target = self.masked_value.compressed()
        # Add the geometries to the index.
        r_add = si.add
        for idx, geom in iter_array(target, return_value=True):
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
        super(GeometryVariable, self).update_crs(to_crs)
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


def geometryvariable_get_mask_from_intersects(gvar, geometry, use_spatial_index=True, keep_touches=False,
                                              original_mask=None):
    # Create the fill array and reference the mask. This is the output geometry value array.
    if original_mask is None:
        original_mask = gvar.get_mask()
    fill = original_mask.copy()
    fill.fill(True)
    ref_fill_mask = fill.reshape(-1)

    # Track global indices because spatial operations only occur on non-masked values.
    global_index = np.arange(original_mask.size)
    global_index = np.ma.array(global_index, mask=original_mask).compressed()
    # Select the geometry targets. If an original mask is provided, use this. It may be modified to limit the search
    # area for intersects operations. Useful for speeding up grid subsetting operations.
    geometry_target = np.ma.array(gvar.value, mask=original_mask).compressed()

    if use_spatial_index:
        si = gvar.get_spatial_index(target=geometry_target)
        # Return the indices of the geometries intersecting the target geometry, and update the mask accordingly.
        for idx in si.iter_intersects(geometry, geometry_target, keep_touches=keep_touches):
            ref_fill_mask[global_index[idx]] = False
    else:
        # Prepare the polygon for faster spatial operations.
        prepared = prep(geometry)
        # We are not keeping touches at this point. Remember the mask is an inverse.
        for idx, geom in iter_array(geometry_target, return_value=True):
            bool_value = False
            if prepared.intersects(geom):
                if not keep_touches and geometry.touches(geom):
                    bool_value = True
            else:
                bool_value = True
            ref_fill_mask[global_index[idx]] = bool_value

    return fill
