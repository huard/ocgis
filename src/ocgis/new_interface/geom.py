from abc import ABCMeta, abstractmethod
from copy import copy

import fiona
import numpy as np
from numpy.core.multiarray import ndarray
from shapely import wkb
from shapely.geometry import Point, Polygon, MultiPolygon, mapping
from shapely.prepared import prep

from ocgis import constants
from ocgis import env, CoordinateReferenceSystem
from ocgis.exc import EmptySubsetError
from ocgis.new_interface.variable import Variable
from ocgis.util.environment import ogr
from ocgis.util.helpers import iter_array, get_none_or_slice, get_trimmed_array_by_mask, get_added_slice

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint


class AbstractSpatialVariable(Variable):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self._crs = None

        self.crs = kwargs.pop('crs', None)

        super(AbstractSpatialVariable, self).__init__(**kwargs)

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        if value is not None:
            assert isinstance(value, CoordinateReferenceSystem)
        self._crs = value

    @abstractmethod
    def get_mask(self):
        """Return the object mask."""

    @abstractmethod
    def set_mask(self):
        """Set the object mask."""

    @abstractmethod
    def update_crs(self, to_crs):
        """Update coordinate system in-place."""


class PointArray(AbstractSpatialVariable):
    def __init__(self, **kwargs):
        self._grid = kwargs.pop('grid', None)
        self._geom_type = kwargs.pop('geom_type', 'auto')

        kwargs['name'] = kwargs.get('name', 'geom')

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
    def geom_type(self):
        # Geometry objects may change part counts during operations. It is better to scan and update the geometry types
        # to account for these operations.
        if self._geom_type == 'auto':
            for geom in self.value.data.flat:
                if geom.geom_type.startswith('Multi'):
                    break
            self._geom_type = geom.geom_type
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
        # tdk: for polygon subsets we should use_bounds=False
        # First, subset the grid by the bounding box.
        if self._grid is not None:
            minx, miny, maxx, maxy = args[0].bounds
            _, slc = self.grid.get_subset_bbox(minx, miny, maxx, maxy, return_indices=True, use_bounds=True)
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
            ret._grid.set_mask(self.get_mask())

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

    def get_nearest(self, target, return_index=False):
        target = target.centroid
        distances = {}
        for select_nearest_index, geom in iter_array(self.value, return_value=True):
            distances[target.distance(geom)] = select_nearest_index
        select_nearest_index = distances[min(distances.keys())]
        ret = self.value[select_nearest_index]

        if return_index:
            ret = (ret, select_nearest_index[0],)

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

    def update_crs(self, to_crs):
        # Be sure and project masked geometries to maintain underlying geometries.
        r_value = self.value.data
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

    def write_fiona(self, path, driver='ESRI Shapefile'):
        name_uid = constants.HEADERS.ID_GEOMETRY.upper()
        schema = {'geometry': self.geom_type,
                  'properties': {name_uid: 'int'}}
        ref_prep = self._write_fiona_prep_geom_
        with fiona.open(path, 'w', driver=driver, crs=self.crs.value, schema=schema) as f:
            for uid, geom in enumerate(self.value.compressed()):
                geom = ref_prep(geom)
                feature = {'properties': {name_uid: uid}, 'geometry': mapping(geom)}
                f.write(feature)

        return path

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

    @staticmethod
    def _write_fiona_prep_geom_(geom):
        return geom
