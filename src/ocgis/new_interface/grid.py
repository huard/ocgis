from abc import abstractmethod, ABCMeta
from copy import copy

import numpy as np
from shapely.geometry import box

from ocgis import constants, CoordinateReferenceSystem
from ocgis.exc import EmptySubsetError
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.variable import Variable, VariableCollection, AbstractContainer
from ocgis.util.environment import ogr
from ocgis.util.helpers import get_reduced_slice

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint


def expand_needed(func):
    def func_wrapper(*args, **kwargs):
        obj = args[0]
        obj.expand()
        return func(*args, **kwargs)

    return func_wrapper


class AbstractSpatialObject(AbstractInterfaceObject):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self._crs = None

        self.crs = kwargs.pop('crs', None)

        super(AbstractSpatialObject, self).__init__(**kwargs)

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        if value is not None:
            assert isinstance(value, CoordinateReferenceSystem)
        self._crs = value

    @property
    def envelope(self):
        return box(*self.extent)

    @property
    def extent(self):
        return self._get_extent_()

    @abstractmethod
    def update_crs(self, to_crs):
        """Update coordinate system in-place."""

    def write_netcdf(self, dataset):
        if self.crs is not None:
            self.crs.write_to_rootgrp(dataset)

    @abstractmethod
    def _get_extent_(self):
        """
        :returns: A tuple with order (minx, miny, maxx, maxy).
        :rtype: tuple
        """


class AbstractSpatialContainer(AbstractContainer, AbstractSpatialObject):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        crs = kwargs.pop('crs', None)
        backref = kwargs.pop('backref', None)

        AbstractContainer.__init__(self, backref=backref)
        AbstractSpatialObject.__init__(self, crs=crs)

    def write_netcdf(self, dataset, **kwargs):
        self._variables.write_netcdf(dataset, **kwargs)
        AbstractSpatialObject.write_netcdf(self, dataset)


class GridXY(AbstractSpatialContainer):
    ndim = 2

    def __init__(self, x, y, crs=None, backref=None):
        x = x.copy()
        y = y.copy()
        x.attrs['axis'] = 'X'
        y.attrs['axis'] = 'Y'

        self._x_name = x.name
        self._y_name = y.name

        self._variables = VariableCollection(variables=(x, y))

        super(GridXY, self).__init__(crs=crs, backref=backref)

        if self.is_vectorized:
            try:
                assert not x.get_mask().any()
                assert not y.get_mask().any()
            except AssertionError:
                msg = 'Vector coordinates may not be masked.'
                raise ValueError(msg)

    def _getitem_main_(self, ret, slc):
        # tdk: order
        """
        :param slc: The slice sequence with indices corresponding to:

         0 --> y-dimension
         1 --> x-dimension

        :type slc: sequence of slice-compatible arguments
        :returns: Sliced grid.
        :rtype: :class:`ocgis.new_interface.grid.GridXY`
        """

        if ret.is_vectorized:
            y = ret.y[slc[0]]
            x = ret.x[slc[1]]
        else:
            y = ret.y[slc[0], slc[1]]
            x = ret.x[slc[0], slc[1]]
        ret._variables = VariableCollection(variables=(x, y))

    def expand(self):
        # tdk: doc

        if self.y.ndim == 1:
            new_x_value, new_y_value = np.meshgrid(self.x.value, self.y.value)
            new_dimensions = self.dimensions

            self.x = self.x.copy()
            self.x.value = None
            self.x.dimensions = new_dimensions
            self.x.value = new_x_value

            self.y = self.y.copy()
            self.y.value = None
            self.y.dimensions = new_dimensions
            self.y.value = new_y_value

            assert self.y.ndim == 2
            assert self.x.ndim == 2

            if self.y.bounds is not None:
                self.y.bounds = None
                self.x.bounds = None
                # tdk: this should leverage the bounds already in place on the vectors
                self.set_extrapolated_bounds()

    @property
    def dimensions(self):
        if self.is_vectorized:
            try:
                ret = (self.y.dimensions[0], self.x.dimensions[0])
            except TypeError:
                # Assume dimensions are none.
                ret = None
        else:
            ret = self.y.dimensions
        return ret

    @property
    def has_bounds(self):
        if self._archetype.bounds is None:
            ret = False
        else:
            ret = True
        return ret

    @property
    def is_vectorized(self):
        if self.y.ndim == 1:
            ret = True
        else:
            ret = False
        return ret

    @property
    def x(self):
        return self._variables[self._x_name]

    @x.setter
    def x(self, value):
        assert isinstance(value, Variable)
        self._variables[self._x_name] = value

    @property
    def y(self):
        return self._variables[self._y_name]

    @y.setter
    def y(self, value):
        assert isinstance(value, Variable)
        self._variables[self._y_name] = value

    @property
    def resolution(self):
        y = self.y
        x = self.x
        if self.is_vectorized:
            to_mean = [y.resolution, x.resolution]
        else:
            resolution_limit = constants.RESOLUTION_LIMIT
            targets = [np.diff(y.value[0:resolution_limit, :], axis=0),
                       np.diff(x.value[:, 0:resolution_limit], axis=1)]
            to_mean = [np.mean(t) for t in targets]
        ret = np.mean(to_mean)
        return ret

    @property
    def shape(self):
        if self.is_vectorized:
            ret = (self.y.shape[0], self.x.shape[0])
        else:
            ret = self._archetype.shape
        return ret

    @property
    def _archetype(self):
        return self.y

    def copy(self):
        ret = copy(self)
        return ret

    def create_dimensions(self, names=None):
        if names is None:
            names = [self.y.name, self.x.name]
        if self.is_vectorized:
            y_name, x_name = names
            self.y.create_dimensions(names=y_name)
            self.x.create_dimensions(names=x_name)
        else:
            self.y.create_dimensions(names=names)
            self.x.create_dimensions(names=names)

    def get_mask(self):
        if self.is_vectorized:
            ret = np.zeros(self.shape, dtype=bool)
        else:
            ret = self._archetype.get_mask()
        return ret

    @expand_needed
    def set_mask(self, value):
        super(GridXY, self).set_mask(value)
        for target in (self.y, self.x):
            target.set_mask(value)

    @expand_needed
    def iter(self, **kwargs):
        name_x = self.x.name
        value_x = self.x.value
        for idx, record in self.y.iter(**kwargs):
            record[name_x] = value_x[idx]
            yield idx, record

    def get_subset_bbox(self, min_col, min_row, max_col, max_row, return_indices=False, closed=True, use_bounds=True):
        assert min_row <= max_row
        assert min_col <= max_col

        if self.y.ndim == 2:
            assert not use_bounds
            r_row = self.y.value.data
            r_col = self.x.value.data
            real_idx_row = np.arange(0, r_row.shape[0])
            real_idx_col = np.arange(0, r_col.shape[1])

            if closed:
                lower_row = r_row > min_row
                upper_row = r_row < max_row
                lower_col = r_col > min_col
                upper_col = r_col < max_col
            else:
                lower_row = r_row >= min_row
                upper_row = r_row <= max_row
                lower_col = r_col >= min_col
                upper_col = r_col <= max_col

            idx_row = np.logical_and(lower_row, upper_row)
            idx_col = np.logical_and(lower_col, upper_col)

            keep_row = np.any(idx_row, axis=1)
            keep_col = np.any(idx_col, axis=0)

            # Slice reduction may fail due to empty bounding box returns. Catch these value errors and re-purpose as
            # subset errors.
            try:
                row_slc = get_reduced_slice(real_idx_row[keep_row])
            except ValueError:
                if real_idx_row[keep_row].shape[0] == 0:
                    raise EmptySubsetError(origin='Y')
                else:
                    raise
            try:
                col_slc = get_reduced_slice(real_idx_col[keep_col])
            except ValueError:
                if real_idx_col[keep_col].shape[0] == 0:
                    raise EmptySubsetError(origin='X')
                else:
                    raise
        else:
            new_row, row_indices = self.y.get_between(min_row, max_row, return_indices=True, closed=closed,
                                                      use_bounds=use_bounds)
            new_col, col_indices = self.x.get_between(min_col, max_col, return_indices=True, closed=closed,
                                                      use_bounds=use_bounds)
            row_slc = get_reduced_slice(row_indices)
            col_slc = get_reduced_slice(col_indices)

        ret = self[row_slc, col_slc]

        if return_indices:
            ret = (ret, (row_slc, col_slc))

        return ret

    def set_extrapolated_bounds(self):
        """
        Extrapolate corners from grid centroids.
        """

        for target in [self.y, self.x]:
            target.set_extrapolated_bounds()

    @expand_needed
    def update_crs(self, to_crs):
        """
        Update the coordinate system in place.

        :param to_crs: The destination coordinate system.
        :type to_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        """

        assert self.crs is not None
        src_sr = self.crs.sr
        to_sr = to_crs.sr

        # Transforming the coordinate system will result in a non-vectorized grid (i.e. cannot be representated as row
        # and column vectors).
        y = self.y
        x = self.x
        value_row = y.value.data.reshape(-1)
        value_col = x.value.data.reshape(-1)
        update_crs_with_geometry_collection(src_sr, to_sr, value_row, value_col)

        if y.bounds is not None:
            corner_row = y.bounds.value.data.reshape(-1)
            corner_col = x.bounds.value.data.reshape(-1)
            update_crs_with_geometry_collection(src_sr, to_sr, corner_row, corner_col)

        self.crs = to_crs

    def _get_extent_(self):
        #tdk: test, doc
        if not self.is_vectorized:
            corners = self.corners
            if corners is not None:
                minx = corners[1].min()
                miny = corners[0].min()
                maxx = corners[1].max()
                maxy = corners[0].max()
            else:
                value = self.value
                minx = value[1, :, :].min()
                miny = value[0, :, :].min()
                maxx = value[1, :, :].max()
                maxy = value[0, :, :].max()
        else:
            row = self.y
            col = self.x
            if row.bounds is None:
                minx = col.value.min()
                miny = row.value.min()
                maxx = col.value.max()
                maxy = row.value.max()
            else:
                minx = col.bounds.value.min()
                miny = row.bounds.value.min()
                maxx = col.bounds.value.max()
                maxy = row.bounds.value.max()
        return minx, miny, maxx, maxy

        # tdk: remove
        #
        # def _get_shape_(self):
        #     if self.is_vectorized:
        #         ret = (self._y.shape[0], self._x.shape[0])
        #     else:
        #         ret = self._y.shape
        #     return ret
        # tdk: remove
        # ret = super(GridXY, self)._get_shape_()
        # if len(ret) == 0:
        #     if self.is_vectorized:
        # # Trim the first dimension. It is always 2 for grids.
        # if len(ret) == 3:
        #     ret = tuple(list(ret)[1:])
        # assert len(ret) == 2
        return ret

        # tdk: remove
        # def _get_value_(self):
        #     if self._value is None:
        #         x = self._x
        #         y = self._y
        #         if self.is_vectorized:
        #             new_x, new_y = np.meshgrid(x.value, y.value)
        #             shp = (2, len(y), len(x))
        #         else:
        #             new_x, new_y = x.value, y.value
        #             shp = [2] + list(new_x.shape)
        #
        #         fill = np.zeros(shp)
        #         fill[0, ...] = new_y
        #         fill[1, ...] = new_x
        #
        #         self._set_value_(fill)
        #
        #     return self._value
        #
        # def _validate_value_(self, value):
        #     if self._dimensions is not None and self._y is not None:
        #         assert value.shape[1:] == self.shape
        #         assert value.shape[0] == 2


# tdk: remove
# def get_dimension_variable(axis_string, gridxy, idx, variable_name):
#     if gridxy.dimensions is not None:
#         dimensions = gridxy.dimensions
#     else:
#         dimensions = None
#     attrs = OrderedDict({'axis': axis_string})
#     # Only write the corners if they have been loaded.
#     if gridxy._corners is not None:
#         if dimensions is not None:
#             dim_ncorners = Dimension(constants.DEFAULT_NAME_CORNERS_DIMENSION, length=4)
#             dimensions_corners = list(dimensions) + [dim_ncorners]
#         else:
#             dimensions_corners = None
#         corners = Variable(name='{}_corners'.format(variable_name), dimensions=dimensions_corners,
#                            value=gridxy.corners[idx, :, :, :])
#         attrs.update({'bounds': corners.name})
#     else:
#         corners = None
#     ret = BoundedVariable(name=variable_name, dimensions=dimensions, attrs=attrs,
#                           bounds=corners, value=gridxy.value[idx, :, :])
#     return ret


def update_crs_with_geometry_collection(src_sr, to_sr, value_row, value_col):
    """
    Update coordinate vectors in place to match the destination coordinate system.

    :param src_sr: The source coordinate system.
    :type src_sr: :class:`osgeo.osr.SpatialReference`
    :param to_sr: The destination coordinate system.
    :type to_sr: :class:`osgeo.osr.SpatialReference`
    :param value_row: Vector of row or Y values.
    :type value_row: :class:`numpy.ndarray`
    :param value_col: Vector of column or X values.
    :type value_col: :class:`numpy.ndarray`
    """

    geomcol = Geometry(wkbGeometryCollection)
    for ii in range(value_row.shape[0]):
        point = Geometry(wkbPoint)
        point.AddPoint(value_col[ii], value_row[ii])
        geomcol.AddGeometry(point)
    geomcol.AssignSpatialReference(src_sr)
    geomcol.TransformTo(to_sr)
    for ii, geom in enumerate(geomcol):
        value_col[ii] = geom.GetX()
        value_row[ii] = geom.GetY()


#tdk: remove
# def update_xy_dimensions(grid):
#     """
#     Update dimensions on "x" and "y" grid components.
#
#     :param grid: The target grid object.
#     :type grid: :class:`ocgis.new_interface.grid.GridXY`
#     """
#     if grid.__y__ is not None:
#         for n, d in zip(grid.name, (grid._y, grid._x)):
#             d.name = n
#         if grid.dimensions is not None:
#             if grid.is_vectorized:
#                 grid._y.dimensions = grid.dimensions[0]
#                 grid._x.dimensions = grid.dimensions[1]
#             else:
#                 grid._y.dimensions = grid.dimensions
#                 grid._x.dimensions = grid.dimensions


