import itertools
from copy import copy

import numpy as np
from shapely.geometry import Polygon, Point, box

from ocgis import constants
from ocgis.exc import EmptySubsetError, GridDeficientError
from ocgis.new_interface.geom import GeometryVariable, AbstractSpatialContainer
from ocgis.new_interface.variable import VariableCollection
from ocgis.util.environment import ogr
from ocgis.util.helpers import get_reduced_slice, iter_array, get_trimmed_array_by_mask, get_local_to_global_slices

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint


def expand_needed(func):
    def func_wrapper(*args, **kwargs):
        obj = args[0]
        obj.expand()
        return func(*args, **kwargs)

    return func_wrapper


class GridXY(AbstractSpatialContainer):
    ndim = 2

    def __init__(self, x, y, point=None, polygon=None, abstraction='auto', crs=None, backref=None):
        self._abstraction = None

        self.abstraction = abstraction

        x = x.copy()
        y = y.copy()
        x.attrs['axis'] = 'X'
        y.attrs['axis'] = 'Y'

        self._x_name = x.name
        self._y_name = y.name

        new_variables = [x, y]
        for target, attr_name in zip([point, polygon], ['point', 'polygon']):
            if target is not None:
                target = target.copy()
                target.attrs['axis'] = 'geom'
                new_variables.append(target)

        self._variables = VariableCollection(variables=new_variables)

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
        new_variables = [x, y]

        if 'point' in ret._variables:
            point = ret.point[slc[0], slc[1]]
            new_variables.append(point)
        if 'polygon' in ret._variables:
            polygon = ret.polygon[slc[0], slc[1]]
            new_variables.append(polygon)

        ret._variables = VariableCollection(variables=new_variables)

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
    def point(self):
        try:
            ret = self._variables['point']
        except KeyError:
            ret = get_geometry_variable(get_point_geometry_array, self, name='point', attrs={'axis': 'geom'})
            self._variables.add_variable(ret)
        return ret

    @point.setter
    def point(self, value):
        if value is not None:
            self._variables[value.name] = value

    @property
    def polygon(self):
        try:
            ret = self._variables['polygon']
        except KeyError:
            if not self.has_bounds:
                ret = None
            else:
                ret = get_geometry_variable(get_polygon_geometry_array, self, name='polygon', attrs={'axis': 'geom'})
                self._variables.add_variable(ret)
        return ret

    @polygon.setter
    def polygon(self, value):
        if value is not None:
            self._variables[value.name] = value

    @property
    def x(self):
        return self._variables[self._x_name]

    @x.setter
    def x(self, value):
        self._variables[self._x_name] = value

    @property
    def y(self):
        return self._variables[self._y_name]

    @y.setter
    def y(self, value):
        self._variables[self._y_name] = value

    @property
    def resolution(self):
        y = self.y
        x = self.x
        if self.is_vectorized:
            to_mean = [y.resolution, x.resolution]
        else:
            resolution_limit = constants.RESOLUTION_LIMIT
            targets = [np.abs(np.diff(np.abs(y.value[0:resolution_limit, :]), axis=0)),
                       np.abs(np.diff(np.abs(x.value[:, 0:resolution_limit]), axis=1))]
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
    @expand_needed
    def value_stacked(self):
        y = self.y.value
        x = self.x.value
        fill = np.zeros([2] + list(y.shape))
        fill[0, :, :] = y
        fill[1, :, :] = x
        return fill

    @property
    def _archetype(self):
        return self.y

    def copy(self):
        ret = copy(self)
        ret._variables = ret._variables.copy()
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
        # The grid uses its variables for mask management. Remove the mask reference so get_mask returns from variables.
        self._mask = None
        for target in self._variables.values():
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

        if not self.is_vectorized:
            r_row = self.y.value
            r_col = self.x.value
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

            if self.has_bounds:
                row_start, row_stop = row_slc.start, row_slc.stop
                col_start, col_stop = col_slc.start, col_slc.stop
                subset_polygon = box(min_col, min_row, max_col, max_row)

                add_row_top = False
                if row_start != 0:
                    if col_start != 0:
                        col_select_start = col_start - 1
                    else:
                        col_select_start = 0
                    col_select_stop = col_stop + 1
                    row_select = row_start - 1
                    for col_select in range(col_select_start, col_select_stop):
                        corners_x = self.x.bounds.value[row_select, col_select].flatten()
                        corners_y = self.y.bounds.value[row_select, col_select].flatten()
                        if is_subset_polygon_in_corners(corners_x, corners_y, subset_polygon, closed=closed):
                            add_row_top = True
                            break

                add_row_bottom = False
                if row_stop != (self.x.shape[0] - 1):
                    if col_start != 0:
                        col_select_start = col_start - 1
                    else:
                        col_select_start = 0
                    col_select_stop = col_stop + 1
                    row_select = row_stop
                    for col_select in range(col_select_start, col_select_stop):
                        corners_x = self.x.bounds.value[row_select, col_select].flatten()
                        corners_y = self.y.bounds.value[row_select, col_select].flatten()
                        if is_subset_polygon_in_corners(corners_x, corners_y, subset_polygon, closed=closed):
                            add_row_bottom = True
                            break

                add_col_left = False
                if col_start != 0:
                    if row_start != 0:
                        row_select_start = row_start - 1
                    else:
                        row_select_start = row_start
                    row_select_stop = row_stop + 1
                    col_select = col_start
                    for row_select in range(row_select_start, row_select_stop):
                        corners_x = self.x.bounds.value[row_select, col_select].flatten()
                        corners_y = self.y.bounds.value[row_select, col_select].flatten()
                        if is_subset_polygon_in_corners(corners_x, corners_y, subset_polygon, closed=closed):
                            add_col_left = True
                            break

                add_col_right = False
                if col_stop != self.y.shape[1]:
                    if row_start != 0:
                        row_select_start = row_start - 1
                    else:
                        row_select_start = row_start
                    row_select_stop = row_stop + 1
                    col_select = col_stop
                    for row_select in range(row_select_start, row_select_stop):
                        corners_x = self.x.bounds.value[row_select, col_select].flatten()
                        corners_y = self.y.bounds.value[row_select, col_select].flatten()
                        if is_subset_polygon_in_corners(corners_x, corners_y, subset_polygon, closed=closed):
                            add_col_right = True
                            break

                if add_row_top:
                    row_slc_start = row_slc.start - 1
                else:
                    row_slc_start = row_slc.start
                if add_row_bottom:
                    row_slc_stop = row_slc.stop + 1
                else:
                    row_slc_stop = row_slc.stop
                row_slc = slice(row_slc_start, row_slc_stop)

                if add_col_left:
                    col_slc_start = col_slc.start - 1
                else:
                    col_slc_start = col_slc.start
                if add_col_right:
                    col_slc_stop = col_slc.stop + 1
                else:
                    col_slc_stop = col_slc.stop
                col_slc = slice(col_slc_start, col_slc_stop)

        else:
            new_row, row_indices = self.y.get_between(min_row, max_row, return_indices=True, closed=closed,
                                                      use_bounds=use_bounds)
            new_col, col_indices = self.x.get_between(min_col, max_col, return_indices=True, closed=closed,
                                                      use_bounds=use_bounds)
            row_slc = get_reduced_slice(row_indices)
            col_slc = get_reduced_slice(col_indices)

        ret = self[row_slc, col_slc]
        # Set the mask to update variables only for non-vectorized grids.
        if not self.is_vectorized:
            ret.set_mask(self.get_mask()[row_slc, col_slc])

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
        value_row = y.value.reshape(-1)
        value_col = x.value.reshape(-1)
        update_crs_with_geometry_collection(src_sr, to_sr, value_row, value_col)

        if y.bounds is not None:
            corner_row = y.bounds.value.reshape(-1)
            corner_col = x.bounds.value.reshape(-1)
            update_crs_with_geometry_collection(src_sr, to_sr, corner_row, corner_col)

        self.crs = to_crs

    def _get_extent_(self):
        #tdk: test, doc
        if not self.is_vectorized:
            if self.has_bounds:
                x_bounds = self.x.bounds.value
                y_bounds = self.y.bounds.value
                minx = x_bounds.min()
                miny = y_bounds.min()
                maxx = x_bounds.max()
                maxy = y_bounds.max()
            else:
                x_value = self.x.value
                y_value = self.y.value
                minx = x_value.min()
                miny = y_value.min()
                maxx = x_value.max()
                maxy = y_value.max()
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

    @property
    def abstraction(self):
        if self._abstraction == 'auto':
            if self.has_bounds:
                ret = 'polygon'
            else:
                ret = 'point'
        else:
            ret = self._abstraction
        return ret

    @abstraction.setter
    def abstraction(self, abstraction):
        self._abstraction = abstraction

    @property
    def abstraction_geometry(self):
        return getattr(self, self.abstraction)

    def get_intersects(self, *args, **kwargs):
        if self.abstraction == 'polygon':
            use_bounds = True
        else:
            use_bounds = False

        return_indices = kwargs.pop('return_indices', False)
        minx, miny, maxx, maxy = args[0].bounds
        ret, slc = self.get_subset_bbox(minx, miny, maxx, maxy, return_indices=True, use_bounds=use_bounds)
        new_mask = ret.get_intersects_masked(*args, **kwargs).get_mask()
        # Barbed and circular geometries may result in rows and or columns being entirely masked. These rows and
        # columns should be trimmed.
        _, adjust = get_trimmed_array_by_mask(new_mask, return_adjustments=True)
        # Use the adjustments to trim the returned data object.
        ret = ret.__getitem__(adjust)
        # Adjust the returned slices.
        if return_indices:
            ret_slc = get_local_to_global_slices(slc, adjust)
            ret = (ret, ret_slc)
        return ret

    def get_intersects_masked(self, *args, **kwargs):
        ret = self.copy()
        new_mask = self.abstraction_geometry.get_intersects_masked(*args, **kwargs).get_mask()
        # tdk: this unecessarily sets the mask of the abstraction geometry twice.
        ret.set_mask(new_mask)
        return ret

    def get_nearest(self, *args, **kwargs):
        ret = self.copy()
        _, slc = self.abstraction_geometry.get_nearest(*args, **kwargs)
        ret = ret.__getitem__(slc)
        return ret

    def get_spatial_index(self, *args, **kwargs):
        return self.abstraction_geometry.get_spatial_index(*args, **kwargs)

    def iter_records(self, *args, **kwargs):
        return self.abstraction_geometry.iter_records(self, *args, **kwargs)

    def write_fiona(self, *args, **kwargs):
        return self.abstraction_geometry.write_fiona(*args, **kwargs)


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


def get_geometry_fill(shape, mask=False):
    return np.ma.array(np.zeros(shape), mask=mask, dtype=object)


def get_polygon_geometry_array(grid):
    fill = get_geometry_fill(grid.shape)
    r_data = fill.data

    if grid.is_vectorized and grid.has_bounds:
        ref_row_bounds = grid.y.bounds.value
        ref_col_bounds = grid.x.bounds.value
        for idx_row, idx_col in itertools.product(range(ref_row_bounds.shape[0]), range(ref_col_bounds.shape[0])):
            row_min, row_max = ref_row_bounds[idx_row, :].min(), ref_row_bounds[idx_row, :].max()
            col_min, col_max = ref_col_bounds[idx_col, :].min(), ref_col_bounds[idx_col, :].max()
            r_data[idx_row, idx_col] = Polygon([(col_min, row_min), (col_min, row_max),
                                                (col_max, row_max), (col_max, row_min)])
    elif not grid.is_vectorized and grid.has_bounds:
        # We want geometries for everything even if masked.
        x_corners = grid.x.bounds.value
        y_corners = grid.y.bounds.value
        # tdk: we should be able to avoid the creation of this corners array
        corners = np.vstack((y_corners, x_corners))
        corners = corners.reshape([2] + list(x_corners.shape))
        range_row = range(grid.shape[0])
        range_col = range(grid.shape[1])

        for row, col in itertools.product(range_row, range_col):
            current_corner = corners[:, row, col]
            coords = np.hstack((current_corner[1, :].reshape(-1, 1),
                                current_corner[0, :].reshape(-1, 1)))
            polygon = Polygon(coords)
            r_data[row, col] = polygon
    else:
        msg = 'A grid must have bounds/corners to construct polygons. Consider using "set_extrapolated_bounds".'
        raise GridDeficientError(msg)

    return fill


def get_point_geometry_array(grid):
    fill = get_geometry_fill(grid.shape)

    # Create geometries for all the underlying coordinates regardless if the data is masked.
    x_data = grid.x.value
    y_data = grid.y.value

    r_data = fill.data
    if grid.is_vectorized:
        for idx_row in range(y_data.shape[0]):
            for idx_col in range(x_data.shape[0]):
                pt = Point(x_data[idx_col], y_data[idx_row])
                r_data[idx_row, idx_col] = pt
    else:
        for idx_row, idx_col in iter_array(y_data, use_mask=False):
            y = y_data[idx_row, idx_col]
            x = x_data[idx_row, idx_col]
            pt = Point(x, y)
            r_data[idx_row, idx_col] = pt
    return fill


def get_geometry_variable(func, grid, **kwargs):
    kwargs = kwargs.copy()
    value = func(grid)
    kwargs['value'] = value
    return GeometryVariable(**kwargs)


def is_subset_polygon_in_corners(corners_x, corners_y, subset_polygon, closed=True):
    coordinates_polygon = []
    for cx, cy in zip(corners_x.flat, corners_y.flat):
        coordinates_polygon.append([cx, cy])

    polygon = Polygon(coordinates_polygon)
    touches = subset_polygon.touches(polygon)
    intersects = subset_polygon.intersects(polygon)

    if intersects:
        if closed and touches:
            ret = False
        else:
            ret = True
    else:
        ret = False

    return ret
