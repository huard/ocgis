import itertools
from copy import copy, deepcopy

import numpy as np
from shapely.geometry import Polygon, Point

from ocgis import constants
from ocgis.exc import GridDeficientError, EmptySubsetError, AllElementsMaskedError
from ocgis.new_interface.geom import GeometryVariable, AbstractSpatialContainer
from ocgis.new_interface.mpi import MPI_RANK, get_optimal_splits, create_nd_slices, MPI_SIZE, MPI_COMM
from ocgis.new_interface.variable import VariableCollection
from ocgis.util.environment import ogr
from ocgis.util.helpers import iter_array, get_trimmed_array_by_mask, get_local_to_global_slices, get_formatted_slice

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

    def __setitem__(self, slc, grid):
        slc = get_formatted_slice(slc, self.ndim)
        if not grid.is_vectorized and self.is_vectorized:
            self.expand()
        if self.is_vectorized:
            self.x[slc[1]] = grid.x
            self.y[slc[0]] = grid.y
        else:
            self.x[slc] = grid.x
            self.y[slc] = grid.y
        if 'point' in grid._variables:
            self.point[slc] = grid.point
        if 'polygon' in grid._variables:
            self.polygon[slc] = grid.polygon

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

    def get_subset_bbox(self, minx, miny, maxx, maxy, return_slice=False, use_bounds=True):
        assert minx <= maxx
        assert miny <= maxy

        ret, slc = grid_get_subset_bbox(self, minx, miny, maxx, maxy, use_bounds=use_bounds)

        if MPI_RANK == 0:
            # Set the mask to update variables only for non-vectorized grids.
            if not self.is_vectorized:
                ret.set_mask(self.get_mask()[slc])

            if return_slice:
                ret = (ret, slc)
        else:
            if return_slice:
                ret = None, None
            else:
                ret = None

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

        return_slice = kwargs.pop('return_slice', False)
        minx, miny, maxx, maxy = args[0].bounds
        ret, slc = self.get_subset_bbox(minx, miny, maxx, maxy, return_slice=True, use_bounds=use_bounds)
        new_mask = ret.get_intersects_masked(*args, **kwargs).get_mask()
        # Barbed and circular geometries may result in rows and or columns being entirely masked. These rows and
        # columns should be trimmed.
        _, adjust = get_trimmed_array_by_mask(new_mask, return_adjustments=True)
        # Use the adjustments to trim the returned data object.
        ret = ret.__getitem__(adjust)
        # Adjust the returned slices.
        if return_slice:
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


def get_arr_intersects_bounds(arr, lower, upper, keep_touches=True):
    assert lower <= upper

    if keep_touches:
        ret = np.logical_and(arr >= lower, arr <= upper)
    else:
        ret = np.logical_and(arr > lower, arr < upper)
    return ret


def grid_get_subset_bbox(grid, bounds_sequence, keep_touches=True, use_bounds=True, mpi_comm=None):
    if mpi_comm is None:
        mpi_comm = MPI_COMM
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        splits = get_optimal_splits(MPI_SIZE, grid.shape)
        slices_grid = create_nd_slices(splits, grid.shape)
        if len(slices_grid) < mpi_size:
            slices_grid = list(slices_grid)
            difference = mpi_size - len(slices_grid)
            slices_grid += ([None] * difference)
    else:
        slices_grid = None

    slc_grid = mpi_comm.scatter(slices_grid, root=0)

    if slc_grid is None:
        slc = None
        grid_sliced = None
    else:
        grid_sliced = grid[slc_grid]
        try:
            slc = grid_get_subset_bbox_slice(grid_sliced, bounds_sequence, use_bounds=use_bounds,
                                             keep_touches=keep_touches)
        except EmptySubsetError:
            slc = None
            grid_sliced = None

    slices_global = mpi_comm.gather(slc_grid, root=0)
    slices_local = mpi_comm.gather(slc, root=0)
    grid_subs = mpi_comm.gather(grid_sliced, root=0)

    if mpi_rank == 0:
        raise_empty_subset = False
        if all([e is None for e in grid_subs]):
            raise_empty_subset = True
        else:
            filled_grid, slc = get_filled_grid_and_slice(grid, grid_subs, slices_global, slices_local)
            ret = filled_grid, slc
    else:
        raise_empty_subset = None
        ret = None, None

    raise_empty_subset = mpi_comm.bcast(raise_empty_subset, root=0)

    if raise_empty_subset:
        raise EmptySubsetError('grid')
    else:
        return ret


def grid_get_subset_bbox_slice(grid, subset, use_bounds=True, keep_touches=True):
    try:
        minx, miny, maxx, maxy = subset.bounds
    except AttributeError:  # Assume a bounds tuple.
        minx, miny, maxx, maxy = subset

    has_bounds, is_vectorized = grid.has_bounds, grid.is_vectorized

    res_x = get_coordinate_boolean_array(grid.x, has_bounds, is_vectorized, keep_touches, maxx, minx, use_bounds)
    res_y = get_coordinate_boolean_array(grid.y, has_bounds, is_vectorized, keep_touches, maxy, miny, use_bounds)

    try:
        if is_vectorized:
            _, x_slice = get_trimmed_array_by_mask(res_x, return_adjustments=True)
            _, y_slice = get_trimmed_array_by_mask(res_y, return_adjustments=True)
            x_slice, y_slice = x_slice[0], y_slice[0]
        else:
            res = np.invert(np.logical_and(res_x.reshape(*grid.shape), res_y.reshape(*grid.shape)))
            _, (y_slice, x_slice) = get_trimmed_array_by_mask(res, return_adjustments=True)
    except AllElementsMaskedError:
        raise EmptySubsetError('grid')

    return y_slice, x_slice


def get_filled_grid_and_slice(grid, grid_subs, slices_global, slices_local):
    as_global = []
    for global_slice, local_slice in zip(slices_global, slices_local):
        if local_slice is not None:
            app = get_local_to_global_slices(global_slice, local_slice)
        else:
            app = None
        as_global.append(app)

    slice_map_template = {'starts': [], 'stops': []}
    slice_map = {}
    keys = ['row', 'col']
    for key in keys:
        slice_map[key] = deepcopy(slice_map_template)
    for idx, sub in enumerate(grid_subs):
        if sub is not None:
            for key, idx_slice in zip(keys, [0, 1]):
                slice_map[key]['starts'].append(as_global[idx][idx_slice].start)
                slice_map[key]['stops'].append(as_global[idx][idx_slice].stop)
    row, col = slice_map['row'], slice_map['col']
    start_row, stop_row = min(row['starts']), max(row['stops'])
    start_col, stop_col = min(col['starts']), max(col['stops'])

    slc_ret = (slice(start_row, stop_row), slice(start_col, stop_col))
    fill_grid = grid[slc_ret]
    return fill_grid, slc_ret


def get_coordinate_boolean_array(grid_target, has_bounds, is_vectorized, keep_touches, max_target, min_target,
                                 use_bounds):
    target_centers = grid_target.value

    if has_bounds and use_bounds:
        if is_vectorized:
            n_bounds = 2
        else:
            n_bounds = 4
        target_bounds = grid_target.bounds.value.reshape(-1, n_bounds)

    res_target_centers = np.array(
        get_arr_intersects_bounds(target_centers, min_target, max_target, keep_touches=keep_touches))

    if has_bounds and use_bounds:
        res_target_bounds = np.array(
            get_arr_intersects_bounds(target_bounds, min_target, max_target, keep_touches=keep_touches))
        res_target_bounds = np.any(res_target_bounds, axis=1)
        res_target_bounds = res_target_bounds.reshape(-1)

    res_target_centers = res_target_centers.reshape(-1)

    if has_bounds and use_bounds:
        res_target = np.logical_or(res_target_centers, res_target_bounds)
    else:
        res_target = res_target_centers

    if is_vectorized:
        res_target = np.invert(res_target)
    return res_target
