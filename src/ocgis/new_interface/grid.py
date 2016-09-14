import itertools
from copy import deepcopy

import numpy as np
from pyproj import Proj, transform
from shapely.geometry import Polygon, Point, box
from shapely.geometry.base import BaseGeometry

from ocgis import constants
from ocgis.exc import GridDeficientError, EmptySubsetError, AllElementsMaskedError
from ocgis.interface.base.crs import CFRotatedPole
from ocgis.new_interface.geom import GeometryVariable, AbstractSpatialContainer
from ocgis.new_interface.mpi import get_optimal_splits, create_nd_slices, MPI_SIZE, MPI_COMM
from ocgis.new_interface.ocgis_logging import log, log_entry_exit
from ocgis.new_interface.variable import VariableCollection, get_dslice, get_dimension_lengths
from ocgis.util.environment import ogr
from ocgis.util.helpers import iter_array, get_trimmed_array_by_mask, get_formatted_slice

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint

_NAMES_2D = ['ocgis_yc', 'ocgis_xc']


class GridXY(AbstractSpatialContainer):
    ndim = 2

    def __init__(self, x, y, abstraction='auto', crs=None, is_vectorized='auto', parent=None):
        if x.dimensions is None or y.dimensions is None:
            raise ValueError('Grid variables must have dimensions.')

        self._abstraction = None
        self._is_vectorized = None

        self.abstraction = abstraction

        x.attrs['axis'] = 'X'
        y.attrs['axis'] = 'Y'

        self._x_name = x.name
        self._y_name = y.name

        self._point_name = 'ocgis_point'
        self._polygon_name = 'ocgis_polygon'

        new_variables = [x, y]
        if parent is None:
            parent = VariableCollection(variables=new_variables)
        else:
            for var in new_variables:
                parent.add_variable(var, force=True)

        super(GridXY, self).__init__(crs=crs, parent=parent)

        self.is_vectorized = is_vectorized

    def __getitem__(self, slc):
        """
        :param slc: The slice sequence with indices corresponding to:

         0 --> y-dimension
         1 --> x-dimension

        :type slc: sequence of slice-compatible arguments
        :returns: Sliced grid.
        :rtype: :class:`ocgis.new_interface.grid.GridXY`
        """
        slc = get_dslice(self.dimensions, slc)
        ret = self.copy()
        new_parent = ret.parent[slc]
        ret.parent = new_parent
        return ret

    def __setitem__(self, slc, grid):
        slc = get_formatted_slice(slc, self.ndim)

        if self._point_name in grid.parent:
            self.point[slc] = grid.point
        if self._polygon_name in grid.parent:
            self.polygon[slc] = grid.polygon

        self.x[slc] = grid.x
        self.y[slc] = grid.y

    def get_member_variables(self, include_bounds=False):
        targets = [self._x_name, self._y_name, self._point_name, self._polygon_name]
        ret = []
        for target in targets:
            try:
                var = self.parent[target]
            except KeyError:
                pass
            else:
                ret.append(var)
                if include_bounds and var.has_bounds:
                    ret.append(var.bounds)
        return ret

    @property
    def has_initialized_point(self):
        if self._point_name in self.parent:
            return True
        else:
            return False

    @property
    def has_initialized_polygon(self):
        if self._polygon_name in self.parent:
            return True
        else:
            return False

    @property
    def has_initialized_abstraction_geometry(self):
        if self.abstraction == 'point':
            return self.has_initialized_point
        elif self.abstraction == 'polygon':
            return self.has_initialized_polygon
        else:
            raise NotImplementedError(self.abstraction)

    @property
    def is_vectorized(self):
        return self._is_vectorized

    @is_vectorized.setter
    def is_vectorized(self, value):
        # Select vectorization state from the dimension count on the x/y coordinate variables.
        if value == 'auto':
            if self._archetype.ndim == 1:
                value = True
            else:
                value = False
        self._is_vectorized = value
        # Also update the parent if it is field-like to ensure this state passes between grid instances.
        if self.parent is not None:
            if hasattr(self.parent, 'grid_is_vectorized'):
                self.parent.grid_is_vectorized = value

    @property
    def dimensions(self):
        ret = self._archetype.dimensions
        if len(ret) == 1:
            ret = (self.parent[self._y_name].dimensions[0], self.parent[self._x_name].dimensions[0])
        return ret

    # tdk: REMOVE
    @property
    def dist(self):
        raise NotImplementedError

    @property
    def has_bounds(self):
        return self._archetype.has_bounds

    @property
    def point(self):
        try:
            ret = self.parent[self._point_name]
        except KeyError:
            ret = grid_set_geometry_variable_on_parent(get_point_geometry_array, self, self._point_name)
        return ret

    @point.setter
    def point(self, value):
        if value is not None:
            self.parent[value.name] = value
            self._point_name = value.name
        else:
            self.parent.pop(self._point_name, None)

    @property
    def polygon(self):
        try:
            ret = self.parent[self._polygon_name]
        except KeyError:
            if not self.has_bounds:
                ret = None
            else:
                ret = grid_set_geometry_variable_on_parent(get_polygon_geometry_array, self, self._polygon_name)
        return ret

    @polygon.setter
    def polygon(self, value):
        if value is not None:
            self.parent[value.name] = value
            self._polygon_name = value.name
        else:
            self.parent.pop(self._polygon_name, None)

    @property
    def x(self):
        ret = self.parent[self._x_name]
        if ret.ndim == 1:
            expand_grid(self)
        return ret

    @x.setter
    def x(self, value):
        self.parent[self._x_name] = value

    @property
    def y(self):
        ret = self.parent[self._y_name]
        if ret.ndim == 1:
            expand_grid(self)
        return ret

    @y.setter
    def y(self, value):
        self.parent[self._y_name] = value

    @property
    def resolution(self):
        y = self.y
        x = self.x
        resolution_limit = constants.RESOLUTION_LIMIT
        targets = [np.abs(np.diff(np.abs(y.value[0:resolution_limit, :]), axis=0)),
                   np.abs(np.diff(np.abs(x.value[:, 0:resolution_limit]), axis=1))]
        to_mean = [np.mean(t) for t in targets]
        ret = np.mean(to_mean)
        return ret

    @property
    def shape(self):
        return get_dimension_lengths(self.dimensions)

    @property
    def value_stacked(self):
        y = self.y.value
        x = self.x.value
        fill = np.zeros([2] + list(y.shape))
        fill[0, :, :] = y
        fill[1, :, :] = x
        return fill

    @property
    def masked_value_stacked(self):
        y = self.y.masked_value
        x = self.x.masked_value
        fill = np.ma.zeros([2] + list(y.shape))
        fill[0, :, :] = y
        fill[1, :, :] = x
        return fill

    @property
    def _archetype(self):
        return self.parent[self._y_name]

    def allocate_geometry_variable(self, target):
        targets = {'point': {'name': self._point_name, 'func': get_point_geometry_array},
                   'polygon': {'name': self._polygon_name, 'func': get_polygon_geometry_array}}
        grid_set_geometry_variable_on_parent(targets[target]['func'], self, targets[target]['name'], alloc_only=True)

    def create_dimensions(self, names=None):
        if names is None:
            names = [self.y.name, self.x.name]
        self.y.create_dimensions(names=names)
        self.x.create_dimensions(names=names)

    def get_mask(self):
        return self.y.get_mask()

    def set_mask(self, value, cascade=False):
        for v in self.get_member_variables():
            v.set_mask(value)
        if cascade:
            grid_set_mask_cascade(self)

    def iter(self, **kwargs):
        name_x = self.x.name
        value_x = self.x.value
        for idx, record in self.y.iter(**kwargs):
            record[name_x] = value_x[idx]
            yield idx, record

    def get_intersects(self, *args, **kwargs):
        args = list(args)
        args.insert(0, 'intersects')
        return self.get_spatial_operation(*args, **kwargs)

    def get_intersection(self, *args, **kwargs):
        args = list(args)
        args.insert(0, 'intersection')
        return self.get_spatial_operation(*args, **kwargs)

    def get_spatial_operation(self, *args, **kwargs):
        spatial_op, original_subset_target = args
        args = (original_subset_target,)

        use_bounds = kwargs.pop('use_bounds', 'auto')
        return_slice = kwargs.get('return_slice', False)
        original_mask = kwargs.get('original_mask')

        if use_bounds == 'auto':
            if self.abstraction == 'polygon':
                use_bounds = True
            else:
                use_bounds = False

        if original_mask is None:
            if not isinstance(original_subset_target, BaseGeometry):
                subset_target = box(*original_subset_target)
            else:
                subset_target = original_subset_target
            subset_target = subset_target.buffer(1.25 * self.resolution)
            original_mask = get_hint_mask_from_geometry_bounds(self, subset_target)
            original_mask = np.logical_or(self.get_mask(), original_mask)
        kwargs['original_mask'] = original_mask

        if use_bounds:
            if spatial_op == 'intersects':
                spatial_op_return = self.polygon.get_intersects(*args, **kwargs)
            else:
                spatial_op_return = self.polygon.get_intersection(*args, **kwargs)
        else:
            if spatial_op == 'intersects':
                spatial_op_return = self.point.get_intersects(*args, **kwargs)
            else:
                spatial_op_return = self.point.get_intersection(*args, **kwargs)

        if return_slice:
            gvar, the_return_slice = spatial_op_return
        else:
            gvar = spatial_op_return

        ret = self.copy()
        ret.parent = gvar.parent

        if return_slice:
            ret = (ret, the_return_slice)

        return ret

    def set_extrapolated_bounds(self, name_x_variable, name_y_variable, name_dimension):
        """
        Extrapolate corners from grid centroids.
        """
        self.x.set_extrapolated_bounds(name_x_variable, name_dimension)
        self.y.set_extrapolated_bounds(name_y_variable, name_dimension)
        self.parent = self.y.parent

    def update_crs(self, to_crs):
        """
        Update the coordinate system in place.

        :param to_crs: The destination coordinate system.
        :type to_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        """

        super(GridXY, self).update_crs(to_crs)

        if isinstance(self.crs, CFRotatedPole):
            self.crs.update_with_rotated_pole_transformation(self, inverse=False)
        elif isinstance(to_crs, CFRotatedPole):
            to_crs.update_with_rotated_pole_transformation(self, inverse=True)
        else:
            src_proj4 = self.crs.proj4
            dst_proj4 = to_crs.proj4

            src_proj4 = Proj(src_proj4)
            dst_proj4 = Proj(dst_proj4)

            y = self.y
            x = self.x

            value_row = self.y.value.reshape(-1)
            value_col = self.x.value.reshape(-1)

            tvalue_col, tvalue_row = transform(src_proj4, dst_proj4, value_col, value_row)

            self.x.value = tvalue_col.reshape(self.shape)
            self.y.value = tvalue_row.reshape(self.shape)

            if self.has_bounds:
                corner_row = y.bounds.value.reshape(-1)
                corner_col = x.bounds.value.reshape(-1)
                # update_crs_with_geometry_collection(src_sr, to_sr, corner_row, corner_col)
                tvalue_col, tvalue_row = transform(src_proj4, dst_proj4, corner_col, corner_row)
                y.bounds.value = tvalue_row.reshape(y.bounds.shape)
                x.bounds.value = tvalue_col.reshape(x.bounds.shape)

        # import ipdb;
        # ipdb.set_trace()
        # src_sr = self.crs.sr
        # to_sr = to_crs.sr
        #
        # # Transforming the coordinate system will result in a non-vectorized grid (i.e. cannot be representated as row
        # # and column vectors).
        # y = self.y
        # x = self.x
        # value_row = y.value.reshape(-1)
        # value_col = x.value.reshape(-1)
        # update_crs_with_geometry_collection(src_sr, to_sr, value_row, value_col)
        # y.value = value_row.reshape(self.shape)
        # x.value = value_col.reshape(self.shape)
        #
        # if self.has_bounds:
        #     corner_row = y.bounds.value.reshape(-1)
        #     corner_col = x.bounds.value.reshape(-1)
        #     update_crs_with_geometry_collection(src_sr, to_sr, corner_row, corner_col)
        #     y.bounds.value = corner_row.reshape(y.bounds.shape)
        #     x.bounds.value = corner_col.reshape(x.bounds.shape)

        self.crs = to_crs

        # Regenerate geometries.
        self.point = None
        self.polygon = None

        # Rotated pole transformations maintain grid vectorizations.
        if not isinstance(self.crs, CFRotatedPole) or not isinstance(to_crs, CFRotatedPole):
            self.is_vectorized = False

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
            row = self.y[:, 0]
            col = self.x[0, :]
            if not self.has_bounds:
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

    def get_nearest(self, *args, **kwargs):
        ret = self.copy()
        _, slc = self.abstraction_geometry.get_nearest(*args, **kwargs)
        ret = ret.__getitem__(slc)
        return ret

    def get_spatial_index(self, *args, **kwargs):
        return self.abstraction_geometry.get_spatial_index(*args, **kwargs)

    def iter_records(self, *args, **kwargs):
        return self.abstraction_geometry.iter_records(self, *args, **kwargs)

    def remove_bounds(self):
        self.x.bounds = None
        self.y.bounds = None

    def write_fiona(self, *args, **kwargs):
        return self.abstraction_geometry.write_fiona(*args, **kwargs)

    def write(self, *args, **kwargs):
        from ocgis.api.request.driver.nc import DriverNetcdf
        driver = kwargs.pop('driver', DriverNetcdf)
        args = list(args)
        args.insert(0, self)
        driver.write_gridxy(*args, **kwargs)


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


def get_geometry_fill(shape, mask):
    return np.ma.array(np.zeros(shape), mask=mask, dtype=object)


def get_polygon_geometry_array(grid, fill):
    r_data = fill.data

    if grid.has_bounds:
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


def get_point_geometry_array(grid, fill):
    # Create geometries for all the underlying coordinates regardless if the data is masked.
    x_data = grid.x.value
    y_data = grid.y.value

    r_data = fill.data
    for idx_row, idx_col in iter_array(y_data, use_mask=False):
        y = y_data[idx_row, idx_col]
        x = x_data[idx_row, idx_col]
        pt = Point(x, y)
        r_data[idx_row, idx_col] = pt
    return fill


def get_geometry_variable(func, grid, **kwargs):
    alloc_only = kwargs.pop('alloc_only', False)
    value = get_geometry_fill(grid.shape, grid.get_mask())
    if not alloc_only:
        value = func(grid, value)
    kwargs['value'] = value
    kwargs['crs'] = grid.crs
    kwargs['parent'] = grid.parent
    kwargs['dimensions'] = grid.dimensions
    return GeometryVariable(**kwargs)


def get_arr_intersects_bounds(arr, lower, upper, keep_touches=True):
    assert lower <= upper

    if keep_touches:
        arr_lower = arr >= lower
        arr_upper = arr <= upper
    else:
        arr_lower = arr > lower
        arr_upper = arr < upper

    ret = np.logical_and(arr_lower, arr_upper)
    return ret


def grid_get_intersects(grid, subset, keep_touches=True, use_bounds=True, mpi_comm=None,
                        use_spatial_index=True):
    # Get local communicator attributes.
    if mpi_comm is None:
        mpi_comm = MPI_COMM
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # The first subset uses bounds sequence. Extract the sequence from the subset geometry if required. Set a flag to
    # indicate if we are using a geometry for the subset.
    try:
        bounds_sequence = subset.bounds
        with_geometry = True
    except AttributeError:  # Assume a bounds tuple.
        bounds_sequence = subset
        subset = box(*bounds_sequence)
        with_geometry = False

    # If we are dealing with bounds, always trigger spatial operations.
    if use_bounds and grid.has_bounds:
        # Buffer the sequence to ensure centroids for cells with bounds are captured. Bounds always require spatial
        # operations.
        buffered = box(*bounds_sequence).buffer(1.5 * grid.resolution)
        bounds_sequence = buffered.bounds
        with_geometry = True

    if mpi_rank == 0:
        log.info('split')
        splits = get_optimal_splits(MPI_SIZE, grid.shape)
        slices_grid = create_nd_slices(splits, grid.shape)
        if len(slices_grid) < mpi_size:
            slices_grid = list(slices_grid)
            difference = mpi_size - len(slices_grid)
            slices_grid += ([None] * difference)
    else:
        slices_grid = None

    log.info('apply')
    slc_grid = mpi_comm.scatter(slices_grid, root=0)

    grid_sliced = None
    if slc_grid is not None:
        grid_sliced = grid[slc_grid]
        try:
            grid_update_mask(grid_sliced, bounds_sequence, keep_touches=keep_touches)
        except EmptySubsetError:
            grid_sliced = None
        else:
            if with_geometry:
                try:
                    grid_sliced = grid_sliced.get_intersects_masked(subset, keep_touches=keep_touches,
                                                                    use_spatial_index=use_spatial_index)
                except EmptySubsetError:
                    grid_sliced = None

    slices_global = mpi_comm.gather(slc_grid, root=0)
    grid_subs = mpi_comm.gather(grid_sliced, root=0)

    if mpi_rank == 0:
        log.info('combine')
        raise_empty_subset = False
        if all([e is None for e in grid_subs]):
            raise_empty_subset = True
        else:
            filled_grid, slc = get_filled_grid_and_slice(grid, grid_subs, slices_global)
            ret = filled_grid, slc
    else:
        raise_empty_subset = None
        ret = None, None

    raise_empty_subset = mpi_comm.bcast(raise_empty_subset, root=0)

    if raise_empty_subset:
        raise EmptySubsetError('grid')
    else:
        return ret


def grid_update_mask(grid, bounds_sequence, keep_touches=True):
    minx, miny, maxx, maxy = bounds_sequence

    res_x = get_coordinate_boolean_array(grid.x, keep_touches, maxx, minx)
    res_y = get_coordinate_boolean_array(grid.y, keep_touches, maxy, miny)

    try:
        res = np.invert(np.logical_and(res_x.reshape(*grid.shape), res_y.reshape(*grid.shape)))
        if np.all(res):
            raise AllElementsMaskedError
        grid.set_mask(res)
    except AllElementsMaskedError:
        raise EmptySubsetError('grid')


def remove_nones(target):
    ret = filter(lambda x: x is not None, target)
    return ret


@log_entry_exit
def get_filled_grid_and_slice(grid, grid_subs, slices_global):
    slice_map_template = {'starts': [], 'stops': []}
    slice_map = {}
    keys = ['row', 'col']
    for key in keys:
        slice_map[key] = deepcopy(slice_map_template)
    for idx, sub in enumerate(grid_subs):
        if sub is not None:
            for key, idx_slice in zip(keys, [0, 1]):
                slice_map[key]['starts'].append(slices_global[idx][idx_slice].start)
                slice_map[key]['stops'].append(slices_global[idx][idx_slice].stop)
    row, col = slice_map['row'], slice_map['col']
    start_row, stop_row = min(row['starts']), max(row['stops'])
    start_col, stop_col = min(col['starts']), max(col['stops'])

    slc_remaining = (slice(start_row, stop_row), slice(start_col, stop_col))
    fill_grid = grid[slc_remaining]

    as_local = []
    for target in slices_global:
        if target is None:
            continue
        to_append = []
        for target_remaining, target_element in zip(slc_remaining, target):
            new_start = target_element.start - target_remaining.start
            new_stop = target_element.stop - target_remaining.start
            to_append.append(slice(new_start, new_stop))
        as_local.append(to_append)

    # The grid should be entirely masked. Section grids are subsetted to the extent of the local overlap.
    if grid.is_vectorized:
        for target in [fill_grid.x, fill_grid.y]:
            new_mask = target.get_mask()
            new_mask.fill(True)
            target.set_mask(new_mask)
    else:
        new_mask = fill_grid.get_mask()
        new_mask.fill(True)
        fill_grid.set_mask(new_mask)

    # Fill the parent grid with its subsets. Allocate the abstraction geometry to avoid loading all the polygons.
    build = True
    for idx, gs in enumerate(grid_subs):
        if gs is not None:
            if build and gs.has_allocated_abstraction_geometry:
                if not fill_grid.has_allocated_abstraction_geometry:
                    fill_grid.allocate_geometry_variable(fill_grid.abstraction)
                build = False
            fill_grid[as_local[idx]] = gs

    _, slc_ret = get_trimmed_array_by_mask(fill_grid.get_mask(), return_adjustments=True)

    fill_grid = fill_grid[slc_ret]

    new_slc_ret = [None] * len(slc_ret)
    for idx, sr in enumerate(slc_ret):
        target_slice_remaining = slc_remaining[idx]
        new_sr = slice(sr.start + target_slice_remaining.start, sr.stop + target_slice_remaining.start)
        new_slc_ret[idx] = new_sr
    slc_ret = tuple(new_slc_ret)

    return fill_grid, slc_ret


def get_coordinate_boolean_array(grid_target, keep_touches, max_target, min_target):
    target_centers = grid_target.value

    res_target = np.array(get_arr_intersects_bounds(target_centers, min_target, max_target, keep_touches=keep_touches))
    res_target = res_target.reshape(-1)

    return res_target


def get_hint_mask_from_geometry_bounds(grid, geometry):
    minx, miny, maxx, maxy = geometry.bounds
    grid_x = grid.x.value
    grid_y = grid.y.value

    select_x = np.logical_and(grid_x >= minx, grid_x <= maxx)
    select_y = np.logical_and(grid_y >= miny, grid_y <= maxy)
    select = np.logical_and(select_x, select_y)

    return np.invert(select)


def grid_set_geometry_variable_on_parent(func, grid, name, alloc_only=False):
    dimensions = [d.name for d in grid.dimensions]
    ret = get_geometry_variable(func, grid, name=name, attrs={'axis': 'geom'}, alloc_only=alloc_only,
                                dimensions=dimensions)
    # ret.create_dimensions(names=[d.name for d in grid.dimensions])
    # grid.parent.add_variable(ret)
    return ret


def grid_set_mask_cascade(grid):
    members = grid.get_member_variables(include_bounds=True)
    members = [m.name for m in members]
    grid.parent.set_mask(grid._archetype, exclude=members)


def expand_grid(grid):
    # y = grid.y
    y = grid.parent[grid._y_name]
    # x = grid.x
    x = grid.parent[grid._x_name]
    if y.ndim == 1:
        new_x_value, new_y_value = np.meshgrid(x.value, y.value)
        new_dimensions = [y.dimensions[0], x.dimensions[0]]

        x.value = None
        x._dimensions = None
        x.value = new_x_value

        y.value = None
        y._dimensions = None
        y.value = new_y_value

        assert y.ndim == 2
        assert x.ndim == 2

        if y.bounds is not None:
            name_y = y.bounds.name
            name_x = x.bounds.name
            name_dimension = y.bounds.dimensions[1].name
            y.bounds = None
            x.bounds = None
            y.dimensions = new_dimensions
            x.dimensions = new_dimensions
            # tdk: this should leverage the bounds already in place on the vectors
            grid.set_extrapolated_bounds(name_x, name_y, name_dimension)
        else:
            y.dimensions = new_dimensions
            x.dimensions = new_dimensions
