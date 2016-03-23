import itertools
from copy import deepcopy

import numpy as np
from shapely.geometry import Polygon, Point, box

from ocgis import constants
from ocgis.exc import GridDeficientError, EmptySubsetError, AllElementsMaskedError
from ocgis.new_interface.geom import GeometryVariable, AbstractSpatialContainer
from ocgis.new_interface.mpi import MPI_RANK, get_optimal_splits, create_nd_slices, MPI_SIZE, MPI_COMM
from ocgis.new_interface.ocgis_logging import log, log_entry_exit
from ocgis.new_interface.variable import VariableCollection, get_dslice
from ocgis.util.environment import ogr
from ocgis.util.helpers import iter_array, get_trimmed_array_by_mask, get_formatted_slice

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint

_NAMES_2D = ['ocgis_yc', 'ocgis_xc']


class GridXY(AbstractSpatialContainer):
    ndim = 2

    def __init__(self, x, y, abstraction='auto', crs=None, parent=None):
        self._abstraction = None

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

        if self.dimensions is None:
            if self._archetype.ndim == 1:
                self.x.create_dimensions(names='ocgis_xc')
                self.y.create_dimensions(names='ocgis_yc')
            else:
                self.x.create_dimensions(names=_NAMES_2D)
                self.y.create_dimensions(names=_NAMES_2D)

        if self._archetype.ndim == 1:
            self.is_vectorized = True
            expand_grid(self)
        else:
            self.is_vectorized = False

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

        # No way to know if the data is still vectorized...
        self.is_vectorized = False

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
    def has_allocated_point(self):
        if self._point_name in self.parent:
            return True
        else:
            return False

    @property
    def has_allocated_polygon(self):
        if self._polygon_name in self.parent:
            return True
        else:
            return False

    @property
    def has_allocated_abstraction_geometry(self):
        if self.abstraction == 'point':
            return self.has_allocated_point
        elif self.abstraction == 'polygon':
            return self.has_allocated_polygon
        else:
            raise NotImplementedError(self.abstraction)

    @property
    def dimensions(self):
        return self._archetype.dimensions

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

    @property
    def x(self):
        return self.parent[self._x_name]

    @x.setter
    def x(self, value):
        self.parent[self._x_name] = value

    @property
    def y(self):
        return self.parent[self._y_name]

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
        return self._archetype.shape

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
        return self.y

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
        return self._archetype.get_mask()

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

    def get_intersects(self, bounds_or_geometry, return_slice=False, use_bounds='auto', use_spatial_index=True,
                       cascade=False):
        if use_bounds == 'auto':
            if self.abstraction == 'polygon':
                use_bounds = True
            else:
                use_bounds = False

        ret, slc = grid_get_intersects(self, bounds_or_geometry, use_bounds=use_bounds,
                                       use_spatial_index=use_spatial_index)

        if cascade:
            grid_set_mask_cascade(ret)

        if MPI_RANK == 0:
            if return_slice:
                ret = (ret, slc)
        else:
            if return_slice:
                ret = None, None
            else:
                ret = None

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

    def get_intersects_masked(self, *args, **kwargs):
        ret = self.copy()
        cascade = kwargs.pop('cascade', False)
        fill = self.abstraction_geometry.get_mask_from_intersects(*args, **kwargs)
        self.set_mask(fill, cascade=cascade)
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

    def write_netcdf(self, dataset, **kwargs):
        popped = []
        for target in [self._point_name, self._polygon_name]:
            popped.append(self.parent.pop(target, None))
        if self.is_vectorized:
            original_x = self.x
            original_y = self.y

            self.x = self.x[0, :]
            self.x.dimensions = None
            self.x.value = self.x.value.reshape(-1)
            self.x._mask = None
            self.x.dimensions = list(original_x.dimensions)[1]

            self.y = self.y[:, 0]
            self.y.dimensions = None
            self.y.value = self.y.value.reshape(-1)
            self.y._mask = None
            self.y.dimensions = list(original_y.dimensions)[0]
            #tdk: test mask is returned after write
        try:
            super(GridXY, self).write_netcdf(dataset, **kwargs)
        finally:
            for p in popped:
                if p is not None:
                    self.parent[p.name] = p
            if self.is_vectorized:
                self.x = original_x
                self.y = original_y


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


@log_entry_exit
def get_geometry_variable(func, grid, **kwargs):
    kwargs = kwargs.copy()
    alloc_only = kwargs.pop('alloc_only', False)
    value = get_geometry_fill(grid.shape, grid.get_mask())
    if not alloc_only:
        value = func(grid, value)
    kwargs['value'] = value
    kwargs['crs'] = grid.crs
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


@log_entry_exit
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

    if with_geometry and grid.is_vectorized:
        grid.expand()

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

    has_bounds, is_vectorized = grid.has_bounds, grid.is_vectorized

    res_x = get_coordinate_boolean_array(grid.x, is_vectorized, keep_touches, maxx, minx)
    res_y = get_coordinate_boolean_array(grid.y, is_vectorized, keep_touches, maxy, miny)

    try:
        if is_vectorized:
            if any([np.all(r) for r in [res_x, res_y]]):
                raise AllElementsMaskedError
            else:
                grid.x.set_mask(res_x)
                grid.y.set_mask(res_y)
        else:
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

    if grid.is_vectorized:
        _, y_slice = get_trimmed_array_by_mask(fill_grid.y.get_mask(), return_adjustments=True)
        _, x_slice = get_trimmed_array_by_mask(fill_grid.x.get_mask(), return_adjustments=True)
        slc_ret = (y_slice[0], x_slice[0])
    else:
        _, slc_ret = get_trimmed_array_by_mask(fill_grid.get_mask(), return_adjustments=True)

    fill_grid = fill_grid[slc_ret]

    new_slc_ret = [None] * len(slc_ret)
    for idx, sr in enumerate(slc_ret):
        target_slice_remaining = slc_remaining[idx]
        new_sr = slice(sr.start + target_slice_remaining.start, sr.stop + target_slice_remaining.start)
        new_slc_ret[idx] = new_sr
    slc_ret = tuple(new_slc_ret)

    return fill_grid, slc_ret


def get_coordinate_boolean_array(grid_target, is_vectorized, keep_touches, max_target, min_target):
    target_centers = grid_target.value

    res_target = np.array(get_arr_intersects_bounds(target_centers, min_target, max_target, keep_touches=keep_touches))
    res_target = res_target.reshape(-1)

    if is_vectorized:
        res_target = np.invert(res_target)
    return res_target


def grid_set_geometry_variable_on_parent(func, grid, name, alloc_only=False):
    ret = get_geometry_variable(func, grid, name=name, attrs={'axis': 'geom'}, alloc_only=alloc_only)
    ret.create_dimensions(names=[d.name for d in grid.dimensions])
    grid.parent.add_variable(ret)
    return ret


def grid_set_mask_cascade(grid):
    members = grid.get_member_variables(include_bounds=True)
    members = [m.name for m in members]
    grid.parent.set_mask(grid._archetype, exclude=members)


def expand_grid(grid):
    y = grid.y
    if y.ndim == 1:
        x = grid.x
        new_x_value, new_y_value = np.meshgrid(x.value, y.value)
        new_dimensions = [y.dimensions[0], x.dimensions[0]]

        x.value = None
        x.dimensions = new_dimensions
        x.value = new_x_value

        y.value = None
        y.dimensions = new_dimensions
        y.value = new_y_value

        assert y.ndim == 2
        assert x.ndim == 2

        if y.bounds is not None:
            name_y = y.bounds.name
            name_x = x.bounds.name
            name_dimension = y.bounds.dimensions[1].name
            y.bounds = None
            x.bounds = None
            # tdk: this should leverage the bounds already in place on the vectors
            grid.set_extrapolated_bounds(name_x, name_y, name_dimension)

