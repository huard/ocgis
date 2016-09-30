import logging
from collections import OrderedDict
from copy import deepcopy
from warnings import warn

import netCDF4 as nc
import numpy as np
from netCDF4._netCDF4 import VLType

from ocgis import env
from ocgis.api.request.driver.base import AbstractDriver, get_group, driver_scope
from ocgis.constants import MPIWriteMode, MPIDistributionMode, DimensionMapKeys
from ocgis.exc import ProjectionDoesNotMatch, PayloadProtectedError, OcgWarning
from ocgis.interface.base.crs import CFCoordinateReferenceSystem
from ocgis.new_interface.base import orphaned
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.mpi import MPI_COMM
from ocgis.new_interface.temporal import TemporalVariable
from ocgis.new_interface.variable import SourcedVariable, ObjectType, VariableCollection, \
    get_slice_sequence_using_local_bounds
from ocgis.util.helpers import itersubclasses, get_iter, get_formatted_slice, get_by_key_list
from ocgis.util.logging_ocgis import ocgis_lh


class DriverNetcdf(AbstractDriver):
    extensions = ('.*\.nc', 'http.*')
    key = 'netcdf'
    output_formats = 'all'

    @classmethod
    def get_variable_for_writing_temporal(cls, temporal_variable):
        return temporal_variable.value_numtime

    def get_variable_collection(self):
        with driver_scope(self) as ds:
            ret = read_from_collection(ds, self.rd, parent=None)
        return ret

    def get_variable_value(self, variable):
        return get_value_from_request_dataset(variable)

    @classmethod
    def write_variable(cls, var, dataset, write_mode=MPIWriteMode.NORMAL, **kwargs):
        """
        Write a variable to an open netCDF dataset object.

        :param var: Variable object.
        :param dataset: Open netCDF dataset object.
        :param kwargs: Arguments to netCDF variable creation with additional keyword arguments below.
        :keyword bool file_only: (``=False``) If ``True``, do not write the value to the output file. Create an empty
         netCDF file.
        :keyword bool unlimited_to_fixedsize: (``=False``) If ``True``, convert the unlimited dimension to a fixed size.
        """

        # Write the parent collection if available on the variable.
        if not var.is_orphaned:
            return var.parent.write(dataset, variable_kwargs=kwargs)

        assert isinstance(dataset, nc.Dataset)

        file_only = kwargs.pop('file_only', False)
        unlimited_to_fixedsize = kwargs.pop('unlimited_to_fixedsize', False)

        # No data should be written during a global write. Data will be filled in during the append process.
        if write_mode == MPIWriteMode.TEMPLATE:
            file_only = True

        if var.name is None:
            msg = 'A variable "name" is required.'
            raise ValueError(msg)

        # Dimension creation should not occur during a fill operation. The dimensions and variables have already been
        # created.
        if write_mode != MPIWriteMode.FILL:
            dimensions = var.dimensions

            dtype = cls.get_variable_write_dtype(var)
            if isinstance(dtype, ObjectType):
                dtype = dtype.create_vltype(dataset, dimensions[0].name + '_VLType')

            if len(dimensions) > 0:
                dimensions = list(dimensions)
                # Convert the unlimited dimension to fixed size if requested.
                for idx, d in enumerate(dimensions):
                    if d.is_unlimited and unlimited_to_fixedsize:
                        dimensions[idx] = Dimension(d.name, size=var.shape[idx])
                        break
                # Create the dimensions.
                for dim in dimensions:
                    create_dimension_or_pass(dim, dataset, write_mode=write_mode)
                dimensions = [d.name for d in dimensions]

            # Only use the fill value if something is masked.
            if len(dimensions) > 0 and not file_only and var.has_masked_values:
                fill_value = cls.get_variable_write_fill_value(var)
            else:
                # Copy from original attributes.
                if '_FillValue' not in var.attrs:
                    fill_value = None
                else:
                    fill_value = cls.get_variable_write_fill_value(var)

        if write_mode == MPIWriteMode.FILL:
            ncvar = dataset.variables[var.name]
        else:
            ncvar = dataset.createVariable(var.name, dtype, dimensions=dimensions, fill_value=fill_value, **kwargs)

        # Do not fill values on file_only calls. Also, only fill values for variables with dimension greater than zero.
        if not file_only and var.ndim > 0 and not var.is_empty:
            if isinstance(var.dtype, ObjectType) and not isinstance(var, TemporalVariable):
                bounds_local = var.dimensions[0].bounds_local
                for idx in range(bounds_local[0], bounds_local[1]):
                    ncvar[idx] = np.array(var.value[idx - bounds_local[0]])
            else:
                fill_slice = get_slice_sequence_using_local_bounds(var)
                data_value = cls.get_variable_write_value(var)
                ncvar.__setitem__(fill_slice, data_value)

        # Only set variable attributes if this is not a fill operation.
        if write_mode != MPIWriteMode.FILL:
            var.write_attributes_to_netcdf_object(ncvar)
            if var.units is not None:
                ncvar.units = var.units

        dataset.sync()

    @classmethod
    def _write_variable_collection_main_(cls, vc, opened_or_path, comm, rank, size, write_mode, archetype_rank,
                                         **kwargs):
        """
        Write the variable collection to an open netCDF dataset or file path.

        :param dataset: The open dataset object or path for the write.
        :type dataset: :class:`netCDF4.Dataset` or str
        :param bool file_only: If ``True``, we are not filling the value variables. Only the file schema and dimension
         values will be written.
        :param bool unlimited_to_fixedsize: If ``True``, convert the unlimited dimension to fixed size.
        :param kwargs: Extra keyword arguments in addition to ``dimensions`` and ``fill_value`` to pass to
         ``createVariable``. See http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html#createVariable
        """
        from ocgis.new_interface.geom import GeometryVariable

        assert write_mode is not None
        dataset_kwargs = kwargs.get('dataset_kwargs', {})
        variable_kwargs = kwargs.get('variable_kwargs', {})

        # When filling a dataset, we use append mode.
        if write_mode == MPIWriteMode.FILL:
            mode = 'a'
        else:
            mode = 'w'

        # Check for distributed variables. If there are no distributed variables, do not add a barrier.
        distributions = [variable.dist for variable in vc.values()]
        if any([d is not None for d in distributions]):
            add_barrier = True
        else:
            add_barrier = False

        # Write the data on each rank.
        for rank_to_write in range(size):
            # The template write only occurs on the first rank.
            if write_mode == MPIWriteMode.TEMPLATE and rank_to_write != archetype_rank:
                pass
            # If this is not a template write, fill the data.
            elif rank == rank_to_write:
                with driver_scope(cls, opened_or_path=opened_or_path, mode=mode, **dataset_kwargs) as dataset:
                    # Write global attributes if we are not filling data.
                    if write_mode != MPIWriteMode.FILL:
                        vc.write_attributes_to_netcdf_object(dataset)
                    # This is the main variable write loop.
                    for variable in vc.values():
                        if isinstance(variable, GeometryVariable):
                            continue
                        # For isolated and replicated variables, only write once.
                        if write_mode != MPIWriteMode.TEMPLATE:
                            if variable.dist is not None and variable.dist != MPIDistributionMode.DISTRIBUTED:
                                if variable.dist == MPIDistributionMode.REPLICATED:
                                    if rank != 0:
                                        continue
                                else:
                                    if rank != variable.ranks[0]:
                                        continue
                        # Load the variable's data before orphaning. The variable needs its parent to know which group
                        # it is in.
                        variable.load()
                        # Call the individual variable write method in fill mode. Orphaning is required as a variable
                        # will attempt to write its parent first.
                        with orphaned(variable, keep_dimensions=True):
                            variable.write(dataset, write_mode=write_mode, **variable_kwargs)
                    # Recurse the children.
                    for child in vc.children.values():
                        if write_mode != MPIWriteMode.FILL:
                            group = nc.Group(dataset, child.name)
                        else:
                            group = dataset.groups[child.name]
                        child.write(group, write_mode=write_mode, **kwargs)
                    dataset.sync()

            # Allow each rank to finish it's write process. Only one rank can have the dataset open at a given time.
            if add_barrier:
                comm.Barrier()

    def _get_dimensions_main_(self, group_metadata):
        return tuple(get_dimensions_from_netcdf_metadata(group_metadata, group_metadata['dimensions'].keys()))

    def _get_metadata_main_(self):
        with driver_scope(self) as ds:
            ret = parse_metadata(ds)
        return ret

    def _init_variable_from_source_main_(self, variable, variable_object):
        init_variable_using_metadata_for_netcdf(self, variable, self.rd.metadata)

    @staticmethod
    def _open_(uri, mode='r', **kwargs):
        """
        :rtype: object
        """
        group_indexing = kwargs.pop('group_indexing', None)

        if isinstance(uri, basestring):
            ret = nc.Dataset(uri, mode=mode, **kwargs)
        else:
            ret = nc.MFDataset(uri, **kwargs)

        if group_indexing is not None:
            for group_name in get_iter(group_indexing):
                ret = ret.groups[group_name]

        return ret


class DriverNetcdfCF(DriverNetcdf):
    key = 'netcdf-cf'
    _default_crs = env.DEFAULT_COORDSYS
    _priority = True

    def get_dimension_map(self, metadata):
        # tdk: we'll need to handle hierarchical fields at some point...dimension map only works on the first group
        # Get dimension variable metadata. This involves checking for the presence of any bounds variables.
        variables = metadata['variables']
        dimensions = metadata['dimensions']
        axes = {'realization': 'R', 'time': 'T', 'level': 'L', 'x': 'X', 'y': 'Y'}
        check_bounds = axes.keys()
        check_bounds.pop(check_bounds.index('realization'))

        # Get the main entry for each axis.
        for k, v in axes.items():
            axes[k] = get_dimension_map_entry(v, variables, dimensions)

        # Attempt to find bounds for each entry (ignoring realizations).
        for k in check_bounds:
            if axes[k] is not None:
                keys = ['bounds']
                if k == 'time':
                    keys += ['climatology']
                bounds_var = get_by_key_list(variables[axes[k]['variable']]['attributes'], keys)
                if bounds_var is not None:
                    if bounds_var not in variables:
                        msg = 'Bounds listed for variable "{0}" but the destination bounds variable "{1}" does not exist.'. \
                            format(axes[k]['variable'], bounds_var)
                        ocgis_lh(msg, logger='nc.driver', level=logging.WARNING)
                        bounds_var = None
                axes[k]['bounds'] = bounds_var

        # Create the template dimension map dictionary.
        ret = {k: v for k, v in axes.items() if v is not None}

        # Check for coordinate system variables. This will check every variable.
        crs_name = None
        if self.rd._crs is not None and self.rd._crs != 'auto':
            crs_name = self.rd._crs.name
        else:
            crs = self.get_crs(metadata)
            if crs is not None:
                crs_name = crs.name
        if crs_name is not None:
            ret['crs'] = {'variable': crs_name}

        return ret

    @staticmethod
    def get_dimensioned_variables(field):
        axes_needed = ['time', 'x', 'y']
        dvars = []
        group_dimension_map = field.dimension_map

        for v in field.iter_data_variables():
            vname = v.name
            found_axis = []
            for a in axes_needed:
                to_append = False
                d = group_dimension_map.get(a)
                if d is not None:
                    for dname in (d.name for d in v.dimensions):
                        if dname in d['names']:
                            to_append = True
                            break
                found_axis.append(to_append)
            if all(found_axis):
                dvars.append(vname)

        return tuple(dvars)

    def get_distributed_dimension_name(self, dimension_map, dimensions_metadata):
        if DimensionMapKeys.X in dimension_map and DimensionMapKeys.Y in dimension_map:
            sizes = np.zeros(2, dtype={'names': ['dim', 'size'], 'formats': [object, int]})

            dimension_name_x = dimension_map[DimensionMapKeys.X]['names'][0]
            dimension_name_y = dimension_map[DimensionMapKeys.Y]['names'][0]

            sizes[0] = (dimension_name_x, dimensions_metadata[dimension_name_x]['size'])
            sizes[1] = (dimension_name_y, dimensions_metadata[dimension_name_y]['size'])
            max_index = np.argmax(sizes['size'])
            ret = sizes['dim'][max_index]
        else:
            ret = None
        return ret

    def _get_crs_main_(self, group_metadata):
        return get_crs_variable(group_metadata)

    @classmethod
    def _get_variable_collection_write_target_(cls, field):
        from ocgis.new_interface.variable import Variable

        if field.crs is not None:
            for dimensioned_variable_name in cls.get_dimensioned_variables(field):
                field[dimensioned_variable_name].attrs['grid_mapping_name'] = field.crs.name

        if field.grid is not None:
            gridxy = field.grid
            has_bounds = gridxy.has_bounds
            if gridxy.is_vectorized:
                to_write = gridxy.parent.copy()

                y = Variable(name=gridxy.y.name, value=gridxy.y.value[:, 0].reshape(-1),
                             dimensions=gridxy.dimensions[0], attrs=gridxy.y.attrs, dist=gridxy.y.dist,
                             ranks=gridxy.y.ranks)
                x = Variable(name=gridxy.x.name, value=gridxy.x.value[0, :].reshape(-1),
                             dimensions=gridxy.dimensions[1], attrs=gridxy.x.attrs, dist=gridxy.x.dist,
                             ranks=gridxy.x.ranks)

                if has_bounds:
                    x_bounds = np.squeeze(gridxy.x.bounds.value[0, :, :])
                    x_bounds_fill = np.zeros((x.shape[0], 2), dtype=x.dtype)
                    x_bounds_fill[:, 0] = np.min(x_bounds, axis=1)
                    x_bounds_fill[:, 1] = np.max(x_bounds, axis=1)

                    y_bounds = np.squeeze(gridxy.y.bounds.value[:, 0, :])
                    y_bounds_fill = np.zeros((y.shape[0], 2), dtype=y.dtype)
                    y_bounds_fill[:, 0] = np.min(y_bounds, axis=1)
                    y_bounds_fill[:, 1] = np.max(y_bounds, axis=1)

                    y_bounds_dimensions = [y.dimensions[0], Dimension(gridxy._original_bounds_dimension_name, 2)]
                    x_bounds_dimensions = [x.dimensions[0], Dimension(gridxy._original_bounds_dimension_name, 2)]
                    x_bounds_var = Variable(name=gridxy.x.bounds.name, value=x_bounds_fill,
                                            dimensions=x_bounds_dimensions, dist=gridxy.x.bounds.dist,
                                            ranks=gridxy.x.bounds.ranks)
                    y_bounds_var = Variable(name=gridxy.y.bounds.name, value=y_bounds_fill,
                                            dimensions=y_bounds_dimensions, dist=gridxy.y.bounds.dist,
                                            ranks=gridxy.y.bounds.ranks)
                    x.bounds = x_bounds_var
                    y.bounds = y_bounds_var

                to_write.add_variable(x, force=True)
                to_write.add_variable(y, force=True)
            else:
                to_write = field
        else:
            to_write = field

        return to_write


def parse_metadata(rootgrp, fill=None):
    if fill is None:
        fill = OrderedDict()
    if 'groups' not in fill:
        fill['groups'] = OrderedDict()
    update_group_metadata(rootgrp, fill)
    for group in rootgrp.groups.values():
        new_fill = fill['groups'][group.name] = OrderedDict()
        parse_metadata(group, fill=new_fill)
    return fill


def read_from_collection(target, request_dataset, parent=None, name=None):
    ret = VariableCollection(attrs=deepcopy(target.__dict__), parent=parent, name=name)
    for varname, ncvar in target.variables.iteritems():
        ret[varname] = SourcedVariable(name=varname, request_dataset=request_dataset, parent=ret)
    for group_name, ncgroup in target.groups.items():
        child = read_from_collection(ncgroup, request_dataset, parent=ret, name=group_name)
        ret.add_child(child)
    return ret


def update_group_metadata(rootgrp, fill):
    fill.update({'global_attributes': deepcopy(rootgrp.__dict__)})

    # get file format
    fill.update({'file_format': rootgrp.file_format})

    # get variables
    variables = OrderedDict()
    for key, value in rootgrp.variables.iteritems():
        subvar = OrderedDict()
        for attr in value.ncattrs():
            subvar.update({attr: getattr(value, attr)})

        # Remove scale factors and offsets from the metadata.
        if 'scale_factor' in subvar:
            dtype_packed = value[0].dtype
            fill_value_packed = np.ma.array([], dtype=dtype_packed).fill_value
        else:
            dtype_packed = None
            fill_value_packed = None

        # make two attempts at missing value attributes otherwise assume the default from a numpy masked array
        try:
            fill_value = value.fill_value
        except AttributeError:
            try:
                fill_value = value.missing_value
            except AttributeError:
                fill_value = np.ma.array([], dtype=value.dtype).fill_value

        variables.update({key: {'dimensions': value.dimensions,
                                'attributes': subvar,
                                'dtype': value.dtype,
                                'name': value._name,
                                'fill_value': fill_value,
                                'dtype_packed': dtype_packed,
                                'fill_value_packed': fill_value_packed}})
    fill.update({'variables': variables})

    # get dimensions
    dimensions = OrderedDict()
    for key, value in rootgrp.dimensions.iteritems():
        subdim = {key: {'name': key, 'size': len(value), 'isunlimited': value.isunlimited()}}
        dimensions.update(subdim)
    fill.update({'dimensions': dimensions})


def init_variable_using_metadata_for_netcdf(driver, variable, metadata):
    source = get_group(metadata, variable.group, has_root=False)
    desired_name = variable.name or variable._request_dataset.variable
    var = source['variables'][desired_name]

    # Use the distribution to identify if this is an isolated variable. Isolated variables exist on select ranks.
    dist_for_var = driver.dist.get_variable(variable)

    # tdk: understand issues with the null communicator
    # rank = driver.rd.comm.Get_rank()
    rank = MPI_COMM.Get_rank()

    if dist_for_var['dist'] == MPIDistributionMode.ISOLATED and rank not in dist_for_var['ranks']:
        variable._is_empty = True

    # Update the variable's distribution and rank information.
    variable.dist = dist_for_var['dist']
    if variable.dist == MPIDistributionMode.ISOLATED:
        variable.ranks = dist_for_var['ranks']
    else:
        variable.ranks = 'all'

    # Update data type and fill value.
    if variable._dtype is None:
        var_dtype = var['dtype']
        desired_dtype = deepcopy(var_dtype)
        if isinstance(var_dtype, VLType):
            desired_dtype = ObjectType(var_dtype)
        elif var['dtype_packed'] is not None:
            desired_dtype = deepcopy(var['dtype_packed'])
        variable._dtype = desired_dtype

    if variable._fill_value is None:
        if var['fill_value_packed'] is not None:
            desired_fill_value = var['fill_value_packed']
        else:
            desired_fill_value = var['fill_value']
        variable._fill_value = deepcopy(desired_fill_value)

    variable_attrs = variable._attrs
    # Offset and scale factors are not supported by OCGIS. The data is unpacked when written to a new output file.
    # tdk: consider supporting offset and scale factors
    exclude = ['add_offset', 'scale_factor']
    for k, v in var['attributes'].items():
        if k in exclude:
            continue
        if k not in variable_attrs:
            variable_attrs[k] = deepcopy(v)


def get_dimensions_from_netcdf_metadata(metadata, desired_dimensions):
    new_dimensions = []
    for dim_name in desired_dimensions:
        dim = metadata['dimensions'][dim_name]
        dim_length = dim['size']
        if dim['isunlimited']:
            length = None
            length_current = dim_length
        else:
            length = dim_length
            length_current = None
        # tdk: identify best method to remove the need to set 'auto' when creating a source index
        new_dim = Dimension(dim_name, size=length, size_current=length_current, src_idx='auto')
        new_dimensions.append(new_dim)
    return new_dimensions


def get_value_from_request_dataset(variable):
    if variable.protected:
        raise PayloadProtectedError

    rd = variable._request_dataset
    with driver_scope(rd.driver) as source:
        if variable.group is not None:
            for vg in variable.group:
                if vg is None:
                    continue
                else:
                    source = source.groups[vg]
        desired_name = variable.name or rd.variable
        ncvar = source.variables[desired_name]
        ret = get_variable_value(ncvar, variable.dimensions)
    return ret


def get_variable_value(variable, dimensions):
    if dimensions is not None and len(dimensions) > 0:
        to_format = [None] * len(dimensions)
        for idx in range(len(dimensions)):
            current_dimension = dimensions[idx]
            if current_dimension._src_idx is None:
                if current_dimension.bounds_local is None:
                    to_insert = slice(0, len(current_dimension))
                else:
                    to_insert = slice(*current_dimension.bounds_local)
            else:
                to_insert = current_dimension._src_idx
            to_format[idx] = to_insert
        slc = get_formatted_slice(to_format, len(dimensions))
    else:
        slc = slice(None)
    ret = variable.__getitem__(slc)
    return ret


def create_dimension_or_pass(dim, dataset, write_mode=MPIWriteMode.NORMAL):
    if dim.name not in dataset.dimensions:
        if write_mode == MPIWriteMode.TEMPLATE:
            lower, upper = dim.bounds_global
            size = upper - lower
        else:
            size = dim.size
        dataset.createDimension(dim.name, size)


def get_crs_variable(metadata, to_search=None):
    found = []
    variables = metadata['variables']

    for vname, var in variables.items():
        if to_search is not None:
            if vname not in to_search:
                continue
        for potential in itersubclasses(CFCoordinateReferenceSystem):
            try:
                crs = potential.load_from_metadata(vname, metadata)
                found.append(crs)
                break
            except ProjectionDoesNotMatch:
                continue

    fset = set([f.name for f in found])
    if len(fset) > 1:
        msg = 'Multiple coordinate systems found. There should be only one.'
        raise ValueError(msg)
    elif len(fset) == 0:
        crs = None
    else:
        crs = found[0]

    return crs


def get_dimension_map_entry(axis, variables, dimensions):
    axis_vars = []
    for variable in variables.values():
        vattrs = variable['attributes']
        if vattrs.get('axis') == axis:
            axis_vars.append(variable['name'])
    if len(axis_vars) == 1:
        var_name = axis_vars[0]
        ret = {'variable': var_name, 'names': list(variables[var_name]['dimensions'])}
    elif len(axis_vars) > 1:
        msg = 'Multiple axis attributes with value "{}" found on variables "{}". Use a dimension map to specify the ' \
              'appropriate coordinate dimensions.'
        w = OcgWarning(msg.format(axis, axis_vars))
        warn(w)
        ret = None
    else:
        ret = None
    return ret
