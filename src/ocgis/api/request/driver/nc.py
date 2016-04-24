import logging
from collections import OrderedDict
from copy import deepcopy
from warnings import warn

import netCDF4 as nc
import numpy as np
from netCDF4._netCDF4 import VLType

from ocgis import messages, env
from ocgis.api.request.driver.base import AbstractDriver
from ocgis.exc import ProjectionDoesNotMatch, PayloadProtectedError
from ocgis.interface.base.crs import CFCoordinateReferenceSystem
from ocgis.interface.base.dimension.spatial import SpatialDimension
from ocgis.interface.base.variable import Variable
from ocgis.interface.nc.dimension import NcVectorDimension
from ocgis.interface.nc.field import NcField
from ocgis.interface.nc.spatial import NcSpatialGridDimension
from ocgis.new_interface.base import orphaned
from ocgis.new_interface.dimension import SourcedDimension, Dimension
from ocgis.new_interface.variable import SourcedVariable, ObjectType, VariableCollection
from ocgis.util.helpers import itersubclasses, get_iter, get_formatted_slice, get_by_key_list, iter_array
from ocgis.util.logging_ocgis import ocgis_lh


# tdk: rename key to "netcdf-cf"

class DriverNetcdf(AbstractDriver):
    extensions = ('.*\.nc', 'http.*')
    key = 'netCDF'
    output_formats = 'all'
    _default_crs = env.DEFAULT_COORDSYS

    def get_metadata(self):
        ds = self.open()
        try:
            ret = parse_metadata(ds)
        finally:
            self.close(ds)
        return ret

    def _open_(self, group_indexing=None, mode='r'):
        try:
            ret = nc.Dataset(self.rd.uri, mode=mode)
        except (TypeError, RuntimeError):
            try:
                ret = nc.MFDataset(self.rd.uri)
            except KeyError as e:
                # it is possible the variable is not in one of the data URIs. check for this to raise a cleaner error.
                for uri in get_iter(self.rd.uri):
                    # tdk: this should use request datasets?
                    ds = nc.Dataset(uri, 'r')
                    try:
                        for variable in get_iter(self.rd.variable):
                            try:
                                ds.variables[variable]
                            except KeyError:
                                msg = 'The variable "{0}" was not found in URI "{1}".'.format(variable, uri)
                                raise KeyError(msg)
                    finally:
                        ds.close()

                # If all variables were found, raise the other error.
                raise e

        if group_indexing is not None:
            for group_name in get_iter(group_indexing):
                ret = ret.groups[group_name]

        return ret

    def _close_(self, obj):
        obj.close()

    def get_crs(self):
        return get_crs_variable(self.metadata)

    def get_dimension_map(self, metadata):
        # Get dimension variable metadata. This involves checking for the presence of any bounds variables.
        variables = metadata['variables']
        axes = {'realization': 'R', 'time': 'T', 'level': 'L', 'x': 'X', 'y': 'Y'}
        check_bounds = axes.keys()
        check_bounds.pop(check_bounds.index('realization'))
        for k, v in axes.items():
            axes[k] = get_dimension_map_entry(v, variables)
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
        elif self.crs is not None:
            crs_name = self.crs.name
        if crs_name is not None:
            ret['crs'] = {'variable': crs_name}

        return ret

    def get_dimensioned_variables(self, dimension_map, metadata):
        # tdk: add option to specify needed axes
        axes_needed = ['time', 'x', 'y']
        dvars = []

        for vname, v in metadata['variables'].items():
            found_axis = []
            for a in axes_needed:
                to_append = False
                d = dimension_map.get(a)
                if d is not None:
                    for dname in v['dimensions']:
                        if dname in d['names']:
                            to_append = True
                            break
                found_axis.append(to_append)
            if all(found_axis):
                dvars.append(vname)

        return dvars

    def get_dump_report(self):
        lines = get_dump_report_for_group(self.metadata)
        lines.insert(0, 'netcdf {')
        for group_name, group_metadata in self.metadata['groups'].items():
            lines.append('')
            lines.append('group: ' + group_name + ' {')
            lines += get_dump_report_for_group(group_metadata, global_attributes_name=group_name, indent=2)
            lines.append('  }')
        lines.append('}')
        return lines

    def get_variable_collection(self):
        ds = self.open()
        try:
            ret = read_from_collection(ds, self.rd, parent=None)
        finally:
            self.close(ds)
        return ret

    def allocate_variable_without_value(self, variable):
        allocate_variable_using_metadata(variable, self.rd.metadata)

    def get_variable_value(self, variable):
        return get_value_from_request_dataset(variable)

    @staticmethod
    def write_gridxy(gridxy, dataset, **kwargs):
        popped = []
        for target in [gridxy._point_name, gridxy._polygon_name]:
            popped.append(gridxy.parent.pop(target, None))
        if gridxy.is_vectorized:
            original_dimensions = deepcopy(gridxy.dimensions)
            original_x = gridxy.x.copy()
            original_y = gridxy.y.copy()

            gridxy.x = gridxy.x[0, :]
            gridxy.y = gridxy.y[:, 0]
            x, y = gridxy.x, gridxy.y

            x.dimensions = None
            x.value = x.value.reshape(-1)
            x._mask = None
            x.dimensions = original_dimensions[1]

            y.dimensions = None
            y.value = y.value.reshape(-1)
            y._mask = None
            y.dimensions = original_dimensions[0]
        try:
            gridxy.parent.write(dataset, **kwargs)
        finally:
            for p in popped:
                if p is not None:
                    gridxy.parent[p.name] = p
            if gridxy.is_vectorized:
                gridxy.x = original_x
                gridxy.y = original_y

    @staticmethod
    def write_variable(var, dataset, **kwargs):
        if var.parent is not None:
            return var.parent.write(dataset, **kwargs)

        if var.name is None:
            msg = 'A variable "name" is required.'
            raise ValueError(msg)

        file_only = kwargs.pop('file_only', False)
        unlimited_to_fixedsize = kwargs.pop('unlimited_to_fixedsize', False)

        if var.dimensions is None:
            new_names = ['dim_ocgis_{}_{}'.format(var.name, ctr) for ctr in range(var.ndim)]
            var.create_dimensions(new_names)

        dimensions = var.dimensions

        dtype = var._get_netcdf_dtype_()
        if isinstance(dtype, ObjectType):
            dtype = dtype.create_vltype(dataset, dimensions[0].name + '_VLType')

        if len(dimensions) > 0:
            dimensions = list(dimensions)
            # Convert the unlimited dimension to fixed size if requested.
            for idx, d in enumerate(dimensions):
                if d.length is None and unlimited_to_fixedsize:
                    dimensions[idx] = Dimension(d.name, length=var.shape[idx])
                    break
            # Create the dimensions.
            for dim in dimensions:
                create_dimension_or_pass(dim, dataset)
            dimensions = [d.name for d in dimensions]

        # Only use the fill value if something is masked.
        if len(dimensions) > 0 and not file_only and var.get_mask().any():
            fill_value = var.fill_value
        else:
            # Copy from original attributes.
            if '_FillValue' not in var.attrs:
                fill_value = None
            else:
                fill_value = var._get_netcdf_fill_value_()

        ncvar = dataset.createVariable(var.name, dtype, dimensions=dimensions, fill_value=fill_value, **kwargs)
        if not file_only:
            try:
                ncvar[:] = var._get_netcdf_value_()
            except AttributeError:
                # Assume ObjectType.
                for idx, v in iter_array(var.value, use_mask=False, return_value=True):
                    ncvar[idx] = np.array(v)

        var.write_attributes_to_netcdf_object(ncvar)

        if var.units is not None:
            ncvar.units = var.units

        dataset.sync()

    @staticmethod
    def write_variable_collection(vc, dataset_or_path, **kwargs):
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

        if not isinstance(dataset_or_path, (nc.Dataset, nc.Group)):
            dataset = nc.Dataset(dataset_or_path, 'w')
            close_dataset = True
        else:
            dataset = dataset_or_path
            close_dataset = False

        try:
            vc.write_attributes_to_netcdf_object(dataset)
            for variable in vc.values():
                with orphaned(vc, variable):
                    variable.write(dataset, **kwargs)
            for child in vc.children.values():
                group = nc.Group(dataset, child.name)
                child.write(group, **kwargs)
            dataset.sync()
        finally:
            if close_dataset:
                dataset.close()

    # tdk: remove
    def _get_field_(self, format_time=True):
        """
        :param bool format_time:
        :raises ValueError:
        """

        # reference the request dataset's source metadata
        source_metadata = self.rd.source_metadata

        def _get_temporal_adds_(ref_attrs):
            # calendar should default to standard if it is not present and the t_calendar overload is not used.
            calendar = self.rd.t_calendar or ref_attrs.get('calendar', None) or 'standard'

            return {'units': self.rd.t_units or ref_attrs['units'], 'calendar': calendar, 'format_time': format_time,
                    'conform_units_to': self.rd.t_conform_units_to}

        # parameters for the loading loop
        to_load = {'temporal': {'cls': NcTemporalDimension, 'adds': _get_temporal_adds_, 'axis': 'T', 'name_uid': 'tid',
                                'name': 'time'},
                   'level': {'cls': NcVectorDimension, 'adds': None, 'axis': 'Z', 'name_uid': 'lid', 'name': 'level'},
                   'row': {'cls': NcVectorDimension, 'adds': None, 'axis': 'Y', 'name_uid': 'yc_id', 'name': 'yc'},
                   'col': {'cls': NcVectorDimension, 'adds': None, 'axis': 'X', 'name_uid': 'xc_id', 'name': 'xc'},
                   'realization': {'cls': NcVectorDimension, 'adds': None, 'axis': 'R', 'name_uid': 'rlz_id',
                                   'name_value': 'rlz'}}

        loaded = {}
        kwds_grid = {}
        has_row_column = True
        for k, v in to_load.iteritems():
            fill = self._get_vector_dimension_(k, v, source_metadata)
            if k != 'realization' and not isinstance(fill, NcVectorDimension) and fill is not None:
                assert k in ('row', 'col')
                has_row_column = False
                kwds_grid[k] = fill
            loaded[k] = fill

        loaded_keys = set([k for k, v in loaded.iteritems() if v is not None])
        if has_row_column:
            if not {'temporal', 'row', 'col'}.issubset(loaded_keys):
                raise ValueError('Target variable must at least have temporal, row, and column dimensions.')
            kwds_grid = {'row': loaded['row'], 'col': loaded['col']}
        else:
            shape_src_idx = [source_metadata['dimensions'][xx]['len'] for xx in kwds_grid['row']['dimensions']]
            src_idx = {'row': np.arange(0, shape_src_idx[0], dtype=np.int32),
                       'col': np.arange(0, shape_src_idx[1], dtype=np.int32)}
            name_row = kwds_grid['row']['name']
            name_col = kwds_grid['col']['name']
            kwds_grid = {'name_row': name_row, 'name_col': name_col, 'request_dataset': self.rd, 'src_idx': src_idx}

        grid = NcSpatialGridDimension(**kwds_grid)

        spatial = SpatialDimension(name_uid='gid', grid=grid, crs=self.rd.crs, abstraction=self.rd.s_abstraction)

        vc = VariableCollection()
        for vdict in self.rd:
            variable_meta = deepcopy(source_metadata['variables'][vdict['variable']])
            variable_units = vdict['units'] or variable_meta['attrs'].get('units')
            attrs = variable_meta['attrs'].copy()
            if variable_meta['dtype_packed'] is None:
                dtype = np.dtype(variable_meta['dtype'])
                fill_value = variable_meta['fill_value']
            else:
                dtype = np.dtype(variable_meta['dtype_packed'])
                fill_value = variable_meta['fill_value_packed']
                # Remove scale factors and offsets from the metadata.
                attrs.pop('scale_factor')
                attrs.pop('add_offset', None)
                attrs.pop('missing_value', None)
                attrs.pop('_Fill_Value', None)
            variable = Variable(vdict['variable'], vdict['alias'], units=variable_units, meta=variable_meta,
                                request_dataset=self.rd, conform_units_to=vdict['conform_units_to'], dtype=dtype,
                                fill_value=fill_value, attrs=attrs)
            vc.add_variable(variable)

        ret = NcField(variables=vc, spatial=spatial, temporal=loaded['temporal'], level=loaded['level'],
                      realization=loaded['realization'], meta=source_metadata.copy(), uid=self.rd.did,
                      name=self.rd.name, attrs=source_metadata['dataset'].copy())

        # tdk: RESUME: apply any subset parameters
        # Apply any subset parameters after the field is loaded.
        if self.rd.time_range is not None:
            ret = ret.get_between('temporal', min(self.rd.time_range), max(self.rd.time_range))
        if self.rd.time_region is not None:
            ret = ret.get_time_region(self.rd.time_region)
        if self.rd.time_subset_func is not None:
            ret = ret.get_time_subset_by_function(self.rd.time_subset_func)
        if self.rd.level_range is not None:
            try:
                ret = ret.get_between('level', min(self.rd.level_range), max(self.rd.level_range))
            except AttributeError:
                # there may be no level dimension
                if ret.level is None:
                    msg = messages.M4.format(self.rd.alias)
                    raise ValueError(msg)
                else:
                    raise

        return ret

    # tdk: remove
    @staticmethod
    def _get_name_bounds_dimension_(source_metadata):
        """
        :param dict source_metadata: Metadata dictionary as returned from :attr:`~ocgis.RequestDataset.source_metadata`.
        :returns: The name of the bounds suffix to use when creating dimensions. If no bounds are found in the source
         metadata return ``None``.
        :rtype: str or None
        """

        name_bounds_suffix = None
        for v2 in source_metadata['dim_map'].itervalues():
            # it is possible the dimension itself is none
            try:
                if v2 is not None and v2['bounds'] is not None:
                    name_bounds_suffix = source_metadata['variables'][v2['bounds']]['dimensions'][1]
                    break
            except KeyError:
                # bounds key is likely just not there
                if 'bounds' in v2:
                    raise
        return name_bounds_suffix


def read_from_collection(target, request_dataset, parent=None, name=None):
    ret = VariableCollection(attrs=deepcopy(target.__dict__), parent=parent, name=name)
    for name, ncvar in target.variables.iteritems():
        ret[name] = SourcedVariable(name=name, request_dataset=request_dataset, parent=ret)
    for name, ncgroup in target.groups.items():
        child = read_from_collection(ncgroup, request_dataset, parent=ret, name=name)
        ret.add_child(child)
    return ret


def parse_metadata(rootgrp):
    fill = OrderedDict()
    fill['groups'] = OrderedDict()
    update_group_metadata(rootgrp, fill)
    for group in rootgrp.groups.values():
        fill['groups'][group.name] = OrderedDict()
        fill_group = fill['groups'][group.name]
        update_group_metadata(group, fill_group)
    return fill


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
        subdim = {key: {'len': len(value), 'isunlimited': value.isunlimited()}}
        dimensions.update(subdim)
    fill.update({'dimensions': dimensions})


def get_dump_report_for_group(group, global_attributes_name='global', indent=0):
    lines = ['dimensions:']
    template = '    {0} = {1} ;{2}'
    for key, value in group['dimensions'].iteritems():
        if value['isunlimited']:
            one = 'ISUNLIMITED'
            two = ' // {0} currently'.format(value['len'])
        else:
            one = value['len']
            two = ''
        lines.append(template.format(key, one, two))

    lines.append('')
    lines.append('variables:')
    var_template = '    {0} {1}({2}) ;'
    attr_template = '      {0}:{1} = {2} ;'
    for key, value in group['variables'].iteritems():
        dims = [str(d) for d in value['dimensions']]
        dims = ', '.join(dims)
        lines.append(var_template.format(value['dtype'], key, dims))
        for key2, value2 in value['attributes'].iteritems():
            lines.append(attr_template.format(key, key2, format_attribute_for_dump_report(value2)))

    lines.append('')
    lines.append('// {} attributes:'.format(global_attributes_name))
    template = '    :{0} = {1} ;'
    for key, value in group['global_attributes'].iteritems():
        try:
            lines.append(template.format(key, format_attribute_for_dump_report(value)))
        except UnicodeEncodeError:
            # for a unicode string, if "\u" is in the string and an inappropriate unicode character is used, then
            # template formatting will break.
            msg = 'Unable to encode attribute "{0}". Skipping printing of attribute value.'.format(key)
            warn(msg)

    if indent > 0:
        indent_string = ''
        for _ in range(indent):
            indent_string += ' '
        for idx, current in enumerate(lines):
            if len(current) > 0:
                lines[idx] = indent_string + current

    return lines


def format_attribute_for_dump_report(attr_value):
    if isinstance(attr_value, basestring):
        ret = '"{}"'.format(attr_value)
    else:
        ret = attr_value
    return ret


def allocate_variable_using_metadata(variable, metadata):
    source = metadata
    if variable.parent is not None:
        if variable.parent.parent is not None:
            source = metadata['groups'][variable.parent.name]

    desired_name = variable.name or variable._request_dataset.variable

    var = source['variables'][desired_name]

    if variable._dimensions is None:
        desired_dimensions = var['dimensions']
        new_dimensions = get_dimensions_from_netcdf(source, desired_dimensions)
        super(SourcedVariable, variable)._set_dimensions_(new_dimensions)

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
    exclude = ['add_offset', 'scale_factor']
    for k, v in var['attributes'].items():
        if k in exclude:
            continue
        if k not in variable_attrs:
            variable_attrs[k] = deepcopy(v)

    variable._allocated = True


def get_dimensions_from_netcdf(dataset, desired_dimensions):
    new_dimensions = []
    for dim_name in desired_dimensions:
        dim = dataset['dimensions'][dim_name]
        dim_length = dim['len']
        if dim['isunlimited']:
            length = None
            length_current = dim_length
        else:
            length = dim_length
            length_current = None
        new_dim = SourcedDimension(dim_name, length=length, length_current=length_current)
        new_dimensions.append(new_dim)
    return new_dimensions


def get_value_from_request_dataset(variable):
    if variable.protected:
        raise PayloadProtectedError

    rd = variable._request_dataset

    ds = rd.driver.open()
    source = ds
    if variable.parent is not None:
        if variable.parent.parent is not None:
            source = ds.groups[variable.parent.name]
    desired_name = variable.name or rd.variable
    try:
        ncvar = source.variables[desired_name]
        ret = get_variable_value(ncvar, variable.dimensions)
        return ret
    finally:
        ds.close()


def get_variable_value(variable, dimensions):
    if dimensions is not None and len(dimensions) > 0:
        slc = get_formatted_slice([d._src_idx for d in dimensions], len(dimensions))
    else:
        slc = slice(None)
    ret = variable.__getitem__(slc)
    return ret


def create_dimension_or_pass(dim, dataset):
    if dim.name not in dataset.dimensions:
        dataset.createDimension(dim.name, dim.length)


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


def get_dimension_map_entry(axis, variables):
    axis_vars = []
    for variable in variables.values():
        vattrs = variable['attributes']
        if vattrs.get('axis') == axis:
            axis_vars.append(variable['name'])
    assert len(axis_vars) <= 1
    if len(axis_vars) == 1:
        var_name = axis_vars[0]
        ret = {'variable': var_name, 'names': list(variables[var_name]['dimensions'])}
    else:
        ret = None
    return ret