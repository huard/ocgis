import abc
import json
from contextlib import contextmanager
from copy import deepcopy
from warnings import warn

from ocgis.constants import MPIDistributionMode, MPIWriteMode
from ocgis.exc import DefinitionValidationError, OcgMpiError
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.mpi import find_dimension_in_sequence, MPI_COMM, OcgMpi
from ocgis.new_interface.variable import SourcedVariable, VariableCollection
from ocgis.util.logging_ocgis import ocgis_lh


class AbstractDriver(object):
    """
    :param rd: The input request dataset object.
    :type rd: :class:`~ocgis.RequestDataset`
    """

    __metaclass__ = abc.ABCMeta

    _default_crs = None
    _priority = False

    def __init__(self, rd):
        self.rd = rd
        self._metadata_raw = None
        self._dist = None
        self._dimension_map_raw = None

    def __eq__(self, other):
        return self.key == other.key

    def __str__(self):
        return '"{0}"'.format(self.key)

    # tdk: remove me
    @property
    def crs(self):
        raise NotImplementedError

    @property
    def dimension_map_raw(self):
        if self._dimension_map_raw is None:
            self._dimension_map_raw = get_dimension_map_raw(self, self.metadata_raw)
        return self._dimension_map_raw

    @property
    def dist(self):
        if self._dist is None:
            self._dist = self.get_dist()
        return self._dist

    @property
    def metadata_raw(self):
        if self._metadata_raw is None:
            self._metadata_raw = self.get_metadata()
        return self._metadata_raw

    @property
    def metadata_source(self):
        return self.rd.metadata

    @abc.abstractproperty
    def extensions(self):
        """
        :returns: A sequence of regular expressions used to match appropriate URIs.
        :rtype: (str,)

        >>> ('.*\.shp',)
        """

    @abc.abstractproperty
    def key(self):
        """:rtype: str"""

    @abc.abstractproperty
    def output_formats(self):
        """
        :returns: A list of acceptable output formats for the driver. If this is `'all'`, then the driver's data may be
         converted to all output formats.
        :rtype: list[str, ...]
        """

    @classmethod
    def close(cls, obj, rd=None):
        # If the request dataset has an opened file object, do not close the file as we expect the client to handle
        # closing/finalization options.
        if rd is not None and rd.opened is not None:
            pass
        else:
            cls._close_(obj)

    def get_crs(self, group_metadata):
        """:rtype: ~ocgis.interface.base.crs.CoordinateReferenceSystem"""

        crs = self._get_crs_main_(group_metadata)
        if crs is None:
            ret = self._default_crs
        else:
            ret = crs
        return ret

    def get_dimension_map(self, group_metadata):
        """:rtype: dict"""
        return {}

    def get_dimensioned_variables(self, dimension_map, metadata):
        """:rtype: tuple(str, ...)"""
        return None

    def get_dist(self):
        """
        :return: The dimension distribution object.
        :rtype: :class:`ocgis.new_interface.mpi.OcgMpi`
        """
        # Allow the request dataset to overload the distribution
        if self.rd.dist is not None:
            # Ensure the distribution is updated.
            assert self.rd.dist.has_updated_dimensions
            return self.rd.dist
        # Otherwise, create the template distribution object.
        else:
            dist = OcgMpi(size=self.rd.comm.Get_size())

        # Convert metadata into a grouping consistent with the MPI dimensions.
        metadata = {None: self.metadata_source}
        for group_index in iter_all_group_keys(metadata):
            group_meta = get_group(metadata, group_index)

            # Add the dimensions to the distribution object.
            dimensions = self._get_dimensions_main_(group_meta)
            for dimension_name, dimension_meta in group_meta['dimensions'].items():
                target_dimension = find_dimension_in_sequence(dimension_name, dimensions)
                target_dimension.dist = group_meta['dimensions'][dimension_name].get('dist', False)
                dist.add_dimension(target_dimension, group=group_index)

            # Configure the default distribution.
            if self.rd.use_default_dist:
                dimension_map = get_group(self.rd.dimension_map, group_index, has_root=False)
                distributed_dimension_name = self.get_distributed_dimension_name(dimension_map,
                                                                                 group_meta['dimensions'])
                # Allow no distributed dimensions to be returned.
                if distributed_dimension_name is not None:
                    distributed_dimension = dist.get_dimension(distributed_dimension_name, group=group_index)
                    distributed_dimension.dist = True

            # Add the variables to the distribution object.
            for variable_name, variable_meta in group_meta['variables'].items():
                # If the variable has any distributed dimensions, the variable is always distributed. Otherwise, allow
                # the variable to determine if it is replicated or isolated. Default is replicated.
                ranks = 'all'
                variable_dist = variable_meta.get('dist')
                if any([dim.dist for dim in dist.get_dimensions(variable_meta['dimensions'], group=group_index)]):
                    if variable_dist is not None and variable_dist != MPIDistributionMode.DISTRIBUTED:
                        msg = 'The replicated or isolated variable "{}" must not have distributed dimensions.'
                        raise OcgMpiError(msg.format(variable_name))
                    variable_dist = MPIDistributionMode.DISTRIBUTED
                else:
                    variable_dist = variable_dist or MPIDistributionMode.REPLICATED
                    # Distributed variables require distributed dimensions.
                    if variable_dist == MPIDistributionMode.DISTRIBUTED:
                        msg = 'The distributed variable "{}" requires distributed dimensions.'
                        raise OcgMpiError(msg.format(variable_name))
                    # If the variable is isolated, identify the ranks it should be on.
                    if variable_dist == MPIDistributionMode.ISOLATED:
                        ranks = variable_meta.get('ranks', (0,))
                dist.add_variable(variable_name, dimensions=variable_meta['dimensions'], group=group_index,
                                  dist=variable_dist, ranks=ranks)

        # tdk: this will have to be moved to account for slicing
        dist.update_dimension_bounds()
        return dist

    def get_distributed_dimension_name(self, dimension_map, dimensions_metadata):
        """Return the preferred distributed dimension name."""
        return None

    def get_dump_report(self, indent=0, group_metadata=None, first=True, global_attributes_name='global'):
        lines = []
        if first:
            lines.append('OCGIS Driver Key: ' + self.key + ' {')
            group_metadata = group_metadata or self.metadata_source
        else:
            indent += 2
        lines += get_dump_report_for_group(group_metadata, global_attributes_name=global_attributes_name, indent=indent)
        for group_name, group_metadata in group_metadata.get('groups', {}).items():
            lines.append('')
            lines.append(' ' * indent + 'group: ' + group_name + ' {')
            dump_lines = self.get_dump_report(group_metadata=group_metadata, first=False, indent=indent,
                                              global_attributes_name=group_name)
            lines += dump_lines
            lines.append(' ' * indent + '  }' + ' // group: {}'.format(group_name))
        if first:
            lines.append('}')
        return lines

    def get_field(self, *args, **kwargs):
        # tdk: test dimension map overloading
        vc = kwargs.pop('vc', None)
        if vc is None:
            # Get the raw variable collection from source.
            vc = self.get_variable_collection()

        # Get the appropriate metadata for the collection.
        group_metadata = self.get_group_metadata(vc.group, self.metadata_source)
        # Always pull the dimension map from the request dataset. This allows it to be overloaded.
        dimension_map = self.get_group_metadata(vc.group, self.rd.dimension_map)

        # # If there is a group index, extract the appropriate child for the target field.
        # field_group = self.rd.field_group
        # if field_group is not None:
        #     for fg in field_group:
        #         vc = vc.children[fg]

        # Modify the coordinate system variable. If it is overloaded on the request dataset, then the variable
        # collection needs to be updated to hold the variable and any alternative coordinate systems needs to be
        # removed.
        to_remove = None
        to_add = None
        crs = self.get_crs(group_metadata)
        if self.rd._crs is not None and self.rd._crs != 'auto':
            to_add = self.rd._crs
            if crs is not None:
                to_remove = crs.name
        elif crs is not None:
            to_add = crs
        if to_remove is not None:
            vc.pop(to_remove, None)
        if to_add is not None:
            vc.add_variable(to_add, force=True)

        # Convert the raw variable collection to a field.
        kwargs['dimension_map'] = dimension_map
        field = OcgField.from_variable_collection(vc, *args, **kwargs)

        # If this is a source grid for regridding, ensure the flag is updated.
        if self.rd.regrid_source:
            field._should_regrid = True
        # Update the assigned coordinate system flag.
        if self.rd._has_assigned_coordinate_system:
            field._has_assigned_coordinate_system = True

        # tdk: incorporate spatial subset
        # Apply any requested subsets.
        if self.rd.time_range is not None:
            field = field.time.get_between(*self.rd.time_range).parent
        if self.rd.time_region is not None:
            field = field.time.get_time_region(self.rd.time_region).parent
        if self.rd.time_subset_func is not None:
            field = field.time.get_subset_by_function(self.rd.time_subset_func).parent
        if self.rd.level_range is not None:
            field = field.level.get_between(*self.rd.level_range).parent

        for child in field.children.values():
            kwargs['vc'] = child
            field.children[child.name] = self.get_field(*args, **kwargs)

        return field

    @staticmethod
    def get_group_metadata(group_index, metadata, has_root=False):
        return get_group(metadata, group_index, has_root=has_root)

    def get_metadata(self):
        """
        :rtype: dict
        """

        return self._get_metadata_main_()

    def get_source_metadata_as_json(self):
        # tdk: test

        def _jsonformat_(d):
            for k, v in d.iteritems():
                if isinstance(v, dict):
                    _jsonformat_(v)
                else:
                    try:
                        v = v.tolist()
                    except AttributeError:
                        # NumPy arrays need to be converted to lists.
                        pass
                d[k] = v

        meta = deepcopy(self.metadata_source)
        _jsonformat_(meta)
        return json.dumps(meta)

    def get_variable_collection(self):
        """
        :rtype: :class:`ocgis.new_interface.variable.VariableCollection`
        """

        dimension = self.dist.get_group()['dimensions'].values()[0]
        ret = VariableCollection()
        for v in self.metadata_source['variables'].values():
            nvar = SourcedVariable(name=v['name'], dimensions=dimension, dtype=v['dtype'], request_dataset=self.rd)
            ret.add_variable(nvar)
        return ret

    def get_variable_metadata(self, variable_object):
        variable_metadata = get_variable_metadata_from_request_dataset(self, variable_object)
        return variable_metadata

    @classmethod
    def get_variable_for_writing(cls, variable):
        """
        Allows variables to overload which member to use for writing. For example, temporal variables always want
        numeric times.
        """
        from ocgis.new_interface.temporal import TemporalVariable
        if isinstance(variable, TemporalVariable):
            ret = cls.get_variable_for_writing_temporal(variable)
        else:
            ret = variable
        return ret

    @classmethod
    def get_variable_for_writing_temporal(cls, temporal_variable):
        # Only return datetime objects directly if time is being formatted. Otherwise return the standard value, these
        # could be datetime objects if passed during initialization.
        if temporal_variable.format_time:
            ret = temporal_variable.value_datetime
        else:
            ret = temporal_variable.value
        return ret

    @abc.abstractmethod
    def get_variable_value(self, variable):
        """Get value for the variable."""

    @classmethod
    def get_variable_write_dtype(cls, variable):
        return cls.get_variable_for_writing(variable).dtype

    @classmethod
    def get_variable_write_fill_value(cls, variable):
        return cls.get_variable_for_writing(variable).fill_value

    @classmethod
    def get_variable_write_value(cls, variable):
        from ocgis.new_interface.temporal import TemporalVariable
        if isinstance(variable, TemporalVariable):
            ret = cls.get_variable_for_writing(variable)
        else:
            ret = cls.get_variable_for_writing(variable).masked_value
        return ret

    def init_variable_from_source(self, variable):
        variable_metadata = self.get_variable_metadata(variable)

        # Create the dimensions if they are not present.
        if variable._dimensions is None:
            dist = self.dist
            desired_dimensions = variable_metadata['dimensions']
            new_dimensions = []
            for d in desired_dimensions:
                to_append = dist.get_dimension(d, group=variable.group)
                new_dimensions.append(to_append)
            super(SourcedVariable, variable)._set_dimensions_(new_dimensions)

        # Call the subclass variable initialization routine.
        self._init_variable_from_source_main_(variable, variable_metadata)
        # The variable is now allocated.
        variable._allocated = True

    def init_variable_value(self, variable):
        """Set the variable value from source data conforming units in the process."""

        value = self.get_variable_value(variable)
        variable._set_value_(value)
        # Conform the units if requested. Need to check if this variable is inside a group to find the appropriate
        # metadata.
        meta = get_variable_metadata_from_request_dataset(self, variable)
        conform_units_to = meta.get('conform_units_to')
        if conform_units_to is not None:
            variable.cfunits_conform(conform_units_to)

    @staticmethod
    def inquire_opened_state(opened_or_path):
        """
        Return ``True`` if the input is an opened file object.

        :param opened_or_path: Output file path or an open file object.
        :rtype: bool
        """
        if isinstance(opened_or_path, (basestring, tuple, list)):
            ret = False
        else:
            ret = True
        return ret

    def inspect(self):
        """
        Inspect the request dataset printing information to stdout.
        """

        for line in self.get_dump_report():
            print line

    @classmethod
    def open(cls, uri=None, mode='r', rd=None, **kwargs):
        if uri is None and rd is None:
            raise ValueError('A URI or request dataset is required.')

        if rd is not None and rd.opened is not None:
            ret = rd.opened
        else:
            if rd is not None and uri is None:
                uri = rd.uri
            ret = cls._open_(uri, mode=mode, **kwargs)
        return ret

    def validate_field(self, field):
        pass

    @classmethod
    def validate_ops(cls, ops):
        """
        :param ops: An operation object to validate.
        :type ops: :class:`~ocgis.OcgOperations`
        :raises: DefinitionValidationError
        """

        if cls.output_formats != 'all':
            if ops.output_format not in cls.output_formats:
                msg = 'Output format not supported for driver "{0}". Supported output formats are: {1}'.format(cls.key, cls.output_formats)
                ocgis_lh(logger='driver', exc=DefinitionValidationError('output_format', msg))

    def write_gridxy(self, *args, **kwargs):
        """Write a grid."""
        raise NotImplementedError

    def write_variable(self, *args, **kwargs):
        """Write a variable. Not applicable for tabular formats."""
        raise NotImplementedError

    @classmethod
    def write_variable_collection(cls, vc, opened_or_path, **kwargs):
        comm = kwargs.pop('comm', None) or MPI_COMM
        rank = comm.Get_rank()
        size = comm.Get_size()

        write_mode = kwargs.pop('write_mode', None)

        if size > 1:
            if cls.inquire_opened_state(opened_or_path):
                raise ValueError('Only paths allowed for parallel writes.')

        if write_mode is None:
            if size > 1:
                write_modes = [MPIWriteMode.TEMPLATE, MPIWriteMode.FILL]
            else:
                write_modes = [MPIWriteMode.NORMAL]
        else:
            write_modes = [write_mode]

        for write_mode in write_modes:
            cls._write_variable_collection_main_(vc, opened_or_path, comm, rank, size, write_mode, **kwargs)
            # MPI_COMM.Barrier()

    def _get_crs_main_(self, group_metadata):
        """Return the coordinate system variable or None if not found."""
        return None

    def _get_dimensions_main_(self, group_metadata):
        """
        :param dict group_metadata: Metadata dictionary for the target group.
        :return: A sequence of dimension objects.
        :rtype: sequence
        """

        gmd = group_metadata['dimensions']
        dims = [Dimension(dref['name'], size=dref['size'], src_idx='auto') for dref in gmd.values()]
        return tuple(dims)

    @abc.abstractmethod
    def _get_metadata_main_(self):
        """
        Return the base metadata object. The superclass will finalize the metadata doing things like adding dimension
        maps for each group.

        :rtype: dict
        """

    @staticmethod
    def _close_(obj):
        """
        Close and finalize the open file object.
        """

        obj.close()

    @abc.abstractmethod
    def _init_variable_from_source_main_(self, variable_object, variable_metadata):
        """Initialize everything but dimensions on the target variable."""

    @staticmethod
    def _open_(uri, mode='r', **kwargs):
        """
        :rtype: object
        """

        return open(uri, mode=mode, **kwargs)

    @classmethod
    @abc.abstractmethod
    def _write_variable_collection_main_(cls, vc, opened_or_path, comm, rank, size, write_mode, **kwargs):
        """
        :param vc: :class:`~ocgis.new_interface.variable.VariableCollection`
        :param opened_or_path: Opened file object or path to the file object to open.
        :param comm: The MPI communicator.
        :param rank: The MPI rank.
        :param size: The MPI size.
        """


@contextmanager
def driver_scope(ocgis_driver, opened_or_path=None, mode='r', **kwargs):
    if opened_or_path is None:
        try:
            # Attempt to get the request dataset from the driver. If not there, assume we are working with the driver
            # class and not an instance created with a request dataset.
            rd = ocgis_driver.rd
        except AttributeError:
            rd = None
        if rd is None:
            raise ValueError('Without a driver instance and no open object or file path, nothing can be scoped.')
        else:
            if rd.opened is not None:
                opened_or_path = rd.opened
            else:
                opened_or_path = rd.uri
    else:
        rd = None

    if ocgis_driver.inquire_opened_state(opened_or_path):
        should_close = False
    else:
        should_close = True
        opened_or_path = ocgis_driver.open(uri=opened_or_path, mode=mode, rd=rd, **kwargs)

    try:
        yield opened_or_path
    finally:
        if should_close:
            ocgis_driver.close(opened_or_path)


def find_variable_by_attribute(variables_metadata, attribute_name, attribute_value):
    ret = []
    for variable_name, variable_metadata in variables_metadata.items():
        for k, v in variable_metadata['attributes'].items():
            if k == attribute_name and v == attribute_value:
                ret.append(variable_name)
    return ret


def format_attribute_for_dump_report(attr_value):
    if isinstance(attr_value, basestring):
        ret = '"{}"'.format(attr_value)
    else:
        ret = attr_value
    return ret


def get_dimension_map_raw(driver, group_metadata, group_name=None, update_target=None):
    dimension_map = driver.get_dimension_map(group_metadata)
    if update_target is None:
        update_target = dimension_map
    if 'groups' not in dimension_map:
        dimension_map['groups'] = {}
    if group_name is not None:
        update_target['groups'][group_name] = dimension_map
    if 'groups' in group_metadata:
        for group_name, sub_group_metadata in group_metadata['groups'].items():
            get_dimension_map_raw(driver, sub_group_metadata, group_name=group_name,
                                  update_target=update_target['groups'])
    return update_target


def get_dump_report_for_group(group, global_attributes_name='global', indent=0):
    lines = []

    if len(group['dimensions']) > 0:
        lines.append('dimensions:')
        template = '    {0} = {1} ;{2}'
        for key, value in group['dimensions'].items():
            if value.get('isunlimited', False):
                one = 'ISUNLIMITED'
                two = ' // {0} currently'.format(value['size'])
            else:
                one = value['size']
                two = ''
            lines.append(template.format(key, one, two))

    if len(group['variables']) > 0:
        lines.append('variables:')
        var_template = '    {0} {1}({2}) ;'
        attr_template = '      {0}:{1} = {2} ;'
        for key, value in group['variables'].items():
            dims = [str(d) for d in value['dimensions']]
            dims = ', '.join(dims)
            lines.append(var_template.format(value['dtype'], key, dims))
            for key2, value2 in value.get('attributes', {}).iteritems():
                lines.append(attr_template.format(key, key2, format_attribute_for_dump_report(value2)))

    global_attributes = group.get('global_attributes', {})
    if len(global_attributes) > 0:
        lines.append('')
        lines.append('// {} attributes:'.format(global_attributes_name))
        template = '    :{0} = {1} ;'
        for key, value in global_attributes.items():
            try:
                lines.append(template.format(key, format_attribute_for_dump_report(value)))
            except UnicodeEncodeError:
                # for a unicode string, if "\u" is in the string and an inappropriate unicode character is used, then
                # template formatting will break.
                msg = 'Unable to encode attribute "{0}". Skipping printing of attribute value.'.format(key)
                warn(msg)

    if indent > 0:
        indent_string = ' ' * indent
        for idx, current in enumerate(lines):
            if len(current) > 0:
                lines[idx] = indent_string + current

    return lines


def get_group(ddict, keyseq, has_root=True):
    keyseq = deepcopy(keyseq)

    if keyseq is None:
        keyseq = [None]
    elif isinstance(keyseq, basestring):
        keyseq = [keyseq]

    if keyseq[0] is not None:
        keyseq.insert(0, None)

    curr = ddict
    for key in keyseq:
        if key is None:
            if has_root:
                curr = curr[None]
        else:
            curr = curr['groups'][key]
    return curr


def get_variable_metadata_from_request_dataset(driver, variable):
    return get_group(driver.metadata_source, variable.group, has_root=False)['variables'][variable.name]


def iter_all_group_keys(ddict, entry=None):
    if entry is None:
        entry = [None]
    yield entry
    for keyseq in iter_group_keys(ddict, entry):
        for keyseq2 in iter_all_group_keys(ddict, keyseq):
            yield keyseq2


def iter_group_keys(ddict, keyseq):
    for key in get_group(ddict, keyseq).get('groups', {}):
        yld = deepcopy(keyseq)
        yld.append(key)
        yield yld
