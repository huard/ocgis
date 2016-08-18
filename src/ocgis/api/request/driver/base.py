import abc
import json
from contextlib import contextmanager
from copy import deepcopy
from warnings import warn

from ocgis.exc import DefinitionValidationError
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.mpi import find_dimension_in_sequence, MPI_COMM
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
        self._metadata = None
        self._dimension_map = None
        self._dimensions = None
        self._crs = None

    def __eq__(self, other):
        return self.key == other.key

    def __str__(self):
        return '"{0}"'.format(self.key)

    @property
    def crs(self):
        if self._crs is None:
            self._crs = self.get_crs() or self._default_crs
        return self._crs

    @property
    def dimensions(self):
        if self._dimensions is None:
            self._dimensions = self.get_dimensions()
        return self._dimensions

    @property
    def dimension_map(self):
        if self._dimension_map is None:
            self._dimension_map = self.get_dimension_map(self.metadata)
        return self._dimension_map

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.get_metadata()
        return self._metadata

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

    def get_crs(self):
        """:rtype: ~ocgis.interface.base.crs.CoordinateReferenceSystem"""
        return None

    def get_dimension_map(self, metadata):
        """:rtype: dict"""
        return {}

    def get_dimensioned_variables(self, dimension_map, metadata):
        """:rtype: tuple(str, ...)"""
        return None

    def get_dimensions(self):
        """
        :return: The dimension distribution object.
        :rtype: :class:`ocgis.new_interface.mpi.OcgMpi`
        """

        # Convert metadata into a grouping consistent with the MPI dimensions.
        target_metdata = {None: self.rd.metadata}
        for group_name in iter_all_group_keys(target_metdata):
            group_meta = get_group(target_metdata, group_name)
            dimensions = self._get_dimensions_main_(group_meta)
            for dimension_name, dimension_meta in group_meta['dimensions'].items():
                target_dimension = find_dimension_in_sequence(dimension_name, dimensions)
                target_dimension.dist = group_meta['dimensions'][dimension_name].get('dist', False)
                self.rd.dist.add_dimension(target_dimension, group=group_name)
        self.rd.dist.update_dimension_bounds()
        return self.rd.dist

    def get_dump_report(self):
        lines = get_dump_report_for_group(self.metadata)
        lines.insert(0, 'OCGIS Driver Key: ' + self.key + ' {')
        for group_name, group_metadata in self.metadata.get('groups', {}).items():
            lines.append('')
            lines.append('group: ' + group_name + ' {')
            lines += get_dump_report_for_group(group_metadata, global_attributes_name=group_name, indent=2)
            lines.append('  }')
        lines.append('}')
        return lines

    @abc.abstractmethod
    def get_variable_value(self, variable):
        """Get value for the variable."""

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

        meta = deepcopy(self.metadata)
        _jsonformat_(meta)
        return json.dumps(meta)

    def get_variable_collection(self):
        """
        :rtype: :class:`ocgis.new_interface.variable.VariableCollection`
        """

        dimension = self.dimensions.get_group()['dimensions'][0]
        ret = VariableCollection()
        for v in self.rd.metadata['variables'].values():
            nvar = SourcedVariable(name=v['name'], dimensions=dimension, dtype=v['dtype'], request_dataset=self.rd)
            ret.add_variable(nvar)
        return ret

    def get_variable_metadata(self, variable_object):
        variable_metadata = get_variable_metadata_from_request_dataset(self, variable_object)
        return variable_metadata

    def get_field(self, *args, **kwargs):
        # tdk: test dimension map overloading
        # Get the raw variable collection from source.
        vc = self.get_variable_collection()

        # If there is a group index, extract the appropriate child for the target field.
        field_group = self.rd.field_group
        if field_group is not None:
            for fg in field_group:
                vc = vc.children[fg]

        # Modify the coordinate system variable. If it is overloaded on the request dataset, then the variable
        # collection needs to be updated to hold the variable and any alternative coordinate systems needs to be
        # removed.
        to_remove = None
        to_add = None
        if self.rd._crs is not None and self.rd._crs != 'auto':
            to_add = self.rd._crs
            to_remove = self.crs.name
        elif self.crs is not None:
            to_add = self.crs
        if to_remove is not None:
            vc.pop(to_remove, None)
        if to_add is not None:
            vc.add_variable(to_add, force=True)

        # Convert the raw variable collection to a field.
        kwargs['dimension_map'] = self.rd.dimension_map
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

        return field

    @abc.abstractmethod
    def get_metadata(self):
        """
        :rtype: dict
        """

    def init_variable_from_source(self, variable):
        variable_metadata = self.get_variable_metadata(variable)

        # Create the dimensions if they are not present.
        if variable._dimensions is None:
            dist = self.dimensions
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

        from ocgis.util.inspect import Inspect

        for line in Inspect(request_dataset=self.rd).get_report_possible():
            print line

    @classmethod
    def open(cls, uri=None, mode='r', rd=None, **kwargs):
        if rd is not None and rd.opened is not None:
            ret = rd.opened
        else:
            if rd is not None and uri is None:
                uri = rd.uri
            ret = cls._open_(uri, mode=mode, **kwargs)
        return ret

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
    def write_variable_collection(cls, vc, opened_or_path, comm=None, **kwargs):
        comm = comm or MPI_COMM
        rank = comm.Get_rank()
        size = comm.Get_size()

        if size > 1:
            if cls.inquire_opened_state(opened_or_path):
                raise ValueError('Only paths allowed for parallel writes.')

        cls._write_variable_collection_main_(vc, opened_or_path, comm, rank, size, **kwargs)

    def _get_dimensions_main_(self, group_metadata):
        """
        :param dict group_metadata: Metadata dictionary for the target group.
        :return: A sequence of dimension objects.
        :rtype: sequence
        """

        gmd = group_metadata['dimensions']
        dims = [Dimension(dref['name'], size=dref['size'], src_idx='auto') for dref in gmd.values()]
        return tuple(dims)

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
    @abc.abstractmethod
    def _open_(uri, mode='r', **kwargs):
        """
        :rtype: object
        """

        return open(uri, mode=mode, **kwargs)

    @classmethod
    @abc.abstractmethod
    def _write_variable_collection_main_(cls, vc, opened_or_path, comm, rank, size):
        """
        :param vc: :class:`~ocgis.new_interface.variable.VariableCollection`
        :param opened_or_path: Opened file object or path to the file object to open.
        :param comm: The MPI communicator.
        :param rank: The MPI rank.
        :param size: The MPI size.
        """


def format_attribute_for_dump_report(attr_value):
    if isinstance(attr_value, basestring):
        ret = '"{}"'.format(attr_value)
    else:
        ret = attr_value
    return ret


def get_dump_report_for_group(group, global_attributes_name='global', indent=0):
    lines = ['dimensions:']
    template = '    {0} = {1} ;{2}'
    for key, value in group['dimensions'].iteritems():
        if value.get('isunlimited', False):
            one = 'ISUNLIMITED'
            two = ' // {0} currently'.format(value['size'])
        else:
            one = value['size']
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
        for key2, value2 in value.get('attributes', {}).iteritems():
            lines.append(attr_template.format(key, key2, format_attribute_for_dump_report(value2)))

    lines.append('')
    lines.append('// {} attributes:'.format(global_attributes_name))
    template = '    :{0} = {1} ;'
    for key, value in group.get('global_attributes', {}).iteritems():
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
    return get_group(driver.rd.metadata, variable.group, has_root=False)['variables'][variable.name]


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


@contextmanager
def driver_scope(driver, opened_or_path=None, mode='r', **kwargs):
    if opened_or_path is None:
        try:
            # Attempt to get the request dataset from the driver. If not there, assume we are working with the driver
            # class and not an instance created with a request dataset.
            rd = driver.rd
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

    if driver.inquire_opened_state(opened_or_path):
        should_close = False
    else:
        should_close = True
        opened_or_path = driver.open(uri=opened_or_path, mode=mode, rd=rd, **kwargs)

    try:
        yield opened_or_path
    finally:
        if should_close:
            driver.close(opened_or_path)
