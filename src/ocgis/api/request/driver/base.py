import abc
import json
from copy import deepcopy

from ocgis.exc import DefinitionValidationError
from ocgis.new_interface.field import OcgField
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

    def close(self, obj):
        if self.rd.opened is None:
            self._close_(obj)
        else:
            pass

    # tdk: order
    @abc.abstractmethod
    def _close_(self, obj):
        """
        Close and finalize the open file object.
        """

    def get_crs(self):
        """:rtype: ~ocgis.interface.base.crs.CoordinateReferenceSystem"""
        return None

    def get_dimension_map(self, metadata):
        """:rtype: dict"""
        return {}

    def get_dimensioned_variables(self, dimension_map, metadata):
        """:rtype: tuple(str, ...)"""
        return None

    @abc.abstractmethod
    def get_dimensions(self):
        """
        :return: A dimension object dictionary. The key is the dimension name. The value is the dimension object. This
         method accounts for any MPI distribution.
        :rtype: dict
        """

        dimensions = self._get_dimensions_main_()
        for dim in dimensions.values():
            self.rd.mpi.add_dimension(dim)
        return self.rd.mpi.get_group()

    @abc.abstractmethod
    def get_dump_report(self):
        """
        :returns: A sequence of strings containing the metadata dump from the source request dataset.
        :rtype: list[str, ...]
        """

    def initialize_variable_value(self, variable):
        """Set the variable value from source data conforming units in the process."""

        value = self.get_variable_value(variable)
        variable._set_value_(value)
        # Conform the units if requested. Need to check if this variable is inside a group to find the appropriate
        # metadata.
        meta = get_variable_metadata_from_request_dataset(self, variable)
        conform_units_to = meta.get('conform_units_to')
        if conform_units_to is not None:
            variable.cfunits_conform(conform_units_to)

    def allocate_variable_value(self, variable):
        raise NotImplementedError

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

    @abc.abstractmethod
    def get_variable_collection(self):
        """
        :rtype: :class:`ocgis.new_interface.variable.VariableCollection`
        """

    def get_field(self, *args, **kwargs):
        # tdk: test dimension map overloading
        # Get the raw variable collection from source.
        vc = self.get_variable_collection()

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

    def inspect(self):
        """
        Inspect the request dataset printing information to stdout.
        """

        from ocgis.util.inspect import Inspect

        for line in Inspect(request_dataset=self.rd).get_report_possible():
            print line

    def open(self, *args, **kwargs):
        if self.rd.opened is None:
            ret = self._open_(*args, **kwargs)
        else:
            ret = self.rd.opened
        return ret

    # tdk: order
    @abc.abstractmethod
    def _open_(self):
        """
        :rtype: object
        """

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

    @abc.abstractmethod
    def write_gridxy(self, *args, **kwargs):
        """Write a grid."""

    @abc.abstractmethod
    def write_variable(self, *args, **kwargs):
        """Write a variable."""

    @abc.abstractmethod
    def write_variable_collection(self, *args, **kwargs):
        """Write a variable collection to file(s)."""

    @abc.abstractmethod
    def _get_dimensions_main_(self):
        """
        :return: A dimension object dictionary. The key is the dimension name. The value is the dimension object.
        :rtype: dict
        """


def get_variable_metadata_from_request_dataset(driver, variable):
    to_check = variable.parent
    current_meta = driver.rd.metadata['groups']
    ctr = 0
    while to_check is not None:
        try:
            current_meta = current_meta[to_check.name]
        except KeyError:
            to_check = None
        else:
            to_check = to_check.parent
            ctr += 1
    if ctr > 0:
        meta = current_meta
    else:
        meta = driver.rd.metadata

    return meta['variables'][variable.name]
