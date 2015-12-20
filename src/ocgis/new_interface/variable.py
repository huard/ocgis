from copy import copy, deepcopy
from itertools import izip
from netCDF4 import Dataset

import numpy as np
from numpy.ma import MaskedArray

from ocgis import constants
from ocgis.api.collection import AbstractCollection
from ocgis.exc import VariableInCollectionError, BoundsAlreadyAvailableError, EmptySubsetError, \
    ResolutionError, NoUnitsError
from ocgis.interface.base.attributes import Attributes
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.dimension import Dimension, SourcedDimension
from ocgis.util.helpers import get_iter, get_formatted_slice, get_bounds_from_1d
from ocgis.util.units import get_units_object, get_conformed_units


class Variable(AbstractInterfaceObject, Attributes):
    # tdk:doc
    # tdk:support unlimited dimensions

    def __init__(self, name=None, value=None, dimensions=None, dtype=None, alias=None, attrs=None, fill_value=None,
                 units=None):
        self._is_init = True
        self._alias = None
        self._dimensions = None
        self._value = None
        self._dtype = None
        self._fill_value = None
        self._units = None

        self.name = name
        self.units = units
        self.dtype = dtype
        self.fill_value = fill_value
        self.alias = alias
        self.dimensions = dimensions
        self.value = value

        Attributes.__init__(self, attrs=attrs)

        self._is_init = False

        if self._value is not None:
            self._validate_value_(self._value)
            # Update any unlimited dimension length.
            if self._dimensions is not None:
                for idx, d in enumerate(self.dimensions):
                    if d.length is None and d.length_current is None:
                        d.length_current = self.shape[idx]

    def __getitem__(self, slc):
        ret = copy(self)
        slc = get_formatted_slice(slc, self.ndim)
        ret.dimensions = [d[s] for d, s in izip(self.dimensions, get_iter(slc, dtype=slice))]
        if self._value is not None:
            ret.value = self.value.__getitem__(slc)
            ret.value.unshare_mask()
        return ret

    def __len__(self):
        return self.shape[0]

    @property
    def alias(self):
        if self._alias is None:
            ret = self.name
        else:
            ret = self._alias
        return ret

    @alias.setter
    def alias(self, value):
        self._alias = value

    @property
    def cfunits(self):
        return get_units_object(self.units)

    @property
    def dtype(self):
        if self._value is not None:
            ret = self._value.dtype
        else:
            ret = self._dtype
        return ret

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def dimensions(self):
        if self._dimensions is None:
            try:
                self._dimensions = self._get_dimensions_()
            except NotImplementedError:
                self.create_dimensions()
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        self._set_dimensions_(value)

    def _get_dimensions_(self):
        raise NotImplementedError

    def _set_dimensions_(self, value):
        if value is not None:
            value = tuple(get_iter(value, dtype=Dimension))
        self._dimensions = value

    @property
    def extent(self):
        target = self._get_extent_target_()
        return target.compressed().min(), target.compressed().max()

    def _get_extent_target_(self):
        return self.value

    @property
    def fill_value(self):
        if self._value is not None:
            ret = self._value.fill_value
        else:
            ret = self._fill_value
        return ret

    @fill_value.setter
    def fill_value(self, value):
        self._fill_value = value

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def resolution(self):
        # tdk: test
        # tdk: not sure where this belongs exactly. maybe on a value dimension?
        if self.value.shape[0] < 2:
            msg = 'With only a single coordinate, approximate resolution may not be determined.'
            raise ResolutionError(msg)
        res_array = np.diff(self.value.data[0:constants.RESOLUTION_LIMIT])
        ret = np.abs(res_array).mean()
        return ret

    @property
    def shape(self):
        return self._get_shape_()

    @property
    def shape_dimensions(self):
        return tuple([d.length for d in get_iter(self.dimensions, dtype=Dimension)])

    def _get_shape_(self):
        return get_shape_from_variable(self)

    @property
    def units(self):
        if self._units is None:
            self._units = self._get_units_()
        return self._units

    @units.setter
    def units(self, value):
        self._set_units_(value)

    def _get_units_(self):
        return None

    def _set_units_(self, value):
        if value is not None:
            value = str(value)
        self._units = value

    @property
    def value(self):
        if self._value is None:
            self._value = self._get_value_()
        return self._value

    @value.setter
    def value(self, value):
        self._set_value_(value)

    def _get_value_(self):
        return self._value

    def _set_value_(self, value):
        if value is not None:
            if not isinstance(value, MaskedArray):
                value = np.ma.array(value, dtype=self._dtype, fill_value=self._fill_value, mask=False)
            self._validate_value_(value)
        self._value = value

    def _validate_value_(self, value):
        if not self._is_init:
            dimensions = self.dimensions
            # Account for unlimited dimensions.
            value_shape = value.shape
            dim_shape = self.shape_dimensions
            for idx, d in enumerate(dimensions):
                if d.length is not None:
                    assert value_shape[idx] == dim_shape[idx]

    def cfunits_conform(self, to_units, from_units=None):
        """
        Conform value units in-place. If there are scale or offset parameters in the attribute dictionary, they will
        be removed.

        :param to_units: Target conform units.
        :type to_units: str or units object
        :param from_units: Overload source units.
        :type from_units: str or units object
        :raises: NoUnitsError
        """
        if from_units is None and self.units is None:
            raise NoUnitsError(self.name)

        # Use overloaded value for source units.
        from_units = self.cfunits if from_units is None else from_units

        # Get the conform value before swapping the units. Conversion inside time dimensions may be negatively affected
        # otherwise.
        to_conform_value = self._get_to_conform_value_()

        # Update the units attribute with the destination units. Do this before conversion to not enter recursion when
        # setting the new value.
        self.units = to_units

        # Conform the units.
        new_value = get_conformed_units(to_conform_value, from_units, to_units)
        self._set_to_conform_value_(new_value)

        # Let the data type load from the value array.
        self._dtype = None
        # Remove any compression attributes if present.
        for remove in constants.NETCDF_ATTRIBUTES_TO_REMOVE_ON_VALUE_CHANGE:
            self.attrs.pop(remove, None)

    def create_dimensions(self, names=None):
        value = self._value
        if value is None:
            new_dimensions = ()
        else:
            if names is None:
                assert self.name is not None
                names = [self.name]
                for ii in range(1, value.ndim):
                    names.append('{0}_{1}'.format(self.name, ii))
            new_dimensions = []
            for name, shp in izip(get_iter(names), value.shape):
                new_dimensions.append(Dimension(name, length=shp))
        self.dimensions = new_dimensions

    def get_mask(self):
        """Return the object mask."""
        return self.value.mask

    def set_mask(self, value):
        """Set the object mask."""
        assert value.ndim == self.ndim
        self.value.mask = value

    def write_netcdf(self, dataset, **kwargs):
        """
        Write the field object to an open netCDF dataset object.

        :param dataset: The open dataset object or path for the write.
        :type dataset: :class:`netCDF4.Dataset` or str
        :param bool file_only: If ``True``, we are not filling the value variables. Only the file schema and dimension
         values will be written.
        :param bool unlimited_to_fixedsize: If ``True``, convert the unlimited dimension to fixed size.
        :param kwargs: Extra keyword arguments in addition to ``dimensions`` and ``fill_value`` to pass to
         ``createVariable``. See http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html#createVariable
        """

        file_only = kwargs.pop('file_only', False)
        unlimited_to_fixedsize = kwargs.pop('unlimited_to_fixedsize', False)

        if self.dimensions is not None:
            dimensions = list(self.dimensions)
            # Convert the unlimited dimension to fixed size if requested.
            for idx, d in enumerate(dimensions):
                if d.length is None and unlimited_to_fixedsize:
                    dimensions[idx] = Dimension(d.name, length=self.shape[idx])
                    break
            # Create the dimensions.
            for dim in dimensions:
                create_dimension_or_pass(dim, dataset)
            dimensions = [d.name for d in dimensions]
        else:
            dimensions = []
        # Only use the fill value if something is masked.
        if len(dimensions) > 0 and not file_only and self.get_mask().any():
            fill_value = self.fill_value
        else:
            # Copy from original attributes.
            if '_FillValue' not in self.attrs:
                fill_value = None
            else:
                fill_value = self.fill_value
        var = dataset.createVariable(self.alias, self.dtype, dimensions=dimensions, fill_value=fill_value, **kwargs)
        if not file_only:
            var[:] = self.value
        self.write_attributes_to_netcdf_object(var)
        if self.units is not None:
            var.units = self.units
        dataset.sync()

    def _get_to_conform_value_(self):
        return self.value

    def _set_to_conform_value_(self, value):
        self.value = value


class SourcedVariable(Variable):
    def __init__(self, *args, **kwargs):
        self._conform_units_to = None

        # Flag to indicate if metadata already from source.
        self._allocated = False

        self.conform_units_to = kwargs.pop('conform_units_to', None)
        self._request_dataset = kwargs.pop('request_dataset', None)

        super(SourcedVariable, self).__init__(*args, **kwargs)

        if self._value is None and self._request_dataset is None:
            msg = 'A "value" or "request_dataset" is required.'
            raise ValueError(msg)
        if self._value is not None and self._conform_units_to is not None:
            msg = '"conform_units_to" only applicable when loading from source.'
            raise ValueError(msg)

    def __getitem__(self, slc):
        if self._value is None:
            slc = get_formatted_slice(slc, len(self.dimensions))
            ret = copy(self)
            ret.dimensions = [d[s] for d, s in izip(self.dimensions, get_iter(slc, dtype=slice))]
        else:
            ret = super(SourcedVariable, self).__getitem__(slc)
        return ret

    @property
    def conform_units_to(self):
        return self._conform_units_to

    @conform_units_to.setter
    def conform_units_to(self, value):
        if value is not None:
            value = get_units_object(value)
        self._conform_units_to = value

    @property
    def dtype(self):
        if self._dtype is None and not self._allocated:
            if self._value is None:
                self._set_metadata_from_source_()
            else:
                return self._value.dtype
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def fill_value(self):
        if self._fill_value is None and not self._allocated:
            if self._value is None:
                self._set_metadata_from_source_()
            else:
                return self._value.fill_value
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value):
        self._fill_value = value

    def _get_dimensions_(self):
        if self._request_dataset is None:
            ret = super(SourcedVariable, self)._get_dimensions_()
        else:
            self._set_metadata_from_source_()
            ret = self._dimensions
        return ret

    def _set_metadata_from_source_(self):
        if self._allocated:
            raise ValueError('Variable metadata already read from source.')
        else:
            set_variable_metadata_from_source(self)

    def _set_metadata_from_source_finalize_(self, *args, **kwargs):
        pass

    def _get_units_(self):
        if not self._allocated and self._units is None and self._request_dataset is not None:
            self._set_metadata_from_source_()
        return self._units

    def _get_value_(self):
        if self._value is None:
            value = self._get_value_from_source_()
            super(SourcedVariable, self)._set_value_(value)
        return super(SourcedVariable, self)._get_value_()

    def _get_value_from_source_(self):
        ds = self._request_dataset.driver.open()
        try:
            dimensions = self.dimensions
            if len(dimensions) > 0:
                slc = get_formatted_slice([d._src_idx for d in dimensions], len(self.shape))
            else:
                slc = slice(None)
            var = ds.variables[self.name]
            ret = var.__getitem__(slc)

            # Conform the units if requested.
            if self.conform_units_to is not None:
                ret = get_conformed_units(ret, self.cfunits, self.conform_units_to)
                self.units = self.conform_units_to
                self.conform_units_to = None
            return ret
        finally:
            ds.close()


class BoundedVariable(SourcedVariable):
    def __init__(self, *args, **kwargs):
        self._bounds = None

        bounds = kwargs.pop('bounds', None)
        self._has_extrapolated_bounds = False
        self._name_bounds_dimension = kwargs.pop('name_bounds_dimension', constants.OCGIS_BOUNDS)

        super(BoundedVariable, self).__init__(*args, **kwargs)

        # Setting bounds requires checking the units of the value variable.
        self.bounds = bounds

        # tdk: try to get this to one
        assert self.ndim <= 2

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, self.ndim)
        ret = super(BoundedVariable, self).__getitem__(slc)
        if ret.bounds is not None:
            if ret.bounds.ndim == 2:
                bounds = ret.bounds[slc, :]
            else:
                bounds = ret.bounds[slc[0], slc[1], :]
        else:
            bounds = None
        ret.bounds = bounds
        return ret

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        if value is not None:
            assert value.ndim <= 3
            assert isinstance(value, Variable)

            # Update source dimensions for vector variables.
            if self.ndim == 1:
                bounds_dimension = copy(value.dimensions[1])
                bounds_dimension.name = self._name_bounds_dimension
                value.dimensions = [self.dimensions[0], bounds_dimension]

            # Set units and calendar for bounds variables.
            if value.units is None:
                value.units = self.units
            if hasattr(self, 'calendar'):
                value.calendar = self.calendar
            try:
                if value.conform_units_to is None:
                    value.conform_units_to = self.conform_units_to
            except AttributeError:
                # Account for non-sourced variables.
                if not isinstance(value, Variable):
                    raise
        self._bounds = value

    @property
    def resolution(self):
        # tdk: test
        if self.bounds is None and self.value.shape[0] < 2:
            msg = 'With no bounds and a single coordinate, approximate resolution may not be determined.'
            raise ResolutionError(msg)
        if self.bounds is None:
            ret = super(BoundedVariable, self).resolution
        if self.bounds is not None:
            res_bounds = self.bounds.value[0:constants.RESOLUTION_LIMIT]
            res_array = res_bounds[:, 1] - res_bounds[:, 0]
            ret = np.abs(res_array).mean()
        return ret

    def cfunits_conform(self, to_units, from_units=None):
        # The units will cascade when updated on the host variable.
        units_original = self.units
        super(BoundedVariable, self).cfunits_conform(to_units, from_units=None)
        if self.bounds is not None:
            self.bounds.cfunits_conform(to_units, from_units=units_original)

    def get_between(self, lower, upper, return_indices=False, closed=False, use_bounds=True):
        # tdk: refactor to function
        assert (lower <= upper)

        # Determine if data bounds are contiguous (if bounds exists for the data). Bounds must also have more than one
        # row.
        is_contiguous = False
        if self.bounds is not None:
            bounds_value = self.bounds.value
            try:
                if len(set(bounds_value[0, :]).intersection(set(bounds_value[1, :]))) > 0:
                    is_contiguous = True
            except IndexError:
                # There is likely not a second row.
                if bounds_value.shape[0] == 1:
                    pass
                else:
                    raise

        # Subset operation when bounds are not present.
        if self.bounds is None or use_bounds == False:
            value = self.value
            if closed:
                select = np.logical_and(value > lower, value < upper)
            else:
                select = np.logical_and(value >= lower, value <= upper)
        # Subset operation in the presence of bounds.
        else:
            # Determine which bound column contains the minimum.
            if bounds_value[0, 0] <= bounds_value[0, 1]:
                lower_index = 0
                upper_index = 1
            else:
                lower_index = 1
                upper_index = 0
            # Reference the minimum and maximum bounds.
            bounds_min = bounds_value[:, lower_index]
            bounds_max = bounds_value[:, upper_index]

            # If closed is True, then we are working on a closed interval and are not concerned if the values at the
            # bounds are equivalent. It does not matter if the bounds are contiguous.
            if closed:
                select_lower = np.logical_or(bounds_min > lower, bounds_max > lower)
                select_upper = np.logical_or(bounds_min < upper, bounds_max < upper)
            else:
                # If the bounds are contiguous, then preference is given to the lower bound to avoid duplicate
                # containers (contiguous bounds share a coordinate).
                if is_contiguous:
                    select_lower = np.logical_or(bounds_min >= lower, bounds_max > lower)
                    select_upper = np.logical_or(bounds_min <= upper, bounds_max < upper)
                else:
                    select_lower = np.logical_or(bounds_min >= lower, bounds_max >= lower)
                    select_upper = np.logical_or(bounds_min <= upper, bounds_max <= upper)
            select = np.logical_and(select_lower, select_upper)

        if select.any() == False:
            raise EmptySubsetError(origin=self.name)

        ret = self[select]

        if return_indices:
            indices = np.arange(select.shape[0])
            ret = (ret, indices[select])

        return ret

    def set_extrapolated_bounds(self, name=None):
        """Set the bounds variable using extrapolation."""
        # Only allow extrapolation with a 1d value.
        assert self.ndim == 1
        if self.bounds is not None:
            raise BoundsAlreadyAvailableError
        name = name or '{0}_{1}'.format(self.name, 'bounds')
        bounds_value = get_bounds_from_1d(self.value)
        dims = [self.dimensions[0], Dimension(constants.OCGIS_BOUNDS, length=2)]
        var = Variable(name, value=bounds_value, dimensions=dims)
        self.bounds = var
        self._has_extrapolated_bounds = True

    def write_netcdf(self, *args, **kwargs):
        super(BoundedVariable, self).write_netcdf(*args, **kwargs)
        if self.bounds is not None:
            self.bounds.write_netcdf(*args, **kwargs)

    def write_attributes_to_netcdf_object(self, target):
        super(BoundedVariable, self).write_attributes_to_netcdf_object(target)
        if self.bounds is not None:
            target.bounds = self.bounds.name

    def _set_units_(self, value):
        super(BoundedVariable, self)._set_units_(value)
        # Only update the units if they are not being conformed.
        if self.bounds is not None:
            cut = getattr(self.bounds, '_conform_units_to', None)
            if cut is None:
                self.bounds.units = value

    def _get_extent_target_(self):
        if self.bounds is None:
            ret = super(BoundedVariable, self)._get_extent_target_()
        else:
            ret = self.bounds.value
        return ret


class VariableCollection(AbstractInterfaceObject, AbstractCollection, Attributes):
    # tdk: should test for equivalence of dimensions on variables

    def __init__(self, variables=None, attrs=None):
        AbstractCollection.__init__(self)
        Attributes.__init__(self, attrs)

        if variables is not None:
            for variable in get_iter(variables, dtype=Variable):
                self.add_variable(variable)

    @property
    def dimensions(self):
        ret = {}
        for variable in self.itervalues():
            for d in variable.dimensions:
                if d not in ret:
                    ret[d.name] = d
                else:
                    assert d.length == ret[d.name].length
        return ret

    def add_variable(self, variable):
        """
        :param :class:`ocgis.interface.base.variable.Variable`
        """

        assert isinstance(variable, Variable)
        if variable.alias in self:
            raise VariableInCollectionError(variable)
        self[variable.alias] = variable

    def write_netcdf(self, dataset_or_path, **kwargs):
        """
        Write the field object to an open netCDF dataset object.

        :param dataset: The open dataset object or path for the write.
        :type dataset: :class:`netCDF4.Dataset` or str
        :param bool file_only: If ``True``, we are not filling the value variables. Only the file schema and dimension
         values will be written.
        :param bool unlimited_to_fixedsize: If ``True``, convert the unlimited dimension to fixed size.
        :param kwargs: Extra keyword arguments in addition to ``dimensions`` and ``fill_value`` to pass to
         ``createVariable``. See http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html#createVariable
        """

        if not isinstance(dataset_or_path, Dataset):
            dataset = Dataset(dataset_or_path, 'w')
            close_dataset = True
        else:
            dataset = dataset_or_path
            close_dataset = False

        try:
            self.write_attributes_to_netcdf_object(dataset)
            for variable in self.itervalues():
                variable.write_netcdf(dataset, **kwargs)
            dataset.sync()
        finally:
            if close_dataset:
                dataset.close()

    @staticmethod
    def read_netcdf(path):
        from ocgis import RequestDataset
        rd = RequestDataset(uri=path)
        ds = Dataset(path)
        try:
            ret = VariableCollection(attrs=deepcopy(ds.__dict__))
            for name, ncvar in ds.variables.iteritems():
                ret[name] = SourcedVariable(name=name, request_dataset=rd)
        finally:
            ds.close()
        return ret


def create_dimension_or_pass(dim, dataset):
    if dim.name not in dataset.dimensions:
        dataset.createDimension(dim.name, dim.length)


def get_dimension_lengths(dimensions):
    return tuple([len(d) for d in dimensions])


def get_shape_from_variable(variable):
    _dimensions = variable._dimensions
    _value = variable._value
    if _dimensions is not None and not has_unlimited_dimension(_dimensions):
        ret = get_dimension_lengths(_dimensions)
    elif _value is None:
        dimensions = variable.dimensions
        if dimensions is None:
            ret = tuple()
        else:
            ret = get_dimension_lengths(dimensions)
    else:
        ret = _value.shape

    return ret


def has_unlimited_dimension(dimensions):
    ret = False
    for d in dimensions:
        if d.length is None:
            ret = True
    return ret


def set_variable_metadata_from_source(variable):
    ds = variable._request_dataset.driver.open()
    try:
        var = ds.variables[variable.name]

        if variable._dimensions is None:
            new_dimensions = []
            for dim_name in var.dimensions:
                dim = ds.dimensions[dim_name]
                dim_length = len(dim)
                if dim.isunlimited():
                    length = None
                    length_current = dim_length
                else:
                    length = dim_length
                    length_current = None
                new_dim = SourcedDimension(dim.name, length=length, length_current=length_current)
                new_dimensions.append(new_dim)
            super(SourcedVariable, variable)._set_dimensions_(new_dimensions)

        if variable._dtype is None:
            variable.dtype = deepcopy(var.dtype)

        if variable._fill_value is None:
            variable.fill_value = deepcopy(var.__dict__.get('_FillValue'))

        if variable._units is None:
            variable.units = deepcopy(var.__dict__.get('units'))

        variable.attrs.update(deepcopy(var.__dict__))

        variable._set_metadata_from_source_finalize_(var)
    finally:
        ds.close()
    variable._allocated = True

