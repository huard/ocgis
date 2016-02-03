import itertools
from abc import ABCMeta, abstractproperty
from collections import OrderedDict
from copy import copy, deepcopy
from itertools import izip

import numpy as np
from netCDF4 import Dataset, VLType
from netCDF4._netCDF4 import Group
from numpy.core.multiarray import ndarray
from numpy.ma import MaskedArray

from ocgis import constants
from ocgis.api.collection import AbstractCollection
from ocgis.exc import VariableInCollectionError, BoundsAlreadyAvailableError, EmptySubsetError, \
    ResolutionError, NoUnitsError, DimensionsRequiredError, PayloadProtectedError
from ocgis.interface.base.attributes import Attributes
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.dimension import Dimension, SourcedDimension, create_dimension_or_pass
from ocgis.new_interface.mpi import create_nd_slices
from ocgis.util.helpers import get_iter, get_formatted_slice, get_bounds_from_1d, get_extrapolated_corners_esmf, \
    get_ocgis_corners_from_esmf_corners, iter_array
from ocgis.util.units import get_units_object, get_conformed_units


class AbstractContainer(AbstractInterfaceObject):
    __metaclass__ = ABCMeta

    def __init__(self, mask, parent=None):
        self._mask = None

        if parent is not None:
            assert isinstance(parent, VariableCollection)

        self.parent = parent
        if mask is not None:
            self.set_mask(mask)

    def __getitem__(self, slc):
        ret, slc = self._getitem_initialize_(slc)
        self._getitem_main_(ret, slc)
        if self.parent is not None:
            # Update the parent collection with the returned slice. This is skipped in the update occurring in
            # set_sliced...
            self.parent[ret.name] = ret
            set_sliced_backref_variables(ret, slc)
        self._getitem_finalize_(ret, slc)
        return ret

    @abstractproperty
    def ndim(self):
        pass

    @abstractproperty
    def dimensions(self):
        pass

    def get_mask(self):
        ret = self._mask
        if ret is None:
            if self.value is not None:
                ret = np.zeros(self.shape, dtype=bool)
                fill_value = self.fill_value
                if fill_value is not None:
                    is_equal = self.value == fill_value
                    ret[is_equal] = True
                else:
                    self._fill_value = np.ma.array([], dtype=self.dtype).fill_value
        return ret

    def set_mask(self, mask):
        mask = np.array(mask, dtype=bool)
        assert mask.shape == self.shape

        if self.parent is not None:
            names_container = [d.name for d in self.dimensions]
            new_backref = VariableCollection(attrs=self.parent.attrs.copy())
            mask_container = mask
            for k, v in self.parent.items():
                names_variable = [d.name for d in v.dimensions]
                mask_variable = v.get_mask()
                for slc, value_mask_container in iter_array(mask_container, return_value=True, use_mask=False):
                    if value_mask_container:
                        mapped_slice = get_mapped_slice(slc, names_container, names_variable)
                        mask_variable[mapped_slice] = True
                v.set_mask(mask_variable)
                new_backref.add_variable(v)
            self.parent = new_backref

        self._mask = mask

    def _getitem_initialize_(self, slc):
        try:
            slc = get_formatted_slice(slc, self.ndim)
        except (NotImplementedError, IndexError) as e:
            # Assume it is a dictionary slice.
            try:
                slc = {k: get_formatted_slice(v, 1) for k, v in slc.items()}
            except:
                raise e

        ret = self.copy()
        return ret, slc

    def _getitem_main_(self, ret, slc):
        """Perform major slicing operations in-place."""

    def _getitem_finalize_(self, ret, slc):
        """Finalize the returned sliced object in-place."""


class ObjectType(object):
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def create_vltype(self, dataset, name):
        if self.dtype == object:
            msg = 'Object/ragged arrays required a non-object datatype when writing to netCDF.'
            raise ValueError(msg)
        return dataset.createVLType(self.dtype, name)


class Variable(AbstractContainer, Attributes):
    # tdk:doc

    def __init__(self, name=None, value=None, mask=None, dimensions=None, dtype=None, attrs=None, fill_value=None,
                 units=None, parent=None):
        Attributes.__init__(self, attrs=attrs)

        self._dimensions = None
        self._value = None
        self._dtype = None
        self._units = None
        self._mask = None

        self._fill_value = fill_value

        self.name = name
        self.units = units
        self.dtype = dtype

        create_dimensions = False
        if dimensions is not None:
            if isinstance(list(get_iter(dimensions, dtype=(basestring, Dimension)))[0], basestring):
                create_dimensions = True
            else:
                self.dimensions = dimensions
        self.value = value
        if create_dimensions:
            self.create_dimensions(names=dimensions)

        # The mask is updated in _set_value_. Use the internal reference to ensure it is not overwritten.
        if mask is None:
            mask = self._mask
        AbstractContainer.__init__(self, mask, parent=parent)

    def _getitem_main_(self, ret, slc):
        dimensions = ret.dimensions

        if isinstance(slc, dict):
            if dimensions is None:
                msg = 'Dimensions are required for dictionary slices.'
                raise DimensionsRequiredError(msg)
            else:
                names_src, new_slc = slc.keys(), slc.values()
                names_dst = [d.name for d in dimensions]
                new_slc = get_mapped_slice(new_slc, names_src, names_dst)
                self._getitem_main_(ret, new_slc)
        else:
            new_dimensions = None
            new_value = None
            new_mask = None

            if dimensions is not None:
                new_dimensions = [d[s] for d, s in izip(dimensions, get_iter(slc, dtype=(slice, ndarray)))]
            if ret._value is not None:
                new_value = ret.value.__getitem__(slc)
            if ret._mask is not None:
                new_mask = ret._mask.__getitem__(slc)

            ret.dimensions = None
            ret.value = None

            ret.dimensions = new_dimensions
            ret.value = new_value
            if new_mask is not None:
                ret.set_mask(new_mask)

    def __setitem__(self, slc, variable_or_value):
        # tdk: order
        # tdk:
        slc = get_formatted_slice(slc, self.ndim)
        try:
            self.value[slc] = variable_or_value.value
        except AttributeError:  # Assume this is an array or other object.
            self.value[slc] = variable_or_value
        else:
            new_mask = self.get_mask()
            new_mask[slc] = variable_or_value.get_mask()
            self.set_mask(new_mask)

    def __len__(self):
        return self.shape[0]

    @property
    def cfunits(self):
        return get_units_object(self.units)

    @property
    def dtype(self):
        if self._dtype is None:
            try:
                ret = self._value.dtype
                if ret == object:
                    ret = ObjectType(object)
            except AttributeError:
                # Assume None.
                ret = None
        else:
            ret = self._dtype
        return ret

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def dimensions(self):
        if self._dimensions is None:
            self._dimensions = self._get_dimensions_()
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        self._set_dimensions_(value)

    @property
    def dimensions_dict(self):
        ret = OrderedDict()
        try:
            for d in self.dimensions:
                ret[d.name] = d
        except TypeError:
            # Assume None.
            pass
        return ret

    def _get_dimensions_(self):
        return None

    def _set_dimensions_(self, dimensions):
        if dimensions is not None:
            dimensions = tuple(get_iter(dimensions, dtype=Dimension))
        self._dimensions = dimensions
        update_unlimited_dimension_length(self._value, dimensions)

    @property
    def extent(self):
        target = self._get_extent_target_()
        return target.compressed().min(), target.compressed().max()

    def _get_extent_target_(self):
        return self.masked_value

    @property
    def fill_value(self):
        return self._fill_value

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
        res_array = np.diff(np.abs(self.value[0:constants.RESOLUTION_LIMIT]))
        ret = np.abs(res_array).mean()
        return ret

    @property
    def shape(self):
        return self._get_shape_()

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

    @property
    def masked_value(self):
        if isinstance(self.dtype, ObjectType):
            dtype = object
        else:
            dtype = self.dtype
        ret = np.ma.array(self.value, mask=self.get_mask(), dtype=dtype, fill_value=self.fill_value)
        return ret

    def _get_value_(self):
        dimensions = self._dimensions
        if dimensions is None or len(dimensions) == 0:
            ret = None
        else:
            if has_unlimited_dimension(dimensions):
                msg = 'Value shapes for variables with unlimited dimensions are undetermined.'
                raise ValueError(msg)
            elif len(dimensions) > 0:
                new_shape = get_dimension_lengths(dimensions)
                ret = np.zeros(new_shape, dtype=self.dtype)
        return ret

    def _set_value_(self, value):
        if value is not None:
            if isinstance(value, MaskedArray):
                self._fill_value = value.fill_value
                mask = value.mask.copy()
                if np.isscalar(mask):
                    new_mask = np.zeros(value.shape, dtype=bool)
                    new_mask.fill(mask)
                    mask = new_mask
                self._mask = mask
                value = value.data
            else:
                self._mask = None

            desired_dtype = self._dtype

            if not isinstance(value, ndarray):
                array_type = desired_dtype
                if isinstance(array_type, ObjectType):
                    array_type = object
                value = np.array(value, dtype=array_type)
                if isinstance(desired_dtype, ObjectType):
                    if desired_dtype.dtype != object:
                        for idx in range(value.shape[0]):
                            value[idx] = np.array(value[idx], dtype=desired_dtype.dtype)
            if desired_dtype is not None and desired_dtype != value.dtype and value.dtype != object:
                try:
                    value = value.astype(desired_dtype, copy=False)
                except TypeError:
                    value = value.astype(desired_dtype)

        update_unlimited_dimension_length(value, self._dimensions)

        self._value = value

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

    def copy(self):
        ret = copy(self)
        try:
            ret._value = ret._value.view()
            ret._mask = ret._mask.view()
        except AttributeError:  # Assume None.
            pass
        ret.attrs = ret._attrs
        return ret

    def create_dimensions(self, names=None):
        if names is None and self.name is None:
            msg = 'Variable "name" is required when no "names" provided to "create_dimensions".'
            raise ValueError(msg)

        names = tuple(get_iter(names or self.name))

        value = self._value
        if value is None:
            new_dimensions = ()
        else:
            new_dimensions = []
            if len(names) != value.ndim:
                msg = "The number of dimension 'names' must equal the number of dimensions (ndim)."
                raise ValueError(msg)
            for name, shp in izip(names, value.shape):
                new_dimensions.append(Dimension(name, length=shp))
        self.dimensions = new_dimensions

    def get_scatter_slices(self, splits):
        slices = create_nd_slices(splits, self.shape)
        return slices

    def iter(self, use_mask=True):
        name = self.name
        for idx, value in iter_array(self.masked_value, use_mask=use_mask, return_value=True):
            yld = OrderedDict()
            try:
                for idx_d, d in enumerate(self.dimensions):
                    try:
                        yld[d._variable.name] = d._variable.value[idx[idx_d]]
                    except AttributeError:  # Assume None.
                        pass
            except TypeError:  # Assume None.
                pass
            yld[name] = value
            yield idx, yld

    def reshape(self, *args, **kwargs):
        dimension_name = kwargs.pop('dimension_name', None)
        if self.parent is not None:
            # tdk: needs implementation
            raise NotImplementedError('backref cannot be reshaped')
        # tdk: test with source index
        # tdk: test with unlimited dimensions
        ret = self.copy()
        ret.dimensions = None
        mask = ret._mask
        ret.value = ret.value.reshape(*args)
        ret.create_dimensions(names=dimension_name)
        if mask is not None:
            ret.set_mask(mask.reshape(*args))
        return ret

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

        if self.name is None:
            msg = 'A variable "name" is required.'
            raise ValueError(msg)

        file_only = kwargs.pop('file_only', False)
        unlimited_to_fixedsize = kwargs.pop('unlimited_to_fixedsize', False)

        if self.dimensions is None:
            new_names = ['dim_ocgis_{}_{}'.format(self.name, ctr) for ctr in range(self.ndim)]
            self.create_dimensions(new_names)

        dimensions = self.dimensions

        dtype = self.dtype
        if isinstance(dtype, ObjectType):
            dtype = dtype.create_vltype(dataset, dimensions[0].name + '_VLType')

        if len(dimensions) > 0:
            dimensions = list(dimensions)
            # Convert the unlimited dimension to fixed size if requested.
            for idx, d in enumerate(dimensions):
                if d.length is None and unlimited_to_fixedsize:
                    dimensions[idx] = Dimension(d.name, length=self.shape[idx])
                    break
            # Create the dimensions.
            for dim in dimensions:
                create_dimension_or_pass(dim, dataset)
            dimensions = [d.name for d in dimensions]

        # Only use the fill value if something is masked.
        if len(dimensions) > 0 and not file_only and self.get_mask().any():
            fill_value = self.fill_value
        else:
            # Copy from original attributes.
            if '_FillValue' not in self.attrs:
                fill_value = None
            else:
                fill_value = self.fill_value

        var = dataset.createVariable(self.name, dtype, dimensions=dimensions, fill_value=fill_value, **kwargs)
        if not file_only:
            try:
                var[:] = self.masked_value
            except AttributeError:
                # Assume ObjectType.
                for idx, v in iter_array(self.value, use_mask=False, return_value=True):
                    var[idx] = np.array(v)

        self.write_attributes_to_netcdf_object(var)

        if self.units is not None:
            var.units = self.units

        dataset.sync()

    def _get_to_conform_value_(self):
        return self.masked_value

    def _set_to_conform_value_(self, value):
        self.value = value


class SourcedVariable(Variable):
    def __init__(self, *args, **kwargs):
        self._conform_units_to = None

        # Flag to indicate if metadata already from source.
        self._allocated = False

        self.protected = kwargs.pop('protected', False)

        self.conform_units_to = kwargs.pop('conform_units_to', None)
        self._request_dataset = kwargs.pop('request_dataset', None)

        super(SourcedVariable, self).__init__(*args, **kwargs)

        if self._value is None and self._request_dataset is None:
            msg = 'A "value" or "request_dataset" is required.'
            raise ValueError(msg)
        if self._value is not None and self._conform_units_to is not None:
            msg = '"conform_units_to" only applicable when loading from source.'
            raise ValueError(msg)

    def _get_shape_(self):
        # tdk: order
        if self._request_dataset is not None and not self._allocated:
            self._set_metadata_from_source_()
        return super(SourcedVariable, self)._get_shape_()

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
        if self._fill_value is None and not self._allocated and self._request_dataset is not None:
            self._set_metadata_from_source_()
        return self._fill_value

    def _get_dimensions_(self):
        if self._request_dataset is None:
            ret = super(SourcedVariable, self)._get_dimensions_()
        else:
            self._set_metadata_from_source_()
            ret = self._dimensions
        return ret

    def _set_metadata_from_source_(self):
        if not self._allocated:
            set_variable_metadata_from_source(self)

    def _set_metadata_from_source_finalize_(self, *args, **kwargs):
        pass

    def _get_attrs_(self):
        self._attrs = Attributes._get_attrs_(self)
        if not self._allocated and self._request_dataset is not None:
            self._set_metadata_from_source_()
        return self._attrs

    def _get_units_(self):
        if not self._allocated and self._units is None and self._request_dataset is not None:
            self._set_metadata_from_source_()
        return self._units

    def _get_value_(self):
        if self._value is None:
            value = self._get_value_from_source_()
            super(SourcedVariable, self)._set_value_(value)
            ret = self._value
        else:
            ret = super(SourcedVariable, self)._get_value_()
        return ret

    def _get_value_from_source_(self):
        if self.protected:
            raise PayloadProtectedError

        ds = self._request_dataset.driver.open()
        source = ds
        if self.parent is not None:
            if self.parent.parent is not None:
                source = ds.groups[self.parent.name]

        desired_name = self.name or self._request_dataset.variable
        try:
            variable = source.variables[desired_name]
            ret = get_variable_value(variable, self.dimensions)

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

        try:
            assert self.ndim <= 2
        except DimensionsRequiredError:
            assert self._value.ndim <= 2

    def __setitem__(self, slc, variable_or_value):
        super(BoundedVariable, self).__setitem__(slc, variable_or_value)
        try:
            if variable_or_value.bounds is not None:
                if self.bounds is None:
                    self.set_extrapolated_bounds()
                self.bounds.value[slc] = variable_or_value.bounds.value
        except AttributeError:  # Assume array or other object.
            pass

    def _getitem_main_(self, ret, slc):
        # tdk: order
        bounds = ret.bounds
        ret.bounds = None
        try:
            super(BoundedVariable, self)._getitem_main_(ret, slc)
            if bounds is not None:
                bounds = bounds[slc]
        finally:
            ret.bounds = bounds


    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        if value is not None:
            try:
                assert value.ndim <= 3
            except DimensionsRequiredError as e:
                if value._value is None:
                    msg = "Dimensions are required on the 'bounds' variable when its 'value' is None: {}".format(e.message)
                    raise DimensionsRequiredError(msg)
                else:
                    assert value._value.ndim <= 3

            # Update source dimensions for vector variables.
            try:
                ndim = self.ndim
            except DimensionsRequiredError:
                ndim = self._value.ndim
            if ndim == 1:
                if value._dimensions is not None:
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

        if self.bounds is not None:
            raise BoundsAlreadyAvailableError
        name = name or '{0}_{1}'.format(self.name, 'bounds')

        if self.ndim == 1:
            bounds_value = get_bounds_from_1d(self.value)
        else:
            # tdk: consider renaming this functions to get_bounds_from_2d
            bounds_value = get_extrapolated_corners_esmf(self.value)
            bounds_value = get_ocgis_corners_from_esmf_corners(bounds_value)

        dimensions = self.dimensions
        if dimensions is None:
            dims = None
        else:
            if self.ndim == 1:
                dims = [dimensions[0], Dimension(constants.OCGIS_BOUNDS, length=2)]
            else:
                dims = list(dimensions) + [Dimension(constants.DEFAULT_NAME_CORNERS_DIMENSION, length=4)]

        var = Variable(name, value=bounds_value, dimensions=dims)
        self.bounds = var
        self._has_extrapolated_bounds = True

        # This will synchronize the bounds mask with the variable's mask.
        set_bounds_mask_from_parent(self.get_mask(), self.bounds)

    def create_dimensions(self, names=None):
        super(BoundedVariable, self).create_dimensions(names)
        if self.bounds is not None:
            if self.ndim == 1:
                if self.bounds._dimensions is not None:
                    name_bounds = self.bounds.dimensions[1].name
                else:
                    name_bounds = constants.OCGIS_BOUNDS
                names = [self.dimensions[0].name, name_bounds]
            else:
                names = [d.name for d in self.dimensions]
                if self.bounds.dimensions is not None:
                    name_corners = self.bounds.dimensions[2].name
                else:
                    name_corners = constants.DEFAULT_NAME_CORNERS_DIMENSION
                names.append(name_corners)
            self.bounds.create_dimensions(names=names)

            synced_dimensions = list(self.bounds.dimensions)
            synced_dimensions[0] = self.dimensions[0]
            self.bounds.dimensions = synced_dimensions

    def iter(self, **kwargs):
        itr = super(BoundedVariable, self).iter(**kwargs)
        for idx, element in itr:
            if self.bounds is not None:
                row = self.bounds.value[idx, :]
                lb, ub = np.min(row), np.max(row)
                element['lb_{}'.format(self.name)] = lb
                element['ub_{}'.format(self.name)] = ub
            yield idx, element

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
            ret = self.bounds.masked_value
        return ret

    def set_mask(self, mask):
        super(BoundedVariable, self).set_mask(mask)
        if self.bounds is not None:
            set_bounds_mask_from_parent(self.get_mask(), self.bounds)


class VariableCollection(AbstractInterfaceObject, AbstractCollection, Attributes):
    def __init__(self, name=None, variables=None, attrs=None, parent=None):
        self.name = name
        self.children = OrderedDict()

        AbstractCollection.__init__(self)
        Attributes.__init__(self, attrs)
        AbstractInterfaceObject.__init__(self, parent=parent)

        if variables is not None:
            for variable in get_iter(variables, dtype=Variable):
                self.add_variable(variable)

    def add_child(self, child):
        child.parent = self
        self.children[child.name] = child

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

    @property
    def shapes(self):
        return OrderedDict([[k, v.shape] for k, v in self.items()])

    def add_variable(self, variable):
        """
        :param :class:`ocgis.interface.base.variable.Variable`
        """

        if variable.name in self:
            raise VariableInCollectionError(variable)
        self[variable.name] = variable
        variable.parent = self

    def copy(self):
        ret = AbstractCollection.copy(self)
        ret.attrs = ret.attrs.copy()
        for k, v in ret.items():
            ret[k] = v.copy()
        return ret

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

        if not isinstance(dataset_or_path, (Dataset, Group)):
            dataset = Dataset(dataset_or_path, 'w')
            close_dataset = True
        else:
            dataset = dataset_or_path
            close_dataset = False

        try:
            self.write_attributes_to_netcdf_object(dataset)
            for variable in self.values():
                variable.write_netcdf(dataset, **kwargs)
            for child in self.children.values():
                group = Group(dataset, child.name)
                child.write_netcdf(group, **kwargs)
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
            ret = VariableCollection._read_from_collection_(ds, rd, parent=None)
        finally:
            ds.close()
        return ret

    @staticmethod
    def _read_from_collection_(target, request_dataset, parent=None, name=None):
        ret = VariableCollection(attrs=deepcopy(target.__dict__), parent=parent, name=name)
        for name, ncvar in target.variables.iteritems():
            ret[name] = SourcedVariable(name=name, request_dataset=request_dataset, parent=ret)
        for name, ncgroup in target.groups.iteritems():
            child = VariableCollection._read_from_collection_(ncgroup, request_dataset, parent=ret, name=name)
            ret.add_child(child)
        return ret


def get_dimension_lengths(dimensions):
    return tuple([len(d) for d in dimensions])


def get_shape_from_variable(variable):
    dimensions = variable._dimensions
    value = variable._value
    try:
        shape = get_dimension_lengths(dimensions)
    except TypeError:
        try:
            shape = value.shape
        except AttributeError:
            shape = tuple()
    return shape


def has_unlimited_dimension(dimensions):
    ret = False
    for d in dimensions:
        if d.length is None:
            ret = True
    return ret


def set_variable_metadata_from_source(variable):
    dataset = variable._request_dataset.driver.open()
    source = dataset
    if variable.parent is not None:
        if variable.parent.parent is not None:
            source = dataset.groups[variable.parent.name]

    desired_name = variable.name or variable._request_dataset.variable
    try:
        var = source.variables[desired_name]

        if variable._dimensions is None:
            desired_dimensions = var.dimensions
            new_dimensions = get_dimensions_from_netcdf(source, desired_dimensions)
            super(SourcedVariable, variable)._set_dimensions_(new_dimensions)

        if variable._dtype is None:
            desired_dtype = deepcopy(var.dtype)
            if isinstance(var.datatype, VLType):
                desired_dtype = ObjectType(var.dtype)
            variable.dtype = desired_dtype

        if variable._fill_value is None:
            variable._fill_value = deepcopy(var.__dict__.get('_FillValue'))

        if variable._units is None:
            variable.units = deepcopy(var.__dict__.get('units'))

        variable.attrs.update(deepcopy(var.__dict__))

        variable._set_metadata_from_source_finalize_(var)
    finally:
        dataset.close()
    variable._allocated = True


def get_dimensions_from_netcdf(dataset, desired_dimensions):
    new_dimensions = []
    for dim_name in desired_dimensions:
        dim = dataset.dimensions[dim_name]
        dim_length = len(dim)
        if dim.isunlimited():
            length = None
            length_current = dim_length
        else:
            length = dim_length
            length_current = None
        new_dim = SourcedDimension(dim.name, length=length, length_current=length_current)
        new_dimensions.append(new_dim)
    return new_dimensions


def update_unlimited_dimension_length(variable_value, dimensions):
    """
    Update unlimited dimension length if present on the variable. Update only occurs if the variable's value is
    allocated.
    """
    if variable_value is not None:
        # Update any unlimited dimension length.
        if dimensions is not None:
            aq = are_variable_and_dimensions_shape_equal(variable_value, dimensions)
            if not aq:
                msg = "Variable and dimension shapes must be equal."
                raise ValueError(msg)
            for idx, d in enumerate(dimensions):
                if d.length is None:
                    d.length_current = variable_value.shape[idx]


def are_variable_and_dimensions_shape_equal(variable_value, dimensions):
    to_test = []
    vshape = variable_value.shape
    dshape = get_dimension_lengths(dimensions)

    if len(vshape) != len(dshape):
        ret = False
    else:
        is_unlimited = [d.length is None for d in dimensions]
        for v, d, iu in zip(vshape, dshape, is_unlimited):
            if iu:
                to_append = True
            else:
                to_append = v == d
            to_test.append(to_append)
        ret = all(to_test)

    return ret


def set_sliced_backref_variables(ret, slc):
    if not isinstance(slc, dict):
        if ret.dimensions is None:
            raise DimensionsRequiredError('Dimensions required when slicing parent variables.')
        else:
            slc = {ret.dimensions[idx].name: slc[idx] for idx in range(ret.ndim)}
    backref = ret.parent
    for key, variable in backref.items():
        if ret.name != variable.name:
            variable.parent = None
            try:
                backref[key] = variable.__getitem__(slc)
            except DimensionsRequiredError:
                pass
            backref[key].parent = backref


def get_mapped_slice(slc_src, names_src, names_dst):
    ret = [slice(None)] * len(names_dst)
    for idx, name in enumerate(names_dst):
        try:
            idx_src = names_src.index(name)
        except ValueError:
            continue
        else:
            ret[idx] = slc_src[idx_src]
    return tuple(ret)


def set_bounds_mask_from_parent(mask, bounds):
    if mask.ndim == 1:
        mask_bounds = mask.reshape(-1, 1)
        mask_bounds = np.hstack((mask_bounds, mask_bounds))
    elif mask.ndim == 2:
        mask_bounds = np.zeros(list(mask.shape) + [4], dtype=bool)
        for idx_row, idx_col in itertools.product(range(mask.shape[0]), range(mask.shape[1])):
            if mask[idx_row, idx_col]:
                mask_bounds[idx_row, idx_col, :] = True
    else:
        raise NotImplementedError(mask.ndim)
    bounds.set_mask(mask_bounds)


def get_variable_value(variable, dimensions):
    if dimensions is not None and len(dimensions) > 0:
        slc = get_formatted_slice([d._src_idx for d in dimensions], len(dimensions))
    else:
        slc = slice(None)
    ret = variable.__getitem__(slc)
    return ret
