import itertools
from abc import ABCMeta, abstractproperty, abstractmethod
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from numpy.core.multiarray import ndarray
from numpy.ma import MaskedArray

from ocgis import constants
from ocgis.api.collection import AbstractCollection
from ocgis.exc import VariableInCollectionError, BoundsAlreadyAvailableError, EmptySubsetError, \
    ResolutionError, NoUnitsError, DimensionsRequiredError, DimensionMismatchError
from ocgis.interface.base.attributes import Attributes
from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.new_interface.base import AbstractInterfaceObject, orphaned
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.mpi import create_nd_slices, get_global_to_local_slice
from ocgis.util.helpers import get_iter, get_formatted_slice, get_bounds_from_1d, get_extrapolated_corners_esmf, \
    get_ocgis_corners_from_esmf_corners, iter_array
from ocgis.util.units import get_units_object, get_conformed_units


def handle_empty(func):
    def wrapped(*args, **kwargs):
        self = args[0]
        if self.is_empty:
            return self
        else:
            return func(*args, **kwargs)

    return wrapped


class AbstractContainer(AbstractInterfaceObject):
    __metaclass__ = ABCMeta

    def __init__(self, name, parent=None):
        self._name = name
        self._parent = parent

        if parent is None:
            self._initialize_parent_()

    def __getitem__(self, slc):
        ret, slc = self._getitem_initialize_(slc)
        if self._parent is None:
            self._getitem_main_(ret, slc)
            self._getitem_finalize_(ret, slc)
        else:
            if not isinstance(slc, dict):
                slc = get_dslice(self.dimensions, slc)
            new_parent = ret.parent[slc]
            ret = new_parent[ret.name]
        return ret

    @abstractproperty
    def dimensions(self):
        pass

    @property
    def group(self):
        if self.parent is None:
            ret = None
        else:
            curr = self.parent
            ret = [curr.name]
            while True:
                if curr.parent is None:
                    break
                else:
                    curr = curr.parent
                    ret.append(curr.name)
            ret.reverse()
        return ret

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        if self._parent is None:
            self._initialize_parent_()
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None:
            assert isinstance(value, VariableCollection)
        self._parent = value

    @abstractmethod
    def get_mask(self):
        """:rtype: :class:`numpy.ndarray`"""
        raise NotImplementedError

    @abstractmethod
    def set_mask(self, mask):
        raise NotImplementedError

    def _getitem_initialize_(self, slc):
        try:
            slc = get_formatted_slice(slc, self.ndim)
        except (NotImplementedError, IndexError) as e:
            # Assume it is a dictionary slice.
            try:
                slc = {k: get_formatted_slice(v, 1)[0] for k, v in slc.items()}
            except:
                raise e
        return self, slc

    def _getitem_main_(self, ret, slc):
        """Perform major slicing operations in-place."""

    def _getitem_finalize_(self, ret, slc):
        """Finalize the returned sliced object in-place."""

    def _initialize_parent_(self):
        self._parent = VariableCollection()


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

    def __init__(self, name=None, value=None, mask=None, dimensions=None, dtype=None, attrs=None, fill_value=None,
                 units='auto', parent=None, bounds=None, is_empty=None, dist=None, ranks=None):
        self._is_init = True

        Attributes.__init__(self, attrs=attrs)

        self._dimensions = None
        self._value = None
        self._dtype = None
        self._mask = None
        self._is_empty = is_empty

        self.dist = dist
        self.ranks = ranks
        self.dtype = dtype

        self._fill_value = fill_value
        if bounds is not None:
            self._bounds_name = bounds.name
        else:
            self._bounds_name = None

        AbstractContainer.__init__(self, name, parent=parent)

        if self.name is None:
            self._name = 'var_{}_{}'.format(self.parent.name, self.parent._variable_name_ctr)
            self.parent._variable_name_ctr += 1

        # Units on sourced variables may check for the presence of a parent. Units may be used by bounds, so set the
        # units here.
        if str(units) != 'auto':
            self.units = units

        # create_dimensions = False
        # if dimensions is not None:
        #     if isinstance(list(get_iter(dimensions, dtype=(basestring, Dimension)))[0], basestring):
        #         create_dimensions = True
        #     else:
        #         self.dimensions = dimensions

        self.set_value(value)
        if value is not None and dimensions is None:
            self.create_dimensions(shape=self.value.shape)
        else:
            self.set_dimensions(dimensions)
        if value is not None:
            update_unlimited_dimension_length(self.value, self.dimensions)
        # if create_dimensions:
        #     self.create_dimensions(names=dimensions)

        if bounds is not None:
            self.bounds = bounds

        # # Add to the parent.
        # if self.parent is not None:
        #     self.parent.add_variable(self, force=True)

        if mask is not None:
            self.set_mask(mask)

        self._is_init = False

    def __add_to_collection_finalize__(self, vc):
        """
        Finalize adding the variable to the collection.

        :param vc: :class:`ocgis.VariableCollection`
        """
        if self.has_bounds:
            vc.add_variable(self.bounds, force=True)

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
                new_dimensions = [d[s] for d, s in zip(dimensions, slc)]
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

    def __setitem__(self, slc, variable):
        # tdk: order
        slc = get_formatted_slice(slc, self.ndim)
        self.value[slc] = variable.value
        new_mask = self.get_mask()
        new_mask[slc] = variable.get_mask()

        if self.has_bounds:
            names_src = [d.name for d in self.dimensions]
            names_dst = [d.name for d in self.bounds.dimensions]
            slc = get_mapped_slice(slc, names_src, names_dst)
            with orphaned(self.bounds):
                self.bounds[slc] = variable.bounds

        self.set_mask(new_mask)

    def __len__(self):
        return self.shape[0]

    @property
    def bounds(self):
        if self._bounds_name is None or self.parent is None:
            ret = None
        else:
            ret = self.parent[self._bounds_name]
        return ret

    @bounds.setter
    def bounds(self, value):
        self._set_bounds_(value)

    def _set_bounds_(self, value):
        if value is None:
            if self._bounds_name is not None:
                self.parent.pop(self._bounds_name)
                self.attrs.pop('bounds', None)
            self._bounds_name = None
        else:
            assert value.name != self.name
            self._bounds_name = value.name
            self.attrs['bounds'] = value.name
            self.parent.add_variable(value, force=True)
            value.units = self.units

    @property
    def cfunits(self):
        return get_units_object(self.units)

    @property
    def dtype(self):
        if self._dtype is None:
            ret = self._get_dtype_()
        else:
            ret = self._dtype
        return ret

    def _get_dtype_(self):
        try:
            ret = self._value.dtype
            if ret == object:
                ret = ObjectType(object)
        except AttributeError:
            # Assume None.
            ret = None
        return ret

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def dimensions(self):
        return self._get_dimensions_()

    @property
    def dimensions_dict(self):
        ret = OrderedDict()
        for d in self.dimensions:
            ret[d.name] = d
        return ret

    @property
    def dimension_names(self):
        return self._dimensions

    def _get_dimensions_(self):
        if self._dimensions is None:
            ret = tuple()
        else:
            ret = tuple([self.parent.dimensions[name] for name in self._dimensions])
        return ret

    def set_dimensions(self, dimensions):
        if dimensions is not None:
            dimensions = list(get_iter(dimensions, dtype=(Dimension, basestring)))
            dimension_names = [None] * len(dimensions)
            for idx, dimension in enumerate(dimensions):
                try:
                    dimension_name = dimension.name
                    self.parent.add_dimension(dimension)
                except AttributeError:
                    dimension_name = dimension
                assert dimension_name in self.parent.dimensions
                dimension_names[idx] = dimension_name
            self._dimensions = tuple(dimension_names)
        else:
            self._dimensions = dimensions
        update_unlimited_dimension_length(self._value, self.dimensions)
        # Only update the bounds dimensions if this is not part of the variable initialization process. Bounds are
        # configured normally during initialization.
        if not self._is_init and self.has_bounds:
            if dimensions is None:
                bounds_dimensions = None
            else:
                bounds_dimensions = list(self.bounds.dimensions)
                bounds_dimensions[0:len(self.dimensions)] = self.dimensions
            self.bounds.dimensions = bounds_dimensions

    @property
    def extent(self):
        target = self._get_extent_target_()
        return target.compressed().min(), target.compressed().max()

    def _get_extent_target_(self):
        if self.has_bounds:
            ret = self.bounds.masked_value
        else:
            ret = self.masked_value
        return ret

    @property
    def fill_value(self):
        return self._get_fill_value_()

    def _get_fill_value_(self):
        return self._fill_value

    @property
    def has_dimensions(self):
        """
        :return: ``True`` if the variable has dimensions.
        :rtype: bool
        """

        if self.dimensions is not None:
            if len(self.dimensions) > 0:
                ret = True
            else:
                ret = False
        else:
            ret = False
        return ret

    @property
    def has_distributed_dimension(self):
        """
        :return: ``True`` if the variable has a distributed dimension.
        :rtype: bool
        """

        ret = False
        dimensions = self.dimensions
        if dimensions is not None:
            for d in dimensions:
                if d.dist:
                    ret = True
                    break
        return ret

    @property
    def has_masked_values(self):
        """
        Return ``True`` if any values are masked.

        :rtype: bool
        """

        if self._mask is None:
            ret = False
        else:
            mask = self.get_mask()
            ret = mask.any()
        return ret

    @property
    def is_empty(self):
        if self._is_empty is None:
            ret = False
            if self.has_distributed_dimension:
                for dim in self.dimensions:
                    if dim.is_empty:
                        ret = True
                        break
        else:
            ret = self._is_empty
        return ret

    @property
    def ndim(self):
        return len(self._dimensions)

    @property
    def resolution(self):
        # tdk: test
        # tdk: not sure where this belongs exactly. maybe on a value dimension?

        if not self.has_bounds and self.value.shape[0] < 2:
            msg = 'With no bounds and only a single coordinate, approximate resolution may not be determined.'
            raise ResolutionError(msg)
        elif self.has_bounds:
            res_bounds = self.bounds.value[0:constants.RESOLUTION_LIMIT]
            res_array = res_bounds[:, 1] - res_bounds[:, 0]
            ret = np.abs(res_array).mean()
        else:
            res_array = np.diff(np.abs(self.value[0:constants.RESOLUTION_LIMIT]))
            ret = np.abs(res_array).mean()
        return ret

    @property
    def shape(self):
        return self._get_shape_()

    @property
    def size(self):
        ret = 1
        if len(self.shape) == 0:
            ret = 0
        else:
            for s in self.shape:
                ret *= s
        return ret

    def _get_shape_(self):
        return get_shape_from_variable(self)

    @property
    def units(self):
        return self._get_units_()

    @units.setter
    def units(self, value):
        self._set_units_(value)

    def _get_units_(self):
        return get_attribute_property(self, 'units')

    def _set_units_(self, value):
        if value is not None:
            value = str(value)
        set_attribute_property(self, 'units', value)
        if self.bounds is not None:
            set_attribute_property(self.bounds, 'units', value)

    @property
    def value(self):
        if self._value is None:
            self._value = self._get_value_()
        return self._value

    @property
    def masked_value(self):
        if isinstance(self.dtype, ObjectType):
            dtype = object
        else:
            dtype = self.dtype
        ret = np.ma.array(self.value, mask=self.get_mask(), dtype=dtype, fill_value=self.fill_value)
        return ret

    def _get_value_(self):
        if self.is_empty:
            ret = None
        else:
            dimensions = self.dimensions
            if len(dimensions) == 0:
                ret = None
            else:
                if has_unlimited_dimension(dimensions):
                    msg = 'Value shapes for variables with unlimited dimensions are undetermined.'
                    raise ValueError(msg)
                elif len(dimensions) > 0:
                    ret = variable_get_zeros(dimensions, self.dtype)
        return ret

    def set_value(self, value):
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

        if not self._is_init:
            update_unlimited_dimension_length(value, self.dimensions)

        self._value = value

    def copy(self):
        if self._parent is None:
            ret = AbstractContainer.copy(self)
            ret.attrs = ret.attrs.copy()
        else:
            ret = self.parent.copy()[self.name]
        return ret

    def deepcopy(self, eager=False):
        deepcopied = self.copy()

        if eager:
            raise NotImplementedError
        else:
            with orphaned(deepcopied):
                deepcopied.__dict__ = deepcopy(deepcopied.__dict__)
            if deepcopied.parent is not None:
                deepcopied.parent.add_variable(deepcopied, force=True)
            if deepcopied.has_bounds:
                deepcopied.bounds = deepcopied.bounds.deepcopy()

        return deepcopied

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

        # Let the data type and fill value load from the value array.
        self._dtype = None
        self._fill_value = None

        if self.has_bounds:
            self.bounds.cfunits_conform(to_units, from_units=from_units)

    def create_dimensions(self, names=None, shape=None):
        if shape is None:
            shape = self._value.shape
        if names is None:
            names = [None] * len(shape)
            for idx in range(len(names)):
                names[idx] = 'dim_{}_{}'.format(self.parent.name, self.parent._dimension_name_ctr)
                self.parent._dimension_name_ctr += 1
        else:
            names = tuple(get_iter(names))

        new_dimensions = []
        for name, shp in zip(names, shape):
            new_dimension = Dimension(name, size=shp)
            new_dimensions.append(new_dimension)
        self.set_dimensions(new_dimensions)

    def reshape(self, *args, **kwargs):
        assert not self.has_bounds

        new_dimensions = kwargs.get('dimensions')
        self.dimensions = None
        mask = self.get_mask()
        if mask is not None:
            new_mask = mask.reshape(*args)
        self.value = self.value.reshape(*args)
        self.set_mask(new_mask)
        if new_dimensions is not None:
            self.create_dimensions(new_dimensions)

    def set_extrapolated_bounds(self, name_variable, name_dimension):
        """Set the bounds variable using extrapolation."""

        if self.bounds is not None:
            raise BoundsAlreadyAvailableError
        if self.dimensions is None:
            raise DimensionsRequiredError('Dimensions are required on the bounded variable.')

        bounds_value = None
        if self.ndim == 1:
            if not self.is_empty:
                bounds_value = get_bounds_from_1d(self.value)
            bounds_dimension_size = 2
        else:
            # tdk: consider renaming this functions to get_bounds_from_2d
            if not self.is_empty:
                bounds_value = get_extrapolated_corners_esmf(self.value)
                bounds_value = get_ocgis_corners_from_esmf_corners(bounds_value)
            bounds_dimension_size = 4

        dimensions = list(self.dimensions)
        dimensions.append(Dimension(name=name_dimension, size=bounds_dimension_size))

        var = Variable(name=name_variable, value=bounds_value, dimensions=dimensions, units=self.units)
        self.bounds = var

        # This will synchronize the bounds mask with the variable's mask.
        if not self.is_empty:
            self.set_mask(self.get_mask())

    @property
    def has_allocated_value(self):
        if self._value is None:
            ret = False
        else:
            ret = True
        return ret

    @property
    def has_bounds(self):
        if self.bounds is not None:
            ret = True
        else:
            ret = False
        return ret

    def get_mask(self):
        if self.is_empty:
            ret = None
        else:
            ret = self._mask
            if ret is None:
                if self.value is not None:
                    ret = np.zeros(self.shape, dtype=bool)
                    fill_value = self.fill_value
                    if fill_value is not None:
                        is_equal = self.value == fill_value
                        ret[is_equal] = True
        return ret

    def set_mask(self, mask, cascade=False):
        mask = np.array(mask, dtype=bool)

        assert mask.shape == self.shape
        self._mask = mask
        if cascade and self.parent is not None:
            self.parent.set_mask(self)
        else:
            # Bounds will be updated if there is a parent. Otherwise, update the bounds directly.
            if self.has_bounds:
                set_mask_by_variable(self, self.bounds)

    def allocate_value(self):
        self.value = variable_get_zeros(self.dimensions, self.dtype)

    def get_between(self, lower, upper, return_indices=False, closed=False, use_bounds=True):
        # tdk: refactor to function
        assert lower <= upper

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

    def get_distributed_slice(self, slc, comm=None):
        slc = get_formatted_slice(slc, self.ndim)
        new_dimensions = [None] * self.ndim
        dimensions = self.dimensions
        for idx in range(self.ndim):
            new_dimensions[idx] = dimensions[idx].get_distributed_slice(slc[idx], comm=comm)

        is_or_will_be_empty = self.is_empty or any([nd.is_empty for nd in new_dimensions])

        if is_or_will_be_empty:
            ret = self.copy()
        else:
            slc = get_formatted_slice(slc, self.ndim)
            local_slc = [slice(None)] * self.ndim
            for idx in range(self.ndim):
                if slc[idx] != slice(None):
                    local_slc_args = get_global_to_local_slice([slc[idx].start, slc[idx].stop],
                                                               dimensions[idx].bounds_local)
                    local_slc[idx] = slice(*local_slc_args)
            ret = self[local_slc]

        # Synchronize the dimensions.
        if is_or_will_be_empty:
            ret._dimensions = tuple(new_dimensions)
            # Ensure all variables in shared collection are also considered empty.
            if ret.parent is not None:
                for var in ret.parent.values():
                    var._is_empty = True
        else:
            ret.dimensions = new_dimensions

        return ret

    def get_scatter_slices(self, splits):
        slices = create_nd_slices(splits, self.shape)
        return slices

    def iter(self, use_mask=False, add_bounds=False, formatter=None):
        has_bounds = self.has_bounds
        name = self.name
        for idx, value in iter_array(self._get_iter_value_(), use_mask=use_mask, return_value=True):
            yld = OrderedDict()
            yld[name] = value

            if has_bounds and add_bounds:
                row = self.bounds.value[idx, :]
                lb, ub = np.min(row), np.max(row)
                yld['lb_{}'.format(name)] = lb
                yld['ub_{}'.format(name)] = ub

            if formatter is not None:
                for k, v in yld.items():
                    yld[k] = formatter(v)

            yield idx, yld

    def load(self, *args, **kwargs):
        """
        Allows variables to be fake-loaded in the case of mixed pure variables and sourced variables. Actual
        implementations is in :class:`ocgis.new_interface.variable.SourcedVariable`
        """

    def _get_iter_value_(self):
        return self.masked_value

    def write(self, *args, **kwargs):
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
        from ocgis.api.request.driver.nc import DriverNetcdf
        driver = kwargs.pop('driver', DriverNetcdf)
        args = list(args)
        args.insert(0, self)
        driver.write_variable(*args, **kwargs)

    def _get_to_conform_value_(self):
        return self.masked_value

    def _set_to_conform_value_(self, value):
        self.value = value


class SourcedVariable(Variable):
    def __init__(self, *args, **kwargs):
        # Flag to indicate if value has been loaded from sourced. This allows the value to be set to None and not have a
        # reload from source.
        self._has_initialized_value = False
        self.protected = kwargs.pop('protected', False)
        self._request_dataset = kwargs.pop('request_dataset', None)
        kwargs['attrs'] = kwargs.get('attrs') or OrderedDict()
        bounds = kwargs.pop('bounds', None)
        super(SourcedVariable, self).__init__(*args, **kwargs)

        init_from_source(self)

        if bounds is not None:
            self.bounds = bounds

    def load(self, eager=False):
        """Load all variable data from source.

        :param bool eager: If ``False``, only load this variable's data form source. If ``True``, load all data from
         source including any variables on its parent object.
        """

        # Only load the value if it has not been initialized and it is None.
        if not self._has_initialized_value:
            self._get_value_()
            if eager and self.parent is not None:
                for var in self.parent.values():
                    var.load()

    def _get_value_(self):
        if not self.is_empty and self._value is None and not self._has_initialized_value:
            self._request_dataset.driver.init_variable_value(self)
            ret = self._value
            self._has_initialized_value = True
        else:
            ret = super(SourcedVariable, self)._get_value_()
        return ret

    def _set_value_(self, value):
        # Allow value to be set to None. This will remove dimensions.
        if self._has_initialized_value and value is None:
            self._dimensions = None
        super(SourcedVariable, self)._set_value_(value)


# tdk: variable collection should inherit from abstract container
class VariableCollection(AbstractInterfaceObject, AbstractCollection, Attributes):
    def __init__(self, name=None, variables=None, attrs=None, parent=None, children=None):
        self._dimensions = OrderedDict()
        self._dimension_name_ctr = 0
        self._variable_name_ctr = 0

        self.name = name
        self.children = children or OrderedDict()
        self.parent = parent

        AbstractCollection.__init__(self)
        Attributes.__init__(self, attrs)
        AbstractInterfaceObject.__init__(self)

        if variables is not None:
            for variable in get_iter(variables, dtype=Variable):
                self.add_variable(variable)

    def __getitem__(self, item_or_slc):
        if not isinstance(item_or_slc, dict):
            ret = AbstractCollection.__getitem__(self, item_or_slc)
        else:
            # Assume a dictionary slice.
            ret = self.copy()
            for dimension_name, slc in item_or_slc.items():
                ret.dimensions[dimension_name] = self.dimensions[dimension_name].__getitem__(slc)
            names = set(item_or_slc.keys())
            for k, v in ret.items():
                with orphaned(v):
                    if v.ndim > 0:
                        v_dimension_names = set(v.dimension_names)
                        if len(v_dimension_names.intersection(names)) > 0:
                            mapped_slc = [item_or_slc[d] for d in v.dimension_names]
                            v_sub = v.__getitem__(mapped_slc)
                    else:
                        v_sub = v.copy()
                ret.add_variable(v_sub, force=True)
        return ret

    # tdk: dimensions and group can be removed with inheritance from abstractcontainer
    @property
    def dimensions(self):
        return self._dimensions

    @property
    def group(self):
        if self.parent is None:
            ret = None
        else:
            curr = self.parent
            ret = [curr.name]
            while True:
                if curr.parent is None:
                    break
                else:
                    curr = curr.parent
                    ret.append(curr.name)
            ret.reverse()
        return ret

    @property
    def shapes(self):
        return OrderedDict([[k, v.shape] for k, v in self.items() if not isinstance(v, CoordinateReferenceSystem)])

    def add_child(self, child, force=False):
        if child.name in self.children and not force:
            raise ValueError("Child with name '{}' already in parent with name '{}'.".format(child.name, self.name))
        child.parent = self
        self.children[child.name] = child

    def add_dimension(self, dimension, force=False):
        existing_dim = self.dimensions.get(dimension.name)
        if existing_dim is not None and not force:
            if existing_dim != dimension:
                raise DimensionMismatchError(dimension.name, self.name)
        else:
            self.dimensions[dimension.name] = dimension

    def add_variable(self, variable, force=False):
        """
        :param :class:`ocgis.interface.base.variable.Variable`
        """
        if not force and variable.name in self:
            raise VariableInCollectionError(variable)
        self[variable.name] = variable
        for dimension in variable.dimensions:
            self.add_dimension(dimension, force=force)

        # Allow variables to optionally overload how they are added to the collection.
        try:
            variable.__add_to_collection_finalize__(self)
        except AttributeError:
            # It is okay that is not overloaded. The variable is simply added without special operations.
            pass

        variable.parent = self

    def copy(self):
        ret = AbstractCollection.copy(self)
        ret._dimensions = ret._dimensions.copy()
        for v in ret.values():
            with orphaned(v):
                ret[v.name] = v.copy()
            ret[v.name].parent = ret
        ret.children = ret.children.copy()
        return ret

    def load(self):
        """Load all variables from source."""

        for v in self.values():
            v.load()

    def iter_data_variables(self):
        from ocgis.new_interface.geom import GeometryVariable
        for v in self.values():
            if not isinstance(v, (CoordinateReferenceSystem, GeometryVariable)):
                yield v

    @staticmethod
    def read(*args, **kwargs):
        from ocgis import RequestDataset
        rd = RequestDataset(*args, **kwargs)
        return rd.driver.get_variable_collection()

    def set_mask(self, variable, exclude=None):
        names_container = [d.name for d in variable.dimensions]
        for k, v in self.items():
            if exclude is not None and k in exclude:
                continue
            if variable.name != k and v.ndim > 0:
                names_variable = [d.name for d in v.dimensions]
                slice_map = get_mapping_for_slice(names_container, names_variable)
                if len(slice_map) > 0:
                    set_mask_by_variable(variable, v, slice_map)

    def strip(self):
        self._storage = OrderedDict()
        self.children = OrderedDict()

    def write(self, *args, **kwargs):
        # tdk: the driver argument should accept the string key of the target driver
        from ocgis.api.request.driver.nc import DriverNetcdf
        driver = kwargs.pop('driver', DriverNetcdf)
        args = list(args)
        args.insert(0, self)
        driver.write_variable_collection(*args, **kwargs)


def are_variable_and_dimensions_shape_equal(variable_value, dimensions):
    to_test = []
    vshape = variable_value.shape
    dshape = get_dimension_lengths(dimensions)

    if len(vshape) != len(dshape):
        ret = False
    else:
        is_unlimited = [d.is_unlimited for d in dimensions]
        for v, d, iu in zip(vshape, dshape, is_unlimited):
            if iu:
                to_append = True
            else:
                to_append = v == d
            to_test.append(to_append)
        ret = all(to_test)

    return ret


def get_attribute_property(variable, name):
    return variable.attrs.get(name)


def get_dimension_lengths(dimensions):
    ret = [len(d) for d in dimensions]
    return tuple(ret)


def get_dslice(dimensions, slc):
    return {d.name: s for d, s in zip(dimensions, slc)}


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


def get_mapping_for_slice(names_source, names_destination):
    to_map = set(names_source).intersection(names_destination)
    ret = []
    for name in to_map:
        ret.append([names_source.index(name), names_destination.index(name)])
    return ret


def get_shape_from_variable(variable):
    dimensions = variable._dimensions
    value = variable._value
    if dimensions is None and value is None:
        ret = tuple()
    elif dimensions is not None:
        ret = get_dimension_lengths(variable.dimensions)
    elif value is not None:
        ret = value.shape
    else:
        raise NotImplementedError()
    return ret


def get_slice_sequence_using_local_bounds(variable):
    ndim = variable.ndim
    ret = [None] * ndim
    for idx, dim in enumerate(variable.dimensions):
        lower, upper = dim.bounds_local
        ret[idx] = slice(lower, upper)
    return ret


def has_unlimited_dimension(dimensions):
    ret = False
    for d in dimensions:
        if d.is_unlimited:
            ret = True
            break
    return ret


def init_from_source(variable):
    request_dataset = variable._request_dataset
    if request_dataset is not None:
        request_dataset.driver.init_variable_from_source(variable)


def set_attribute_property(variable, name, value):
    variable.attrs[name] = value


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


def set_mask_by_variable(source_variable, target_variable, slice_map=None):
    if slice_map is None:
        names_source = [d.name for d in source_variable.dimensions]
        names_destination = [d.name for d in target_variable.dimensions]
        slice_map = get_mapping_for_slice(names_source, names_destination)
    mask_source = source_variable.get_mask()
    mask_target = target_variable.get_mask()
    # If dimensions are equivalent, do not execute the loop.
    if all([ii[0] == ii[1] for ii in slice_map]) and source_variable.ndim == target_variable.ndim:
        mask_target = mask_source
    else:
        template = [slice(None)] * target_variable.ndim
        for slc in itertools.product(*[range(ii) for ii in source_variable.shape]):
            slc = [slice(s, s + 1) for s in slc]
            if mask_source[slc]:
                for m in slice_map:
                    template[m[1]] = slc[m[0]]
                mask_target[template] = True
    target_variable.set_mask(mask_target)


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
                if d.size is None:
                    d._size_current = variable_value.shape[idx]


def variable_get_zeros(dimensions, dtype):
    new_shape = get_dimension_lengths(dimensions)
    ret = np.zeros(new_shape, dtype=dtype)
    return ret

