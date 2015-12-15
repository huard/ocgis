from copy import copy
from itertools import izip

import numpy as np
from numpy.ma import MaskedArray

from ocgis import constants
from ocgis.api.collection import AbstractCollection
from ocgis.exc import VariableInCollectionError, VariableShapeMismatch, BoundsAlreadyAvailableError, EmptySubsetError, \
    ResolutionError
from ocgis.interface.base.attributes import Attributes
from ocgis.interface.base.dimension.base import get_none_or_array
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.dimension import Dimension, SourcedDimension
from ocgis.util.helpers import get_iter, get_formatted_slice, get_bounds_from_1d


class Variable(AbstractInterfaceObject, Attributes):
    # tdk:doc
    # tdk:support unlimited dimensions

    def __init__(self, name=None, value=None, dimensions=None, dtype=None, alias=None, attrs=None, fill_value=None):
        self._alias = None
        self._dimensions = None
        self._value = None
        self._dtype = None
        self._fill_value = None

        self.name = name

        self.dtype = dtype
        self.fill_value = fill_value
        self.alias = alias
        self.dimensions = dimensions
        self.value = value

        Attributes.__init__(self, attrs=attrs)

    def __getitem__(self, slc):
        ret = copy(self)
        slc = get_formatted_slice(slc, len(self.shape))
        value = self.value.__getitem__(slc)
        if self.dimensions is not None:
            ret.dimensions = [d[s] for d, s in izip(self.dimensions, get_iter(slc, dtype=slice))]
        ret.value = value
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
        return self._get_dimensions_()

    @dimensions.setter
    def dimensions(self, value):
        self._set_dimensions_(value)

    def _get_dimensions_(self):
        return self._dimensions

    def _set_dimensions_(self, value):
        if value is not None:
            value = tuple(get_iter(value, dtype=Dimension))
        self._dimensions = value

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
        if self.dimensions is None:
            if self.value is None:
                ret = tuple()
            else:
                ret = self.value.shape
        else:
            ret = tuple([len(d) for d in self.dimensions])
        return ret

    # property(value) ##################################################################################################

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
                value = np.ma.array(value, dtype=self.dtype, fill_value=self.fill_value, mask=False)
            self._validate_value_(value)
        self._value = value

    def _validate_value_(self, value):
        if self.dimensions is not None:
            assert value.shape == self.shape

    ####################################################################################################################

    def create_dimensions(self, names=None):
        assert self.dimensions is None
        if names is None:
            names = [self.name]
            for ii in range(1, self.ndim):
                names.append('{0}_{1}'.format(self.name, ii))
        new_dimensions = []
        for name, shp in izip(get_iter(names), self.shape):
            new_dimensions.append(Dimension(name, length=shp))
        self.dimensions = new_dimensions

    def write_netcdf(self, dataset, file_only=False, **kwargs):
        """
        Write the field object to an open netCDF dataset object.

        :param dataset: The open dataset object.
        :type dataset: :class:`netCDF4.Dataset`
        :param bool file_only: If ``True``, we are not filling the value variables. Only the file schema and dimension
         values will be written.
        :param kwargs: Extra keyword arguments in addition to ``dimensions`` and ``fill_value`` to pass to
         ``createVariable``. See http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html#createVariable
        """

        if self.dimensions is None:
            self.create_dimensions()
        for dim in self.dimensions:
            create_dimension_or_pass(dim, dataset)
        dimensions = [d.name for d in self.dimensions]
        var = dataset.createVariable(self.name, self.dtype, dimensions=dimensions, fill_value=self.fill_value, **kwargs)
        if not file_only:
            var[:] = self.value
        self.write_attributes_to_netcdf_object(var)


class BoundedVariable(Variable):
    def __init__(self, *args, **kwargs):
        self._bounds = None

        self.bounds = kwargs.pop('bounds', None)
        self._has_extrapolated_bounds = False

        super(BoundedVariable, self).__init__(*args, **kwargs)

        assert self.ndim == 1

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, 1)
        ret = super(BoundedVariable, self).__getitem__(slc)
        if self.bounds is not None:
            bounds = self.bounds[slc, :]
        else:
            bounds = None
        ret.bounds = bounds
        return ret

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        value = get_none_or_array(value, 2)
        if value is not None:
            assert value.ndim == 2
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

    def get_between(self, lower, upper, return_indices=False, closed=False, use_bounds=True):
        # tdk: refactor to function
        assert (lower <= upper)

        # Determine if data bounds are contiguous (if bounds exists for the data). Bounds must also have more than one
        # row.
        is_contiguous = False
        if self.bounds is not None:
            bounds_value = self.bounds
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
        bounds_value = get_bounds_from_1d(self.value)
        self.bounds = Variable(name, value=bounds_value)
        self._has_extrapolated_bounds = True

    def write_netcdf(self, dataset, **kwargs):
        super(BoundedVariable, self).write_netcdf(dataset, **kwargs)
        if self.bounds is not None:
            self.bounds.write_netcdf(dataset, **kwargs)
            var = dataset.variables[self.name]
            var.bounds = self.bounds.name


class SourcedVariable(Variable):
    # tdk: allow multiple variables to be opened with a single dataset open call?
    # tdk: rename 'data' to 'request_dataset'
    def __init__(self, *args, **kwargs):
        if kwargs.get('value') is None and kwargs.get('request_dataset') is None:
            msg = 'A "value" or "request_dataset" is required.'
            raise ValueError(msg)

        self._request_dataset = kwargs.pop('request_dataset', None)

        super(SourcedVariable, self).__init__(*args, **kwargs)

    def __getitem__(self, slc):
        if self._value is None:
            slc = get_formatted_slice(slc, len(self.dimensions))
            ret = copy(self)
            ret.dimensions = [d[s] for d, s in izip(self.dimensions, get_iter(slc, dtype=slice))]
        else:
            ret = super(SourcedVariable, self).__getitem__(slc)
        return ret

    @property
    def dtype(self):
        if self._dtype is None:
            self._set_metadata_from_source_()
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def fill_value(self):
        if self._fill_value is None:
            self._set_metadata_from_source_()
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value):
        self._fill_value = value

    def _get_dimensions_(self):
        if self._dimensions is None:
            self._set_metadata_from_source_()
        return self._dimensions

    def _set_metadata_from_source_(self):
        ds = self._request_dataset.driver.open()
        try:
            var = ds.variables[self.name]

            if self._dimensions is None:
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
                super(SourcedVariable, self)._set_dimensions_(new_dimensions)

            if self._dtype is None:
                self.dtype = var.dtype

            if self._fill_value is None:
                self.fill_value = var.__dict__.get('_FillValue')

            self.attrs.update(var.__dict__)
        finally:
            ds.close()

    def _get_value_(self):
        if self._value is None:
            value = self._get_value_from_source_()
            super(SourcedVariable, self)._set_value_(value)
        return super(SourcedVariable, self)._get_value_()

    def _get_value_from_source_(self):
        ds = self._request_dataset.driver.open()
        try:
            var = ds.variables[self.name]
            slc = get_formatted_slice([d._src_idx for d in self.dimensions], len(self.shape))
            return var.__getitem__(slc)
        finally:
            ds.close()


class VariableCollection(AbstractInterfaceObject, AbstractCollection):
    # tdk: should test for equivalence of dimensions on variables

    def __init__(self, variables=None):
        super(VariableCollection, self).__init__()

        if variables is not None:
            for variable in get_iter(variables, dtype=Variable):
                self.add_variable(variable)

    def __getitem__(self, slc):
        variables = [v.__getitem__(slc) for v in self.itervalues()]
        return VariableCollection(variables=variables)

    @property
    def shape(self):
        return self.first().shape

    def add_variable(self, variable):
        """
        :param :class:`ocgis.interface.base.variable.Variable`
        """

        assert isinstance(variable, Variable)
        if variable.alias in self:
            raise VariableInCollectionError(variable)
        if len(self) > 0:
            if variable.shape != self.shape:
                raise VariableShapeMismatch(variable, self.shape)

        self[variable.alias] = variable


def create_dimension_or_pass(dim, dataset):
    if dim.name not in dataset.dimensions:
        dataset.createDimension(dim.name, dim.length)
