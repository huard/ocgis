from itertools import izip
from copy import copy

from numpy.ma import MaskedArray
import numpy as np

from ocgis.exc import VariableInCollectionError, VariableShapeMismatch
from ocgis.new_interface.dimension import Dimension, SourcedDimension
from ocgis.util.helpers import get_iter, get_formatted_slice
from ocgis.api.collection import AbstractCollection
from ocgis.interface.base.attributes import Attributes
from ocgis.new_interface.base import AbstractInterfaceObject


class Variable(AbstractInterfaceObject, Attributes):
    # tdk:doc
    # tdk:support unlimited dimensions

    def __init__(self, name, value=None, dimensions=None, dtype=None, alias=None, attrs=None, fill_value=None):
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
        return ret

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
        if self.value is not None:
            ret = self.value.dtype
        else:
            ret = self._dtype
        return ret

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    # property(dimensions) #############################################################################################

    def _get_dimensions_(self):
        return self._dimensions

    def _set_dimensions_(self, value):
        if value is not None:
            value = tuple(get_iter(value, dtype=Dimension))
        self._dimensions = value

    dimensions = property(_get_dimensions_, _set_dimensions_)

    ####################################################################################################################

    @property
    def fill_value(self):
        if self.value is not None:
            ret = self.value.fill_value
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

    def _get_value_(self):
        return self._value

    def _set_value_(self, value):
        if value is not None:
            if not isinstance(value, MaskedArray):
                value = np.ma.array(value, dtype=self.dtype, fill_value=self.fill_value, mask=False)
            if self.dimensions is not None:
                assert value.shape == self.shape
        self._value = value

    value = property(_get_value_, _set_value_)

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


class SourcedVariable(Variable):
    def __init__(self, *args, **kwargs):
        self._data = kwargs.pop('data')

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

    dimensions = property(_get_dimensions_, Variable._set_dimensions_)

    def _set_metadata_from_source_(self):
        ds = self._data.driver.open()
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
                self.fill_value = var.__dict__.get('_Fill_Value')

            if self._attrs is None:
                self.attrs = var.__dict__
        finally:
            ds.close()

    def _get_value_(self):
        if self._value is None:
            value = self._get_value_from_source_()
            super(self.__class__, self)._set_value_(value)
        return super(self.__class__, self)._get_value_()

    value = property(_get_value_, Variable._set_value_)

    def _get_value_from_source_(self):
        ds = self._data.driver.open()
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
