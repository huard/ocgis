from itertools import izip

from copy import copy

from numpy.ma import MaskedArray
import numpy as np

from ocgis.exc import VariableInCollectionError
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

    def _get_dimensions_(self):
        return self._dimensions

    def _set_dimensions_(self, value):
        if value is not None:
            value = tuple(get_iter(value, dtype=Dimension))
        self._dimensions = value

    dimensions = property(_get_dimensions_, _set_dimensions_)

    @property
    def shape(self):
        if self.dimensions is None:
            ret = tuple()
        else:
            ret = tuple([len(d) for d in self.dimensions])
        return ret

    def _get_value_(self):
        return self._value

    def _set_value_(self, value):
        if value is not None:
            if not isinstance(value, MaskedArray):
                value = np.ma.array(value, dtype=self.dtype, fill_value=self.fill_value)
            if self.dimensions is not None:
                assert value.shape == self.shape
        self._value = value

    value = property(_get_value_, _set_value_)


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

    def _get_dimensions_(self):
        if self._dimensions is None:
            self._dimensions = self._get_dimensions_from_source_data_()
        return self._dimensions

    dimensions = property(_get_dimensions_, Variable._set_dimensions_)

    def _get_dimensions_from_source_data_(self):
        ds = self._data.driver.open()
        try:
            var = ds.variables[self.name]
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
            return tuple(new_dimensions)
        finally:
            ds.close()

    def _get_value_(self):
        if self._value is None:
            value = self._get_value_from_source_data_()
            super(self.__class__, self)._set_value_(value)
        return super(self.__class__, self)._get_value_()

    value = property(_get_value_, Variable._set_value_)

    def _get_value_from_source_data_(self):
        ds = self._data.driver.open()
        try:
            var = ds.variables[self.name]
            slc = get_formatted_slice([d._src_idx for d in self.dimensions], len(self.shape))
            return var.__getitem__(slc)
        finally:
            ds.close()


class VariableCollection(AbstractInterfaceObject, AbstractCollection):
    # tdk: doc
    # tdk: test

    def __init__(self, variables=None):
        super(VariableCollection, self).__init__()

        if variables is not None:
            for variable in get_iter(variables, dtype=Variable):
                self.add_variable(variable)

    def add_variable(self, variable):
        """
        :param :class:`ocgis.interface.base.variable.Variable`
        """

        assert isinstance(variable, Variable)
        try:
            assert variable.alias not in self
        except AssertionError:
            raise VariableInCollectionError(variable)

        self[variable.alias] = variable

    def get_sliced_variables(self, slc):
        variables = [v.__getitem__(slc) for v in self.itervalues()]
        ret = VariableCollection(variables=variables)
        return ret
