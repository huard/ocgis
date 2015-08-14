from itertools import izip

from copy import copy, deepcopy

from numpy.ma import MaskedArray
import numpy as np

from ocgis.exc import VariableInCollectionError
from ocgis.new_interface.dimension import Dimension
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
        # tdk: consider option for not deepcopying the dimensions here
        ret.dimensions = [Dimension(dim.name, length=shp) for dim, shp in izip(self.dimensions, value.shape)]
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
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        if value is not None:
            value = deepcopy(value)
            value = tuple(get_iter(value, dtype=self.__class__))
        self._dimensions = value

    @property
    def shape(self):
        if self.dimensions is None:
            ret = tuple()
        else:
            ret = tuple([d.length for d in self.dimensions])
        return ret

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is not None:
            if not isinstance(value, MaskedArray):
                value = np.ma.array(value, dtype=self.dtype, fill_value=self.fill_value)
        if self.dimensions is not None:
            assert value.shape == self.shape
        self._value = value


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
