from abc import abstractmethod

from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import get_bounds_from_1d, get_formatted_slice
from ocgis.exc import BoundsAlreadyAvailableError
from ocgis.new_interface.base import AbstractInterfaceObject


class AbstractAdapter(AbstractInterfaceObject):
    def write_netcdf(self, dataset, **kwargs):
        for var in self._iter_variables_():
            var.write_netcdf(dataset, **kwargs)

    @abstractmethod
    def _iter_variables_(self):
        """"""


class BoundedVariable(AbstractAdapter):

    def __init__(self, variable, bounds=None):
        assert variable.ndim == 1
        if bounds is not None:
            assert bounds.ndim == 2

        self._has_extrapolated_bounds = False

        self.variable = variable
        self.bounds = bounds

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, 1)
        variable = self.variable[slc]
        if self.bounds is not None:
            bounds = self.bounds[slc, :]
        else:
            bounds = None
        return BoundedVariable(variable, bounds=bounds)

    def set_extrapolated_bounds(self, name=None):
        """Set the bounds variable using extrapolation."""

        if self.bounds is not None:
            raise BoundsAlreadyAvailableError
        name = name or '{0}_{1}'.format(self.variable.name, 'bounds')
        bounds_value = get_bounds_from_1d(self.variable.value)
        self.bounds = Variable(name, value=bounds_value)
        self._has_extrapolated_bounds = True

    def write_netcdf(self, dataset, **kwargs):
        super(BoundedVariable, self).write_netcdf(dataset, **kwargs)
        if self.bounds is not None:
            var = dataset.variables[self.variable.name]
            var.bounds = self.bounds.name

    def _iter_variables_(self):
        for attr in ['variable', 'bounds']:
            yield getattr(self, attr)
