from ocgis.util.helpers import get_bounds_from_1d, get_formatted_slice
from ocgis.exc import BoundsAlreadyAvailableError
from ocgis.new_interface.base import AbstractInterfaceObject


class AbstractAdapter(AbstractInterfaceObject):
    pass


class BoundedVariable(AbstractAdapter):
    def __init__(self, variable, bounds=None):
        assert variable.ndim == 1
        if bounds is None:
            assert variable.ndim == 2

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

    def set_extrapolated_bounds(self):
        """Set the bounds variable using extrapolation."""

        if self.bounds is not None:
            raise BoundsAlreadyAvailableError
        self.bounds = get_bounds_from_1d(self.value)
        self._has_extrapolated_bounds = True
