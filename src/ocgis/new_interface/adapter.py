from abc import abstractmethod

import numpy as np

from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import get_bounds_from_1d, get_formatted_slice
from ocgis.exc import BoundsAlreadyAvailableError, EmptySubsetError
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

    def get_between(self, lower, upper, return_indices=False, closed=False, use_bounds=True):
        assert (lower <= upper)

        # Determine if data bounds are contiguous (if bounds exists for the data). Bounds must also have more than one
        # row.
        is_contiguous = False
        if self.bounds is not None:
            try:
                if len(set(self.bounds[0, :]).intersection(set(self.bounds[1, :]))) > 0:
                    is_contiguous = True
            except IndexError:
                # There is likely not a second row.
                if self.bounds.shape[0] == 1:
                    pass
                else:
                    raise

        # Subset operation when bounds are not present.
        if self.bounds is None or use_bounds == False:
            value = self.variable.value
            if closed:
                select = np.logical_and(value > lower, value < upper)
            else:
                select = np.logical_and(value >= lower, value <= upper)
        # Subset operation in the presence of bounds.
        else:
            # Determine which bound column contains the minimum.
            if self.bounds[0, 0] <= self.bounds[0, 1]:
                lower_index = 0
                upper_index = 1
            else:
                lower_index = 1
                upper_index = 0
            # Reference the minimum and maximum bounds.
            bounds_min = self.bounds[:, lower_index]
            bounds_max = self.bounds[:, upper_index]

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
            raise EmptySubsetError(origin=self.variable.name)

        ret = self[select]

        if return_indices:
            indices = np.arange(select.shape[0])
            ret = (ret, indices[select])

        return ret

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
