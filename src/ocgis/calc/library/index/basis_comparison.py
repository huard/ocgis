from ocgis import constants
from ocgis.calc.base import AbstractParameterizedFunction, AbstractUnivariateFunction
import numpy as np


class BasisFunction(AbstractUnivariateFunction, AbstractParameterizedFunction):
    description = ''
    dtype = constants.np_float
    key = 'basis_function'
    long_name = 'Operation with Basis Time Series'
    standard_name = 'basis_function'
    parms_definition = {'pyfunc': None, 'basis': '_variable_', 'match': None}

    def calculate(self, values, basis=None, pyfunc=None, match=None):
        assert(basis.field is not None)
        assert(values.ndim == 5)
        assert(len(basis.variables.keys()) == 1)

        ret = np.empty_like(values)

        basis_dict = {}
        for ii, dt in enumerate(basis.field.temporal._get_datetime_value_().flat):
            basis_dict[self._get_key_(dt, match)] = ii

        for it in range(self.field.temporal.shape[0]):
            curr_datetime = self.field.temporal._get_datetime_value_()[it]
            key = self._get_key_(curr_datetime, match)
            basis_index = basis_dict[key]
            basis_slice = basis.variables.first().value[:, basis_index, :, :, :]
            values_slice = values[:, it, :, :, :]
            ret[:, it, :, :, :] = pyfunc(values_slice, basis_slice)

        return ret

    @staticmethod
    def _get_key_(dt, match):
        return tuple([getattr(dt, m) for m in match])