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
        """
        Execute an arbitrary Python function ``pyfunc`` using values from ``values`` and ``basis`` on dates joined by
        ``match``. If there is no date match between ``values`` and ``basis``, then the output array is masked for the
        unmatched time steps.

        :param values: Five-dimensional array from the target variable.
        :type values: :class:`numpy.ma.core.MaskedArray`
        :param basis: The variable to use as the basis for the executable Python function.
        :type basis: :class:`ocgis.interface.base.variable.Variable`

        At the :class:`~ocgis.OcgOperations` level this is the string alias to a variable in a
        :class:`~ocgis.RequestDataset`:

        >>> kwds = [{'basis': 'tas', 'pyfunc': lambda a,b: a+b, 'match': ['month', 'day']}]
        >>> calc = [{'func': 'basis_function', 'name': 'some_basis', 'kwds': kwds}]

        If the variable is contained in a :class:`~ocgis.RequestDataset` with a different name:

        >>> basis = 'name:tas'

        :param pyfunc: The Python function to execute on slices of the arrays. The function takes two arguments. The
         first argument is the time slice from ``values``. The second argument is the time slice from ``basis``. Output
         from the function should be returned as a :class:`numpy.ma.core.MaskedArray` and return the same dimension as
         the first argument.
        :type pyfunc: function

        >>> def my_func(a, b):
        >>>     return a + b
        >>> pyfunc = my_func

        :param match: The :class:`datetime.datetime` attributes to use for date joining.
        :type match: list[str, ...]

        Match ``values`` and ``basis`` by month and year.

        >>> match = ['month', 'year']

        Only match by month. Note: Duplicate key values will be overwritten in the basis dictionary if more than one
        unique month is present in ``basis``.

        >>> match = ['month']

        :raises: AssertionError
        :rtype: :class:`numpy.ma.core.MaskedArray`
        """

        assert(basis.field is not None)
        assert(values.ndim == 5)
        assert(len(match) >= 1)

        # the returned array is masked unless the value is set during date matching. it is possible that not all dates
        # have a match. if they do not, they should remain masked.
        ret = np.ma.array(np.zeros(values.shape), mask=True, dtype=values.dtype)

        # construct a dictionary matching datetime keys to the corresponding index on the basis field. this is used to
        # select the value from the basis when it is time to execute the python function.
        basis_dict = {}
        _get_key_ = self._get_key_
        for ii, dt in enumerate(basis.field.temporal._get_datetime_value_().flat):
            basis_dict[_get_key_(dt, match)] = ii

        # for each time step, look up the corresponding matching value and execute the python function.
        value_datetime = self.field.temporal._get_datetime_value_()
        for it in range(self.field.temporal.shape[0]):
            curr_datetime = value_datetime[it]
            key = _get_key_(curr_datetime, match)
            try:
                basis_index = basis_dict[key]
            except KeyError:
                # if the date match is not available leave it as masked
                continue
            # extract the time slices from the input values and basis and execute the python function.
            basis_slice = basis.value[:, basis_index, :, :, :]
            values_slice = values[:, it, :, :, :]
            ret[:, it, :, :, :] = pyfunc(values_slice, basis_slice)

        return ret

    @staticmethod
    def _get_key_(dt, match):
        return tuple([getattr(dt, m) for m in match])