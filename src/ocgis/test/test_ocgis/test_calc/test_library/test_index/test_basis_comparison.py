from ocgis.calc.library.index.basis_comparison import BasisFunction
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
import numpy as np


class TestBasisFunction(AbstractTestField):

    def test_init(self):
        bc = BasisFunction()

    def test_get_basis(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()

        keywords = dict(
            source=[
                {(12, 30): np.array([[1, 2], [3, 4]])},
                {(12, 30, 1945): np.array([[1, 2], [3, 4]])},
                {(12,): np.array([[1, 2], [3, 4]])},
                field,
                rd
            ],
            match=[['month'], ['month', 'day'], ['month', 'day', 'year']]
        )

    def test_register(self):
        raise NotImplementedError

    def test_execute(self):

        def pyfunc(a, b):
            return a+b

        field = self.get_field(with_value=True)
        basis = self.get_field(with_value=True)
        parms = dict(basis=basis, pyfunc=pyfunc, match=['month', 'day'])
        bc = BasisFunction(field=field, parms=parms)
        ret = bc.execute()
        actual = field.variables.first().value + basis.variables.first().value
        actual = actual.astype(BasisFunction.dtype)
        self.assertNumpyAll(ret.first().value, actual)

    def test_with_grouping(self):
        """Test grouping methods not supported."""
        raise NotImplementedError