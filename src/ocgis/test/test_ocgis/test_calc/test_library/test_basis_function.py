from ocgis import SpatialCollection, OcgOperations, RequestDataset
from ocgis.api.parms.definition import Calc
from ocgis.calc.library.basis_function import BasisFunction
from ocgis.conv.nc import NcConverter
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
import numpy as np
from ocgis.util.helpers import iter_array


class TestBasisFunction(AbstractTestField):

    def test_init(self):
        bc = BasisFunction()

    #todo: test with different field alias
    #todo: test with basis as field v. string
    #todo: test with different time lengths
    #todo: test with dates missing

    def test_register(self):
        calc = [{'func': 'basis_function', 'name': 'basis'}]
        Calc(calc)

    def test_in_operations(self):

        def pyfunc(a, b):
            return a*b

        calc_kwds = {'pyfunc': pyfunc, 'basis': 'sub_tas', 'match': ['month', 'day']}
        calc = [{'func': 'basis_function', 'name': 'basis', 'kwds': calc_kwds}]

        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()[:, :, :, 20:22, 30:32]
        field.variables['tas'].value[:] = 2
        coll = SpatialCollection()
        coll.add_field(1, None, field)
        conv = NcConverter([coll], self._test_dir, 'all_time')
        path_field = conv.write()

        sub = field.get_time_region({'year': [2010]})
        sub.variables['tas'].value[:] = 3
        coll = SpatialCollection()
        coll.add_field(1, None, sub)
        conv = NcConverter([coll], self._test_dir, 'subsetted_time')
        path_sub = conv.write()

        rd1 = RequestDataset(uri=path_field, variable='tas')
        rd2 = RequestDataset(uri=path_sub, variable='tas', alias='sub_tas')
        ops = OcgOperations(dataset=[rd1, rd2], calc=calc)
        ret = ops.execute()
        ref = ret[1]['tas'].variables['basis']
        self.assertEqual(ref.value.min(), 6.0)
        self.assertEqual(ref.value.max(), 6.0)
        self.assertEqual(ref.value.shape, (1, 3650, 1, 2, 2))
        self.assertEqual(ret.keys(), [1])
        self.assertEqual(ret[1].keys(), ['tas'])
        self.assertEqual(ret[1]['tas'].variables.keys(), ['basis'])

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