from collections import OrderedDict
from ocgis import SpatialCollection, OcgOperations, RequestDataset
from ocgis.api.parms.definition import Calc
from ocgis.calc.library.basis_function import BasisFunction
from ocgis.conv.nc import NcConverter
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
import numpy as np
from ocgis.util.helpers import iter_array
from ocgis.util.itester import itr_products_keywords


class TestBasisFunction(AbstractTestField):

    def test_init(self):
        bc = BasisFunction()

    def test_register(self):
        calc = [{'func': 'basis_function', 'name': 'basis'}]
        Calc(calc)

    def get_nc_subsets(self):
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
        return path_field, path_sub

    def test_in_operations_complex(self):
        """Test a more complicated operations involving geometry subsets and different time selections."""

        geom = 'state_boundaries'
        select_ugid = [15, 24]

        def pyfunc(a, b):
            return a-b

        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas', kwds={'alias': 'sub_tas', 'time_region': {'year': [2001]}})
        calc_kwds = {'pyfunc': pyfunc, 'basis': 'sub_tas', 'match': ['month', 'day']}
        calc = [{'func': 'basis_function', 'name': 'basis', 'kwds': calc_kwds}]
        ops = OcgOperations(calc=calc, dataset=[rd1, rd2], geom=geom, select_ugid=select_ugid)
        ret = ops.execute()
        self.assertEqual(ret.keys(), [15, 24])
        for ugid, field_dict in ret.iteritems():
            self.assertEqual(field_dict.keys(), ['tas'])

    def test_in_operations(self):

        path_field, path_sub = self.get_nc_subsets()

        def pyfunc(a, b):
            return a*b

        keywords = dict(rd2_name=[None, 'rd2'],
                        basis_alias=['rd2:sub_tas', 'sub_tas'])

        for ctr, k in enumerate(itr_products_keywords(keywords, as_namedtuple=True)):

            calc_kwds = {'pyfunc': pyfunc, 'basis': k.basis_alias, 'match': ['month', 'day']}
            calc = [{'func': 'basis_function', 'name': 'basis', 'kwds': calc_kwds}]

            rd1 = RequestDataset(uri=path_field, variable='tas')
            rd2 = RequestDataset(uri=path_sub, variable='tas', alias='sub_tas', name=k.rd2_name)
            ops = OcgOperations(dataset=[rd1, rd2], calc=calc, prefix=str(ctr))

            try:
                ret = ops.execute()
            except KeyError:
                to_test = k._asdict()
                if to_test == OrderedDict([('basis_alias', 'rd2:sub_tas'), ('rd2_name', None)]):
                    continue
                elif to_test == OrderedDict([('basis_alias', 'sub_tas'), ('rd2_name', 'rd2')]):
                    continue
                else:
                    raise

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
        parms = dict(basis=basis.variables['tmax'], pyfunc=pyfunc, match=['month', 'day'])
        bc = BasisFunction(field=field, parms=parms)
        ret = bc.execute()
        actual = field.variables.first().value + basis.variables.first().value
        actual = actual.astype(BasisFunction.dtype)
        self.assertNumpyAll(ret.first().value, actual)

    def test_execute_missing_dates(self):
        """Test missing dates are appropriately masked in the data output."""

        def pyfunc(a, b):
            return a+b

        field = self.get_field(with_value=True)
        basis = self.get_field(with_value=True)

        idx = range(0, 2) + range(3, 7) + range(8, basis.temporal.shape[0])
        basis = basis[:, idx, :, :, :]
        parms = dict(basis=basis.variables['tmax'], pyfunc=pyfunc, match=['month', 'day'])
        bc = BasisFunction(field=field, parms=parms)
        ret = bc.execute()

        ref = ret['basis_function'].value
        self.assertTrue(ref[:, 2, :, :, :].mask.all())
        self.assertTrue(ref[:, 7, :, :, :].mask.all())
