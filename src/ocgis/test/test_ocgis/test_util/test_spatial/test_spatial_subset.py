from copy import deepcopy

import numpy as np
from shapely import wkt

from ocgis import CoordinateReferenceSystem, RequestDataset
from ocgis import env
from ocgis.constants import WrappedState
from ocgis.exc import EmptySubsetError
from ocgis.interface.base.crs import CFWGS84, CFRotatedPole
from ocgis.interface.base.dimension.spatial import SpatialDimension
from ocgis.interface.base.field import Field
from ocgis.test.base import TestBase, attr
from ocgis.test.strings import GERMANY_WKT, NEBRASKA_WKT
from ocgis.test.test_ocgis.test_api.test_parms.test_definition import TestGeom
from ocgis.util.helpers import make_poly
from ocgis.util.itester import itr_products_keywords
from ocgis.util.spatial.spatial_subset import SpatialSubsetOperation


class TestSpatialSubsetOperation(TestBase):

    def __init__(self, *args, **kwargs):
        self._target = None
        super(TestSpatialSubsetOperation, self).__init__(*args, **kwargs)

    def __iter__(self):
        keywords = dict(target=self.target,
                        output_crs=self.get_output_crs(),
                        wrap=[None, True, False])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            kwargs = k._asdict()
            target = kwargs.pop('target')
            ss = SpatialSubsetOperation(target, **kwargs)
            yield (ss, k)

    @property
    def germany(self):
        germany = self.get_buffered(wkt.loads(GERMANY_WKT))
        germany = {'geom': germany, 'properties': {'UGID': 2, 'DESC': 'Germany'}}
        return germany

    @property
    def nebraska(self):
        nebraska = self.get_buffered(wkt.loads(NEBRASKA_WKT))
        nebraska = {'geom': nebraska, 'properties': {'UGID': 1, 'DESC': 'Nebraska'}}
        return nebraska

    @property
    def rd_rotated_pole(self):
        rd = self.test_data.get_rd('rotated_pole_cccma')
        return rd

    @property
    def target(self):
        if self._target is None:
            self._target = self.get_target()
        return self._target

    def get_buffered(self, geom):
        ret = geom.buffer(0)
        self.assertTrue(ret.is_valid)
        return ret

    def get_output_crs(self):
        crs_wgs84 = CFWGS84()
        ret = ['input', crs_wgs84]
        return ret

    def get_subset_sdim(self):

        # 1: nebraska
        nebraska = self.nebraska

        # 2: germany
        germany = self.germany

        # 3: nebraska and germany

        ret = [SpatialDimension.from_records(d) for d in [[nebraska], [germany], [nebraska, germany]]]

        return ret

    def get_target(self):
        # 1: standard input file - geographic coordinate system, unwrapped
        rd_standard = self.test_data.get_rd('cancm4_tas')

        # 2: standard field - geographic coordinate system
        field_standard = rd_standard.get()

        # 3: field with rotated pole coordinate system
        field_rotated_pole = self.rd_rotated_pole.get()

        # 4: field with lambert conformal coordinate system
        rd = self.test_data.get_rd('narccap_lambert_conformal')
        field_lambert = rd.get()

        # 5: standard input field - geographic coordinate system, wrapped
        field_wrapped = rd_standard.get()
        field_wrapped.spatial.wrap()

        # 6: spatial dimension - standard geographic coordinate system
        sdim = field_standard.spatial

        ret = [rd_standard, field_standard, field_rotated_pole, field_lambert, field_wrapped, sdim]

        return ret

    @attr('data')
    def test_init_output_crs(self):
        for ss, k in self:
            if k.output_crs is None:
                if isinstance(k.target, Field):
                    self.assertEqual(ss.sdim.crs, k.target.spatial.crs)

    @attr('data')
    def test_field(self):
        for ss, k in self:
            try:
                self.assertIsInstance(ss.field, Field)
            except AttributeError:
                if isinstance(k.target, SpatialDimension):
                    continue
                else:
                    raise

    def test_get_buffered_subset_sdim(self):
        proj4 = '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs'
        buffer_crs_list = [None, CoordinateReferenceSystem(proj4=proj4)]
        poly = make_poly((36, 44), (-104, -95))

        for buffer_crs in buffer_crs_list:
            subset_sdim = SpatialDimension.from_records([{'geom': poly, 'properties': {'UGID': 1}}], crs=CFWGS84())
            self.assertEqual(subset_sdim.crs, CFWGS84())
            if buffer_crs is None:
                buffer_value = 1
            else:
                buffer_value = 10

            ret = SpatialSubsetOperation._get_buffered_subset_sdim_(subset_sdim, buffer_value, buffer_crs=buffer_crs)
            ref = ret.geom.polygon.value[0, 0]

            if buffer_crs is None:
                self.assertEqual(ref.bounds, (-105.0, 35.0, -94.0, 45.0))
            else:
                self.assertNumpyAllClose(np.array(ref.bounds), np.array((-104.00013263459613, 35.9999147913708, -94.99986736540386, 44.00008450528758)))
            self.assertEqual(subset_sdim.crs, ret.crs)

            # check deepcopy
            ret.geom.polygon.value[0, 0] = make_poly((1, 2), (3, 4))
            ref_buffered = ret.geom.polygon.value[0, 0]
            ref_original = subset_sdim.geom.polygon.value[0, 0]
            with self.assertRaises(AssertionError):
                self.assertNumpyAllClose(np.array(ref_buffered.bounds), np.array(ref_original.bounds))

    @attr('data')
    def test_get_should_wrap(self):
        # a 360 dataset
        field_360 = self.test_data.get_rd('cancm4_tas').get()
        ss = SpatialSubsetOperation(field_360, wrap=True)
        self.assertTrue(ss._get_should_wrap_(ss.target))
        ss = SpatialSubsetOperation(field_360, wrap=False)
        self.assertFalse(ss._get_should_wrap_(ss.target))
        ss = SpatialSubsetOperation(field_360, wrap=None)
        self.assertFalse(ss._get_should_wrap_(ss.target))

        # wrapped dataset
        field_360.spatial.wrap()
        ss = SpatialSubsetOperation(field_360, wrap=True)
        self.assertFalse(ss._get_should_wrap_(ss.target))
        ss = SpatialSubsetOperation(field_360, wrap=False)
        self.assertFalse(ss._get_should_wrap_(ss.target))
        ss = SpatialSubsetOperation(field_360, wrap=None)
        self.assertFalse(ss._get_should_wrap_(ss.target))

    @attr('slow')
    def test_get_spatial_subset(self):
        ctr_test = 0
        ctr = 0
        for ss, k in self:
            for subset_sdim in self.get_subset_sdim():
                for operation in ['intersects', 'clip', 'foo']:

                    use_subset_sdim = deepcopy(subset_sdim)
                    use_ss = deepcopy(ss)

                    # ctr += 1
                    # print ctr
                    # if ctr != 73:
                    #     continue
                    # else:

                    try:
                        ret = use_ss.get_spatial_subset(operation, use_subset_sdim, use_spatial_index=True,
                                                        select_nearest=False, buffer_value=None, buffer_crs=None)
                    except ValueError:
                        # 'foo' is not a valid type of subset operation.
                        if operation == 'foo':
                            continue
                        # only one polygon for a spatial operation
                        elif use_subset_sdim.shape != (1, 1):
                            continue
                        else:
                            raise
                    except EmptySubsetError:
                        # subset tests occur on the spatial dimension operations
                        continue
                    try:
                        self.assertIsInstance(ret, type(use_ss.target))
                    except AssertionError:
                        # if the target is a request datasets, then the output should be a field
                        if isinstance(use_ss.target, RequestDataset):
                            self.assertIsInstance(ret, Field)
                    ctr_test += 1
        self.assertGreater(ctr_test, 5)

    @attr('data')
    def test_get_spatial_subset_circular_geometries(self):
        """Test circular geometries. They were causing wrapping errors."""

        geoms = TestGeom.get_geometry_dictionaries()
        rd = self.test_data.get_rd('cancm4_tas')
        ss = SpatialSubsetOperation(rd, wrap=True)
        buffered = [element['geom'].buffer(rd.get().spatial.grid.resolution*2) for element in geoms]
        for buff in buffered:
            record = [{'geom': buff, 'properties': {'UGID': 1}}]
            subset_sdim = SpatialDimension.from_records(record)
            ret = ss.get_spatial_subset('intersects', subset_sdim)
            self.assertTrue(np.all(ret.spatial.grid.extent > 0))

    @attr('data')
    def test_get_spatial_subset_output_crs(self):
        """Test subsetting with an output CRS."""

        # test with default crs converting to north american lambert
        proj4 = '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs'
        output_crs = CoordinateReferenceSystem(proj4=proj4)
        subset_sdim = SpatialDimension.from_records([self.nebraska])
        rd = self.test_data.get_rd('cancm4_tas')
        ss = SpatialSubsetOperation(rd, output_crs=output_crs)
        ret = ss.get_spatial_subset('intersects', subset_sdim)
        self.assertEqual(ret.spatial.crs, output_crs)
        self.assertAlmostEqual(ret.spatial.grid.value.mean(), -35065.750850951554)

        # test with an input rotated pole coordinate system
        rd = self.rd_rotated_pole
        ss = SpatialSubsetOperation(rd, output_crs=env.DEFAULT_COORDSYS)
        subset_sdim = SpatialDimension.from_records([self.germany])
        ret = ss.get_spatial_subset('intersects', subset_sdim)
        self.assertEqual(ret.spatial.crs, env.DEFAULT_COORDSYS)

    @attr('data')
    def test_get_spatial_subset_rotated_pole(self):
        """Test input has rotated pole with now output CRS."""

        rd = self.rd_rotated_pole
        ss = SpatialSubsetOperation(rd)
        subset_sdim = SpatialDimension.from_records([self.germany])
        ret = ss.get_spatial_subset('intersects', subset_sdim)
        self.assertEqual(ret.spatial.crs, rd.get().spatial.crs)
        self.assertAlmostEqual(ret.spatial.grid.value.data.mean(), -2.0600000000000009)

    @attr('data')
    def test_get_spatial_subset_wrap(self):
        """Test subsetting with wrap set to a boolean value."""

        subset_sdim = SpatialDimension.from_records([self.nebraska])
        rd = self.test_data.get_rd('cancm4_tas')
        self.assertEqual(rd.get().spatial.wrapped_state, WrappedState.UNWRAPPED)
        ss = SpatialSubsetOperation(rd, wrap=True)
        ret = ss.get_spatial_subset('intersects', subset_sdim)
        self.assertEqual(ret.spatial.wrapped_state, WrappedState.WRAPPED)
        self.assertAlmostEqual(ret.spatial.grid.value.data[1].mean(), -99.84375)

        # test with wrap false
        ss = SpatialSubsetOperation(rd, wrap=False)
        ret = ss.get_spatial_subset('intersects', subset_sdim)
        self.assertEqual(ret.spatial.wrapped_state, WrappedState.UNWRAPPED)
        self.assertAlmostEqual(ret.spatial.grid.value.data[1].mean(), 260.15625)

    @attr('data')
    def test_prepare_target(self):
        for ss, k in self:
            self.assertIsNone(ss._original_rotated_pole_state)
            if isinstance(ss.sdim.crs, CFRotatedPole):
                ss._prepare_target_()
                self.assertIsInstance(ss._original_rotated_pole_state, CFRotatedPole)
                self.assertIsInstance(ss.sdim.crs, CFWGS84)
            else:
                ss._prepare_target_()
                self.assertIsNone(ss._original_rotated_pole_state)

    @attr('data')
    def test_prepare_subset_sdim(self):
        for subset_sdim in self.get_subset_sdim():
            for ss, k in self:
                try:
                    prepared = ss._prepare_subset_sdim_(subset_sdim)
                    # check that a deepcopy has occurred
                    self.assertFalse(np.may_share_memory(prepared.uid, subset_sdim.uid))
                except KeyError:
                    # the target has a rotated pole coordinate system. transformations to rotated pole for the subset
                    # geometry is not supported.
                    if isinstance(ss.sdim.crs, CFRotatedPole):
                        continue
                    else:
                        raise
                self.assertEqual(prepared.crs, ss.sdim.crs)

        # test nebraska against an unwrapped dataset specifically
        nebraska = SpatialDimension.from_records([self.nebraska])
        field = self.test_data.get_rd('cancm4_tas').get()
        ss = SpatialSubsetOperation(field)
        prepared = ss._prepare_subset_sdim_(nebraska)
        self.assertEqual(prepared.wrapped_state, WrappedState.UNWRAPPED)

    @attr('data')
    def test_sdim(self):
        for ss, k in self:
            self.assertIsInstance(ss.sdim, SpatialDimension)

    @attr('data')
    def test_should_update_crs(self):
        # no output crs provided
        target = self.test_data.get_rd('cancm4_tas')
        ss = SpatialSubsetOperation(target)
        self.assertFalse(ss.should_update_crs)

        # output crs different than input
        ss = SpatialSubsetOperation(target, output_crs=CoordinateReferenceSystem(epsg=2136))
        self.assertTrue(ss.should_update_crs)

        # same output crs as input
        ss = SpatialSubsetOperation(target, output_crs=ss.sdim.crs)
        self.assertFalse(ss.should_update_crs)
