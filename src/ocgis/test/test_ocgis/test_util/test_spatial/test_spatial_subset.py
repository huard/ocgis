from copy import deepcopy

import numpy as np
from shapely import wkt

from ocgis import CoordinateReferenceSystem
from ocgis import env
from ocgis.constants import WrappedState
from ocgis.exc import EmptySubsetError
from ocgis.interface.base.crs import CFWGS84, CFRotatedPole, WGS84, CFSpherical
from ocgis.interface.base.field import Field
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.geom import GeometryVariable
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
            ss = SpatialSubsetOperation(target.values()[0], **kwargs)
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

    def get_subset_geometries(self):
        nebraska = self.nebraska
        germany = self.germany
        ret = [nebraska, germany]
        return ret

    def get_target(self):
        # 1: standard input file - geographic coordinate system, unwrapped
        rd_standard = self.test_data.get_rd('cancm4_tas')

        # 2: standard field - geographic coordinate system
        field_standard = rd_standard.get()

        # 3: field with rotated pole coordinate system
        field_rotated_pole = self.rd_rotated_pole.get()

        # 4: field with lambert conformal coordinate system
        lambert_dmap = {'crs': {'variable': u'Lambert_Conformal'}, 'groups': {},
                        'time': {'variable': u'time', 'names': [u'time'], 'bounds': u'time_bnds'},
                        'x': {'variable': 'xc', 'names': ['xc']},
                        'y': {'variable': 'yc', 'names': ['yc']}}
        rd = self.test_data.get_rd('narccap_lambert_conformal', kwds={'dimension_map': lambert_dmap})
        field_lambert = rd.get()

        # 5: standard input field - geographic coordinate system, wrapped
        field_wrapped = rd_standard.get()
        field_wrapped.wrap()

        fields = [{'standard': field_standard}, {'lambert': field_lambert}, {'wrapped': field_wrapped},
                  {'rotated_pole': field_rotated_pole}]

        return fields

    @attr('data')
    def test_init_output_crs(self):
        for ss, k in self:
            if k.output_crs is None:
                if isinstance(k.target, Field):
                    self.assertEqual(ss.sdim.crs, k.target.spatial.crs)

    def test_get_buffered_geometry(self):
        proj4 = '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs'
        buffer_crs_list = [None, CoordinateReferenceSystem(proj4=proj4)]
        poly = make_poly((36, 44), (-104, -95))

        for buffer_crs in buffer_crs_list:
            # subset_sdim = SpatialDimension.from_records([{'geom': poly, 'properties': {'UGID': 1}}], crs=CFWGS84())
            gvar = GeometryVariable(value=poly, name='geoms', dimensions='dim', crs=CFWGS84())
            self.assertEqual(gvar.crs, CFWGS84())
            if buffer_crs is None:
                buffer_value = 1
            else:
                buffer_value = 10

            ret = SpatialSubsetOperation._get_buffered_geometry_(gvar, buffer_value, buffer_crs=buffer_crs)
            ref = ret.value[0]

            if buffer_crs is None:
                self.assertEqual(ref.bounds, (-105.0, 35.0, -94.0, 45.0))
            else:
                self.assertNumpyAllClose(np.array(ref.bounds), np.array((-104.00013263459613, 35.9999147913708, -94.99986736540386, 44.00008450528758)))
            self.assertEqual(gvar.crs, ret.crs)

            # check deepcopy
            ret.value[0] = make_poly((1, 2), (3, 4))
            ref_buffered = ret.value[0]
            ref_original = gvar.value[0]
            with self.assertRaises(AssertionError):
                self.assertNumpyAllClose(np.array(ref_buffered.bounds), np.array(ref_original.bounds))

    @attr('data')
    def test_get_should_wrap(self):
        # A 360 dataset.
        field_360 = self.test_data.get_rd('cancm4_tas').get()
        ss = SpatialSubsetOperation(field_360, wrap=True)
        self.assertTrue(ss._get_should_wrap_(ss.field))
        ss = SpatialSubsetOperation(field_360, wrap=False)
        self.assertFalse(ss._get_should_wrap_(ss.field))
        ss = SpatialSubsetOperation(field_360, wrap=None)
        self.assertFalse(ss._get_should_wrap_(ss.field))

        # A wrapped dataset.
        field_360.wrap()
        ss = SpatialSubsetOperation(field_360, wrap=True)
        self.assertFalse(ss._get_should_wrap_(ss.field))
        ss = SpatialSubsetOperation(field_360, wrap=False)
        self.assertFalse(ss._get_should_wrap_(ss.field))
        ss = SpatialSubsetOperation(field_360, wrap=None)
        self.assertFalse(ss._get_should_wrap_(ss.field))

    @attr('slow')
    def test_get_spatial_subset(self):
        ctr_test = 0
        for ss, k in self:

            # for var in k.target.values()[0].values():
            #     if not isinstance(var, CoordinateReferenceSystem):
            #         log.debug(var._request_dataset.uri)
            #         break

            for geometry_record in self.get_subset_geometries():
                for operation in ['intersects', 'clip', 'foo']:
                    ctr_test += 1
                    use_geometry = deepcopy(geometry_record['geom'])
                    use_ss = deepcopy(ss)
                    try:
                        ret = use_ss.get_spatial_subset(operation, use_geometry, use_spatial_index=True,
                                                        buffer_value=None, buffer_crs=None, geom_crs=WGS84())
                    except ValueError:
                        # 'foo' is not a valid type of subset operation.
                        if operation == 'foo':
                            continue
                        else:
                            raise
                    except EmptySubsetError:
                        try:
                            self.assertEqual(k.target.keys()[0], 'lambert')
                            self.assertEqual(geometry_record['properties']['DESC'], 'Germany')
                        except AssertionError:
                            self.assertEqual(k.target.keys()[0], 'rotated_pole')
                            self.assertEqual(geometry_record['properties']['DESC'], 'Nebraska')
                        continue
                    self.assertIsInstance(ret, OcgField)

        self.assertGreater(ctr_test, 5)

    @attr('data')
    def test_get_spatial_subset_circular_geometries(self):
        """Test circular geometries. They were causing wrapping errors."""

        geoms = TestGeom.get_geometry_dictionaries()
        rd = self.test_data.get_rd('cancm4_tas')
        buffered = [element['geom'].buffer(rd.get().grid.resolution * 2) for element in geoms]
        for buff in buffered:
            ss = SpatialSubsetOperation(rd.get(), wrap=True)
            gvar = GeometryVariable(value=buff, name='geom', dimensions='dim', crs=WGS84())
            ret = ss.get_spatial_subset('intersects', gvar)
            self.assertTrue(np.all(ret.grid.extent > 0))

    @attr('data')
    def test_get_spatial_subset_output_crs(self):
        """Test subsetting with an output CRS."""

        # test with default crs converting to north american lambert
        proj4 = '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs'
        output_crs = CoordinateReferenceSystem(proj4=proj4)
        rd = self.test_data.get_rd('cancm4_tas')
        ss = SpatialSubsetOperation(rd.get(), output_crs=output_crs)
        ret = ss.get_spatial_subset('intersects', self.nebraska['geom'], geom_crs=WGS84())
        self.assertEqual(ret.crs, output_crs)
        self.assertAlmostEqual(ret.grid.value_stacked.mean(), -23341.955070198124)

        # test with an input rotated pole coordinate system
        rd = self.rd_rotated_pole
        ss = SpatialSubsetOperation(rd.get(), output_crs=env.DEFAULT_COORDSYS)
        ret = ss.get_spatial_subset('intersects', self.germany['geom'], geom_crs=WGS84())
        self.assertEqual(ret.crs, env.DEFAULT_COORDSYS)

    @attr('data')
    def test_get_spatial_subset_rotated_pole(self):
        """Test input has rotated pole with now output CRS."""

        rd = self.rd_rotated_pole
        ss = SpatialSubsetOperation(rd.get())
        ret = ss.get_spatial_subset('intersects', self.germany['geom'], geom_crs=WGS84())
        self.assertEqual(ret.crs, rd.get().crs)
        self.assertAlmostEqual(ret.grid.value_stacked.mean(), -2.1699999954751132)

    @attr('data')
    def test_get_spatial_subset_wrap(self):
        """Test subsetting with wrap set to a boolean value."""

        rd = self.test_data.get_rd('cancm4_tas')
        self.assertEqual(rd.get().wrapped_state, WrappedState.UNWRAPPED)
        ss = SpatialSubsetOperation(rd.get(), wrap=True)
        ret = ss.get_spatial_subset('intersects', self.nebraska['geom'], geom_crs=WGS84())
        self.assertEqual(ret.wrapped_state, WrappedState.WRAPPED)
        self.assertAlmostEqual(ret.grid.x.value[1].mean(), -99.84375)

        # test with wrap false
        ss = SpatialSubsetOperation(rd.get(), wrap=False)
        ret = ss.get_spatial_subset('intersects', self.nebraska['geom'], geom_crs=WGS84())
        self.assertEqual(ret.wrapped_state, WrappedState.UNWRAPPED)
        self.assertAlmostEqual(ret.grid.x.value[1].mean(), 260.15625)

    @attr('data')
    def test_prepare_target(self):
        for ss, k in self:
            self.assertIsNone(ss._original_rotated_pole_state)
            if isinstance(ss.field.crs, CFRotatedPole):
                ss._prepare_target_()
                self.assertIsInstance(ss._original_rotated_pole_state, CFRotatedPole)
                self.assertIsInstance(ss.field.crs, CFSpherical)
            else:
                ss._prepare_target_()
                self.assertIsNone(ss._original_rotated_pole_state)

    @attr('data')
    def test_prepare_geometry(self):
        for geometry_record in self.get_subset_geometries():
            for ss, k in self:
                gvar = GeometryVariable(value=geometry_record['geom'], dimensions='dim', crs=WGS84())
                self.assertIsNotNone(gvar.crs)
                prepared = ss._prepare_geometry_(gvar)
                self.assertNotEqual(gvar.value[0].bounds, prepared.value[0].bounds)
                self.assertFalse(np.may_share_memory(gvar.value, prepared.value))
                try:
                    self.assertEqual(prepared.crs, ss.field.crs)
                except AssertionError:
                    # Rotated pole on the fields become spherical.
                    self.assertEqual(prepared.crs, CFSpherical())
                    self.assertIsInstance(ss.field.crs, CFRotatedPole)

        # Test nebraska against an unwrapped dataset.
        nebraska = GeometryVariable(value=self.nebraska['geom'], dimensions='d', crs=WGS84())
        field = self.test_data.get_rd('cancm4_tas').get()
        ss = SpatialSubsetOperation(field)
        prepared = ss._prepare_geometry_(nebraska)
        self.assertEqual(prepared.wrapped_state, WrappedState.UNWRAPPED)

    @attr('data')
    def test_should_update_crs(self):
        # no output crs provided
        target = self.test_data.get_rd('cancm4_tas').get()
        ss = SpatialSubsetOperation(target)
        self.assertFalse(ss.should_update_crs)

        # output crs different than input
        ss = SpatialSubsetOperation(target, output_crs=CoordinateReferenceSystem(epsg=2136))
        self.assertTrue(ss.should_update_crs)

        # same output crs as input
        ss = SpatialSubsetOperation(target, output_crs=ss.field.crs)
        self.assertFalse(ss.should_update_crs)

