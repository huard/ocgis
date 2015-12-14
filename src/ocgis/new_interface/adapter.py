from abc import ABCMeta

import fiona
from shapely import wkb
from shapely.geometry import mapping

from ocgis import constants
from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.util.environment import ogr
from ocgis.util.helpers import iter_array


class AbstractAdapter(AbstractInterfaceObject):
    __metaclass__ = ABCMeta


class SpatialAdapter(AbstractAdapter):
    def __init__(self, crs=None, geom_type='auto', spatial_index=None):
        if crs is not None:
            assert isinstance(crs, CoordinateReferenceSystem)

        self._geom_type = geom_type
        self._crs = crs
        self._spatial_index = spatial_index

    @property
    def crs(self):
        return self._crs

    @property
    def geom_type(self):
        # Geometry objects may change part counts during operations. It is better to scan and update the geometry types
        # to account for these operations.
        if self._geom_type == 'auto':
            for geom in self.value.data.flat:
                if geom.geom_type.startswith('Multi'):
                    break
            self._geom_type = geom.geom_type
        return self._geom_type

    def get_spatial_index(self):
        # "rtree" is an optional dependency.
        from ocgis.util.spatial.index import SpatialIndex
        # Fill the spatial index with unmasked values only.
        si = SpatialIndex()
        r_add = si.add
        # Add the geometries to the index.
        # tdk: test with the value having a mask already
        for idx, geom in iter_array(self.value.reshape(-1), return_value=True, use_mask=True):
            r_add(idx[0], geom)

        return si

    def update_crs(self, to_crs):
        # Be sure and project masked geometries to maintain underlying geometries.
        r_value = self.value.data
        r_loads = wkb.loads
        r_create = ogr.CreateGeometryFromWkb
        to_sr = to_crs.sr
        from_sr = self.crs.sr
        for idx, geom in enumerate(r_value.flat):
            ogr_geom = r_create(geom.wkb)
            ogr_geom.AssignSpatialReference(from_sr)
            ogr_geom.TransformTo(to_sr)
            r_value[idx] = r_loads(ogr_geom.ExportToWkb())
        self._crs = to_crs

    def write_fiona(self, path, driver='ESRI Shapefile'):
        name_uid = constants.HEADERS.ID_GEOMETRY.upper()
        schema = {'geometry': self.geom_type,
                  'properties': {name_uid: 'int'}}
        ref_prep = self._write_fiona_prep_geom_
        with fiona.open(path, 'w', driver=driver, crs=self.crs.value, schema=schema) as f:
            for uid, geom in enumerate(self.value.compressed()):
                geom = ref_prep(geom)
                feature = {'properties': {name_uid: uid}, 'geometry': mapping(geom)}
                f.write(feature)

        return path

    @staticmethod
    def _write_fiona_prep_geom_(geom):
        return geom
