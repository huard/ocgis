from abc import ABCMeta

from shapely import wkb

from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.util.environment import ogr


class AbstractAdapter(AbstractInterfaceObject):
    __metaclass__ = ABCMeta


class SpatialAdapter(AbstractAdapter):
    def __init__(self, crs=None, geom_type='auto'):
        if crs is not None:
            assert isinstance(crs, CoordinateReferenceSystem)

        self._geom_type = geom_type
        self._crs = crs

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

    def get_intersects(self):
        raise NotImplementedError

    def get_intersection(self):
        raise NotImplementedError
