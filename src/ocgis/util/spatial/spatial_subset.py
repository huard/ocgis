from copy import deepcopy, copy

from ocgis.constants import WrappedState
from ocgis.interface.base.crs import CFRotatedPole, CFWGS84
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.geom import GeometryVariable


class SpatialSubsetOperation(object):
    """
    Perform spatial subsets on various types of ``ocgis`` data objects.

    :param field: The target field to subset.
    :type field: :class:`ocgis.new_interface.field.OcgField`
    :param output_crs: If provided, all output coordinates will be remapped to match.
    :type output_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
    :param wrap: This is only relevant for spherical coordinate systems on ``field`` or when selected as the
     ``output_crs``. If ``None``, leave the wrapping the same as ``field``. If ``True``, wrap the coordinates. If
     ``False``, unwrap the coordinates. A "wrapped" spherical coordinate system has a longitudinal domain from -180 to
     180 degrees.
    :type wrap: bool
    """

    _rotated_pole_destination_crs = CFWGS84

    def __init__(self, field, output_crs='input', wrap=None):
        if not isinstance(field, OcgField):
            raise ValueError('"field" must be an "OcgField" object.')

        self.field = field
        self.output_crs = output_crs
        self.wrap = wrap

        self._original_rotated_pole_state = None

    @property
    def should_update_crs(self):
        """Return ``True`` if output from ``get_spatial_subset`` needs to have its CRS updated."""

        if self.output_crs == 'input':
            ret = False
        elif self.output_crs != self.field.crs:
            ret = True
        else:
            ret = False
        return ret

    def get_spatial_subset(self, operation, geom, use_spatial_index=True, buffer_value=None, buffer_crs=None,
                           geom_crs=None):
        """
        Perform a spatial subset operation on ``target``.

        :param str operation: Either 'intersects' or 'clip'.
        :param geom: The input geometry object to use for subsetting of ``target``.
        :type geom: :class:`shapely.geometry.base.BaseGeometry`
        :param bool use_spatial_index: If ``True``, use an ``rtree`` spatial index.
        :param bool select_nearest: If ``True``, select the geometry nearest ``polygon`` using
         :meth:`shapely.geometry.base.BaseGeometry.distance`.
        :rtype: Same as ``target``. If ``target`` is a :class:`ocgis.RequestDataset`,
         then a :class:`ocgis.interface.base.field.Field` will be returned.
        :param float buffer_value: The buffer radius to use in units of the coordinate system of ``subset_sdim``.
        :param buffer_crs: If provided, then ``buffer_value`` are not in units of the coordinate system of
         ``subset_sdim`` but in units of ``buffer_crs``.
        :type buffer_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :raises: ValueError
        """

        if not isinstance(geom, GeometryVariable):
            geom = GeometryVariable(value=geom, name='geom', dimensions='one', crs=geom_crs)
        if geom.value.flatten().shape != (1,):
            msg = 'Only one subset geometry allowed. The shape of the geometry variable is {}.'.format(geom.shape)
            raise ValueError(msg)

        # Buffer the subset if a buffer value is provided.
        if buffer_value is not None:
            geom = self._get_buffered_geometry_(geom, buffer_value, buffer_crs=buffer_crs)

        if operation == 'clip':
            method = self.field.geom.get_intersection
        elif operation == 'intersects':
            method = self.field.geom.get_intersects
        else:
            msg = 'The spatial operation "{0}" is not supported.'.format(operation)
            raise ValueError(msg)

        self._prepare_target_()

        prepared = self._prepare_geometry_(geom)
        base_geometry = prepared.value.flatten()[0]

        # execute the spatial operation
        ret = method(base_geometry, use_spatial_index=use_spatial_index, cascade=True).parent

        # check for rotated pole and convert back to default CRS
        if self._original_rotated_pole_state is not None and self.output_crs == 'input':
            try:
                ret.spatial.update_crs(self._original_rotated_pole_state)
            except AttributeError:
                # like a spatial dimension
                ret.update_crs(self._original_rotated_pole_state)

        # wrap the data...
        if self._get_should_wrap_(ret):
            ret.wrap()

        # convert the coordinate system if requested...
        if self.should_update_crs:
            ret.update_crs(self.output_crs)

        return ret

    @staticmethod
    def _get_buffered_geometry_(geom, buffer_value, buffer_crs=None):
        """
        Buffer a spatial dimension. If ``buffer_crs`` is provided, then ``buffer_value`` are in units of ``buffer_crs``
        and the coordinate system of ``geom`` may need to be updated.

        :param subset_sdim: The spatial dimension object to buffer.
        :type subset_sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        :param float buffer_value: The buffer radius to use in units of the coordinate system of ``subset_sdim``.
        :param buffer_crs: If provided, then ``buffer_value`` are not in units of the coordinate system of
         ``subset_sdim`` but in units of ``buffer_crs``.
        :type buffer_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :rtype: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        to_buffer = deepcopy(geom)
        if buffer_crs is not None:
            to_buffer.update_crs(buffer_crs)
        to_buffer.value[0] = to_buffer.value[0].buffer(buffer_value, cap_style=3)
        if buffer_crs is not None:
            to_buffer.update_crs(geom.crs)
        return to_buffer

    def _get_should_wrap_(self, field):
        """
        Return ``True`` if the output from ``get_spatial_subset`` should be wrapped.

        :param field: The target field to test for wrapped stated.
        :type fiedl: :class:`ocgis.new_interface.field.OcgField`
        """

        # The output needs to be wrapped and the input data is unwrapped. Output from get_spatial_subset is always
        # wrapped relative to the input target.
        if self.wrap and field.wrapped_state == WrappedState.UNWRAPPED:
            ret = True
        else:
            ret = False

        return ret

    def _prepare_geometry_(self, geom):
        """
        Compare ``subset_sdim`` geographic state with ``target`` and perform any necessary transformations to ensure a
        smooth subset operation.

        :param subset_sdim: The input object to use for subsetting of ``target``.
        :type subset_sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        :rtype: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        prepared = deepcopy(geom)
        prepared.update_crs(self.field.crs)
        field_wrapped_state = self.field.wrapped_state
        prepared_wrapped_state = prepared.wrapped_state
        if field_wrapped_state == WrappedState.UNWRAPPED:
            if prepared_wrapped_state == WrappedState.WRAPPED:
                prepared.unwrap()
        elif field_wrapped_state == WrappedState.WRAPPED:
            if prepared_wrapped_state == WrappedState.UNWRAPPED:
                prepared.wrap()

        return prepared

    def _prepare_target_(self):
        """
        Perform any transformations on ``target`` in preparation for spatial subsetting.
        """

        if isinstance(self.field.crs, CFRotatedPole):
            self._original_rotated_pole_state = copy(self.field.crs)
            self.field.update_crs(self._rotated_pole_destination_crs())
