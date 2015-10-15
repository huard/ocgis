import fiona
from shapely.geometry import Polygon, MultiPolygon, shape, mapping
from shapely.geometry.base import BaseMultipartGeometry
import numpy as np

from ocgis.util.helpers import iter_array


def convert_multipart_to_singlepart(path_in, path_out, new_uid_name='MID', start=0):
    """
    Convert a vector GIS file from multipart to singlepart geometries. The function copies all attributes and
    maintains the coordinate system.

    :param str path_in: Path to the input file containing multipart geometries.
    :param str path_out: Path to the output file.
    :param str new_uid_name: Use this name as the default for the new unique identifier.
    :param int start: Start value for the new unique identifier.
    """

    with fiona.open(path_in) as source:
        source.meta['schema']['properties'][new_uid_name] = 'int'
        with fiona.open(path_out, mode='w', **source.meta) as sink:
            for record in source:
                geom = shape(record['geometry'])
                if isinstance(geom, BaseMultipartGeometry):
                    for element in geom:
                        record['properties'][new_uid_name] = start
                        record['geometry'] = mapping(element)
                        sink.write(record)
                        start += 1
                else:
                    record['properties'][new_uid_name] = start
                    sink.write(record)
                    start += 1


def get_coordinate_array_from_polygon(polygon, node_count, dtype=None):
    polygon = get_valid(polygon)
    ret = np.array(polygon.exterior.coords, dtype=dtype)
    fill = np.ma.array(np.zeros((2, node_count), dtype=dtype), mask=True)
    # Unused array slots are left masked.
    fill[0, 0:ret.shape[0]] = ret[:, 1]
    fill[1, 0:ret.shape[0]] = ret[:, 0]
    return fill


def get_mesh_array_from_polygons(polygons, dtype=None):
    node_count = 0
    # First loop is to determine max node count.
    for ctr, (_, polygon) in enumerate(iter_polygons_and_indices(polygons)):
        polygon_node_count = len(polygon.exterior.coords)
        if polygon_node_count > node_count:
            node_count = polygon_node_count
    # Second loop fills array with node coordinates.
    fill = np.ma.array(np.zeros((ctr + 1, 2, node_count)), dtype=dtype)
    for idx, (_, polygon) in enumerate(iter_polygons_and_indices(polygons)):
        res = get_coordinate_array_from_polygon(polygon, node_count, dtype=dtype)
        fill[idx] = res
    return fill


def get_polygon_array_from_mesh_array(mesh_arr):
    fill = np.zeros((1, mesh_arr.shape[0]), dtype=object)
    for idx in range(mesh_arr.shape[0]):
        x = mesh_arr[idx][1, :].compressed().reshape(-1, 1)
        y = mesh_arr[idx][0, :].compressed().reshape(-1, 1)
        xy = np.hstack((x, y))
        polygon = Polygon(xy)
        fill[0, idx] = polygon
    return np.ma.array(fill, mask=False)


def get_valid(geom):
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def iter_polygons_and_indices(arr):
    for idx, geom in iter_array(arr, return_value=True):
        if isinstance(geom, MultiPolygon):
            for yld in geom:
                yield idx, yld
        else:
            yield idx, geom
