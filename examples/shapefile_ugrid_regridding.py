"""
This script uses ESMPy and OCGIS to regrid a structured precipitation dataset to unstructured catchment areas.

contact: esmf_support@list.woc.noaa.gov
"""

import os
import tempfile

import ESMF
import numpy as np

import ocgis


# path to a small catchments shapefile
PATH_SHP = 'catchment_San_Guad_3reaches/catchment_San_Guad_3reaches.shp'
# path to an example precipitation dataset
PATH_PR = 'nldas_met_update.obs.daily.pr.1990.nc'

# this example writes all newly created files to a temporary directory
DIR_TMP = tempfile.mkdtemp()
# path the output UGRID NetCDF file
PATH_OUT_NC = os.path.join(DIR_TMP, 'ugrid_catchments.nc')
# tell OCGIS to write to the temporary directory
ocgis.env.DIR_OUTPUT = DIR_TMP


def get_ugridnc_and_subsetnc():
    """
    :returns: Tuple with indices corresponding to:
     1. Path to UGRID NetCDF file.
     2. Path to subsetted NetCDF file.
     3. An ESMPy field object created from the subsetted NetCDF file.
    :rtype: (str, str, :class:`ESMF.api.field.Field`)
    """

    ugridnc = ocgis.OcgOperations(dataset={'uri': PATH_SHP}, output_format='nc-ugrid-2d-flexible-mesh',
                                  add_auxiliary_files=False).execute()

    # select_ugid = [xx['properties']['UGID'] for xx in ShpCabinetIterator(path=path_ugid_shp)]
    rd = ocgis.RequestDataset(uri=PATH_PR)
    ops = ocgis.OcgOperations(dataset=rd, geom=PATH_SHP, agg_selection=True, prefix='subset_nc', output_format='nc',
                              add_auxiliary_files=False)
    subset_nc = ops.execute()

    # convert the subsetted NetCDF file to and ESMPy field object
    ops = ocgis.OcgOperations(dataset={'uri': subset_nc}, output_format='esmpy')
    srcfield = ops.execute()

    return ugridnc, subset_nc, srcfield

# create a manager object with multiprocessor logging in debug mode
# ESMF.Manager(logkind=ESMF.LogKind.MULTI, debug=True)

# use OCGIS to subset a 1990 Maurer downscaled precipitation data file and return source field values for ESMPy. also
# create a ugrid formatted file from a shapefile
ugridnc, subset_nc, srcfield = get_ugridnc_and_subsetnc()

# create an ESMPy Mesh and destination Field from UGRID file
dstgrid = ESMF.Mesh(filename=ugridnc, filetype=ESMF.FileFormat.UGRID, meshname="Mesh2")
dstfield = ESMF.Field(dstgrid, "dstfield", meshloc=ESMF.MeshLoc.ELEMENT, ndbounds=[365])

# create an object to regrid data from the source to the destination field
regrid = ESMF.Regrid(srcfield, dstfield, regrid_method=ESMF.RegridMethod.CONSERVE,
                     unmapped_action=ESMF.UnmappedAction.IGNORE)

# do the regridding from source to destination field
dstfield = regrid(srcfield, dstfield)

# turn the shapefile into an OCGIS field
ofield = ocgis.RequestDataset(PATH_SHP).get()
# get the time dimension from the original netCDF file
otime = ocgis.RequestDataset(PATH_PR).get().temporal
# create an OCGIS variable from the regridded data values
variable = ocgis.Variable(name='pr', value=np.array(dstfield.reshape(1, otime.shape[0], 1, 1, ofield.shape[-1])))
# combine the spatial data with time and the regridded values
ofield2 = ocgis.Field(temporal=otime, spatial=ofield.spatial, variables=variable)
# write this to shapefile
path_out_shp = ocgis.OcgOperations(dataset=ofield2, output_format='shp', prefix='pr_catchments',
                                   add_auxiliary_files=False).execute()
