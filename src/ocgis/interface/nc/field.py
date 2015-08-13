import numpy as np

from ocgis.interface.base.field import Field


class NcField(Field):
    @staticmethod
    def _get_formatted_slice_for_netcdf4_(axis_slc, var):
        slc = [axis_slc[k] for k in var.dimensions]
        return slc

    def _get_new_shape_for_output_array_(self, axis_reshape):
        new_shape = [1 if e is None else len(e) for e in [axis_reshape.get(a) for a in self._axes]]
        return new_shape

    def _get_value_from_source_(self, data, variable_name):
        """
        :param data: The netCDF request dataset object.
        :type data: :class:`ocgis.RequestDataset`
        :param variable_name: Name of the target variable to load.
        :type variable_name: str
        :returns: A NumPy masked array object.
        :rtype: :class:`numpy.ma.core.MaskedArray`
        """

        # Collect the dimension slices.
        axis_slc = {}
        # Holds mapping for reshaping when axes are missing.
        axis_reshape = {}
        temporal = self.temporal
        if temporal is not None:
            axis_slc[temporal.name] = temporal._src_idx
            axis_reshape[temporal.axis] = temporal._src_idx
        realization = self.realization
        if realization is not None:
            axis_slc[realization.name] = realization._src_idx
            axis_reshape[realization.axis] = realization._src_idx
        level = self.level
        if level is not None:
            axis_slc[level.name] = level._src_idx
            axis_reshape[level.axis] = level._src_idx

        # Update the spatial slicing.
        self._update_spatial_slice_(axis_reshape, axis_slc)

        ds = data.driver.open()
        try:
            try:
                var = ds.variables[variable_name]
                slc = self._get_formatted_slice_for_netcdf4_(axis_slc, var)
                raw = var.__getitem__(slc)
            # If the slice list are all single element numpy vectors, convert to slice objects to avoid index errors.
            except IndexError:
                if all([len(a) == 1 for a in slc]):
                    slc2 = [slice(a[0], a[0] + 1) for a in slc]
                    raw = ds.variables[variable_name].__getitem__(slc2)
                else:
                    raise
            # Always return a masked array.
            if not isinstance(raw, np.ma.MaskedArray):
                raw = np.ma.array(raw, mask=False)
            # Reshape the data adding singleton axes where necessary.
            new_shape = self._get_new_shape_for_output_array_(axis_reshape)
            raw = raw.reshape(*new_shape)

            return raw
        finally:
            data.driver.close(ds)

    def _update_spatial_slice_(self, axis_reshape, axis_slc):
        grid = self.spatial.grid
        try:
            row = grid.row
            col = grid.col
            axis_slc[row.name] = row._src_idx
            axis_slc[col.name] = col._src_idx
            axis_reshape[row.axis] = row._src_idx
            axis_reshape[col.axis] = col._src_idx
        # If grid and row are not present on the grid object. The source indices are attached to the grid object.
        except AttributeError:
            axis_slc[grid.name_row] = grid._src_idx['row']
            axis_slc[grid.name_col] = grid._src_idx['col']
            axis_reshape['Y'] = grid._src_idx['row']
            axis_reshape['X'] = grid._src_idx['col']


class NcUgridField(NcField):
    @staticmethod
    def _get_formatted_slice_for_netcdf4_(axis_slc, var):
        slc = []
        for k in var.dimensions:
            try:
                slc.append(axis_slc[k])
            except KeyError:
                slc.append(axis_slc['__polygon__'].reshape(-1))
        return slc

    def _get_new_shape_for_output_array_(self, axis_reshape):
        new_shape = [1 if e is None else len(e) for e in [axis_reshape.get(a) for a in self._axes]]
        new_shape[-2:] = axis_reshape['__polygon__'].shape
        return new_shape

    def _update_spatial_slice_(self, axis_reshape, axis_slc):
        src_idx = self.spatial.geom.polygon._src_idx
        axis_slc['__polygon__'] = src_idx
        axis_reshape['__polygon__'] = src_idx

        # tdk: remove commented code
        # def old_get_value_from_source_(self, data, variable_name):
        #     """
        #     :param data: The netCDF request dataset object.
        #     :type data: :class:`ocgis.RequestDataset`
        #     :param variable_name: Name of the target variable to load.
        #     :type variable_name: str
        #     :returns: A NumPy masked array object.
        #     :rtype: :class:`numpy.ma.core.MaskedArray`
        #     """
        #
        #     # Collect the dimension slices.
        #     axis_slc = {}
        #     if self.temporal is not None:
        #         axis_slc['T'] = self.temporal._src_idx
        #     try:
        #         axis_slc['Y'] = self.spatial.grid.row._src_idx
        #         axis_slc['X'] = self.spatial.grid.col._src_idx
        #     # If grid and row are not present on the grid object. The source indices are attached to the grid object.
        #     except AttributeError:
        #         axis_slc['Y'] = self.spatial.grid._src_idx['row']
        #         axis_slc['X'] = self.spatial.grid._src_idx['col']
        #     if self.realization is not None:
        #         axis_slc['R'] = self.realization._src_idx
        #     if self.level is not None:
        #         axis_slc['Z'] = self.level._src_idx
        #
        #     dim_map = data.source_metadata['dim_map']
        #     slc = [None for v in dim_map.values() if v is not None]
        #     axes = deepcopy(slc)
        #     for k, v in dim_map.iteritems():
        #         if v is not None:
        #             slc[v['pos']] = axis_slc[k]
        #             axes[v['pos']] = k
        #     # Ensure axes ordering is as expected.
        #     possible = [['T', 'Y', 'X'], ['T', 'Z', 'Y', 'X'], ['R', 'T', 'Y', 'X'], ['R', 'T', 'Z', 'Y', 'X']]
        #     check = [axes == poss for poss in possible]
        #     assert any(check)
        #
        #     ds = data.driver.open()
        #     try:
        #         try:
        #             raw = ds.variables[variable_name].__getitem__(slc)
        #         # If the slice list are all single element numpy vectors, convert to slice objects to avoid index errors.
        #         except IndexError:
        #             if all([len(a) == 1 for a in slc]):
        #                 slc2 = [slice(a[0], a[0] + 1) for a in slc]
        #                 raw = ds.variables[variable_name].__getitem__(slc2)
        #             else:
        #                 raise
        #         # Always return a masked array.
        #         if not isinstance(raw, np.ma.MaskedArray):
        #             raw = np.ma.array(raw, mask=False)
        #         # Reshape the data adding singleton axes where necessary.
        #         new_shape = [1 if e is None else len(e) for e in [axis_slc.get(a) for a in self._axes]]
        #         raw = raw.reshape(*new_shape)
        #
        #         return raw
        #     finally:
        #         data.driver.close(ds)
