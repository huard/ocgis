from copy import deepcopy
import itertools
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.interface.base.variable import VariableCollection
from ocgis.interface.base.field import DerivedMultivariateField, DerivedField
from ocgis.calc.base import AbstractMultivariateFunction, AbstractParameterizedFunction
import logging
from ocgis.calc.eval_function import EvalFunction
import numpy as np


class OcgCalculationEngine(object):
    '''
    :type grouping: list of temporal groupings (e.g. ['month','year'])
    :type funcs: list of function dictionaries
    :param raw: If True, perform calculations on raw data values.
    :type raw: bool
    :param agg: If True, data needs to be spatially aggregated (using weights) following a calculation.
    :type agg: bool
    :param bool calc_sample_size:
    :param :class:`ocgis.util.logging_ocgis.ProgressOcgOperations` progress:
    '''
    
    def __init__(self,grouping,funcs,raw=False,agg=False,calc_sample_size=False,
                 progress=None):
        self.raw = raw
        self.agg = agg
        self.grouping = grouping
        self.funcs = funcs
        self.calc_sample_size = calc_sample_size
        
        ## select which value data to pull based on raw and agg arguments
        if self.raw and self.agg is False:
            self.use_raw_values = False
        elif self.raw is False and self.agg is True:
            self.use_raw_values = False
        elif self.raw and self.agg:
            self.use_raw_values = True
        elif not self.raw and not self.agg:
            self.use_raw_values = False
        else:
            raise(NotImplementedError)

        self._tgds = {}
        self._progress = progress
    
    @staticmethod
    def _check_calculation_members_(funcs,klass):
        '''
        Return True if a subclass of type `klass` is contained in the calculation
        list.
        
        :param funcs: Sequence of calculation dictionaries.
        :param klass: `ocgis.calc.base.OcgFunction`
        '''
        check = [issubclass(f['ref'],klass) for f in funcs]
        ret = True if any(check) else False
        return(ret)
        
    def execute(self,coll,file_only=False,tgds=None):
        '''
        :param :class:~`ocgis.SpatialCollection` coll:
        :param bool file_only:
        :param dict tgds: {'field_alias': :class:`ocgis.interface.base.dimension.temporal.TemporalGroupDimension`,...}
        '''

        ## switch field type based on the types of calculations present
        if self._check_calculation_members_(self.funcs,AbstractMultivariateFunction):
            klass = DerivedMultivariateField
        elif self._check_calculation_members_(self.funcs,EvalFunction):
            ## if the input field has more than one variable, assumed this is a 
            ## multivariate calculation
            klass = DerivedField
            for field_container in coll.itervalues():
                for field in field_container.itervalues():
                    if len(field.variables.keys()) > 1:
                        klass = DerivedMultivariateField
                        break
        else:
            klass = DerivedField
        
        ## select which dictionary will hold the temporal group dimensions
        if tgds == None:
            tgds_to_use = self._tgds
            tgds_overloaded = False
        else:
            tgds_to_use = tgds
            tgds_overloaded = True
                                    
        ## group the variables. if grouping is None, calculations are performed
        ## on each element. array computations are taken advantage of.
        if self.grouping is not None:
            ocgis_lh('Setting temporal groups: {0}'.format(self.grouping),'calc.engine')
            for v in coll.itervalues():
                for k2,v2 in v.iteritems():
                    if tgds_overloaded:
                        assert(k2 in tgds_to_use)
                    else:
                        if k2 not in tgds_to_use:
                            tgds_to_use[k2] = v2.temporal.get_grouping(self.grouping)

        ## iterate over functions
        alias_fields_to_remove = {}
        for ugid,dct in coll.iteritems():
            for alias_field,field in dct.iteritems():

                # retrieve the parameters for the function. this only does something in the case where one of the
                # requested parameters is a variable.
                parms = [None]*len(self.funcs)
                process_field = True
                for ii, f in enumerate(self.funcs):
                    parms[ii], alias_field_for_parms = self._get_parms_for_function_(f['ref'], ugid, f, coll)
                    if alias_field_for_parms == alias_field:
                        process_field = False
                        # collect the ugid/fields to remove from the output collection
                        alias_fields_to_remove[ugid] = alias_field_for_parms
                        break
                if not process_field:
                    continue

                ## choose a representative data type based on the first variable
                dtype = field.variables.values()[0].dtype
                
                new_temporal = tgds_to_use.get(alias_field)
                ## if the engine has a grouping, ensure it is equivalent to the
                ## new temporal dimension.
                if self.grouping is not None:
                    try:
                        compare = set(new_temporal.grouping) == set(self.grouping)
                    ## types may be unhashable, compare directly
                    except TypeError:
                        compare = new_temporal.grouping == self.grouping
                    if compare == False:
                        msg = ('Engine temporal grouping and field temporal grouping '
                               'are not equivalent. Perhaps optimizations are incorrect?')
                        ocgis_lh(logger='calc.engine',exc=ValueError(msg))
                
                out_vc = VariableCollection()
                for f, p in itertools.izip(self.funcs, parms):

                    try:
                        ocgis_lh('Calculating: {0}'.format(f['func']),logger='calc.engine')

                        ## initialize the function
                        function = f['ref'](alias=f['name'],dtype=dtype,field=field,file_only=file_only,vc=out_vc,
                             parms=p,tgd=new_temporal,use_raw_values=self.use_raw_values,
                             calc_sample_size=self.calc_sample_size,meta_attrs=f.get('meta_attrs'))
                    except KeyError:
                        ## likely an eval function which does not have the name
                        ## key
                        function = EvalFunction(field=field,file_only=file_only,vc=out_vc,
                                                expr=self.funcs[0]['func'],meta_attrs=self.funcs[0].get('meta_attrs'))
                        
                    ocgis_lh('calculation initialized',logger='calc.engine',level=logging.DEBUG)
                    
                    ## return the variable collection from the calculations
                    out_vc = function.execute()
                    
                    for dv in out_vc.itervalues():
                        ## any outgoing variables from a calculation must have a 
                        ## data type associated with it
                        try:
                            assert(dv.dtype != None)
                        except AssertionError:
                            assert(isinstance(dv.dtype,np.dtype))
                        ## if this is a file only operation, then there should
                        ## be no values.
                        if file_only:
                            assert(dv._value == None)
                    
                    ocgis_lh('calculation finished',logger='calc.engine',level=logging.DEBUG)
                    
                    ## try to mark progress
                    try:
                        self._progress.mark()
                    except AttributeError:
                        pass
                    
                new_temporal = new_temporal or field.temporal
                new_field = klass(variables=out_vc,temporal=new_temporal,spatial=field.spatial,
                                  level=field.level,realization=field.realization,meta=field.meta,
                                  uid=field.uid,name=field.name)
                coll[ugid][alias_field] = new_field

        # if there was nothing computed for the alias, remove it from the output collection
        for k, v in alias_fields_to_remove.iteritems():
            coll[k].pop(v)

        return coll

    @staticmethod
    def _get_parms_for_function_(klass, ugid, dct, coll):
        """
        Return the parameters for an input function. If there are no parameters, return ``None``.

        :param klass: The input function class.
        :type klass: :class:`ocgis.calc.base.AbstractFunction`
        :param int ugid: The corresponding geometry for which this is calculated.
        :param dict dct: The function definition dictionary.
        :param coll: The input collection that a special parameter may need to be extracted from.
        :type coll: :class:`ocgis.api.collection.SpatialCollection`
        :rtype: tuple(dict|None, str|None)
        """

        alias_field = None
        try:
            ret = dct['kwds']
        except KeyError:
            # likely an eval function - return...
            ret = None
        else:
            try:
                variable_parameter_definition_string = AbstractParameterizedFunction._variable_parameter_definition_string
                for k, v in klass.parms_definition.iteritems():
                    if v == variable_parameter_definition_string:
                        alias = dct['kwds'][k]
                        variable, alias_field = AbstractParameterizedFunction.get_variable_from_collection(ugid, coll, alias)
                        ret = deepcopy(ret)
                        ret[k] = variable
            except AttributeError:
                # likely not a parameterized function, so just return the defaults.
                pass
        return ret, alias_field
