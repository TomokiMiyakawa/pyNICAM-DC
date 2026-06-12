import numpy as np

class StateContainer:
    
    _instance = None
    
    def __init__(self):
        pass

    def load(self, name, content):
        setattr(self, name, content)

        # import types
        # print(f"\nLoading '{attr_name}' into state container...")
        # print(f"  -> Type of content: {type(content)}")     
        # # The robust loop using dir() and getattr()
        # for attr_name in dir(content):
        #     # Skip special "dunder" methods and private attributes
        #     if not attr_name.startswith('_'):
        #         # Get the actual attribute value from the content object
        #         attr_value = getattr(content, attr_name)

        #         # Now, filter out the methods (callables)
        #         # We check against a few types to be thorough
        #         if not isinstance(attr_value, (types.MethodType, types.FunctionType)):
        #             print(f"  -> Setting attribute '{attr_name}'...")
        #             setattr(self, attr_name, attr_value)

        # print("\nCopying data attributes from", name, "to state container...")
        # for attr_name, attr_value in dir(content).items():
        #     # Exclude private/special attributes and callables (methods)
        #     if not attr_name.startswith('_') and not callable(attr_value):
        #         print(f"  -> Setting attribute '{attr_name}'...")
        #         setattr(self, attr_name, attr_value)

#        setattr(self, name, content)


#    def get(self, name):
#        return getattr(self, name, None)

#    def load(self, adm, comm, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, bndc, cnvv, bsst, numf, vi, src, srctr, trcadv, pre):
        # setattr(self, "administrative", adm)
        # setattr(self, "communication", comm)
        # setattr(self, "constants", cnst)
        # setattr(self, "grid", grd)
        # setattr(self, "grid_metric", gmtr)
        # setattr(self, "operator", oprt)
        # setattr(self, "vertical_metric", vmtr)
        # setattr(self, "time", tim)
        # setattr(self, "runconfiguration", rcnf)
        # setattr(self, "prognostic_variables", prgv)
        # setattr(self, "tdynamics", tdyn)
        # setattr(self, "boundary_conditions", bndc)
        # setattr(self, "cnvv", cnvv)
        # setattr(self, "base_state", bsst)
        # setattr(self, "numfilter", numf)
        # setattr(self, "vertical_implicit", vi)
        # setattr(self, "source_terms", src)
        # setattr(self, "source_tracer", srctr)
        # setattr(self, "tracer_advection", trcadv)
        # setattr(self, "data_precision", pre)

    # def load_config(self, rcnf, adm):
    #     setattr(self, "runconfiguration", rcnf)
    #     setattr(self, "administrative", adm)

    # # adm rcnf cnst  numf_settings

    # def load_static(self):
    #     pass

    # # grd bsst vmtr_coef numf_coef  vi(? Mc Mu Ml)  (frc)

    # def load_variable(self):
    #     pass

    # tim, prgv (PROG/DIAG)   


    # excluded    prf, prc, std
    #             comm?


        # if std.io_l: 
        #     with open(std.fname_log, 'a') as log_file:
        #         print("+++ Module[grd]/Category[common share]", file=log_file)        
        #         print(f"*** input toml file is ", fname_in, file=log_file)
 
        # with open(fname_in, 'r') as  file:
        #     cnfs = toml.load(file)

        # if 'grdparam' not in cnfs:
        #     with open(std.fname_log, 'a') as log_file:
        #         print("*** grdparam not found in toml file! STOP.", file=log_file)
        #         prc.prc_mpistop(std.io_l, std.fname_log)

        # else:
        #     cnfs = cnfs['grdparam']
        #     self.GRD_grid_type = cnfs['GRD_grid_type']

        # if std.io_nml: 
        #     if std.io_l:
        #         with open(std.fname_log, 'a') as log_file: 
        #             print(cnfs,file=log_file)