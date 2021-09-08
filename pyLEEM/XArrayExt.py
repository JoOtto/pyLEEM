from xarray.backends import BackendEntrypoint
from pyLEEM.LEEMAnalysis import load_NLP
import xarray 
class NLPBackend(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *args,
        drop_variables=None,
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ):
        return load_NLP(filename_or_obj)

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self,filename_or_obj):
        try:
            with open(filename_or_obj, "rb") as f:
                header = f.read(5).decode()
        except TypeError:
            return False
        return header == 'NLP4\n'
    
    