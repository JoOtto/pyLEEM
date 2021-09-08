from xarray.backends import BackendEntrypoint,BackendArray
from xarray.core import indexing
import numpy as np
from pyLEEM.LEEMAnalysis import load_NLP
import xarray 
import os
class NLPBackend(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *args,
        drop_variables=None,
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ):
        ds = load_NLP(filename_or_obj,frame_loading = 'none')
        backend_array = NLPImageArray(ds)
        data = indexing.LazilyIndexedArray(backend_array)
        ds['intensity'] = (['time', 'y', 'x'], data)
        
        return ds

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self,filename_or_obj):
        try:
            with open(filename_or_obj, "rb") as f:
                header = f.read(5).decode()
        except TypeError:
            return False
        return header == 'NLP4\n'
    


class NLPImageArray(BackendArray):
    def __init__(
        self,
        ds, 
        
        # other backend specific keyword arguments
    ):
        self.ds = ds
        self.shape =    (1, 2, 2)# (ds.image_address.shape[0],ds.height.data.max(),ds.width.data.max())
        self.dtype= np.double
         

    def __getitem__(self, key: xarray.core.indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(key,self.shape, indexing.IndexingSupport.BASIC,self._raw_indexing_method)

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        # thread safe method that access to data on disk
            
            print(key)
            return [[[111,112],[121,122]]]
        
        
        
backend = NLPBackend()
ds=backend.open_dataset(r"C:\Git\pyLEEM\tests\data\20190223_185646_5.7um_349.0_test_ESCHER.nlp")
print(ds.intensity.data[:,1,:])
print('#2')

print(ds.intensity.data[0,0,0])