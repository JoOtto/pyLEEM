from xarray import *
import pkg_resources
import os

def test_ESCHERnlp_read():
    path = os.path.join("tests", "data", "20190223_185646_5.7um_349.0_test_ESCHER.nlp")
    Test = open_dataset(path, engine='pyLEEM')
    assert Test.intensity.values[-1,-1,-1] == 1163.0

def test_ESCHER_frameMetaData():
    path = os.path.join("tests", "data", "20190223_185646_5.7um_349.0_test_ESCHER.nlp")   
    Test = open_dataset(path, engine='pyLEEM')
    assert Test.GUN_HV.values[0] == '+15000.000000'
    
def test_ESCHER_metaData():
    path = os.path.join("tests", "data", "20190223_185646_5.7um_349.0_test_ESCHER.nlp")    
    Test = open_dataset(path, engine='pyLEEM')
    assert Test.attrs["UPRISM_ST"]== 0.01975
    
def test_XArrayBackendReg():
    assert 'pyLEEM' in [x.name for x in list(pkg_resources.iter_entry_points('xarray.backends'))]
        
def test_XArrayBackend():
    path = os.path.join("tests", "data", "20190223_185646_5.7um_349.0_test_ESCHER.nlp")
    Test = open_dataset(path, engine='pyLEEM')
    assert Test.intensity.values[-1,-1,-1] == 1163.0