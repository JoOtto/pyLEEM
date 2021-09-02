from pyLEEM.LEEMAnalysis import load_NLP
import os

def test_ESCHERnlp_read():
    path = os.path.join("tests", "data", "20190223_185646_5.7um_349.0_test_ESCHER.nlp")
    Test = load_NLP(path)
    assert Test.intensity.values[-1,-1,-1] == 1163.0

def test_ESCHER_frameMetaData():
    path = os.path.join("tests", "data", "20190223_185646_5.7um_349.0_test_ESCHER.nlp")   
    Test = load_NLP(path)
    assert Test.GUN_HV.values[0] == '+15000.000000'
    
def test_ESCHER_metaData():
    path = os.path.join("tests", "data", "20190223_185646_5.7um_349.0_test_ESCHER.nlp")    
    Test = load_NLP(path)
    assert Test.attrs["UPRISM_ST"]== 0.01975