from pyLEEM.LEEMAnalysis import SpecsNLP
import os

def test_ESCHERnlp_read():
    path = os.path.join("./tests", "data", "20190223_185646_5.7um_349.0_test_ESCHER.nlp")
    Test = SpecsNLP(path)
    assert Test.ds.intensity.shape == (1, 1024, 1280)
