from pyLEEM.LEEMAnalysis import SpecsNLP
import os

#%%
#Test = SpecsNLP(r"C:\Data\LEEM Data\20210714-203124.nlp")
#Test = SpecsNLP(r"C:\Data\LEEM Data\20210715-115719_1.nlp")
Test = SpecsNLP(os.path.join("./tests", "data", "20190223_185646_5.7um_349.0_test_ESCHER.nlp"))

#%%
import napari
viewer = napari.view_image(Test.ds.intensity)
