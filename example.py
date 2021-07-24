from LEEMAnalysis import *

#%%
#Test = SpecsNLP(r"C:\Data\LEEM Data\20210714-203124.nlp")
#Test = SpecsNLP(r"C:\Data\LEEM Data\20210715-115719_1.nlp")
Test = SpecsNLP(r"test\20190223_190508_6.8um_349.0_test.nlp")

#%%
import napari
viewer = napari.view_image(Test.ds.intensity)