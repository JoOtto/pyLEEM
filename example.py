from pyLEEM.LEEMAnalysis import load_NLP
import os

#%%
Test = load_NLP(os.path.join("tests", "data", "20190223_185646_5.7um_349.0_test_ESCHER.nlp"))

#%%
import napari
viewer = napari.view_image(Test.intensity)
