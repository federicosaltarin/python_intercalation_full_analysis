# Random forest pixel segmentation for timelapse analysis

These script and Jupyter Notebook allow to compare classical thresholding and Random Forest Classifier for pixel classification and object segmentation in microscopy images.
In particular, source images are obtained from time-lapse acquisition of biological experiments.

The scripts are a Python alternative, all-in-one solution to other pipelines that involve 2 steps analysis in ImageJ Macros and R.
e.g. see https://github.com/federicosaltarin/intercalation_fiji_macro and https://github.com/federicosaltarin/Intercalation_Analysis

In the first part we can load the original images and respective labeled masks for model training. Then the thresholding is also used for comparison.

The random forest classifier is trained based on the given images and then saved for application.

In the last part the trained model is applied to all images through a folder together with the "classical" thresholding aproach.

Finally the results can be visualized.
