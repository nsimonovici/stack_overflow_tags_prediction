# stack_overflow_tags_prediction

Project part of OpenClassrooms Data Scientist path

Our aim is to develop a tag predictor algorithm for questions asked on StackOverflow.

Steps are as follow :

1) Merge datafiles of data extraction from stackexchange
2) Clean merge dataset
3) Process text from title and body of questions
4) Train supervised classifiers for main tags
5) Predict tags of input based on classifiers predictions with highest confidence

Files of interest :

- modules.py : Gather the functions for the programs and more
- main.py : Train + Predict on dataframe
- training.py : Train classifiers with hard-coded options
- predicting.py : Predict on input dataframe from already saved training material
- cleaning_exploring.ipynb : Data cleaning and exploring notebook
- tags_predicting.ipynb : Text processing and data modelisation notebook
