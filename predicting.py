#coding:utf-8

import os
import sys

import scipy
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing

from modules import create_binary_target_df, pick_load, get_most_probable_tags, dynamic_std_print, get_scores, text_processing

## Predict tags on input queries

# Getting current path
local_path = os.getcwd()

## 1) Loadings

## 1.1) Load training material

# Load training tags
training_tags = pick_load("training_material/training_tags")
# Load prediction trusted threshold
pred_threshold = pick_load("training_material/predictions_trust_threshold")
# Load body vectorizer, title vectorizer and features to keep (above minimum volumetry)
body_train_vectz, title_train_vectz, indices_features = pick_load("training_material/vectorizing_material")
# Load training classifiers
training_classifiers = []
for tag in training_tags:
    training_classifiers.append(pick_load("training_material/classifiers/clf_" + tag))

## 1.2) Load input queries
try:
    print("Loading clean input queries dataset")
    input_data = pd.read_csv(local_path + "/data/InputQueries_clean.csv", sep=',', index_col=0)
    print("Input queries clean dataset loaded")
    need_to_clean = False
except FileNotFoundError:
    print("The 'InputQueries_clean.csv' file is not in the 'data' folder, need to clean the dataset")
    need_to_clean = True

## 2) Clean input queries if needed

# If needed run cleaning script
if need_to_clean:
    # Making sure data to predict on are available
    input_file_in_folder = False
    while not input_file_in_folder:
        try:
            pd.read_csv(local_path + "/data/InputQueries.csv", sep=',')
            input_file_in_folder = True
        except FileNotFoundError:
            print("The 'InputQueries.csv' file is not in the 'data' folder, impossible to predict tags")
            print("Please add an input csv file named 'InputQueries.csv' in the 'data' folder")

    os.system("python cleaning.py /data/InputQueries.csv predict")

    # Loading cleant data
    print("Loading clean input queries dataset")
    input_data = pd.read_csv(local_path + "/data/InputQueries_clean.csv", sep=',')

## 3) Process input

print("Processing input text")
# Create new dataframe for processed text inputs
input_data_words_clean = pd.DataFrame(columns=['body_words', 'title_words'])
input_data_words_clean['body_words'] = input_data.Body.apply(text_processing)
input_data_words_clean['title_words'] = input_data.Title.apply(text_processing)
# Vectorize
print("Vectorizing input text with training vectorizers")
input_data_body_features = body_train_vectz.transform(input_data_words_clean.body_words)
input_data_title_features = title_train_vectz.transform(input_data_words_clean.title_words)
# Combine body and text features
input_words_features = scipy.sparse.hstack((input_data_body_features, input_data_title_features))
# Select features of interest
X_input = input_words_features.tocsc()[:, indices_features]

print("\nCreating target matrice for input data\n")
# Compute tags target matrices
y_input = create_binary_target_df(training_tags, input_data)

## 4) Predict probability for each training classifier on input set

print("\nPredict probability for each question of input set to belong to each selected tags\n")
input_predictions = []
count = 1
zero_array = np.zeros((X_input.shape[0]))
for clf in training_classifiers:
    dynamic_std_print("Predicting on classifier {:d} / {:d}".format(count, len(training_classifiers)))
    prediction = clf.predict_proba(X_input)

    if prediction.shape[1] == 2:
        input_predictions.append(prediction[:,1])
    elif prediction.shape[1] == 1:
        input_predictions.append(zero_array)
    count += 1

## 5) Predict input tags

print("\nInput tags prediction\n")
# Convert list of arrays to dataframe and keep probabilities over trusted threshold as tags predictions

# Convert to df
input_predictions_df = pd.DataFrame(np.array(input_predictions).T, columns=y_input.columns)
# Map threshold
input_predictions_df_thresh = input_predictions_df.copy()
input_predictions_df_thresh[input_predictions_df_thresh < pred_threshold] = 0

# Build Predicted tags from classifiers, if more than 5 keep the 5 first with highest probability
y_input_pred = pd.Series(name='Pred_tags')
y_input_pred = input_predictions_df_thresh.apply(get_most_probable_tags, args=(training_tags,), axis=1)

# Save output
output_data = pd.concat((input_data, pd.DataFrame(y_input_pred, columns=['tags_pred'])), axis=1)
output_data.to_csv("data/OutputQueries.csv")