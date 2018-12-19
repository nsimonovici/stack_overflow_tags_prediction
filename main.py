#coding:utf-8

import os
import sys
import time
import math
import itertools
import random
import numpy as np
import pandas as pd
import seaborn as sns

import scipy
import pandas as pd
import seaborn as sns
from sklearn import model_selection, naive_bayes, decomposition, ensemble, multiclass, tree, metrics, preprocessing, svm, linear_model

from modules import missing_values, count_tags, data_preprocessing, create_binary_target_df, get_most_probable_tags, get_scores

## 1) Retrieving data ##

# Load clean dataset if available

# Getting current path
local_path = os.getcwd()
try:
    print("Trying to load clean questions full dataset")
    data_raw = pd.read_csv(local_path + "/data/data_questions_clean.csv", sep=',', index_col=0)
    print("Clean questions full dataset loaded")
    need_to_clean = False
except:
    print("The 'data_questions_clean.csv' file is not in the 'data' folder")
    need_to_clean = True

## 2) Cleaning if not already done

# If needed run cleaning script
if need_to_clean:
    os.system("python cleaning.py /data/data_questions.csv")

    # Loading cleant data
    print("Loading clean questions full dataset")
    data_raw = pd.read_csv(local_path + "/data/data_questions_clean.csv", sep=',')

## 3) Data preparation

# import pdb;pdb.set_trace()

# Last cleaning
data_raw = data_raw.dropna(subset=['New_Tags_syn'])
print("Final shape : ", data_raw.shape)

# Gather unique tags
# After removing synonyms :
temp_list = [x.split('/') for x in data_raw.New_Tags_syn.values.tolist()]
tags_list = [y for x in temp_list for y in x]
unique_tags_syn = list(set(tags_list))
# Remove nan
for value in unique_tags_syn:
    try:
        if np.isnan(value):
            unique_tags_syn.remove(value)
    except:
        pass

# Retrieve tags counts and output most popular tag
count_tags_df = count_tags(data_raw)
most_popular_tag = count_tags_df.Tag[0]

# Gather all tags in a pandas Serie to train binarizers for later computing f1-score
unique_tags_serie = pd.Series(unique_tags_syn).apply(lambda x: [x])

## 4) Train / Test data split and Text processing

print("Splitting dataset into train and test sets")
# Selecting Train et Test sets
data_train, data_test = model_selection.train_test_split(data_raw, test_size=0.3)

print("\nProcessing train set\n")
# Processing train text features
X_train = data_preprocessing(data_train)
print("\nProcessing test set\n")
# Processing test text features
X_test = data_preprocessing(data_test)

# Define min volumetry in tags frequency to learn about
min_tag_volumetry = 20
# Retrieve training tags
count_tags_train = count_tags(data_train)
# Select training tags with minimum volumetry
sel_tags_train = count_tags_train[count_tags_train.Count >= min_tag_volumetry].Tag
print("Given our training set we selected %i Tags with at least %i occurences in the training set" % (sel_tags_train.shape[0], min_tag_volumetry))

print("\nCreating train and test target matrices\n")
# Compute tags target matrices
y_train = create_binary_target_df(sel_tags_train, data_train)
y_test = create_binary_target_df(sel_tags_train, data_test)

# Get target strings
y_test_tags = data_test.New_Tags_syn

## 5) Train classifiers

classifiers = []
count = 1
for tag in sel_tags_train:
    print("Training classifier for '%s' tag (%i / %i)" % (tag, count, len(sel_tags_train)))
    classifier = linear_model.LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train[tag])
    classifiers.append(classifier)
    count += 1

## 6) Predict probability for each classifier on testing set

print("\nPredict probability for each question to belong to each selected tags\n")
test_predictions = []
count = 0
zero_array = np.zeros((X_test.shape[0]))
for clf in classifiers:
    print(count, " / ", len(classifiers))
    prediction = clf.predict_proba(X_test)

    if prediction.shape[1] == 2:
        test_predictions.append(prediction[:,1])
    elif prediction.shape[1] == 1:
        test_predictions.append(zero_array)
    else:
        import pdb;pdb.set_trace()
    count += 1

# Convert list of arrays to dataframe and keep probabilities over trusted threshold as tags predictions

# Convert to df
test_predictions_df = pd.DataFrame(np.array(test_predictions).T, columns=y_test.columns)
# Set threshold
trusted_threshold = 0
# Map threshold
test_predictions_df_thresh = test_predictions_df.copy()
test_predictions_df_thresh[test_predictions_df_thresh < trusted_threshold] = 0

# Build Predicted tags from classifiers, if more than 5 keep the 5 first with highest probability
y_pred = pd.Series()
y_pred = test_predictions_df_thresh.apply(get_most_probable_tags, args=(sel_tags_train,), axis=1)

# Concatenate predicted tags and true tags for comparison
y_pred.index = y_test_tags.index
comparison_df = pd.concat((y_pred, y_test_tags), axis=1)
comparison_df.columns = ['Predicted_Tags', 'True_Tags']

# Get metrics
print("Metrics for tags predicting algorithm :")
get_scores(comparison_df, unique_tags_serie)

## 7) Predict tags on input queries

# Load input queries
try:
    print("Loading input queries dataset")
    data_raw = pd.read_csv(local_path + "/data/InputQueries.csv", sep=',')
    print("Input queries dataset loaded")
    need_to_clean = False
except:
    print("The 'data_questions_clean.csv' file is not in the 'data' folder")
    need_to_clean = True