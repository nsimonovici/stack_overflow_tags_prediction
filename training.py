#coding:utf-8

import os
import sys
import random
import numpy as np
import pandas as pd

from sklearn import model_selection, metrics, preprocessing, linear_model

from modules import count_tags, data_preprocessing, create_binary_target_df, pick_dump, get_most_probable_tags, dynamic_std_print, get_scores

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
    os.system("python cleaning.py /data/data_questions.csv train")

    # Loading cleant data
    print("Loading clean questions full dataset")
    data_raw = pd.read_csv(local_path + "/data/data_questions_clean.csv", sep=',')

# Data reduction for debugging purpose
# data_raw = data_raw.iloc[:10000]

## 3) Data preparation

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

# Paste train and test sets for processing
data_train_test = pd.concat((data_train, data_test), axis=0)
# Process
print("\nProcessing train and test sets\n")
X_train_test, body_train_vectz, title_train_vectz, indices_features = data_preprocessing(data_train_test)

# Recover train text features
X_train = X_train_test[:data_train.shape[0], :]
# Processing test text features
X_test = X_train_test[data_train.shape[0]:, :]

# Define min volumetry in tags frequency to learn about
min_tag_volumetry = 20
# Retrieve training tags
count_tags_train = count_tags(data_train)
# Select training tags with minimum volumetry
sel_tags_train = count_tags_train[count_tags_train.Count >= min_tag_volumetry].Tag
# Restrain tag selection for application purpose
sel_tags_train = sel_tags_train[:500]
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
    dynamic_std_print("Training classifier for '{}' tag ({:d} / {:d})      ".format(tag, count, len(sel_tags_train)))
    classifier = linear_model.LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train[tag])
    classifiers.append(classifier)
    count += 1

## 6) Save (pickle) training material for later prediction usage

print("\nSaving training material")
# Save training tags
pick_dump(sel_tags_train, "training_material/training_tags")

# Save body vectorizer, title vectorizer and features to keep (above minimum volumetry)
vectorizing_material = [body_train_vectz, title_train_vectz, indices_features]
pick_dump(vectorizing_material, "training_material/vectorizing_material")

print("\nSaving trained classifiers")
# Save classifiers
for clf, tag in zip(classifiers, sel_tags_train):
    pick_dump(clf, "training_material/classifiers/clf_" + tag)

## 7) Predict probability for each classifier on testing set

print("\nPredict probability for each question to belong to each selected tags\n")
test_predictions = []
count = 1
zero_array = np.zeros((X_test.shape[0]))
for clf in classifiers:
    dynamic_std_print("Predicting on classifier {:d} / {:d}".format(count, len(classifiers)))
    prediction = clf.predict_proba(X_test)

    if prediction.shape[1] == 2:
        test_predictions.append(prediction[:,1])
    elif prediction.shape[1] == 1:
        test_predictions.append(zero_array)
    count += 1

# Convert list of arrays to dataframe and keep probabilities over trusted threshold as tags predictions

# Convert to df
test_predictions_df = pd.DataFrame(np.array(test_predictions).T, columns=y_test.columns)
# Set threshold
trusted_threshold = 0.01

# Save prediction trusted threshold
pick_dump(trusted_threshold, "training_material/predictions_trust_threshold")

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
print("\nMetrics for tags predicting algorithm :\n")
get_scores(comparison_df, unique_tags_serie)