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
<<<<<<< HEAD
    print("Loading clean input queries dataset")
    input_data = pd.read_csv(local_path + "/data/InputQueries_clean.csv", sep=',')
    print("Input queries clean dataset loaded")
    need_to_clean = False
except FileNotFoundError:
    print("The 'InputQueries_clean.csv' file is not in the 'data' folder, need to clean the dataset")
    need_to_clean = True

## 7.2) Clean input queries if needed

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

    os.system("python cleaning.py /data/InputQueries.csv")

    # Loading cleant data
    print("Loading clean input queries dataset")
    input_data = pd.read_csv(local_path + "/data/InputQueries_clean.csv", sep=',')

## 7.3) Process input

X_input = data_preprocessing(input_data)

print("\nCreating target matrice for input data\n")
# Compute tags target matrices
y_input = create_binary_target_df(sel_tags_train, input_data)

# Get target strings
y_input_tags = input_data.New_Tags_syn

<<<<<<< HEAD
# Gather unique tags of input
# After removing synonyms :
temp_list = [x.split('/') for x in input_data.New_Tags_syn.values.tolist()]
tags_list = [y for x in temp_list for y in x]
input_unique_tags_syn = list(set(tags_list))
# Remove nan
for value in input_unique_tags_syn:
    try:
        if np.isnan(value):
            input_unique_tags_syn.remove(value)
    except:
        pass

# Concatenate training unique tags and input tags for prediction
complete_unique_tags_syn = list(set(sel_tags_train.tolist() + input_unique_tags_syn))
# Gather all tags in a pandas Serie to train binarizers for later computing f1-score
complete_unique_tags_serie = pd.Series(complete_unique_tags_syn).apply(lambda x: [x])
print("\n%i tags in input questions have not been learn in training\n" % (len(np.setdiff1d(complete_unique_tags_syn, sel_tags_train))))

## 7.4) Predict probability for each training classifier on input set
=======
## 7.4) Predict probability for each classifier on input set
>>>>>>> parent of dcf8f2b... End of main programming

print("\nPredict probability for each question of input set to belong to each selected tags\n")
input_predictions = []
count = 0
zero_array = np.zeros((X_input.shape[0]))
for clf in classifiers:
    print(count, " / ", len(classifiers))
    prediction = clf.predict_proba(X_input)

    if prediction.shape[1] == 2:
        input_predictions.append(prediction[:,1])
    elif prediction.shape[1] == 1:
        input_predictions.append(zero_array)
    else:
        import pdb;pdb.set_trace()
    count += 1

## 7.5) Predict input tags

# Convert list of arrays to dataframe and keep probabilities over trusted threshold as tags predictions

# Convert to df
input_predictions_df = pd.DataFrame(np.array(input_predictions).T, columns=y_input.columns)
# Map threshold
input_predictions_df_thresh = input_predictions_df.copy()
input_predictions_df_thresh[input_predictions_df_thresh < trusted_threshold] = 0

# Build Predicted tags from classifiers, if more than 5 keep the 5 first with highest probability
y_input_pred = pd.Series(name='Pred_tags')
y_input_pred = input_predictions_df_thresh.apply(get_most_probable_tags, args=(sel_tags_train,), axis=1)

# Concatenate predicted tags and true tags for comparison
y_input_pred.index = y_input_tags.index
input_comparison_df = pd.concat((y_input_pred, y_input_tags), axis=1)
input_comparison_df.columns = ['Predicted_Tags', 'True_Tags']

# Get metrics
print("Metrics for tags predicting algorithm :")
get_scores(input_comparison_df, unique_tags_serie)

# Save output
<<<<<<< HEAD
output_data = pd.concat((input_data, pd.DataFrame(y_input_pred, columns=['tags_pred'])), axis=1)
output_data.to_csv("data/OutputQueries.csv")
=======
    print("Loading input queries dataset")
    data_raw = pd.read_csv(local_path + "/data/InputQueries.csv", sep=',')
    print("Input queries dataset loaded")
    need_to_clean = False
except:
    print("The 'data_questions_clean.csv' file is not in the 'data' folder")
    need_to_clean = True
>>>>>>> parent of 99bf8b9... End coding input prediction before debugging
=======
output_data = pd.concat((input_data, y_input_pred))
>>>>>>> parent of dcf8f2b... End of main programming
