#coding:utf-8

import os
import sys
import numpy as np
import pandas as pd

from modules import replace_tag_synonym, join_tags_minus_nans

## 1) Retrieving data ##

# Retrieve argument
dataset_name = sys.argv[1]

# Getting current path
local_path = os.getcwd()

if dataset_name == 'data_questions.csv':
    # List files of interest
    data_files = [file for file in os.listdir(local_path + '/data') if file.startswith('QueryResults')]

    # If main dataset does not exist, Loop on files and construct main dataset
    try:
        data_raw = pd.DataFrame()
        print("Loading questions full dataset")
        data_raw = pd.read_csv(local_path + "/data/data_questions.csv", sep=',')
    except FileNotFoundError:
        # Loading one file to get columns names
        file = data_files[0]
        try:
            data_col_names = pd.read_csv(local_path + "/data/" + file, sep=',')
        except FileNotFoundError:
            print("Please check if the file %s is in the 'data' folder at the current location" % file)
            sys.exit(1)
        data_columns = data_col_names.columns

        # Initialise main df
        data_raw = pd.DataFrame(columns=data_columns)
        # Loop over separate files to build main dataframe
        for file in data_files:
            print("Treating file : %s" % file)
            # Verifying data presence
            try:
                data_temp = pd.read_csv(local_path + "/data/" + file, sep=',')
            except FileNotFoundError:
                print("Please check if the file %s is in the 'data' folder at the current location" % file)
                sys.exit(1)

            # Save data
            data_raw = data_raw.append(data_temp)

        # Save data
        print("Saving")
        data_raw.to_csv("data/data_questions.csv", index=False)
else:
    # Load dataset
    try:
        data_raw = pd.DataFrame()
        print("Loading '%s' dataset" % dataset_name)
        data_raw = pd.read_csv(local_path + dataset_name, sep=',')
    except FileNotFoundError:
        print("Please check if the file '%s' is in the 'data' folder at the current location" % dataset_name)
        sys.exit(1)

# Only keep relevant features
data_raw = data_raw.loc[:, ['Body', 'Title', 'Tags']]

# Load Synonyms dataset
try:
    print("Loading Tags Synonyms dataset")
    tags_synonyms = pd.read_csv(local_path + "/data/Tags_Synonyms.csv", sep=',')
except FileNotFoundError:
    print("Please check if the file 'Tags_Synonyms.csv' is in the 'data' folder at the current location")
    sys.exit(1)

print("Begin cleaning dataset")
# Getting rid of the duplicates
print("initial shape : ", data_raw.shape)
dup = data_raw[data_raw.duplicated()].shape[0]
if dup > 0:
    print("duplicates found : ", dup)
    data_raw = data_raw.drop_duplicates(keep='first')
    print("Shape without duplicates: ", data_raw.shape)
else:
    print("No duplicate")

# Define tags features
tags_features = ['Tag_1', 'Tag_2', 'Tag_3', 'Tag_4', 'Tag_5']

# Drop rows without tags
data_raw = data_raw.dropna(subset=['Tags'])
print("New shape without missing tags : ", data_raw.shape)

# We will use Body and Title to train our models, delete rows with missing values
data_raw = data_raw.dropna(subset=['Body', 'Title'])
print("New shape without missing body or title : ", data_raw.shape)

## 3) Feature engineering

# Body and Title may both contains interesting clues, we will concatenate those into one new string
data_raw['TitleBody'] = data_raw.Title + data_raw.Body

# Removing Tags chevrons
data_raw['New_Tags'] = data_raw.Tags.apply(lambda x: x.strip('<').strip('>').replace('>', '').replace('<', '/'))

# Counting Tags
data_raw['n_Tags'] = data_raw.New_Tags.apply(lambda x: len(x.split('/')))

# # Separating tags in indiviuals features (Tag 1 to 5)
# tags_lists = data_raw.New_Tags.apply(lambda x: x.split('/')).values
# # Initialise new list of tags
# filled_tags_list = []
# # Loop over lists of tags
# for inner_list in tags_lists:
#     # Get list length
#     length = len(inner_list)
#     # While length not equal to 5 append nans
#     while length < 5:
#         inner_list.append(np.nan)
#         length = len(inner_list)
#     # Add extended list to new list
#     filled_tags_list.append(inner_list)

# # Convert lists of tags into dataframe
# tags_df = pd.DataFrame(filled_tags_list)
# # Remove empty label if needed
# try:
#     tags_df = tags_df.drop(labels=5, axis=1)
# except:
#     pass
# tags_df.index = data_raw.index
# tags_df.columns = tags_features

# # Add separated tags to main dataframe
# data_raw = pd.concat((data_raw, tags_df), axis=1)

#Looking for tags that can be replaced with synonyms

# Before removing synonyms :
temp_list = [x.split('/') for x in data_raw.New_Tags.values.tolist()]
tags_list = [y for x in temp_list for y in x]
unique_tags = set(tags_list)
print("Total of %i unique Tags" % len(unique_tags))

tags_syns_in_tags = []
for sourcetag in tags_synonyms.SourceTagName:
    if sourcetag in unique_tags :
        tags_syns_in_tags.append(sourcetag)
print("%i Tags are synonyms and can be replaced" % len(tags_syns_in_tags))

# Get synonyms dictionnary
synonyms_dict = dict(zip(tags_synonyms.SourceTagName.values, tags_synonyms.TargetTagName.values))

# Replace tags that can be replaced
print("Replacing synonyms")
data_raw['New_Tags_syn'] = data_raw.loc[:, 'New_Tags'].apply(replace_tag_synonym, args=(synonyms_dict,))

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
print("Total of %i unique Tags" % len(unique_tags_syn))

print("Saving clean dataset")
# Saving processed data
data_raw.to_csv("data/data_questions_clean.csv")