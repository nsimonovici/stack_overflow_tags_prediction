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

from modules import missing_values, count_tags

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
    os.system("python cleaning.py")

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