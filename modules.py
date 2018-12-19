#coding:utf-8

import os
import time
import math
import re
import pickle
import itertools

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from bs4 import BeautifulSoup
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics, preprocessing

# Functions for cleaning

def missing_values(df):
    """
    Function looking for missing values in a dataframe
    Returning informative dataframe on missing values
    """
    missing_val_df = df.isnull().sum(axis=0).reset_index()
    missing_val_df.columns = ['feature', 'missing values']
    missing_val_df['missing values (%)'] = 100 - ((df.shape[0] - missing_val_df['missing values']) / df.shape[0] * 100)
    missing_val_df = missing_val_df.sort_values('missing values (%)', ascending=False)
    missing_val_df
    print(missing_val_df)
    
    return missing_val_df

def replace_tag_synonym(source_tag_string, synonyms_dict):
    """
    Short function to replace a tag with its synonyms
    """
    source_tags = source_tag_string.split("/")
    for i in range(len(source_tags)):
        if source_tags[i] in synonyms_dict.keys():
            replaced_tag = synonyms_dict[source_tags[i]]
            source_tags[i] = replaced_tag
        else:
            pass
    
    return "/".join(source_tags)

def join_tags_minus_nans(row):
    """
    Function to join list of tags without taking nans
    """
    row_clean = row.dropna()
    joined_row = "/".join(row_clean)
    return joined_row

def count_tags(data):
    """
    Function counting tags in dataframe and outputing informative dataframe on tags
    """
    values = list(itertools.chain.from_iterable(data.loc[:, 'New_Tags_syn'].apply(lambda x: x.split("/")).values))
    count_df = pd.Series(values).value_counts()
    ct_df = pd.DataFrame({'Tag': count_df.index,
                          'Count': count_df.values,
                          'Prcentage (%)': (100 * (count_df / count_df.sum())).values})
    return ct_df

# Functions for predicting

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """
    # Get tag first letter
    tag = nltk.pos_tag([word])[0][1][0].upper()
    # Create dict of relevant tags
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    # Return relevant tag otherwise Noun
    return tag_dict.get(tag, wordnet.NOUN)

def body_to_words(raw_body):
    """
    Function to convert a raw body to a string of words
    The input is a single string (a raw body string), and the output is a single string (a preprocessed body)
    """
    # 1. Remove HTML
    body_text = BeautifulSoup(raw_body, "lxml").get_text() 
    #
    # 2. Remove non-letters non-numbers      
    letters_numbers_only = re.sub("[^a-zA-Z0-9#+]", " ", body_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_numbers_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatize_meaningful_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in meaningful_words]
    #
    # 7. Join the words back into one string separated by space, and return the result.
    return( " ".join( lemmatize_meaningful_words ))   

def text_processing(raw_text):
    """
    Function to convert a raw text to a string of words
    The input is a single string (a raw body string), and the output is a single string (a preprocessed body)
    """
    # 1. Remove HTML
    striped_text = BeautifulSoup(raw_text, "lxml").get_text() 
    #
    # 2. Remove non-letters non-numbers      
    letters_numbers_only = re.sub("[^a-zA-Z0-9#+]", " ", striped_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_numbers_only.lower()
    
    return words

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        string_out = " ".join([feature_names[i] + " ({:.0f}) ".format(topic[i]) for i in topic.argsort()[:-no_top_words - 1:-1]])
        print(string_out)

def get_most_probable_tags(row, selected_tags):
    """
    Take a row of predicted tags probabilities and output the corresponding string of tags
    Take 5 tags at most, 5 with best classifier accuracy score if more than 5 tags are predicted by the classifiers
    """
    # Sort probabilities
    probs = row.sort_values(ascending=False)

    # How many tags have been predicted ?
    n_pred_tags = len(probs[probs > 0])
    # If at least 5, take 5 firsts
    if n_pred_tags >= 5 :
        # Retrieve associated tags
        out_tags = probs[:5].index.values
    elif n_pred_tags == 0:
        # Predict most regular tag
        out_tags = np.array(selected_tags[0])
    else:
        out_tags = probs[:n_pred_tags].index.values

    # Output string
    if n_pred_tags > 1:
        out_string = "/".join(out_tags.tolist())
    elif n_pred_tags == 1:
        out_string = out_tags.tolist()[0]
    else:
        out_string = out_tags.tolist()
    
    return out_string

def local_accuracy(row):
    """
    Input is string of tags separated with '/'
    Input is turned into arrays of lists of tags
    Compute an accuracy score specific to the number of tags of the question and shrink it to a 1-based value
    Output average.
    If the number of predicted tags is greater than the number of true tags, score will be limited to 1.
    """
    # Get lists of tags
    y_true = row.True_Tags.split("/")
    y_pred = row.Predicted_Tags.split("/")
    # Compute difference
    diff = np.setdiff1d(y_true, y_pred)
    len_diff = len(y_true) - len(diff)
    score_entry = len_diff / len(y_true)
    # Return local score
    return score_entry

def pick_dump(obj_name, file_name):
    """ Save with pickle"""
    with open(file_name + '.pickle', 'wb') as file:
        pickle.dump(obj_name, file, protocol=pickle.HIGHEST_PROTOCOL)
    
def pick_load(file_name):
    """ Load with pickle"""
    with open(file_name + '.pickle', 'rb') as file:
        obj = pickle.load(file)
    return obj

def create_binary_target_df(selected_tags, data_set):
    """
    Use selected tags and build a matrix of ones and zeros (but not in sparse format) indicating what tags are related
    to each question 
    """
    tags_feature = ['New_Tags_syn']
    print("Computing binary target dataframe for %i selected tags" % selected_tags.shape[0])
    # Creating numpy array for faster computation
    temp_array = np.zeros((data_set.shape[0], selected_tags.shape[0]))
    # Loop over every popular tag
    for i, tag in zip(range(temp_array.shape[1]), selected_tags.values):
        # Find if tag countained in question
        temp_array[:, i] = np.sum(data_set.loc[:, tags_feature].split("/") == tag, axis=1)

    # Limiting array values to 0 and 1
    temp_array[temp_array > 1] = 1
    # Converting numpy array to pandas dataframe
    target_binary = pd.DataFrame(temp_array, columns=selected_tags, dtype='int64')

    return target_binary

def data_preprocessing(data):
    """
    Function preparing data for unsupervised and supervised learning.
    Input is data as pandas dataframe.
    Output is resulting sparse matrix.
    """
    
    print("Cleaning Body text")
    # Apply treatment to body
    # Create clean body df
    body_words_clean = pd.DataFrame(columns=['body_words'])
    # Apply text processing to clean body df
    body_words_clean['body_words'] = data.Body.apply(text_processing)

    print("Cleaning Title text")
    # Apply treatment to title
    # Create clean title df
    title_words_clean = pd.DataFrame(columns=['title_words'])
    # Apply text processing to clean title df
    title_words_clean['title_words'] = data.Title.apply(text_processing)

    # Initialise Vectorizer to create binary bag of words (1 if the word is present in the document)
    vectorizer_kag = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, binary=True) 

    print("Vectorizing body and title into bag of words")
    # Vectorise our clean body and title words
    body_words_features = vectorizer_kag.fit_transform(body_words_clean.body_words)
    title_words_features = vectorizer_kag.fit_transform(title_words_clean.title_words)

    print("Body vectorized of shape (%i, %i) and Title vectorized of shape (%i, %i)" % (body_words_features.shape[0],
                                                                                        body_words_features.shape[1],
                                                                                        title_words_features.shape[0],
                                                                                        title_words_features.shape[1]))
    
    # Drop features with less than N occurences*
    min_feat_occ = 20
    print("Droping features with less than %i occurences" % min_feat_occ)
    
    # For body
    words_counts_body = body_words_features.sum(axis=0)
    indices = np.where(words_counts_body >= min_feat_occ)[1]
    body_words_features_threshold = body_words_features.tocsc()[:,indices]
    
    # For title
    words_counts_title = title_words_features.sum(axis=0)
    indices = np.where(words_counts_title >= 20)[1]
    title_words_features_threshold = title_words_features.tocsc()[:,indices]

    print("Body vectorized of shape (%i, %i) and Title vectorized of shape (%i, %i)" % (body_words_features_threshold.shape[0],
                                                                                        body_words_features_threshold.shape[1],
                                                                                        title_words_features_threshold.shape[0],
                                                                                        title_words_features_threshold.shape[1]))

    # Combine body and text features
    words_features = scipy.sparse.hstack((body_words_features_threshold, title_words_features_threshold))
    print("Final features matrix size : ", words_features.shape)
    
    return words_features

def get_enough_frequent_tags(row, min_frequency, most_frequent_tag):
    """
    Return the tags most frequently associated to the lda topics, if frequency is above certain threshold.
    If no tags over min frequency threshold : attribute most frequent tag
    """
    # Sort row
    sorted_row = row.sort_values(ascending=False)
    # Test if at least one tag over min frequency threshold
    if sorted_row[0] >= min_frequency:
        acceptable_values = sorted_row[sorted_row >= min_frequency].index.values
        if len(acceptable_values) > 5 :
            output = acceptable_values[:5]
        else:
            output = acceptable_values
    # else return most frequent tag
    else:
        output = [most_frequent_tag]
    
    return output

def get_scores(truth_pred_comparison_df, unique_tags_serie):
    """
    Function that outputs lcoal metric and f1-score
    """
    ### Local hand-made metric ###
    truth_pred_comparison_df['local_metric'] = truth_pred_comparison_df.apply(local_accuracy, axis=1)
    local_metric = truth_pred_comparison_df.local_metric.mean()
    
    print("Average hand-made local accuracy score : %.2f" % local_metric)
    
    ### Calculating f1_score ###

    # Split tags and make tags lists for multilabel binarizer to work with
    y_true_splitted = truth_pred_comparison_df.True_Tags.apply(lambda x: x.split("/"))
    # From dataframe to series
    y_pred_splitted = truth_pred_comparison_df.Predicted_Tags.apply(lambda x: x.split("/"))
    
    # Fit test true tags
    binarizer = preprocessing.MultiLabelBinarizer().fit(unique_tags_serie)

    # Check what have been learnt by the binarizer
    print("sample of binarizer vocabulary", binarizer.classes_)
    
    # Compute f1_score average on samples because of multi-label multi-class classification
    f1_score = metrics.f1_score(binarizer.transform(y_true_splitted),
                                binarizer.transform(y_pred_splitted),
                                average='samples')

    print("F1 Score : %.3f" % f1_score)

