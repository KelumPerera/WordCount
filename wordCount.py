# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:02:28 2017

@author: Gishan
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             min_df = 0,          \
                             max_features = 50) 

# Import the data
data_original = pd.read_csv('C:/Users/123/Downloads/File.csv', index_col=0)

data_original.head(5)
data_original.describe()

text1 = data_original['ColumnToAnalyse']  

# Example data
text = ["Hello I am going to I with hello am"]

# Count
train_data_features = vectorizer.fit_transform(text)
vocab = vectorizer.get_feature_names()

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features.toarray(), axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print (count, tag)