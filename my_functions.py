# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:12:33 2023

@author: dcg_1
"""

import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
import demoji
from gensim.parsing.preprocessing import remove_stopwords
import re
from textblob import Word
import nltk
nltk.download('omw-1.4')
import subprocess
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

# The command you want to run
command = "python -m textblob.download_corpora"

try:
    subprocess.run(command, shell=True, check=True)
    print("Corpora downloaded successfully.")
except subprocess.CalledProcessError as e:
    print("Error:", e)

#here I am copying code from my project to process the data to be fed in the model
def data_process(List):
    def unemoji(string_list):
        preprocessed_strings = []
        for text in string_list:
            if demoji.findall(text):
                preprocessed_strings.append(demoji.replace_with_desc(text, sep=" "))
            else:
                preprocessed_strings.append(text)
        return preprocessed_strings

    def stopwords(List):
        for i in range(len(List)):
            List[i] = remove_stopwords(List[i])
        return List

    def clean(List):
        for i in range(len(List)):
            List[i] = re.sub(r'\|\|\|', r' ', List[i]) 
            List[i] = List[i].replace('„','')
            List[i] = List[i].replace('“','')
            List[i] = List[i].replace('"','')
            List[i] = List[i].replace('\'','')
            List[i] = List[i].replace('-','')
            List[i] = List[i].replace("\n", " ")
            List[i] = List[i].replace("\'", "'")
            #I will check if cleaning the text from more standard punctuation benefits the model accuracy
            List[i] = List[i].replace(",", "'")
            List[i] = List[i].replace(".", " ")
            #These two punctuations might be significant in a sentence's meaning
            #List[i] = List[i].replace("!", " ")
            #List[i] = List[i].replace("?", " ")
            List[i] = List[i].lower()
        return List

    def lemmatization(List):
        for i in range(len(List)):
            text = List[i]
            lem = []
            for w in text.split():
                word1 = Word(w).lemmatize("n")
                word2 = Word(word1).lemmatize("v")
                word3 = Word(word2).lemmatize("a")
                lem.append(Word(word3).lemmatize())
            List[i] = text
        return List

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        return tokens

    List = unemoji(List)
    List = stopwords(List)
    List = clean(List)
    List = lemmatization(List)
    
    documents = [TaggedDocument(string, [i]) for i, string in enumerate(List)]
    vector_model = Doc2Vec(vector_size=300, window=15, min_count=5)
    vector_model.build_vocab(documents)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit_transform(List)

    feature_names = tfidf_vectorizer.get_feature_names_out()

    word_tfidf_scores = dict(zip(feature_names, tfidf_vectorizer.idf_))

    word_weights = {word: word_tfidf_scores.get(word, 1.0) for word in vector_model.wv.key_to_index}

    tokenized_texts = [word_tokenize(text) for text in List]
    vectorized_text = []
    for tokens in tokenized_texts:
        text_vector = np.zeros(vector_model.vector_size)  # Initialize a zero vector
        total_weight = 0.0

        for token in tokens:
            if token in vector_model.wv.key_to_index:
                word_vector = vector_model.wv[token]
                weight = word_weights.get(token, 1.0)  # Default weight of 1.0 if word not found in word_weights
                text_vector += weight * word_vector
                total_weight += weight

        # Normalize the text vector by dividing by the total weight
        if total_weight > 0:
            text_vector /= total_weight

        vectorized_text.append(text_vector)

    return vectorized_text

class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 0.1, alpha =1):
        self.threshold = threshold
        self.alpha = alpha
        self.selected_indices = None
        
    def fit(self, X, y=None):
        lasso = Lasso(alpha=self.alpha)  # You can adjust the alpha value if needed
        lasso.fit(X, y)
        
        # Identify features with coefficients above the threshold
        self.selected_indices = np.where(np.abs(lasso.coef_) >= self.threshold)[0]
        
        return self
    
    def transform(self, X):
        return X[:, self.selected_indices]