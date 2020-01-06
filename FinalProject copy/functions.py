import os
import logging
import pandas
from datetime import datetime 
# from google.cloud import bigquery, storage
# from google_pandas_load import Loader, LoaderQuickSetup
# from google_pandas_load import LoadConfig

import nltk
from nltk.corpus import stopwords
import string
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
import requests
import json
from time import sleep, time
import lxml
import re
from urllib.parse import urljoin
import unicodedata
from collections import defaultdict, Counter
# import WebScrape_Indeed
# import WebScrape_LinkedIn
import streamlit as st 
import terms 
import Cities 
import functions
import time

search_terms = terms.total_terms

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    clean = re.sub(cleanr, ' ', str(raw_html))
    cleaner = clean.strip()
    cleantext = re.sub('\n', ' ', cleaner)
    return cleantext

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)    
def clean(raw):
    raw = ' '.join(raw.tolist())
    for char in '-.,\n':
        raw = raw.replace(char,' ')
    
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    raw = nltk.regexp_tokenize(raw, pattern)
    return raw

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def cleanC(raw):
    raw = ' '.join(raw.tolist())
    for char in '-.,\n':
        raw = raw.replace(char,' ')
    return raw

def cleanC2(raw):
    # raw = ' '.join(raw.tolist())
    for char in '-.,\n':
        raw = raw.replace(char,' ')
    return raw

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def C_plus(raw):
    Cplus = re.findall(r'(?i)\bC\+\+(?!\w)', str(raw))
    return Cplus

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def C_sharp(raw):
    Csharp = re.findall(r'(?i)\bC\#(?!\w)', str(raw))
    return Csharp

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def tokenize(raw):
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    raw = nltk.regexp_tokenize(raw, pattern)
    return raw 

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)    
def word_count(string):
    freq = defaultdict(int)
    List1 = [x.lower() for x in search_terms]
    List2 = [x.lower() for x in string]

    for item in List1:
        if item in List2:
            freq[item] = List2.count(item)
        else:
            freq[item] = 0 
    return freq





# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def compare(word, result):
    test = []
    List1 = [x.lower() for x in word]
    List2 = [x.lower() for x in result.index]
    for x in List1:
        if x in result.index: 
            test.append(result.loc[x])
    return test