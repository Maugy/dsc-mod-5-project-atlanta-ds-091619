# TriGrams

import requests
import time

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import re
import glob
import random
import seaborn as sns
import string

from IPython.display import clear_output

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# http://www.nltk.org/howto/wordnet.html

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.wsd import lesk

import os
import logging
import pandas
from datetime import datetime 

import nltk
from nltk.corpus import stopwords
import string
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go
import plotly.figure_factory as ff


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
import WebScrape_Indeed
import WebScrape_LinkedIn
import streamlit as st 
import terms 
import Cities 
import functions
import time
from google.cloud import bigquery, storage
from google_pandas_load import Loader, LoaderQuickSetup
from google_pandas_load import LoadConfig

import chart_studio.plotly 
import plotly.graph_objects as go
import plotly
import pandas as pd

# import pydeck as pdk
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)   

import string
import collections
 
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)
from sklearn import metrics
from sklearn.decomposition import PCA

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # for plot styling
import numpy as np

from ipywidgets import interact
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing import text, sequence

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

data = pd.read_csv('total_data_date.csv')

dtitle = data[data['title'].astype(str).str.contains(option)]
search_terms = terms.total_terms
text = dtitle.description
text_clean = functions.clean(text)

def tech_count(text):
    tech_skills = []
    List1 = [x.lower() for x in search_terms]
    List2 = [x.lower() for x in text_clean]

    for item in List2:
        if item in List1:
            tech_skills.append(item)
        else:
            None 
    return tech_skills

tech_list = tech_count(text_clean)

ngrams = {}
words = 3

words_tokens = tech_list
for i in range(len(words_tokens)-words):
    seq = ' '.join(words_tokens[i:i+words])
    print(seq)
    if seq not in ngrams.keys():
        ngrams[seq] = []
    ngrams[seq].append(words_tokens[i+words])

curr_sequence = ' '.join(words_tokens[0:words])
output = curr_sequence
for i in range(100):
    if curr_sequence not in ngrams.keys():
        break
    possible_words = ngrams[curr_sequence]
    next_word = possible_words[random.randrange(len(possible_words))]
    output += ' ' + next_word
    seq_words = nltk.word_tokenize(output)
    curr_sequence = ' '.join(seq_words[len(seq_words)-words:len(seq_words)])

output_list = output.split()
output_total = Counter(output_list)
total = sum(output_total.values())
output_length = len(output_total)

output_dict = [(i, output_total[i] / total * 100.0) for i in output_total]
output_result = pd.DataFrame(output_dict, columns=['Tech','Percentage'])

output_result_chart = output_result.sort_values('Percentage', ascending=False).head(10)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=output_result_chart['Tech'],
    y=output_result_chart['Percentage'],
    name='Tech Prediction',
    marker_color='rgb(55, 83, 109)'
))
fig.update_layout(xaxis_tickfont_size=14, yaxis=dict(title='Skill Distribution (%)', titlefont_size=16, tickfont_size=14,), barmode='group', xaxis_tickangle=-45)
st.plotly_chart(fig)

def lookout(text):
    lookout_skill = []
    List1 = [x.lower() for x in total_result_chart['Tech']]
    List2 = [x.lower() for x in output_result_chart['Tech']]

    for item in List2:
        if item not in List1:
            lookout_skill.append(item)
        else:
            None 
    return lookout_skill

st.write(lookout(output_result_chart))

