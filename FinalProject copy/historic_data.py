
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
import WebScrape_Indeed
import WebScrape_LinkedIn
import streamlit as st 
import terms 
import Cities 
import functions
import time
import plotly.figure_factory as ff
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
import WebScrape_Indeed
import WebScrape_LinkedIn
import streamlit as st 
import terms 
import Cities 
import functions


search_terms = terms.total_terms

def clean(raw):
    raw = ' '.join(raw.tolist())
    for char in '-.,\n':
        raw = raw.replace(char,' ')
    
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    raw = nltk.regexp_tokenize(raw, pattern)
    return raw


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

dtotal = pd.read_csv('total_data.csv')
dtitle = dtotal[dtotal['title'].astype(str).str.contains(option)]

dtotal = dtotal['description']
dtotal = clean(dtotal)
dtotal = word_count(dtotal)

ctotal = Counter(dtotal)
total = sum(ctotal.values())
Ctotaldict = [(i, ctotal[i] / total * 100.0) for i in ctotal]

total_result = pd.DataFrame(Ctotaldict, columns=['Tech','Percentage'])

total_result_chart = total_result.set_index('Tech',drop=True)