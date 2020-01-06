# Final Project

import os
import logging
import pandas
from datetime import datetime 
import altair as alt
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

import nltk
from nltk.corpus import stopwords
import string
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go
import plotly.figure_factory as ff
# import pydeck as pdk
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)     

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
import mapbox
from google.cloud import bigquery, storage
from google_pandas_load import Loader, LoaderQuickSetup
from google_pandas_load import LoadConfig

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.wsd import lesk

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/users/dmauger/Flatiron/FinalProject/WebScraping-676f89d97ac9.json"

project_id = 'webscraping-261119'
dataset_id = 'Web_Scraping'
bucket_name = 'web_scrape_data'
gs_dir_path = 'https://storage.googleapis.com'
local_dir_path = '/users/dmauger/Flatiron/FinalProject/'
job_config = bigquery.LoadJobConfig()
job_config.autodetect = True
job_config.source_format = bigquery.SourceFormat.CSV
# bq_schema = [bigquery.SchemaField(name='title', field_type='STRING'),
#              bigquery.SchemaField(name='company', field_type='STRING'),
#              bigquery.SchemaField(name='location', field_type='STRING'),
#              bigquery.SchemaField(name='description', field_type='STRING')]

if not os.path.isdir(local_dir_path):
    os.makedirs(local_dir_path)

bq_client = bigquery.Client(
    project=project_id,
    credentials=None)


dataset_ref = bigquery.dataset.DatasetReference(
    project=project_id,
    dataset_id=dataset_id)


gs_client = storage.Client(
    project=project_id,
    credentials=None)

bucket = storage.bucket.Bucket(
    client=gs_client,
    name=bucket_name)


gpl = Loader(
    bq_client=bq_client,
    dataset_ref=dataset_ref,
    bucket=bucket,
    gs_dir_path=gs_dir_path,
    local_dir_path=local_dir_path, compress=False)


search_terms = terms.total_terms

st.title('Skill Distribution + Trends')

option = st.sidebar.text_input('Enter position title: ')
dtotal = pd.read_csv('total_data_date.csv')

if st.sidebar.button('Search: ', key=1) == True:
    dtotal= dtotal[dtotal['title'].astype(str).str.contains(option)]
total_length = len(dtotal)
dtotal2 = dtotal['description']
dtotal1 = functions.clean(dtotal2)
dtotal1 = functions.word_count(dtotal1)
c_let3 = functions.cleanC(dtotal2)
c_p3 = functions.C_plus(c_let3)
c_s3 = functions.C_sharp(c_let3)
test3a = Counter(c_p3) + Counter(c_s3)

ctotal = Counter(dtotal1) + Counter(test3a)
total = sum(ctotal.values())
Ctotaldict = [(i, ctotal[i] / total * 100.0) for i in ctotal]

total_result = pd.DataFrame(Ctotaldict, columns=['Tech','Percentage'])

total_resulty = pd.DataFrame(Ctotaldict, columns=['Tech','Percentage'])
total_resulty = total_resulty.set_index('Tech',drop=True)
total_result_chart = total_result.sort_values('Percentage', ascending=False).head(10)


periods = len(total_resulty)
date1 = dtotal.date
date1 = date1.replace({'date': "12-15-2019"})
date1 = pd.to_datetime(date1, format="%m-%d-%Y")
low = min(date1)
high = max(date1)
date2 = pd.date_range(low, high, periods=periods)
date3 = date2.strftime("%m-%d-%Y")
total_resulty['Date'] = date3
total_resulty['Title'] = option
total_resulty.to_csv('/users/dmauger/Flatiron/FinalProject/' + 'time_series.csv', mode='a', header=False, encoding='utf-8')

st.write('Database Results: ', total_length)

title_ind = str(option.replace(' ', '+'))
title_li = str(option.replace(' ', '%20'))

ind = "https://www.indeed.com/jobs?q=title%3A(%22"+title_ind+"%22)&l=United+States&sort=date"  
li = "https://www.linkedin.com/jobs/search?keywords="+title_li+"&location=United%20States&trk=guest_job_search_jobs-search-bar_search-submit&redirect=false&position=1&pageNum=0&f_TP=1%2C2"

indeed = WebScrape_Indeed.extract_title(ind) 
linkedin = WebScrape_LinkedIn.extract_title(li)

data_li = linkedin['description']
data_len = len(data_li)
data_li2 = functions.clean(data_li)
test1 = functions.word_count(data_li2)
c_let = functions.cleanC(data_li)
c_p = functions.C_plus(c_let)
c_s = functions.C_sharp(c_let)
test1a = Counter(c_p) + Counter(c_s)

data_ind = indeed['description']
data_len2 = len(data_ind)
data_ind2 = functions.clean(data_ind)
test2 = functions.word_count(data_ind2)
c_let2 = functions.cleanC(data_ind)
c_p2 = functions.C_plus(c_let2)
c_s2 = functions.C_sharp(c_let2)
test2a = Counter(c_p2) + Counter(c_s2)
web_len = data_len + data_len2

c = Counter(test1) + Counter(test2) + Counter(test1a) + Counter(test2a)
total = sum(c.values())
Cdict = [(i, c[i] / total * 100.0) for i in c]

results = pd.DataFrame(Cdict, columns=['Tech','Percentage'])

resulty = pd.DataFrame(Cdict, columns=['Tech','Percentage'])
resulty = resulty.set_index('Tech',drop=True)
result_chart = results.sort_values('Percentage', ascending=False).head(10)
merged = total_result.set_index('Tech').join(results.set_index('Tech'), lsuffix='Baseline', rsuffix='Current')
chart_comb = merged.sort_values('PercentageCurrent', ascending=False).head(10)

resulty['Date'] = datetime.now().strftime("%m-%d-%Y")
resulty['Title'] = option
resulty.to_csv('/users/dmauger/Flatiron/FinalProject/' + 'time_series.csv', mode='a', header=False, encoding='utf-8')

st.write('Web Results: ', web_len)


st.header("Skill Comparison: ")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=chart_comb.index,
    y=chart_comb['PercentageBaseline'],
    name='Baseline',
    marker_color='rgb(55, 83, 109)'
))
fig.add_trace(go.Bar(
    x=chart_comb.index,
    y=chart_comb['PercentageCurrent'],
    name='Current',
    marker_color='rgb(26, 118, 255)'
))

fig.update_layout(xaxis_tickfont_size=14, yaxis=dict(title='Skill Distribution', titlefont_size=16, tickfont_size=14,), barmode='group', xaxis_tickangle=-45)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=10,)
fig.update_yaxes(ticksuffix="%", rangemode="tozero", ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=10)    
st.plotly_chart(fig)

st.header("Location Distribution: ")
census = pd.read_csv('census_location.csv', index_col='city')
dblocation = dtotal['location']
dblocation = dblocation.to_list()
dbcity = [x.split(',')[0] for x in dblocation if isinstance(x, str)]

@st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def locate(word):
    location = []
    for x in census.index:
        if x in cities:
            location.append(census.loc[x])
    return location

@st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def dblocate(word):
    locationdb = []
    for x in census.index:
        if x in dbcity:
            locationdb.append(census.loc[x])
    return locationdb

db_city = dblocate(dbcity)
db_graph = pd.DataFrame(db_city, columns=['lat','lon'])

location = linkedin['location']
location = location.to_list()
city = [x.split(',')[0] for x in location if isinstance(x, str)]

location2 = indeed['location']
location2 = location2.to_list()
city2 = [x.split(',')[0] for x in location2 if isinstance(x, str)]

cities = city+city2

city_list = locate(cities)
map_graph = pd.DataFrame(city_list, columns=['lat','lon'])



st.deck_gl_chart(
            viewport={
                'latitude': map_graph['lat'].median(),
                'longitude':  map_graph['lon'].median(),
                'zoom': 2.5},
            layers=[{
                'type': 'ScatterplotLayer',
                'data': map_graph,
                'radiusScale': 10,
                'getLatitude': ['lat'],
                'getLongitude': ['lon'],
                'radiusMinPixels': 2,
                'getFillColor': [180, 0, 200, 140],
                'pickable': True,
                'auto_highlight': True},
                {'type': 'ScatterplotLayer',
                'data': db_graph,
                'getLatitude': ['lat'],
                'getLongitude': ['lon'],
                'radiusScale': 10,
                'radiusMinPixels': 2,
                'getFillColor': [55,83,109, 140],
                'pickable': True,
                'auto_highlight': True}])
    


time_series = pd.read_csv('time_series.csv')

skill = st.sidebar.selectbox('Skill Comparison: ', options=search_terms, index=3)
st.header('Trend Observed: ')
ts_graph = time_series[time_series['Title'].astype(str).str.contains(option)]
ts_graph2 = ts_graph[ts_graph['Tech'].astype(str).str.contains(skill.lower())]
ts_graph3 = ts_graph2.groupby(['Date','Tech','Title'], as_index=False)['Percentage'].mean()
ts_date = pd.to_datetime(ts_graph3['Date'], format="%m-%d-%Y")
ts_date2 = ts_date.dt.strftime('%b %d, %Y')
fig = go.Figure(data=[go.Scatter(x=ts_date2,y=ts_graph3['Percentage'])])
fig.update_layout(
    yaxis_title="Tech Distribution",
    xaxis_title="Date",
    xaxis_tickangle=-45,
    xaxis = dict(
        tickmode = 'linear',
        tick0 = [ts_date2.min()],
        dtick = 1.0,
        ))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=10,)
fig.update_yaxes(ticksuffix="%", rangemode="tozero", ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=10)           
st.plotly_chart(fig)

data = pd.read_csv('total_data_date.csv')
dtitle = data[data['title'].astype(str).str.contains(option)]
search_terms = terms.total_terms
search_terms = [x.lower() for x in search_terms]
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
            pass 
    return tech_skills

tech_list = tech_count(text_clean)

ngrams = {}
words = 3

words_tokens = tech_list
for i in range(len(words_tokens)-words):
    seq = ' '.join(words_tokens[i:i+words])

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
out_chart = output_result_chart.set_index('Tech', drop=True)


def lookout(text):
    lookout_skill = []
    List1 = [x.lower() for x in total_result_chart['Tech']]
    List2 = [x.lower() for x in out_chart.index]

    for item in List2:
        if item not in List1:
            lookout_skill.append(out_chart.loc[item])
        else:
            pass 
    return lookout_skill



lookout_chart = lookout(output_result_chart)
look_chart = pd.DataFrame(lookout_chart).reset_index()
look_chart = look_chart.rename(columns={'index': 'Tech'})
# look_chart = look_chart.set_index('Tech', drop=True)
outy = output_result_chart.reset_index(drop=True)
mergey = pd.merge(outy, look_chart, how='outer', indicator=True)
exist = mergey.loc[mergey._merge == 'left_only']
emerge = mergey.loc[mergey._merge == 'both']
x_label = list(outy.Tech)

st.header('Emerging Technology: ')
fig = go.Figure()
fig.add_trace(go.Bar(
    x=exist.index,
    y=exist['Percentage'],
    name='Existing'))

fig.add_trace(go.Bar(
    x=emerge.index,
    y=emerge['Percentage'],
    name='Emerging',
    marker_color='crimson'))

fig.update_layout(
    yaxis_title="Tech Distribution",
    xaxis_tickangle=-45,
    xaxis = dict(
        tickmode = 'array',
        tickvals = [0,1,2,3,4,5,6,7,8,9],
        ticktext = [x_label[0], x_label[1], x_label[2], x_label[3], x_label[4], x_label[5], x_label[6], x_label[7], x_label[8],x_label[9]]))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=10,)
fig.update_yaxes(ticksuffix="%", rangemode="tozero", ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=10)        
st.plotly_chart(fig)









WebScrape_LinkedIn.export_storage(linkedin)
WebScrape_Indeed.export_storage(indeed)


