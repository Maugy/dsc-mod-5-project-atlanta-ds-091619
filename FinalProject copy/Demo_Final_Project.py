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
from PIL import Image
from IPython.display import clear_output

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

import streamlit as st 
import terms 
import Cities 
import functions
import time
import mapbox

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.wsd import lesk
import SessionState

search_terms = terms.total_terms

st.title('Predicting Technology Trends')

session_state = SessionState.get(a='Data Scientist', b=0, c=0)
option = st.sidebar.text_input('Enter position title: ', value=session_state.a, key=9)
dtotal = pd.read_csv('cleaned_df.csv')


submit = st.sidebar.button('Search: ', key=1)
if submit == True:
    session_state.a = option
    dtotal= dtotal[dtotal['title'].astype(str).str.contains(option)]
else:
    pass

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

indeed = pd.read_csv('demo_indeed_webscrape.csv') 
linkedin = pd.read_csv('demo_webscrape.csv')

linkedin = linkedin[linkedin['title'].astype(str).str.contains(option)]
data_li = linkedin['description']
data_len = len(data_li)
data_li2 = functions.clean(data_li)
test1 = functions.word_count(data_li2)
c_let = functions.cleanC(data_li)
c_p = functions.C_plus(c_let)
c_s = functions.C_sharp(c_let)
test1a = Counter(c_p) + Counter(c_s)

indeed = indeed[indeed['title'].astype(str).str.contains(option)]
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

st.write('**Database Results:** ', total_length, '|     **Web Results:** ', web_len)
st.write("  \n")
st.write("  \n")
st.write("  \n")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=chart_comb.index,
    y=chart_comb['PercentageBaseline'],
    name='Database',
    marker_color='rgb(55, 83, 109)'
))
fig.add_trace(go.Bar(
    x=chart_comb.index,
    y=chart_comb['PercentageCurrent'],
    name='Web',
    marker_color='rgb(26, 118, 255)'
))

fig.update_layout(title="<b>Skill Comparison:</b>", font=dict(size=18), xaxis_tickfont_size=14, yaxis=dict(title='Skill Distribution', titlefont_size=16, tickfont_size=14,), barmode='group', xaxis_tickangle=-45)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='rgb(250, 20, 160)', ticklen=10,)
fig.update_yaxes(ticksuffix="%", rangemode="tozero", ticks="outside", tickwidth=2, tickcolor='rgb(250, 20, 160)', ticklen=10)    
st.plotly_chart(fig)

st.header("**Location Distribution:**")
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

image_legend = Image.open('legend2.png')
st.image(image_legend, use_column_width=True)

st.deck_gl_chart(
            viewport={'mapStyle': "mapbox://styles/mapbox/dark-v10",
                'latitude': map_graph['lat'].median(),
                'longitude':  map_graph['lon'].median(),
                'zoom': 2.6},
            layers=[{
                'type': 'ScatterplotLayer',
                'data': map_graph,
                'legend': True,
                'radiusScale': 20,
                'getLatitude': ['lat'],
                'getLongitude': ['lon'],
                'radiusMinPixels': 5,
                'getFillColor': [250, 20, 160],
                'pickable': True,
                'auto_highlight': True},
                {'type': 'ScatterplotLayer',
                'data': db_graph,
                'getLatitude': ['lat'],
                'getLongitude': ['lon'],
                'radiusScale': 20,
                'radiusMinPixels': 4,
                'getFillColor': [65, 105, 225, 180],
                'pickable': True,
                'auto_highlight': True}])
    

data = pd.read_csv('cleaned_df.csv')

if submit:
    session_state.b = option
    dtitle = data[data['title'].astype(str).str.contains(option)]
else:
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
outy = output_result_chart.reset_index(drop=True)
mergey = pd.merge(outy, look_chart, how='outer', indicator=True)
exist = mergey.loc[mergey._merge == 'left_only']
emerge = mergey.loc[mergey._merge == 'both']
x_label = list(outy.Tech)

st.write("  \n")
st.write("  \n")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=exist.index,
    y=exist['Percentage'],
    name='Existing'))

fig.add_trace(go.Bar(
    x=emerge.index,
    y=emerge['Percentage'],
    name='Emerging',
    marker_color='rgb(250, 20, 160)'))

fig.update_layout(title="<b>Emerging Technology:</b>", font=dict(size=18),xaxis_tickfont_size=14, yaxis=dict(title='Skill Distribution', titlefont_size=16, tickfont_size=14),
    xaxis_tickangle=-45,
    xaxis = dict(
        tickmode = 'array',
        tickvals = [0,1,2,3,4,5,6,7,8,9],
        ticktext = [x_label[0], x_label[1], x_label[2], x_label[3], x_label[4], x_label[5], x_label[6], x_label[7], x_label[8],x_label[9]]))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='rgb(250, 20, 160)', ticklen=10,)
fig.update_yaxes(ticksuffix="%", rangemode="tozero", ticks="outside", tickwidth=2, tickcolor='rgb(250, 20, 160)', ticklen=10)        
st.plotly_chart(fig)


time_series = pd.read_csv('time_series.csv')
searchy = terms.total_terms

skill = st.sidebar.selectbox('Skill Comparison: ', options=searchy, index=3)
if submit:
    session_state.c = skill
    ts_graph = time_series[time_series['Title'].astype(str).str.contains(option)]
else:
    ts_graph = time_series[time_series['Title'].astype(str).str.contains(option)]

ts_graph2 = ts_graph[ts_graph['Tech'].astype(str).str.contains(skill.lower())]
ts_graph3 = ts_graph2.groupby(['Date','Tech','Title'], as_index=False)['Percentage'].mean()
ts_date = pd.to_datetime(ts_graph3['Date'], format="%m-%d-%Y")
ts_date2 = ts_date.dt.strftime('%b %d, %Y')
fig = go.Figure(data=[go.Scatter(x=ts_date2,y=ts_graph3['Percentage'])])
fig.update_layout(title="<b>Trend:</b>" + skill, font=dict(size=18), yaxis=dict(title='Skill Distribution', titlefont_size=16, tickfont_size=14), 
    xaxis_tickangle=-45,
    xaxis = dict(title='Date', titlefont_size=16, tickfont_size=12,
        tickmode = 'linear',
        tick0 = [ts_date2.min(axis=0)],
        dtick = 1.0,
        ))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='rgb(250, 20, 160)', ticklen=10,)
fig.update_yaxes(ticksuffix="%", rangemode="tozero", ticks="outside", tickwidth=2, tickcolor='rgb(250, 20, 160)', ticklen=10)           
st.plotly_chart(fig)




