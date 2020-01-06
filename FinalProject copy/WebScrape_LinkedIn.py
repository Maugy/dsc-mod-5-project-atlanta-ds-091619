# Web Scraping - LinkedIn
import os
import logging
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
from time import sleep, time
import lxml
import re
from urllib.parse import urljoin
import unicodedata

from datetime import datetime 
from google.cloud import bigquery, storage
from google_pandas_load import Loader, LoaderQuickSetup
from google_pandas_load import LoadConfig

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





data = {'title': [],
        'company': [], 
        'location': [],
        'description': [],
        'date': [],}

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    clean = re.sub(cleanr, ' ', str(raw_html))
    cleaner = clean.strip()
    cleantext = re.sub('\n', ' ', cleaner)
    return cleantext

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def desc_clean(text):
    clean = unicodedata.normalize("NFKD", str(text))
    testy = clean.replace('\\xa0','')
    return testy

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def export_table(data):
    table = pd.DataFrame(data, columns=['title', 'company', 'location', 'description', 'date'])
    table.index = table.index + 1
    # table.to_csv('/users/dmauger/Flatiron/FinalProject/' + 'total_data.csv', mode='a', encoding='utf-8', index=False)
    # gpl.load(source='dataframe', destination='bq', data_name='total_data', dataframe=table, write_disposition='WRITE_APPEND', compress=False)
    return table 
    
    # print('Scraping done. Here are the results:')
    # print(table.info())
# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def export_storage(table):
    table.to_csv('/users/dmauger/Flatiron/FinalProject/' + 'total_data_date.csv', mode='a', encoding='utf-8', index=False, header=False)
    gpl.load(source='local', destination='gs', data_name='total_data_date.csv', write_disposition='WRITE_APPEND')
    print('Process completed.')

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def job_details(job):
    
    sleep(1)

    r = requests.get(job)
    r.encoding = 'utf-8'

    html_content = r.text
    soup = BeautifulSoup(html_content, 'html.parser')
    info = soup.find('div', class_='topcard__content')

    try:
        title = info.find('h1').text
    except:
        title = 'NaN'

    try:
        company = soup.find('div', class_='topcard__content').find_all('span')[0].text
    except:
        company = 'NaN'

    try:
        location = info.find('h3', class_='topcard__flavor-row').find_all('span')[1].text
    except:
        location = 'NaN'

    try:
        description = soup.find('div', class_="description__text")
    except:
        description = soup.find('div', class_="description__text").find_all('p')

    data['title'].append(cleanhtml(title))
    data['company'].append(cleanhtml(company))
    data['location'].append(cleanhtml(location))
    data['description'].append(desc_clean(cleanhtml(description)))
    data['date'] = datetime.now().strftime("%m-%d-%Y")

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def extract_title(li_url):

    sleep(1)
    page = requests.get(li_url)
    page.encoding = 'utf-8'
    bs = BeautifulSoup(page.content, 'html.parser')

    links = []
    for div in bs.find('div', class_="results__container").find_all('li'):
        for a in div.find_all('a', attrs={'href': re.compile("^https://www.linkedin.com/jobs/")}):
            links.append(a['href'])

    for job in links:
        job_details(job)

    return export_table(data)



# extract_title(search_url)


# cit = input('Please, enter a city:\n')
# stat = input('Please, enter a state:\n')
# city = str(cit.replace(' ', '+'))
# state = str(stat.replace(' ','+'))
# location = city+"%2C%20"+state

# print(f'Searching {city},{state}. Please wait...')

# search_url = "https://www.linkedin.com/jobs/search?keywords=Data%20Scientist&location="+location+"&trk=guest_job_search_jobs-search-bar_search-submit&redirect=false&position=1&pageNum=0&f_TP=1%2C2"