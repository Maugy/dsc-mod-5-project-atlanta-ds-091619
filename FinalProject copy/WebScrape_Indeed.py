# Web Scrape - Indeed

import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
from time import sleep, time
import lxml
import re
from urllib.parse import urljoin
import unicodedata
import streamlit as st

import os
import logging
import pandas
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

    r = requests.get(job)
    r.encoding = 'utf-8'
    sleep(1)

    html_content = r.text
    soup = BeautifulSoup(html_content, 'html.parser')

    try:
        title = soup.find('h3', class_='jobsearch-JobInfoHeader-title').text
    except:
        title = 'NaN'

    try:
        company = soup.find_all(
            'div', class_="jobsearch-InlineCompanyRating")[-1].find_all('div')[0].text
    except:
        company = 'NaN'

    try:
        location = soup.find_all(
            'div', class_="jobsearch-InlineCompanyRating")[-1].find_all('div')[-1].text
    except:
        location = 'NaN'

    try:
        description = soup.find_all(
            'div', class_="jobsearch-JobComponent-description")
    except:
        description = 'NaN'

    data['title'].append(title)
    data['company'].append(company)
    data['location'].append(location)
    data['description'].append(cleanhtml(description))
    data['date'] = datetime.now().strftime("%m-%d-%Y")

# @st.cache(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def extract_title(ind_url):

    sleep(1)
    page = requests.get(ind_url)
    bs = BeautifulSoup(page.content, 'html.parser')

    links = []
    for div in bs.find_all('div', class_="title"):
        for a in div.find_all('a', href=True):
            links.append(urljoin('https://indeed.com', a['href']))

    for job in links:
        job_details(job)

    # next_page_text = bs.find('div', class_="pagination").find_all('a')
    # next_page = [link.get('href') for link in next_page_text][-1]

    # if '&start=20' not in next_page:
    #     next_page_url = (urljoin('https://indeed.com', cleanhtml(next_page)))
    #     print(next_page_url)
    #     extract_title(next_page_url)
    # else:
    #     export_table(data)

    return export_table(data)

# cit = st.text_input('Enter city')
# stat = st.text_input('Enter state')
# city = str(cit.replace(' ', '+'))
# state = str(stat.replace(' ','+'))
# location = city+"%2C+"+state+"&radius=50&sort=date"

# search_url = "https://www.indeed.com/jobs?q=title%3A(%22data+scientist%22+OR+%22data+science%22+OR+%22data+analyst%22)&l="+location

# extract_title(search_url)