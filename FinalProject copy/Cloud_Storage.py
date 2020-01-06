# Cloud Storage

# from google.cloud import bigquery, storage
# from google_pandas_load import Loader, LoaderQuickSetup
# from google_pandas_load import LoadConfig

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/users/dmauger/Flatiron/FinalProject/WebScraping-676f89d97ac9.json"

# project_id = 'webscraping-261119'
# dataset_id = 'Web_Scraping'
# bucket_name = 'web_scrape_data'
# gs_dir_path = 'gs://web_scrape_data'
# local_dir_path = '/users/dmauger/Flatiron/FinalProject/'
# job_config = bigquery.LoadJobConfig()
# job_config.autodetect = True
# job_config.source_format = bigquery.SourceFormat.CSV
# bq_schema = [bigquery.SchemaField(name='title', field_type='STRING'),
#              bigquery.SchemaField(name='company', field_type='STRING'),
#              bigquery.SchemaField(name='location', field_type='STRING'),
#              bigquery.SchemaField(name='description', field_type='STRING')]
             
# if not os.path.isdir(local_dir_path):
#     os.makedirs(local_dir_path)

# bq_client = bigquery.Client(
#     project=project_id,
#     credentials=None)

# dataset_ref = bigquery.dataset.DatasetReference(
#     project=project_id,
#     dataset_id=dataset_id)

# gs_client = storage.Client(
#     project=project_id,
#     credentials=None)

# bucket = storage.bucket.Bucket(
#     client=gs_client,
#     name=bucket_name)

# gpl = Loader(
#     bq_client=bq_client,
#     dataset_ref=dataset_ref,
#     bucket=bucket,
#     gs_dir_path=gs_dir_path,
#     local_dir_path=local_dir_path)