import pandas as pd
import numpy as np
import os
import json
os.environ['KAGGLE_CONFIG_DIR'] = '/home/eliormi/PycharmProjects/TellMeWhy'
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# print(f"Pandas version: {pd.__version__}")
# print(f"NumPy version: {np.__version__}")


# Load Kaggle API credentials
with open('kaggle.json') as f:
    kaggle_creds = json.load(f)

# Set Kaggle credentials as environment variables
os.environ['KAGGLE_USERNAME'] = kaggle_creds['username'] 
os.environ['KAGGLE_KEY'] = kaggle_creds['key']

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download dataset (replace with desired dataset)
# Format: api.dataset_download_files('username/dataset-name', path='.')
api.dataset_download_files('atharvasoundankar/chocolate-sales', path='.')
print('Dataset downloaded successfully')

with zipfile.ZipFile("chocolate-sales.zip", 'r') as zip_ref:
    zip_ref.extractall("chocolate-sales")

    



