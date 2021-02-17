import pandas as pd
import numpy as np
from datetime import datetime


def load_dataset(data_content):
    # Data Initialization
    print('Loading Data from CSV File...', flush=True)

    # Preparing tours.csv
    data_content.tours_df = pd.read_csv(data_content.base_dir + 'tours.csv')
    print('tours.csv loaded...', flush=True)

    # Preparing bikers_network.csv
    data_content.bikers_network_df = pd.read_csv(data_content.base_dir + 'bikers_network.csv')
    print('bikers_network.csv loaded...', flush=True)

    # Preparing tour_convoy.csv
    data_content.tour_convoy_df = pd.read_csv(data_content.base_dir + 'tour_convoy.csv')
    print('tour_convoy.csv loaded...', flush=True)

    # Preparing train.csv
    data_content.train_df = pd.read_csv(data_content.base_dir + 'train.csv')
    data_content.train_df['timestamp'] = data_content.train_df['timestamp'].apply(
        lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M:%S'))
    print('train.csv loaded...', flush=True)

    # Preparing test.csv
    data_content.test_df = pd.read_csv(data_content.base_dir + 'test.csv')
    data_content.test_df['timestamp'] = data_content.test_df['timestamp'].apply(
        lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M:%S'))
    print('test.csv loaded...', flush=True)

    # Prepating bikers.csv
    data_content.bikers_df = pd.read_csv(data_content.base_dir + 'bikers.csv')
    print('Data Loaded...100%\n', flush=True)

    return data_content
