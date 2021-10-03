import os
import pandas as pd
def get_olist_dataset (dataset):
    csv_dir = 'csv'
    path = os.path.join(csv_dir, f'olist_{dataset}_dataset.csv')
    return pd.read_csv(path)