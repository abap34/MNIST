import glob
import pandas as pd
import os


def _load_data(full_path):
    files = glob.glob(full_path)
    df = pd.DataFrame()
    for file in files:
        fe = pd.read_feather(file)
        df = pd.concat([df, fe], axis=1)
    df.sort_index(axis=1, inplace=True)
    return df

def load_data(file_dir):
    train = _load_data(os.path.join(file_dir, 'train'))
    test = _load_data(os.path.join(file_dir, 'test'))
    
    return train, test



