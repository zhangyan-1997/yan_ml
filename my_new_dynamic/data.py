import pandas as pd


def load_data(filename):
    data = pd.read_excel(filename)
    return data


def save_data(data, to_path):
    data.to_csv(to_path, index=False)
