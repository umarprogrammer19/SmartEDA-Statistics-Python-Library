import pandas as pd


def detect_duplicates(df):
    return df.duplicated().sum()


def get_duplicate_rows(df):
    return df[df.duplicated(keep=False)]


def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)
