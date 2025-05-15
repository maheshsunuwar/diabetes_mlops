from sklearn.datasets import load_diabetes
import pandas as pd

def load_and_save():
    data = load_diabetes(as_frame=True) # load data as dataframe
    df = data.frame
    df.to_csv('data/raw.csv', index=False)

    print('Data saved to data/raw.csv')

if __name__ == '__main__':
    load_and_save()
