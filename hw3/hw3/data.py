import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


def load_data(path):
    cat_dtypes = dict(x_2='category', x_21='category', x_24='category')
    df = pd.read_csv(path, header=0, dtype=cat_dtypes)
    # first col is just a sequence
    df = df.iloc[:, 1:]
    return df


def encode_categorical(df: pd.DataFrame):
    df = df.copy()
    ord_enc = OrdinalEncoder(categories='auto')
    cat_cols = list(df.columns[df.dtypes == 'category'])
    df[cat_cols] = ord_enc.fit_transform(df[cat_cols])
    return df


def get_training_data(df: pd.DataFrame, scale_covariates=False):
    ncols = len(df.columns)
    n_covariates = ncols - 2  # T and Y

    X = df.iloc[:, :n_covariates].to_numpy()
    t = df['T'].to_numpy()
    y = df['Y'].to_numpy()

    if scale_covariates:
        X = StandardScaler().fit_transform(X)

    return X, y, t
