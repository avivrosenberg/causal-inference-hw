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
    """
    Encodes the categorical features of a dataset as ordinal integers.
    @param df: A Dataframe.
    @return: A new dataframe, with the categorical features replaced.
    """
    df = df.copy()
    ord_enc = OrdinalEncoder(categories='auto')
    cat_cols = list(df.columns[df.dtypes == 'category'])
    df[cat_cols] = ord_enc.fit_transform(df[cat_cols])
    return df


def get_training_data(df: pd.DataFrame, scale_covariates=False,
                      col_t='T', col_y='Y', col_prop='propensity'):
    """
    Extracts covariates X, outcome y, treatment assignment t and propensity
    scores p from a dataset.
    @param df: Dataframe containing a dataset.
    @param scale_covariates: Whether to normalize each covariate to zero
    mean and unit variance.
    @param col_t: Name of treatment assignment column
    @param col_y: Name of outcome column
    @param col_prop: Name of propensity column
    @return: Tuple (X, y, t, p). If any of y, t, p are not found, they will
    be returned as None.
    """

    cols = df.columns
    cols_y_t_p = (col_y, col_t, col_prop)
    cols_X = [col for col in cols if col not in cols_y_t_p]

    X = df[cols_X].to_numpy()
    y, t, p = tuple(df[col].to_numpy() if col else None for col in cols_y_t_p)

    if scale_covariates:
        X = StandardScaler().fit_transform(X)

    return X, y, t, p
