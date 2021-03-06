import re
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

CASTRR_METADATA_PATTERN = \
    re.compile(r'^.*<age-range>: (?P<age>\d+).*<sex>: (?P<sex>\w+).*$',
               re.DOTALL | re.MULTILINE)


def load_mhrv_xls(
        path: str, sheet_names: list, df_meta: pd.DataFrame = None
) -> Dict[str, pd.DataFrame]:
    """
    Loads raw data from excel file containing HRV features generated by mhrv.
    @param path: Path to file.
    @param sheet_names: Names of excel sheets to load.
    @param df_meta: Extra metadata about each record.
    Should be a dataframe where the index is the record name.
    @return: A dict from sheet name to pandas dataframe. The summary
    statistics rows will be removed. Each dataframe will have a multiindex
    of (rec, win) which are are record name and window number.
    """

    dfs: Dict[str, pd.DataFrame] = pd.read_excel(
        path, header=0, index_col=0,
        sheet_name=list(sheet_names),
    )

    summary_rows = ('Mean', 'Median', 'SE')
    for name, df in dfs.items():
        # Remove the rows which contain summary statistics
        df = df.loc[~df.index.isin(summary_rows), :]

        # Convert dtypes
        df = df.astype(np.float32, copy=False)
        df = df.astype(dict(RR=np.int32, NN=np.int32), copy=False)

        # Impute NaN with mean
        df = df.fillna(value=df.mean(skipna=True))

        # Replace index with multi-index
        df.index = pd.MultiIndex.from_tuples(
            [i.split("_") for i in df.index],
            names=['rec', 'win']
        )

        if df_meta is not None:
            # Join, but only keep rows from df
            df = df.join(df_meta, how='left')

        dfs[name] = df

        print(f'Loaded {name}: {len(df)} samples, {len(df.columns)} features')

    return dfs


def castrr_load_metadata(db_dir: str) -> pd.DataFrame:
    """
    Parses metadata (age and sex) from records of the physionet CAST RR
    database (crisdb).
    The database is available here.
    https://physionet.org/content/crisdb/1.0.0/
    @param db_dir:
    @return: A DataFrame where index is record name, and columns are (AGE,
    SEX).
    """
    meta = {}

    db_dir = Path(db_dir)
    assert db_dir.is_dir()

    header_paths = db_dir.glob('**/*.hea')
    for header_path in header_paths:
        rec_name = header_path.stem
        with open(str(header_path), 'r') as f:
            m = CASTRR_METADATA_PATTERN.match(f.read())
            if not m:
                print(f'WARNING: header does not match expected format in '
                      f'file {header_path}', file=sys.stderr)
                continue
            meta[rec_name] = m.groupdict()

    df_meta: pd.DataFrame = pd.DataFrame(meta).transpose().sort_index(axis=0)
    df_meta.index.name = 'rec'
    df_meta.columns = [c.upper() for c in df_meta.columns]
    df_meta = df_meta.astype({'AGE': np.int32, 'SEX': 'category'})
    return df_meta


def castrr_ci_dataset(
        df_control: pd.DataFrame, df_treated: pd.DataFrame,
        psd_type: str = 'AR',
        outcome_mse=True, outcome_dfa=True, outcome_beta=True,
        ignore_features=(), include_counterfactuals=False,
        random_seed=None,
) -> pd.DataFrame:
    """
    Creates a causal inference dataset from a set for control and treated
    features from the mhrv analysis of the CASTRR data.

    To create a causal inference dataset, we'll create our own control
    and treatment groups by randomly assigning each subject for either group.
    For subjects assigned to our control group, we take their data as is.
    For subjects assigned to
    the treatment group, we take the pre-treatment HRV features as
    covariates and the post-treatment computed outcomes as outcome variables.

    @param df_control: DataFrame with control data.
    @param df_treated: DataFrame with post-treatment data.
    @param psd_type: Type of PSD estimate to take from dataset,
    which contains multiple estimates as e.g. "VLF_POWER_AR". The
    selected variables will be renamed to e.g. "VLF_POWER".
    @param outcome_mse: Whether to generate MSE-based outcome variables.
    @param outcome_dfa: Whether to generate DFA-based outcome variables
    @param ignore_features: List of features to drop.
    @param include_counterfactuals: Whether to include counterfactual
    outcome columns in the output. Due to the nature of the CAST RR dataset,
    we can do this. For our control group, the counterfatual outcomes will
    come from the CASTRR treated records of the corresponding patient,
    and for our treated group they'll come from the CASTRR control records.
    @param random_seed: Random seed for splitting patients into our own
    control and treated groups.
    @return: Single dataframe with consolidated PSD features, marked
    covariate and output features ("X_" and "Y_") prefix, a treatment
    column 'T' denoting our control/treated split. The index of the
    dataframe will be (rec, win) where rec is the patient id (without suffix
    'a'/'b').
    """
    psd_suffix = f'_{psd_type.upper()}'

    # First, take only matching rows from control and treated, by moving
    # them to a common index
    df_control, df_treated = _castrr_common_index(df_control, df_treated)

    # Extract patient ids. The common index is ('rec', 'win') where rec
    # is e.g. 'e001'. Take level 0 of index ('rec').
    patient_ids = list(df_control.index.levels[0])

    dfs = {'control': df_control, 'treated': df_treated}

    covariates, outcomes = [], []

    for name, df in dfs.items():
        # Treat PSD columns: Keep only the requested type,
        # remove the other type and rename the columns
        df = _consolidate_psd(df, psd_suffix)

        # Compute the outcome
        df, outcome_cols = _create_outcome_columns(
            df, outcome_mse, outcome_dfa, outcome_beta, prefix=''
        )

        # Mark the columns with prefixes and reorder dataframe
        df, covariates, outcomes = _mark_dataset(
            df, outcome_cols, ignore=ignore_features,
            covariates_prefix='X_', outcomes_prefix='Y_'
        )

        dfs[name] = df

    # Create our own control and treated patient groups
    if random_seed is not None:
        np.random.seed(random_seed)
    ids_control = sorted(
        np.random.choice(patient_ids, len(patient_ids) // 2, replace=False)
    )
    ids_treated = sorted(set(patient_ids).difference(ids_control))

    # For our control group, we'll only use the control data as is
    df_control: pd.DataFrame = dfs['control'].loc[ids_control]

    # Add counterfactual outcome from their post-treatment records
    df_control_cf = dfs['treated'].loc[ids_control][outcomes]
    if include_counterfactuals:
        df_control = df_control.join(df_control_cf, how='left', rsuffix='_CF')

    # For our treatment group, we need to take their covariates
    # from the pre-treatment data and their outcome from the post-treatment
    # data.
    df_treated_covariates = dfs['control'].loc[ids_treated][covariates]
    df_treated_outcomes = dfs['treated'].loc[ids_treated][outcomes]
    df_treated: pd.DataFrame = df_treated_covariates.join(df_treated_outcomes,
                                                          how='left')
    # Add counterfactual outcome from the pre-treatment records
    df_treated_cf = dfs['control'].loc[ids_treated][outcomes]
    if include_counterfactuals:
        df_treated = df_treated.join(df_treated_cf, how='left', rsuffix='_CF')

    # Assign treatment variable
    df_control = df_control.assign(T=np.int32(0))
    df_treated = df_treated.assign(T=np.int32(1))

    df = df_control.append(df_treated)
    return df


def _castrr_common_index(
        df_control: pd.DataFrame, df_treated: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a common index from control and treated dataframes loaded from
    the CASTRR dataset.
    @param df_control:
    @param df_treated:
    @return:
    """

    def transform_idx(idx):
        rec, win = idx
        try:
            win = int(win)
        except Exception as e:
            win = 1
            print(f"WARNING: Can't convert window to int rec={rec}, win={win}:"
                  f" {e}", file=sys.stderr)
        return rec[:-1], win

    idx_all = set()
    idx_names = None
    dfs = {'control': df_control, 'treated': df_treated}
    for name, df in dfs.items():
        if not idx_names:
            idx_names = df.index.names
        else:
            # Make sure both dfs have the same index names
            assert idx_names == df.index.names

        # Indices in the CAST RR db are e.g. 'e001a' for control and 'e001b'
        # for treated. Remove the 'a'/'b' part so we can create a common index.
        new_idx = pd.MultiIndex.from_tuples(
            [transform_idx(i) for i in df.index], names=idx_names
        )
        idx_all.update(new_idx.values)
        df = df.copy()
        df.index = new_idx
        dfs[name] = df

    # Find index tuples that exist both in control and treated
    idx_common = set(idx_all)
    for name, df in dfs.items():
        curr_idx = df.index.values
        idx_common = idx_common.intersection(curr_idx)

    idx_common = sorted(idx_common)
    idx_common = pd.MultiIndex.from_tuples(idx_common, names=idx_names)

    # Keep only rows from the common index
    for name, df in dfs.items():
        dfs[name] = df.loc[idx_common]

    return dfs['control'], dfs['treated']


def _create_outcome_columns(
        df: pd.DataFrame, mse=True, dfa=True, beta=True,
        prefix="Y_", drop_features=True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates outcome columns based on Multiscale Entropy (MSE) and Detrended
    Fluctuation Analysis (DFA) features, and drops these columns from the
    dataset.
    @param df: A dataframe with features from mhrv.
    @param mse: Whether to create MSE outcome column and drop the MSE
    feature columns.
    @param dfa: Whether to create DFA outcome column and drop the DFA
    feature column.
    @param prefix: Prefix to prepend to new columns.
    @return: A tuple (df, outcome_colums) where df is a new dataframe with the
    added columns and outcome_colums is a list containing their names.
    """

    assert mse or dfa
    outcome_columns = []
    outcomes = {}
    re_flags = re.IGNORECASE

    def fn_mean(df_cols: pd.DataFrame):
        return df_cols.mean(axis=1)

    def fn_neg_mean(df_cols: pd.DataFrame):
        return - df_cols.mean(axis=1)

    if mse:
        outcomes['MSE_LO'] = (re.compile(r'^MSE\d$', re_flags),
                              fn_mean)
        outcomes['MSE_HI'] = (re.compile(r'^MSE[1-9][0-9]$', re_flags),
                              fn_mean)
    if dfa:
        outcomes['ALPHA1'] = (re.compile(r'^alpha1$', re_flags),
                              fn_mean)
        outcomes['ALPHA2'] = (re.compile(r'^alpha2$', re_flags),
                              fn_mean)
    if beta:
        outcomes['BETA'] = (re.compile(r'^BETA$', re_flags),
                            fn_neg_mean)

    for outcome_name, (column_pattern, outcome_fn) in outcomes.items():
        cols = [c for c in df.columns if column_pattern.match(c)]
        assert len(cols) > 0

        outcome_value = outcome_fn(df[cols])

        outcome_column = f'{prefix}{outcome_name}'
        df = df.assign(**{outcome_column: outcome_value})

        if drop_features and outcome_column not in cols:
            df = df.drop(columns=cols)

        outcome_columns.append(outcome_column)

    return df, outcome_columns


def _mark_dataset(
        df: pd.DataFrame,
        outcomes: List[str],
        ignore=(),
        covariates_prefix="X_",
        outcomes_prefix="Y_",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Marks dataset variables as covariates or outcome variables.
    (Assumes no treatment variable exists).
    @param df: DataFrame containing the dataset to split.
    @param outcomes: Names of columns with outcome variables.
    @param ignore: Covariate names to remove.
    @param covariates_prefix: Prefix to give to covariate variables in the
    output.
    @param outcomes_prefix: Prefix to give to outcome variables in the
    output.
    @return: Tuple (df, covariates, outcomes) where df is a DataFrame with
    marked covariates and outcomes in this order and marked
    according to the prefixes; covariates and outcomes are the column names
    after marking them and removing the ignored features.
    """

    # Remove ignored columns
    if ignore:
        df = df.drop(columns=ignore, inplace=False)

    df_Y = df[outcomes]
    outcomes_map = {o: f'{outcomes_prefix}{o}' for o in outcomes}
    df_Y = df_Y.rename(columns=outcomes_map, inplace=False)

    covariates = [c for c in df.columns if c not in outcomes]
    df_X = df[covariates]
    covariates_map = {c: f'{covariates_prefix}{c}' for c in covariates}
    df_X = df_X.rename(columns=covariates_map, inplace=False)

    # Reorder columns
    return (
        pd.concat([df_X, df_Y], axis=1),
        list(covariates_map.values()), list(outcomes_map.values())
    )


def _consolidate_psd(df: pd.DataFrame, psd_suffix):
    """
    Keep only one type of PSD features.
    @param df: A dataframe from mhrv output.
    @param psd_suffix: Suffix of PSD features we wish to keep, e.g. "AR".
    @return: The dataframe with other PSD types removed, and the requested
    features renamed to not include the prefix.
    """
    # Discover PSD columns
    cols = df.columns
    psd_cols = set(
        map(lambda c: c.split(psd_suffix)[0],
            filter(lambda c: c.endswith(psd_suffix), cols))
    )

    # Discover PDS colums of other types
    other_psd_type_cols = filter(lambda c: any(
        [c.startswith(n) and not c.endswith(psd_suffix) for n in psd_cols]
    ), cols)

    df = df.drop(columns=other_psd_type_cols)
    df = df.rename(
        lambda c: c if not c.endswith(psd_suffix) else c.split(psd_suffix)[0],
        axis='columns'
    )
    return df


def split_dataset(
        df: pd.DataFrame,
        covariates_prefix="X_",
        outcomes_prefix="Y_",
        treatment='T',
        scale_covariates=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    @param df: DataFrame to split.
    @param covariates_prefix: Prefix of the covariate variables in the
    input.
    @param outcomes_prefix: Prefix of the outcome variables in the
    input.
    @param treatment: Name of treatment variable.
    @param scale_covariates: Whether to scale the covariates using a
    standard scaler.
    @return: Tuple (X, y, y) containing numpy arrays with the covariates,
    outcomes and treatment respectively.
    """
    df = df.copy()

    # Handle categorical data by converting categories to integers
    ord_enc = OrdinalEncoder(categories='auto')
    cat_cols = list(df.columns[df.dtypes == 'category'])
    df[cat_cols] = ord_enc.fit_transform(df[cat_cols])

    covariates = [c for c in df.columns if c.startswith(covariates_prefix)]
    outcomes = [c for c in df.columns if c.startswith(outcomes_prefix)]

    X = df[covariates].values.astype(np.float32)
    y = df[outcomes].values.astype(np.float32)
    t = df[treatment].values.reshape(-1).astype(np.int32)

    if y.shape[1] == 1:
        y = y.reshape(-1)

    if scale_covariates:
        X = StandardScaler().fit_transform(X)

    return X, y, t
