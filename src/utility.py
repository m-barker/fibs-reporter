import pandas as pd
import numpy as np

import config

def get_same_variables(correlation_dfs: list, feature_df: pd.DataFrame) -> list:
    """

    :param correlation_dfs:
    :param feature_df:
    :return:
    """

    all_corr_vars = []
    for df in correlation_dfs:
        if df is not None:
            variables = list(df.index.values)
            [all_corr_vars.append(v) for v in variables]

    all_corr_vars = set(all_corr_vars)
    feature_vars_1 = set(feature_df.iloc[:, 0])
    feature_vars_2 = set(feature_df.iloc[:, 2])

    common_vars = all_corr_vars.intersection(feature_vars_1 | feature_vars_2)
    return list(common_vars)


def remove_nans_and_missing(df: pd.DataFrame):
    """
    Removes missing values and NANs from self.df
    and stores number of NANs
    :return: Df of number_of_nans in each variable
    """

    # First, convert inf and -infs to NANs
    df.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
    df.replace(to_replace=["inf", "-inf"], value=np.nan, inplace=True)
    df_nan = df.isna()
    number_of_nans_df = df_nan.sum()
    number_of_nans_df = pd.DataFrame(number_of_nans_df)
    number_of_nans_df.columns = ["Number of missing/NAN values"]
    number_of_nans = number_of_nans_df.sum().iloc[0]

    if number_of_nans == 0:
        number_of_nans_df = None
    else:
        number_of_nans_df = number_of_nans_df[number_of_nans_df["Number of missing/NAN values"] > 0]

    df.dropna(inplace=True)
    return number_of_nans_df


def remove_constant_vars(df, output_var: str) -> list:
    """
    Removes independent variables that have constant values from self.df
    and stores list of removed variables
    :return: List of constant var names
    """
    constant_vars = []
    for var in df.columns:
        if var == output_var:
            continue
        if len(list(pd.unique(df[var]))) == 1:
            constant_vars.append(var)
            df.drop([var], axis=1, inplace=True)

    return constant_vars


def is_categorical(df, var_name: str) -> bool:
    """
    Determines whether a given variable within self.df is categorical, according
    to its number of unique values being less than cutoff
    :param var_name: Name of the variable within self.df to check for type
    :return: Bool, True if variable is deemed to be categorical, else False
    """
    max_categories = config.MAX_CAT
    data_type = df[var_name].dtypes
    if data_type == "float64":
        return False
    if data_type == "float32":
        return False
    if data_type == "string":
        return True
    if data_type == "boolean":
        return True
    if data_type == "object":
        return True

    rows, _ = df.shape
    number_unique = len(list(pd.unique(df[var_name])))
    percent_unique = number_unique / rows * 100

    if rows < 500:
        if percent_unique <= 10:
            return True
        return False

    if number_unique <= max_categories:
        return True
    return False
