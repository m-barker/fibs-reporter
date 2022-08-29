import pandas as pd
import numpy as np
import src.config as config

from src.utility import is_categorical
from src.utility import remove_constant_vars
from src.utility import remove_nans_and_missing

config.define_constants()


def test_is_categorical():
    df = pd.DataFrame()
    df["feature_1"] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    df["feature_2"] = ["A", "B", "C", "0.4", "0.5", "1.6", "1.7", "1.8", "1.9", "2.0"]
    df["feature_3"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df["feature_4"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    df["output_var"] = [True, False, True, False, True, True, True, False, False, False]
    assert is_categorical(df, "output_var")
    assert not is_categorical(df, "feature_1")
    assert is_categorical(df, "feature_2")
    assert not is_categorical(df, "feature_3")
    assert not is_categorical(df, "feature_4")

    df = pd.DataFrame()
    df["output_var"] = list(np.arange(1000))
    zeros = list(np.zeros(1000))
    zeros[999] = 1
    zeros = [int(x) for x in zeros]
    df["feature_1"] = zeros
    print(df)
    assert is_categorical(df, "feature_1")


def test_remove_constants():
    df = pd.DataFrame()

    df["var_1"] = list(np.arange(5))
    df["var_2"] = [1, 1, 1, 1, 1]
    df["var_3"] = [1.0, 1.0, 1.0, 1.0, 1.0]
    df["var_4"] = [False, False, False, False, False]
    df["var_5"] = [True, True, True, True, True]
    df["var_6"] = ["A", "A", "A", "A", "A"]
    df["var_7"] = ["1", "1", "1", "1", "1"]

    assert remove_constant_vars(df, "var_2") == ["var_3", "var_4", "var_5", "var_6", "var_7"]

    df_2 = pd.DataFrame()
    df_2["var_1"] = list(np.arange(5))
    df_2["var_2"] = [1, 1, 1, 1, 1]

    assert df_2.shape == df.shape


def test_remove_nans_and_infs():
    df = pd.DataFrame()

    df["var_1"] = list(np.arange(5))
    df["var_2"] = [1, 1, np.inf, 1, 1]
    df["var_3"] = [1.0, 1.0, 1.0, 1.0, 1.0]
    df["var_4"] = [False, False, False, "inf", False]
    df["var_5"] = [True, True, True, True, True]
    df["var_6"] = ["A", "A", "A", "A", "A"]
    df["var_7"] = ["1", "1", "1", "1", "-inf"]

    df_2 = pd.DataFrame()
    df_2["var_1"] = [0, 1]
    df_2["var_2"] = [1, 1]
    df_2["var_3"] = [1.0, 1.0]
    df_2["var_4"] = [False, False]
    df_2["var_5"] = [True, True]
    df_2["var_6"] = ["A", "A"]
    df_2["var_7"] = ["1", "1"]

    remove_nans_and_missing(df)

    assert list(df["var_1"]) == list(df_2["var_1"])
    assert list(df["var_2"]) == list(df_2["var_2"])
    assert list(df["var_3"]) == list(df_2["var_3"])
    assert list(df["var_4"]) == list(df_2["var_4"])
    assert list(df["var_5"]) == list(df_2["var_5"])
    assert list(df["var_6"]) == list(df_2["var_6"])
    assert list(df["var_7"]) == list(df_2["var_7"])


