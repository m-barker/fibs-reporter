import pandas as pd
import src.config as config

from src.Data_analyser import DataAnalyser

config.define_constants()


def test_bool_categorical():
    df = pd.DataFrame()
    df["feature"] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    df["output_var"] = [True, False, True, False, True, True, True, False, False, False]
    analyser = DataAnalyser(df, "output_var")
    assert (analyser.is_categorical("output_var"))


def test_string_categorical():
    df = pd.DataFrame()
    df["feature"] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    df["output_var"] = ["T", "F", "T", "F", "T", "T", "T", "F", "F", "F"]
    analyser = DataAnalyser(df, "output_var")
    assert (analyser.is_categorical("output_var"))


def test_float_cont():
    df = pd.DataFrame()
    df["feature"] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    df["output_var"] = [True, False, True, False, True, True, True, False, False, False]
    analyser = DataAnalyser(df, "output_var")
    assert (not analyser.is_categorical("feature"))


def test_constant_indep():
    df = pd.DataFrame()
    df["output_var"] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    df["feature"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    analyser = DataAnalyser(df, "output_var")
    assert (analyser.kruskal_wallis_h(indep_var="feature_1") == (0, 1))


def test_constant_dep():
    df = pd.DataFrame()
    df["feature"] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    df["output_var"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    analyser = DataAnalyser(df, "output_var")
    assert (analyser.kruskal_wallis_h(indep_var="feature", categorical="output") == (0, 1))


def test_stat():
    df = pd.DataFrame()
    df["feature"] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    df["output_var"] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    analyser = DataAnalyser(df, "output_var")
    assert (analyser.kruskal_wallis_h(indep_var="feature", categorical="output") == (6.818181818181813,
                                                                                     0.009023438818080334))


def test_spearman():
    df = pd.DataFrame()
    df["feature"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df["output_var"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    analyser = DataAnalyser(df, "output_var")
    assert(analyser.spearman_correlation("feature") == (0.9999999999999999, 6.646897422032013e-64))


def test_point_biserial():
    df = pd.DataFrame()
    df["feature"] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    df["output_var"] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    analyser = DataAnalyser(df, "output_var")
    assert(analyser.point_biserial_correlation(["feature"]) == [0.8558475122559045])


def test_chi_squared():
    df = pd.DataFrame()
    df["feature"] = [True, False, True, False, True, True, True, False, False, False]
    df["output_var"] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    analyser = DataAnalyser(df, "output_var")
    assert(analyser.chi_squared_stat_vals(variables=["feature_True", "feature_False"]) == [0, 0])
    df["feature"] = [True, True, True, True, False, False, False, False, False, False]
    df["output_var"] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    analyser = DataAnalyser(df, "output_var")
    assert(analyser.chi_squared_stat_vals(variables=["feature_True", "feature_False"]) == [6.267361111111111,
                                                                                           6.267361111111111])

