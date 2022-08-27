import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

import src.config as config


class DataAnalyser:
    def __init__(self, df: pd.DataFrame, output_var: str,
                 remove_nans: bool = False, audio: bool = False):
        """
        Stores the given dataframe and output variable as data members
        :param df: Pandas df that contains the dataset to be analysed
        :param output_var: The output variable (dependent variable) name in the df
        :param remove_nans bool (default=False) whether to remove nans from df
        """
        self.df = df.copy()
        self.output_var = output_var
        self.max_categories = config.MAX_CAT
        self.constant_vars = []
        self.number_of_nans = 0
        self.number_of_nans_df = None

        # Used for calculating MI metrics
        self.encoder = None

        if remove_nans:
            self.remove_nans_and_missing()
            self.remove_constant_vars()

        # Used for storing correlation dfs calculated later
        self.spearman = None
        self.cramers = None
        self.kruskal = None
        self.correlations = False

        self.get_dummies(audio=audio)

    def get_dummies(self, audio: bool = False):
        """

        :param audio:
        :return:
        """
        if not audio:
            for var in self.df.columns:
                if var == self.output_var:
                    continue
                if self.is_categorical(var):
                    dummies = pd.get_dummies(self.df[var], drop_first=False)
                    dummies.columns = [f"{var}_{cat}" for cat in dummies.columns]
                    self.df.drop(var, axis=1, inplace=True)
                    self.df = pd.concat([self.df, dummies], axis=1)

    def drop_variable(self, var_name: str) -> None:
        """
        Drops a variable from self.df
        :param var_name: String, name of the variable in self.df to be dropped
        :return: None
        """
        self.df = self.df.drop([var_name], axis=1)

    def remove_nans_and_missing(self) -> None:
        """
        Removes missing values and NANs from self.df
        and stores number of NANs
        :return: None
        """

        # First, convert inf and -infs with NANs
        self.df.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
        self.df.replace(to_replace=["inf", "-inf"], value=np.nan, inplace=True)
        df_nan = self.df.isna()
        self.number_of_nans_df = df_nan.sum()
        self.number_of_nans_df = pd.DataFrame(self.number_of_nans_df)
        self.number_of_nans_df.columns = ["Number of missing/NAN values"]
        self.number_of_nans = self.number_of_nans_df.sum().iloc[0]

        if self.number_of_nans == 0:
            self.number_of_nans_df = None
        else:
            self.number_of_nans_df = self.number_of_nans_df[self.number_of_nans_df["Number of missing/NAN values"] > 0]

        self.df.dropna(inplace=True)

    def remove_constant_vars(self) -> None:
        """
        Removes independent variables that have constant values from self.df
        and stores list of removed variables
        :return: None
        """
        for var in self.df.columns:
            if var == self.output_var:
                continue
            if len(list(pd.unique(self.df[var]))) == 1:
                self.constant_vars.append(var)
                self.df.drop([var], axis=1, inplace=True)

    def cross_tabulate(self, indep_var: str) -> pd.DataFrame:
        """
        Creates a cross-tabulation table between the output_var and indep var.
        Only useful when both output_var and indep_var are categorical variables
        :param indep_var: name of the independent variable to cross_tabulate with
                          the output variable
        :return: pandas dataframe containing the cross tabulation
        """
        return pd.crosstab(self.df[indep_var], self.df[self.output_var])

    def is_categorical(self, var_name: str) -> bool:
        """
        Determines whether a given variable within self.df is categorical, according
        to its number of unique values being less than cutoff
        :param var_name: Name of the variable within self.df to check for type
        :return: Bool, True if variable is deemed to be categorical, else False
        """

        data_type = self.df[var_name].dtypes
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

        rows, _ = self.df.shape
        number_unique = len(list(pd.unique(self.df[var_name])))
        percent_unique = number_unique / rows * 100

        if rows < 500:
            if percent_unique <= 10:
                return True
            return False

        if number_unique <= config.MAX_CAT:
            return True
        return False

    def bias_corrected_cramers_v(self, indep_var: str) -> float:
        """
        Calculates the bias corrected Cramer's V measure of correlation, as given
        by Bergsma https://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf, between self.output_var
        and indep_var. Measure is only valid if both variables are categorical.
        :param indep_var: variable in self.df to perform the Cramer's V measure with
        :return: float between 0 and 1. 0 = no correlation (i.e., variables are independent)
                 1 = maximum correlation (i.e., one variable can completely explain the other)
        """

        cont_table = self.cross_tabulate(indep_var)
        (chi2, p, _, _) = scipy.stats.chi2_contingency(cont_table)
        n = cont_table.to_numpy().sum()
        rows = cont_table.shape[0]
        cols = cont_table.shape[1]
        # if rows == 1 or cols == 1:
        # return "Error - constant var detected"
        phi = chi2 / n
        phi_b_adj = max(phi - ((1 / (n - 1)) * (rows - 1) * (cols - 1)), 0)
        rows_b_adj = rows - ((1 / (n - 1)) * (rows - 1) ** 2)
        cols_b_adj = cols - ((1 / (n - 1)) * (cols - 1) ** 2)
        denominator = min(rows_b_adj - 1, cols_b_adj - 1)
        return (phi_b_adj / denominator) ** 0.5

    def get_output_correlations(self, audio_file: str = None, ) -> tuple:
        """
        Calculates the relevant correlations between the output variable and all other variables
        within the dataframe. Chooses the appropriate correlation metric depending on whether the
        output variable, and given independent variable, is categorical of continuous
        :param audio_file (str, default = None), name of the column containing audio files that should be ignored
        :return: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

        """
        """
        First, determine whether the variables are categorical
        or continuous
        """

        if self.correlations:
            return self.spearman, self.cramers, self.kruskal

        if self.is_categorical(self.output_var):
            out_type = "cat"
        else:
            out_type = "cont"

        variables = list(self.df.columns)
        spearman = {}
        cramers = {}
        kruskal = {}
        for var in variables:
            if var == self.output_var:
                continue
            if var == audio_file:
                continue
            if self.is_categorical(var):
                if out_type == "cat":
                    cramers[var] = self.bias_corrected_cramers_v(var)
                elif out_type == "cont":
                    kruskal[var] = self.kruskal_wallis_h(var)
            else:
                if out_type == "cat":
                    kruskal[var] = self.kruskal_wallis_h(var, categorical="dep")
                elif out_type == "cont":
                    spearman_stat, p_value = self.spearman_correlation(var)
                    dist_from_max = min(abs(1 - spearman_stat), abs(-1 - spearman_stat))
                    spearman[var] = [spearman_stat, p_value, dist_from_max]

        spearman = pd.DataFrame(spearman, index=["Spearman Correlation Coefficient", "P-value", "Dist from max"]).T. \
            sort_values(by=["Dist from max"], axis=0).drop("Dist from max", axis=1)
        cramers = pd.DataFrame(cramers, index=["Cramer's V"]).T.sort_values(by=["Cramer's V"], axis=0, ascending=False)
        kruskal = pd.DataFrame(kruskal, index=["Kruskal-Wallis Stat", "P-value"]).T.sort_values(by=["P-value"],
                                                                                                axis=0)

        self.spearman = spearman
        self.cramers = cramers
        self.kruskal = kruskal
        self.correlations = True

        return spearman, cramers, kruskal

    def kruskal_wallis_h(self, indep_var: str, categorical: str = "indep") -> tuple:
        """
        Calculates the kruskal and Wallis statistics for determining whether the median
        value of a continuous variable is different according to different given categories
        :param indep_var: The name of the independent variable to perform the calculation with
        :param categorical: (default = "indep") Which of the two variables (self.output_var or
                            indep_var) is the categorical variable
        :return: Tuple (statistics value, p-value)
        """
        """ https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html """
        """ https://www.jstor.org/stable/2280779 """

        if categorical == "indep":
            cat_var = indep_var
            cont_var = self.output_var
        else:
            cat_var = self.output_var
            cont_var = indep_var

        if len(set(self.df[cont_var])) == 1 or len(set(self.df[cat_var])) == 1:
            return 0, 1
        categories = list(pd.unique(self.df[cat_var]))
        category_values = []
        for category in categories:
            values = self.df[self.df[cat_var] == category]
            category_values.append(list(values[cont_var]))

        vals = scipy.stats.kruskal(*category_values)
        return vals

    def spearman_correlation(self, indep_var: str) -> tuple:
        """
        Calculates the Spearman Correlation coefficient between two continuous variables
        :param indep_var: Name of the variable in self.df to perform this calculation with self.output_var
        :return: tuple (correlation value, p-value)
        """
        """ https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html"""
        return scipy.stats.spearmanr(self.df[self.output_var], self.df[indep_var])

    def add_to_df(self, var_name: str, var_values: list) -> None:
        """
        Adds a variable to self.df

        :param var_name: String - variable (column) name
        :param var_values: List - variable values
        :return: None
        """
        rows = self.df.shape[0]
        assert len(var_values) == rows
        self.df[var_name] = var_values

    def mutual_information(self, audio: bool = False) -> pd.DataFrame:
        """
        Calculates mutual information

        Makes use of the sklearn functionality
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
        :return: pd.Dataframe, MI values between independent and dependent variable
        """

        x = self.df.drop([self.output_var], axis=1)
        y = self.df[self.output_var]
        var_names = list(self.df.drop([self.output_var], axis=1))

        if self.is_categorical(self.output_var):
            mutual_information = mutual_info_classif(x, y)
        else:
            mutual_information = mutual_info_regression(x, y)

        mi_df = pd.DataFrame(mutual_information, index=var_names, columns=["Mutual Information"])

        if audio:
            for var in mi_df.index:
                if len(pd.unique(self.df[var])) == 1:
                    mi_df.at[var, "Mutual Information"] = 0

        mi_df = mi_df.sort_values(by=["Mutual Information"], axis=0, ascending=False)

        return mi_df

    def spearman_r_values(self, variables: list):
        """
        Calculates the Spearman correlation statistic for a list of variables
        :param variables: list of variable names
        :return: list
        """
        if len(variables) == 0:
            return []

        corr = []
        for variable in variables:
            r, p = self.spearman_correlation(variable)
            corr.append(abs(r))
        return corr

    def chi_squared_stat_vals(self, variables: list):
        """
        Calculates the Chi-Squared statistic for a list of variables
        :param variables: list of variable names
        :return: list
        """
        if len(variables) == 0:
            return []

        stats = []
        for variable in variables:
            cont_table = self.cross_tabulate(variable)
            (chi2, p, _, _) = scipy.stats.chi2_contingency(cont_table)
            stats.append(chi2)
        return stats

    def kw_stat_vals(self, variables: list, audio: bool = False):
        """
        Calculates the Kruskal-Wallis statistic for a list of variables
        :param audio: bool (default=False) whether the variables are for audio or tabular data
        :param variables: list of variable names
        :return: list
        """
        if len(variables) == 0:
            return []

        stats = []
        for variable in variables:
            if audio:
                cat = "output"
            elif self.is_categorical(variable):
                cat = "indep"
            else:
                cat = "output"
            stat, p = self.kruskal_wallis_h(indep_var=variable, categorical=cat)
            stats.append(stat)
        return stats

    def point_biserial_correlation(self, var_list: list):
        """
        Calculates the point biserial correlation between a list of variables and the output var
        :param var_list: list of variable names
        :return: list
        """

        if len(var_list) == 0:
            return []

        corr_list = []

        for variable in var_list:
            r, p_val = stats.pointbiserialr(self.df[variable], self.df[self.output_var])
            corr_list.append(abs(r))
        return corr_list
