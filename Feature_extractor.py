import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from itertools import compress

from src.utility import is_categorical


class FeatureExtractor:
    def __init__(self, df: pd.DataFrame, output_var: str, audio: bool = False):
        """
        Stores the dataframe as a data member
        :param df: pd.DataFrame containing data
        :param output_var: String, name of the output variable to be modelled
        """
        # Want to store a deep copy as we are modifying the dataframe in place
        self.df = df.copy()
        self.output_var = output_var
        self.le = LabelEncoder()
        self.le.fit(self.df[self.output_var])

        # Pre-process df
        self.pre_process(audio=audio)

    def pre_process(self, audio: bool = False) -> None:
        """
        Normalises all continuous variables in self.df and adds one-hot encoding for each categorical
        variable in self.df
        :param audio: bool (default=False) if true, assumes that all independent variables in self.df
        are audio features, and so are continuous with certainty.
        :return: None, normalisation is done in place
        """
        features = list(self.df.columns)
        for feature in features:
            # Don't want to process the output variable
            if feature == self.output_var:
                continue
            if audio:
                self.normalize_var(feature)
            else:
                if is_categorical(self.df, feature):
                    self.one_hot_encoding(feature)
                else:
                    self.normalize_var(feature)

    def one_hot_encoding(self, var_name: str) -> None:
        """
        Performs one-hot encoding of the variable called var_name in self.df
        :param var_name: string, name of variable to encode
        :return: None - modifies self.df
        """
        dummies = pd.get_dummies(self.df[var_name], drop_first=False)
        dummies.columns = [f"{var_name}_{cat}" for cat in dummies.columns]
        self.df.drop(var_name, axis=1, inplace=True)
        self.df = pd.concat([self.df, dummies], axis=1)

    def normalize_var(self, var_name: str) -> None:
        """
        Normalizes the variable var_name in self.df in-place
        :param var_name: string, name of the variable to normalize
        :return: None, normalization is done in-place
        """
        normalizer = StandardScaler()
        data = np.array(self.df[var_name])
        data = data.reshape(-1, 1)
        normal_data = normalizer.fit_transform(data)
        self.df[var_name] = normal_data

    def lasso_features(self, no_of_features: int = 20) -> dict:
        """
        Extracts the most relevant features using Lasso regression
        :param no_of_features: Int, (default=20), maximum number of features to return
        :param output_var: String, name of the output variable to be modelled
        :return: dict of features:coeff of most relevant features
        """

        indep_vars = self.df.drop([self.output_var], axis=1)
        indep_names = list(indep_vars.columns)
        lasso_model = linear_model.Lasso()
        fitted_lasso = lasso_model.fit(indep_vars, self.df[self.output_var])
        return self.get_relevant_features(fitted_lasso, indep_names, no_of_features)

    def get_relevant_features(self, model, feature_names, no_of_features: int = 100) -> dict:
        """
        Extracts the relevant features from a pre-fit sklearn model
        :param feature_names: list of all feature names used to train the model
        :param model: estimator object containing pre-fit model
        :param no_of_features: Int, (default=100), maximum number of features to return
        :return: dict {feature_name: feature_coefficient}
        """

        features = SelectFromModel(model, prefit=True, threshold="median", max_features=no_of_features)
        # relevant_features = list(compress(feature_names, list(features.get_support())))
        relevant_features = list(feature_names)

        try:
            coeff = model.feature_importances_
        except AttributeError:
            coeff = model.coef_

        """
        Linear SVC returns the feature coefficients for each class
        To account for this, coefficients for each class are scaled according
        to the proportion of observations each class accounts for. 
        """

        if isinstance(model, LinearSVC):
            categories = model.classes_
            if len(categories) > 2:
                weights = []

                for cat in categories:
                    weights.append(self.df[self.df[self.output_var] == cat].shape[0])

                observations = sum(weights)
                weights = [w / observations for w in weights]

                for index, _ in enumerate(coeff):
                    coeff[index] = abs(coeff[index]) * weights[index]
                coeff = np.sum(coeff, axis=0)

            else:
                coeff = coeff.flatten()


        features = model.feature_names_in_
        feature_coeff = dict(zip(features, coeff))
        relevant_fc = {f: c for f, c in feature_coeff.items() if f in relevant_features}

        return relevant_fc

    def elastic_net_features(self, no_of_features: int = 20) -> dict:
        """
        Extracts the most relevant features using the elastic net regression
        :param no_of_features: Int, (default=20), maximum number of features to return
        :return: dict of features:coeff of most relevant features
        """

        indep_vars = self.df.drop([self.output_var], axis=1)
        indep_names = list(indep_vars.columns)
        elastic_net = linear_model.ElasticNet()
        fitted_net = elastic_net.fit(indep_vars, self.df[self.output_var])
        return self.get_relevant_features(fitted_net, indep_names, no_of_features)

    def svm_rfe(self, no_of_features: int = 10) -> list:
        """
        Extracts the relevant features using SVM-RFE algorithm
        :param no_of_features: Int, (default=20), maximum number of features to return
        :return: list of strings containing the most relevant feature names
        """

        indep_vars = self.df.drop([self.output_var], axis=1)
        indep_names = list(indep_vars.columns)

        # From documentation, if n_samples > n_features, dual should be False
        if len(indep_vars) >= self.df.shape[0]:
            dual = True
        else:
            dual = False

        svc = LinearSVC(dual=dual)
        rfe_selector = RFE(svc, n_features_to_select=no_of_features, verbose=100)
        fitted_rfe = rfe_selector.fit(indep_vars, self.df[self.output_var])
        return list(compress(indep_names, fitted_rfe.support_))

    def linear_svc_features(self, no_of_features: int = 20) -> dict:
        """
        Extracts the most relevant features according to a Support Vector Classification model
        :param no_of_features: Int, (default=20), maximum number of features to return
        :return: dict of features: coeff of most relevant features
        """
        indep_vars = self.df.drop([self.output_var], axis=1)
        indep_names = list(indep_vars.columns)

        # From documentation, if n_samples > n_features, dual should be False
        if len(indep_vars) >= self.df.shape[0]:
            dual = True
        else:
            dual = False

        svc = LinearSVC(dual=dual)
        svc_fit = svc.fit(indep_vars, self.df[self.output_var])
        return self.get_relevant_features(svc_fit, indep_names, no_of_features)

    def random_forest_features(self, no_of_features: int = 20) -> dict:
        """
        Extracts the most relevant features according to a random forest model
        :param no_of_features: Int, (default=20), maximum number of features to return
        :return: dict of features:coeff of most relevant features
        """
        indep_vars = self.df.drop([self.output_var], axis=1)
        indep_names = list(indep_vars.columns)
        rf_estimator = RandomForestClassifier()
        rf_estimator_fit = rf_estimator.fit(indep_vars, self.df[self.output_var])
        return self.get_relevant_features(rf_estimator_fit, indep_names, no_of_features)

    def xgboost_features(self):

        x = self.df.drop([self.output_var], axis=1)
        y = self.df[self.output_var]
        y = self.le.transform(y)
        column_encoder = LabelEncoder()
        original_columns = list(x.columns).copy()
        x.columns = list(column_encoder.fit_transform(original_columns))
        x.columns = [str(x) for x in x.columns]
        dmatrix = xgb.DMatrix(data=x, label=y)

        no_classes = len(set(y))
        if no_classes == 2:
            objective = "binary:logistic"
            params = {"max_depth": 6, "objective": objective}
        else:
            objective = "multi:softmax"
            params = {"max_depth": 6, "objective": objective, "num_class": no_classes}

        bst = xgb.train(params, dmatrix)
        feature_scores = bst.get_score(importance_type="gain")

        coeff = list(feature_scores.values())
        features = list(feature_scores.keys())
        features = [int(x) for x in features]
        features = list(column_encoder.inverse_transform(features))

        zero_features = []
        for feature in original_columns:
            if feature not in features:
                zero_features.append(feature)

        features.extend(zero_features)
        zeros = [0]*len(zero_features)
        coeff.extend(zeros)

        feature_scores = dict(zip(features, coeff))
        return feature_scores

    def get_regression_features(self, no_of_features: int = 20) -> tuple:
        """
        Gets regression (lasso & ElasticNet) most important features
        :param no_of_features: Int, (default=20), maximum number of features to return
        :return: tuple[dict, dict] == tuple[lasso features: lasso coeff, elastic features: elastic coeff]
        """
        return (self.lasso_features(no_of_features),
                self.elastic_net_features(no_of_features))

    def get_classification_features(self, no_of_features: int = 20) -> tuple:
        """
        Gets classification (svm_rfe & random_forest) most important features
        :param no_of_features: Int, (default=20), maximum number of features to return
        :return: tuple[dict, dict] == tuple[rf features: rf coeff, svc features: svc coeff]
        """
        return (self.xgboost_features(),
                self.linear_svc_features(no_of_features))

    def extract_features(self, no_of_features: int = 20) -> pd.DataFrame:
        """
        Extracts the most relevant features (as specified by no_of_features) and returns
        them as a pd.Dataframe
        :param no_of_features: option int (default=20) maximum number of features to extract
        :return: pd.DataFrame containing relevant features according to method used
        """

        df = pd.DataFrame()
        df2 = pd.DataFrame()

        if is_categorical(self.df, self.output_var):
            rf, svc = self.get_classification_features(no_of_features)
            df["RF Features"] = list(rf.keys())
            df["RF Gini"] = [round(val, 4) for val in list(rf.values())]
            df = df.sort_values(by=["RF Gini"], ascending=False)
            df2["Linear SVC Features"] = list(svc.keys())
            df2["SVC Coeff"] = [round(val, 4) for val in list(svc.values())]
            df2 = df2.sort_values(by=["SVC Coeff"], ascending=False)
        else:
            lasso, elastic = self.get_regression_features(no_of_features)
            df["Lasso Features"] = list(lasso.keys())
            df["Lasso Coeff"] = [round(val, 4) for val in list(lasso.values())]
            df = df.sort_values(by=["Lasso Coeff"], ascending=False)
            df2["Elastic Net Features"] = list(elastic.keys())
            df2["EN Coeff"] = [round(val, 4) for val in list(elastic.values())]
            df2 = df2.sort_values(by=["EN Coeff"], ascending=False)

        df.reset_index(inplace=True, drop=True)
        df2.reset_index(inplace=True, drop=True)
        df = pd.concat([df, df2], axis=1)
        return df

    def tsne(self):
        """
        Calculates TSNE of dataset features
        :return: np.Array
        """
        tsnse = TSNE(learning_rate="auto")
        return tsnse.fit_transform(self.df.drop([self.output_var], axis=1))


