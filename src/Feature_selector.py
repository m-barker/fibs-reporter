import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from itertools import compress


class FeatureExtractor:
    def __init__(self, df: pd.DataFrame, output_var: str):
        """
        Stores the dataframe as a data member
        :param df: pd.DataFrame containing data
        :param output_var: String, name of the output variable to be modelled
        """
        # Want to store a deep copy as we are modifying the dataframe in place
        self.df = df.copy()
        self.output_var = output_var

        # Variables used if training and evaluating a model
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.split = False

        # Variables used to store trained models
        self.trained_lasso = None
        self.trained_en = None
        self.trained_rf = None
        self.trained_svc = None
        self.trained_pca = None

    def one_hot_encoding(self, var_name: str) -> None:
        """
        Performs one-hot encoding of the variable called var_name in self.df
        :param var_name: string, name of variable to encode
        :return: None - modifies self.df
        """
        dummies = pd.get_dummies(self.df[var_name], drop_first=True)
        dummies.columns = [f"{var_name}_{cat}" for cat in dummies.columns]
        self.df.drop(var_name, axis=1, inplace=True)
        self.df = pd.concat([self.df, dummies], axis=1)

    def min_max_var(self, var_name: str) -> None:
        """
        Scales the variable var_name in self.df in-place
        :param var_name: string, name of the variable to scale
        :return: None, scaling is done in-place
        """
        min_max_scaler = MinMaxScaler()
        data = np.array(self.df[var_name])
        data = data.reshape(-1, 1)
        min_max_scaler.fit(data)
        scaled_data = min_max_scaler.transform(data)
        self.df[var_name] = list(scaled_data)

    def normalize_var(self, var_name: str) -> None:
        """
        Normalizes the variable var_name in self.df in-place
        :param var_name: string, name of the variable to normalize
        :return: None, normalization is done in-place
        """
        normalizer = StandardScaler()
        data = np.array(self.df[var_name])
        data = data.reshape(-1, 1)
        normalizer.fit(data)
        normal_data = normalizer.transform(data)
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

    def get_relevant_features(self, model, feature_names, no_of_features: int = 20,
                              ) -> dict:
        """
        Extracts the relevant features from a pre-fit sklearn model
        :param feature_names: list of all feature names used to train the model
        :param model: estimator object containing pre-fit model
        :param no_of_features: Int, (default=20), maximum number of features to return
        :return: dict {feature_name: feature_coefficient}
        """
        features = SelectFromModel(model, prefit=True, threshold="median", max_features=no_of_features)
        relevant_features = list(compress(feature_names, list(features.get_support())))

        try:
            coeff = model.feature_importances_
        except AttributeError:
            coeff = model.coef_

        if isinstance(model, LinearSVC):
            avg_coeff = []
            categories = model.classes_
            weights = []

            for cat in categories:
                weights.append(self.df[self.df[self.output_var] == cat].shape[0])

            observations = sum(weights)
            weights = [w / observations for w in weights]

            for index, _ in enumerate(coeff):
                coeff[index] = coeff[index] * weights[index]

            coeff = np.sum(coeff, axis=0)

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

    def pca(self, components: int = 2, data: str = "all", reset=False) -> list:
        """
        Extracts the principal components within the dataset
        :param components: optional int (default=20), number of principal components to compute
        :return: ndarray of shape (n_samples, n_components)
        """

        if data == "all":
            indep_vars = self.df.drop([self.output_var], axis=1)
        elif data == "train":
            indep_vars = self.x_train
        else:
            # Can't transform the test set if no trained model
            if self.trained_pca is None:
                raise ValueError("PCA error - can't transform test set before a model has been trained")
            indep_vars = self.x_test

        if self.trained_pca is None or reset:
            pca_model = PCA(n_components=components)
            self.trained_pca = pca_model.fit(indep_vars)

        data = self.trained_pca.transform(indep_vars)
        return data

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
        # return (self.svm_rfe(output_var, no_of_features),
        #       self.random_forest_features(output_var, no_of_features))
        return (self.random_forest_features(no_of_features),
                self.linear_svc_features(no_of_features))

    def get_train_test_split(self, test_prop: int = 0.2) -> None:
        """
        Generates train/test splits
        :param test_prop: int (default=0.2). Proportion of the data to test
        :return: None
        """

        x = self.df.drop([self.output_var], axis=1)
        y = self.df[self.output_var]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_prop)
        self.split = True

    def train_rf(self) -> object:
        """
        Trains a random forest model
        :return: Trained RandomForest Classifier
        """
        rf_estimator = RandomForestClassifier()
        rf_trained = rf_estimator.fit(self.x_train, self.y_train)
        self.trained_rf = rf_trained
        return rf_trained

    def train_linear_svc(self) -> object:
        """
        Trains a linear support vector classifier
        :return: Trained svc model
        """
        # From documentation, if n_samples > n_features, dual should be False
        if self.x_train.shape[1] >= self.x_train.shape[0]:
            dual = True
        else:
            dual = False
        svc = LinearSVC(dual=dual, fit_intercept=False, max_iter=10000)
        svc_trained = svc.fit(self.x_train, self.y_train)
        self.trained_svc = svc_trained
        return svc_trained

    def train_lasso(self) -> object:
        """
        Trains a lasso regression model
        :return: Trained lasso model
        """
        lasso_model = linear_model.Lasso()
        trained_lasso = lasso_model.fit(self.x_train, self.y_train)
        self.trained_lasso = trained_lasso
        return trained_lasso

    def train_elastic_net(self) -> object:
        """
        Trains an elastic net regression model
        :return: Trained elastic net model
        """
        elastic_net = linear_model.ElasticNet()
        trained_net = elastic_net.fit(self.x_train, self.y_train)
        self.trained_en = trained_net
        return trained_net

    def evaluate_model(self, trained_model: object) -> float:
        """
        Evaluates a trained model
        :param trained_model: Instance of a trained model
        :return: Float, accuracy (or R^2 if regression) on test data
        """

        y_hat = trained_model.predict(self.x_test)
        return recall_score(self.y_test, y_hat, average="macro")

    def evaluate_classifier_models(self) -> tuple:
        """
        Trains and evaluates an RF and SVC model
        :return: tuple(float,float) class weighted f1 on test set
        """
        if not self.split:
            self.get_train_test_split()

        y_train_mode = self.y_train.mode().iloc[0]
        most_freq_total = len(self.y_train.loc[self.y_train == y_train_mode])
        most_freq_prop = most_freq_total / self.y_train.shape[0]

        rf_f1 = self.evaluate_model(self.train_rf())
        svc_f1 = self.evaluate_model(self.train_linear_svc())
        return rf_f1, svc_f1

    def evaluate_regression_models(self) -> tuple:
        """
        Trains and evaluates Lasso and Elastic Net Regression models
        :return: tuple(float, float) of R^2 scores on test set
        """
        if not self.split:
            self.get_train_test_split()

        lasso_r2 = self.evaluate_model(self.train_lasso())
        elastic_r2 = self.evaluate_model(self.train_elastic_net())
        return lasso_r2, elastic_r2

    def get_confusion_matrix(self, trained_model: object) -> np.ndarray:
        """
        Calculates the confusion matrix from a given trained model on the test set
        :param trained_model: object instance of trained classification model
        :return: np.ndarray of shape (no_of_classes, no_of_classes)
        """
        y_hat = trained_model.predict(self.x_test)
        return confusion_matrix(self.y_test, y_hat, normalize="true")

