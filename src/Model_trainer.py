import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from math import sqrt

from utility import is_categorical


class ModelTrainer:
    def __init__(self, main_df: pd.DataFrame, output_var: str, test_df: pd.DataFrame = None, audio: bool = False):
        """

        :param main_df: Dataframe either containing all data, or train data, in the case that test_df is None
        :param output_var: Str, name of the output var
        :param test_df: Optional, df containing the test data. If None, main_df is shuffled and split according to
        test/train split
        :param audio: bool (default=False). Specifies whether the dataframes contain audio features
        """

        self.output_var = output_var
        self.audio = audio
        self.le = LabelEncoder()

        if test_df is not None:
            self.x_train = main_df.drop([self.output_var], axis=1)
            self.y_train = main_df[self.output_var]
            self.x_test = test_df.drop([self.output_var], axis=1)
            self.y_test = test_df[self.output_var]
        else:
            x = main_df.drop([self.output_var], axis=1)
            y = main_df[self.output_var]
            self.x_train, self.x_test, self.y_train, self.y_test = \
                train_test_split(x, y, test_size=0.2)

        # Preprocessing encoders
        self.standardiser = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
        self.le.fit(self.y_train)

        # Variables used to store trained models
        self.trained_lasso = None
        self.trained_en = None
        self.trained_rf = None
        self.trained_svc = None
        self.trained_pca = None
        self.bst_y_hat = None
        self.svc_y_hat = None
        self.lasso_y_hat = None
        self.en_y_hat = None
        self.bst = None

        # Special train/test data for PCA, as PCA should only contain
        # Continuous variables
        self.pca_train = None
        self.pca_test = None

        self.pre_process()

    def pre_process(self):
        """
        Pre-processes train and test data. Normalises continuous variables and one-hot-encodes
        categorical variables
        :return:
        """

        if self.audio:
            self.x_train = self.standardiser.fit_transform(self.x_train)
            self.x_test = self.standardiser.transform(self.x_test)
            self.pca_train = self.x_train
            self.pca_test = self.x_test
            return

        else:
            # Get boolean mask of categorical/continuous variables
            categorical_bools = [is_categorical(self.x_train, var) for var in self.x_train.columns]
            continuous_bools = [not cat for cat in categorical_bools]

            cat = sum(categorical_bools) > 0
            cont = sum(continuous_bools) > 0

            if cont:
                cont_train_df = self.x_train.loc[:, continuous_bools]
                cont_test_df = self.x_test.loc[:, continuous_bools]
                cont_train = self.standardiser.fit_transform(cont_train_df)
                cont_test = self.standardiser.transform(cont_test_df)
                self.pca_train = cont_train
                self.pca_test = cont_test

            if cat:
                cat_train_df = self.x_train.loc[:, categorical_bools]
                cat_test_df = self.x_test.loc[:, categorical_bools]
                cat_train = self.one_hot_encoder.fit_transform(cat_train_df).toarray()
                cat_test = self.one_hot_encoder.transform(cat_test_df).toarray()

            if cont and cat:
                self.x_train = np.concatenate((cont_train, cat_train), axis=1)
                self.x_test = np.concatenate((cont_test, cat_test), axis=1)
            elif cont:
                self.x_train = cont_train
                self.x_test = cont_test
            else:
                self.x_train = cat_train
                self.x_test = cat_test

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
        lasso_model = linear_model.Lasso(max_iter=10000, fit_intercept=False)
        trained_lasso = lasso_model.fit(self.x_train, self.y_train)
        self.trained_lasso = trained_lasso
        self.lasso_y_hat = trained_lasso.predict(self.x_test)
        return trained_lasso

    def train_elastic_net(self) -> object:
        """
        Trains an elastic net regression model
        :return: Trained elastic net model
        """
        elastic_net = linear_model.ElasticNet(max_iter=10000, fit_intercept=False)
        trained_net = elastic_net.fit(self.x_train, self.y_train)
        self.trained_en = trained_net
        self.en_y_hat = trained_net.predict(self.x_test)
        return trained_net

    def train_eval_xgb(self):
        """

        :return:
        """
        if self.bst_y_hat is None:
            x = self.x_train
            y = self.y_train
            y = self.le.fit_transform(y)
            dmatrix_train = xgb.DMatrix(data=x, label=y)
            dmatrix_test = xgb.DMatrix(data=self.x_test)
            no_classes = len(set(y))
            if no_classes == 2:
                objective = "binary:logistic"
                params = {"max_depth": 6, "objective": objective}
            else:
                objective = "multi:softmax"
                params = {"max_depth": 6, "objective": objective, "num_class": no_classes}

            self.bst = xgb.train(params, dmatrix_train)
            y_hat = self.bst.predict(dmatrix_test)
            y_hat = [round(y) for y in y_hat]
            self.bst_y_hat = self.le.inverse_transform(y_hat)

        return recall_score(self.y_test, self.bst_y_hat, average="macro", zero_division=0)

    def train_pca(self, no_of_components: int):
        """
        Trains a PCA model on self.pca_train
        :param no_of_components: Number of principal components
        :return: Instance of trained PCA
        """
        if no_of_components > min(self.pca_train.shape[0], self.pca_train.shape[1]):
            raise ValueError("Error, no. of PCA components must be <= min(no_features, no_samples)")

        pca = PCA(n_components=no_of_components)
        self.trained_pca = pca.fit(self.pca_train)
        return self.trained_pca

    def evaluate_class_model(self, trained_model: object) -> float:
        """
        Evaluates a trained classification model
        :param trained_model: Instance of a trained model
        :return: Float, Unweighted Average Recall (UAR) on test data
        """

        y_hat = trained_model.predict(self.x_test)
        self.svc_y_hat = y_hat
        return recall_score(self.y_test, y_hat, average="macro", zero_division=0)

    def evaluate_reg_model(self, trained_model: object) -> tuple:
        """
        Evaluates a trained regression model
        :param trained_model: Instance of trained model
        :return: tuple[R2, root_mean_squared_error] on test set
        """
        y_hat = trained_model.predict(self.x_test)
        r2 = r2_score(self.y_test, y_hat)
        rmse = sqrt(mean_squared_error(self.y_test, y_hat))
        return r2, rmse

    def evaluate_classifier_models(self) -> tuple:
        """
        Trains and evaluates an RF and SVC model
        :return: tuple(float,float) class weighted f1 on test set
        """

        # rf_f1 = self.evaluate_class_model(self.train_rf())
        xboost = self.train_eval_xgb()
        svc_f1 = self.evaluate_class_model(self.train_linear_svc())
        return xboost, svc_f1

    def evaluate_regression_models(self) -> tuple:
        """
        Trains and evaluates Lasso and Elastic Net Regression models
        :return: tuple(float, float, float, float) of R^2, rmse scores on test set
        """

        lasso_r2, lasso_mse = self.evaluate_reg_model(self.train_lasso())
        elastic_r2, elastic_mse = self.evaluate_reg_model(self.train_elastic_net())
        return lasso_r2, lasso_mse, elastic_r2, elastic_mse

    def get_confusion_matrix(self, trained_model: object = None, y_pred = None) -> tuple:
        """
        Calculates the confusion matrix from a given trained model on the test set
        :param trained_model: object instance of trained classification model
        :return: np.ndarray of shape (no_of_classes, no_of_classes)
        """
        if y_pred is None:
            y_hat = trained_model.predict(self.x_test)
        else:
            y_hat = y_pred
        return confusion_matrix(self.y_test, y_hat, normalize="true"), y_hat

    def get_regression_predictions(self) -> tuple:
        """
        Gets y_hat for Elastic net and Lasso regression
        :return: tuple(list, list)
        """
        if self.trained_lasso is None or self.trained_en is None:
            raise ValueError("Error - trying to get regression predictions when models haven't been trained")

        lasso_y_hat = list(self.trained_lasso.predict(self.x_test))
        en_y_hat = list(self.trained_en.predict(self.x_test))

        return lasso_y_hat, en_y_hat

    def get_confusion_matrix_class(self):
        """

        :return:
        """
        """
        if self.trained_rf is None or self.trained_svc is None:
            raise ValueError("Error - trying to get classification predictions when models haven't been trained")
        """

        # rf_cm, rf_y_hat = self.get_confusion_matrix(self.trained_rf)
        xgb_cm, xgb_y_hat = self.get_confusion_matrix(y_pred=self.bst_y_hat)
        svc_cm, svc_y_hat = self.get_confusion_matrix(self.trained_svc)

        return xgb_cm, xgb_y_hat, svc_cm, svc_y_hat

    def evaluate_pca_model(self, reg: bool = False) -> tuple:
        """
        Trains and evaluates a series of SVM models including an increasing number of PCA components
        :param reg: bool, is task regression (default = False). If true, treats task as regression and trains
        a Lasso model instead of a SVC model.
        :return: list of UAR scores (if classification task) or rmse (if regression task)
        """

        if self.pca_train is None:
            return None, None

        start = 1
        # Enforce a limit of 200 components
        stop = min(min(self.pca_train.shape[0], self.pca_train.shape[1]) + 1, 200)
        trained_pca = self.train_pca(stop - 1)
        normaliser = StandardScaler()

        trained_pca_vals = trained_pca.transform(self.pca_train)
        test_pca_vals = trained_pca.transform(self.pca_test)

        train_pca_normal = normaliser.fit_transform(trained_pca_vals)
        test_pca_normal = normaliser.transform(test_pca_vals)

        if stop < 20:
            step = 1
        elif stop < 200:
            step = 5
        else:
            step = 10

        scores = []
        components = []
        no_of_components = start
        while no_of_components < stop:
            if no_of_components == 1:
                pca_train = train_pca_normal[:, 0:no_of_components].reshape(-1, 1)
                test_pca = test_pca_normal[:, 0:no_of_components].reshape(-1, 1)
            else:
                pca_train = train_pca_normal[:, 0:no_of_components]
                test_pca = test_pca_normal[:, 0:no_of_components]

            if reg:
                model = linear_model.Lasso(max_iter=1000, fit_intercept=False)
                trained_model = model.fit(pca_train, self.y_train)
                y_hat = trained_model.predict(test_pca)
                scores.append(sqrt(mean_squared_error(self.y_test, y_hat)))
            else:
                # From documentation, if n_samples > n_features, dual should be False
                if train_pca_normal.shape[1] >= train_pca_normal.shape[0]:
                    dual = True
                else:
                    dual = False
                model = LinearSVC(dual=dual, fit_intercept=False, max_iter=1000, C=0.0001)
                trained_model = model.fit(pca_train, self.y_train)
                y_hat = trained_model.predict(test_pca)
                scores.append(recall_score(self.y_test, y_hat, average="macro", zero_division=0))

            components.append(no_of_components)
            if no_of_components < 30:
                no_of_components += 1
            else:
                no_of_components += step

        return scores, components

