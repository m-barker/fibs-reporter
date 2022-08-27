import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Data_analyser import DataAnalyser
from Audio_processor import AudioProcessor
from Feature_extractor import FeatureExtractor
from Model_trainer import ModelTrainer

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from random import randint
from statistics import mode
from math import sqrt

from utility import is_categorical
from utility import remove_constant_vars
from utility import remove_nans_and_missing


class DataBuilder:
    """
    Generates the relevant data to be used in the report
    """

    def __init__(self, df: pd.DataFrame, output_var, audio_col: str = None,
                 test_df: pd.DataFrame = None):
        """

        :param df: dataframe containing features
        :param output_var: str, name of the output variable
        :param audio_col: Optional str name of the audio column
        :param test_df: Optional dataframe objecting containing separate test data
        """

        self.output_var = output_var
        self.no_of_nans_df = remove_nans_and_missing(df)
        self.constant_vars = remove_constant_vars(df, output_var)

        if is_categorical(df, output_var):
            self.task = "classification"
        else:
            self.task = "regression"

        if test_df is not None:
            remove_nans_and_missing(test_df)
            if len(self.constant_vars) > 0:
                for variable in self.constant_vars:
                    test_df = test_df.drop([variable], axis=1)

        # Variables for storing trained model feature dfs
        self.audio_model_features = None
        self.tabular_model_features = None
        self.sorted_audio_features = None
        self.sorted_tabular_features = None
        self.audio_trained = False
        self.tabular_trained = False

        # Analyser used if audio data is provided
        self.audio_analyser = None
        self.audio_modeller = None

        if audio_col is not None:
            self.audio_processor = AudioProcessor(df, output_var, audio_col)
            self.audio_features = self.audio_processor.audio_features
            self.audio_analyser = DataAnalyser(self.audio_features, output_var, remove_nans=False, audio=True)
            self.audio_fe = FeatureExtractor(self.audio_features, output_var, audio=True)
            self.tabular_df = df.drop([audio_col], axis=1)
            self.test_audio_features = None
            self.tabular_test_df = None

            if test_df is not None:
                self.audio_test_processor = AudioProcessor(test_df, output_var, audio_col)
                self.test_audio_features = self.audio_test_processor.audio_features
                self.tabular_test_df = test_df.drop([audio_col], axis=1)

            self.audio_modeller = ModelTrainer(self.audio_features, output_var, self.test_audio_features,
                                               audio=True)

        else:
            self.tabular_df = df
            self.tabular_test_df = test_df

        # If there is at least one non-output, non-audio variable
        if len(self.tabular_df.columns) > 1:
            self.tabular_modeller = ModelTrainer(self.tabular_df, output_var, self.tabular_test_df)
            self.tabular_fe = FeatureExtractor(self.tabular_df, output_var)
            self.tabular_analyser = DataAnalyser(self.tabular_df, output_var)
            self.get_feature_ranking()
        else:
            self.tabular_modeller = None
            self.tabular_fe = None
            self.tabular_analyser = None

        if audio_col is not None:
            self.get_feature_ranking(audio=True)

    def get_top_features_by_type(self, categorical=True, top_x: int = 10, audio: bool = False):
        """
        Get the names of the categorical or continuous variables that appear in the top_x of feature correlations
        :param categorical: bool (default=True) Whether to extract categorical or continuous feature names
        :param top_x: int (default=10), top number of variables to consider
        :param audio: bool (default=False), whether to consider audio or tabular features
        :return: List
        """
        # All audio features are floats and so are not categorical
        if audio:
            if self.sorted_audio_features is None:
                raise ValueError("Error - attempting to get audio features when model features have not"
                                 " yet been extracted")
            if categorical:
                return []
            return self.sorted_audio_features.head(top_x)["Feature Name"]
        else:
            if self.sorted_tabular_features is None:
                raise ValueError("Error - attempting to get tabular features when model features have not"
                                 " yet been extracted")
            features = self.sorted_tabular_features.head(top_x)
            cat_features = []
            cont_features = []
            for v in features["Feature Name"]:
                if is_categorical(self.tabular_fe.df, var_name=v):
                    cat_features.append(v)
                else:
                    cont_features.append(v)
            if categorical:
                return cat_features
            else:
                return cont_features

    def get_stat_feature_ranking(self, audio: bool = False):
        """
        Gets the statistical test feature ranking metric
        :param audio: bool (default = False), whether the data is tabular or audio
        :return: dataframe of sorted ranking
        """
        if audio:
            analyser = self.audio_analyser
        else:
            analyser = self.tabular_analyser

        cont_vars = []
        cat_vars = []
        for variable in analyser.df.columns:
            if variable == self.output_var:
                continue
            elif audio:
                cont_vars.append(variable)
            elif is_categorical(analyser.df, variable):
                cat_vars.append(variable)
            else:
                cont_vars.append(variable)

        if self.task == "classification":
            if audio:
                metrics = analyser.kw_stat_vals(cont_vars, audio=True)
                var_names = cont_vars
            else:
                metrics = analyser.kw_stat_vals(cont_vars)
                var_names = cont_vars
                metrics.extend(analyser.chi_squared_stat_vals(cat_vars))
                var_names.extend(cat_vars)
        else:
            if audio:
                metrics = analyser.spearman_r_values(cont_vars)
                var_names = cont_vars
            else:
                metrics = analyser.spearman_r_values(cont_vars)
                var_names = cont_vars
                metrics.extend(analyser.point_biserial_correlation(cat_vars))
                var_names.extend(cat_vars)

        df = pd.DataFrame(data=metrics, index=var_names, columns=["Stat_tests"])
        return df.sort_values(by=["Stat_tests"], axis=0, ascending=False)

    def get_feature_ranking(self, top_x: int = 10, no_of_features: int = 100, audio: bool = False) -> pd.DataFrame:
        """
        Get the top x features and their scaled relevance, according to
        mutual information criterion, RF model, and Linear SVC model.
        :param audio: Bool (default = False), whether to extract tabular or audio features
        :param no_of_features: int (default = 1000), number of features to consider
        :param top_x: int (default = 10), number of features to return
        :return: pd.Dataframe with columns [MI, XGBoost, Linear SVC, Average]
        """
        if audio:
            if self.audio_trained:
                return self.sorted_audio_features.head(top_x)
            print("Extracting most important audio features....")
            mutual_info = self.audio_analyser.mutual_information(audio=True)
            stat_features = self.get_stat_feature_ranking(audio=True)
            model_features = self.audio_fe.extract_features(no_of_features=no_of_features)
            self.audio_model_features = model_features
            print("Done!")
        else:
            if self.tabular_trained:
                return self.sorted_tabular_features.head(top_x)
            print("Extracting most important tabular features...")
            mutual_info = self.tabular_analyser.mutual_information()
            stat_features = self.get_stat_feature_ranking()
            model_features = self.tabular_fe.extract_features(no_of_features=no_of_features)
            self.tabular_model_features = model_features
            print("Done!")

        model_1_df = model_features.iloc[:, :2]
        model_2_df = model_features.iloc[:, 2:]

        model_1_df.iloc[:, 1] = [abs(x) for x in model_1_df.iloc[:, 1]]
        model_2_df.iloc[:, 1] = [abs(x) for x in model_2_df.iloc[:, 1]]

        mutual_info_features = list(mutual_info.index)
        stat_names = list(stat_features.index)
        model_1_features = list(model_1_df.iloc[:, 0])
        model_2_features = list(model_2_df.iloc[:, 0])

        common_features = set(mutual_info_features).intersection(set(model_1_features)). \
            intersection(set(model_2_features).intersection(set(stat_names)))

        mutual_info = mutual_info[mutual_info.index.isin(common_features)]
        stat_features = stat_features[stat_features.index.isin(common_features)]
        model_1_df = model_1_df[model_1_df.iloc[:, 0].isin(common_features)]
        model_2_df = model_2_df[model_2_df.iloc[:, 0].isin(common_features)]

        model_1_labels = list(model_1_df.columns)
        model_2_labels = list(model_2_df.columns)

        # Combine dataframes into one
        mutual_info.sort_index(inplace=True)
        stat_features.sort_index(inplace=True)
        model_1_df.sort_values(by=[model_1_labels[0]], inplace=True)
        model_2_df.sort_values(by=[model_2_labels[0]], inplace=True)

        features_df = pd.DataFrame()
        features_df["MI"] = list(mutual_info["Mutual Information"])
        features_df["Stat"] = list(stat_features["Stat_tests"])
        features_df[model_1_labels[0]] = list(model_1_df[model_1_labels[1]])
        features_df[model_2_labels[0]] = list(model_2_df[model_2_labels[1]])

        # Scale data between 0 and 1 for each metric
        scaler = MinMaxScaler()
        features_df[["MI", "Stat", model_1_labels[0], model_2_labels[0]]] = scaler.fit_transform(features_df)

        # Compute the average of each feature's scaled importance
        features_df["Average"] = list(features_df.mean(axis=1))

        # Add the feature names to the df
        features_df["Feature Name"] = list(model_1_df[model_1_labels[0]])

        # Finally, sort by this average
        features_df.sort_values(by=["Average"], inplace=True, ascending=False)

        labels = ["Feature Name", "MI", "Stat", model_1_labels[0], model_2_labels[0], "Average"]

        features_df = features_df[labels]

        if audio:
            self.sorted_audio_features = features_df
            self.audio_trained = True
        else:
            self.sorted_tabular_features = features_df
            self.tabular_trained = True

        return features_df.head(top_x)

    def plot_feature_ranking_heatmap(self, audio: bool = False):
        """
        Plots the feature ranking heatmap
        :param audio: bool (default = False) whether to extract audio or tabular features
        :return: matplotlib.Figure object of plotted figure
        """

        features_df = self.get_feature_ranking(audio=audio)

        if self.task == "classification":
            labels = ["Mutual\nInfo.", "Statistic\ntests", "XGBoost", "Linear\nSVC", "Average"]
        else:
            labels = ["Mutual\nInfo.", "Statistic\ntests", "LASSO", "Elastic\nNet", "Average"]

        fig, ax = plt.subplots()
        if audio:
            feature_names = list(features_df["Feature Name"])
        else:
            feature_names = [name[:40] for name in list(features_df["Feature Name"])]
        graph = sns.heatmap(features_df.drop(["Feature Name"], axis=1), annot=True, ax=ax, vmin=0, vmax=1, linewidth=0.5,
                            yticklabels=feature_names, xticklabels=labels, cmap="Greens")

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        c_bar = graph.collections[0].colorbar
        c_bar.ax.set_ylabel("Feature Importance\n 1 = Most Important, 0 = Least Important", rotation=-90,
                         va="bottom", fontweight="bold")
        if audio:
            ax.set_ylabel("Audio Features", fontweight="bold")
        else:
            ax.set_ylabel("Tabular Features", fontweight="bold")
        ax.set_xlabel("Feature Extraction Methods", fontweight="bold")
        fig.tight_layout()

        if audio:
            plt.subplots_adjust(left=0.5, right=0.9)
        else:
            plt.subplots_adjust(left=0.35, right=0.9)

        return fig

    def baseline_model_class(self, audio: bool = False):
        """
        Gets baseline classification model performance
        :param audio: bool (default=False). Whether to train on audio data or tabular
        :return: tuple(RF UAR, SVC UAR, RF CM, SVC CM)
        """

        if audio:
            print("Training baseline audio classifier models...")
            rf_uar, svc_uar = self.audio_modeller.evaluate_classifier_models()
            rf_cm, rf_y_hat, svc_cm, svc_y_hat = self.audio_modeller.get_confusion_matrix_class()
            print("Done!")
        else:
            print("Training baseline tabular classifier models...")
            rf_uar, svc_uar = self.tabular_modeller.evaluate_classifier_models()
            rf_cm, rf_y_hat, svc_cm, svc_y_hat = self.tabular_modeller.get_confusion_matrix_class()
            print("Done!")

        return rf_uar, svc_uar, rf_cm, svc_cm, rf_y_hat, svc_y_hat

    def baseline_model_reg(self, audio: bool = False):
        """
        Gets baseline regression model performance
        :param audio: bool (default=False). Whether to train on audio data or tabular
        :return: tuple(LASSO R2, LASSO RMSE, EN R2, EN RMSE)
        """
        if audio:
            print("Training baseline audio regression models...")
            result = self.audio_modeller.evaluate_regression_models()
            print("Done!")
        else:
            print("Training baseline tabular regression models...")
            result = self.tabular_modeller.evaluate_regression_models()
            print("Done!")
        return result

    def plot_confusion_matrix(self, cm: np.ndarray, y_hat, audio, title: str, bar_label: bool = False):
        """
        Plots the confusion matrix
        :param bar_label: bool (default=False) whether to add the color bar label
        :param audio: bool (default=False) whether the data is audio or tabular
        :param y_hat: list or predicted values
        :param title: Title of the plot
        :param cm: np array containing confusion matrix
        :return: matplotlib.Fig
        """

        if audio:
            classes = list(np.unique(self.audio_modeller.y_test))
        else:
            classes = list(np.unique(self.tabular_modeller.y_test))

        classes = set(classes)
        y_hat = set(list(y_hat))
        classes = classes.union(y_hat)
        classes = list(classes)

        shortened_classes = []
        for cls in classes:
            if len(str(cls)) > 10:
                shortened_classes.append(str(cls)[:10])
            else:
                shortened_classes.append(cls)

        classes = shortened_classes
        classes.sort()
        extra_txt = ""
        if len(classes) > 10:
            extra_txt = "Top 10 performing classes shown"
            # Get the top 10 performing classes to plot
            no_of_classes = len(classes)

            accuracies = []
            for i in range(no_of_classes):
                accuracies.append(cm[i, i])

            mean_accuracy = sum(accuracies)/len(accuracies)
            accuracies = [abs(x - mean_accuracy) for x in accuracies]
            sorted_accuracies = accuracies.copy()
            sorted_accuracies.sort()

            # Get 10th highest accuracy in confusion matrix
            threshold = sorted_accuracies[-10]

            top_10_classes = []
            positions = []
            top_10_accuracies = []
            for i in range(no_of_classes):
                if accuracies[i] >= threshold:
                    top_10_classes.append(classes[i])
                    positions.append(i)
                    top_10_accuracies.append(cm[i, i])

            classes = top_10_classes
            if len(classes) > 10:
                classes = classes[:10]
                positions = positions[:10]

            positions_to_delete = [i for i in range(no_of_classes) if i not in positions]

            for i in range(len(positions_to_delete)):
                cm = np.delete(cm, positions_to_delete[i], 0)
                cm = np.delete(cm, positions_to_delete[i], 1)
                positions_to_delete = [x - 1 if x > positions_to_delete[i] else x for x in positions_to_delete]

        fig, ax = plt.subplots()

        # Plot the heatmaps on the created figures
        graph = sns.heatmap(cm, annot=True, linewidths=0.5, vmin=0, vmax=1,  xticklabels=classes,
                            yticklabels=classes, ax=ax, fmt=".2f", cmap="Greens")

        if bar_label:
            c_bar = graph.collections[0].colorbar
            c_bar.ax.set_ylabel("Proportion of each class", rotation=-90,
                                va="bottom", fontweight="bold")

        title += f"\n{extra_txt}"
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        fig.tight_layout()

        return fig

    def plot_pca(self, audio: bool = False):
        """
        Plots the graph on model performance accross an increasing number of PCA components
        :return: matplotlib.Fig object
        """

        if audio:
            print("Training audio PCA model...")
            modeller = self.audio_modeller
        else:
            print("Training tabular PCA model...")
            modeller = self.tabular_modeller

        if self.task == "regression":
            performance, components = modeller.evaluate_pca_model(reg=True)
            title = "LASSO model performance\nacross an increasing number of PCA components"
            y_label = "Root Mean Squared Error"
        else:
            performance, components = modeller.evaluate_pca_model()
            title = "Linear Support Vector Classification performance\nacross an increasing number of PCA components"
            y_label = "Unweighted Average Recall (UAR)"

        # Edge case where there is not enough components to do any PCA
        if performance is None:
            return None

        # Don't want to add plot to report if less than 5 components
        if components[-1] < 5:
            return None

        fig, ax = plt.subplots()

        sns.lineplot(x=components, y=performance, ax=ax, marker="o")
        ax.set_xlabel("Number of principal components in model")
        ax.set_title(title)
        ax.set_ylabel(y_label)
        print("Done!")
        return fig

    def get_correlation_metric(self, audio: bool = False) -> tuple:
        """
        Returns dataFrames of correlation metrics with output var
        :param audio: bool (default=False) Whether to get audio or tabular correlations
        :return: tuple[spearman_df, cramers_v_df, kruskal_df]
        """
        if audio:
            return self.audio_analyser.get_output_correlations()
        else:
            if self.tabular_analyser is None:
                raise ValueError("Error, trying to get tabular correlations when no tabular"
                                 "variables exist")
            return self.tabular_analyser.get_output_correlations()

    def get_model_features(self, audio: bool = False):
        """
        Returns df containing model features
        :param audio: bool (default=False) Whether to get audio or tabular features
        :return: pd.Dataframe of features
        """

        if audio:
            features = self.audio_model_features
        else:
            features = self.tabular_model_features

        if features is None:
            raise ValueError("Error, trying to extract model features before training models")

        return features

    def get_feature_importance(self, feature_name: str, audio: bool = False):
        """
        Extracts the relative feature importance from Mutual Information and model feature importance
        :param feature_name: String, name of the feature to extract its relative importance
        :param audio: bool (default=False) Whether to extract audio or tabular features
        :return: tuple(MI_importance, RF_importance, SVC_importance)
        """
        if audio:
            if self.sorted_audio_features is None:
                self.get_feature_ranking(audio=True)
            feature_df = self.sorted_audio_features
        else:
            if self.sorted_tabular_features is None:
                self.get_feature_ranking()
            feature_df = self.sorted_tabular_features

        # labels = ["Feature Name", "MI", model_1_labels[0], model_2_labels[0], "Average"]
        if feature_name not in list(feature_df["Feature Name"]):
            raise ValueError(f"Error in getting feature importance: {feature_name} does not exist")

        feature_df = feature_df[feature_df["Feature Name"] == feature_name]
        return feature_df.iloc[0, 1], feature_df.iloc[0, 2], feature_df.iloc[0, 3], feature_df.iloc[0, 4]

    def get_simple_audio_feature_importance(self):
        """
        Extracts simple audio features relative importance
        :return: pd.DF
        """

        sample_rate = list(self.get_feature_importance("sample_rate", audio=True))
        loudness = list(self.get_feature_importance("loudness", audio=True))
        duration = list(self.get_feature_importance("duration", audio=True))

        if self.task == "classification":
            df = pd.DataFrame(index=["Mutual\nInformation", "Statistical\nTests",
                                     "XGBoost", "Support\nVector Class."])
        else:
            df = pd.DataFrame(index=["Mutual\nInformation", "Statistical\nTests",
                                     "Lasso\nReg.", "Elastic Net\nReg."])

        df["sample_rate"] = sample_rate
        df["loudness"] = loudness
        df["duration"] = duration
        df = df.T

        return df

    def simple_audio_heatmap(self):
        """
        Plots a heatmap of the simple audio features
        :return: matplotlib.Fig object
        """
        simple_features = self.get_simple_audio_feature_importance()

        fig, ax = plt.subplots()

        graph = sns.heatmap(simple_features, annot=True, ax=ax, vmin=0, vmax=1, linewidth=0.5,
                            cmap="Reds")

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        c_bar = graph.collections[0].colorbar
        c_bar.ax.set_ylabel("Feature Importance\n1 = Most Important, 0 = Least Important", rotation=-90,
                            va="bottom", fontweight="bold")
        ax.set_ylabel("Simple Audio Features", fontweight="bold")
        ax.set_xlabel("Feature Extraction Methods", fontweight="bold")
        fig.tight_layout()

        return fig

    def plot_tsne(self):
        """
        Plots the TSNE plot from the dataset features
        :return: matplotlib Fig
        """

        tsne_data = self.audio_fe.tsne()
        tsne_df = pd.DataFrame(tsne_data)
        tsne_df.columns = ["First TSNE Feature", "Second TSNE Feature"]
        tsne_df[self.output_var] = self.audio_fe.df[self.output_var]

        plt.figure()
        ax = sns.jointplot(data=tsne_df, x="First TSNE Feature", y="Second TSNE Feature", hue=self.output_var,
                           palette="crest")
        ax.fig.suptitle(f"Distribution of {self.output_var} across first two TSNE Features")

        return ax._figure

    def add_feature_importance_to_ax(self, feature_name, ax, audio: bool = False):
        """
        Retrieves and adds the feature importance text to a given axes object
        :param audio: bool (default=False) Whether the feature is audio or tabular
        :param feature_name: Name of the feature in the plot
        :param ax: matplotlib.axes object to add the text to
        :return: None
        """

        mutual_info, stat_test, model_1, model_2 = self.get_feature_importance(feature_name, audio=audio)

        if self.task == "classification":
            model_1_name = "XGBoost"
            model_2_name = "Linear SVC"
        else:
            model_1_name = "LASSO Regression"
            model_2_name = "Elastic Net Regression"

        text = (f"Mutual Information Relative Importance: {mutual_info:.2f}\n"
                f"Statistical Tests Relative Importance: {stat_test:.2f}\n"
                f"{model_1_name} Relative Importance: {model_1:.2f}\n"
                f"{model_2_name} Relative Importance: {model_2:.2f}")

        ax.text(0, 0.7, text, transform=ax.transAxes, bbox=dict(facecolor="white"))

    def get_simple_baseline(self, audio: bool = False):
        """
        Gets random chance and most_frequent_class heuristic performance on the test set
        :param audio: bool (default=False). Whether to use the audio or tabular modeller
        :return: tuple(random_chance, most_frequent_class)
        """

        if audio:
            modeller = self.audio_modeller
        else:
            modeller = self.tabular_modeller

        classes = list(set(list(modeller.y_train)))

        most_frequent_class = mode(modeller.y_train)

        number_of_classes = len(classes)
        number_of_predictions = len(list(modeller.y_test))

        random_predictions = [classes[randint(0, number_of_classes-1)] for i in range(0, number_of_predictions)]
        most_frequent_predictions = [most_frequent_class] * number_of_predictions

        random_uar = recall_score(modeller.y_test, random_predictions, average="macro", zero_division=0)
        most_frequent_uar = recall_score(modeller.y_test, most_frequent_predictions, average="macro", zero_division=0)

        return random_uar, most_frequent_uar

    def get_regression_baseline(self, audio: bool = False):
        """
        Gets simple baseline heuristic metrics for regression task
        :param audio: bool (default=False) whether the data is audio or tabular
        :return: tuple(float, float)
        """

        if audio:
            modeller = self.audio_modeller
        else:
            modeller = self.tabular_modeller

        y_train = list(modeller.y_train)
        y_mean = sum(y_train)/len(y_train)

        y_test = modeller.y_test
        y_mean_pred = [y_mean] * len(y_test)

        r2 = r2_score(y_test, y_mean_pred)
        rmse = sqrt(mean_squared_error(y_test, y_mean_pred))

        return r2, rmse

    def traffic_light_score(self, audio: bool = False):
        """
        Calculates the traffic light score for the report
        :param audio: bool (default = False) whether the score is for audio or tabular data
        :return: dict
        """

        baseline_score = None
        feature_score = None
        overall_score = None
        simple_feature_score = None

        if self.task == "classification":
            model_1_uar, model_2_uar, _, _, _, _ = self.baseline_model_class(audio=audio)
            _, most_freq_uar = self.get_simple_baseline(audio=audio)

            average_uar = (model_1_uar + model_2_uar) / 2

            if average_uar >= 1.75 * most_freq_uar:
                baseline_score = "red"
            elif average_uar >= 1.5 * most_freq_uar:
                baseline_score = "amber"
            else:
                baseline_score = "green"

        else:
            _, model_1_rmse, _, model_2_rmse = self.baseline_model_reg(audio=audio)
            _, simple_rmse = self.get_regression_baseline(audio=audio)

            average_rmse = (model_1_rmse + model_2_rmse) / 2

            if average_rmse <= simple_rmse / 1.75:
                baseline_score = "red"
            elif average_rmse <= simple_rmse / 1.5:
                baseline_score = "amber"
            else:
                baseline_score = "green"

        if audio:
            simple_features = self.get_simple_audio_feature_importance()

            values = list(simple_features.values)
            simple_feature_score = "green"
            for vals in values:
                if any(x >= 0.5 for x in vals):
                    simple_feature_score = "red"
                    break
                if any(x >= 0.15 for x in vals):
                    simple_feature_score = "amber"

            most_important_feature_average = self.sorted_audio_features.iloc[0, 5]
            x_feature_average = self.sorted_audio_features.iloc[4, 5]
            end_point = 4
        else:
            most_important_feature_average = self.sorted_tabular_features.iloc[0, 5]
            end_point = min(self.sorted_tabular_features.shape[0] - 1, 4)
            x_feature_average = self.sorted_tabular_features.iloc[end_point, 5]

        if most_important_feature_average >= 2 * x_feature_average:
            feature_score = "red"
        elif most_important_feature_average >= 1.5 * x_feature_average:
            feature_score = "amber"
        else:
            feature_score = "green"

        if audio:
            if baseline_score == "green":
                if simple_feature_score == "green":
                    overall_score = "green"
                else:
                    overall_score = "amber"
            elif baseline_score == "amber":
                if simple_feature_score == "green":
                    overall_score = feature_score
                elif simple_feature_score == "amber":
                    if feature_score == "green" or feature_score == "amber":
                        overall_score = "amber"
                    else:
                        overall_score = "red"
                elif simple_feature_score == "red":
                    overall_score = "red"
            elif baseline_score == "red":
                if simple_feature_score == "amber" or simple_feature_score == "red":
                    overall_score = "red"
                elif simple_feature_score == "green":
                    if feature_score == "green" or feature_score == "amber":
                        overall_score = "amber"
                    else:
                        overall_score = "red"

        else:
            if baseline_score == "green":
                overall_score = "green"
            elif feature_score == "green" or feature_score == "amber":
                overall_score = "amber"
            else:
                overall_score = "red"

        if self.task == "classification":
            return {"overall_score": overall_score, "model_score": baseline_score, "feature_score": feature_score,
                    "average_uar": average_uar, "most_freq_uar": most_freq_uar, "most_important_feature_average":
                    most_important_feature_average, "x_feature_average": x_feature_average,
                    "simple_audio": simple_feature_score, "last_feature": end_point + 1}
        else:
            return {"overall_score": overall_score, "model_score": baseline_score, "feature_score": feature_score,
                    "average_rmse": average_rmse, "simple_rmse": simple_rmse, "most_important_feature_average":
                        most_important_feature_average, "x_feature_average": x_feature_average,
                    "simple_audio": simple_feature_score, "last_feature": end_point + 1}

    def plot_class_performance(self, y_hat: list, audio: bool = False, model_name: str = ""):
        """
        Plots class performance UAR graphs
        :param model_name: Str, name of the model to put on the Figure
        :param audio: bool (default = False) whether the data is tabular or audio
        :param y_hat: predicted values
        :return: matplotlib.fig object
        """

        if audio:
            modeller = self.audio_modeller
        else:
            modeller = self.tabular_modeller

        y_true = list(modeller.y_test)
        labels = set(y_true)
        classes = labels.union(y_hat)
        classes = sorted(classes)
        recall = list(recall_score(y_true=y_true, y_pred=y_hat, average=None, zero_division=0))
        recall_df = pd.DataFrame()

        new_classes = []
        rotate = False
        for cl in classes:
            if len(str(cl)) > 8:
                cl = str(cl)[:8]
                new_classes.append(cl)
                rotate = True
            else:
                new_classes.append(cl)

        recall_df["Class"] = new_classes
        recall_df["Recall"] = recall

        uar = sum(recall)/len(labels)
        extra_txt = ""

        # Only want 10 most interesting classes to plot on graph
        if len(labels) > 10:
            diff_from_uar = [abs(x - uar) for x in recall]
            recall_df["Diff"] = diff_from_uar
            recall_df = recall_df.sort_values(by=["Diff"], ascending=False)
            recall_df = recall_df.drop(["Diff"], axis=1)
            recall_df = recall_df.head(10)
            extra_txt = "\nmost interesting 10 classes selected"

        fig, ax = plt.subplots()
        plot = sns.barplot(data=recall_df, x="Class", y="Recall", ax=ax)
        plot.axhline(y=uar, label=f"Overall UAR = {uar:.2f}", color="r", ls="--", alpha=0.75)

        ax.set_title(f"Recall by class of {model_name} model{extra_txt}")
        ax.legend()

        if rotate:
            plt.xticks(rotation=45)

        fig.tight_layout()

        return fig

    def plot_residuals(self, y_hat, audio: bool = False, model_name: str = ""):
        """
        Plots the residual graphs from regression models
        :param model_name: Str, name of regression model (elastic net or LASSO)
        :param y_hat: List of predicted values
        :param audio: bool (default = False) whether the data is audio or tabular
        :return: matplotlib.Fig object
        """
        if audio:
            modeller = self.audio_modeller
        else:
            modeller = self.tabular_modeller

        y_true = list(modeller.y_test)

        percentage_diff = [((y_hat[index] - y_true[index]) / y_true[index]) * 100 for index in range(len(y_true))]
        residual_df = pd.DataFrame()
        residual_df["Residuals %-difference"] = percentage_diff
        residual_df["y_true"] = y_true

        fig, ax = plt.subplots()
        plot = sns.scatterplot(data=residual_df, x="y_true", y="Residuals %-difference", ax=ax)
        ax.set_title(f"Residuals %-difference\nof {model_name} model")
        plot.axhline(y=0, color="r")

        return fig

    def get_model_performance_graphs(self, audio: bool = False):
        """
        Plots the model performance graphs
        :param audio: bool (default = False) whether the data is audio or tabular
        :return: tuple of matplotlib.figs
        """

        if audio:
            modeller = self.audio_modeller
        else:
            modeller = self.tabular_modeller

        if self.task == "classification":
            xgboost_y_hat = list(modeller.bst_y_hat)
            svc_y_hat = list(modeller.svc_y_hat)
            fig_1 = self.plot_class_performance(xgboost_y_hat, audio=audio, model_name="XGBoost")
            fig_2 = self.plot_class_performance(svc_y_hat, audio=audio, model_name="SVC")
        else:
            lasso_y_hat = list(modeller.lasso_y_hat)
            en_y_hat = list(modeller.en_y_hat)
            fig_1 = self.plot_residuals(lasso_y_hat, audio=audio, model_name="Lasso")
            fig_2 = self.plot_residuals(en_y_hat, audio=audio, model_name="Elastic Net")

        return fig_1, fig_2
