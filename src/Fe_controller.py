import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.Feature_selector import FeatureExtractor
from src.Data_analyser import DataAnalyser
from src.Audio_processor import AudioProcessor
from sklearn.preprocessing import StandardScaler


class FeController:
    def __init__(self, analyser: DataAnalyser,
                 audio_processor: AudioProcessor = None,
                 pre_process = True):
        """
        Stores DataAnalyser and AudioProcessor objects
        :param analyser: DataAnalyser object
        :param audio_processor: Optional AudioProcessor Object
        """
        self.analyser = analyser
        self.audio_processor = audio_processor

        # Store output_variable as data_member
        self.output_var = self.analyser.output_var
        # Store tabular df as data_member
        self.tabular_df = self.analyser.df
        # Store audio_feature df as data_member
        if self.audio_processor is not None:
            self.audio_features = self.audio_processor.audio_features

        # Create FeatureExtractor objects
        self.tabular_fe = FeatureExtractor(self.tabular_df, self.output_var)

        if pre_process:
            self.pre_process()

        if self.audio_processor is not None:
            self.audio_fe = FeatureExtractor(self.audio_features, self.output_var)
            if pre_process:
                self.pre_process(audio=True)

    def pre_process(self, audio: bool = False) -> None:
        """
        Pre-processes the feature extractors' dataframes by min-max scaling continuous variables
        and adding K-1 dummy variables for each categorical variable, where K = number of categories
        :param audio: bool (default=False) If True, pre-processes the audio_fe, else, pre-processes
                      the tabular_fe
        :return: None (modifies FeatureExtractor objects)
        """
        if audio:
            fe = self.audio_fe
            features = list(self.audio_features.columns)
        else:
            fe = self.tabular_fe
            features = list(self.tabular_df.columns)

        for feature in features:
            # Don't want to process the output variable
            if feature == self.output_var:
                continue
            if audio:
                fe.normalize_var(feature)
            else:
                if self.analyser.is_categorical(feature):
                    fe.one_hot_encoding(feature)
                else:
                    fe.normalize_var(feature)

    def extract_features(self, no_of_features: int = 20, audio: bool = False) -> pd.DataFrame:
        """
        Extracts the most relevant features (as specified by no_of_features) and returns
        them as a pd.Dataframe
        :param audio: bool (default=false) if true, extracts audio features, else extracts tabular features
        :param no_of_features: option int (default=20) maximum number of features to extract
        :return: pd.DataFrame containing relevant features according to method used
        """
        fe = self.tabular_fe
        if audio:
            fe = self.audio_fe

        df = pd.DataFrame()
        df2 = pd.DataFrame()

        if self.analyser.is_categorical(self.output_var):
            rf, svc = fe.get_classification_features(no_of_features)
            df["RF Features"] = list(rf.keys())
            df["RF Gini"] = [round(val, 4) for val in list(rf.values())]
            df = df.sort_values(by=["RF Gini"], ascending=False)
            df2["Linear SVC Features"] = list(svc.keys())
            df2["SVC Coeff"] = [round(val, 4) for val in list(svc.values())]
            df2 = df2.sort_values(by=["SVC Coeff"], ascending=False)
        else:
            lasso, elastic = fe.get_regression_features(no_of_features)
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

    def get_pca(self, no_of_components: int = 2, data: str = "all", audio: bool = False,
                reset = False) -> pd.DataFrame:
        """
        Extracts the principal components of the data
        :param data: {"all", "train", "test"}
        :param no_of_components: optional int (default=2), number of principal components to extract
        :param audio: bool (default=false) if true, extracts audio features, else extracts tabular features
        :return: pd.DataFrame containing principal component values

        Raises ValueError if PCA is not possible: no_of_components must be between 0 and min(n_samples, n_features)
        """
        fe = self.tabular_fe
        if audio:
            fe = self.audio_fe

        # Need to remove categorical variables from df, and restore df on function exit
        original_df = fe.df
        analyser_df = self.analyser.df
        self.analyser.df = fe.df
        for variable in list(fe.df.columns):
            if variable == self.output_var:
                continue
            elif not audio and self.analyser.is_categorical(variable):
                fe.df.drop([variable], inplace=True, axis=1)

        self.analyser.df = analyser_df

        # After augmenting fe.df, check whether PCA is possible
        rows, cols = fe.df.shape
        # Subtract 1 to cols as we don't want to include output variable
        if no_of_components > min(rows, cols - 1):
            raise ValueError("Not enough data/features to do PCA")

        # Transpose the numpy array to iterate over pca components not observations
        pca_values = fe.pca(no_of_components, data, reset=reset).T
        df = pd.DataFrame()
        for component in range(0, len(pca_values)):
            df[f"Component {component + 1}"] = list(pca_values[component])

        # Restore dataframe on exit
        fe.df = original_df
        return df

    def plot_first2_pca(self, audio: bool = False) -> list:
        """
        Plots the first two principal components as a scatter, coloured by the output var class
        :param audio: bool (default=false) if true, extracts audio features, else extracts tabular features
        :return: List of MatPlotLib figure objects
        """
        fe = self.tabular_fe
        if audio:
            fe = self.audio_fe
        components_df = self.get_pca(audio=audio)

        output_var = fe.df[self.output_var]
        assert output_var.shape[0] == components_df.shape[0]

        components_df["output_var"] = list(output_var)
        figure_list = []
        if self.analyser.is_categorical(self.output_var):
            categories = list(pd.unique(output_var))
            categories.sort()

            # Plot a maximum of 5 categories per scatter
            end_point = 3
            start_point = 0
            while end_point < len(categories) + 3:
                subset = categories[start_point:end_point]
                filtered_df = components_df[components_df["output_var"].isin(subset)]
                plt.figure()
                ax = sns.jointplot(data=filtered_df, x="Component 1", y="Component 2", hue="output_var",
                                   palette="crest")

                figure_list.append(ax._figure)
                end_point += 3
                start_point += 3
        else:
            fig, ax = plt.subplots()
            observations = output_var.shape[0]

            median = components_df["output_var"].median()
            low_df = components_df[components_df["output_var"] < median]
            high_df = components_df[components_df["output_var"] >= median]

            x_val_low = low_df["Component 1"]
            y_val_low = low_df["Component 2"]
            x_val_high = high_df["Component 1"]
            y_val_high = high_df["Component 2"]

            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_title("Scatter of first two principle components")

            ax.scatter(x_val_low, y_val_low, s=0.5, alpha=0.5)
            ax.scatter(x_val_high, y_val_high, s=0.5, alpha=0.5)
            ax.legend()

            figure_list.append(fig)

        return figure_list

    def evaluate_models(self, audio: bool = False) -> tuple:
        """
        Trains and evaluates appropriate models according to ML task
        :param audio: bool (default=false) if true, evaluates audio models, else evaluates tabular models
        :return: tuple(float,float) of accuracies or R2 depending on classification/regression task
        """
        fe = self.tabular_fe
        if audio:
            fe = self.audio_fe

        if self.analyser.is_categorical(self.output_var):
            return fe.evaluate_classifier_models()
        else:
            return fe.evaluate_regression_models()

    def plot_confusion_matrix(self, audio: bool = False) -> tuple:
        """
        Calculates and plots a heatmap of a confusion matrix for RF and SVC models
        :param audio: bool (default=false) if true, evaluates audio models, else evaluates tabular models
        :return: tuple[plt.figure, plt.figure], matplotlib figure objects of RF, SVC confusion matrices
        """
        fe = self.tabular_fe
        if audio:
            fe = self.audio_fe

        # This function should never be called before the models have been trained
        assert fe.trained_rf is not None
        assert fe.trained_svc is not None

        rf_cm = fe.get_confusion_matrix(fe.trained_rf)
        svc_cm = fe.get_confusion_matrix(fe.trained_svc)

        rf_fig, rf_ax = plt.subplots()
        svc_fig, svc_ax = plt.subplots()

        classes = list(pd.unique(self.tabular_df[self.output_var]))
        classes.sort()

        # Plot the heatmaps on the created figures
        sns.heatmap(rf_cm, annot=True, linewidths=0.5, xticklabels=classes,
                    yticklabels=classes, ax=rf_ax, fmt=".2f", cmap="Greens")
        sns.heatmap(svc_cm, annot=True, linewidths=0.5, xticklabels=classes,
                    yticklabels=classes, ax=svc_ax, fmt=".2f", cmap="Greens")

        rf_ax.set_title("Random Forest Confusion Matrix")
        rf_ax.set_xlabel("Predicted")
        rf_ax.set_ylabel("Actual")

        svc_ax.set_title("Linear SVC Confusion Matrix")
        svc_ax.set_xlabel("Predicted")
        svc_ax.set_ylabel("Actual")

        rf_fig.tight_layout()
        svc_fig.tight_layout()

        return rf_fig, svc_fig

    def set_test_csv(self, test_csv: str, filename: str = None):
        """

        :param filename: (optional) audio filename col
        :param test_csv: filepath to test csv
        :return:
        """

        test_df = pd.read_csv(test_csv)
        headers = list(test_df.columns)
        if self.output_var not in headers:
            raise ValueError(f"Error - no column in provided in test csv file called {self.output_var}")

        if filename is not None:
            if filename not in headers:
                raise ValueError(f"Error - no audio file column provided in test csv file called {filename}")

            test_audio_processor = AudioProcessor(test_df, self.output_var, filename, self.analyser)
            test_features = test_audio_processor.audio_features

            scaler = StandardScaler()
            self.audio_fe.x_train = self.audio_fe.df.drop([self.output_var], axis=1)
            self.audio_fe.x_test = test_features.drop([self.output_var], axis=1)

            print("Train Features:")
            print(self.audio_fe.x_train)
            print("Test Features:")
            print(self.audio_fe.x_test)

            scaler = scaler.fit(self.audio_fe.x_train)

            self.audio_fe.x_train = scaler.transform(self.audio_fe.x_train)
            self.audio_fe.x_test = scaler.transform(self.audio_fe.x_test)
            self.audio_fe.y_test = test_features[self.output_var]
            self.audio_fe.split = True

            self.audio_fe.y_train = self.audio_fe.df[self.output_var]

            print("X_train scaled:")
            print(self.audio_fe.x_train)
            print("X_test scaled:")
            print(self.audio_fe.x_test)
            print("Y_train:")
            print(self.audio_fe.y_train)
            print("Y_test:")
            print(self.audio_fe.y_test)
