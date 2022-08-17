"""
Using FPDF2 package: https://github.com/PyFPDF/fpdf2
Following tutorial here: https://pyfpdf.github.io/fpdf2/
"""
import math

import pandas as pd
from fpdf import FPDF
from pdfrw import PdfReader, PdfWriter
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl
import numpy as np

import src.config as config
from src.Plotter import Plotter
from src.utility import get_same_variables
from src.Data_builder import DataBuilder

mpl.rcParams['figure.dpi'] = 300


class ReportBuilder:
    def __init__(self, db: DataBuilder, audio: bool = False):
        """
        Stores the relevant objects and correlation dataframes that are used frequently throughout
        the report generation

        :param db: DataBuilder object that generates the data needed for the report
        :param audio: bool (default=False) whether audio data is generated
        """

        self.data_builder = db
        self.output_var = self.data_builder.output_var

        self.tabular_plotter = Plotter(self.data_builder.tabular_df, self.output_var)
        self.audio_plotter = None

        self.pdf = FPDF()
        self.table_number = 1
        self.figure_number = 1
        self.audio_report = False

        if audio:
            self.audio_report = True
            self.audio_plotter = Plotter(self.data_builder.audio_features, self.output_var)

        # Font values set in set_fonts function
        self.title_font = None
        self.large_title = None
        self.small_title = None
        self.body = None
        self.bold_body = None
        self.set_fonts()
        self.build_report()

    def set_fonts(self) -> None:
        """
        Sets the default font styles used in the report
        :return: None
        """
        self.title_font = ('Helvetica', 'B', 24)
        self.large_title = ('Helvetica', 'B', 16)
        self.small_title = ('Helvetica', 'B', 12)
        self.body = ('Helvetica', '', 10)
        self.bold_body = ('Helvetica', 'B', 10)
        self.pdf.set_fill_color(r=166, g=214, b=255)

    def add_title_page(self):
        self.pdf.add_page()
        self.pdf.set_font(*self.title_font)
        # Center cursor position
        title = "Machine Learning Data Report"
        self.pdf.x = self.pdf.epw / 2 - self.pdf.get_string_width(title) / 2 + 10
        self.pdf.y = self.pdf.eph / 2

        self.pdf.cell(txt=title)

    def build_report(self) -> None:
        """
        Calls the functions required to generate the pages, and the page contents, of the PDF report
        :return: None
        """
        self.add_title_page()
        self.pdf.add_page()

        if self.audio_report:
            self.build_audio_summary()

        if self.data_builder.tabular_fe is not None:
            self.build_tabular_summary()

        if self.audio_report:
            self.generate_audio_visuals()

        if self.data_builder.tabular_fe is not None:
            self.generate_visualisation()

        # self.build_data_overview()
        print("Generating PDF report....")
        self.pdf.add_page(orientation="P")
        self.pdf.output(f"{config.REPORT_NAME}.pdf")
        print("Done!")

    def add_baseline_models(self, audio: bool = False):
        """
        Adds the baseline model text and plots to report
        :param audio: Determines whether to do audio or tabular models
        :return: None
        """
        if self.data_builder.task == "classification":

            xb_uar, svc_uar, xb_cm, svc_cm, xb_y_hat, svc_y_hat = self.data_builder.baseline_model_class(audio=audio)
            random_chance, most_frequent_class = self.data_builder.get_simple_baseline(audio=audio)
            self.pdf.set_x(30)
            current_y = self.pdf.get_y()
            self.pdf.multi_cell(txt=f"XGBoost UAR: {xb_uar:.2f}\n"
                                    f"Random Chance UAR: {random_chance:.2f}\n"
                                    f"Most Frequent Class UAR: {most_frequent_class:.2f}",
                                w=self.pdf.epw / 3.5, fill=True)
            self.pdf.set_y(current_y)
            self.pdf.set_x((self.pdf.epw - 60))
            self.pdf.multi_cell(txt=f"Linear SVC UAR: {svc_uar:.2f}\n"
                                    f"Random Chance UAR: {random_chance:.2f}\n"
                                    f"Most Frequent Class UAR {most_frequent_class:.2f}",
                                w=self.pdf.epw / 3.5, fill=True)
            self.pdf.ln(5)

            graph_1 = self.data_builder.plot_confusion_matrix(xb_cm, xb_y_hat, audio=audio,
                                                              title="XGBoost Confusion Matrix", bar_label=True)
            graph_2 = self.data_builder.plot_confusion_matrix(svc_cm, svc_y_hat, audio=audio,
                                                              title="Linear SVC Confusion Matrix")

        else:
            lasso_r2, lasso_rmse, elastic_r2, elastic_rmse = self.data_builder.baseline_model_reg(audio=audio)
            mean_r2, mean_rmse = self.data_builder.get_regression_baseline(audio=audio)
            current_y = self.pdf.get_y()
            self.pdf.multi_cell(txt=f"                Lasso Regression R-Squared: {round(lasso_r2, 2)}\n"
                                    f"                Lasso Regression RMSE: {round(lasso_rmse, 2)}\n"
                                    f"                Predict mean of y_train RMSE: {round(mean_rmse, 2)}",
                                w=self.pdf.epw / 2)
            self.pdf.set_y(current_y)
            self.pdf.set_x((self.pdf.epw * 0.70))
            self.pdf.multi_cell(txt=f"Elastic Net R-Squared: {round(elastic_r2, 2)}\n"
                                    f"Elastic Net RMSE: {round(elastic_rmse, 2)}\n"
                                    f"Predict mean of y_train RMSE: {round(mean_rmse, 2)}", w=self.pdf.epw / 2.5)

            graph_1, graph_2 = self.data_builder.plot_regression_predictions(audio=audio)

        current_y = self.pdf.get_y()
        self.insert_graph(graph_1, 5.5, 4.5, 2, 3.5, x=10)
        self.pdf.set_y(current_y)
        self.insert_graph(graph_2, 5.5, 4.5, 2, 3.5, x=(self.pdf.epw * 0.70) - 20)

        ################################ Add class/residual graphs ####################################################
        self.pdf.add_page(orientation="P")

        if audio:
            type = "Audio"
        else:
            type = "Tabular"

        self.pdf.set_font(*self.large_title)

        if self.data_builder.task == "classification":
            self.pdf.start_section(f"{type} baseline models performance by class", level=1)
            self.pdf.cell(txt=f"{type} baseline models performance by class", border="B", w=0)
        else:
            self.pdf.start_section(f"{type} baseline models residual visualisation", level=1)
            self.pdf.cell(txt=f"{type} baseline models residual plots", border="B", w=0)

        self.pdf.ln(10)
        fig_1, fig_2 = self.data_builder.get_model_performance_graphs(audio=audio)
        self.insert_graph(fig_1, width=7, height=6, p_width=1.2, p_height=2.5, x=35)
        self.pdf.ln(10)
        self.insert_graph(fig_2, width=7, height=6, p_width=1.2, p_height=2.5, x=35)

    def insert_traffic_colour(self, colour: str):
        """

        :param colour:
        :return:
        """
        if colour == "red":
            self.pdf.set_text_color(r=255, g=0, b=0)
            self.pdf.multi_cell(txt=f"RED", w=10)
        elif colour == "amber":
            self.pdf.set_text_color(r=255, g=128, b=0)
            self.pdf.multi_cell(txt=f"AMBER", w=15)
        else:
            self.pdf.set_text_color(r=0, g=255, b=0)
            self.pdf.multi_cell(txt=f"GREEN", w=15)

        self.pdf.ln(5)
        self.pdf.set_text_color(r=0, g=0, b=0)

    def add_traffic_lights(self, audio: bool = False):
        """

        :param audio:
        :return:
        """
        score_dict = self.data_builder.traffic_light_score(audio=audio)
        if audio:
            data = "Audio"
        else:
            data = "Tabular"

        self.pdf.set_font(*self.title_font)
        self.pdf.start_section(f"{data} Traffic Light Rating")
        self.pdf.multi_cell(txt=f"Potential Spurious Correlation\nTraffic Light Rating on {data} Data",
                            w=0, align="C")
        self.pdf.ln(10)

        overall_score = score_dict["overall_score"]
        model_score = score_dict["model_score"]
        feature_score = score_dict["feature_score"]
        most_important_feature_avg = score_dict["most_important_feature_average"]
        x_feature_average = score_dict["x_feature_average"]
        last_feature = score_dict["last_feature"]
        simple_feature_score = score_dict["simple_audio"]

        if self.data_builder.task == "classification":
            model_uar = score_dict["average_uar"]
            most_freq_uar = score_dict["most_freq_uar"]
            performance_gain = model_uar / most_freq_uar
        else:
            model_rmse = score_dict["average_rmse"]
            mean_rmse = score_dict["simple_rmse"]
            performance_gain = mean_rmse / model_rmse

        try:
            feature_importance_gain = most_important_feature_avg / x_feature_average
        except RuntimeWarning:
            feature_importance_gain = math.inf

        self.pdf.set_font(*self.bold_body)
        self.pdf.multi_cell(txt=f"Overall rating is: ", w=30, new_y="LAST")

        extra_audio_txt = ""

        if simple_feature_score is None:
            simple_feature_score = ""

        if simple_feature_score == "red":
            extra_audio_txt = "Very simple audio features show potentially strong correlation with the output " \
                              "variable. " \
                              "It is very import to consider whether any of these simple feature should be true" \
                              " correlates, else there is very likely bias within the dataset. "

        elif simple_feature_score == "amber":
            extra_audio_txt = "Very simple audio features show potentially weak to moderate correlation with the " \
                              "output " \
                              "variable. " \
                              "It is very import to consider whether any of these simple feature should be true" \
                              " correlates, else there is very likely bias within the dataset. "

        if overall_score == "red" and model_score == "red":
            self.insert_traffic_colour("red")
            self.pdf.set_font(*self.body)
            self.pdf.multi_cell(txt=f"{simple_feature_score}There are strong relationships between dataset features "
                                    f"and the output variable. "
                                    f"This does not necessarily mean that these relationships are spurious, but it "
                                    f"warrants investigation to verify whether the identified most important variables "
                                    f"are expected to be correlates with the output variable.", w=0)

        elif overall_score == "amber" and simple_feature_score == "red":
            self.insert_traffic_colour("amber")
            self.pdf.set_font(*self.body)
            self.pdf.multi_cell(txt="Whilst baseline model performance suggests only weak or no relationships between "
                                    "features within the dataset and the output variable, simple audio features have "
                                    "been identified as being relatively important. It is important to check the plots "
                                    "of these features to check whether this relative importance is due to actual "
                                    "correlation between these features and the output variable, which may very likely "
                                    "indicate bias.", w=0)

        elif overall_score == "red" and model_score == "amber" and simple_feature_score == "red":
            self.insert_traffic_colour("red")
            self.pdf.set_font(*self.body)
            self.pdf.multi_cell(txt=f"{simple_feature_score}There are moderately strong "
                                    f"relationships between dataset features and the output variable. "
                                    f"This does not necessarily mean that these relationships are spurious, but it "
                                    f"warrants investigation to verify whether the identified most important variables "
                                    f"are expected to be correlates with the output variable.", w=0)

        elif overall_score == "red" and model_score == "amber" and feature_score == "red":
            self.insert_traffic_colour("red")
            self.pdf.set_font(*self.body)
            self.pdf.multi_cell(txt=f"{simple_feature_score}There are moderately strong "
                                    f"relationships between dataset features and the output variable. "
                                    f"This does not necessarily mean that these relationships are spurious, but it "
                                    f"warrants investigation to verify whether the identified most important variables "
                                    f"are expected to be correlates with the output variable.", w=0)

        elif overall_score == "amber" and model_score == "amber":
            self.insert_traffic_colour("amber")
            self.pdf.set_font(*self.body)
            self.pdf.multi_cell(txt="There are moderately strong relationships between dataset features and the output "
                                    "variable. This does not necessarily mean that these relationships are spurious, but it "
                                    "warrants investigation to verify whether the identified most important variables "
                                    "are expected to be correlates with the output variable.", w=0)
        else:
            self.insert_traffic_colour("green")
            self.pdf.set_font(*self.body)
            self.pdf.multi_cell(txt="Software has identified weak/no relationships between dataset features and the "
                                    "output variable. Deep learning may be required to construct a well-performing "
                                    "model. There may also be no relevant features in the dataset.", w=0)

        if audio:
            self.pdf.ln(15)
            self.pdf.set_font(*self.bold_body)
            self.pdf.multi_cell(txt="Audio simple feature bias rating is: ", w=60, new_y="LAST")

            if simple_feature_score == "red":
                self.insert_traffic_colour("red")
                self.pdf.set_font(*self.body)
                self.pdf.multi_cell(txt="At least one of three simple audio features has been identified as being "
                                        "relatively important to at least one feature extraction method. If you do not "
                                        "expect loudness to correlate with the output variable, this strongly suggest "
                                        "either bias within your dataset, or very poor baseline model performance. It "
                                        "is strongly advised you look at the plots of simple audio features to "
                                        "determine whether this is actual bias or not.", w=0)
            elif simple_feature_score == "amber":
                self.insert_traffic_colour("amber")
                self.pdf.set_font(*self.body)
                self.pdf.multi_cell(txt="At least one of three simple audio features has been identified as being "
                                        "weakly relatively important to at least one feature extraction method. If you do not "
                                        "expect loudness to correlate with the output variable, this strongly suggest "
                                        "either bias within your dataset, or very poor baseline model performance. It "
                                        "is strongly advised you look at the plots of simple audio features to "
                                        "determine whether this is actual bias or not.", w=0)
            else:
                self.insert_traffic_colour("green")
                self.pdf.set_font(*self.body)
                self.pdf.multi_cell(txt="No simple audio features are significantly important. There is therefore no "
                                        "immediately obvious bias in the sample rate, duration, or loudness of the "
                                        " audio data.", w=0)

        self.pdf.ln(15)
        self.pdf.set_font(*self.bold_body)
        self.pdf.multi_cell(txt=f"Baseline model rating is: ", w=44, new_y="LAST")

        if model_score == "red":
            self.insert_traffic_colour("red")
            model_txt = "Baseline model performance thus significantly outperforms the simple predictive heuristic. " \
                        "The overall performance should be examined, and, if the machine learning task is expected to " \
                        "be complex, this may suggest bias within your dataset, with the baseline models picking up on " \
                        "spurious patterns rather than the true signal."
        elif model_score == "amber":
            self.insert_traffic_colour("amber")
            model_txt = "Baseline model performance thus moderately outperforms the simple predictive heuristic. " \
                        "The overall performance should be examined, and, if the machine learning task is expected to " \
                        "be complex, this may suggest bias within your dataset, with the baseline models picking up on " \
                        "spurious patterns rather than the true signal."
        else:
            self.insert_traffic_colour("green")
            if performance_gain > 1:
                model_txt = "Baseline model performance is only slightly better than a simple predictive heuristic. " \
                            "This suggests that there are only weakly relevant features when computing a simple model, or" \
                            " no relevant features (in which case the relative " \
                            "feature importance scores should be ignored). It may also suggests that the machine learning " \
                            "task is complex, and so deep learning may be required to get a better predictive model."
            else:
                model_txt = "Baseline model performance is worse than a simple predictive heuristic. " \
                            "This suggests that there are either no relevant features (in which case the relative " \
                            "feature importance scores should be ignored), or, it suggests that the machine learning " \
                            "task is complex, and so deep learning may be required to get a better predictive model."

        self.pdf.set_font(*self.body)
        self.pdf.x = 10

        if self.data_builder.task == "classification":
            performance_txt = "most frequent class"
        else:
            performance_txt = "mean value of y_train"

        self.pdf.multi_cell(txt=f"Baseline model performance is {performance_gain:.2f} times that of a simple "
                                f"method of always predicting the {performance_txt}. {model_txt}", w=0)

        self.pdf.ln(15)
        self.pdf.set_font(*self.bold_body)
        self.pdf.multi_cell(txt=f"Relative feature importance rating is: ", w=66, new_y="LAST")

        if feature_score == "red":
            self.insert_traffic_colour("red")
            feature_txt = "This suggests that a relatively small subset of features are significantly more important " \
                          "than other features in the dataset. It is important to check whether there is a valid reason " \
                          "as to why these features are important, else, it may indicate bias within the dataset."
        elif feature_score == "amber":
            self.insert_traffic_colour("amber")
            feature_txt = "This suggests that a relatively small subset of features are moderately more important " \
                          "than other features in the dataset. It is important to check whether there is a valid reason " \
                          "as to why these features are important, else, it may indicate bias within the dataset."
        else:
            self.insert_traffic_colour("green")
            feature_txt = "This suggests that there is not a relatively small subset of dominant features. The most " \
                          "important features should still be checked, as if there is not a valid reason for them to be " \
                          "important then there may still be bias within the dataset."

        self.pdf.set_font(*self.body)
        self.pdf.x = 10

        if last_feature == 2:
            prefix = "nd"
        elif last_feature == 3:
            prefix = "rd"
        else:
            prefix = "th"

        self.pdf.multi_cell(
            txt=f"The most important feature is {feature_importance_gain:.2f} times more important than "
                f"the {last_feature}{prefix} most important feature. {feature_txt}", w=0)

        self.pdf.set_text_color(r=0, g=0, b=0)

    def build_audio_summary(self):
        """

        :return:
        """
        self.add_traffic_lights(audio=True)
        self.pdf.add_page()
        self.set_title("Summary of Audio Features")
        self.pdf.y -= 7
        self.pdf.start_section("Summary of audio features")
        self.pdf.set_font(*self.large_title)
        self.pdf.start_section("Simple Audio Feature Importance", level=1)
        self.pdf.cell(txt="Simple Audio Feature Importance", border="B", w=0)
        self.pdf.ln(10)
        self.pdf.set_font(*self.body)
        self.pdf.multi_cell(txt="Below shows a heatmap, and plots, of three very simple audio features: sample_rate, "
                                "loudness, and duration. If any of these are significantly greater than zero, and the "
                                "plots suggest some correlation, this very likely suggests bias within your data set, "
                                "and this needs to be addressed.", w=0)
        self.pdf.ln(15)
        current_y = self.pdf.get_y()
        self.insert_graph(self.data_builder.simple_audio_heatmap(), 6, 5, 1.5, 3, x=40)
        after_fig_y = self.pdf.get_y()
        self.pdf.set_y(current_y - 10)
        self.pdf.set_font(*self.bold_body)
        self.pdf.x += 20
        self.pdf.multi_cell(txt="For each feature extraction method (column), importance metrics have been scaled\n"
                                "between 0 and 1 such that 1 = most important feature, 0 = least "
                                "important "
                            , border=1, align="C", w=self.pdf.epw * 0.75)
        self.pdf.set_y(after_fig_y)
        self.pdf.set_font(*self.large_title)

        if self.data_builder.task == "classification":
            sample_rate_fig, _ = self.audio_plotter.plot_violin(cont_var="sample_rate", max_cat=10)
            loudness_fig, _ = self.audio_plotter.plot_violin(cont_var="loudness", max_cat=10)
            duration_fig, _ = self.audio_plotter.plot_violin(cont_var="duration", max_cat=10)
        else:
            sample_rate_fig, _ = self.audio_plotter.plot_scatter("sample_rate")
            loudness_fig, _ = self.audio_plotter.plot_scatter("loudness")
            duration_fig, _ = self.audio_plotter.plot_scatter("duration")

        current_y = self.pdf.get_y()
        self.insert_graph(sample_rate_fig, 6, 5, 2.2, 4.2, x=15)
        self.pdf.set_y(current_y)
        self.insert_graph(loudness_fig, 5.5, 5, 2.2, 4.2, x=(self.pdf.epw * 0.70) - 20)
        self.insert_graph(duration_fig, 5.5, 5, 2.2, 4.2, x=15)

        self.pdf.add_page()
        self.pdf.cell(txt="Most Important Features", border="B", w=0)
        self.pdf.start_section("Most important audio features", level=1)
        self.pdf.ln(10)
        self.pdf.set_font(*self.body)
        self.pdf.multi_cell(txt="Below shows a heatmap of the top 10 key openSmile audio features according to three "
                                "feature extraction methods. Data has been scaled so a value of 1 means the feature is "
                                "the most important, and a value of 0 means the feature is the least important.", w=0)
        self.pdf.ln(5)
        self.pdf.set_font(*self.bold_body)
        current_y = self.pdf.get_y()
        self.insert_graph(self.data_builder.plot_feature_ranking_heatmap(audio=True), 10, 8, 1.1, 2.2, x=0)
        after_fig_y = self.pdf.get_y()
        self.pdf.set_y(current_y)
        self.pdf.x += 20
        self.pdf.multi_cell(txt="For each feature extraction method (column), importance metrics have been scaled\n"
                                "between 0 and 1 such that 1 = most important feature, 0 = least "
                                "important "
                            , border=1, align="C", w=self.pdf.epw * 0.75)
        self.pdf.set_y(after_fig_y)
        self.pdf.set_font(*self.large_title)
        self.pdf.cell(txt="Baseline Model Performance", border="B", w=0)
        self.pdf.start_section("Baseline audio model performance", level=1)
        self.pdf.ln(10)
        self.pdf.set_font(*self.body)

        self.add_baseline_models(audio=True)

        pca_fig = self.data_builder.plot_pca(audio=True)
        if pca_fig is not None:
            self.pdf.add_page()
            self.pdf.set_font(*self.large_title)
            self.pdf.cell(txt="Principal Component Analysis", w=0, border="B")
            self.pdf.start_section("Principal component analysis of audio features", level=1)
            self.pdf.ln(10)
            self.pdf.set_font(*self.body)
            self.pdf.multi_cell(txt="The following graph shows a baseline model performance on an increasing number"
                                    "of estimated principal components of the audio features. This aims to give an idea"
                                    "as to the relative complexity of the machine learning task. If performance is high"
                                    "with a small number of components, but the task is expected to be complex, this"
                                    " suggests bias/spurious correlation within the dataset.", w=0)
            self.pdf.ln(10)
            self.insert_graph(pca_fig, width=7, height=6, p_width=1.2, p_height=2.5, x=35)
            self.pdf.ln(5)
            self.insert_graph(self.data_builder.plot_tsne(), width=7, height=6, p_width=1.2, p_height=2.5, x=35)

    def build_tabular_summary(self):
        """

        :return:
        """
        if self.audio_report:
            self.pdf.add_page()
        self.add_traffic_lights()

        self.pdf.add_page()

        self.set_title("Summary of Tabular Features")
        self.pdf.set_font(*self.large_title)
        self.pdf.start_section("Summary of tabular features")
        self.pdf.cell(txt="Most Important Features", border="B", w=0)
        self.pdf.start_section("Most important tabular features", level=1)
        self.pdf.ln(10)
        self.pdf.set_font(*self.body)
        self.pdf.multi_cell(txt="Below shows a heatmap of the top 10 key tabular features according to three "
                                "feature extraction methods. Data has been scaled so a value of 1 means the feature is "
                                "the most important, and a value of 0 means the feature is the least important.", w=0)
        self.pdf.ln(10)
        self.pdf.set_font(*self.bold_body)
        current_y = self.pdf.get_y()
        self.insert_graph(self.data_builder.plot_feature_ranking_heatmap(), 7, 6, 1.5, 3, x=self.pdf.epw / 6)
        end_of_fig_y = self.pdf.get_y()
        self.pdf.set_y(current_y - 5)
        self.pdf.x += 20
        self.pdf.multi_cell(txt="For each feature extraction method (column), importance metrics have been scaled\n"
                                "between 0 and 1 such that 1 = most important feature, 0 = least "
                                "important "
                            , border=1, align="C", w=self.pdf.epw * 0.75)
        self.pdf.set_y(end_of_fig_y)
        self.pdf.ln(10)

        self.pdf.set_font(*self.large_title)
        self.pdf.cell(txt="Baseline Model Performance", border="B", w=0)
        self.pdf.start_section("Baseline tabular model performance", level=1)
        self.pdf.ln(10)
        self.pdf.set_font(*self.body)

        self.add_baseline_models()

        pca_fig = self.data_builder.plot_pca()
        if pca_fig is not None:
            self.pdf.add_page()
            self.pdf.set_font(*self.large_title)
            self.pdf.cell(txt="Principal Component Analysis")
            self.pdf.start_section("Principal component analysis of tabular features", level=1)
            self.pdf.ln(10)
            self.pdf.set_font(*self.body)
            self.pdf.multi_cell(txt="The following graph shows a baseline model performance on an increasing number"
                                    "of estimated principal components of the tabular features. This aims to give an idea"
                                    "as to the relative complexity of the machine learning task. If performance is high"
                                    "with a small number of components, but the task is expected to be complex, this"
                                    "suggests bias/spurious correlation within the dataset.", w=0)
            self.pdf.ln(10)
            self.insert_graph(pca_fig, width=7, height=6, p_width=1, p_height=2)

    def set_title(self, txt: str) -> None:
        """
        Sets and outputs the title of the report.
        :return: None
        """
        self.pdf.set_font(*self.title_font)
        # Blank space to centre current position
        self.pdf.cell(txt=txt, new_x='LMARGIN')
        self.pdf.ln(20)

    def generate_table(self, df: pd.DataFrame, title: str = None,
                       word_length: int = 35) -> None:
        """
        Generates a table from values stored in df (IGNORING the index)
        :param word_length: int (default=38) maximum number of characters per cell
        :param df: pd.DataFrame containing data to be tabulated
        :param title: optional str title of the table (default=None)
        :return: None
        """

        column_labels = list(df.columns)
        cell_width = self.pdf.epw / (len(column_labels) + 2)
        cell_height = 6

        self.pdf.set_font(*self.body)
        self.pdf.set_line_width(0.3)

        # Define arguments for a blank cell to centre the table on the page
        blank_cell = (cell_width - 30, cell_height, "")

        self.pdf.cell(*blank_cell)

        # Insert the table headers
        border = "T,B,R,L"
        for index, header in enumerate(column_labels):
            if len(header) > word_length:
                header = header[:word_length]
            # Use case for function means every other column should be shorter
            if index % 2 == 0:
                width = cell_width + 40
            else:
                width = cell_width - 10
            self.pdf.cell(width, cell_height, txt=header, border=border, align='C')

        self.pdf.ln(cell_height)

        self.pdf.set_fill_color(r=166, g=214, b=255)
        normal_border = "L,R"
        start_border = "L,R,T"
        end_border = "L,R,B"
        full_border = "L,R,T,B"
        fill = True
        for row in range(0, df.shape[0]):
            if row == df.shape[0] - 1 and row == 0:
                border = full_border
            elif row == df.shape[0] - 1:
                border = end_border
            elif row == 0:
                border = start_border
            else:
                border = normal_border
            self.pdf.cell(*blank_cell)
            for col in range(0, df.shape[1]):
                content = df.iloc[row, col]
                if isinstance(content, (np.floating, float, int)):
                    content = str(format(content, ".2e"))
                else:
                    content = str(content)

                if len(content) > word_length:
                    content = content[:word_length]

                # Variable names are every other column in feature extract tables
                # So want their width to be longer
                if col % 2 == 0:
                    width = cell_width + 40
                else:
                    width = cell_width - 10

                self.pdf.cell(width, cell_height, txt=content, align='C', border=border, fill=fill)
            self.pdf.ln(cell_height)
            fill = not fill

    def build_data_overview(self) -> None:
        """
        Builds the section in the report containing an overview of the dataset.
        Reports on the number of missing/NAN observations, and reports on the number of
        constant variables that have been removed.
        :return: None
        """
        self.pdf.set_font(*self.large_title)
        self.pdf.cell(txt="Data Overview")
        self.pdf.start_section("Missing values and constant variables")
        self.pdf.ln(10)
        self.pdf.set_font(*self.small_title)
        self.pdf.cell(txt="Missing Values")
        self.pdf.ln(10)
        self.pdf.set_font(*self.body)

        number_of_nans = self.analyser.number_of_nans
        if self.analyser.number_of_nans > 0:
            self.pdf.multi_cell(txt=f"A total of {number_of_nans} missing values or NANs have been found in the "
                                    f"provided dataset. Observations containing NANs have been removed. Table "
                                    f"{self.table_number} below shows the number of missing values/NANs in each"
                                    f" variable:", w=0)
            self.pdf.ln(10)
            self.generate_correlation_tables(self.analyser.number_of_nans_df, title=f"Table {self.table_number}: number"
                                                                                    f" of missing values/NANs in each "
                                                                                    f"variable")
            self.pdf.ln(10)
        else:
            self.pdf.cell(txt="No missing values/NANs were present")
            self.pdf.ln(10)

        self.pdf.set_font(*self.small_title)
        self.pdf.cell(txt="Constant variables")
        self.pdf.ln(10)
        self.pdf.set_font(*self.body)

        number_of_constant_vars = len(self.analyser.constant_vars)
        if number_of_constant_vars > 0:
            self.pdf.multi_cell(txt=f"A total of {number_of_constant_vars} variables are constant across all "
                                    f"observations and thus have been removed from the dataset. The removed "
                                    f"constant variables can be found in Table {self.table_number} below:", w=0)
            self.pdf.ln(10)
            self.generate_table(pd.DataFrame(data=self.analyser.constant_vars, columns=["Constant variables"]),
                                title=f"Table {self.table_number} list of constant variables")
        else:
            self.pdf.cell(txt="There are no constant variables within the provided dataset.")

        self.pdf.ln(10)

    def get_df_edge_x(self, df, x=10, start_from_bottom=False):
        """
        Helper function to get the 'top' (or bottom is specified) x items from df

        :param df: dataframe to get top x from
        :param x: number of cols to get
        :param start_from_bottom: get the bottom x items in df
        :return: df containing the x items
        """
        if df is None:
            return None

        rows, cols = df.shape
        if rows <= x:
            return df

        if start_from_bottom:
            return df.iloc[:-x, :]

        return df.iloc[:x, :]

    def add_kw_description(self) -> None:
        """
        Adds the description of the kruskal_Wallis test to the current location in self.pdf
        :return: None
        """

        if self.data_builder.task == "classification":
            var_type = "categorical"
        else:
            var_type = "continuous"

        self.pdf.multi_cell(
            txt=f"As the provided output variable is {var_type}, the metric used in this section is the "
                f"kruskal-Wallis Test. This tests whether or not the distribution of a continuous variable is "
                f"different across categories. The null hypothesis is that the distributions are all the "
                f"same, so a rejection of the null hypothesis (i.e., a sufficiently low P-value) means that "
                f"at least one category is deemed to have a different distribution. Table {self.table_number} "
                f"below shows the ten variables that have the lowest P-value for the kruskal-Wallis test, "
                f"starting with the lowest:"
            , new_x='LMARGIN', w=0)
        self.pdf.ln(5)

    def add_correlation_description(self, dat_type: str) -> None:
        """
        Adds the description of the relevant correlation metrics used in the current location
        of the report. There are a total of three potential descriptions, which will be selected
        depending on the type of the output variable and the independent variable called dat_type

        :param dat_type: String, either "Cat" or "Cont" - the independent variable type
        :return: Nothing - adds text to the existing self.pdf data member
        """

        if dat_type == "Cont":
            txt_type = "continuous"
        else:
            txt_type = "categorical"

        self.pdf.set_font(*self.small_title)
        self.pdf.cell(txt=f"Correlations with {txt_type} variables", new_x='LMARGIN')
        self.pdf.ln(5)
        self.pdf.set_font(*self.body)

        if self.data_builder.task == "regression":
            if dat_type == "Cont":
                self.pdf.multi_cell(
                    txt=f"As the provided output variable is continuous, the metric used in this section is the "
                        f"Spearman Rank Correlation Coefficient. This metric ranges from -1 to 1, with -1 being the "
                        f"strongest possible negative correlation, 1 being the strongest possible positive correlation,"
                        f" and 0 implying no correlation. Table {self.table_number} below shows the ten continuous "
                        f"variables that correlate most strongly with the given output variable, starting with the "
                        f"strongest correlate in the first row of the table:"
                    , new_x='LMARGIN', w=0)
                self.pdf.ln(5)
            else:
                self.add_kw_description()
        else:
            if dat_type == "Cat":
                self.pdf.multi_cell(
                    txt=f"As the provided output variable is categorical, the metric used in this section is the bias "
                        f"adjusted Cramer's V association metric. This metric ranges from 0, meaning that the two "
                        f"variables are independent, to 1, meaning that one variable completely determines the other. "
                        f"Table {self.table_number} below shows the ten categorical variables that have the highest "
                        f"association with the given output variable, starting with the highest association in the "
                        f"first row of the table: "
                    , new_x='LMARGIN', w=0)
                self.pdf.ln(5)
            else:
                self.add_kw_description()

    def generate_corr_summary(self) -> None:
        """
        Generates the correlation summary of the report, which contains the top correlates of tabular independent
        variables with the output variable
        :return: None
        """
        self.pdf.add_page(orientation="P")
        self.pdf.start_section("Tabular Data: A Detailed Look")
        self.pdf.start_section("Tabular Data Correlation Overview", level=1)
        self.pdf.set_font(*self.large_title)
        self.pdf.cell(txt="Correlation Overview", new_x='LMARGIN')
        self.pdf.ln(10)
        self.pdf.set_font(*self.body)

        self.pdf.multi_cell(
            txt=f"This section highlights variables in the provided dataset "
                f"that most strongly correlate with the provided output variable ({self.output_var}). "
                f"It is important to consider whether these correlations make causal sense, as any machine learning "
                f"model will likely pick up on these correlations. If these correlations are spurious, then any learned"
                f" model will likely have a degree of bias.\n\nCorrelation rankings have been split into two sections, "
                f"depending on whether the given output variable and independent variable are categorical or "
                f"continuous. Full details of the measures used can be found at the end of this report.",
            new_x='LMARGIN', w=0)
        self.pdf.ln(5)

        spearman, cramers, kruskal = self.data_builder.get_correlation_metric()

        if self.data_builder.task == "classification":
            if not kruskal.empty:
                self.add_correlation_description("Cont")
                self.generate_correlation_tables(self.get_df_edge_x(kruskal),
                                                 title=f"Table {self.table_number}: Ten largest kruskal-Wallis "
                                                       f"Stats of {self.output_var} with continuous variables")
            if not cramers.empty:
                self.add_correlation_description("Cat")
                self.generate_correlation_tables(self.get_df_edge_x(cramers),
                                                 title=f"Table {self.table_number}: Ten largest Cramer's V values of "
                                                       f"{self.output_var} with categorical variables")
        else:
            if not spearman.empty:
                self.add_correlation_description("Cont")
                self.generate_correlation_tables(self.get_df_edge_x(spearman),
                                                 title=f"Table {self.table_number}: Ten strongest Spearman Correlation "
                                                       f"Coefficients of {self.output_var} with continuous "
                                                       f"variables")
            if not kruskal.empty:
                self.add_correlation_description("Cat")
                self.generate_correlation_tables(self.get_df_edge_x(kruskal),
                                                 title=f"Table {self.table_number}: Ten largest kruskal-Wallis Stats "
                                                       f"of {self.output_var} with categorical variables")

    def generate_correlation_tables(self, df: pd.DataFrame, title: str = None,
                                    word_length: int = 40) -> None:
        """
        Generates a table from the provided dataframe, and outputs it to the current position in self.pdf

        :param word_length: Int (default=40) maximum length for the variable names in the table
        :param title: Optional title for the table
        :param df: The dataframe that contains the data for the table
        :return: Nothing

        """
        """ Official reference at: https://pyfpdf.github.io/fpdf2/Maths.html """
        """ Draws a table of df at the current cursor position in self.pdf """
        column_labels = list(df.columns)
        cell_width = self.pdf.epw / (len(column_labels) + 2)
        cell_height = 6

        self.pdf.set_font(*self.body)
        self.pdf.set_line_width(0.3)

        if title is not None:
            # Insert blank cell to center table in PDF
            self.pdf.cell((cell_width / 2) - len(title) + 85, cell_height, txt="")
            self.pdf.cell(cell_width * len(column_labels), cell_height, txt=title)
            self.pdf.ln(cell_height)

        # Insert blank cell to center table in PDF
        self.pdf.cell((cell_width / 2) - 20, cell_height, txt="")
        self.pdf.cell(cell_width + 30, cell_height, txt="Variable name", border="B,R,T,L", align='C')

        for label in column_labels:
            # Enforce max 20 chars in table cell
            if len(label) > word_length:
                label = label[:word_length]
            self.pdf.cell(cell_width, cell_height, txt=label, new_x='RIGHT', border="B,R,T,L", align='C')
        self.pdf.ln(cell_height)

        self.pdf.set_font(*self.body)
        row_labels = list(df.index.values)
        self.pdf.set_fill_color(r=166, g=214, b=255)

        fill = True
        for row in range(0, df.shape[0]):
            if row == 0 and row == df.shape[0] - 1:
                border = "T,B,L,R"
            elif row == 0:
                border = "T,L,R"
            elif row == df.shape[0] - 1:
                border = "B,L,R"
            else:
                border = "L,R"

            txt = row_labels[row]

            if len(txt) > word_length:
                txt = txt[:word_length]

            self.pdf.cell((cell_width / 2) - 20, cell_height, txt="")
            self.pdf.cell(cell_width + 30, cell_height, fill=fill, txt=txt, new_x='RIGHT', align='C',
                          border=border)

            for col in range(0, len(column_labels)):
                if isinstance(df.iloc[row, col], str):
                    txt = df.iloc[row, col]
                # Not p-values
                elif col == 0:
                    txt = "{:.2f}".format(df.iloc[row, col])
                else:
                    txt = "{:.2e}".format(df.iloc[row, col])
                self.pdf.cell(cell_width, cell_height, fill=fill, txt=txt,
                              new_x='RIGHT', align='C', border=border)
            self.pdf.ln(cell_height)
            fill = not fill

        self.pdf.ln(cell_height)
        self.table_number += 1

    def insert_graph(self, fig: plt.Figure, width=9.0, height=6.0,
                     p_width=1.0, p_height=1.2, x=None) -> None:
        """
        Inserts a MaplotLib Figure at the current location in the PDF
        :param fig: MatPlotLib Figure object containing graph to be inserted
        :return: None
        """
        """ Code adapted from Official Documentation available at: https://pyfpdf.github.io/fpdf2/Maths.html"""
        fig.set_size_inches(width, height)
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        self.pdf.image(img, w=self.pdf.epw / p_width, h=self.pdf.eph / p_height, x=x)
        self.figure_number += 1
        # Close to prevent memory leaks
        plt.close(fig)

    def generate_visualisation(self) -> None:
        """
        Outputs the relevant graphs to the report for tabular independent variables. Graphs types are determined
        according to the output variable and independent variable(s). Graphs are plotted in order of the strength
        of association of the independent variable(s).
        :return: None
        """
        self.pdf.add_page(orientation="L")
        self.pdf.set_font(*self.large_title)
        self.pdf.cell(txt="Tabular Data Visualisation", new_x='LMARGIN')
        self.pdf.start_section("Tabular Data Visualisation", level=1)
        self.pdf.ln(10)
        self.pdf.set_font(*self.small_title)
        self.insert_visualisations()

        """

        spearman, cramers, kruskal = self.data_builder.get_correlation_metric()
        if self.data_builder.task == "classification":
            out_var_cat = True
        else:
            out_var_cat = False

        if not kruskal.empty:
            variables = list(kruskal.index)
            number_of_vars = len(variables)
            if number_of_vars <= 5:
                labels = [f"KW P-value: {s}" for s in
                          kruskal["P-value"]]
                max_plot_per_fig = 9
                start_index = 0
                end_index = min(max_plot_per_fig, number_of_vars)
                self.insert_graph(self.tabular_plotter.plot_violin_multi(variables[start_index: end_index],
                                                                         out_var_cat,
                                                                         texts=labels[start_index: end_index]))
                start_index = end_index
                end_index += max_plot_per_fig
                while start_index < number_of_vars:
                    self.insert_graph(self.tabular_plotter.plot_violin_multi(variables[start_index: end_index],
                                                                             out_var_cat,
                                                                             texts=labels[start_index: end_index]))
                    start_index = end_index
                    end_index += max_plot_per_fig
            else:
                labels = [f"KW P-value: {s}" for s in
                          kruskal["P-value"]]
                for index, var in enumerate(variables):
                    self.insert_graph(self.tabular_plotter.plot_violin(cat_var=self.output_var, cont_var=var,
                                                                       txt=labels[index]))
                    #  self.insert_graph(self.plotter.sns_violin(cat_var=self.plotter.output_var, cont_var=var,
                    #                                          txt=labels[index]))

        if not spearman.empty:
            variables = list(spearman.index)
            labels = [f"SCC: {s:.2f}" for s in
                      spearman["Spearman Correlation Coefficient"]]
            number_of_vars = len(variables)
            max_plot_per_fig = 9
            start_index = 0
            end_index = min(max_plot_per_fig, number_of_vars)
            self.insert_graph(self.tabular_plotter.plot_scatter_multi(variables[start_index: end_index],
                                                              texts=labels[start_index: end_index]))
            start_index = end_index
            end_index += max_plot_per_fig
            while start_index < number_of_vars:
                self.insert_graph(self.tabular_plotter.plot_scatter_multi(variables[start_index: end_index],
                                                                  texts=labels[start_index: end_index]))
                start_index = end_index
                end_index += max_plot_per_fig

        if not cramers.empty:
            variables = list(cramers.index)
            number_of_vars = len(variables)
            labels = [f"CV: {s:.2f}" for s in
                      cramers["Cramer's V"]]
            for index, var in enumerate(variables):
                # self.insert_graph(self.plotter.plot_heatmap_contingency(var, txt=labels[index]))
                self.insert_graph(self.tabular_plotter.sns_heatmap(var, txt=labels[index]))
        """

    def insert_visualisations(self, audio: bool = False):
        """

        :param audio: bool (default=False). Whether to insert audio or tabular feature graphs
        :return: None
        """
        self.pdf.ln(5)
        self.pdf.set_font(*self.bold_body)
        cat_vars = self.data_builder.get_top_features_by_type(audio=audio)
        cont_vars = self.data_builder.get_top_features_by_type(categorical=False, audio=audio)

        if audio:
            plotter = self.audio_plotter
        else:
            plotter = self.tabular_plotter

        for index, variable in enumerate(cont_vars):
            if index != 0:
                self.pdf.add_page(orientation="L")
            if self.data_builder.task == "regression":
                fig, ax = plotter.plot_scatter(indep_var=variable)
                model_1_txt = "Lasso"
                model_2_txt = "Elastic Net"
            else:
                fig, ax = plotter.plot_violin(cont_var=variable)
                model_1_txt = "XGBoost"
                model_2_txt = "Linear SVC"

            mi, stat, model_1, model_2 = self.data_builder.get_feature_importance(variable, audio=audio)
            self.pdf.multi_cell(txt=f"Mutual Information Relative Importance: {mi:.2f}     "
                                    f"Statistical Tests Relative Importance: {stat:.2f}\n"
                                    f"{model_1_txt} Relative Importance: {model_1:.2f}     "
                                    f"{model_2_txt} Relative Importance: {model_2:.2f}", align="C", w=0,
                                border=1)
            self.pdf.ln(1)
            self.insert_graph(fig)

        for index, variable in enumerate(cat_vars):
            heatmap=False
            if cont_vars:
                self.pdf.add_page(orientation="L")
            elif index != 0:
                self.pdf.add_page(orientation="L")
            if self.data_builder.task == "regression":
                fig, ax = plotter.plot_violin(cont_var=self.output_var, cat_var=variable)
                model_1_txt = "Lasso"
                model_2_txt = "Elastic Net"
            else:
                fig, ax = plotter.sns_heatmap(indep_var=variable)
                model_1_txt = "XGBoost"
                model_2_txt = "Linear SVC"
                heatmap = True

            mi, stat, model_1, model_2 = self.data_builder.get_feature_importance(variable, audio=audio)
            self.pdf.multi_cell(txt=f"Mutual Information Relative Importance: {mi:.2f}     "
                                    f"Statistical Tests Relative Importance: {stat:.2f}\n"
                                    f"{model_1_txt} Relative Importance: {model_1:.2f}     "
                                    f"{model_2_txt} Relative Importance: {model_2:.2f}", align="C", w=0,
                                border=1)
            self.pdf.ln(1)
            if heatmap:
                self.pdf.ln(5)
                self.insert_graph(fig, 11, 7, 1.1, 1.1, x=40)
            else:
                self.insert_graph(fig)

    def generate_audio_visuals(self) -> None:
        """
        Generates the audio feature visualisation section of the report
        Graphs the top 10 highest audio feature correlates
        :return: None
        """
        self.pdf.add_page(orientation='L')
        self.pdf.set_font(*self.large_title)
        self.pdf.cell(txt="Audio Feature Correlation Visualisations")
        self.pdf.start_section("Audio feature visualisation")
        self.pdf.ln(10)
        self.insert_visualisations(audio=True)
        """
        audio_s, audio_c, audio_k = self.data_builder.get_correlation_metric(audio=True)

        if not audio_k.empty:
            top_10_corr = self.get_df_edge_x(audio_k)
            variables = list(top_10_corr.index)
            labels = [f"KW P-value: {s:.2f}" for s in
                      top_10_corr["P-value"]]
            for index, var in enumerate(variables):
                self.insert_graph(self.audio_plotter.plot_violin(cont_var=variables[index], cat_var=self.output_var,
                                                           txt=labels[index]))
        if not audio_s.empty:
            top_10_corr = self.get_df_edge_x(audio_s)
            variables = list(top_10_corr.index)
            labels = [f"SCC: {s:.2f}" for s in
                      top_10_corr["Spearman Correlation Coefficient"]]
            self.insert_graph(self.audio_plotter.plot_scatter_multi(variables[0:9], texts=labels[0:9]))
            self.insert_graph(self.audio_plotter.plot_scatter(variables[9], txt=labels[9]))
        """

    def insert_image(self, image_path: str, width: int = 50, height: int = 0) -> None:
        """
        Inserts the image located at image_path in the current location in self.pdf
        :param height: image height
        :param width: image width
        :param image_path: File path to the image
        :return: None
        """
        self.pdf.image(image_path, w=width, h=height)

    def generate_audio_page(self) -> None:
        """
        Generates the Audio page in the report
        :return: None
        """
        self.pdf.add_page(orientation='P')
        self.pdf.set_font(*self.large_title)
        self.pdf.cell(txt="Audio Feature Correlations")
        self.pdf.start_section("Audio feature correlation measures")
        self.pdf.ln(10)
        self.pdf.set_font(*self.small_title)
        self.pdf.cell(txt="Sample rate")
        self.pdf.ln(10)
        self.pdf.set_font(*self.body)
        self.pdf.multi_cell(txt="Sample rate should never correlate with the output variable...", w=0)
        self.pdf.ln(10)

        """
        if self.audio_processor.same_sample_rate:
            self.pdf.multi_cell(txt="All sample rates are identical; no correlation with the output variable exists",
                                w=0)
            self.pdf.ln(10)

        else:
            if self.analyser.is_categorical(self.analyser.output_var):
                stat, pvalue = self.analyser.kruskal_wallis_h("sample_rate", "out_var")
                label = "kruskal-Wallis Statistic"
                graph = "Violin"
            else:
                stat, pvalue = self.analyser.spearman_correlation("sample_rate")
                label = "Spearman Correlation Coefficient"
                graph = "Scatter"

            sample_corr = {"sample_rate": [stat, pvalue]}
            df = pd.DataFrame(sample_corr, index=[label, "P-value"]).T
            self.generate_table(df, title=f"Table {self.table_number}: association of {self.analyser.output_var} with "
                                          f"audio sampling rate")
            self.pdf.ln(10)

            if graph == "Violin":
                self.plotter.plot_violin("sample_rate", txt=f"KW P-value: {pvalue:.2f}")
            else:
                self.plotter.plot_scatter("sample_rate", txt=f"SCC: {stat:.2f}")
        """
        self.pdf.set_font(*self.small_title)
        self.pdf.cell(txt="Correlations with audio features")
        self.pdf.ln(10)
        self.pdf.set_font(*self.body)
        self.pdf.multi_cell(txt="6374 audio features have been extracted using openSmile. This section provides"
                                " a detailed look at correlation metrics between these audio features and the output"
                                " variable.", w=0)
        self.pdf.ln(10)

        audio_s, audio_c, audio_k = self.data_builder.get_correlation_metric(audio=True)

        if not audio_s.empty:
            self.generate_correlation_tables(self.get_df_edge_x(audio_s, 50))
            self.pdf.ln(10)

        if not audio_c.empty:
            self.generate_correlation_tables(self.get_df_edge_x(audio_c, 50))
            self.pdf.ln(10)

        if not audio_k.empty:
            self.generate_correlation_tables(self.get_df_edge_x(audio_k, 50))
            self.pdf.ln(10)

        self.generate_audio_fe()

    def generate_tabular_fe(self) -> None:
        """
        Adds a section on extracted most relevant tabular features
        :return: None
        """
        self.pdf.add_page(orientation="P")
        self.pdf.set_font(*self.small_title)
        self.pdf.cell(txt="Tabular feature extraction")
        self.pdf.start_section("Tabular data feature extraction tables")
        self.pdf.ln(10)

        features = self.data_builder.get_model_features()
        spearman, cramers, kruskal = self.data_builder.get_correlation_metric()

        self.generate_table(features.head(50))
        self.pdf.ln(10)

        common_features = get_same_variables([cramers.head(50), kruskal.head(50), spearman.head(50)],
                                             features.head(50))
        common_features = pd.DataFrame(data=common_features, columns=["Common Features Across all Methods"])
        if len(common_features) != 0:
            self.generate_table(common_features, word_length=50)

    def generate_audio_fe(self) -> None:
        """
        Adds a section on extracted most relevant audio features
        :return: None
        """
        self.pdf.set_font(*self.small_title)
        self.pdf.cell(txt="Audio feature extraction")
        self.pdf.start_section("Audio data feature extraction data tables")
        self.pdf.ln(10)

        features = self.data_builder.get_model_features(audio=True)
        audio_s, audio_c, audio_k = self.data_builder.get_correlation_metric(audio=True)
        self.generate_table(features.head(50))
        self.pdf.ln(10)

        common_features = get_same_variables([self.get_df_edge_x(audio_s, x=100),
                                              self.get_df_edge_x(audio_k, x=100),
                                              self.get_df_edge_x(audio_c, x=100)],
                                             features.head(100))

        common_features = pd.DataFrame(data=common_features, columns=["Common Audio Features Across all Methods"])
        if len(common_features) != 0:
            self.generate_table(common_features.head(20), word_length=50)


"""
df = pd.DataFrame()
        mi = [0.67, 0.91, 0.67, 0.53, 0.68, 0.73, 0.42, 0.78, 0.68, 0.55]
        rf = [1, 0.78, 1, 0.67, 0.78, 0.78, 1, 0.56, 0.89, 0.89]
        svc = [0.59, 0.47, 0.47, 0.88, 0.59, 0.53, 0.59, 0.59, 0.35, 0.47]
        avg = [0.75, 0.72, 0.71, 0.69, 0.68, 0.68, 0.67, 0.64, 0.64, 0.64]

        df["Mutual Information"] = mi
        df["Random Forest"] = rf
        df["Linear SVC"] = svc
        df["Average"] = avg

        xlabels = ["pcm_fftMag_spectralRollOff25.0_sma_flatness", "mfcc_sma[2]_quartile1",
                   "mfcc_sma[2]_amean", "mfcc_sma[11]_rqmean", "logHNR_sma_posamean",
                   "pcm_fftMag_spectralVariance_sma_de_lpgain", "audspecRasta_lengthL1norm_sma_de_lpc2",
                   "voicingFinalUnclipped_sma_de_iqr1-2", "pcm_RMSenergy_sma_lpc1",
                   "audspec_lengthL1norm_sma_de_lpc1"]

        ylabels = ["Mutual Info.", "Rand. Forest", "Linear SVC", "Average"]

        fig, ax = plt.subplots()
        sns.heatmap(df, annot=True, ax=ax, vmin=0, vmax=1, linewidth=0.5, xticklabels=ylabels,
                    yticklabels=xlabels, cmap="Greens")

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        fig.tight_layout()
        self.insert_graph(fig, 10, 6, 1.2, 3, x=0)
        
            def insert_graph(self, fig: plt.Figure, width=11.69, height=7,
                     p_width=1.0, p_height=1.2, x=None) -> None:
"""
