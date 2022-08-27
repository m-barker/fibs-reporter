import math

from fpdf import FPDF
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl
import numpy as np

import config
from Plotter import Plotter
from Data_builder import DataBuilder

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
        title = f"FIBS automated data report"
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
            self.pdf.set_x(30)
            self.pdf.multi_cell(txt=f"Lasso Regression R-Squared: {round(lasso_r2, 2)}\n"
                                    f"Lasso Regression RMSE: {round(lasso_rmse, 2)}\n"
                                    f"Predict mean of y_train RMSE: {round(mean_rmse, 2)}",
                                w=self.pdf.epw / 2.6, fill=True)
            self.pdf.set_y(current_y)
            self.pdf.set_x((self.pdf.epw * 0.70))
            self.pdf.multi_cell(txt=f"Elastic Net R-Squared: {round(elastic_r2, 2)}\n"
                                    f"Elastic Net RMSE: {round(elastic_rmse, 2)}\n"
                                    f"Predict mean of y_train RMSE: {round(mean_rmse, 2)}", w=self.pdf.epw / 2.6,
                                fill=True)
            graph_1, graph_2 = self.data_builder.get_model_performance_graphs(audio=audio)

        current_y = self.pdf.get_y()
        self.insert_graph(graph_1, 5.5, 4.5, 2, 3.5, x=10)
        self.pdf.set_y(current_y)
        self.insert_graph(graph_2, 5.5, 4.5, 2, 3.5, x=(self.pdf.epw * 0.70) - 20)

        ################################ Add class/residual graphs ####################################################

        if audio:
            data_type = "Audio"
        else:
            data_type = "Tabular"

        self.pdf.set_font(*self.large_title)

        if self.data_builder.task == "classification":
            self.pdf.add_page(orientation="P")
            self.pdf.start_section(f"{data_type} baseline models performance by class", level=1)
            self.pdf.cell(txt=f"{data_type} baseline models performance by class", border="B", w=0)
            self.pdf.ln(10)
            fig_1, fig_2 = self.data_builder.get_model_performance_graphs(audio=audio)
            self.insert_graph(fig_1, width=7, height=6, p_width=1.2, p_height=2.5, x=35)
            self.pdf.ln(10)
            self.insert_graph(fig_2, width=7, height=6, p_width=1.2, p_height=2.5, x=35)

    def insert_traffic_colour(self, colour: str):
        """
        Inserts the traffic light score coloured text
        :param colour: str {red, amber, green} colour of the text
        :return: None
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
        Adds the traffic light score section of the report
        :param audio: bool, (default=False), whether the section is for audio or tabular data
        :return: None
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
            self.pdf.multi_cell(txt=f"{extra_audio_txt}There are strong relationships between dataset features "
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
            self.pdf.multi_cell(txt=f"{extra_audio_txt}There are moderately strong "
                                    f"relationships between dataset features and the output variable. "
                                    f"This does not necessarily mean that these relationships are spurious, but it "
                                    f"warrants investigation to verify whether the identified most important variables "
                                    f"are expected to be correlates with the output variable.", w=0)

        elif overall_score == "red" and model_score == "amber" and feature_score == "red":
            self.insert_traffic_colour("red")
            self.pdf.set_font(*self.body)
            self.pdf.multi_cell(txt=f"{extra_audio_txt}There are moderately strong "
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
                                    "model. There may also be no relevant features in the dataset. However, feature "
                                    "importance ratings and visualisation plots should still be checked, in the context "
                                    "of poor model performance, as there still may be some weak sources of spurious "
                                    "correlation.", w=0)

        if audio:
            self.pdf.ln(15)
            self.pdf.set_font(*self.bold_body)
            self.pdf.multi_cell(txt="Audio simple feature bias rating is: ", w=64, new_y="LAST")

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

        self.pdf.ln(10)
        self.pdf.set_font(*self.bold_body)
        self.pdf.multi_cell(txt=f"Disclaimer: the scores presented here summarise the sources of potential "
                                f"spurious correlation detected by FIBS. The feature importance section of the report"
                                f" considers the top ten most important features. For datasets with a large number of "
                                f"features, or datasets containing audio data, some sources of spurious correlation may "
                                f"be missed, and so "
                                f"there may be spurious relationships within "
                                f"your dataset that FIBS has been unable to detect.", w=0)
        self.pdf.ln(10)
        self.pdf.set_text_color(r=0, g=0, b=0)

    def build_audio_summary(self):
        """
        Builds the summary page for audio data
        :return: None
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
        self.pdf.multi_cell(txt="Below shows a heatmap of the top 10 key openSmile audio features according to four "
                                "feature extraction methods. Data has been scaled for each method so a value of 1 means the feature is "
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
                                    " as to the relative complexity of the machine learning task. If performance is high"
                                    " with a small number of components, but the task is expected to be complex, this"
                                    " suggests bias/spurious correlation within the dataset.", w=0)
            self.pdf.ln(10)
            self.insert_graph(pca_fig, width=7, height=6, p_width=1.2, p_height=2.5, x=35)
            self.pdf.ln(5)
            self.insert_graph(self.data_builder.plot_tsne(), width=7, height=6, p_width=1.2, p_height=2.5, x=35)

    def build_tabular_summary(self):
        """
        Builds the summary page for tabular data
        :return: None
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
        self.pdf.multi_cell(txt="Below shows a heatmap of the top 10 key tabular features according to four "
                                "feature extraction methods. Data has been scaled for each method so a value of 1 means the feature is "
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

    def insert_graph(self, fig: plt.Figure, width=9.0, height=6.0,
                     p_width=1.0, p_height=1.2, x=None) -> None:
        """
        Inserts a MaplotLib Figure at the current location in the PDF
        :param x: (optional) offset of graph x position in report
        :param p_width: float (default = 1.0) what to divide the page width by to scale image
        :param p_height: float (default = 1.2) what to divide the page height by to scale image
        :param height: float (default=6.0) matplotlib figure height parameter
        :param width: float (default=9.0) matplotlib figure width parameter
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

    def insert_image(self, image_path: str, width: int = 50, height: int = 0) -> None:
        """
        Inserts the image located at image_path in the current location in self.pdf
        :param height: image height
        :param width: image width
        :param image_path: File path to the image
        :return: None
        """
        self.pdf.image(image_path, w=width, h=height)
