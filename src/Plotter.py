import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from operator import add
from operator import truediv
from math import ceil

from src.utility import is_categorical


class Plotter:
    def __init__(self, df: pd.DataFrame, output_var: str):
        self.df = df
        self.output_var = output_var

        for var in self.df.columns:
            if var == self.output_var:
                continue
            if is_categorical(self.df, var):
                dummies = pd.get_dummies(self.df[var], drop_first=False)
                dummies.columns = [f"{var}_{cat}" for cat in dummies.columns]
                self.df.drop(var, axis=1, inplace=True)
                self.df = pd.concat([self.df, dummies], axis=1)

    def get_most_important_violin_cats(self, categories: list, cont_var: str, cat_var: str, max_cat) -> list:
        """
        Gets the 'most important' variables for a violin plot in terms of the categories that have
        the largest difference from their median to the overall continuous variable median

        :param cat_var: Name of the categorical variable
        :param categories: list of category names
        :param cont_var: name of the continuous variable
        :return: List of 50 'most important' categories
        """
        overall_median = self.df[cont_var].median()
        medians = [self.df[cont_var][self.df[cat_var] == cat].median() for cat in categories]
        median_deviations = [list(abs(overall_median - m) for m in medians)]
        cat_deviations = zip(categories, median_deviations)
        deviation_dict = dict(cat_deviations)
        df = pd.DataFrame(deviation_dict, index=categories)
        df.sort_values(by=[list(df.columns)[0]], inplace=True, ascending=False)
        top_x = df.iloc[:max_cat]
        return list(top_x.index)

    def plot_violin(self, cont_var: str, cat_var: str = "output_var", txt: str = None, max_cat = 20, **pltkw) -> tuple:
        """

        :param txt:
        :param cont_var:
        :param cat_var:
        :return:
        Documentation can be found at:
        https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
        """

        if cat_var == "output_var":
            cat_var = self.output_var

        data = []
        x_points = []
        categories = list(pd.unique(self.df[cat_var]))
        extra_title = ""
        overall_median = self.df[cont_var].median()
        if len(categories) > max_cat:
            categories = self.get_most_important_violin_cats(categories, cont_var, cat_var, max_cat=max_cat)
            extra_title = f"\n(most relevant {max_cat} categories displayed)"

        categories.sort()
        medians = []
        for index, category in enumerate(categories):
            values = self.df[self.df[cat_var] == category]
            data.append(list(values[cont_var]))
            medians.append(values[cont_var].median())
            x_points.append(index + 1)

        fig, ax = plt.subplots(**pltkw)
        ax.set_title(f"Violin plot of {cont_var} across {cat_var} {extra_title}")
        ax.set_ylabel(f"{cont_var}")
        ax.set_xlabel(f"{cat_var}")

        shortened_cats = []
        for cat in categories:
            if len(str(cat)) > 15:
                shortened_cats.append((str(cat))[:15])
            else:
                shortened_cats.append(cat)

        categories = shortened_cats

        if len(categories) > 15:
            ax.set_xticks(x_points, categories, rotation=30, horizontalalignment="right")
        else:
            ax.set_xticks(x_points, categories)

        plt.subplots_adjust(bottom=0.2)

        if txt is not None:
            self.add_correlation_and_highlight(txt, ax, 0, 0.95, 0.05, less_than=True,
                                               scientific_format=True)

        ax.violinplot(data, showmedians=True)
        ax.plot(np.arange(1, len(categories) + 1), medians, label="Category medians")
        ax.hlines(y=overall_median, xmin=1, xmax=len(categories), color='b', linestyle='--',
                  label=f"Overall {cont_var} median")
        ax.legend()
        return fig, ax

    def plot_scatter(self, indep_var: str, txt: str = None) -> tuple:
        """
        Creates a scatter between self.output_var and indep_var
        :param txt:
        :param indep_var:
        :return:
        """
        fig, ax = plt.subplots()
        ax.scatter(self.df[indep_var], self.df[self.output_var], s=5, alpha=0.5)
        ax.set_xlabel(indep_var)
        ax.set_ylabel(self.output_var)
        if txt is not None:
            self.add_correlation_and_highlight(txt, ax, 0.7, 0.8, 0.2)

        return fig, ax

    def get_relevant_heatmap_cols(self, cont_table: pd.DataFrame, column_names: list) -> list:
        """

        :param cont_table: Pandas DataFrame
        :param column_names: List
        :return: List
        """
        ranges = {}
        for column in column_names:
            max = cont_table[column].max()
            min = cont_table[column].min()
            range = max-min
            ranges[column] = [range]
        df = pd.DataFrame(ranges).T
        df.sort_values(by=[0], inplace=True, ascending=False)
        top_20 = df.iloc[:20]
        return list(top_20.index)

    def get_relevant_heatmap_rows(self, cont_table: pd.DataFrame, row_names: list) -> list:
        """

        :param cont_table:
        :param row_names:
        :return:
        """
        metric = {}
        for row in row_names:
            df = cont_table.drop(row)
            col_means = [df[col].mean() for col in cont_table.columns]
            diff_from_mean = [abs(cont_table.loc[row, col] - col_means[i]) for i, col in enumerate(cont_table.columns)]
            # mean diff from mean
            metric[row] = [sum(diff_from_mean)/len(list(cont_table.columns))]
        df = pd.DataFrame(metric).T
        df.sort_values(by=[0], inplace=True, ascending=False)
        top_20 = df.iloc[:20]
        return list(top_20.index)

    def plot_hist(self, indep_var: str, cat=0, **pltkw):
        """

        :param indep_var:
        :param cat:
        :return:
        """
        if cat == 0:
            cat_var = self.output_var
            cont_var = indep_var
        else:
            cat_var = indep_var
            cont_var = self.output_var

        categories = list(pd.unique(self.df[cat_var]))
        fig, axs = plt.subplots(len(categories) // 4 + 1, min(3, len(categories)), **pltkw)
        ax_num = 0
        row_num = 0
        col_num = 0
        maximum = self.df[cont_var].max()
        for index, category in enumerate(categories):
            data = self.df[self.df[cat_var] == category]
            data = data[cont_var]
            if len(categories) < 4:
                axs[ax_num].hist(data, density=True, range=(0, maximum), edgecolor='black')
                axs[ax_num].set_title(f"Histogram density of {cont_var} when {cat_var} is equal to {category}")
                axs[ax_num].set_xlabel(cont_var)
                axs[ax_num].set_ylabel("Prob. Density")
                ax_num += 1
            else:
                axs[row_num][col_num].hist(data, density=True, range=(0, maximum), edgecolor='black')
                axs[row_num][col_num].set_title(
                    f"Histogram density of {cont_var}\n when {cat_var} is equal to {category}")
                axs[row_num][col_num].set_xlabel(cont_var)
                axs[row_num][col_num].set_ylabel("Prob. Density")
                col_num = (col_num + 1) % 3
                if (index + 1) % 3 == 0 and index != 0:
                    row_num += 1

        return fig

    def plot_violin_multi(self, indep_vars: list, out_var_cat: bool = True,
                          plots_per_row: int = 3, texts: list = None) -> plt.Figure:
        """
        
        :param texts: (optional) String list containing additional text to display on each individual subplot
        :param indep_vars: List of independent variables. One plot will be generated for each variable
        :param out_var_cat: Bool (default=True) whether output variable is categoric or not 
        :param plots_per_row: Int (default=3), number of plots per row of the figure
        :return: MatPlotLib figure object
        """"""
        Documentation can be found at:
        https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
        """
        number_of_plots = len(indep_vars)
        # Dimensions of the figure
        n_rows = ceil(number_of_plots / plots_per_row)
        n_cols = min(number_of_plots, plots_per_row)
        if number_of_plots == 1:
            if out_var_cat:
                return self.plot_violin(indep_vars[0], txt=texts[0])
            else:
                return self.plot_violin(self.output_var, indep_vars[0], txt=texts[0])

        fig, axs = plt.subplots(n_rows, n_cols)
        fig.tight_layout()
        plt_number = 1
        axs = axs.flatten()
        cont_var = None
        cat_var = None
        for index, ax in enumerate(axs):
            if plt_number <= number_of_plots:
                data = []
                x_points = []

                if out_var_cat:
                    categories = list(pd.unique(self.df[self.output_var]))
                    cat_var = self.output_var
                    cont_var = indep_vars[index]
                else:
                    categories = list(pd.unique(self.df[indep_vars[index]]))
                    cat_var = indep_vars[index]
                    cont_var = self.output_var
                categories.sort()
                medians = []
                for i, category in enumerate(categories):
                    values = self.df[self.df[cat_var] == category]
                    data.append(list(values[cont_var]))
                    medians.append(values[cont_var].median())
                    x_points.append(i + 1)

                # ax.set_title(f"Violin plot of {cont_var} across {cat_var}")
                ax.set_ylabel(f"{cont_var}")
                ax.set_xlabel(f"{cat_var}")
                ax.set_xticks(x_points, categories, rotation=45)

                if texts is not None:
                    self.add_correlation_and_highlight(texts[index], ax, 0.7, 0.8, 0.05, less_than=True,
                                                       scientific_format=True)

                ax.violinplot(data, showmedians=True)
                ax.plot(np.arange(1, len(categories) + 1), medians, label="Category medians")

                plt_number += 1
            else:
                ax.remove()

        y = 1
        font_weight = "bold"
        if out_var_cat:
            fig.suptitle(f"Distributions of {cat_var} across all continuous variables", y=y, fontweight=font_weight)
        else:
            fig.suptitle(f"Distribution of {cont_var} across all categorical variables", y=y, fontweight=font_weight)

        return fig

    def plot_scatter_multi(self, indep_vars: list, plots_per_row: int = 3,
                           texts: list = None) -> plt.Figure:
        """

        :param texts: (optional) String list containing additional text to display on each individual subplot
        :param indep_vars: List of independent variables. One plot will be generated for each variable 
        :param plots_per_row: Int (default=3), number of plots per row of the figure
        :return: MatPlotLib figure object
        """"""
        Documentation can be found at:
        https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
        """

        number_of_plots = len(indep_vars)
        # Dimensions of the figure
        n_rows = ceil(number_of_plots / plots_per_row)
        n_cols = min(number_of_plots, plots_per_row)

        if number_of_plots == 1:
            fig = self.plot_scatter(indep_vars[0], txt=texts[0])
            return fig

        fig, axs = plt.subplots(n_rows, n_cols)
        fig.tight_layout()
        plt_number = 1
        axs = axs.flatten()
        for index, ax in enumerate(axs):
            if plt_number <= number_of_plots:
                ax.scatter(self.df[indep_vars[index]], self.df[self.output_var], s=1, alpha=0.2)

                if texts is not None:
                    self.add_correlation_and_highlight(texts[index], ax, 0.7, 0.8, 0.2)

                ax.set_xlabel(indep_vars[index])
                ax.set_ylabel(self.output_var)
                plt_number += 1
            else:
                ax.remove()

        fig.suptitle(f"Scatters of {self.output_var} for each continuous variable (SCC = Spearman Correlation"
                     f" Coefficient)", y=1, fontweight="bold")

        return fig

    def configure_heatmap(self, ax: plt.Axes, rows: int, cols: int, label: bool = True,
                          cont_table: pd.DataFrame = None) -> None:
        """
        Helper function that configures the axis for the contingency heatmap

        :param cols: Number of rows in the heatmap
        :param rows: Number of columns in the heatmap
        :param ax: MatPlotLib axis object
        :param label: Bool (default True) if True, labels the data in cont_table on the graph
        :param cont_table: Pandas DataFrame (default = None) contains the data of the heatmap
        :return: None
        """

        ax.set_xticks(np.arange(cols + 1) - .5, minor=True)
        ax.set_yticks(np.arange(rows + 1) - .5, minor=True)

        ax.grid(which="minor", color='w', linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        if label:
            for r in range(rows):
                for c in range(cols):
                    if cont_table.iloc[r, c] <= 50:
                        colour = "black"
                    else:
                        colour = "white"
                    if cont_table.iloc[r, c] == 100:
                        txt = "100"
                    else:
                        txt = str(cont_table.iloc[r, c].round(1))
                    text = ax.text(c, r, txt,
                                   ha="center", va="center", color=colour)

    def add_correlation_and_highlight(self, corr_txt: str, ax: plt.Axes,
                                      x_pos: float, y_pos: float, cut_off: float,
                                      less_than: bool = False,
                                      scientific_format: bool = False) -> None:
        """

        :param scientific_format: bool (default = False) if true, displays the numbers in the text in
               scientific format
        :param less_than: Bool (default = False), if true, highlight axis if correlation is less than cut_off
        :param cut_off: float, if correlation is greater than cutoff, highlight axis
        :param y_pos: float, y-position of the graph where the text should go
        :param x_pos: float, x-position of the graph where the text should go
        :param ax: MatPlotLib axis object to add the text and highlighting to
        :param corr_txt: str, Correlation text, last 4 chars are assumed to be the correlation/P value
        :return: None
        """

        # Last 4 digits of corr_txt is a number of the form x.xx
        highlight = False
        if less_than:
            if abs(float(corr_txt[corr_txt.find(":") + 1:])) < cut_off:
                highlight = True
        else:
            if abs(float(corr_txt[corr_txt.find(":") + 1:])) >= cut_off:
                highlight = True

        if highlight:
            ax.spines["top"].set_color("red")
            ax.spines["bottom"].set_color("red")
            ax.spines["left"].set_color("red")
            ax.spines["right"].set_color("red")

        if scientific_format:
            number = float(corr_txt[corr_txt.find(":") + 1:])
            words = corr_txt[:corr_txt.find(":") + 1]
            number = str(format(number, ".2e"))
            corr_txt = words + number

        ax.text(x_pos, y_pos, corr_txt, fontsize="small", transform=ax.transAxes,
                bbox={'facecolor': 'white'})

    def sns_heatmap(self, indep_var, txt=None):
        """

        :param indep_var:
        :return:
        """
        contingency_table = pd.crosstab(self.df[self.output_var], self.df[indep_var], normalize="index") * 100
        contingency_table = contingency_table.round(1)

        rows, cols = contingency_table.shape
        # Convert col names to string to prevent implicit conversions
        contingency_table.columns = [str(c) for c in list(contingency_table.columns)]

        # Max number of rows/cols that can be displayed is 20, so need to choose relevant ones
        if rows > 20:
            rows = self.get_relevant_heatmap_rows(contingency_table, contingency_table.index)
        else:
            rows = list(contingency_table.index)
        if cols > 20:
            cols = self.get_relevant_heatmap_cols(contingency_table, contingency_table.columns)
        else:
            cols = list(contingency_table.columns)
            cols = [str(c) for c in cols]

        contingency_table = contingency_table.loc[rows, cols]
        # Sort by index so that the y-axis and x-axis is in order
        try:
            contingency_table.columns = [int(c) for c in list(contingency_table.columns)]
        except ValueError:
            """ No action needed """

        labels = list(contingency_table.index)
        new_labels = []
        for label in labels:
            if len(str(label)) > 10:
                new_labels.append(str(label)[:10])
            else:
                new_labels.append(label)

        contingency_table[self.output_var] = new_labels
        contingency_table = contingency_table.set_index(self.output_var)

        contingency_table.sort_index(inplace=True)
        contingency_table.sort_index(inplace=True, axis=1)
        fig, ax = plt.subplots()
        sns.heatmap(contingency_table, annot=True, ax=ax, vmin=0, vmax=100, linewidth=0.5,
                    fmt=".2f", cmap="Greens")
        ax.set_title(f"Frequency distribution of {indep_var} across {self.output_var}\n Rows sum to 100 (unless"
                     f" there are more than 20 categories, then the most relevant\n rows/columns to display have been "
                     f"selected)")
        ax.set_xlabel(indep_var)

        if txt is not None:
            self.add_correlation_and_highlight(txt, ax, 0.0, 1.0, 0.1)
        return fig, ax

    def sns_violin(self, cont_var: str, cat_var: str = "output_var", txt: str = None):
        """

        :param cont_var:
        :param cat_var:
        :param txt:
        :return:
        """
        if cat_var == "output_var":
            cat_var = self.output_var

        fig, ax = plt.subplots()
        sns.violinplot(x=self.df[cat_var], y=self.df[cont_var])
        if txt is not None:
            self.add_correlation_and_highlight(txt, ax, 0, 0.95, 0.05, less_than=True,
                                               scientific_format=True)

        return fig
