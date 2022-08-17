import math

import numpy as np
import pandas as pd
import opensmile
import audiofile

from pydub import AudioSegment
from src.Data_analyser import DataAnalyser
from scipy import stats
from src.utility import remove_nans_and_missing

import src.config as config


class AudioProcessor:
    def __init__(self, df: pd.DataFrame, output_var_name: str,
                 audio_file_path_col_name: str, analyser: DataAnalyser = None):
        """

        :param df: dataframe containing metadata and audio file links
        :param output_var_name: name of the data frame column that contains the output variable
        :param audio_file_path_col_name: name of the data frame column that contains the audio file paths
        :param analyser (default=None) Optional existing DataAnalyser class object
        """

        self.audio_analyser = None
        self.df = df
        self.output_var_name = output_var_name
        self.audio_file_path_col_name = audio_file_path_col_name
        self.same_sample_rate = False
        self.audio_features_ = None
        self.out_var_added = False

        self.analyser = None
        if analyser is not None:
            self.analyser = analyser

        if config.AUDIO_SIZE == "large":
            f_level = opensmile.FeatureLevel.Functionals
        else:
            f_level = opensmile.FeatureLevel.LowLevelDescriptors

        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                     feature_level=f_level)

        self.process_signal()

    def process_signal(self) -> None:
        """
        Iterates through each audio file observation and extracts audio features and sample rate.
        Stores features in self.audio_features
        :return: None
        """
        files = self.df[self.audio_file_path_col_name]
        data = []
        columns = None
        self.df = self.df.reset_index(drop=True)

        for index, file in enumerate(files):
            if index % 100 == 0:
                print(f"Completed {index}/{len(files)}")

            file_path = ""
            file_path += file

            features = self.smile.process_file(file_path)
            _, sample_rate = audiofile.read(file_path)

            pydub_audio = AudioSegment.from_file(file_path)
            loudness = pydub_audio.dBFS
            duration = pydub_audio.duration_seconds
            if math.isinf(loudness):
                loudness = 0

            audio_data = np.append(features.values[0], sample_rate)
            audio_data = np.append(audio_data, loudness)
            audio_data = np.append(audio_data, duration)
            data.append(audio_data)

            if index == 0:
                columns = features.columns
                columns = np.append(columns, "sample_rate")
                columns = np.append(columns, "loudness")
                columns = np.append(columns, "duration")

        data = pd.DataFrame(data=data, columns=columns)
        self.audio_features_ = data
        if self.analyser is not None and len(list(pd.unique(self.audio_features["sample_rate"]))) > 1:
            self.analyser.add_to_df("sample_rate", self.get_sample_rates())

        output_var = list(self.df[self.output_var_name])
        loudness = list(data["loudness"])
        loudness_corr = stats.spearmanr(a=output_var, b=loudness)

        if len(list(pd.unique(self.audio_features["sample_rate"]))) == 1:
            self.same_sample_rate = True

    def get_sample_rates(self) -> list:
        """
        Returns a list of sample rates for every observation

        :return: list of sample rates for each observation
        """
        return list(self.audio_features["sample_rate"])

    def get_audio_correlations(self) -> tuple:
        """
        Get the correlations between the output variable and all audio features.
        :return: Tuple (df, df, df) containing (spearman, cramersV, kruskal) correlations
        """
        if not self.out_var_added:
            self.audio_features_[self.output_var_name] = list(self.df[self.output_var_name])
            self.out_var_added = True
        self.audio_analyser = DataAnalyser(self.audio_features, self.output_var_name, remove_nans=True)

        return self.audio_analyser.get_output_correlations()

    @property
    def audio_features(self) -> pd.DataFrame:
        """
        Gets a dataFrame of audio features along with the output_var, if they have been extracted
        :return: pd.Dataframe of audio features
        """

        if self.audio_features_ is not None:
            if not self.out_var_added:
                self.audio_features_[self.output_var_name] = list(self.df[self.output_var_name])
                self.out_var_added = True
            return self.audio_features_
        else:
            raise ValueError("Error, audio features have not been extracted")



