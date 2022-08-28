# The Data FIBS Reporter
The Data **F**eature **I**mportance, **B**aseline-modeller and **S**purious correlation Reporter (**FIBS**) is an open-source software for automatic generation of a PDF report to highlight and visualise potential sources of spurious correlation within **any** given tabular or audio dataset stored as a Comma Separated Values (CSV) file. FIBS is run through one-command line command; **all of the calculations, model training, and report generation happen automatically**.

All that is required as input on the command line is the path to the CSV file containing the data, and the name of the output (dependent) variable within the dataset. The toolkit will automatically determine whether the task is regression or classification. Optionally, the toolkit can process and extract audio data, provided the name of the variable within the CSV that contains the audio file for each observation is specified,

## Key features that are generated automatically:
- A traffic light score for potential spurious correlations within the dataset
- Calculation of four different feature importance metrics to highlight the most important features within the given dataset
- Training and evaluation of two baseline models, including visualisation of model results
- Visuals of the most important features, with different visuals depending on the variable types
- Automatic determination of regression or classification task, resulting in different baseline models, feature extraction methods, and visualisations 
- Principal Component Analysis calculation and baseline model to estimate complexity within the dataset
- (Optionally) extract audio data features and run the above on these features
- Output all of the above in a PDF report with accompanying dynamic textual explanations

## Installation
The easiest way to install FIBS is by running the pip command:
```
pip install fibs-reporter
```

Note that FIBS requires Python ≥ V. 3.7

## Usage
It is assumed that the data to be analysed is contained within a CSV file, where each column contains a variable, and each row contains an observation. 

If the CSV file path is _path_to_my_csv_file_ and the output variable is called _target_ then the following command line command will output a report called _data_report.pdf_ in the current working directory:

```
fibs path_to_my_csv_file target --report_name data_report
```

If you wish to use an audio dataset, then a column in the CSV file must contain the file path to the raw WAV audio recording for each observation. If this column is called _filename_ then the following command line command would be used, where the other inputs are as above:

```
fibs path_to_csv_file target -audio filename --report_name data_report
```

Another command line argument, `--test` can be used to specify an additional CSV file which is treated as the test dataset. This CSV file must contain the same variables as the other CSV file, and is used to evaluate baseline model performance. Feature importance scores are calculated on the original CSV (training set) only. 

For example, if the test CSV is stored at _path_to_test_csv_, and the other arguments are as above in the audio case, then the following command line input would be used:

```
fibs path_to_train_csv target -audio filename --test path_to_test_csv --report_name data_report
```

There are a few other optional command line arguments. The full list of optional command line arguments is as follows:

`--help`, displays a full list and description of required and optional command line arguments

`--audio_file_col_name AUDIO_COLUMN_NAME` or `-audio AUDIO_COLUMN_NAME` used to specify the name of the column in the CSV file containing the filepaths to raw audio WAV recordings for each observation.

`--audio_size AUDIO_SIZE`, where `AUDIO_SIZE` must be one of either "small" or "large". Must be used in conjunction with the `-audio` command. Used to control the number of audio features extracted by [openSMILE](https://github.com/audeering/opensmile). By default, this is equal to "large", meaning that over 6000 audio features are extracted from each audio file. If "small" is specified, only 64 features will be extracted from each audio file.

`--test TEST_CSV_FILE_PATH` used to specify the filepath to a test CSV file. Test CSV file must contain identical column names to the original CSV file. Data in the test CSV file will be used to evaluate baseline model performance. If not provided, the original CSV is shuffled and split 80/20 for a train/test split.

`--report_name REPORT_NAME` used to name the PDF file output by the software. For example  
`--report_name data_report` would name the file 'data_report.pdf'

`--max_categories MAX_CATEGORIES` or `-max_cat MAX_CATEGORIES`. MAX_CATEGORIES must be a positive integer, and is used to specify the maximum number of categories in integer categorical variables. By default, any integer variable that has more than 50 unique values will be assumed to be a continuous variable.

## Acknowledgments
FIBS makes use of the following open-source Python packages:

- `audiofile` for extracting sample rate of audio recordings
- `fpdf2` for building the PDF report
- `matplotlib & seaborn` for creating graphs for report
- `numpy` for general data handling
- `openSMILE` for extracting complex audio features
- `pandas` for loading the CSV files and general data handling
- `pillow` for converting the graphs for images to use in the report
- `pydub` for extracting loudness and duration audio features
- `scipy` for calculating test statistics for feature ranking
- `scikit-learn` for variable encoding, model training and feature ranking, model evaluation, and PCA
- `xgboost` for XGBoost modelling and feature ranking



## License
The software has been released under a free for non-commercial use license. For details, please consult the LICENSE.txt file.


