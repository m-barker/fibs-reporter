import argparse


def configure_argparse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Configures the command line options used when running main.py
    :param parser: argparse.ArgumentParser() object
    :return: Arguments
    """
    parser.add_argument("csv_filepath", help="filepath to csv file containing data to be analysed. Filepath "
                                             "should be the full filepath or relative from the current working "
                                             "directory", type=str)
    parser.add_argument("output_var_name", help="name of the output variable stored in the csv file. Name must "
                                                "exactly match the output variable column name in the csv file",
                        type=str)
    parser.add_argument("--audio_file_col_name", "-audio", help="name of the column in the csv file containing the "
                                                                "filepaths "
                                                                "to each observation's audio recording", type=str)

    parser.add_argument("--max_categories", "-max_cat", help="the maximum number of categories in a categorical "
                                                             "variable. By "
                                                             "default, any integer variable that has more than 50 "
                                                             "unique values "
                                                             "will never be deemed to be categorical", type=int)

    parser.add_argument("--test", help="filepath to a train_csv file - must contain the same variables as the "
                                       "other csv. Used to train baseline models. If not provided, the original"
                                       "csv is split 80/20 for train/test", type=str)

    parser.add_argument("--audio_size", help="{small, large} determines the number of audio features to extract. "
                                             "If large, uses", type=str)

    parser.add_argument("--report_name", help="Name of the report (default=report.pdf)", type=str)

    return parser.parse_args()
