import argparse
import pandas as pd
import src

from src.Report_builder import ReportBuilder
from src.Argparser import configure_argparse
from src.Data_builder import DataBuilder
import src.config as config
import csv


def parse_inputs() -> argparse.Namespace:
    """
    Parses the command line inputs using configure_argparse function

    :return: argparse.Namespace containing the command line arguments
    """

    parser = argparse.ArgumentParser()
    args = configure_argparse(parser)
    return args


def get_df(csv_file_name: str, output_var_name: str) -> pd.DataFrame:
    """
    Loads the dataframe from the csv_file_name path.
    Will raise an error terminating the program if the csv_file cannot be loaded/is wrong type of file
    and if no column exists called output_var_name

    :param csv_file_name: String, filepath of CSV, either full filepath or relative from CD
    :param output_var_name: String, name of the output variable column
    :return: pd.Dataframe - DataFrame loaded from the provided csv file
    """
    csv_file = open(csv_file_name, "r")
    csv_file_reader = csv.reader(csv_file, delimiter=",")
    headers = []
    for row in csv_file_reader:
        headers = row
        break
    csv_file.close()

    if len(set(headers)) != len(headers):
        raise ValueError(f"Error - duplicate variable names are present in {csv_file_name}")

    df = pd.read_csv(csv_file_name)
    headers = list(df.columns)
    if output_var_name not in headers:
        raise ValueError(f"Error - no column in provided csv file called {output_var_name}"
                         f"Column names in provided file are: {headers}")

    if len(pd.unique(df[output_var_name])) == 1:
        raise ValueError("Error - all output variable observations are identical")

    if len(headers) == 1:
        raise ValueError(f"Error, only one variable is present in {csv_file_name}")

    if df.shape[0] < 20:
        raise ValueError(f"Error, a minimum of 20 observations are required to use the software."
                         f"\nProvided CSV file contains {df.shape[0]} observations.")

    check_for_all_const_df = df.drop([output_var_name], axis=1)
    all_const = True
    for variable in check_for_all_const_df:
        if len(pd.unique(check_for_all_const_df[variable])) != 1:
            all_const = False
            break

    if all_const:
        raise ValueError(f"Error, all independent variables in {csv_file_name} are constant")

    return df


def validate_and_init(args) -> tuple:
    """

    :return: tuple(main_df, test_df) test_df is None if test_csv is None
    """

    test_df = None
    main_df = get_df(args.csv_filepath, args.output_var_name)
    variables = list(main_df.columns)
    variables.sort()

    if args.audio_file_col_name is not None:
        if args.audio_file_col_name not in variables:
            raise ValueError(f"Error, no audio file column called {args.audio_col} "
                             f"exists in csv file {args.csv_filepath}")

    if args.test is not None:
        test_df = get_df(args.test, args.output_var_name)
        test_vars = list(test_df.columns)
        test_vars.sort()
        if test_vars != variables:
            raise ValueError("Error, test csv does not have identical variable names to the main csv")

    if args.max_categories is not None:
        if args.max_categories <= 0:
            raise ValueError("Error with --max_categories: maximum categories must be a positive integer")
        config.MAX_CAT = args.max_categories

    if args.report_name is not None:
        config.REPORT_NAME = args.report_name

    if args.audio_size is not None:
        if args.audio_size not in ["small", "large"]:
            raise ValueError("Error with --audio_size: argument must be equal to one of: {small, large}")
        config.AUDIO_SIZE = args.audio_size

    return main_df, test_df


def main() -> None:
    """
    Main executable file of the toolkit
    :return: None
    """

    config.define_constants()

    args = parse_inputs()
    output_var = args.output_var_name
    audio_col = None

    audio = False

    if args.max_categories is not None:
        max_cat = args.max_categories

    if args.audio_file_col_name is not None:
        audio = True
        audio_col = args.audio_file_col_name

    main_df, test_df = validate_and_init(args)

    db = DataBuilder(main_df, output_var, audio_col, test_df)

    report = ReportBuilder(db, audio=audio)


if __name__ == '__main__':
    main()

