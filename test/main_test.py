import argparse
import pandas as pd
import numpy as np

from src.main import validate_and_init
from src.main import get_df


def test_duplicate_headers():
    csv_file_name = "test\\csvs\\duplicate_headers.csv"
    error_code = ""
    try:
        get_df(csv_file_name, "test")
    except ValueError as error:
        error_code = str(error)

    assert error_code == f"Error - duplicate variable names are present in {csv_file_name}"


def test_non_existant_csv():
    csv_file_name = "test\\csvs\\duplicate_headerss.csv"
    error_code = ""
    try:
        get_df(csv_file_name, "test")
    except FileNotFoundError as error:
        error_code = str(error)

    assert error_code == "[Errno 2] No such file or directory: 'test\\\\csvs\\\\duplicate_headerss.csv'"


def test_incorrect_output_var():
    csv_file_name = "test\\csvs\\output_var.csv"
    error_code = ""
    try:
        get_df(csv_file_name, "test2")
    except ValueError as error:
        error_code = str(error)

    assert error_code == (f"Error - no column in provided csv file called test2"
                          f"Column names in provided file are: ['var_1', 'test']")


def test_constant_output_var():
    csv_file_name = "test\\csvs\\constant_output.csv"
    error_code = ""
    try:
        get_df(csv_file_name, "output")
    except ValueError as error:
        error_code = str(error)

    assert error_code == f"Error - all output variable observations are identical"


def test_one_var():
    csv_file_name = "test\\csvs\\one_var.csv"
    error_code = ""
    try:
        get_df(csv_file_name, "output")
    except ValueError as error:
        error_code = str(error)

    assert error_code == f"Error, only one variable is present in {csv_file_name}"


def test_few_observations():
    csv_file_name = "test\\csvs\\few_observations.csv"
    error_code = ""
    try:
        get_df(csv_file_name, "output")
    except ValueError as error:
        error_code = str(error)

    assert error_code == (f"Error, a minimum of 20 observations are required to use the software."
                          f"\nProvided CSV file contains 9 observations.")


def test_all_const():
    csv_file_name = "test\\csvs\\all_const.csv"
    error_code = ""
    try:
        get_df(csv_file_name, "output")
    except ValueError as error:
        error_code = str(error)

    assert error_code == f"Error, all independent variables in {csv_file_name} are constant"


def test_no_audio_col():
    csv_file_name = "test\\csvs\\no_audio.csv"
    error_code = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_filepath")
    parser.add_argument("output_var_name")
    parser.add_argument("-audio", "--audio_file_col_name")

    try:
        validate_and_init(parser.parse_args([csv_file_name, "output", "-audio", "filename"]))
    except ValueError as error:
        error_code = str(error)

    assert error_code == (f"Error, no audio file column called filename "
                          f"exists in csv file {csv_file_name}")


def test_negative_max_cat():
    csv_file_name = "test\\csvs\\no_audio.csv"
    error_code = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_filepath")
    parser.add_argument("output_var_name")
    parser.add_argument("-audio", "--audio_file_col_name")
    parser.add_argument("--max_categories", "-max_cat", type=int)
    parser.add_argument("--test", type=str)

    try:
        validate_and_init(parser.parse_args([csv_file_name, "output", "-max_cat", "-1"]))
    except ValueError as error:
        error_code = str(error)

    assert error_code == "Error with --max_categories: maximum categories must be a positive integer"


def test_invalid_audio_size():
    csv_file_name = "test\\csvs\\no_audio.csv"
    error_code = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_filepath")
    parser.add_argument("output_var_name")
    parser.add_argument("-audio", "--audio_file_col_name")
    parser.add_argument("--max_categories", "-max_cat", type=int)
    parser.add_argument("--test", type=str)
    parser.add_argument("--audio_size", type=str)

    parser.add_argument("--report_name", type=str)

    try:
        validate_and_init(parser.parse_args([csv_file_name, "output", "--audio_size", "medium"]))
    except ValueError as error:
        error_code = str(error)

    assert error_code == "Error with --audio_size: argument must be equal to one of: {small, large}"

def test_invalid_test_csv():
    csv_file_name = "test\\csvs\\no_audio.csv"
    test_csv_file_name = "test\\csvs\\test.csv"
    error_code = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_filepath")
    parser.add_argument("output_var_name")
    parser.add_argument("-audio", "--audio_file_col_name")
    parser.add_argument("--max_categories", "-max_cat", type=int)
    parser.add_argument("--test", type=str)
    parser.add_argument("--audio_size", type=str)

    parser.add_argument("--report_name", type=str)

    try:
        validate_and_init(parser.parse_args([csv_file_name, "output", "--test", test_csv_file_name]))
    except ValueError as error:
        error_code = str(error)

    assert error_code == "Error, test csv does not have identical variable names to the main csv"
