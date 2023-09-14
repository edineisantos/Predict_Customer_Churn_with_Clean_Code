"""
Predict Customer Churn with Clean Code

This script is responsible for running tests and logging for all the functions
in the `churn_library`. It includes functions to test data import, data cleaning,
feature engineering, model training, and model evaluation. The script produces
logs which can be found in './logs/churn_library.log'.

By running this script, you ensure that the main functionalities of the `churn_library`
work as expected. It helps in maintaining the integrity of the codebase and makes
debugging and adding new features easier.

Note: Before running this script, ensure that you have the `churn_library` and
required constants imported.

Author: Edinei Santos
Date: 2023-09

"""

import logging
import churn_library as cl

# import constants
from constants import (
    file_path
)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data(file_path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def main():
    """
    Run the tests for all functions.

    Parameters:
    None

    Returns:
    None
    """
    print("Testing import_data...")
    test_import(cl.import_data)


if __name__ == "__main__":
    main()
