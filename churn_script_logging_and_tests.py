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
import os
import logging
import churn_library as cl

# import constants
from constants import (
    file_path, eda_images_path, category_list_constant,
    response_constant
)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import
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


def test_eda(perform_eda, dataframe):
    '''
    test perform_eda function
    '''
    # Run the function
    eda_df = perform_eda(dataframe)

    # Test if new column 'Churn' is created in dataframe
    try:
        assert 'Churn' in eda_df.columns
        logging.info("Testing perform_eda: Churn column creation SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: Churn column was not created")
        raise err

    # Check if plots are saved to the images folder
    eda_plots = ['churn_distribution.png', 'customer_age_distribution.png',
                 'heatmap.png', 'marital_status_distribution.png',
                 'total_transaction_distribution.png']

    for plot_name in eda_plots:
        plot_path = os.path.join(eda_images_path, plot_name)
        try:
            assert os.path.isfile(plot_path)
            logging.info("Testing perform_eda: %s plot SUCCESS", plot_name)
        except AssertionError as err:
            logging.error(
                "Testing perform_eda: %s plot was not created", plot_name)
            raise err


def test_encoder_helper(encoder_helper, dataframe):
    '''
    Test the encoder_helper function

    input:
            encoder_function: function to be tested
            dataframe: pandas DataFrame to test

    output:
            None
    '''
    try:
        # Apply the encoder_helper function
        encoded_df = encoder_helper(dataframe, category_list_constant,
                                    response=response_constant)

        # Check that the DataFrame returned is not None
        assert encoded_df is not None

        # Check that new columns were created
        for col in category_list_constant:
            assert f"{col}_{response_constant}" in encoded_df.columns

        logging.info("Testing encoder_helper: SUCCESS")

    except Exception as err:
        logging.error("Testing encoder_helper: %s", err)
        raise err

def test_perform_feature_engineering(perform_feature_engineering, dataframe):
    '''
    Test perform_feature_engineering function

    input:
        perform_feature_engineering: function to test
        dataframe: pandas DataFrame to test

    output:
        None
    '''
    try:
        # Apply the function
        features_train, features_test, target_train, target_test = \
            perform_feature_engineering(dataframe, response=response_constant)

        # Check that none of the splits are None
        assert features_train is not None
        assert features_test is not None
        assert target_train is not None
        assert target_test is not None

        # Check shapes of the splits
        assert features_train.shape[0] == target_train.shape[0]
        assert features_test.shape[0] == target_test.shape[0]

        logging.info("Testing perform_feature_engineering: SUCCESS")

    except Exception as err:
        logging.error("Testing perform_feature_engineering: %s", err)
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

    print("Testing perform_eda...")
    churn_df = cl.import_data(file_path)
    test_eda(cl.perform_eda, churn_df)

    print("Testing encoder_helper...")
    test_encoder_helper(cl.encoder_helper, churn_df)

    print("Testing perform_feature_engineering...")
    eda_df = cl.perform_eda(churn_df)
    test_perform_feature_engineering(cl.perform_feature_engineering, eda_df)


if __name__ == "__main__":
    main()
