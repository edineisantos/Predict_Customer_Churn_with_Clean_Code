"""
Predict Customer Churn with Clean Code

This script is a part of a project that aims to identify credit card customers
who are most likely to churn. The project as a whole adheres to best coding
practices and engineering standards, including PEP8.

This script can be run either interactively or from the command-line interface
(CLI). It serves as a starting point for running all the data science processes
required for customer churn prediction.

Dataset: The data used in this project is sourced from Kaggle.

Author: Edinei Santos
Date: 2023-09
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import (
    file_path, eda_images_path,
    category_list_constant, response_constant
)
sns.set()

# Set the QT_QPA_PLATFORM as offscreen
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''

    dataframe = pd.read_csv(pth)
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    # Transform 'Attrition_Flag' to 'Churn'
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Churn Distribution
    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.title("Churn Distribution")
    plt.savefig(f"{eda_images_path}churn_distribution.png")
    plt.close()

    # Customer Age Distribution
    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.title("Customer Age Distribution")
    plt.savefig(f"{eda_images_path}customer_age_distribution.png")
    plt.close()

    # Heatmap of correlations
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Correlation Heatmap")
    plt.savefig(f"{eda_images_path}heatmap.png")
    plt.close()

    # Marital Status Distribution
    plt.figure(figsize=(20, 10))
    dataframe['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.title("Marital Status Distribution")
    plt.savefig(f"{eda_images_path}marital_status_distribution.png")
    plt.close()

    # Total Transaction Distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.title("Total Transaction Distribution")
    plt.savefig(f"{eda_images_path}total_transaction_distribution.png")
    plt.close()


def encoder_helper(dataframe, category_list, response=response_constant):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category

    input:
        dataframe: pandas DataFrame
        category_list: list of columns that contain categorical features
        response: string of response name [optional argument that could
        be used for naming variables or index y column]

    output:
        DataFrame with new columns
    '''
    for category in category_list:
        category_groups = dataframe.groupby(category).mean()[response]
        new_column_name = f"{category}_{response}"

        dataframe[new_column_name] = dataframe[category].apply(
            lambda val, category_groups=category_groups: category_groups.loc[val])

    return dataframe


def main():
    """
    Run all the data science processes.

    Parameters:
    None

    Returns:
    None
    """

    # Import data
    churn_df = import_data(file_path)
    print("Dataframe shape:")
    print(churn_df.shape)
    print(type(churn_df))

    # Perform EDA
    print("Performing EDA...")
    perform_eda(churn_df)

    # Test encoder_helper
    encoded_df = encoder_helper(churn_df, category_list_constant)
    print(type(encoded_df))
    print(encoded_df.columns)

    # Print
    print("Process completed!")


if __name__ == '__main__':
    main()
