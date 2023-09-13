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

#import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from PIL import Image
#from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

# import constants
from constants import (
    file_path, eda_images_path,
    category_list_constant, response_constant,
    keep_cols, results_images_path
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


def perform_feature_engineering(dataframe, response=response_constant):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument
              that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    target = dataframe['Churn']
    features = pd.DataFrame()
    dataframe = encoder_helper(
        dataframe,
        category_list_constant,
        response=response)
    features[keep_cols] = dataframe[keep_cols]
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.3, random_state=42)

    return features_train, features_test, target_train, target_test


def train_models(features_train, features_test, target_train, target_test):
    '''
    train, store model results: images + scores, and store models
    input:
              features_train: features training data
              features_test: features testing data
              target_train: target training data
              target_test: target testing data
    output:
              None
    '''
    # train models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    print("Training rfc...")
    cv_rfc.fit(features_train, target_train)
    print("Training lrc...")
    lrc.fit(features_train, target_train)

    # Get the predictions
    print("Getting predictions...")
    # Randon forest
    target_train_preds_rf = cv_rfc.best_estimator_.predict(features_train)
    target_test_preds_rf = cv_rfc.best_estimator_.predict(features_test)
    # Logistic Regression
    target_train_preds_lr = lrc.predict(features_train)
    target_test_preds_lr = lrc.predict(features_test)
    print("...........................................")

    # Scores
    print('Logistic regression results:')
    print('Test results:')
    print(classification_report(target_test, target_test_preds_lr))
    print('Train results:')
    print(classification_report(target_train, target_train_preds_lr))
    print('Random forest results:')
    print('Test results:')
    print(classification_report(target_test, target_test_preds_rf))
    print('Train results:')
    print(classification_report(target_train, target_train_preds_rf))
    print("...........................................")

    # roc_curve_result.png
    print("Saving roc_curve_result.png...")
    # Create a new figure for the ROC curve
    plt.ioff()
    fig, ax_fig = plt.subplots(figsize=(15, 8))
    # Plot ROC curve for each model
    plot_roc_curve(lrc, features_test, target_test, ax=ax_fig,
                   alpha=0.8, name='Logistic Regression')
    plot_roc_curve(cv_rfc.best_estimator_, features_test, target_test,
                   ax=ax_fig, alpha=0.8, name='Random Forest')
    # Add title to the plot
    plt.title("ROC Curve")
    # Save the plot
    save_path = f"{results_images_path}roc_curve_result.png"
    plt.savefig(save_path)

    # Close the plot
    plt.close(fig)
    del fig
    del ax_fig

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # logistic_results.png and rf_results.png
    classification_report_image(target_train,
                                target_test,
                                target_train_preds_lr,
                                target_train_preds_rf,
                                target_test_preds_lr,
                                target_test_preds_rf)

    # feature_importances.png on cv_rfc
    feature_importance_plot(cv_rfc, features_test, results_images_path)

    print("Train models process completed!")

# results images


def classification_report_image(target_train,
                                target_test,
                                target_train_preds_lr,
                                target_train_preds_rf,
                                target_test_preds_lr,
                                target_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            target_train: training response values
            target_test:  test response values
            target_train_preds_lr: training predictions from logistic regression
            target_train_preds_rf: training predictions from random forest
            target_test_preds_lr: test predictions from logistic regression
            target_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # logistic_results.png
    print("Saving logistic_results.png...")
    plt.rc('figure', figsize=(10, 8))
    plt.text(
        0.01, 1.25,
        'Logistic Regression Train',
        {'fontsize': 10},
        fontproperties='monospace'
    )
    plt.text(
        0.01, 0.05,
        str(classification_report(target_train, target_train_preds_lr)),
        {'fontsize': 10},
        fontproperties='monospace'
    )
    plt.text(
        0.01, 0.6,
        'Logistic Regression Test',
        {'fontsize': 10},
        fontproperties='monospace'
    )
    plt.text(
        0.01, 0.7,
        str(classification_report(target_test, target_test_preds_lr)),
        {'fontsize': 10},
        fontproperties='monospace'
    )
    plt.axis('off')

    # Save the figure
    plt.savefig(f"{results_images_path}/logistic_results.png")

    # rf_results.png
    print("Saving rf_results.png...")
    plt.rc('figure', figsize=(10, 8))
    plt.text(
        0.01, 1.25,
        'Random Forest Train',
        {'fontsize': 10},
        fontproperties='monospace'
    )
    plt.text(
        0.01, 0.05,
        str(classification_report(target_test, target_test_preds_rf)),
        {'fontsize': 10},
        fontproperties='monospace'
    )
    plt.text(
        0.01, 0.6,
        'Random Forest Test',
        {'fontsize': 10},
        fontproperties='monospace'
    )
    plt.text(
        0.01, 0.7,
        str(classification_report(target_train, target_train_preds_rf)),
        {'fontsize': 10},
        fontproperties='monospace'
    )
    plt.axis('off')

    # Save the figure
    plt.savefig(f"{results_images_path}/rf_results.png")


def feature_importance_plot(model, feature_data, output_path):
    '''
    Creates and stores the feature importances in path.
    Inputs:
        model: model object containing feature_importances_
        feature_data: pandas DataFrame of X values
        output_path: path to store the figure
    Output:
        None
    '''
    print("Saving feature_importances.png...")

    # Create a figure to hold the SHAP plot
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(feature_data)
    plt.figure(figsize=(20, 5))
    shap.summary_plot(shap_values, feature_data, plot_type="bar",
                      show=False, plot_size=(20, 5))
    shap_plot_path = os.path.join(output_path, "shap_plot.png")
    plt.savefig(shap_plot_path)
    plt.close()

    # Create and Save Feature Importance plot
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [feature_data.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))
    plt.bar(range(feature_data.shape[1]), importances[indices])
    # Reduce font size for x-axis labels
    plt.xticks(range(feature_data.shape[1]), names, rotation=90, fontsize=8)
    # Increase the viewing area below the graph
    plt.subplots_adjust(bottom=0.3)
    # Add a title and ylabel
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    feature_importance_plot_path = os.path.join(
        output_path, "feature_importance_plot.png")
    plt.savefig(feature_importance_plot_path)

    # Automatically adjust spacing between subplots
    plt.tight_layout()

    plt.close()

    # Open the images
    img1 = Image.open(shap_plot_path)
    img2 = Image.open(feature_importance_plot_path)

    # Get dimensions
    img1_width, img1_height = img1.size
    img2_width, img2_height = img2.size

    # Create a new image with white background
    new_img = Image.new(
        "RGB",
        (max(
            img1_width,
            img2_width),
            img1_height +
            img2_height),
        "white")

    # Paste the images
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1_height))

    # Save the new image
    new_img.save(os.path.join(output_path, "feature_importances.png"))

    # Delete the temporary plots
    if os.path.exists(shap_plot_path):
        os.remove(shap_plot_path)
    if os.path.exists(feature_importance_plot_path):
        os.remove(feature_importance_plot_path)

    print("Feature importances saved successfully.")


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

    # Perform Feature Engineering
    print("Performing Feature Engineering...")
    features_train, features_test, target_train, target_test = perform_feature_engineering(
        churn_df)
    print("Shape of dataframes for training:")
    print(features_train.shape)
    print(features_test.shape)
    print(target_train.shape)
    print(target_test.shape)

    # Train models
    train_models(features_train, features_test, target_train, target_test)

    # End of process
    print("Process completed!")


if __name__ == '__main__':
    main()
