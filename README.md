# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This portfolio project contains a Python package designed to predict customer churn in the credit card industry. As a showcase of both coding and data science expertise, the package is built following PEP8 coding standards and incorporates best practices in software engineering, such as modularity, comprehensive documentation, and thorough testing.

## Objectives
This project aims to demonstrate the following competencies:

- Writing modular and maintainable code.
- Adherence to PEP8 coding standards.
- Implementation and execution of unit tests.
- Utilizing logging for debugging and monitoring.
- Creating a Command Line Interface (CLI) for the Python package.

## Files and Data Description
Overview of the folders, files, and data present in the root directory.

### Files in the Root Directory:
- **churn_library.py:** Library of functions for identifying customers likely to churn.
- **churn_notebook.ipynb:** Original code that was refactored into `churn_library.py`.
- **churn_script_logging_and_tests.py:** Tests and logging functionalities for `churn_library.py`.
- **constants.py:** Constants utilized in `churn_library.py`.
- **Dockerfile:** Instructions for building the Docker image required to run the project.
- **Guide.ipynb:** Getting started guide and troubleshooting tips.
- **README.md:** Project overview and setup instructions.
- **requirements.txt:** List of dependencies and libraries required to run the project.

### Folders in the Root Directory:
- **data**: Dataset used for the project.
- **images**: Visualizations generated during the data science process.
  - **eda**: Visualizations generated during the Exploratory Data Analysis (EDA) process.
  - **results**: Visualizations of the results obtained from machine learning models.
- **logs**: Log files generated during testing of `churn_library.py`.
- **models**: Best-performing models saved during the data science process.

### Files in the Data Folder:
- **bank_data.csv:** Dataset containing customer churn information.

### Files in the Image/EDA Folder:
- **churn_distribution.png**: Visualization showcasing the distribution of customer churn.
- **customer_age_distribution.png**: Histogram displaying the distribution of customers' ages.
- **heatmap.png**: A heatmap showing the correlation matrix of the different features in the dataset.
- **marital_status_distribution.png**: Bar chart displaying the distribution of customers based on their marital status.
- **total_transaction_distribution.png**: Visualization depicting the distribution of the total transactions made by customers.

### Files in the Image/results Folder:
- **feature_importances.png**: Visual representation highlighting the importance of each feature used in the machine learning models.
- **logistic_results.png**: Image containing the classification report for the Logistic Regression model on both training and testing sets.
- **rf_results.png**: Image featuring the classification report for the Random Forest modelon both training and testing sets.
- **roc_curve_result.png**: Plot displaying the Receiver Operating Characteristic (ROC) curves for evaluating the performance of the trained models.

### Files in the Logs Folder:
- **churn_library.log**: Text file that records various runtime events, errors, and other information related to the execution of the `churn_library.py` script.

### Files in the Models Folder:
- **logistic_model.pkl**: Serialized logistic regression model trained to predict customer churn, saved in Python's pickle format.
- **rfc_model.pkl**: Serialized Random Forest Classifier model trained to predict customer churn, also saved in Python's pickle format.


## Running the Project

This section provides guidelines for building and running the Docker image, as well as for executing the project's Python scripts.

### Building and Running the Docker Image

1. Open your terminal and navigate to the project's root directory, where the `Dockerfile` is located.

2. Build the Docker image by running the following command:
    ```bash
    docker build -t predict_customer_churn_image .
    ```

3. After the image has been built successfully, you can run a Docker container based on this image with a specified name:
    ```bash
    docker run -it --name predict_customer_churn_container --rm predict_customer_churn_image
    ```

Note: The `--name` flag assigns a name to the running container, which is `predict_customer_churn_container` in this example. The container will provide an interactive shell. From there, you can navigate to the appropriate directories within the container to run the Python scripts or notebooks.

### Running Scripts with IPython

#### churn_library.py
To run the `churn_library.py` script, follow these steps:

1. Open your terminal and navigate to the project's root directory.

2. Start IPython and run the script by entering the following command:
    ```bash
    ipython churn_library.py
    ```

#### churn_script_logging_and_tests.py
To run the `churn_script_logging_and_tests.py` script, you'll also want to be in the project's root directory.

1. Make sure you're in the terminal and start IPython with the script by entering:
    ```bash
    ipython churn_script_logging_and_tests.py
    ```

By following these steps, you can successfully build the Docker container, and run the `churn_library.py` and `churn_script_logging_and_tests.py` scripts using IPython.


