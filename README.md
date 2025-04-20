# Plata Credit Risk Prediction Project

This project analyzes credit risk using LendingClub data.
The analysis, modeling, and deployment simulation are documented in the following Jupyter Notebooks:

1.  **`submission/1_Analytics.ipynb`**: Contains the Exploratory Data Analysis (EDA) and initial data processing steps.
2.  **`submission/2_Modelling.ipynb`**: Details the feature engineering, model training, and evaluation process.
3.  **`submission/3_Deployment.ipynb`**: Demonstrates how to load the trained model artifact and make predictions on new data.

## Running the Notebooks

-   **`1_Analytics.ipynb` and `2_Modelling.ipynb`**: These notebooks include saved outputs from previous runs. You do not need to re-run them to see the results. If you choose to run them, please ensure the file path pointing to the dataset (`accepted_2007_to_2018Q4.csv.gz`) is correctly configured within the notebook.

-   **`3_Deployment.ipynb`**: This notebook simulates the deployment of the model. You can run this notebook to make predictions on sample data (either hardcoded or from `data/sample_applications.csv`). Please verify and update the artifact path (`ARTIFACT_PATH`) and the sample data path (`SAMPLE_CSV_PATH`) within the notebook if necessary before running. 