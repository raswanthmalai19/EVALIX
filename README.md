# Credit Risk Assessment Project

This project aims to assess credit risk using a loan prediction dataset. The workflow includes exploratory data analysis (EDA), preprocessing, model building, evaluation, hyperparameter tuning, and creating a final pipeline for predictions.

## Project Structure

```
credit-risk-assessment
├── data
│   └── Loan_Prediction.csv
├── notebooks
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Building.ipynb
│   ├── 03_Model_Evaluation.ipynb
│   ├── 04_Hyperparameter_Tuning.ipynb
│   └── 05_Final_Pipeline_and_Predictions.ipynb
├── src
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_utils.py
│   └── evaluation.py
├── models
│   └── .gitkeep
├── outputs
│   └── .gitkeep
├── requirements.txt
└── README.md
```

## Overview

The project consists of several components:

1. **Data**: The dataset used for analysis is located in the `data` directory. It contains loan application records with various features.

2. **Notebooks**: 
   - `01_EDA_and_Preprocessing.ipynb`: This notebook performs EDA and preprocessing on the dataset, including data loading, overview, quality assessment, and analysis.
   - `02_Model_Building.ipynb`: This notebook focuses on building the predictive model, defining the architecture, training the model, and saving it for future use.
   - `03_Model_Evaluation.ipynb`: This notebook evaluates the trained model's performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
   - `04_Hyperparameter_Tuning.ipynb`: This notebook performs hyperparameter tuning to optimize the model's performance using techniques like Grid Search or Random Search.
   - `05_Final_Pipeline_and_Predictions.ipynb`: This notebook creates a final pipeline for preprocessing, model training, and making predictions on new data.

3. **Source Code**: The `src` directory contains Python modules for preprocessing, feature engineering, model utilities, and evaluation functions.

4. **Models and Outputs**: The `models` and `outputs` directories are included to store trained models and output files, respectively. They contain `.gitkeep` files to ensure they are tracked in version control.

5. **Requirements**: The `requirements.txt` file lists all necessary Python packages and their versions required to run the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd credit-risk-assessment
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook and start working with the notebooks in the `notebooks` directory.

## Usage Guidelines

- Follow the notebooks in order to complete the project workflow.
- Ensure that the dataset is available in the `data` directory before running the notebooks.
- Modify the source code in the `src` directory as needed to customize preprocessing, feature engineering, or evaluation metrics.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.# EVALIX
