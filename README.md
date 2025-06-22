# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project identifies credit card customers that are most likely to churn. The project includes a Python package for a machine learning model that predicts customer churn.

The project follows best coding practices with:
- Clean, modular, and well-documented code
- Unit tests to ensure code quality
- Logging to track model performance and code execution
- Production-ready code that can be easily maintained and extended

## Files and Data Description

### Main Files
- `churn_library.py`: Contains the main functions for the ML pipeline, including data loading, EDA, feature engineering, model training, and evaluation.
- `churn_script_logging_and_tests.py`: Contains unit tests for the functions in `churn_library.py` and logs results.
- `churn_notebook.ipynb`: Jupyter notebook containing the original exploratory data analysis.
- `README.md`: This file, providing an overview of the project and instructions.

### Data
- `data/bank_data.csv`: The dataset containing customer information and churn status.

### Directories
- `images/`: Contains visualizations generated during EDA and model evaluation.
  - `images/eda/`: Exploratory data analysis plots.
  - `images/results/`: Model performance visualizations.
- `logs/`: Contains log files generated during code execution.
- `models/`: Contains trained machine learning models saved as pickle files.

## Running Files

### Environment Setup
1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements_py3.9.txt
   ```

### Running the Machine Learning Pipeline
To run the full machine learning pipeline:
```
python churn_library.py
```

This will:
1. Load the data
2. Perform EDA and generate visualizations
3. Perform feature engineering
4. Train machine learning models (Random Forest and Logistic Regression)
5. Generate and save model performance metrics and visualizations

### Running Tests
To run the tests and verify the functionality of the code:
```
python churn_script_logging_and_tests.py
```

This will:
1. Test each function in `churn_library.py`
2. Log the results to `./logs/churn_library_test.log`
3. Verify that all outputs (images, models) are created correctly

### Logging
- All logs are stored in the `logs/` directory
- The main pipeline logs to `logs/churn_library_debug.log`
- The tests log to `logs/churn_library_test.log`

The logs include information about:
- Function execution status
- Data shapes and characteristics
- Model performance metrics
- Any errors or warnings that occur during execution
