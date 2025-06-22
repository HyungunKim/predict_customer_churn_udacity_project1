import os
import logging
import churn_library as cls
import pytest
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(
    filename='./logs/churn_library_test.log',
    level=logging.INFO,
    filemode='w',
    format='[%(asctime)s] - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Create logs directory if it doesn't exist
if not os.path.exists('./logs'):
    os.makedirs('./logs')

@pytest.fixture(scope="module")
def import_data():
    """
    Fixture for the import_data function
    """
    return cls.import_data

@pytest.fixture(scope="module")
def perform_eda():
    """
    Fixture for the perform_eda function
    """
    return cls.perform_eda

@pytest.fixture(scope="module")
def encoder_helper():
    """
    Fixture for the encoder_helper function
    """
    return cls.encoder_helper

@pytest.fixture(scope="module")
def perform_feature_engineering():
    """
    Fixture for the perform_feature_engineering function
    """
    return cls.perform_feature_engineering

@pytest.fixture(scope="module")
def train_models():
    """
    Fixture for the train_models function
    """
    return cls.train_models

@pytest.fixture(scope="module")
def classification_report_image():
    """
    Fixture for the classification_report_image function
    """
    return cls.classification_report_image

@pytest.fixture(scope="module")
def feature_importance_plot():
    """
    Fixture for the feature_importance_plot function
    """
    return cls.feature_importance_plot

@pytest.fixture(scope="module")
def predict_models():
    """
    Fixture for the predict_models function
    """
    return cls.predict_models

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    # Run pytest
    retcode = pytest.main([os.path.join(os.path.dirname(__file__), 'churn_script_logging_and_tests.py')])
    exit(retcode)
