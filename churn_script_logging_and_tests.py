"""
A test module for the customer churn prediction library.

This module contains unit tests for all functions in the churn_library.py module,
including data import, EDA, feature engineering, model training and evaluation.
Each test function includes logging to track success and failures of the tests.

Note: This library is part of a Udacity Project. The original work belongs to the Udacity course
        "Machine Learning DevOps Engineer" courer authors.

Author: Hyunkun Kim
Date: 2025-06-22
"""
import os
import sys
import logging
import pytest
import joblib
import pandas as pd
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library_test.log',
    level=logging.INFO,
    filemode='w',
    format='[%(asctime)s] - %(levelname)s - %(message)s')


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


def test_import(import_data): # pylint: disable=redefined-outer-name
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


def test_eda(perform_eda): # pylint: disable=redefined-outer-name
    '''
    test perform eda function
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: ERROR")
        raise err

    # Check if the output images exist
    try:
        assert os.path.isfile('./images/eda/customer_churn_distribution.png')
        assert os.path.isfile('./images/eda/customer_age_distribution.png')
        assert os.path.isfile('./images/eda/martial_status.png')
        assert os.path.isfile('./images/eda/Total_Trans_Ct.png')
        assert os.path.isfile('./images/eda/correlation.png')
        logging.info("Testing perform_eda output files: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda output files: Not all expected files were created")
        raise err


def test_encoder_helper(encoder_helper): # pylint: disable=redefined-outer-name
    '''
    test encoder helper
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        # Add Churn column
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        encoded_df = encoder_helper(df, category_lst)
        logging.info("Testing encoder_helper: SUCCESS")

        # Check that the new columns were created
        try:
            for category in category_lst:
                assert f'{category}_Churn' in encoded_df.columns

            # Check that the encoded columns are not empty
            for category in category_lst:
                assert encoded_df[f'{category}_Churn'].shape[0] > 0

            logging.info("Testing encoder_helper output columns: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing encoder_helper output columns: Not all expected columns were created")
            raise err

    except Exception as err:
        logging.error("Testing encoder_helper: ERROR")
        raise err


def test_perform_feature_engineering(perform_feature_engineering): # pylint: disable=redefined-outer-name
    '''
    test perform_feature_engineering
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")

        # Prepare the dataframe with encoded features
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        df = cls.encoder_helper(df, category_lst)

        # Perform feature engineering
        x_train, x_test, y_train, y_test = perform_feature_engineering(df)
        logging.info("Testing perform_feature_engineering: SUCCESS")

        # Check that the outputs are not empty
        try:
            assert x_train.shape[0] > 0
            assert x_test.shape[0] > 0
            assert len(y_train) > 0
            assert len(y_test) > 0

            # Check that the train/test split was done correctly
            assert x_train.shape[0] == len(y_train)
            assert x_test.shape[0] == len(y_test)

            logging.info(
                "Testing perform_feature_engineering outputs: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing perform_feature_engineering outputs: The outputs are not as expected")
            raise err

    except Exception as err:
        logging.error("Testing perform_feature_engineering: ERROR")
        raise err


def test_train_models(train_models): # pylint: disable=redefined-outer-name
    '''
    test train_models
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")

        # Prepare the dataframe with encoded features
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        df = cls.encoder_helper(df, category_lst)

        # Perform feature engineering
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df)

        # Train models
        y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = train_models(
            x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")

        # Check that the outputs are not empty
        try:
            assert len(y_train_preds_lr) > 0
            assert len(y_train_preds_rf) > 0
            assert len(y_test_preds_lr) > 0
            assert len(y_test_preds_rf) > 0

            # Check that the model files were created
            assert os.path.isfile('./models/rfc_model.pkl')
            assert os.path.isfile('./models/lrc_model.pkl')

            # Check that the classification report images were created
            rf_model_result = cls.ModelResults(
                y_train, y_test, y_train_preds_rf, y_test_preds_rf)
            lr_model_result = cls.ModelResults(
                y_train, y_test, y_train_preds_lr, y_test_preds_lr)
            cls.classification_report_image(rf_model_result, lr_model_result)

            assert os.path.isfile('./images/roc_curve_result.png')
            assert os.path.isfile(
                './images/classification_report_random_forest.png')
            assert os.path.isfile(
                './images/classification_report_logistic_regression.png')

            # Check feature importance plot
            rfc = joblib.load('./models/rfc_model.pkl')
            x_all = pd.concat([x_train, x_test], axis=0)
            cls.feature_importance_plot(rfc, x_all.iloc[::10], './images')

            assert os.path.isfile('./images/feature_importance.png')
            assert os.path.isfile('./images/shap_feature_importance.png')

            logging.info("Testing train_models outputs and files: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing train_models outputs and files: Not all expected outputs or files were created")
            raise err

    except Exception as err:
        logging.error("Testing train_models: ERROR")
        raise err


def test_classification_report_image(classification_report_image): # pylint: disable=redefined-outer-name
    '''
    test classification_report_image function
    '''
    try:
        # Prepare test data
        df = cls.import_data("./data/bank_data.csv")
        df = cls.encoder_helper(df,
                                ['Gender',
                                 'Education_Level',
                                 'Marital_Status',
                                 'Income_Category',
                                 'Card_Category'])
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df)

        # Train models or load existing models
        if os.path.exists(
                './models/rfc_model.pkl') and os.path.exists('./models/lrc_model.pkl'):
            y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = cls.predict_models(
                x_train, x_test)
        else:
            y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = cls.train_models(
                x_train, x_test, y_train, y_test)

        # Create ModelResults objects
        # Note: The order of parameters here matches the usage in the main function
        # even though it's different from the class definition
        rf_model_result = cls.ModelResults(
            y_train, y_test, y_train_preds_rf, y_test_preds_rf)
        lr_model_result = cls.ModelResults(
            y_train, y_test, y_train_preds_lr, y_test_preds_lr)

        # Call the function to test
        classification_report_image(rf_model_result, lr_model_result)
        logging.info("Testing classification_report_image: SUCCESS")

        # Check that the output files exist
        try:
            assert os.path.isfile('./images/roc_curve_result.png')
            assert os.path.isfile(
                './images/classification_report_random_forest.png')
            assert os.path.isfile(
                './images/classification_report_logistic_regression.png')
            logging.info(
                "Testing classification_report_image output files: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing classification_report_image output files: Not all expected files were created")
            raise err

    except Exception as err:
        logging.error("Testing classification_report_image: ERROR")
        raise err


def test_feature_importance_plot(feature_importance_plot): # pylint: disable=redefined-outer-name
    '''
    test feature_importance_plot function
    '''
    try:
        # Prepare test data
        df = cls.import_data("./data/bank_data.csv")
        df = cls.encoder_helper(df,
                                ['Gender',
                                 'Education_Level',
                                 'Marital_Status',
                                 'Income_Category',
                                 'Card_Category'])
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df)

        # Load model
        if not os.path.exists('./models/rfc_model.pkl'):
            # Train model if it doesn't exist
            cls.train_models(x_train, x_test, y_train, y_test)

        rfc = joblib.load('./models/rfc_model.pkl')

        # Call the function to test
        x_all = pd.concat([x_train, x_test], axis=0)
        feature_importance_plot(rfc, x_all.iloc[::10], './images')
        logging.info("Testing feature_importance_plot: SUCCESS")

        # Check that the output files exist
        try:
            assert os.path.isfile('./images/feature_importance.png')
            assert os.path.isfile('./images/shap_feature_importance.png')
            logging.info(
                "Testing feature_importance_plot output files: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing feature_importance_plot output files: Not all expected files were created")
            raise err

    except Exception as err:
        logging.error("Testing feature_importance_plot: ERROR")
        raise err


def test_predict_models(predict_models): # pylint: disable=redefined-outer-name
    '''
    test predict_models function
    '''
    try:
        # Prepare test data
        df = cls.import_data("./data/bank_data.csv")
        df = cls.encoder_helper(df,
                                ['Gender',
                                 'Education_Level',
                                 'Marital_Status',
                                 'Income_Category',
                                 'Card_Category'])
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df)

        # Ensure models exist
        if not (os.path.exists('./models/rfc_model.pkl')
                and os.path.exists('./models/lrc_model.pkl')):
            # Train models if they don't exist
            cls.train_models(x_train, x_test, y_train, y_test)

        # Call the function to test
        y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = predict_models(
            x_train, x_test)
        logging.info("Testing predict_models: SUCCESS")

        # Check that the outputs are not empty
        try:
            assert len(y_train_preds_lr) > 0
            assert len(y_train_preds_rf) > 0
            assert len(y_test_preds_lr) > 0
            assert len(y_test_preds_rf) > 0

            # Check that the predictions have the correct shape
            assert len(y_train_preds_lr) == x_train.shape[0]
            assert len(y_train_preds_rf) == x_train.shape[0]
            assert len(y_test_preds_lr) == x_test.shape[0]
            assert len(y_test_preds_rf) == x_test.shape[0]

            logging.info("Testing predict_models outputs: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing predict_models outputs: The outputs are not as expected")
            raise err

    except Exception as err:
        logging.error("Testing predict_models: ERROR")
        raise err


if __name__ == "__main__":
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        '[%(asctime)s] - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    # Run pytest
    retcode = pytest.main([os.path.join(os.path.dirname(
        __file__), 'churn_script_logging_and_tests.py')])
    sys.exit(retcode)
