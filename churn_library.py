# library doc string

'''
A library for predicting customer churn in banking services.

This library provides functionality for:
- Data import and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and encoding
- Model training (Random Forest and Logistic Regression)
- Model evaluation and result visualization
- Feature importance analysis

Note: This library is part of a Udacity Project. The original work belongs to the Udacity course
        "Machine Learning DevOps Engineer" courer authors.

Author: Hyunkun Kim
Date: 2025-06-22
'''
# import libraries
import logging
import os
from dataclasses import dataclass
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.makedirs('./images/eda', exist_ok=True)
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(filename="./logs/churn_library_debug.log",
                    format='[%(asctime)s] - %(levelname)s - %(message)s',
                    filemode='w',
                    level=logging.INFO)
console = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

@dataclass
class ModelResults:
    """
    Represents the results of a predictive model's performance.

    This data class stores the actual and predicted values for both the
    training and testing datasets. It is designed to provide a
    structured way to encapsulate and access these values, enabling
    downstream analysis or reporting of model performance.
    """
    y_train_preds_: np.ndarray
    y_test_preds_: np.ndarray
    y_train_: np.ndarray
    y_test_: np.ndarray


def import_data(pth_):
    '''
    returns dataframe for the csv found at pth

    input:
            pth_: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_ = pd.read_csv(pth_)
    logging.info("df size: %d", len(df_))
    return df_


def perform_eda(df_):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # plot churn historgram
    df_['Churn'] = df_['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df_['Churn'].hist()
    plt.title('Churn Distribution')
    plt.savefig('./images/eda/customer_churn_distribution.png')

    # plot customer age distribution
    plt.figure(figsize=(20, 10))
    df_['Customer_Age'].hist()
    plt.title('Customer Age Distribution')
    plt.savefig('./images/eda/customer_age_distribution.png')

    # plot martial status
    plt.figure(figsize=(20, 10))
    df_.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Marital Status Distribution')
    plt.savefig('./images/eda/martial_status.png')

    # total trans Ct
    plt.figure(figsize=(20, 10))
    sns.histplot(df_['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Total Trans Ct Distribution')
    plt.savefig('./images/eda/Total_Trans_Ct.png')

    # plot correlation
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df_.select_dtypes(
            include=[
                'int64',
                'float64']).corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.title('Correlation Heatmap')
    plt.savefig('./images/eda/correlation.png')


def encoder_helper(df_, category_list, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
             used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    df_['Churn'] = df_['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    responses = [f'{c}_{response}' for c in category_list]
    for cat, resp in zip(category_list, responses):
        # feature encoded column
        feature_lst = []
        feature_groups = df_.groupby(cat)[response].mean()

        for val in df_[cat]:
            feature_lst.append(feature_groups.loc[val])

        df_[resp] = feature_lst

    return df_


def perform_feature_engineering(df_, response="Churn"):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for
                naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x = pd.DataFrame()
    y = df_[response]
    x[keep_cols] = df_[keep_cols]
    # train test split
    x_train_, x_test_, y_train_, y_test_ = train_test_split(
        x, y, test_size=0.3, random_state=42)
    return x_train_, x_test_, y_train_, y_test_


def classification_report_image(rf_model_results_, lrc_model_results_):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            rf_model_results: ModelResults object containing training results
            lrc_model_results: ModelResults object containing testing results

    output:
             None
    '''
    y_train_ = rf_model_results_.y_train_
    y_test_ = rf_model_results_.y_test_
    y_train_preds_rf_ = rf_model_results_.y_train_preds_
    y_test_preds_rf_ = rf_model_results_.y_test_preds_
    y_train_preds_lr_ = lrc_model_results_.y_train_preds_
    y_test_preds_lr_ = lrc_model_results_.y_test_preds_
    # scores
    logging.info('random forest results')
    logging.info('test results')
    logging.info(classification_report(y_test_, y_test_preds_rf_))
    logging.info('train results')
    logging.info(classification_report(y_train_, y_train_preds_rf_))

    logging.info('logistic regression results')
    logging.info('test results')
    logging.info(classification_report(y_test_, y_test_preds_lr_))
    logging.info('train results')
    logging.info(classification_report(y_train_, y_train_preds_lr_))

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    RocCurveDisplay.from_predictions(y_test_, y_test_preds_rf_, ax=ax)
    RocCurveDisplay.from_predictions(y_test_, y_test_preds_lr_, ax=ax)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('./images/roc_curve_result.png')
    plt.close()

    plt.figure(figsize=(7, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test_, y_test_preds_rf_)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!

    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train_, y_train_preds_rf_)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/classification_report_random_forest.png')
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train_, y_train_preds_lr_)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test_, y_test_preds_lr_)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/classification_report_logistic_regression.png')

    plt.close()


def feature_importance_plot(model, x_data_, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    plt.figure(figsize=(10, 5))
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(x_data_)
    shap.summary_plot(shap_values, x_data_, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.ylabel('SHAP Value')
    plt.savefig(f"{output_pth}/shap_feature_importance.png")

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data_.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data_.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data_.shape[1]), names, rotation=90)
    plt.savefig(f"{output_pth}/feature_importance.png")


def train_models(x_train_, x_test_, y_train_, y_test_):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              y_train_preds_lr: training predictions from logistic regression
              y_train_preds_rf: training predictions from random forest
              y_test_preds_lr: test predictions from logistic regression
              y_test_preds_rf: test predictions from random forest
    '''
    # This cell may take up to 15-20 minutes to run
    # grid search
    rfc_ = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc_ = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc_, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train_, y_train_)

    lrc_.fit(x_train_, y_train_)

    y_train_preds_rf_ = cv_rfc.best_estimator_.predict(x_train_)
    y_test_preds_rf_ = cv_rfc.best_estimator_.predict(x_test_)

    y_train_preds_lr_ = lrc_.predict(x_train_)
    y_test_preds_lr_ = lrc_.predict(x_test_)

    # scores
    logging.info('random forest results')
    logging.info('test results')
    logging.info(classification_report(y_test_, y_test_preds_rf_))
    logging.info('train results')
    logging.info(classification_report(y_train_, y_train_preds_rf_))

    logging.info('logistic regression results')
    logging.info('test results')
    logging.info(classification_report(y_test_, y_test_preds_lr_))
    logging.info('train results')

    logging.info(classification_report(y_train_, y_train_preds_lr_))

    with open('./models/rfc_model.pkl', 'wb') as f:
        joblib.dump(cv_rfc.best_estimator_, f)

    with open('./models/lrc_model.pkl', 'wb') as f:
        joblib.dump(lrc_, f)

    return y_train_preds_lr_, y_train_preds_rf_, y_test_preds_lr_, y_test_preds_rf_


def predict_models(x_train_, x_test_):
    """
    Predicts outcomes using pre-trained models for both the training and test datasets.

    This function loads pre-trained models from pickle files and uses them to make
    predictions on the training and test datasets. It returns the predictions made
    by both models for both datasets.

    Args:
        x_train_: Features for the training dataset.
        x_test_: Features for the test dataset.

    Returns:
        A tuple containing the following:
            - Predictions from the logistic regression model on the training dataset.
            - Predictions from the random forest classifier model on the training dataset.
            - Predictions from the logistic regression model on the test dataset.
            - Predictions from the random forest classifier model on the test dataset.
    """
    with open('./models/rfc_model.pkl', 'rb') as f:
        rfc_ = joblib.load(f)
    with open('./models/lrc_model.pkl', 'rb') as f:
        lrc_ = joblib.load(f)
    y_train_preds_rf_ = rfc_.predict(x_train_)
    y_test_preds_rf_ = rfc_.predict(x_test_)
    y_train_preds_lr_ = lrc_.predict(x_train_)
    y_test_preds_lr_ = lrc_.predict(x_test_)
    return y_train_preds_lr_, y_train_preds_rf_, y_test_preds_lr_, y_test_preds_rf_


if __name__ == "__main__":
    PTH = './data/bank_data.csv'
    df = import_data(PTH)
    category = ['Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category']
    df = encoder_helper(df, category)
    logging.debug('encoded_results')
    logging.debug(str(df.head()))
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    logging.info("performing EDA")
    perform_eda(df)
    if os.path.exists(
            './models/rfc_model.pkl') and os.path.exists('./models/lrc_model.pkl'):
        logging.info('models exist, loading models')
        y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = predict_models(
            x_train, x_test)
    else:
        logging.info('models do not exist, training models')
        y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = train_models(
            x_train, x_test, y_train, y_test)

    rf_model_result = ModelResults(y_train, y_test, y_train_preds_rf, y_test_preds_rf)
    lr_model_result = ModelResults(y_train, y_test, y_train_preds_lr, y_test_preds_lr)
    logging.info("calling classification report image")
    classification_report_image(rf_model_result, lr_model_result)
    rfc = joblib.load('./models/rfc_model.pkl')
    X_all = pd.concat([x_train, x_test], axis=0)
    logging.info("calling feature importance plot")
    # sample only fraction of X_all to save time
    feature_importance_plot(rfc, X_all.iloc[::10], './images')
