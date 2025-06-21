# library doc string


# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
import os
import logging

os.environ['QT_QPA_PLATFORM']='offscreen'
logging.basicConfig(filename="./logs/churn_library_debug.log",
                    format='[%(asctime)s] - %(levelname)s - %(message)s',
                    filemode='w',
                    level=logging.DEBUG)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)
    logging.debug(f"df size: {len(df)}")
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # plot churn historgram
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist();
    plt.savefig('./images/eda/customer_churn_distribution.png')
    
    # plot customer age distribution 
    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist();
    plt.savefig('./images/eda/customer_age_distribution.png')
    
    # plot martial status
    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar');
    plt.savefig('./images/eda/martial_status.png')
    
    # total trans Ct
    plt.figure(figsize=(20,10)) 
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True);
    plt.savefig('./images/eda/Total_Trans_Ct.png')
    
    # plot correlation
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    #     plt.show()
    plt.savefig('./images/eda/correlation.png')
    pass


def encoder_helper(df, category_list, response=[]):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    if len(response) != len(category_list):
        if len(category_list) != 0:
            logging.warn("warning, category_list and \
                         response must have same size")
        response = [f'{c}_churn' for c in category_list]
    for category, resp in zip(category_list, response):
        # feature encoded column
        feature_lst = []
        feature_groups = df.groupby(category).mean()['Churn']

        for val in df[category]:
            feature_lst.append(feature_groups.loc[val])

        df[resp] = feature_lst    
   
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

if __name__=="__main__":
    pth = './data/bank_data.csv'
    df = import_data(pth)
    perform_eda(df)
    category = ['Gender',
               'Education_Level',
               'Marital_Status',
               'Income_Category',
               'Card_Category']
    df = encoder_helper(df, category)
    logging.debug('encoded_results')
    logging.debug(str(df.head()))