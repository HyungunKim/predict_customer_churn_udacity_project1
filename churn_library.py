# library doc string


# import libraries
import logging
import os
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
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


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    logging.info(f"df size: {len(df)}")
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
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('./images/eda/customer_churn_distribution.png')

    # plot customer age distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_distribution.png')

    # plot martial status
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/martial_status.png')

    # total trans Ct
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/Total_Trans_Ct.png')

    # plot correlation
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    #     plt.show()
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
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
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
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    x = pd.DataFrame()
    y = df_[response]
    x[keep_cols] = df_[keep_cols]
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


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
    # scores
    logging.info('random forest results')
    logging.info('test results')
    logging.info(classification_report(y_test, y_test_preds_rf))
    logging.info('train results')
    logging.info(classification_report(y_train, y_train_preds_rf))

    logging.info('logistic regression results')
    logging.info('test results')
    logging.info(classification_report(y_test, y_test_preds_lr))
    logging.info('train results')
    logging.info(classification_report(y_train, y_train_preds_lr))

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_predictions(y_test, y_test_preds_rf, ax=ax)
    lrc_plot = RocCurveDisplay.from_predictions(y_test, y_test_preds_lr, ax=ax)

    plt.savefig(f'./images/roc_curve_result.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!

    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/classification_report_random_forest.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/classification_report_logistic_regression.png')

    plt.close()


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
    plt.figure(figsize=(20, 5))
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    plt.savefig(f"{output_pth}/shap_feature_importance.png")

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    plt.savefig(f"{output_pth}/feature_importance.png")

def train_models(X_train, X_test, y_train, y_test):
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
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    logging.info('random forest results')
    logging.info('test results')
    logging.info(classification_report(y_test, y_test_preds_rf))
    logging.info('train results')
    logging.info(classification_report(y_train, y_train_preds_rf))

    logging.info('logistic regression results')
    logging.info('test results')
    logging.info(classification_report(y_test, y_test_preds_lr))
    logging.info('train results')

    logging.info(classification_report(y_train, y_train_preds_lr))

    with open('./models/rfc_model.pkl', 'wb') as f:
        joblib.dump(cv_rfc, f)

    with open('./models/lrc_model.pkl', 'wb') as f:
        joblib.dump(lrc, f)

    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf

def predict_models(X_train, X_yest, y_train, y_test):
    with open('./models/rfc_model.pkl', 'rb') as f:
        rfc = joblib.load(f)
    with open('./models/lrc_model.pkl', 'rb') as f:
        lrc = joblib.load(f)
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_yest)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_yest)
    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf

if __name__ == "__main__":
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
    X, X_test, y, y_test = perform_feature_engineering(df, 'Churn')
    train_models(X, X_test, y, y_test)
    perform_feature_engineering()
