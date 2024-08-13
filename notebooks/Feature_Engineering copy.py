
# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score, KFold
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
import dask
import dask_ml
import distributed
import catboost
import xgboost
warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns', None)




# %%
from dask.distributed import Client
import joblib
client= Client(processes=False)

df = pd.read_csv('/home/vibhav911/Documents/DS_Projects/shipment_price_prediction/data/train.csv')
# %%
to_drop_columns = ['Customer Id', 'Artist Name', 'Customer Location', 'Scheduled Date', 'Delivery Date']
df.drop(columns=to_drop_columns, inplace=True, axis=1)

# %%
target_feature = ['Cost']
numeric_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
numeric_features.remove(target_feature[0])
non_numeric_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
outlier_features = ['Weight', 'Price Of Sculpture']


# %%
target_feature = ['Cost']
#df.loc[target_feature[0]] = np(df[target_feature[0]])
#df[target_feature[0]] = df[target_feature[0]].values.reshape(-1,1)
#df[target_feature[0]] = pd.DataFrame(df[target_feature[0]])

#cols_at_end = target_feature[0]
#df = df[[c for c in df if c not in cols_at_end] + [c for c in cols_at_end if c in df]]
target_feature = ['Cost']
column_to_move = df.pop(target_feature[0])

# insert column with insert(location, column_name, column_value)
df.insert(len(df.columns), target_feature[0], column_to_move)

train_set, test_set = train_test_split(df, shuffle=True, random_state=42, test_size=0.2)
print(train_set['Cost'])




# %%
df.head()
df.columns



# %%
'''
X=df.drop(columns=['Cost'], axis=1)
y = np.abs(df['Cost'])
pt = PowerTransformer(method='box-cox', standardize=True)
pt.fit(y.values.reshape(-1,1))
target_lambda = pt.lambdas_[0]
y = pt.transform(y.values.reshape(-1,1))

print(target_lambda)
'''


# %%
#X_train, y_train = (train_set.drop(target_feature,axis = 1), train_set[target_feature])
#X_test, y_test = (test_set.drop(target_feature, axis = 1), test_set[target_feature])

'''
X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle=True, test_size=0.2)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
'''

#X_train.head()


# %%
#numeric_features = [feature for feature in numeric_feature if feature not in outlier_features]
#non_numeric_features = [feature for feature in non_numeric_feature if feature not in to_drop_columns]
#target_feature = ['Cost']
#(numeric_features), (non_numeric_features), (outlier_features), (target_feature)

# %%
numeric_features = list(numeric_features)
non_numeric_features = list(non_numeric_features)
outlier_features = list(outlier_features)

target_feature 
target_feature


# %%
numeric_feature_pipeline = Pipeline(
    steps= [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ]
)
categorical_feature_pipeline = Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder(drop='first'))
    ]
)
outlier_feature_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('transformer', PowerTransformer(method='box-cox', standardize=True))
    ]
)
target_feature_pipeline = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ("power_transform",PowerTransformer(method='yeo-johnson', standardize=True) )
    ]
)
preprocessor = ColumnTransformer(
    [
        ('Numeric Pipeline', numeric_feature_pipeline, numeric_features),
        ('Categorical Pipeline', categorical_feature_pipeline, non_numeric_features),
        ('Outliers Feature Pipeline', outlier_feature_pipeline, outlier_features),
        ("Target_Feature_Pipeline", target_feature_pipeline, target_feature),
    ]
)


'''
X=df.drop(columns=['Cost'], axis=1)
y = np.abs(df['Cost'])
pt = PowerTransformer(method='box-cox', standardize=True)
pt.fit(y.values.reshape(-1,1))
target_lambda = pt.lambdas_[0]
y = pt.transform(y.values.reshape(-1,1))

print(target_lambda)
'''


# %%
'''
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

target_preprocessor = preprocessor.named_transformers_['Target_Feature_Pipeline'].named_steps['power_transform']
target_lambdas = target_preprocessor.lambdas_

y_train = preprocessor.fit_transform(y_train)
y_test = preprocessor.transform(y_test)
'''
train_set = preprocessor.fit_transform(train_set)
target_preprocessor = preprocessor.named_transformers_['Target_Feature_Pipeline'].named_steps['power_transform']
target_lambda = target_preprocessor.lambdas_
test_set = preprocessor.transform(test_set)

# %%
print(target_lambda[0])




# %%
X_train, y_train = (train_set[:, :-1], train_set[:, -1])
X_test, y_test = (test_set[:, :-1], test_set[:, -1])

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_test

# %%
# functions which takes true and predicted values to calculate metrics
def evaluate_reg(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_Score = r2_score(true, predicted)
    return mae, mse, rmse, r2_Score


# %%
# function which can evaluate models and return a report
def evaluate_models(X_train, X_test, y_train, y_test, models):
    '''
    This function takes in X and y and models dictionary as input
    It splits the data into Train Test split
    Iterates through the given model dictionary and evaluates the metrics
    Returns: Dataframe which contains report of all models metrics with cost
    '''
    models_list = []
    r2_list = []
    for i in range(0, len(list(models))):
        model = list(models.values())[i]
        with joblib.parallel_backend('dask'):
            model.fit(X_train, y_train)
        # Make Predictions
        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)
        # Evalue train and test datasets
        model_train_mae, model_train_mse, model_train_rmse, model_train_r2Score = evaluate_reg(y_train, y_train_predict)
        model_test_mae, model_test_mse, model_test_rmse, model_test_r2Score = evaluate_reg(y_test, y_test_predict)
        print(list(models.keys())[i])
        models_list.append(list(models.keys())[i])
        print('Model Performance for Training Set')
        print("Mean Absolute Error: {:.4f}".format(model_train_mae))
        print('Mean Squared Error: {:.4f}'.format(model_train_mse))
        print('Root Mean Squared Error: {:.4f}'.format(model_train_rmse))
        print('r2 Score: {:.4f}'.format(model_train_r2Score))
        print('Model Performance for Test Set')
        print("Mean Absolute Error: {:.4f}".format(model_test_mae))
        print('Mean Squared Error: {:.4f}'.format(model_test_mse))
        print('Root Mean Squared Error: {:.4f}'.format(model_test_rmse))
        print('r2 Score: {:.4f}'.format(model_test_r2Score))
        r2_list.append(model_test_r2Score)
        print('='*35)
        print('\n')
    report = pd.DataFrame(list(zip(models_list, r2_list)), columns=['Model Name','r2_Score']).sort_values(by=['r2_Score'], ascending=False)
    return report



# %%
# Initializing Models
models = {
    'Linear Regression': LinearRegression(),
    'K-Neighbour Regression': KNeighborsRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'XGBRegressor': XGBRegressor(),
    'CatBoost Regressor': CatBoostRegressor(verbose=False, max_depth=5),
    'Adaboost Regressor': AdaBoostRegressor(),
    'SVR': SVR()
}


# %%
base_report = evaluate_models(X_train, X_test, y_train, y_test, models)


# %%
base_report


# %%
svr_params = {
    'C': [1],
    'gamma': [0.1],
    'epsilon': [0.01],
    'kernel': ["poly"]
    }
cat_params = {
    'learning_rate': [0.03],
    'depth':[3],
    'l2_leaf_reg': [2],
    'boosting_type': ['Ordered']
}
rf_params = {
    'n_estimators': [150],
    'max_features': ['sqrt'],
    'max_depth': [9],
    'max_leaf_nodes': [6],
}
xg_param = {
    'max_depth': [6],
    'learning_rate': [0.06],
    'subsample': [0.5],
    'n_estimators':[200]
}

# %%
# Model list for hyperparameter tuning
randomsearch_model = [
    ('SVR', SVR(), svr_params),
    ('RandomForestRegressor', RandomForestRegressor(), rf_params),
    #('Decision Tree', DecisionTreeRegressor(), decision_params),
    ('XGBRegressor', XGBRegressor(), xg_param),
    ('CatBoost Regressor',CatBoostRegressor(verbose=False),cat_params)
]
kf = KFold(n_splits=3, random_state=1, shuffle=True)


# %%
model_param = {}
for name, model, params in randomsearch_model:
    random = GridSearchCV(estimator=model,
                                param_grid=params,
                                cv=kf,
                                verbose=1,
                                n_jobs=-1)
    with joblib.parallel_backend('dask'):
        random.fit(X_train, y_train)
        model_param[name] = random.best_params_
for model_name in model_param:
    print(f'--------- Best Params for {model_name} ---------')
    print(model_param[model_name])


# %%
models = {
    'Catboost': CatBoostRegressor(**model_param['CatBoost Regressor'], verbose=False),
    'XGBRegressor': XGBRegressor(**model_param["XGBRegressor"], n_jobs=-1),
    'SVR': SVR(**model_param['SVR'], verbose=False),
   # 'Decision Tree': DecisionTreeRegressor(**model_param['Decision Tree']),
    'Random Forest Regressor': RandomForestRegressor(**model_param['RandomForestRegressor'], verbose=False)

}


# %%
retrained_report = evaluate_models(X_train, X_test, y_train, y_test, models)


# %%
retrained_report




# %%
import json
import sys
import os
import pandas as pd
from pandas import DataFrame
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from typing import Tuple, Union
import yaml
# %%
import yaml
with open("/home/vibhav911/Documents/DS_Projects/shipment_price_prediction/notebooks/DataDriftReport.yaml", 'r') as stream:
    data = yaml.safe_load(stream)

def find(key, dictionary):
    # everything is a dict
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
                
for x in find("dataset_drift", data):
    print(xk)
