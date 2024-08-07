# %%
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
df = pd.read_csv('/home/vibhav911/Documents/DS_Projects/shipment_price_prediction/data/train.csv')
df.head()

# %%
target_feature = 'Cost'
numeric_feature = [feature for feature in df.columns if df[feature].dtypes != 'O']
numeric_feature.remove(target_feature)
non_numeric_feature = [feature for feature in df.columns if df[feature].dtypes == 'O']
print('We have {} Numeric features:{}'.format(len(numeric_feature), numeric_feature))
print('We have {} Non numeric features:{}'.format(len(non_numeric_feature), non_numeric_feature))


# %%
df1 = df.copy()
for i in numeric_feature:
    df1[i].fillna(df1[i].median(), inplace=True)


# %%
outlier_features = ['Weight', 'Price Of Sculpture']
outlier_data = df1[outlier_features]

# %%
outlier_features = ['Weight', 'Price Of Sculpture']
outlier_data = df1[outlier_features]
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox')
df1[outlier_features] = pt.fit_transform(df1[outlier_features])
df_outlier = pd.DataFrame(outlier_data, columns=outlier_features)
df_outlier = pd.DataFrame(outlier_data, columns=outlier_features)

# %%
df1['Cost'] = np.log1p(df1['Cost'])
df1['Cost'].skew()



# %%
# Convert object datatype to datetime
df['Scheduled Date'] = pd.to_datetime(df['Scheduled Date'])
df['Delivery Date'] = pd.to_datetime(df['Delivery Date'])
df['Month'] = pd.to_datetime(df['Scheduled Date']).dt.month
df['Year'] = pd.to_datetime(df['Scheduled Date']).dt.year
numeric_feature.append('Month')
numeric_feature.append('Year')


# %%
to_drop_columns = ['Customer Id', 'Artist Name', 'Customer Location', 'Scheduled Date', 'Delivery Date']
df.drop(columns=to_drop_columns, inplace=True, axis=1)
df['Cost'].fillna(df['Cost'].median(), inplace=True)



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
#from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# %%
from dask.distributed import Client
import joblib
client= Client(processes=False)


# %%
X=df.drop(columns=['Cost'], axis=1)
y = np.abs(df['Cost'])
y = np.log1p(y)


# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle=True, test_size=0.2)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# %%
numeric_features = [feature for feature in numeric_feature if feature not in outlier_features]
Categorical_features = [feature for feature in non_numeric_feature if feature not in to_drop_columns]
(numeric_features), (Categorical_features), (outlier_features)


# %%
numeric_feature_pipeline = Pipeline(
    steps= [
        ('imputer', SimpleImputer(strategy='mean')),
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
preprocessor = ColumnTransformer(
    [
        ('Numeric Pipeline', numeric_feature_pipeline, numeric_features),
        ('Categorical Pipeline', categorical_feature_pipeline, Categorical_features),
        ('Outliers Feature Pipeline', outlier_feature_pipeline, outlier_features)
    ]
)


# %%
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


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
    #'XGBRegressor': XGBRegressor(),
    #'CatBoost Regressor': CatBoostRegressor(verbose=False, max_depth=5),
    'Adaboost Regressor': AdaBoostRegressor(),
    'SVR': SVR()
}


# %%
base_report = evaluate_models(X_train, X_test, y_train, y_test, models)


# %%
base_report


# %%
svr_params = {
    'C': [1, 10, 12, 14, 16, 18, 20, 22],
    'gamma': [0.001, 0.01, 0.1, 1, 2, 5],
    'epsilon': [0.001, 0.01, 0.1, 1, 2, 4],
    'kernel': ("rbf", "poly", "sigmoid")
    }
cat_params = {
    'learning_rate': [0.03, 0.06],
    'depth':[3, 6, 9],
    'l2_leaf_reg': [2, 3, 4],
    'boosting_type': ['Ordered', 'Plain']
}
rf_params = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}


# %%
# Model list for hyperparameter tuning
randomsearch_model = [
    ('SVR', SVR(), svr_params),
    ('RandomForestRegressor', RandomForestRegressor(), rf_params)
]
kf = KFold(n_splits=3, random_state=1, shuffle=True)


# %%
model_param = {}
for name, model, params in randomsearch_model:
    random = RandomizedSearchCV(estimator=model,
                                param_distributions=params,
                                n_iter=50,
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
    'Random Forest Regressor': RandomForestRegressor(**model_param['RandomForestRegressor'], n_jobs=-1),
    #'Catboost': CatBoostRegressor(**model_param['Catboost'], verbose=False),
    'SVR': SVR(**model_param['SVR'], verbose=False)
}


# %%
retrained_report = evaluate_models(X_train, X_test, y_train, y_test, models)


# %%
retrained_report
