from ctypes import Array
import os
from shipment.logger import logging
import sys
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
from typing import Union, Tuple, Any
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer
from shipment.entity.config_entity import DataTransformationConfig
from shipment.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts
)
from shipment.exception import shippingException

class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_transformation_config: DataTransformationConfig
    ):
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_transformation_config = data_transformation_config
        
        # Reading train.csv and test.csv from data ingestion artifacts
        self.train_set = pd.read_csv(self.data_ingestion_artifacts.train_data_file_path)
        self.test_set = pd.read_csv(self.data_ingestion_artifacts.test_data_file_path)
        
    #  This method is used to get the input feature transformer object
    def get_input_feature_transformer_object(self) -> object:
        """
        Method Name :   get_data_transformer_object
        Description :   This method gives preprocessor object.        
        Output      :   Preprocessor Object. 
        """
        
        logging.info('Entered get_data_transformer_object method of Data_Transformation class')
        try:
            # Getting necessary column names from config file
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG['numerical_columns']
            categorical_colulmns = self.data_transformation_config.SCHEMA_CONFIG['categorical_columns']
            #binary_columns = self.data_transformation_config.SCHEMA_CONFIG['binary_columns']
            outliers_columns = self.data_transformation_config.SCHEMA_CONFIG['outlier_columns']
            logging.info("Got numerical cols, one cols, binary cols from schema config")
            
            # Creating transformer objects
            numerical_imputer  = SimpleImputer(strategy="median")
            outlier_imputer = SimpleImputer(strategy="median")
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            numeric_transformer = RobustScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop="first")
            outlier_transformer = PowerTransformer(method='box-cox', standardize=True)
            logging.info("Intialized RobustScaler, OneHotEncoder, SimpleImputer, PowerTransformer")
            
            # Creating Pipeline object for columns
            numeric_feature_pipeline = Pipeline(
                steps= [
                    ('Numerical_Imputer', numerical_imputer),
                    ('Numeric_Transformer', numeric_transformer)
                ]
            )
            categorical_feature_pipeline = Pipeline(
                steps=[
                    ('Categorical_imputer', categorical_imputer),
                    ('Categorical_Transformer', categorical_transformer)
                ]
            )
            outlier_feature_pipeline = Pipeline(
                steps=[
                    ('Outlier_Imputer', outlier_imputer),
                    ('Outlier_Transformer', outlier_transformer)
                ]
            )
            
            # Using transformer objects in column transformer
            preprocessor = ColumnTransformer(
                [
                    ("NumericPipeline", numeric_feature_pipeline, numerical_columns),
                    ("CategoricalPipeline", categorical_feature_pipeline, categorical_colulmns),
                    ("OutlierPipeline", outlier_feature_pipeline, outliers_columns)
                ]
            )
            logging.info(" Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of Data_Transformation class")
            return preprocessor
            
        except Exception as e:
            raise shippingException(e, sys) from e
            
    # This method is used to do the target feature transformation
    def get_target_feature_transformation_object(self, target: Any) -> Tuple[Array, object]:
        """
        Method Name :   get_target_feature_transformation
        Description :   This method gives preprocessed target feature       
        Output      :   Preprocessed numpy array
        """
        logging.info("Entered into get_target_feture_transformation in Data Transformation class")
        
        target.fillna(target.median(), inplace=True)
        target = np.abs(target)
        target = target.values.reshape(-1,1)
        # Using Power Transformer for transformation
        pt = PowerTransformer(method='box-cox', standardize=True)
        
        logging.info(" Created preprocessed feature and preprocessor object for target_feature_transformation")
        logging.info("Exited get_feature_transformer_object method of Data_Transformation class")
        return target, pt
        
        
    
    # This is static method for capping the Outliers
    @staticmethod
    def _outlier_capping(col, df:DataFrame) -> DataFrame:
        """
        Method Name :   _outlier_capping
        Description :   This method performs outlier capping in the dataframe.        
        Output      :   DataFrame. 
        """
        
        logging.info(" Entered _outlier_capping method of Data_Transformation class")
        try:
            logging.info(" Performing _outlier_capping for columns in the dataframe")
            percentile25 = df[col].quantile(0.25) # calculating 25 percentile
            percentile75 = df[col].quantile(0.75) # calculating 75 percentile
            
            # Calculating upper limit and lower linit
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + (1.5*iqr)
            lower_limit = percentile25 - (1.5*iqr)
            
            # Capping the outliers
            df.loc[(df[col] > upper_limit), col] = upper_limit
            df.loc[(df[col]< lower_limit), col] = lower_limit
            logging.info("performed _outlier_capping method of Data_Transformation class")
            
            logging.info("Exited _outlier_capping method of Data_Transformation class")
            return df
            
        except Exception as e:
            raise shippingException(e, sys) from e
            
    # This method is used to initialize data Transformation
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates data transformation.        
        Output      :   Data Transformation Artifacts. 
        """  
        
        
        logging.info("Entered initiate_data_transformation method of Data_Transformation class")
        try:
            # Creating directory for data transforamtion artifacts
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            logging.info(f"Created artifacts directory for {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}")
            
            # Getting preprocessor object
            input_preprocessor = self.get_input_feature_transformer_object()
            logging.info("Got the preprocessor object")
            
            # Getting target column name from schema file
            target_column_name = self.data_transformation_config.SCHEMA_CONFIG['target_column']
            
            # Getting numerical columns from schema file
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG['numerical_columns']
            
            logging.info("Got target column name and numerical columns from schema config")
            
            
            '''
            # Outlier Capping
            
            continuous_columns = [
                feature for feature in numerical_columns if len(self.train_set[feature].unique()) >= 25
            ]
            logging.info('Got a list of continuous_columns')
            
            [self._outlier_capping(col, self.train_set) for col in continuous_columns]
            logging.info("Outlier capped in train df")
            
            [self._outlier_capping(col, self.test_set) for col in continuous_columns]
            logging.info("Outlier capped in test df")
            '''
            
            # Getting the input features and target feature of the Training dataset
            input_feature_train_df = self.train_set.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = self.train_set[target_column_name]
            logging.info("Got train features and test feature")
            
            # Getting the input features and target feature of the testing dataset
            input_feature_test_df = self.test_set.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = self.test_set[target_column_name]
            logging.info("Got train features and test features")
            
            # Applying preprocessing object on training dataframe and testing dataframe
            input_feature_train_arr = input_preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = input_preprocessor.transform(input_feature_test_df)
            logging.info("Used the preprocessor object to transfer the test features")
            
            # Intializing and preparinng target_feature and preprocessor object
            (target_train, target_preprocessor) = self.get_target_feature_transformation_object(target_feature_train_df) 
            (target_test, _) = self.get_target_feature_transformation_object(target_feature_test_df)
            
            # applying preprocessed object on target_train feature
            preprocessed_target_train = target_preprocessor.fit_transform(target_train)
            # Getting the lambda values for inverse transformation in model predictor
            target_lambda = target_preprocessor.lambdas_[0]
            # applying preprocessed object on target_test feature
            preprocessed_target_test = target_preprocessor.transform(target_test)
            
            # Concatinating input feature array and target feature arra of Train dataset
            train_arr = np.c_[input_feature_train_arr, preprocessed_target_train]
            logging.info("Created train array")
            
            # Creating directory for transformed train dataset array and saving the array
            os.makedirs(self.data_transformation_config.TRANSFORMED_TRAIN_DATA_DIR, exist_ok=True)
            transformed_train_file = self.data_transformation_config.UTILS.save_numpy_array_data(
                self.data_transformation_config.TRANSFORMED_TRAIN_FILE_PATH, train_arr
            )
            logging.info(f"Saved train arry to {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}")
            
            # Concatinating input feature array and target feature array of Test dataset
            test_arr = np.c_[input_feature_test_arr, preprocessed_target_test]
            logging.info("Creating test array")
            
            # Creating directory for transformed test dataset array and saving the array
            os.makedirs(self.data_transformation_config.TRANSFORMED_TEST_DATA_DIR, exist_ok=True)
            transformed_test_file = self.data_transformation_config.UTILS.save_numpy_array_data(
                self.data_transformation_config.TRANSFORMED_TEST_FILE_PATH, test_arr
            )
            logging.info(f"Saved test array to {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}")
            
            # Saving the input_preprocessor object to data transformation artifacts directory
            input_preprocessor_obj_file = self.data_transformation_config.UTILS.save_object(
                self.data_transformation_config.INPUT_PREPROCESSOR_FILE_PATH, input_preprocessor
            )
            
            # Saving the target_preprocessor object to data transformation artifacts directiry
            target_preprocessor_obj_file = self.data_transformation_config.UTILS.save_object(
                self.data_transformation_config.TARGET_PREPROCESSOR_FILE_PATH, target_preprocessor
            )
            logging.info("Saved the preprocessor object in DataTransformation artifacts directoory")
            logging.info("Exited initiate_data_transformation method of Data_Transformation class")
            
            # Saving data transformation artifacts
            data_transformation_artifacts = DataTransformationArtifacts(
                transformed_input_object_file_path = input_preprocessor_obj_file,
                transformed_target_object_file_path = target_preprocessor_obj_file,
                transformed_train_file_path = transformed_train_file,
                transformed_test_file_path = transformed_test_file
            )
            
            return data_transformation_artifacts
            
        except Exception as e:
            raise shippingException(e, sys) from e