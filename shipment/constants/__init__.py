import os
from os import environ
from datetime import datetime
from from_root import from_root

MODEL_CONFIG_FILE = "config/model.yaml"
SCHEMA_FILE_PATH = 'config/schema.yaml'

TIMESTAMP: str = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

DB_URL = environ['MONGO_DB_URL']



TARGET_COLUMN = "Cost"
DB_NAME = "shipmentdata"
COLLECTION_NAME = 'ship'
TEST_SIZE = 0.2
ARTIFACTS_DIR = os.path.join(from_root(), 'artifacts', TIMESTAMP)


DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
DATA_INGESTION_TRAIN_DIR = 'Train'
DATA_INGESTION_TEST_DIR = 'Test'
DATA_INGESTON_TRAIN_FILE_NAME = 'train.csv'
DATA_INGESTION_TEST_FILE_NAME = 'test.csv'


DATA_VALIDATION_ARTIFACT_DIR = "DataValidationArtifacts"
DATA_DRIFT_FILE_NAME = "DataDriftReport.yaml"

DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
TRANSFORMED_TRAIN_DATA_DIR = "Transformed_Train"
TRANSFORMED_TEST_DATA_DIR = "Transformed_Test"
TRANSFORMED_TRAIN_DATA_FILE_NAME = "transformed_train_data.npz"
TRANSFORMED_TEST_DATA_FILE_NAME = 'transformed_test_data.npz'
INPUT_PREPROCESSOR_OBJECT_FILE_NAME = 'shipping_input_preprocessor.pkl'
TARGET_PREPROCESSOR_OBJECT_FILE_NAME = 'shipping_target_preprocessor.pkl'

MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
MODEL_FILE_NAME = "shipping_price_model.pkl"
MODEL_SAVE_FORMAL = '.pkl'

BUCKET_NAME = "datascience-projects-bucket"
S3_MODEL_NAME = "shipping_price_model.pkl"

APP_HOST = "0.0.0.0"
APP_PORT = 8080
