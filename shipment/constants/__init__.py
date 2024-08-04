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