from dataclasses import dataclass
from shipment.logger import logging
from from_root import from_root
import os
# from shipment.configuration.s3_operation import S3Operation
from shipment.utils.main_utils import MainUtils
from shipment.constants import *


# Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.UTILS = MainUtils()
        logging.info("MainUtls taken")
        self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(filename=SCHEMA_FILE_PATH)
        self.DB_NAME = DB_NAME
        self.COLLECTION_NAME = COLLECTION_NAME
        self.DROP_COLS = list(self.SCHEMA_CONFIG['drop_columns'])
        
        logging.info('Drop column taken')
        
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(
            from_root(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR
        )
        
        self.TRAIN_DATA_ARTIFACT_FILE_DIR: str = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TRAIN_DIR
        )
        
        self.TEST_DATA_ARTIFACT_FILE_DIR: str = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TEST_DIR
        )
        
        self.TRAIN_DATA_FILE_PATH: str = os.path.join(
            self.TRAIN_DATA_ARTIFACT_FILE_DIR, DATA_INGESTON_TRAIN_FILE_NAME
        )
        
        self.TEST_DATA_FILE_PATH: str = os.path.join(
            self.TEST_DATA_ARTIFACT_FILE_DIR, DATA_INGESTION_TEST_FILE_NAME
        )
# Data Validation Configurations
@dataclass
class DataValidationConfig:
    def __init__(self):
        self.UTILS = MainUtils()
        self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(filename = SCHEMA_FILE_PATH)
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(from_root(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_VALIDATION_ARTIFACTS_DIR: str = os.path.join(from_root(), ARTIFACTS_DIR, DATA_VALIDATION_ARTIFACT_DIR)
        self.DATA_DRIFT_FILE_PATH: str = os.path.join(self.DATA_VALIDATION_ARTIFACTS_DIR, DATA_DRIFT_FILE_NAME)