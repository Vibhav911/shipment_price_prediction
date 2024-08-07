from dataclasses import dataclass

# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    train_data_file_path: str
    test_data_file_path: str

# Data Validation Artifact
@dataclass
class DataValidationArtifacts:
    data_drift_file_path: str
    validation_status: bool
    
# Data Transformation Artifacts
@dataclass
class DataTransformationArtifacts:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    
# Model Trainer
@dataclass
class ModelTrainerArtifacts:
    trained_model_file_path: str