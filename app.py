import sys
from shipment.logger import logging
from shipment.exception import shippingException
from shipment.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
   obj = TrainPipeline()
   obj.run_pipeline()