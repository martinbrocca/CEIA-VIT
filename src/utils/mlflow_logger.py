# src/utils/mlflow_logger.py
"""Helper class for MLflow experiment tracking, supporting both Databricks and local setups."""
import mlflow
import time
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config import PROJECT_ROOT

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Set MLflow tracking to Databricks
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

if DATABRICKS_HOST and DATABRICKS_TOKEN:
    mlflow.set_tracking_uri("databricks")
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
    print(f"✓ Using Databricks MLflow at {DATABRICKS_HOST}")
else:
    # Fallback to local
    MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    print(f"⚠ Using local MLflow at {MLFLOW_TRACKING_URI}")

class MLflowLogger:
    """Helper class for MLflow experiment tracking"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        
        # For Databricks, experiments need the /Users/email@domain.com/ prefix
        if DATABRICKS_HOST:
            # Get or create experiment in Databricks workspace
            try:
                mlflow.set_experiment(f"/Users/{os.getenv('DATABRICKS_USER', 'default')}/{experiment_name}")
            except:
                # Fallback to root level experiment
                mlflow.set_experiment(f"/{experiment_name}")
        else:
            mlflow.set_experiment(experiment_name)
    
    def log_preprocessing(self, stats: dict):
        """Log data preprocessing metrics"""
        with mlflow.start_run(run_name="data_preprocessing"):
            # Log parameters
            mlflow.log_param("dataset_source", "Food.com")
            mlflow.log_param("raw_data_path", stats.get("raw_path"))
            
            # Log metrics
            mlflow.log_metric("total_recipes_raw", stats.get("total_raw", 0))
            mlflow.log_metric("total_recipes_processed", stats.get("total_processed", 0))
            mlflow.log_metric("recipes_filtered_out", stats.get("filtered_out", 0))
            mlflow.log_metric("avg_ingredients_per_recipe", stats.get("avg_ingredients", 0))
            mlflow.log_metric("avg_recipe_text_length", stats.get("avg_text_length", 0))
            
            # Log artifacts
            if "output_path" in stats:
                mlflow.log_param("processed_data_path", stats["output_path"])
            
            mlflow.set_tag("stage", "preprocessing")
            mlflow.set_tag("status", "completed")
    
    def log_embedding_creation(self, model_name: str, stats: dict):
        """Log embedding creation metrics"""
        with mlflow.start_run(run_name=f"embeddings_{model_name.split('/')[-1]}"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("embedding_type", stats.get("embedding_type", "text"))
            mlflow.log_param("batch_size", stats.get("batch_size", 128))
            mlflow.log_param("device", stats.get("device", "cpu"))
            
            # Log metrics
            mlflow.log_metric("num_recipes", stats.get("num_recipes", 0))
            mlflow.log_metric("embedding_dimension", stats.get("embedding_dim", 0))
            mlflow.log_metric("total_time_seconds", stats.get("total_time", 0))
            mlflow.log_metric("recipes_per_second", stats.get("recipes_per_second", 0))
            mlflow.log_metric("embedding_size_mb", stats.get("embedding_size_mb", 0))
            
            # Log artifacts
            if "embedding_path" in stats:
                mlflow.log_param("embedding_output_path", stats["embedding_path"])
            
            mlflow.set_tag("stage", "embedding_creation")
            mlflow.set_tag("model_type", stats.get("model_type", "unknown"))
            mlflow.set_tag("status", "completed")
    
    def log_index_building(self, stats: dict):
        """Log FAISS index building metrics"""
        with mlflow.start_run(run_name="faiss_index_building"):
            # Log parameters
            mlflow.log_param("index_type", stats.get("index_type", "IndexFlatIP"))
            mlflow.log_param("dimension", stats.get("dimension", 0))
            
            # Log metrics
            mlflow.log_metric("num_vectors", stats.get("num_vectors", 0))
            mlflow.log_metric("build_time_seconds", stats.get("build_time", 0))
            
            # Sample search performance
            if "sample_search_time" in stats:
                mlflow.log_metric("avg_search_time_ms", stats["sample_search_time"])
            
            mlflow.set_tag("stage", "index_building")
            mlflow.set_tag("status", "completed")