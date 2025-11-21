# src/models/embeddings.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config import (
    PROCESSED_RECIPES,
    RECIPE_EMBEDDINGS,
    DEFAULT_EMBEDDING_MODEL,
    setup_directories
)
from utils.device import get_device

# Add to imports at top
from utils.mlflow_logger import MLflowLogger
import os

# Update the create_recipe_embeddings function
def create_recipe_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 128,
    max_recipes: int = None,
    log_to_mlflow: bool = True
):
    """
    Create embeddings for all recipes using sentence-transformers
    """
    import time
    start_time = time.time()
    
    setup_directories()
    
    # Load device
    device = get_device()
    
    # Load processed recipes
    print(f"Loading recipes from {PROCESSED_RECIPES}...")
    df = pd.read_parquet(PROCESSED_RECIPES)
    
    if max_recipes:
        df = df.head(max_recipes)
        print(f"Limited to {max_recipes} recipes for testing")
    
    num_recipes = len(df)
    print(f"Processing {num_recipes} recipes")
    
    # Load model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    model = model.to(device)
    
    # Get recipe texts
    recipe_texts = df['recipe_text'].tolist()
    
    # Create embeddings with progress bar
    print("Creating embeddings...")
    embeddings = model.encode(
        recipe_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=str(device)
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Time taken: {total_time:.2f} seconds ({num_recipes/total_time:.0f} recipes/sec)")
    
    # Save embeddings
    print(f"Saving embeddings to {RECIPE_EMBEDDINGS}...")
    np.save(RECIPE_EMBEDDINGS, embeddings)
    
    # Calculate embedding size
    embedding_size_mb = os.path.getsize(RECIPE_EMBEDDINGS) / (1024 * 1024)
    
    # Also save recipe IDs for mapping
    recipe_ids_path = RECIPE_EMBEDDINGS.parent / "recipe_ids.npy"
    np.save(recipe_ids_path, df['id'].values)
    
    print("✓ Done!")
    print(f"Embeddings saved to: {RECIPE_EMBEDDINGS}")
    print(f"Recipe IDs saved to: {recipe_ids_path}")
    print(f"Embedding file size: {embedding_size_mb:.1f} MB")
    
    # Log to MLflow
    if log_to_mlflow:
        print("\nLogging to MLflow...")
        logger = MLflowLogger("recipe-search-pipeline")
        logger.log_embedding_creation(model_name, {
            "embedding_type": "text",
            "model_type": "sentence-transformers",
            "batch_size": batch_size,
            "device": str(device),
            "num_recipes": num_recipes,
            "embedding_dim": embeddings.shape[1],
            "total_time": total_time,
            "recipes_per_second": num_recipes / total_time,
            "embedding_size_mb": embedding_size_mb,
            "embedding_path": str(RECIPE_EMBEDDINGS)
        })
        print("✓ Logged to MLflow")
    
    return embeddings, df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Model name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--max-recipes", type=int, default=None, help="Limit recipes (for testing)")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    
    args = parser.parse_args()
    
    create_recipe_embeddings(
        model_name=args.model,
        batch_size=args.batch_size,
        max_recipes=args.max_recipes,
        log_to_mlflow=not args.no_mlflow
    )